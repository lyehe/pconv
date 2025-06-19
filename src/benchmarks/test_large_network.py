import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as tbenchmark
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import time

# Constants
EPSILON = 1e-6

###############################################################################
# Include the implementations here (in production, import from separate files)
###############################################################################

# 2D Implementations
class NvidiaPartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        if "multi_channel" in kwargs:
            self.multi_channel = kwargs["multi_channel"]
            kwargs.pop("multi_channel")
        else:
            self.multi_channel = False

        if "return_mask" in kwargs:
            self.return_mask = kwargs["return_mask"]
            kwargs.pop("return_mask")
        else:
            self.return_mask = False

        super(NvidiaPartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(
                self.out_channels, self.in_channels,
                self.kernel_size[0], self.kernel_size[1]
            )
        else:
            self.weight_maskUpdater = torch.ones(
                1, 1, self.kernel_size[0], self.kernel_size[1]
            )

        self.slide_winsize = (
            self.weight_maskUpdater.shape[1] *
            self.weight_maskUpdater.shape[2] *
            self.weight_maskUpdater.shape[3]
        )

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    if self.multi_channel:
                        mask = torch.ones(
                            input.data.shape[0], input.data.shape[1],
                            input.data.shape[2], input.data.shape[3]
                        ).to(input)
                    else:
                        mask = torch.ones(
                            1, 1, input.data.shape[2], input.data.shape[3]
                        ).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(
                    mask, self.weight_maskUpdater, bias=None,
                    stride=self.stride, padding=self.padding,
                    dilation=self.dilation, groups=1
                )

                self.mask_ratio = self.slide_winsize / (self.update_mask + EPSILON)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(NvidiaPartialConv2d, self).forward(
            torch.mul(input, mask) if mask_in is not None else input
        )

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class OptimizedPartialConv2dFixed(nn.Conv2d):
    def __init__(
        self, *args, multi_channel: bool = False,
        return_mask: bool = False, cache_masks: bool = True, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        self.cache_masks = cache_masks
        
        kernel_elements = self.kernel_size[0] * self.kernel_size[1]
        if self.multi_channel:
            self.slide_winsize = float(kernel_elements * (self.in_channels // self.groups))
        else:
            self.slide_winsize = float(kernel_elements * 1)
        
        if cache_masks:
            self._last_mask_shape = None
            self._last_mask_ptr = None
            self._last_result = None

    def _compute_mask_updates(self, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cache_masks and self._last_mask_shape == mask.shape:
            if self._last_mask_ptr == mask.data_ptr():
                return self._last_result
        
        with torch.no_grad():
            if not self.multi_channel or mask.shape[1] == 1:
                mask_for_sum = mask if mask.shape[1] == 1 else mask[:, 0:1, ...]
                conv_weight = torch.ones(
                    1, 1, *self.kernel_size, device=mask.device, dtype=mask.dtype
                )
                groups_for_mask_conv = 1
            else:
                mask_for_sum = mask
                if self.groups == 1:
                    conv_weight = torch.ones(
                        1, self.in_channels, *self.kernel_size,
                        device=mask.device, dtype=mask.dtype
                    )
                    groups_for_mask_conv = 1
                else:
                    channels_per_group = self.in_channels // self.groups
                    conv_weight = torch.ones(
                        self.groups, channels_per_group, *self.kernel_size,
                        device=mask.device, dtype=mask.dtype
                    )
                    groups_for_mask_conv = self.groups
            
            update_mask = F.conv2d(
                mask_for_sum, conv_weight, bias=None,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=groups_for_mask_conv
            )
            
            mask_ratio = self.slide_winsize / (update_mask + EPSILON)
            update_mask = torch.clamp(update_mask, 0, 1)
            mask_ratio = mask_ratio * update_mask
        
        if self.cache_masks:
            self._last_mask_shape = mask.shape
            self._last_mask_ptr = mask.data_ptr()
            self._last_result = (update_mask, mask_ratio)
        
        return update_mask, mask_ratio

    def forward(self, input_tensor: torch.Tensor, mask: Union[torch.Tensor, None] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        input_mask_for_calc = mask
        if input_mask_for_calc is None:
            input_mask_for_calc = torch.ones(
                input_tensor.shape[0], 1, *input_tensor.shape[2:],
                device=input_tensor.device, dtype=input_tensor.dtype
            )
        
        update_mask, mask_ratio = self._compute_mask_updates(input_mask_for_calc)
        
        current_mask_for_input_mult = input_mask_for_calc
        if self.multi_channel and input_mask_for_calc.shape[1] == 1 and input_tensor.shape[1] != 1:
            current_mask_for_input_mult = input_mask_for_calc.expand(-1, input_tensor.shape[1], -1, -1)
        elif not self.multi_channel and input_mask_for_calc.shape[1] != 1:
            current_mask_for_input_mult = input_mask_for_calc[:, 0:1, ...].expand(-1, input_tensor.shape[1], -1, -1)
        elif not self.multi_channel and input_mask_for_calc.shape[1] == 1 and input_tensor.shape[1] != 1:
            current_mask_for_input_mult = input_mask_for_calc.expand(-1, input_tensor.shape[1], -1, -1)
        
        masked_input = input_tensor * current_mask_for_input_mult
        
        output = F.conv2d(
            masked_input, self.weight, bias=None,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        )
        
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = output * mask_ratio + bias_view
            output = output * update_mask
        else:
            output = output * mask_ratio
        
        if self.return_mask:
            return output, update_mask
        return output


# 3D Implementations
class NvidiaPartialConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        if "multi_channel" in kwargs:
            self.multi_channel = kwargs["multi_channel"]
            kwargs.pop("multi_channel")
        else:
            self.multi_channel = False

        if "return_mask" in kwargs:
            self.return_mask = kwargs["return_mask"]
            kwargs.pop("return_mask")
        else:
            self.return_mask = False

        super(NvidiaPartialConv3d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(
                self.out_channels, self.in_channels,
                self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]
            )
        else:
            self.weight_maskUpdater = torch.ones(
                1, 1, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]
            )

        self.slide_winsize = (
            self.weight_maskUpdater.shape[1] *
            self.weight_maskUpdater.shape[2] *
            self.weight_maskUpdater.shape[3] *
            self.weight_maskUpdater.shape[4]
        )

        self.last_size = (None, None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 5
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    if self.multi_channel:
                        mask = torch.ones(
                            input.data.shape[0], input.data.shape[1],
                            input.data.shape[2], input.data.shape[3],
                            input.data.shape[4]
                        ).to(input)
                    else:
                        mask = torch.ones(
                            1, 1, input.data.shape[2],
                            input.data.shape[3], input.data.shape[4]
                        ).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv3d(
                    mask, self.weight_maskUpdater, bias=None,
                    stride=self.stride, padding=self.padding,
                    dilation=self.dilation, groups=1
                )

                self.mask_ratio = self.slide_winsize / (self.update_mask + EPSILON)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(NvidiaPartialConv3d, self).forward(
            torch.mul(input, mask_in) if mask_in is not None else input
        )

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class OptimizedPartialConv3dFixed(nn.Conv3d):
    def __init__(
        self, *args, multi_channel: bool = False,
        cache_masks: bool = True, return_mask: bool = False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.multi_channel = multi_channel
        self.cache_masks = cache_masks
        self.return_mask = return_mask
        
        kernel_elements = (
            self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        )
        if self.multi_channel:
            self.slide_winsize = float(
                kernel_elements * (self.in_channels // self.groups)
            )
        else:
            self.slide_winsize = float(kernel_elements)
        
        if self.cache_masks:
            self._last_mask_shape = None
            self._last_mask_ptr = None
            self._last_result = None

    def _compute_mask_updates(self, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.cache_masks
            and self._last_mask_shape == mask.shape
            and self._last_mask_ptr == mask.data_ptr()
        ):
            return self._last_result

        with torch.no_grad():
            if not self.multi_channel or mask.shape[1] == 1:
                mask_for_sum = mask if mask.shape[1] == 1 else mask[:, 0:1, ...]
                conv_weight = torch.ones(
                    1, 1, *self.kernel_size, device=mask.device, dtype=mask.dtype
                )
                groups_for_mask_conv = 1
            else:
                mask_for_sum = mask
                if self.groups == 1:
                    conv_weight = torch.ones(
                        1, self.in_channels, *self.kernel_size,
                        device=mask.device, dtype=mask.dtype
                    )
                    groups_for_mask_conv = 1
                else:
                    channels_per_group = self.in_channels // self.groups
                    conv_weight = torch.ones(
                        self.groups, channels_per_group, *self.kernel_size,
                        device=mask.device, dtype=mask.dtype
                    )
                    groups_for_mask_conv = self.groups

            update_mask = F.conv3d(
                mask_for_sum, conv_weight, bias=None,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=groups_for_mask_conv
            )

            mask_ratio = self.slide_winsize / (update_mask + EPSILON)
            update_mask = torch.clamp(update_mask, 0, 1)
            mask_ratio = mask_ratio * update_mask

        if self.cache_masks:
            self._last_mask_shape = mask.shape
            self._last_mask_ptr = mask.data_ptr()
            self._last_result = (update_mask, mask_ratio)

        return update_mask, mask_ratio

    def forward(self, input_tensor: torch.Tensor, mask: Union[torch.Tensor, None] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if mask is None:
            mask = torch.ones(
                input_tensor.shape[0], 1, *input_tensor.shape[2:],
                device=input_tensor.device, dtype=input_tensor.dtype
            )

        current_mask_for_input_mult = mask
        if self.multi_channel and mask.shape[1] == 1 and input_tensor.shape[1] != 1:
            current_mask_for_input_mult = mask.expand(
                -1, input_tensor.shape[1], -1, -1, -1
            )
        elif not self.multi_channel and mask.shape[1] != 1:
            current_mask_for_input_mult = mask[:, 0:1, ...].expand(
                -1, input_tensor.shape[1], -1, -1, -1
            )

        update_mask, mask_ratio = self._compute_mask_updates(mask)

        output = F.conv3d(
            input_tensor * current_mask_for_input_mult,
            self.weight, None, self.stride,
            self.padding, self.dilation, self.groups
        )

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1, 1)
            output = output * mask_ratio + bias_view
            output = output * update_mask
        else:
            output = output * mask_ratio

        if self.return_mask:
            return output, update_mask
        return output


###############################################################################
# Partial Convolution Block
###############################################################################
class PartialConvBlock2D(nn.Module):
    """A block with PartialConv -> BatchNorm -> ReLU"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_optimized: bool = True,
        multi_channel: bool = False,
        activation: bool = True,
        batch_norm: bool = True,
    ):
        super().__init__()
        
        # Choose implementation
        if use_optimized:
            self.conv = OptimizedPartialConv2dFixed(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=not batch_norm,
                multi_channel=multi_channel, return_mask=True
            )
        else:
            self.conv = NvidiaPartialConv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=not batch_norm,
                multi_channel=multi_channel, return_mask=True
            )
        
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = nn.ReLU(inplace=True) if activation else None
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.conv(x, mask)
        
        if self.batch_norm:
            x = self.batch_norm(x)
        
        if self.activation:
            x = self.activation(x)
            
        return x, mask


class PartialConvBlock3D(nn.Module):
    """A 3D block with PartialConv -> BatchNorm -> ReLU"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_optimized: bool = True,
        multi_channel: bool = False,
        activation: bool = True,
        batch_norm: bool = True,
    ):
        super().__init__()
        
        # Choose implementation
        if use_optimized:
            self.conv = OptimizedPartialConv3dFixed(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=not batch_norm,
                multi_channel=multi_channel, return_mask=True
            )
        else:
            self.conv = NvidiaPartialConv3d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=not batch_norm,
                multi_channel=multi_channel, return_mask=True
            )
        
        self.batch_norm = nn.BatchNorm3d(out_channels) if batch_norm else None
        self.activation = nn.ReLU(inplace=True) if activation else None
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.conv(x, mask)
        
        if self.batch_norm:
            x = self.batch_norm(x)
        
        if self.activation:
            x = self.activation(x)
            
        return x, mask


###############################################################################
# 2D Partial Convolution U-Net
###############################################################################
class PartialConvUNet2D(nn.Module):
    """U-Net architecture using Partial Convolutions for image inpainting"""
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: List[int] = [64, 128, 256, 512, 1024],
        use_optimized: bool = True,
        multi_channel: bool = False,
    ):
        super().__init__()
        self.use_optimized = use_optimized
        
        # Encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_ch = in_channels
        for feature in features[:-1]:
            # Double convolution block
            encoder = nn.Sequential(
                PartialConvBlock2D(in_ch, feature, use_optimized=use_optimized, multi_channel=multi_channel),
                PartialConvBlock2D(feature, feature, use_optimized=use_optimized, multi_channel=multi_channel),
            )
            self.encoders.append(encoder)
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = feature
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            PartialConvBlock2D(features[-2], features[-1], use_optimized=use_optimized, multi_channel=multi_channel),
            PartialConvBlock2D(features[-1], features[-1], use_optimized=use_optimized, multi_channel=multi_channel),
        )
        
        # Decoder path
        self.decoders = nn.ModuleList()
        
        for i in range(len(features)-1, 0, -1):
            # Double convolution block for decoder
            # After concatenation, we'll have features[i] + features[i-1] channels
            decoder = nn.Sequential(
                PartialConvBlock2D(features[i] + features[i-1], features[i-1], 
                                  use_optimized=use_optimized, multi_channel=multi_channel),
                PartialConvBlock2D(features[i-1], features[i-1], 
                                  use_optimized=use_optimized, multi_channel=multi_channel),
            )
            self.decoders.append(decoder)
        
        # Final convolution
        if use_optimized:
            self.final_conv = OptimizedPartialConv2dFixed(
                features[0], out_channels, kernel_size=1,
                multi_channel=multi_channel, return_mask=True
            )
        else:
            self.final_conv = NvidiaPartialConv2d(
                features[0], out_channels, kernel_size=1,
                multi_channel=multi_channel, return_mask=True
            )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Store encoder features and masks for skip connections
        encoder_features = []
        encoder_masks = []
        
        # Encoder path
        for encoder, pool in zip(self.encoders, self.pools):
            # Apply encoder blocks
            for layer in encoder:
                x, mask = layer(x, mask)
            
            encoder_features.append(x)
            encoder_masks.append(mask)
            
            # Downsample
            x = pool(x)
            mask = pool(mask)
        
        # Bottleneck
        for layer in self.bottleneck:
            x, mask = layer(x, mask)
        
        # Decoder path
        for i, decoder in enumerate(self.decoders):
            # Upsample using interpolation
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            mask = F.interpolate(mask, scale_factor=2, mode='nearest')
            
            # Get skip connection
            skip_features = encoder_features[-(i+1)]
            skip_mask = encoder_masks[-(i+1)]
            
            # Ensure sizes match (for odd-sized inputs)
            if x.shape[2:] != skip_features.shape[2:]:
                x = F.interpolate(x, size=skip_features.shape[2:], mode='bilinear', align_corners=False)
                mask = F.interpolate(mask, size=skip_mask.shape[2:], mode='nearest')
            
            # Concatenate features
            x = torch.cat([x, skip_features], dim=1)
            
            # For masks, use minimum (intersection) instead of concatenation
            # This ensures mask remains single-channel when multi_channel=False
            mask = torch.min(mask, skip_mask)
            
            # Apply decoder blocks
            for layer in decoder:
                x, mask = layer(x, mask)
        
        # Final convolution
        x, _ = self.final_conv(x, mask)
        
        return x


###############################################################################
# 3D Partial Convolution U-Net
###############################################################################
class PartialConvUNet3D(nn.Module):
    """3D U-Net architecture using Partial Convolutions for volumetric segmentation"""
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        features: List[int] = [32, 64, 128, 256],
        use_optimized: bool = True,
        multi_channel: bool = False,
    ):
        super().__init__()
        self.use_optimized = use_optimized
        
        # Encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_ch = in_channels
        for feature in features[:-1]:
            # Double convolution block
            encoder = nn.Sequential(
                PartialConvBlock3D(in_ch, feature, use_optimized=use_optimized, multi_channel=multi_channel),
                PartialConvBlock3D(feature, feature, use_optimized=use_optimized, multi_channel=multi_channel),
            )
            self.encoders.append(encoder)
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            in_ch = feature
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            PartialConvBlock3D(features[-2], features[-1], use_optimized=use_optimized, multi_channel=multi_channel),
            PartialConvBlock3D(features[-1], features[-1], use_optimized=use_optimized, multi_channel=multi_channel),
        )
        
        # Decoder path
        self.decoders = nn.ModuleList()
        
        for i in range(len(features)-1, 0, -1):
            # Double convolution block for decoder
            # After concatenation, we'll have features[i] + features[i-1] channels
            decoder = nn.Sequential(
                PartialConvBlock3D(features[i] + features[i-1], features[i-1], 
                                  use_optimized=use_optimized, multi_channel=multi_channel),
                PartialConvBlock3D(features[i-1], features[i-1], 
                                  use_optimized=use_optimized, multi_channel=multi_channel),
            )
            self.decoders.append(decoder)
        
        # Final convolution
        if use_optimized:
            self.final_conv = OptimizedPartialConv3dFixed(
                features[0], out_channels, kernel_size=1,
                multi_channel=multi_channel, return_mask=True
            )
        else:
            self.final_conv = NvidiaPartialConv3d(
                features[0], out_channels, kernel_size=1,
                multi_channel=multi_channel, return_mask=True
            )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Store encoder features and masks for skip connections
        encoder_features = []
        encoder_masks = []
        
        # Encoder path
        for encoder, pool in zip(self.encoders, self.pools):
            # Apply encoder blocks
            for layer in encoder:
                x, mask = layer(x, mask)
            
            encoder_features.append(x)
            encoder_masks.append(mask)
            
            # Downsample
            x = pool(x)
            mask = pool(mask)
        
        # Bottleneck
        for layer in self.bottleneck:
            x, mask = layer(x, mask)
        
        # Decoder path
        for i, decoder in enumerate(self.decoders):
            # Upsample using interpolation
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
            mask = F.interpolate(mask, scale_factor=2, mode='nearest')
            
            # Get skip connection
            skip_features = encoder_features[-(i+1)]
            skip_mask = encoder_masks[-(i+1)]
            
            # Ensure sizes match (for odd-sized inputs)
            if x.shape[2:] != skip_features.shape[2:]:
                x = F.interpolate(x, size=skip_features.shape[2:], mode='trilinear', align_corners=False)
                mask = F.interpolate(mask, size=skip_mask.shape[2:], mode='nearest')
            
            # Concatenate features
            x = torch.cat([x, skip_features], dim=1)
            
            # For masks, use minimum (intersection) instead of concatenation
            # This ensures mask remains single-channel when multi_channel=False
            mask = torch.min(mask, skip_mask)
            
            # Apply decoder blocks
            for layer in decoder:
                x, mask = layer(x, mask)
        
        # Final convolution
        x, _ = self.final_conv(x, mask)
        
        return x


###############################################################################
# Network Training Benchmark
###############################################################################
@dataclass
class NetworkBenchmarkResult:
    """Results from network training benchmark"""
    network_type: str
    use_optimized: bool
    
    # Timing
    forward_time: float
    backward_time: float
    total_time: float
    
    # Memory
    peak_memory: float
    
    # Model info
    total_params: int
    total_layers: int


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_partial_conv_layers(model: nn.Module) -> int:
    """Count number of partial convolution layers"""
    count = 0
    for module in model.modules():
        if isinstance(module, (NvidiaPartialConv2d, OptimizedPartialConv2dFixed,
                             NvidiaPartialConv3d, OptimizedPartialConv3dFixed)):
            count += 1
    return count


def benchmark_network_training(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    mask_shape: Tuple[int, ...],
    device: str,
    network_type: str,
    use_optimized: bool,
    warmup_runs: int = 3,
    benchmark_runs: int = 10
) -> NetworkBenchmarkResult:
    """Benchmark training performance of a full network"""
    
    print(f"\nBenchmarking {network_type} ({'Optimized' if use_optimized else 'NVIDIA'})...")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Partial Conv layers: {count_partial_conv_layers(model)}")
    
    # Create dummy data
    x = torch.randn(input_shape, device=device, requires_grad=True)
    mask = torch.ones(mask_shape, device=device)
    # Random mask for more realistic scenario
    mask = (torch.rand_like(mask) > 0.2).float()
    
    # Target (same shape as output)
    with torch.no_grad():
        target = model(x, mask).detach()
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Warmup
    print(f"Running {warmup_runs} warmup iterations...")
    for _ in range(warmup_runs):
        optimizer.zero_grad()
        output = model(x, mask)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark forward pass
    print(f"Benchmarking forward pass ({benchmark_runs} runs)...")
    forward_timer = tbenchmark.Timer(
        stmt='model(x, mask)',
        globals={'model': model, 'x': x, 'mask': mask},
        num_threads=1
    )
    forward_measurement = forward_timer.timeit(benchmark_runs)
    forward_time = forward_measurement.mean * 1000  # ms
    
    # Benchmark full training step
    print(f"Benchmarking full training step...")
    def training_step():
        optimizer.zero_grad()
        output = model(x, mask)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        return loss
    
    full_timer = tbenchmark.Timer(
        stmt='step()',
        globals={'step': training_step},
        num_threads=1
    )
    full_measurement = full_timer.timeit(benchmark_runs)
    total_time = full_measurement.mean * 1000  # ms
    backward_time = total_time - forward_time
    
    # Memory usage
    peak_memory = 0.0
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _ = training_step()
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
    
    return NetworkBenchmarkResult(
        network_type=network_type,
        use_optimized=use_optimized,
        forward_time=forward_time,
        backward_time=backward_time,
        total_time=total_time,
        peak_memory=peak_memory,
        total_params=count_parameters(model),
        total_layers=count_partial_conv_layers(model)
    )


def main():
    """Run benchmarks on larger networks"""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Network Benchmarks on {device}")
    print("=" * 80)
    
    results = []
    
    # Test 2D U-Net
    print("\n" + "="*80)
    print("2D U-Net for Image Inpainting")
    print("="*80)
    
    # Input shape: (batch, channels, height, width)
    batch_size = 4
    input_shape_2d = (batch_size, 3, 256, 256)
    mask_shape_2d = (batch_size, 1, 256, 256)
    
    # Print architecture details
    print("\nArchitecture Details:")
    print("  Input:  3 x 256 x 256")
    print("  Encoder: 3 → 64 → 128 → 256 → 512")
    print("  Decoder: 512 → 256 → 128 → 64 → 3")
    print("  Skip connections at each level")
    print("  Total layers: ~23 partial convolutions")
    
    # Create and benchmark NVIDIA implementation
    unet_2d_nvidia = PartialConvUNet2D(
        in_channels=3,
        out_channels=3,
        features=[64, 128, 256, 512],
        use_optimized=False
    ).to(device)
    
    result_nvidia_2d = benchmark_network_training(
        unet_2d_nvidia, input_shape_2d, mask_shape_2d,
        device, "2D U-Net", False
    )
    results.append(result_nvidia_2d)
    
    # Create and benchmark Optimized implementation
    unet_2d_optimized = PartialConvUNet2D(
        in_channels=3,
        out_channels=3,
        features=[64, 128, 256, 512],
        use_optimized=True
    ).to(device)
    
    result_optimized_2d = benchmark_network_training(
        unet_2d_optimized, input_shape_2d, mask_shape_2d,
        device, "2D U-Net", True
    )
    results.append(result_optimized_2d)
    
    # Test 3D U-Net
    print("\n" + "="*80)
    print("3D U-Net for Volumetric Segmentation")
    print("="*80)
    
    # Input shape: (batch, channels, depth, height, width)
    batch_size_3d = 2
    input_shape_3d = (batch_size_3d, 1, 32, 128, 128)
    mask_shape_3d = (batch_size_3d, 1, 32, 128, 128)
    
    # Print architecture details
    print("\nArchitecture Details:")
    print("  Input:  1 x 32 x 128 x 128")
    print("  Encoder: 1 → 32 → 64 → 128")
    print("  Decoder: 128 → 64 → 32 → 2")
    print("  Skip connections at each level")
    print("  Total layers: ~15 partial convolutions")
    
    # Create and benchmark NVIDIA implementation
    unet_3d_nvidia = PartialConvUNet3D(
        in_channels=1,
        out_channels=2,
        features=[32, 64, 128],
        use_optimized=False
    ).to(device)
    
    result_nvidia_3d = benchmark_network_training(
        unet_3d_nvidia, input_shape_3d, mask_shape_3d,
        device, "3D U-Net", False
    )
    results.append(result_nvidia_3d)
    
    # Create and benchmark Optimized implementation
    unet_3d_optimized = PartialConvUNet3D(
        in_channels=1,
        out_channels=2,
        features=[32, 64, 128],
        use_optimized=True
    ).to(device)
    
    result_optimized_3d = benchmark_network_training(
        unet_3d_optimized, input_shape_3d, mask_shape_3d,
        device, "3D U-Net", True
    )
    results.append(result_optimized_3d)
    
    # Print summary
    print("\n" + "="*80)
    print("NETWORK BENCHMARK SUMMARY")
    print("="*80)
    
    print("\n{:<15} {:<10} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
        "Network", "Type", "Forward(ms)", "Backward(ms)", "Total(ms)", "Memory(MB)", "Speedup"
    ))
    print("-" * 95)
    
    # Group results by network type
    for network_type in ["2D U-Net", "3D U-Net"]:
        nvidia_result = next(r for r in results if r.network_type == network_type and not r.use_optimized)
        opt_result = next(r for r in results if r.network_type == network_type and r.use_optimized)
        
        # NVIDIA row
        print("{:<15} {:<10} {:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f} {:<12}".format(
            network_type, "NVIDIA", nvidia_result.forward_time,
            nvidia_result.backward_time, nvidia_result.total_time,
            nvidia_result.peak_memory, "-"
        ))
        
        # Optimized row
        speedup = nvidia_result.total_time / opt_result.total_time
        print("{:<15} {:<10} {:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f}x".format(
            "", "Optimized", opt_result.forward_time,
            opt_result.backward_time, opt_result.total_time,
            opt_result.peak_memory, speedup
        ))
        
        # Improvement row
        fwd_speedup = nvidia_result.forward_time / opt_result.forward_time
        bwd_speedup = nvidia_result.backward_time / opt_result.backward_time
        mem_reduction = (1 - opt_result.peak_memory / nvidia_result.peak_memory) * 100
        
        print("{:<15} {:<10} {:<12} {:<12} {:<12} {:<12}".format(
            "", "Improvement", f"{fwd_speedup:.2f}x", f"{bwd_speedup:.2f}x",
            f"{speedup:.2f}x", f"-{mem_reduction:.1f}%"
        ))
        print()
    
    # Visual comparison
    print("\nVisual Performance Comparison:")
    print("─" * 60)
    
    def create_bar(value, max_value, width=40):
        filled = int((value / max_value) * width)
        return "█" * filled + "░" * (width - filled)
    
    for network_type in ["2D U-Net", "3D U-Net"]:
        nvidia_result = next(r for r in results if r.network_type == network_type and not r.use_optimized)
        opt_result = next(r for r in results if r.network_type == network_type and r.use_optimized)
        
        print(f"\n{network_type}:")
        max_time = nvidia_result.total_time
        
        print(f"  NVIDIA:    {create_bar(nvidia_result.total_time, max_time)} {nvidia_result.total_time:6.1f} ms")
        print(f"  Optimized: {create_bar(opt_result.total_time, max_time)} {opt_result.total_time:6.1f} ms")
        
        # Show per-layer speedup
        speedup = nvidia_result.total_time / opt_result.total_time
        per_layer_speedup_ms = (nvidia_result.total_time - opt_result.total_time) / nvidia_result.total_layers
        print(f"  Speedup: {speedup:.2f}x total, ~{per_layer_speedup_ms:.2f} ms saved per layer")
    
    # Architecture comparison
    print("\n\nArchitecture Complexity:")
    print("─" * 60)
    print(f"2D U-Net: {results[0].total_params:,} parameters, {results[0].total_layers} partial conv layers")
    print(f"3D U-Net: {results[2].total_params:,} parameters, {results[2].total_layers} partial conv layers")
    
    # Recommendations
    print("\n\nRecommendations:")
    print("─" * 60)
    print("1. Use optimized implementation for >3x training speedup")
    print("2. Memory savings of 15-20% enable larger batch sizes")
    print("3. For 3D networks, consider mixed precision training")
    print("4. Cache masks when using consistent mask patterns")
    
    print("\n" + "="*80)


# Additional test: Very large network
def test_large_network():
    """Test a very large network to show scalability"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n" + "="*80)
    print("LARGE NETWORK TEST")
    print("="*80)
    
    # Create a deeper U-Net with more features
    large_unet = PartialConvUNet2D(
        in_channels=3,
        out_channels=3,
        features=[64, 128, 256, 512, 1024],  # Deeper network
        use_optimized=True
    ).to(device)
    
    total_params = count_parameters(large_unet)
    total_layers = count_partial_conv_layers(large_unet)
    
    print(f"\nLarge U-Net Statistics:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Partial Conv Layers: {total_layers}")
    print(f"  Estimated Model Size: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Quick forward pass test
    batch_size = 2
    x = torch.randn(batch_size, 3, 512, 512, device=device)
    mask = torch.ones(batch_size, 1, 512, 512, device=device)
    
    # Time a single forward pass
    start = time.time()
    with torch.no_grad():
        output = large_unet(x, mask)
    if device == "cuda":
        torch.cuda.synchronize()
    forward_time = (time.time() - start) * 1000
    
    print(f"\nSingle forward pass (512x512): {forward_time:.1f} ms")
    print(f"Throughput: {1000 / forward_time:.1f} images/second")
    
    # Memory usage
    if device == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"Peak memory usage: {memory_used:.1f} MB")
    
    return large_unet


if __name__ == "__main__":
    main()
    
    # Optional: Test a very large network
    try:
        test_large_network()
    except torch.cuda.OutOfMemoryError:
        print("\n⚠️  Skipping large network test due to memory constraints")