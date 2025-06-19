from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as tbenchmark

EPSILON = 1e-6


###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################


class NvidiaPartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        # whether the mask is multi-channel or not
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
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
            )
        else:
            self.weight_maskUpdater = torch.ones(
                1, 1, self.kernel_size[0], self.kernel_size[1]
            )

        self.slide_winsize = (
            self.weight_maskUpdater.shape[1]
            * self.weight_maskUpdater.shape[2]
            * self.weight_maskUpdater.shape[3]
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
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(
                            input.data.shape[0],
                            input.data.shape[1],
                            input.data.shape[2],
                            input.data.shape[3],
                        ).to(input)
                    else:
                        mask = torch.ones(
                            1, 1, input.data.shape[2], input.data.shape[3]
                        ).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(
                    mask,
                    self.weight_maskUpdater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=1,
                )

                # for mixed precision training, change 1e-6 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-6)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-6)
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


###############################################################################
# Optimized Implementation (from original code)
###############################################################################
class OptimizedPartialConv2d(nn.Conv2d):
    def __init__(
        self,
        *args,
        multi_channel: bool = False,
        return_mask: bool = False,
        cache_masks: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.multi_channel = multi_channel
        self.return_mask = return_mask
        self.cache_masks = cache_masks

        kernel_elements = self.kernel_size[0] * self.kernel_size[1]
        if self.multi_channel:
            self.slide_winsize = float(
                kernel_elements * (self.in_channels // self.groups)
            )
        else:
            self.slide_winsize = float(kernel_elements * 1)

        if cache_masks:
            self._last_mask_shape = None
            self._last_mask_ptr = None
            self._last_result = None

        # Don't register bias_view as a buffer - compute it dynamically!

    def _compute_mask_updates(
        self, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
                        1,
                        self.in_channels,
                        *self.kernel_size,
                        device=mask.device,
                        dtype=mask.dtype,
                    )
                    groups_for_mask_conv = 1
                else:
                    channels_per_group = self.in_channels // self.groups
                    conv_weight = torch.ones(
                        self.groups,
                        channels_per_group,
                        *self.kernel_size,
                        device=mask.device,
                        dtype=mask.dtype,
                    )
                    groups_for_mask_conv = self.groups

            update_mask = F.conv2d(
                mask_for_sum,
                conv_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=groups_for_mask_conv,
            )

            # For mixed precision training, consider using 1e-6 instead of 1e-6
            mask_ratio = self.slide_winsize / (update_mask + 1e-6)
            update_mask = torch.clamp(update_mask, 0, 1)
            mask_ratio = mask_ratio * update_mask

        if self.cache_masks:
            self._last_mask_shape = mask.shape
            self._last_mask_ptr = mask.data_ptr()
            self._last_result = (update_mask, mask_ratio)

        return update_mask, mask_ratio

    def forward(
        self, input_tensor: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        input_mask_for_calc = mask
        if input_mask_for_calc is None:
            input_mask_for_calc = torch.ones(
                input_tensor.shape[0],
                1,
                *input_tensor.shape[2:],
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )

        update_mask, mask_ratio = self._compute_mask_updates(input_mask_for_calc)

        # Determine mask for element-wise multiplication with input_tensor
        current_mask_for_input_mult = input_mask_for_calc
        if (
            self.multi_channel
            and input_mask_for_calc.shape[1] == 1
            and input_tensor.shape[1] != 1
        ):
            current_mask_for_input_mult = input_mask_for_calc.expand(
                -1, input_tensor.shape[1], -1, -1
            )
        elif not self.multi_channel and input_mask_for_calc.shape[1] != 1:
            current_mask_for_input_mult = input_mask_for_calc[:, 0:1, ...].expand(
                -1, input_tensor.shape[1], -1, -1
            )
        elif (
            not self.multi_channel
            and input_mask_for_calc.shape[1] == 1
            and input_tensor.shape[1] != 1
        ):
            current_mask_for_input_mult = input_mask_for_calc.expand(
                -1, input_tensor.shape[1], -1, -1
            )

        masked_input = input_tensor * current_mask_for_input_mult

        output = F.conv2d(
            masked_input,
            self.weight,
            bias=None,  # Bias handled after scaling
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        if self.bias is not None:
            # CRITICAL FIX: Compute bias view dynamically for proper gradient flow
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = output * mask_ratio + bias_view
            output = output * update_mask
        else:
            output = output * mask_ratio

        if self.return_mask:
            return output, update_mask
        return output

    def clear_cache(self):
        if self.cache_masks:
            self._last_mask_shape = None
            self._last_mask_ptr = None
            self._last_result = None


###############################################################################
# Multilayer Network Implementations
###############################################################################
class NvidiaMultilayerPConv(nn.Module):
    """Sequential multilayer partial convolution network using NVIDIA implementation"""

    def __init__(self, layer_configs: list[dict], activation: str = "relu"):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation

        for i, config in enumerate(layer_configs):
            # All layers except the last one return masks
            layer_config = config.copy()
            layer_config["return_mask"] = True
            layer_config["multi_channel"] = layer_config.get("multi_channel", False)

            layer = NvidiaPartialConv2d(**layer_config)
            self.layers.append(layer)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        current_mask = mask

        for i, layer in enumerate(self.layers):
            x, current_mask = layer(x, current_mask)

            # Apply activation (except for last layer in some cases)
            if i < len(self.layers) - 1:  # Not the last layer
                if self.activation == "relu":
                    x = F.relu(x)
                elif self.activation == "leaky_relu":
                    x = F.leaky_relu(x, 0.2)
                elif self.activation == "gelu":
                    x = F.gelu(x)

        return x, current_mask

    def clear_cache(self):
        """Clear internal caches for all layers"""
        for layer in self.layers:
            layer.update_mask = None
            layer.last_size = (None, None, None, None)


class OptimizedMultilayerPConv(nn.Module):
    """Sequential multilayer partial convolution network using optimized implementation"""

    def __init__(self, layer_configs: list[dict], activation: str = "relu"):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation

        for i, config in enumerate(layer_configs):
            # All layers except the last one return masks
            layer_config = config.copy()
            layer_config["return_mask"] = True
            layer_config["multi_channel"] = layer_config.get("multi_channel", False)
            layer_config["cache_masks"] = layer_config.get("cache_masks", True)

            layer = OptimizedPartialConv2d(**layer_config)
            self.layers.append(layer)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        current_mask = mask

        for i, layer in enumerate(self.layers):
            x, current_mask = layer(x, current_mask)

            # Apply activation (except for last layer in some cases)
            if i < len(self.layers) - 1:  # Not the last layer
                if self.activation == "relu":
                    x = F.relu(x)
                elif self.activation == "leaky_relu":
                    x = F.leaky_relu(x, 0.2)
                elif self.activation == "gelu":
                    x = F.gelu(x)

        return x, current_mask

    def clear_cache(self):
        """Clear caches for all layers"""
        for layer in self.layers:
            if hasattr(layer, "clear_cache"):
                layer.clear_cache()


class EncoderDecoderPConv(nn.Module):
    """U-Net style encoder-decoder with partial convolutions"""

    def __init__(self, layer_type: str = "nvidia", base_channels: int = 32):
        super().__init__()
        self.layer_type = layer_type

        # Choose layer class
        if layer_type == "nvidia":
            conv_class = NvidiaPartialConv2d
        else:
            conv_class = OptimizedPartialConv2d

        # Encoder layers (downsampling)
        self.enc1 = conv_class(
            3, base_channels, 3, padding=1, return_mask=True, multi_channel=False
        )
        self.enc2 = conv_class(
            base_channels,
            base_channels * 2,
            3,
            stride=2,
            padding=1,
            return_mask=True,
            multi_channel=False,
        )
        self.enc3 = conv_class(
            base_channels * 2,
            base_channels * 4,
            3,
            stride=2,
            padding=1,
            return_mask=True,
            multi_channel=False,
        )

        # Bottleneck
        self.bottleneck = conv_class(
            base_channels * 4,
            base_channels * 8,
            3,
            padding=1,
            return_mask=True,
            multi_channel=False,
        )

        # Decoder layers (upsampling)
        self.dec3 = conv_class(
            base_channels * 8,
            base_channels * 4,
            3,
            padding=1,
            return_mask=True,
            multi_channel=False,
        )
        self.dec2 = conv_class(
            base_channels * 4,
            base_channels * 2,
            3,
            padding=1,
            return_mask=True,
            multi_channel=False,
        )
        self.dec1 = conv_class(
            base_channels * 2,
            base_channels,
            3,
            padding=1,
            return_mask=True,
            multi_channel=False,
        )

        # Final output layer
        self.output = conv_class(
            base_channels, 3, 3, padding=1, return_mask=False, multi_channel=False
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # Encoder
        x1, m1 = self.enc1(x, mask)
        x1 = F.relu(x1)

        x2, m2 = self.enc2(x1, m1)
        x2 = F.relu(x2)

        x3, m3 = self.enc3(x2, m2)
        x3 = F.relu(x3)

        # Bottleneck
        x_bottle, m_bottle = self.bottleneck(x3, m3)
        x_bottle = F.relu(x_bottle)

        # Decoder
        x_up3 = F.interpolate(x_bottle, scale_factor=2, mode="nearest")
        m_up3 = F.interpolate(m_bottle.float(), scale_factor=2, mode="nearest")
        x_dec3, m_dec3 = self.dec3(x_up3, m_up3)
        x_dec3 = F.relu(x_dec3)

        x_up2 = F.interpolate(x_dec3, scale_factor=2, mode="nearest")
        m_up2 = F.interpolate(m_dec3.float(), scale_factor=2, mode="nearest")
        x_dec2, m_dec2 = self.dec2(x_up2, m_up2)
        x_dec2 = F.relu(x_dec2)

        x_dec1, m_dec1 = self.dec1(x_dec2, m_dec2)
        x_dec1 = F.relu(x_dec1)

        # Final output
        output = self.output(x_dec1, m_dec1)

        return output, m_dec1

    def clear_cache(self):
        """Clear caches for all layers"""
        for module in self.modules():
            if isinstance(module, (NvidiaPartialConv2d, OptimizedPartialConv2d)):
                if hasattr(module, "clear_cache"):
                    module.clear_cache()
                elif hasattr(module, "update_mask"):
                    module.update_mask = None
                    module.last_size = (None, None, None, None)


###############################################################################
# Multilayer Benchmark Suite
###############################################################################
@dataclass
class MultilayerBenchmarkResult:
    test_name: str
    nvidia_time: float
    optimized_time: float
    speedup: float
    outputs_match: bool
    max_diff: float
    memory_nvidia: float = 0.0
    memory_optimized: float = 0.0
    num_layers: int = 0
    total_params: int = 0


class MultilayerBenchmark:
    def __init__(
        self,
        device="cuda" if torch.cuda.is_available() else "cpu",
        warmup_runs=3,
        benchmark_measurements=10,
    ):
        self.device = device
        self.warmup_runs = warmup_runs
        self.benchmark_measurements = benchmark_measurements
        self.results = []

        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def create_multilayer_networks(
        self, layer_configs: list[dict], activation: str = "relu"
    ):
        """Create both NVIDIA and optimized multilayer networks"""
        nvidia_net = (
            NvidiaMultilayerPConv(layer_configs, activation).to(self.device).eval()
        )
        optimized_net = (
            OptimizedMultilayerPConv(layer_configs, activation).to(self.device).eval()
        )

        # Copy weights from NVIDIA to optimized network
        with torch.no_grad():
            for nvidia_layer, opt_layer in zip(nvidia_net.layers, optimized_net.layers):
                opt_layer.weight.data.copy_(nvidia_layer.weight.data)
                if nvidia_layer.bias is not None:
                    opt_layer.bias.data.copy_(nvidia_layer.bias.data)
                    if hasattr(opt_layer, "_bias_view"):
                        opt_layer._bias_view.copy_(nvidia_layer.bias.view(1, -1, 1, 1))

        return nvidia_net, optimized_net

    def create_encoder_decoder_networks(self, base_channels: int = 32):
        """Create encoder-decoder networks"""
        nvidia_net = EncoderDecoderPConv("nvidia", base_channels).to(self.device).eval()
        optimized_net = (
            EncoderDecoderPConv("optimized", base_channels).to(self.device).eval()
        )

        # Copy weights
        with torch.no_grad():
            nvidia_modules = [
                m for m in nvidia_net.modules() if isinstance(m, NvidiaPartialConv2d)
            ]
            opt_modules = [
                m
                for m in optimized_net.modules()
                if isinstance(m, OptimizedPartialConv2d)
            ]

            for nvidia_layer, opt_layer in zip(nvidia_modules, opt_modules):
                opt_layer.weight.data.copy_(nvidia_layer.weight.data)
                if nvidia_layer.bias is not None:
                    opt_layer.bias.data.copy_(nvidia_layer.bias.data)
                    if hasattr(opt_layer, "_bias_view"):
                        opt_layer._bias_view.copy_(nvidia_layer.bias.view(1, -1, 1, 1))

        return nvidia_net, optimized_net

    def count_parameters(self, model):
        """Count total parameters in model"""
        return sum(p.numel() for p in model.parameters())

    def measure_memory(self, model_callable, input_tensor, mask_arg):
        """Measure peak memory usage"""
        if self.device != "cuda":
            return 0.0
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        _ = model_callable(input_tensor, mask_arg)
        torch.cuda.synchronize(self.device)
        peak_memory = torch.cuda.max_memory_allocated(self.device) / 1e6  # MB
        return peak_memory

    def benchmark_multilayer_config(self, config: dict):
        """Benchmark a multilayer configuration"""
        print(f"--- Testing {config['name']} ---")

        if config["type"] == "sequential":
            nvidia_net, optimized_net = self.create_multilayer_networks(
                config["layer_configs"], config.get("activation", "relu")
            )
            num_layers = len(config["layer_configs"])
        elif config["type"] == "encoder_decoder":
            nvidia_net, optimized_net = self.create_encoder_decoder_networks(
                config.get("base_channels", 32)
            )
            num_layers = sum(
                1 for _ in nvidia_net.modules() if isinstance(_, NvidiaPartialConv2d)
            )
        else:
            raise ValueError(f"Unknown network type: {config['type']}")

        input_shape = config["input_shape"]
        input_tensor = torch.randn(input_shape, device=self.device)

        # Create mask based on configuration
        mask_input_arg = None
        if config.get("mask_type") == "random":
            mask_input_arg = (
                torch.rand(input_shape[0], 1, *input_shape[2:], device=self.device)
                > 0.3
            ).float()
        elif config.get("mask_type") == "center_hole":
            mask_input_arg = torch.ones(
                input_shape[0], 1, *input_shape[2:], device=self.device
            )
            h, w = input_shape[2], input_shape[3]
            mask_input_arg[:, :, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 0
        elif config.get("mask_type") != "none":
            mask_input_arg = torch.ones(
                input_shape[0], 1, *input_shape[2:], device=self.device
            )

        # Warmup
        for _ in range(self.warmup_runs):
            _ = nvidia_net(input_tensor, mask_input_arg)
            _ = optimized_net(input_tensor, mask_input_arg)
        if self.device == "cuda":
            torch.cuda.synchronize()

        # Clear caches
        nvidia_net.clear_cache()
        optimized_net.clear_cache()

        # Equivalence check
        print("1. Equivalence:")
        with torch.no_grad():
            out_nvidia, _ = nvidia_net(input_tensor, mask_input_arg)
            out_optimized, _ = optimized_net(input_tensor, mask_input_arg)

            out_nvidia = out_nvidia.to("cpu")
            out_optimized = out_optimized.to("cpu")

        are_close = torch.allclose(out_nvidia, out_optimized, rtol=1e-4, atol=1e-4)
        max_diff = (
            (out_nvidia - out_optimized).abs().max().item() if not are_close else 0.0
        )
        print(
            f"   - Outputs Match: {'✓' if are_close else '✗'} (Max Diff: {max_diff:.2e})"
        )

        # Speed benchmark
        print(f"\n2. Speed (avg over {self.benchmark_measurements} measurements):")

        timer_nvidia = tbenchmark.Timer(
            stmt="model(inp, mask)",
            globals={"model": nvidia_net, "inp": input_tensor, "mask": mask_input_arg},
            num_threads=1,
        )
        nvidia_measurement = timer_nvidia.timeit(self.benchmark_measurements)
        nvidia_time = nvidia_measurement.mean * 1000

        timer_optimized = tbenchmark.Timer(
            stmt="model(inp, mask)",
            globals={
                "model": optimized_net,
                "inp": input_tensor,
                "mask": mask_input_arg,
            },
            num_threads=1,
        )
        optimized_measurement = timer_optimized.timeit(self.benchmark_measurements)
        optimized_time = optimized_measurement.mean * 1000

        speedup = nvidia_time / optimized_time if optimized_time > 0 else float("inf")

        print(f"   - NVIDIA Network:    {nvidia_time:10.3f} ms")
        print(f"   - Optimized Network: {optimized_time:10.3f} ms")
        print(f"   - Speedup:           {speedup:10.2f}x")
        print(f"   - Layers:            {num_layers:10d}")

        total_params = self.count_parameters(nvidia_net)
        print(f"   - Parameters:        {total_params:10,d}")

        # Memory benchmark
        mem_nvidia = self.measure_memory(nvidia_net, input_tensor, mask_input_arg)
        mem_optimized = self.measure_memory(optimized_net, input_tensor, mask_input_arg)

        print("\n3. Peak Memory Usage:")
        print(f"   - NVIDIA Network:    {mem_nvidia:10.2f} MB")
        print(f"   - Optimized Network: {mem_optimized:10.2f} MB")
        if mem_nvidia > 0:
            mem_reduction = (1 - mem_optimized / mem_nvidia) * 100
            print(f"   - Reduction:         {mem_reduction:10.2f}%")

        print("=" * 60)

        result = MultilayerBenchmarkResult(
            config["name"],
            nvidia_time,
            optimized_time,
            speedup,
            are_close,
            max_diff,
            mem_nvidia,
            mem_optimized,
            num_layers,
            total_params,
        )
        self.results.append(result)
        return result

    def run_benchmark(self):
        """Run the complete multilayer benchmark suite"""
        print(f"Running Multilayer Partial Convolution Benchmark on {self.device}")
        print("=" * 80)

        configurations = [
            # Sequential networks with different depths
            {
                "name": "2-Layer Sequential",
                "type": "sequential",
                "layer_configs": [
                    {
                        "in_channels": 32,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 64,
                        "out_channels": 32,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                ],
                "input_shape": (8, 32, 128, 128),
                "mask_type": "full",
                "activation": "relu",
            },
            {
                "name": "3-Layer Sequential",
                "type": "sequential",
                "layer_configs": [
                    {
                        "in_channels": 32,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 64,
                        "out_channels": 128,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 128,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                ],
                "input_shape": (8, 32, 128, 128),
                "mask_type": "random",
                "activation": "relu",
            },
            {
                "name": "5-Layer Deep Network",
                "type": "sequential",
                "layer_configs": [
                    {
                        "in_channels": 16,
                        "out_channels": 32,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 32,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 64,
                        "out_channels": 128,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 128,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 64,
                        "out_channels": 32,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                ],
                "input_shape": (4, 16, 256, 256),
                "mask_type": "center_hole",
                "activation": "relu",
            },
            {
                "name": "10-Layer Very Deep",
                "type": "sequential",
                "layer_configs": [
                    {
                        "in_channels": 8,
                        "out_channels": 16,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 16,
                        "out_channels": 32,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 32,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 64,
                        "out_channels": 128,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 128,
                        "out_channels": 256,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 256,
                        "out_channels": 128,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 128,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 64,
                        "out_channels": 32,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 32,
                        "out_channels": 16,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 16,
                        "out_channels": 8,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                ],
                "input_shape": (2, 8, 128, 128),
                "mask_type": "random",
                "activation": "relu",
            },
            # Strided convolutions (downsampling)
            {
                "name": "Strided Encoder",
                "type": "sequential",
                "layer_configs": [
                    {
                        "in_channels": 32,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 64,
                        "out_channels": 128,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 128,
                        "out_channels": 256,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                        "bias": True,
                    },
                ],
                "input_shape": (4, 32, 256, 256),
                "mask_type": "random",
                "activation": "relu",
            },
            # Grouped convolutions
            {
                "name": "Grouped 3-Layer",
                "type": "sequential",
                "layer_configs": [
                    {
                        "in_channels": 32,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "padding": 1,
                        "groups": 4,
                        "bias": True,
                    },
                    {
                        "in_channels": 64,
                        "out_channels": 128,
                        "kernel_size": 3,
                        "padding": 1,
                        "groups": 8,
                        "bias": True,
                    },
                    {
                        "in_channels": 128,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "padding": 1,
                        "groups": 4,
                        "bias": True,
                    },
                ],
                "input_shape": (8, 32, 128, 128),
                "mask_type": "center_hole",
                "activation": "relu",
            },
            # Encoder-Decoder architectures
            {
                "name": "Small U-Net",
                "type": "encoder_decoder",
                "base_channels": 16,
                "input_shape": (2, 3, 128, 128),
                "mask_type": "random",
            },
            {
                "name": "Medium U-Net",
                "type": "encoder_decoder",
                "base_channels": 32,
                "input_shape": (2, 3, 256, 256),
                "mask_type": "center_hole",
            },
            # Different activations
            {
                "name": "3-Layer with LeakyReLU",
                "type": "sequential",
                "layer_configs": [
                    {
                        "in_channels": 32,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 64,
                        "out_channels": 128,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 128,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                ],
                "input_shape": (8, 32, 128, 128),
                "mask_type": "random",
                "activation": "leaky_relu",
            },
            {
                "name": "3-Layer with GELU",
                "type": "sequential",
                "layer_configs": [
                    {
                        "in_channels": 32,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 64,
                        "out_channels": 128,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 128,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                ],
                "input_shape": (8, 32, 128, 128),
                "mask_type": "random",
                "activation": "gelu",
            },
            # High-resolution test
            {
                "name": "High-Res 2-Layer",
                "type": "sequential",
                "layer_configs": [
                    {
                        "in_channels": 16,
                        "out_channels": 32,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                    {
                        "in_channels": 32,
                        "out_channels": 16,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": True,
                    },
                ],
                "input_shape": (2, 16, 512, 512),
                "mask_type": "center_hole",
                "activation": "relu",
            },
        ]

        for config in configurations:
            try:
                self.benchmark_multilayer_config(config)
            except Exception as e:
                print(f"Error testing {config['name']}: {e}")
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                continue

        self.print_summary()

    def print_summary(self):
        """Print comprehensive benchmark summary"""
        if not self.results:
            print("No benchmark results to summarize.")
            return

        print(f"\n{'=' * 100}")
        print("MULTILAYER PARTIAL CONVOLUTION BENCHMARK SUMMARY")
        print(f"{'=' * 100}\n")

        # Statistics
        speedups = [r.speedup for r in self.results if np.isfinite(r.speedup)]
        if speedups:
            print("Speed Performance:")
            print(f"  Average Speedup: {np.mean(speedups):.2f}x")
            print(f"  Maximum Speedup: {np.max(speedups):.2f}x")
            print(f"  Minimum Speedup: {np.min(speedups):.2f}x")
            print(f"  Speedup Std Dev: {np.std(speedups):.2f}x\n")

        # Memory statistics
        if self.device == "cuda":
            mem_reductions = []
            for r in self.results:
                if r.memory_nvidia > 0:
                    reduction = (1 - r.memory_optimized / r.memory_nvidia) * 100
                    mem_reductions.append(reduction)

            if mem_reductions:
                print("Memory Performance:")
                print(f"  Average Memory Reduction: {np.mean(mem_reductions):.1f}%")
                print(f"  Maximum Memory Reduction: {np.max(mem_reductions):.1f}%")
                print(f"  Minimum Memory Reduction: {np.min(mem_reductions):.1f}%\n")

        # Detailed results table
        print("Detailed Results:")
        print(
            f"{'Test Name':<25} {'Layers':<7} {'Params':<10} {'NVIDIA(ms)':<12} {'Opt(ms)':<10} {'Speedup':<9} {'Match':<6} {'Mem Reduction':<12}"
        )
        print("-" * 110)

        for r in self.results:
            test_name = r.test_name[:24]  # Truncate long names
            params_str = (
                f"{r.total_params/1000:.0f}K"
                if r.total_params < 1000000
                else f"{r.total_params/1000000:.1f}M"
            )
            speedup_str = f"{r.speedup:.1f}x" if np.isfinite(r.speedup) else "inf"
            match_str = "✓" if r.outputs_match else "✗"

            if r.memory_nvidia > 0:
                mem_reduction = (1 - r.memory_optimized / r.memory_nvidia) * 100
                mem_str = f"{mem_reduction:.1f}%"
            else:
                mem_str = "N/A"

            print(
                f"{test_name:<25} {r.num_layers:<7} {params_str:<10} {r.nvidia_time:<12.1f} {r.optimized_time:<10.1f} {speedup_str:<9} {match_str:<6} {mem_str:<12}"
            )

        # Analysis by network depth
        print(f"\n{'=' * 60}")
        print("Analysis by Network Depth:")
        print(f"{'=' * 60}")

        depth_analysis = {}
        for r in self.results:
            if r.num_layers not in depth_analysis:
                depth_analysis[r.num_layers] = []
            depth_analysis[r.num_layers].append(r.speedup)

        for depth in sorted(depth_analysis.keys()):
            speedups = [s for s in depth_analysis[depth] if np.isfinite(s)]
            if speedups:
                avg_speedup = np.mean(speedups)
                print(
                    f"{depth:2d} layers: {avg_speedup:.2f}x average speedup ({len(speedups)} tests)"
                )

        # Accuracy summary
        all_match = all(r.outputs_match for r in self.results)
        print(
            f"\nAccuracy: {'✓' if all_match else '✗'} All outputs match within tolerance"
        )

        if not all_match:
            print("Tests with accuracy issues:")
            for r in self.results:
                if not r.outputs_match:
                    print(f"  - {r.test_name}: Max difference = {r.max_diff:.2e}")


def main():
    """Run the multilayer benchmark"""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    benchmark = MultilayerBenchmark(
        device="cuda" if torch.cuda.is_available() else "cpu",
        warmup_runs=3,
        benchmark_measurements=10,
    )
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()
