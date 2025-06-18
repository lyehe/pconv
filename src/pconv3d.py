import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-6  # Small constant to avoid division by zero in mask updates


class PConv3d(nn.Conv3d):
    def __init__(
        self,
        *args,
        multi_channel: bool = False,
        cache_masks: bool = True,
        return_mask: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.multi_channel = multi_channel
        self.cache_masks = cache_masks
        self.return_mask = return_mask

        # Calculate sliding window size
        kernel_elements = (
            self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        )
        if self.multi_channel:
            self.slide_winsize = float(
                kernel_elements * (self.in_channels // self.groups)
            )
        else:
            self.slide_winsize = float(kernel_elements)

        # Initialize cache
        if self.cache_masks:
            self._last_mask_shape = None
            self._last_mask_ptr = None
            self._last_result = None

        # Register persistent buffer for bias view
        if self.bias is not None:
            self.register_buffer("_bias_view", self.bias.view(1, -1, 1, 1, 1))

    def _compute_mask_updates(
        self, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Return from cache if possible
        if (
            self.cache_masks
            and self._last_mask_shape == mask.shape
            and self._last_mask_ptr == mask.data_ptr()
        ):
            return self._last_result

        with torch.no_grad():
            # Create weight for sum pooling on-the-fly
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

            # Perform sum pooling
            update_mask = F.conv3d(
                mask_for_sum,
                conv_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=groups_for_mask_conv,
            )

            # Calculate ratio and clamp mask
            mask_ratio = self.slide_winsize / (update_mask + EPSILON)
            update_mask = torch.clamp(update_mask, 0, 1)
            mask_ratio = mask_ratio * update_mask

        # Update cache
        if self.cache_masks:
            self._last_mask_shape = mask.shape
            self._last_mask_ptr = mask.data_ptr()
            self._last_result = (update_mask, mask_ratio)

        return update_mask, mask_ratio

    def forward(
        self, input_tensor: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = torch.ones(
                input_tensor.shape[0],
                1,
                *input_tensor.shape[2:],
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )

        # Handle mask expansion for input multiplication
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

        # Main convolution without bias
        output = F.conv3d(
            input_tensor * current_mask_for_input_mult,
            self.weight,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        # Apply partial conv formula with in-place operations
        if self.bias is not None:
            output.mul_(mask_ratio)
            output.add_(self._bias_view)
            output.mul_(update_mask)
        else:
            output.mul_(mask_ratio)

        if self.return_mask:
            return output, update_mask

        return output

    def clear_cache(self):
        if self.cache_masks:
            self._last_mask_shape = None
            self._last_mask_ptr = None
            self._last_result = None
