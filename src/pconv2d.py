import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-6  # Small constant to avoid division by zero in mask updates


class PConv2d(nn.Conv2d):
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

        if self.bias is not None:
            self.register_buffer("_bias_view", self.bias.view(1, -1, 1, 1))

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

            mask_ratio = self.slide_winsize / (update_mask + EPSILON)
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
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

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
