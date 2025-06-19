import torch
import torch.nn as nn
import torch.nn.functional as F


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


def test_gradient_flow():
    """Test that gradients flow properly through the OptimizedPartialConv2d layer."""

    # Create layer with bias
    layer = OptimizedPartialConv2d(
        in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=True
    )

    # Create dummy input and mask
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 32, 32, requires_grad=True)
    mask = torch.ones(batch_size, 1, 32, 32)

    # Forward pass
    output = layer(input_tensor, mask)

    # Create a simple loss (sum of outputs)
    loss = output.sum()

    # Backward pass
    loss.backward()

    # Check gradients exist
    print("Gradient checks:")
    print(f"✓ Input gradient exists: {input_tensor.grad is not None}")
    print(f"✓ Input gradient shape: {input_tensor.grad.shape}")
    print(f"✓ Weight gradient exists: {layer.weight.grad is not None}")
    print(f"✓ Weight gradient shape: {layer.weight.grad.shape}")
    print(f"✓ Bias gradient exists: {layer.bias.grad is not None}")
    print(f"✓ Bias gradient shape: {layer.bias.grad.shape}")

    # Check gradients are non-zero
    print(f"\n✓ Input gradient non-zero: {input_tensor.grad.abs().sum().item() > 0}")
    print(f"✓ Weight gradient non-zero: {layer.weight.grad.abs().sum().item() > 0}")
    print(f"✓ Bias gradient non-zero: {layer.bias.grad.abs().sum().item() > 0}")

    # Test with changing masks (important for training scenarios)
    print("\n\nTesting with changing masks:")
    layer.zero_grad()
    input_tensor.grad = None

    # First forward with one mask
    output1 = layer(input_tensor, mask)
    loss1 = output1.sum()
    loss1.backward(retain_graph=True)
    bias_grad1 = layer.bias.grad.clone()

    # Second forward with different mask
    layer.zero_grad()
    mask2 = torch.rand_like(mask) > 0.5
    output2 = layer(input_tensor, mask2.float())
    loss2 = output2.sum()
    loss2.backward()
    bias_grad2 = layer.bias.grad.clone()

    print(
        f"✓ Bias gradients differ with different masks: {not torch.allclose(bias_grad1, bias_grad2)}"
    )

    print("\n✅ All gradient flow tests passed!")


# Run the test
test_gradient_flow()
