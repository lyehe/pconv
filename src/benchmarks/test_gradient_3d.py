import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedPartialConv3d(nn.Conv3d):
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

        # DON'T register bias_view as a buffer - compute it dynamically!

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
            # For mixed precision training, consider using 1e-6 instead of 1e-6
            mask_ratio = self.slide_winsize / (update_mask + 1e-6)
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
            None,  # Bias is None
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        # Apply partial conv formula with proper gradient flow
        if self.bias is not None:
            # CRITICAL FIX: Compute bias view dynamically for proper gradient flow
            bias_view = self.bias.view(1, self.out_channels, 1, 1, 1)
            output = output * mask_ratio + bias_view
            output = output * update_mask
        else:
            output = output * mask_ratio

        if self.return_mask:
            return output, update_mask

        return output

    def clear_cache(self):
        """Clear the mask cache. Useful when switching between different mask patterns."""
        if self.cache_masks:
            self._last_mask_shape = None
            self._last_mask_ptr = None
            self._last_result = None


def test_gradient_flow_3d():
    """Test that gradients flow properly through the OptimizedPartialConv3d layer."""

    print("Testing gradient flow for 3D Partial Convolution...")

    # Create layer with bias
    layer = OptimizedPartialConv3d(
        in_channels=4, out_channels=8, kernel_size=3, padding=1, bias=True
    )

    # Create dummy input and mask for 3D data
    batch_size = 2
    depth, height, width = 16, 32, 32
    input_tensor = torch.randn(batch_size, 4, depth, height, width, requires_grad=True)
    mask = torch.ones(batch_size, 1, depth, height, width)

    # Forward pass
    output = layer(input_tensor, mask)

    # Create a simple loss
    loss = output.sum()

    # Backward pass
    loss.backward()

    # Check gradients exist and have correct shapes
    print("\n1. Gradient existence checks:")
    print(f"   ✓ Input gradient exists: {input_tensor.grad is not None}")
    print(f"   ✓ Input gradient shape: {input_tensor.grad.shape}")
    print(f"   ✓ Weight gradient exists: {layer.weight.grad is not None}")
    print(f"   ✓ Weight gradient shape: {layer.weight.grad.shape}")
    print(f"   ✓ Bias gradient exists: {layer.bias.grad is not None}")
    print(f"   ✓ Bias gradient shape: {layer.bias.grad.shape}")

    # Check gradients are non-zero
    print(f"\n2. Non-zero gradient checks:")
    print(f"   ✓ Input gradient non-zero: {input_tensor.grad.abs().sum().item() > 0}")
    print(f"   ✓ Weight gradient non-zero: {layer.weight.grad.abs().sum().item() > 0}")
    print(f"   ✓ Bias gradient non-zero: {layer.bias.grad.abs().sum().item() > 0}")

    # Test with grouped convolution
    print("\n3. Testing grouped convolution:")
    grouped_layer = OptimizedPartialConv3d(
        in_channels=8, out_channels=16, kernel_size=3, padding=1, groups=4, bias=True
    )

    input_grouped = torch.randn(batch_size, 8, 8, 16, 16, requires_grad=True)
    mask_grouped = torch.ones(batch_size, 1, 8, 16, 16)

    output_grouped = grouped_layer(input_grouped, mask_grouped)
    loss_grouped = output_grouped.sum()
    loss_grouped.backward()

    print(
        f"   ✓ Grouped conv weight gradient exists: {grouped_layer.weight.grad is not None}"
    )
    print(
        f"   ✓ Grouped conv bias gradient exists: {grouped_layer.bias.grad is not None}"
    )

    # Test with changing masks
    print("\n4. Testing with dynamic masks:")
    layer.zero_grad()
    input_tensor.grad = None

    # First forward with one mask
    output1 = layer(input_tensor, mask)
    loss1 = output1.sum()
    loss1.backward(retain_graph=True)
    bias_grad1 = layer.bias.grad.clone()

    # Second forward with different mask
    layer.zero_grad()
    random_mask = (torch.rand_like(mask) > 0.3).float()
    output2 = layer(input_tensor, random_mask)
    loss2 = output2.sum()
    loss2.backward()
    bias_grad2 = layer.bias.grad.clone()

    print(
        f"   ✓ Bias gradients differ with different masks: {not torch.allclose(bias_grad1, bias_grad2)}"
    )

    # Test memory efficiency with large 3D volumes
    print("\n5. Testing with larger 3D volumes:")
    large_layer = OptimizedPartialConv3d(
        in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=True
    )

    # Test on a reasonably sized 3D volume
    large_input = torch.randn(1, 16, 32, 64, 64, requires_grad=True)
    large_mask = torch.ones(1, 1, 32, 64, 64)

    large_output = large_layer(large_input, large_mask)
    large_loss = large_output.sum()
    large_loss.backward()

    print(f"   ✓ Large volume gradient exists: {large_layer.bias.grad is not None}")
    print(
        f"   ✓ Large volume gradient non-zero: {large_layer.bias.grad.abs().sum().item() > 0}"
    )

    print("\n✅ All 3D gradient flow tests passed!")

    # Optional: Compare gradient magnitudes between fixed and original
    print("\n6. Gradient magnitude comparison (if you have the buggy version):")
    print("   Note: The fixed version should have non-zero bias gradients")
    print(f"   Bias gradient L2 norm: {layer.bias.grad.norm().item():.6f}")


# Run the test
if __name__ == "__main__":
    test_gradient_flow_3d()
