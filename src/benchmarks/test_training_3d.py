import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as tbenchmark
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

# Constants
EPSILON = 1e-6


###############################################################################
# NVIDIA Original Implementation for 3D
###############################################################################
class NvidiaPartialConv3d(nn.Conv3d):
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

        super(NvidiaPartialConv3d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
                self.kernel_size[2],
            )
        else:
            self.weight_maskUpdater = torch.ones(
                1, 1, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]
            )

        self.slide_winsize = (
            self.weight_maskUpdater.shape[1]
            * self.weight_maskUpdater.shape[2]
            * self.weight_maskUpdater.shape[3]
            * self.weight_maskUpdater.shape[4]
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
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(
                            input.data.shape[0],
                            input.data.shape[1],
                            input.data.shape[2],
                            input.data.shape[3],
                            input.data.shape[4],
                        ).to(input)
                    else:
                        mask = torch.ones(
                            1,
                            1,
                            input.data.shape[2],
                            input.data.shape[3],
                            input.data.shape[4],
                        ).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv3d(
                    mask,
                    self.weight_maskUpdater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=1,
                )

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-6)
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


###############################################################################
# Fixed Optimized Implementation for 3D Training
###############################################################################
class OptimizedPartialConv3dFixed(nn.Conv3d):
    """Fixed version with proper gradient flow for bias"""

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


###############################################################################
# Training Benchmark for 3D
###############################################################################
@dataclass
class TrainingBenchmarkResult3D:
    """Results from 3D training benchmark"""

    # Timing
    nvidia_forward_time: float
    optimized_forward_time: float
    nvidia_backward_time: float
    optimized_backward_time: float
    forward_speedup: float
    backward_speedup: float
    total_speedup: float

    # Accuracy
    output_match: bool
    output_max_diff: float
    grad_input_match: bool
    grad_input_max_diff: float
    grad_weight_match: bool
    grad_weight_max_diff: float
    grad_bias_match: bool
    grad_bias_max_diff: float

    # Memory
    nvidia_peak_memory: float
    optimized_peak_memory: float
    memory_reduction_pct: float


def benchmark_training_step_3d(
    nvidia_layer: nn.Module,
    optimized_layer: nn.Module,
    input_tensor: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    loss_fn: nn.Module,
    device: str,
    warmup_runs: int = 5,
    benchmark_runs: int = 20,
) -> TrainingBenchmarkResult3D:
    """Benchmark a single 3D training step including forward and backward passes"""

    print("Starting 3D training benchmark...")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Device: {device}")
    print(f"Warmup runs: {warmup_runs}, Benchmark runs: {benchmark_runs}")
    print("-" * 60)

    # Clone inputs to ensure fair comparison
    input_nvidia = input_tensor.clone().requires_grad_(True)
    input_optimized = input_tensor.clone().requires_grad_(True)

    # Ensure both layers have same weights
    with torch.no_grad():
        optimized_layer.weight.data.copy_(nvidia_layer.weight.data)
        if nvidia_layer.bias is not None:
            optimized_layer.bias.data.copy_(nvidia_layer.bias.data)

    # 1. Test Equivalence
    print("\n1. Testing Forward Pass Equivalence:")
    nvidia_layer.eval()
    optimized_layer.eval()

    with torch.no_grad():
        out_nvidia = nvidia_layer(input_nvidia, mask)
        out_optimized = optimized_layer(input_optimized, mask)

    output_match = torch.allclose(out_nvidia, out_optimized, rtol=1e-4, atol=1e-5)
    output_max_diff = (out_nvidia - out_optimized).abs().max().item()
    print(
        f"   Outputs match: {'✓' if output_match else '✗'} (max diff: {output_max_diff:.2e})"
    )

    # 2. Test Gradient Equivalence
    print("\n2. Testing Gradient Equivalence:")
    nvidia_layer.train()
    optimized_layer.train()

    # Reset gradients
    nvidia_layer.zero_grad()
    optimized_layer.zero_grad()
    input_nvidia.grad = None
    input_optimized.grad = None

    # Forward and backward for NVIDIA
    out_nvidia = nvidia_layer(input_nvidia, mask)
    loss_nvidia = loss_fn(out_nvidia, target)
    loss_nvidia.backward()

    # Forward and backward for Optimized
    out_optimized = optimized_layer(input_optimized, mask)
    loss_optimized = loss_fn(out_optimized, target)
    loss_optimized.backward()

    # Check gradient equivalence
    grad_input_match = torch.allclose(
        input_nvidia.grad, input_optimized.grad, rtol=1e-4, atol=1e-5
    )
    grad_input_max_diff = (input_nvidia.grad - input_optimized.grad).abs().max().item()

    grad_weight_match = torch.allclose(
        nvidia_layer.weight.grad, optimized_layer.weight.grad, rtol=1e-4, atol=1e-5
    )
    grad_weight_max_diff = (
        (nvidia_layer.weight.grad - optimized_layer.weight.grad).abs().max().item()
    )

    grad_bias_match = True
    grad_bias_max_diff = 0.0
    if nvidia_layer.bias is not None:
        grad_bias_match = torch.allclose(
            nvidia_layer.bias.grad, optimized_layer.bias.grad, rtol=1e-4, atol=1e-5
        )
        grad_bias_max_diff = (
            (nvidia_layer.bias.grad - optimized_layer.bias.grad).abs().max().item()
        )

    print(
        f"   Input gradients match: {'✓' if grad_input_match else '✗'} (max diff: {grad_input_max_diff:.2e})"
    )
    print(
        f"   Weight gradients match: {'✓' if grad_weight_match else '✗'} (max diff: {grad_weight_max_diff:.2e})"
    )
    print(
        f"   Bias gradients match: {'✓' if grad_bias_match else '✗'} (max diff: {grad_bias_max_diff:.2e})"
    )

    # Verify bias gradients are non-zero (important for training)
    if nvidia_layer.bias is not None:
        nvidia_bias_grad_norm = nvidia_layer.bias.grad.norm().item()
        optimized_bias_grad_norm = optimized_layer.bias.grad.norm().item()
        print(f"\n   Bias gradient norms (should be non-zero):")
        print(f"   NVIDIA:    {nvidia_bias_grad_norm:.6f}")
        print(f"   Optimized: {optimized_bias_grad_norm:.6f}")
        if optimized_bias_grad_norm == 0:
            print(
                "   ⚠️  WARNING: Optimized bias gradients are zero! Check bias implementation."
            )

    # 3. Warmup
    print(f"\n3. Running {warmup_runs} warmup iterations...")
    for _ in range(warmup_runs):
        # NVIDIA
        nvidia_layer.zero_grad()
        out = nvidia_layer(input_tensor.clone().requires_grad_(True), mask)
        loss = loss_fn(out, target)
        loss.backward()

        # Optimized
        optimized_layer.zero_grad()
        out = optimized_layer(input_tensor.clone().requires_grad_(True), mask)
        loss = loss_fn(out, target)
        loss.backward()

    if device == "cuda":
        torch.cuda.synchronize()

    # 4. Benchmark Forward Pass
    print(f"\n4. Benchmarking Forward Pass ({benchmark_runs} runs):")

    # NVIDIA forward
    timer_nvidia_fwd = tbenchmark.Timer(
        stmt="layer(inp, m)",
        globals={"layer": nvidia_layer, "inp": input_tensor, "m": mask},
        num_threads=1,
    )
    nvidia_fwd_measurement = timer_nvidia_fwd.timeit(benchmark_runs)
    nvidia_forward_time = nvidia_fwd_measurement.mean * 1000  # ms

    # Optimized forward
    timer_optimized_fwd = tbenchmark.Timer(
        stmt="layer(inp, m)",
        globals={"layer": optimized_layer, "inp": input_tensor, "m": mask},
        num_threads=1,
    )
    optimized_fwd_measurement = timer_optimized_fwd.timeit(benchmark_runs)
    optimized_forward_time = optimized_fwd_measurement.mean * 1000  # ms

    forward_speedup = (
        nvidia_forward_time / optimized_forward_time
        if optimized_forward_time > 0
        else float("inf")
    )

    print(f"   NVIDIA:    {nvidia_forward_time:8.3f} ms")
    print(f"   Optimized: {optimized_forward_time:8.3f} ms")
    print(f"   Speedup:   {forward_speedup:8.2f}x")

    # 5. Benchmark Full Training Step (Forward + Backward)
    print(f"\n5. Benchmarking Full Training Step:")

    def training_step_nvidia():
        inp = input_tensor.clone().requires_grad_(True)
        nvidia_layer.zero_grad()
        out = nvidia_layer(inp, mask)
        loss = loss_fn(out, target)
        loss.backward()
        return loss

    def training_step_optimized():
        inp = input_tensor.clone().requires_grad_(True)
        optimized_layer.zero_grad()
        out = optimized_layer(inp, mask)
        loss = loss_fn(out, target)
        loss.backward()
        return loss

    # NVIDIA full step
    timer_nvidia_full = tbenchmark.Timer(
        stmt="step()", globals={"step": training_step_nvidia}, num_threads=1
    )
    nvidia_full_measurement = timer_nvidia_full.timeit(benchmark_runs)
    nvidia_total_time = nvidia_full_measurement.mean * 1000  # ms

    # Optimized full step
    timer_optimized_full = tbenchmark.Timer(
        stmt="step()", globals={"step": training_step_optimized}, num_threads=1
    )
    optimized_full_measurement = timer_optimized_full.timeit(benchmark_runs)
    optimized_total_time = optimized_full_measurement.mean * 1000  # ms

    # Calculate backward time by subtraction
    nvidia_backward_time = nvidia_total_time - nvidia_forward_time
    optimized_backward_time = optimized_total_time - optimized_forward_time
    backward_speedup = (
        nvidia_backward_time / optimized_backward_time
        if optimized_backward_time > 0
        else float("inf")
    )
    total_speedup = (
        nvidia_total_time / optimized_total_time
        if optimized_total_time > 0
        else float("inf")
    )

    print(f"   NVIDIA:")
    print(f"      Forward:  {nvidia_forward_time:8.3f} ms")
    print(f"      Backward: {nvidia_backward_time:8.3f} ms")
    print(f"      Total:    {nvidia_total_time:8.3f} ms")
    print(f"   Optimized:")
    print(f"      Forward:  {optimized_forward_time:8.3f} ms")
    print(f"      Backward: {optimized_backward_time:8.3f} ms")
    print(f"      Total:    {optimized_total_time:8.3f} ms")
    print(f"   Speedup:")
    print(f"      Forward:  {forward_speedup:8.2f}x")
    print(f"      Backward: {backward_speedup:8.2f}x")
    print(f"      Total:    {total_speedup:8.2f}x")

    # 6. Memory Usage (if CUDA)
    nvidia_peak_memory = 0.0
    optimized_peak_memory = 0.0
    memory_reduction_pct = 0.0

    if device == "cuda":
        print(f"\n6. Memory Usage:")

        # NVIDIA memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _ = training_step_nvidia()
        torch.cuda.synchronize()
        nvidia_peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB

        # Optimized memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _ = training_step_optimized()
        torch.cuda.synchronize()
        optimized_peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB

        memory_reduction_pct = (
            (1 - optimized_peak_memory / nvidia_peak_memory) * 100
            if nvidia_peak_memory > 0
            else 0
        )

        print(f"   NVIDIA:    {nvidia_peak_memory:8.2f} MB")
        print(f"   Optimized: {optimized_peak_memory:8.2f} MB")
        print(f"   Reduction: {memory_reduction_pct:8.1f}%")

    return TrainingBenchmarkResult3D(
        nvidia_forward_time=nvidia_forward_time,
        optimized_forward_time=optimized_forward_time,
        nvidia_backward_time=nvidia_backward_time,
        optimized_backward_time=optimized_backward_time,
        forward_speedup=forward_speedup,
        backward_speedup=backward_speedup,
        total_speedup=total_speedup,
        output_match=output_match,
        output_max_diff=output_max_diff,
        grad_input_match=grad_input_match,
        grad_input_max_diff=grad_input_max_diff,
        grad_weight_match=grad_weight_match,
        grad_weight_max_diff=grad_weight_max_diff,
        grad_bias_match=grad_bias_match,
        grad_bias_max_diff=grad_bias_max_diff,
        nvidia_peak_memory=nvidia_peak_memory,
        optimized_peak_memory=optimized_peak_memory,
        memory_reduction_pct=memory_reduction_pct,
    )


def main():
    """Run 3D training benchmark with test cases"""
    # Set seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running 3D Partial Convolution Training Benchmark on {device}")
    print("=" * 80)

    # Test configurations for 3D
    test_configs = [
        {
            "name": "Small 3D Volume",
            "batch_size": 4,
            "in_channels": 32,
            "out_channels": 64,
            "depth": 16,
            "height": 64,
            "width": 64,
            "kernel_size": 3,
            "padding": 1,
        },
        {
            "name": "Medical Imaging Size",
            "batch_size": 2,
            "in_channels": 16,
            "out_channels": 32,
            "depth": 32,
            "height": 128,
            "width": 128,
            "kernel_size": 3,
            "padding": 1,
        },
        {
            "name": "Video Processing",
            "batch_size": 1,
            "in_channels": 64,
            "out_channels": 128,
            "depth": 8,  # temporal dimension
            "height": 256,
            "width": 256,
            "kernel_size": 3,
            "padding": 1,
        },
    ]

    # Run benchmark for each configuration
    for config in test_configs:
        print(f"\n{'='*80}")
        print(f"Testing: {config['name']}")
        print(f"{'='*80}")

        # Create layers
        print("Creating layers...")
        nvidia_layer = NvidiaPartialConv3d(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            kernel_size=config["kernel_size"],
            padding=config["padding"],
            bias=True,
            multi_channel=False,
        ).to(device)

        optimized_layer = OptimizedPartialConv3dFixed(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            kernel_size=config["kernel_size"],
            padding=config["padding"],
            bias=True,
            multi_channel=False,
        ).to(device)

        # Create test data
        print("Creating test data...")
        input_shape = (
            config["batch_size"],
            config["in_channels"],
            config["depth"],
            config["height"],
            config["width"],
        )

        input_tensor = torch.randn(input_shape, device=device)

        # Test with different mask types
        mask_configs = [
            (
                "Random mask (80% valid)",
                (
                    torch.rand(
                        config["batch_size"],
                        1,
                        config["depth"],
                        config["height"],
                        config["width"],
                        device=device,
                    )
                    > 0.2
                ).float(),
            ),
            (
                "Random mask (50% valid)",
                (
                    torch.rand(
                        config["batch_size"],
                        1,
                        config["depth"],
                        config["height"],
                        config["width"],
                        device=device,
                    )
                    > 0.5
                ).float(),
            ),
            (
                "Full mask",
                torch.ones(
                    config["batch_size"],
                    1,
                    config["depth"],
                    config["height"],
                    config["width"],
                    device=device,
                ),
            ),
        ]

        # Target for loss calculation (same spatial size as output)
        target_shape = (
            config["batch_size"],
            config["out_channels"],
            config["depth"],
            config["height"],
            config["width"],
        )
        target = torch.randn(target_shape, device=device)

        # Loss function
        loss_fn = nn.MSELoss()

        # Run benchmark for first mask type only (to save time)
        mask_name, mask = mask_configs[0]  # Use random 80% mask

        print(f"\nUsing: {mask_name}")

        try:
            result = benchmark_training_step_3d(
                nvidia_layer=nvidia_layer,
                optimized_layer=optimized_layer,
                input_tensor=input_tensor,
                mask=mask,
                target=target,
                loss_fn=loss_fn,
                device=device,
                warmup_runs=3,  # Fewer warmup runs for 3D due to memory
                benchmark_runs=10,  # Fewer benchmark runs for 3D
            )

            # Print summary for this configuration
            print("\n" + "=" * 80)
            print(f"TRAINING BENCHMARK SUMMARY - {config['name']}")
            print("=" * 80)

            print("\n✓ Correctness:")
            all_match = all(
                [
                    result.output_match,
                    result.grad_input_match,
                    result.grad_weight_match,
                    result.grad_bias_match,
                ]
            )
            print(f"   All outputs and gradients match: {'✓' if all_match else '✗'}")

            print("\n✓ Performance:")
            print(f"   Forward pass speedup:  {result.forward_speedup:.2f}x")
            print(f"   Backward pass speedup: {result.backward_speedup:.2f}x")
            print(f"   Total training speedup: {result.total_speedup:.2f}x")

            if device == "cuda":
                print("\n✓ Memory:")
                print(f"   Peak memory reduction: {result.memory_reduction_pct:.1f}%")

            # Visual performance comparison
            print("\n✓ Visual Performance Comparison:")
            print("   " + "─" * 50)

            def create_bar(value, max_value, width=30):
                filled = int((value / max_value) * width)
                return "█" * filled + "░" * (width - filled)

            max_time = max(result.nvidia_forward_time, result.nvidia_backward_time)

            print(f"   Forward Pass:")
            print(
                f"     NVIDIA:    {create_bar(result.nvidia_forward_time, max_time)} {result.nvidia_forward_time:6.2f} ms"
            )
            print(
                f"     Optimized: {create_bar(result.optimized_forward_time, max_time)} {result.optimized_forward_time:6.2f} ms"
            )

            print(f"\n   Backward Pass:")
            print(
                f"     NVIDIA:    {create_bar(result.nvidia_backward_time, max_time)} {result.nvidia_backward_time:6.2f} ms"
            )
            print(
                f"     Optimized: {create_bar(result.optimized_backward_time, max_time)} {result.optimized_backward_time:6.2f} ms"
            )

        except torch.cuda.OutOfMemoryError:
            print(f"\n⚠️  Out of memory for {config['name']}. Skipping...")
            torch.cuda.empty_cache()
            continue

        # Only run first configuration for quick test
        # Remove this break to test all configurations
        break

    print("\n" + "=" * 80)
    print("3D Training Benchmark completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
