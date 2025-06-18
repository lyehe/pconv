from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as tbenchmark

# A small constant to prevent division by zero
EPSILON = 1e-8


###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
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

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
        #     self.update_mask.to(input)
        #     self.mask_ratio.to(input)

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
# Optimized Implementation
# (Same as your provided code)
###############################################################################
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
        if (
            self.multi_channel
        ):  # This should be based on the effective input channels for the mask conv
            self.slide_winsize = float(
                kernel_elements * (self.in_channels // self.groups)
            )
        else:  # If not multi_channel, mask is treated as single channel for sum pooling
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
            and self._last_mask_ptr
            == mask.data_ptr()  # Check if it's the same tensor object
        ):
            return self._last_result

        with torch.no_grad():
            # Create weight for sum pooling on-the-fly
            if not self.multi_channel or mask.shape[1] == 1:
                # If not multi_channel OR if the input mask is already single-channel,
                # treat the mask as single-channel for sum pooling.
                mask_for_sum = mask if mask.shape[1] == 1 else mask[:, 0:1, ...]
                conv_weight = torch.ones(
                    1, 1, *self.kernel_size, device=mask.device, dtype=mask.dtype
                )
                # For sum pooling a single channel (or effectively single channel) mask, groups is 1
                groups_for_mask_conv = 1
            else:  # multi_channel is True and mask.shape[1] > 1 (presumably == self.in_channels)
                mask_for_sum = mask
                if (
                    self.groups == 1
                ):  # Standard convolution, sum over all in_channels of the mask
                    conv_weight = torch.ones(
                        1,
                        self.in_channels,
                        *self.kernel_size,
                        device=mask.device,
                        dtype=mask.dtype,
                    )
                    groups_for_mask_conv = 1
                else:  # Grouped convolution, sum mask channels within each group
                    channels_per_group = self.in_channels // self.groups
                    # The conv_weight for mask sum pooling should reflect the grouping structure
                    # It should be (self.groups, channels_per_group_mask, k, k, k)
                    # where channels_per_group_mask is channels_per_group if mask is grouped like input,
                    # or 1 if we sum all channels within a group of the mask to a single channel.
                    # Given the slide_winsize calculation, it implies summing all relevant input channels.
                    conv_weight = torch.ones(
                        self.groups,  # Number of output groups for the mask conv
                        channels_per_group,  # Number of input channels per group for the mask
                        *self.kernel_size,
                        device=mask.device,
                        dtype=mask.dtype,
                    )
                    groups_for_mask_conv = (
                        self.groups
                    )  # Mask conv is grouped like main conv

            # Perform sum pooling
            update_mask = F.conv3d(
                mask_for_sum,
                conv_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=groups_for_mask_conv,  # Use the determined groups for mask conv
            )

            # Calculate ratio and clamp mask
            mask_ratio = self.slide_winsize / (update_mask + EPSILON)
            update_mask = torch.clamp(update_mask, 0, 1)  # Clamp sum to 0-1 range
            mask_ratio = (
                mask_ratio * update_mask
            )  # Apply correction factor only to valid areas

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
            # If no mask is provided, create a single-channel mask of ones.
            # This will be handled correctly by _compute_mask_updates.
            mask = torch.ones(
                input_tensor.shape[0],
                1,  # Always create a single channel mask here if None
                *input_tensor.shape[2:],
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )

        # If multi_channel is True and the input mask is single channel but the input tensor is multi-channel,
        # the mask should be expanded to match the input tensor's channels for element-wise multiplication.
        # This is for `input_tensor * mask`. The `_compute_mask_updates` handles its own logic.
        current_mask_for_input_mult = mask
        if self.multi_channel and mask.shape[1] == 1 and input_tensor.shape[1] != 1:
            current_mask_for_input_mult = mask.expand(
                -1, input_tensor.shape[1], -1, -1, -1
            )
        elif (
            not self.multi_channel and mask.shape[1] != 1
        ):  # Using first channel of mask if not multi_channel
            current_mask_for_input_mult = mask[:, 0:1, ...].expand(
                -1, input_tensor.shape[1], -1, -1, -1
            )

        update_mask, mask_ratio = self._compute_mask_updates(
            mask
        )  # Pass original mask to compute updates

        # Main convolution is performed *without* bias, as it's handled manually
        output = F.conv3d(
            input_tensor
            * current_mask_for_input_mult,  # Use potentially expanded/selected mask
            self.weight,
            None,  # Bias is None
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        # Apply partial conv formula with in-place operations
        if self.bias is not None:
            output.mul_(mask_ratio)  # In-place multiplication by ratio
            output.add_(self._bias_view)  # In-place addition of bias
            output.mul_(update_mask)  # In-place application of final mask
        else:
            output.mul_(mask_ratio)

        if self.return_mask:
            return output, update_mask

        return output


###############################################################################
# Benchmarking Suite
# (BenchmarkResult and print_summary are same as your provided code)
###############################################################################
@dataclass
class BenchmarkResult:
    """A simple dataclass to store benchmark results for summarization."""

    name: str
    nvidia_time: float
    optimized_time: float
    speedup: float
    outputs_match: bool
    max_diff: float
    memory_nvidia: float
    memory_optimized: float


def print_summary(results: list[BenchmarkResult], device: str):
    """Prints a formatted summary of all benchmark results."""
    print(f"\n{'=' * 80}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 80}")

    if not results:
        print("No results to summarize.")
        return

    # --- Speedup Statistics ---
    # Filter out inf and NaN before calculating mean for speedups
    valid_speedups = [
        r.speedup
        for r in results
        if r.speedup is not None and np.isfinite(r.speedup) and r.speedup > 0
    ]
    if valid_speedups:
        avg_speedup = np.mean(valid_speedups) if valid_speedups else 0.0
        # For max/min, we might want to include inf if it's meaningful (e.g. one time is zero)
        # but for a general summary, using only finite speedups is safer.
        all_finite_speedups = [
            r.speedup
            for r in results
            if r.speedup is not None and np.isfinite(r.speedup)
        ]
        if all_finite_speedups:
            max_speedup = max(all_finite_speedups)
            min_speedup = min(
                s for s in all_finite_speedups if s > 0
            )  # Min positive speedup
        else:  # Handle case where all speedups might be inf or 0
            max_speedup = (
                float("inf") if any(r.speedup == float("inf") for r in results) else 0.0
            )
            min_speedup = 0.0

        print("\nSpeedup Statistics (excluding non-finite values for avg):")
        print(f"   Average: {avg_speedup:.2f}x")
        print(f"   Maximum: {max_speedup:.2f}x")  # This might still show inf if present
        print(f"   Minimum (positive): {min_speedup:.2f}x")

    # --- Detailed Results Table ---
    print(f"\n{'Detailed Results':^80}")
    print(
        f"{'Test':<25} {'NVIDIA (ms)':>15} {'Optimized (ms)':>18} {'Speedup':>10} {'Match':>8}"
    )
    print("-" * 80)

    for r in results:
        match_str = "✓" if r.outputs_match else "✗"
        if r.optimized_time == 0 and r.nvidia_time > 0:
            speedup_str = "inf"
        elif r.nvidia_time == 0 and r.optimized_time > 0:
            speedup_str = "0.00x"  # or N/A
        elif r.optimized_time == 0 and r.nvidia_time == 0:
            speedup_str = "N/A"
        elif r.speedup is None or not np.isfinite(r.speedup):
            speedup_str = "N/A"
        else:
            speedup_str = f"{r.speedup:.2f}x"
        print(
            f"{r.name:<25} {r.nvidia_time:>15.3f} {r.optimized_time:>18.3f} "
            f"{speedup_str:>9} {match_str:>8}"
        )

    # --- Memory Usage Summary (if on CUDA) ---
    if device == "cuda" and any(r.memory_nvidia > 0 for r in results):
        print(f"\n{'Memory Usage Summary (MB)':^80}")
        print(
            f"{'Test':<25} {'NVIDIA Peak':>15} {'Optimized Peak':>18} {'Reduction':>15}"
        )
        print("-" * 80)
        for r in results:
            reduction = r.memory_nvidia - r.memory_optimized
            reduction_pct = (
                (reduction / r.memory_nvidia * 100) if r.memory_nvidia > 0 else 0
            )
            reduction_str = f"{reduction:.2f} ({reduction_pct:.1f}%)"
            print(
                f"{r.name:<25} {r.memory_nvidia:>15.2f} {r.memory_optimized:>18.2f} {reduction_str:>15}"
            )

    # --- Final Accuracy Check ---
    all_match = all(r.outputs_match for r in results)
    print(f"\nAll outputs match: {'✓' if all_match else '✗'}")

    if not all_match:
        print("\nTests with mismatched outputs:")
        for r in results:
            if not r.outputs_match:
                print(f"   - {r.name} (max diff: {r.max_diff:.2e})")


def benchmark():
    """Runs a full benchmark comparing the two implementations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmark on {device}")
    print("=" * 80)

    results = []
    warmup_iterations = 5
    # For torch.utils.benchmark, num_threads=1 is often good for GPU to avoid interference.
    # The `number` parameter within timeit controls inner loops.
    # `timeit(N)` runs N measurements.
    num_benchmark_measurements = 20  # Number of measurements to average for tbenchmark

    # Define test configurations (same as your provided code)
    configs = [
        {
            "name": "Small (128x128x16)",
            "layer_config": {
                "in_channels": 64,
                "out_channels": 128,
                "kernel_size": 3,
                "padding": 1,
                "multi_channel": False,
            },
            "input_shape": (8, 64, 16, 128, 128),
            "mask_type": "full",
        },
        {
            "name": "Medium (256x256x16)",
            "layer_config": {
                "in_channels": 128,
                "out_channels": 256,
                "kernel_size": 3,
                "padding": 1,
                "multi_channel": False,
            },
            "input_shape": (2, 128, 16, 256, 256),
            "mask_type": "full",
        },
        {
            "name": "Large (512x512x16)",
            "layer_config": {
                "in_channels": 64,
                "out_channels": 128,
                "kernel_size": 3,
                "padding": 1,
                "multi_channel": False,
            },
            "input_shape": (1, 64, 16, 512, 512),
            "mask_type": "full",
        },
        {
            "name": "Grouped Conv",
            "layer_config": {
                "in_channels": 128,
                "out_channels": 256,
                "kernel_size": 3,
                "padding": 1,
                "groups": 4,
                "multi_channel": False,  # False for NVIDIA original compatibility
            },
            "input_shape": (4, 128, 16, 128, 128),
            "mask_type": "full",
        },
        {
            "name": "Strided Conv",
            "layer_config": {
                "in_channels": 64,
                "out_channels": 128,
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "multi_channel": False,
            },
            "input_shape": (4, 64, 32, 256, 256),
            "mask_type": "full",
        },
        {
            "name": "Large Kernel",
            "layer_config": {
                "in_channels": 64,
                "out_channels": 128,
                "kernel_size": 5,
                "padding": 2,
                "multi_channel": False,
            },
            "input_shape": (4, 64, 16, 128, 128),
            "mask_type": "full",
        },
        {
            "name": "Dilated Conv",
            "layer_config": {
                "in_channels": 64,
                "out_channels": 128,
                "kernel_size": 3,
                "padding": 2,
                "dilation": 2,
                "multi_channel": False,
            },
            "input_shape": (4, 64, 16, 128, 128),
            "mask_type": "full",
        },
        {
            "name": "Random Mask",
            "layer_config": {
                "in_channels": 128,
                "out_channels": 256,
                "kernel_size": 3,
                "padding": 1,
                "multi_channel": False,
            },
            "input_shape": (4, 128, 16, 128, 128),
            "mask_type": "random",
        },
        {
            "name": "No Mask",
            "layer_config": {
                "in_channels": 128,
                "out_channels": 256,
                "kernel_size": 3,
                "padding": 1,
                "multi_channel": False,
            },
            "input_shape": (4, 128, 16, 128, 128),
            "mask_type": "none",
        },
        {
            "name": "High Channels",
            "layer_config": {
                "in_channels": 256,
                "out_channels": 512,
                "kernel_size": 3,
                "padding": 1,
                "multi_channel": False,
            },
            "input_shape": (1, 256, 8, 64, 64),
            "mask_type": "full",
        },
        {
            "name": "Large Batch",
            "layer_config": {
                "in_channels": 32,
                "out_channels": 64,
                "kernel_size": 3,
                "padding": 1,
                "multi_channel": False,
            },
            "input_shape": (16, 32, 8, 64, 64),
            "mask_type": "full",
        },
        {
            "name": "Sequential Data",
            "layer_config": {
                "in_channels": 32,
                "out_channels": 64,
                "kernel_size": 3,
                "padding": 1,
                "multi_channel": False,
            },
            "input_shape": (4, 32, 64, 32, 32),
            "mask_type": "random",
        },
    ]

    for config in configs:
        print(f"\n--- Testing {config['name']} Configuration ---")
        # Ensure 'multi_channel' is in cfg_params if it's in layer_config
        # The NvidiaPConv pops it, OptimizedPConv uses it as a kwarg.
        cfg_params_nvidia = config["layer_config"].copy()
        cfg_params_optimized = config["layer_config"].copy()

        input_shape = config["input_shape"]
        mask_type = config.get("mask_type", "full")

        try:
            # --- Initialization ---
            # NvidiaPartialConv3d pops 'multi_channel'
            nvidia_layer = (
                NvidiaPartialConv3d(**cfg_params_nvidia, bias=True).to(device).eval()
            )
            # OptimizedPartialConv3d takes 'multi_channel' as a direct argument
            optimized_layer = (
                OptimizedPartialConv3d(**cfg_params_optimized, bias=True)
                .to(device)
                .eval()
            )

            with torch.no_grad():
                optimized_layer.weight.data.copy_(nvidia_layer.weight.data)
                if nvidia_layer.bias is not None:
                    optimized_layer.bias.data.copy_(nvidia_layer.bias.data)
                    if hasattr(
                        optimized_layer, "_bias_view"
                    ):  # Ensure _bias_view exists
                        optimized_layer._bias_view.copy_(
                            nvidia_layer.bias.view(1, -1, 1, 1, 1)
                        )

            # --- Create Input Data ---
            input_tensor = torch.randn(input_shape, device=device)
            if mask_type == "none":
                mask_input_arg = (
                    None  # Use None to test the layer's internal mask generation
                )
            elif mask_type == "random":
                mask_input_arg = (
                    torch.rand(input_shape[0], 1, *input_shape[2:], device=device) > 0.5
                ).float()
            else:  # 'full'
                mask_input_arg = torch.ones(
                    input_shape[0], 1, *input_shape[2:], device=device
                )

            # --- Warmup ---
            for _ in range(warmup_iterations):
                _ = nvidia_layer(input_tensor, mask_input_arg)
                _ = optimized_layer(input_tensor, mask_input_arg)
            if device == "cuda":
                torch.cuda.synchronize()

            # --- 1. Equivalence Check ---
            print("1. Equivalence:")
            with torch.no_grad():
                # For equivalence, ensure both layers get the same mask_input_arg
                out_nvidia = nvidia_layer(input_tensor, mask_input_arg).to("cpu")
                out_optimized = optimized_layer(input_tensor, mask_input_arg).to("cpu")

            are_close = torch.allclose(out_nvidia, out_optimized, atol=1e-5, rtol=1e-4)
            max_diff = (
                (out_nvidia - out_optimized).abs().max().item()
                if not are_close
                else 0.0
            )
            print(f"   - Outputs Match: {'✓' if are_close else '✗'}")
            if not are_close:
                print(f"   - Max Absolute Difference: {max_diff:.2e}")

            # --- 2. Speed Benchmark with torch.utils.benchmark ---
            print(f"\n2. Speed (avg over {num_benchmark_measurements} measurements):")

            timer_nvidia = tbenchmark.Timer(
                stmt="layer(inp, m_arg)",
                globals={
                    "layer": nvidia_layer,
                    "inp": input_tensor,
                    "m_arg": mask_input_arg,
                },
                num_threads=1,  # Good for GPU benchmarks
            )
            nvidia_measurement = timer_nvidia.timeit(num_benchmark_measurements)
            nvidia_time = nvidia_measurement.mean * 1000  # ms

            timer_optimized = tbenchmark.Timer(
                stmt="layer(inp, m_arg)",
                globals={
                    "layer": optimized_layer,
                    "inp": input_tensor,
                    "m_arg": mask_input_arg,
                },
                num_threads=1,
            )
            optimized_measurement = timer_optimized.timeit(num_benchmark_measurements)
            optimized_time = optimized_measurement.mean * 1000  # ms

            if optimized_time > 0:
                speedup = nvidia_time / optimized_time
            elif (
                nvidia_time > 0 and optimized_time == 0
            ):  # Optimized is immeasurably fast
                speedup = float("inf")
            else:  # Both 0 or nvidia_time is 0 and optimized_time > 0
                speedup = (
                    0.0 if nvidia_time == 0 and optimized_time > 0 else 1.0
                )  # Or handle as N/A

            print(f"   - NVIDIA Impl:   {nvidia_time:9.3f} ms")
            print(f"   - Optimized Impl:{optimized_time:9.3f} ms")
            if speedup == float("inf"):
                print("   - Speedup:           infx")
            else:
                print(f"   - Speedup:         {speedup:.2f}x")

            # --- 3. Memory Benchmark ---
            nvidia_mem, optimized_mem = 0.0, 0.0
            if device == "cuda":
                print("\n3. Peak Memory Usage (during forward pass):")
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                _ = nvidia_layer(input_tensor, mask_input_arg)
                torch.cuda.synchronize()  # Ensure op is done
                nvidia_mem = torch.cuda.max_memory_allocated(device) / 1e6

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                _ = optimized_layer(input_tensor, mask_input_arg)
                torch.cuda.synchronize()  # Ensure op is done
                optimized_mem = torch.cuda.max_memory_allocated(device) / 1e6

                print(f"   - NVIDIA Impl:   {nvidia_mem:9.2f} MB")
                print(f"   - Optimized Impl:{optimized_mem:9.2f} MB")
                if (
                    nvidia_mem > 0 and optimized_mem <= nvidia_mem
                ):  # Avoid division by zero or negative reduction
                    reduction = (1 - optimized_mem / nvidia_mem) * 100
                    print(f"   - Reduction:       {reduction:.2f}%")
                elif optimized_mem > nvidia_mem:
                    increase = (optimized_mem / nvidia_mem - 1) * 100
                    print(f"   - Increase:        {increase:.2f}%")

            results.append(
                BenchmarkResult(
                    config["name"],
                    nvidia_time,
                    optimized_time,
                    speedup,
                    are_close,
                    max_diff,
                    nvidia_mem,
                    optimized_mem,
                )
            )
            print("-" * 40)

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"   - FAILED: {e}")
            results.append(
                BenchmarkResult(config["name"], -1, -1, 0, False, -1, -1, -1)
            )
            if device == "cuda":
                torch.cuda.empty_cache()  # Try to free memory
            continue

    print_summary(results, device)


if __name__ == "__main__":
    # Set seeds for reproducibility of random data, not for benchmark timing itself
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    benchmark()
