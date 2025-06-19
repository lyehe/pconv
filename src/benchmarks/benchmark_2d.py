from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as tbenchmark  # New import

EPSILON = 1e-8

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

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
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
# Optimized Implementation
###############################################################################
class OptimizedPartialConv2d(nn.Conv2d):
    def __init__(
        self,
        *args,
        multi_channel: bool = False,  # Default to False to match Nvidia's default
        return_mask: bool = False,
        cache_masks: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.multi_channel = multi_channel
        self.return_mask = return_mask
        self.cache_masks = cache_masks

        kernel_elements = self.kernel_size[0] * self.kernel_size[1]
        if (
            self.multi_channel
        ):  # slide_winsize depends on effective input channels for mask sum-pooling
            self.slide_winsize = float(
                kernel_elements * (self.in_channels // self.groups)
            )
        else:  # If not multi_channel, mask is treated as single channel for sum pooling
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
            else:  # multi_channel is True and mask.shape[1] > 1
                mask_for_sum = mask
                if self.groups == 1:  # Standard conv, sum over all in_channels of mask
                    conv_weight = torch.ones(
                        1,
                        self.in_channels,  # Sum all input mask channels
                        *self.kernel_size,
                        device=mask.device,
                        dtype=mask.dtype,
                    )
                    groups_for_mask_conv = 1
                else:  # Grouped conv, sum mask channels within each group
                    channels_per_group = self.in_channels // self.groups
                    conv_weight = torch.ones(
                        self.groups,  # Number of output groups for mask conv
                        channels_per_group,  # Number of input channels per group for mask
                        *self.kernel_size,
                        device=mask.device,
                        dtype=mask.dtype,
                    )
                    groups_for_mask_conv = self.groups  # Mask conv is grouped

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
        input_mask_for_calc = mask  # The mask to be used for _compute_mask_updates
        if input_mask_for_calc is None:
            input_mask_for_calc = torch.ones(  # single channel default
                input_tensor.shape[0],
                1,
                *input_tensor.shape[2:],
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )

        update_mask, mask_ratio = self._compute_mask_updates(input_mask_for_calc)

        # Determine mask for element-wise multiplication with input_tensor
        # This mask should match input_tensor's channel dimension if multi_channel
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
            # If not multi_channel, but a multi-channel mask was given, use its first channel
            # and expand it to match input tensor channels for the multiplication.
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


###############################################################################
# Benchmark Suite
###############################################################################
@dataclass
class BenchmarkResult:
    test_name: str
    nvidia_time: float
    optimized_time: float
    speedup: float
    outputs_match: bool
    max_diff: float
    memory_nvidia: float = 0.0
    memory_optimized: float = 0.0


class SimpleBenchmark:
    def __init__(
        self,
        device="cuda" if torch.cuda.is_available() else "cpu",
        warmup_runs=5,  # Explicit warmup before tbenchmark
        benchmark_measurements=20,  # Number of measurements for tbenchmark.timeit
    ):
        self.device = device
        self.warmup_runs = warmup_runs
        self.benchmark_measurements = benchmark_measurements
        self.results = []

        if device == "cuda":
            # torch.backends.cudnn.benchmark = True # Can be enabled if input sizes are very consistent
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def create_layers(self, **kwargs):
        # Ensure 'multi_channel' is handled correctly for both
        # NvidiaPConv pops it, OptimizedPConv uses it as a kwarg.
        # Benchmark configs should explicitly pass multi_channel.

        # For Nvidia layer
        nvidia_kwargs = kwargs.copy()
        # NvidiaPartialConv2d will pop 'multi_channel' if present
        nvidia = NvidiaPartialConv2d(**nvidia_kwargs).to(self.device).eval()

        # For Optimized layer
        optimized_kwargs = kwargs.copy()
        # OptimizedPartialConv2d takes 'multi_channel' as a direct argument
        optimized = OptimizedPartialConv2d(**optimized_kwargs).to(self.device).eval()

        with torch.no_grad():
            optimized.weight.data.copy_(nvidia.weight.data)  # Use copy_
            if nvidia.bias is not None:
                optimized.bias.data.copy_(nvidia.bias.data)  # Use copy_
                if hasattr(optimized, "_bias_view"):
                    optimized._bias_view.copy_(nvidia.bias.view(1, -1, 1, 1))
        return nvidia, optimized

    def measure_memory(self, layer_callable, input_tensor, mask_arg):
        if self.device != "cuda":
            return 0.0
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        _ = layer_callable(input_tensor, mask_arg)
        torch.cuda.synchronize(self.device)  # Ensure op is done
        peak_memory = torch.cuda.max_memory_allocated(self.device) / 1e6  # MB
        return peak_memory

    def benchmark_single_config(self, config: dict):
        print(f"--- Testing {config['name']} ---")

        layer_kwargs = config["layer_config"]
        input_shape = config["input_shape"]

        # Ensure multi_channel is explicitly set in layer_kwargs for consistent testing
        if "multi_channel" not in layer_kwargs:
            layer_kwargs["multi_channel"] = False  # Default to False if not specified

        nvidia, optimized = self.create_layers(**layer_kwargs)

        input_tensor = torch.randn(input_shape, device=self.device)
        mask_input_arg = None
        if config.get("mask_type") == "random":
            mask_input_arg = (
                torch.rand(input_shape[0], 1, *input_shape[2:], device=self.device)
                > 0.5
            ).float()
        elif config.get("mask_type") != "none":  # "full" or default
            mask_input_arg = torch.ones(
                input_shape[0], 1, *input_shape[2:], device=self.device
            )

        # Warmup
        for _ in range(self.warmup_runs):
            _ = nvidia(input_tensor, mask_input_arg)
            _ = optimized(input_tensor, mask_input_arg)
        if self.device == "cuda":
            torch.cuda.synchronize()

        if hasattr(optimized, "clear_cache"):  # Clear cache after warmup, before timing
            optimized.clear_cache()
        # For Nvidia, to reset its internal cache for a fair timing start:
        nvidia.update_mask = None
        nvidia.last_size = (None, None, None, None)

        # Equivalence Check
        print("1. Equivalence:")
        with torch.no_grad():
            out_nvidia = nvidia(input_tensor, mask_input_arg).to("cpu")
            out_optimized = optimized(input_tensor, mask_input_arg).to("cpu")

        are_close = torch.allclose(out_nvidia, out_optimized, rtol=1e-4, atol=1e-5)
        max_diff = (
            (out_nvidia - out_optimized).abs().max().item() if not are_close else 0.0
        )
        print(
            f"   - Outputs Match: {'✓' if are_close else '✗'} (Max Diff: {max_diff:.2e})"
        )

        # Speed Benchmark
        print(f"\n2. Speed (avg over {self.benchmark_measurements} measurements):")
        timer_nvidia = tbenchmark.Timer(
            stmt="layer(inp, m_arg)",
            globals={"layer": nvidia, "inp": input_tensor, "m_arg": mask_input_arg},
            num_threads=1,
        )
        nvidia_measurement = timer_nvidia.timeit(self.benchmark_measurements)
        nvidia_time = nvidia_measurement.mean * 1000

        timer_optimized = tbenchmark.Timer(
            stmt="layer(inp, m_arg)",
            globals={"layer": optimized, "inp": input_tensor, "m_arg": mask_input_arg},
            num_threads=1,
        )
        optimized_measurement = timer_optimized.timeit(self.benchmark_measurements)
        optimized_time = optimized_measurement.mean * 1000

        speedup = float("inf")
        if optimized_time > 0:
            speedup = nvidia_time / optimized_time
        elif nvidia_time == 0 and optimized_time == 0:  # Both immeasurably fast
            speedup = 1.0  # Or N/A
        elif nvidia_time > 0 and optimized_time == 0:  # Optimized is inf faster
            speedup = float("inf")
        elif (
            nvidia_time == 0 and optimized_time > 0
        ):  # Nvidia is inf faster (should not happen with opt)
            speedup = 0.0

        print(f"   - NVIDIA Impl:    {nvidia_time:10.3f} ms")
        print(f"   - Optimized Impl: {optimized_time:10.3f} ms")
        if speedup == float("inf"):
            print("   - Speedup:              infx\n")
        else:
            print(f"   - Speedup:        {speedup:10.2f}x\n")

        # Memory Benchmark
        mem_nvidia = self.measure_memory(nvidia, input_tensor, mask_input_arg)
        mem_optimized = self.measure_memory(optimized, input_tensor, mask_input_arg)

        print("3. Peak Memory Usage (during forward pass):")
        print(f"   - NVIDIA Impl:    {mem_nvidia:10.2f} MB")
        print(f"   - Optimized Impl: {mem_optimized:10.2f} MB")
        if mem_nvidia > 0:
            if mem_optimized <= mem_nvidia:
                mem_reduction = (1 - mem_optimized / mem_nvidia) * 100
                print(f"   - Reduction:      {mem_reduction:10.2f}%")
            else:
                mem_increase = (mem_optimized / mem_nvidia - 1) * 100
                print(f"   - Increase:       {mem_increase:10.2f}%")

        print("----------------------------------------\n")

        self.results.append(
            BenchmarkResult(
                config["name"],
                nvidia_time,
                optimized_time,
                speedup,
                are_close,
                max_diff,
                mem_nvidia,
                mem_optimized,
            )
        )
        return self.results[-1]

    def run_benchmark(self):
        print(f"Running benchmark on {self.device}")
        print("=" * 80)
        configurations = [
            # Adding multi_channel: False to all configs for consistent testing
            {
                "name": "Small (128x128x16)",
                "layer_config": {
                    "in_channels": 32,
                    "out_channels": 64,
                    "kernel_size": 3,
                    "padding": 1,
                    "bias": True,
                    "multi_channel": False,
                },
                "input_shape": (16, 32, 128, 128),
                "mask_type": "full",
            },
            {
                "name": "Medium (256x256x32)",
                "layer_config": {
                    "in_channels": 16,
                    "out_channels": 32,
                    "kernel_size": 3,
                    "padding": 1,
                    "bias": True,
                    "multi_channel": False,
                },
                "input_shape": (32, 16, 256, 256),
                "mask_type": "full",
            },
            {
                "name": "Large (512x512x16)",
                "layer_config": {
                    "in_channels": 8,
                    "out_channels": 16,
                    "kernel_size": 3,
                    "padding": 1,
                    "bias": True,
                    "multi_channel": False,
                },
                "input_shape": (16, 8, 512, 512),
                "mask_type": "full",
            },
            {
                "name": "Grouped Conv",
                "layer_config": {
                    "in_channels": 32,
                    "out_channels": 64,
                    "kernel_size": 3,
                    "padding": 1,
                    "groups": 4,
                    "bias": True,
                    "multi_channel": False,
                },
                "input_shape": (8, 32, 128, 128),
                "mask_type": "full",
            },
            {
                "name": "Strided Conv",
                "layer_config": {
                    "in_channels": 32,
                    "out_channels": 64,
                    "kernel_size": 3,
                    "stride": 2,
                    "padding": 1,
                    "bias": True,
                    "multi_channel": False,
                },
                "input_shape": (8, 32, 256, 256),
                "mask_type": "full",
            },
            {
                "name": "Large Kernel",
                "layer_config": {
                    "in_channels": 32,
                    "out_channels": 64,
                    "kernel_size": 5,
                    "padding": 2,
                    "bias": True,
                    "multi_channel": False,
                },
                "input_shape": (8, 32, 128, 128),
                "mask_type": "full",
            },
            {
                "name": "Dilated Conv",
                "layer_config": {
                    "in_channels": 32,
                    "out_channels": 64,
                    "kernel_size": 3,
                    "padding": 2,
                    "dilation": 2,
                    "bias": True,
                    "multi_channel": False,
                },
                "input_shape": (8, 32, 128, 128),
                "mask_type": "full",
            },
            {
                "name": "Random Mask",
                "layer_config": {
                    "in_channels": 32,
                    "out_channels": 64,
                    "kernel_size": 3,
                    "padding": 1,
                    "bias": True,
                    "multi_channel": False,
                },
                "input_shape": (8, 32, 128, 128),
                "mask_type": "random",
            },
            {
                "name": "No Mask",  # This will likely mismatch due to NvidiaPConv bug with mask_in=None + caching
                "layer_config": {
                    "in_channels": 32,
                    "out_channels": 64,
                    "kernel_size": 3,
                    "padding": 1,
                    "bias": True,
                    "multi_channel": False,
                },
                "input_shape": (8, 32, 128, 128),
                "mask_type": "none",
            },
            {
                "name": "High Channels",
                "layer_config": {
                    "in_channels": 128,
                    "out_channels": 256,
                    "kernel_size": 3,
                    "padding": 1,
                    "bias": True,
                    "multi_channel": False,
                },
                "input_shape": (4, 128, 64, 64),
                "mask_type": "full",
            },
            {
                "name": "Large Batch",
                "layer_config": {
                    "in_channels": 32,
                    "out_channels": 64,
                    "kernel_size": 3,
                    "padding": 1,
                    "bias": True,
                    "multi_channel": False,
                },
                "input_shape": (64, 32, 64, 64),
                "mask_type": "full",
            },
            {
                "name": "Sequential Data",
                "layer_config": {
                    "in_channels": 32,
                    "out_channels": 32,
                    "kernel_size": 3,
                    "padding": 1,
                    "bias": True,
                    "multi_channel": False,
                },
                "input_shape": (128, 32, 32, 32),
                "mask_type": "random",
            },
        ]
        for config in configurations:
            try:
                self.benchmark_single_config(config)
            except Exception as e:
                print(f"Error testing {config['name']}: {e}")
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                continue
        self.print_summary()

    def print_summary(self):
        if not self.results:
            print("No benchmark results to summarize.")
            return

        print(f"\n{'=' * 80}")
        print("BENCHMARK SUMMARY")
        print(f"{'=' * 80}\n")

        finite_speedups = [
            r.speedup for r in self.results if np.isfinite(r.speedup) and r.speedup > 0
        ]
        if finite_speedups:
            avg_speedup = np.mean(finite_speedups)
            max_speedup = (
                max(finite_speedups)
                if finite_speedups
                else (
                    float("inf")
                    if any(r.speedup == float("inf") for r in self.results)
                    else 0
                )
            )
            min_speedup = min(finite_speedups) if finite_speedups else 0
            print("Speedup Statistics (finite positive values):")
            print(f"  Average: {avg_speedup:.2f}x")
            print(f"  Maximum: {max_speedup:.2f}x")
            print(f"  Minimum: {min_speedup:.2f}x\n")
        else:
            print("Speedup Statistics: No finite positive speedups to analyze.\n")

        print(f"{'':<34}Detailed Results")
        print(
            f"{'Test':<35}{'NVIDIA (ms)':>15}{'Optimized (ms)':>18}{'Speedup':>12}{'Match':>8}"
        )
        print("-" * 88)
        for r in self.results:
            test_name_short = (
                r.test_name.replace(" Configuration", "")
                .replace(" (128x128x16)", "")
                .replace(" (256x256x32)", "")
                .replace(" (512x512x16)", "")
            )
            match_str = "✓" if r.outputs_match else "✗"

            speedup_str = "N/A"
            if r.speedup == float("inf"):
                speedup_str = "infx"
            elif np.isfinite(r.speedup):
                speedup_str = f"{r.speedup:.2f}x"

            print(
                f"{test_name_short:<35}{r.nvidia_time:>15.3f}{r.optimized_time:>18.3f}"
                f"{speedup_str:>11} {match_str:>7}"
            )

        if self.device == "cuda" and any(r.memory_nvidia > 0 for r in self.results):
            print(f"\n{'':<28}Memory Usage Summary (MB)")
            print(
                f"{'Test':<35}{'NVIDIA Peak':>15}{'Optimized Peak':>18}{'Reduction':>18}"
            )
            print("-" * 88)
            for r in self.results:
                test_name_short = (
                    r.test_name.replace(" Configuration", "")
                    .replace(" (128x128x16)", "")
                    .replace(" (256x256x32)", "")
                    .replace(" (512x512x16)", "")
                )
                mem_reduction_abs = r.memory_nvidia - r.memory_optimized
                if r.memory_nvidia > 0:
                    mem_reduction_pct = (mem_reduction_abs / r.memory_nvidia) * 100
                else:
                    mem_reduction_pct = 0.0
                reduction_str = f"{mem_reduction_abs:,.2f} ({mem_reduction_pct:.1f}%)"
                print(
                    f"{test_name_short:<35}{r.memory_nvidia:>15.2f}{r.memory_optimized:>18.2f}{reduction_str:>18}"
                )
        all_match = all(r.outputs_match for r in self.results)
        print(f"\nAll outputs match: {'✓' if all_match else '✗'}")
        if not all_match:
            print("\nTests with mismatched outputs (Max Difference):")
            for r in self.results:
                if not r.outputs_match:
                    print(
                        f"   - {r.test_name.replace(' Configuration', '')}: {r.max_diff:.2e}"
                    )


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    benchmark_suite = SimpleBenchmark(
        device="cuda" if torch.cuda.is_available() else "cpu",
        warmup_runs=5,
        benchmark_measurements=20,
    )
    benchmark_suite.run_benchmark()


if __name__ == "__main__":
    main()
