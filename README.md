# Optimized Partial Convolution 

Optimized implementation of Partial Convolution from Nvidia's [nvidia/partialconv] for image inpainting and related tasks. 
The original implementation is slow especially for 3D convolutions, and has high memory overhead due to persistent tensors.
This optimized version addresses these issues by dynamically creating necessary tensors, improving caching mechanisms, and refining the handling of mask operations.
Benchmark cases generated with Claude.


## Performance Summary (w/ Nvidia 3090ti)

### 1. 2-D Partial Convolution Benchmarks

#### Speedup Statistics

| Metric      | Value      |
| ----------- | ---------- |
| **Average** | **1.36 ×** |
| **Maximum** | **1.62 ×** |
| **Minimum** | **1.13 ×** |

<details>
<summary>Detailed Per-Test Results</summary>

| Test            | NVIDIA (ms) | Optimized (ms) | Speedup   | Match |
| --------------- | ----------- | -------------- | --------- | ----- |
| Small           | 1.341       | 0.971          | **1.38×** | ✓     |
| Medium          | 4.791       | 3.355          | **1.43×** | ✓     |
| Large           | 5.303       | 3.277          | **1.62×** | ✓     |
| Grouped Conv    | 0.649       | 0.447          | **1.45×** | ✓     |
| Strided Conv    | 0.923       | 0.728          | **1.27×** | ✓     |
| Large Kernel    | 1.084       | 0.892          | **1.21×** | ✓     |
| Dilated Conv    | 0.690       | 0.499          | **1.38×** | ✓     |
| Random Mask     | 0.690       | 0.499          | **1.38×** | ✓     |
| No Mask         | 0.614       | 0.541          | **1.13×** | ✓     |
| High Channels   | 0.598       | 0.484          | **1.23×** | ✓     |
| Large Batch     | 1.311       | 0.955          | **1.37×** | ✓     |
| Sequential Data | 0.385       | 0.261          | **1.48×** | ✓     |


---

### 2. 3-D Partial Convolution Benchmarks

#### Speedup Statistics

| Metric      | Value      |
| ----------- | ---------- |
| **Average** | **1.32 ×** |
| **Maximum** | **1.81 ×** |
| **Minimum** | **1.07 ×** |

<details>
<summary>Detailed Per-Test Results</summary>

| Test             | NVIDIA (ms) | Optimized (ms) | Speedup   | Match |
| ---------------- | ----------- | -------------- | --------- | ----- |
| Small (128³×16)  | 43.528      | 32.315         | **1.35×** | ✓     |
| Medium (256³×16) | 128.449     | 112.828        | **1.14×** | ✓     |
| Large (512²×16)  | 91.856      | 69.727         | **1.32×** | ✓     |
| Grouped Conv     | 44.288      | 36.211         | **1.22×** | ✓     |
| Strided Conv     | 33.479      | 27.002         | **1.24×** | ✓     |
| Large Kernel     | 72.408      | 56.667         | **1.28×** | ✓     |
| Dilated Conv     | 21.440      | 15.790         | **1.36×** | ✓     |
| Random Mask      | 62.169      | 54.212         | **1.15×** | ✓     |
| No Mask          | 57.588      | 53.939         | **1.07×** | ✓     |
| High Channels    | 6.655       | 6.213          | **1.07×** | ✓     |
| Large Batch      | 4.996       | 2.775          | **1.80×** | ✓     |
| Sequential Data  | 2.579       | 1.422          | **1.81×** | ✓     |


---

### 3. Multilayer **2-D** Network Benchmarks

#### Speed Performance

| Metric              | Value                    |
| ------------------- | ------------------------ |
| **Average Speedup** | **1.38 ×**               |
| **Maximum Speedup** | **2.23 ×** (Small U-Net) |
| **Minimum Speedup** | **1.24 ×**               |
| **Std Dev**         | 0.27 ×                   |

#### Memory Performance

| Metric                | Value      |
| --------------------- | ---------- |
| **Average Reduction** | **11.3 %** |
| **Maximum Reduction** | **21.4 %** |
| **Minimum Reduction** | **0 %**    |

<details>
<summary>Detailed Per-Network Results</summary>

| Network (Depth)     | Params | NVIDIA (ms) | Opt (ms) | Speedup  | Memory Δ |
| ------------------- | ------ | ----------- | -------- | -------- | -------- |
| 2-Layer Seq.        | 37 K   | 1.3         | 1.0      | **1.3×** | 21.4 %   |
| 3-Layer Seq.        | 166 K  | 3.5         | 2.8      | **1.3×** | 19.3 %   |
| 5-Layer Deep        | 189 K  | 8.7         | 7.0      | **1.2×** | 4.4 %    |
| 10-Layer Very Deep  | 786 K  | 4.0         | 3.1      | **1.3×** | 0 %      |
| Strided Encoder     | 388 K  | 1.2         | 0.9      | **1.4×** | 0 %      |
| Grouped 3-Layer     | 33 K   | 3.2         | 2.5      | **1.3×** | 19.5 %   |
| **Small U-Net**     | 195 K  | 3.3         | 1.5      | **2.2×** | 3.1 %    |
| Medium U-Net        | 777 K  | 4.5         | 3.6      | **1.2×** | 3.5 %    |
| 3-Layer + LeakyReLU | 166 K  | 3.5         | 2.8      | **1.3×** | 16.1 %   |
| 3-Layer + GELU      | 166 K  | 3.5         | 2.8      | **1.3×** | 16.1 %   |
| High-Res 2-Layer    | 9 K    | 2.4         | 1.8      | **1.3×** | 20.9 %   |

</details>

---

## 4. Multilayer **3-D** Network Benchmarks

### Speed Performance

| Metric              | Value                         |
| ------------------- | ----------------------------- |
| **Average Speedup** | **2.98 ×**                    |
| **Maximum Speedup** | **6.82 ×** (High-Res Spatial) |
| **Minimum Speedup** | **1.85 ×**                    |
| **Std Dev**         | 1.35 ×                        |

### Memory Performance

| Metric                | Value                    |
| --------------------- | ------------------------ |
| **Average Reduction** | **2.1 %**                |
| **Maximum Reduction** | **20.9 %** (Grouped 3-D) |
| **Minimum Reduction** | **0 %**                  |

<details>
<summary>Detailed Per-Network Results</summary>

| Network              | L | Params | Voxels | NVIDIA (ms) | Opt (ms) | Speedup  | Memory Δ   |
| -------------------- | - | ------ | ------ | ----------- | -------- | -------- | ---------- |
| 2-Layer Video Proc.  | 2 | 28 K   | 1.0 M  | 1.1         | 0.4      | **2.5×** | 0 %        |
| 3-Layer Temporal     | 3 | 31 K   | 1.0 M  | 2.6         | 1.0      | **2.6×** | 0 %        |
| 4-Layer Deep 3-D     | 4 | 32 K   | 2.1 M  | 11.1        | 3.9      | **2.8×** | 0 %        |
| Temporal Strided     | 2 | 17 K   | 1.0 M  | 1.3         | 0.5      | **2.8×** | 0 %        |
| Grouped 3-D          | 2 | 7 K    | 1.0 M  | 1.3         | 0.7      | **2.0×** | **20.9 %** |
| Small Video U-Net    | 8 | 147 K  | 98 K   | 3.6         | 1.6      | **2.2×** | 0 %        |
| Medium Video U-Net   | 8 | 584 K  | 786 K  | 10.5        | 5.7      | **1.9×** | 0 %        |
| 2-Layer + LeakyReLU  | 2 | 7 K    | 524 K  | 1.0         | 0.3      | **3.1×** | 0 %        |
| Long Temporal Seq.   | 2 | 2 K    | 131 K  | 1.2         | 0.4      | **3.3×** | 0 %        |
| **High-Res Spatial** | 2 | 0 K    | 524 K  | 2.0         | 0.3      | **6.8×** | 0 %        |

</details>

---

### Specific Optimizations

1.  **Mask Sum-Pooling Weight Generation:**
    *   **Nvidia:** Stores a `weight_maskUpdater` tensor (composed of ones) as an instance variable. This tensor is moved to the input's device/type on the first forward pass if they don't match.
    *   **Optimized:** Dynamically creates the `conv_weight` (tensor of ones for mask sum-pooling) on-the-fly within the `_compute_mask_updates` method. This weight is immediately created on the correct `mask.device` and with `mask.dtype`.
    *   **Benefit:** Reduces persistent memory footprint by not storing `weight_maskUpdater`. Avoids potential overhead of device/type transfer for `weight_maskUpdater` during the forward pass. Ensures the temporary convolution weight for mask processing is always correctly configured.

2.  **Mask Caching Mechanism:**
    *   **Nvidia:** Caches `update_mask` and `mask_ratio` based on the input tensor's shape (`last_size`) and whether `mask_in` was provided. Recomputes if `mask_in` is new or input shape changes.
    *   **Optimized:** Implements a more precise caching strategy in `_compute_mask_updates`. It caches the `update_mask` and `mask_ratio` based on the input `mask`'s shape *and* its `data_ptr()`.
    *   **Benefit:** Provides more reliable caching. The cache is hit only if the exact same mask tensor object (not just a tensor of the same shape) is reused, preventing unnecessary recomputations and ensuring correctness if different masks of the same shape are used sequentially.

3.  **Refined `multi_channel` Logic and Grouped Convolution Support for Mask Processing:**
    *   **`slide_winsize` Calculation:**
        *   Nvidia: `slide_winsize` is based on the dimensions of `weight_maskUpdater`. For `multi_channel=True`, this effectively means `in_channels * kernel_height * kernel_width`.
        *   Optimized: Explicitly calculates `slide_winsize` as `kernel_elements * (self.in_channels // self.groups)` if `multi_channel=True`, and `kernel_elements * 1` otherwise.
        *   Benefit: Provides a more accurate `slide_winsize` for grouped convolutions when `multi_channel=True`, correctly reflecting the number of elements summed per group in the mask.
    *   **Mask Sum-Pooling Convolution for `update_mask`:**
        *   Nvidia: Always uses `groups=1` for the `F.conv2d` operation that computes `update_mask`.
        *   Optimized: If `multi_channel=True` and the main convolution is grouped (`self.groups > 1`), the `F.conv2d` operation for mask sum-pooling also uses `groups=self.groups`. The `conv_weight` for this operation is shaped `(self.groups, channels_per_group, k, k)`.
        *   Benefit: Ensures that for grouped convolutions with `multi_channel=True`, the mask channels are summed appropriately within their respective groups, aligning the mask processing with the main convolution's structure. This is crucial for correct normalization.

4.  **Optimized Tensor Operations and State Management:**
    *   **In-place Operations:** Where appropriate (e.g., applying `mask_ratio`, `bias`, and `update_mask` to the output), the optimized version uses in-place PyTorch operations (`mul_`, `add_`).
    *   **Benefit:** Can reduce memory allocations for intermediate tensors and potentially improve performance.
    *   **Bias View Buffer:** The `_bias_view` tensor is registered as a persistent buffer using `self.register_buffer()`.
    *   **Benefit:** Ensures `_bias_view` is correctly handled as part of the model's state (e.g., moved to devices with `model.to(device)`, included in `state_dict`).

5.  **Clearer Input Mask Application:**
    *   **Nvidia:** The logic for multiplying the input tensor by the mask before the main convolution was `torch.mul(input, mask) if mask_in is not None else input`. If `mask_in` was `None`, a default `mask` of ones was created for calculating `update_mask` and `mask_ratio`, but the raw `input` (not `input * mask`) was passed to the convolution.
    *   **Optimized:** Always applies an element-wise mask multiplication: `masked_input = input_tensor * current_mask_for_input_mult`. If the input `mask` is `None`, `current_mask_for_input_mult` defaults to a tensor of ones matching the input tensor's spatial dimensions (and potentially channel dimensions based on `multi_channel`).
    *   **Benefit:** While numerically equivalent when `mask_in is None` (as `input * ones_mask == input`), the optimized approach makes the input masking step explicit and consistent with the partial convolution formulation `W^T (x⊙M)`. It also provides clearer handling for expanding/selecting mask channels based on `self.multi_channel`.
