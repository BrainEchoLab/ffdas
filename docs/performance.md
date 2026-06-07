# Performance Guide

This page provides practical guidance for getting the most out of ffdas. It covers algorithm selection, precision, batch sizing, and scaling behavior. For the underlying design rationale, see [how it works](how-it-works.md).

## Algorithm selection

The `algorithm` parameter on `das` and `das_sparse` selects the kernel variant. The choice affects throughput, memory usage, and which data types and configurations are supported.

| Variant | Layout permutation | Batch required | float64 | SM requirement | Relative speed |
|---|---|---|---|---|---|
| `ALG1` | None | No | Yes | SM 70+ | 1× (baseline) |
| `ALG2` | Yes (internal) | Yes | No | SM 70+ | ~1.4× |
| `ALG4` | Yes (internal) | Yes | No | SM 70+ | ~1.6–2× with FP16 |

`DEFAULT` currently dispatches to `ALG1` unconditionally. It does not auto-select.

**When to use which:**

`ALG1` is the right choice when you are reconstructing a single frame (no batch dimension), when you need `float64` or `complex128` precision, or when GPU memory is tight — it uses the standard input layout and avoids the temporary allocation for the permuted copy. It is also the simplest path for verifying correctness before optimizing.

`ALG2` is the recommended default for batched reconstruction. If you are processing more than one frame at a time and your data fits in GPU memory, `ALG2` will be faster than `ALG1` in virtually all configurations. The batch size must be a multiple of the internal vector width (typically 4 or 8 depending on the data type).

`ALG4` provides the highest throughput, particularly with FP16 inputs, and is the variant benchmarked in the paper. Its advantage over `ALG2` is most pronounced for large, dense voxel grids where spatial locality ensures tight clustering of sample indices within each warp tile. The speedup is geometry-dependent: voxels far from the array benefit most (tight index clustering), while voxels close to the array see less improvement (more index spread per tile). For typical imaging geometries where most of the volume is at moderate to large depth, `ALG4` with FP16 input is the fastest option available.

`ALG3` is reserved and not currently implemented.

=== "Python"

    ```python
    from ffdas import Algorithm

    # explicit algorithm selection
    output = ffdas.das(rf, srcpos, dstpos, offsets, weights,
                       algorithm=Algorithm.ALG4, use_fp16=True)
    ```

=== "MATLAB"

    ```matlab
    % algorithm is an int32: 0=DEFAULT, 1=ALG1, 2=ALG2, 4=ALG4
    output = ffdas.das(rf, srcpos, dstpos, offsets, weights, [], wavenum, int32(4), true);
    ```

## Precision

### Input precision

ffdas supports three input precisions for `das`:

**FP16 (half-precision)** halves memory traffic compared to FP32 and doubles the effective bandwidth for the optimized algorithms. Most ultrasound acquisition systems produce data well within FP16's dynamic range (~3.3 decimal digits, ±65504). The output is always accumulated in FP32 regardless of input precision, so the final image maintains full 32-bit accuracy. FP16 provides the largest speedup when combined with `ALG2` or `ALG4`.

In Python, set `use_fp16=True` to have the library convert FP32 inputs to FP16 internally, or provide a `TensorView` wrapping half-precision data directly:

=== "Python"

    ```python
    # automatic conversion (fused with layout permutation)
    output = ffdas.das(rf, srcpos, dstpos, offsets, weights, use_fp16=True)

    # manual: wrap a float16 array as a half TensorView
    rf_half = ffdas.view(cp.asarray(rf, dtype=cp.float16), ffdas.half)
    output = ffdas.das(rf_half, srcpos, dstpos, offsets, weights)

    # complex IQ data: use half2 view
    rf_half2 = ffdas.view(cp.asarray(rf.view(cp.float16).reshape(*rf.shape, 2)), ffdas.half2)
    ```

=== "MATLAB"

    ```matlab
    % set use_fp16 to true (8th positional argument)
    output = ffdas.das(rf, srcpos, dstpos, offsets, weights, [], wavenum, int32(0), true);
    ```

**FP32 (single-precision)** is the default and sufficient for the vast majority of ultrasound imaging. All internal geometric calculations (delays, weights, distances) use FP32 regardless of input precision.

**FP64 (double-precision)** is supported only by `ALG1`. Spatial operations (`das`, `greens`, `interpolate`) use `float3`/`float4` position types internally, so positions are always single-precision even when the signal data is double. Double precision is primarily useful for algebraic operations like `einsum` and `truncate_rank` where the computation does not involve spatial coordinates.

### Compute precision

The `use_fp16` flag on `das` controls only the *storage* format, and accumulation always happens in FP32 (or FP64 for double-precision inputs). 
<!-- 
## Batch sizing

The batch dimension is the primary lever for GPU utilization with `ALG2` and `ALG4`. These algorithms tile over the batch dimension so that threads in a warp access consecutive batch items at each sample position. Larger batches mean wider coalesced memory transactions and better amortization of per-voxel geometric calculations (delay and weight computation are shared across all batch items for a given voxel).

Practical guidance:

- **Batch 1** (no batch dimension): only `ALG1` is available.
- **Batch 8–32**: `ALG2` becomes effective. The GPU is partially saturated; throughput scales roughly linearly with batch size in this range.
- **Batch 64–256**: both `ALG2` and `ALG4` reach near-peak throughput. The per-voxel-per-frame cost plateaus as the GPU becomes fully saturated.
- **Batch >256**: throughput continues to scale linearly (more frames at the same per-frame cost). Memory becomes the main constraint.

The batch size must be a multiple of the algorithm's internal vector width. `ALG2` and `ALG4` will return an error if the batch size is not divisible by this value (typically 4 for complex data, 8 for real data). If your natural batch size doesn't divide evenly, pad with zeros.

## Scaling behavior

Performance scales near-linearly with problem dimensions once the GPU is saturated:

**Grid size** (number of voxels): linear. Doubling the grid doubles the runtime. This is expected since each voxel is independent and the workload is dominated by the per-voxel sum over channels.

**Array size** (number of receivers): linear. Each additional receiver adds one distance calculation and one sample interpolation per voxel per transmit event.

**Sequence events**: linear. DAS processes each transmit event independently and sums the results. Achievable frame rates for compounding with $Q$ events are approximately $1/Q$ times the single-event rate. For example, a configuration that achieves 5 kHz with $Q = 1$ reaches ~300 Hz with $Q = 16$.

**Sample count**: affects input data size and memory footprint but not the per-voxel cost, since each voxel interpolates at most one sample per receiver per event regardless of $K$.

**GPU architecture**: the optimizations in ffdas are designed around the memory hierarchy of NVIDIA GPUs (warp-level coalescing, sector-aligned access, tensor core MMA tiles). Performance characteristics have been validated on Ada Lovelace (RTX 40-series) but the principles apply to Volta, Turing, and Ampere as well. Different architectures have different sector sizes, cache hierarchies, and tensor core tile dimensions, so absolute throughput will vary. Hopper and Blackwell architectures may benefit from different tile configurations. -->

## Data layout

### channels_trailing

The `channels_trailing` parameter controls how the channel data dimensions are interpreted. The C API always expects `(batch, channels, sequence, samples)` with samples contiguous. When `channels_trailing` is `True`, the binding swaps the channels and sequence dimensions in the tensor descriptor, so the user can pass data in `(batch, sequence, channels, samples)` order.

In Python, the default is `False` (channels before sequence). In MATLAB, the default is `True` (channels after sequence, which after the column-major reversal means channels are the faster-varying dimension).

For `ALG2` and `ALG4`, the library permutes the data internally to the batch-inner layout regardless of `channels_trailing`, so the flag does not affect kernel throughput. It does affect the permutation cost (which is small) and whether the user needs to transpose their data before calling `das`.

### Contiguity

`ALG1` checks whether the input is C-contiguous and copies it if not. `ALG2` and `ALG4` always copy the input into the batch-inner layout, so contiguity of the input does not matter for those variants.

All position arrays (`srcpos`, `dstpos`) and parameter arrays (`offsets`, `weights`) are cast to contiguous `float32` internally if they are not already.

## Timing and profiling

### Timer

`ffdas.utils.Timer` records CUDA events on the library's internal stream, so it measures actual GPU execution time rather than host-side wall time:

=== "Python"

    ```python
    with ffdas.utils.Timer() as t:
        output = ffdas.das(rf, srcpos, dstpos, offsets, weights,
                           algorithm=Algorithm.ALG4, use_fp16=True)
    print(f"{t.elapsed_ms():.1f} ms")
    ```

=== "MATLAB"

    ```matlab
    t = ffdas.utils.Timer();
    t.start();
    output = ffdas.das(rf, srcpos, dstpos, offsets, weights);
    t.stop();
    fprintf("%.1f ms\n", t.elapsed_ms());
    ```

The timer synchronizes the stop event before returning, so `elapsed_ms()` reflects the completed kernel time. For benchmarking, run a warmup iteration before timing to avoid measuring JIT compilation and allocation overhead.

### NVTX annotations

ffdas emits NVTX range annotations for all major operations when built with `FFDAS_USE_NVTX=ON` (the default). These are visible in NVIDIA Nsight Systems and Nsight Compute, making it straightforward to profile ffdas calls alongside the rest of your application. The annotations are zero-cost when no profiling tool is attached.

### Nsight Compute

For kernel-level analysis, Nsight Compute can capture hardware performance counters. Useful metrics to look at:

- **L1 Payload Efficiency** (`l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio`): bytes used vs. bytes transferred. ALG1 ~40%, ALG2/ALG4 ~100%.
- **DRAM throughput** (`dram__bytes.sum.per_second`): how close to peak memory bandwidth.
- **Warp stall reasons** (`smsp__warps_issue_stalled_*`): Long Scoreboard (memory latency), Math Throttle (compute saturation), Branch Resolving (control flow).
- **Tensor core utilization** (for ALG4): fraction of cycles using MMA instructions.

## Performance of other operations

**`greens`**: uses tensor core instructions natively. Performance scales with source count × frequency count × target count. The kernel is compute-bound once the source and target counts are large enough. Requires SM 70+.

**`truncate_rank`**: dominated by the cuSOLVER SVD, which scales as $O(\min(m,n)^2 \cdot \max(m,n))$ where $m$ is the number of frames and $n$ is the product of spatial dimensions. For a 64-frame sequence on a 64³ grid, $m = 64$ and $n \approx 2.6 \times 10^5$, so the SVD processes a tall-skinny matrix. The reconstruction step (matrix multiply of the selected singular vectors) is fast by comparison.

**`einsum`**: dispatches to cuBLAS GEMM. Performance matches cuBLAS for the equivalent matrix dimensions after the contraction modes are mapped to a GEMM problem.

**`interpolate`**: the initial `Interpolator` construction builds a spatial hash over the grid, which takes a fraction of a millisecond for typical grid sizes. Each subsequent interpolation call performs a point-location lookup per query point followed by weight evaluation. Using `preprocess=True` caches the lookup results so that subsequent calls with different value arrays skip the point-location step — this is relevant when interpolating many frames on the same grid and query positions.

**`gather` / `scatter`**: simple GPU kernels. Vectorized loads are used when alignment and data type permit. Performance is bandwidth-limited.

## Stream management

The Python API currently uses a single internal CUDA stream per device, created automatically on first use. There is no public API to set or query the stream. For applications that need to overlap ffdas calls with other GPU work or enforce ordering with respect to external streams, use the C API directly, which provides `ffdas_set_stream` and `ffdas_get_stream` on the library handle.

This is a known limitation of the Python bindings. Stream-aware Python APIs are planned for a future release.

## Quick reference

For a 128³ grid, 32×32 array, batch 128, single transmit event ($Q = 1$), on an RTX 4080 Super:

| Configuration | Approximate throughput |
|---|---|
| ALG1, FP32 | ~170 volumes/s |
| ALG1, FP16 | ~190 volumes/s |
| ALG2, FP32 | ~260 volumes/s |
| ALG2, FP16 | ~320 volumes/s |
| ALG4, FP32 | ~370 volumes/s |
| ALG4, FP16 | ~1000 volumes/s |

These numbers are derived from profiling on a single GPU with locked clock speed. For $Q$ compounded transmit events, divide by $Q$. Actual throughput depends on GPU model, clock speed, imaging geometry, and thermal conditions. Measure on your own hardware with `Timer` for authoritative numbers.
