# Performance Guide

This page provides practical guidance for getting the most out of ffdas. It covers algorithm selection, precision, batch sizing, and scaling behavior. For the underlying design rationale, see [how it works](how-it-works.md).

## Algorithm selection

The `algorithm` parameter on `das` and `das_sparse` selects the kernel variant. The choice affects throughput, memory usage, and which data types and configurations are supported.

| Variant | Layout permutation | Batch required | float64 | SM requirement | Relative speed |
|---|---|---|---|---|---|
| `ALG1` | None | No | Yes | SM 53+ | 1× (baseline) |
| `ALG2` | Yes (internal) | Yes | No | SM 53+ | ~1.4× |
| `ALG4` | Yes (internal) | Yes | No | SM 70+ | ~1.6–2× with FP16 |

`DEFAULT` currently dispatches to `ALG1` unconditionally. It does not yet auto-select.

**When to use which:**

`ALG1` is the right choice when you are reconstructing a single frame (no batch dimension), when you need `float64` or `complex128` precision, or when GPU memory is tight. It uses the standard input layout and avoids a temporary allocation for a permuted copy if possible. It is also the simplest path for verifying correctness before optimizing.

`ALG2` is the recommended default for batched reconstruction. If you can process more than one frame at a time and your data fits in GPU memory, `ALG2` will be faster than `ALG1` in virtually all configurations. The batch size must be a multiple of the internal vector width (typically 4 or 8 depending on the data type).

`ALG4` provides the highest throughput, particularly with FP16 inputs. Its advantage over `ALG2` is most pronounced for large, dense voxel grids where spatial locality ensures tight clustering of sample indices across neighboring voxels. The speedup is typically geometry-dependent: voxels far from the array benefit most (tight index clustering), while voxels close to the array see less improvement (larger index range across subsequent voxels). For typical imaging geometries where most of the volume is at moderate to large depth, `ALG4` with FP16 input is the fastest option available.

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

## Data layout

### channels_trailing

The `channels_trailing` parameter controls how the channel data dimensions are interpreted. The C API always expects `(batch, channels, sequence, samples)` with samples contiguous. When `channels_trailing` is `True`, the binding swaps the channels and sequence dimensions in the tensor descriptor, so you can pass data in `(batch, sequence, channels, samples)` order. Note that this may trigger a contiguous copy of the input data internally.

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

**`greens`**: uses tensor core instructions natively. Performance scales with source count * frequency count * target count. The kernel is compute-bound once the source and target counts are large enough. Requires SM 70+, but a scalar fallback option exists for SM 53+.

**`truncate_rank`**: dominated by the cuSOLVER SVD, which scales as $O(\min(m,n)^2 \cdot \max(m,n))$ where $m$ is the number of frames and $n$ is the product of spatial dimensions. For a 64-frame sequence on a 64³ grid, $m = 64$ and $n \approx 2.6 \times 10^5$, so the SVD processes a tall-skinny matrix. The reconstruction step (matrix multiply of the selected singular vectors) is fast by comparison.

**`einsum`**: dispatches to cuBLAS GEMM. Performance matches cuBLAS for the equivalent matrix dimensions after the contraction modes are mapped to a GEMM problem.

**`interpolate`**: the initial `Interpolator` construction builds a spatial hash over the grid, which takes a fraction of a millisecond for typical grid sizes. Each subsequent interpolation call performs a point-location lookup per query point followed by weight evaluation. Using `preprocess=True` caches the lookup results so that subsequent calls with different value arrays skip the point-location step — this is relevant when interpolating many frames on the same grid and query positions.

**`gather` / `scatter`**: simple GPU kernels. Vectorized loads are used when alignment and data type permit. Performance is bandwidth-limited.

## Stream management

The Python API currently uses a single internal CUDA stream per device, created automatically on first use. There is no public API to set or query the stream. For applications that need to overlap ffdas calls with other GPU work or enforce ordering with respect to external streams, use the C API directly, which provides `ffdas_set_stream` and `ffdas_get_stream` on the library handle.

This is a known limitation of the Python bindings. Stream-aware Python APIs are planned for a future release.
