# How It Works

This page explains the design of the delay-and-sum kernels in ffdas: why GPU implementations of DAS tend to underperform, and what ffdas does differently. Understanding these internals is not required to use the library, but it helps when choosing algorithm variants and reasoning about performance. For practical tuning advice, see the [performance guide](performance.md).

For a detailed treatment with profiling data and roofline analysis, see the accompanying paper:

> L. Verhoef and P. Kruizinga, "ffdas: Volumetric ultrasound reconstruction at warp speed," arXiv preprint arXiv:2606.13259, 2026. [https://arxiv.org/abs/2606.13259](https://arxiv.org/abs/2606.13259)

If you use ffdas in your work, please cite this paper:

```bibtex
@article{verhoef2026ffdas,
    title   = {ffdas: Volumetric ultrasound reconstruction at warp speed},
    author  = {Verhoef, Luuk and Kruizinga, Pieter},
    journal = {arXiv preprint arXiv:2606.13259},
    year    = {2026},
    eprint  = {2606.13259},
    archivePrefix = {arXiv},
    primaryClass  = {physics.med-ph},
    url     = {https://arxiv.org/abs/2606.13259},
}
```

## The computational challenge

Delay-and-sum reconstructs an image by, for each target voxel, interpolating and summing aligned samples from every receiver and transmit event. On a 128³ grid with a 32×32 matrix array and 16 transmit events, a single volume requires roughly $128^3 \times 1024 \times 16 \approx 3.4 \times 10^{10}$ accumulation operations. At kilohertz volume rates, this pushes sustained throughput to the limits of even high-end consumer GPUs.

A natural thought is to precompute the delay from every receiver to every voxel and store these in an array. At this scale, that would take $N \times M \times Q \times 4$ bytes $\approx 128$ GiB — impractical on most systems. ffdas instead computes all delays in-kernel from the receiver and voxel positions, which keeps memory requirements manageable and makes the reconstruction independent of any precomputed delay table.

## Why memory access is the bottleneck

The delay-and-sum computation is embarrassingly parallel: each voxel is independent, so it maps directly onto GPU threads. Yet standard implementations achieve only a fraction of the GPU's theoretical throughput. The reason is not insufficient compute capacity — even a consumer GPU has far more arithmetic capacity than this workload requires. The bottleneck is in how the data is accessed.

GPUs transfer data from memory in fixed 32-byte sectors. When a warp (a group of 32 threads executing in lockstep) issues memory requests, the hardware loads whichever sectors contain the requested addresses. If many threads request nearby addresses within the same sector, most of the loaded bytes are wasted. This fraction of loaded bytes that are actually used is called **payload efficiency**.

In the standard DAS approach, each thread handles a different voxel. Because neighboring voxels in a regular grid have similar distances to any given receiver, threads in a warp tend to request nearly the same sample index from each channel. In a layout where samples are contiguous along the time axis, these requests map to overlapping addresses within the same few sectors. The hardware loads those sectors, but each 32-byte sector contains consecutive time samples that no other thread needs. Payload efficiency in this configuration is roughly 40% — more than half of all transferred data goes unused.

The GPU's caches partially compensate: since threads request similar indices, the L1 and L2 caches serve most requests without going to DRAM. Cache hit rates can exceed 99%. But the fundamental problem remains: the many small, fragmented memory transactions stall the execution pipeline. Threads spend most of their time waiting for the memory system to process individually small requests, even though the data they need is often already in cache. The issue is not bandwidth but transaction overhead.

## Three optimization strategies

ffdas addresses this with three strategies that progressively improve memory utilization and arithmetic throughput. Each corresponds to an algorithm variant in the `Algorithm` enum.

### ALG1: per-voxel baseline

`ALG1` uses the straightforward approach described above: each thread processes one voxel, iterating over all receivers and transmit events. It tiles over a small block of consecutive batch items per thread to amortize the geometric calculations (delay and weight computation) across frames.

The input data uses the standard `(batch, channels, sequence, samples)` layout with samples contiguous. No data permutation is required.

This is the simplest variant and the one with the lowest memory overhead. It supports all data types including `float64` and `complex128`, and does not require a batch dimension. Its throughput is limited by the fragmented memory access pattern described above, but for small problems, single-frame reconstruction, or when GPU memory is scarce, it is the appropriate choice.

### ALG2: batch-inner tiling

`ALG2` rearranges the problem so that memory access aligns with the GPU's transfer granularity.

Instead of assigning each thread its own voxel, multiple threads in a warp process the **same voxel** for **different batch items**. The input data is permuted to place the batch index as the innermost (contiguous) dimension: `(channels, sequence, samples, batch)`. Now, when a group of threads needs the same sample from the same receiver, they each access a different batch item at that sample position. Because the batch dimension is contiguous, these accesses map to consecutive memory addresses, and the hardware can coalesce them into wide transactions that fully utilize each 32-byte sector.

This raises payload efficiency from ~40% to 100%. The GPU generates fewer but wider memory requests, and threads spend less time stalled waiting for the memory system. The permutation from the user's input layout to the batch-inner layout is done internally as part of the kernel launch, and for moderate and large data sizes its cost is negligible compared to the reconstruction itself.

The warp is divided into groups of threads, where each group shares a voxel and processes a tile of batch items. Within a group, threads that share a voxel can distribute geometric calculations through warp shuffles, avoiding redundant delay and weight computation.

For typical configurations, `ALG2` is roughly 40% faster than `ALG1`, with an additional ~20% improvement when combined with half-precision input storage (see below). It requires a batch dimension and the batch size must be a multiple of the group's vector width.

### ALG4: tensor core acceleration

`ALG4` takes the memory efficiency of the batch-inner layout and adds tensor core acceleration to increase arithmetic throughput.

Modern GPUs provide tensor cores — specialized matrix multiply-accumulate (MMA) units that operate on small dense matrix tiles with significantly higher throughput than scalar floating-point units. DAS does not naturally map to a dense matrix operation, since each voxel needs only two samples per receiver. But spatial locality makes it possible to repackage the computation into dense tiles.

For a group of neighboring voxels, the required sample indices from any given receiver cluster tightly: because the voxels are close in space, their path lengths to the receiver are similar, so their sample indices differ by at most a few positions. `ALG4` exploits this by processing receivers in tiles and iterating over the distinct sample positions that appear within each tile. For each unique sample index, it assembles a weight matrix (voxels × receivers) and a sample matrix (receivers × batch items), then multiplies them with an MMA instruction. Accumulating over all unique indices, receiver tiles, and transmit events yields the final image.

The tighter the sample indices cluster within a voxel tile, the fewer MMA iterations are needed and the higher the throughput. Clustering depends on imaging geometry: voxels far from the array have similar path lengths to each receiver and share most sample indices, while voxels close to the array show more variation. This means the tensor core advantage is largest for deep imaging and diminishes near the transducer surface.

`ALG4` provides the highest throughput of the three variants, particularly with FP16 input storage where tensor core units are optimized for mixed-precision processing. Like `ALG2`, it requires a batch dimension and permutes the data internally.

!!! note
    `ALG4` uses WMMA instructions and requires compute capability 7.0 (Volta) or higher.

### Mixed-precision storage

Orthogonal to the algorithm choice, ffdas supports storing input samples in 16-bit floating-point (FP16) format. Most practical ultrasound systems produce data well within FP16's precision and dynamic range. Halving the storage per sample doubles the effective memory bandwidth.

Samples are converted to 32-bit precision before interpolation and accumulation, so the output maintains full FP32 accuracy. The conversion can be fused with the data permutation, so the overhead is minimal.

The benefit of FP16 depends on the algorithm. `ALG1`'s poor payload efficiency means it wastes most transferred bytes regardless of element size, so halving the element size helps less. `ALG2` and `ALG4` already achieve full payload efficiency, so FP16 translates directly into proportionally less data transfer.

In Python, FP16 input is enabled by passing `use_fp16=True` to `das` or `das_sparse`, or by constructing a `TensorView` with `ffdas.view(array, ffdas.half)` or `ffdas.view(array, ffdas.half2)` for complex data.

## DEFAULT algorithm selection

`Algorithm.DEFAULT` currently dispatches to `ALG1`. It does not auto-select based on the problem configuration. If you want `ALG2` or `ALG4`, you must request them explicitly. Auto-selection based on batch size, grid dimensions, and hardware capabilities is planned for a future release.

## Other operations

The optimization strategies above apply specifically to `das` and `das_sparse`. The other operations in ffdas use different approaches appropriate to their computation:

- **`greens`** uses WMMA (tensor core) instructions directly, as the frequency-domain summation maps naturally to dense matrix operations. It requires SM 70+.
- **`truncate_rank`** dispatches to cuSOLVER for the SVD and cuBLAS for the reconstruction, so its performance is determined by those libraries.
- **`einsum`** dispatches to cuBLAS GEMM after mapping the contraction to a matrix multiply. Performance is essentially cuBLAS throughput for the given tile sizes.
- **`interpolate`** builds a spatial hash over the grid vertices for fast point location, then evaluates interpolation weights per query point.
- **`gather`** and **`scatter`** are straightforward GPU kernels with vectorized loads where possible.
