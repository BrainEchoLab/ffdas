# How It Works

This page explains the design of the delay-and-sum kernels in ffdas: why GPU implementations of delay-and-sum tend to underperform, and what ffdas does differently. Understanding these internals is not required to use the library, but it helps when choosing algorithm variants and reasoning about performance. For practical tuning advice, see the [performance guide](performance.md).

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

Delay-and-sum reconstructs an image by interpolating and summing (compounding) aligned samples from every receiver and transmit event for each target voxel. For volumetric imaging with matrix arrays, this amounts to tens of billions of accumulations per volume, often at kilohertz volume rates.

ffdas computes all delays in-kernel from the receiver and voxel positions rather than precomputing and storing them. A precomputed delay table for a typical volumetric configuration would require over 100 GiB.

## Why memory access is the bottleneck

Each voxel is independent, so DAS maps directly onto GPU threads. Standard implementations assign one thread per voxel and iterate over receivers and transmit events. The limiting factor in this approach is memory access efficiency.

GPU memory transfers happen in fixed 32-byte sectors. When threads in a warp request addresses that fall within the same few sectors, most loaded bytes go unused. The ratio of used to transferred bytes is the **payload efficiency**.

In the standard per-voxel approach, neighboring threads request nearly the same sample index from each channel because their voxels are close in space. With a time-contiguous sample layout, these requests map to overlapping sectors containing consecutive time samples that no other thread needs. Payload efficiency in this configuration is around 40%.

The L1 and L2 caches compensate partially, with hit rates exceeding 99%. But the large number of small, fragmented memory transactions still stalls the execution pipeline. The limiting factor is transaction overhead rather than raw bandwidth. See the paper for profiling data and roofline analysis.

## Three optimization strategies

ffdas provides three algorithm variants that progressively improve memory utilization and arithmetic throughput.

### ALG1: per-voxel baseline

`ALG1` assigns each thread one voxel and iterates over all receivers and transmit events. It tiles over a small block of consecutive batch items per thread to amortize geometric calculations across frames.

The input data uses the standard `(batch, channels, sequence, samples)` layout with samples contiguous. No data permutation is required.

`ALG1` is the simplest variant and has the lowest memory overhead. It supports most data types, including `float64` and `complex128`, and does not require a batch dimension. Its throughput is limited by the fragmented access pattern described above, but for small problems, single-frame reconstruction, or limited GPU memory, it is the appropriate choice.

### ALG2: batch-inner tiling

`ALG2` rearranges the data so that memory access aligns with the GPU's transfer granularity.

Instead of each thread processing its own voxel, multiple threads in a warp process the **same voxel** for **different batch items**. The input data is permuted to place the batch index as the innermost (contiguous) dimension: `(channels, sequence, samples, batch)`. When threads sharing a voxel access the same sample from the same receiver, they each read a different batch item at that position. Because the batch dimension is contiguous, these reads map to consecutive addresses and coalesce into wide transactions that fully utilize each 32-byte sector.

This raises payload efficiency from around 40% to around 100%. The permutation is done internally as part of the kernel launch and is negligible for moderate and large data sizes.

The warp is divided into groups that share a voxel and process a tile of batch items. Within a group, threads distribute geometric calculations through warp shuffles, avoiding redundant delay and weight computation.

`ALG2` is roughly 40% faster than `ALG1`, with an additional ~20% improvement when combined with half-precision input storage. It requires a batch dimension and the batch size must be a multiple of the group's vector width.

### ALG4: tensor core acceleration

`ALG4` combines the batch-inner layout with tensor core (MMA) instructions to increase arithmetic throughput.

DAS does not naturally map to a dense matrix operation since each voxel needs only two samples per receiver. But for a tile of neighboring voxels, the required sample indices from a given receiver cluster within a narrow range because the voxels have similar path lengths. `ALG4` processes receivers in tiles and iterates over the distinct sample positions within each tile. For each unique sample index, it assembles a weight matrix (voxels x receivers) and a sample matrix (receivers x batch items), multiplies them with an MMA instruction, and accumulates the result. Iterating over all unique indices, receiver tiles, and transmit events yields the final image.

Tighter clustering of sample indices means fewer MMA iterations. This is geometry-dependent: voxels far from the array share most sample indices, while voxels close to the array show more variation. The advantage of `ALG4` over `ALG2` is largest for deep imaging and smallest near the transducer surface.

`ALG4` provides the highest throughput of the three variants, particularly with FP16 input where tensor cores operate at peak mixed-precision throughput. Like `ALG2`, it requires a batch dimension and permutes the data internally.

!!! note
    `ALG4` uses WMMA instructions and requires compute capability 7.0 (Volta) or higher.

### Mixed-precision storage

Orthogonal to the algorithm choice, ffdas supports storing input samples in half-precision (FP16). Most ultrasound acquisition systems produce data within FP16's dynamic range. Halving the storage per sample doubles the effective memory bandwidth.

Samples are converted to FP32 before interpolation and accumulation, so the output maintains full single-precision accuracy. The conversion can be fused with the data permutation at negligible overhead.

The benefit depends on the algorithm. `ALG1`'s low payload efficiency means it wastes most transferred bytes regardless of element size, so halving the element size helps less. `ALG2` and `ALG4` achieve full payload efficiency, so FP16 translates directly into proportionally less data transfer.

In Python, FP16 input is enabled by passing `use_fp16=True` to `das` or `das_sparse`, or by constructing a `TensorView` with `ffdas.view(array, ffdas.half)` or `ffdas.view(array, ffdas.half2)` for complex data.

## DEFAULT algorithm selection

`Algorithm.DEFAULT` currently dispatches to `ALG1`. It does not yet auto-select based on the problem configuration. If you want `ALG2` or `ALG4`, you must request them explicitly. Auto-selection based on batch size, grid dimensions, and hardware capabilities is planned for a future release.

## Other operations

The optimization strategies above apply specifically to `das` and `das_sparse`. The other operations in ffdas use different approaches:

- **`greens`** uses WMMA instructions directly, as the frequency-domain summation maps naturally to dense matrix operations. It requires SM 70+.
- **`truncate_rank`** dispatches to cuSOLVER and cuBLAS for the SVD (or eigen decomposition) and reconstruction, so its performance is determined by those libraries.
- **`einsum`** dispatches to cuBLAS GEMM after mapping the contraction to a matrix multiply. Performance is essentially cuBLAS throughput for the given tile sizes.
- **`interpolate`** builds a spatial hash over the grid vertices for fast point location, then evaluates interpolation weights per query point.
- **`gather`** and **`scatter`** are straightforward GPU kernels with vectorized loads where possible.
