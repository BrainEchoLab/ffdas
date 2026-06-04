# API Reference

All functions accept GPU arrays from any DLPack-compatible library (CuPy, PyTorch, MATLAB gpuArrays). Position and offset arrays are cast to `float32` internally; index arrays are cast to `int32`. See [conventions](../conventions.md) for the unit system and array layout.

## Core Operations

| Function | Description |
|---|---|
| [`das`](das.md) | Delay-and-sum, compounding over the sequence dimension |
| [`das_sparse`](das.md#das_sparse) | Delay-and-sum with per-target sequence subsets |
| [`greens`](greens.md) | Frequency-domain Green's function summation (requires SM 70+) |
| [`eigfilter`](eigfilter.md) | Eigenspace-based clutter filtering |
| [`einsum`](einsum.md) | Binary tensor contraction |
| [`interpolate`](interpolation.md) | Structured 3D grid interpolation |
| [`Interpolator`](interpolation.md#interpolator) | Reusable interpolation plan (Python only) |

## Tensor Operations

| Function | Description |
|---|---|
| [`gather`](tensor.md#gather) | Gather elements along a dimension |
| [`scatter`](tensor.md#scatter) | Scatter elements along a dimension |
| [`contiguous_copy`](tensor.md#contiguous_copy) | Copy a strided tensor into a contiguous buffer |

## Utilities

| Function | Description |
|---|---|
| [`cdist`](utilities.md#cdist) | Pairwise Euclidean distance |
| [`rect_dist`](utilities.md#rect_dist) | Distance from points to an axis-aligned rectangle |
| [`angle`](utilities.md#angle) | Angle between vectors |
| [`Timer`](utilities.md#timer) | GPU timing via CUDA events (Python only) |
