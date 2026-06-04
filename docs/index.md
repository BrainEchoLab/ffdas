# ffdas

ffdas is a CUDA-accelerated library of delay-and-sum and related primitives for image reconstruction in high frame rate ultrasound, photoacoustics, and similar domains. Python and MATLAB bindings integrate directly with GPU arrays (CuPy, PyTorch, MATLAB gpuArrays) via zero-copy interop.

## Getting Started

Install the Python package with pip:

```bash
pip install ffdas-cu13
```

For MATLAB, download the prebuilt MEX binaries from the [GitHub Releases](https://github.com/luukverhoef/ffdas/releases) page. For building from source, see the [installation guide](getting-started/installation.md).

The [quickstart](getting-started/quickstart.md) walks through computing transmit delays and apodization weights for a diverging-wave setup and passing them to `das`.

## Operations

The library provides GPU-accelerated implementations of:

- [`das`](api/das.md), [`das_sparse`](api/das.md#das_sparse) — delay-and-sum with multiple algorithm variants, sparse compounding, half-precision compute, and directivity masking.
- [`greens`](api/greens.md) — frequency-domain Green's function summation for wave propagation modeling. Requires SM 70+.
- [`eigfilter`](api/eigfilter.md) — eigenspace-based clutter filtering (truncated SVD reconstruction).
- [`einsum`](api/einsum.md) — binary tensor contraction using Einstein summation notation.
- [`interpolate`](api/interpolation.md) — structured 3D grid interpolation (nearest, linear).
- [`gather`, `scatter`](api/tensor.md) — index-based GPU tensor operations.
- [Spatial utilities](api/utilities.md) — `cdist`, `rect_dist`, `angle` for computing transmit geometry.

## Conventions

All positions and time offsets passed to `das` are in sampling wavelengths (`c / f_s`), avoiding per-element scaling on the GPU. Array layouts follow the conventions of each language: coordinate-last `(..., 3)` in Python, coordinate-first `(3, ...)` in MATLAB. See the [conventions](conventions.md) page for full details.
