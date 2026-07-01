# ffdas

This package contains the Python bindings for ffdas, a CUDA-accelerated delay-and-sum 
and related primitives for image reconstruction in high frame rate ultrasound, photoacoustics, 
and similar domains. 
The bindings integrate directly with GPU array libraries (CuPy, PyTorch) via zero-copy interop.

## Installation

```bash
pip install ffdas[cuda12]   # or ffdas[cuda13]
```

Requires Python 3.12+ and a CUDA-capable GPU (compute capability 5.3+).
CuPy or another DLPack-compatible GPU array library is needed to create input
arrays.

## Example

```python
import cupy as cp
import ffdas

# reconstruct a volume from acquired channel data
output = ffdas.das(
    channel_data,  # (batch, channels, sequence, samples)
    channel_pos,   # (channels, 3)
    voxel_pos,     # (..., 3), e.g., (nz, ny, nx, 3)
    delays,        # (sequence, ...)
    weights,       # (sequence, ...)
)

# remove tissue clutter for flow imaging
filtered = ffdas.truncate_rank(output, start=64)

# interpolate to arbitrary positions
interpolated = ffdas.interpolate(
    voxel_pos,     # (nz, ny, nx, 3)
    filtered,      # (batch, nz, ny, nx)
    interp_pos,    # (..., 3)
)
```

## Operations

The library provides GPU-accelerated implementations of delay-and-sum
beamforming (`das`, `das_sparse`), Green's function summation (`greens`),
rank truncation (`truncate_rank`), tensor contraction (`einsum`), structured
grid interpolation (`interpolate`), and gather/scatter primitives.

## Links

- [Documentation](https://brainecholab.github.io/ffdas/)
- [GitHub](https://github.com/BrainEchoLab/ffdas)
- [Paper (arXiv:2606.13259)](https://arxiv.org/abs/2606.13259)
