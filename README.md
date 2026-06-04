# ffdas

ffdas is a CUDA-accelerated library of delay-and-sum and related primitives for image reconstruction in high frame rate ultrasound, photoacoustics, and similar domains. Python and MATLAB bindings integrate directly with GPU arrays (CuPy, PyTorch, MATLAB gpuArrays) via zero-copy interop.

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
filtered = ffdas.eigfilter(output, k0=64)

# interpolate to arbitrary positions
interpolated = ffdas.interpolate(
    voxel_pos,     # (nz, ny, nx, 3)
    filtered,      # (batch, nz, ny, nx)
    interp_pos,    # (..., 3)
)
```

## Installation

### Python

```bash
pip install ffdas-cu13
```

Requires Python 3.12+ and a CUDA-capable GPU. CuPy or another DLPack-compatible GPU array library is needed to create the input arrays.

### MATLAB

Download the prebuilt MEX binaries for your platform from the [GitHub Releases](https://github.com/luukverhoef/ffdas/releases) page and add the bindings directory to your MATLAB path:

```matlab
addpath("/path/to/ffdas/bindings/matlab")
```

Requires MATLAB R2018b+ with the Parallel Computing Toolbox.

### Building from Source

See the [installation guide](docs/getting-started/installation.md) for building from source, including platform-specific instructions and CMake options.

## Operations

The library provides GPU-accelerated implementations of:

- **Delay-and-sum** (`das`, `das_sparse`) with multiple algorithm variants, sparse compounding, half-precision compute paths, and per-channel directivity masking.
- **Green's function summation** (`greens`) for frequency-domain wave propagation modeling. Requires SM 70+ (Volta or newer).
- **Eigenspace filtering** (`eigfilter`) for subspace-based clutter rejection.
- **Tensor contraction** (`einsum`) for arbitrary binary contractions on GPU arrays.
- **Structured grid interpolation** (`interpolate`) with nearest-neighbor and linear modes.
- **Gather/scatter** and contiguous-copy utilities for GPU tensor manipulation.

Python also provides spatial utility functions (`cdist`, `rect_dist`, `angle`) for computing transmit geometry. The MATLAB bindings include these plus `take` and `take_along_axis`.

## Conventions

Positions and time offsets passed to `das` are in **sampling wavelengths** (`c / f_s`), making the kernel unitless. In practice this means scaling spatial coordinates by `f_s / c` and expressing time offsets in samples. `greens` and `interpolate` are unit-agnostic — positions just need to be consistent with the wavenumbers or query points respectively. The [quickstart](docs/getting-started/quickstart.md) walks through this conversion for a concrete transmit setup.

The Python and MATLAB bindings follow the layout conventions of their respective languages. In Python, arrays are row-major with the coordinate dimension last: positions have shape `(..., 3)`. In MATLAB, arrays are column-major with the coordinate dimension first: `(3, ...)`. The rest of the API is analogous between the two. See the [conventions reference](docs/conventions.md) for details.

## Examples

The [`bindings/python/examples/`](bindings/python/examples/) directory contains self-contained scripts that simulate data and demonstrate each part of the pipeline:

- **`reconstruct.py`** — End-to-end volume reconstruction: simulate diverging-wave channel data via Green's function propagation, compute transmit geometry, and reconstruct a 3D image with `das`.
- **`simulation.py`** — Frequency-domain acoustic simulation with `greens`, followed by a receive-only DAS reconstruction (as in photoacoustics).
- **`eigfilter.py`** — Clutter filtering on a slow-time sequence: separate stationary tissue from moving flow using `eigfilter`.
- **`interpolation.py`** — Reconstruct on a spherical grid, then interpolate back to Cartesian coordinates.

## Requirements

ffdas requires an NVIDIA GPU with compute capability 5.3 (Maxwell) or higher. The `greens` function uses tensor core instructions and requires compute capability 7.0 (Volta) or higher; calling it on an older GPU produces a runtime error.

## License

MIT License (see `LICENSE.txt`)
