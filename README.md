# ffdas

A CUDA-accelerated library for delay-and-sum beamforming and related operations, with bindings for Python and MATLAB.

ffdas targets real-time and high-throughput ultrasound imaging pipelines. The core is a C/CUDA library exposing a handle-based API, with high-level bindings that integrate directly with GPU array libraries (CuPy, Torch, MATLAB gpuArrays) via zero-copy interop.

## Features

The library provides GPU-accelerated implementations of:

- **Delay-and-sum beamforming** with multiple algorithm variants, including sparse compounding (per-target transmission subsets) and half-precision compute paths.
- **Structured grid interpolation** (nearest, linear) for mapping between coordinate systems.
- **Green's function summation** for wave propagation modeling.
- **Eigenvalue-based clutter filtering** for removing tissue signal from blood flow data.
- **Tensor contraction** (einsum) for arbitrary binary contractions on GPU arrays.
- **Gather/scatter** and contiguous-copy operations on GPU tensors.

## Getting Started

### Prerequisites

ffdas requires CMake 3.18+, a CUDA toolkit, and a C++17-capable compiler. The Python bindings require Python 3.10+ and NumPy 2.0+. The MATLAB bindings require MATLAB R2018b+ with the Parallel Computing Toolbox.

nanobind (for the Python bindings) and NVTX (for profiling annotations) are fetched automatically during the build.

### Building

```bash
cmake -S . -B build
cmake --build build --config Release
```

This builds the shared and static libraries and, if the dependencies are found, the Python and MATLAB bindings. To skip either set of bindings:

```bash
cmake -S . -B build -DBUILD_PYTHON=OFF   # skip Python
cmake -S . -B build -DBUILD_MEX=OFF      # skip MATLAB
```

See the [installation guide](docs/getting-started/installation.md) for platform-specific instructions and troubleshooting.

### Quick Example

```python
import cupy as cp
import ffdas

# reconstruct a batch of sample data
output = ffdas.das(
    sample_data,  # (batch, channels, sequence, samples)
    channel_pos,  # (channels, 3)
    voxel_pos,    # (..., 3)
    delays,       # (sequence, ...)
    weights,      # (sequence, ...)
)

# remove clutter via eigenfilter
filtered = ffdas.eigfilter(beamformed, k0=64)
```

## Project Structure

```
ffdas/
├── include/           C API headers
├── src/               CUDA implementation
├── bindings/
│   ├── python/        Python package (nanobind + DLPack)
│   └── matlab/        MATLAB package (MEX)
├── docs/              Documentation
└── profile/           Profiling scripts
```

## Array Layout Conventions

The Python and MATLAB bindings follow the conventions of their respective languages. In Python, coordinate arrays are row-major with the coordinate dimension last: positions have shape `(..., 3)`. In MATLAB, they are column-major with the coordinate dimension first: `(3, ...)`. The rest of the API is analogous between the two.

## License

MIT
