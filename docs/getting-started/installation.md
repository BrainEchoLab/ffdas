# Installation

## Python

=== "CUDA 13"

    ```bash
    pip install ffdas[cu13]
    ```

=== "CUDA 12"

    ```bash
    pip install ffdas[cu12]
    ```

Requires Python 3.12+ and a CUDA-capable GPU (SM 53+, SM 70+ recommended). CuPy or another DLPack-compatible GPU array library is needed to create the input arrays.

The `ffdas` package contains the Python bindings. The `[cu12]` / `[cu13]` extra pulls in the matching core library (`ffdas-core-cu12` or `ffdas-core-cu13`), which contains the GPU-accelerated shared library built against that CUDA version.

## MATLAB

Download the prebuilt MEX archive for your platform and CUDA version from the [GitHub Releases](https://github.com/brainecholab/ffdas/releases) page. Extract it and add the directory to your MATLAB path:

```matlab
addpath("/path/to/ffdas-matlab")
```

Requires MATLAB R2018b+ with the Parallel Computing Toolbox and a system CUDA installation matching the archive's CUDA version.

## Building from source

### Requirements

- CMake 3.26+
- CUDA Toolkit (12.x or 13.x)
- C++ compiler with C++17 support (GCC 9+, Clang 10+, or MSVC 2019+)
- For Python bindings: Python 3.12+, nanobind, scikit-build-core
- For MATLAB bindings: MATLAB R2018b+, Parallel Computing Toolbox

### Core library

```bash
cmake --preset default
cmake --build _build
```

This builds the `libffdas.so` (Linux) or `ffdas.dll` (Windows) shared library. The library links CUDA privately — the public API is pure C with no CUDA types.

To target a specific GPU architecture:

```bash
cmake --preset default -DCMAKE_CUDA_ARCHITECTURES=86
```

The minimum supported architecture is SM 53. SM 70+ enables tensor-core-accelerated Green's function summation. Common values:

| Architecture | GPUs |
|---|---|
| 53 | Jetson Nano, GTX 10-series (mobile) |
| 70 | V100, Titan V |
| 75 | RTX 20-series, Quadro RTX |
| 80 | A100 |
| 86 | RTX 30-series, A-series |
| 89 | RTX 40-series, L-series |
| 90 | H100, GH200 |

If `CMAKE_CUDA_ARCHITECTURES` is not specified, CMake targets the architecture of the locally installed GPU (`native`).

To use a specific CUDA toolkit:

```bash
cmake --preset default \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13/bin/nvcc \
    -DCUDAToolkit_ROOT=/usr/local/cuda-13
```

### Python bindings

The bindings are built as a standalone project that links against the core library. Build the core library first, then install the bindings with pip:

```bash
# 1. Build the core library
cmake --preset default
cmake --build _build

# 2. Install the Python bindings (editable)
CMAKE_PREFIX_PATH=$PWD/_build pip install -e bindings/python --no-build-isolation
```

scikit-build-core invokes CMake for the nanobind extension, using `CMAKE_PREFIX_PATH` to locate the core library. On Linux, the extension's build-tree RPATH handles runtime library resolution automatically.

On Windows, set `FFDAS_LIB_DIR` to the build directory so the library can be found at import time:

```powershell
$env:FFDAS_LIB_DIR = "$PWD\_build"
CMAKE_PREFIX_PATH="$PWD\_build" pip install -e bindings\python --no-build-isolation
```

nanobind must be installed before running the editable install (it is listed in the build requirements, but `--no-build-isolation` skips automatic installation):

```bash
pip install nanobind
```

### MATLAB bindings

```bash
# 1. Build the core library (if not already done)
cmake --preset default
cmake --build _build

# 2. Build the MATLAB MEX bindings
cmake -S bindings/matlab -B _build_matlab -DCMAKE_PREFIX_PATH=$PWD/_build
cmake --build _build_matlab

# 3. Install
cmake --install _build_matlab --prefix /path/to/install
```

The MEX files use `$ORIGIN` RPATH on Linux, so the shared library must be in the same directory. The install step places both the MEX files and the shared library under `+ffdas/+core/`.

Add the install prefix to your MATLAB path:

```matlab
addpath("/path/to/install")
```

### CMake options

| Option | Default | Description |
|---|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | `native` | Target CUDA architectures (minimum: 53) |
| `FFDAS_USE_NVTX` | `ON` | Enable NVTX profiling annotations |

### Release builds

The `release.py` script at the repository root automates the full release workflow: building the core library for multiple CUDA versions, packaging Python wheels, and creating MATLAB archives.

```bash
CUDA12_ROOT=/usr/local/cuda-12 CUDA13_ROOT=/usr/local/cuda-13 python release.py
```

See `python release.py --help` for options.

## Troubleshooting

**CUDA toolkit not found.** Verify `nvcc --version` works. If the toolkit is installed in a non-standard location, set `CUDAToolkit_ROOT` and `CMAKE_CUDA_COMPILER`.

**Python import error.** If `import ffdas` fails with a missing `_ffdas` module, the nanobind extension was not built or is not in the expected location. For editable installs, verify the core library build completed and that `CMAKE_PREFIX_PATH` was set correctly during `pip install`.

**CUDA runtime not found at import time.** If the import fails with an `OSError` loading the shared library, the CUDA runtime libraries are not on the library search path. Install the CUDA toolkit system-wide, or use `pip install cuda-toolkit[cudart,cublas,cusolver]==12.*` to install them via pip.

**Windows: DLL not found.** Set `FFDAS_LIB_DIR` to the directory containing `ffdas.dll` for development builds.

**Architecture error during build.** If you see errors about unsupported SM versions, set `CMAKE_CUDA_ARCHITECTURES` to match your GPU. Run `nvidia-smi` to check your GPU model.
