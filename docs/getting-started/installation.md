# Installation

## Python

=== "CUDA 13"

    ```bash
    pip install ffdas-cu13
    ```

=== "CUDA 12"

    ```bash
    pip install ffdas-cu12
    ```

Requires Python 3.12+ and a CUDA-capable GPU (SM 70+). CuPy or another DLPack-compatible GPU array library is needed to create the input arrays.

By default, the pip package does not install CUDA runtime libraries. If you do not have a CUDA toolkit installed system-wide, install with the `cuda` extra to pull them from PyPI:

=== "CUDA 13"

    ```bash
    pip install ffdas-cu13[cuda]
    ```

=== "CUDA 12"

    ```bash
    pip install ffdas-cu12[cuda]
    ```

## MATLAB

Download the prebuilt MEX binaries for your platform from the [GitHub Releases](https://github.com/brainecholab/ffdas/releases) page and add the bindings directory to your MATLAB path:

```matlab
addpath("/path/to/ffdas/bindings/matlab")
```

Requires MATLAB R2018b+ with the Parallel Computing Toolbox.

## Building from Source

### Requirements

- CMake 3.26 or later
- CUDA Toolkit (13.x or 12.x)
- C++ compiler with C++17 support (GCC 9+, Clang 10+, or MSVC 2019+)
- For Python bindings: Python 3.12+, nanobind, array-api-compat
- For MATLAB bindings: MATLAB R2018b+, Parallel Computing Toolbox

### Core Library

Configure and build from the repository root:

```bash
cmake -S . -B build
cmake --build build --config Release
```

This builds the `ffdas` shared library (`libffdas.so` / `ffdas.dll`) and places it in the `build/` directory. A second shared library target (`ffdas_static_runtime`) is built with the CUDA runtime linked statically — this is used by the MATLAB bindings to avoid conflicts with MATLAB's own CUDA runtime.

### CUDA Architecture

If `CMAKE_CUDA_ARCHITECTURES` is not specified, CMake targets all architectures. To build for a specific GPU generation, set it explicitly:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
```

The minimum supported architecture is SM 70 (required for half-precision and tensor core support). Common values:

| Architecture | GPUs |
|---|---|
| 70 | V100, Titan V |
| 75 | RTX 20-series, Quadro RTX |
| 80 | A100 |
| 86 | RTX 30-series, A-series |
| 89 | RTX 40-series, L-series |
| 90 | H100, GH200 |

Specifying the exact architecture for your GPU avoids compiling unused PTX/SASS and speeds up the build.

### CMake Options

| Option | Default | Description |
|---|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | `all` | Target CUDA architectures (minimum: 70) |
| `BUILD_PYTHON` | `OFF` | Build Python bindings |
| `BUILD_MEX` | `OFF` | Build MATLAB MEX bindings |
| `FFDAS_USE_NVTX` | `ON` | Enable NVTX profiling annotations |
| `FETCH_NANOBIND` | `OFF` | Fetch nanobind via FetchContent if not found |

### Python Development Install

The recommended way to build and install the Python bindings for development is directly through CMake:

```bash
git clone https://github.com/brainecholab/ffdas.git
cd ffdas
pip install nanobind array-api-compat
cmake -S . -B build -DBUILD_PYTHON=ON
cmake --build build --config Release
cmake --install build --prefix bindings/python
```

This places the compiled extension (`_ffdas`) and shared library (`libffdas`) into `bindings/python/ffdas/_core/`, making the package importable from `bindings/python/`. Either add that directory to your Python path or install in editable mode:

```bash
pip install -e ".[dev]" --no-build-isolation
```

After modifying C++ or CUDA source files, re-run the cmake build and install steps.

If nanobind is not installed, CMake can fetch it automatically:

```bash
cmake -S . -B build -DBUILD_PYTHON=ON -DFETCH_NANOBIND=ON
```

#### Using pip to build

An alternative to invoking cmake directly is building entirely through pip, which handles nanobind discovery and cmake invocation automatically:

```bash
pip install -e ".[dev]"
```

To point the build at a specific CUDA toolkit, set environment variables:

```bash
CUDA_ROOT=/usr/local/cuda-13 pip install -e ".[dev]"
```

Additional cmake flags can be passed through `FFDAS_CMAKE_ARGS`:

```bash
FFDAS_CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86" pip install -e ".[dev]"
```

### MATLAB Bindings

The MATLAB bindings are built when `BUILD_MEX` is enabled and CMake finds a MATLAB installation:

```bash
cmake -S . -B build -DBUILD_MEX=ON
cmake --build build --config Release
```

The compiled MEX files are placed into the build directory and can be installed with:

```bash
cmake --install build --prefix /path/to/install
```

This copies the MEX files and MATLAB wrapper scripts into the install prefix. Add the install directory to your MATLAB path:

```matlab
addpath("/path/to/install/+ffdas/..")
```

The MEX bindings link against `ffdas_static_runtime` (static CUDA runtime), so no runtime dependency on the CUDA shared libraries is needed. On Linux the MEX files also link libstdc++ and libgcc statically, so they should work without needing to match the system compiler version.

If CMake cannot find MATLAB automatically, specify the installation path:

```bash
cmake -S . -B build -DBUILD_MEX=ON -DMatlab_ROOT_DIR="/path/to/matlab"
```

## Platform Notes

### Linux

Install the CUDA toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads) or your distribution's package manager. Make sure `nvcc` is on your PATH:

```bash
nvcc --version
```

If you have multiple CUDA versions installed, point CMake to the right one:

```bash
cmake -S . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13/bin/nvcc \
    -DCUDAToolkit_ROOT=/usr/local/cuda-13
```

Or via environment variable when building through pip:

```bash
CUDA_ROOT=/usr/local/cuda-13 pip install -e .
```

### Windows

Install Visual Studio 2019 or later with the C++ workload, and the CUDA Toolkit from NVIDIA. Use the Visual Studio generator:

```bash
cmake -S . -B build -G "Visual Studio 16 2019" -A x64
cmake --build build --config Release
```

Check the [CUDA/Visual Studio compatibility table](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#visual-studio-support) if you run into compiler errors.

### Debug Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
```

This enables device-side debug info and verbose PTX output, which is useful for kernel debugging with tools like compute-sanitizer and Nsight Compute.

## Troubleshooting

**CUDA toolkit not found.** Verify `nvcc --version` works. If the toolkit is installed in a non-standard location, set `CUDAToolkit_ROOT` or `CMAKE_CUDA_COMPILER`:

```bash
cmake -S . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

**Python not found or wrong version.** CMake uses `find_package(Python)` which searches the PATH. If the wrong interpreter is picked, specify it explicitly:

```bash
cmake -S . -B build -DPython_EXECUTABLE=$(which python3.12)
```

Or when using pip:

```bash
python3.12 -m pip install -e .
```

**Architecture error during build.** If you see errors about unsupported SM versions, set `CMAKE_CUDA_ARCHITECTURES` to match your GPU. The minimum is SM 70. Run `nvidia-smi` to check your GPU model, then look up its compute capability.

**Linux: MEX files fail to load.** If MATLAB reports missing symbols, check that you built with a compiler version compatible with your MATLAB release. Recent MATLAB versions require GCC 12 or 13; older GCC versions may produce incompatible binaries.

**Import error in Python.** If `import ffdas` fails with a missing `_ffdas` module, the nanobind extension was not built or is not in the expected location. Verify the build completed without errors and confirm that `bindings/python/ffdas/_core/_ffdas*.so` (or `.pyd` on Windows) exists.
