# Installation

## Requirements

**Required:**

- CMake 3.26 or later
- CUDA Toolkit (13.x recommended)
- C++ compiler with C++17 support (GCC 9+, Clang 10+, or MSVC 2019+)

**For Python bindings:**

- Python 3.12 or later
- array-api-compat

**For MATLAB bindings:**

- MATLAB R2018b or later
- Parallel Computing Toolbox (for gpuArray support)

nanobind can be downloaded automatically during the CMake configure step via FetchContent.

## Building the Core Library

Configure and build from the repository root:

```bash
cmake -S . -B build
cmake --build build --config Release
```

By default this builds both shared (`libffdas.so` / `ffdas.dll`) and static (`libffdas_static.a` / `ffdas_static.lib`) libraries, placing them in `bin/`.

### CUDA Architecture

If `CMAKE_CUDA_ARCHITECTURES` is not specified, CMake targets all architectures. To build for a specific GPU generation, set it explicitly:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
```

The minimum supported architecture is SM 53 (required for half-precision support). Common values:

| Architecture | GPUs |
|---|---|
| 53 | Jetson Nano, TX1 |
| 75 | RTX 20-series, Quadro RTX |
| 86 | RTX 30-series, A-series |
| 89 | RTX 40-series, L-series |
| 90 | H100, GH200 |

Specifying the exact architecture for your GPU avoids compiling unused PTX/SASS and speeds up the build.

### CMake Options

| Option | Default | Description |
|---|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | `all` | Target CUDA architectures (minimum: 53) |
| `BUILD_PYTHON` | `ON` | Build Python bindings |
| `BUILD_MEX` | `ON` | Build MATLAB MEX bindings |

## Python Bindings

The Python bindings are built automatically when `BUILD_PYTHON` is enabled and CMake finds a suitable Python installation. The compiled extension module (`_ffdas`) is placed directly into the source tree at `bindings/python/ffdas/_core/`, so you can use the package immediately by adding the bindings directory to your path or installing it with pip.

### Installing as a Package

After building the core library and the Python extension:

```bash
cd bindings/python
pip install -e .
```

This installs ffdas in editable mode, so changes to the Python source are reflected immediately. The compiled extension must already be present from the CMake build step.

### Dependencies

The Python package depends on NumPy 2.0+ and array-api-compat. These are installed automatically by pip. CuPy is not a hard dependency, but in practice you will need it (or another DLPack-compatible GPU array library) to create the GPU arrays that ffdas operates on.

## MATLAB Bindings

The MATLAB bindings are built automatically when `BUILD_MEX` is enabled and CMake finds a MATLAB installation. The compiled MEX files are placed into `bindings/matlab/+ffdas/+core/`.

To use ffdas in MATLAB, add the bindings directory to your path:

```matlab
addpath("/path/to/ffdas/bindings/matlab")
```

The bindings link against the static library, so no runtime dependency on the shared library is needed. On Linux the MEX files also link libstdc++ and libgcc statically, so they should work without needing to match the system compiler version.

### MATLAB Path

If CMake cannot find MATLAB automatically, specify the installation path:

```bash
cmake -S . -B build -DMatlab_ROOT_DIR="/path/to/matlab"
```

## Platform Notes

### Linux

Install the CUDA toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads) or your distribution's package manager. Make sure `nvcc` is on your PATH:

```bash
nvcc --version
```

If you have multiple CUDA versions installed, point CMake to the right one:

```bash
export CUDA_PATH=/usr/local/cuda-12.6
cmake -S . -B build
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

**CUDA toolkit not found.** Verify `nvcc --version` works. If the toolkit is installed in a non-standard location, set `CUDA_PATH` or `CMAKE_CUDA_COMPILER`:

```bash
cmake -S . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc
```

**Python not found or wrong version.** CMake uses `find_package(Python)` which searches the PATH. If the wrong interpreter is picked, specify it explicitly:

```bash
cmake -S . -B build -DPython_EXECUTABLE=$(which python3.12)
```

**Architecture error during build.** If you see errors about unsupported SM versions, set `CMAKE_CUDA_ARCHITECTURES` to match your GPU. Run `nvidia-smi` to check your GPU model, then look up its compute capability.

**Linux: MEX files fail to load.** If MATLAB reports missing symbols, check that you built with a compiler version compatible with your MATLAB release. Recent MATLAB versions require GCC 12 or 13; older GCC versions may produce incompatible binaries.

**Import error in Python.** If `import ffdas` fails with a missing `_ffdas` module, the nanobind extension was not built or is not in the expected location. Verify the CMake build completed without errors and that `bindings/python/ffdas/_core/_ffdas*.so` (or `.pyd` on Windows) exists.
