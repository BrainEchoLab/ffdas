# Installation

## Python

=== "CUDA 13"

    ```bash
    pip install ffdas[cuda13]
    ```

=== "CUDA 12"

    ```bash
    pip install ffdas[cuda12]
    ```

Requires Python 3.12+ and a CUDA-capable GPU (SM 53+, SM 70+ recommended). CuPy or another DLPack-compatible GPU array library is needed to create the input arrays.

The `ffdas` package contains the Python bindings. The `[cuda12]` / `[cuda13]` extra pulls in the matching core library (`ffdas-core-cuda12` or `ffdas-core-cuda13`), which contains the GPU-accelerated shared library built against that CUDA version.

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

### Windows: Visual Studio developer environment

On Windows, the CUDA compiler requires the MSVC toolchain to be on PATH. Open a **Developer Command Prompt** or activate the environment from a regular terminal:

```powershell
# Find the VS installation
$vsPath = & "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" `
    -latest -property installationPath

# Activate the developer environment
Import-Module "$vsPath\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $vsPath -Arch amd64 -SkipAutomaticLocation
```

Alternatively, launch the **x64 Native Tools Command Prompt** from the Start menu (search for "x64 Native Tools"). All build commands below assume this environment is active on Windows.

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
| 53 | Maxwell, Jetson Nano, GTX 10-series (mobile) |
| 60 | Pascal |
| 70 | Volta, V100, Titan V |
| 75 | Turing, RTX 20-series, Quadro RTX |
| 80 | Ampere, A100 |
| 86 | Ampere, RTX 30-series, A-series |
| 89 | Ada, RTX 40-series, L-series |
| 90 | Hopper, H100, GH200 |
| 100 | Blackwell, B100 |
| 120 | Blackwell, RTX 50-series, RTX PRO |

If `CMAKE_CUDA_ARCHITECTURES` is not specified, CMake targets the architecture of the locally installed GPU (`native`).

On Windows, the default preset does not specify a generator. Ninja is recommended for CUDA builds:

```powershell
cmake --preset default -G Ninja
cmake --build _build
```

To use a specific CUDA toolkit:

=== "Linux"

    ```bash
    cmake --preset default \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13/bin/nvcc \
        -DCUDAToolkit_ROOT=/usr/local/cuda-13
    ```

=== "Windows"

    ```powershell
    cmake --preset default -G Ninja `
        -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe" `
        -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"
    ```

### Python bindings

The bindings are a standalone project that links against an installed core library. Build and install the core library first, then install the bindings with pip:

=== "Linux"

    ```bash
    # 1. Build and install the core library
    cmake --preset default
    cmake --build _build
    cmake --install _build --prefix _install

    # 2. Make the shared library findable at runtime
    export LD_LIBRARY_PATH=$PWD/_install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

    # 3. Install the Python bindings (editable)
    pip install nanobind
    CMAKE_PREFIX_PATH=$PWD/_install pip install -e bindings/python --no-build-isolation
    ```

=== "Windows"

    ```powershell
    # 1. Build and install the core library
    cmake --preset default -G Ninja
    cmake --build _build
    cmake --install _build --prefix _install

    # 2. Make the DLL findable at runtime
    $env:Path = "$PWD\_install\bin;$env:Path"

    # 3. Install the Python bindings (editable)
    pip install nanobind
    $env:CMAKE_PREFIX_PATH = "$PWD\_install"
    pip install -e bindings\python --no-build-isolation
    ```

scikit-build-core invokes CMake for the nanobind extension, using `CMAKE_PREFIX_PATH` to locate the installed core library. The shared library must be on the system library search path (`LD_LIBRARY_PATH` on Linux, `PATH` on Windows) for `import ffdas` to work.

After modifying C++ or CUDA source files, rebuild and reinstall the core library, then reinstall the bindings.

### MATLAB bindings

=== "Linux"

    ```bash
    # 1. Build and install the core library (if not already done)
    cmake --preset default
    cmake --build _build
    cmake --install _build --prefix _install

    # 2. Build the MATLAB MEX bindings
    cmake -S bindings/matlab -B _build_matlab -DCMAKE_PREFIX_PATH=$PWD/_install
    cmake --build _build_matlab

    # 3. Install
    cmake --install _build_matlab --prefix /path/to/install
    ```

=== "Windows"

    ```powershell
    cmake --preset default -G Ninja
    cmake --build _build
    cmake --install _build --prefix _install

    cmake -S bindings\matlab -B _build_matlab -G Ninja `
        -DCMAKE_PREFIX_PATH="$PWD\_install"
    cmake --build _build_matlab

    cmake --install _build_matlab --prefix C:\path\to\install
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

=== "Linux"

    ```bash
    CUDA12_ROOT=/usr/local/cuda-12 CUDA13_ROOT=/usr/local/cuda-13 python release.py
    ```

=== "Windows"

    ```powershell
    $env:CUDA12_ROOT = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
    $env:CUDA13_ROOT = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
    python release.py
    ```

See `python release.py --help` for options.

## Troubleshooting

**Core library not found at import time.** If `import ffdas` raises `ImportError: Could not find the ffdas core library`, the CUDA extra was not included when installing. The correct command is `pip install ffdas[cuda12]` or `pip install ffdas[cuda13]` — the bracket suffix pulls in the GPU-accelerated core library. Some shells (e.g. zsh) require quoting: `pip install 'ffdas[cuda12]'`.

**CUDA toolkit not found.** Verify `nvcc --version` works. If the toolkit is installed in a non-standard location, set `CUDAToolkit_ROOT` and `CMAKE_CUDA_COMPILER`.

**Python import error.** If `import ffdas` fails with a missing `_ffdas` module, the nanobind extension was not built or is not in the expected location. For editable installs, verify the core library build completed and that `CMAKE_PREFIX_PATH` was set correctly during `pip install`.

**CUDA runtime not found at import time.** If the import fails with an `OSError` loading the shared library, the CUDA runtime libraries are not on the library search path. Install the CUDA toolkit system-wide, or use `pip install cuda-toolkit[cudart,cublas,cusolver]==12.*` to install them via pip.

**Architecture error during build.** If you see errors about unsupported SM versions, set `CMAKE_CUDA_ARCHITECTURES` to match your GPU. Run `nvidia-smi` to check your GPU model.

**Shared library not found at import time.** `import ffdas` requires the core shared library to be findable. For installed packages this is handled automatically by `ffdas-core-cu*`. For development builds, ensure the install prefix is on the library search path (`LD_LIBRARY_PATH` on Linux, `PATH` on Windows). Alternatively, set `FFDAS_LIB_DIR` to the directory containing the shared library.

**Windows: compiler not found.** Make sure you are in a Visual Studio developer command prompt. See the [Windows: Visual Studio developer environment](#windows-visual-studio-developer-environment) section above.
