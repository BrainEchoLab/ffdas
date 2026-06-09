#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST_DIR="$REPO_ROOT/dist"

CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
TARGET="${1:-all}"
FFDAS_VERSION="0.1.0"

fail() { echo "error: $1" >&2; exit 1; }

# Validate prerequisites
[ -x "$CUDA_ROOT/bin/nvcc" ] || fail "nvcc not found at $CUDA_ROOT/bin/nvcc (set CUDA_ROOT)"
command -v cmake &>/dev/null || fail "cmake not found"

# Detect CUDA major version from the toolkit
CUDA_MAJOR=$("$CUDA_ROOT/bin/nvcc" --version | grep -oP 'release \K[0-9]+')
echo "info: detected CUDA major version: $CUDA_MAJOR"

if [ "$CUDA_MAJOR" = "13" ]; then
    CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90-real;100-real;120"
elif [ "$CUDA_MAJOR" = "12" ]; then
    CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90"
else
    fail "unsupported CUDA major version $CUDA_MAJOR (expected 12 or 13)"
fi

# auditwheel excludes — CUDA shared libraries provided by the system or pip
AUDITWHEEL_EXCLUDES=(
    --exclude libcudart.so.*
    --exclude libcublas.so.*
    --exclude libcusolver.so.*
    --exclude libcublasLt.so.*
    --exclude libcusparse.so.*
    --exclude libnvToolsExt.so.*
)

if [ "$TARGET" = "matlab" ] || [ "$TARGET" = "all" ]; then
    command -v matlab &>/dev/null || echo "warning: matlab not on PATH (cmake may still find it)"
    command -v zip &>/dev/null || fail "zip not found"
fi

if [ "$TARGET" = "python" ] || [ "$TARGET" = "all" ]; then
    command -v python &>/dev/null || fail "python not found"
    python -c "import build" 2>/dev/null || fail "python build module not found (pip install build)"
    command -v auditwheel &>/dev/null || fail "auditwheel not found (pip install auditwheel)"
fi

echo "info: CUDA_ROOT=$CUDA_ROOT"
echo "info: CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES"
echo "info: TARGET=$TARGET"
echo "info: DIST_DIR=$DIST_DIR"

mkdir -p "$DIST_DIR"

# Stage 1: build the core library
LIB_BUILD_DIR="$REPO_ROOT/_build_lib_cu$CUDA_MAJOR"
echo ""
echo "info: building libffdas_cu$CUDA_MAJOR"
cmake -S "$REPO_ROOT" -B "$LIB_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
    -DCUDAToolkit_ROOT="$CUDA_ROOT" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES"
cmake --build "$LIB_BUILD_DIR" -j

# Verify the library was built
[ -f "$LIB_BUILD_DIR/libffdas_cu$CUDA_MAJOR.so" ] || fail "libffdas_cu$CUDA_MAJOR.so not found in $LIB_BUILD_DIR"

# Stage 2a: MATLAB toolbox
if [ "$TARGET" = "matlab" ] || [ "$TARGET" = "all" ]; then
    MATLAB_BUILD_DIR="$REPO_ROOT/_build_matlab_cu$CUDA_MAJOR"
    MATLAB_DIST_NAME="ffdas_cu$CUDA_MAJOR-$FFDAS_VERSION-matlab-linux_x86_64"
    echo ""
    echo "info: building MATLAB toolbox"
    cmake -S "$REPO_ROOT/bindings/matlab" -B "$MATLAB_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DFFDAS_LIB_DIR="$LIB_BUILD_DIR" \
        -DFFDAS_INCLUDE_DIR="$REPO_ROOT/include"
    cmake --build "$MATLAB_BUILD_DIR" -j
    cmake --install "$MATLAB_BUILD_DIR" --prefix "$DIST_DIR/$MATLAB_DIST_NAME"
    (cd "$DIST_DIR" && zip -qr "$MATLAB_DIST_NAME.zip" "$MATLAB_DIST_NAME/")
    echo ""
fi

# Stage 2b: Python wheel
if [ "$TARGET" = "python" ] || [ "$TARGET" = "all" ]; then
    WHEEL_DIR="$REPO_ROOT/_build_wheel_cu$CUDA_MAJOR"
    echo ""
    echo "info: building Python wheel (ffdas-cu$CUDA_MAJOR)"
    export FFDAS_LIB_DIR="$LIB_BUILD_DIR"
    export CUDA_ROOT
    cd "$REPO_ROOT"
    rm -rf "$WHEEL_DIR"
    python -m build --wheel --outdir "$WHEEL_DIR"
    auditwheel repair "$WHEEL_DIR"/*.whl \
        --wheel-dir "$DIST_DIR/" \
        "${AUDITWHEEL_EXCLUDES[@]}"
    echo ""
fi

echo "info: build outputs"
ls -lh "$DIST_DIR"/*.zip "$DIST_DIR"/*.whl 2>/dev/null || echo "(none)"
