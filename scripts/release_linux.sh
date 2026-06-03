#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST_DIR="$REPO_ROOT/dist"

CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-13}"
CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90-real;100-real;120"
TARGET="${1:-all}"

fail() { echo "error: $1" >&2; exit 1; }

# Validate prerequisites
[ -x "$CUDA_ROOT/bin/nvcc" ] || fail "nvcc not found at $CUDA_ROOT/bin/nvcc (set CUDA_ROOT)"
command -v cmake &>/dev/null || fail "cmake not found"

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

rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

if [ "$TARGET" = "matlab" ] || [ "$TARGET" = "all" ]; then
    echo "info: building MATLAB toolbox"
    cmake -S "$REPO_ROOT" -B "$REPO_ROOT/_build_matlab" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
        -DCUDAToolkit_ROOT="$CUDA_ROOT" \
        -DBUILD_MEX=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES"
    cmake --build "$REPO_ROOT/_build_matlab" -j
    cmake --install "$REPO_ROOT/_build_matlab" --prefix "$DIST_DIR/ffdas-matlab-linux-x86_64"
    (cd "$DIST_DIR" && zip -qr ffdas-matlab-linux-x86_64.zip ffdas-matlab-linux-x86_64/)
    echo ""
fi

if [ "$TARGET" = "python" ] || [ "$TARGET" = "all" ]; then
    echo "info: building Python wheel"
    cd "$REPO_ROOT"
    python -m build --wheel --outdir "$REPO_ROOT/_build_wheel" \
        -Ccmake.define.CMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
        -Ccmake.define.CUDAToolkit_ROOT="$CUDA_ROOT"
    auditwheel repair "$REPO_ROOT/_build_wheel"/*.whl --wheel-dir "$DIST_DIR/" \
        --exclude libcudart.so.13 \
        --exclude libcublas.so.13 \
        --exclude libcusolver.so.13 \
        --exclude libnvToolsExt.so.1
    echo ""
fi

echo "info: build outputs"
ls -lh "$DIST_DIR"/*.zip "$DIST_DIR"/*.whl 2>/dev/null || echo "(none)"
