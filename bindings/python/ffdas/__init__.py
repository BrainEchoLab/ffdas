import ctypes
import os
import sys
from pathlib import Path


def _load_core():
    """Preload the ffdas shared library so the nanobind extension can find it."""
    lib_name = "ffdas.dll" if sys.platform == "win32" else "libffdas.so"
    lib_path = None

    try:
        import ffdas_core
        lib_path = Path(ffdas_core.__file__).parent / ffdas_core.LIB_NAME
    except ImportError:
        pass

    # Development: cmake installs the core library alongside the extension
    if lib_path is None:
        candidate = Path(__file__).parent / "_core" / lib_name
        if candidate.exists():
            lib_path = candidate

    if lib_path is None:
        lib_dir = os.environ.get("FFDAS_LIB_DIR")
        if lib_dir:
            lib_path = Path(lib_dir) / lib_name

    if lib_path is None:
        return

    if sys.platform == "win32":
        os.add_dll_directory(str(lib_path.parent))
        try:
            from cuda.pathfinder import load_nvidia_dynamic_lib
            lib = load_nvidia_dynamic_lib("cudart")
            if lib.abs_path is not None:
                os.add_dll_directory(
                    os.path.dirname(os.path.realpath(lib.abs_path)))
        except Exception:
            pass

    try:
        ctypes.CDLL(
            str(lib_path),
            mode=ctypes.RTLD_GLOBAL if sys.platform != "win32" else 0,
        )
    except OSError as e:
        cuda_ver = ""
        try:
            import ffdas_core
            cuda_ver = f" (CUDA {ffdas_core.CUDA_VERSION})"
        except Exception:
            pass
        raise ImportError(
            f"Failed to load ffdas shared library{cuda_ver}. "
            f"Ensure the matching CUDA runtime is installed.\n"
            f"Original error: {e}"
        ) from e


_load_core()


from ._core._ffdas import (  # noqa: E402
    half,
    float32,
    float64,
    int16,
    int32,
    half2,
    float2,
    complex64,
    complex128,
    double2,
    short2,
    int2,
    dtype,
    view,
    TensorView,
    InterpMode,
    Algorithm,
)
from ._core.tensor import (  # noqa: E402
    gather,
    scatter,
    contiguous_copy,
    empty_like,
    zeros_like,
    full_like,
    astype,
    DowncastWarning,
)
from ._core.library import (  # noqa: E402
    get_cuda_device,
    set_cuda_device,
)
from .das import (  # noqa: E402
    das,
    das_sparse,
)
from .truncate_rank import truncate_rank  # noqa: E402
from .interpolation import (  # noqa: E402
    Interpolator,
    interpolate,
)
from .greens import greens  # noqa: E402
from .einsum import einsum  # noqa: E402
from . import utils  # noqa: E402
from .utils.spatial import (  # noqa: E402
    cdist,
    rect_dist,
    angle,
)

__version__ = "0.1.0"

__all__ = [
    "half",
    "float32",
    "float64",
    "int16",
    "int32",
    "half2",
    "float2",
    "complex64",
    "complex128",
    "double2",
    "short2",
    "int2",
    "dtype",
    "view",
    "TensorView",
    "empty_like",
    "zeros_like",
    "full_like",
    "astype",
    "DowncastWarning",
    "get_cuda_device",
    "set_cuda_device",
    "InterpMode",
    "Algorithm",
    "gather",
    "scatter",
    "contiguous_copy",
    "das",
    "das_sparse",
    "truncate_rank",
    "greens",
    "einsum",
    "Interpolator",
    "interpolate",
    "cdist",
    "rect_dist",
    "angle",
    "utils",
]
