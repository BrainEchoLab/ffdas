import ctypes
import os
import sys
from pathlib import Path

_dll_dirs = []


def _load_core():
    lib_name = "ffdas.dll" if sys.platform == "win32" else "libffdas.so"
    lib_path = None

    try:
        import ffdas_core
        lib_path = Path(ffdas_core.__file__).parent / ffdas_core.LIB_NAME
    except ImportError:
        pass

    if lib_path is None:
        lib_dir = os.environ.get("FFDAS_LIB_DIR")
        if lib_dir:
            lib_path = Path(lib_dir) / lib_name

    if lib_path is None:
        raise ImportError(
            "Could not find the ffdas core library. "
            "If you installed ffdas from PyPI, make sure you included the "
            "CUDA extra, e.g.: pip install ffdas[cuda12] or pip install ffdas[cuda13]. "
            "For development builds, set the FFDAS_LIB_DIR environment variable "
            "to the directory containing the shared library."
        )

    if sys.platform == "win32":
        _dll_dirs.append(os.add_dll_directory(str(lib_path.parent)))

        # Python 3.8+ does not search PATH for DLL dependencies.
        # Add the CUDA toolkit bin directory explicitly so that
        # cudart, cublas, cusolver etc. are found when loading ffdas.dll.
        cuda_bin_found = False
        try:
            from cuda.pathfinder import load_nvidia_dynamic_lib
            lib = load_nvidia_dynamic_lib("cudart")
            if lib.abs_path is not None:
                _dll_dirs.append(os.add_dll_directory(
                    os.path.dirname(os.path.realpath(lib.abs_path))))
                cuda_bin_found = True
        except Exception:
            pass

        if not cuda_bin_found:
            cuda_path = os.environ.get("CUDA_PATH", "")
            if cuda_path:
                cuda_bin = os.path.join(cuda_path, "bin")
                if os.path.isdir(cuda_bin):
                    _dll_dirs.append(os.add_dll_directory(cuda_bin))
                    cuda_bin_found = True

    try:
        if sys.platform == "win32":
            # winmode=0 uses the standard Windows search order, which
            # includes PATH. The CUDA installer adds its bin directories
            # to PATH, so this finds cudart, cublas, cusolver etc.
            # regardless of toolkit layout (e.g. CUDA 13 uses bin\x64).
            ctypes.CDLL(str(lib_path), winmode=0)
        else:
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        cuda_ver = ""
        try:
            import ffdas_core
            cuda_ver = f" (CUDA {ffdas_core.CUDA_VERSION})"
        except Exception:
            pass
        hint = ""
        if sys.platform == "win32" and not os.environ.get("CUDA_PATH"):
            hint = (
                " The CUDA_PATH environment variable is not set — if the "
                "CUDA toolkit is installed, ensure its bin directory is "
                "accessible (e.g. set CUDA_PATH).\n"
            )
        raise ImportError(
            f"Failed to load the ffdas shared library{cuda_ver} from {lib_path}. "
            f"Ensure the matching CUDA runtime is installed and that your "
            f"GPU driver supports the required CUDA version.\n"
            f"{hint}"
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
