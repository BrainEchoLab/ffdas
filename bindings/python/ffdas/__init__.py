import sys
import os


def find_win32_dlls():
    if sys.platform != "win32":
        return

    # ffdas.dll and _ffdas.pyd are in _core/
    core_dir = os.path.join(os.path.dirname(__file__), "_core")
    if os.path.isdir(core_dir):
        os.add_dll_directory(core_dir)

    try:
        from cuda.pathfinder import (
            load_nvidia_dynamic_lib, DynamicLibNotFoundError)
    except ImportError:
        return
    try:
        lib = load_nvidia_dynamic_lib("cudart")
    except DynamicLibNotFoundError:
        pass
    else:
        if lib.abs_path is not None:
            cuda_dll_path = os.path.dirname(os.path.realpath(lib.abs_path))
            os.add_dll_directory(cuda_dll_path)


find_win32_dlls()


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
