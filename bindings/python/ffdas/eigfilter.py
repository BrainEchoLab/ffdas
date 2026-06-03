from typing import overload
import math

from ._core import _ffdas
from ._core.library import get_library_handle
from ._core.tensor_like import TensorLike, T
from ._core.tensor import empty_like, reshape


@overload
def eigfilter(
    x: T,
    k0: int,
    k1: int | None = None,
    *,
    out: None = ...,
) -> T: ...
@overload
def eigfilter(
    x: TensorLike,
    k0: int,
    k1: int | None = None,
    *,
    out: T,
) -> T: ...


def eigfilter(
    x: TensorLike,
    k0: int,
    k1: int | None = None,
    *,
    out: TensorLike | None = None,
) -> TensorLike:
    """Eigenvalue-based clutter filter.

    Reconstructs x using only singular vectors k0 through k1, removing
    all other components. This is a fast approximation of truncated SVD
    reconstruction.

    The input is reshaped to a 2D matrix (m, n) where m = x.shape[0] and
    n = product of remaining dimensions. The output has the same shape
    as x.

    Args:
        x: Input array with at least 2 dimensions.
        k0: Index of the first singular vector to keep (0-based).
        k1: Index past the last singular vector to keep (exclusive).
            Defaults to min(m, n).
        out: Pre-allocated output array.

    Returns:
        Filtered array with the same shape as x.
    """
    if x.ndim < 2:
        raise ValueError(f"x must have at least 2 dimensions, got {x.ndim}")

    shp = x.shape
    m = shp[0]
    n = math.prod(x.shape[1:])

    if k0 < 0 or k0 >= min(m, n):
        raise ValueError(f"k0 must be >= 0 and < min(m, n) ({m}, {n}), got {k0}")
    if k1 is None:
        k1 = min(m, n)
    elif k1 <= k0 or k1 > min(m, n):
        raise ValueError(
            f"k1 must be > k0 ({k0}) and <= min(m, n) ({m}, {n}), got {k1}"
        )

    if out is not None and out.shape != x.shape:
        raise ValueError("out must have the same dimensions as x")

    if out is None:
        out = empty_like(x, shape=x.shape)

    if x.ndim > 2:
        x = reshape(x, (m, n), order="C")
        # don't reassign out so that we can return the original reference to out
        # i.e., `result = eigfilter(..., out=out); result is out  # True`
        out_2d = reshape(out, (m, n), order="C")
    else:
        out_2d = out

    _ffdas.eigfilter(
        get_library_handle(),
        x,
        k0,
        k1,
        out_2d,
    )

    return out
