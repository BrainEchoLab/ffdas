from typing import overload
import math

from ._core import _ffdas
from ._core.library import get_library_handle
from ._core.tensor_like import TensorLike, T
from ._core.tensor import empty_like, reshape


@overload
def truncate_rank(
    x: T,
    start: int,
    stop: int | None = None,
    *,
    out: None = ...,
) -> T: ...
@overload
def truncate_rank(
    x: TensorLike,
    start: int,
    stop: int | None = None,
    *,
    out: T,
) -> T: ...


def truncate_rank(
    x: TensorLike,
    start: int,
    stop: int | None = None,
    *,
    out: TensorLike | None = None,
) -> TensorLike:
    """Rank truncation filter via truncated SVD reconstruction.

    Reconstructs x using only singular vectors start through stop,
    removing all other components.

    The input is reshaped to a 2D matrix (m, n) where m = x.shape[0] and
    n = product of remaining dimensions. The output has the same shape
    as x.

    Args:
        x: Input array with at least 2 dimensions.
        start: Index of the first singular vector to keep (0-based).
        stop: Index past the last singular vector to keep (exclusive).
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

    if start < 0 or start >= min(m, n):
        raise ValueError(f"start must be >= 0 and < min(m, n) ({m}, {n}), got {start}")
    if stop is None:
        stop = min(m, n)
    elif stop <= start or stop > min(m, n):
        raise ValueError(
            f"stop must be > start ({start}) and <= min(m, n) ({m}, {n}), got {stop}"
        )

    if out is not None and out.shape != x.shape:
        raise ValueError("out must have the same dimensions as x")

    if out is None:
        out = empty_like(x, shape=x.shape)

    if x.ndim > 2:
        x = reshape(x, (m, n), order="C")
        out_2d = reshape(out, (m, n), order="C")
    else:
        out_2d = out

    _ffdas.truncate_rank(
        get_library_handle(),
        x,
        start,
        stop,
        out_2d,
    )

    return out
