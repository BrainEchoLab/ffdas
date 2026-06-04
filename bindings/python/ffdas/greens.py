from typing import overload
import math

from ._core import _ffdas
from ._core.library import get_library_handle
from ._core.tensor_like import TensorLike, T
from ._core.tensor import empty_like, reshape, astype


@overload
def greens(
    xpos: TensorLike,
    wavenums: TensorLike,
    x: T,
    ypos: TensorLike,
    *,
    out: None = ...,
) -> T: ...
@overload
def greens(
    xpos: TensorLike,
    wavenums: TensorLike,
    x: TensorLike,
    ypos: TensorLike,
    *,
    out: T,
) -> T: ...

def greens(
    xpos: TensorLike,
    wavenums: TensorLike,
    x: TensorLike,
    ypos: TensorLike,
    *,
    out: TensorLike | None = None,
) -> TensorLike:
    """Green's function summation over source positions.

    For each frequency and target, sums contributions from all sources
    weighted by the free-space Green's function exp(i*k*r) / r, where r
    is the source-to-target distance and k is the wavenumber at that
    frequency.

    Args:
        xpos: Source positions, shape (sources, 3).
        wavenums: Wavenumber per frequency bin, shape (frequencies,).
        x: Input field, shape (batch, sources, frequencies) Complex-valued (frequency domain).
        ypos: Target positions, shape (..., 3).
        out: Pre-allocated output array.

    Returns:
        Propagated field, shape (batch, num_targets, frequencies), where
        num_targets = product of ypos.shape[:-1].
    """
    xpos = astype(xpos, "float32")
    ypos = astype(ypos, "float32")
    wavenums = astype(wavenums, "float32")

    full_shape = (x.shape[0], *ypos.shape[:-1], x.shape[2])
    shape = (x.shape[0], math.prod(ypos.shape[:-1]), x.shape[2])
    if out is None:
        out = empty_like(x, shape=shape)

    _ffdas.greens_sum(
        get_library_handle(),
        xpos,
        wavenums,
        x,
        ypos,
        out,
    )

    return reshape(out, full_shape)
