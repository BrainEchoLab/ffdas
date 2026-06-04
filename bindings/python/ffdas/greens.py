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

    out_shape = (*ypos.shape[:-1], x.shape[-1])
    flat_shape = (math.prod(ypos.shape[:-1]), x.shape[-1])

    if x.ndim == 3:
        out_shape = (x.shape[0],) + out_shape
        flat_shape = (x.shape[0],) + flat_shape
    elif x.ndim == 2:
        flat_shape = (1,) + flat_shape
    else:
        raise ValueError(f"input must have 2 or 3 dimensions, got shape {tuple(x.shape)}")

    if out is None:
        out = empty_like(x, shape=out_shape)
    elif out.shape != out_shape:
        raise ValueError(f"invalid output shape: expected {out_shape}, got {tuple(out.shape)}")
    
    out_3d = reshape(out, flat_shape, order="C")

    _ffdas.greens_sum(
        get_library_handle(),
        xpos,
        wavenums,
        x,
        ypos,
        out_3d,
    )

    return reshape(out, out_shape)
