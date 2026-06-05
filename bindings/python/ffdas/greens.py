from typing import overload
import math

from ._core import _ffdas
from ._core.library import get_library_handle
from ._core.tensor_like import TensorLike, T
from ._core.tensor import empty_like, reshape, astype


@overload
def greens(
    srcpos: TensorLike,
    wavenums: TensorLike,
    x: T,
    dstpos: TensorLike,
    *,
    out: None = ...,
) -> T: ...
@overload
def greens(
    srcpos: TensorLike,
    wavenums: TensorLike,
    x: TensorLike,
    dstpos: TensorLike,
    *,
    out: T,
) -> T: ...

def greens(
    srcpos: TensorLike,
    wavenums: TensorLike,
    x: TensorLike,
    dstpos: TensorLike,
    *,
    out: TensorLike | None = None,
) -> TensorLike:
    """Green's function summation over source positions.

    For each frequency and target, sums contributions from all sources
    weighted by the free-space Green's function exp(i*k*r) / r, where r
    is the source-to-target distance and k is the wavenumber at that
    frequency.

    Args:
        srcpos: Source positions, shape (sources, 3).
        wavenums: Wavenumber per frequency bin, shape (frequencies,).
        x: Input field, shape (batch, sources, frequencies) Complex-valued (frequency domain).
        dstpos: Destination (target) positions, shape (..., 3).
        out: Pre-allocated output array.

    Returns:
        Propagated field, shape (batch, num_targets, frequencies), where
        num_targets = product of dstpos.shape[:-1].
    """
    srcpos = astype(srcpos, "float32")
    dstpos = astype(dstpos, "float32")
    wavenums = astype(wavenums, "float32")

    out_shape = (*dstpos.shape[:-1], x.shape[-1])
    flat_shape = (math.prod(dstpos.shape[:-1]), x.shape[-1])

    if x.ndim == 3:
        out_shape = (x.shape[0],) + out_shape
        flat_shape = (x.shape[0],) + flat_shape
    elif x.ndim == 2:
        flat_shape = (1,) + flat_shape
        x = reshape(x, (1,) + x.shape)
    else:
        raise ValueError(f"input must have 2 or 3 dimensions, got shape {tuple(x.shape)}")

    if out is None:
        out = empty_like(x, shape=out_shape)
    elif out.shape != out_shape:
        raise ValueError(f"invalid output shape: expected {out_shape}, got {tuple(out.shape)}")
    
    out_3d = reshape(out, flat_shape, order="C")

    _ffdas.greens_sum(
        get_library_handle(),
        srcpos,
        wavenums,
        x,
        dstpos,
        out_3d,
    )

    return reshape(out, out_shape)
