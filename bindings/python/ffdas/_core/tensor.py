from typing import Any, overload
import warnings

from array_api_compat import array_namespace
import array_api_compat

from . import _ffdas
from .library import get_library_handle
from .tensor_like import TensorLike, T


class DowncastWarning(UserWarning):
    pass


@overload
def astype(x: None, dtype: Any) -> None: ...
@overload
def astype(x: T, dtype: Any) -> T: ...

def astype(x: T | None, dtype: Any) -> T | None:
    """Cast x to dtype, returning x unchanged if it already matches.

    dtype may be a string ("float32", "int32", ...) or a backend dtype
    object. Works for any array backend supported by array_api_compat.
    """
    if x is None:
        return None
    backend = array_namespace(x)
    if isinstance(dtype, str):
        dtype_name = dtype
        dtype = getattr(backend, dtype)
    else:
        dtype_name = str(dtype)
    if x.dtype == dtype:
        return x
    warnings.warn(
        f"input of type {x.dtype} will be downcast to {dtype_name}",
        DowncastWarning,
        stacklevel=3,
    )
    return backend.astype(x, dtype)


def reshape(x: T, shape: tuple[int, ...], order="C") -> T:
    if order != "C":
        raise ValueError("only C-contiguous reshapes are supported")
    if isinstance(x, _ffdas.TensorView):
        if x.dtype in (_ffdas.half2, _ffdas.float2, _ffdas.short2, _ffdas.int2):
            shape = shape + (2,)
        base = reshape(x.base, shape, order=order)
        return _ffdas.view(base, dtype=x.dtype)

    backend = array_namespace(x)
    return backend.reshape(x, shape, order=order)


def empty_like(x: T, shape: tuple[int, ...] | None = None, device=None) -> T:
    if shape is None:
        shape = x.shape
    if device is None:
        device = array_api_compat.device(x)

    if isinstance(x, _ffdas.TensorView):
        if x.dtype in (_ffdas.half2, _ffdas.float2, _ffdas.short2, _ffdas.int2):
            shape = shape + (2,)
        base = empty_like(x.base, shape=shape, device=device)
        return _ffdas.view(base, dtype=x.dtype)

    backend = array_namespace(x)
    return array_api_compat.to_device(backend.empty_like(x, shape=shape, device=device), device)


def zeros_like(x: T, shape: tuple[int, ...] | None = None, device=None) -> T:
    if shape is None:
        shape = x.shape
    if device is None:
        device = array_api_compat.device(x)

    if isinstance(x, _ffdas.TensorView):
        if x.dtype in (_ffdas.half2, _ffdas.float2, _ffdas.short2, _ffdas.int2):
            shape = shape + (2,)
        base = zeros_like(x.base, shape=shape, device=device)
        return _ffdas.view(base, dtype=x.dtype)

    backend = array_namespace(x)
    return array_api_compat.to_device(backend.zeros_like(x, shape=shape, device=device), device)


def full_like(x: T, fill_value: complex, shape: tuple[int, ...] | None = None, device=None) -> T:
    if shape is None:
        shape = x.shape
    if device is None:
        device = array_api_compat.device(x)

    if isinstance(x, _ffdas.TensorView):
        if x.dtype in (_ffdas.half2, _ffdas.float2, _ffdas.short2, _ffdas.int2):
            shape = shape + (2,)
        base = full_like(x.base, fill_value=fill_value, shape=shape, device=device)
        return _ffdas.view(base, dtype=x.dtype)

    backend = array_namespace(x)
    return array_api_compat.to_device(backend.full_like(x, fill_value=fill_value, shape=shape, device=device), device)


@overload
def gather(x: T, indices: TensorLike, axis: int, out: None = ...) -> T: ...
@overload
def gather(x: TensorLike, indices: TensorLike, axis: int, out: T) -> T: ...

def gather(
    x: TensorLike,
    indices: TensorLike,
    axis: int,
    out: TensorLike | None = None,
) -> TensorLike:
    """Gather elements from ``x`` along ``axis`` at ``indices``.

    Args:
        x: Input array on a CUDA device.
        indices: 1-D int32 index array on a CUDA device.
        axis: Dimension along which to gather.
        out: Optional pre-allocated output array. If ``None``, a new array
            is allocated using the same library and device as ``x``.

    Returns:
        The gathered result, same object as ``out`` if provided.
    """
    indices = astype(indices, "int32")

    shape = list(x.shape)
    shape[axis] = indices.shape[0]

    if out is None:
        out = empty_like(x, shape=tuple(shape))

    _ffdas.gather(get_library_handle(), x, out, axis, indices)
    return out

@overload
def scatter(x: T, indices: TensorLike, axis: int, out: None = ...) -> T: ...
@overload
def scatter(x: TensorLike, indices: TensorLike, axis: int, out: T) -> T: ...

def scatter(
    x: TensorLike,
    indices: TensorLike,
    axis: int,
    out: TensorLike | None = None,
) -> TensorLike:
    """Scatter elements of ``x`` into ``out`` along ``axis`` at ``indices``.

    Args:
        x: Input array on a CUDA device.
        indices: 1-D int32 index array on a CUDA device.
        axis: Dimension along which to scatter.
        out: Pre-allocated output array. Must be on a CUDA device.

    Returns:
        The output array (same object as ``out``).
    """
    indices = astype(indices, "int32")

    shape = list(x.shape)
    shape[axis] = indices.shape[0]

    if out is None:
        out = empty_like(x, shape=tuple(shape))

    _ffdas.scatter(get_library_handle(), x, out, axis, indices)
    return out

@overload
def contiguous_copy(x: T, out: None = ...) -> T: ...
@overload
def contiguous_copy(x: TensorLike, out: T) -> T: ...

def contiguous_copy(x: TensorLike, out: TensorLike | None = None) -> TensorLike:
    """Copy a strided array into a contiguous array.

    Args:
        x: Input array on a CUDA device (may be non-contiguous).
        out: Optional pre-allocated contiguous output array.

    Returns:
        Contiguous copy of ``x``.
    """
    if out is None:
        out = empty_like(x)

    _ffdas.contiguous_copy(get_library_handle(), x, out)
    return out
