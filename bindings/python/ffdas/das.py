from typing import overload

from array_api_compat import array_namespace

from ._core import _ffdas
from ._core.library import get_library_handle
from ._core.tensor_like import TensorLike, T
from ._core.tensor import empty_like, zeros_like, astype


def _compute_type(x, use_fp16):
    if use_fp16:
        return _ffdas.ComputeType.FP16
    if isinstance(x, _ffdas.TensorView):
        if x.dtype in (_ffdas.float64, _ffdas.complex128, _ffdas.double2):
            return _ffdas.ComputeType.FP64
        return _ffdas.ComputeType.FP32
    xp = array_namespace(x)
    if x.dtype == xp.float64 or x.dtype == xp.complex128:
        return _ffdas.ComputeType.FP64
    return _ffdas.ComputeType.FP32


@overload
def das(
    x: T,
    xpos: TensorLike,
    ypos: TensorLike,
    offsets: TensorLike,
    weights: TensorLike,
    *,
    xdir: TensorLike | None = None,
    wavenum: float = 0.0,
    algorithm: _ffdas.Algorithm = _ffdas.Algorithm.DEFAULT,
    use_fp16: bool = False,
    channels_trailing: bool = False,
    out: None = ...,
) -> T: ...
@overload
def das(
    x: TensorLike,
    xpos: TensorLike,
    ypos: TensorLike,
    offsets: TensorLike,
    weights: TensorLike,
    *,
    xdir: TensorLike | None = None,
    wavenum: float = 0.0,
    algorithm: _ffdas.Algorithm = _ffdas.Algorithm.DEFAULT,
    use_fp16: bool = False,
    channels_trailing: bool = False,
    out: T,
) -> T: ...

def das(
    x: TensorLike,
    xpos: TensorLike,
    ypos: TensorLike,
    offsets: TensorLike,
    weights: TensorLike,
    *,
    xdir: TensorLike | None = None,
    wavenum: float = 0.0,
    algorithm: _ffdas.Algorithm = _ffdas.Algorithm.DEFAULT,
    use_fp16: bool = False,
    channels_trailing: bool = False,
    out: TensorLike | None = None,
) -> TensorLike:
    """Delay-and-sum beamforming, compounding over the sequence dimension.

    For each target, computes a weighted sum of interpolated samples from
    all channels and sequence events. Positions should be in sampling
    wavelengths (c / fs) and offsets in samples.

    Args:
        x: Input RF data, shape (channels, sequence, samples) or
            (batch, channels, sequence, samples).
        xpos: Channel positions, shape (channels, 3).
        ypos: Target positions, shape (..., 3). The spatial dimensions
            are ypos.shape[:-1].
        offsets: Per-target time offsets in samples, shape (sequence, ...).
        weights: Per-target apodization weights, shape (sequence, ...).
        xdir: Channel directivity vectors, shape (channels, 4). The first
            three components are the unit surface normal; the fourth is
            the cosine of the sensitivity half-angle. Channels whose
            angle to a target exceeds this are excluded.
        wavenum: Wavenumber for phase rotation, typically -2*pi*fc/fs.
            Set to 0 to disable.
        algorithm: Algorithm variant.
        use_fp16: Use half-precision arithmetic.
        channels_trailing: If true, the input dimensions are interpreted as (batch, sequence, channels, samples)
            instead of (batch, channels, sequence, samples). Default: False.
        out: Pre-allocated output array.

    Returns:
        Beamformed output with shape (...) or (batch, ...).
    """
    if ypos.ndim < 2 or ypos.shape[-1] != 3:
        raise ValueError("ypos must have dimensions (..., 3)")

    spatial_dims = ypos.shape[:-1]

    out_dims = spatial_dims
    if x.ndim == 4:
        out_dims = (x.shape[0], *out_dims)

    if offsets.shape[1:] != spatial_dims:
        raise ValueError(
            f"offsets shape {offsets.shape} must match target layout {spatial_dims}"
        )
    if weights.shape[1:] != spatial_dims:
        raise ValueError(
            f"weights shape {weights.shape} must match target layout {spatial_dims}"
        )

    xpos = astype(xpos, "float32")
    ypos = astype(ypos, "float32")
    offsets = astype(offsets, "float32")
    weights = astype(weights, "float32")
    xdir = astype(xdir, "float32")

    if out is None:
        out = empty_like(x, shape=out_dims)

    beta = zeros_like(x, shape=(1,), device="cpu")
    compute_type = _compute_type(x, use_fp16)

    _ffdas.das(
        get_library_handle(),
        x,
        xpos,
        ypos,
        offsets,
        weights,
        xdir,
        wavenum,
        beta,
        out,
        algorithm,
        compute_type,
        channels_trailing,
    )
    return out


@overload
def das_sparse(
    x: T,
    xpos: TensorLike,
    ypos: TensorLike,
    offsets: TensorLike,
    weights: TensorLike,
    sparse_indices: TensorLike,
    *,
    xdir: TensorLike | None = None,
    wavenum: float = 0.0,
    algorithm: _ffdas.Algorithm = _ffdas.Algorithm.DEFAULT,
    use_fp16: bool = False,
    channels_trailing: bool = False,
    out: None = ...,
) -> T: ...
@overload
def das_sparse(
    x: TensorLike,
    xpos: TensorLike,
    ypos: TensorLike,
    offsets: TensorLike,
    weights: TensorLike,
    sparse_indices: TensorLike,
    *,
    xdir: TensorLike | None = None,
    wavenum: float = 0.0,
    algorithm: _ffdas.Algorithm = _ffdas.Algorithm.DEFAULT,
    use_fp16: bool = False,
    channels_trailing: bool = False,
    out: T,
) -> T: ...

def das_sparse(
    x: TensorLike,
    xpos: TensorLike,
    ypos: TensorLike,
    offsets: TensorLike,
    weights: TensorLike,
    sparse_indices: TensorLike,
    *,
    xdir: TensorLike | None = None,
    wavenum: float = 0.0,
    algorithm: _ffdas.Algorithm = _ffdas.Algorithm.DEFAULT,
    use_fp16: bool = False,
    channels_trailing: bool = False,
    out: TensorLike | None = None,
) -> TensorLike:
    """Sparse compounding delay-and-sum beamforming.

    Like ``das``, but each target compounds over a per-target subset of n
    sequence events, selected by sparse_indices.

    Args:
        x: Input RF data, shape (channels, sequence, samples) or
            (batch, channels, sequence, samples).
        xpos: Channel positions, shape (channels, 3).
        ypos: Target positions, shape (..., 3).
        offsets: Per-target time offsets in samples, shape (n, ...).
        weights: Per-target apodization weights, shape (n, ...).
        sparse_indices: Indices into the sequence dimension of x,
            shape (n, ...). Each target compounds the n events given
            by these indices.
        xdir: Channel directivity vectors, shape (channels, 4).
            See ``das`` for details.
        wavenum: Wavenumber for phase rotation (-2*pi*fc/fs).
        algorithm: Algorithm variant.
        use_fp16: Use half-precision arithmetic.
        channels_trailing: If true, the input dimensions are interpreted as (batch, sequence, channels, samples). Default: False.
        out: Pre-allocated output array.

    Returns:
        Beamformed output with shape (...) or (batch, ...).
    """
    if ypos.ndim < 2 or ypos.shape[-1] != 3:
        raise ValueError("ypos must have dimensions (..., 3)")

    spatial_dims = ypos.shape[:-1]

    out_dims = spatial_dims
    if x.ndim == 4:
        out_dims = (x.shape[0], *out_dims)

    if offsets.shape[1:] != spatial_dims:
        raise ValueError(
            f"offsets shape {offsets.shape} must match target layout {spatial_dims}"
        )
    if weights.shape[1:] != spatial_dims:
        raise ValueError(
            f"weights shape {weights.shape} must match target layout {spatial_dims}"
        )
    if sparse_indices.shape[1:] != spatial_dims:
        raise ValueError(
            f"sparse_indices shape {sparse_indices.shape} must match target layout {spatial_dims}"
        )

    sparse_count = sparse_indices.shape[0]

    if offsets.shape[0] != sparse_count or weights.shape[0] != sparse_count:
        raise ValueError(
            f"offsets and weights leading dimension must match sparse count {sparse_count}"
        )

    xpos = astype(xpos, "float32")
    ypos = astype(ypos, "float32")
    offsets = astype(offsets, "float32")
    weights = astype(weights, "float32")
    xdir = astype(xdir, "float32")
    sparse_indices = astype(sparse_indices, "int32")

    if out is None:
        out = empty_like(x, shape=out_dims)

    beta = zeros_like(x, shape=(1,), device="cpu")
    compute_type = _compute_type(x, use_fp16)

    _ffdas.das_sparse(
        get_library_handle(),
        x,
        xpos,
        ypos,
        offsets,
        weights,
        xdir,
        wavenum,
        beta,
        out,
        sparse_indices,
        algorithm,
        compute_type,
        channels_trailing,
    )
    return out
