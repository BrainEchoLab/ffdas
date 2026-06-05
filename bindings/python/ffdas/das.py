from typing import overload

from array_api_compat import array_namespace

from ._core import _ffdas
from ._core.library import get_library_handle
from ._core.tensor_like import TensorLike, T
from ._core.tensor import empty_like, zeros_like, astype


def _ensure_contiguous(x):
    if x is None or isinstance(x, _ffdas.TensorView):
        return x
    if hasattr(x, "flags") and not x.flags["C_CONTIGUOUS"]:
        xp = array_namespace(x)
        return xp.ascontiguousarray(x)
    return x


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
    srcpos: TensorLike,
    dstpos: TensorLike,
    offsets: TensorLike,
    weights: TensorLike,
    *,
    srcdir: TensorLike | None = None,
    wavenum: float = 0.0,
    algorithm: _ffdas.Algorithm = _ffdas.Algorithm.DEFAULT,
    use_fp16: bool = False,
    channels_trailing: bool = False,
    out: None = ...,
) -> T: ...
@overload
def das(
    x: TensorLike,
    srcpos: TensorLike,
    dstpos: TensorLike,
    offsets: TensorLike,
    weights: TensorLike,
    *,
    srcdir: TensorLike | None = None,
    wavenum: float = 0.0,
    algorithm: _ffdas.Algorithm = _ffdas.Algorithm.DEFAULT,
    use_fp16: bool = False,
    channels_trailing: bool = False,
    out: T,
) -> T: ...

def das(
    x: TensorLike,
    srcpos: TensorLike,
    dstpos: TensorLike,
    offsets: TensorLike,
    weights: TensorLike,
    *,
    srcdir: TensorLike | None = None,
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
        srcpos: Source (channel) positions, shape (channels, 3).
        dstpos: Destination (target) positions, shape (..., 3). The spatial
            dimensions are dstpos.shape[:-1].
        offsets: Per-target time offsets in samples, shape (sequence, ...).
        weights: Per-target apodization weights, shape (sequence, ...).
        srcdir: Source directivity vectors, shape (channels, 4). The first
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
    if dstpos.ndim < 2 or dstpos.shape[-1] != 3:
        raise ValueError("dstpos must have dimensions (..., 3)")

    spatial_dims = dstpos.shape[:-1]

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

    srcpos = _ensure_contiguous(astype(srcpos, "float32"))
    dstpos = _ensure_contiguous(astype(dstpos, "float32"))
    offsets = _ensure_contiguous(astype(offsets, "float32"))
    weights = _ensure_contiguous(astype(weights, "float32"))
    srcdir = _ensure_contiguous(astype(srcdir, "float32"))

    if out is None:
        out = empty_like(x, shape=out_dims)

    beta = zeros_like(x, shape=(1,), device="cpu")
    compute_type = _compute_type(x, use_fp16)

    _ffdas.das(
        get_library_handle(),
        x,
        srcpos,
        dstpos,
        offsets,
        weights,
        srcdir,
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
    srcpos: TensorLike,
    dstpos: TensorLike,
    offsets: TensorLike,
    weights: TensorLike,
    sparse_indices: TensorLike,
    *,
    srcdir: TensorLike | None = None,
    wavenum: float = 0.0,
    algorithm: _ffdas.Algorithm = _ffdas.Algorithm.DEFAULT,
    use_fp16: bool = False,
    channels_trailing: bool = False,
    out: None = ...,
) -> T: ...
@overload
def das_sparse(
    x: TensorLike,
    srcpos: TensorLike,
    dstpos: TensorLike,
    offsets: TensorLike,
    weights: TensorLike,
    sparse_indices: TensorLike,
    *,
    srcdir: TensorLike | None = None,
    wavenum: float = 0.0,
    algorithm: _ffdas.Algorithm = _ffdas.Algorithm.DEFAULT,
    use_fp16: bool = False,
    channels_trailing: bool = False,
    out: T,
) -> T: ...

def das_sparse(
    x: TensorLike,
    srcpos: TensorLike,
    dstpos: TensorLike,
    offsets: TensorLike,
    weights: TensorLike,
    sparse_indices: TensorLike,
    *,
    srcdir: TensorLike | None = None,
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
        srcpos: Source (channel) positions, shape (channels, 3).
        dstpos: Destination (target) positions, shape (..., 3).
        offsets: Per-target time offsets in samples, shape (n, ...).
        weights: Per-target apodization weights, shape (n, ...).
        sparse_indices: Indices into the sequence dimension of x,
            shape (n, ...). Each target compounds the n events given
            by these indices.
        srcdir: Source directivity vectors, shape (channels, 4).
            See ``das`` for details.
        wavenum: Wavenumber for phase rotation (-2*pi*fc/fs).
        algorithm: Algorithm variant.
        use_fp16: Use half-precision arithmetic.
        channels_trailing: If true, the input dimensions are interpreted as (batch, sequence, channels, samples). Default: False.
        out: Pre-allocated output array.

    Returns:
        Beamformed output with shape (...) or (batch, ...).
    """
    if dstpos.ndim < 2 or dstpos.shape[-1] != 3:
        raise ValueError("dstpos must have dimensions (..., 3)")

    spatial_dims = dstpos.shape[:-1]

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

    srcpos = _ensure_contiguous(astype(srcpos, "float32"))
    dstpos = _ensure_contiguous(astype(dstpos, "float32"))
    offsets = _ensure_contiguous(astype(offsets, "float32"))
    weights = _ensure_contiguous(astype(weights, "float32"))
    srcdir = _ensure_contiguous(astype(srcdir, "float32"))
    sparse_indices = _ensure_contiguous(astype(sparse_indices, "int32"))

    if out is None:
        out = empty_like(x, shape=out_dims)

    beta = zeros_like(x, shape=(1,), device="cpu")
    compute_type = _compute_type(x, use_fp16)

    _ffdas.das_sparse(
        get_library_handle(),
        x,
        srcpos,
        dstpos,
        offsets,
        weights,
        srcdir,
        wavenum,
        beta,
        out,
        sparse_indices,
        algorithm,
        compute_type,
        channels_trailing,
    )
    return out
