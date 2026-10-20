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


def _channels_and_events(x, srcpos, srcdir, channels_trailing):
    """Check x against the channel arrays; return (channels, events) of x."""
    if x.ndim not in (3, 4):
        raise ValueError(
            f"x must have dimensions ([batch,] channels, sequence, samples); got {x.ndim} dimensions"
        )
    channel_axis, event_axis = (-2, -3) if channels_trailing else (-3, -2)
    channels, events = x.shape[channel_axis], x.shape[event_axis]
    if tuple(srcpos.shape) != (channels, 3):
        raise ValueError(f"srcpos shape {srcpos.shape} must be ({channels}, 3) for {channels} channels")
    if srcdir is not None and tuple(srcdir.shape) not in ((channels, 4), (events, channels, 4)):
        raise ValueError(
            f"srcdir shape {srcdir.shape} must be ({channels}, 4), or ({events}, {channels}, 4) "
            f"for one look direction per sequence event and channel"
        )
    return channels, events


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
        weights: Per-target apodization weights, shape (sequence, ...). An
            event that weighs 0 for a target is skipped for it: it costs
            nothing and contributes nothing, even where x is NaN. Not yet
            in ALG4 when a warp mixes zero and nonzero weights: such a
            target can still take up a NaN or Inf that another target of
            its warp reads, or a NaN in its own offset.
        srcdir: Receive directivity: per channel a unit look direction
            (first three components) and the cosine of the sensitivity
            half-angle (fourth). A channel contributes to a target only if
            the target lies inside its cone. Shape (channels, 4), the same
            for every sequence event, or (sequence, channels, 4): one look
            direction per event and channel, e.g. a sub-array pre-steered
            at each transmit's receive focus.
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

    _, events = _channels_and_events(x, srcpos, srcdir, channels_trailing)
    if offsets.shape[0] != events or weights.shape[0] != events:
        raise ValueError(
            f"offsets and weights leading dimension must match the {events} sequence events of x"
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
            by these indices. Entries that weigh 0 (padding) and indices
            outside x are skipped, so the cost follows each target's
            nonzero entries; put those first, in ascending event order,
            so neighbouring targets stay in step.
        srcdir: Receive directivity, shape (channels, 4) or
            (sequence, channels, 4). See ``das``. A per-event srcdir is
            indexed by the event an entry selects (its sparse index), not
            by the entry.
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

    _channels_and_events(x, srcpos, srcdir, channels_trailing)

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
