import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose

import ffdas

from das_reference import das_reference


def _das_args(**overrides):
    """Minimal valid cupy argument set."""
    n_ch, n_seq, n_samp, n_tgt = 4, 2, 16, 8
    args = dict(
        x=cp.zeros((n_ch, n_seq, n_samp), dtype="float32"),
        srcpos=cp.zeros((n_ch, 3), dtype="float32"),
        dstpos=cp.zeros((n_tgt, 3), dtype="float32"),
        offsets=cp.zeros((n_seq, n_tgt), dtype="float32"),
        weights=cp.ones((n_seq, n_tgt), dtype="float32"),
    )
    args.update(overrides)
    return args


def _das_sparse_args(**overrides):
    n_ch, n_seq, n_samp, n_tgt, n_sparse = 4, 8, 16, 6, 3
    args = dict(
        x=cp.zeros((n_ch, n_seq, n_samp), dtype="float32"),
        srcpos=cp.zeros((n_ch, 3), dtype="float32"),
        dstpos=cp.zeros((n_tgt, 3), dtype="float32"),
        offsets=cp.zeros((n_sparse, n_tgt), dtype="float32"),
        weights=cp.ones((n_sparse, n_tgt), dtype="float32"),
        sparse_indices=cp.zeros((n_sparse, n_tgt), dtype="int32"),
    )
    args.update(overrides)
    return args


# raises


def test_das_dstpos_wrong_trailing_dim():
    with pytest.raises(ValueError):
        ffdas.das(**_das_args(dstpos=cp.zeros((8, 2), dtype="float32")))


def test_das_dstpos_1d():
    with pytest.raises(ValueError):
        ffdas.das(**_das_args(dstpos=cp.zeros((3,), dtype="float32")))


def test_das_offsets_shape_mismatch():
    with pytest.raises(ValueError):
        ffdas.das(**_das_args(offsets=cp.zeros((2, 99), dtype="float32")))


def test_das_weights_shape_mismatch():
    with pytest.raises(ValueError):
        ffdas.das(**_das_args(weights=cp.zeros((2, 99), dtype="float32")))


# raises


def test_das_sparse_indices_shape_mismatch():
    with pytest.raises(ValueError):
        ffdas.das_sparse(
            **_das_sparse_args(
                sparse_indices=cp.zeros((99, 6), dtype="int32"),
            )
        )


def test_das_sparse_offsets_count_mismatch():
    with pytest.raises(ValueError):
        ffdas.das_sparse(
            **_das_sparse_args(
                offsets=cp.zeros((5, 6), dtype="float32"),
            )
        )


# General tests


def test_das_zero_input_zero_output():
    result = ffdas.das(**_das_args())
    assert_allclose(cp.asnumpy(result), 0.0, atol=1e-6)


def test_das_zero_weights_zero_output(rng):
    args = _das_args()
    args["x"] = cp.array(rng.standard_normal(args["x"].shape).astype("float32"))
    args["weights"] = cp.zeros_like(args["weights"])
    result = ffdas.das(**args)
    assert_allclose(cp.asnumpy(result), 0.0, atol=1e-6)


def test_das_output_shape_3d():
    n_ch, n_seq, n_samp = 4, 2, 32
    nz, nx = 8, 6
    result = ffdas.das(
        x=cp.zeros((n_ch, n_seq, n_samp), dtype="float32"),
        srcpos=cp.zeros((n_ch, 3), dtype="float32"),
        dstpos=cp.zeros((nz, nx, 3), dtype="float32"),
        offsets=cp.zeros((n_seq, nz, nx), dtype="float32"),
        weights=cp.ones((n_seq, nz, nx), dtype="float32"),
    )
    assert result.shape == (nz, nx)


def test_das_output_shape_4d_batched():
    batch, n_ch, n_seq, n_samp, n_tgt = 3, 4, 2, 32, 8
    result = ffdas.das(
        x=cp.zeros((batch, n_ch, n_seq, n_samp), dtype="float32"),
        srcpos=cp.zeros((n_ch, 3), dtype="float32"),
        dstpos=cp.zeros((n_tgt, 3), dtype="float32"),
        offsets=cp.zeros((n_seq, n_tgt), dtype="float32"),
        weights=cp.ones((n_seq, n_tgt), dtype="float32"),
    )
    assert result.shape == (batch, n_tgt)


def test_das_single_sample_impulse():
    """A nonzero sample at a known offset should produce nonzero output."""
    x = cp.zeros((1, 1, 64), dtype="float32")
    x[0, 0, 32] = 1.0
    result = ffdas.das(
        x=x,
        srcpos=cp.zeros((1, 3), dtype="float32"),
        dstpos=cp.zeros((1, 3), dtype="float32"),
        offsets=cp.full((1, 1), 32.0, dtype="float32"),
        weights=cp.ones((1, 1), dtype="float32"),
    )
    assert abs(float(result.squeeze())) > 0.5


# out-of-range sample indices (#2): a sample index outside [0, samples - 2] contributes exactly 0, and
# nothing outside the data is read for it -- in every algorithm. ALG4 exists only for complex input on
# SM 8.0+; those cases skip, and any error from any other case fails the test.

ALGORITHMS = [
    pytest.param(ffdas.Algorithm.ALG1, False, id="ALG1"),
    pytest.param(ffdas.Algorithm.ALG2, False, id="ALG2"),
    pytest.param(ffdas.Algorithm.ALG2, True, id="ALG2-fp16"),
    pytest.param(ffdas.Algorithm.ALG4, False, id="ALG4"),
    pytest.param(ffdas.Algorithm.ALG4, True, id="ALG4-fp16"),
]


def _das(**kwargs):
    """ffdas.das, skipping only the combinations ALG4 does not implement."""
    if kwargs["algorithm"] == ffdas.Algorithm.ALG4:
        if not np.issubdtype(kwargs["x"].dtype, np.complexfloating):
            pytest.skip("ALG4 takes complex input only")
        if int(cp.cuda.Device().compute_capability) < 80:
            pytest.skip("ALG4 needs compute capability 8.0 or higher")
    return ffdas.das(**kwargs)


def _nan_padded(shape, dtype, pad=4096):
    """Zeros sitting inside a NaN-filled buffer. A read before the first or past the last sample lands
    in the padding and shows up as NaN in the output (0 * NaN), instead of depending on whether the
    neighbouring device memory happens to be mapped. (ALG2 and ALG4 re-lay x into a buffer of their
    own first: for them see test_das_out_of_range_sample_is_not_read_from_the_relayout_buffer.)"""
    n = int(np.prod(shape))
    buf = cp.full((n + 2 * pad,), np.nan, dtype=dtype)
    buf[pad:pad + n] = 0
    return buf[pad:pad + n].reshape(shape)


def _co_located(n_ch, n_tgt, r):
    """Every target r samples from every channel (all channels at the origin)."""
    dstpos = cp.zeros((n_tgt, 3), dtype="float32")
    dstpos[:, 2] = r
    return cp.zeros((n_ch, 3), dtype="float32"), dstpos


def _assert_exact_zero(out):
    cp.cuda.Device().synchronize()
    out = cp.asnumpy(out)
    assert np.isfinite(out).all() and np.all(out == 0), out.ravel()[:4]


@pytest.mark.parametrize("algorithm,use_fp16", ALGORITHMS)
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_das_issue2_nan_in_last_sample_is_not_used(algorithm, use_fp16, dtype):
    """The reproduction from #2: the index lands on the last sample (NaN), so k + 1 is past the end.
    (With a batch axis: ALG2 only takes batched input.)"""
    samples = 4
    x = cp.asarray(np.array([[[[1.0, 1.0, 1.0, np.nan]]]]).astype(dtype))
    out = _das(
        x=x,
        srcpos=cp.zeros((1, 3), dtype="float32"),
        dstpos=cp.zeros((1, 3), dtype="float32"),
        offsets=cp.array(np.array([[samples - 1]], dtype=np.float32)),
        weights=cp.ones((1, 1), dtype="float32"),
        algorithm=algorithm,
        use_fp16=use_fp16,
    )
    _assert_exact_zero(out)


@pytest.mark.parametrize("algorithm,use_fp16", ALGORITHMS)
@pytest.mark.parametrize("target_sample", [-60.0, 250.0], ids=["before-start", "past-end"])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("batch", [1, 3, 16])
def test_das_out_of_range_sample_is_not_read(algorithm, use_fp16, target_sample, dtype, batch):
    n_ch, n_seq, n_samp, n_tgt, r = 4, 2, 191, 8, 10.0
    x = _nan_padded((batch, n_ch, n_seq, n_samp), dtype)
    srcpos, dstpos = _co_located(n_ch, n_tgt, r)
    out = _das(
        x=x,
        srcpos=srcpos,
        dstpos=dstpos,
        offsets=cp.full((n_seq, n_tgt), target_sample - r, dtype="float32"),  # phase = offset + r
        weights=cp.ones((n_seq, n_tgt), dtype="float32"),
        wavenum=-0.5,
        algorithm=algorithm,
        use_fp16=use_fp16,
    )
    _assert_exact_zero(out)


@pytest.mark.parametrize("algorithm,use_fp16", ALGORITHMS)
@pytest.mark.parametrize("target_sample", [-60.0, 250.0], ids=["before-start", "past-end"])
@pytest.mark.parametrize("nan_at", [0, 190], ids=["first-sample", "last-sample"])
def test_das_out_of_range_target_ignores_nan_in_the_recording(algorithm, use_fp16, target_sample, nan_at):
    """Out-of-range targets must contribute an exact 0 whatever the recording holds: a NaN at the
    first or last sample of every row must not reach them (an index merely clamped into the data
    would still multiply a 0 weight by that NaN)."""
    n_ch, n_seq, n_samp, n_tgt, r = 8, 2, 191, 64, 10.0
    x = cp.ones((1, n_ch, n_seq, n_samp), dtype="complex64")
    x[..., nan_at] = complex(np.nan, np.nan)
    srcpos, dstpos = _co_located(n_ch, n_tgt, r)
    out = _das(
        x=x,
        srcpos=srcpos,
        dstpos=dstpos,
        offsets=cp.full((n_seq, n_tgt), target_sample - r, dtype="float32"),
        weights=cp.ones((n_seq, n_tgt), dtype="float32"),
        wavenum=-0.5,
        algorithm=algorithm,
        use_fp16=use_fp16,
    )
    _assert_exact_zero(out)


@pytest.mark.parametrize("algorithm,use_fp16", ALGORITHMS)
def test_das_directivity_rejected_sample_is_not_used(algorithm, use_fp16):
    """A channel the target is outside the sensitivity cone of contributes an exact 0, like an
    out-of-range sample: the (in-range) data it would have used is NaN here."""
    n_ch, n_seq, n_samp, n_tgt, r = 8, 2, 191, 64, 10.0
    x = cp.full((1, n_ch, n_seq, n_samp), complex(np.nan, np.nan), dtype="complex64")
    srcpos, dstpos = _co_located(n_ch, n_tgt, r)  # targets at +z ...
    srcdir = cp.zeros((n_ch, 4), dtype="float32")
    srcdir[:, 2] = -1.0  # ... every channel looks along -z, half-angle 90 degrees
    out = _das(
        x=x,
        srcpos=srcpos,
        dstpos=dstpos,
        offsets=cp.full((n_seq, n_tgt), 50.0 - r, dtype="float32"),
        weights=cp.ones((n_seq, n_tgt), dtype="float32"),
        srcdir=srcdir,
        wavenum=-0.5,
        algorithm=algorithm,
        use_fp16=use_fp16,
    )
    _assert_exact_zero(out)


@pytest.mark.parametrize("target_sample", [-60.0, 250.0], ids=["before-start", "past-end"])
def test_das_sparse_out_of_range_sample_is_not_read(target_sample):
    n_ch, n_seq, n_samp, n_tgt, n_sparse, r = 4, 3, 191, 8, 2, 10.0
    x = _nan_padded((1, n_ch, n_seq, n_samp), "complex64")
    srcpos, dstpos = _co_located(n_ch, n_tgt, r)
    out = ffdas.das_sparse(
        x=x,
        srcpos=srcpos,
        dstpos=dstpos,
        offsets=cp.full((n_sparse, n_tgt), target_sample - r, dtype="float32"),
        weights=cp.ones((n_sparse, n_tgt), dtype="float32"),
        sparse_indices=cp.tile(cp.asarray([[0], [2]], dtype="int32"), (1, n_tgt)),
        wavenum=-0.5,
    )
    _assert_exact_zero(out)


@pytest.mark.parametrize("algorithm,use_fp16", ALGORITHMS)
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_das_single_sample_recording_reads_nothing_outside(algorithm, use_fp16, dtype):
    """With one sample there is no valid interpolation pair: every target is out of range, and the
    second interpolation tap must not step past the row either."""
    x = _nan_padded((1, 1, 1, 1), dtype)  # (batch, channels, sequence, samples)
    x[...] = np.nan  # the one sample itself is NaN too: it must not be used
    srcpos, dstpos = _co_located(1, 4, 2.0)
    out = _das(
        x=x,
        srcpos=srcpos,
        dstpos=dstpos,
        offsets=cp.zeros((1, 4), dtype="float32"),
        weights=cp.ones((1, 4), dtype="float32"),
        algorithm=algorithm,
        use_fp16=use_fp16,
    )
    _assert_exact_zero(out)


# ALG2 / ALG4 re-lay x out into a buffer from the library's caching allocator, so neither a NaN-padded
# input nor compute-sanitizer can see a read past THAT buffer. The allocator hands out bins of
# 512 B * 8**n and reuses a freed block for the next request in the same bin: a NaN-filled call whose
# buffer fills the bin leaves NaN in the slack, and the real (zero-filled, smaller) call that follows
# reuses the block -- an out-of-range load then returns NaN into the output. A false PASS is possible
# if the allocator stops reusing blocks; a false failure is not.


def _das_native(samples, fill, k, *, algorithm, use_fp16, n_ch=8, n_seq=2, n_tgt=64):
    x = cp.full((1, n_ch, n_seq, samples), fill, dtype="complex64")
    srcpos, dstpos = _co_located(n_ch, n_tgt, 10.0)
    return _das(
        x=x,
        srcpos=srcpos,
        dstpos=dstpos,
        offsets=cp.full((n_seq, n_tgt), k - 10.0, dtype="float32"),  # phase = ofs + r = k
        weights=cp.ones((n_seq, n_tgt), dtype="float32"),
        wavenum=-0.5,
        algorithm=algorithm,
        use_fp16=use_fp16,
    )


@pytest.mark.parametrize("algorithm,use_fp16", ALGORITHMS)
def test_das_out_of_range_sample_is_not_read_from_the_relayout_buffer(algorithm, use_fp16):
    _das_native(255, complex(np.nan, np.nan), 50.0, algorithm=algorithm, use_fp16=use_fp16)  # fills the 32 KiB bin with NaN
    out = _das_native(191, 0j, 250.0, algorithm=algorithm, use_fp16=use_fp16)  # same bin; targets 59 samples past the end
    _assert_exact_zero(out)


# ALG2 vectorises over the batch (2 or 4 columns per load, whenever the batch size allows it) and tiles
# it 16..64 columns per block. The batch sizes below cover every combination of load width (1, 2, 4) and
# tile width (16, 32, 64), one and several tiles, and a partly filled last tile. Its per-thread accumulators were sized M * N / 32, which is too few when a
# warp's subtile is wider than the tile (small even batches), and every thread wrote back up to the full
# batch rather than its own tile (batches > 64): both returned garbage or NaN for whole batch rows.


@pytest.mark.parametrize("use_fp16", [False, True], ids=["fp32", "fp16"])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("batch", [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 33, 40, 48, 64, 65, 96, 128, 129, 130])
def test_das_alg2_matches_alg1_across_batch_sizes(batch, dtype, use_fp16):
    rng = np.random.default_rng(batch)
    n_ch, n_seq, n_samp, n_tgt = 8, 2, 191, 64
    xr = rng.standard_normal((batch, n_ch, n_seq, n_samp))
    if dtype == "complex64":
        xr = xr + 1j * rng.standard_normal(xr.shape)
    x = cp.asarray(xr.astype(dtype))
    dstpos = np.zeros((n_tgt, 3), dtype="float32")
    dstpos[:, 0] = np.linspace(-5, 5, n_tgt)
    dstpos[:, 2] = 60.0
    args = dict(
        srcpos=cp.asarray(np.stack([np.linspace(-5, 5, n_ch), np.zeros(n_ch), np.zeros(n_ch)], 1).astype("float32")),
        dstpos=cp.asarray(dstpos),
        offsets=cp.full((n_seq, n_tgt), 10.0, dtype="float32"),
        weights=cp.ones((n_seq, n_tgt), dtype="float32"),
        wavenum=-0.5,
    )
    ref = cp.asnumpy(ffdas.das(x=x, **args, algorithm=ffdas.Algorithm.ALG1))
    out = cp.asnumpy(ffdas.das(x=x, **args, algorithm=ffdas.Algorithm.ALG2, use_fp16=use_fp16))
    # ALG2 carries the interpolation fraction in fp16 even in fp32 mode
    tol = (3e-2 if use_fp16 else 5e-3) * np.abs(ref).max()
    assert np.isfinite(out).all()
    assert_allclose(out, ref, rtol=0, atol=tol)


# ALG1 picks its batch tile as the largest power of two up to 128 bytes of accumulators per thread. For
# real (4-byte) output that gave 32, a width no kernel was compiled for, so real input at batch >= 32
# was refused ("Invalid dimensions"). The tile is now capped at 16.


@pytest.mark.parametrize("use_fp16", [False, True], ids=["fp32", "fp16"])
@pytest.mark.parametrize("batch", [32, 33, 64, 100])
def test_das_alg1_real_input_at_large_batch(batch, use_fp16):
    rng = np.random.default_rng(batch)
    n_ch, n_seq, n_samp, n_tgt = 8, 2, 191, 64
    x = cp.asarray(rng.standard_normal((batch, n_ch, n_seq, n_samp)).astype("float32"))
    dstpos = np.zeros((n_tgt, 3), dtype="float32")
    dstpos[:, 0] = np.linspace(-5, 5, n_tgt)
    dstpos[:, 2] = 60.0
    args = dict(
        srcpos=cp.asarray(np.stack([np.linspace(-5, 5, n_ch), np.zeros(n_ch), np.zeros(n_ch)], 1).astype("float32")),
        dstpos=cp.asarray(dstpos),
        offsets=cp.full((n_seq, n_tgt), 10.0, dtype="float32"),
        weights=cp.ones((n_seq, n_tgt), dtype="float32"),
        algorithm=ffdas.Algorithm.ALG1,
        use_fp16=use_fp16,
    )
    out = cp.asnumpy(ffdas.das(x=x, **args))
    chunks = np.concatenate([cp.asnumpy(ffdas.das(x=x[b:b + 16], **args)) for b in range(0, batch, 16)])
    assert_allclose(out, chunks, rtol=1e-5, atol=1e-6 * np.abs(chunks).max())


# Each sequence event's own offsets and weights. Without a batch dimension (3-D x) every event used
# event 0's offsets / weights / sparse indices: their row stride was taken from the output's batch
# stride, which is 0 then. Checked against a float64 numpy reference (das_reference.py).


def _problem(rng, *, x_shape, n_seq, n_tgt, dtype="complex64"):
    """Channels on a line, targets 40-60 samples deep, offsets and weights that differ per event."""
    n_ch = x_shape[-3]
    xr = rng.standard_normal(x_shape)
    if dtype == "complex64":
        xr = xr + 1j * rng.standard_normal(x_shape)
    dstpos = np.stack([rng.uniform(-5, 5, n_tgt), rng.uniform(-2, 2, n_tgt), rng.uniform(40, 60, n_tgt)], 1)
    srcpos = np.stack([np.linspace(-5, 5, n_ch), np.zeros(n_ch), np.zeros(n_ch)], 1)
    return dict(
        x=xr.astype(dtype),
        srcpos=srcpos.astype("float32"),
        dstpos=dstpos.astype("float32"),
        offsets=rng.uniform(0, 20, (n_seq, n_tgt)).astype("float32"),
        weights=rng.uniform(0.5, 1.5, (n_seq, n_tgt)).astype("float32"),
    )


def _on_gpu(args):
    return {k: cp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in args.items()}


def _assert_matches(out, ref, rel):
    out = cp.asnumpy(out)
    assert out.shape == ref.shape
    assert_allclose(out, ref, rtol=0, atol=rel * np.abs(ref).max())


@pytest.mark.parametrize("algorithm,use_fp16", ALGORITHMS)
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("batch", [1, 4])
def test_das_matches_reference(algorithm, use_fp16, dtype, batch, rng):
    n_ch, n_seq, n_samp, n_tgt = 8, 3, 191, 200
    args = _problem(rng, x_shape=(batch, n_ch, n_seq, n_samp), n_seq=n_seq, n_tgt=n_tgt, dtype=dtype)
    ref = das_reference(**args, wavenum=-0.5)
    out = _das(**_on_gpu(args), wavenum=-0.5, algorithm=algorithm, use_fp16=use_fp16)
    _assert_matches(out, ref, 3e-2 if use_fp16 else 5e-3)


@pytest.mark.parametrize("algorithm", [ffdas.Algorithm.ALG1, ffdas.Algorithm.ALG4], ids=["ALG1", "ALG4"])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_das_unbatched_input_uses_each_events_offsets_and_weights(algorithm, dtype, rng):
    n_ch, n_seq, n_samp, n_tgt = 8, 3, 191, 200
    args = _problem(rng, x_shape=(n_ch, n_seq, n_samp), n_seq=n_seq, n_tgt=n_tgt, dtype=dtype)
    ref = das_reference(**args, wavenum=-0.5)
    gpu = _on_gpu(args)
    out = _das(**gpu, wavenum=-0.5, algorithm=algorithm)
    # ALG4 multiplies on the tensor cores (tf32 / fp16 operands)
    _assert_matches(out, ref, 5e-3 if algorithm == ffdas.Algorithm.ALG4 else 1e-5)
    batched = _das(**{**gpu, "x": gpu["x"][None]}, wavenum=-0.5, algorithm=algorithm)
    assert_allclose(cp.asnumpy(out), cp.asnumpy(batched)[0], rtol=0, atol=0)


def test_das_sparse_unbatched_input_uses_each_entrys_event(rng):
    n_ch, n_seq, n_samp, n_tgt, n_sparse = 8, 6, 191, 200, 3
    args = _problem(rng, x_shape=(n_ch, n_seq, n_samp), n_seq=n_sparse, n_tgt=n_tgt)
    args["sparse_indices"] = np.stack([rng.permutation(n_seq)[:n_sparse] for _ in range(n_tgt)], 1).astype("int32")
    ref = das_reference(**args, wavenum=-0.5)
    gpu = _on_gpu(args)
    out = ffdas.das_sparse(**gpu, wavenum=-0.5)
    _assert_matches(out, ref, 1e-5)
    batched = ffdas.das_sparse(**{**gpu, "x": gpu["x"][None]}, wavenum=-0.5)
    assert_allclose(cp.asnumpy(out), cp.asnumpy(batched)[0], rtol=0, atol=0)


@pytest.mark.parametrize("algorithm,use_fp16", ALGORITHMS)
def test_das_zero_targets_gives_an_empty_result(algorithm, use_fp16):
    n_ch, n_seq, n_samp = 4, 2, 64
    out = _das(
        x=cp.ones((2, n_ch, n_seq, n_samp), dtype="complex64"),
        srcpos=cp.zeros((n_ch, 3), dtype="float32"),
        dstpos=cp.zeros((0, 3), dtype="float32"),
        offsets=cp.zeros((n_seq, 0), dtype="float32"),
        weights=cp.ones((n_seq, 0), dtype="float32"),
        algorithm=algorithm,
        use_fp16=use_fp16,
    )
    assert out.shape == (2, 0)


def test_das_sparse_zero_targets_gives_an_empty_result():
    n_ch, n_seq, n_samp, n_sparse = 4, 3, 64, 2
    out = ffdas.das_sparse(
        x=cp.ones((n_ch, n_seq, n_samp), dtype="float32"),
        srcpos=cp.zeros((n_ch, 3), dtype="float32"),
        dstpos=cp.zeros((0, 3), dtype="float32"),
        offsets=cp.zeros((n_sparse, 0), dtype="float32"),
        weights=cp.ones((n_sparse, 0), dtype="float32"),
        sparse_indices=cp.zeros((n_sparse, 0), dtype="int32"),
    )
    assert out.shape == (0,)


@pytest.mark.parametrize("algorithm,use_fp16", ALGORITHMS)
def test_das_empty_batch_gives_an_empty_result(algorithm, use_fp16):
    n_ch, n_seq, n_samp, n_tgt = 4, 2, 64, 8
    out = _das(
        x=cp.ones((0, n_ch, n_seq, n_samp), dtype="complex64"),
        srcpos=cp.zeros((n_ch, 3), dtype="float32"),
        dstpos=cp.zeros((n_tgt, 3), dtype="float32"),
        offsets=cp.zeros((n_seq, n_tgt), dtype="float32"),
        weights=cp.ones((n_seq, n_tgt), dtype="float32"),
        algorithm=algorithm,
        use_fp16=use_fp16,
    )
    assert out.shape == (0, n_tgt)


def test_das_offsets_event_count_mismatch():
    with pytest.raises(ValueError):
        ffdas.das(**_das_args(offsets=cp.zeros((3, 8), dtype="float32"), weights=cp.ones((3, 8), dtype="float32")))


def test_das_srcpos_channel_count_mismatch():
    with pytest.raises(ValueError):
        ffdas.das(**_das_args(srcpos=cp.zeros((5, 3), dtype="float32")))


def test_das_srcdir_channel_count_mismatch():
    with pytest.raises(ValueError):
        ffdas.das(**_das_args(srcdir=cp.zeros((5, 4), dtype="float32")))


def test_das_sparse_srcpos_channel_count_mismatch():
    with pytest.raises(ValueError):
        ffdas.das_sparse(**_das_sparse_args(srcpos=cp.zeros((5, 3), dtype="float32")))


def test_das_non_contiguous_out_is_refused():
    """The kernels write the output as one dense block; a strided out= used to be written (and its
    offsets read) at the wrong places."""
    args = _das_args(x=cp.zeros((2, 4, 2, 16), dtype="float32"))
    out = cp.zeros((2, 16), dtype="float32")[:, ::2]
    with pytest.raises(RuntimeError):
        ffdas.das(**args, out=out)
