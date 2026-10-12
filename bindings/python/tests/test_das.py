import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose

import ffdas


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
# nothing outside the data is read for it


def _nan_padded(shape, dtype, pad=4096):
    """Zeros sitting inside a NaN-filled buffer. A read before the first or past the last sample lands
    in the padding and shows up as NaN in the output (0 * NaN), instead of depending on whether the
    neighbouring device memory happens to be mapped."""
    n = int(np.prod(shape))
    buf = cp.full((n + 2 * pad,), np.nan, dtype=dtype)
    buf[pad:pad + n] = 0
    return buf[pad:pad + n].reshape(shape)


def _co_located(n_ch, n_tgt, r):
    """Every target r samples from every channel (all channels at the origin)."""
    dstpos = cp.zeros((n_tgt, 3), dtype="float32")
    dstpos[:, 2] = r
    return cp.zeros((n_ch, 3), dtype="float32"), dstpos


def test_das_issue2_nan_in_last_sample_is_not_used():
    """The reproduction from #2: the index lands on the last sample (NaN), so k + 1 is past the end."""
    samples = 4
    x = cp.array(np.array([[[1.0, 1.0, 1.0, np.nan]]], dtype=np.float32))
    out = ffdas.das(
        x=x,
        srcpos=cp.zeros((1, 3), dtype="float32"),
        dstpos=cp.zeros((1, 3), dtype="float32"),
        offsets=cp.array(np.array([[samples - 1]], dtype=np.float32)),
        weights=cp.ones((1, 1), dtype="float32"),
        algorithm=ffdas.Algorithm.ALG1,
    )
    value = cp.asnumpy(out)[0]
    assert np.isfinite(value) and value == 0.0, value


@pytest.mark.parametrize("target_sample", [-60.0, 250.0], ids=["before-start", "past-end"])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("batch", [1, 3, 16])
def test_das_out_of_range_sample_is_not_read(target_sample, dtype, batch):
    n_ch, n_seq, n_samp, n_tgt, r = 4, 2, 191, 8, 10.0
    x = _nan_padded((batch, n_ch, n_seq, n_samp), dtype)
    srcpos, dstpos = _co_located(n_ch, n_tgt, r)
    out = ffdas.das(
        x=x,
        srcpos=srcpos,
        dstpos=dstpos,
        offsets=cp.full((n_seq, n_tgt), target_sample - r, dtype="float32"),  # phase = offset + r
        weights=cp.ones((n_seq, n_tgt), dtype="float32"),
        wavenum=-0.5,
        algorithm=ffdas.Algorithm.ALG1,
    )
    cp.cuda.Device().synchronize()
    out = cp.asnumpy(out)
    assert not np.isnan(out).any(), "a sample outside the data was read"
    assert_allclose(out, 0.0)


@pytest.mark.parametrize("nan_at", [0, 190], ids=["first-sample", "last-sample"])
def test_das_out_of_range_target_ignores_nan_in_the_recording(nan_at):
    """Out-of-range targets must contribute an exact 0 whatever the recording holds: a NaN at the
    first or last sample of every row must not reach them."""
    n_ch, n_seq, n_samp, n_tgt, r = 8, 2, 191, 64, 10.0
    x = cp.ones((1, n_ch, n_seq, n_samp), dtype="complex64")
    x[..., nan_at] = complex(np.nan, np.nan)
    srcpos, dstpos = _co_located(n_ch, n_tgt, r)
    out = ffdas.das(
        x=x,
        srcpos=srcpos,
        dstpos=dstpos,
        offsets=cp.full((n_seq, n_tgt), 250.0 - r, dtype="float32"),  # every target needs sample 250
        weights=cp.ones((n_seq, n_tgt), dtype="float32"),
        wavenum=-0.5,
        algorithm=ffdas.Algorithm.ALG1,
    )
    cp.cuda.Device().synchronize()
    out = cp.asnumpy(out)
    assert np.isfinite(out).all() and np.all(out == 0), out.ravel()[:4]


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
    cp.cuda.Device().synchronize()
    out = cp.asnumpy(out)
    assert not np.isnan(out).any(), "a sample outside the data was read"
    assert_allclose(out, 0.0)


def test_das_single_sample_recording_reads_nothing_outside():
    """With one sample there is no valid interpolation pair: every target is out of range, and the
    second interpolation tap must not step past the row either."""
    x = _nan_padded((1, 1, 1), "float32")
    x[...] = np.nan  # the one sample itself is NaN too: it must not be used
    srcpos, dstpos = _co_located(1, 4, 2.0)
    out = ffdas.das(
        x=x,
        srcpos=srcpos,
        dstpos=dstpos,
        offsets=cp.zeros((1, 4), dtype="float32"),
        weights=cp.ones((1, 4), dtype="float32"),
        algorithm=ffdas.Algorithm.ALG1,
    )
    cp.cuda.Device().synchronize()
    out = cp.asnumpy(out)
    assert np.isfinite(out).all() and np.all(out == 0), out
