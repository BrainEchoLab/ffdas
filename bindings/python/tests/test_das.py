import cupy as cp
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
