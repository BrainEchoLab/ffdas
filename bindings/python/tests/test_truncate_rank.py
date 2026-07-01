import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose

import ffdas


# raisees

def test_truncate_rank_1d_rejected():
    with pytest.raises(ValueError):
        ffdas.truncate_rank(cp.ones(10), start=0)


def test_truncate_rank_start_negative():
    with pytest.raises(ValueError):
        ffdas.truncate_rank(cp.ones((4, 8)), start=-1)


def test_truncate_rank_start_too_large():
    with pytest.raises(ValueError):
        ffdas.truncate_rank(cp.ones((4, 8)), start=4)


def test_truncate_rank_stop_before_start():
    with pytest.raises(ValueError):
        ffdas.truncate_rank(cp.ones((4, 8)), start=2, stop=1)


def test_truncate_rank_stop_too_large():
    with pytest.raises(ValueError):
        ffdas.truncate_rank(cp.ones((4, 8)), start=0, stop=5)


# output

def _svd_filter(x, start, stop):
    """Reference: reconstruct x using singular vectors start:stop."""
    m = x.shape[0]
    n = np.prod(x.shape[1:])
    U, S, Vh = np.linalg.svd(x.reshape(m, n), full_matrices=False)
    return (U[:, start:stop] @ np.diag(S[start:stop]) @ Vh[start:stop, :]).reshape(x.shape)


def test_truncate_rank_keep_all(rng):
    x_np = rng.standard_normal((8, 32)).astype("float32")
    m, n = x_np.shape
    result = ffdas.truncate_rank(cp.array(x_np), start=0, stop=min(m, n))
    assert_allclose(cp.asnumpy(result), x_np, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("start, stop", [(0, 4), (1, 8), (2, 6)])
def test_truncate_rank_truncated(start, stop, rng):
    x_np = rng.standard_normal((16, 64)).astype("float32")
    expected = _svd_filter(x_np, start, stop)
    result = ffdas.truncate_rank(cp.array(x_np), start=start, stop=stop)
    assert_allclose(cp.asnumpy(result), expected, rtol=1e-2, atol=1e-3)


def test_truncate_rank_output_shape_3d(rng):
    x = cp.array(rng.standard_normal((8, 4, 16)).astype("float32"))
    assert ffdas.truncate_rank(x, start=0, stop=4).shape == x.shape


@pytest.mark.parametrize("shape", [(16, 64), (64, 16), (32, 32)])
def test_truncate_rank_shape_variants(shape, rng):
    x_np = rng.standard_normal(shape).astype("float32")
    k = min(shape)
    start, stop = 1, k - 1
    expected = _svd_filter(x_np, start, stop)
    result = ffdas.truncate_rank(cp.array(x_np), start=start, stop=stop)
    assert_allclose(cp.asnumpy(result), expected, rtol=1e-2, atol=1e-3)


def test_truncate_rank_more_rows_than_cols(rng):
    x_np = rng.standard_normal((64, 8)).astype("float32")
    expected = _svd_filter(x_np, 0, 8)
    result = ffdas.truncate_rank(cp.array(x_np), start=0, stop=8)
    assert_allclose(cp.asnumpy(result), expected, rtol=1e-3, atol=1e-4)


def test_truncate_rank_preallocated(rng):
    x = cp.array(rng.standard_normal((4, 16)).astype("float32"))
    out = cp.zeros_like(x)
    result = ffdas.truncate_rank(x, start=0, out=out)
    assert result is out
