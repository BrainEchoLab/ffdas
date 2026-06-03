import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose

import ffdas


# raisees

def test_eigfilter_1d_rejected():
    with pytest.raises(ValueError):
        ffdas.eigfilter(cp.ones(10), k0=0)


def test_eigfilter_k0_negative():
    with pytest.raises(ValueError):
        ffdas.eigfilter(cp.ones((4, 8)), k0=-1)


def test_eigfilter_k0_too_large():
    with pytest.raises(ValueError):
        ffdas.eigfilter(cp.ones((4, 8)), k0=4)


def test_eigfilter_k1_before_k0():
    with pytest.raises(ValueError):
        ffdas.eigfilter(cp.ones((4, 8)), k0=2, k1=1)


def test_eigfilter_k1_too_large():
    with pytest.raises(ValueError):
        ffdas.eigfilter(cp.ones((4, 8)), k0=0, k1=5)


# output

def _svd_filter(x, k0, k1):
    """Reference: reconstruct x using singular vectors k0:k1."""
    m = x.shape[0]
    n = np.prod(x.shape[1:])
    U, S, Vh = np.linalg.svd(x.reshape(m, n), full_matrices=False)
    return (U[:, k0:k1] @ np.diag(S[k0:k1]) @ Vh[k0:k1, :]).reshape(x.shape)


def test_eigfilter_keep_all(rng):
    x_np = rng.standard_normal((8, 32)).astype("float32")
    m, n = x_np.shape
    result = ffdas.eigfilter(cp.array(x_np), k0=0, k1=min(m, n))
    assert_allclose(cp.asnumpy(result), x_np, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("k0, k1", [(0, 4), (1, 8), (2, 6)])
def test_eigfilter_truncated(k0, k1, rng):
    x_np = rng.standard_normal((16, 64)).astype("float32")
    expected = _svd_filter(x_np, k0, k1)
    result = ffdas.eigfilter(cp.array(x_np), k0=k0, k1=k1)
    assert_allclose(cp.asnumpy(result), expected, rtol=1e-2, atol=1e-3)


def test_eigfilter_output_shape_3d(rng):
    x = cp.array(rng.standard_normal((8, 4, 16)).astype("float32"))
    assert ffdas.eigfilter(x, k0=0, k1=4).shape == x.shape


def test_eigfilter_preallocated(rng):
    x = cp.array(rng.standard_normal((4, 16)).astype("float32"))
    out = cp.zeros_like(x)
    result = ffdas.eigfilter(x, k0=0, out=out)
    assert result is out
