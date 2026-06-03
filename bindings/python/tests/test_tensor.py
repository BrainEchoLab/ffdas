import warnings

import cupy as cp
import numpy as np

from numpy.testing import assert_allclose

import ffdas


# astype

def test_astype_noop():
    x = np.ones(4, dtype="float32")
    assert ffdas.astype(x, "float32") is x


def test_astype_cast_warns():
    x = np.ones(4, dtype="float64")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        y = ffdas.astype(x, "float32")
    assert y.dtype == np.float32
    assert any(issubclass(wi.category, ffdas.DowncastWarning) for wi in w)


def test_astype_none():
    assert ffdas.astype(None, "float32") is None


# allocation helpers

def test_zeros_like_shape():
    x = cp.zeros((3, 4), dtype="float32")
    assert ffdas.zeros_like(x, shape=(5, 6)).shape == (5, 6)


def test_zeros_like_values():
    x = cp.ones(4, dtype="float32")
    assert_allclose(cp.asnumpy(ffdas.zeros_like(x)), 0.0)


def test_full_like_values():
    x = cp.zeros(3, dtype="float32")
    assert_allclose(cp.asnumpy(ffdas.full_like(x, 7.0)), 7.0)


def test_empty_like_shape():
    x = cp.zeros((2, 3), dtype="float32")
    assert ffdas.empty_like(x, shape=(8,)).shape == (8,)


# contiguous_copy

def test_contiguous_copy_strided():
    x = cp.arange(12, dtype="float32").reshape(3, 4)
    strided = x[:, ::2]
    y = ffdas.contiguous_copy(strided)
    assert_allclose(cp.asnumpy(y), cp.asnumpy(strided))


# gather

def test_gather_axis0():
    x = cp.arange(12, dtype="float32").reshape(4, 3)
    indices = cp.array([0, 2], dtype="int32")
    y = ffdas.gather(x, indices, axis=0)
    assert_allclose(cp.asnumpy(y), cp.asnumpy(x)[[0, 2], :])


# def test_gather_axis1():
#     x = cp.arange(12, dtype="float32").reshape(4, 3)
#     indices = cp.array([1], dtype="int32")
#     y = ffdas.gather(x, indices, axis=1)
#     assert_allclose(cp.asnumpy(y), cp.asnumpy(x)[:, [1]])


def test_gather_preallocated():
    x = cp.arange(12, dtype="float32").reshape(4, 3)
    indices = cp.array([0, 3], dtype="int32")
    out = cp.zeros((2, 3), dtype="float32")
    result = ffdas.gather(x, indices, axis=0, out=out)
    assert result is out
    assert_allclose(cp.asnumpy(out), cp.asnumpy(x)[[0, 3], :])


# scatter

def test_scatter_axis0():
    x = cp.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype="float32")
    indices = cp.array([1, 3], dtype="int32")
    out = cp.zeros((4, 3), dtype="float32")
    ffdas.scatter(x, indices, axis=0, out=out)
    result = cp.asnumpy(out)
    assert_allclose(result[1], [10.0, 20.0, 30.0])
    assert_allclose(result[3], [40.0, 50.0, 60.0])


def test_gather_scatter_roundtrip():
    x = cp.arange(6, dtype="float32").reshape(2, 3)
    indices = cp.array([0, 2], dtype="int32")
    scattered = cp.zeros((4, 3), dtype="float32")
    ffdas.scatter(x, indices, axis=0, out=scattered)
    recovered = ffdas.gather(scattered, indices, axis=0)
    assert_allclose(cp.asnumpy(recovered), cp.asnumpy(x))
