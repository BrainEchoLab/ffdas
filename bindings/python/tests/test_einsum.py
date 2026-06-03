"""Tests for einsum: subscript parsing, validation, and GPU correctness."""

import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose

import ffdas
from ffdas.einsum import implicit_output_modes


# implicit_output_modes (python function)

@pytest.mark.parametrize("modes, expected", [
    (["ij", "jk"], "ik"),
    (["i", "j"], "ij"),
    (["ii", ""], ""),
    (["ij", "ij"], ""),
    (["bij", "bjk"], "ik"),
    (["ki", "ij"], "kj"),
])
def test_implicit_output_modes(modes, expected):
    assert implicit_output_modes(modes) == expected


# validation

def test_einsum_missing_comma():
    with pytest.raises(ValueError):
        ffdas.einsum("ij", cp.ones((3, 3)), cp.ones((3, 3)))


def test_einsum_multiple_arrows():
    with pytest.raises(ValueError):
        ffdas.einsum("ij,jk->ik->bad", cp.ones((3, 3)), cp.ones((3, 3)))


def test_einsum_dimension_mismatch():
    with pytest.raises(ValueError):
        ffdas.einsum("ij,jk->ik", cp.ones((3, 4)), cp.ones((5, 6)))


def test_einsum_bad_output_mode():
    with pytest.raises(ValueError):
        ffdas.einsum("ij,jk->iz", cp.ones((3, 4)), cp.ones((4, 5)))


# Compare einsum against cpu implementation

EINSUM_CASES = [
    ("ij,jk->ik", (4, 8), (8, 6)),
    ("bij,bjk->bik", (2, 4, 8), (2, 8, 6)),
    ("i,i->", (16,), (16,)),
    ("i,j->ij", (4,), (5,)),
    ("ij,jk", (4, 8), (8, 6)),
    ("ij,kj->ik", (4, 8), (6, 8)),
]


@pytest.mark.parametrize("subscripts, a_shape, b_shape", EINSUM_CASES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_einsum_correctness(subscripts, a_shape, b_shape, dtype, rng):
    a_np = rng.standard_normal(a_shape).astype(dtype)
    b_np = rng.standard_normal(b_shape).astype(dtype)
    expected = np.einsum(subscripts, a_np, b_np)
    result = ffdas.einsum(subscripts, cp.array(a_np), cp.array(b_np))
    assert_allclose(cp.asnumpy(result), expected, rtol=1e-4, atol=1e-5)
