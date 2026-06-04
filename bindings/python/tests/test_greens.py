import cupy as cp
import numpy as np
from numpy.testing import assert_allclose

import ffdas


def _greens_reference(xpos, wavenums, x, ypos):
    """Reference: y[b,t,f] = sum_s x[b,s,f] * exp(i*k[f]*r[s,t]) / r[s,t]."""
    flat_ypos = ypos.reshape(-1, 3)
    diff = xpos[None, :, :] - flat_ypos[:, None, :]
    r = np.maximum(np.sqrt(np.sum(diff ** 2, axis=-1)), 1e-10)
    phase = wavenums[None, None, :] * r[:, :, None]
    kernel = np.exp(1j * phase) / r[:, :, None]
    out = np.einsum("bsf,tsf->btf", x, kernel)
    return out.reshape(x.shape[0], *ypos.shape[:-1], x.shape[2])


def test_greens_against_reference(rng):
    batch_size, nin, nk, nout = 1, 4, 8, 6
    xpos = rng.standard_normal((nin, 3)).astype("float32")
    ypos = rng.standard_normal((nout, 3)).astype("float32")
    wavenums = rng.uniform(1.0, 10.0, (nk,)).astype("float32")
    x = (rng.standard_normal((batch_size, nin, nk))
         + 1j * rng.standard_normal((batch_size, nin, nk))).astype("complex64")

    expected = _greens_reference(xpos, wavenums, x, ypos)
    result = ffdas.greens(
        cp.array(xpos), cp.array(wavenums), cp.array(x), cp.array(ypos),
    )
    assert_allclose(cp.asnumpy(result), expected, rtol=1e-3, atol=1e-4)


def test_greens_output_shape_flat():
    batch_size, nin, nk, nout = 2, 4, 8, 6
    result = ffdas.greens(
        cp.zeros((nin, 3), dtype="float32"),
        cp.ones(nk, dtype="float32"),
        cp.zeros((batch_size, nin, nk), dtype="complex64"),
        cp.zeros((nout, 3), dtype="float32"),
    )
    assert result.shape == (batch_size, nout, nk)


def test_greens_output_shape_2d_targets():
    batch_size, nin, nk = 1, 4, 8
    nz, nx = 3, 5
    result = ffdas.greens(
        cp.zeros((nin, 3), dtype="float32"),
        cp.ones(nk, dtype="float32"),
        cp.zeros((batch_size, nin, nk), dtype="complex64"),
        cp.zeros((nz, nx, 3), dtype="float32"),
    )
    assert result.shape == (batch_size, nz, nx, nk)


def test_greens_zero_input():
    result = ffdas.greens(
        cp.zeros((4, 3), dtype="float32"),
        cp.ones(8, dtype="float32"),
        cp.zeros((1, 4, 8), dtype="complex64"),
        cp.ones((6, 3), dtype="float32"),
    )
    assert_allclose(cp.asnumpy(cp.abs(result)), 0.0, atol=1e-6)
