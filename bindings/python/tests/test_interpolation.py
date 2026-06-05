import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose

import ffdas


# raises


def test_interpolator_wrong_rank():
    with pytest.raises(ValueError):
        ffdas.Interpolator(cp.zeros((4, 3), dtype="float32"))


def test_interpolator_wrong_trailing_dim():
    with pytest.raises(ValueError):
        ffdas.Interpolator(cp.zeros((2, 3, 4, 2), dtype="float32"))


def test_interpolator_invalid_mode():
    with pytest.raises(ValueError):
        ffdas.Interpolator(cp.zeros((2, 3, 4, 3), dtype="float32"), mode="cubic")


# output


def _regular_grid(nz, ny, nx):
    """Regular [0,1]^3 grid as a cupy array."""
    z = np.linspace(0, 1, nz, dtype="float32")
    y = np.linspace(0, 1, ny, dtype="float32")
    x = np.linspace(0, 1, nx, dtype="float32")
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return cp.array(np.stack([xx, yy, zz], axis=-1))


@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_exact_at_grid_points(mode):
    nz, ny, nx = 3, 4, 5
    grid = _regular_grid(nz, ny, nx)
    x = cp.arange(nz * ny * nx, dtype="float32").reshape(nz, ny, nx)
    interp = ffdas.Interpolator(grid, mode=mode)
    queries = grid.reshape(-1, 3)
    result = interp(x, queries)
    assert_allclose(cp.asnumpy(result), cp.asnumpy(x).flatten(), atol=1e-4)


def test_linear_midpoint():
    nz, ny, nx = 2, 2, 2
    grid = _regular_grid(nz, ny, nx)
    x = cp.zeros((nz, ny, nx), dtype="float32")
    x[0, 0, 0] = 0.0
    x[0, 0, 1] = 1.0
    interp = ffdas.Interpolator(grid, mode="linear")
    result = interp(x, cp.array([[0.5, 0.0, 0.0]], dtype="float32"))
    assert_allclose(cp.asnumpy(result), [0.5], atol=1e-4)


@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_fill_value_outside_grid(mode):
    grid = _regular_grid(2, 2, 2)
    x = cp.ones((2, 2, 2), dtype="float32")
    interp = ffdas.Interpolator(grid, mode=mode)
    result = interp(x, cp.array([[5.0, 5.0, 5.0]], dtype="float32"), fill=-1.0)
    assert_allclose(cp.asnumpy(result), [-1.0], atol=1e-5)


def test_output_shape():
    grid = _regular_grid(2, 3, 4)
    x = cp.ones((2, 3, 4), dtype="float32")
    queries = cp.zeros((10, 3), dtype="float32")
    result = ffdas.Interpolator(grid, mode="linear")(x, queries)
    assert result.shape == (10,)


@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_batch_dim(mode):
    batch_size, nz, ny, nx = 4, 3, 4, 5
    grid = _regular_grid(nz, ny, nx)
    x = cp.arange(batch_size * nz * ny * nx, dtype="float32").reshape(batch_size, nz, ny, nx)
    interp = ffdas.Interpolator(grid, mode=mode)
    queries = grid.reshape(-1, 3)
    result = interp(x, queries)
    assert_allclose(cp.asnumpy(result), cp.asnumpy(x).reshape(batch_size, -1), atol=1e-4)


def test_convenience_function():
    nz, ny, nx = 3, 4, 5
    grid = _regular_grid(nz, ny, nx)
    x = cp.arange(nz * ny * nx, dtype="float32").reshape(nz, ny, nx)
    queries = grid.reshape(-1, 3)
    result = ffdas.interpolate(grid, x, queries, mode="nearest")
    assert_allclose(cp.asnumpy(result), cp.asnumpy(x).flatten(), atol=1e-5)


def test_values_leading_dims_mismatch():
    grid = _regular_grid(2, 3, 4)
    interp = ffdas.Interpolator(grid, mode="linear")
    with pytest.raises(ValueError):
        interp(cp.ones((3, 3, 4), dtype="float32"), cp.zeros((1, 3), dtype="float32"))
