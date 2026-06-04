# interpolate

Interpolation from a structured 3D grid to arbitrary query positions. The grid has a regular index-space lattice `(nz, ny, nx)` but its vertex positions can be arbitrary 3D coordinates, enabling interpolation from curvilinear grids (spherical, cylindrical, etc.) to Cartesian coordinates or any other target geometry.

## interpolate

One-shot interpolation. Creates an internal plan, evaluates the query points, and discards the plan. Use `Interpolator` instead when interpolating multiple value arrays or query point sets on the same grid.

### Signature

=== "Python"

    ```python
    ffdas.interpolate(
        grid_points,       # grid vertex positions
        values,            # values on the grid
        query_points,      # evaluation points
        *,
        mode="linear",     # "nearest" or "linear"
        fill=None,         # fill value for out-of-grid points (default 0)
        out=None,
    )
    ```

=== "MATLAB"

    ```matlab
    result = ffdas.interpolate(grid_points, values, query_points)
    result = ffdas.interpolate(grid_points, values, query_points, mode, fill_value)
    ```

### Parameters

| Parameter | Python | MATLAB | Description |
|---|---|---|---|
| `grid_points` | `(nz, ny, nx, 3)` | `(3, nx, ny, nz)` | Grid vertex positions. |
| `values` | `([batch], nz, ny, nx)` | `(nx, ny, nz[, batch])` | Values defined on the grid. |
| `query_points` | `(..., 3)` | `(3, ...)` | Points at which to evaluate. |
| `mode` | `str` | `char` | `"nearest"` or `"linear"` (default). |
| `fill` | `float` or `None` | `numeric` | Value assigned to query points outside the grid. Default `0`. |

### Returns

Interpolated values at the query points, with shape matching the spatial dimensions of `query_points` and an optional batch dimension.

---

## Interpolator {#interpolator}

Python only. Creates a reusable interpolation plan for a fixed grid. Avoids rebuilding internal lookup structures when interpolating multiple value arrays or query point sets.

### Constructor

```python
interp = ffdas.Interpolator(grid_points, mode="linear")
```

| Parameter | Shape | Description |
|---|---|---|
| `grid_points` | `(nz, ny, nx, 3)` | Grid vertex positions (`float32`). |
| `mode` | `str` | `"nearest"` or `"linear"`. |

### Calling

```python
result = interp(values, query_points, *, fill=None, out=None, preprocess=False)
```

| Parameter | Shape | Description |
|---|---|---|
| `values` | `([batch], nz, ny, nx)` | Values on the grid. |
| `query_points` | `(..., 3)` | Evaluation points. |
| `fill` | `float` or `None` | Fill value for out-of-grid points. Default `0`. |
| `preprocess` | `bool` | If `True`, cache lookup structures for these query points so that subsequent calls with different values skip this step. |

### Example

=== "Python"

    ```python
    # build the interpolator once
    interp = ffdas.Interpolator(spherical_grid)  # (nz, ny, nx, 3)

    # reuse for multiple frames
    for frame in frames:
        cart = interp(frame, cart_points, fill=0.0)  # (cart_nz, cart_ny, cart_nx)
    ```

=== "MATLAB"

    ```matlab
    % one-shot interpolation
    cart = ffdas.interpolate(spherical_grid, frame, cart_points, 'linear', 0.0);
    ```
