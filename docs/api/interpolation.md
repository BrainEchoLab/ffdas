# interpolate

Interpolation from a structured 3D grid to arbitrary query positions. The grid has a regular index-space lattice `(nz, ny, nx)` but its vertex positions can be arbitrary 3D coordinates, enabling interpolation from curvilinear grids (spherical, cylindrical, etc.) to Cartesian coordinates or any other target geometry.

The grid vertex positions must define a non-self-intersecting mapping from index space to physical space. Self-intersecting grids (where different index-space cells map to overlapping physical regions) will produce unpredictable results.

The dimensions of the grid can be in arbitrary order (e.g., `(nz, ny, nx)` or `(ny, nx, nz)`) as long as the input and query points are consistent.

## interpolate

One-shot interpolation. Creates an internal plan, evaluates the query points, and discards the plan. Use `Interpolator` instead when interpolating multiple value arrays or query point sets on the same grid.

### Signature

=== "Python"

    ```python
    ffdas.interpolate(
        gridpos, 
        x, 
        querypos, 
        *, 
        mode="linear", 
        fill=None, 
        out=None, 
    )
    ```

=== "MATLAB"

    ```matlab
    result = ffdas.interpolate(gridpos, x, querypos)
    result = ffdas.interpolate(gridpos, x, querypos, mode, fill_value)
    ```

### Parameters

| Parameter | Python | MATLAB | Description |
|---|---|---|---|
| `gridpos` | `(nz, ny, nx, 3)` | `(3, nx, ny, nz)` | Grid vertex positions. |
| `x` | `([batch,] nz, ny, nx)` | `(nx, ny, nz[, batch])` | Values defined on the grid. |
| `querypos` | `(..., 3)` | `(3, ...)` | Points at which to evaluate. |
| `mode` | `str` | `char` | `"nearest"` or `"linear"` (default). |
| `fill` | `float` or `None` | `numeric` | Value assigned to query points outside the grid. Default `0`. |

### Returns

Interpolated values at the query points, with shape matching the spatial dimensions of `querypos` and an optional batch dimension: `([batch,] ...)` in Python, `(...[, batch])` in MATLAB.

---

## Interpolator

Python only. Creates a reusable interpolation plan for a fixed grid. Avoids rebuilding internal lookup structures when interpolating multiple value arrays or query point sets.

### Constructor

```python
interp = ffdas.Interpolator(gridpos, mode="linear")
```

| Parameter | Shape | Description |
|---|---|---|
| `gridpos` | `(nz, ny, nx, 3)` | Grid vertex positions (`float32`). |
| `mode` | `str` | `"nearest"` or `"linear"`. |

### Calling

```python
result = interp(x, querypos, *, fill=None, out=None, preprocess=False)
```

| Parameter | Shape | Description |
|---|---|---|
| `x` | `([batch,] nz, ny, nx)` | Values on the grid. |
| `querypos` | `(..., 3)` | Evaluation points. |
| `fill` | `float` or `None` | Fill value for out-of-grid points. Default `0`. |
| `preprocess` | `bool` | If `True`, cache lookup structures for these query points so that subsequent calls with different values skip this step. |

### Example

=== "Python"

    ```python
    # build the interpolator once
    interp = ffdas.Interpolator(spherical_grid)  # (nz, ny, nx, 3)

    # reuse for multiple frames
    for frame in frames:
        cart = interp(frame, cart_points, fill=0.0)
    ```
