# Utilities

Spatial helper functions and GPU timing. The spatial functions use `array_api_compat` in Python and native MATLAB operations, so they work with any supported array backend (CuPy, NumPy, PyTorch).

## cdist {#cdist}

Pairwise Euclidean distance between two sets of points.

=== "Python"

    ```python
    ffdas.cdist(a, b)
    # a: (...A, D), b: (...B, D) -> (*a.shape[:-1], *b.shape[:-1])
    ```

=== "MATLAB"

    ```matlab
    d = ffdas.utils.cdist(a, b)
    % a: (D, ...A), b: (D, ...B) -> (...A, ...B)
    ```

The coordinate dimension is last in Python and first in MATLAB, matching the library-wide convention. Broadcasting applies to all other dimensions.

### Example

=== "Python"

    ```python
    sources = cp.array([[0.0, 0.0, -0.01]], dtype=cp.float32)        # (1, 3)
    targets = cp.stack([xx, yy, zz], axis=-1)                         # (nz, ny, nx, 3)
    dist = ffdas.cdist(sources, targets)                               # (1, nz, ny, nx)
    ```

=== "MATLAB"

    ```matlab
    sources = gpuArray(single([0; 0; -0.01]));                        % (3, 1)
    targets = permute(cat(4, xx, yy, zz), [4 3 1 2]);                % (3, nz, ny, nx)
    dist = ffdas.utils.cdist(sources, targets);                       % (1, nz, ny, nx)
    ```

---

## rect_dist {#rect_dist}

Minimum Euclidean distance from 3D points to an axis-aligned rectangle centered at the origin in the $z = 0$ plane. Used to compute the reference distance for diverging-wave transmit delay calculations (the point on the aperture closest to the virtual source).

=== "Python"

    ```python
    ffdas.rect_dist(points, size)
    # points: (..., 3), size: (2,) -> (...)
    ```

=== "MATLAB"

    ```matlab
    d = ffdas.utils.rect_dist(points, size)
    % points: (3, ...), size: [width; height] -> (...)
    ```

`size` is the full width and height of the rectangle (not half-extents).

---

## angle {#angle}

Angle in radians between vectors `a` and `b`.

=== "Python"

    ```python
    ffdas.angle(a, b, eps=1e-7)
    # a: (..., D), b: (..., D) -> (...)
    ```

=== "MATLAB"

    ```matlab
    theta = ffdas.utils.angle(a, b)
    theta = ffdas.utils.angle(a, b, eps)
    % a: (D, ...), b: (D, ...) -> (...)
    ```

Broadcasting applies to all dimensions except the coordinate dimension. The `eps` parameter prevents division by zero for near-zero-length vectors.

### Example

Computing the off-axis angle for apodization:

=== "Python"

    ```python
    direction = cp.array([0.0, 0.0, 1.0], dtype=cp.float32)
    theta = ffdas.angle(targets - source, direction)  # angle from forward axis
    ```

=== "MATLAB"

    ```matlab
    direction = gpuArray(single([0; 0; 1]));
    theta = ffdas.utils.angle(targets - source, direction);
    ```

---

## Timer {#timer}

Python only. Context manager for timing GPU operations using CUDA events. Records events on the library's internal CUDA stream before and after the wrapped block, then synchronizes and reports elapsed GPU time.

```python
with ffdas.utils.Timer() as t:
    output = ffdas.das(rf, channel_pos, voxel_pos, offsets, weights)
print(f"{t.elapsed_ms():.1f} ms")
```

The timer can also be used without a context manager via `start()` and `stop()`.

---

## MATLAB-only Utilities

### take

Take the first $k$ entries along a dimension.

```matlab
y = ffdas.utils.take(x, k, axis)
```

### take_along_axis

Gather elements from `x` using per-element indices along a dimension, equivalent to NumPy's `take_along_axis`.

```matlab
y = ffdas.utils.take_along_axis(x, indices, axis)
```

`indices` are 1-based and must have the same shape as `x` except along the gather axis.
