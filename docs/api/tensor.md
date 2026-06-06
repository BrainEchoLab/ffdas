# gather / scatter

GPU tensor indexing operations. These avoid host-device round trips that occur when using Python-side fancy indexing on large arrays.

## gather

Gather elements from `x` along a dimension at the given indices.

=== "Python"

    ```python
    ffdas.gather(x, indices, axis, *, out=None)
    ```

=== "MATLAB"

    ```matlab
    y = ffdas.gather(x, indices, axis)
    y = ffdas.gather(x, indices, axis, permutation)
    y = ffdas.gather(x, indices, axis, permutation, dtype)
    ```

### Parameters

| Parameter | Description |
|---|---|
| `x` | Input GPU array. |
| `indices` | 1D index array (`int32`). 0-based in Python, 1-based in MATLAB. |
| `axis` | Dimension along which to gather. 0-based in Python, 1-based in MATLAB. |

MATLAB additionally supports:

| Parameter | Description |
|---|---|
| `permutation` | Optional permutation vector (1-based, `int32`). Effectively, this corresponds to permuting the dimensions of `x` to order `permutation` *before* gathering, but the library performs both operations at the same time, avoiding the extra copy that MATLAB's `permute` would create. |
| `dtype` | Optional output datatype (`'single'`, `'double'`, `'int16'`, `'int32'`). |

### Returns

Array with the same shape as `x` except along the gather dimension, where the size equals `length(indices)`.

---

## scatter

Python only. Scatter elements of `x` into an output array along a dimension at the given indices. The inverse of gather.

```python
ffdas.scatter(x, indices, axis, *, out=None)
```

### Parameters

| Parameter | Description |
|---|---|
| `x` | Input GPU array. |
| `indices` | 1D index array (`int32`). |
| `axis` | Dimension along which to scatter. |

---

## contiguous_copy

Python only. Copy a strided (non-contiguous) GPU array into a contiguous buffer. Useful after slicing or transposing when a subsequent operation requires contiguous memory.

```python
ffdas.contiguous_copy(x, *, out=None)
```

### Parameters

| Parameter | Description |
|---|---|
| `x` | Input GPU array (may be non-contiguous). |

### Returns

Contiguous copy of `x`.
