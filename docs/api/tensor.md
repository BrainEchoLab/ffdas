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
    y = ffdas.gather(..., 'interleaved_complex', true)
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
| `'interleaved_complex'` | Name-value option (default `false`). Reads a real `x` as interleaved (real, imag) pairs along its first dimension: an `int16` or `single` array of size `[2*N, ...]` is treated as a complex array of size `[N, ...]`, without a copy. `axis` and `permutation` refer to that complex array. |

### Returns

Array with the same shape as `x` except along the gather dimension, where the size equals `length(indices)`.

### Raw interleaved IQ data

Acquisition systems often deliver IQ data as `int16` with real and imaginary parts interleaved along the sample dimension and the channels in hardware order. One `gather` maps the channels to element order, permutes the dimensions and converts to complex single in a single pass:

=== "Python"

    ```python
    # raw: (..., 2 * samples) int16; the last axis holds (real, imag) pairs
    x = ffdas.view(raw.reshape(*raw.shape[:-1], -1, 2), ffdas.short2)  # complex int16, no copy
    shape = list(x.shape)
    shape[axis] = len(channel_map)
    data = ffdas.gather(x, channel_map, axis, out=cp.empty(shape, dtype="complex64"))
    ```

=== "MATLAB"

    ```matlab
    % raw: [2 * n_samples, n_beams, n_repeats, n_channels] int16
    data = ffdas.gather(raw, channel_map, 4, [1 2 4 3], 'single', ...
                        'interleaved_complex', true);
    % data: [n_samples, n_beams, n_channels, n_repeats] complex single
    ```

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
