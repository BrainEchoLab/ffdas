# das

Delay-and-sum reconstruction, compounding over the sequence (transmit) dimension.

For each target, `das` computes a weighted sum of interpolated samples from all channels and sequence events. The receive delay from each channel to each target is computed internally from the channel and target positions. The `offsets` array provides the per-target transmit delay (one-way, in samples), and `weights` provides per-target apodization.

## Signature

=== "Python"

    ```python
    ffdas.das(
        x,                    # channel data
        srcpos,                 # channel positions
        dstpos,                 # target positions
        offsets,              # transmit delays
        weights,              # apodization weights
        *,
        srcdir=None,            # channel directivity
        wavenum=0.0,          # phase rotation wavenumber
        algorithm=Algorithm.DEFAULT,
        out=None,
        use_fp16=False,
    )
    ```

=== "MATLAB"

    ```matlab
    out = ffdas.das(x, srcpos, dstpos, offsets, weights)
    out = ffdas.das(x, srcpos, dstpos, offsets, weights, srcdir, wavenum, algorithm, use_fp16, channels_trailing)
    ```

## Parameters

| Parameter | Python | MATLAB | Description |
|---|---|---|---|
| `x` | `([batch,] channels, seq, samples)` | `(samples, seq, channels[, batch])` | Channel data (RF or IQ). |
| `srcpos` | `(channels, 3)` | `(3, channels)` | Channel positions in sampling wavelengths. |
| `dstpos` | `(..., 3)` | `(3, ...)` | Target positions in sampling wavelengths. |
| `offsets` | `(seq, ...)` | `(..., seq)` | Per-target transmit time offsets in samples. Spatial dimensions must match `dstpos`. |
| `weights` | `(seq, ...)` | `(..., seq)` | Per-target apodization weights. Same shape as `offsets`. |
| `srcdir` | `(channels, 4)` or `None` | `(4, channels)` or `[]` | Directivity vectors. The first three components are the unit surface normal of each channel element; the fourth is the cosine of the sensitivity half-angle. Targets outside a channel's cone receive zero contribution from that channel. |
| `wavenum` | `float` | `single` | Wavenumber for phase rotation, typically `-2*pi*fc/fs` for IQ data. Set to `0` to disable. |
| `algorithm` | `Algorithm` | `int32` | Algorithm variant. `DEFAULT` (0) selects automatically. |
| `use_fp16` | `bool` | `logical` | Use half-precision arithmetic. |
| `channels_trailing` | — | `logical` (default `true`) | If `true`, channel data layout is `(samples, seq, channels[, batch])`. If `false`, `(samples, channels, seq[, batch])`. |

## Returns

Reconstructed output with the spatial dimensions of `dstpos`: `([batch,] ...)` in Python, `(...[, batch])` in MATLAB.

## Example

=== "Python"

    ```python
    k = sampling_freq / sound_speed

    output = ffdas.das(
        rf,                        # (batch, 1024, 1, 512) — 1024 channels, 1 transmit, 512 samples
        channel_pos * k,           # (1024, 3)
        voxel_pos * k,             # (64, 64, 64, 3)
        offsets,                   # (1, 64, 64, 64) — one transmit event
        weights,                   # (1, 64, 64, 64)
        wavenum=-2 * math.pi * center_freq / sampling_freq,
    )
    # output: (batch, 64, 64, 64)
    ```

=== "MATLAB"

    ```matlab
    k = sampling_freq / sound_speed;

    output = ffdas.das( ...
        rf, ...                    % (512, 1, 1024, batch) — 512 samples, 1 transmit, 1024 channels
        channel_pos * k, ...       % (3, 1024)
        voxel_pos * k, ...         % (3, 64, 64, 64)
        offsets, ...               % (64, 64, 64, 1) — one transmit event
        weights ...                % (64, 64, 64, 1)
    );
    % output: (64, 64, 64, batch)
    ```

---

## das_sparse

Like `das`, but each target compounds over a per-target subset of $n$ sequence events selected by `sparse_indices`, rather than all sequence events. This is useful for synthetic aperture setups where each target only uses a subset of transmissions.

### Additional Parameters

| Parameter | Python | MATLAB | Description |
|---|---|---|---|
| `sparse_indices` | `(n, ...)`, `int32`, 0-based | `(..., n)`, `int32`, 1-based | Indices into the sequence dimension of `x`. Each target compounds the $n$ events given by these indices. |

The `offsets` and `weights` arrays have shape `(n, ...)` in Python and `(..., n)` in MATLAB, matching `sparse_indices` rather than the full sequence dimension.

### Signature

=== "Python"

    ```python
    ffdas.das_sparse(
        x, srcpos, dstpos,
        offsets, weights, sparse_indices,
        *,
        srcdir=None, wavenum=0.0,
        algorithm=Algorithm.DEFAULT,
        out=None, use_fp16=False,
    )
    ```

=== "MATLAB"

    ```matlab
    out = ffdas.das_sparse(x, srcpos, dstpos, offsets, weights, sparse_indices)
    out = ffdas.das_sparse(x, srcpos, dstpos, offsets, weights, sparse_indices, srcdir, wavenum, algorithm, use_fp16, channels_trailing)
    ```
