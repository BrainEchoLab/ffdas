# das

Delay-and-sum reconstruction, compounding over the sequence (transmit) dimension.

For each target, `das` computes a weighted sum of interpolated samples from all channels and sequence events. The receive delay from each channel to each target is computed internally from the channel and target positions. The `offsets` array provides the per-target transmit delay (one-way, in samples), and `weights` provides per-target apodization.

## Signature

=== "Python"

    ```python
    ffdas.das(
        x, 
        srcpos, 
        dstpos, 
        offsets, 
        weights, 
        *,
        srcdir=None, 
        wavenum=0.0, 
        algorithm=Algorithm.DEFAULT, 
        use_fp16=False, 
        channels_trailing=False, 
        out=None, 
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
| `offsets` | `(seq, ...)` | `(..., seq)` | Per-target transmit time offsets in samples, one row per sequence event of `x`. Spatial dimensions must match `dstpos`. In MATLAB a single event on a flat list of targets is `(targets, 1)`. |
| `weights` | `(seq, ...)` | `(..., seq)` | Per-target apodization weights. Same shape as `offsets`. An event that weighs exactly 0 for a target is skipped for it: it costs nothing and contributes nothing, even where `x` is NaN. Not yet in `ALG4` when a warp mixes zero and nonzero weights: such a target can still take up a NaN or Inf that another target of its warp reads, or a NaN in its own offset. |
| `srcdir` | `(channels, 4)`, `(seq, channels, 4)` or `None` | `(4, channels)`, `(4, channels, seq)` or `[]` | Receive directivity: per channel a unit look direction (first three components) and the cosine of the sensitivity half-angle (fourth). Targets outside a channel's cone receive zero contribution from that channel. With a `seq` dimension every transmit event has its own look directions. See [Receive directivity](#receive-directivity). |
| `wavenum` | `float` | `single` | Wavenumber for phase rotation, typically `-2*pi*fc/fs` for IQ data. Set to `0` to disable. |
| `algorithm` | `Algorithm` | `int32` | Algorithm variant. `DEFAULT` (0) runs `ALG1`; see [Performance](../performance.md). |
| `use_fp16` | `bool` | `logical` | Use half-precision arithmetic. |
| `channels_trailing` | `bool` | `logical` (default `true`) | If `true`, channel data layout is `(samples, seq, channels[, batch])`. If `false`, `(samples, channels, seq[, batch])`. |

## Returns

Reconstructed output with the spatial dimensions of `dstpos`: `([batch,] ...)` in Python, `(...[, batch])` in MATLAB. With zero targets (or an empty batch) the result is empty. A preallocated `out` must be contiguous.

## Example

=== "Python"

    ```python
    k = sampling_freq / sound_speed

    output = ffdas.das(
        rf,               # (batch, 1024, 1, 512) — 1024 channels, 1 transmit, 512 samples
        channel_pos * k,  # (1024, 3)
        voxel_pos * k,    # (64, 64, 64, 3)
        offsets,          # (1, 64, 64, 64) — one transmit event
        weights,          # (1, 64, 64, 64)
        wavenum=-2 * math.pi * center_freq / sampling_freq,  # wavenumber in sampling wavelengths
    )
    # output: (batch, 64, 64, 64)
    ```

=== "MATLAB"

    ```matlab
    k = sampling_freq / sound_speed;

    output = ffdas.das( ...
        rf, ...               % (512, 1, 1024, batch) — 512 samples, 1 transmit, 1024 channels
        channel_pos * k, ...  % (3, 1024)
        voxel_pos * k, ...    % (3, 64, 64, 64)
        offsets, ...          % (64, 64, 64, 1) — one transmit event
        weights, ...          % (64, 64, 64, 1)
        [], ...               % srcdir: none
        -2 * pi * center_freq / sampling_freq ...  % wavenumber in sampling wavelengths
    );
    % output: (64, 64, 64, batch)
    ```


## Receive directivity

`srcdir` limits which targets each channel contributes to. With $\hat{d}$ the unit vector from a channel to a target and $(\hat{\ell}, w)$ the channel's row of `srcdir`, the channel contributes only if

$$
\hat{d} \cdot \hat{\ell} > w ,
$$

a cone of half-angle $\arccos w$ around the look direction $\hat{\ell}$. $w < -1$ disables the cut. A rejected sample contributes an exact 0, whatever `x` holds there. Not yet in `ALG4` when other targets of its warp read a NaN or Inf: the rejected channel's zero coefficients still multiply their samples (0 × NaN = NaN).

**Per transmit event.** A `(seq, channels, 4)` `srcdir` gives each channel a look direction and cut per sequence event, for receive channels whose directivity changes with the transmit. The case it was made for: channels that are sub-arrays pre-steered at the receive focus of the current transmit (micro-beamforming ASICs). The group sums its elements with delays that point it at the focus, so it hears well only around that direction, and the direction changes with every transmit; its look direction is then the unit vector from the group to that transmit's receive focus. In `das_sparse`, the row is the event a sparse entry selects, not the entry.

=== "Python"

    ```python
    # one receive focus per transmit; every group looks at it, with a 20 degree cut
    look = rx_focus[:, None, :] - group_pos[None, :, :]                       # (transmits, groups, 3)
    look /= np.linalg.norm(look, axis=-1, keepdims=True)
    cut = np.full(look.shape[:2] + (1,), np.cos(np.deg2rad(20)))
    srcdir = cp.asarray(np.concatenate([look, cut], axis=-1), dtype=cp.float32)  # (transmits, groups, 4)

    image = ffdas.das_sparse(rf, group_pos * k, voxel_pos * k, offsets, weights, sparse_indices,
                             srcdir=srcdir, wavenum=wavenum)
    ```

=== "MATLAB"

    ```matlab
    % one receive focus per transmit; every group looks at it, with a 20 degree cut
    look = reshape(rx_focus, 3, 1, []) - group_pos;                           % (3, groups, transmits)
    look = look ./ vecnorm(look, 2, 1);
    srcdir = cat(1, look, cos(20 * pi / 180) * ones(1, size(look, 2), size(look, 3)));  % (4, groups, transmits)

    image = ffdas.das_sparse(rf, group_pos * k, voxel_pos * k, offsets, weights, sparse_indices, ...
        gpuArray(single(srcdir)), wavenum);
    ```

This is the same image as one call per transmit, each over the targets that transmit serves with that transmit's `srcdir`, at the cost of one row lookup per event instead of one call per transmit.

---

## das_sparse

Like `das`, but each target compounds over a per-target subset of $n$ sequence events selected by `sparse_indices`, rather than all sequence events. This is useful for synthetic aperture setups where each target only uses a subset of transmissions.

### Additional Parameters

| Parameter | Python | MATLAB | Description |
|---|---|---|---|
| `sparse_indices` | `(n, ...)`, `int32`, 0-based | `(..., n)`, `int32`, 1-based | Indices into the sequence dimension of `x`. Each target compounds the $n$ events given by these indices. An entry that weighs 0, or whose index is outside `x`, is skipped. |

The `offsets` and `weights` arrays have shape `(n, ...)` in Python and `(..., n)` in MATLAB, matching `sparse_indices` rather than the full sequence dimension. A per-event `srcdir` keeps the full sequence dimension: it is indexed by the event an entry selects.

The work follows each target's nonzero entries, so $n$ can be the largest number of events any target needs, with the rest padded with weight 0. Targets are processed 32 at a time (a warp), at the pace of the target with the most nonzero entries, and fastest when those 32 read the same event at the same time. `ffdas.utils.sort_sparse` arranges both, once per geometry: each target's nonzero entries first, in ascending event order, and the targets grouped by those event lists. For 16 diverging transmits this measured 1.7-2.5x faster than grid order, and faster than one call per transmit. The output comes back in the sorted order; put it back with one indexed assignment.

=== "Python"

    ```python
    order, idx, ofs, w = ffdas.utils.sort_sparse(sparse_indices, offsets, weights)   # (n, targets)
    image = cp.empty((batch, n_targets), cp.complex64)
    image[:, order] = ffdas.das_sparse(rf, srcpos, dstpos[order], ofs, w, idx, srcdir=srcdir, wavenum=wavenum)
    ```

=== "MATLAB"

    ```matlab
    [order, idx, ofs, w] = ffdas.utils.sort_sparse(sparse_indices, offsets, weights);  % (targets, n)
    image = complex(gpuArray.zeros(n_targets, batch, "single"));
    image(order, :) = ffdas.das_sparse(rf, srcpos, dstpos(:, order), ofs, w, idx, srcdir, wavenum);
    ```

### Signature

=== "Python"

    ```python
    ffdas.das_sparse(
        x, 
        srcpos, 
        dstpos,
        offsets, 
        weights, 
        sparse_indices,
        *,
        srcdir=None, 
        wavenum=0.0,
        algorithm=Algorithm.DEFAULT,
        use_fp16=False,
        channels_trailing=False,
        out=None, 
    )
    ```

=== "MATLAB"

    ```matlab
    out = ffdas.das_sparse(x, srcpos, dstpos, offsets, weights, sparse_indices)
    out = ffdas.das_sparse(x, srcpos, dstpos, offsets, weights, sparse_indices, srcdir, wavenum, algorithm, use_fp16, channels_trailing)
    ```
