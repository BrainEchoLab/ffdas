# Conventions

## Notation

Throughout this documentation, `(...)` in a shape description means one or more dimensions of arbitrary size. For example, `(..., 3)` means an array of any shape whose last dimension is 3 — this could be `(3,)`, `(64, 3)`, `(64, 64, 64, 3)`, and so on.

## Units

`das` operates in **sampling wavelengths**, $\lambda_s = c / f_s$, where $c$ is the propagation speed and $f_s$ is the sampling frequency. All spatial coordinates passed to `das` should be in units of $\lambda_s$, and all time offsets should be in samples.

In practice this means multiplying physical positions by $f_s / c$:

=== "Python"

    ```python
    k = sampling_freq / sound_speed

    channel_pos_scaled = channel_pos * k   # (channels, 3) in sampling wavelengths
    voxel_pos_scaled = voxel_pos * k       # (..., 3) in sampling wavelengths
    delay_samples = delay_seconds * sampling_freq  # offsets in samples
    ```

=== "MATLAB"

    ```matlab
    k = sampling_freq / sound_speed;

    channel_pos_scaled = channel_pos * k;   % (3, channels) in sampling wavelengths
    voxel_pos_scaled = voxel_pos * k;       % (3, ...) in sampling wavelengths
    delay_samples = delay_seconds * sampling_freq;  % offsets in samples
    ```

This convention exists for two reasons. First, the `das` kernel internally computes the distance from each channel to each target to determine the receive sample index. Working in sampling wavelengths means these distances are directly in samples, avoiding a per-element division by $c / f_s$ on the GPU. Second, it makes the kernel independent of the acquisition parameters — the same kernel handles any combination of sound speed and sampling rate without reconfiguration.

The `wavenum` parameter for IQ phase rotation follows the same convention: $k = -2\pi f_c / f_s$. Set it to 0 for real-valued RF data.

`greens` and `interpolate` are unit-agnostic. For `greens`, positions can be in any unit as long as the wavenumbers are in the reciprocal unit. For `interpolate`, grid and query positions just need to share the same unit. The spatial utilities (`cdist`, `rect_dist`, `angle`) similarly operate in whatever units their inputs use.

## Array Layout

The C core uses row-major (C-contiguous) memory layout. The Python bindings expose this directly. The MATLAB bindings reverse the dimension order so that arrays are column-major, matching MATLAB's native layout. This means the same data is accessed efficiently in both languages without transposition.

The general pattern: where Python has dimensions `(a, b, c)`, MATLAB has `(c, b, a)`.

For position arrays specifically: Python places the coordinate dimension last, `(..., 3)`. MATLAB places it first, `(3, ...)`.

| | Python | MATLAB |
|---|---|---|
| Channel positions | `(channels, 3)` | `(3, channels)` |
| Target positions | `(..., 3)` | `(3, ...)` |
| Channel data | `([batch,] channels, seq, samples)` | `(samples, seq, channels[, batch])` |

See the individual [API reference](api/index.md) pages for the exact shapes of each function's inputs and outputs.

## Batch Dimension

Most operations accept an optional batch dimension that lets you process multiple independent inputs sharing the same geometry in a single kernel launch. For example, `das` can reconstruct many frames at once if they share the same channel positions, target grid, and transmit delays — only the channel data differs across the batch. In Python the batch dimension is leading; in MATLAB it is trailing.

In shape descriptions, square brackets denote optional dimensions: `([batch,] channels, seq, samples)` means the array can be either 3D (no batch) or 4D (with batch).

## Coordinate System

ffdas does not enforce a specific coordinate system. The convention used in the examples and quickstart is right-handed with z along the imaging axis (away from the transducer), but any consistent system works. Position arrays are always ordered (x, y, z) along the coordinate dimension.
