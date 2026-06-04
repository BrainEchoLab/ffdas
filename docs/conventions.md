# Conventions

This page documents the unit system, array layout, and data format conventions used by ffdas.

## Units

`das` operates in **sampling wavelengths**:

$\lambda_s = \frac{c}{f_s}$

where $c$ is the speed of sound (or propagation speed) and $f_s$ is the sampling frequency. All spatial coordinates and channel positions passed to `das` should be in units of $\lambda_s$, and all time offsets should be in samples.

In practice this means multiplying physical positions (in meters) by $f_s / c$ and multiplying physical times (in seconds) by $f_s$:

=== "Python"

    ```python
    k = sampling_freq / sound_speed  # conversion factor [1/m]

    channel_pos_scaled = channel_pos * k   # (channels, 3) in sampling wavelengths
    voxel_pos_scaled = voxel_pos * k       # (..., 3) in sampling wavelengths
    delay_samples = delay_seconds * sampling_freq  # offsets in samples
    ```

=== "MATLAB"

    ```matlab
    k = sampling_freq / sound_speed;  % conversion factor [1/m]

    channel_pos_scaled = channel_pos * k;   % (3, channels) in sampling wavelengths
    voxel_pos_scaled = voxel_pos * k;       % (3, ...) in sampling wavelengths
    delay_samples = delay_seconds * sampling_freq;  % offsets in samples
    ```

This makes the kernel unitless: it doesn't need to know the speed of sound or sampling frequency. Different operations can use different values of $c$ or $f_s$ without any library-level configuration.

The `wavenum` parameter used by `das` for IQ phase rotation follows the same convention. For complex baseband data sampled at $f_s$ with center frequency $f_c$:

$$k = -\frac{2\pi f_c}{f_s}$$

Set `wavenum = 0` when working with real-valued RF data or when no phase rotation is needed.

The spatial utility functions (`cdist`, `rect_dist`, `angle`) operate in whatever units their inputs are in. They are typically called with physical coordinates (meters) before the conversion to sampling wavelengths.

`greens` and `interpolate` are unit-agnostic. For `greens`, positions can be in any unit as long as the wavenumbers are in the corresponding reciprocal unit (e.g., positions in meters and wavenumbers in 1/m). For `interpolate`, grid and query positions just need to share the same unit.

## Array Layout

The Python and MATLAB bindings use the natural memory layout of their respective languages. Dimensions are reversed between the two, so the same data is accessed efficiently in both.

### Positions

Position arrays have one coordinate dimension of size 3 (x, y, z).

| Array | Python | MATLAB |
|---|---|---|
| Channel positions | `(channels, 3)` | `(3, channels)` |
| Target positions | `(..., 3)` | `(3, ...)` |
| Directivity vectors | `(channels, 4)` | `(4, channels)` |

### Channel Data

The input to `das` and `das_sparse` is the acquired channel data (RF or IQ), with a batch dimension that is optional in Python and trailing in MATLAB.

| Dimension | Python | MATLAB |
|---|---|---|
| Without batch | `(channels, sequence, samples)` | `(samples, sequence, channels)` |
| With batch | `(batch, channels, sequence, samples)` | `(samples, sequence, channels, batch)` |

The **sequence** dimension indexes transmit events. In MATLAB, the `channels_trailing` parameter (default `true`) controls whether the channel dimension comes after the sequence dimension; setting it to `false` gives `(samples, channels, sequence[, batch])`.

### Offsets and Weights

The `offsets` and `weights` arrays passed to `das` have a sequence dimension (indexing transmit events) and spatial dimensions matching the target grid.

| Array | Python | MATLAB |
|---|---|---|
| Offsets | `(sequence, ...)` | `(..., sequence)` |
| Weights | `(sequence, ...)` | `(..., sequence)` |

For `das_sparse`, the leading/trailing dimension indexes into a per-target subset of size $n$ rather than the full sequence, and `sparse_indices` provides the mapping back to the sequence dimension of the input data.

### Green's Function

The `greens` function operates in the frequency domain. The input and output have a frequency dimension and an optional batch dimension.

| Array | Python | MATLAB |
|---|---|---|
| Input field | `(batch, sources, frequencies)` | `(frequencies, sources[, batch])` |
| Output field | `(batch, targets, frequencies)` | `(frequencies, targets[, batch])` |
| Wavenumbers | `(frequencies,)` | `(frequencies, 1)` |

### Eigenfilter

`eigfilter` treats its first dimension (Python) or last dimension (MATLAB) as the slow-time axis and decomposes across it. The remaining dimensions are flattened into a single spatial dimension internally.

| | Python | MATLAB |
|---|---|---|
| Input | `(frames, ...)` | `(..., frames)` |
| k0/k1 indexing | 0-based | 1-based |

### Interpolation

Grid positions have shape `(nz, ny, nx, 3)` in Python and `(3, nx, ny, nz)` in MATLAB. Values on the grid follow the same spatial ordering without the coordinate dimension. Query positions follow the same convention as target positions: `(..., 3)` in Python, `(3, ...)` in MATLAB.

## Data Types

All position arrays (`xpos`, `ypos`, `grid_points`, `query_points`) are cast to `float32` internally. Offsets and weights are similarly `float32`. Indices (`sparse_indices`, gather/scatter indices) are `int32`.

Channel data and values passed to `eigfilter`, `einsum`, `greens`, and `interpolate` can be any of the supported floating-point or complex types. The main compute path uses `float32` / `complex64`. Double precision (`float64` / `complex128`) is supported. Half precision (`float16`) is available in `das` via the `use_fp16` flag.

## Coordinate System

ffdas does not enforce a specific coordinate system. The convention used in the examples and quickstart is right-handed with z along the imaging axis (away from the transducer), but any consistent coordinate system works. Position arrays are always ordered (x, y, z) along the coordinate dimension.
