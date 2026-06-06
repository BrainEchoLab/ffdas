# greens

Frequency-domain Green's function summation over source positions.

For each frequency bin and target point, `greens` sums contributions from all sources weighted by the free-space Green's function:

$$G(\mathbf{r}_s, \mathbf{r}_t, k) = \frac{e^{ik|\mathbf{r}_t - \mathbf{r}_s|}}{|\mathbf{r}_t - \mathbf{r}_s|}$$

where $\mathbf{r}_s$ and $\mathbf{r}_t$ are the source and target positions and $k$ is the wavenumber at that frequency.

The output at each target is the sum over all sources of $x_s \cdot G$, computed independently per frequency bin and per batch element.

!!! note
    `greens` uses tensor core (WMMA) instructions and requires compute capability 7.0 (Volta) or higher. Calling it on an older GPU produces a runtime error.

## Signature

=== "Python"

    ```python
    ffdas.greens(
        srcpos, 
        wavenums, 
        x, 
        dstpos, 
        *,
        out=None, 
    )
    ```

=== "MATLAB"

    ```matlab
    out = ffdas.greens(srcpos, wavenums, x, dstpos)
    ```

## Parameters

| Parameter | Python | MATLAB | Description |
|---|---|---|---|
| `srcpos` | `(sources, 3)` | `(3, sources)` | Source positions. Any unit is fine as long as `wavenums` uses the reciprocal unit. |
| `wavenums` | `(frequencies,)` | `(frequencies, 1)` | Wavenumber at each frequency bin. For outgoing waves, use $k = -2\pi f / c$. |
| `x` | `([batch,] sources, frequencies)` | `(frequencies, sources[, batch])` | Complex-valued input field in the frequency domain. |
| `dstpos` | `(..., 3)` | `(3, ...)` | Target positions. Same units as `srcpos`. |

## Returns

Propagated field at the target positions: `([batch,] ..., frequencies)` in Python, `(frequencies, ...[, batch])` in MATLAB, where `...` corresponds to the spatial dimensions of `dstpos`.

## Example

=== "Python"

    ```python
    freqs = cp.fft.fftfreq(n_samples, d=1.0 / sampling_freq) + center_freq
    wavenums = (-2 * math.pi * freqs / sound_speed).astype(cp.float32)

    pulse = cp.exp(-0.5 * ((freqs - center_freq) / sigma_f) ** 2).astype(cp.complex64)

    # propagate from sources to receivers
    received = ffdas.greens(
        source_pos,                         # (sources, 3)
        wavenums,                           # (frequencies,)
        pulse[None, None, :] * amplitudes,  # (batch, sources, frequencies)
        receiver_pos,                       # (receivers, 3)
    )
    # received: (batch, receivers, frequencies)

    # convert to time domain
    rf = cp.fft.ifft(received, axis=-1)
    ```

=== "MATLAB"

    ```matlab
    freqs = fftfreq(n_samples, 1.0 / sampling_freq) + center_freq;
    wavenums = single(-2 * pi * freqs / sound_speed);

    pulse = exp(-0.5 * ((freqs - center_freq) / sigma_f).^2);

    received = ffdas.greens( ...
        source_pos, ...           % (3, sources)
        wavenums, ...             % (frequencies, 1)
        pulse .* amplitudes, ...  % (frequencies, sources, batch)
        receiver_pos ...          % (3, receivers)
    );
    % received: (frequencies, receivers, batch)
    ```

## Sign Convention

The kernel computes $e^{ik r}/r$. To get the outgoing-wave Green's function $e^{-i 2\pi f r / c}/r$, pass negative wavenumbers: `wavenums = -2 * pi * freqs / sound_speed`. This convention matches the examples in `simulation.py` and `reconstruct.py`.
