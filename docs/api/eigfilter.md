# eigfilter

Eigenspace-based clutter filter. Reconstructs the input using only singular vectors $k_0$ through $k_1$, removing all other components. In high frame rate ultrasound (Doppler, flow imaging), the first singular components capture stationary tissue clutter. Removing them isolates the weaker blood flow signal.

The input is internally reshaped to a 2D matrix $(m, n)$ where $m$ is the size of the first dimension (the frame axis) and $n$ is the product of all remaining dimensions (spatial). The SVD is computed on this matrix, and the output is reconstructed from the selected singular components.

## Signature

=== "Python"

    ```python
    ffdas.eigfilter(
        x,           # input array (frames, ...)
        k0,          # first singular vector to keep (0-based)
        k1=None,     # past the last singular vector (exclusive)
        *,
        out=None,
    )
    ```

=== "MATLAB"

    ```matlab
    y = ffdas.eigfilter(x, k0)
    y = ffdas.eigfilter(x, k0, k1)
    ```

## Parameters

| Parameter | Python | MATLAB | Description |
|---|---|---|---|
| `x` | at least 2D | at least 2D | Input array. The first dimension (Python) or last dimension (MATLAB) is the frame axis. |
| `k0` | `int`, 0-based | `int`, 1-based | Index of the first singular vector to keep. |
| `k1` | `int` or `None` | `int` or `[]` | Index past the last singular vector to keep (exclusive in Python, inclusive in MATLAB). Defaults to $\min(m, n)$. |

## Returns

Filtered array with the same shape and dtype as the input.

## Example

In high frame rate ultrasound imaging, the first few singular components typically capture stationary tissue (high energy, low rank). Setting `k0` to skip these isolates the weaker flow signal.

=== "Python"

    ```python
    # volume: (n_frames, nz, ny, nx) — reconstructed frame sequence
    # tissue is approximately rank 1; skip the first singular vector
    flow = ffdas.eigfilter(volume, k0=1)

    # keep only components 5 through 20
    band = ffdas.eigfilter(volume, k0=5, k1=20)
    ```

=== "MATLAB"

    ```matlab
    % volume: (nz, ny, nx, n_frames)
    flow = ffdas.eigfilter(volume, 2);         % skip the first singular vector (1-based)
    band = ffdas.eigfilter(volume, 6, 20);     % keep components 6 through 20
    ```

## Notes

The implementation uses cuSOLVER for the SVD computation. For large spatial dimensions, this is substantially faster than host-side SVD followed by reconstruction, since it avoids the device-to-host transfer of the full data matrix.
