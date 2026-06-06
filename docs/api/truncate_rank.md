# truncate_rank

Rank truncation filter via truncated SVD reconstruction. Reconstructs the input using only singular vectors `start` through `stop`, removing all other components. In high frame rate ultrasound (Doppler, flow imaging), the first singular components capture stationary tissue clutter. Removing them isolates the weaker blood flow signal.

The input is internally reshaped to a 2D matrix $(m, n)$ where $m$ is the size of the first dimension (the frame axis) and $n$ is the product of all remaining dimensions (spatial). The SVD is computed on this matrix, and the output is reconstructed from the selected singular components.

## Signature

=== "Python"

    ```python
    ffdas.truncate_rank(
        x,
        start,
        stop=None,
        *,
        out=None,
    )
    ```

=== "MATLAB"

    ```matlab
    y = ffdas.truncate_rank(x, start)
    y = ffdas.truncate_rank(x, start, stop)
    ```

## Parameters

| Parameter | Python | MATLAB | Description |
|---|---|---|---|
| `x` | at least 2D | at least 2D | Input array. The first dimension (Python) or last dimension (MATLAB) is the frame axis. |
| `start` | `int`, 0-based | `int`, 1-based | Index of the first singular vector to keep. |
| `stop` | `int` or `None` | `int` or `[]` | Index past the last singular vector to keep (exclusive in Python, inclusive in MATLAB). Defaults to $\min(m, n)$. |

## Returns

Filtered array with the same shape and dtype as the input.

## Example

In high frame rate ultrasound imaging, the first few singular components typically capture stationary tissue (high energy, low rank). Setting `start` to skip these isolates the weaker flow signal.

=== "Python"

    ```python
    # volumes: (n_frames, nz, ny, nx)
    # tissue is approximately rank 1; skip the first singular vector
    flow = ffdas.truncate_rank(volumes, start=1)

    # keep only components 5 through 20
    band = ffdas.truncate_rank(volumes, start=5, stop=20)
    ```

=== "MATLAB"

    ```matlab
    % volumes: (nz, ny, nx, n_frames)
    % skip the first singular vector (1-based)
    flow = ffdas.truncate_rank(volumes, 2);

    % keep components 6 through 20
    band = ffdas.truncate_rank(volumes, 6, 20);
    ```

## Notes

The implementation uses cuSOLVER for the SVD computation.
