# truncate_rank

Reconstruct the input from a subset of its singular vectors. Given the SVD $X = U \Sigma V^H$, the output is $U_k \Sigma_k V_k^H$ where $k$ indexes the selected components. Singular vectors are indexed in descending order of singular value: index 0 corresponds to the largest.

The input is reshaped to a 2D matrix $(m, n)$ where $m$ is the size of the first dimension (Python) or last dimension (MATLAB), and $n$ is the product of all remaining dimensions. The function eigendecomposes the smaller of the two Gram matrices ($X^H X$ or $X X^H$, whichever has lower rank), builds a projection matrix from the selected eigenvectors, and applies it. This is equivalent to truncated SVD reconstruction but avoids computing the full decomposition.

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
| `x` | at least 2D | at least 2D | Input array. The first dimension (Python) or last dimension (MATLAB) is preserved; all others are flattened for the decomposition. |
| `start` | `int`, 0-based | `int`, 1-based | First singular vector to keep, counting from the largest. |
| `stop` | `int` or `None` | `int` or `[]` | One past the last singular vector to keep (Python, exclusive) or last to keep (MATLAB, inclusive). Defaults to $\min(m, n)$. |

## Returns

Filtered array with the same shape and dtype as the input.

## Example

In high frame rate ultrasound, the first singular components typically capture stationary tissue (high energy, low rank). Setting `start` to skip them isolates the weaker flow signal.

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

The eigendecomposition is computed via `cusolverDnXsyevd`. The Gram matrix is formed with `cublasSgemm` (or the complex/double equivalent) rather than `ssyrk`/`cherk` to avoid rounding errors when the matrix dimensions differ substantially.

The cost is dominated by the eigendecomposition of the $p \times p$ Gram matrix, where $p = \min(m, n)$, plus two matrix multiplications involving the full $m \times n$ input.
