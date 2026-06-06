# einsum

Binary tensor contraction using Einstein summation notation.

`einsum` contracts two GPU tensors according to a subscript string, dispatching to cuBLAS for the computation. It supports both explicit output modes (e.g., `"ij,jk->ik"`) and implicit mode where repeated indices are summed and unique indices are kept in order of first appearance.

## Signature

=== "Python"

    ```python
    ffdas.einsum(subscripts, a, b, *, out=None)
    ```

=== "MATLAB"

    ```matlab
    out = ffdas.einsum(subscripts, a, b)
    ```

## Parameters

| Parameter | Description |
|---|---|
| `subscripts` | Subscript string, e.g., `"ij,jk->ik"`. Exactly two comma-separated operands. The `->` and output modes are optional; if omitted, output modes are inferred (indices appearing exactly once, in order of first appearance). |
| `a` | First operand. |
| `b` | Second operand. Must have the same dtype as `a`. |

## Returns

Result of the contraction. Scalar outputs (all indices contracted) are returned with shape `(1,)` in Python and squeezed to a scalar in MATLAB.

## Examples

=== "Python"

    ```python
    # matrix multiplication
    C = ffdas.einsum("ij,jk->ik", A, B)

    # batched matrix multiplication
    C = ffdas.einsum("bij,bjk->bik", A, B)

    # outer product
    C = ffdas.einsum("i,j->ij", a, b)
    ```

=== "MATLAB"

    ```matlab
    % matrix multiplication
    C = ffdas.einsum('ij,jk->ik', A, B);

    % batched matmul
    C = ffdas.einsum('bij,bjk->bik', A, B);

    % outer product
    C = ffdas.einsum('i,j->ij', a, b);
    ```

## Notes

Both operands must have the same dtype. The contraction is dispatched to cuBLAS, which handles the underlying GEMM or batched GEMM operation. For contractions that reduce to standard matrix multiplication, performance is comparable to calling cuBLAS directly.
