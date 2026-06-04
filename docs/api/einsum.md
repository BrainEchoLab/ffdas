# einsum

Binary tensor contraction using Einstein summation notation.

`einsum` contracts two GPU tensors according to a subscript string, dispatching to cuBLAS for the computation. It supports both explicit output modes (e.g., `"ij,jk->ik"`) and implicit mode where repeated indices are summed and unique indices are kept in order of first appearance.

## Signature

=== "Python"

    ```python
    ffdas.einsum(subscripts, a, x, *, out=None)
    ```

=== "MATLAB"

    ```matlab
    y = ffdas.einsum(subscripts, a, x)
    ```

## Parameters

| Parameter | Description |
|---|---|
| `subscripts` | Subscript string, e.g., `"ij,jk->ik"`. Exactly two comma-separated operands. The `->` and output modes are optional; if omitted, output modes are inferred (indices appearing exactly once, in order of first appearance). |
| `a` | First operand. |
| `x` | Second operand. Must have the same dtype as `a`. |

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

    # trace (implicit output — all indices repeated, so scalar output)
    t = ffdas.einsum("ii", A, eye)  # returns shape (1,)
    ```

=== "MATLAB"

    ```matlab
    C = ffdas.einsum('ij,jk->ik', A, B);     % matrix multiplication
    C = ffdas.einsum('bij,bjk->bik', A, B);   % batched matmul
    C = ffdas.einsum('i,j->ij', a, b);        % outer product
    ```

## Notes

Both operands must have the same dtype. The contraction is dispatched to cuBLAS, which handles the underlying GEMM or batched GEMM operation. For contractions that reduce to standard matrix multiplication, performance is comparable to calling cuBLAS directly.
