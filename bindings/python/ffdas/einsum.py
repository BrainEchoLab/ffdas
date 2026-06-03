from typing import overload

from ._core import _ffdas
from ._core.library import get_library_handle
from ._core.tensor_like import TensorLike, T
from ._core.tensor import reshape, empty_like


def implicit_output_modes(modes: list[str]):
    # count number of times modes appear across all operands
    mode_count = {}
    for mode in "".join(modes):
        mode_count[mode] = mode_count.get(mode, 0) + 1

    # output modes appear exactly once, and are ordered by first appearance
    output_modes = []
    seen = set()
    for mode in "".join(modes):
        if mode_count[mode] == 1 and mode not in seen:
            output_modes.append(mode)
            seen.add(mode)

    return "".join(output_modes)


@overload
def einsum(
    subscripts: str,
    a: T,
    x: T,
    *,
    out: None = None,
) -> T: ...
@overload
def einsum(
    subscripts: str,
    a: TensorLike,
    x: TensorLike,
    *,
    out: T,
) -> T: ...

def einsum(
    subscripts: str,
    a: TensorLike,
    x: TensorLike,
    *,
    out: TensorLike | None = None,
) -> TensorLike:
    """Binary tensor contraction using Einstein summation notation.

    Supports explicit output modes (e.g. "ij,jk->ik") and implicit mode
    where repeated indices are summed and unique indices are kept in order
    of first appearance. Both operands must have the same dtype.

    Scalar outputs (all indices contracted) are returned with shape (1,).

    Args:
        subscripts: Subscript string, e.g. "ij,jk->ik".
        a: First operand.
        x: Second operand.
        out: Pre-allocated output array.

    Returns:
        Result of the contraction.
    """
    sides = subscripts.split("->")

    if len(sides) == 1:
        # No '->' found, infer output modes
        modes = sides[0].split(",")
        if len(modes) != 2:
            raise ValueError(
                f"subscripts must contain exactly one ',' (got '{subscripts}')"
            )
        am, xm = modes
        ym = implicit_output_modes(modes)
    elif len(sides) == 2:
        # Explicit output modes provided
        modes = sides[0].split(",")
        if len(modes) != 2:
            raise ValueError(
                f"subscripts must contain exactly one ',' (got '{subscripts}')"
            )

        am, xm = modes
        ym = sides[1]
    else:
        raise ValueError(
            f"subscripts must contain at most one '->' (got '{subscripts}')"
        )

    dims = {m: d for m, d in zip(am, a.shape)}

    for m, d in zip(xm, x.shape):
        prev = dims.setdefault(m, d)
        if prev != d:
            raise ValueError(f"input dimensions don't match for subscript '{m}'")

    if len(set(ym) - set(dims.keys())) > 0:
        raise ValueError(f"unexpected modes in output: {set(ym) - dims.keys()}")

    # Handle scalar output case by adding a dummy dimension
    scalar_output = len(ym) == 0
    if scalar_output:
        # Add a dummy mode and reshape one of the operands to have an extra dimension of size 1
        dummy_mode = chr(max(ord(m) for m in am + xm) + 1)  # Find an unused mode
        ym = dummy_mode
        # Add dummy dimension to the second operand (x)
        x = reshape(x, x.shape + (1,))
        xm = xm + dummy_mode
        dims[dummy_mode] = 1

    out_shape = tuple(dims[m] for m in ym)

    if out is None:
        out = empty_like(x, shape=out_shape)
    elif out.shape != out_shape:
        raise ValueError(
            f"invalid output shape: {tuple(out.shape)} (expected {out_shape})"
        )

    am = [ord(m) for m in am]
    xm = [ord(m) for m in xm]
    ym = [ord(m) for m in ym]

    plan = _ffdas.create_contraction(
        get_library_handle(),
        x,
        xm,
        a,
        am,
        out,
        ym,
    )

    _ffdas.contraction(
        get_library_handle(),
        plan,
        x,
        a,
        out,
    )

    # Remove dummy dimension for scalar output
    if scalar_output:
        out = reshape(out, (1,))

    return out
