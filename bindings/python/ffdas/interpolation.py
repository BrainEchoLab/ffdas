from typing import overload

from ._core import _ffdas
from ._core.library import get_library_handle
from ._core.tensor_like import TensorLike, T
from ._core.tensor import empty_like, full_like, astype


_interp_modes = {
    "nearest": _ffdas.InterpMode.NEAREST,
    "linear": _ffdas.InterpMode.LINEAR,
}


class Interpolator:
    """Structured 3D grid interpolation.

    Creates a reusable interpolation plan for a fixed grid. Use this when
    interpolating multiple value arrays or query point sets on the same grid.

    Args:
        gridpos: Grid vertex positions, shape (nz, ny, nx, 3).
        mode: "nearest" or "linear".
    """

    def __init__(self, gridpos: TensorLike, mode: str = "linear"):
        if len(gridpos.shape) != 4 or gridpos.shape[3] != 3:
            raise ValueError("gridpos must have shape (nz, ny, nx, 3)")
        gridpos = astype(gridpos, "float32")
        self.shape = gridpos.shape[:3]
        self.mode = _interp_modes.get(mode)
        if not self.mode:
            raise ValueError(
                f"invalid interpolation mode '{mode}' (must be one of {tuple(_interp_modes.keys())})"
            )

        nz, ny, nx = self.shape
        self._plan = _ffdas.create_interpolation_plan(
            get_library_handle(), nx, ny, nz, gridpos, self.mode
        )

    @overload
    def __call__(
        self,
        x: T,
        querypos: TensorLike,
        *,
        fill: float | complex | None = None,
        out: None = ...,
        preprocess: bool = False,
    ) -> T: ...
    @overload
    def __call__(
        self,
        x: TensorLike,
        querypos: TensorLike,
        *,
        fill: float | complex | None = None,
        out: T,
        preprocess: bool = False,
    ) -> T: ...

    def __call__(
        self,
        x: TensorLike,
        querypos: TensorLike,
        *,
        fill: float | complex | None = None,
        out: TensorLike | None = None,
        preprocess: bool = False,
    ) -> TensorLike:
        """Interpolate values at query points.

        Args:
            x: Values on the grid, shape ([batch], nz, ny, nx).
            querypos: Evaluation points, shape (..., 3).
            fill: Fill value for points outside the grid (matching dtype of x). Default 0.
            out: Pre-allocated output array.
            preprocess: If True, cache internal lookup structures for
                the given query points so that subsequent calls with
                different values can skip this step.

        Returns:
            Interpolated values at the query points.
        """
        if self._plan is None:
            raise RuntimeError("Interpolator has been destroyed")

        if len(x.shape) < 3 or len(x.shape) > 4:
            raise ValueError(
                f"x must be three or four-dimensional (got {tuple(x.shape)})"
            )
        if x.shape[-3:] != self.shape:
            raise ValueError(
                f"x must have the same leading dimensions as the configured interpolation points {tuple(x.shape[-3:])} != {tuple(self.shape)})"
            )

        if preprocess:
            self.preprocess(querypos)

        querypos = astype(querypos, "float32")

        if fill is None:
            fill = 0.0

        fill_v = full_like(x, fill, shape=(1,), device="cpu")

        out_shape = querypos.shape[:-1]
        if x.ndim == 4:
            out_shape = (x.shape[0], *out_shape)

        if out is None:
            out = empty_like(x, shape=out_shape)
        elif out.shape != out_shape:
            raise ValueError(f"incorrect dimensions for out: expected {tuple(out_shape)}, got {tuple(out.shape)})")

        _ffdas.interpolation(
            get_library_handle(), self._plan, querypos, x, out, fill_v
        )
        return out

    def preprocess(self, querypos: TensorLike) -> None:
        """Cache internal lookup structures for the given query points.

        Args:
            querypos: Points to preprocess, shape (..., 3).
        """
        if self._plan is None:
            raise RuntimeError("Interpolator has been destroyed")
        querypos = astype(querypos, "float32")
        _ffdas.interpolation_preprocess(
            get_library_handle(), self._plan, querypos
        )

    def close(self) -> None:
        """Release the internal plan and its GPU resources."""
        self._plan = None

    def __del__(self):
        self.close()


@overload
def interpolate(
    gridpos: TensorLike,
    x: T,
    querypos: TensorLike,
    *,
    mode: str = "linear",
    fill: float | None = None,
    out: None = ...,
) -> T: ...
@overload
def interpolate(
    gridpos: TensorLike,
    x: TensorLike,
    querypos: TensorLike,
    *,
    mode: str = "linear",
    fill: float | None = None,
    out: T,
) -> T: ...


def interpolate(
    gridpos: TensorLike,
    x: TensorLike,
    querypos: TensorLike,
    *,
    mode: str = "linear",
    fill: float | None = None,
    out: TensorLike | None = None,
) -> TensorLike:
    """Interpolate values on a structured 3D grid at arbitrary points.

    Convenience wrapper around ``Interpolator`` for one-shot use. If the
    same grid is reused, construct an ``Interpolator`` directly.

    Args:
        gridpos: Grid vertex positions, shape (nz, ny, nx, 3).
        x: Values on the grid, shape ([batch,] nz, ny, nx).
        querypos: Evaluation points, shape (..., 3).
        mode: "nearest" or "linear".
        fill: Fill value for points outside the grid. Default 0.
        out: Pre-allocated output array.

    Returns:
        Interpolated values at the query points.
    """
    return Interpolator(gridpos, mode)(x, querypos, fill=fill, out=out)
