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
        grid_points: Grid vertex positions, shape (nz, ny, nx, 3).
        mode: "nearest" or "linear".
    """

    def __init__(self, grid_points: TensorLike, mode: str = "linear"):
        if len(grid_points.shape) != 4 or grid_points.shape[3] != 3:
            raise ValueError("grid_points must have shape (nz, ny, nx, 3)")
        grid_points = astype(grid_points, "float32")
        self.shape = grid_points.shape[:3]
        self.mode = _interp_modes.get(mode)
        if not self.mode:
            raise ValueError(
                f"invalid interpolation mode '{mode}' (must be one of {tuple(_interp_modes.keys())})"
            )

        nz, ny, nx = self.shape
        self._plan = _ffdas.create_interpolation_plan(
            get_library_handle(), nx, ny, nz, grid_points, self.mode
        )

    @overload
    def __call__(
        self,
        values: T,
        query_points: TensorLike,
        *,
        fill: float | complex | None = None,
        out: None = ...,
        preprocess: bool = False,
    ) -> T: ...
    @overload
    def __call__(
        self,
        values: TensorLike,
        query_points: TensorLike,
        *,
        fill: float | complex | None = None,
        out: T,
        preprocess: bool = False,
    ) -> T: ...

    def __call__(
        self,
        values: TensorLike,
        query_points: TensorLike,
        *,
        fill: float | complex | None = None,
        out: TensorLike | None = None,
        preprocess: bool = False,
    ) -> TensorLike:
        """Interpolate values at query points.

        Args:
            values: Values on the grid, shape ([batch], nz, ny, nx).
            query_points: Evaluation points, shape (..., 3).
            fill: Fill value for points outside the grid (matching dtype of values). Default 0.
            out: Pre-allocated output array.
            preprocess: If True, cache internal lookup structures for
                the given query points so that subsequent calls with
                different values can skip this step.

        Returns:
            Interpolated values at the query points.
        """
        if len(values.shape) < 3 or len(values.shape) > 4:
            raise ValueError(
                f"values must be three or four-dimensional (got {tuple(values.shape)})"
            )
        if values.shape[-3:] != self.shape:
            raise ValueError(
                f"values must have the same leading dimensions as the configured interpolation points {tuple(values.shape[-3:])} != {tuple(self.shape)})"
            )

        if preprocess:
            self.preprocess(query_points)

        query_points = astype(query_points, "float32")

        if fill is None:
            fill = 0.0

        fill_v = full_like(values, fill, shape=(1,), device="cpu")

        out_shape = query_points.shape[:-1]
        if values.ndim == 4:
            out_shape = (values.shape[0], *out_shape)

        if out is None:
            out = empty_like(values, shape=out_shape)
        elif out.shape != out_shape:
            raise ValueError(f"incorrect dimensions for out: expected {tuple(out_shape)}, got {tuple(out.shape)})")

        _ffdas.interpolation(
            get_library_handle(), self._plan, query_points, values, out, fill_v
        )
        return out

    def preprocess(self, query_points: TensorLike) -> None:
        """Cache internal lookup structures for the given query points.

        Args:
            query_points: Points to preprocess, shape (..., 3).
        """
        query_points = astype(query_points, "float32")
        _ffdas.interpolation_preprocess(
            get_library_handle(), self._plan, query_points
        )


@overload
def interpolate(
    grid_points: TensorLike,
    values: T,
    query_points: TensorLike,
    *,
    mode: str = "linear",
    fill: float | None = None,
    out: None = ...,
) -> T: ...
@overload
def interpolate(
    grid_points: TensorLike,
    values: TensorLike,
    query_points: TensorLike,
    *,
    mode: str = "linear",
    fill: float | None = None,
    out: T,
) -> T: ...


def interpolate(
    grid_points: TensorLike,
    values: TensorLike,
    query_points: TensorLike,
    *,
    mode: str = "linear",
    fill: float | None = None,
    out: TensorLike | None = None,
) -> TensorLike:
    """Interpolate values on a structured 3D grid at arbitrary points.

    Convenience wrapper around ``Interpolator`` for one-shot use. If the
    same grid is reused, construct an ``Interpolator`` directly.

    Args:
        grid_points: Grid vertex positions, shape (nz, ny, nx, 3).
        values: Values on the grid, shape (nz, ny, nx, ...).
        query_points: Evaluation points, shape (..., 3).
        mode: "nearest" or "linear".
        fill: Fill value for points outside the grid. Default 0.
        out: Pre-allocated output array.

    Returns:
        Interpolated values at the query points.
    """
    return Interpolator(grid_points, mode)(values, query_points, fill=fill, out=out)
