function result = interpolate(grid_points, values, query_points, mode, fill_value)
% INTERPOLATE Structured 3D grid interpolation.
%   RESULT = INTERPOLATE(GRID_POINTS, VALUES, QUERY_POINTS) interpolates
%   values defined on a regular 3D grid at arbitrary query points using
%   linear interpolation.
%
%   RESULT = INTERPOLATE(..., MODE, FILL_VALUE) specifies the interpolation
%   mode and fill value for points outside the grid.
%
%   Inputs:
%     GRID_POINTS  - Grid vertex positions (gpuArray, single),
%                    shape (3, nx, ny, nz).
%     VALUES       - Values on the grid (gpuArray), shape (nx, ny, nz, ...).
%     QUERY_POINTS - Evaluation points (gpuArray, single), shape (3, ...).
%     MODE         - 'nearest' or 'linear' (default: 'linear').
%     FILL_VALUE   - Value for points outside the grid (default: 0).
%
%   Output:
%     RESULT - Interpolated values (gpuArray), shape matching query points.

    arguments
        grid_points gpuArray
        values gpuArray
        query_points gpuArray
        mode char {mustBeMember(mode, {'nearest', 'linear'})} = 'linear'
        fill_value (1, 1) {mustBeNumeric} = zeros([1, 1], underlyingType(values))
    end
    
    grid_points = ffdas.core.astype(grid_points, 'single');
    query_points = ffdas.core.astype(query_points, 'single');

    if ndims(grid_points) ~= 4 || size(grid_points, 1) ~= 3
        error('grid_points must have shape (3, k, n, m)');
    end
    if ndims(values) < 3
        error('values must have at least 3 dimensions');
    end
    if size(values, 1) ~= size(grid_points, 2) || size(values, 2) ~= size(grid_points, 3) || size(values, 3) ~= size(grid_points, 4)
        error('values shape must match grid shape');
    end
    if ndims(query_points) < 2 || size(query_points, 1) ~= 3
        error('query_points must have shape (3, ...)');
    end

    h = ffdas.core.get_handle();
    result = ffdas.core.ffdas_interpolate(h, grid_points, values, query_points, mode, fill_value);
end
