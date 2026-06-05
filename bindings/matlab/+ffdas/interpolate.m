function result = interpolate(gridpos, x, querypos, mode, fill_value)
% INTERPOLATE Structured 3D grid interpolation.
%   RESULT = INTERPOLATE(GRIDPOS, X, QUERYPOS) interpolates
%   values defined on a regular 3D grid at arbitrary query points using
%   linear interpolation.
%
%   RESULT = INTERPOLATE(..., MODE, FILL_VALUE) specifies the interpolation
%   mode and fill value for points outside the grid.
%
%   Inputs:
%     GRIDPOS    - Grid vertex positions (gpuArray, single),
%                  shape (3, nx, ny, nz).
%     X          - Values on the grid (gpuArray), shape (nx, ny, nz, ...).
%     QUERYPOS   - Evaluation points (gpuArray, single), shape (3, ...).
%     MODE       - 'nearest' or 'linear' (default: 'linear').
%     FILL_VALUE - Value for points outside the grid (default: 0).
%
%   Output:
%     RESULT - Interpolated values (gpuArray), shape matching query points.

    arguments
        gridpos gpuArray
        x gpuArray
        querypos gpuArray
        mode char {mustBeMember(mode, {'nearest', 'linear'})} = 'linear'
        fill_value (1, 1) {mustBeNumeric} = zeros([1, 1], underlyingType(x))
    end
    
    gridpos = ffdas.core.astype(gridpos, 'single');
    querypos = ffdas.core.astype(querypos, 'single');

    if ndims(gridpos) ~= 4 || size(gridpos, 1) ~= 3
        error('gridpos must have shape (3, k, n, m)');
    end
    if ndims(x) < 3
        error('x must have at least 3 dimensions');
    end
    if size(x, 1) ~= size(gridpos, 2) || size(x, 2) ~= size(gridpos, 3) || size(x, 3) ~= size(gridpos, 4)
        error('x shape must match grid shape');
    end
    if ndims(querypos) < 2 || size(querypos, 1) ~= 3
        error('querypos must have shape (3, ...)');
    end

    h = ffdas.core.get_handle();
    result = ffdas.core.ffdas_interpolate(h, gridpos, x, querypos, mode, fill_value);
end
