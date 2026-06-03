function d = rect_dist(points, sz)
% RECT_DIST Minimum distance to an axis-aligned rectangle.
%   D = RECT_DIST(POINTS, SZ) computes the minimum Euclidean distance from
%   3D points to a rectangle centered at the origin in the z=0 plane.
%
%   POINTS: (3, ...)
%   SZ: [width, height]

    trailing = size(points);
    trailing = trailing(2:end);
    points = reshape(points, 3, []);

    half = sz(:) / 2;
    dx = max(abs(points(1, :)) - half(1), 0);
    dy = max(abs(points(2, :)) - half(2), 0);
    dz = abs(points(3, :));
    d = sqrt(dx.^2 + dy.^2 + dz.^2);
    d = reshape(d, trailing);
end
