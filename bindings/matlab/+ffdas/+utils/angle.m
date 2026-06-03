function theta = angle(a, b, eps)
% ANGLE Angle between vectors.
%   THETA = ANGLE(A, B) returns the angle in radians between vectors A and B.
%   The first dimension is the coordinate dimension; implicit expansion
%   applies to all remaining dimensions.
%
%   A: (D, ...), B: (D, ...) -> (...)

    arguments
        a
        b
        eps (1,1) double = 1e-7
    end

    dot_ab = sum(a .* b, 1);
    norm_a = sqrt(sum(a .* a, 1));
    norm_b = sqrt(sum(b .* b, 1));
    cos_theta = dot_ab ./ (norm_a .* norm_b + eps);
    theta = acos(min(max(cos_theta, -1), 1));

    % sum over dim 1 leaves a leading singleton; compute the broadcast
    % output shape and reshape to remove it
    sz_a = size(a); sz_b = size(b);
    ndim = max(numel(sz_a), numel(sz_b));
    sz_a(end+1:ndim) = 1; sz_b(end+1:ndim) = 1;
    out_shape = max(sz_a, sz_b);
    theta = reshape(theta, out_shape(2:end));
end
