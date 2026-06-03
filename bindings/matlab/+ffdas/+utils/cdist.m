function d = cdist(a, b)
% CDIST Pairwise Euclidean distance.
%   D = CDIST(A, B) computes the distance from each point in A to each
%   point in B. The first dimension is the coordinate dimension.
%
%   A: (D, ...A), B: (D, ...B) -> (...A, ...B)

    D = size(a, 1);
    sz_a = size(a); 
    trailing_a = sz_a(2:end);
    sz_b = size(b); 
    trailing_b = sz_b(2:end);

    a = reshape(a, [D, trailing_a, ones(1, numel(trailing_b))]);
    b = reshape(b, [D, ones(1, numel(trailing_a)), trailing_b]);

    diff = a - b;
    d = sqrt(sum(diff .* diff, 1));
    d = reshape(d, [trailing_a, trailing_b]);
end
