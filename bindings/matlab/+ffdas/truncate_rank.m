function out = truncate_rank(x, start, stop)
% TRUNCATE_RANK Rank truncation filter via truncated SVD reconstruction.
%   OUT = TRUNCATE_RANK(X, START) reconstructs X using only singular vectors
%   from START onward, removing the first START-1 components.
%
%   OUT = TRUNCATE_RANK(X, START, STOP) keeps only singular vectors START
%   through STOP.
%
%   The input is reshaped to a 2D matrix (m, n) where n is the last
%   dimension and m is the product of all preceding dimensions. The output
%   has the same shape as X.
%
%   Inputs:
%     X     - Input array (gpuArray), at least 2 dimensions.
%     START - Index of the first singular vector to keep (1-based).
%     STOP  - Index of the last singular vector to keep (1-based, inclusive).
%             Default: min(m, n).

    arguments 
        x gpuArray
        start int64
        stop int64 = []
    end

    shp = size(x);
    x = reshape(x, [], size(x, ndims(x)));
    m = size(x, 1);
    n = size(x, 2);

    if isempty(stop)
        stop = min(m, n);
    end

    h = ffdas.core.get_handle();
    out = ffdas.core.ffdas_truncate_rank(h, x, start, stop);
    out = reshape(out, shp);
end
