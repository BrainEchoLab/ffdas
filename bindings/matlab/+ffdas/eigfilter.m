function y = eigfilter(x, k0, k1)
% EIGFILTER Eigenvalue-based clutter filter.
%   Y = EIGFILTER(X, K0) reconstructs X using only singular vectors from
%   K0 onward, removing the first K0-1 components. This is a fast
%   approximation of truncated SVD reconstruction.
%
%   Y = EIGFILTER(X, K0, K1) keeps only singular vectors K0 through K1.
%
%   The input is reshaped to a 2D matrix (m, n) where n is the last
%   dimension and m is the product of all preceding dimensions. The output
%   has the same shape as X.
%
%   Inputs:
%     X  - Input array (gpuArray), at least 2 dimensions.
%     K0 - Index of the first singular vector to keep (1-based).
%     K1 - Index past the last singular vector to keep (1-based, inclusive).
%          Default: min(m, n).

    arguments 
        x gpuArray
        k0 int64
        k1 int64 = []
    end

    shp = size(x);
    x = reshape(x, [], size(x, ndims(x)));
    m = size(x, 1);
    n = size(x, 2);

    if isempty(k1)
        k1 = min(m, n);
    end

    h = ffdas.core.get_handle();
    y = ffdas.core.ffdas_eigfilter(h, x, k0, k1);
    y = reshape(y, shp);
end
