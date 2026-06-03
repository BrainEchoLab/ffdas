function y = gather(x, indices, axis, permutation, dtype)
% GATHER Gather elements along a dimension.
%   Y = GATHER(X, INDICES, AXIS) gathers elements from X at positions
%   INDICES along dimension AXIS. INDICES are 1-based.
%
%   Y = GATHER(X, INDICES, AXIS, PERMUTATION) simultaneously permutes X
%   during the gather, avoiding the extra copy that MATLAB's permute
%   would create.
%
%   Y = GATHER(..., DTYPE) casts the output to DTYPE. Supported types:
%   'single', 'int16', 'int32'. Complexity is inferred from the input.
%
%   Inputs:
%     X           - Input array (gpuArray).
%     INDICES     - 1-based index vector (gpuArray, int32).
%     AXIS        - Dimension along which to gather (1-based).
%     PERMUTATION - Optional permutation vector (1-based, int32).
%     DTYPE       - Optional output datatype (char).
%
%   Output:
%     Y - Gathered array (gpuArray). Same shape as X except along AXIS,
%         where the size equals length(INDICES).

    arguments 
        x gpuArray
        indices gpuArray
        axis int32
        permutation (1, :) int32 = []
        dtype char = ''
    end

    h = ffdas.core.get_handle();

    indices = ffdas.core.astype(indices, 'int32');
    indices = indices - 1;

    if isempty(permutation) && isempty(dtype)
        y = ffdas.core.ffdas_gather(h, x, indices, axis);
    elseif ~isempty(permutation) && isempty(dtype)
        y = ffdas.core.ffdas_gather(h, x, indices, axis, permutation);
    else
        y = ffdas.core.ffdas_gather(h, x, indices, axis, permutation, dtype);
    end
end
