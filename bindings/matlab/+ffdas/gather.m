function out = gather(x, indices, axis, permutation, dtype, options)
% GATHER Gather elements along a dimension.
%   OUT = GATHER(X, INDICES, AXIS) gathers elements from X at positions
%   INDICES along dimension AXIS. INDICES are 1-based.
%
%   OUT = GATHER(X, INDICES, AXIS, PERMUTATION) simultaneously permutes X
%   during the gather, avoiding the extra copy that MATLAB's permute
%   would create.
%
%   OUT = GATHER(..., DTYPE) casts the output to DTYPE. Supported types:
%   'single', 'int16', 'int32', and 'uint16' for IEEE half precision (held
%   in a uint16 gpuArray; see FFDAS.TO_FP16). Complexity is inferred from
%   the input.
%
%   OUT = GATHER(..., 'interleaved_complex', true) reads a real X as
%   interleaved (real, imag) pairs along its first dimension: an int16 or
%   single array of size [2*N, ...] is treated as a complex array of size
%   [N, ...], without a copy. AXIS and PERMUTATION refer to that complex
%   array. For raw IQ data from an acquisition system, for example:
%
%       data = ffdas.gather(raw, channel_map, 4, [1 2 4 3], 'single', ...
%                           'interleaved_complex', true);
%
%   Inputs:
%     X           - Input array (gpuArray).
%     INDICES     - 1-based index vector (gpuArray, int32).
%     AXIS        - Dimension along which to gather (1-based).
%     PERMUTATION - Optional permutation vector (1-based, int32).
%     DTYPE       - Optional output datatype (char).
%
%   Name-value options:
%     interleaved_complex - Read a real X as interleaved (real, imag)
%                           pairs (logical, default false).
%
%   Output:
%     OUT - Gathered array (gpuArray). Same shape as X except along AXIS,
%         where the size equals length(INDICES).

    arguments
        x gpuArray
        indices gpuArray
        axis int32
        permutation (1, :) int32 = []
        dtype char = ''
        options.interleaved_complex (1, 1) logical = false
    end

    h = ffdas.core.get_handle();

    indices = ffdas.core.astype(indices, 'int32');
    indices = indices - 1;

    out = ffdas.core.ffdas_gather(h, x, indices, axis, permutation, dtype, options.interleaved_complex);
end
