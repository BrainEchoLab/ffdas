function out = to_fp16(x)
% TO_FP16 Convert channel data to half precision once, for DAS / DAS_SPARSE.
%   OUT = TO_FP16(X) returns X (a single gpuArray, real or complex) in IEEE
%   half precision. MATLAB has no half-precision gpuArray, so OUT is a
%   uint16 gpuArray of the same size and complexity holding the half bits;
%   DAS and DAS_SPARSE read it as half-precision data and return single.
%
%   USE_FP16 = true converts the whole input on every call. Converting once
%   and passing OUT instead gives the same image without that cost, for
%   example when several grids are beamformed from the same buffer:
%
%       xh = ffdas.to_fp16(x);
%       ortho = ffdas.das_sparse(xh, srcpos, ortho_pos, ...);
%       volume = ffdas.das_sparse(xh, srcpos, volume_pos, ...);

    arguments
        x gpuArray
    end

    % nothing to convert (e.g. an empty batch); gather refuses an empty index list
    if isempty(x)
        out = zeros(size(x), 'uint16', 'gpuArray');
        if ~isreal(x)
            out = complex(out, out);
        end
        return
    end

    % a gather of every element along the last dimension, with a type conversion
    d = max(ndims(x), 2);
    out = ffdas.gather(x, gpuArray(int32(1:size(x, d))), d, [], 'uint16');
end
