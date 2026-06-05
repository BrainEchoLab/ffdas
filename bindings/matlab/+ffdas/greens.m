function out = greens(srcpos, wavenums, x, dstpos)
% GREENS Green's function summation over source positions.
%   OUT = GREENS(SRCPOS, WAVENUMS, X, DSTPOS) propagates a frequency-domain
%   field from source positions to target positions. For each frequency
%   and target, sums contributions from all sources weighted by the
%   free-space Green's function exp(i*k*r) / r.
%
%   Inputs:
%     SRCPOS   - Source positions (gpuArray, single), shape (3, sources).
%     WAVENUMS - Wavenumber per frequency bin (gpuArray), shape (frequencies, 1).
%     X        - Input field (gpuArray), shape (frequencies, sources[, batch]).
%                Typically complex-valued.
%     DSTPOS   - Destination (target) positions (gpuArray, single), shape (3, ...).
%
%   Output:
%     OUT - Propagated field (gpuArray), shape (frequencies, ...[, batch]).

    arguments 
        srcpos gpuArray
        wavenums gpuArray
        x gpuArray
        dstpos gpuArray
    end

    h = ffdas.core.get_handle();

    srcpos = ffdas.core.astype(srcpos, 'single');
    wavenums = ffdas.core.astype(wavenums, 'single');
    dstpos = ffdas.core.astype(dstpos, 'single');

    out = ffdas.core.ffdas_greens(h, srcpos, wavenums, x, dstpos);
end
