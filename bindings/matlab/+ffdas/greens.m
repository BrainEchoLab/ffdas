function y = greens(xpos, wavenums, x, ypos)
% GREENS Green's function summation over source positions.
%   Y = GREENS(XPOS, WAVENUMS, X, YPOS) propagates a frequency-domain
%   field from source positions to target positions. For each frequency
%   and target, sums contributions from all sources weighted by the
%   free-space Green's function exp(i*k*r) / r.
%
%   Inputs:
%     XPOS     - Source positions (gpuArray, single), shape (3, sources).
%     WAVENUMS - Wavenumber per frequency bin (gpuArray), shape (frequencies, 1).
%     X        - Input field (gpuArray), shape (frequencies, sources[, batch]).
%                Typically complex-valued.
%     YPOS     - Target positions (gpuArray, single), shape (3, ...).
%
%   Output:
%     Y - Propagated field (gpuArray), shape (frequencies, ...[, batch]).

    arguments 
        xpos gpuArray
        wavenums gpuArray
        x gpuArray
        ypos gpuArray
    end

    h = ffdas.core.get_handle();

    xpos = ffdas.core.astype(xpos, 'single');
    wavenums = ffdas.core.astype(wavenums, 'single');
    ypos = ffdas.core.astype(ypos, 'single');

    y = ffdas.core.ffdas_greens(h, xpos, wavenums, x, ypos);
end
