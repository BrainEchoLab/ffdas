function y = das(x, xpos, ypos, offsets, weights, xdir, wavenum, algorithm, use_fp16, channels_trailing)
% DAS Delay-and-sum beamforming, compounding over the sequence dimension.
%   Y = DAS(X, XPOS, YPOS, OFFSETS, WEIGHTS) beamforms input data X by
%   computing a weighted sum of interpolated samples from all channels and
%   sequence events for each target position.
%
%   Positions should be in sampling wavelengths (c / fs) and offsets in
%   samples.
%
%   Inputs:
%     X       - Input RF data (gpuArray). Shape:
%               (samples, sequence, channels[, batch]) if CHANNELS_TRAILING, or
%               (samples, channels, sequence[, batch]) otherwise.
%     XPOS    - Channel positions (gpuArray, single), shape (3, channels).
%     YPOS    - Target positions (gpuArray, single), shape (3, ...).
%     OFFSETS - Per-target time offsets in samples (gpuArray, single),
%               shape (..., sequence).
%     WEIGHTS - Per-target apodization weights (gpuArray, single),
%               shape (..., sequence).
%     XDIR    - Channel directivity vectors (gpuArray, single),
%               shape (4, channels). The first three rows are the unit
%               surface normal; the fourth is the cosine of the sensitivity
%               half-angle. Default: [] (disabled).
%     WAVENUM - Wavenumber for phase rotation (single), typically
%               -2*pi*fc/fs. Default: 0.0 (disabled).
%     ALGORITHM      - Algorithm variant (int32, default: 0).
%     USE_FP16       - Use half-precision arithmetic (logical, default: false).
%     CHANNELS_TRAILING - If true, the channel dimension follows the sequence
%                         dimension in X. Default: true.
%
%   Output:
%     Y - Beamformed output (gpuArray), shape (...[, batch]).

    arguments 
        x gpuArray
        xpos gpuArray
        ypos gpuArray
        offsets gpuArray
        weights gpuArray
        xdir gpuArray = []
        wavenum single = 0.0
        algorithm int32 = 0
        use_fp16 logical = false
        channels_trailing logical = true
    end

    h = ffdas.core.get_handle();

    xpos = ffdas.core.astype(xpos, 'single');
    ypos = ffdas.core.astype(ypos, 'single');
    offsets = ffdas.core.astype(offsets, 'single');
    weights = ffdas.core.astype(weights, 'single');
    xdir = ffdas.core.astype(xdir, 'single');

    y = ffdas.core.ffdas_das(h, x, xpos, ypos, offsets, weights, xdir, wavenum, algorithm, use_fp16, channels_trailing);
end
