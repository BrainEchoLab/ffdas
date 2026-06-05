function out = das(x, srcpos, dstpos, offsets, weights, srcdir, wavenum, algorithm, use_fp16, channels_trailing)
% DAS Delay-and-sum beamforming, compounding over the sequence dimension.
%   OUT = DAS(X, SRCPOS, DSTPOS, OFFSETS, WEIGHTS) beamforms input data X by
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
%     SRCPOS  - Source (channel) positions (gpuArray, single), shape (3, channels).
%     DSTPOS  - Destination (target) positions (gpuArray, single), shape (3, ...).
%     OFFSETS - Per-target time offsets in samples (gpuArray, single),
%               shape (..., sequence).
%     WEIGHTS - Per-target apodization weights (gpuArray, single),
%               shape (..., sequence).
%     SRCDIR  - Source directivity vectors (gpuArray, single),
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
%     OUT - Beamformed output (gpuArray), shape (...[, batch]).

    arguments 
        x gpuArray
        srcpos gpuArray
        dstpos gpuArray
        offsets gpuArray
        weights gpuArray
        srcdir gpuArray = []
        wavenum single = 0.0
        algorithm int32 = 0
        use_fp16 logical = false
        channels_trailing logical = true
    end

    h = ffdas.core.get_handle();

    srcpos = ffdas.core.astype(srcpos, 'single');
    dstpos = ffdas.core.astype(dstpos, 'single');
    offsets = ffdas.core.astype(offsets, 'single');
    weights = ffdas.core.astype(weights, 'single');
    srcdir = ffdas.core.astype(srcdir, 'single');

    out = ffdas.core.ffdas_das(h, x, srcpos, dstpos, offsets, weights, srcdir, wavenum, algorithm, use_fp16, channels_trailing);
end
