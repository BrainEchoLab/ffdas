function out = das_sparse(x, srcpos, dstpos, offsets, weights, sparse_indices, srcdir, wavenum, algorithm, use_fp16, channels_trailing)
% DAS_SPARSE Sparse compounding delay-and-sum beamforming.
%   OUT = DAS_SPARSE(X, SRCPOS, DSTPOS, OFFSETS, WEIGHTS, SPARSE_INDICES)
%   beamforms like DAS, but each target compounds over a per-target subset
%   of n sequence events selected by SPARSE_INDICES.
%
%   Inputs:
%     X       - Input RF data (gpuArray). Shape:
%               (samples, sequence, channels[, batch]) if CHANNELS_TRAILING, or
%               (samples, channels, sequence[, batch]) otherwise.
%     SRCPOS  - Source (channel) positions (gpuArray, single), shape (3, channels).
%     DSTPOS  - Destination (target) positions (gpuArray, single), shape (3, ...).
%     OFFSETS - Per-target time offsets in samples (gpuArray, single),
%               shape (..., n).
%     WEIGHTS - Per-target apodization weights (gpuArray, single),
%               shape (..., n).
%     SPARSE_INDICES - Indices into the sequence dimension of X (gpuArray,
%               int32, 1-based), shape (..., n).
%     SRCDIR  - Source directivity vectors (gpuArray, single),
%               shape (4, channels). See DAS.
%     WAVENUM - Wavenumber for phase rotation (single, default: 0.0).
%     ALGORITHM      - Algorithm variant (int32, default: 0).
%                      Possible values: 0 (auto), 1, 2, 4.
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
        sparse_indices gpuArray
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
    sparse_indices = ffdas.core.astype(sparse_indices, 'int32');
    sparse_indices = sparse_indices - 1;
    srcdir = ffdas.core.astype(srcdir, 'single');

    out = ffdas.core.ffdas_das_sparse(h, x, srcpos, dstpos, offsets, weights, sparse_indices, srcdir, wavenum, algorithm, use_fp16, channels_trailing);
end
