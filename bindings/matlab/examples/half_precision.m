% Half-precision channel data, converted once and reused.
%
% USE_FP16 = true converts the whole input to half precision inside every
% das / das_sparse call. When one buffer is beamformed several times (an
% orthoview and a volume, say), convert it once with ffdas.to_fp16 and pass
% the result: the image is the same, and the conversion is paid once.
% MATLAB has no half-precision gpuArray, so the half bits travel in a
% uint16 gpuArray; das returns single.
%
% Raw int16 IQ can go to half precision directly, in the gather that
% reorders the channels.

rng(0);

n_samples = 191;
n_events = 3;
n_channels = 8;
n_pixels = 2000;
batch = 16;

x = complex(randn(n_samples, n_events, n_channels, batch, "single", "gpuArray"), ...
            randn(n_samples, n_events, n_channels, batch, "single", "gpuArray"));
channel_pos = gpuArray(single([linspace(-5, 5, n_channels); zeros(2, n_channels)]));
pixel_pos = gpuArray(single([10 * rand(1, n_pixels) - 5; zeros(1, n_pixels); 20 * rand(1, n_pixels) + 40]));
offsets = gpuArray(single(20 * rand(n_pixels, n_events)));
weights = gpuArray(single(0.5 + rand(n_pixels, n_events)));
wavenum = single(-0.5);

per_call = ffdas.das(x, channel_pos, pixel_pos, offsets, weights, [], wavenum, int32(0), true);

x_half = ffdas.to_fp16(x);
assert(isa(x_half, "gpuArray") && strcmp(classUnderlying(x_half), "uint16") && ~isreal(x_half), ...
    "to_fp16 should give a complex uint16 gpuArray");
once = ffdas.das(x_half, channel_pos, pixel_pos, offsets, weights, [], wavenum);
assert(strcmp(classUnderlying(once), "single"), "half-precision input should give single output");
assert(isequal(gather(once), gather(per_call)), "converting once should give the use_fp16 image");
fprintf("to_fp16 once == use_fp16 per call: identical\n");

% an empty batch stays empty, with its size and complexity, and beamforms to an empty image
empty_half = ffdas.to_fp16(x(:, :, :, []));
assert(isequal(size(empty_half), [n_samples, n_events, n_channels, 0]) && ~isreal(empty_half) ...
    && strcmp(classUnderlying(empty_half), "uint16"), "an empty batch should give an empty complex uint16 gpuArray");
assert(isempty(ffdas.das(empty_half, channel_pos, pixel_pos, offsets, weights, [], wavenum)), ...
    "an empty batch should give an empty image");

% raw int16 (real, imag) pairs straight to half precision in the gather
raw = int16(round(1000 * randn(2 * n_samples, n_events, n_channels, batch, "single", "gpuArray")));
half_direct = ffdas.gather(raw, gpuArray(int32(1:n_channels)), 3, [], 'uint16', 'interleaved_complex', true);
half_via_single = ffdas.to_fp16(ffdas.gather(raw, gpuArray(int32(1:n_channels)), 3, [], 'single', ...
    'interleaved_complex', true));
assert(isequal(gather(half_direct), gather(half_via_single)), "int16 -> half should equal int16 -> single -> half");
fprintf("gather int16 -> half: identical to int16 -> single -> half\n");
