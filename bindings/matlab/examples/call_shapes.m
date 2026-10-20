% Array shapes das and das_sparse accept, checked against each other.
%
% MATLAB drops trailing singleton dimensions: a single frame of data has
% no batch dimension, and the offsets/weights of a single event on a flat
% list of pixels are (pixels x 1). This example builds one small problem
% and checks that every way of expressing it gives the same image:
%
%   1. das over all events of a single frame (3-D x) equals the sum of one
%      das call per event (each with (pixels x 1) offsets and weights);
%   2. das_sparse with one entry per pixel (n = 1, default srcdir) equals
%      das over the event that entry selects;
%   3. zero pixels give an empty result.

rng(0);

n_samples = 191;
n_events = 3;
n_channels = 8;
n_pixels = 200;

x = complex(randn(n_samples, n_events, n_channels, "single", "gpuArray"), ...
            randn(n_samples, n_events, n_channels, "single", "gpuArray"));
channel_pos = gpuArray(single([linspace(-5, 5, n_channels); zeros(2, n_channels)]));
pixel_pos = gpuArray(single([10 * rand(1, n_pixels) - 5; ...
                             4 * rand(1, n_pixels) - 2; ...
                             20 * rand(1, n_pixels) + 40]));
% every event its own delays and weights
offsets = gpuArray(single(20 * rand(n_pixels, n_events)));
weights = gpuArray(single(0.5 + rand(n_pixels, n_events)));
wavenum = single(-0.5);
tol = 1e-5;

% 1. all events at once vs one call per event
image_all = ffdas.das(x, channel_pos, pixel_pos, offsets, weights, [], wavenum);
image_sum = complex(gpuArray.zeros(n_pixels, 1, "single"));
for e = 1:n_events
    image_sum = image_sum + ffdas.das(x(:, e, :), channel_pos, pixel_pos, ...
        offsets(:, e), weights(:, e), [], wavenum);
end
err = max(abs(image_all(:) - image_sum(:))) / max(abs(image_sum(:)));
fprintf("all events vs one call per event: relative difference %.1e\n", err);
assert(err < tol, "das over all events differs from the per-event sum");

% 2. das_sparse with a single entry per pixel (n = 1) and the default srcdir
e = 2;
image_sparse = ffdas.das_sparse(x, channel_pos, pixel_pos, offsets(:, e), ...
    weights(:, e), int32(e) * gpuArray.ones(n_pixels, 1, "int32"));
image_event = ffdas.das(x(:, e, :), channel_pos, pixel_pos, offsets(:, e), weights(:, e));
err = max(abs(image_sparse(:) - image_event(:))) / max(abs(image_event(:)));
fprintf("das_sparse (n = 1) vs das over that event: relative difference %.1e\n", err);
assert(err < tol, "das_sparse with one entry per pixel differs from das");

% 3. no pixels at all
image_none = ffdas.das(x, channel_pos, gpuArray.zeros(3, 0, "single"), ...
    gpuArray.zeros(0, n_events, "single"), gpuArray.zeros(0, n_events, "single"));
assert(isempty(image_none), "zero pixels should give an empty result");
fprintf("zero pixels: empty result\n");
