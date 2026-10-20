% Receive directivity that changes with the transmit event.
%
% Some probes sum groups of elements into one channel, with delays that
% pre-steer each group at the receive focus of the current transmit (a
% micro-beamforming ASIC). Each group then hears best around a direction
% that changes from transmit to transmit. An SRCDIR of shape
% (4, channels, sequence) gives every channel its own look direction per
% event, in one das / das_sparse call.
%
% This example builds diverging-wave transmits on a 4 x 4 steering grid for
% a 16 x 12 grid of 0.72 mm groups (4 x 4 elements of 0.18 mm), and checks
% that one das_sparse call with the per-event SRCDIR equals one call per
% transmit, each over the pixels that transmit serves with that transmit's
% SRCDIR.

rng(0);

sound_speed = 1540;
center_freq = 5e6;
sampling_freq = 3.125e6;            % complex IQ rate
ks = sampling_freq / sound_speed;   % metres -> sampling wavelengths
n_samples = 256;
batch = 8;

% receive groups
pitch = 0.72e-3;
[gx, gy] = ndgrid(((0:15) - 7.5) * pitch, ((0:11) - 5.5) * pitch);
groups = [gx(:)'; gy(:)'; zeros(1, numel(gx))];
n_channels = size(groups, 2);

% 16 diverging transmits; each one's receive focus is 40 mm along it
steer = linspace(-22.5, 22.5, 4) * pi / 180;
[az, el] = ndgrid(steer, steer);
dirs = [sin(az(:)') .* cos(el(:)'); sin(el(:)'); cos(az(:)') .* cos(el(:)')];
n_tx = size(dirs, 2);
rx_focus = 0.040 * dirs;
virtual_source = -0.0328 * dirs;

% an azimuth plane of pixels
[px, pz] = ndgrid(linspace(-0.02, 0.02, 121), linspace(0.008, 0.06, 121));
pixels = [px(:)'; zeros(1, numel(px)); pz(:)'];
n_pixels = size(pixels, 2);

% transmit delay (samples) and weight of every (pixel, transmit): the beams
% within one steering step of the pixel's direction, raised-cosine weighted
tx_dist = squeeze(sqrt(sum((reshape(pixels, 3, [], 1) - reshape(virtual_source, 3, 1, [])).^2, 1)));
offsets = single((tx_dist - 0.0328) * ks);                                   % (pixels, transmits)
pixel_dir = pixels ./ sqrt(sum(pixels.^2, 1));
angle_off = acos(min(1, pixel_dir' * dirs));                                % (pixels, transmits)
step = 15 * pi / 180;
weights = single((angle_off < step) .* (0.5 + 0.5 * cos(pi * angle_off / step)));

% sparse lists: up to 4 transmits per pixel, the nonzero ones first
k = 4;
[~, order] = sort(weights, 2, "descend");
idx = order(:, 1:k);
sp_w = ffdas.utils.take_along_axis(weights, idx, 2);
sp_off = ffdas.utils.take_along_axis(offsets, idx, 2);

x = complex(randn(n_samples, n_tx, n_channels, batch, "single", "gpuArray"), ...
            randn(n_samples, n_tx, n_channels, batch, "single", "gpuArray"));
srcpos = gpuArray(single(groups * ks));
dstpos = gpuArray(single(pixels * ks));
wavenum = single(-2 * pi * center_freq / sampling_freq);

% every group looks at the current transmit's receive focus, 15 degree cut
look = reshape(rx_focus, 3, 1, []) - groups;                    % (3, channels, transmits)
look = look ./ vecnorm(look, 2, 1);
srcdir = gpuArray(single(cat(1, look, cos(15 * pi / 180) * ones(1, n_channels, n_tx))));

timer = ffdas.utils.Timer();
timer.start();
one_call = ffdas.das_sparse(x, srcpos, dstpos, gpuArray(sp_off), gpuArray(sp_w), ...
    gpuArray(int32(idx)), srcdir, wavenum);
timer.stop();
fprintf("das_sparse, per-event srcdir, one call: %.1f ms\n", timer.elapsed_ms());

timer.start();
per_tx = complex(gpuArray.zeros(n_pixels, batch, "single"));
for t = 1:n_tx
    [served, entry] = find(idx == t);
    lin = sub2ind(size(idx), served, entry);
    per_tx(served, :) = per_tx(served, :) + ffdas.das_sparse(x, srcpos, ...
        dstpos(:, served), gpuArray(sp_off(lin)), gpuArray(sp_w(lin)), ...
        gpuArray(int32(t) * ones(numel(served), 1, "int32")), srcdir(:, :, t), wavenum);
end
timer.stop();
fprintf("das_sparse, one call per transmit:      %.1f ms\n", timer.elapsed_ms());

err = max(abs(one_call(:) - per_tx(:))) / max(abs(per_tx(:)));
fprintf("relative difference: %.1e\n", err);
assert(err < 1e-5, "one call with a per-event srcdir differs from one call per transmit");

% the same call with the pixels grouped by the transmits they compound, so
% that the 32 pixels the GPU runs together read the same transmit (once
% per geometry); the image comes back in grid order
[order, s_idx, s_off, s_w] = ffdas.utils.sort_sparse(gpuArray(int32(idx)), ...
    gpuArray(sp_off), gpuArray(sp_w));
timer.start();
sorted = complex(gpuArray.zeros(n_pixels, batch, "single"));
sorted(order, :) = ffdas.das_sparse(x, srcpos, dstpos(:, order), s_off, s_w, s_idx, srcdir, wavenum);
timer.stop();
fprintf("das_sparse, one call, pixels sorted:    %.1f ms\n", timer.elapsed_ms());
err = max(abs(sorted(:) - one_call(:))) / max(abs(one_call(:)));
assert(err < 1e-5, "sorting the pixels changed the image");
