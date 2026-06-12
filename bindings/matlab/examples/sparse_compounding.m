% Sparse compounding with das_sparse.
%
% Standard compounding (as in compound.m) sums over all transmit events
% for every voxel. When each transmit only illuminates a portion of the
% volume, most voxels receive zero weight from most transmits, wasting
% work. das_sparse lets each voxel specify which events to compound,
% so only meaningful contributions are summed.
%
% This example sets up 25 diverging-wave transmits from virtual sources
% at different steering angles. Each source illuminates a cone-shaped
% region, so any given voxel falls within only a few beams. We select
% the k highest-weight transmits per voxel and compare das_sparse
% against full das (which iterates over all 25, including the zeros).

rng(0);

sound_speed = 1540.0;
center_freq = 5.0e6;
sampling_freq = center_freq;
n_samples = 512;

pitch = 250e-6;
el = gpuArray(single((0:31) - 15.5) * pitch);
[ex, ey] = ndgrid(el, el);
n_channels = 1024;
channel_pos = gpuArray(single([ex(:)'; ey(:)'; zeros(1, n_channels)]));

xmin = -0.008; xmax = 0.008;
ymin = -0.008; ymax = 0.008;
zmin = 0.010;  zmax = 0.026;

% virtual sources behind the aperture at z < 0. a 5x5 angular grid
% gives 25 diverging-wave transmits that together cover the volume
rho = 0.03;
theta_arr = gpuArray(single(linspace(-15, 15, 5))) * (pi / 180);
phi_arr   = gpuArray(single(linspace(-15, 15, 5))) * (pi / 180);
[pp, tt] = ndgrid(phi_arr, theta_arr);
tt = tt(:)';
pp = pp(:)';
sources = [rho * sin(tt) .* cos(pp); ...
           rho * sin(pp); ...
          -rho * cos(tt) .* cos(pp)];
n_sources = size(sources, 2);


n_scatterers = 8192;
tube_sigma = single(0.0001);
bg_sigma = single(0.008);
t_param = 2 * pi * rand(1, n_scatterers, "single", "gpuArray");
is_tube = rand(1, n_scatterers, "single", "gpuArray") < 0.2;
sigma = is_tube * tube_sigma + (~is_tube) * bg_sigma;
scatter_pos = trefoil(t_param) ...
    + randn(3, n_scatterers, "single", "gpuArray") .* sigma;


% simulate RF data for all transmits. each source produces a diverging
% wave: channels are delayed by their distance to the source, minus
% the reference distance to the nearest aperture edge
k_fft = gpuArray(single([0:n_samples/2-1, -n_samples/2:-1]'));
freqs = k_fft * sampling_freq / n_samples + center_freq;
wavenums = single(-2 * pi) * freqs / sound_speed;
sigma_f = 0.6 * center_freq / (2 * sqrt(2 * log(2)));
pulse = complex(exp(-0.5 * ((freqs - center_freq) / sigma_f).^2));

aperture_size = single([32 * pitch; 32 * pitch]);
src_ref = ffdas.utils.rect_dist(sources, aperture_size);
channel_delay = (ffdas.utils.cdist(sources, channel_pos) - src_ref) / sound_speed;

rf = complex(gpuArray.zeros(n_samples, n_sources, n_channels, "single"));
for i = 1:n_sources
    tx_signal = pulse .* exp(-2j * pi * freqs .* channel_delay(i,:));
    tx = ffdas.greens(channel_pos, wavenums, tx_signal, scatter_pos);
    rx = ffdas.greens(scatter_pos, wavenums, tx, channel_pos);
    rf(:, i, :) = reshape(conj(ifft(rx, [], 1)), n_samples, 1, []);
end


% reconstruction grid
nz = 64; ny = 64; nx = 64;
x = gpuArray(single(linspace(xmin, xmax, nx)));
y = gpuArray(single(linspace(ymin, ymax, ny)));
z = gpuArray(single(linspace(zmin, zmax, nz)));
[xx, yy, zz] = ndgrid(x, y, z);
voxel_pos = gpuArray.zeros(3, nx, ny, nz, "single");
voxel_pos(1,:,:,:) = xx;
voxel_pos(2,:,:,:) = yy;
voxel_pos(3,:,:,:) = zz;

ks = sampling_freq / sound_speed;
wavenum = single(-2 * pi * center_freq / sampling_freq);

% transmit delay from each source to each voxel (in samples),
% relative to the nearest-aperture-edge reference
% cdist returns (n_sources, nx, ny, nz); permute to (nx, ny, nz, n_sources)
offsets = permute(ffdas.utils.cdist(sources, voxel_pos) - src_ref, [2 3 4 1]) * ks;
offsets = min(max(offsets, 0), single(n_samples - 1));

% beam mask: a voxel is illuminated if its angle from the source's
% forward direction falls within the geometric opening angle
src_norm = sqrt(sum(sources .* sources, 1));
forward = -sources ./ src_norm;
angle_vol = ffdas.utils.angle( ...
    voxel_pos - reshape(sources, 3, 1, 1, 1, []), ...
    reshape(forward, 3, 1, 1, 1, []));
opening_angle = atan2(0.5 * 32 * pitch, rho);
weights = single(angle_vol < opening_angle);

% element directivity
srcdir = gpuArray.zeros(4, n_channels, "single");
srcdir(3,:) = 1.0;
srcdir(4,:) = 0.707;

% full das: sums all 25 transmits per voxel, including zeros
timer_full = ffdas.utils.Timer();
timer_full.start();
image_full = ffdas.das( ...
    rf, channel_pos * ks, voxel_pos * ks, offsets, weights, srcdir, wavenum);
timer_full.stop();
fprintf("das (all %d transmits): %.1f ms\n", n_sources, timer_full.elapsed_ms());

% sparse compounding: pick the k transmits with highest weight per
% voxel. this is the "sparse" part — instead of iterating over all
% n_sources events, each voxel only processes k of them
k = 8;
[~, sort_idx] = sort(weights, 4, "descend");
sparse_idx = sort_idx(:,:,:,1:k);
sparse_weights = ffdas.utils.take_along_axis(weights, sparse_idx, 4);
sparse_offsets = ffdas.utils.take_along_axis(offsets, sparse_idx, 4);

timer_sparse = ffdas.utils.Timer();
timer_sparse.start();
image_sparse = ffdas.das_sparse( ...
    rf, channel_pos * ks, voxel_pos * ks, ...
    sparse_offsets, sparse_weights, sparse_idx, srcdir, wavenum);
timer_sparse.stop();
fprintf("das_sparse (k=%d of %d): %.1f ms\n", k, n_sources, timer_sparse.elapsed_ms());

diff_db = 20 * log10(max(abs(image_full(:) - image_sparse(:))) ...
    / max(abs(image_full(:))) + 1e-10);
fprintf("max difference: %.1f dB\n", diff_db);


magnitude = abs(image_sparse);

x_mm = gather(x) * 1e3;
y_mm = gather(y) * 1e3;
z_mm = gather(z) * 1e3;

mip_xz = squeeze(max(magnitude, [], 2));
db_xz = 20 * log10(mip_xz / max(mip_xz(:)) + 1e-10);

mip_yz = squeeze(max(magnitude, [], 1));
db_yz = 20 * log10(mip_yz / max(mip_yz(:)) + 1e-10);

figure;
tiledlayout(1, 2);

nexttile;
imagesc(x_mm, z_mm, gather(db_xz)');
colormap("gray"); clim([-32 0]); axis image;
xlabel("x [mm]"); ylabel("z [mm]");
title("xz max projection");

nexttile;
imagesc(y_mm, z_mm, gather(db_yz)');
colormap("gray"); clim([-32 0]); axis image;
xlabel("y [mm]");
title("yz max projection");

exportgraphics(gcf, "sparse_compounding.png", Resolution=150);


function pos = trefoil(t)
    pos = cat(1, ...
        (sin(t) + 2 * sin(2*t)) * 0.0016, ...
        -sin(3*t) * 0.0024, ...
        (cos(t) - 2 * cos(2*t)) * 0.0016 + 0.018);
end
