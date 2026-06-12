% Clutter filtering with truncate_rank.
%
% In high frame rate ultrasound sequences (Doppler, flow imaging), the
% first singular components capture largely static tissue clutter.
% truncate_rank removes these, isolating the weaker flow signal.
%
% This example simulates a tissue background with stationary scatterers
% (coherent across frames, low rank in slow time) and a flow component
% consisting of two circular vessels in perpendicular planes whose
% scatterers shift between frames. truncate_rank separates the two,
% revealing the vessel geometry.

rng(0);

sound_speed = 1540.0;
center_freq = 3.08e6;
sampling_freq = center_freq;
n_samples = 512;
n_frames = 64;

% 32-by-32 matrix array
pitch = 500e-6;
el = gpuArray(single((0:31) - 15.5) * pitch);
[ex, ey] = ndgrid(el, el);
n_channels = 1024;
channel_pos = gpuArray(single([ex(:)'; ey(:)'; zeros(1, n_channels)]));

xmin = -0.008; xmax = 0.008;
ymin = -0.008; ymax = 0.008;
zmin = 0.010;  zmax = 0.026;


% flow: two circular vessels in perpendicular planes.
% ring 1 lives in the xz plane, ring 2 in the yz plane
vessel_radius = 0.0025;
n_rings = 128;
t = gpuArray(single((0:n_rings/2 - 1) * (2 * pi / (n_rings / 2))));

right_depth_1 = zmin + 0.25 * (zmax - zmin);
right_depth_2 = zmin + 0.75 * (zmax - zmin);

% tissue: stationary scatterers throughout the volume
n_scatterers = 16384;
background_pos = [ ...
    xmin + (xmax - xmin) * rand(1, n_scatterers - n_rings, "single", "gpuArray"); ...
    ymin + (ymax - ymin) * rand(1, n_scatterers - n_rings, "single", "gpuArray"); ...
    zmin + (zmax - zmin) * rand(1, n_scatterers - n_rings, "single", "gpuArray")];

scatter_values = cat(2, ...
    single(0.1) * rand(1, n_rings, "single", "gpuArray"), ...
    rand(1, n_scatterers - n_rings, "single", "gpuArray"));

k_fft = gpuArray(single([0:n_samples/2-1, -n_samples/2:-1]'));
freqs = k_fft * sampling_freq / n_samples + center_freq;
wavenums = single(-2 * pi) * freqs / sound_speed;
sigma_f = 0.6 * center_freq / (2 * sqrt(2 * log(2)));
pulse = complex(exp(-0.5 * ((freqs - center_freq) / sigma_f).^2));

channel_delay = zeros(1, n_channels, "single", "gpuArray");
tx_signal = pulse .* exp(-2j * pi * freqs .* channel_delay);

% each frame has a different flow scatterer configuration: positions
% are shifted along the vessel by a fraction of the period
rot_per_frame = 0.1 * 2 * pi / n_frames;
rf = complex(zeros(n_samples, 1, n_channels, n_frames, "single", "gpuArray"));

for i = 1:n_frames
    phase = (i - 1) * rot_per_frame;
    ct = vessel_radius * cos(t + phase);
    st = vessel_radius * sin(t + phase);

    ring_pos_1 = [ct; zeros(1, n_rings / 2, "single", "gpuArray"); right_depth_1 + st];
    ring_pos_2 = [zeros(1, n_rings / 2, "single", "gpuArray"); ct; right_depth_2 + st];
    scatter_pos = cat(2, ring_pos_1, ring_pos_2, ...
        background_pos + [0; 0; single((i - 1) * 0.00001)]);

    tx = ffdas.greens(channel_pos, wavenums, tx_signal, scatter_pos);
    rx = ffdas.greens(scatter_pos, wavenums, tx .* scatter_values, channel_pos);

    rf(:, 1, :, i) = reshape(conj(ifft(rx, [], 1)), n_samples, 1, n_channels);
end


% reconstruct a 64^3 volume for each slow-time frame
nz = 64; ny = 64; nx = 64;
x = gpuArray(single(linspace(xmin, xmax, nx)));
y = gpuArray(single(linspace(ymin, ymax, ny)));
z = gpuArray(single(linspace(zmin, zmax, nz)));
[xx, yy, zz] = ndgrid(x, y, z);
voxel_pos = zeros(3, nx, ny, nz, "single", "gpuArray");
voxel_pos(1,:,:,:) = xx;
voxel_pos(2,:,:,:) = yy;
voxel_pos(3,:,:,:) = zz;

ks = sampling_freq / sound_speed;

offsets = squeeze(voxel_pos(3,:,:,:)) * ks;
offsets = reshape(offsets, nx, ny, nz, 1);
weights = ones(size(offsets), "single", "gpuArray");

wavenum = single(-2 * pi * center_freq / sampling_freq);

timer = ffdas.utils.Timer();
timer.start();
volume = ffdas.das( ...
    rf, channel_pos * ks, voxel_pos * ks, offsets, weights, [], wavenum);
timer.stop();
fprintf("das: %dx%dx%d grid, %d frames: %.1f ms\n", ...
    nz, ny, nx, n_frames, timer.elapsed_ms());

tissue_rank = 5;

timer2 = ffdas.utils.Timer();
timer2.start();
filtered = ffdas.truncate_rank(volume, tissue_rank + 1);
timer2.stop();
fprintf("truncate_rank: %.1f ms\n", timer2.elapsed_ms());


% xz and yz slices through the center of the volume
y_mid = ny / 2;
x_mid = nx / 2;

before_env = gather(sum(abs(volume), 4));
after_env  = gather(sum(abs(filtered), 4));

before_xz = squeeze(before_env(:, y_mid, :))';
before_yz = squeeze(before_env(x_mid, :, :))';
after_xz  = squeeze(after_env(:, y_mid, :))';
after_yz  = squeeze(after_env(x_mid, :, :))';

db_before_xz = 20 * log10(before_xz / max(before_xz(:)) + 1e-10);
db_before_yz = 20 * log10(before_yz / max(before_yz(:)) + 1e-10);
db_after_xz  = 20 * log10(after_xz / max(after_xz(:)) + 1e-10);
db_after_yz  = 20 * log10(after_yz / max(after_yz(:)) + 1e-10);

x_mm = gather(x) * 1e3;
y_mm = gather(y) * 1e3;
z_mm = gather(z) * 1e3;

figure;
tiledlayout(2, 2);

nexttile;
imagesc(x_mm, z_mm, db_before_xz);
colormap("gray"); clim([-32 0]); axis image;
xlabel("x [mm]"); ylabel("z [mm]");
title("before — xz");

nexttile;
imagesc(y_mm, z_mm, db_before_yz);
colormap("gray"); clim([-32 0]); axis image;
xlabel("y [mm]");
title("before — yz");

nexttile;
imagesc(x_mm, z_mm, db_after_xz);
colormap("hot"); clim([-32 0]); axis image;
xlabel("x [mm]"); ylabel("z [mm]");
title(sprintf("after truncate\\_rank (start=%d) — xz", tissue_rank + 1));

nexttile;
imagesc(y_mm, z_mm, db_after_yz);
colormap("hot"); clim([-32 0]); axis image;
xlabel("y [mm]");
title(sprintf("after truncate\\_rank (start=%d) — yz", tissue_rank + 1));

exportgraphics(gcf, "clutter_filter.png", Resolution=150);
