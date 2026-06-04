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

pitch = 300e-6;
el = gpuArray(single((0:31) - 15.5) * pitch);
[ex, ey] = ndgrid(el, el);
n_channels = 1024;
channel_pos = gpuArray(single([ex(:)'; ey(:)'; zeros(1, n_channels)]));

sound_speed = 1540.0;
center_freq = 3e6;
sampling_freq = center_freq;
n_samples = 512;
n_frames = 64;

vessel_radius = 0.004;
n_flow = 64;
t_ring1 = gpuArray(single((0:n_flow-1) * 2 * pi / n_flow));
t_ring2 = gpuArray(single((0:n_flow-1) * 2 * pi / n_flow));
noise1 = randn(3, n_flow, "single", "gpuArray") * 0.0003;
noise2 = randn(3, n_flow, "single", "gpuArray") * 0.0003;

n_tissue = 16384;
tissue_pos = zeros(3, n_tissue, "single", "gpuArray");
tissue_pos(1,:) = -0.015 + 0.030 * rand(1, n_tissue, "single", "gpuArray");
tissue_pos(2,:) = -0.015 + 0.030 * rand(1, n_tissue, "single", "gpuArray");
tissue_pos(3,:) =  0.005 + 0.030 * rand(1, n_tissue, "single", "gpuArray");


% simulate channel data
source = gpuArray(single([0; 0; -0.005]));

k_fft = gpuArray(single([0:n_samples/2-1, -n_samples/2:-1]'));
freqs = k_fft * sampling_freq / n_samples + center_freq;
wavenums = single(-2 * pi) * freqs / sound_speed;
sigma_f = 0.6 * center_freq / (2 * sqrt(2 * log(2)));
pulse = complex(exp(-0.5 * ((freqs - center_freq) / sigma_f).^2));

tissue_amp = single(10.0) + complex( ...
    randn(1, n_tissue, "single", "gpuArray"), ...
    randn(1, n_tissue, "single", "gpuArray"));

tx_tissue = ffdas.greens(source, wavenums, pulse, tissue_pos);
rx_tissue = ffdas.greens(tissue_pos, wavenums, tx_tissue .* tissue_amp, channel_pos);

shift_per_frame = 0.1 * 2 * pi / n_frames;
rf = complex(zeros(n_samples, 1, n_channels, n_frames, "single", "gpuArray"));

for i = 1:n_frames
    shift = (i - 1) * shift_per_frame;
    flow_pos = cat(2, ...
        ring_xz(t_ring1 + shift, vessel_radius) + noise1, ...
        ring_yz(t_ring2 + shift, vessel_radius) + noise2);
    tx_flow = ffdas.greens(source, wavenums, pulse, flow_pos);
    rx_flow = ffdas.greens(flow_pos, wavenums, tx_flow, channel_pos);
    rx_total = rx_tissue + rx_flow;
    noise = complex(randn(size(rx_total), "like", rx_total), ...
                    randn(size(rx_total), "like", rx_total));
    rx_total = rx_total + single(0.05) * noise;
    rf_frame = conj(ifft(rx_total, [], 1));
    rf(:, 1, :, i) = reshape(rf_frame, n_samples, 1, n_channels);
end


% reconstruct a 64^3 volume for each slow-time frame
nz = 64; ny = 64; nx = 64;
x = gpuArray(single(linspace(-0.010, 0.010, nx)));
y = gpuArray(single(linspace(-0.010, 0.010, ny)));
z = gpuArray(single(linspace(0.010, 0.030, nz)));
[xx, yy, zz] = ndgrid(x, y, z);
voxel_pos = zeros(3, nx, ny, nz, "single", "gpuArray");
voxel_pos(1,:,:,:) = xx;
voxel_pos(2,:,:,:) = yy;
voxel_pos(3,:,:,:) = zz;

k = sampling_freq / sound_speed;
d = ffdas.utils.cdist(source, voxel_pos);
offsets = reshape(d * k, nx, ny, nz, 1);
weights = ones(size(offsets), "single", "gpuArray");

wavenum = single(-2 * pi * center_freq / sampling_freq);

timer = ffdas.utils.Timer();
timer.start();
volume = ffdas.das( ...
    rf, channel_pos * k, voxel_pos * k, offsets, weights, [], wavenum);
timer.stop();
fprintf("das: %dx%dx%d grid, %d frames: %.1f ms\n", ...
    nz, ny, nx, n_frames, timer.elapsed_ms());

tissue_rank = 1;

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


function pos = ring_xz(t, r)
    pos = cat(1, r * cos(t), zeros(size(t), "like", t), 0.015 + r * sin(t));
end

function pos = ring_yz(t, r)
    pos = cat(1, zeros(size(t), "like", t), r * cos(t), 0.025 + r * sin(t));
end
