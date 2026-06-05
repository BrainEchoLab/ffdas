% Reconstruct a 3D ultrasound volume from simulated plane-wave data.
%
% Given channel data, element positions, a reconstruction grid, and
% transmit parameters, ffdas.das produces a volumetric image. The
% simulation section below generates synthetic data using a trefoil
% knot phantom; skip it if you have your own channel data.

rng(0);

sound_speed = 1540.0;
center_freq = 3e6;
sampling_freq = center_freq;
n_samples = 512;

pitch = 300e-6;
el = gpuArray(single((0:31) - 15.5) * pitch);
[ex, ey] = ndgrid(el, el);
channel_pos = gpuArray(single([ex(:)'; ey(:)'; zeros(1, 1024)]));


% simulate rf data for a normal-incidence plane wave

batch_size = 128;
n_scatterers = 4096;
t = 2 * pi * rand(1, n_scatterers, "single", "gpuArray");
scatter_pos = trefoil(t) + randn(3, n_scatterers, "single", "gpuArray") * 0.00015;

k_fft = gpuArray(single([0:n_samples/2-1, -n_samples/2:-1]'));
freqs = k_fft * sampling_freq / n_samples + center_freq;
wavenums = single(-2 * pi) * freqs / sound_speed;
sigma_f = 0.6 * center_freq / (2 * sqrt(2 * log(2)));
pulse = complex(exp(-0.5 * ((freqs - center_freq) / sigma_f).^2));

% plane wave: all channels transmit simultaneously (zero delay)
channel_delay = zeros(1, 1024, "single", "gpuArray");
transmission = pulse .* exp(-2j * pi * freqs .* channel_delay);
scatter_values = rand(1, n_scatterers, batch_size, "single", "gpuArray");

tx = ffdas.greens(channel_pos, wavenums, transmission, scatter_pos);
rx = ffdas.greens(scatter_pos, wavenums, tx .* scatter_values, channel_pos);
rf = conj(ifft(rx, [], 1));

% (samples, channels, batch) -> (samples, 1, channels, batch)
rf = reshape(rf, n_samples, 1, [], batch_size);


% reconstruction grid: 64^3 voxels centered on the phantom
nz = 64; ny = 64; nx = 64;
x = gpuArray(single(linspace(-0.007, 0.007, nx)));
y = gpuArray(single(linspace(-0.007, 0.007, ny)));
z = gpuArray(single(linspace(0.003, 0.017, nz)));
[xx, yy, zz] = ndgrid(x, y, z);
voxel_pos = zeros(3, nx, ny, nz, "single", "gpuArray");
voxel_pos(1,:,:,:) = xx;
voxel_pos(2,:,:,:) = yy;
voxel_pos(3,:,:,:) = zz;

ks = sampling_freq / sound_speed;

% transmit offset: one-way plane-wave delay from z=0 to each voxel
% (nx, ny, nz, 1) — trailing dimension is the sequence axis
offsets = squeeze(voxel_pos(3,:,:,:)) * ks;
offsets = reshape(offsets, nx, ny, nz, 1);
weights = ones(size(offsets), "single", "gpuArray");

% element directivity: (4, channels), rows 1-3 are the normal, row 4
% is the cosine of the sensitivity half-angle
srcdir = zeros(4, 1024, "single", "gpuArray");
srcdir(3,:) = 1.0;
srcdir(4,:) = 0.5;

wavenum = single(-2 * pi * center_freq / sampling_freq);

timer = ffdas.utils.Timer();
timer.start();
image = ffdas.das( ...
    rf, channel_pos * ks, voxel_pos * ks, offsets, weights, srcdir, wavenum);
timer.stop();
fprintf("das: %dx%dx%d, %d ch, batch %d: %.1f ms\n", ...
    nz, ny, nx, 1024, batch_size, timer.elapsed_ms());


magnitude = abs(image(:,:,:,1));

mip_xz = squeeze(max(magnitude, [], 2));
db_xz = 20 * log10(mip_xz / max(mip_xz(:)) + 1e-10);

mip_yz = squeeze(max(magnitude, [], 1));
db_yz = 20 * log10(mip_yz / max(mip_yz(:)) + 1e-10);

figure;
tiledlayout(1, 2);

nexttile;
imagesc(gather(x) * 1e3, gather(z) * 1e3, gather(db_xz)');
colormap("gray"); clim([-24 0]);
xlabel("x [mm]"); ylabel("z [mm]");
title("xz max projection");
axis image;

nexttile;
imagesc(gather(y) * 1e3, gather(z) * 1e3, gather(db_yz)');
colormap("gray"); clim([-24 0]);
xlabel("y [mm]");
title("yz max projection");
axis image;

exportgraphics(gcf, "reconstruct.png", Resolution=150);


function pos = trefoil(t)
    pos = cat(1, ...
        (sin(t) + 2 * sin(2*t)) * 0.0015, ...
        -sin(3*t) * 0.002, ...
        (cos(t) - 2 * cos(2*t)) * 0.0015 + 0.01);
end
