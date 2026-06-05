% Reconstruct a 3D ultrasound volume from simulated plane-wave data.
%
% Given channel data, element positions, a reconstruction grid, and
% transmit parameters, ffdas.das produces a volumetric image. The
% simulation section below generates synthetic data using a trefoil
% knot phantom; skip it if you have your own channel data.

rng(0);

sound_speed = 1540.0;
center_freq = 3.08e6;
sampling_freq = center_freq;
n_samples = 512;

pitch = 500e-6;
el = gpuArray(single((0:31) - 15.5) * pitch);
[ex, ey] = ndgrid(el, el);
n_channels = 1024;
channel_pos = gpuArray(single([ex(:)'; ey(:)'; zeros(1, n_channels)]));

xmin = -0.008; xmax = 0.008;
ymin = -0.008; ymax = 0.008;
zmin = 0.010;  zmax = 0.026;


% simulate rf data for a normal-incidence plane wave.
% see simulation.py for a walkthrough of greens propagation

batch_size = 128;
n_scatterers = 16384;
n_knot = 8192;

t = 2 * pi * rand(1, n_knot, "single", "gpuArray");
knot_pos = trefoil(t) + randn(3, n_knot, "single", "gpuArray") * 0.00015;
background_pos = [ ...
    xmin + (xmax - xmin) * rand(1, n_scatterers - n_knot, "single", "gpuArray"); ...
    ymin + (ymax - ymin) * rand(1, n_scatterers - n_knot, "single", "gpuArray"); ...
    zmin + (zmax - zmin) * rand(1, n_scatterers - n_knot, "single", "gpuArray")];

scatter_pos = cat(2, knot_pos, background_pos);
scatter_values = cat(2, ...
    rand(1, n_knot, batch_size, "single", "gpuArray"), ...
    rand(1, n_scatterers - n_knot, batch_size, "single", "gpuArray"));

k_fft = gpuArray(single([0:n_samples/2-1, -n_samples/2:-1]'));
freqs = k_fft * sampling_freq / n_samples + center_freq;
wavenums = single(-2 * pi) * freqs / sound_speed;
sigma_f = 0.6 * center_freq / (2 * sqrt(2 * log(2)));
pulse = complex(exp(-0.5 * ((freqs - center_freq) / sigma_f).^2));

% plane wave: all channels transmit simultaneously (zero delay)
channel_delay = zeros(1, n_channels, "single", "gpuArray");
transmission = pulse .* exp(-2j * pi * freqs .* channel_delay);

tx = ffdas.greens(channel_pos, wavenums, transmission, scatter_pos);
rx = ffdas.greens(scatter_pos, wavenums, tx .* scatter_values, channel_pos);
rf = conj(ifft(rx, [], 1));

% (samples, channels, batch) -> (samples, 1, channels, batch)
rf = reshape(rf, n_samples, 1, [], batch_size);


% reconstruction grid: 64^3 voxels centered on the phantom
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

% transmit offset: one-way plane-wave delay from z=0 to each voxel
% (nx, ny, nz, 1) — trailing dimension is the sequence axis
offsets = squeeze(voxel_pos(3,:,:,:)) * ks;
offsets = reshape(offsets, nx, ny, nz, 1);
weights = ones(size(offsets), "single", "gpuArray");

wavenum = single(-2 * pi * center_freq / sampling_freq);

timer = ffdas.utils.Timer();
timer.start();
image = ffdas.das( ...
    rf, channel_pos * ks, voxel_pos * ks, offsets, weights, [], wavenum);
timer.stop();
fprintf("das (srcdir=[]): %dx%dx%d, %d ch, batch %d: %.1f ms\n", ...
    nz, ny, nx, n_channels, batch_size, timer.elapsed_ms());

% element directivity: (4, channels), rows 1-3 are the normal, row 4
% is the cosine of the sensitivity half-angle
srcdir = zeros(4, n_channels, "single", "gpuArray");
srcdir(3,:) = 1.0;
srcdir(4,:) = 0.707;  % ~45 degree half-angle

timer2 = ffdas.utils.Timer();
timer2.start();
image = ffdas.das( ...
    rf, channel_pos * ks, voxel_pos * ks, offsets, weights, srcdir, wavenum);
timer2.stop();
fprintf("das (with srcdir): %dx%dx%d, %d ch, batch %d: %.1f ms\n", ...
    nz, ny, nx, n_channels, batch_size, timer2.elapsed_ms());


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
        (sin(t) + 2 * sin(2*t)) * 0.0016, ...
        -sin(3*t) * 0.0024, ...
        (cos(t) - 2 * cos(2*t)) * 0.0016 + 0.018);
end
