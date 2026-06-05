% Coherent plane-wave compounding.
%
% Plane-wave compounding transmits at several steering angles and
% combines the resulting images to improve lateral resolution and
% reduce artifacts. ffdas.das handles this as follows: the
% sequence dimension of the rf data holds per-angle channel data,
% and the matching offsets array provides the per-angle transmit
% delay at each voxel. DAS sums over the sequence dimension,
% producing a coherently compounded image.

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

% 2D angle grid: 3 angles along x times 3 along y = 9 compound angles
angle_range = gpuArray(single(linspace(-10, 10, 3))) * (pi / 180);
[angle_x, angle_y] = ndgrid(angle_range, angle_range);
angle_x = angle_x(:);
angle_y = angle_y(:);
n_angles = numel(angle_x);


% simulate rf data for each steering angle.
% a plane wave at angle (theta_x, theta_y) steers by delaying element i
% by (sin(theta_x) * element_x + sin(theta_y) * element_y) / c

batch_size = 16;
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

rf = zeros(n_samples, n_angles, n_channels, batch_size, "single", "gpuArray");
for i = 1:n_angles
    delay = (sin(angle_x(i)) * channel_pos(1,:) ...
           + sin(angle_y(i)) * channel_pos(2,:)) / sound_speed;
    tx_signal = pulse .* exp(-2j * pi * freqs .* delay);
    tx = ffdas.greens(channel_pos, wavenums, tx_signal, scatter_pos);
    rx = ffdas.greens(scatter_pos, wavenums, tx .* scatter_values, channel_pos);
    rf_angle = conj(ifft(rx, [], 1));
    rf(:, i, :, :) = reshape(rf_angle, n_samples, 1, [], batch_size);
end


% reconstruction grid
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

% per-angle transmit offsets: plane-wave delay at (theta_x, theta_y)
sin_ax = reshape(sin(angle_x), 1, 1, 1, []);
sin_ay = reshape(sin(angle_y), 1, 1, 1, []);
cos_ax = reshape(cos(angle_x), 1, 1, 1, []);
cos_ay = reshape(cos(angle_y), 1, 1, 1, []);
offsets = (xx .* sin_ax + yy .* sin_ay + zz .* cos_ax .* cos_ay) * ks;
weights = ones(size(offsets), "single", "gpuArray");

wavenum = single(-2 * pi * center_freq / sampling_freq);

% compounded: das sums over the sequence dimension (all angles at once)
timer = ffdas.utils.Timer();
timer.start();
compounded = ffdas.das( ...
    rf, channel_pos * ks, voxel_pos * ks, offsets, weights, [], wavenum);
timer.stop();
fprintf("compounded (%d angles): %.1f ms\n", n_angles, timer.elapsed_ms());

% single angle at normal incidence for comparison
mid = ceil(n_angles / 2);
timer2 = ffdas.utils.Timer();
timer2.start();
single_img = ffdas.das( ...
    rf(:, mid:mid, :, :), ...
    channel_pos * ks, voxel_pos * ks, ...
    offsets(:,:,:,mid:mid), weights(:,:,:,mid:mid), ...
    [], wavenum);
timer2.stop();
fprintf("single angle: %.1f ms\n", timer2.elapsed_ms());


mag_single = abs(single_img(:,:,:,1));
mag_compounded = abs(compounded(:,:,:,1));

x_mm = gather(x) * 1e3;
y_mm = gather(y) * 1e3;
z_mm = gather(z) * 1e3;

figure;
tiledlayout(2, 2);

mip = squeeze(max(mag_single, [], 2));
db = 20 * log10(mip / max(mip(:)) + 1e-10);
nexttile;
imagesc(x_mm, z_mm, gather(db)');
colormap("gray"); clim([-24 0]); axis image;
xlabel("x [mm]"); ylabel("z [mm]");
title("single angle — xz");

mip = squeeze(max(mag_single, [], 1));
db = 20 * log10(mip / max(mip(:)) + 1e-10);
nexttile;
imagesc(y_mm, z_mm, gather(db)');
colormap("gray"); clim([-24 0]); axis image;
xlabel("y [mm]");
title("single angle — yz");

mip = squeeze(max(mag_compounded, [], 2));
db = 20 * log10(mip / max(mip(:)) + 1e-10);
nexttile;
imagesc(x_mm, z_mm, gather(db)');
colormap("gray"); clim([-24 0]); axis image;
xlabel("x [mm]"); ylabel("z [mm]");
title(sprintf("compounded (%d angles) — xz", n_angles));

mip = squeeze(max(mag_compounded, [], 1));
db = 20 * log10(mip / max(mip(:)) + 1e-10);
nexttile;
imagesc(y_mm, z_mm, gather(db)');
colormap("gray"); clim([-24 0]); axis image;
xlabel("y [mm]");
title(sprintf("compounded (%d angles) — yz", n_angles));

exportgraphics(gcf, "compound.png", Resolution=150);


function pos = trefoil(t)
    pos = cat(1, ...
        (sin(t) + 2 * sin(2*t)) * 0.0016, ...
        -sin(3*t) * 0.0024, ...
        (cos(t) - 2 * cos(2*t)) * 0.0016 + 0.018);
end
