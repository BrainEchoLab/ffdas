% Interpolation from structured grids to arbitrary positions.
%
% A structured grid has a regular index-space lattice (nz, ny, nx) but
% its vertex positions can be arbitrary 3D coordinates. This enables
% reconstruction on curvilinear grids (e.g. spherical) followed by
% interpolation to Cartesian coordinates or any other query positions.
%
% This example places scatterers on a regular Cartesian grid, then
% reconstructs on a spherical grid where the pattern appears warped.
% After interpolation back to Cartesian coordinates, the grid is
% recovered.

rng(0);

sound_speed = 1540.0;
center_freq = 3.08e6;
sampling_freq = center_freq;
n_samples = 512;

% probe: 32x32 matrix array in the xy-plane at z=0
pitch = 250e-6;
el = gpuArray(single((0:31) - 15.5) * pitch);
[ex, ey] = ndgrid(el, el);
n_channels = 1024;
channel_pos = gpuArray(single([ex(:)'; ey(:)'; zeros(1, n_channels)]));

xmin = -0.016; xmax = 0.016;
ymin = -0.016; ymax = 0.016;
zmin = 0.006;  zmax = 0.038;

source = gpuArray(single([0; 0; -0.005]));

k_fft = gpuArray(single([0:n_samples/2-1, -n_samples/2:-1]'));
freqs = k_fft * sampling_freq / n_samples + center_freq;
wavenums = single(-2 * pi) * freqs / sound_speed;
sigma_f = 0.6 * center_freq / (2 * sqrt(2 * log(2)));
pulse = complex(exp(-0.5 * ((freqs - center_freq) / sigma_f).^2));

delay = ffdas.utils.cdist(source, channel_pos) / sound_speed;
tx_signal = pulse .* exp(-2j * pi * freqs .* delay);

% render "ff" text to a binary image for the scatterer phantom
fig = figure('Visible', 'off', 'Color', 'k');
set(fig, 'Units', 'pixels', 'Position', [100 100 64 96]);
ax = axes(fig, 'Units', 'normalized', 'Position', [0 0 1 1]);
set(ax, 'Color', 'k', 'XLim', [0 1], 'YLim', [0 1], 'Visible', 'off');
text(ax, 0.47, 0.5, 'ff', 'Color', 'w', 'FontSize', 48, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'Units', 'normalized');
drawnow;
frame = getframe(ax);
close(fig);
grid_values = gpuArray(single(mean(double(frame.cdata), 3) > 128));
img_h = size(grid_values, 1);
img_w = size(grid_values, 2);

n_scatterers = 16384;
scatter_pos_unit = rand(3, n_scatterers, "single", "gpuArray");

row = min(floor(scatter_pos_unit(3,:) * single(img_h - 1)) + 1, single(img_h));
col = min(floor(scatter_pos_unit(1,:) * single(img_w - 1)) + 1, single(img_w));
scatter_amp = grid_values(row + (col - 1) * img_h);
scatter_values = single(0.001) * (scatter_amp ...
    + randn(1, n_scatterers, "single", "gpuArray") * 0.1);

scatter_pos = gpuArray(single([xmin; ymin; zmin])) ...
    + scatter_pos_unit .* gpuArray(single([xmax - xmin; ymax - ymin; zmax - zmin]));

tx = ffdas.greens(channel_pos, wavenums, tx_signal, scatter_pos);
rx = ffdas.greens(scatter_pos, wavenums, tx .* scatter_values, channel_pos);
rf = conj(ifft(rx, [], 1));
rf = reshape(rf, n_samples, 1, []);


% spherical reconstruction grid
nr = 96; ntheta = 96; nphi = 96;
r     = gpuArray(single(linspace(zmin + 0.005, zmax, nr)));
theta = gpuArray(single(linspace(-0.5, 0.5, ntheta)));
phi   = gpuArray(single(linspace(-0.5, 0.5, nphi)));

% MATLAB convention: gridpos is (3, nphi, ntheta, nr)
[pp, tt, rr] = ndgrid(phi, theta, r);
gridpos = zeros(3, nphi, ntheta, nr, "single", "gpuArray");
gridpos(1,:,:,:) = rr .* sin(tt);
gridpos(2,:,:,:) = rr .* sin(pp);
gridpos(3,:,:,:) = rr .* cos(tt) .* cos(pp);

ks = sampling_freq / sound_speed;
d = ffdas.utils.cdist(source, gridpos);
offsets = reshape(d * ks, nphi, ntheta, nr, 1);
weights = ones(size(offsets), "single", "gpuArray");

spherical = ffdas.das( ...
    rf, channel_pos * ks, gridpos * ks, offsets, weights, ...
    [], single(-2 * pi * center_freq / sampling_freq));

envelope = abs(spherical);  % (nphi, ntheta, nr)


% interpolate to Cartesian coordinates
cart_nz = 128; cart_ny = 128; cart_nx = 128;
cx = gpuArray(single(linspace(xmin, xmax, cart_nx)));
cy = gpuArray(single(linspace(ymin, ymax, cart_ny)));
cz = gpuArray(single(linspace(zmin, zmax, cart_nz)));
[cxx, cyy, czz] = ndgrid(cx, cy, cz);
cart_points = zeros(3, cart_nx, cart_ny, cart_nz, "single", "gpuArray");
cart_points(1,:,:,:) = cxx;
cart_points(2,:,:,:) = cyy;
cart_points(3,:,:,:) = czz;

timer = ffdas.utils.Timer();
timer.start();
cart_volume = ffdas.interpolate(gridpos, envelope, cart_points, "linear", 0.0);
timer.stop();
fprintf("interpolate to %dx%dx%d: %.1f ms\n", ...
    cart_nz, cart_ny, cart_nx, timer.elapsed_ms());


% spherical volume: max projection over phi shows the Cartesian grid
% warped by the curvilinear coordinates. after interpolation, recovered
sph_mip = gather(squeeze(max(envelope, [], 1)))';
db_sph = 20 * log10(sph_mip / max(sph_mip(:)) + 1e-10);

cart_mip = gather(squeeze(max(cart_volume, [], 2)))';
db_cart = 20 * log10(cart_mip / max(cart_mip(:)) + 1e-10);

figure;
tiledlayout(1, 2);

nexttile;
imagesc(gather(theta), gather(r) * 1e3, db_sph);
clim([-40 0]); colormap("gray");
xlabel("\theta [rad]"); ylabel("r [mm]");
title("spherical grid (warped)");

nexttile;
imagesc(gather(cx) * 1e3, gather(cz) * 1e3, db_cart);
clim([-40 0]); colormap("gray"); axis image;
xlabel("x [mm]"); ylabel("z [mm]");
title("Cartesian (interpolated)");

exportgraphics(gcf, "interpolation.png", Resolution=150);
