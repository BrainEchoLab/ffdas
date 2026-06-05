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


% spherical reconstruction grid
nr = 96; ntheta = 64; nphi = 64;
r     = gpuArray(single(linspace(0.005, 0.035, nr)));
theta = gpuArray(single(linspace(-0.4, 0.4, ntheta)));
phi   = gpuArray(single(linspace(-0.4, 0.4, nphi)));

% MATLAB convention: gridpos is (3, nx, ny, nz) = (3, nphi, ntheta, nr)
[pp, tt, rr] = ndgrid(phi, theta, r);
gridpos = zeros(3, nphi, ntheta, nr, "single", "gpuArray");
gridpos(1,:,:,:) = rr .* sin(tt);
gridpos(2,:,:,:) = rr .* sin(pp);
gridpos(3,:,:,:) = rr .* cos(tt) .* cos(pp);


% simulate and reconstruct (see reconstruct.m for details)
pitch = 150e-6;
el = gpuArray(single((0:31) - 15.5) * pitch);
[ex, ey] = ndgrid(el, el);
n_channels = 1024;
channel_pos = gpuArray(single([ex(:)'; ey(:)'; zeros(1, n_channels)]));

sound_speed = 1540.0;
center_freq = 5e6;
sampling_freq = center_freq;
n_samples = 512;
batch_size = 32;

xs = gpuArray(single(linspace(-0.004, 0.004, 9)));
ys = gpuArray(single(linspace(-0.003, 0.003, 3)));
zs = gpuArray(single(linspace(0.010, 0.026, 9)));
[gxs, gys, gzs] = ndgrid(xs, ys, zs);
scatterers = gpuArray(single([gxs(:)'; gys(:)'; gzs(:)']));
n_scatterers = size(scatterers, 2);

source = gpuArray(single([0; 0; -0.005]));

k_fft = gpuArray(single([0:n_samples/2-1, -n_samples/2:-1]'));
freqs = k_fft * sampling_freq / n_samples + center_freq;
wavenums = single(-2 * pi) * freqs / sound_speed;
sigma_f = 0.6 * center_freq / (2 * sqrt(2 * log(2)));
pulse = complex(exp(-0.5 * ((freqs - center_freq) / sigma_f).^2));

phases = exp(2i * pi * rand(1, n_scatterers, batch_size, "single", "gpuArray"));

tx = ffdas.greens(source, wavenums, pulse, scatterers);
rx = ffdas.greens(scatterers, wavenums, tx .* phases, channel_pos);
rf = conj(ifft(rx, [], 1));
rf = reshape(rf, n_samples, 1, [], batch_size);

k = sampling_freq / sound_speed;
d = ffdas.utils.cdist(source, gridpos);
offsets = reshape(d * k, nphi, ntheta, nr, 1);
weights = ones(size(offsets), "single", "gpuArray");

volume = ffdas.das( ...
    rf, channel_pos * k, gridpos * k, offsets, weights, ...
    [], single(-2 * pi * center_freq / sampling_freq));

envelope = abs(volume);  % (nphi, ntheta, nr, batch)


% interpolate to Cartesian coordinates
cart_nz = 128; cart_ny = 64; cart_nx = 64;
cx = gpuArray(single(linspace(-0.010, 0.010, cart_nx)));
cy = gpuArray(single(linspace(-0.010, 0.010, cart_ny)));
cz = gpuArray(single(linspace(0.005, 0.035, cart_nz)));
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
sph_mip = gather(squeeze(max(envelope(:,:,:,1), [], 1)))';
db_sph = 20 * log10(sph_mip / max(sph_mip(:)) + 1e-10);

cart_mip = gather(squeeze(max(cart_volume(:,:,:,1), [], 2)))';
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
