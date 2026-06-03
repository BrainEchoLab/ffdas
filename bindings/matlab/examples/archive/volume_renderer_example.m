imgsize = [64, 64, 128];

theta = linspace(-30, 30, imgsize(1)) * pi / 180;
phi = linspace(-30, 30, imgsize(2)) * pi / 180;
rho = linspace(5e-3, 50e-3, imgsize(3));

grid = sim.grids.SphericalGrid(theta, phi, rho);

extent = [min(grid.xyz, [], 1); max(grid.xyz, [], 1)];
itp_dims = int32([128, 128, 128]);
itp_spacing = single(1e3 * (extent(2, :) - extent(1, :)) ./ single(itp_dims));
itp_origin = single(1e3 * extent(1, :));

points = reshape(single(1e3 * grid.xyz'), [3, imgsize]);

data = single(randn(imgsize) .* linspace(0, 1, imgsize(2))) * 25 + 25;

% lut = single([0, 0, 0, 0; 50, 1, 1, 1]');

lut = single([linspace(0, 50, 64)' hot(64)]');
% data = single(imgaussfilt3(data, 1.0));

for i = 1:64
%     data = single(randn(128, 128, 128) * 25 + 25);
    data = circshift(data, 1, 3);
    tic();
    volume_renderer_mex(data, points, 'dims', itp_dims, 'spacing', itp_spacing, 'origin', itp_origin, 'lut', lut);
%     volume_renderer_mex(data, [], 'spacing', single([0.1, 0.1, 0.05]));
    toc();
    pause(0.1);
end

% volume_renderer_mex(data, points, 'dims', [128, 128, 128], 'spacing', [0.2, 0.2, 0.2], 'origin', [0, 0, 0]);
% volume_renderer_mex(data, [], 'dims', [128, 128, 128], 'spacing', [0.2, 0.2, 0.2], 'origin', [0, 0, 0]);
