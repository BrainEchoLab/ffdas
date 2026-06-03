addpath('../bin');

grid_size = [64, 64, 64];
grid = ffdas.util.spherical_grid_fromrange(...
    [-30, 30], ...  % theta range
    [-30, 30], ...  % phi range  
    [10e-3, 50e-3], ...  % rho range
    grid_size, ...
    true ...  % degrees = true
);

query_size = [128, 128, 128];
query_points = ffdas.util.cartesian_grid_fromrange([-25e-3, 25e-3], [-25e-3, 25e-3], [0,50e-3], query_size);

values = gpuArray.rand(grid_size, 'single');

% class-based: fast to interpolate many times with the same geometry
interp = ffdas.StructuredInterpolator(grid, query_points, 'linear');
out1 = interp(values);

% function: easy to use for single interpolations
out2 = ffdas.interpolate(grid, values, query_points, 'linear');

subplot(2, 3, 1);
imagesc(squeeze(max(out1, [], 1))');
colormap gray;

subplot(2, 3, 2);
imagesc(squeeze(max(out1, [], 2))');
colormap gray;

subplot(2, 3, 3);
imagesc(squeeze(max(out1, [], 3))');
colormap gray;

subplot(2, 3, 4);
imagesc(squeeze(max(out2, [], 1))');
colormap gray;

subplot(2, 3, 5);
imagesc(squeeze(max(out2, [], 2))');
colormap gray;

subplot(2, 3, 6);
imagesc(squeeze(max(out2, [], 3))');
colormap gray;

output_file = 'interpolation_output.png';
saveas(gcf, output_file);
fprintf('Saved image to %s\n', output_file);
