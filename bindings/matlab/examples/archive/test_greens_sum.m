addpath("bin")
parallel.gpu.enableCUDAForwardCompatibility(true);

imgsize = [64, 64, 64];
array_size = [64, 48];
sound_speed = 1540;
center_frequency = 3e6;

%% set up input grids
% create focus points using spherical grid
focus = ffdas.util.spherical_grid_fromrange(...
    [-30, 30], ...  % theta range  
    [-30, 30], ...  % phi range
    20e-3, ...  % rho
    [4, 2, 1], ...  % grid size
    true  ...  % degrees = true
);

focus = reshape(focus, [3, 8]);  % (3, batch)
focus(3, :) = -focus(3, :);  % flip z coord for diverging wave

channel_pos = ffdas.util.cartesian_grid_fromrange( ...
    [-5e-3, 5e-3], ...  % x range
    [-5e-3, 5e-3], ...  % y range
    0, ...  % z range
    [array_size, 1] ...
);
channel_pos = reshape(channel_pos, 3, []);  % (3, num_channels)

% note: for points close to the aperture, this method is not very accurate
target_pos = ffdas.util.cartesian_grid_fromrange( ...
    [-20e-3, 20e-3], ...  % x range
    [-20e-3, 20e-3], ...  % y range
    [0, 40e-3], ...  % z range
    imgsize ...
);  % (3, nx, ny, nz)
target_pos = reshape(target_pos, 3, []);  % (3, num_targets)

%% compute impulse response for channels (from focused/divergin delays)
focus = reshape(focus, [3, 1, 8]);  % (3, 1, batch)
channel_pos = reshape(channel_pos, [3, size(channel_pos, 2), 1]);  % (3, num_channels, 1)

% compute distances and delays
delays = sqrt(sum((focus - channel_pos).^2, 1));  % (1, num_channels, batch)
delays = delays - min(delays(:));

% create frequency and wave number arrays
frequencies = gpuArray(single(linspace(0, 6e6, 128)));  % (num_samples, 1, 1)
wave_numbers = -2 * pi * frequencies / sound_speed;

% add simple gaussian envelope and compute complex phase
envelope = exp(-(frequencies - center_frequency).^2 / (4 * (0.25e6)^2));  % (1, num_samples)
inputs = envelope' .* exp(1i * wave_numbers' .* delays); % (num_samples, num_channels, batch)

%% compute the propagated impulse response from channel_pos to target_pos
% note: the output is permuted to shape (batch, num_targets, num_samples)
% note: currently, the batch size (e.g., focus points) must be exactly 16
fprintf('running green''s sum...\n');
tic;
result = ffdas.greens(channel_pos, wave_numbers, inputs, target_pos);
elapsed = toc;
fprintf('done. (%.4f s)\n', elapsed);

disp(size(result))

% compute the power by summing over all frequencies
power = mean(abs(result).^2, 1);  % (batch, num_targets)
disp(size(power))

%% display the first 8 outputs
figure();

for i = 1:4
    % extract volume for focus point i
    vol = reshape(power(:, :, i), imgsize);
    
    % Create maximum intensity projections
    subplot(3, 4, i);
    imagesc(squeeze(max(vol, [], 1))');
    if i == 1, ylabel('Max X'); end
    if i <= 8, title(sprintf('Focus %d', i)); end

    subplot(3, 4, i + 4);
    imagesc(squeeze(max(vol, [], 2))');
    if i == 1, ylabel('Max Y'); end
    
    subplot(3, 4, i + 8);
    imagesc(squeeze(max(vol, [], 3))');
    if i == 1, ylabel('Max Z'); end
end

% Save figure
output_file = 'greens.png';
saveas(gcf, output_file);
fprintf('Saved visualization to %s\n', output_file);
