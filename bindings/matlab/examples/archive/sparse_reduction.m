% load sample data and parameters
addpath('../bin');

data_dir = fullfile(...
    getenv('CLINICSHARE'), ...
    'Patients/SBD-04_29-01-2025/usx-recording-2025-01-29T114649/' ...
);

fprintf('Loading data from: %s\n', data_dir);
[params, data] = load_sample_data(data_dir, 2);  % load 4 chunks

% data is of shape (num_samples, num_obs, batch_size, num_channels, num_chunks)
num_samples = size(data, 1);

% create imaging grid using spherical coordinates
% points are more dense close to aperture, sparse further away
start_depth = 15e-3;
max_depth = num_samples/2 * params.sound_speed / (2 * params.sampling_frequency);
num_targets = [64, 64, 128];  % [theta, phi, rho]

target_pos = ffdas.util.spherical_grid_fromrange(...
    [-30, 30], ...  % theta range
    [-30, 30], ...  % phi range  
    [start_depth, max_depth], ...  % rho range
    num_targets, ...
    true ...  % degrees = true
);

aperture = max(params.channel_pos(1:2, :), [], 2) - min(params.channel_pos(1:2, :), [], 2);  % total aperture size

% Use zeros for delays and weights as requested (skip focused transmit functions)
delays = ffdas.util.focused_transmit_delay(target_pos, params.focus, aperture, params.sound_speed);
weights = ffdas.util.focused_transmit_weight(target_pos, params.focus, [0, 0, 0]', 2.7, @hamming);

% find the indices along 'axis' (4) corresponding to the k highest weight values
k = 4;
[~, sorted_indices] = sort(weights, 4, 'descend');
sparse_indices = ffdas.util.take(sorted_indices, k, 4);

% take the delay and weights along 'axis' (4) according to these indices
weights = ffdas.util.take_along_axis(weights, sparse_indices, 4);
delays = ffdas.util.take_along_axis(delays, sparse_indices, 4);
sparse_indices = int32(sparse_indices);

% apply channel mapping using gather and then permute so that we can collapse 
% the batch and chunk dimensions:
% note: a permute() from matlab would create an extra copy of the data, which 
% we want to avoid. `ffdas.gather()` accepts the optional `permutation` argument,
% which will simultaneously permute the data while gathering.
% note: similarly for the datatype, we use `ffdas.gather()` to simultaneously
% cast to single precision type
data = ffdas.gather(data, params.channel_map+1, 4, [1, 2, 4, 3, 5], 'complex64');

% collapse batch and chunk dims into batch to get: 
% (num_samples, num_obs, num_channels, batch)
data = reshape(data, size(data, 1), size(data, 2), size(data, 3), []);

% Apply delay and sum to the sample data
% Normalize positions and delays as expected by ffdas:
% 1. positions in wavelengths of sampling frequency
% 2. delays in samples
sample_wavenum = params.sampling_frequency / params.sound_speed;
phase_scale = -2 * pi * params.demod_frequency / params.sampling_frequency;

fprintf('beamforming %d frames...', size(data, 3));
tic;
beamformed = ffdas.das_sparse(...
    data, ...
    params.channel_pos * sample_wavenum, ...
    target_pos * sample_wavenum, ...
    delays * params.sampling_frequency, ...
    weights, ...
    (sparse_indices-1) * num_samples, ...  % note: the beamformer expects sparse offsets in samples and 0-based
    0, ...  % algorithm = DEFAULT
    phase_scale, ...
    1.0 ...  % incidence_power = 1.0
); 
delta = toc;
fprintf('done (%.4f s)\n', delta);

% remove stationary clutter using fast, approximate eigen-based filtering
filtered = ffdas.eigfilter(beamformed, 64); % remove first 64 singular values

% compute power over time
power_doppler = mean(abs(filtered), 4);  % mean over batch dimension

query_size = [256, 256, 256];
lo = min(target_pos, [], [2, 3, 4]);
hi = max(target_pos, [], [2, 3, 4]);

query_points = ffdas.util.cartesian_grid_fromrange([lo(1), hi(1)], [lo(2), hi(2)], [lo(3), hi(3)], query_size);
power_doppler_cartesian = ffdas.interpolate(target_pos, power_doppler, query_points, 'linear');

% log-compress for visualization
eps_val = 1e-9;
power_doppler = 20 * log10(power_doppler / max(power_doppler(:)) + eps_val);
power_doppler_cartesian = 20 * log10(power_doppler_cartesian / max(power_doppler_cartesian(:)) + eps_val);

% Copy to CPU memory for visualization
power_doppler = gather(power_doppler);
power_doppler_cartesian = gather(power_doppler_cartesian);

% Show maximum intensity projections along each axis
subplot(2, 3, 1);
imagesc(squeeze(max(power_doppler, [], 1))');
ylabel('Spherical'); colormap gray; caxis([-18, 0]); daspect([1, 1, 1]);

subplot(2, 3, 2);
imagesc(squeeze(max(power_doppler, [], 2))');
colormap gray; caxis([-18, 0]); daspect([1, 1, 1]);

subplot(2, 3, 3);
imagesc(squeeze(max(power_doppler, [], 3))');
colormap gray; caxis([-18, 0]); daspect([1, 1, 1]);

subplot(2, 3, 4);
imagesc(squeeze(max(power_doppler_cartesian, [], 1))');
ylabel('Cartesian'); colormap gray; caxis([-18, 0]); daspect([1, 1, 1]);

subplot(2, 3, 5);
imagesc(squeeze(max(power_doppler_cartesian, [], 2))');
colormap gray; caxis([-18, 0]); daspect([1, 1, 1]);

subplot(2, 3, 6);
imagesc(squeeze(max(power_doppler_cartesian, [], 3))');
colormap gray; caxis([-18, 0]); daspect([1, 1, 1]);

output_file = 'sparse_reduction_output.png';
saveas(gcf, output_file);
fprintf('Saved image to %s\n', output_file);
