% Reordering raw channel data with gather.
%
% Acquisition systems often deliver IQ data as int16, with the real and
% imaginary parts interleaved along the sample dimension and the channels
% in hardware order. One gather call maps the channels to element order,
% permutes the dimensions into the layout das expects and converts to
% complex single, in a single pass over the data.

rng(0);

n_samples = 256;
n_beams = 8;
n_repeats = 4;
n_channels = 128;

% raw data as delivered: [2 * n_samples, n_beams, n_repeats, n_channels],
% with (real, imag) pairs interleaved along the first dimension
raw = gpuArray(int16(randi([-2000, 2000], 2 * n_samples, n_beams, n_repeats, n_channels)));

% hardware channel order -> element order
channel_map = gpuArray(int32(randperm(n_channels)));

data = ffdas.gather(raw, channel_map, 4, [1 2 4 3], 'single', ...
                    'interleaved_complex', true);
% data: [n_samples, n_beams, n_channels, n_repeats], complex single

% the same in plain MATLAB, which makes three copies of the data
ref = complex(single(raw(1:2:end, :, :, :)), single(raw(2:2:end, :, :, :)));
ref = permute(ref(:, :, :, channel_map), [1 2 4 3]);

assert(isequal(size(data), [n_samples, n_beams, n_channels, n_repeats]), 'unexpected size');
assert(~isreal(data) && underlyingType(data) == "single", 'expected complex single output');
assert(isequal(gather(data), gather(ref)), 'result differs from the plain MATLAB reference');
fprintf('data: %s complex %s, identical to the plain MATLAB reference\n', mat2str(size(data)), underlyingType(data));

% gather without the option: plain indexing, with and without permutation and type
x = gpuArray(single(randn(6, 5, 4)));
idx = gpuArray(int32([4 1 3]));
assert(isequal(gather(ffdas.gather(x, idx, 2)), gather(x(:, idx, :))));
assert(isequal(gather(ffdas.gather(x, idx, 2, [3 1 2])), gather(permute(x(:, idx, :), [3 1 2]))));
assert(isequal(gather(ffdas.gather(int16(x * 100), idx, 2, [], 'single')), gather(single(int16(x(:, idx, :) * 100)))));
