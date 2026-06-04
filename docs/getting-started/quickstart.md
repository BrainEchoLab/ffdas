# Quickstart

This page walks through computing transmit parameters (delays and apodization weights) for a diverging transmit. These parameters are the inputs to `das` and `das_sparse`.

## Setup

Define a virtual source behind the aperture, the aperture dimensions, and a 3D target grid.

=== "Python"

    ```python
    import cupy as cp
    import ffdas

    source = cp.array([0.0, 0.0, -0.01], dtype=cp.float32)
    aperture_size = cp.array([0.02, 0.02], dtype=cp.float32)
    sound_speed = 1540.0
    f_number = 2.0

    x = cp.linspace(-0.015, 0.015, 32, dtype=cp.float32)
    y = cp.linspace(-0.015, 0.015, 32, dtype=cp.float32)
    z = cp.linspace(0.005, 0.04, 64, dtype=cp.float32)
    zz, yy, xx = cp.meshgrid(z, y, x, indexing="ij")
    targets = cp.stack([xx, yy, zz], axis=-1)  # (64, 32, 32, 3)
    ```

=== "MATLAB"

    ```matlab
    source = gpuArray(single([0; 0; -0.01]));
    aperture_size = single([0.02; 0.02]);
    sound_speed = 1540.0;
    f_number = 2.0;

    x = single(gpuArray.linspace(-0.015, 0.015, 32));
    y = single(gpuArray.linspace(-0.015, 0.015, 32));
    z = single(gpuArray.linspace(0.005, 0.04, 64));
    [xx, yy, zz] = meshgrid(x, y, z);
    targets = permute(cat(4, xx, yy, zz), [4 3 1 2]);  % (3, 64, 32, 32)
    ```

The source is placed 10 mm behind the aperture center at `z = 0`, producing a diverging wavefront. Coordinates are in meters throughout.

## Transmit Delay

The transmit delay to each target is the difference between the source-to-target distance and the source-to-aperture distance, divided by the speed of sound. `rect_dist` gives the shortest distance from the source to the aperture rectangle, which is the reference point where the wavefront first enters the medium.

=== "Python"

    ```python
    dist = ffdas.cdist(source, targets)                   # (64, 32, 32)
    ref = ffdas.rect_dist(source, aperture_size)           # scalar
    delay = (dist - ref) / sound_speed                       # seconds
    ```

=== "MATLAB"

    ```matlab
    dist = ffdas.utils.cdist(source, targets);              % (1, 64, 32, 32)
    ref = ffdas.utils.rect_dist(source, aperture_size);     % scalar
    delay = squeeze(dist - ref) / sound_speed;               % (64, 32, 32), seconds
    ```

To pass these delays to `das`, multiply by the sampling frequency to convert from seconds to samples.

## Angular Weight

The angular weight controls which targets are considered within the effective beam. Compute the off-axis angle of each target relative to the forward direction from the source, then apply a window function scaled by the f-number.

=== "Python"

    ```python
    direction = cp.array([0.0, 0.0, 1.0], dtype=cp.float32)
    theta = ffdas.angle(targets - source, direction)       # (64, 32, 32)

    ratio = f_number * cp.tan(theta)
    weight = cp.where(
        cp.abs(ratio) <= 0.5,
        0.5 + 0.5 * cp.cos(2 * cp.pi * ratio),
        0.0,
    )
    ```

=== "MATLAB"

    ```matlab
    direction = gpuArray(single([0; 0; 1]));
    theta = ffdas.utils.angle(targets - source, direction); % (64, 32, 32)

    ratio = f_number * tan(theta);
    weight = (abs(ratio) <= 0.5) .* (0.5 + 0.5 * cos(2 * pi * ratio));
    ```

The window used here is a Hamming window, but any function of `theta` works.

## Multiple Sources

For multiple transmit events, stack the source positions along a leading dimension. `cdist` produces a pairwise output with one entry per source–target pair.

=== "Python"

    ```python
    sources = cp.array([                                      # (3, 3)
        [ 0.000, 0.0, -0.01],
        [-0.005, 0.0, -0.01],
        [ 0.005, 0.0, -0.01],
    ], dtype=cp.float32)

    dist = ffdas.cdist(sources, targets)                    # (3, 64, 32, 32)
    ref = ffdas.rect_dist(sources, aperture_size)           # (3,)
    delay = (dist - ref[..., None, None, None]) / sound_speed # (3, 64, 32, 32)
    ```

=== "MATLAB"

    ```matlab
    sources = gpuArray(single([0, -0.005, 0.005; 0, 0, 0; -0.01, -0.01, -0.01]));  % (3, 3)

    dist = ffdas.utils.cdist(sources, targets);              % (3, 64, 32, 32)
    ref = ffdas.utils.rect_dist(sources, aperture_size);     % (3, 1)
    delay = (dist - ref) ./ sound_speed;                      % (3, 64, 32, 32)
    ```

The delay and weight arrays can then be passed directly to `das` or `das_sparse`.
