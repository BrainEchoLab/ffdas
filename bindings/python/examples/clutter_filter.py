"""Clutter filtering with truncate_rank.

In high frame rate ultrasound sequences (Doppler, flow imaging), the first
singular components capture largely static tissue clutter. truncate_rank
removes these, isolating the weaker flow signal.

This example simulates a tissue background with stationary scatterers
(coherent across frames, low rank in slow time) and a flow component
consisting of two circular vessels in perpendicular planes whose
scatterers shift between frames. truncate_rank separates the two,
revealing the vessel geometry.
"""

import math

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

import ffdas


cp.random.seed(0)

sound_speed = 1540.0
center_freq = 3.08e6
sampling_freq = center_freq
n_samples = 512
n_frames = 64

# 32-by-32 matrix array
pitch = 500e-6
el = (cp.arange(32, dtype=cp.float32) - 15.5) * pitch
ex, ey = cp.meshgrid(el, el, indexing="ij")  # type: ignore
channel_pos = cp.stack(
    [ex.ravel(), ey.ravel(), cp.zeros(1024, dtype=cp.float32)], axis=-1
)
n_channels = channel_pos.shape[0]

xmin, xmax = -0.008, 0.008
ymin, ymax = -0.008, 0.008
zmin, zmax = 0.01, 0.026


# flow: two circular vessels in perpendicular planes.
# ring_xz lives in the xz plane and is visible as a circle in the xz slice;
# ring_yz lives in the yz plane and is visible as a circle in the yz slice
vessel_radius = 0.0025

n_rings = 128

t = cp.linspace(0, 2 * math.pi, n_rings // 2, endpoint=False, dtype=cp.float32)

right_depth_1 = zmin + 0.25 * (zmax - zmin)
right_depth_2 = zmin + 0.75 * (zmax - zmin)

# tissue: 1024 stationary scatterers throughout the volume
n_scatterers = 16384
background_pos = cp.random.uniform(
    [xmin, ymin, zmin],  # type: ignore
    [xmax, ymax, zmax],  # type: ignore
    size=(n_scatterers - n_rings, 3),
    dtype=cp.float32,  # type: ignore
)

scatter_values = cp.concat(
    [
        0.1 * cp.random.rand(n_rings, dtype=cp.float32),
        cp.random.rand(n_scatterers - n_rings, dtype=cp.float32),
    ],
    axis=0,
)

freqs = cp.fft.fftfreq(n_samples, d=1.0 / sampling_freq).astype(cp.float32) + center_freq
wavenums = (-2 * math.pi * freqs / sound_speed).astype(cp.float32)
sigma_f = 0.6 * center_freq / (2 * math.sqrt(2 * math.log(2)))
pulse = cp.exp(-0.5 * ((freqs - center_freq) / sigma_f) ** 2).astype(cp.complex64)

channel_delay = cp.zeros(channel_pos.shape[0], dtype=cp.float32)
tx_signal = pulse * cp.exp(-2j * cp.pi * freqs * channel_delay[:, None])

# each frame has a different flow scatterer configuration: positions
# are shifted along the vessel by a fraction of the period
rot_per_frame = 0.1 * 2 * cp.pi / n_frames
rf = cp.empty((n_frames, n_channels, 1, n_samples), dtype=cp.complex64)
for i in range(n_frames):
    phase = i * rot_per_frame
    ct = vessel_radius * cp.cos(t + phase)  # type: ignore
    st = vessel_radius * cp.sin(t + phase)  # type: ignore

    ring_pos_1 = cp.stack([ct, cp.zeros_like(ct), right_depth_1 + st], axis=-1)
    ring_pos_2 = cp.stack([cp.zeros_like(ct), ct, right_depth_2 + st], axis=-1)
    scatter_pos = cp.concat([
        ring_pos_1, 
        ring_pos_2, 
        background_pos + cp.array([0.0, 0.0, i * 0.00001], dtype=cp.float32)
    ], axis=0)

    tx = ffdas.greens(channel_pos, wavenums, tx_signal, scatter_pos)
    rx = ffdas.greens(scatter_pos, wavenums, tx * scatter_values[:, None], channel_pos)

    rf[i, :, 0] = cp.fft.ifft(rx, axis=-1).conj()

# rf: (n_frames, channels, 1 transmit, samples)

# reconstruct a 64^3 volume for each slow-time frame
nz, ny, nx = 64, 64, 64
x = cp.linspace(xmin, xmax, nx, dtype=cp.float32)
y = cp.linspace(ymin, ymax, ny, dtype=cp.float32)
z = cp.linspace(zmin, zmax, nz, dtype=cp.float32)
zz, yy, xx = cp.meshgrid(z, y, x, indexing="ij")  # type: ignore
voxel_pos = cp.stack([xx, yy, zz], axis=-1)

ks = sampling_freq / sound_speed

# offsets and weights as in `reconstruct.py`
offsets = voxel_pos[None, ..., 2] * ks
weights = cp.ones_like(offsets)

with ffdas.utils.Timer() as t:
    volume = ffdas.das(
        rf,
        channel_pos * ks,
        voxel_pos * ks,
        offsets,
        weights,
        wavenum=-2 * math.pi * center_freq / sampling_freq,
    )
print(f"das: {nz}x{ny}x{nx} grid, {n_frames} frames: {t.elapsed_ms():.1f} ms")

# truncate_rank decomposes along the first axis (frames) and
# reconstructs using singular vectors start through stop.
# tissue is rank 1, so start=1 removes it entirely
tissue_rank = 5

with ffdas.utils.Timer() as t:
    filtered = ffdas.truncate_rank(volume, start=tissue_rank)
print(f"truncate_rank: {t.elapsed_ms():.1f} ms")


# xz and yz slices through the center of the volume
y_mid = ny // 2
x_mid = nx // 2

before_env = cp.abs(volume).sum(axis=0)  # (nz, ny, nx)
after_env = cp.abs(filtered).sum(axis=0)

before_xz = cp.asnumpy(before_env[:, y_mid, :])
before_yz = cp.asnumpy(before_env[:, :, x_mid])
after_xz = cp.asnumpy(after_env[:, y_mid, :])
after_yz = cp.asnumpy(after_env[:, :, x_mid])

db_before_xz = 20 * np.log10(before_xz / before_xz.max() + 1e-10)
db_before_yz = 20 * np.log10(before_yz / before_yz.max() + 1e-10)

db_after_xz = 20 * np.log10(after_xz / after_xz.max() + 1e-10)
db_after_yz = 20 * np.log10(after_yz / after_yz.max() + 1e-10)

extent_xz = [float(x[0]) * 1e3, float(x[-1]) * 1e3,
             float(z[-1]) * 1e3, float(z[0]) * 1e3]
extent_yz = [float(y[0]) * 1e3, float(y[-1]) * 1e3,
             float(z[-1]) * 1e3, float(z[0]) * 1e3]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(db_before_xz, extent=extent_xz, cmap="gray", vmin=-32,
                  vmax=0, aspect="equal")
axes[0, 0].set_xlabel("x [mm]")
axes[0, 0].set_ylabel("z [mm]")
axes[0, 0].set_title("before — xz")

axes[0, 1].imshow(db_before_yz, extent=extent_yz, cmap="gray", vmin=-32,
                  vmax=0, aspect="equal")
axes[0, 1].set_xlabel("y [mm]")
axes[0, 1].set_title("before — yz")

axes[1, 0].imshow(db_after_xz, extent=extent_xz, cmap="hot", vmin=-32,
                  vmax=0, aspect="equal")
axes[1, 0].set_xlabel("x [mm]")
axes[1, 0].set_ylabel("z [mm]")
axes[1, 0].set_title(f"after truncate_rank (start={tissue_rank}) — xz")

axes[1, 1].imshow(db_after_yz, extent=extent_yz, cmap="hot", vmin=-32,
                  vmax=0, aspect="equal")
axes[1, 1].set_xlabel("y [mm]")
axes[1, 1].set_title(f"after truncate_rank (start={tissue_rank}) — yz")

plt.tight_layout()
plt.savefig("clutter_filter.png", dpi=150)
