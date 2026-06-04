"""Clutter filtering with truncate_rank.

In high frame rate ultrasound sequences (Doppler, flow imaging), the first
singular components capture largely static tissue clutter. truncate_rank
removes these, isolating the weaker flow signal.

This example simulates a tissue background with stationary scatterers
(coherent across frames, low rank in slow time) and a flow component
whose scatterers move along a trefoil knot between frames. truncate_rank
separates the two, revealing the knot shape.
"""

import math

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

import ffdas


cp.random.seed(0)

# 32-by-32 matrix array
pitch = 150e-6
el = (cp.arange(32, dtype=cp.float32) - 15.5) * pitch
ex, ey = cp.meshgrid(el, el, indexing="ij")  # type: ignore
channel_pos = cp.stack(
    [ex.ravel(), ey.ravel(), cp.zeros(1024, dtype=cp.float32)], axis=-1
)

sound_speed = 1540.0
center_freq = 5e6
sampling_freq = center_freq
n_samples = 512
n_frames = 64


def trefoil(t):
    return cp.stack([
        (cp.sin(t) + 2 * cp.sin(2 * t)) * 0.0025,
        (cp.cos(t) - 2 * cp.cos(2 * t)) * 0.0025,
        -cp.sin(3 * t) * 0.0025 + 0.020,
    ], axis=-1)


# tissue: 1024 stationary scatterers throughout the volume
n_tissue = 1024
tissue_pos = cp.zeros((n_tissue, 3), dtype=cp.float32)
tissue_pos[:, 0] = cp.random.uniform(-0.015, 0.015, n_tissue).astype(cp.float32)
tissue_pos[:, 1] = cp.random.uniform(-0.015, 0.015, n_tissue).astype(cp.float32)
tissue_pos[:, 2] = cp.random.uniform(0.005, 0.035, n_tissue).astype(cp.float32)

# flow: 128 scatterers on the trefoil knot, shifting position each frame
n_flow = 128
t_base = cp.linspace(0, 2 * math.pi, n_flow, endpoint=False, dtype=cp.float32)
flow_noise = cp.random.normal(0, 0.0003, (n_flow, 3)).astype(cp.float32)


# simulate channel data (see simulation.py for details).
# tissue contribution is constant across frames, computed once
source = cp.array([[0.0, 0.0, -0.005]], dtype=cp.float32)

freqs = cp.fft.fftfreq(n_samples, d=1.0 / sampling_freq) + center_freq
wavenums = (-2 * math.pi * freqs / sound_speed).astype(cp.float32)
sigma_f = 0.6 * center_freq / (2 * math.sqrt(2 * math.log(2)))
pulse = cp.exp(-0.5 * ((freqs - center_freq) / sigma_f) ** 2).astype(cp.complex64)

tissue_amp = 1.0 * cp.exp(
    2j * cp.pi * cp.random.rand(1, n_tissue, 1)
).astype(cp.complex64)
tx_tissue = ffdas.greens(source, wavenums, pulse[None, None, :], tissue_pos)
rx_tissue = ffdas.greens(tissue_pos, wavenums, tx_tissue * tissue_amp, channel_pos)

# each frame has a different flow scatterer configuration: positions
# are shifted along the knot by a fraction of the period
shift_per_frame = 0.1 * 2 * math.pi / n_frames
rf_list = []
for i in range(n_frames):
    flow_pos = trefoil(t_base + i * shift_per_frame) + flow_noise  # type: ignore
    tx_flow = ffdas.greens(source, wavenums, pulse[None, None, :], flow_pos)
    rx_flow = ffdas.greens(flow_pos, wavenums, tx_flow, channel_pos)
    rx_total = rx_tissue + rx_flow
    rf_list.append(cp.fft.ifft(rx_total, axis=-1).astype(cp.complex64).conj())

rf = cp.concatenate(rf_list, axis=0)[:, :, None, :]
# rf: (n_frames, channels, 1 transmit, samples)


# reconstruct a 64^3 volume for each slow-time frame
nz, ny, nx = 64, 64, 64
x = cp.linspace(-0.010, 0.010, nx, dtype=cp.float32)
y = cp.linspace(-0.010, 0.010, ny, dtype=cp.float32)
z = cp.linspace(0.010, 0.030, nz, dtype=cp.float32)
zz, yy, xx = cp.meshgrid(z, y, x, indexing="ij")  # type: ignore
voxel_pos = cp.stack([xx, yy, zz], axis=-1)

k = sampling_freq / sound_speed
offsets = (ffdas.cdist(source[0], voxel_pos) * k)[None]
weights = cp.ones_like(offsets)

with ffdas.utils.Timer() as t:
    volume = ffdas.das(
        rf,
        channel_pos * k,
        voxel_pos * k,
        offsets,
        weights,
        wavenum=-2 * math.pi * center_freq / sampling_freq,
    )
print(
    f"das: {nz}x{ny}x{nx} grid, {n_frames} frames: {t.elapsed_ms():.1f} ms"
)

# truncate_rank decomposes along the first axis (frames) and
# reconstructs using singular vectors start through stop.
# tissue is rank 1, so start=1 removes it entirely
tissue_rank = 1

with ffdas.utils.Timer() as t:
    filtered = ffdas.truncate_rank(volume, start=tissue_rank)
print(f"truncate_rank: {t.elapsed_ms():.1f} ms")


before = cp.asnumpy(cp.abs(volume).sum(axis=0).max(axis=0))  # MIP over z
db_before = 20 * np.log10(before / before.max() + 1e-10)

after = cp.asnumpy(cp.abs(filtered).sum(axis=0).max(axis=0))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

extent = [float(x[0]) * 1e3, float(x[-1]) * 1e3,
          float(z[-1]) * 1e3, float(z[0]) * 1e3]

axes[0].imshow(db_before, extent=extent, cmap="gray", vmin=-40, vmax=0,
               aspect="equal")
axes[0].set_xlabel("x [mm]")
axes[0].set_ylabel("y [mm]")
axes[0].set_title("before filtering")

axes[1].imshow(after, extent=extent, cmap="hot", aspect="equal")
axes[1].set_xlabel("x [mm]")
axes[1].set_ylabel("y [mm]")
axes[1].set_title(f"after truncate_rank (start={tissue_rank})")

plt.tight_layout()
plt.savefig("clutter_filter.png", dpi=150)
