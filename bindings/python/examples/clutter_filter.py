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

# 32-by-32 matrix array
pitch = 300e-6
el = (cp.arange(32, dtype=cp.float32) - 15.5) * pitch
ex, ey = cp.meshgrid(el, el, indexing="ij")  # type: ignore
channel_pos = cp.stack(
    [ex.ravel(), ey.ravel(), cp.zeros(1024, dtype=cp.float32)], axis=-1
)

sound_speed = 1540.0
center_freq = 3e6
sampling_freq = center_freq
n_samples = 512
n_frames = 64


# flow: two circular vessels in perpendicular planes.
# ring_xz lives in the xz plane and is visible as a circle in the xz slice;
# ring_yz lives in the yz plane and is visible as a circle in the yz slice
vessel_radius = 0.004

def ring_xz(t):
    return cp.stack([
        vessel_radius * cp.cos(t),
        cp.zeros_like(t),
        0.015 + vessel_radius * cp.sin(t),
    ], axis=-1)

def ring_yz(t):
    return cp.stack([
        cp.zeros_like(t),
        vessel_radius * cp.cos(t),
        0.025 + vessel_radius * cp.sin(t),
    ], axis=-1)


n_flow = 64  # per vessel
t_ring1 = cp.linspace(0, 2 * math.pi, n_flow, endpoint=False, dtype=cp.float32)
t_ring2 = cp.linspace(0, 2 * math.pi, n_flow, endpoint=False, dtype=cp.float32)
noise1 = cp.random.normal(0, 0.0003, (n_flow, 3)).astype(cp.float32)
noise2 = cp.random.normal(0, 0.0003, (n_flow, 3)).astype(cp.float32)

# tissue: 1024 stationary scatterers throughout the volume
n_tissue = 16384
tissue_pos = cp.zeros((n_tissue, 3), dtype=cp.float32)
tissue_pos[:, 0] = cp.random.uniform(-0.015, 0.015, n_tissue).astype(cp.float32)
tissue_pos[:, 1] = cp.random.uniform(-0.015, 0.015, n_tissue).astype(cp.float32)
tissue_pos[:, 2] = cp.random.uniform(0.005, 0.035, n_tissue).astype(cp.float32)


# simulate channel data (see simulation.py for details).
# tissue contribution is constant across frames, computed once
source = cp.array([[0.0, 0.0, -0.005]], dtype=cp.float32)

freqs = cp.fft.fftfreq(n_samples, d=1.0 / sampling_freq) + center_freq
wavenums = (-2 * math.pi * freqs / sound_speed).astype(cp.float32)
sigma_f = 0.6 * center_freq / (2 * math.sqrt(2 * math.log(2)))
pulse = cp.exp(-0.5 * ((freqs - center_freq) / sigma_f) ** 2).astype(cp.complex64)

tissue_amp = 10.0 + cp.random.randn(1, n_tissue, 1, dtype=cp.float32) + 1j * cp.random.randn(1, n_tissue, 1, dtype=cp.float32)

tx_tissue = ffdas.greens(source, wavenums, pulse[None, None, :], tissue_pos)
rx_tissue = ffdas.greens(tissue_pos, wavenums, tx_tissue * tissue_amp, channel_pos)

# each frame has a different flow scatterer configuration: positions
# are shifted along the vessel by a fraction of the period
shift_per_frame = 0.1 * 2 * math.pi / n_frames
rf_list = []
for i in range(n_frames):
    shift = i * shift_per_frame
    flow_pos = cp.concatenate([
        ring_xz(t_ring1 + shift) + noise1,  # type: ignore
        ring_yz(t_ring2 + shift) + noise2,  # type: ignore
    ], axis=0)
    tx_flow = ffdas.greens(source, wavenums, pulse[None, None, :], flow_pos)
    rx_flow = ffdas.greens(flow_pos, wavenums, tx_flow, channel_pos)
    rx_total = rx_tissue + rx_flow
    noise = cp.random.normal(*rx_total.shape, dtype=cp.float32) + 1j * cp.random.normal(*rx_total.shape, dtype=cp.float32)  # type: ignore
    rx_total = rx_total + 0.05 * noise
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
print(f"das: {nz}x{ny}x{nx} grid, {n_frames} frames: {t.elapsed_ms():.1f} ms")

# truncate_rank decomposes along the first axis (frames) and
# reconstructs using singular vectors start through stop.
# tissue is rank 1, so start=1 removes it entirely
tissue_rank = 1

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
