"""Reconstruct a 3D ultrasound volume from simulated diverging-wave data.

Covers the core workflow: simulate channel data via Green's function
propagation, compute transmit parameters, and call ffdas.das to
reconstruct a volumetric image. The trefoil knot phantom produces a
recognizable, distinct shape in each maximum intensity projection.
"""

import math

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

import ffdas


cp.random.seed(0)

# 32x32 matrix array in the z=0 plane
pitch = 250e-6
el = (cp.arange(32, dtype=cp.float32) - 15.5) * pitch
ex, ey = cp.meshgrid(el, el, indexing="ij")  # type: ignore
channel_pos = cp.stack(
    [ex.ravel(), ey.ravel(), cp.zeros(1024, dtype=cp.float32)], axis=-1
)

sound_speed = 1540.0
center_freq = 5e6
sampling_freq = center_freq  # complex baseband
n_samples = 512
batch_size = 128


# 2048 scatterers distributed around a trefoil knot
n_scatterers = 4096
t = cp.linspace(0, 2 * math.pi, n_scatterers, endpoint=False, dtype=cp.float32)
knot = cp.stack([
    (cp.sin(t) + 2 * cp.sin(2 * t)) * 0.0025,
    (cp.cos(t) - 2 * cp.cos(2 * t)) * 0.0025,
    -cp.sin(3 * t) * 0.0025 + 0.020,
], axis=-1)
scatterers = knot + cp.random.normal(0, 0.0004, (n_scatterers, 3)).astype(cp.float32)


# simulate channel data (see simulation.py for a detailed walkthrough)
source = cp.array([[0.0, 0.0, -0.01]], dtype=cp.float32)

freqs = cp.fft.fftfreq(n_samples, d=1.0 / sampling_freq) + center_freq
wavenums = (-2 * math.pi * freqs / sound_speed).astype(cp.float32)
sigma_f = 0.6 * center_freq / (2 * math.sqrt(2 * math.log(2)))
pulse = cp.exp(-0.5 * ((freqs - center_freq) / sigma_f) ** 2).astype(cp.complex64)

phases = cp.exp(
    2j * cp.pi * cp.random.rand(batch_size, n_scatterers, 1)
).astype(cp.complex64)

tx = ffdas.greens(source, wavenums, pulse[None, None, :], scatterers)
rx = ffdas.greens(scatterers, wavenums, tx * phases, channel_pos)
rf = cp.fft.ifft(rx, axis=-1).astype(cp.complex64).conj()
rf = rf[:, :, None, :]  # (batch, channels, 1 transmit, samples)


# reconstruction grid: 128^3 voxels
nz, ny, nx = 64, 64, 64
x = cp.linspace(-0.010, 0.010, nx, dtype=cp.float32)
y = cp.linspace(-0.010, 0.010, ny, dtype=cp.float32)
z = cp.linspace(0.010, 0.030, nz, dtype=cp.float32)
zz, yy, xx = cp.meshgrid(z, y, x, indexing="ij")  # type: ignore
voxel_pos = cp.stack([xx, yy, zz], axis=-1)  # (nz, ny, nx, 3)


# transmit parameters.
# all positions passed to das are in units of c/fs (sampling wavelengths)
k = sampling_freq / sound_speed

# one-way transmit delay from virtual source to each voxel, in samples
offsets = (ffdas.cdist(source[0], voxel_pos) * k)[None]  # (1, nz, ny, nx)
weights = cp.ones_like(offsets)

# element directivity: xdir is (channels, 4) where the first three
# columns are the unit normal of each element and the fourth is the
# minimum cosine of the angle between a voxel and the normal. voxels
# outside this cone receive zero weight from that channel
xdir = cp.zeros((channel_pos.shape[0], 4), dtype=cp.float32)
xdir[:, 2] = 1.0   # normals point in +z
xdir[:, 3] = 0.5   # half-angle cutoff at arccos(0.5) = 60 degrees


with ffdas.utils.Timer() as t:
    image = ffdas.das(
        rf,
        channel_pos * k,
        voxel_pos * k,
        offsets,
        weights,
        xdir=xdir,
        wavenum=-2 * math.pi * center_freq / sampling_freq,
    )
print(
    f"das: {nz}x{ny}x{nx} grid, {channel_pos.shape[0]} channels, "
    f"batch {batch_size}: {t.elapsed_ms():.1f} ms"
)
# image: (batch, nz, ny, nx)


magnitude = cp.abs(image[0])

mip_xy = cp.asnumpy(magnitude.max(axis=0))  # max over z
db_xy = 20 * np.log10(mip_xy / mip_xy.max() + 1e-10)

mip_yz = cp.asnumpy(magnitude.max(axis=2))  # max over x
db_yz = 20 * np.log10(mip_yz / mip_yz.max() + 1e-10)

fig, axes = plt.subplots(1, 2, figsize=(10, 8))

axes[0].imshow(
    db_xy,
    extent=[float(x[0]) * 1e3, float(x[-1]) * 1e3,
            float(y[-1]) * 1e3, float(y[0]) * 1e3],
    cmap="gray",
    vmin=-24,
    vmax=0,
    aspect="equal",
)
axes[0].set_xlabel("x [mm]")
axes[0].set_ylabel("y [mm]")
axes[0].set_title("xy max projection")

axes[1].imshow(
    db_yz.T,  # plot y vertically
    extent=[float(z[0]) * 1e3, float(z[-1]) * 1e3,
            float(y[-1]) * 1e3, float(y[0]) * 1e3],
    cmap="gray",
    vmin=-24,
    vmax=0,
    aspect="equal",
)
axes[1].set_xlabel("z [mm]")
axes[1].set_ylabel("y [mm]")
axes[1].set_title("yz max projection")

plt.tight_layout()
plt.savefig("reconstruct.png", dpi=150)
