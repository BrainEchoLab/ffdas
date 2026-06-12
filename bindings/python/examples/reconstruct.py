"""Reconstruct a 3D ultrasound volume from simulated plane-wave data.

Given channel data, element positions, a reconstruction grid, and
transmit parameters, ffdas.das produces a volumetric image. The
simulation section below generates synthetic data using a trefoil
knot phantom; skip it if you have your own channel data.
"""

import cupy as cp
import matplotlib.pyplot as plt

import ffdas


cp.random.seed(0)

sound_speed = 1540.0
center_freq = 3.08e6
sampling_freq = center_freq  # complex baseband
n_samples = 512

# probe: 32x32 matrix array in the xy-plane at z=0
pitch = 500e-6
el = (cp.arange(32, dtype=cp.float32) - 15.5) * pitch
ex, ey = cp.meshgrid(el, el, indexing="ij")  # type: ignore
channel_pos = cp.column_stack([ex.ravel(), ey.ravel(), cp.zeros_like(ex).ravel()])
n_channels = channel_pos.shape[0]

xmin, xmax = -0.008, 0.008
ymin, ymax = -0.008, 0.008
zmin, zmax = 0.010, 0.026


# simulate rf data for a normal-incidence plane wave


def trefoil(t):
    return cp.stack(
        [
            (cp.sin(t) + 2 * cp.sin(2 * t)) * 0.1 * (xmax - xmin),
            -cp.sin(3 * t) * 0.15 * (ymax - ymin),
            (cp.cos(t) - 2 * cp.cos(2 * t)) * 0.1 * (zmax - zmin) + 0.5 * (zmin + zmax),
        ],
        axis=-1,
    )


batch_size = 128
n_scatterers = 4096

tube_sigma = 0.0001
bg_sigma = 0.008

p = 0.3
t = 2 * cp.pi * cp.random.rand(batch_size, n_scatterers, dtype=cp.float32)
displacement = cp.random.normal(
    scale=cp.random.choice([tube_sigma, bg_sigma], size=(1, n_scatterers, 1), p=[p, 1-p]), 
    size=(batch_size, n_scatterers, 3),
    dtype=cp.float32,  # type: ignore
)
scatter_pos = trefoil(t) + displacement

freqs = cp.fft.fftfreq(n_samples, d=1.0 / sampling_freq).astype(cp.float32) + center_freq
wavenums = (-2 * cp.pi * freqs / sound_speed).astype(cp.float32)
sigma_f = 0.6 * center_freq / (2 * cp.sqrt(2 * cp.log(2)))
pulse = cp.exp(-0.5 * ((freqs - center_freq) / sigma_f) ** 2).astype(cp.complex64)

# plane wave: all channels transmit simultaneously (zero delay)
channel_delay = cp.zeros(channel_pos.shape[0], dtype=cp.float32)
transmission = pulse * cp.exp(-2j * cp.pi * freqs * channel_delay[:, None])

tx = ffdas.greens(channel_pos, wavenums, transmission, scatter_pos)
rx = ffdas.greens(scatter_pos, wavenums, tx, channel_pos)
rf = cp.fft.ifft(rx, axis=-1).astype(cp.complex64).conj()

# rf: (batch, channels, samples). add the sequence dimension (one
# transmission here; see compound.py for multi-angle compounding)
rf = rf[:, :, None, :]  # (batch, channels, 1, samples)


# reconstruction grid: 64^3 voxels centered on the phantom
nz, ny, nx = 64, 64, 64
x = cp.linspace(xmin, xmax, nx, dtype=cp.float32)
y = cp.linspace(ymin, ymax, ny, dtype=cp.float32)
z = cp.linspace(zmin, zmax, nz, dtype=cp.float32)
zz, yy, xx = cp.meshgrid(z, y, x, indexing="ij")  # type: ignore
voxel_pos = cp.stack([xx, yy, zz], axis=-1)

# all positions passed to das are in sampling wavelengths (c/fs)
ks = sampling_freq / sound_speed

# transmit offset: one-way plane-wave delay from z=0 to each voxel.
# shape is (n_transmissions, nz, ny, nx), matching the sequence dimension
offsets = voxel_pos[None, ..., 2] * ks

# transmit apodization (uniform for a plane wave)
weights = cp.ones_like(offsets)

with ffdas.utils.Timer() as t:
    image = ffdas.das(
        rf,
        channel_pos * ks,
        voxel_pos * ks,
        offsets,
        weights,
        srcdir=None,
        wavenum=-2 * cp.pi * center_freq / sampling_freq,
    )
print(f"das (srcdir=None): {nz}x{ny}x{nx}, {channel_pos.shape[0]} ch, batch {batch_size}: {t.elapsed_ms():.1f} ms")

# element directivity: (channels, 4) where columns 0-2 are the element
# normal and column 3 is the cosine cutoff. voxels outside the cone
# receive zero weight from that channel
srcdir = cp.zeros((channel_pos.shape[0], 4), dtype=cp.float32)
srcdir[:, 2] = 1.0  # normals point in +z
srcdir[:, 3] = 0.707  # 60 degree half-angle

with ffdas.utils.Timer() as t:
    image = ffdas.das(
        rf,
        channel_pos * ks,
        voxel_pos * ks,
        offsets,
        weights,
        srcdir=srcdir,
        wavenum=-2 * cp.pi * center_freq / sampling_freq,
    )
print(f"das (with srcdir): {nz}x{ny}x{nx}, {channel_pos.shape[0]} ch, batch {batch_size}: {t.elapsed_ms():.1f} ms")


magnitude = cp.abs(image[0])

mip_xz = magnitude.max(axis=1)
db_xz = 20 * cp.log10(mip_xz / mip_xz.max() + 1e-10)

mip_yz = magnitude.max(axis=2)
db_yz = 20 * cp.log10(mip_yz / mip_yz.max() + 1e-10)

fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

axes[0].imshow(
    cp.asnumpy(db_xz),
    extent=[float(x[0]) * 1e3, float(x[-1]) * 1e3, float(z[-1]) * 1e3, float(z[0]) * 1e3],
    cmap="gray",
    vmin=-24,
    vmax=0,
    aspect="equal",
)
axes[0].set_xlabel("x [mm]")
axes[0].set_ylabel("z [mm]")
axes[0].set_title("xz max projection")

axes[1].imshow(
    cp.asnumpy(db_yz),
    extent=[float(y[0]) * 1e3, float(y[-1]) * 1e3, float(z[-1]) * 1e3, float(z[0]) * 1e3],
    cmap="gray",
    vmin=-24,
    vmax=0,
    aspect="equal",
)
axes[1].set_xlabel("y [mm]")
axes[1].set_title("yz max projection")

plt.tight_layout()
plt.savefig("reconstruct.png", dpi=150)
