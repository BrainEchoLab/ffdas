"""Reconstruct a 3D ultrasound volume from simulated diverging-wave data.

Covers the core workflow: simulate channel data via Green's function
propagation, compute transmit parameters, and call ffdas.das to
reconstruct a volumetric image. The trefoil knot phantom produces a
recognizable, distinct shape in each maximum intensity projection.
"""

import cupy as cp
import matplotlib.pyplot as plt

import ffdas


cp.random.seed(0)

# 32-by-32 matrix array in the xy-plane at z=0
pitch = 300e-6
el = (cp.arange(32, dtype=cp.float32) - 15.5) * pitch
ex, ey = cp.meshgrid(el, el, indexing="ij")  # type: ignore
channel_pos = cp.column_stack([ex.ravel(), ey.ravel(), cp.zeros_like(ex).ravel()])

sound_speed = 1540.0
center_freq = 3e6
sampling_freq = center_freq  # complex baseband
n_samples = 512
batch_size = 128


# scatterers distributed around a trefoil knot
def trefoil(t):
    return cp.stack(
        [
            (cp.sin(t) + 2 * cp.sin(2 * t)) * 0.0015,
            -cp.sin(3 * t) * 0.002,
            (cp.cos(t) - 2 * cp.cos(2 * t)) * 0.0015 + 0.01,
        ],
        axis=-1,
    )


n_scatterers = 4096
t = 2 * cp.pi * cp.random.rand(n_scatterers, dtype=cp.float32)
scatter_pos = trefoil(t) + cp.random.normal(
    0,
    0.00015,
    (n_scatterers, 3),
    dtype=cp.float32,  # type: ignore
)

# simulate channel data for a plane wave transmission
channel_delay = cp.zeros(channel_pos.shape[0], dtype=cp.float32)  # (channels,)

freqs = cp.fft.fftfreq(n_samples, d=1.0 / sampling_freq).astype(cp.float32) + center_freq
wavenums = (-2 * cp.pi * freqs / sound_speed).astype(cp.float32)
sigma_f = 0.6 * center_freq / (2 * cp.sqrt(2 * cp.log(2)))
pulse = cp.exp(-0.5 * ((freqs - center_freq) / sigma_f) ** 2).astype(cp.complex64)  # (samples,)

transmission = pulse * cp.exp(-2j * cp.pi * freqs * channel_delay[:, None])  # (channels, samples)
scatter_values = cp.random.rand(batch_size, n_scatterers, 1, dtype=cp.float32)

tx = ffdas.greens(channel_pos, wavenums, transmission, scatter_pos)  # (channels, samples)
rx = ffdas.greens(scatter_pos, wavenums, tx * scatter_values, channel_pos)  # (channels, samples)
rf = cp.fft.ifft(rx, axis=-1).astype(cp.complex64).conj()  # (batch, channels, sequence, samples)
rf = rf[:, :, None, :]  # add dummy sequence dimension

# reconstruction grid: 64^3 voxels
nz, ny, nx = 64, 64, 64
x = cp.linspace(-0.007, 0.007, nx, dtype=cp.float32)
y = cp.linspace(-0.007, 0.007, ny, dtype=cp.float32)
z = cp.linspace(0.003, 0.017, nz, dtype=cp.float32)
zz, yy, xx = cp.meshgrid(z, y, x, indexing="ij")  # type: ignore
voxel_pos = cp.stack([xx, yy, zz], axis=-1)  # (nz, ny, nx, 3)

# transmit parameters.
# all positions passed to das are in units of c/fs (sampling wavelengths)
ks = sampling_freq / sound_speed

# one-way transmit delay from array at z=0 to each voxel, in samples
offsets = voxel_pos[None, ..., 2] * ks  # (1, nz, ny, nx) add dummy sequence dimension

# transmit intensity weighting at each voxel
weights = cp.ones_like(offsets)

# element directivity: xdir is (channels, 4) where the first three
# columns are the unit normal of each element and the fourth is the
# minimum cosine of the angle between a voxel and the normal. voxels
# outside this cone receive zero weight from that channel
xdir = cp.zeros((channel_pos.shape[0], 4), dtype=cp.float32)
xdir[:, 2] = 1.0  # normals point in +z
xdir[:, 3] = 0.5  # half-angle cutoff at arccos(0.5) = 60 degrees


with ffdas.utils.Timer() as t:
    # das outputs (batch, nz, ny, nx)
    image = ffdas.das(
        rf,
        channel_pos * ks,
        voxel_pos * ks,
        offsets,
        weights,
        xdir=xdir,
        wavenum=-2 * cp.pi * center_freq / sampling_freq,
    )
print(f"das: {nz}x{ny}x{nx} grid, {channel_pos.shape[0]} channels, batch {batch_size}: {t.elapsed_ms():.1f} ms")

magnitude = cp.abs(image[0])

mip_xz = magnitude.max(axis=1)  # max over y
db_xz = 20 * cp.log10(mip_xz / mip_xz.max() + 1e-10)

mip_yz = magnitude.max(axis=2)  # max over x
db_yz = 20 * cp.log10(mip_yz / mip_yz.max() + 1e-10)

fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

axes[0].imshow(
    cp.asnumpy(db_xz),
    extent=[
        float(x[0]) * 1e3,
        float(x[-1]) * 1e3,
        float(z[-1]) * 1e3,
        float(z[0]) * 1e3,
    ],
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
    extent=[
        float(y[0]) * 1e3,
        float(y[-1]) * 1e3,
        float(z[-1]) * 1e3,
        float(z[0]) * 1e3,
    ],
    cmap="gray",
    vmin=-24,
    vmax=0,
    aspect="equal",
)
axes[1].set_xlabel("y [mm]")
axes[1].set_title("yz max projection")

plt.tight_layout()
plt.savefig("reconstruct.png", dpi=150)
