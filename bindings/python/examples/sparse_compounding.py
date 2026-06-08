"""Reconstruct a 3D ultrasound volume from simulated focused-wave data.

Given channel data, element positions, a reconstruction grid, and
transmit parameters, ffdas.das produces a volumetric image. The
simulation section below generates synthetic data using a trefoil
knot phantom; skip it if you have your own channel data.
"""

import cupy as cp
import matplotlib.pyplot as plt

import ffdas

ffdas.set_cuda_device(1)


cp.random.seed(0)

sound_speed = 1540.0
center_freq = 8.0e6
wavelen = sound_speed / center_freq

sampling_freq = center_freq  # complex baseband
n_samples = 512

pitch = 250e-6  # 1 wavelength
el = (cp.arange(32, dtype=cp.float32) - 15.5) * pitch
ex, ey = cp.meshgrid(el, el, indexing="ij")  # type: ignore
channel_pos = cp.column_stack([ex.ravel(), ey.ravel(), cp.zeros_like(ex).ravel()])
n_channels = channel_pos.shape[0]

xmin, xmax = -0.008, 0.008
ymin, ymax = -0.008, 0.008
zmin, zmax = 0.010, 0.026

# 2D angle grid: 4 angles along x times 4 along y = 16 compound angles
# rho = (zmax + zmin) / 2
rho = 0.03
opening_angle = cp.arctan2(0.5 * 32 * pitch, rho)

theta = cp.linspace(-5, 5, 11, dtype=cp.float32) * cp.pi / 180  # type: ignore
phi = cp.linspace(-5, 5, 11, dtype=cp.float32) * cp.pi / 180  # type: ignore
# angle_range = cp.linspace(-45, 45, 9, dtype=cp.float32) * cp.pi / 180  # type: ignore
# angle_range[:] = angle_range[0]
phi, theta = cp.meshgrid(phi, theta, indexing="ij")  # type: ignore
theta = theta.ravel()
phi = phi.ravel()
sources = cp.stack(
    [
        rho * cp.sin(theta) * cp.cos(phi),
        rho * cp.sin(phi),
        rho * cp.cos(theta) * cp.cos(phi),
    ],
    axis=-1
)

# sources = cp.stack(
#     [
#         rho * cp.sin(theta) * cp.cos(phi),
#         rho * cp.sin(theta) * cp.sin(phi),
#         rho * cp.cos(theta),
#     ],
#     axis=-1,
# )
n_sources = sources.shape[0]

# print(angle_x)
# print(angle_y)
# print(sources)
# print(phi)

# print(sources)
# simulate rf data for a normal-incidence plane wave.
# see simulation.py for a walkthrough of greens propagation


def trefoil(t):
    return cp.stack(
        [
            (cp.sin(t) + 2 * cp.sin(2 * t)) * 0.1 * (xmax - xmin),
            -cp.sin(3 * t) * 0.15 * (ymax - ymin),
            (cp.cos(t) - 2 * cp.cos(2 * t)) * 0.1 * (zmax - zmin) + 0.5 * (zmin + zmax),
        ],
        axis=-1,
    )


freqs = cp.fft.fftfreq(n_samples, d=1.0 / sampling_freq).astype(cp.float32) + center_freq
wavenums = (-2 * cp.pi * freqs / sound_speed).astype(cp.float32)
sigma_f = 0.6 * center_freq / (2 * cp.sqrt(2 * cp.log(2)))
pulse = cp.exp(-0.5 * ((freqs - center_freq) / sigma_f) ** 2).astype(cp.complex64)

batch_size = 128
n_scatterers = 4096*4

tube_sigma = 0.0001
bg_sigma = 0.008

p = 0.2
t = 2 * cp.pi * cp.random.rand(n_scatterers, dtype=cp.float32)
displacement = cp.random.normal(
    scale=cp.random.choice([tube_sigma, bg_sigma], size=(n_scatterers, 1), p=[p, 1-p]), 
    size=(n_scatterers, 3),
    dtype=cp.float32,  # type: ignore
)
scatter_pos = trefoil(t) + displacement

# scatter_pos = cp.random.uniform(
#     [xmin, ymin, zmin],  # type: ignore
#     [xmax, ymax, zmax],  # type: ignore
#     size=(n_scatterers, 3),
#     dtype=cp.float32,  # type: ignore
# )

freqs = cp.fft.fftfreq(n_samples, d=1.0 / sampling_freq).astype(cp.float32) + center_freq
wavenums = (-2 * cp.pi * freqs / sound_speed).astype(cp.float32)
sigma_f = 0.6 * center_freq / (2 * cp.sqrt(2 * cp.log(2)))
pulse = cp.exp(-0.5 * ((freqs - center_freq) / sigma_f) ** 2).astype(cp.complex64)

# plane wave: all channels transmit simultaneously (zero delay)
# channel_delay = cp.zeros(channel_pos.shape[0], dtype=cp.float32)
# transmission = pulse * cp.exp(-2j * cp.pi * freqs * channel_delay[:, None])

# channel_delay = (
#     cp.sin(angle_x[:, None]) * channel_pos[:, 0] + cp.sin(angle_y[:, None]) * channel_pos[:, 1]
# ) / sound_speed

sz = sources[..., 2]
print(sz)
size = cp.array([32 * pitch, 32 * pitch], dtype=cp.float32)

# reference offset from focal point
r = cp.maximum(0.0, sources[..., :2] + cp.sign(sz[..., None]) * 0.5 * size)
ref = cp.sign(sz) * cp.sqrt(sz ** 2 + r[..., 0] ** 2 + r[..., 1] ** 2)

# channel_delay = cp.sign(-sources[..., 2])[..., None] * (-ref[..., None] + ffdas.cdist(sources, channel_pos)) / sound_speed
# print(channel_delay.min(axis=1))
# channel_delay = cp.sign(-sources[..., 2])[..., None] * channel_delay
# channel_delay = channel_delay - channel_delay.min(axis=-1)[..., None]

channel_delay = (cp.sign(-sources[..., 2])[..., None] * ffdas.cdist(sources, channel_pos) + ref[..., None]) / sound_speed

# channel_delay = channel_delay - channel_delay.min(axis=-1)[..., None]
# channel_delay = -channel_delay
# channel_delay = channel_delay - channel_delay.min(axis=-1)[..., None]

# channel_delay = ffdas.cdist(sources, channel_pos) / sound_speed
# channel_delay = channel_delay.max(axis=-1)[..., None] - channel_delay

# print(channel_delay.min(axis=1) * sampling_freq)
# print(ref * sampling_freq / sound_speed)
# channel_delay = channel_delay.max(axis=-1)[..., None] - channel_delay #- ref[..., None] / sound_speed
# channel_delay = channel_delay - ref[..., None] / sound_speed

# print(channel_delay.min(axis=1))
# print(ref / sound_speed)

transmission = pulse * cp.exp(-2j * cp.pi * freqs * channel_delay[:, :, None])

tx = ffdas.greens(channel_pos, wavenums, transmission, scatter_pos)
rx = ffdas.greens(scatter_pos, wavenums, tx, channel_pos) / scatter_pos.shape[0]
rf = cp.fft.ifft(rx, axis=-1).astype(cp.complex64).conj()
rf = rf.transpose([1, 0, 2])
rf = cp.ascontiguousarray(rf[None, :, :, :])

# rf: (batch, channels, samples). add the sequence dimension (one
# transmission here; see compound.py for multi-angle compounding)
# rf = rf[:, :, None, :]  # (batch, channels, 1, samples)
print(rf.shape)


# reconstruction grid: 64^3 voxels centered on the phantom
nz, ny, nx = 128, 128, 128
x = cp.linspace(xmin, xmax, nx, dtype=cp.float32)
y = cp.linspace(ymin, ymax, ny, dtype=cp.float32)
z = cp.linspace(zmin, zmax, nz, dtype=cp.float32)
zz, yy, xx = cp.meshgrid(z, y, x, indexing="ij")  # type: ignore
voxel_pos = cp.stack([xx, yy, zz], axis=-1)

# all positions passed to das are in sampling wavelengths (c/fs)
ks = sampling_freq / sound_speed

# transmit offset: one-way plane-wave delay from z=0 to each voxel.
# shape is (n_transmissions, nz, ny, nx), matching the sequence dimension
# offsets = voxel_pos[None, ..., 2] * ks
# offsets = ffdas.cdist(sources, voxel_pos) * ks
# offsets = cp.clip(offsets, 0, None)

# sz = sources[..., 2]
# size = cp.array([32 * pitch, 32 * pitch], dtype=cp.float32)

# # reference offset from focal point
# r = cp.maximum(0.0, sources[..., :2] + cp.sign(sz[..., None]) * 0.5 * size)
# ref = cp.sign(sz) * cp.sqrt(sz ** 2 + r[..., 0] ** 2 + r[..., 1] ** 2)

print(wavelen)

# apodization
proj = voxel_pos[..., :2] - (zz / sz[:, None, None, None])[..., None] * sources[:, None, None, None, :2]
lim = cp.hypot(0.5 * size * (1 - zz / sz[:, None, None, None])[..., None], wavelen * sz[:, None, None, None, None] / size)
weights = cp.all(cp.abs(proj) < lim, axis=-1).astype(cp.float32)
# weights = weights / (weights.sum(axis=0) + 1e-9)

# delay
diff = voxel_pos - sources[:, None, None, None, :]
offsets = ref[:, None, None, None] + cp.sign(diff[..., 2]) * cp.linalg.norm(diff, axis=-1)

offsets = offsets * ks

# indices = cp.argmax(weights, axis=0)[None, ...].astype(cp.int32)
indices = cp.argpartition(weights, -8, axis=0)[-8:].astype(cp.int32)
weights = cp.take_along_axis(weights, indices, axis=0)
offsets = cp.take_along_axis(offsets, indices, axis=0)

# weights = weights / (weights.sum(axis=0) + 1e-9)

# print(rf.shape)
# print(weights.shape)
# print(offsets.shape)
# print(indices.shape)
# print(indices.min(), indices.max())

offsets = cp.clip(offsets, 0.0, n_samples-1)

# print(weights.shape, offsets.shape)
# print((weights * offsets * ks).min())
# print(opening_angle)

# transmit apodization (uniform for a plane wave)
# weights = cp.ones_like(offsets)
# print(ffdas.angle(-sources[:, None, None, None, :], voxel_pos).min(), ffdas.angle(-sources[:, None, None, None, :], voxel_pos).max())
# weights = cp.where(-ffdas.angle(sources[:, None, None, None, :], voxel_pos) < opening_angle, 1.0, 0.0).astype(cp.float32)
# weights = cp.abs(ffdas.angle(-sources[:, None, None, None, :], voxel_pos - sources[:, None, None, None, :]))

# with ffdas.utils.Timer() as t:
#     image = ffdas.das(
#         rf,
#         channel_pos * ks,
#         voxel_pos * ks,
#         offsets * ks,
#         weights,
#         srcdir=None,
#         wavenum=-2 * cp.pi * center_freq / sampling_freq,
#     )
# print(f"das (srcdir=None): {nz}x{ny}x{nx}, {channel_pos.shape[0]} ch, batch {batch_size}: {t.elapsed_ms():.1f} ms")

# element directivity: (channels, 4) where columns 0-2 are the element
# normal and column 3 is the cosine cutoff. voxels outside the cone
# receive zero weight from that channel
srcdir = cp.zeros((channel_pos.shape[0], 4), dtype=cp.float32)
srcdir[:, 2] = 1.0  # normals point in +z
srcdir[:, 3] = 0.707  # 60 degree half-angle

with ffdas.utils.Timer() as t:
    # image = ffdas.das(
    #     rf,
    #     channel_pos * ks,
    #     voxel_pos * ks,
    #     offsets,
    #     weights,
    #     srcdir=srcdir,
    #     wavenum=-2 * cp.pi * center_freq / sampling_freq,
    # )
    image = ffdas.das_sparse(
        rf,
        channel_pos * ks,
        voxel_pos * ks,
        offsets,
        weights,
        sparse_indices=indices,
        srcdir=srcdir,
        wavenum=-2 * cp.pi * center_freq / sampling_freq,
    )

print(f"das (with srcdir): {nz}x{ny}x{nx}, {channel_pos.shape[0]} ch, batch {batch_size}: {t.elapsed_ms():.1f} ms")

print(image.shape, rf.shape)

magnitude = cp.abs(image[0])
# magnitude = weights[0]

mip_xz = magnitude.max(axis=1)
# mip_xz = magnitude[:, 64]
db_xz = 20 * cp.log10(mip_xz / mip_xz.max() + 1e-10)

mip_yz = magnitude.max(axis=2)
# mip_yz = magnitude[:, :, 64]
db_yz = 20 * cp.log10(mip_yz / mip_yz.max() + 1e-10)

fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

axes[0].imshow(
    cp.asnumpy(db_xz),
    extent=[float(x[0]) * 1e3, float(x[-1]) * 1e3, float(z[-1]) * 1e3, float(z[0]) * 1e3],
    cmap="gray",
    vmin=-32,
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
    vmin=-32,
    vmax=0,
    aspect="equal",
)
axes[1].set_xlabel("y [mm]")
axes[1].set_title("yz max projection")

plt.tight_layout()
plt.savefig("reconstruct.png", dpi=150)
