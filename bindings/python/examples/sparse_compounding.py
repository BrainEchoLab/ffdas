"""Sparse compounding with das_sparse.

Standard compounding (as in compound.py) sums over all transmit events
for every voxel. When each transmit only illuminates a portion of the
volume, most voxels receive zero weight from most transmits, wasting
work. das_sparse lets each voxel specify which events to compound,
so only meaningful contributions are summed.

This example sets up 25 diverging-wave transmits from virtual sources
at different steering angles. Each source illuminates a cone-shaped
region, so any given voxel falls within only a few beams. We select
the k highest-weight transmits per voxel and compare das_sparse
against full das (which iterates over all 25, including the zeros).
"""

import cupy as cp
import matplotlib.pyplot as plt

import ffdas


cp.random.seed(0)

sound_speed = 1540.0
center_freq = 5.0e6
sampling_freq = center_freq
n_samples = 512

pitch = 250e-6
el = (cp.arange(32, dtype=cp.float32) - 15.5) * pitch
ex, ey = cp.meshgrid(el, el, indexing="ij")  # type: ignore
channel_pos = cp.column_stack([ex.ravel(), ey.ravel(), cp.zeros_like(ex).ravel()])
n_channels = channel_pos.shape[0]

xmin, xmax = -0.008, 0.008
ymin, ymax = -0.008, 0.008
zmin, zmax = 0.010, 0.026

# virtual sources behind the aperture at z < 0. a 5x5 angular grid
# gives 25 diverging-wave transmits that together cover the volume
rho = 0.03
theta = cp.linspace(-15, 15, 5, dtype=cp.float32) * cp.pi / 180  # type: ignore
phi = cp.linspace(-15, 15, 5, dtype=cp.float32) * cp.pi / 180  # type: ignore
pp, tt = cp.meshgrid(phi, theta, indexing="ij")  # type: ignore
sources = cp.stack([
    rho * cp.sin(tt.ravel()) * cp.cos(pp.ravel()),
    rho * cp.sin(pp.ravel()),
    -rho * cp.cos(tt.ravel()) * cp.cos(pp.ravel()),
], axis=-1)
n_sources = sources.shape[0]


def trefoil(t):
    return cp.stack(
        [
            (cp.sin(t) + 2 * cp.sin(2 * t)) * 0.1 * (xmax - xmin),
            -cp.sin(3 * t) * 0.15 * (ymax - ymin),
            (cp.cos(t) - 2 * cp.cos(2 * t)) * 0.1 * (zmax - zmin) + 0.5 * (zmin + zmax),
        ],
        axis=-1,
    )


n_scatterers = 8192
tube_sigma = 0.0001
bg_sigma = 0.008
t = 2 * cp.pi * cp.random.rand(n_scatterers, dtype=cp.float32)
displacement = cp.random.normal(
    scale=cp.random.choice([tube_sigma, bg_sigma], size=(n_scatterers, 1), p=[0.2, 0.8]),
    size=(n_scatterers, 3),
    dtype=cp.float32,  # type: ignore
)
scatter_pos = trefoil(t) + displacement


# simulate RF data for all transmits. each source produces a diverging
# wave: channels are delayed by their distance to the source, minus
# the reference distance to the nearest aperture edge
freqs = cp.fft.fftfreq(n_samples, d=1.0 / sampling_freq).astype(cp.float32) + center_freq
wavenums = (-2 * cp.pi * freqs / sound_speed).astype(cp.float32)
sigma_f = 0.6 * center_freq / (2 * cp.sqrt(2 * cp.log(2)))
pulse = cp.exp(-0.5 * ((freqs - center_freq) / sigma_f) ** 2).astype(cp.complex64)

aperture_size = cp.array([32 * pitch, 32 * pitch], dtype=cp.float32)
src_ref = ffdas.rect_dist(sources, aperture_size)
channel_delay = (ffdas.cdist(sources, channel_pos) - src_ref[:, None]) / sound_speed
transmission = pulse * cp.exp(-2j * cp.pi * freqs * channel_delay[:, :, None])

tx = ffdas.greens(channel_pos, wavenums, transmission, scatter_pos)
rx = ffdas.greens(scatter_pos, wavenums, tx, channel_pos)
rf = cp.fft.ifft(rx, axis=-1).astype(cp.complex64).conj()

# rf: (n_sources, n_channels, n_samples). rearrange so the source
# index becomes the sequence dimension: (1, channels, n_sources, samples)
rf = cp.ascontiguousarray(rf.transpose(1, 0, 2)[None, :, :, :])


# reconstruction grid
nz, ny, nx = 64, 64, 64
x = cp.linspace(xmin, xmax, nx, dtype=cp.float32)
y = cp.linspace(ymin, ymax, ny, dtype=cp.float32)
z = cp.linspace(zmin, zmax, nz, dtype=cp.float32)
zz, yy, xx = cp.meshgrid(z, y, x, indexing="ij")  # type: ignore
voxel_pos = cp.stack([xx, yy, zz], axis=-1)

ks = sampling_freq / sound_speed
wavenum = -2 * cp.pi * center_freq / sampling_freq

# transmit delay from each source to each voxel (in samples),
# relative to the nearest-aperture-edge reference
offsets = (ffdas.cdist(sources, voxel_pos) - src_ref[:, None, None, None]) * ks
offsets = cp.clip(offsets, 0.0, n_samples - 1)

# beam mask: a voxel is illuminated if its angle from the source's
# forward direction falls within the geometric opening angle of the
# aperture as seen from the focal point
forward = -sources / cp.linalg.norm(sources, axis=-1, keepdims=True)
angle = ffdas.angle(
    voxel_pos - sources[:, None, None, None, :],
    forward[:, None, None, None, :],
)
opening_angle = float(cp.arctan2(0.5 * 32 * pitch, rho))
weights = (angle < opening_angle).astype(cp.float32)

# element directivity
srcdir = cp.zeros((n_channels, 4), dtype=cp.float32)
srcdir[:, 2] = 1.0
srcdir[:, 3] = 0.707

# full das: sums all 25 transmits per voxel, including zeros
with ffdas.utils.Timer() as t_full:
    image_full = ffdas.das(
        rf, channel_pos * ks, voxel_pos * ks, offsets, weights,
        srcdir=srcdir, wavenum=wavenum,
    )
print(f"das (all {n_sources} transmits): {t_full.elapsed_ms():.1f} ms")

# sparse compounding: pick the k transmits with the highest weight
# per voxel. this is the "sparse" part — instead of iterating over
# all n_sources events, each voxel only processes k of them
k = 8
sparse_idx = cp.argpartition(weights, -k, axis=0)[-k:].astype(cp.int32)
sparse_weights = cp.take_along_axis(weights, sparse_idx, axis=0)
sparse_offsets = cp.take_along_axis(offsets, sparse_idx, axis=0)

with ffdas.utils.Timer() as t_sparse:
    image_sparse = ffdas.das_sparse(
        rf, channel_pos * ks, voxel_pos * ks,
        sparse_offsets, sparse_weights, sparse_idx,
        srcdir=srcdir, wavenum=wavenum,
    )
print(f"das_sparse (k={k} of {n_sources}): {t_sparse.elapsed_ms():.1f} ms")

# the two images are identical: das processes zero-weight events that
# don't contribute, das_sparse skips them
diff_db = 20 * cp.log10(
    cp.abs(image_full - image_sparse).max() / cp.abs(image_full).max() + 1e-10
)
print(f"max difference: {float(diff_db):.1f} dB")


magnitude = cp.abs(image_sparse[0])

mip_xz = magnitude.max(axis=1)
db_xz = 20 * cp.log10(mip_xz / mip_xz.max() + 1e-10)

mip_yz = magnitude.max(axis=2)
db_yz = 20 * cp.log10(mip_yz / mip_yz.max() + 1e-10)

fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

axes[0].imshow(
    cp.asnumpy(db_xz),
    extent=[float(x[0]) * 1e3, float(x[-1]) * 1e3, float(z[-1]) * 1e3, float(z[0]) * 1e3],
    cmap="gray", vmin=-32, vmax=0, aspect="equal",
)
axes[0].set_xlabel("x [mm]")
axes[0].set_ylabel("z [mm]")
axes[0].set_title("xz max projection")

axes[1].imshow(
    cp.asnumpy(db_yz),
    extent=[float(y[0]) * 1e3, float(y[-1]) * 1e3, float(z[-1]) * 1e3, float(z[0]) * 1e3],
    cmap="gray", vmin=-32, vmax=0, aspect="equal",
)
axes[1].set_xlabel("y [mm]")
axes[1].set_title("yz max projection")

plt.tight_layout()
plt.savefig("sparse_compounding.png", dpi=150)
