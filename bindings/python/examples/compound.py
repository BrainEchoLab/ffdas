"""Coherent plane-wave compounding.

Plane-wave compounding transmits at several steering angles and
combines the resulting images to improve lateral resolution and
reduce artifacts. ffdas.das handles this as follows: the
sequence dimension of the rf data holds per-angle channel data,
and the matching offsets array provides the per-angle transmit
delay at each voxel. DAS sums over the sequence dimension,
producing a coherently compounded image.
"""

import cupy as cp
import matplotlib.pyplot as plt

import ffdas


cp.random.seed(0)

sound_speed = 1540.0
center_freq = 3.08e6
sampling_freq = center_freq
n_samples = 512

pitch = 500e-6  # 1 wavelength
el = (cp.arange(32, dtype=cp.float32) - 15.5) * pitch
ex, ey = cp.meshgrid(el, el, indexing="ij")  # type: ignore
channel_pos = cp.column_stack([ex.ravel(), ey.ravel(), cp.zeros_like(ex).ravel()])
n_channels = channel_pos.shape[0]

xmin, xmax = -0.008, 0.008
ymin, ymax = -0.008, 0.008
zmin, zmax = 0.010, 0.026

# 2D angle grid: 3 angles along x times 3 along y = 9 compound angles
angle_range = cp.linspace(-10, 10, 3, dtype=cp.float32) * cp.pi / 180  # type: ignore
angle_x, angle_y = cp.meshgrid(angle_range, angle_range, indexing="ij")  # type: ignore
angle_x = angle_x.ravel()
angle_y = angle_y.ravel()
n_angles = angle_x.shape[0]


# simulate rf data for each steering angle.
# a plane wave at angle (theta_x, theta_y) steers by delaying element i
# by (sin(theta_x) * element_x + sin(theta_y) * element_y) / c


def trefoil(t):
    return cp.stack(
        [
            (cp.sin(t) + 2 * cp.sin(2 * t)) * 0.1 * (xmax - xmin),
            -cp.sin(3 * t) * 0.15 * (ymax - ymin),
            (cp.cos(t) - 2 * cp.cos(2 * t)) * 0.1 * (zmax - zmin) + 0.5 * (zmin + zmax),
        ],
        axis=-1,
    )


batch_size = 16
n_scatterers = 16384
n_knot = 8192

t = 2 * cp.pi * cp.random.rand(n_knot, dtype=cp.float32)
knot_pos = trefoil(t) + 0.00015 * cp.random.randn(n_knot, 3, dtype=cp.float32)
background_pos = cp.random.uniform(
    [xmin, ymin, zmin],  # type: ignore
    [xmax, ymax, zmax],  # type: ignore
    size=(n_scatterers - n_knot, 3),
    dtype=cp.float32,  # type: ignore
)

scatter_pos = cp.concat([knot_pos, background_pos], axis=0)
scatter_values = cp.concat(
    [
        cp.random.rand(batch_size, n_knot, dtype=cp.float32),
        cp.random.rand(batch_size, n_scatterers - n_knot, dtype=cp.float32),
    ],
    axis=1,
)

freqs = cp.fft.fftfreq(n_samples, d=1.0 / sampling_freq).astype(cp.float32) + center_freq
wavenums = (-2 * cp.pi * freqs / sound_speed).astype(cp.float32)
sigma_f = 0.6 * center_freq / (2 * cp.sqrt(2 * cp.log(2)))
pulse = cp.exp(-0.5 * ((freqs - center_freq) / sigma_f) ** 2).astype(cp.complex64)

rf = cp.empty((batch_size, n_channels, n_angles, n_samples), dtype=cp.complex64)
for i in range(n_angles):
    delay = (
        float(cp.sin(angle_x[i])) * channel_pos[:, 0] + float(cp.sin(angle_y[i])) * channel_pos[:, 1]
    ) / sound_speed
    tx_signal = pulse * cp.exp(-2j * cp.pi * freqs * delay[:, None])
    tx = ffdas.greens(channel_pos, wavenums, tx_signal, scatter_pos)
    rx = ffdas.greens(scatter_pos, wavenums, tx * scatter_values[:, :, None], channel_pos)
    rf[:, :, i] = cp.fft.ifft(rx, axis=-1).conj()

# reconstruction grid
nz, ny, nx = 64, 64, 64
x = cp.linspace(xmin, xmax, nx, dtype=cp.float32)
y = cp.linspace(ymin, ymax, ny, dtype=cp.float32)
z = cp.linspace(zmin, zmax, nz, dtype=cp.float32)
zz, yy, xx = cp.meshgrid(z, y, x, indexing="ij")  # type: ignore
voxel_pos = cp.stack([xx, yy, zz], axis=-1)

ks = sampling_freq / sound_speed

# per-angle transmit offsets: one-way plane-wave delay at angles (theta_x, theta_y)
# is (x*sin(theta_x) + y*sin(theta_y) + z*cos(theta_x)*cos(theta_y)) * fs/c
sin_ax = cp.sin(angle_x)[:, None, None, None]
sin_ay = cp.sin(angle_y)[:, None, None, None]
cos_ax = cp.cos(angle_x)[:, None, None, None]
cos_ay = cp.cos(angle_y)[:, None, None, None]
offsets = (xx * sin_ax + yy * sin_ay + zz * cos_ax * cos_ay) * ks  # (n_angles, nz, ny, nx)
weights = cp.ones_like(offsets)

wavenum = -2 * cp.pi * center_freq / sampling_freq

# compounded: das sums over the sequence dimension (all angles at once)
with ffdas.utils.Timer() as t:
    compounded = ffdas.das(
        rf,
        channel_pos * ks,
        voxel_pos * ks,
        offsets,
        weights,
        wavenum=wavenum,
    )
print(f"compounded ({n_angles} angles): {t.elapsed_ms():.1f} ms")

# single angle at normal incidence for comparison
mid = n_angles // 2
with ffdas.utils.Timer() as t:
    single = ffdas.das(
        rf[:, :, mid : mid + 1, :],
        channel_pos * ks,
        voxel_pos * ks,
        offsets[mid : mid + 1],
        weights[mid : mid + 1],
        wavenum=wavenum,
    )
print(f"single angle: {t.elapsed_ms():.1f} ms")


mag_single = cp.abs(single[0])
mag_compounded = cp.abs(compounded[0])

fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharey="row", sharex="col")

extent_xz = [float(x[0]) * 1e3, float(x[-1]) * 1e3, float(z[-1]) * 1e3, float(z[0]) * 1e3]
extent_yz = [float(y[0]) * 1e3, float(y[-1]) * 1e3, float(z[-1]) * 1e3, float(z[0]) * 1e3]

mip = mag_single.max(axis=1)
db = 20 * cp.log10(mip / mip.max() + 1e-10)
axes[0, 0].imshow(cp.asnumpy(db), extent=extent_xz, cmap="gray", vmin=-24, vmax=0, aspect="equal")
axes[0, 0].set_xlabel("x [mm]")
axes[0, 0].set_ylabel("z [mm]")
axes[0, 0].set_title("single angle — xz")

mip = mag_single.max(axis=2)
db = 20 * cp.log10(mip / mip.max() + 1e-10)
axes[0, 1].imshow(cp.asnumpy(db), extent=extent_yz, cmap="gray", vmin=-24, vmax=0, aspect="equal")
axes[0, 1].set_xlabel("y [mm]")
axes[0, 1].set_title("single angle — yz")

mip = mag_compounded.max(axis=1)
db = 20 * cp.log10(mip / mip.max() + 1e-10)
axes[1, 0].imshow(cp.asnumpy(db), extent=extent_xz, cmap="gray", vmin=-24, vmax=0, aspect="equal")
axes[1, 0].set_xlabel("x [mm]")
axes[1, 0].set_ylabel("z [mm]")
axes[1, 0].set_title(f"compounded ({n_angles} angles) — xz")

mip = mag_compounded.max(axis=2)
db = 20 * cp.log10(mip / mip.max() + 1e-10)
axes[1, 1].imshow(cp.asnumpy(db), extent=extent_yz, cmap="gray", vmin=-24, vmax=0, aspect="equal")
axes[1, 1].set_xlabel("y [mm]")
axes[1, 1].set_title(f"compounded ({n_angles} angles) — yz")

plt.tight_layout()
plt.savefig("compound.png", dpi=150)
