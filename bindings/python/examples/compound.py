"""Coherent plane-wave compounding.

Plane-wave compounding transmits at several steering angles and
combines the resulting images to improve lateral resolution and
reduce sidelobe artifacts. ffdas.das handles this natively: the
sequence dimension of the rf data holds per-angle channel data,
and the matching offsets array provides the per-angle transmit
delay at each voxel. DAS sums over the sequence dimension,
producing a coherently compounded image in a single call.
"""

import cupy as cp
import matplotlib.pyplot as plt

import ffdas


cp.random.seed(0)

# same probe and medium as reconstruct.py
pitch = 300e-6
el = (cp.arange(32, dtype=cp.float32) - 15.5) * pitch
ex, ey = cp.meshgrid(el, el, indexing="ij")  # type: ignore
channel_pos = cp.column_stack([ex.ravel(), ey.ravel(), cp.zeros_like(ex).ravel()])

sound_speed = 1540.0
center_freq = 3e6
sampling_freq = center_freq
n_samples = 512

n_angles = 11
angles = cp.linspace(-15, 15, n_angles, dtype=cp.float32) * (cp.pi / 180)


# simulate rf data for each steering angle.
# a plane wave at angle theta steers by delaying element i
# by sin(theta) * element_x / c

def trefoil(t):
    return cp.stack(
        [
            (cp.sin(t) + 2 * cp.sin(2 * t)) * 0.0015,
            -cp.sin(3 * t) * 0.002,
            (cp.cos(t) - 2 * cp.cos(2 * t)) * 0.0015 + 0.01,
        ],
        axis=-1,
    )


batch_size = 32
n_scatterers = 4096
t = 2 * cp.pi * cp.random.rand(n_scatterers, dtype=cp.float32)
scatter_pos = trefoil(t) + cp.random.normal(
    0, 0.00015, (n_scatterers, 3), dtype=cp.float32  # type: ignore
)

freqs = cp.fft.fftfreq(n_samples, d=1.0 / sampling_freq).astype(cp.float32) + center_freq
wavenums = (-2 * cp.pi * freqs / sound_speed).astype(cp.float32)
sigma_f = 0.6 * center_freq / (2 * cp.sqrt(2 * cp.log(2)))
pulse = cp.exp(-0.5 * ((freqs - center_freq) / sigma_f) ** 2).astype(cp.complex64)
scatter_values = cp.random.rand(batch_size, n_scatterers, 1, dtype=cp.float32)

rf_angles = []
for angle in angles:
    delay = float(cp.sin(angle)) * channel_pos[:, 0] / sound_speed
    tx_signal = pulse * cp.exp(-2j * cp.pi * freqs * delay[:, None])
    tx = ffdas.greens(channel_pos, wavenums, tx_signal, scatter_pos)
    rx = ffdas.greens(scatter_pos, wavenums, tx * scatter_values, channel_pos)
    rf_angles.append(cp.fft.ifft(rx, axis=-1).astype(cp.complex64).conj())

rf = cp.stack(rf_angles, axis=2)  # (batch, channels, n_angles, samples)


# reconstruction grid
nz, ny, nx = 64, 64, 64
x = cp.linspace(-0.007, 0.007, nx, dtype=cp.float32)
y = cp.linspace(-0.007, 0.007, ny, dtype=cp.float32)
z = cp.linspace(0.003, 0.017, nz, dtype=cp.float32)
zz, yy, xx = cp.meshgrid(z, y, x, indexing="ij")  # type: ignore
voxel_pos = cp.stack([xx, yy, zz], axis=-1)

ks = sampling_freq / sound_speed

# per-angle transmit offsets: one-way plane-wave delay at angle theta
# is (x*sin(theta) + z*cos(theta)) * fs/c
sin_a = cp.sin(angles)[:, None, None, None]
cos_a = cp.cos(angles)[:, None, None, None]
offsets = (xx * sin_a + zz * cos_a) * ks  # (n_angles, nz, ny, nx)
weights = cp.ones_like(offsets)

xdir = cp.zeros((channel_pos.shape[0], 4), dtype=cp.float32)
xdir[:, 2] = 1.0
xdir[:, 3] = 0.5

wavenum = -2 * cp.pi * center_freq / sampling_freq

# compounded: das sums over the sequence dimension (all angles at once)
with ffdas.utils.Timer() as t:
    compounded = ffdas.das(
        rf, channel_pos * ks, voxel_pos * ks, offsets, weights,
        xdir=xdir, wavenum=wavenum,
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
        xdir=xdir,
        wavenum=wavenum,
    )
print(f"single angle: {t.elapsed_ms():.1f} ms")


def xz_mip_db(vol, dynamic_range=-24):
    mip = cp.abs(vol[0]).max(axis=1)
    db = 20 * cp.log10(mip / mip.max() + 1e-10)
    return cp.asnumpy(cp.clip(db, dynamic_range, 0))


extent = [float(x[0]) * 1e3, float(x[-1]) * 1e3,
          float(z[-1]) * 1e3, float(z[0]) * 1e3]

fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

axes[0].imshow(xz_mip_db(single), extent=extent, cmap="gray",
               vmin=-24, vmax=0, aspect="equal")
axes[0].set_xlabel("x [mm]")
axes[0].set_ylabel("z [mm]")
axes[0].set_title("single angle (0°)")

axes[1].imshow(xz_mip_db(compounded), extent=extent, cmap="gray",
               vmin=-24, vmax=0, aspect="equal")
axes[1].set_xlabel("x [mm]")
axes[1].set_title(f"compounded ({n_angles} angles, ±15°)")

plt.tight_layout()
plt.savefig("compound.png", dpi=150)
