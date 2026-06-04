"""Interpolation from structured grids to arbitrary positions.

A structured grid has a regular index-space lattice (nz, ny, nx) but
its vertex positions can be arbitrary 3D coordinates. This enables
reconstruction on curvilinear grids (e.g. spherical) followed by
interpolation to Cartesian coordinates or any other query positions.

This example places scatterers on a regular Cartesian grid, then
reconstructs on a spherical grid where the pattern appears warped.
After interpolation back to Cartesian coordinates, the grid is
recovered.
"""

import cupy as cp
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import ffdas


cp.random.seed(0)


sound_speed = 1540.0
center_freq = 3.08e6
sampling_freq = center_freq  # complex baseband
n_samples = 512

# probe: 32x32 matrix array in the xy-plane at z=0
pitch = 250e-6
el = (cp.arange(32, dtype=cp.float32) - 15.5) * pitch
ex, ey = cp.meshgrid(el, el, indexing="ij")  # type: ignore
channel_pos = cp.column_stack([ex.ravel(), ey.ravel(), cp.zeros_like(ex).ravel()])
n_channels = channel_pos.shape[0]

xmin, xmax = -0.016, 0.016
ymin, ymax = -0.016, 0.016
zmin, zmax = 0.006, 0.038

source = cp.array([[0.0, 0.0, -0.005]], dtype=cp.float32)

freqs = cp.fft.fftfreq(n_samples, d=1.0 / sampling_freq).astype(cp.float32) + center_freq
wavenums = (-2 * cp.pi * freqs / sound_speed).astype(cp.float32)
sigma_f = 0.6 * center_freq / (2 * cp.sqrt(2 * cp.log(2)))
pulse = cp.exp(-0.5 * ((freqs - center_freq) / sigma_f) ** 2).astype(cp.complex64)

delay = ffdas.cdist(source, channel_pos) / sound_speed
tx_signal = pulse * cp.exp(-2j * cp.pi * freqs * delay[:, :, None])

n_scatterers = 16384
scatter_pos = cp.random.uniform(
    size=(n_scatterers, 3),
    dtype=cp.float32,  # type: ignore
)

img = Image.new("L", (64, 96))
ImageDraw.Draw(img).text(
    (img.width // 2 - 3, img.height // 2), "ff", fill=1, anchor="mm", font_size=56
)
grid_values = cp.array(img, dtype=cp.float32)
# scatter_values = (cp.abs(scatter_pos[:, 1] - 0.5) < 0.05) * grid_values[(scatter_pos[:, 2] * (img.height-1)).astype(cp.int32), (scatter_pos[:, 0] * (img.width-1)).astype(cp.int32)]
scatter_values = grid_values[
    (scatter_pos[:, 2] * (img.height - 1)).astype(cp.int32), (scatter_pos[:, 0] * (img.width - 1)).astype(cp.int32)
]
scatter_values = 0.001 * (scatter_values + cp.random.randn(*scatter_values.shape, dtype=cp.float32) * 0.1)

scatter_pos = cp.array([xmin, ymin, zmin], dtype=cp.float32) + scatter_pos * cp.array(
    [xmax - xmin, ymax - ymin, zmax - zmin], dtype=cp.float32
)

tx = ffdas.greens(channel_pos, wavenums, tx_signal, scatter_pos)
rx = ffdas.greens(scatter_pos, wavenums, tx * scatter_values[:, None], channel_pos)
rf = cp.fft.ifft(rx, axis=-1).conj()
rf = rf[:, :, None, :]


# spherical reconstruction grid
nr, ntheta, nphi = 96, 96, 96
r = cp.linspace(zmin + 0.005, zmax, nr, dtype=cp.float32)
theta = cp.linspace(-0.5, 0.5, ntheta, dtype=cp.float32)
phi = cp.linspace(-0.5, 0.5, nphi, dtype=cp.float32)

rr, tt, pp = cp.meshgrid(r, theta, phi, indexing="ij")  # type: ignore
grid_points = cp.stack(
    [
        rr * cp.sin(tt),
        rr * cp.sin(pp),
        rr * cp.cos(tt) * cp.cos(pp),
    ],
    axis=-1,
)

ks = sampling_freq / sound_speed
offsets = ffdas.cdist(source, grid_points) * ks
weights = cp.ones_like(offsets)

spherical = ffdas.das(
    rf,
    channel_pos * ks,
    grid_points * ks,
    offsets,
    weights,
    wavenum=-2 * cp.pi * center_freq / sampling_freq,
)

envelope = cp.abs(spherical)  # (batch, nr, ntheta, nphi)


# the Interpolator builds a lookup structure over the grid vertices
# and can be reused for multiple query point sets
interp = ffdas.Interpolator(grid_points)

cart_nz, cart_ny, cart_nx = 128, 128, 128
cx = cp.linspace(xmin, xmax, cart_nx, dtype=cp.float32)
cy = cp.linspace(ymin, ymax, cart_ny, dtype=cp.float32)
cz = cp.linspace(zmin, zmax, cart_nz, dtype=cp.float32)
czz, cyy, cxx = cp.meshgrid(cz, cy, cx, indexing="ij")  # type: ignore
cart_points = cp.stack([cxx, cyy, czz], axis=-1)

with ffdas.utils.Timer() as t:
    cartesian = interp(envelope, cart_points, fill=0.0)
print(f"interpolate to {cart_nz}x{cart_ny}x{cart_nx}: {t.elapsed_ms():.1f} ms")


# the spherical volume in index space: a max projection over phi
# shows the Cartesian grid warped by the curvilinear coordinates.
# after interpolation to Cartesian, the grid is recovered
sph_mip = envelope[0].max(axis=2)  # (nr, ntheta)
db_sph = 20 * cp.log10(sph_mip / sph_mip.max() + 1e-10)

cart_mip = cartesian[0].max(axis=1)  # (cart_nz, cart_nx)
db_cart = 20 * cp.log10(cart_mip / cart_mip.max() + 1e-10)

fig, axes = plt.subplots(1, 2, figsize=(10, 6))

axes[0].imshow(
    cp.asnumpy(db_sph),
    extent=[float(theta[0]), float(theta[-1]), float(r[-1]) * 1e3, float(r[0]) * 1e3],
    cmap="gray",
    vmin=-40,
    vmax=0,
    aspect="auto",
)
axes[0].set_box_aspect(1)
axes[0].set_xlabel("θ [rad]")
axes[0].set_ylabel("r [mm]")
axes[0].set_title("spherical grid (warped)")

axes[1].imshow(
    cp.asnumpy(db_cart),
    extent=[float(cx[0]) * 1e3, float(cx[-1]) * 1e3, float(cz[-1]) * 1e3, float(cz[0]) * 1e3],
    cmap="gray",
    vmin=-40,
    vmax=0,
    aspect="equal",
)
axes[1].set_xlabel("x [mm]")
axes[1].set_ylabel("z [mm]")
axes[1].set_title("Cartesian (interpolated)")

plt.tight_layout()
plt.savefig("interpolation.png", dpi=150)
