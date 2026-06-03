"""Acoustic simulation using Green's function propagation.

For each target point, the greens function sums exp(i*k*r)/r 
over all sources in the frequency domain. This enables fast 
acoustic simulations with (tens of) thousands of sources and 
receivers on the GPU.

After simulating, a receive-only DAS reconstruction (like in 
photoacoustics) recovers the source distribution.
"""

import math

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

import ffdas


cp.random.seed(0)

sound_speed = 1540.0
center_freq = 5e6
sampling_freq = center_freq
n_samples = 512

# frequency-domain setup for complex baseband simulation.
# fftfreq returns bins centered at DC; adding center_freq shifts them
# to physical frequencies around the carrier. the resulting time-domain
# signal after ifft is the complex analytic signal at baseband rate
freqs = cp.fft.fftfreq(n_samples, d=1.0 / sampling_freq) + center_freq

# negative wavenumber so that the kernel's exp(i*k*r)/r gives the
# outgoing-wave Green's function exp(-i*2*pi*f*r/c)/r
wavenums = (-2 * math.pi * freqs / sound_speed).astype(cp.float32)

# broadband gaussian pulse
sigma_f = 0.6 * center_freq / (2 * math.sqrt(2 * math.log(2)))
pulse = cp.exp(-0.5 * ((freqs - center_freq) / sigma_f) ** 2).astype(cp.complex64)

# 8192 point sources distributed in a 3D volume
n_sources = 8192
sources = cp.zeros((n_sources, 3), dtype=cp.float32)
sources[:, 0] = cp.random.uniform(-0.005, 0.005, n_sources).astype(cp.float32)
sources[:, 1] = cp.random.uniform(-0.005, 0.005, n_sources).astype(cp.float32)
sources[:, 2] = cp.random.uniform(0.015, 0.025, n_sources).astype(cp.float32)

# 32x32 matrix receiver array
pitch = 150e-6
el = (cp.arange(32, dtype=cp.float32) - 15.5) * pitch
rx, ry = cp.meshgrid(el, el, indexing="ij")  # type: ignore
receiver_pos = cp.stack(
    [rx.ravel(), ry.ravel(), cp.zeros(1024, dtype=cp.float32)], axis=-1
)

# random source amplitudes per batch frame
batch_size = 32
amplitudes = cp.random.randn(batch_size, n_sources, 1).astype(cp.complex64)
x = pulse[None, None, :] * amplitudes  # (batch, sources, frequencies)

with ffdas.utils.Timer() as t:
    received = ffdas.greens(x, sources, receiver_pos, wavenums)
print(
    f"greens: {n_sources} sources, {receiver_pos.shape[0]} receivers, "
    f"batch {batch_size}: {t.elapsed_ms():.1f} ms"
)

# convert to time domain; conj matches the sign convention used by das
rf = cp.fft.ifft(received, axis=-1).astype(cp.complex64).conj()
# rf: (batch, n_receivers, n_samples)


# receive-only reconstruction (e.g. photoacoustic imaging): the sources
# emit directly, so the transmit offset is zero and DAS uses only the
# one-way receive delay from each voxel to each channel
nz, ny, nx = 64, 64, 64
gx = cp.linspace(-0.010, 0.010, nx, dtype=cp.float32)
gy = cp.linspace(-0.010, 0.010, ny, dtype=cp.float32)
gz = cp.linspace(0.010, 0.030, nz, dtype=cp.float32)
gzz, gyy, gxx = cp.meshgrid(gz, gy, gx, indexing="ij")  # type: ignore
voxel_pos = cp.stack([gxx, gyy, gzz], axis=-1)

k = sampling_freq / sound_speed
offsets = cp.zeros((1, nz, ny, nx), dtype=cp.float32)
weights = cp.ones((1, nz, ny, nx), dtype=cp.float32)

with ffdas.utils.Timer() as t:
    image = ffdas.das(
        rf[:, :, None, :],
        receiver_pos * k,
        voxel_pos * k,
        offsets,
        weights,
        wavenum=-2 * math.pi * center_freq / sampling_freq,
    )
print(
    f"das: {nz}x{ny}x{nx} grid, batch {batch_size}: {t.elapsed_ms():.1f} ms"
)


envelope = cp.asnumpy(cp.abs(rf[0]))

mip = cp.asnumpy(cp.abs(image[0]).max(axis=1))  # max over y
db = 20 * np.log10(mip / mip.max() + 1e-10)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

axes[0].imshow(
    envelope.T,
    aspect="auto",
    cmap="hot",
    extent=[0, receiver_pos.shape[0], n_samples, 0],
)
axes[0].set_xlabel("receiver")
axes[0].set_ylabel("sample")
axes[0].set_title(f"received signal ({n_sources} sources)")

axes[1].imshow(
    db,
    extent=[float(gx[0]) * 1e3, float(gx[-1]) * 1e3,
            float(gz[-1]) * 1e3, float(gz[0]) * 1e3],
    cmap="gray",
    vmin=-40,
    vmax=0,
    aspect="equal",
)
axes[1].set_xlabel("x [mm]")
axes[1].set_ylabel("z [mm]")
axes[1].set_title("xz max projection")

plt.tight_layout()
plt.savefig("simulation.png", dpi=150)
