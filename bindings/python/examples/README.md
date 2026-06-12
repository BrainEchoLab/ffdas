# Examples

Self-contained scripts demonstrating each part of the ffdas pipeline. Each example simulates synthetic data, reconstructs an image, and saves the result as a PNG.

The examples use conservative data sizes by default so they run on most CUDA-capable GPUs. If your GPU has more memory, increase `batch_size`, grid resolution (`nz`, `ny`, `nx`), and `n_scatterers` to see what the library can do at scale.

## Scripts

- **`reconstruct.py`** — Single plane-wave reconstruction. Simulates diverging-wave channel data with `greens`, then reconstructs a 3D volume with `das`.
- **`compound.py`** — Coherent plane-wave compounding. Transmits at several steering angles and combines the resulting images with `das`.
- **`sparse_compounding.py`** — Sparse compounding with `das_sparse`. Diverging waves from virtual sources at different angles each illuminate only part of the volume. Each voxel selects the relevant transmit events, avoiding iteration over zero-weight contributions.
- **`clutter_filter.py`** — Clutter filtering. Reconstructs a slow-time sequence, then separates stationary tissue from moving flow using `truncate_rank`.
- **`interpolation.py`** — Grid interpolation. Reconstructs on a spherical grid, then interpolates back to Cartesian coordinates.

## Requirements

All examples require CuPy and matplotlib:

```bash
pip install cupy-cuda12x matplotlib
```

The `interpolation.py` example also requires Pillow (`pip install Pillow`).
