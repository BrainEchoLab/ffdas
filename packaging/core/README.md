# ffdas-core

Prebuilt shared library for [ffdas](https://pypi.org/project/ffdas/), a
CUDA-accelerated library of delay-and-sum and related primitives for ultrasound
image reconstruction.

This package is an internal dependency of `ffdas` and is installed
automatically when you specify a CUDA extra:

```bash
pip install ffdas[cuda12]   # or ffdas[cuda13]
```

See the [ffdas documentation](https://brainecholab.github.io/ffdas/) for
usage and API reference.
