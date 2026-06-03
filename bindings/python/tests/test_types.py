import cupy as cp
import pytest

import ffdas


SCALAR_DTYPES = {
    "half": ffdas.half,
    "float32": ffdas.float32,
    "float64": ffdas.float64,
    "int16": ffdas.int16,
    "int32": ffdas.int32,
}

PACKED_DTYPES = {
    "complex64": ffdas.complex64,
    "complex128": ffdas.complex128,
    "half2": ffdas.half2,
    "float2": ffdas.float2,
    "double2": ffdas.double2,
    "short2": ffdas.short2,
    "int2": ffdas.int2,
}

ALL_DTYPES = {**SCALAR_DTYPES, **PACKED_DTYPES}


# dtype properties

@pytest.mark.parametrize("name, dt", ALL_DTYPES.items())
def test_dtype_name(name, dt):
    assert dt.name == name


@pytest.mark.parametrize("name, dt", ALL_DTYPES.items())
def test_dtype_repr_contains_name(name, dt):
    assert name in repr(dt)


def test_dtype_hashable():
    d = {ffdas.float32: "f32", ffdas.float64: "f64"}
    assert d[ffdas.float32] == "f32"


# enums

def test_algorithm_values():
    assert ffdas.Algorithm.DEFAULT.value == 0
    assert ffdas.Algorithm.ALG1.value == 1


def test_interp_mode_values():
    assert ffdas.InterpMode.NEAREST.value == 0
    assert ffdas.InterpMode.LINEAR.value == 1


# TensorView

def test_view_float2_shape():
    base = cp.zeros((4, 3, 2), dtype="float32")
    v = ffdas.view(base, ffdas.float2)
    assert v.dtype == ffdas.float2
    assert v.shape == (4, 3)


def test_view_preserves_base():
    base = cp.ones((8, 2), dtype="float32")
    v = ffdas.view(base, ffdas.float2)
    assert v.base is base


def test_view_dlpack_device():
    base = cp.zeros((4, 2), dtype="float32")
    v = ffdas.view(base, ffdas.float2)
    dev_type, _ = v.__dlpack_device__()
    assert dev_type == 2  # kDLCUDA
