"""A C-level ffdas error must surface as a Python exception, whatever was imported before ffdas.

On Linux, libffdas.so (<= 0.1.0) exported its statically linked C++ runtime: with CuPy imported
first, the first error the binding raised crashed the interpreter. Each case runs in a fresh
interpreter, so the import order is the one under test and a crash fails the test instead of
ending the test run.
"""
import subprocess
import sys
import textwrap

import pytest

SCRIPT = textwrap.dedent("""
    import {first}
    import {second}
    import cupy as cp
    import ffdas

    args = dict(
        srcpos=cp.zeros((4, 3), dtype="float32"),
        dstpos=cp.zeros((8, 3), dtype="float32"),
        offsets=cp.zeros((2, 8), dtype="float32"),
        weights=cp.ones((2, 8), dtype="float32"),
    )
    try:  # ALG2 has no fp64 path: the C API returns FFDAS_ERROR_UNSUPPORTED_TYPE
        ffdas.das(x=cp.zeros((4, 2, 64), dtype="float64"), **args, algorithm=ffdas.Algorithm.ALG2)
    except RuntimeError as e:
        assert "Unsupported data type" in str(e), e
    else:
        raise AssertionError("no error raised")
    out = ffdas.das(x=cp.zeros((4, 2, 64), dtype="float32"), **args)  # still usable afterwards
    assert out.shape == (8,)
    print("ok")
""")


@pytest.mark.parametrize("first,second", [("cupy", "ffdas"), ("ffdas", "cupy")], ids=["cupy-first", "ffdas-first"])
def test_error_is_raised_not_a_crash(first, second):
    r = subprocess.run([sys.executable, "-c", SCRIPT.format(first=first, second=second)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), (r.returncode, r.stderr[-2000:])
