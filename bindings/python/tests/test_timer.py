import cupy as cp
import pytest

from ffdas.utils import Timer


def test_context_manager():
    with Timer() as t:
        pass
    ms = t.elapsed_ms()
    assert isinstance(ms, float)
    assert ms >= 0.0


def test_elapsed_before_stop_raises():
    with pytest.raises(RuntimeError):
        Timer().elapsed_ms()


def test_double_start_raises():
    t = Timer()
    t.start()
    with pytest.raises(RuntimeError):
        t.start()
    t.stop()


def test_manual_start_stop():
    t = Timer()
    t.start()
    t.stop()
    assert t.elapsed_ms() >= 0.0


def test_measures_nonzero_work():
    with Timer() as t:
        a = cp.ones((4096, 4096), dtype="float32")
        cp.matmul(a, a)
    assert t.elapsed_ms() > 0.0
