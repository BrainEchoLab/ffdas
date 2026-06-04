from .._core import _ffdas
from .._core.library import get_library_handle


class Timer:
    """Context manager for timing GPU operations using CUDA events.

    Records events on the library's internal stream before and after
    the wrapped block, then synchronizes the stream and reports elapsed 
    GPU time.

    Example::

        with Timer() as t:
            output = ffdas.das(rf, channel_pos, voxel_pos, offsets, weights)
        print(f"{t.elapsed_ms():.1f} ms")
    """

    def __init__(self):
        self._handle = get_library_handle()
        self._start = _ffdas.event_create()
        self._stop = _ffdas.event_create()
        self._running = False
        self._elapsed = None

    def start(self):
        if self._running:
            raise RuntimeError("Timer is already running")
        self._running = True
        _ffdas.event_record(self._handle, self._start)

    def stop(self):
        if not self._running:
            return
        _ffdas.event_record(self._handle, self._stop)
        _ffdas.event_synchronize(self._stop)
        self._elapsed = _ffdas.event_elapsed_time(self._start, self._stop)
        self._running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    def elapsed_ms(self):
        if self._elapsed is None:
            raise RuntimeError("Timer has not been stopped yet")
        return self._elapsed

    def __del__(self):
        try:
            if self._start:
                _ffdas.event_destroy(self._start)
                self._start = 0
            if self._stop:
                _ffdas.event_destroy(self._stop)
                self._stop = 0
        except Exception:
            pass
