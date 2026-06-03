import atexit

from ._ffdas import Handle


_handle: Handle | None = None


def get_library_handle() -> Handle:
    global _handle
    if _handle is None:
        _handle = Handle()
        atexit.register(_cleanup)
    return _handle


def _cleanup() -> None:
    global _handle
    if _handle is not None:
        _handle.destroy()
        _handle = None
