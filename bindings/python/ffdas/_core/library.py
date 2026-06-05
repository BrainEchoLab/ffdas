import atexit

from ._ffdas import Handle, device_get, device_set


_handles: dict[int, Handle | None] = {}


def get_cuda_device() -> int:
    return device_get()


def set_cuda_device(device: int):
    device_set(device)


def get_library_handle() -> Handle:
    device = device_get()
    h = _handles.get(device)
    if h is None:
        h = Handle()
        _handles[device] = h
    return h


@atexit.register
def _cleanup() -> None:
    for device in _handles.keys():
        h = _handles.get(device)
        if h is not None:
            h.destroy()
        _handles[device] = None
