from array_api_compat import array_namespace

__all__ = [
    "cdist",
    "rect_dist",
    "angle",
]


def cdist(a, b):
    """Euclidean distance from each point in a to each point in b.

    a: (...A, D), b: (...B, D) -> (*a.shape[:-1], *b.shape[:-1])
    """
    backend = array_namespace(a, b)
    na = a.ndim - 1
    nb = b.ndim - 1
    d = a.shape[-1]
    a = backend.reshape(a, a.shape[:-1] + (1,) * nb + (d,))
    b = backend.reshape(b, (1,) * na + b.shape)
    diff = a - b
    return backend.sqrt(backend.sum(diff * diff, axis=-1))


def rect_dist(points, size):
    """Minimum distance from 3D points to an axis-aligned rectangle in the z=0 plane,
    centered at the origin.

    points: (..., 3)
    size: (2,) — full width and height of the rectangle
    """
    backend = array_namespace(points, size)
    half = size / 2
    dx = backend.clip(backend.abs(points[..., 0]) - half[0], min=0)
    dy = backend.clip(backend.abs(points[..., 1]) - half[1], min=0)
    dz = backend.abs(points[..., 2])
    return backend.sqrt(dx * dx + dy * dy + dz * dz)


def angle(a, b, eps=1e-7):
    """Angle in radians between vectors a and b.

    a: (..., D), b: (..., D) -> (...)
    """
    backend = array_namespace(a, b)
    dot = backend.sum(a * b, axis=-1)
    norm_a = backend.sqrt(backend.sum(a * a, axis=-1))
    norm_b = backend.sqrt(backend.sum(b * b, axis=-1))
    cos_theta = dot / (norm_a * norm_b + eps)
    return backend.acos(backend.clip(cos_theta, min=-1.0, max=1.0))
