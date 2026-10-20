"""Entry and target order for ``ffdas.das_sparse``."""

from array_api_compat import array_namespace

__all__ = [
    "sort_sparse",
]


def sort_sparse(sparse_indices, offsets, weights):
    """Reorder a das_sparse problem so that its cost follows the nonzero (target, event) pairs.

    das_sparse runs 32 targets together (a warp), at the pace of the target with the most nonzero
    entries, and reads fastest when those targets read the same event at the same time. This
    puts each target's nonzero entries first, in ascending event order, and orders the targets by
    those event lists -- keeping their own order within a group, so neighbours stay neighbours.
    Measured on 16 diverging transmits: 1.7-2.5x faster than grid order, and faster than one call
    per transmit. Compute it once per geometry.

    sparse_indices, offsets, weights: (n, targets), as passed to das_sparse (flatten the spatial
    dimensions first). Returns ``(order, sparse_indices, offsets, weights)``: the (targets,)
    permutation and the three arrays reordered. Pass ``dstpos[order]`` with them, and put the
    result back with ``image[..., order] = result``.
    """
    xp = array_namespace(sparse_indices, offsets, weights)
    n, n_targets = sparse_indices.shape
    # an entry that weighs 0 is padding: it sorts after every event
    key = xp.where(weights != 0, sparse_indices, xp.max(sparse_indices) + 1)
    entries = xp.argsort(key, axis=0, stable=True)
    key = xp.take_along_axis(key, entries, axis=0)
    # stable sorts from the last slot to the first: a lexicographic sort of the event lists
    order = xp.arange(n_targets)
    for slot in range(n - 1, -1, -1):
        order = xp.take(order, xp.argsort(xp.take(key[slot], order), stable=True))
    entries = xp.take(entries, order, axis=1)
    return (order,) + tuple(xp.take_along_axis(xp.take(a, order, axis=1), entries, axis=0)
                            for a in (sparse_indices, offsets, weights))
