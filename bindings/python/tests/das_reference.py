"""A float64 numpy reference for ffdas.das / ffdas.das_sparse, written from the kernels' rules.

For every target n, compounded entry i (sequence event i, or the event sparse_indices[i, n]) and
channel m:

    r     = |dstpos[n] - srcpos[m]|                      (squared distance clamped at 1e-7)
    phase = offsets[i, n] + r                            (in samples)
    k     = floor(phase), t = phase - k
    used  = 0 <= k < samples - 1                         (and inside channel m's srcdir cone)
    out  += weights[i, n] * exp(1j * wavenum * phase) * ((1 - t) x[.., m, e, k] + t x[.., m, e, k + 1])

The phase rotation applies to complex data only. srcdir is (channels, 4), or (events, channels, 4)
indexed by the event e. A sample that is not used contributes an exact 0, and so does an entry that
weighs 0 or (sparse) selects an event outside x.
"""

import numpy as np


def das_reference(x, srcpos, dstpos, offsets, weights, *, srcdir=None, wavenum=0.0,
                  sparse_indices=None, channels_trailing=False):
    x = np.asarray(x)
    is_complex = np.iscomplexobj(x)
    x = x.astype(np.complex128 if is_complex else np.float64)
    unbatched = x.ndim == 3
    if unbatched:
        x = x[None]
    if channels_trailing:
        x = x.swapaxes(1, 2)
    _, n_ch, _, n_samp = x.shape

    dstpos = np.asarray(dstpos, dtype=np.float64)
    spatial = dstpos.shape[:-1]
    dst = dstpos.reshape(-1, 3)
    n_tgt = dst.shape[0]
    src = np.asarray(srcpos, dtype=np.float64)

    ofs = np.asarray(offsets, dtype=np.float64).reshape(-1, n_tgt)
    wgt = np.asarray(weights, dtype=np.float64).reshape(-1, n_tgt)
    n_ent = ofs.shape[0]
    if sparse_indices is None:
        event = np.broadcast_to(np.arange(n_ent)[:, None], (n_ent, n_tgt))
    else:
        event = np.asarray(sparse_indices).reshape(n_ent, n_tgt)

    d = dst[None, :, :] - src[:, None, :]                       # (channels, targets, 3)
    r = np.sqrt(np.maximum((d ** 2).sum(-1), 1e-7))             # (channels, targets)
    phase = ofs[:, None, :] + r[None]                           # (entries, channels, targets)
    k = np.floor(phase)
    t = phase - k
    k = k.astype(np.int64)
    used = (k >= 0) & (k < n_samp - 1)
    n_ev = x.shape[2]
    valid = (wgt != 0) & (event >= 0) & (event < n_ev)                # entries that count at all
    used &= valid[:, None, :]
    event = np.where(valid, event, 0)

    weight = np.broadcast_to(wgt[:, None, :], phase.shape).copy()
    if srcdir is not None:
        srcdir = np.asarray(srcdir, dtype=np.float64)
        if srcdir.ndim == 2:
            srcdir = np.broadcast_to(srcdir, (n_ev,) + srcdir.shape)
        axes = srcdir[event]                                         # (entries, targets, channels, 4)
        u = d / r[..., None]                                         # (channels, targets, 3)
        proj = np.einsum("ctk,etck->ect", u, axes[..., :3])
        used &= proj > axes[..., 3].transpose(0, 2, 1)
    if is_complex:
        weight = weight * np.exp(1j * wavenum * phase)

    kc = np.where(used, k, 0)
    m = np.arange(n_ch)[None, :, None]
    e = event[:, None, :]
    s0 = x[:, m, e, kc]                                          # (batch, entries, channels, targets)
    s1 = x[:, m, e, np.minimum(kc + 1, n_samp - 1)]
    sample = s0 + t * (s1 - s0)
    out = np.where(used, weight * sample, 0).sum(axis=(1, 2))   # (batch, targets)

    out = out.reshape((out.shape[0], *spatial))
    return out[0] if unbatched else out
