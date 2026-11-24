"""Small-support target filtering for sampled decoding.

Upstream ``filter_target_probs`` ranks the whole vocabulary (two argsorts over
248K ids per verified row) for every top-k / top-p request. Only the ids that
can survive the filter matter:

* top-k: the k most likely ids (``argpartition`` + a sort of k values);
* top-p alone: the most likely ids whose mass reaches top_p, found in the
  top ``SUPPORT`` ids (full fallback when they don't reach it);
* no top-k / top-p (temperature and min-p only): no ranking at all.

The kept ids, their order, the prefix sums and the dense normalisation are the
same as upstream's, so the filtered distribution is the same. Two float-level
differences remain: which of several exactly equal probabilities wins a top-k
tie, and with top_p = 1 we keep the near-zero tail ids that upstream drops when
its float32 prefix sum rounds past 1.0.
"""

from __future__ import annotations

import mlx.core as mx

SUPPORT = 1024


from dspine_mlx.propose import filter_target_probs as _upstream  # noqa: E402  (kept before install())


def _from_support(probs, ids, top_k, top_p, min_p):
    """Filter using the candidate ids ``ids`` [rows, M] (all ids that may survive)."""
    rows, V = probs.shape
    p = mx.take_along_axis(probs, ids, axis=-1)
    order = mx.argsort(-p, axis=-1)
    ids = mx.take_along_axis(ids, order, axis=-1)
    sp = mx.take_along_axis(p, order, axis=-1)
    before = mx.cumsum(sp, axis=-1) - sp
    rank = mx.arange(sp.shape[-1])[None]
    keep = (rank < int(top_k)) & (before <= float(top_p))
    if min_p > 0:
        keep = keep & (sp >= float(min_p) * mx.max(probs, axis=-1, keepdims=True))
    w = mx.put_along_axis(mx.zeros((rows, V), dtype=probs.dtype), ids, mx.where(keep, sp, 0.0), axis=-1)
    return w / mx.sum(w, axis=-1, keepdims=True)


def filter_target_probs(probs: mx.array, top_k: int, top_p: float, min_p: float) -> mx.array:
    rows, V = probs.shape
    if V == 0:
        return probs
    k = V if top_k <= 0 else min(int(top_k), V)
    if k >= V and top_p >= 1.0:
        if min_p <= 0:
            return probs / mx.sum(probs, axis=-1, keepdims=True)
        w = mx.where(probs >= float(min_p) * mx.max(probs, axis=-1, keepdims=True), probs, 0.0)
        return w / mx.sum(w, axis=-1, keepdims=True)
    if k < V:
        ids = mx.argpartition(-probs, k - 1, axis=-1)[:, :k]
        return _from_support(probs, ids, k, top_p, min_p)
    M = min(SUPPORT, V)
    ids = mx.argpartition(-probs, M - 1, axis=-1)[:, :M]
    # every id outside the top M sits after at least their mass; drop them only if that mass
    # already exceeds top_p in every row (otherwise rank the whole vocabulary as upstream does)
    mass = mx.sum(mx.take_along_axis(probs, ids, axis=-1), axis=-1)
    if M < V and bool(mx.all(mass > float(top_p) + 1e-6).item()):
        return _from_support(probs, ids, V, top_p, min_p)
    return _upstream(probs, k, top_p, min_p)

_installed = False


def install() -> None:
    """Use the small-support filter in dspine_mlx's sampler and generator."""
    global _installed
    if _installed:
        return
    import dspine_mlx.generate as g
    import dspine_mlx.propose as p
    import dspine_mlx.sampler as s

    for mod in (g, p, s):
        if hasattr(mod, "filter_target_probs"):
            mod.filter_target_probs = filter_target_probs
    _installed = True
