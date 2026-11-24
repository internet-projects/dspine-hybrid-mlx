"""Multi-row attention kernel for target checks at long context (experimental, Metal only).

At 32K cached tokens MLX's attention gets expensive for a check of 6 to 15 rows (2.1 to 3.0 ms
per layer on an M5 Max, against 1.1 ms for 5 rows and 1.5 ms for 16). mlx-dspark's multi-row
kernel (``multirow_attn.py``, MIT, vendored unchanged) reads each key tile once for all rows and
costs 1.0 to 2.2 ms there. It loses for one row and for 16, and makes little difference at 8K, so
it is used only inside a window: rows ``min_q..max_q`` and at least ``min_kv`` cached keys.

``DSPINE_LONG_ATTN=0`` turns it off; ``DSPINE_LONG_ATTN=6,15,16384`` sets the window.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import mlx.core as mx

DEFAULT_WINDOW = (6, 15, 16384)
# The tree check keeps MLX's masked attention: measured on the M5 Max the kernel's tree variant was
# no faster at 32K and 6 ms per check slower at 16K. DSPINE_LONG_ATTN_TREE=1 routes it anyway.
TREE = os.environ.get("DSPINE_LONG_ATTN_TREE", "0") == "1"


def config(attn, window=None):
    """A gate for ``attn``'s shape, or None (no Metal, switched off, or an unsupported shape)."""
    env = os.environ.get("DSPINE_LONG_ATTN", "")
    if env == "0" or attn is None or not mx.metal.is_available():
        return None
    if window is None:
        window = tuple(int(x) for x in env.split(",")) if env.count(",") == 2 else DEFAULT_WINDOW
    hq, hk, d = attn.num_attention_heads, attn.num_key_value_heads, attn.head_dim
    if d % 8 or d > 256 or hq % hk:
        return None
    from . import multirow_attn as mr

    return mr.MultirowConfig((mr.Window(hq, hk, d, *window),))


@contextmanager
def active(cfg):
    """Route attention calls built inside the block through the kernel where the gate allows."""
    if cfg is None:
        yield False
        return
    from . import multirow_attn as mr

    with mr.multirow_attention(cfg):
        yield True


def tree_bits(ancestors: mx.array) -> mx.array | None:
    """uint32 per tree row: bit j set when row j is the row itself or one of its ancestors."""
    m = ancestors.shape[0]
    if m > 32:
        return None
    w = mx.array([1 << j for j in range(m)], dtype=mx.uint32)
    return (ancestors.astype(mx.uint32) * w).sum(axis=1).astype(mx.uint32)


def tree_attention(cfg, q, k, v, scale, bits):
    """The tree check's attention through the kernel, or None when the gate says stock."""
    if cfg is None or bits is None:
        return None
    from . import multirow_attn as mr

    if not mr.eligible(q, k, v, None, None, cfg):
        return None
    return mr.multirow_sdpa(q, k, v, scale=scale, causal=False, tree_anc=bits)
