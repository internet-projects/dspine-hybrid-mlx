"""A second drafter pass that knows the first drafted token (experimental, greedy tree rounds).

The DSpine drafter fills its whole block in one pass. Inside the block every row gets a message
from the row before it, but only as a soft code read from that row's hidden state; the real token
is used once, in the refinement after the last layer. On prose the first token is often a toss-up
between two words, and the rows after it are drafted for a blend of both.

The twin pass runs the block again (block only, against the context already in the drafter's
cache) once per chosen first token, and in every layer's write replaces the soft code of the
first drafted row by the codebook code of that token, the same code the anchor row always uses.
The rows after it are then drafted for that token. The tree takes each first token's subtree
from its own pass, so a round offers two coherent continuations instead of one blended one.

Whether the drafter makes good use of a hard code there is an empirical question (it was trained
with soft codes); the output is unchanged either way, since the target checks every node.
"""

from __future__ import annotations

import contextlib
import math

import mlx.core as mx

from dspine_mlx.bridge import _F32_TINY, DSpineCodeBridge

_FORCE = {"ids": None}       # [B] token ids for the first drafted row of the running forward, or None


class TwinBridge(DSpineCodeBridge):
    """``DSpineCodeBridge.__call__`` with the first drafted row's outgoing code forced.

    A copy of the upstream method; with nothing forced it computes the same values. The forced
    code only replaces what row 1 SENDS to row 2 (its predecessor gate and value); the row's own
    gate query and the stash keep the soft code, so its own refinement is untouched.
    """

    def __call__(self, hidden, residual, input_ids, layer, next_norm, stash=None):
        force = _FORCE["ids"]
        if force is None:
            return super().__call__(hidden, residual, input_ids, layer, next_norm, stash)
        B, T, H = hidden.shape
        if hidden.dtype != mx.bfloat16:
            raise ValueError("The DSpine code bridge requires BF16 activations")
        if T < 3 or int(force.shape[0]) != B:
            raise ValueError("a forced first token needs a block of 3+ rows and one token per batch row")
        m = B * T
        sqrt_c = math.sqrt(self.code_size)

        raw = hidden if residual is None else (hidden + residual)
        raw = raw.astype(mx.bfloat16)
        f = raw.astype(mx.float32)
        rms = mx.sqrt(mx.maximum(mx.mean(f * f, axis=-1), _F32_TINY))
        normalized = (f / (rms[..., None] + self.eps)).astype(mx.bfloat16)

        projected = mx.matmul(normalized, self.layer_read_weight(layer).T).astype(mx.float32)
        proj_norm = mx.sqrt(mx.sum(projected * projected, axis=-1, keepdims=True))
        codes = (sqrt_c * projected / (proj_norm + self.eps)).astype(mx.bfloat16)

        anchors = input_ids[:, 0]
        anchor_codes = (sqrt_c * self.codebook[anchors].astype(mx.float32)).astype(mx.bfloat16)
        codes = mx.concatenate([anchor_codes[:, None, :], codes[:, 1:, :]], axis=1)

        packed = mx.matmul(codes, self.packed_weight.T)
        if stash is not None:
            stash.update(raw=raw.reshape(m, H), rms=rms.reshape(m), packed=packed.reshape(m, 3 * self.rank))

        q, s, v = mx.split(packed.astype(mx.float32), [self.rank, 2 * self.rank], axis=-1)
        hard = (sqrt_c * self.codebook[force].astype(mx.float32)).astype(mx.bfloat16)       # [B, C]
        _, s1, v1 = mx.split(mx.matmul(hard, self.packed_weight.T).astype(mx.float32), [self.rank, 2 * self.rank], axis=-1)
        s = mx.concatenate([s[:, :1], s1[:, None, :], s[:, 2:]], axis=1)
        v = mx.concatenate([v[:, :1], v1[:, None, :], v[:, 2:]], axis=1)

        gate = mx.sigmoid(q[:, 1:, :] + s[:, :-1, :] + self.gate.bias.astype(mx.float32))
        message = mx.concatenate([mx.zeros_like(q[:, :1, :]), gate * v[:, :-1, :]], axis=1).astype(mx.bfloat16)
        delta = mx.matmul(message, self.feature_write_weights[layer].T).astype(mx.float32)

        not_anchor = (mx.arange(m).reshape(B, T) % T != 0).astype(mx.float32)[..., None]
        out = (f + (rms[..., None] * delta) * not_anchor).astype(mx.bfloat16)

        x = out.astype(mx.float32)
        y = (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + next_norm.eps)).astype(mx.bfloat16)
        normed = (y.astype(mx.float32) * next_norm.weight.astype(mx.float32)).astype(mx.bfloat16)
        return normed, out


def install(draft) -> None:
    bridge = draft.selected_prefix_bridge
    if type(bridge) is DSpineCodeBridge:
        bridge.__class__ = TwinBridge
    elif not isinstance(bridge, TwinBridge):
        raise TypeError(f"unsupported bridge {type(bridge).__name__}")


@contextlib.contextmanager
def forced(tokens):
    """Run drafter forwards with the first drafted row's outgoing code set to ``tokens`` ([B] ids)."""
    _FORCE["ids"] = mx.array(list(tokens), dtype=mx.int32)
    try:
        yield
    finally:
        _FORCE["ids"] = None


def passes(draft, anchor: int, first_tokens, rows: int, caches):
    """Block-only drafter pass per first token. Returns [(proposal rows [rows, H], stash)] in order.

    ``caches`` must already hold the context (the round's normal pass has just put it there)."""
    B, T = len(first_tokens), rows + 1
    block = mx.array([[int(anchor)] + [draft.mask_token_id] * rows] * B, dtype=mx.int32)
    with forced(first_tokens):
        out, stash = draft(None, block, caches)
    res = []
    for b in range(B):
        sl = slice(b * T, (b + 1) * T)
        res.append((out[b, 1:], {k: stash[k][sl] for k in ("raw", "rms", "packed")}))
    return res
