"""Pipelined GatedDeltaNet recurrence for prefill chunks (experimental, Metal).

48 of the 64 layers of Qwen3.8-27B are GatedDeltaNet layers, whose recurrence walks the chunk
token by token. mlx-lm's kernel runs one step at a time; oMLX ships a kernel
(``omlx.custom_kernels.attention.gated_delta.gated_delta_pipelined``, Apache-2.0, compiled at
run time, no build step) that stages 12 tokens a block and overlaps the steps. Sushi routes its
Qwen prefill through the same kernel and reports +4 to 5% prefill on an M5 Max.

Same per-step math as mlx-lm's kernel; only the order of the float32 sums differs, so the result
is close but not bit-identical. Only chunks of at least ``min_tokens`` rows take it: checks and
single-token steps keep mlx-lm's kernel, so decoding is unchanged.

Needs ``omlx`` importable and Metal; without them nothing changes.
"""

from __future__ import annotations

import dataclasses

import mlx.core as mx
from mlx_lm.models.gated_delta import compute_g


@dataclasses.dataclass(frozen=True)
class Config:
    min_tokens: int = 64


_K = {"mod": None, "tried": False}
stats = {"pipelined": 0, "stock": 0}


def kernels():
    """oMLX's gated-delta kernel module, or None when it or Metal is missing."""
    if not _K["tried"]:
        _K["tried"] = True
        try:
            if mx.metal.is_available() and mx.default_device() == mx.gpu:
                from omlx.custom_kernels.attention import gated_delta as mod

                _K["mod"] = mod
        except Exception:  # oMLX is optional
            _K["mod"] = None
    return _K["mod"]


def update(q, k, v, a, b, A_log, dt_bias, state, cfg: Config | None):
    """(y, state) of the recurrence over a wide chunk, or None when the chunk does not qualify.

    Arguments as mlx-lm's ``gated_delta_update`` after ``normalize_qk`` (no mask)."""
    mod = kernels() if cfg is not None else None
    if mod is None or q.ndim != 4 or q.shape[1] < cfg.min_tokens or a.ndim != 3:
        stats["stock"] += 1
        return None
    if not mod.gated_delta_pipelined_supported(q, k, v):
        stats["stock"] += 1
        return None
    stats["pipelined"] += 1
    return mod.gated_delta_pipelined(q, k, v, compute_g(A_log, a, dt_bias), mx.sigmoid(b), state)
