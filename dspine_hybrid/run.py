"""Load a hybrid target + DSpine draft, with an MLX buffer-cache cap and memory limit applied."""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn

from dspine_mlx.generate import DSpineGenerator
from dspine_mlx.model import quantize_draft

from .draft import load_draft
from .hybrid_target import HybridTargetAdapter

GB = 1 << 30

# Verify block used when none is given. On Apple GPUs a quantized 27B forward costs about the
# same for 1-5 tokens and then steps up (M5 Max: q4 31 ms at T=1, 42 ms at T=5, 88 ms at T=16),
# so short blocks win there even though fewer tokens are accepted per round. The bf16 target
# keeps the drafter's trained block of 16.
QUANTIZED_BLOCK = 5


def default_block_size(model) -> int | None:
    quantized = any(isinstance(m, nn.QuantizedLinear) for _, m in model.named_modules())
    return QUANTIZED_BLOCK if quantized else None


def apply_memory_rules(memory_limit_gb: float | None = None) -> None:
    """2 GB buffer cache cap (cleared per request by callers) + memory limit."""
    mx.set_cache_limit(int(float(os.environ.get("DSPINE_CACHE_GB", "2")) * GB))
    lim = memory_limit_gb or float(os.environ.get("DSPINE_MEMORY_LIMIT_GB", "60"))
    mx.set_memory_limit(int(lim * GB))


def load(target: str, draft: str, *, draft_bits: int | None = None, block_size=None):
    d, cfg, window = load_draft(draft)
    if draft_bits:
        d = quantize_draft(d, bits=int(draft_bits), group_size=64, mode="affine")
    t = HybridTargetAdapter.load(target, cfg.target_layer_ids)
    d.bind(t.model)
    if block_size is None:
        block_size = default_block_size(t.model)
    gen = DSpineGenerator(t, d, block_size=block_size)
    return gen, t, cfg, window
