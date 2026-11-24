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


# Drafter backbone bits when none are given. A 4-bit drafter proposes almost the same tokens and
# runs faster; the output is still the target's own (the target verifies every token).
DEFAULT_DRAFT_BITS = 4
FAST_DRAFT_WINDOW = 2048   # drafter context with the verify kernels on


def load(target: str, draft: str, *, draft_bits: int | None = DEFAULT_DRAFT_BITS, block_size=None,
         adaptive: bool = False, alpha: float = 1.0, prompt_cache: bool = True,
         confidence: str = "vocab", calibration: str | None = None,
         copy: bool = False, copy_min: int = 12, tree: bool = False, fast_verify: bool = False, long_attn: bool = False, a8_prefill: bool = False,
         gdn_prefill: bool = False, twin: int = 0, twin_below: float = 1.0,
         draft_window: int | None = None, live_cost: bool | None = None, cost: list | None = None):
    """Return (generator, target adapter, draft config, window).

    draft_bits: 4 or 8 quantizes the drafter backbone, 0/None keeps it bf16.
    adaptive: choose the verify length every round from the drafter's confidence and this
    machine's measured target cost (greedy only; sampling uses the fixed block).
    confidence: with adaptive, "vocab" (drafter head probability, alpha applies) or "calibrated"
    (Tencent's per-depth acceptance maps for Qwen3.8-27B, which also makes sampling adaptive;
    ``calibration`` = optional JSON in their
    format with your own fit).
    copy: propose context copies after ``copy_min`` matching tokens instead of running the drafter.
    tree: verify a best-first tree of the drafter's candidates instead of one chain (greedy).
    fast_verify: use oMLX's verify-shape Metal kernels for the target check and the drafter when the
    ``omlx`` package is installed (see dspine_hybrid.fast_verify); ``target.fast_verify`` says if it took.
    long_attn: with fast_verify on Metal, checks of 6 to 15 rows at 16K+ cached tokens use the multi-row
    attention kernel (see dspine_hybrid.long_attn); off by default until measured end to end.
    a8_prefill: read the prompt through oMLX's int8-activation kernels when they are built (see
    dspine_hybrid.a8_prefill): about 1.3x faster prefill on an M5 Max, logits move slightly (KL ~0.002).
    twin: with tree, run the drafter a second time for this many most likely first tokens, each pass
    knowing its first token (see dspine_hybrid.twin), when the best one's chance is below twin_below;
    off by default until measured.
    gdn_prefill: run the GatedDeltaNet recurrence of prefill chunks through oMLX's pipelined kernel
    (see dspine_hybrid.gdn_prefill); off by default until measured.
    draft_window: let the drafter see only this many latest tokens (default with fast_verify: 2048,
    measured on the M5 Max at 8k context: 42 -> 49.6 tok/s; otherwise the drafter's own 8192).
    live_cost (default: on with fast_verify): adaptive rounds add what attention costs at the prompt's
    length to the cost table (a many-row check is much dearer at 32k than on a short prompt).
    cost: the target cost table of an earlier ``adaptive.measure_cost`` on this machine and setup (it is
    measured here when not given, a few seconds).
    copy and tree keep the output unchanged; with neither, the generators are the ones v0.3 was measured with.
    prompt_cache: keep the target cache between ``generate`` calls and prefill only the part
    of the next prompt that is new (send the whole conversation each turn, as usual).
    """
    from .fast_sampling import install as _fast_sampling

    _fast_sampling()  # small-support top-k / top-p for sampled decoding
    d, cfg, window = load_draft(draft)
    if draft_bits:
        d = quantize_draft(d, bits=int(draft_bits), group_size=64, mode="affine")
    t = HybridTargetAdapter.load(target, cfg.target_layer_ids)
    d.bind(t.model)
    fast = bool(fast_verify and t.enable_fast_verify(True))
    if fast:
        d.fast_kernels = True
        t.enable_flush(8)                 # measured on the M5 Max: 16-row check 49 -> 46 ms
        if long_attn:
            t.enable_long_attention(True)
        if draft_window is None:
            draft_window = FAST_DRAFT_WINDOW
    if a8_prefill:
        t.enable_a8_prefill(True)       # int8 prefill, needs oMLX's built kernels; rounds activations
    if gdn_prefill:
        t.enable_gdn_prefill(True)
    if draft_window is not None and (window is None or int(draft_window) < window):
        from .draft import set_window

        window = int(draft_window)
        set_window(d, window)
    if block_size is None:
        # with the kernels a 16-row check costs about what 5 rows cost, so the full block pays
        block_size = 16 if fast else default_block_size(t.model)
    if live_cost is None:
        live_cost = fast
    if copy or tree or (live_cost and adaptive):
        from . import calibration as cal
        from .adaptive import measure_cost
        from .lanes import LaneDSpineGenerator

        calibrated = confidence == "calibrated"
        gen = LaneDSpineGenerator(
            t, d, block_size=block_size, adaptive=adaptive,
            cost=(cost or measure_cost(t, list(range(1000, 1256)))) if adaptive else None, alpha=alpha,
            confidence=confidence, calibration=cal.load(calibration) if calibrated else None,
            sampled_calibration=cal.load(calibration, "sampled") if calibrated else None,
            copy=copy, copy_min=copy_min, tree=tree, live_cost=live_cost, twin=twin, twin_below=twin_below)
    elif adaptive:
        from .adaptive import AdaptiveDSpineGenerator, measure_cost

        cost = cost or measure_cost(t, list(range(1000, 1256)))
        from . import calibration as cal

        calibrated = confidence == "calibrated"
        gen = AdaptiveDSpineGenerator(t, d, cost=cost, alpha=alpha, block_size=16, fixed_block_size=block_size,
                                      confidence=confidence,
                                      calibration=cal.load(calibration) if calibrated else None,
                                      sampled_calibration=cal.load(calibration, "sampled") if calibrated else None)
    else:
        gen = DSpineGenerator(t, d, block_size=block_size)
    t.enable_prompt_cache(prompt_cache)
    return gen, t, cfg, window
