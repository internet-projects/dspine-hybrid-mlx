"""Optional verify-shape Metal kernels for the target check (experimental).

Stock MLX's quantized matmul is tuned for one row (decoding) and for many rows (prefill). A
verify pass multiplies a few rows (1 + drafted tokens) against every quantized weight, and that
shape pays a steep per-row price: on an M5 Max a 4-bit Qwen3.8-27B check costs about 34 ms for
one row, 42 ms for 5 and 89 ms for 16. oMLX ships Metal kernels written for exactly this shape
(``omlx.patches.qwen35_verify_qmm``, Apache-2.0, adapted there from MTPLX, dflash-mlx and
Splash). They read each weight once for all rows, so wide checks get much cheaper.

This module does not copy those kernels. If the ``omlx`` package is importable it applies oMLX's
``nn.QuantizedLinear`` routing and arms it only while a verify (or, optionally, a draft) forward
is being built; everything else in the process keeps the stock path. Without ``omlx`` it is a
no-op.

The kernels sum in a different order than stock MLX, so logits differ in the last bits. Every
token is still the target's own argmax under these kernels, but at a near-tie it can differ
from plain decoding (the same class of difference as block versus single-step passes).
"""

from __future__ import annotations

from contextlib import contextmanager

_STATE = {"mod": None, "tried": False}


def install() -> bool:
    """Apply oMLX's verify routing once. False if oMLX or its kernels are not available."""
    if not _STATE["tried"]:
        _STATE["tried"] = True
        try:
            from omlx.patches import qwen35_verify_qmm as mod

            if mod.apply_verify_qmm_patch():
                _STATE["mod"] = mod
        except Exception:  # oMLX is optional
            _STATE["mod"] = None
    return _STATE["mod"] is not None


def available() -> bool:
    return _STATE["mod"] is not None


@contextmanager
def armed(on: bool = True):
    """Route quantized projections built inside the block through the verify kernels."""
    mod = _STATE["mod"] if on else None
    if mod is None:
        yield False
        return
    mod.set_verify_qmm_armed(True)
    try:
        yield True
    finally:
        mod.set_verify_qmm_armed(False)


def routes(model, rows: int, dtype) -> dict[str, int]:
    """How many of ``model``'s quantized projections would leave stock qmm at ``rows`` rows."""
    import mlx.nn as nn

    mod = _STATE["mod"]
    out: dict[str, int] = {}
    for _, m in model.named_modules():
        if not isinstance(m, nn.QuantizedLinear):
            continue
        route = None
        if mod is not None and getattr(m, "mode", "affine") == "affine":
            K = m.weight.shape[-1] * 32 // m.bits
            route = mod._verify_route(rows, K, m.scales.shape[0], m.bits, m.group_size, dtype)
        out[route or "stock"] = out.get(route or "stock", 0) + 1
    return out

