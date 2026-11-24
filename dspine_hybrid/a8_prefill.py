"""Int8-activation prefill through oMLX's native kernels (experimental, M5 and later).

On Apple's M5 GPUs the tensor unit multiplies int8 at about twice the bf16 rate. oMLX ships
kernels (``omlx.custom_kernels.quant_matmul``, Apache-2.0, a compiled extension) that feed the
4-bit weight codes straight into an int8 product and round only the activations; Splash describes
the same idea. This module routes the target's wide prefill projections through them:

* the MLP's gate, up and down projections, and
* the GatedDeltaNet's qkv, z and output projections, and
* optionally (part ``"attn"``) the attention layers' q, k, v and output projections,

for chunks of at least ``min_tokens`` rows. Checks and single-token steps never qualify, so
decoding is unchanged; only the prompt pass is rounded.

Activations get one int8 scale per 64 inputs. oMLX's own default for 4-bit weights is one scale
per row, which on Qwen3.8-27B moved the next-token distribution ten times more (KL 0.024 against
0.002, measured on an M5 Max) for the same speed.

Needs ``omlx`` importable with its kernel extension built; without it nothing changes.
"""

from __future__ import annotations

import dataclasses

import mlx.core as mx

G64 = 1          # oMLX activation mode: one scale per group of 64 inputs
PARTS = ("mlp", "gdn")


@dataclasses.dataclass(frozen=True)
class Config:
    min_tokens: int = 128
    parts: tuple = PARTS
    scale: int = G64


_OQ = {"mod": None, "tried": False}
_TAG = "_dspine_a8"
_STOCK = "_dspine_a8_stock_call"
stats = {"int8": 0, "stock": 0}


def kernels():
    """oMLX's int8 prefill module, or None when it or its built kernels are missing."""
    if not _OQ["tried"]:
        _OQ["tried"] = True
        try:
            from omlx.custom_kernels.quant_matmul import fast
            from omlx.patches import qwen35_oq_a8 as oq

            if fast.oq_a8_available():
                _OQ["mod"] = oq
        except Exception:  # oMLX is optional
            _OQ["mod"] = None
    return _OQ["mod"]


def project(linears, x: mx.array, cfg: Config | None):
    """``[linear(x) for linear in linears]`` through the int8 kernels with one shared activation
    quantization, or None when this call does not qualify (the caller then runs them as usual)."""
    oq = _OQ["mod"]
    if (cfg is None or oq is None or x.ndim != 3 or x.shape[-2] < cfg.min_tokens
            or x.dtype not in (mx.bfloat16, mx.float16) or x.shape[-1] % 64):
        return None
    plans = [oq.classify_linear(m) for m in linears]
    if any(p is None for p in plans):
        stats["stock"] += 1
        return None
    stats["int8"] += 1
    stages = {}                      # one activation quantization per weight layout
    for p in plans:
        if p.layout not in stages:
            stages[p.layout] = oq.stage_a(x, cfg.scale, p.layout)
    return [oq.apply_plan(m, stages[p.layout], dataclasses.replace(p, act_mode=cfg.scale)) for m, p in zip(linears, plans)]


def _mlp_call(stock):
    def call(self, x, *args, **kwargs):
        cfg = getattr(self, _TAG, None)
        if cfg is None or args or kwargs:
            return stock(self, x, *args, **kwargs)
        front = project((self.gate_proj, self.up_proj), x, cfg)
        if front is None:
            return stock(self, x)
        from mlx_lm.models.activations import swiglu

        h = swiglu(front[0], front[1])
        down = project((self.down_proj,), h, cfg)
        return down[0] if down is not None else self.down_proj(h)

    return call


def enable(target, on: bool = True, min_tokens: int = 128, parts=PARTS) -> bool:
    """Switch int8 prefill on or off for ``target``. False when the kernels are not available."""
    cfg = Config(int(min_tokens), tuple(parts)) if on and kernels() is not None else None
    for layer in target.plain_layers():          # the kit's layer hooks hide these from the model
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "gate_proj") and hasattr(mlp, "down_proj"):
            cls = type(mlp)
            if cfg is not None and _STOCK not in vars(cls):
                setattr(cls, _STOCK, cls.__call__)
                cls.__call__ = _mlp_call(getattr(cls, _STOCK))
            setattr(mlp, _TAG, cfg if cfg is not None and "mlp" in cfg.parts else None)
    for mod in getattr(target, "_linear", []):
        if mod is not None:
            mod._box.a8 = cfg if cfg is not None and "gdn" in cfg.parts else None
    for mod in getattr(target, "_attn", []):
        if mod is not None:
            mod._box.a8 = cfg if cfg is not None and "attn" in cfg.parts else None
    target.a8_prefill = cfg
    return cfg is not None
