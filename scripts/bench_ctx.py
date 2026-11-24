"""Why a many-row check costs more at long context: the check split by layer kind, per context.

For a short, an 8k and a 32k prompt it times the target check of 1..16 rows three ways: as the kit
runs it (verify kernels on), with a sync around every block so the time splits into attention,
recurrent (GatedDeltaNet) and MLP parts, and with the kernels off. Two micro-benchmarks on arrays
of the same shape follow: the attention product alone, and writing T new rows into the key/value
buffer. Takes about four minutes (the 32k prefill is most of it).

    python scripts/bench_ctx.py --target <4-bit build> --draft <drafter> --out ctx.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402

from dspine_hybrid.hybrid_target import _inner_layers  # noqa: E402
from dspine_hybrid.run import apply_memory_rules, load  # noqa: E402

ROWS = (1, 2, 4, 5, 6, 8, 9, 12, 15, 16)


def med(xs):
    return round(1000 * sorted(xs)[len(xs) // 2], 1)


class Split:
    """Wraps the three block kinds of the target so each call is timed on its own (with a sync)."""

    def __init__(self, layers):
        self.on = False
        self.t = dict(attention=0.0, recurrent=0.0, mlp=0.0)
        seen = {}
        for layer in layers:
            for kind, name in (("attention", "self_attn"), ("recurrent", "linear_attn"), ("mlp", "mlp")):
                mod = getattr(layer, name, None)
                if mod is not None and type(mod) not in seen:
                    seen[type(mod)] = kind
        for cls, kind in seen.items():
            inner = cls.__call__

            def call(mod, x, *args, _inner=inner, _kind=kind, **kw):
                if not self.on:
                    return _inner(mod, x, *args, **kw)
                mx.eval(x)
                t0 = time.perf_counter()
                out = _inner(mod, x, *args, **kw)
                mx.eval(out)
                self.t[_kind] += time.perf_counter() - t0
                return out

            cls.__call__ = call


def check(target, tok, T, reps, split=None):
    ts, parts = [], []
    for r in range(reps + 1):
        if split is not None:
            split.t = dict.fromkeys(split.t, 0.0)
            split.on = True
        t0 = time.perf_counter()
        lg, h = target.verify(mx.array([[tok] * T]))
        mx.eval(lg, h)
        dt = time.perf_counter() - t0
        if split is not None:
            split.on = False
        if r:
            ts.append(dt)
            if split is not None:
                parts.append(dict(split.t))
        target.trim(T - 1)
    if split is None:
        return med(ts)
    return dict(total=med(ts), **{k: med([p[k] for p in parts]) for k in parts[0]})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--context", default="512,8192,32768")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--memory-limit-gb", type=float, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    apply_memory_rules(a.memory_limit_gb)
    _, target, _, _ = load(a.target, a.draft, fast_verify=True)
    target.enable_prompt_cache(False)
    fast = bool(target.fast_verify)
    layers = [getattr(layer, "_layer", layer) for layer in _inner_layers(target.model)]
    split = Split(layers)
    att = next(layer.self_attn for layer in layers if getattr(layer, "self_attn", None) is not None)
    H, KV, hd = att.num_attention_heads, att.num_key_value_heads, att.head_dim
    res = {"target": a.target, "fast_verify": fast, "heads": [H, KV, hd], "ctx": {}}
    print("[ctx] fast verify kernels:", fast, "heads", H, "kv heads", KV, "head dim", hd, flush=True)

    for ctx in [int(c) for c in a.context.split(",")]:
        ids = [(1000 + (i * 7919) % 20000) % target.vocab_size for i in range(ctx)]
        target.reset()
        t0 = time.perf_counter()
        target.prefill(mx.array([ids]), capture=False)
        mx.eval([c.state for c in target.cache])
        row = {"prefill_s": round(time.perf_counter() - t0, 1), "rows": {}}
        tok = ids[-1]
        for T in ROWS:
            r = {"kit": check(target, tok, T, a.reps), "split": check(target, tok, T, a.reps, split)}
            if fast:
                target.enable_fast_verify(False)
                r["kernels off"] = check(target, tok, T, a.reps)
                target.enable_fast_verify(True)
            row["rows"][T] = r
            print(f"[ctx] ctx={ctx} rows={T} {json.dumps(r)}", flush=True)
        target.reset()
        mx.clear_cache()
        # the attention product and the buffer write alone, on arrays of the same shape
        size = ((ctx + 255) // 256) * 256
        keys = mx.random.normal((1, KV, size, hd)).astype(mx.bfloat16)
        vals = mx.random.normal((1, KV, size, hd)).astype(mx.bfloat16)
        mx.eval(keys, vals)
        micro = {}
        for T in ROWS:
            q = mx.random.normal((1, H, T, hd)).astype(mx.bfloat16)
            new = mx.random.normal((1, KV, T, hd)).astype(mx.bfloat16)
            mx.eval(q, new)
            ta, tw = [], []
            for r in range(a.reps + 3):
                t0 = time.perf_counter()
                out = mx.fast.scaled_dot_product_attention(q, keys[..., :ctx, :], vals[..., :ctx, :], scale=hd ** -0.5)
                mx.eval(out)
                t1 = time.perf_counter()
                keys[..., ctx - T : ctx, :] = new
                vals[..., ctx - T : ctx, :] = new
                mx.eval(keys, vals)
                t2 = time.perf_counter()
                if r > 1:
                    ta.append(t1 - t0)
                    tw.append(t2 - t1)
            micro[T] = dict(attention_one_layer=med(ta), write_one_layer=med(tw))
        row["micro"] = micro
        print(f"[ctx] ctx={ctx} one attention layer alone {json.dumps(micro)}", flush=True)
        del keys, vals
        mx.clear_cache()
        res["ctx"][ctx] = row
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
