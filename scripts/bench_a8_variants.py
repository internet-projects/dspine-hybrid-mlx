"""Int8-activation prefill settings: speed against how far the next-token distribution moves.

ds-48 measured oMLX's default (per-row activation scale on all three MLP projections): prefill
1.42-1.47x faster at 8K-32K, but next-token KL 0.023 and the top token changed on 3 of 7 prompts.
This tries the same kernels with a finer activation scale (one per 64 inputs) and with single
projections left exact, to find a setting that keeps the speed with much less drift.

    python scripts/bench_a8_variants.py --target <4bit> --draft <drafter> --out a8v.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402

from bench_accept import prompts  # noqa: E402
from bench_long import build_ids  # noqa: E402
from dspine_hybrid.run import apply_memory_rules, load  # noqa: E402

ROW, G64 = 0, 1
# name, activation scale, int8 on (gate+up), int8 on down
VARIANTS = [
    ("row scale, gate+up+down (oMLX default)", ROW, True, True),
    ("row scale, gate+up only", ROW, True, False),
    ("row scale, down only", ROW, False, True),
    ("g64 scale, gate+up+down", G64, True, True),
    ("g64 scale, gate+up only", G64, True, False),
    ("g64 gate+up, row down", (G64, ROW), True, True),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--context", default="8192,32768")
    p.add_argument("--short", type=int, default=8, help="prompts of about 1400 tokens for the quality check")
    p.add_argument("--memory-limit-gb", type=float, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    apply_memory_rules(a.memory_limit_gb)
    _, target, _, _ = load(a.target, a.draft, fast_verify=True)
    target.enable_prompt_cache(False)

    from mlx_lm.models.activations import swiglu
    from omlx.custom_kernels.quant_matmul import fast
    from omlx.patches import qwen35_oq_a8 as oq

    print("[a8v] kernels available:", fast.oq_a8_available(), flush=True)
    mods = [m for layer in target.plain_layers() for _, m in layer.named_modules()]
    cls = type(next(m for m in mods if hasattr(m, "gate_proj") and hasattr(m, "down_proj")))
    stock_call = getattr(cls, "_omlx_oq_a8_original_call", cls.__call__)
    count = {"int8": 0, "stock": 0}

    def make(scale, front, down):
        s_front, s_down = scale if isinstance(scale, tuple) else (scale, scale)

        def call(self, x, *args, **kw):
            if x.ndim != 3 or x.shape[-2] < 128 or x.dtype not in (mx.bfloat16, mx.float16):
                return stock_call(self, x, *args, **kw)
            plans = [oq.classify_linear(m) for m in (self.gate_proj, self.up_proj, self.down_proj)]
            if any(pl is None for pl in plans):
                count["stock"] += 1
                return stock_call(self, x, *args, **kw)
            count["int8"] += 1
            gp, up, dp = plans
            if front:
                st = oq.stage_a(x, s_front, gp.layout)
                h = swiglu(oq.apply_plan(self.gate_proj, st, dataclasses.replace(gp, act_mode=s_front)),
                           oq.apply_plan(self.up_proj, st, dataclasses.replace(up, act_mode=s_front)))
            else:
                h = swiglu(self.gate_proj(x), self.up_proj(x))
            if down:
                return oq.apply_plan(self.down_proj, oq.stage_a(h, s_down, dp.layout),
                                     dataclasses.replace(dp, act_mode=s_down))
            return self.down_proj(h)

        return call

    tok = target.tokenizer
    pool = [x for xs in prompts(20).values() for x in xs]
    sets = {f"ctx {c}": build_ids(tok, pool, int(c)) for c in a.context.split(",")}
    for i in range(a.short):
        sets[f"short {i}"] = build_ids(tok, pool[i:] + pool[:i], 400)

    def run(ids):
        target.reset()
        mx.clear_cache()
        t0 = time.perf_counter()
        logits, _ = target.prefill(mx.array([ids]), capture=False)
        last = logits[0, -1].astype(mx.float32)
        mx.eval(last)
        t = time.perf_counter() - t0
        logp = last - mx.logsumexp(last)
        mx.eval(logp)
        target.reset()
        return t, logp

    run(sets[next(iter(sets))][:512])
    stock = {name: run(ids) for name, ids in sets.items()}
    res = {"stock": {n: dict(tokens=len(sets[n]), s=round(t, 2), tps=round(len(sets[n]) / t)) for n, (t, _) in stock.items()
                     if n.startswith("ctx")}}
    print("[a8v] stock", json.dumps(res["stock"]), flush=True)

    for name, scale, front, down in VARIANTS:
        cls.__call__ = make(scale, front, down)
        count.update(int8=0, stock=0)
        row = {}
        try:
            run(sets[next(iter(sets))][:512])
            kls, same = [], 0
            for n, ids in sets.items():
                t, logp = run(ids)
                ref = stock[n][1]
                kl = float(mx.sum(mx.exp(ref) * (ref - logp)).item())
                top = bool((mx.argmax(ref) == mx.argmax(logp)).item())
                if n.startswith("ctx"):
                    row[n] = dict(s=round(t, 2), tps=round(len(ids) / t), speedup=round(stock[n][0] / t, 3),
                                  kl=round(kl, 5), same_top=top)
                kls.append(kl)
                same += top
            row.update(kl_mean=round(sum(kls) / len(kls), 5), kl_max=round(max(kls), 5), same_top=f"{same}/{len(kls)}",
                       mlp_calls=dict(count))
        except Exception as e:  # noqa: BLE001 - a setting the kernel refuses is a result too
            row["error"] = f"{type(e).__name__}: {e}"[:300]
        finally:
            cls.__call__ = stock_call
        res[name] = row
        print(f"[a8v] {name} {json.dumps(row)}", flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
