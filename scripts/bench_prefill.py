"""What int8-activation prefill is worth on this Mac, on the kit's own target.

Times the target prefill of a 2k, an 8k and a 32k prompt with the stock projections, then turns on
oMLX's "oQ A8" prefill (activations quantized to int8 per row and 64-input group, weights exact,
the M5 tensor unit multiplies int8 at twice the bf16 rate; the idea Splash describes as
int8-activation prefill) and times them again. It also reports how far the next-token
distribution moves (KL, same top token) at each length and on short prefixes.

Needs oMLX with its native kernel extension built (omlx.custom_kernels.quant_matmul._ext); without
it the script says so and reports the stock times only.

    python scripts/bench_prefill.py --target <4-bit build> --draft <drafter> --out prefill.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402

from bench_accept import prompts  # noqa: E402
from bench_long import build_ids  # noqa: E402
from dspine_hybrid.run import apply_memory_rules, load  # noqa: E402


def run(target, ids, reps):
    """(median seconds, last-token log-probabilities) of a fresh prefill of ``ids``."""
    ts, logp = [], None
    for _ in range(reps):
        target.reset()
        mx.clear_cache()
        t0 = time.perf_counter()
        logits, _ = target.prefill(mx.array([ids]), capture=False)
        last = logits[0, -1].astype(mx.float32)
        mx.eval(last)
        ts.append(time.perf_counter() - t0)
        logp = last - mx.logsumexp(last)
        mx.eval(logp)
    target.reset()
    return sorted(ts)[len(ts) // 2], logp


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--context", default="2048,8192,32768")
    p.add_argument("--short", type=int, default=12, help="short prefixes (about 400 tokens) for the quality check")
    p.add_argument("--min-tokens", type=int, default=128)
    p.add_argument("--memory-limit-gb", type=float, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    apply_memory_rules(a.memory_limit_gb)
    _, target, _, _ = load(a.target, a.draft, fast_verify=True)
    target.enable_prompt_cache(False)
    tok = target.tokenizer
    pool = [x for xs in prompts(20).values() for x in xs]
    sets = {f"ctx {c}": (build_ids(tok, pool, c), 1 if c >= 16384 else 2) for c in (int(c) for c in a.context.split(","))}
    for i in range(a.short):
        sets[f"short {i}"] = (build_ids(tok, pool[i:] + pool[:i], 400), 1)
    res = {"target": a.target, "rows": {}}

    stock = {}
    run(target, sets[next(iter(sets))][0][:256], 1)                    # warm up
    for name, (ids, reps) in sets.items():
        stock[name] = run(target, ids, reps)
        if name.startswith("ctx"):
            print(f"[prefill] stock {name}: {len(ids)} tokens {stock[name][0]:.2f} s "
                  f"{len(ids) / stock[name][0]:.0f} tok/s", flush=True)

    from dspine_hybrid import a8_prefill

    res["kernels"] = a8_prefill.kernels() is not None
    print(f"[prefill] int8 prefill kernels available: {res['kernels']}", flush=True)
    for label, parts in (("int8 mlp+gdn", ("mlp", "gdn")), ("int8 attn", ("attn",)), ("int8 mlp+gdn+attn", ("mlp", "gdn", "attn"))):
        if not target.enable_a8_prefill(True, a.min_tokens, parts):
            break
        a8_prefill.stats.update(int8=0, stock=0)
        run(target, sets[next(iter(sets))][0][:512], 1)
        kls, same, table = [], 0, {}
        for name, (ids, reps) in sets.items():
            t, logp = run(target, ids, reps)
            ref = stock[name][1]
            kl = float(mx.sum(mx.exp(ref) * (ref - logp)).item())
            top = bool((mx.argmax(ref) == mx.argmax(logp)).item())
            kls.append(kl)
            same += top
            if name.startswith("ctx"):
                table[name] = dict(tokens=len(ids), stock_s=round(stock[name][0], 2), s=round(t, 2), tps=round(len(ids) / t),
                                   speedup=round(stock[name][0] / t, 3), kl=round(kl, 5), same_top=top)
                print(f"[prefill] {label} {name}: {t:.2f} s {len(ids) / t:.0f} tok/s x{stock[name][0] / t:.3f} "
                      f"KL {kl:.5f} same top token {top}", flush=True)
        table.update(kl_mean=round(sum(kls) / len(kls), 5), kl_max=round(max(kls), 5), same_top=f"{same}/{len(kls)}",
                     calls=dict(a8_prefill.stats))
        res["rows"][label] = table
        print(f"[prefill] {label}: KL mean {table['kl_mean']} max {table['kl_max']}, same top token {table['same_top']}, "
              f"projection groups {table['calls']}", flush=True)
    target.enable_a8_prefill(False)
    # stock again at the end: the baseline moved between earlier runs (8K 810 and 991 tok/s)
    for name, (ids, reps) in sets.items():
        if name.startswith("ctx"):
            t, _ = run(target, ids, reps)
            res["rows"][f"stock again {name}"] = dict(s=round(t, 2), tps=round(len(ids) / t))
            print(f"[prefill] stock again {name}: {t:.2f} s {len(ids) / t:.0f} tok/s", flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
