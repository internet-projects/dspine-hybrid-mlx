"""Two engine ideas from mlx-dspark (ARahim3, MIT), measured on this Mac without its drafters.

    --part attn     its multi-row attention kernel against mlx's own attention call, for the
                    target's attention shape, 1-16 query rows over 8k and 32k cached keys (the
                    many-row check at long context is where our kit loses time). mlx-dspark keeps
                    this kernel off on M5 chips until someone verifies it there; this is that check.
    --part prefill  its prefill paths on our target: dequantize the weight once ("wide"), and
                    hand a measured share of the rows to the CPU's matrix units while the GPU does
                    the rest ("CPU co-prefill"). Reports prefill time and how far the next-token
                    distribution moves.

mlx-dspark's two modules are loaded by file path from a source checkout (--dspark-src), so
nothing of it is installed and none of its other code runs.

    python scripts/bench_dspark_ideas.py --part attn --dspark-src <checkout> --out attn.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402


def module(src, name):
    spec = importlib.util.spec_from_file_location(f"dspark_{name}", os.path.join(src, "src", "mlx_dspark", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def ms(fn, reps=5):
    ts = []
    for r in range(reps + 2):
        t0 = time.perf_counter()
        mx.eval(fn())
        if r > 1:
            ts.append(time.perf_counter() - t0)
    return round(1000 * sorted(ts)[len(ts) // 2], 2)


def part_attn(a, res):
    mr = module(a.dspark_src, "multirow_attn")
    H, KV, D = [int(x) for x in a.shape.split(",")]
    scale = 1.0 / math.sqrt(D)
    for kv in (8192, 32768):
        k = mx.random.normal((1, KV, kv, D)).astype(mx.bfloat16)
        v = mx.random.normal((1, KV, kv, D)).astype(mx.bfloat16)
        mx.eval(k, v)
        for ql in (1, 2, 4, 5, 6, 8, 9, 12, 15, 16):
            q = mx.random.normal((1, H, ql, D)).astype(mx.bfloat16)
            mx.eval(q)
            mask = "causal" if ql > 1 else None
            row = {"mlx_ms": ms(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask))}
            try:
                ref = mx.fast.scaled_dot_product_attention(
                    q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), scale=scale, mask=mask)
                out = mr.multirow_sdpa(q, k, v, scale=scale, causal=ql > 1)
                row["kernel_err"] = round(float(mx.abs(out.astype(mx.float32) - ref).max()), 4)
                row["mlx_err"] = round(float(mx.abs(mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask).astype(mx.float32) - ref).max()), 4)
                row["kernel_ms"] = ms(lambda: mr.multirow_sdpa(q, k, v, scale=scale, causal=ql > 1))
                row["speedup"] = round(row["mlx_ms"] / row["kernel_ms"], 2)
            except Exception as e:  # noqa: BLE001 - a width the kernel cannot launch here
                row["kernel"] = f"{type(e).__name__}: {e}"[:160]
            res[f"q{ql}@{kv}"] = row
            print(f"[dspark] attention {H}/{KV}/{D} keys={kv} rows={ql} {json.dumps(row)}", flush=True)
        del k, v
        mx.clear_cache()
    window, report = mr.measure_window(H, KV, D)
    res["measure_window"] = {"window": window, "report": report}
    print("[dspark] its own gate for this chip:", json.dumps(window), json.dumps(report), flush=True)


def part_prefill(a, res):
    from bench_accept import prompts
    from bench_long import build_ids
    from dspine_hybrid.run import apply_memory_rules, load

    wg = module(a.dspark_src, "wide_gemm")
    apply_memory_rules(a.memory_limit_gb)
    _, target, _, _ = load(a.target, a.draft, fast_verify=True)
    target.enable_prompt_cache(False)
    pool = [x for xs in prompts(20).values() for x in xs]
    sets = {c: build_ids(target.tokenizer, pool, c) for c in (int(c) for c in a.context.split(","))}

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

    run(next(iter(sets.values()))[:512])
    layers = target.plain_layers()      # the kit's hooks hide the layers from model.named_modules()
    print("[dspark] quantized projections found:", sum(1 for _ in wg.quantized_linears(*layers)), flush=True)
    cross = wg.measure_crossover(*layers)
    split = wg.measure_cpu_split(*layers)
    res["crossover"], res["cpu_split"] = (None if cross is None else [cross[0], len(cross[1])]), split
    print("[dspark] wide path from rows:", None if cross is None else cross[0],
          "| CPU share per width:", json.dumps(split), flush=True)
    for c, ids in sets.items():
        t0, ref = run(ids)
        row = {"tokens": len(ids), "stock_s": round(t0, 2), "stock_tps": round(len(ids) / t0)}
        arms = {}
        if cross is not None:
            arms["wide"] = dict(min_rows=cross[0], shapes=cross[1])
        if split is not None:
            arms["wide + cpu"] = dict(min_rows=cross[0] if cross else None, shapes=cross[1] if cross else None,
                                      cpu_split=split)
        for name, kw in arms.items():
            with wg.wide_matmul(kw.pop("min_rows"), **kw):
                t, logp = run(ids)
            row[name] = dict(s=round(t, 2), tps=round(len(ids) / t), speedup=round(t0 / t, 3),
                             kl=round(float(mx.sum(mx.exp(ref) * (ref - logp)).item()), 5),
                             same_top=bool((mx.argmax(ref) == mx.argmax(logp)).item()))
        res[f"ctx {c}"] = row
        print(f"[dspark] prefill ctx={c} {json.dumps(row)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=("attn", "prefill"), required=True)
    p.add_argument("--dspark-src", required=True)
    p.add_argument("--shape", default="24,4,256", help="attention heads, key/value heads, head size")
    p.add_argument("--target")
    p.add_argument("--draft")
    p.add_argument("--context", default="2048,8192")
    p.add_argument("--memory-limit-gb", type=float, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res: dict = {"part": a.part}
    try:
        (part_attn if a.part == "attn" else part_prefill)(a, res)
    finally:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
