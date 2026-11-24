"""Prefill ideas read in Sushi's engine notes, measured on the kit's own target.

1. kernel: one GatedDeltaNet recurrence at the target's real layout (random inputs), mlx-lm's
   kernel against oMLX's blocked and pipelined kernels: ms per call and how far the results differ.
2. prefill: stock, pipelined recurrence, int8 projections (if oMLX's built kernels are there),
   int8 + pipelined, stock again, at 2k / 8k / 32k, with KL against stock on short prompts.
3. chunk: the prefill chunk width (the kit uses 2048; Sushi found wider chunks give nothing).

    python scripts/bench_gdn_prefill.py --target <4-bit build> --draft <drafter> --out gdn.json
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
from bench_prefill import run  # noqa: E402
from dspine_hybrid import a8_prefill, gdn_prefill, hybrid_target  # noqa: E402
from dspine_hybrid.run import apply_memory_rules, load  # noqa: E402


def kernel_part(target, res):
    from mlx_lm.models.gated_delta import gated_delta_kernel

    mod = gdn_prefill.kernels()
    gdn = next(m for m in target._linear if m is not None)
    hk, hv, dk, dv = gdn.num_k_heads, gdn.num_v_heads, gdn.head_k_dim, gdn.head_v_dim
    print(f"[gdn] layout: {hk} key heads, {hv} value heads, key dim {dk}, value dim {dv}; "
          f"oMLX kernels: {mod is not None}", flush=True)
    res["layout"] = dict(hk=hk, hv=hv, dk=dk, dv=dv)
    if mod is None:
        return
    for T in (256, 2048):
        mx.random.seed(T)
        q = mx.random.normal((1, T, hk, dk))
        k = mx.random.normal((1, T, hk, dk))
        q = (q / mx.linalg.norm(q, axis=-1, keepdims=True) * dk**-0.5).astype(mx.bfloat16)
        k = (k / mx.linalg.norm(k, axis=-1, keepdims=True)).astype(mx.bfloat16)
        v = mx.random.normal((1, T, hv, dv)).astype(mx.bfloat16)
        g = mx.random.uniform(0.9, 1.0, (1, T, hv))
        beta = mx.random.uniform(0.0, 1.0, (1, T, hv))
        st = mx.random.normal((1, hv, dv, dk)) * 0.1
        mx.eval(q, k, v, g, beta, st)
        ok = mod.gated_delta_pipelined_supported(q, k, v)
        routes = {"mlx-lm": lambda: gated_delta_kernel(q, k, v, g, beta, st, None),
                  "blocked": lambda: mod.gated_delta_blocked_seq(q, k, v, g, beta, st)}
        if ok:
            routes["pipelined"] = lambda: mod.gated_delta_pipelined(q, k, v, g, beta, st)
        ref = None
        for name, fn in routes.items():
            try:
                y, s = fn()
                mx.eval(y, s)
                ts = []
                for _ in range(20):
                    t0 = time.perf_counter()
                    y, s = fn()
                    mx.eval(y, s)
                    ts.append(time.perf_counter() - t0)
                ms = sorted(ts)[len(ts) // 2] * 1e3
                y32, s32 = y.astype(mx.float32), s.astype(mx.float32)
                if ref is None:
                    ref = (y32, s32)
                ey = float((mx.abs(y32 - ref[0]).max() / (mx.abs(ref[0]).max() + 1e-9)).item())
                es = float((mx.abs(s32 - ref[1]).max() / (mx.abs(ref[1]).max() + 1e-9)).item())
                res.setdefault("kernel", {})[f"{name} {T}"] = dict(ms=round(ms, 3), err_y=ey, err_state=es)
                print(f"[gdn] {T} tokens {name}: {ms:.2f} ms per call, max error against mlx-lm "
                      f"y {ey:.2e} state {es:.2e}", flush=True)
            except Exception as e:  # a kernel that does not compile here must not stop the run
                print(f"[gdn] {T} tokens {name}: FAILED {type(e).__name__}: {e}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--context", default="2048,8192,32768")
    p.add_argument("--short", type=int, default=8)
    p.add_argument("--memory-limit-gb", type=float, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    apply_memory_rules(a.memory_limit_gb)
    _, target, _, _ = load(a.target, a.draft, fast_verify=True)
    target.enable_prompt_cache(False)
    res = {"target": a.target, "rows": {}}
    kernel_part(target, res)

    tok = target.tokenizer
    pool = [x for xs in prompts(20).values() for x in xs]
    sets = {f"ctx {c}": (build_ids(tok, pool, c), 1 if c >= 16384 else 2) for c in (int(c) for c in a.context.split(","))}
    for i in range(a.short):
        sets[f"short {i}"] = (build_ids(tok, pool[i:] + pool[:i], 400), 1)
    first = sets[next(iter(sets))][0]

    def sweep(label, stock):
        """Time every prompt; with ``stock`` given also report KL and top token against it."""
        run(target, first[:512], 1)
        out, kls, same = {}, [], 0
        for name, (ids, reps) in sets.items():
            if stock is not None and label.startswith("stock") and not name.startswith("ctx"):
                continue
            t, logp = run(target, ids, reps)
            out[name] = (t, logp)
            line = f"[prefill] {label} {name}: {len(ids)} tokens {t:.2f} s {len(ids) / t:.0f} tok/s"
            row = dict(tokens=len(ids), s=round(t, 2), tps=round(len(ids) / t))
            if stock is not None:
                ref = stock[name][1]
                kl = float(mx.sum(mx.exp(ref) * (ref - logp)).item())
                top = bool((mx.argmax(ref) == mx.argmax(logp)).item())
                kls.append(kl)
                same += top
                row.update(speedup=round(stock[name][0] / t, 3), kl=round(kl, 5), same_top=top)
                line += f" x{stock[name][0] / t:.3f} KL {kl:.5f} same top token {top}"
            if name.startswith("ctx"):
                res["rows"][f"{label} {name}"] = row
                print(line, flush=True)
        if kls and not label.startswith("stock"):
            res["rows"][f"{label} quality"] = dict(kl_mean=round(sum(kls) / len(kls), 5), kl_max=round(max(kls), 5),
                                                   same_top=f"{same}/{len(kls)}")
            print(f"[prefill] {label}: KL mean {sum(kls) / len(kls):.5f} max {max(kls):.5f}, "
                  f"same top token {same}/{len(kls)}", flush=True)
        return out

    stock = sweep("stock", None)
    if target.enable_gdn_prefill(True):
        gdn_prefill.stats.update(pipelined=0, stock=0)
        sweep("pipelined", stock)
        print(f"[prefill] pipelined calls {dict(gdn_prefill.stats)}", flush=True)
        res["calls"] = dict(gdn_prefill.stats)
        target.enable_gdn_prefill(False)
    else:
        print("[prefill] pipelined recurrence not available (omlx or Metal missing)", flush=True)
    if target.enable_a8_prefill(True):
        sweep("int8", stock)
        if target.enable_gdn_prefill(True):
            sweep("int8+pipelined", stock)
            target.enable_gdn_prefill(False)
        target.enable_a8_prefill(False)
    else:
        print("[prefill] int8 kernels not available, int8 rows skipped", flush=True)
    sweep("stock again", stock)

    # chunk width, stock projections, 8k prompt (or the longest one given below 16k)
    name = max((n for n in sets if n.startswith("ctx") and len(sets[n][0]) < 16384), key=lambda n: len(sets[n][0]), default=None)
    if name is not None:
        ids = sets[name][0]
        keep = hybrid_target.PREFILL_CHUNK
        for width in (1024, 2048, 4096):
            hybrid_target.PREFILL_CHUNK = width
            try:
                t, _ = run(target, ids, 2)
                res["rows"][f"chunk {width}"] = dict(tokens=len(ids), s=round(t, 2), tps=round(len(ids) / t))
                print(f"[prefill] chunk {width} {name}: {t:.2f} s {len(ids) / t:.0f} tok/s", flush=True)
            finally:
                hybrid_target.PREFILL_CHUNK = keep
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
