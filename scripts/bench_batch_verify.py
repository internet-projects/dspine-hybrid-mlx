"""How much does one target check cost when several requests are checked together?

Times one cached forward pass of the target for B requests x M rows (B in 1, 2, 4; M in 1, 5, 16),
on stock mlx-lm and through oMLX's verify kernels (when installed), after a shared prompt of
--ctx tokens. It uses the plain mlx-lm model, not the kit's single-request adapter, so it only
answers one question: is a batched check cheap enough to be worth building?

    python scripts/bench_batch_verify.py --target mlx-community/Qwen3.8-27B-4bit --ctx 1024
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import mlx.core as mx


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--batches", default="1,2,4")
    p.add_argument("--rows", default="1,5,16")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--memory-limit-gb", type=float, default=40)
    p.add_argument("--out")
    a = p.parse_args()
    mx.set_memory_limit(int(a.memory_limit_gb * 1e9))
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    from dspine_hybrid import fast_verify

    model, tok = load(a.target)
    have = fast_verify.install()
    print(f"[batch] verify kernels: {'on' if have else 'not available'}", flush=True)
    text = "The quick brown fox jumps over the lazy dog. " * (a.ctx // 8)
    ids = tok.encode(text)[: a.ctx]
    probe = tok.encode(" def merge(a, b):\n    out = []\n    i = j = 0\n    while i < len(a)")[:16]
    probe = (probe * 16)[:16]
    results = []
    for B in [int(x) for x in a.batches.split(",")]:
        try:
            cache = make_prompt_cache(model)
            x = mx.array([ids] * B)
            for s in range(0, x.shape[1], 2048):
                model(x[:, s:s + 2048], cache=cache)
                mx.eval([c.state for c in cache])
        except Exception as e:  # batched prefill unsupported
            print(f"[batch] B={B}: prefill failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        for M in [int(x) for x in a.rows.split(",")]:
            rows = mx.array([probe[:M]] * B)
            for mode in (["stock", "kernels"] if have else ["stock"]):
                try:
                    times = []
                    for r in range(a.reps + 2):
                        c = copy.deepcopy(cache)
                        t0 = time.perf_counter()
                        with fast_verify.armed(mode == "kernels"):
                            out = model(rows, cache=c)
                            mx.eval(out)
                        times.append(time.perf_counter() - t0)
                    ms = 1000 * sorted(times[2:])[len(times[2:]) // 2]
                    results.append(dict(B=B, M=M, mode=mode, ms=round(ms, 2)))
                    print(f"[batch] B={B} M={M:2d} {mode:8s} {ms:7.1f} ms  ({ms / B:6.1f} ms per request)", flush=True)
                except Exception as e:
                    print(f"[batch] B={B} M={M} {mode}: failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
        del cache
        mx.clear_cache()
    base = {(r["M"], r["mode"]): r["ms"] for r in results if r["B"] == 1}
    for r in results:
        if r["B"] > 1 and (r["M"], r["mode"]) in base:
            print(f"[batch] ratio B={r['B']} M={r['M']:2d} {r['mode']:8s} x{r['ms'] / base[(r['M'], r['mode'])]:.2f} "
                  f"the cost of one request (x{r['B']} would be no gain)")
    print(f"[batch] peak memory {mx.get_peak_memory() / 1e9:.1f} GB")
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
