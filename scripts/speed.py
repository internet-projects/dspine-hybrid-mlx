"""Decode speed: plain mlx-lm greedy vs DSpine on the same target and prompts.

Run on an otherwise idle machine. Per target: load once, then for each
prompt run plain greedy and DSpine greedy (T=0, same max tokens) and sum
generated tokens / decode seconds (prefill excluded for both).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402

from bench_accept import encode, prompts  # noqa: E402
from dspine_hybrid.run import apply_memory_rules, load  # noqa: E402
from dspine_mlx.generate import _load_tokenizer_eos, baseline_generate  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", action="append", required=True, help="name=path, repeatable")
    p.add_argument("--draft", required=True)
    p.add_argument("--n", type=int, default=2, help="prompts per benchmark")
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--memory-limit-gb", type=float, default=None)
    p.add_argument("--save-prompts", default=None, help="write the prompt texts (to time another server on the same prompts)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    apply_memory_rules(a.memory_limit_gb)

    sets = prompts(a.n)
    texts = [(name, t) for name, ts in sets.items() for t in ts[: a.n]]
    if a.save_prompts:
        with open(a.save_prompts, "w") as f:
            json.dump([{"bench": n, "text": t} for n, t in texts], f)

    res = {"tokens": a.tokens, "thinking": a.thinking, "prompts": len(texts), "targets": {}}
    for spec in a.target:
        name, path = spec.split("=", 1)
        gen, target, _, _ = load(path, a.draft)
        tok = target.tokenizer
        stop = _load_tokenizer_eos(tok)
        # warm-up (kernels, tables)
        w = encode(tok, "Say hello.", a.thinking)
        gen.generate(w, max_tokens=16, stop_ids=stop)
        baseline_generate(target, w, max_tokens=16, stop_ids=stop)
        mx.clear_cache()

        ar_tok = ar_s = sp_tok = sp_s = 0.0
        rounds = committed = exact = 0
        per = {}
        for bench, text in texts:
            ids = encode(tok, text, a.thinking)
            b, bs = baseline_generate(target, ids, max_tokens=a.tokens, stop_ids=stop)
            mx.clear_cache()
            s, ss = gen.generate(ids, max_tokens=a.tokens, stop_ids=stop)
            mx.clear_cache()
            ar_tok += bs.generated_tokens; ar_s += bs.decode_seconds
            sp_tok += ss.generated_tokens; sp_s += ss.decode_seconds
            rounds += ss.rounds; committed += ss.committed_tokens
            exact += s == b
            pb = per.setdefault(bench, [0.0, 0.0, 0.0, 0.0])
            pb[0] += bs.generated_tokens; pb[1] += bs.decode_seconds
            pb[2] += ss.generated_tokens; pb[3] += ss.decode_seconds
            print(f"[speed] {name} {bench} plain {bs.decode_tps:.1f} tok/s  dspine {ss.decode_tps:.1f} tok/s  "
                  f"tokens/round {ss.tokens_per_round:.2f}", flush=True)
        ar, sp = ar_tok / ar_s, sp_tok / sp_s
        res["targets"][name] = {
            "path": path,
            "plain_tok_s": round(ar, 2),
            "dspine_tok_s": round(sp, 2),
            "speedup": round(sp / ar, 2),
            "tokens_per_round": round(committed / max(rounds, 1), 2),
            "token_exact_prompts": f"{exact}/{len(texts)}",
            "per_bench_speedup": {k: round((v[2] / v[3]) / (v[0] / v[1]), 2) for k, v in per.items()},
            "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
        }
        print(f"[speed] {name}: plain {ar:.1f} tok/s, dspine {sp:.1f} tok/s, x{sp / ar:.2f}", flush=True)
        del gen, target
        gc.collect()
        mx.clear_cache()
        mx.reset_peak_memory()

    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
