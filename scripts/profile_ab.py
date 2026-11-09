"""Where DSpine's time goes, and whether a quantized draft helps.

Per target: for each draft precision (bf16 / 8-bit / 4-bit), run the same
prompts with phase timing on (draft, refinement tail, target verify, hybrid
rollback) and report decode tok/s, tokens/round and the time split. Also times
one plain target step and one 16-token verify, so verify cost per round is
visible against plain decoding. Phase timing adds syncs, so tok/s here is a
little below an unprofiled run; compare the rows with each other.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402

from bench_accept import encode, prompts  # noqa: E402
from dspine_hybrid.run import apply_memory_rules, load  # noqa: E402
from dspine_mlx.generate import DSpineGenerator, _load_tokenizer_eos  # noqa: E402


def _cache_arrays(target):
    out = []
    for c in target.cache:
        st = getattr(c, "state", None)
        if isinstance(st, (list, tuple)):
            out += [x for x in st if isinstance(x, mx.array)]
    return out


def time_target(target, ids, reps=8):
    """Seconds for one plain decode step and one 16-token verify (median)."""
    target.reset()
    target.prefill(mx.array([ids]), capture=False)
    one, blk = [], []
    tok = ids[-1]
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(target.step(tok))
        one.append(time.perf_counter() - t0)
    for _ in range(reps):
        t0 = time.perf_counter()
        lg, _ = target.verify(mx.array([[tok] * 16]), capture=False)
        mx.eval(lg, *_cache_arrays(target))
        blk.append(time.perf_counter() - t0)
        target.trim(15)
        mx.eval(*_cache_arrays(target))
    target.reset()
    mx.clear_cache()
    return sorted(one)[reps // 2], sorted(blk)[reps // 2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", action="append", required=True, help="name=path, repeatable")
    p.add_argument("--draft", required=True)
    p.add_argument("--bits", default="0,8,4", help="draft precisions, 0 = bf16")
    p.add_argument("--n", type=int, default=2, help="prompts per benchmark")
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--memory-limit-gb", type=float, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    apply_memory_rules(a.memory_limit_gb)

    texts = [(n, t) for n, ts in prompts(a.n).items() for t in ts[: a.n]]
    res = {"tokens": a.tokens, "thinking": a.thinking, "prompts": len(texts), "targets": {}}
    for spec in a.target:
        name, path = spec.split("=", 1)
        rows = {}
        for bits in [int(b) for b in a.bits.split(",")]:
            gen, target, _, _ = load(path, a.draft, draft_bits=bits or None)
            gen = DSpineGenerator(target, gen.draft, profile=True)
            tok = target.tokenizer
            stop = _load_tokenizer_eos(tok)
            if "step_s" not in res["targets"].get(name, {}):
                w = encode(tok, texts[0][1], a.thinking)
                s1, s16 = time_target(target, w)
                res["targets"].setdefault(name, {})["step_s"] = round(s1, 4)
                res["targets"][name]["verify16_s"] = round(s16, 4)
                print(f"[profile] {name} plain step {s1 * 1e3:.1f} ms, 16-token verify {s16 * 1e3:.1f} ms", flush=True)

            # time the hybrid rollback (forced to run inside the timer)
            roll = [0.0]
            orig_trim = target.trim

            def timed_trim(n, _orig=orig_trim, _t=target):
                t0 = time.perf_counter()
                _orig(n)
                mx.eval(*_cache_arrays(_t))
                roll[0] += time.perf_counter() - t0

            target.trim = timed_trim
            w = encode(tok, "Say hello.", a.thinking)
            gen.generate(w, max_tokens=16, stop_ids=stop)
            mx.clear_cache()
            roll[0] = 0.0
            g = rd = cm = 0
            sec = dr = tl = vf = 0.0
            for _, text in texts:
                ids = encode(tok, text, a.thinking)
                _, st = gen.generate(ids, max_tokens=a.tokens, stop_ids=stop)
                mx.clear_cache()
                g += st.generated_tokens; sec += st.decode_seconds
                rd += st.rounds; cm += st.committed_tokens
                dr += st.draft_seconds; tl += st.tail_seconds; vf += st.verify_seconds
            label = "bf16" if not bits else f"{bits}bit"
            rows[label] = {
                "decode_tok_s": round(g / sec, 2),
                "tokens_per_round": round(cm / max(rd, 1), 2),
                "ms_per_round": {
                    "draft": round(1e3 * dr / rd, 2),
                    "tail": round(1e3 * tl / rd, 2),
                    "verify": round(1e3 * vf / rd, 2),
                    "rollback": round(1e3 * roll[0] / rd, 2),
                    "total": round(1e3 * sec / rd, 2),
                },
                "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
            }
            print(f"[profile] {name} draft {label}: {json.dumps(rows[label])}", flush=True)
            del gen, target
            gc.collect()
            mx.clear_cache()
            mx.reset_peak_memory()
        res["targets"][name]["drafts"] = rows

    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
