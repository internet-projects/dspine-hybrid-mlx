"""Verify cost vs block size, and DSpine speed at smaller verify blocks.

1. Target forward time for T = 1, 2, 4, 6, 8, 12, 16 tokens, both as a plain
   model call (no recording) and through HybridTargetAdapter.verify (recording
   on, then rollback), median of several reps after warm-up.
2. DSpine decode tok/s and tokens/round with block size 16, 12, 8, 6, 4 on the
   same prompts (bf16 draft, greedy).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402

from bench_accept import encode, prompts  # noqa: E402
from dspine_hybrid.run import apply_memory_rules, load  # noqa: E402
from dspine_mlx.generate import DSpineGenerator, _load_tokenizer_eos  # noqa: E402
from profile_ab import _cache_arrays  # noqa: E402


def med(xs):
    return sorted(xs)[len(xs) // 2]


def time_forward(t, ids, T, reps=7):
    t.reset()
    t.prefill(mx.array([ids]), capture=False)
    tok = ids[-1]
    raw, rec = [], []
    for i in range(reps + 2):
        t._set_recording(False)
        x = mx.array([[tok] * T])
        t0 = time.perf_counter()
        lg = t.model(x, t.cache)
        mx.eval(lg, *_cache_arrays(t))
        if i >= 2:
            raw.append(time.perf_counter() - t0)
        for c in t.cache:  # undo: KV trim; linear state is not restored (timing only)
            if hasattr(c, "trim") and c.is_trimmable():
                c.trim(T)
        t0 = time.perf_counter()
        lg, _ = t.verify(x, capture=True)
        mx.eval(lg, *_cache_arrays(t))
        t.trim(T - 1)
        mx.eval(*_cache_arrays(t))
        if i >= 2:
            rec.append(time.perf_counter() - t0)
    t.reset()
    mx.clear_cache()
    return med(raw), med(rec)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--T", default="1,2,4,6,8,12,16")
    p.add_argument("--blocks", default="16,12,8,6,4")
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--memory-limit-gb", type=float, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    apply_memory_rules(a.memory_limit_gb)

    gen, t, _, _ = load(a.target, a.draft)
    tok = t.tokenizer
    stop = _load_tokenizer_eos(tok)
    texts = [(n, x) for n, xs in prompts(a.n).items() for x in xs[: a.n]]
    ids0 = encode(tok, texts[0][1], a.thinking)

    fwd = {}
    for T in [int(x) for x in a.T.split(",")]:
        r, v = time_forward(t, ids0, T)
        fwd[T] = {"plain_ms": round(1e3 * r, 2), "verify_rollback_ms": round(1e3 * v, 2)}
        print(f"[sweep] {a.name} forward T={T}: plain {1e3 * r:.1f} ms, verify+rollback {1e3 * v:.1f} ms", flush=True)

    blocks = {}
    for b in [int(x) for x in a.blocks.split(",")]:
        g = DSpineGenerator(t, gen.draft, block_size=b)
        g.generate(encode(tok, "Say hello.", a.thinking), max_tokens=16, stop_ids=stop)
        mx.clear_cache()
        n = sec = rd = cm = 0
        for _, text in texts:
            _, st = g.generate(encode(tok, text, a.thinking), max_tokens=a.tokens, stop_ids=stop)
            mx.clear_cache()
            n += st.generated_tokens; sec += st.decode_seconds; rd += st.rounds; cm += st.committed_tokens
        blocks[b] = {"decode_tok_s": round(n / sec, 2), "tokens_per_round": round(cm / max(rd, 1), 2),
                     "ms_per_round": round(1e3 * sec / max(rd, 1), 2)}
        print(f"[sweep] {a.name} block {b}: {json.dumps(blocks[b])}", flush=True)

    res = {"target": a.name, "thinking": a.thinking, "prompts": len(texts), "forward": fwd, "blocks": blocks,
           "peak_gb": round(mx.get_peak_memory() / 1e9, 2)}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
