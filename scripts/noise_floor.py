"""Real-model rounding noise: logits from 16-token verify blocks vs one-token steps.

No draft involved. For each prompt: plain greedy decode N tokens, then replay
the same tokens once as 16-token blocks and once token by token from the same
prefill, and compare the logits. Also reports how often the top-2 gap is below
that noise (where a spec-decoding / plain-decoding flip can happen at all).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402

from bench_accept import encode, prompts  # noqa: E402
from dspine_hybrid.hybrid_target import HybridTargetAdapter  # noqa: E402
from dspine_hybrid.run import apply_memory_rules  # noqa: E402
from dspine_mlx.generate import _load_tokenizer_eos, baseline_generate  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--n", type=int, default=2, help="prompts per benchmark")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--memory-limit-gb", type=float, default=None)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    apply_memory_rules(a.memory_limit_gb)

    t = HybridTargetAdapter.load(a.target, (1,))
    tok = t.tokenizer
    stop = _load_tokenizer_eos(tok)
    diffs, gaps = [], []
    for name, texts in prompts(a.n).items():
        for text in texts[: a.n]:
            ids = encode(tok, text, a.thinking)
            seq, _ = baseline_generate(t, ids, max_tokens=a.tokens, stop_ids=stop)
            seq = seq[: (len(seq) // 16) * 16]
            # 16-token blocks
            t.reset()
            t.prefill(mx.array([ids]), capture=False)
            blk = []
            for s in range(0, len(seq), 16):
                lg, _ = t.verify(mx.array([seq[s : s + 16]]), capture=False)
                t.trim(0)
                blk.append(lg[0].astype(mx.float32))
                mx.eval(blk[-1])
            # single steps
            t.reset()
            t.prefill(mx.array([ids]), capture=False)
            one = []
            for x in seq:
                one.append(t.step(x)[0, -1].astype(mx.float32))
                mx.eval(one[-1])
            B = mx.concatenate(blk, axis=0)
            O = mx.stack(one, axis=0)
            d = mx.max(mx.abs(B - O), axis=-1)
            top2 = mx.sort(O, axis=-1)[:, -2:]
            g = top2[:, 1] - top2[:, 0]
            diffs += d.tolist()
            gaps += g.tolist()
            mx.clear_cache()
            print(f"[noise] {name} max|dlogit|={max(d.tolist()):.4f} min top2 gap={min(g.tolist()):.4f}", flush=True)
    ds = sorted(diffs)
    q = lambda f: ds[min(len(ds) - 1, int(f * len(ds)))]  # noqa: E731
    floor = q(0.99)
    res = {
        "positions": len(ds),
        "max_abs_dlogit_p50": round(q(0.5), 4),
        "max_abs_dlogit_p99": round(floor, 4),
        "max_abs_dlogit_max": round(ds[-1], 4),
        "share_top2_gap_below_p99_noise": round(sum(1 for x in gaps if x <= floor) / len(gaps), 4),
    }
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
