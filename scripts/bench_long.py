"""Where the time goes at long context, with the verify kernels on.

At 8k and 32k tokens of prompt (a reference document, then a request for a written summary), runs
greedy decoding for: the fixed block of 16, adaptive with the cost table measured on a short prompt,
adaptive with the cost table measured at this context, the same with a narrower drafter window, and
with the draft tree. Every row reports the drafter and check time per round. The prompt cache is on,
so the long prompt is only read once per context. Takes about eight minutes.

    python scripts/bench_long.py --target <4-bit build> --draft <drafter> --out long.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402

from bench_accept import prompts  # noqa: E402
from dspine_hybrid import calibration as cal  # noqa: E402
from dspine_hybrid.adaptive import measure_cost  # noqa: E402
from dspine_hybrid.draft import set_window  # noqa: E402
from dspine_hybrid.lanes import LaneDSpineGenerator  # noqa: E402
from dspine_hybrid.run import apply_memory_rules, load  # noqa: E402
from dspine_mlx.generate import _load_tokenizer_eos  # noqa: E402

TASK = ("\n\nAbove is a set of reference problems. Write a summary of them for a teacher: which topics "
        "they cover, how hard they are, and what a student needs to know to solve them. Use full paragraphs.")


def build_ids(tok, pool, n_tokens):
    doc, i = [], 0
    while True:
        doc.append(f"Problem {i + 1}. {pool[i % len(pool)]}")
        i += 1
        if i % 20 == 0:
            ids = tok.apply_chat_template([{"role": "user", "content": "\n\n".join(doc) + TASK}],
                                          add_generation_prompt=True, enable_thinking=False)
            if len(ids) >= n_tokens:
                return ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--context", default="8192,32768")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--memory-limit-gb", type=float, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    apply_memory_rules(a.memory_limit_gb)
    base, target, _, _ = load(a.target, a.draft, fast_verify=True)
    draft = base.draft
    window0 = draft.window
    tok = target.tokenizer
    stop = _load_tokenizer_eos(tok)
    pool = [x for xs in prompts(20).values() for x in xs]
    res = {"target": a.target, "tokens": a.tokens, "fast_verify": bool(target.fast_verify), "runs": {}}
    print("[long] fast verify kernels:", res["fast_verify"], "drafter window", window0, flush=True)

    short = measure_cost(target, list(range(1000, 1256)))
    res["cost_ms_short"] = [round(1000 * c, 1) for c in short[1:]]
    print("[long] cost ms short", res["cost_ms_short"], flush=True)

    def make(cost=None, **opts):
        adaptive = cost is not None
        return LaneDSpineGenerator(
            target, draft, block_size=16, adaptive=adaptive, cost=cost,
            confidence="calibrated" if adaptive else "vocab", calibration=cal.load(None) if adaptive else None,
            sampled_calibration=cal.load(None, "sampled") if adaptive else None, **{"copy": False, **opts})

    for ctx in [int(c) for c in a.context.split(",")]:
        ids = build_ids(tok, pool, ctx)
        here = measure_cost(target, ids)                     # check cost at this context length
        table = {"prompt_tokens": len(ids), "cost_ms": [round(1000 * c, 1) for c in here[1:]]}
        print(f"[long] ctx={len(ids)} cost ms", table["cost_ms"], flush=True)
        target.enable_prompt_cache(True)
        # (name, cost table, drafter window, generator options)
        rows = [
            ("fixed 16", None, window0, {}),
            ("adaptive, costs at this context", here, window0, {}),
            ("adaptive, short costs + live cost", short, window0, dict(live_cost=True)),
            ("adaptive, costs here, tree", here, window0, dict(tree=True)),
            ("adaptive, short costs + live cost, tree", short, window0, dict(tree=True, live_cost=True)),
            ("live cost + copy drafts after 12", short, window0, dict(live_cost=True, copy=True, copy_min=12)),
            ("live cost + copy drafts after 6", short, window0, dict(live_cost=True, copy=True, copy_min=6)),
            ("live cost + tree + copy branch 8 after 3", short, window0,
             dict(live_cost=True, tree=True, tree_copy=8, tree_copy_min=3)),
            ("live cost + tree + copy branch 12 after 2", short, window0,
             dict(live_cost=True, tree=True, tree_copy=12, tree_copy_min=2)),
        ]
        for name, cost, window, opts in rows:
            set_window(draft, window)
            gen = make(cost, **opts)
            _, st = gen.generate(ids, max_tokens=a.tokens, stop_ids=stop)
            rounds = max(st.rounds, 1)
            row = dict(tps=round(st.generated_tokens / st.decode_seconds, 1),
                       tokens_per_round=round(st.generated_tokens / rounds, 2),
                       ms_per_round=round(1000 * st.decode_seconds / rounds, 1),
                       phase_ms={k: round(1000 * v / rounds, 1) for k, v in gen.phase_seconds.items()},
                       prefill_s=round(st.prefill_seconds, 1))
            lanes = {k: v for k, v in gen.lane_stats.items() if v and k != "draft_rounds"}
            if lanes:
                row["lanes"] = lanes
            if gen.len_hist:
                row["verify_len_hist"] = dict(sorted(gen.len_hist.items()))
            table[name] = row
            print(f"[long] ctx={len(ids)} {name} {json.dumps(row)}", flush=True)
            mx.clear_cache()
        set_window(draft, window0)
        res["runs"][ctx] = table
    res["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
