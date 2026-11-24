"""Does a second drafter pass that knows the first token raise tokens per round? (greedy, tree on)

Runs prose and code with adaptive + tree, then with the twin pass (dspine_hybrid.twin) in a few
settings: for the best first token only, for the best two, and only in rounds where the best first
token is uncertain. Reports tok/s, tokens per round, ms per round by phase and how many rounds
ran the twin pass. The output is the same in every row (the target checks every node).

    python scripts/bench_twin.py --target <4-bit build> --draft <drafter> --out twin.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402

from bench_accept import prompts  # noqa: E402
from bench_lanes import chat_ids  # noqa: E402
from dspine_hybrid import calibration as cal  # noqa: E402
from dspine_hybrid.adaptive import measure_cost  # noqa: E402
from dspine_hybrid.lanes import LaneDSpineGenerator  # noqa: E402
from dspine_hybrid.run import apply_memory_rules, load  # noqa: E402
from dspine_mlx.generate import _load_tokenizer_eos  # noqa: E402

VARIANTS = (
    ("tree", {}),
    ("tree + twin 1", dict(twin=1)),
    ("tree + twin 2", dict(twin=2)),
    ("tree + twin 2 below 0.8", dict(twin=2, twin_below=0.8)),
    ("tree + twin 2 below 0.6", dict(twin=2, twin_below=0.6)),
    ("tree + twin 3 below 0.6", dict(twin=3, twin_below=0.6)),
    ("tree again", {}),
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--tokens", type=int, default=384)
    p.add_argument("--memory-limit-gb", type=float, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    apply_memory_rules(a.memory_limit_gb)
    base, target, _, _ = load(a.target, a.draft, fast_verify=True)
    target.enable_prompt_cache(False)
    tok = target.tokenizer
    stop = _load_tokenizer_eos(tok)
    raw = prompts(a.n)
    sets = {"prose": raw["mtbench"][: a.n], "code": raw["humaneval"][: a.n]}
    cost = measure_cost(target, list(range(1000, 1256)))
    res = {"target": a.target, "cost_ms": [round(1000 * c, 1) for c in cost[1:]], "greedy": {}}
    for bench, texts in sets.items():
        table, first = {}, None
        for name, opts in VARIANTS:
            gen = LaneDSpineGenerator(
                target, base.draft, block_size=16, adaptive=True, cost=cost, confidence="calibrated",
                calibration=cal.load(None), sampled_calibration=cal.load(None, "sampled"), copy=False, tree=True, **opts)
            n = s = rounds = 0.0
            phase, lanes, outs = {}, {}, []
            for text in texts:
                out, st = gen.generate(chat_ids(tok, text), max_tokens=a.tokens, stop_ids=stop)
                outs.append(out)
                n += st.generated_tokens
                s += st.decode_seconds
                rounds += st.rounds
                for k, v in gen.phase_seconds.items():
                    phase[k] = phase.get(k, 0.0) + v
                for k, v in gen.lane_stats.items():
                    lanes[k] = lanes.get(k, 0) + v
                mx.clear_cache()
            first = first or outs
            rounds = max(rounds, 1)
            table[name] = dict(tps=round(n / s, 1), tokens_per_round=round(n / rounds, 2),
                               ms_per_round=round(1000 * s / rounds, 1),
                               phase_ms={k: round(1000 * v / rounds, 1) for k, v in phase.items()},
                               twin_rounds=lanes.get("twin_rounds", 0), draft_rounds=lanes.get("draft_rounds", 0),
                               tree_side_tokens=lanes.get("tree_side_tokens", 0),
                               same_output=sum(x == y for x, y in zip(outs, first)), prompts=len(outs))
            print("[twin]", bench, name, table[name], flush=True)
        res["greedy"][bench] = table
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
