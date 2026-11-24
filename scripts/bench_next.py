"""Three follow-up tests with the verify kernels on, on one target.

1. The drafter at 4-bit, 8-bit and full precision: does a more exact drafter get more tokens accepted?
2. A two-level adaptive rule (full_from): a chosen length of N or more becomes the whole block.
3. A wider draft tree (31 nodes) on top of adaptive.
Greedy and sampled (Qwen settings) on code and prose. Takes about ten minutes.

    python scripts/bench_next.py --target <4-bit build> --draft <drafter> --out next.json
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
from dspine_hybrid.draft import load_draft  # noqa: E402
from dspine_hybrid.lanes import LaneDSpineGenerator  # noqa: E402
from dspine_hybrid.run import apply_memory_rules, load  # noqa: E402
from dspine_mlx.generate import _load_tokenizer_eos  # noqa: E402
from dspine_mlx.model import quantize_draft  # noqa: E402


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
    base, target, _, _ = load(a.target, a.draft, fast_verify=True)      # kernels, drafter kernels, flush
    target.enable_prompt_cache(False)
    tok = target.tokenizer
    stop = _load_tokenizer_eos(tok)
    raw = prompts(a.n)
    sets = {"code": raw["humaneval"][: a.n], "prose": raw["mtbench"][: a.n]}
    res = {"target": a.target, "tokens": a.tokens, "fast_verify": bool(target.fast_verify)}
    print("[next] fast verify kernels:", res["fast_verify"], flush=True)

    drafts = {4: base.draft}

    def draft_for(bits):
        if bits not in drafts:
            d, _, _ = load_draft(a.draft)
            if bits:
                d = quantize_draft(d, bits=int(bits), group_size=64, mode="affine")
            d.bind(target.model)
            d.fast_kernels = bool(target.fast_verify)
            drafts[bits] = d
        return drafts[bits]

    cost = measure_cost(target, list(range(1000, 1256)), max_T=33)
    res["cost_ms"] = [round(1000 * c, 1) for c in cost[1:]]
    print("[next] cost ms", res["cost_ms"], flush=True)

    def make(bits=4, adaptive=True, **opts):
        return LaneDSpineGenerator(
            target, draft_for(bits), block_size=16, adaptive=adaptive, cost=cost if adaptive else None,
            confidence="calibrated" if adaptive else "vocab", calibration=cal.load(None) if adaptive else None,
            sampled_calibration=cal.load(None, "sampled") if adaptive else None, copy=False, **opts)

    make().generate(chat_ids(tok, "Say hello."), max_tokens=16, stop_ids=stop)

    def run(gen, texts, **kw):
        n = s = rounds = 0.0
        outs, phase = [], {}
        for text in texts:
            mx.random.seed(0)
            out, st = gen.generate(chat_ids(tok, text), max_tokens=a.tokens, stop_ids=stop, **kw)
            n += st.generated_tokens
            s += st.decode_seconds
            rounds += st.rounds
            outs.append(out)
            for k, v in gen.phase_seconds.items():
                phase[k] = phase.get(k, 0.0) + v
            mx.clear_cache()
        rounds = max(rounds, 1)
        return dict(tps=round(n / s, 1), tokens_per_round=round(n / rounds, 2),
                    ms_per_round=round(1000 * s / rounds, 1),
                    phase_ms={k: round(1000 * v / rounds, 1) for k, v in phase.items()}), outs

    # (name, drafter bits (0 = as published), adaptive, options)
    greedy = [
        ("q4 adaptive", 4, True, {}),
        ("q4 fixed 16", 4, False, {}),
        ("q4 adaptive full from 6", 4, True, dict(full_from=6)),
        ("q4 adaptive full from 3", 4, True, dict(full_from=3)),
        ("q4 adaptive + tree 15", 4, True, dict(tree=True)),
        ("q4 adaptive + tree 31", 4, True, dict(tree=True, tree_nodes=31)),
        ("q8 adaptive", 8, True, {}),
        ("q8 fixed 16", 8, False, {}),
        ("full adaptive", 0, True, {}),
        ("full fixed 16", 0, False, {}),
        ("full adaptive + tree 31", 0, True, dict(tree=True, tree_nodes=31)),
    ]
    sampled = [
        ("q4 adaptive", 4, True, {}),
        ("q4 adaptive full from 6", 4, True, dict(full_from=6)),
        ("q8 adaptive", 8, True, {}),
        ("full adaptive", 0, True, {}),
        ("full fixed 16", 0, False, {}),
    ]
    kw = dict(temperature=0.7, top_k=20, top_p=0.8)
    for mode, rows, extra in (("greedy", greedy, {}), ("sampled", sampled, kw)):
        res[mode] = {}
        for bench, texts in sets.items():
            table, ref = {}, None
            for name, bits, adaptive, opts in rows:
                row, outs = run(make(bits, adaptive, **opts), texts, **extra)
                if mode == "greedy":
                    ref = outs if ref is None else ref
                    row["same_as_first"] = sum(o == r for o, r in zip(outs, ref))
                table[name] = row
                print("[next]", mode, bench, name, row, flush=True)
            res[mode][bench] = table
    res["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
