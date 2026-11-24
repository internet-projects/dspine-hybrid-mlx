"""Long-context checks with and without the multi-row attention kernel (dspine_hybrid.long_attn).

For each context length: the check cost per width with stock attention and with the kernel, then
the same generations with it off and on (speed, tokens per round, and whether the text matches).

    python scripts/bench_long_attn.py --target <4bit> --draft <drafter> --out long_attn.json
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
    p.add_argument("--context", default="16384,32768")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--windows", default="6-15-16384,5-15-8192", help="rows from-to-minimum keys, comma separated")
    p.add_argument("--memory-limit-gb", type=float, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    apply_memory_rules(a.memory_limit_gb)
    base, target, _, _ = load(a.target, a.draft, fast_verify=True)
    draft = base.draft
    tok = target.tokenizer
    stop = _load_tokenizer_eos(tok)
    pool = [x for xs in prompts(20).values() for x in xs]
    res = {"target": a.target, "tokens": a.tokens, "fast_verify": bool(target.fast_verify), "runs": {}}
    modes = [("stock", None)] + [(f"kernel {w}", tuple(int(x) for x in w.split("-"))) for w in a.windows.split(",")]

    def switch(window):
        on = target.enable_long_attention(window is not None, window)
        return on or window is None

    short = measure_cost(target, list(range(1000, 1256)))

    def make(cost, **opts):
        return LaneDSpineGenerator(
            target, draft, block_size=16, adaptive=True, cost=cost, confidence="calibrated",
            calibration=cal.load(None), sampled_calibration=cal.load(None, "sampled"), **{"copy": False, **opts})

    for ctx in [int(c) for c in a.context.split(",")]:
        ids = build_ids(tok, pool, ctx)
        table = {"prompt_tokens": len(ids)}
        texts = {}
        for mode, window in modes:
            if not switch(window):
                print(f"[lattn] ctx={len(ids)} {mode}: kernel not available here", flush=True)
                continue
            here = measure_cost(target, ids)
            table[f"cost_ms {mode}"] = [round(1000 * c, 1) for c in here[1:]]
            print(f"[lattn] ctx={len(ids)} {mode} cost ms", table[f"cost_ms {mode}"], flush=True)
            target.enable_prompt_cache(True)
            rows = [
                ("adaptive, costs here", here, {}),
                ("adaptive, costs here, tree", here, dict(tree=True)),
                ("live cost + tree + copy branch 8 after 3", short, dict(live_cost=True, tree=True, tree_copy=8, tree_copy_min=3)),
            ]
            for name, cost, opts in rows:
                gen = make(cost, **opts)
                out, st = gen.generate(ids, max_tokens=a.tokens, stop_ids=stop)
                rounds = max(st.rounds, 1)
                row = dict(tps=round(st.generated_tokens / st.decode_seconds, 1),
                           tokens_per_round=round(st.generated_tokens / rounds, 2),
                           ms_per_round=round(1000 * st.decode_seconds / rounds, 1),
                           phase_ms={k: round(1000 * v / rounds, 1) for k, v in gen.phase_seconds.items()})
                if gen.len_hist:
                    row["verify_len_hist"] = dict(sorted(gen.len_hist.items()))
                out = [int(t) for t in out]
                ref = texts.setdefault(name, out)
                same = sum(1 for x, y in zip(ref, out) if x == y)
                row["same_tokens_as_stock"] = f"{same}/{min(len(ref), len(out))}"
                table[f"{mode} | {name}"] = row
                print(f"[lattn] ctx={len(ids)} {mode} | {name} {json.dumps(row)}", flush=True)
                mx.clear_cache()
            target.enable_prompt_cache(False)
        switch(None)
        res["runs"][ctx] = table
    res["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
