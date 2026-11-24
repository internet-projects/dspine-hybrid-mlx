"""Throughput of batched rounds: the same conversations decoded 1, 2 and 4 at a time.

    python scripts/bench_batch_gen.py --target mlx-community/Qwen3.8-27B-4bit \\
        --draft netprojects/Qwen3.8-27B-DSpine-b16-MLX --out batch.json

Greedy, thinking off. Per prompt set (code, chat) it decodes --n conversations with max_batch
1 (one after the other), 2 and 4, and reports tokens per second of round time (prefill
excluded), tokens per round and whether every conversation got the tokens it gets alone.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402

from dspine_hybrid.adaptive import measure_cost  # noqa: E402
from dspine_hybrid.batch_gen import BatchGenerator, Conversation, DSpineProposer, measure_batch_cost  # noqa: E402
from dspine_hybrid.run import apply_memory_rules, load  # noqa: E402
from dspine_mlx.generate import _load_tokenizer_eos  # noqa: E402

CODE = [
    "Write a Python function that merges two sorted lists, with a docstring and three tests.",
    "Write a Python class LRUCache with get and put in O(1) and five unit tests.",
    "Write a Python function that parses an ISO 8601 date string without external libraries, with tests.",
    "Write a Python implementation of Dijkstra's algorithm with a heap and a small example.",
    "Write a Python function that validates an IPv4 address without regular expressions, with tests.",
    "Write a Python class for a fixed-size ring buffer with append, iteration and four unit tests.",
    "Write a Python function that flattens a nested list of any depth, iteratively, with tests.",
    "Write a Python command line tool with argparse that counts words, lines and characters in a file.",
]
CHAT = [
    "Explain to a ten year old why the sky is blue and why sunsets are red.",
    "Give me a three day plan for visiting Athens with a family, with one tip per day.",
    "What are the pros and cons of working from home? Write four short paragraphs.",
    "Tell a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a refrigerator works to someone who has never studied physics.",
    "Write a friendly email inviting a new neighbour to a street party next Saturday.",
    "What should I think about before adopting a dog? Give me five points with a sentence each.",
    "Describe a walk through an old harbour town on a rainy morning, in three paragraphs.",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--batches", default="1,2,4")
    p.add_argument("--per-conversation", action="store_true",
                   help="older length choice: each conversation sizes its own check from the one-conversation cost")
    p.add_argument("--no-kernels", action="store_true")
    p.add_argument("--calibration", default=None, help="acceptance fit from scripts/fit_acceptance.py")
    p.add_argument("--block", type=int, default=None, help="drafter block (default 16)")
    p.add_argument("--drafter-per-conversation", action="store_true", help="run the drafter once per conversation (as before the batched drafter pass)")
    p.add_argument("--copy", action="store_true", help="copy drafts (the drafter rests when the text repeats)")
    p.add_argument("--modes", default="chain", help="comma list of chain, tree (draft tree per greedy conversation)")
    p.add_argument("--memory-limit-gb", type=float, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    apply_memory_rules(a.memory_limit_gb)
    fast = not a.no_kernels
    fitted = None
    if a.calibration:
        from dspine_hybrid import calibration as cal

        fitted = cal.load(a.calibration, "greedy")
        print(f"[batch] acceptance fit {a.calibration}: scale {fitted['scale']:.3f}", flush=True)
    gen, target, _, _ = load(a.target, a.draft, fast_verify=fast, a8_prefill=fast, prompt_cache=False,
                             block_size=a.block or 16)
    proposer = DSpineProposer(gen, calibration=fitted)
    proposer.batched = not a.drafter_per_conversation
    print(f"[batch] fast verify kernels: {'on' if target.fast_verify else 'off'}, "
          f"int8 prefill: {'on' if target.a8_prefill is not None else 'off'}", flush=True)
    tok = target.tokenizer
    stop = tuple(_load_tokenizer_eos(tok))
    if a.per_conversation:
        cost = measure_cost(target, list(range(1000, 1256)))
        table = {1: cost}
    else:
        cost = table = measure_batch_cost(target, list(range(1000, 1256)), max_batch=max(int(x) for x in a.batches.split(",")), max_T=max(proposer.max_block, 16) + 1)
    for B, c in table.items():
        print(f"[batch] ms per T-row check, {B} conversation(s):", " ".join(f"{1000 * x:.0f}" for x in c[1:]), flush=True)
    tree_cost = cost
    if "tree" in a.modes.split(",") and not a.per_conversation:
        tree_cost = measure_batch_cost(target, list(range(1000, 1256)), max_batch=max(table), max_T=max(proposer.max_block, 16) + 1, tree=True)
        for B, c in tree_cost.items():
            print(f"[batch] ms per T-row TREE check, {B} conversation(s):", " ".join(f"{1000 * x:.0f}" for x in c[1:]), flush=True)
    res = {"drafter": "dspine", "tokens": a.tokens, "n": a.n, "cost_ms": {B: [round(1000 * x, 1) for x in c[1:]] for B, c in table.items()}, "sets": {}}
    for name, texts in (("code", CODE), ("chat", CHAT)):
        ids = [tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                       enable_thinking=False) for t in texts[: a.n]]
        for mode in a.modes.split(","):
            ref = None
            rows = {}
            for B in [int(x) for x in a.batches.split(",")]:
                for rep in range(2):                    # the first pass of a new batch shape warms up
                    bg = BatchGenerator(target, proposer, max_batch=B, cost=tree_cost if mode == "tree" else cost,
                                        tree=mode == "tree", tree_from=1, copy=a.copy)
                    t0 = time.perf_counter()
                    convs = bg.run([Conversation(x, max_tokens=a.tokens, stop_ids=stop) for x in ids])
                    wall = time.perf_counter() - t0
                    mx.clear_cache()
                outs = [c.tokens for c in convs]
                if ref is None:
                    ref = outs
                n_tok = sum(len(o) for o in outs)
                per = sorted(len(c.tokens) / max(c.stats.decode_seconds, 1e-9) for c in convs)   # what each user sees
                r = dict(tok_s=round(n_tok / bg.round_seconds, 1), wall_tok_s=round(n_tok / wall, 1),
                         per_user_tok_s=[round(per[0], 1), round(per[len(per) // 2], 1)], tokens=n_tok, rounds=bg.rounds, ms_per_round=round(1000 * bg.round_seconds / bg.rounds, 1),
                         tokens_per_conv_round=round(sum(c.stats.committed_tokens for c in convs)
                                                     / max(sum(c.stats.rounds for c in convs), 1), 2),
                         same_as_one_at_a_time=f"{sum(o == q for o, q in zip(outs, ref))}/{len(outs)}",
                         rows=sorted((f"{b}x{m}", n) for (b, m), n in bg.row_hist.items())[-6:],
                         tree=dict(bg.tree_stats), side_scale=round(bg.side_scale, 2), check_ratio={k: round(v, 2) for k, v in bg._check_ratio.items()}, kept={k: round(v, 2) for k, v in bg._kept.items()}, copy=dict(bg.copy_stats),
                         phase_ms={k: round(1000 * v / bg.rounds, 1) for k, v in bg.phase_seconds.items()}, peak_gb=round(mx.get_peak_memory() / 1e9, 1))
                rows[str(B)] = r
                print(f"[batch] {name} {mode} {B} at a time: {json.dumps(r)}", flush=True)
            base = rows[min(rows, key=int)]["tok_s"]
            for B, r in rows.items():
                r["speedup"] = round(r["tok_s"] / base, 2)
            print(f"[batch] {name} {mode}: " + ", ".join(f"{B} at a time {r['tok_s']} tok/s (x{r['speedup']})" for B, r in rows.items()), flush=True)
            res["sets"][f"{name}/{mode}"] = rows
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
