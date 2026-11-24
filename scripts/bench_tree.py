"""What a draft tree costs next to a chain, with the verify kernels on, on one target.

Part 1 times the target check and the cache update after it, for a chain and for trees of the same
number of rows (a comb: a trunk with one extra leaf per level; two and four equal branches).
Part 2 runs greedy prose and code with adaptive and with adaptive + tree and reports the time per
round in four parts: drafter, choosing the tree, the check, and the rest. Takes about four minutes.

    python scripts/bench_tree.py --target <4-bit build> --draft <drafter> --out tree.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402

from bench_accept import prompts  # noqa: E402
from bench_lanes import chat_ids  # noqa: E402
from dspine_hybrid import calibration as cal  # noqa: E402
from dspine_hybrid.adaptive import measure_cost  # noqa: E402
from dspine_hybrid.lanes import LaneDSpineGenerator  # noqa: E402
from dspine_hybrid.run import apply_memory_rules, load  # noqa: E402
from dspine_mlx.generate import _load_tokenizer_eos  # noqa: E402


def shapes(rows):
    """Parent lists (row 0 = pending token, parent -1) with ``rows`` rows each."""
    chain = [-1] + list(range(rows - 1))
    comb, trunk = [-1], 0                       # trunk node, then a sibling leaf of the next trunk node
    while len(comb) < rows:
        comb.append(trunk)
        nxt = len(comb) - 1
        if len(comb) < rows:
            comb.append(trunk)
        trunk = nxt

    def branches(k):
        parents, size = [-1], max(1, (rows - 1 + k - 1) // k)
        for i in range(1, rows):
            parents.append(0 if (i - 1) % size == 0 else i - 1)
        return parents

    return {"chain as tree": chain, "comb": comb, "two branches": branches(2), "four branches": branches(4)}


def first_path(parents):
    """Rows of the path that follows each node's first child from row 0."""
    path = [0]
    while True:
        kids = [i for i, p in enumerate(parents) if p == path[-1]]
        if not kids:
            return path
        path.append(kids[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--tokens", type=int, default=384)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--memory-limit-gb", type=float, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    apply_memory_rules(a.memory_limit_gb)
    base, target, _, _ = load(a.target, a.draft, fast_verify=True)
    target.enable_prompt_cache(False)
    res = {"target": a.target, "fast_verify": bool(target.fast_verify), "check_ms": {}}
    print("[tree] fast verify kernels:", res["fast_verify"], flush=True)

    def states():
        mx.eval([c.state for c in target.cache])

    def fresh():
        target.reset()
        target.prefill(mx.array([list(range(1000, 1256))]), capture=False)
        states()

    def med(ts):
        return round(1000 * sorted(ts)[len(ts) // 2], 1)

    for rows in (4, 8, 16):
        row = {}
        toks = [1000 + i for i in range(rows)]
        keep = max(2, rows // 2)                # rows of the accepted path that get committed
        tv, tc = [], []
        for r in range(a.reps + 1):
            fresh()
            t0 = time.perf_counter()
            lg, h = target.verify(mx.array([toks]))
            mx.eval(lg, h)
            t1 = time.perf_counter()
            target.trim(rows - keep)
            states()
            t2 = time.perf_counter()
            if r:
                tv.append(t1 - t0)
                tc.append(t2 - t1)
        row["chain"] = dict(check=med(tv), update=med(tc))
        for name, parents in shapes(rows).items():
            path = first_path(parents)[:keep]
            tv, tc = [], []
            for r in range(a.reps + 1):
                fresh()
                t0 = time.perf_counter()
                lg, h = target.verify_tree(toks, parents)
                mx.eval(lg, h)
                t1 = time.perf_counter()
                target.commit_tree(path)
                states()
                t2 = time.perf_counter()
                if r:
                    tv.append(t1 - t0)
                    tc.append(t2 - t1)
            row[name] = dict(check=med(tv), update=med(tc), kept=len(path))
        res["check_ms"][rows] = row
        print("[tree] rows", rows, row, flush=True)

    import dspine_hybrid.hybrid_target as ht

    res["tree_by_node"], res["tree_kernel_ok"] = bool(ht.TREE_BY_NODE), ht._tree_kernel_ok
    print("[tree] recurrence by node:", ht.TREE_BY_NODE, "tree kernel passed its check:", ht._tree_kernel_ok, flush=True)
    tok = target.tokenizer
    stop = _load_tokenizer_eos(tok)
    raw = prompts(a.n)
    sets = {"prose": raw["mtbench"][: a.n], "code": raw["humaneval"][: a.n]}
    cost = measure_cost(target, list(range(1000, 1256)))
    res["cost_ms"] = [round(1000 * c, 1) for c in cost[1:]]
    res["greedy"] = {}
    for bench, texts in sets.items():
        table = {}
        for name, opts in (("adaptive", {}), ("adaptive + tree", dict(tree=True)),
                           ("adaptive + tree, 7 nodes", dict(tree=True, tree_nodes=7)),
                           ("adaptive + tree + copy branch 4", dict(tree=True, tree_copy=4)),
                           ("adaptive + tree + copy branch 8 after 3", dict(tree=True, tree_copy=8, tree_copy_min=3))):
            gen = LaneDSpineGenerator(
                target, base.draft, block_size=16, adaptive=True, cost=cost, confidence="calibrated",
                calibration=cal.load(None), sampled_calibration=cal.load(None, "sampled"), copy=False, **opts)
            n = s = rounds = 0.0
            phase, lanes, lens = {}, {}, {}
            for text in texts:
                _, st = gen.generate(chat_ids(tok, text), max_tokens=a.tokens, stop_ids=stop)
                n += st.generated_tokens
                s += st.decode_seconds
                rounds += st.rounds
                for k, v in gen.phase_seconds.items():
                    phase[k] = phase.get(k, 0.0) + v
                for k, v in gen.lane_stats.items():
                    lanes[k] = lanes.get(k, 0) + v
                for k, v in gen.len_hist.items():
                    lens[k] = lens.get(k, 0) + v
                mx.clear_cache()
            rounds = max(rounds, 1)
            table[name] = dict(tps=round(n / s, 1), tokens_per_round=round(n / rounds, 2),
                               ms_per_round=round(1000 * s / rounds, 1),
                               phase_ms={k: round(1000 * v / rounds, 1) for k, v in phase.items()},
                               rows_checked=dict(sorted(lens.items())),
                               tree_rounds=lanes.get("tree_rounds", 0), tree_nodes=lanes.get("tree_nodes", 0),
                               tree_side_tokens=lanes.get("tree_side_tokens", 0),
                               graft_nodes=lanes.get("graft_nodes", 0), graft_tokens=lanes.get("graft_tokens", 0))
            print("[tree]", bench, name, table[name], flush=True)
        res["greedy"][bench] = table
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
