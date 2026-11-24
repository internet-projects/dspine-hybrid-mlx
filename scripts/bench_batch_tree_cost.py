"""Where a tree check of one conversation spends more than a chain check (Mac job ds-69).

Times, on a short prompt, rounds of 15 rows: check + keep for a chain, and for trees of
several shapes with a kept path that is a prefix of the rows or not. Each line is the median
of --reps rounds, the keep included (its work is lazy and lands in the next check).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402

from dspine_hybrid.batch_target import BatchTarget  # noqa: E402
from dspine_hybrid.run import apply_memory_rules, load  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--rows", type=int, default=15)
    p.add_argument("--reps", type=int, default=12)
    p.add_argument("--memory-limit-gb", type=float, default=None)
    a = p.parse_args()
    apply_memory_rules(a.memory_limit_gb)
    _, target, _, _ = load(a.target, a.draft, fast_verify=True, a8_prefill=True, prompt_cache=False, block_size=16)
    M = a.rows
    chain = list(range(-1, M - 1))
    comb = [-1, 0] + [(i - 2 if i % 2 == 0 else i - 3) for i in range(2, M)]
    binary = [-1] + [(i - 1) // 2 for i in range(1, M)]

    def spine(par):                     # the longest path from row 0, always the first child
        path = [0]
        while True:
            kids = [i for i, q in enumerate(par) if q == path[-1]]
            if not kids:
                return path
            path.append(kids[0])

    def last_kids(par):                 # a path that takes the last child: not a prefix of the rows
        path = [0]
        while True:
            kids = [i for i, q in enumerate(par) if q == path[-1]]
            if not kids:
                return path
            path.append(kids[-1])

    cases = [("chain check, keep 8", None, 8), ("chain check, keep all", None, M),
             ("chain as tree, keep 8", chain, list(range(8))),
             ("comb tree, keep spine", comb, spine(comb)), ("comb tree, keep 2 rows", comb, spine(comb)[:2]),
             ("binary tree, keep first children", binary, spine(binary)),
             ("binary tree, keep last children", binary, last_kids(binary)),
             ("binary tree, keep nothing", binary, [])]
    ids = mx.array([list(range(1000, 1256))])
    tok = mx.full((1, M), 1255, dtype=mx.int32)
    for name, par, keep in cases:
        target.reset()
        target.prefill(ids, capture=False)
        bt = BatchTarget(target)
        bt.join()
        ts = []
        for r in range(a.reps + 2):
            t0 = time.perf_counter()
            if par is None:
                lg, h = bt.verify(tok)
                mx.eval(mx.argmax(lg, axis=-1), h)
                bt.keep([keep])
            else:
                lg, h = bt.verify_tree(tok, [par])
                mx.eval(mx.argmax(lg, axis=-1), h)
                bt.keep_paths([keep])
            mx.eval([c.state for c in bt.cache])
            if r >= 2:
                ts.append(time.perf_counter() - t0)
        ts.sort()
        print(f"[treecost] {name}: {1000 * ts[len(ts) // 2]:.1f} ms (min {1000 * ts[0]:.1f})", flush=True)
        del bt
        mx.clear_cache()


if __name__ == "__main__":
    main()
