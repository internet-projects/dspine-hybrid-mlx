"""The cheap wide check: stock MLX against oMLX's verify kernels, on one target.

Needs the ``omlx`` package importable (source install with --no-deps is enough). Measures the
target check per width, then greedy decode speed on code and prose for the measured v0.3 rounds
and for the same rounds with the kernels, the drafter on the kernels too, the graph flush, wide
fixed blocks, drafting fewer rows when few are verified, and copy drafts plus the tree on top.
About fifteen minutes.

    python scripts/bench_fast_verify.py --target <4-bit build> --draft <drafter> --out fast.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402

from bench_accept import prompts  # noqa: E402
from bench_lanes import EDIT, chat_ids  # noqa: E402
from dspine_hybrid import calibration as cal  # noqa: E402
from dspine_hybrid import fast_verify  # noqa: E402
from dspine_hybrid.adaptive import measure_cost  # noqa: E402
from dspine_hybrid.lanes import LaneDSpineGenerator  # noqa: E402
from dspine_hybrid.run import apply_memory_rules, load  # noqa: E402
from dspine_mlx.generate import _load_tokenizer_eos  # noqa: E402


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


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
    base, target, _, _ = load(a.target, a.draft)
    target.enable_prompt_cache(False)
    draft = base.draft
    tok = target.tokenizer
    stop = _load_tokenizer_eos(tok)
    raw = prompts(a.n)
    sets = {"code": raw["humaneval"][: a.n], "prose": raw["mtbench"][: a.n], "edit": EDIT[: a.n]}
    res = {"target": a.target, "tokens": a.tokens, "block": base.block_size}

    have = fast_verify.install()
    res["omlx_kernels"] = have
    print("[fast] oMLX verify kernels available:", have, flush=True)
    if have:
        from omlx.patches import qwen35_verify_qmm as q

        res["nax"] = bool(q.is_nax_available())
        res["routes"] = {rows: fast_verify.routes(target.model, rows, mx.bfloat16) for rows in (2, 5, 6, 9, 16)}
        print("[fast] tensor-unit GPU:", res["nax"], "routes", res["routes"], flush=True)

    if have:
        target.enable_fast_verify(True)      # installs the routed head; the flags below switch it per row

    def setup(kernels=False, draft_kernels=False, flush=0):
        # the drafter's head logits follow target.fast_verify, its backbone follows draft.fast_kernels
        target.fast_verify = bool(kernels and have)
        draft.fast_kernels = bool(draft_kernels and have)
        target.enable_flush(flush) if (flush or target._flush_box is not None) else None

    costs = {}

    def cost_for(name, **kw):
        setup(**kw)
        if name not in costs:
            costs[name] = measure_cost(target, list(range(1000, 1256)))
            print("[fast] cost ms", name, [round(1000 * c, 1) for c in costs[name][1:]], flush=True)
        return costs[name]

    def make(cost=None, block=None, copy=False, tree=False, auto_rows=False):
        adaptive = cost is not None
        return LaneDSpineGenerator(
            target, draft, block_size=block or base.block_size, adaptive=adaptive, cost=cost,
            confidence="calibrated" if adaptive else "vocab", calibration=cal.load(None) if adaptive else None,
            sampled_calibration=cal.load(None, "sampled") if adaptive else None, copy=copy, tree=tree,
            auto_rows=auto_rows)

    make().generate(chat_ids(tok, "Say hello."), max_tokens=16, stop_ids=stop)

    # (row name, setup flags, cost table name or None for a fixed block, generator options)
    K = dict(kernels=True)
    KD = dict(kernels=True, draft_kernels=True)
    KDF = dict(kernels=True, draft_kernels=True, flush=8)
    AR = dict(auto_rows=True)
    rows = [
        ("stock fixed 5", {}, None, {}),
        ("stock adaptive", {}, "stock", {}),
        ("stock adaptive + auto rows", {}, "stock", AR),
    ]
    if have:
        rows += [
            ("kernels fixed 16", K, None, dict(block=16)),
            ("kernels+drafter+flush fixed 16", KDF, None, dict(block=16)),
            ("kernels+drafter+flush adaptive", KDF, "kernels+flush", {}),
            ("kernels+drafter+flush adaptive + auto rows", KDF, "kernels+flush", AR),
            ("kernels+drafter+flush adaptive + auto rows + tree", KDF, "kernels+flush", dict(AR, tree=True)),
            ("kernels+drafter+flush adaptive + auto rows + copy", KDF, "kernels+flush", dict(AR, copy=True)),
            ("kernels+drafter+flush adaptive + auto rows + copy + tree", KDF, "kernels+flush",
             dict(AR, copy=True, tree=True)),
        ]
    res["greedy"] = {}
    for bench, texts in sets.items():
        table, ref = {}, None
        for name, flags, cost_name, opts in rows:
            cost = cost_for(cost_name, **flags) if cost_name else None
            setup(**flags)
            gen = make(cost=cost, **opts)
            n = s = rounds = 0.0
            outs = []
            phase = {}
            for text in texts:
                out, st = gen.generate(chat_ids(tok, text), max_tokens=a.tokens, stop_ids=stop)
                for k, v in gen.phase_seconds.items():
                    phase[k] = phase.get(k, 0.0) + v
                n += st.generated_tokens
                s += st.decode_seconds
                rounds += st.rounds
                outs.append(out)
                mx.clear_cache()
            if ref is None:
                ref = outs
            diffs = [first_diff(o, r) for o, r in zip(outs, ref)]
            table[name] = dict(tps=round(n / s, 1), tokens_per_round=round(n / max(rounds, 1), 2),
                               ms_per_round=round(1000 * s / max(rounds, 1), 1),
                               phase_ms={k: round(1000 * v / max(rounds, 1), 1) for k, v in phase.items()},
                               same_as_stock=sum(d is None for d in diffs), first_diffs=diffs)
            print("[fast]", bench, name, table[name], flush=True)
        res["greedy"][bench] = table
    # sampled decoding (Qwen settings): the plain fixed block against the best kernel rows
    kw = dict(temperature=0.7, top_k=20, top_p=0.8)
    srows = [("stock fixed 5", {}, None, {}), ("stock adaptive", {}, "stock", {})]
    if have:
        srows += [("kernels+drafter+flush adaptive", KDF, "kernels+flush", {}),
                  ("kernels+drafter+flush adaptive + auto rows", KDF, "kernels+flush", AR),
                  ("kernels+drafter+flush adaptive + auto rows + copy", KDF, "kernels+flush", dict(AR, copy=True))]
    res["sampled"] = {}
    for bench, texts in sets.items():
        table = {}
        for name, flags, cost_name, opts in srows:
            cost = cost_for(cost_name, **flags) if cost_name else None
            setup(**flags)
            gen = make(cost=cost, **opts)
            n = s = rounds = 0.0
            for text in texts:
                mx.random.seed(0)
                _, st = gen.generate(chat_ids(tok, text), max_tokens=a.tokens, stop_ids=stop, **kw)
                n += st.generated_tokens
                s += st.decode_seconds
                rounds += st.rounds
                mx.clear_cache()
            table[name] = dict(tps=round(n / s, 1), tokens_per_round=round(n / max(rounds, 1), 2),
                               ms_per_round=round(1000 * s / max(rounds, 1), 1))
            print("[fast] sampled", bench, name, table[name], flush=True)
        res["sampled"][bench] = table
    setup()
    res["cost_ms"] = {k: [round(1000 * c, 1) for c in v[1:]] for k, v in costs.items()}
    res["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
