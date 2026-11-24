"""Which verify kernel is best at each row count, on one target.

oMLX's kernel file holds four kernel families and picks one per row count and chip: its own
tensor-unit kernel ("skinny", 9 to 48 rows on M5), "sg8" (4 to 8 rows, operand trick from Splash),
"mma" (7 to 24 rows, adapted from dflash-mlx) and "vk" (3 to 6 rows, split-K adapted from MTPLX).
This script overrides that choice and times the target check for 1 to 17 rows under each variant,
then runs greedy prose and code with adaptive for every variant. It measures oMLX's adaptations
of those kernels, not the upstream packages themselves. Takes about five minutes.

    python scripts/bench_routes.py --target <4-bit build> --draft <drafter> --out routes.json
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
from dspine_hybrid import fast_verify  # noqa: E402
from dspine_hybrid.adaptive import measure_cost  # noqa: E402
from dspine_hybrid.lanes import LaneDSpineGenerator  # noqa: E402
from dspine_hybrid.run import apply_memory_rules, load  # noqa: E402
from dspine_mlx.generate import _load_tokenizer_eos  # noqa: E402


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
    mod = fast_verify._STATE["mod"]
    res = {"target": a.target, "fast_verify": bool(target.fast_verify), "variants": {}}
    print("[routes] fast verify kernels:", res["fast_verify"], flush=True)
    if mod is None:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        return
    tok = target.tokenizer
    stop = _load_tokenizer_eos(tok)
    raw = prompts(a.n)
    sets = {"code": raw["humaneval"][: a.n], "prose": raw["mtbench"][: a.n]}
    keep = dict(skinny=mod.skinny_eligible, sg8=mod.sg8_eligible, nax_min=mod._SG8_SINGLE_MIN_ROWS_NAX,
                min_n=mod._MIN_ROUTE_N)

    def restore():
        mod.skinny_eligible, mod.sg8_eligible = keep["skinny"], keep["sg8"]
        mod._SG8_SINGLE_MIN_ROWS_NAX, mod._MIN_ROUTE_N = keep["nax_min"], keep["min_n"]

    def v_default():
        pass

    def v_mma():                      # dflash-mlx's tile kernel instead of the tensor-unit one at 9..16 rows
        mod.skinny_eligible = lambda *x: False

    def v_sg8_4():                    # sg8 from 4 rows instead of 6
        mod._SG8_SINGLE_MIN_ROWS_NAX = 4

    def v_vk():                       # MTPLX's split-K for every projection at 3..6 rows
        mod.sg8_eligible = lambda M, *x: False if int(M) <= 6 else keep["sg8"](M, *x)
        mod._MIN_ROUTE_N = 1024

    def v_vk_head():                  # split-K only for wide outputs (oMLX's own rule), sg8 off below 6 rows
        mod.sg8_eligible = lambda M, *x: False if int(M) <= 5 else keep["sg8"](M, *x)

    variants = [("oMLX default", v_default), ("mma at 9-16 rows", v_mma), ("sg8 from 4 rows", v_sg8_4),
                ("split-K at 3-6 rows, all projections", v_vk), ("split-K at 3-5 rows, wide outputs only", v_vk_head)]
    for name, setup in variants:
        restore()
        setup()
        row = {"routes": {r: fast_verify.routes(target.model, r, mx.bfloat16) for r in (3, 4, 5, 6, 9, 16)}}
        try:
            cost = measure_cost(target, list(range(1000, 1256)))
            row["cost_ms"] = [round(1000 * c, 1) for c in cost[1:]]
            print("[routes]", name, "cost ms", row["cost_ms"], flush=True)
            for bench, texts in sets.items():
                gen = LaneDSpineGenerator(
                    target, base.draft, block_size=16, adaptive=True, cost=cost, confidence="calibrated",
                    calibration=cal.load(None), sampled_calibration=cal.load(None, "sampled"), copy=False)
                n = s = rounds = 0.0
                for text in texts:
                    _, st = gen.generate(chat_ids(tok, text), max_tokens=a.tokens, stop_ids=stop)
                    n += st.generated_tokens
                    s += st.decode_seconds
                    rounds += st.rounds
                    mx.clear_cache()
                row[bench] = dict(tps=round(n / s, 1), tokens_per_round=round(n / max(rounds, 1), 2),
                                  ms_per_round=round(1000 * s / max(rounds, 1), 1))
                print("[routes]", name, bench, row[bench], flush=True)
        except Exception as e:      # a kernel that does not build for some shape: record and go on
            row["error"] = repr(e)[:300]
            print("[routes]", name, "ERROR", row["error"], flush=True)
        res["variants"][name] = row
    restore()
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
