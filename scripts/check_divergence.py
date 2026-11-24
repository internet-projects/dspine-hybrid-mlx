"""When two settings give different greedy text: is it a rounding tie or a real difference?

Greedy decoding must commit the target's most likely token whatever was drafted. On the GPU the
target's check of a chain, of a tree and of a single token round slightly differently, so where
two tokens are nearly tied the winner can flip, and the texts differ from that point on.

For prose prompts this generates with plain token-by-token decoding, with fixed blocks, with
adaptive, with the tree and with the tree + twin pass. For every setting whose text differs from
plain decoding it reports the first differing position and, from a fresh one-pass read of the
shared prefix, the logit gap between the two tokens there, the gap between the best two tokens,
and the rounding noise of the target on that prompt (one block pass against single steps). A gap
within the noise is a tie; a gap well above it would be a bug.

    python scripts/check_divergence.py --target <4-bit build> --draft <drafter> --out div.json
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
from dspine_mlx.generate import _load_tokenizer_eos, baseline_generate  # noqa: E402


def gap_at(target, prompt, base, out):
    """First differing index and the logits there (None when the texts agree as far as both go)."""
    i = next((j for j in range(min(len(out), len(base))) if out[j] != base[j]), None)
    if i is None:
        return None
    target.reset()
    logits, _ = target.prefill(mx.array([list(prompt) + list(base[:i])]), capture=False)
    row = logits[0, -1].astype(mx.float32)
    top = mx.sort(row)[-2:]
    mx.eval(row, top)
    return dict(at=i, of=len(base), plain_token=int(base[i]), other_token=int(out[i]),
                gap=round(abs(float(row[base[i]].item()) - float(row[out[i]].item())), 4),
                top2_gap=round(float((top[1] - top[0]).item()), 4),
                plain_is_top=bool(int(mx.argmax(row).item()) == int(base[i])),
                other_is_top=bool(int(mx.argmax(row).item()) == int(out[i])))


def noise(target, prompt, toks):
    """Largest logit difference between one pass over ``toks`` and single-token passes."""
    target.reset()
    target.prefill(mx.array([list(prompt)]), capture=False)
    block, _ = target._forward(mx.array([list(toks)]), capture=False)
    mx.eval(block)
    target.reset()
    target.prefill(mx.array([list(prompt)]), capture=False)
    worst = 0.0
    for j, t in enumerate(toks):
        one, _ = target._forward(mx.array([[int(t)]]), capture=False)
        worst = max(worst, float(mx.max(mx.abs(block[0, j].astype(mx.float32) - one[0, 0].astype(mx.float32))).item()))
    return round(worst, 4)


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
    base_gen, target, _, _ = load(a.target, a.draft, fast_verify=True)
    target.enable_prompt_cache(False)
    tok = target.tokenizer
    stop = _load_tokenizer_eos(tok)
    texts = prompts(a.n)["mtbench"][: a.n]
    cost = measure_cost(target, list(range(1000, 1256)))
    kw = dict(confidence="calibrated", calibration=cal.load(None), sampled_calibration=cal.load(None, "sampled"), copy=False)
    settings = (
        ("fixed 16", dict(block_size=16)),
        ("adaptive", dict(block_size=16, adaptive=True, cost=cost)),
        ("tree", dict(block_size=16, adaptive=True, cost=cost, tree=True)),
        ("tree + twin 2", dict(block_size=16, adaptive=True, cost=cost, tree=True, twin=2)),
    )
    res = {"target": a.target, "prompts": []}
    for k, text in enumerate(texts):
        ids = chat_ids(tok, text)
        target.reset()
        base, _ = baseline_generate(target, ids, max_tokens=a.tokens, stop_ids=stop)
        nz = noise(target, ids, base[:16])
        row = {"noise": nz, "settings": {}}
        print(f"[div] prompt {k}: {len(ids)} prompt tokens, {len(base)} plain tokens, rounding noise {nz}", flush=True)
        for name, opts in settings:
            gen = LaneDSpineGenerator(target, base_gen.draft, **kw, **opts)
            out, _ = gen.generate(ids, max_tokens=a.tokens, stop_ids=stop)
            d = gap_at(target, ids, base, out)
            row["settings"][name] = d or "same as plain"
            print(f"[div] prompt {k} {name}: {d or 'same as plain decoding'}", flush=True)
            mx.clear_cache()
        res["prompts"].append(row)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
