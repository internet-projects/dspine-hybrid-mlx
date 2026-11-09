"""DSpine speed vs verify block at long context (8k / 32k tokens of prompt).

Builds a long prompt from real text (GSM8K / MBPP / MATH-500 problems, joined
as a reference document) followed by a task that makes the model write, then
runs DSpine greedy for each config and context length. Configs:
name:block=N[,draft_bits=B], name:alpha=A[,draft_bits=B] (adaptive verify
length) or name:plain=1. Thinking off.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402

from bench_accept import prompts  # noqa: E402
from dspine_hybrid.adaptive import AdaptiveDSpineGenerator, measure_cost  # noqa: E402
from dspine_hybrid.draft import load_draft  # noqa: E402
from dspine_hybrid.hybrid_target import HybridTargetAdapter  # noqa: E402
from dspine_hybrid.run import apply_memory_rules  # noqa: E402
from dspine_mlx.generate import DSpineGenerator, _load_tokenizer_eos, baseline_generate  # noqa: E402
from dspine_mlx.model import quantize_draft  # noqa: E402

TASK = ("\n\nAbove is a set of reference problems. Write a Python module that solves the first five of them, "
        "one function per problem with a short docstring, then explain each solution in two sentences.")


def parse(spec):
    name, _, rest = spec.partition(":")
    return name, dict(x.split("=") for x in rest.split(",") if x)


def build_ids(tok, pool, n_tokens):
    doc, i = [], 0
    while True:
        doc.append(f"Problem {i + 1}. {pool[i % len(pool)]}")
        i += 1
        if i % 20 == 0:
            text = "\n\n".join(doc)
            ids = tok.apply_chat_template([{"role": "user", "content": text + TASK}], add_generation_prompt=True,
                                          enable_thinking=False)
            if len(ids) >= n_tokens:
                return ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--context", default="8192,32768")
    p.add_argument("--config", action="append", required=True)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--memory-limit-gb", type=float, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    apply_memory_rules(a.memory_limit_gb)

    d16, cfg, _ = load_draft(a.draft)
    t = HybridTargetAdapter.load(a.target, cfg.target_layer_ids)
    d16.bind(t.model)
    drafts = {0: d16}

    def draft_for(bits):
        if bits not in drafts:
            dq, _, _ = load_draft(a.draft)
            dq = quantize_draft(dq, bits=bits, group_size=64, mode="affine")
            dq.bind(t.model)
            drafts[bits] = dq
        return drafts[bits]

    tok = t.tokenizer
    stop = _load_tokenizer_eos(tok)
    pool = [x for xs in prompts(20).values() for x in xs]
    res = {"target": a.name, "tokens": a.tokens, "runs": {}}
    for ctx in [int(c) for c in a.context.split(",")]:
        ids = build_ids(tok, pool, ctx)
        res["runs"][ctx] = {"prompt_tokens": len(ids)}
        costs = {}
        for spec in a.config:
            name, kv = parse(spec)
            if kv.get("plain") == "1":
                _, bs = baseline_generate(t, ids, max_tokens=a.tokens, stop_ids=stop)
                row = {"decode_tok_s": round(bs.decode_tps, 2), "prefill_s": round(bs.prefill_seconds, 1)}
            else:
                d = draft_for(int(kv.get("draft_bits", "0")))
                if "alpha" in kv:
                    if "cost" not in costs:
                        costs["cost"] = measure_cost(t, ids)  # cost at this context length
                    gen = AdaptiveDSpineGenerator(t, d, cost=costs["cost"], alpha=float(kv["alpha"]), block_size=16)
                else:
                    gen = DSpineGenerator(t, d, block_size=int(kv.get("block", "16")))
                _, st = gen.generate(ids, max_tokens=a.tokens, stop_ids=stop)
                row = {"decode_tok_s": round(st.decode_tps, 2), "tokens_per_round": round(st.tokens_per_round, 2),
                       "prefill_s": round(st.prefill_seconds, 1), "generated": st.generated_tokens}
                if hasattr(gen, "len_hist"):
                    row["verify_len_hist"] = dict(sorted(gen.len_hist.items()))
            mx.clear_cache()
            res["runs"][ctx][name] = row
            print(f"[long] ctx={len(ids)} {name} {json.dumps(row)}", flush=True)
    res["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
