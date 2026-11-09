"""Answer quality: GSM8K accuracy of DSpine greedy output, per target.

DSpine is lossless (output = the target's own greedy output up to rounding),
so this measures the target build's quality as served through DSpine, and
logs acceptance alongside so quant level vs acceptance can be compared.
Thinking off, short answers, exact match on the final number.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402

from dspine_hybrid.run import apply_memory_rules, load  # noqa: E402
from dspine_mlx.generate import _load_tokenizer_eos  # noqa: E402

INSTR = ("Solve the problem. Show brief working, then give the final answer on the last line "
         "as '#### <number>'.\n\n")


def number(s: str) -> str | None:
    m = re.findall(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", s)
    if not m:
        m = re.findall(r"(-?[\d,]*\.?\d+)", s)
    if not m:
        return None
    x = m[-1].replace(",", "").rstrip(".")
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--memory-limit-gb", type=float, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    apply_memory_rules(a.memory_limit_gb)

    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(a.n))
    gen, target, _, _ = load(a.target, a.draft)
    tok = target.tokenizer
    stop = _load_tokenizer_eos(tok)
    correct = rounds = committed = 0
    for i, r in enumerate(ds):
        ids = tok.apply_chat_template([{"role": "user", "content": INSTR + r["question"]}],
                                      add_generation_prompt=True, enable_thinking=False)
        out, st = gen.generate(ids, max_tokens=a.tokens, stop_ids=stop)
        mx.clear_cache()
        text = tok.decode(out)
        gold = number(r["answer"].split("####")[-1])
        pred = number(text)
        correct += pred is not None and pred == gold
        rounds += st.rounds
        committed += st.committed_tokens
        if i % 10 == 9:
            print(f"[quality] {a.name} {i + 1}/{a.n} acc={correct / (i + 1):.3f}", flush=True)
    res = {"target": a.name, "path": a.target, "gsm8k_n": a.n, "accuracy": round(correct / a.n, 4),
           "tokens_per_round": round(committed / max(rounds, 1), 2), "thinking": False}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
