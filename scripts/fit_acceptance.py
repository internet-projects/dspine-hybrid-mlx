"""Fit a drafter's confidence to what the target really accepts, on this Mac and this build.

    python scripts/fit_acceptance.py --target mlx-community/Qwen3.8-27B-4bit \\
        --draft netprojects/Qwen3.8-27B-DSpine-b16-MLX --out dspine-fit.json

Greedy, thinking off, one conversation at a time, the drafter's whole block checked every round.
Every drafted token the target reached gives one sample (depth, the drafter's own confidence,
accepted or not); the fit is p = sigmoid(scale * logit(confidence) + offset[depth]) (the form
Tencent published for DSpine). The prompts here are not the ones the benches time. The file is
read by ``calibration.load`` and by ``bench_batch_gen.py --calibration``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402

from dspine_hybrid import calibration as cal  # noqa: E402
from dspine_hybrid.batch_gen import BatchGenerator, Conversation, DSpineProposer  # noqa: E402
from dspine_hybrid.run import apply_memory_rules, load  # noqa: E402
from dspine_mlx.generate import _load_tokenizer_eos  # noqa: E402

CODE = [
    "Write a Python function that checks whether a string of brackets is balanced, with tests.",
    "Write a Python class for a min-heap with push, pop and peek, and four unit tests.",
    "Write a Python function that converts a Roman numeral to an integer and back, with tests.",
    "Write a Python script that counts word frequencies in a text file and prints the top ten.",
    "Write a Python function for binary search on a sorted list, with a docstring and edge case tests.",
    "Write a Python generator that yields prime numbers, and a function that uses it to factor an integer.",
]
CHAT = [
    "Explain how a refrigerator works to someone who has never studied physics.",
    "Write a friendly email inviting a colleague to a team lunch next Friday.",
    "What should I consider before adopting a dog? Give five points with a sentence each.",
    "Describe a walk through an old harbour town at dawn, in three paragraphs.",
    "Compare trains and planes for a 500 km trip. Which would you choose and why?",
    "Summarise the plot of a made-up detective novel set on a small island.",
]


def samples_of(log: list) -> list:
    out = []
    for raw, checked, accepted in log:
        for d in range(min(checked, accepted + 1)):
            out.append((d, raw[d], 1 if d < accepted else 0))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--block", type=int, default=None)
    p.add_argument("--no-kernels", action="store_true")
    p.add_argument("--memory-limit-gb", type=float, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    apply_memory_rules(a.memory_limit_gb)
    fast = not a.no_kernels
    gen, target, _, _ = load(a.target, a.draft, fast_verify=fast, a8_prefill=fast, prompt_cache=False,
                             block_size=a.block or 16)
    proposer = DSpineProposer(gen)
    tok = target.tokenizer
    stop = tuple(_load_tokenizer_eos(tok))
    depths = proposer.max_block - 1
    art = {"format": cal.FIT_FORMAT, "drafter": "dspine", "draft": a.draft, "target": a.target, "block": proposer.max_block}
    logs = {}
    for name, texts in (("code", CODE), ("chat", CHAT)):
        bg = BatchGenerator(target, proposer, max_batch=1)
        bg.accept_log = logs[name] = []
        ids = [tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                       enable_thinking=False) for t in texts]
        bg.run([Conversation(x, max_tokens=a.tokens, stop_ids=stop) for x in ids])
        mx.clear_cache()
        s = samples_of(logs[name])
        acc = sum(k for _, _, k in logs[name]) / max(len(logs[name]), 1)
        print(f"[fit] {name}: {len(logs[name])} rounds, {len(s)} samples, {acc:.2f} drafts accepted per round", flush=True)
    for mode, names in (("greedy", ("code", "chat")), ("code", ("code",)), ("chat", ("chat",))):
        s = [x for n in names for x in samples_of(logs[n])]
        m = cal.fit(s, depths)
        art[mode] = m
        print(f"[fit] {mode}: scale {m['scale']:.3f}, offsets {' '.join(f'{v:.2f}' for v in m['offset'])}; "
              f"log loss {m['base_loss']:.3f} -> {m['loss']:.3f}; samples per depth {m['samples']}", flush=True)
    with open(a.out, "w") as f:
        json.dump(art, f, indent=1)


if __name__ == "__main__":
    main()
