"""Losslessness + acceptance length for DSpine on a hybrid MLX target.

1. lossless: DSpine greedy output vs plain greedy on the same target
   (token-exact match, first divergence).
2. accept:   tokens per verify round on short subsets of the benchmarks the
   model card reports (GSM8K, MATH-500, HumanEval, MBPP, MT-Bench), T=0.

Aggregates only. Decode tok/s is logged too, but use scripts/speed.py on an
otherwise idle machine for speed. Prompts come from Hugging Face datasets.
"""

from __future__ import annotations

import argparse
import json
import time

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402

from dspine_hybrid.run import apply_memory_rules, load
from dspine_mlx.generate import _load_tokenizer_eos, baseline_generate

CARD_T0 = {"gsm8k": 7.20, "math500": 6.49, "humaneval": 5.71, "mbpp": 6.14, "mtbench": 4.48}


def prompts(n: int) -> dict[str, list[str]]:
    from datasets import load_dataset

    specs = {
        "gsm8k": (("openai/gsm8k", "main"), "test", lambda r: r["question"]),
        "math500": (("HuggingFaceH4/MATH-500",), "test", lambda r: r["problem"]),
        "humaneval": (("openai/openai_humaneval",), "test",
                      lambda r: "Complete the following Python function.\n\n" + r["prompt"]),
        "mbpp": (("google-research-datasets/mbpp", "full"), "test",
                 lambda r: r["text"] + "\nYour code should pass these tests:\n" + "\n".join(r["test_list"])),
        "mtbench": (("HuggingFaceH4/mt_bench_prompts",), "train", lambda r: r["prompt"][0]),
    }
    out = {}
    for name, (args, split, fmt) in specs.items():
        try:
            ds = load_dataset(*args, split=split)
            out[name] = [fmt(r) for r in ds.select(range(n))]
        except Exception as e:  # keep going; report the gap
            print(f"[data] skipped {name}: {type(e).__name__}: {e}", flush=True)
    if not out:
        raise SystemExit("no benchmark data could be loaded")
    return out


def encode(tok, text: str, thinking: bool):
    return tok.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--draft-bits", type=int, default=None)
    p.add_argument("--n-accept", type=int, default=10)
    p.add_argument("--accept-tokens", type=int, default=512)
    p.add_argument("--n-lossless", type=int, default=2, help="per benchmark")
    p.add_argument("--lossless-tokens", type=int, default=256)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--memory-limit-gb", type=float, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    apply_memory_rules(a.memory_limit_gb)
    t0 = time.time()
    gen, target, cfg, window = load(a.target, a.draft, draft_bits=a.draft_bits)
    tok = target.tokenizer
    stop = _load_tokenizer_eos(tok)
    print(f"[load] {time.time() - t0:.1f}s window={window} block={gen.block_size}", flush=True)

    sets = prompts(max(a.n_accept, a.n_lossless))
    res = {"target": a.target, "draft": a.draft, "draft_bits": a.draft_bits,
           "thinking": a.thinking, "window": window, "lossless": {}, "accept": {}}

    # 1. losslessness
    match = total = 0
    first_div = []
    for name, texts in sets.items():
        for text in texts[: a.n_lossless]:
            ids = encode(tok, text, a.thinking)
            mx.random.seed(0)
            s, _ = gen.generate(ids, max_tokens=a.lossless_tokens, stop_ids=stop)
            mx.clear_cache()
            b, _ = baseline_generate(target, ids, max_tokens=a.lossless_tokens, stop_ids=stop)
            mx.clear_cache()
            n = min(len(s), len(b))
            div = next((i for i in range(n) if s[i] != b[i]), None)
            if div is None and len(s) != len(b):
                div = n
            total += 1
            match += div is None
            margin = None
            if div is not None and div < n:
                target.reset()
                lg, _ = target.prefill(mx.array([list(ids) + b[:div]]), capture=False)
                row = lg[0, -1].astype(mx.float32)
                margin = round(abs(float(row[s[div]].item()) - float(row[b[div]].item())), 4)
                mx.clear_cache()
            if div is not None:
                first_div.append({"bench": name, "token": div, "margin": margin})
            print(f"[lossless] {name} exact={div is None} div={div} margin={margin} len={len(s)}/{len(b)}", flush=True)
    res["lossless"] = {"prompts": total, "exact": match, "first_divergence": first_div}

    # 2. acceptance
    for name, texts in sets.items():
        rounds = committed = gen_tokens = 0
        secs = 0.0
        for text in texts[: a.n_accept]:
            ids = encode(tok, text, a.thinking)
            _, st = gen.generate(ids, max_tokens=a.accept_tokens, stop_ids=stop)
            mx.clear_cache()
            rounds += st.rounds
            committed += st.committed_tokens
            gen_tokens += st.generated_tokens
            secs += st.decode_seconds
        tpr = committed / max(rounds, 1)
        res["accept"][name] = {
            "prompts": min(len(texts), a.n_accept),
            "tokens_per_round": round(tpr, 3),
            "card_t0": CARD_T0[name],
            "decode_tok_s_not_quiet": round(gen_tokens / max(secs, 1e-9), 2),
        }
        print(f"[accept] {name} tokens/round={tpr:.2f} (card {CARD_T0[name]})", flush=True)

    res["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
