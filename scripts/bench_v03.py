"""v0.3 checks on one target: prompt cache across turns, sampled decoding filter, adaptive confidence.

Run on an otherwise idle machine:
    python scripts/bench_v03.py --target <4-bit build> --draft <drafter> --out v03.json
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
from dspine_hybrid import fast_sampling  # noqa: E402
from dspine_hybrid.adaptive import AdaptiveDSpineGenerator, measure_cost  # noqa: E402
from dspine_hybrid.run import apply_memory_rules, load  # noqa: E402
from dspine_mlx.generate import _load_tokenizer_eos  # noqa: E402


def chat_ids(tok, msgs):
    return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)


def multi_turn(gen, target, tok, stop, turns, max_tokens, cache_on):
    target.enable_prompt_cache(cache_on)
    msgs, rows = [], []
    for user in turns:
        msgs.append({"role": "user", "content": user})
        ids = chat_ids(tok, msgs)
        t0 = time.perf_counter()
        out, st = gen.generate(ids, max_tokens=max_tokens, stop_ids=stop)
        wall = time.perf_counter() - t0
        rows.append(dict(prompt=len(ids), reused=getattr(target, "last_reused", 0), prefill_s=round(st.prefill_seconds, 3),
                         decode_tps=round(st.decode_tps, 1), tokens=st.generated_tokens, wall_s=round(wall, 2)))
        msgs.append({"role": "assistant", "content": tok.decode([t for t in out if t not in set(stop)])})
        mx.clear_cache()
    return rows, [m["content"] for m in msgs if m["role"] == "assistant"]


def decode_set(gen, tok, stop, texts, max_tokens, **kw):
    n = s = 0.0
    outs = []
    for text in texts:
        mx.random.seed(0)
        out, st = gen.generate(chat_ids(tok, [{"role": "user", "content": text}]), max_tokens=max_tokens,
                               stop_ids=stop, **kw)
        n += st.generated_tokens; s += st.decode_seconds
        outs.append(out)
        mx.clear_cache()
    return round(n / s, 1), outs


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
    gen, target, _, _ = load(a.target, a.draft)
    tok = target.tokenizer
    stop = _load_tokenizer_eos(tok)
    raw = prompts(a.n)
    sets = {"code": raw["humaneval"], "prose": raw["mtbench"]}
    res = {"target": a.target, "tokens": a.tokens}
    gen.generate(chat_ids(tok, [{"role": "user", "content": "Say hello."}]), max_tokens=16, stop_ids=stop)

    # 1. prompt cache: same 3-turn chat with the cache off and on
    turns = [sets["code"][0], "Now add type hints and a short docstring.", "Write two pytest tests for it."]
    off, ans_off = multi_turn(gen, target, tok, stop, turns, a.tokens, False)
    on, ans_on = multi_turn(gen, target, tok, stop, turns, a.tokens, True)
    res["prompt_cache"] = {"off": off, "on": on, "same_answers": ans_off == ans_on}
    print("[v03] prompt cache off", off, "\n[v03] on", on, flush=True)
    target.enable_prompt_cache(False)

    # 2. sampled decoding (Qwen settings), upstream filter vs small-support filter
    import dspine_mlx.generate as g
    import dspine_mlx.sampler as s

    texts = [t for name in ("code", "prose") for t in sets[name][: a.n]]
    kw = dict(temperature=0.7, top_k=20, top_p=0.8)
    s.filter_target_probs = g.filter_target_probs = fast_sampling._upstream
    up_tps, up_out = decode_set(gen, tok, stop, texts, a.tokens, **kw)
    s.filter_target_probs = g.filter_target_probs = fast_sampling.filter_target_probs
    fa_tps, fa_out = decode_set(gen, tok, stop, texts, a.tokens, **kw)
    cost = measure_cost(target, list(range(1000, 1256)))
    cal_gen = AdaptiveDSpineGenerator(target, gen.draft, cost=cost, block_size=16, fixed_block_size=gen.block_size,
                                      confidence="calibrated")
    ad_tps, _ = decode_set(cal_gen, tok, stop, texts, a.tokens, **kw)
    res["sampled"] = {"upstream_filter_tps": up_tps, "fast_filter_tps": fa_tps, "same_tokens": up_out == fa_out,
                      "adaptive_calibrated_tps": ad_tps, "adaptive_len_hist": cal_gen.len_hist}
    print("[v03] sampled", res["sampled"], flush=True)

    # 3. greedy: fixed block vs adaptive (vocab) vs adaptive (calibrated), per bench
    gens = {"fixed": gen}
    for conf in ("vocab", "calibrated"):
        gens[conf] = AdaptiveDSpineGenerator(target, gen.draft, cost=cost, block_size=16,
                                             fixed_block_size=gen.block_size, confidence=conf)
    res["greedy"] = {}
    for bench in ("code", "prose"):
        row, outs = {}, {}
        for name, gg in gens.items():
            row[name], outs[name] = decode_set(gg, tok, stop, sets[bench][: a.n], a.tokens)
        row["calibrated_same_as_fixed"] = outs["calibrated"] == outs["fixed"]
        res["greedy"][bench] = row
        print("[v03] greedy", bench, row, flush=True)
    res["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
