"""Command line: DSpine speculative decoding on a hybrid Qwen3.8-27B MLX target.

    dspine-hybrid-generate --target mlx-community/Qwen3.8-27B-4bit \
        --draft tencent/Qwen3.8-27B-DSpine-b16 --prompt "Explain quicksort."
"""

from __future__ import annotations

import argparse

import mlx.core as mx

from dspine_mlx.generate import DSpineGenerator, _load_tokenizer_eos, baseline_generate

from .run import apply_memory_rules, load


def main() -> None:
    p = argparse.ArgumentParser("dspine-hybrid-generate")
    p.add_argument("--target", required=True, help="MLX Qwen3.8-27B build (local path or HF repo id)")
    p.add_argument("--draft", default="tencent/Qwen3.8-27B-DSpine-b16", help="DSpine draft (path or HF repo id)")
    p.add_argument("--prompt", default="What is 1+1? Answer briefly.")
    p.add_argument("--thinking", action="store_true", help="enable the chat template's thinking mode")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block-size", type=int, default=None, help="verify block (default: 5 for quantized targets, 16 for bf16)")
    p.add_argument("--draft-bits", type=int, default=None, help="quantize the draft backbone (8 or 4)")
    p.add_argument("--memory-limit-gb", type=float, default=None)
    p.add_argument("--profile", action="store_true", help="per-phase timing")
    p.add_argument("--baseline", action="store_true", help="also run plain greedy and print the speedup")
    a = p.parse_args()

    apply_memory_rules(a.memory_limit_gb)
    mx.random.seed(a.seed)
    gen, target, _, _ = load(a.target, a.draft, draft_bits=a.draft_bits, block_size=a.block_size)
    if a.profile:
        gen = DSpineGenerator(target, gen.draft, block_size=gen.block_size, profile=True)
    tok = target.tokenizer
    ids = tok.apply_chat_template([{"role": "user", "content": a.prompt}],
                                  add_generation_prompt=True, enable_thinking=a.thinking)
    stop = _load_tokenizer_eos(tok)
    kw = dict(max_tokens=a.max_tokens, temperature=a.temperature, top_k=a.top_k, top_p=a.top_p,
              min_p=a.min_p, stop_ids=stop)
    out, st = gen.generate(ids, **kw)
    mx.clear_cache()
    print(tok.decode(out))
    print(f"[dspine] tokens={st.generated_tokens} rounds={st.rounds} tokens/round={st.tokens_per_round:.2f} "
          f"decode={st.decode_tps:.1f} tok/s prefill={st.prefill_seconds:.2f}s peak={st.peak_memory_gb:.1f}GB")
    if a.profile:
        print(f"[profile] draft={st.draft_seconds:.2f}s tail={st.tail_seconds:.2f}s verify={st.verify_seconds:.2f}s")
    if a.baseline:
        _, bs = baseline_generate(target, ids, **kw)
        print(f"[baseline] decode={bs.decode_tps:.1f} tok/s speedup={st.decode_tps / max(bs.decode_tps, 1e-9):.2f}x")


if __name__ == "__main__":
    main()
