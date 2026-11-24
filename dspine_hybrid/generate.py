"""Command line: DSpine speculative decoding on a hybrid Qwen3.8-27B MLX target.

    dspine-hybrid-generate --target mlx-community/Qwen3.8-27B-4bit \
        --draft netprojects/Qwen3.8-27B-DSpine-b16-MLX --prompt "Explain quicksort."
"""

from __future__ import annotations

import argparse

import mlx.core as mx

from dspine_mlx.generate import DSpineGenerator, _load_tokenizer_eos, baseline_generate

from .run import DEFAULT_DRAFT_BITS, apply_memory_rules, load


def main() -> None:
    p = argparse.ArgumentParser("dspine-hybrid-generate")
    p.add_argument("--target", required=True, help="MLX Qwen3.8-27B build (local path or HF repo id)")
    p.add_argument("--draft", default="netprojects/Qwen3.8-27B-DSpine-b16-MLX", help="DSpine draft (path or HF repo id)")
    p.add_argument("--prompt", default="What is 1+1? Answer briefly.")
    p.add_argument("--thinking", action="store_true", help="enable the chat template's thinking mode")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block-size", type=int, default=None, help="verify block (default: 5 for quantized targets, 16 for bf16)")
    p.add_argument("--draft-bits", type=int, default=None,
                   help="drafter backbone bits: 4 (default), 8, or 16 for the unquantized bf16 drafter")
    p.add_argument("--adaptive", action="store_true",
                   help="choose how many drafted tokens to verify each round (greedy; sampling too with --confidence calibrated)")
    p.add_argument("--confidence", choices=["vocab", "calibrated"], default="vocab",
                   help="--adaptive: drafter confidence source (calibrated = Tencent's fitted acceptance maps; also adaptive when sampling)")
    p.add_argument("--calibration", default=None, help="--confidence calibrated: your own calibration JSON")
    p.add_argument("--copy", action="store_true",
                   help="copy drafts: when the last --copy-min tokens also occur earlier, propose what followed them")
    p.add_argument("--copy-min", type=int, default=12, help="--copy: matching tokens needed (default 12)")
    p.add_argument("--tree", action="store_true",
                   help="draft tree: verify the drafter's most likely alternatives in the same pass (greedy)")
    p.add_argument("--fast-verify", action="store_true",
                   help="use oMLX's verify-shape Metal kernels for the target check (needs the omlx package)")
    p.add_argument("--a8-prefill", action="store_true",
                   help="read the prompt through oMLX's int8-activation kernels (needs omlx with its native kernels built; M5 and later)")
    p.add_argument("--fast", action="store_true",
                   help="the fastest measured setup: --fast-verify --adaptive --confidence calibrated --tree --a8-prefill")
    p.add_argument("--chat", action="store_true",
                   help="interactive multi-turn chat; each turn prefills only the new text")
    p.add_argument("--no-prompt-cache", action="store_true", help="prefill the whole conversation every turn")
    p.add_argument("--memory-limit-gb", type=float, default=None)
    p.add_argument("--profile", action="store_true", help="per-phase timing")
    p.add_argument("--baseline", action="store_true", help="also run plain greedy and print the speedup")
    a = p.parse_args()

    if a.fast:
        a.fast_verify = a.adaptive = a.tree = a.a8_prefill = True
        a.confidence = "calibrated"
    apply_memory_rules(a.memory_limit_gb)
    mx.random.seed(a.seed)
    bits = DEFAULT_DRAFT_BITS if a.draft_bits is None else (None if a.draft_bits >= 16 else a.draft_bits)
    gen, target, _, _ = load(a.target, a.draft, draft_bits=bits, block_size=a.block_size, adaptive=a.adaptive,
                             prompt_cache=not a.no_prompt_cache, confidence=a.confidence,
                             calibration=a.calibration, copy=a.copy, copy_min=a.copy_min, tree=a.tree,
                             fast_verify=a.fast_verify, a8_prefill=a.a8_prefill)
    if a.a8_prefill:
        print(f"[dspine] int8 prefill: {'on' if target.a8_prefill is not None else 'not available (omlx native kernels not built?)'}")
    if a.profile and not (a.adaptive or a.copy or a.tree):
        gen = DSpineGenerator(target, gen.draft, block_size=gen.block_size, profile=True)
    if a.fast_verify:
        print(f"[dspine] fast verify kernels: {'on' if target.fast_verify else 'not available (is omlx installed?)'}")
    tok = target.tokenizer
    stop = _load_tokenizer_eos(tok)
    kw = dict(max_tokens=a.max_tokens, temperature=a.temperature, top_k=a.top_k, top_p=a.top_p,
              min_p=a.min_p, stop_ids=stop)
    if a.chat:
        return _chat(gen, target, tok, a, kw)
    ids = tok.apply_chat_template([{"role": "user", "content": a.prompt}],
                                  add_generation_prompt=True, enable_thinking=a.thinking)
    out, st = gen.generate(ids, **kw)
    mx.clear_cache()
    print(tok.decode(out))
    print(f"[dspine] tokens={st.generated_tokens} rounds={st.rounds} tokens/round={st.tokens_per_round:.2f} "
          f"decode={st.decode_tps:.1f} tok/s prefill={st.prefill_seconds:.2f}s peak={st.peak_memory_gb:.1f}GB")
    if getattr(gen, "lane_stats", None):
        ls = gen.lane_stats
        print(f"[lanes] drafter rounds={ls['draft_rounds']} copy rounds={ls['copy_rounds']} "
              f"copied={ls['copy_accepted']}/{ls['copy_offered']} tree rounds={ls['tree_rounds']} "
              f"tree nodes={ls['tree_nodes']} tokens via a side branch={ls['tree_side_tokens']}")
    if a.profile:
        print(f"[profile] draft={st.draft_seconds:.2f}s tail={st.tail_seconds:.2f}s verify={st.verify_seconds:.2f}s")
    if a.baseline:
        _, bs = baseline_generate(target, ids, **kw)
        print(f"[baseline] decode={bs.decode_tps:.1f} tok/s speedup={st.decode_tps / max(bs.decode_tps, 1e-9):.2f}x")


def _chat(gen, target, tok, a, kw) -> None:
    msgs = []
    print("[dspine] chat: empty line or Ctrl-D ends")
    while True:
        try:
            user = input(">>> ").strip()
        except EOFError:
            break
        if not user:
            break
        msgs.append({"role": "user", "content": user})
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=a.thinking)
        out, st = gen.generate(ids, **kw)
        stop = set(kw["stop_ids"])
        text = tok.decode([t for t in out if t not in stop])
        print(text)
        reused = getattr(target, "last_reused", 0)
        print(f"[dspine] prompt={len(ids)} reused={reused} prefill={st.prefill_seconds:.2f}s "
              f"tokens={st.generated_tokens} decode={st.decode_tps:.1f} tok/s")
        msgs.append({"role": "assistant", "content": text})
        mx.clear_cache()


if __name__ == "__main__":
    main()
