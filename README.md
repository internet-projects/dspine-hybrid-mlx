# dspine-hybrid-mlx

**DSpine speculative decoding for Qwen3.8-27B on Apple Silicon (MLX).**

[DSpine](https://github.com/Tencent/DSpine) is Tencent Hunyuan's parallel drafter: a small 5-layer draft model proposes a block of up to 16 tokens, and the target model checks the whole block in one pass. The output is the target's own output, only faster.

Tencent's MLX package (`dspine_mlx`) supports pure-attention targets such as Qwen3-8B. Qwen3.8-27B is a hybrid model, with 48 GatedDeltaNet (linear-attention) layers and 16 full-attention layers. Its recurrent state can't simply be cut back when part of a draft is rejected. This kit adds the two missing pieces, so [`tencent/Qwen3.8-27B-DSpine-b16`](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16) runs on any MLX build of Qwen3.8-27B:

- **Exact rollback for GatedDeltaNet layers.** During verification the kit records each linear layer's inputs and starting state. When only part of the block is accepted, it replays the recurrence over the accepted tokens, so the state matches a plain decode of that prefix.
- **The draft's 8192-token sliding window**, which the checkpoint uses and `dspine_mlx` 0.1.0 rejects.

## Results (MacBook Pro M5 Max, 128 GB)

Decode speed. Greedy, thinking on, 512 new tokens, 10 prompts (2 each from GSM8K, MATH-500, HumanEval, MBPP and MT-Bench), bf16 draft, default settings:

| Target build | Plain mlx-lm | With DSpine | Speedup | Verify block | Peak memory |
|---|---|---|---|---|---|
| 4-bit (`mlx-community/Qwen3.8-27B-4bit`) | 31.9 tok/s | **59.5 tok/s** | **×1.87** | 5 | 26 GB |
| 8-bit (group 64)¹ | 17.9 tok/s | **49.7 tok/s** | **×2.77** | 5 | 39 GB |
| bf16 | 9.1 tok/s | **39.6 tok/s** | **×4.34** | 16 | 60 GB |

¹ Made with `mlx_lm.convert --hf-path Qwen/Qwen3.8-27B -q --q-bits 8 --q-group-size 64`.

**Why the block is 5 on quantized builds.** The drafter is trained to propose 16 tokens. On an Apple GPU, though, a quantized 27B forward pass costs about the same for 1 to 5 tokens and then steps up. On the 4-bit build that's 31 ms for 1 token, 42 ms for 5 and 88 ms for 16. Checking 5 tokens per round accepts fewer tokens than checking 16 (3.4 vs 4.2 per round), but each round is much cheaper, so the net speed is higher. Use `--block-size` to choose a different block.

| 4-bit target, verify block | 3 | 4 | 5 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| DSpine tok/s | 55.2 | 59.5 | 59.5 | 50.3 | 46.1 | 35.0 | 41.9 |
| Tokens per round | 2.50 | 2.99 | 3.35 | 3.55 | 3.81 | 4.07 | 4.18 |

Acceptance with the drafter's full 16-token block, as average tokens committed per verify round, plus answer quality. Greedy, thinking off, first 50 GSM8K test questions:

| Target build | Tokens per round | GSM8K accuracy |
|---|---|---|
| 4-bit | 7.25 | 90% |
| 8-bit | 7.32 | 84% |
| bf16 | – | 86% |
| Tencent's model card (GSM8K, T=0) | 7.20 | |

**Same output as plain decoding.** On a real model, logits from a 16-token verify pass differ slightly from one-token steps through rounding (99th percentile 1.2 on Qwen3.8-27B 4-bit). Every point where DSpine's greedy output departed from plain greedy was a near-tie of at most 0.25 logit, well inside that rounding. On small fp32 test models the outputs are token-identical.

## Install

Needs Python 3.11 or newer on Apple Silicon.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "git+https://github.com/internet-projects/dspine-hybrid-mlx.git"
```

This installs `mlx-lm` 0.32.2 and Tencent's `dspine_mlx` at the tested commit (`0cc92d8`).

## Run

```bash
dspine-hybrid-generate \
  --target mlx-community/Qwen3.8-27B-4bit \
  --draft tencent/Qwen3.8-27B-DSpine-b16 \
  --prompt "Write a Python function that checks whether a number is prime." \
  --max-tokens 512 --baseline
```

`--baseline` also runs plain decoding and prints the speedup. The verify block defaults to 5 on quantized targets and 16 on bf16. `--thinking` turns on the chat template's thinking mode, `--temperature` enables sampling, and `--profile` prints a per-phase time split.

From Python:

```python
from dspine_hybrid.run import load

gen, target, _, _ = load("mlx-community/Qwen3.8-27B-4bit", "tencent/Qwen3.8-27B-DSpine-b16")
tok = target.tokenizer
ids = tok.apply_chat_template([{"role": "user", "content": "Hello!"}],
                              add_generation_prompt=True, enable_thinking=False)
out, stats = gen.generate(ids, max_tokens=256)
print(tok.decode(out), stats.decode_tps, "tok/s")
```

By default `load` caps MLX's buffer cache at 2 GB and sets a 60 GB memory limit. Override these with `DSPINE_CACHE_GB` and `DSPINE_MEMORY_LIMIT_GB`.

## Tests and benchmarks

```bash
pip install -e ".[bench]"
pytest -q                                   # 17 tiny-model tests
python scripts/bench_accept.py --help       # acceptance vs Tencent's card + output check
python scripts/speed.py --help              # plain vs DSpine decode speed
python scripts/quality.py --help            # GSM8K accuracy through DSpine
python scripts/noise_floor.py --help        # block-vs-single rounding on a real model
python scripts/block_sweep.py --help        # forward cost and DSpine speed per verify block
python scripts/profile_ab.py --help         # time split per round (draft / tail / verify / rollback)
```

## Notes

- Tested with mlx 0.32.4 and mlx-lm 0.32.2. `dspine_hybrid/hybrid_target.py` includes a copy of mlx-lm 0.32.2's GatedDeltaNet forward pass, plus recording. Other mlx-lm versions print a warning; run the tests before trusting them.
- Batch size 1 only, which is the same as `dspine_mlx`.
- Where the time goes, on the 4-bit build with block 16: target verify 88%, drafter 8%, the hybrid-state rollback under 1%.
- Speeds come from the same machine. The plain baselines and the bf16 row were measured on an idle Mac. The block-5 rows ran with other local services up, which left the plain single-step time unchanged (31.0 vs 31.3 ms on 4-bit).

## License and credits

Apache-2.0 (see `LICENSE` and `NOTICE`). Built on [Tencent DSpine](https://github.com/Tencent/DSpine) (Apache-2.0) and [mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT). Not affiliated with Tencent, Alibaba/Qwen or Apple.

Made by [NetProjects](https://netprojects.com).
