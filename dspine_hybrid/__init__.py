"""DSpine speculative decoding for hybrid (GatedDeltaNet) MLX targets.

Builds on Tencent's ``dspine_mlx`` (Apache-2.0) and adds:
  * exact rollback of linear-attention state after a partly rejected draft
    (record-and-replay of the verified recurrence inputs), so Qwen3.8-27B
    (qwen3_5: 48 GatedDeltaNet + 16 full-attention layers) can be the target;
  * the draft's native sliding window (Qwen3.8-27B-DSpine-b16: 8192);
  * a chunked prefill that keeps only last-position logits (long prompts);
  * an optional adaptive verify length (``dspine_hybrid.adaptive``), with
    Tencent's calibrated acceptance map as a confidence option;
  * a prompt cache across turns (target cache + drafter context);
  * small-support top-k / top-p filtering for sampled decoding.
"""

__version__ = "0.4.0"

# Upstream code this kit is tested against.
DSPINE_COMMIT = "d211869ac168b756e197f7607ff37cd342b50fb2"
MLX_LM_VERSION = "0.32.2"
