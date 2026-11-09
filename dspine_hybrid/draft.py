"""Draft loading with the native sliding window of Qwen3.8-27B-DSpine-b16.

``dspine_mlx`` rejects drafts that declare ``sliding_attention``. The 27B
draft declares all five layers ``sliding_attention`` with an 8192 window
(SGLang serves it with FlashAttention ``window_left = 8191``): a query at
absolute position p sees keys at positions p-8191 .. p. Below 8192 total
positions this is identical to full causal attention, and the original code
path (``mask="causal"``) is used unchanged.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from dspine_mlx.config import DraftConfig, resolve_draft_path
from dspine_mlx.model import DraftAttention, DSpineDraftModel, _load_weights

SUPPORTED_WINDOW = 8192


def parse_window(raw: dict) -> int | None:
    """Return the sliding window, None for full attention; reject anything else."""
    types = raw.get("layer_types")
    use = bool(raw.get("use_sliding_window", False))
    declares = use or (types is not None and "sliding_attention" in types)
    if not declares:
        return None
    n = int(raw["num_hidden_layers"])
    window = raw.get("sliding_window")
    if (
        types != ["sliding_attention"] * n
        or not use
        or type(window) is not int
        or window != SUPPORTED_WINDOW
        or int(raw.get("max_window_layers", n)) != n
        or raw.get("is_causal") is not True
    ):
        raise ValueError(
            "only the all-layer sliding window of 8192 used by "
            "Qwen3.8-27B-DSpine-b16 is supported"
        )
    return window


def config_without_window(raw: dict) -> dict:
    out = copy.deepcopy(raw)
    n = int(out["num_hidden_layers"])
    out["layer_types"] = ["full_attention"] * n
    out["use_sliding_window"] = False
    out["sliding_window"] = None
    return out


class WindowedDraftAttention(DraftAttention):
    """``DraftAttention`` with a sliding window over absolute positions."""

    window: int | None = None

    def __call__(self, x, x_ctx, rope, cache=None):
        B, L, _ = x.shape
        S = 0 if x_ctx is None else x_ctx.shape[1]

        queries = self.q_proj(x)
        prop_keys = self.k_proj(x)
        prop_values = self.v_proj(x)

        queries = self.q_norm(
            queries.reshape(B, L, self.n_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        prop_keys = self.k_norm(
            prop_keys.reshape(B, L, self.n_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        prop_values = prop_values.reshape(
            B, L, self.n_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        if cache is not None:
            offset = cache.offset
            queries = rope(queries, offset=offset + S)
            prop_keys = rope(prop_keys, offset=offset + S)
            if S > 0:
                ctx_keys, ctx_values = self._context_kv(x_ctx, rope, offset)
                keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
            elif cache.keys is None:
                keys = prop_keys
                values = prop_values
            else:
                keys = cache.keys[:, :, :offset, :]
                values = cache.values[:, :, :offset, :]
            keys = mx.concatenate([keys, prop_keys], axis=2)
            values = mx.concatenate([values, prop_values], axis=2)
        else:
            if x_ctx is None:
                raise ValueError(
                    "DraftAttention needs target context or a prefilled cache"
                )
            ctx_keys, ctx_values = self._context_kv(x_ctx, rope, 0)
            queries = rope(queries)
            prop_keys = rope(prop_keys)
            keys = mx.concatenate([ctx_keys, prop_keys], axis=2)
            values = mx.concatenate([ctx_values, prop_values], axis=2)

        K = keys.shape[2]
        W = self.window
        if W is None or K <= W:
            mask = "causal"
        else:
            # Absolute positions: key j sits at j, query i at K - L + i.
            q0 = K - L
            start = max(0, q0 - (W - 1))
            keys = keys[:, :, start:, :]
            values = values[:, :, start:, :]
            kpos = mx.arange(start, K)[None, :]
            qpos = mx.arange(q0, K)[:, None]
            mask = (kpos <= qpos) & (kpos > qpos - W)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


def set_window(model: DSpineDraftModel, window: int | None) -> None:
    for layer in model.layers:
        layer.self_attn.__class__ = WindowedDraftAttention
        layer.self_attn.window = window


def build_draft(raw: dict) -> tuple[DSpineDraftModel, DraftConfig, int | None]:
    window = parse_window(raw)
    cfg = DraftConfig.from_dict(config_without_window(raw) if window else raw)
    model = DSpineDraftModel(cfg)
    set_window(model, window)
    return model, cfg, window


def load_draft(path) -> tuple[DSpineDraftModel, DraftConfig, int | None]:
    """Strict-name/shape loader (same checks as ``dspine_mlx.model.load_draft``)."""
    root = Path(resolve_draft_path(path))
    with open(root / "config.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    model, cfg, window = build_draft(raw)
    weights = _load_weights(root)

    expected = dict(tree_flatten(model.parameters()))
    cleaned: dict[str, mx.array] = {}
    for name, tensor in weights.items():
        short = name.removeprefix("model.")
        if short in ("draft_block_causal_version", "last_write_refine_version"):
            if int(tensor.item()) != 1:
                raise ValueError(f"Unsupported {short}")
            continue
        cleaned[short] = tensor
    missing = sorted(set(expected) - set(cleaned))
    extra = sorted(set(cleaned) - set(expected))
    if missing or extra:
        raise ValueError(f"DSpine draft weight mismatch: missing={missing} extra={extra}")
    for name, dest in expected.items():
        src = cleaned[name]
        if tuple(src.shape) != tuple(dest.shape):
            raise ValueError(
                f"DSpine draft shape mismatch for {name}: {src.shape} vs {dest.shape}"
            )
        if src.dtype != dest.dtype and dest.dtype in (mx.int32, mx.int64):
            cleaned[name] = src.astype(dest.dtype)
    model.load_weights(list(cleaned.items()), strict=True)
    model.selected_prefix_bridge.prepare()
    return model, cfg, window
