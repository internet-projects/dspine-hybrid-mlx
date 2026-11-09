"""Hybrid-target support: exact rollback for GatedDeltaNet layers.

mlx-lm keeps a GatedDeltaNet layer's state in an ``ArraysCache``
(``[conv_state, ssm_state]``) that cannot be trimmed like a KV cache. After a
verify pass over T tokens of which only ``n`` are accepted, the state must be
the state after exactly ``n`` tokens.

During a verify pass each linear layer records its pre-verify state and the
recurrence inputs (conv window, q, k, v, a, b). ``trim`` then rebuilds the
state for the accepted prefix:

* conv state = rows ``n .. n+K-2`` of the recorded conv input (a slice);
* ssm state  = ``gated_delta_update`` replayed over the first ``n`` recorded
  steps from the pre-verify state. The recurrence runs step by step in the
  same kernel, so the result equals the state the verify pass held after
  step ``n`` (and the state plain decoding would reach).

No projections or attention are recomputed; full-attention layers keep using
the normal ``KVCache.trim``.
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk

from dspine_mlx.target import TargetAdapter

_BASE_GDN = qwen3_5.GatedDeltaNet


class _Record:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _RecordBox:
    def __init__(self):
        self.recording = False
        self.record: Optional[_Record] = None


class RecordingGatedDeltaNet(_BASE_GDN):
    """mlx-lm 0.32.2 ``qwen3_5.GatedDeltaNet.__call__`` plus recording.

    The compute path is a line-for-line copy of the upstream method (tests
    check it is bit-identical); the only additions are the ``_record`` writes.
    """

    # State lives in a plain holder: mlx Modules treat dict/array attributes
    # as children, which must not happen for recorded activations.
    _box: "_RecordBox"

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = inputs.shape

        if self.sharding_group is not None:
            raise NotImplementedError("sharded targets are not supported")

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        q, k = normalize_qk(q, k, inv_scale=self.head_k_dim**-0.5, eps=1e-6)

        if self._box.recording and cache is not None:
            if mask is not None or cache.lengths is not None:
                raise NotImplementedError("rollback supports unpadded batch-1 decode")
            self._box.record = _Record(
                conv_input=conv_input, ssm0=state, q=q, k=k, v=v, a=a, b=b, steps=S
            )

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state
            cache.advance(S)

        out = self.norm(out, z)
        out = self.out_proj(out.reshape(B, S, -1))
        return out

    def rollback(self, cache: ArraysCache, accepted: int) -> None:
        """Set ``cache`` to the state after ``accepted`` of the recorded steps."""
        rec = self._box.record
        if rec is None:
            raise RuntimeError("no recorded verify pass to roll back")
        steps = int(rec.steps)
        if not 1 <= accepted <= steps:
            raise ValueError(f"accepted={accepted} outside [1,{steps}]")
        if accepted < steps:
            n_keep = self.conv_kernel_size - 1
            cache[0] = mx.contiguous(
                rec.conv_input[:, accepted : accepted + n_keep, :]
            )
            sl = slice(0, accepted)
            _, state = gated_delta_update(
                rec.q[:, sl],
                rec.k[:, sl],
                rec.v[:, sl],
                rec.a[:, sl],
                rec.b[:, sl],
                self.A_log,
                self.dt_bias,
                rec.ssm0,
                None,
                use_kernel=not self.training,
            )
            cache[1] = state
        self._box.record = None


def _inner_layers(model) -> list:
    if hasattr(model, "language_model"):
        return model.language_model.model.layers
    return model.model.layers


def install_recording(model) -> list:
    """Swap every qwen3_5 GatedDeltaNet for the recording subclass.

    Returns the per-cache-slot list of linear modules (None for KV layers).
    """
    import warnings

    import mlx_lm

    from . import MLX_LM_VERSION

    if getattr(mlx_lm, "__version__", None) != MLX_LM_VERSION:
        warnings.warn(
            f"dspine_hybrid copies GatedDeltaNet from mlx-lm {MLX_LM_VERSION}; found "
            f"{getattr(mlx_lm, '__version__', '?')}. Run the tests before trusting the output.",
            stacklevel=2,
        )
    linear = []
    found = 0
    for layer in _inner_layers(model):
        layer = getattr(layer, "_layer", layer)  # unwrap a DSpine hook
        mod = getattr(layer, "linear_attn", None)
        if mod is None:
            linear.append(None)
            continue
        if type(mod) not in (_BASE_GDN, RecordingGatedDeltaNet):
            raise TypeError(
                f"unsupported linear-attention module {type(mod).__name__}; "
                "only mlx_lm.models.qwen3_5.GatedDeltaNet is validated"
            )
        mod.__class__ = RecordingGatedDeltaNet
        if not isinstance(getattr(mod, "_box", None), _RecordBox):
            object.__setattr__(mod, "_box", _RecordBox())
        linear.append(mod)
        found += 1
    if not found:
        raise TypeError("target has no GatedDeltaNet layers; use dspine_mlx directly")
    return linear


class HybridTargetAdapter(TargetAdapter):
    """``TargetAdapter`` for qwen3_5 hybrid targets (e.g. Qwen3.8-27B)."""

    def __init__(self, model, tokenizer, target_layer_ids):
        super().__init__(model, tokenizer, target_layer_ids)
        self._linear = install_recording(model)
        self._verify_steps = 0

    @classmethod
    def load(cls, path, target_layer_ids) -> "HybridTargetAdapter":
        import mlx_lm

        model, tokenizer = mlx_lm.load(str(path))
        return cls(model, tokenizer, target_layer_ids)

    def _set_recording(self, on: bool) -> None:
        for mod in self._linear:
            if mod is not None:
                mod._box.recording = on

    def verify(self, tokens: mx.array, capture: bool = True):
        self._set_recording(True)
        try:
            out = self._forward(tokens, capture=capture)
        finally:
            self._set_recording(False)
        self._verify_steps = int(tokens.shape[1])
        return out

    def trim(self, n: int) -> None:
        if n <= 0:
            for mod in self._linear:
                if mod is not None:
                    mod._box.record = None
            return
        steps = self._verify_steps
        if steps <= 0 or n >= steps:
            raise RuntimeError(
                f"trim({n}) needs a preceding verify of more than {n} tokens"
            )
        accepted = steps - n
        if len(self.cache) != len(self._linear):
            raise RuntimeError("cache / layer count mismatch")
        for c, mod in zip(self.cache, self._linear):
            if mod is None:
                if not c.is_trimmable():
                    raise RuntimeError(f"cache {type(c).__name__} is not trimmable")
                c.trim(n)
            else:
                mod.rollback(c, accepted)
        self._verify_steps = 0
