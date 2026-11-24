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

import os
import warnings
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import ArraysCache, make_prompt_cache
from mlx_lm.models.gated_delta import compute_g, gated_delta_ops, gated_delta_update, normalize_qk
from mlx_lm.models.qwen3_5 import Attention as _BASE_ATTN

from dspine_mlx.target import TargetAdapter

_BASE_GDN = qwen3_5.GatedDeltaNet

try:  # dspine_mlx after Tencent/DSpine#3 installs its own recording class; we take the layer over.
    from dspine_mlx.hybrid import RecordingGatedDeltaNet as _UPSTREAM_GDN
except ImportError:
    _UPSTREAM_GDN = None

# Prompt tokens per prefill forward (mlx-lm's generate uses 2048 too).
PREFILL_CHUNK = 2048


class _Record:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _RecordBox:
    def __init__(self):
        self.recording = False
        self.record: Optional[_Record] = None
        self.flush = 0                                # layers between graph submissions (verify only)
        self.tree: Optional["TreeSpec"] = None        # set for the duration of a tree verify
        self.tree_record: Optional[_Record] = None
        self.a8 = None                                # int8 prefill settings (a8_prefill.Config) or None
        self.gdn = None                               # pipelined prefill recurrence (gdn_prefill.Config) or None


TREE_BY_NODE = os.environ.get("DSPINE_TREE_PATHS", "0") != "1"
_tree_kernel = None
_tree_kernel_ok: Optional[bool] = None      # None: not tried yet; False: failed its first-use check


def _make_tree_kernel():
    """mlx-lm's gated-delta step (scalar gate, explicit reduction tree) walking a tree: node n
    starts from its parent's state, which every thread keeps for all M nodes (Dk / 32 floats each)."""
    source = """
        auto hv_idx = thread_position_in_grid.z;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in: [1, Hv, Dv, Dk]
        auto i_state = state_in + (hv_idx * Dv + dv_idx) * Dk;
        float kept[M * n_per_t];
        float state[n_per_t];

        for (int n = 0; n < M; ++n) {
          int p = par[n];
          for (int i = 0; i < n_per_t; ++i) {
            state[i] = p < 0 ? static_cast<float>(i_state[n_per_t * dk_idx + i]) : kept[p * n_per_t + i];
          }
          // q, k: [1, M, Hk, Dk]; v, y: [1, M, Hv, Dv]; g, beta: [1, M, Hv]
          auto q_ = q + (n * Hk + hk_idx) * Dk;
          auto k_ = k + (n * Hk + hk_idx) * Dk;
          auto v_ = v + (n * Hv + hv_idx) * Dv;
          auto gate = g[n * Hv + hv_idx];

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * gate;
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem += simd_shuffle_xor(kv_mem, 1);
          kv_mem += simd_shuffle_xor(kv_mem, 2);
          kv_mem += simd_shuffle_xor(kv_mem, 4);
          kv_mem += simd_shuffle_xor(kv_mem, 8);
          kv_mem += simd_shuffle_xor(kv_mem, 16);

          auto delta = (v_[dv_idx] - kv_mem) * beta[n * Hv + hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
            kept[n * n_per_t + i] = state[i];
          }
          out += simd_shuffle_xor(out, 1);
          out += simd_shuffle_xor(out, 2);
          out += simd_shuffle_xor(out, 4);
          out += simd_shuffle_xor(out, 8);
          out += simd_shuffle_xor(out, 16);
          if (thread_index_in_simdgroup == 0) {
            y[(n * Hv + hv_idx) * Dv + dv_idx] = static_cast<InT>(out);
          }
        }
    """
    return mx.fast.metal_kernel(
        name="dspine_gated_delta_tree",
        input_names=["q", "k", "v", "g", "beta", "state_in", "par"],
        output_names=["y"],
        source=source,
    )


def _tree_delta_ops(q, k, v, g, beta, state0, parents):
    """Reference: one mlx-lm ops step per node, each from its parent's state."""
    states, ys = [], []
    for i, p in enumerate(parents):
        sl = slice(i, i + 1)
        y, st = gated_delta_ops(q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], state0 if p < 0 else states[p])
        states.append(st)
        ys.append(y)
    return mx.concatenate(ys, axis=1)


def tree_delta(q, k, v, a, b, A_log, dt_bias, state0, tree: "TreeSpec", *, use_kernel: bool = True) -> mx.array:
    """Gated-delta outputs [1, M, Hv, Dv] for the M tree nodes, node i stepped from its parent's state."""
    global _tree_kernel, _tree_kernel_ok
    beta = mx.sigmoid(b)
    g = compute_g(A_log, a, dt_bias)
    _, M, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if state0 is None:
        state0 = mx.zeros((1, Hv, Dv, Dk), dtype=mx.float32)
    kernel = (use_kernel and _tree_kernel_ok is not False and mx.default_device() == mx.gpu
              and mx.metal.is_available() and Dk >= 32 and Dk % 32 == 0)
    if not kernel:
        return _tree_delta_ops(q, k, v, g, beta, state0, tree.parents)

    def run():
        return _tree_kernel(
            inputs=[q, k, v, g, beta, state0, tree.par_arr],
            template=[("InT", q.dtype), ("StT", state0.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("M", M)],
            grid=(32, Dv, Hv),
            threadgroup=(32, 4, 1),
            output_shapes=[(1, M, Hv, Dv)],
            output_dtypes=[q.dtype],
        )[0]

    if _tree_kernel_ok is None:              # first use: the kernel must agree with the ops reference
        try:
            _tree_kernel = _make_tree_kernel()
            y = run()
            ref = _tree_delta_ops(q, k, v, g, beta, state0, tree.parents)
            err = mx.max(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32))).item()
            scale = mx.max(mx.abs(ref.astype(mx.float32))).item()
            _tree_kernel_ok = bool(err <= 2e-2 * max(scale, 1e-3))
            if not _tree_kernel_ok:
                warnings.warn(f"dspine tree kernel differs from the reference (max error {err:.3g}); not used")
        except Exception as e:  # noqa: BLE001 - any Metal compile or run error: use the reference
            _tree_kernel_ok = False
            warnings.warn(f"dspine tree kernel not available ({type(e).__name__}: {e}); using the ops path")
        if not _tree_kernel_ok:
            return _tree_delta_ops(q, k, v, g, beta, state0, tree.parents)
        return y
    return run()


_forest_kernel = None
_forest_kernel_ok: Optional[bool] = None


def _make_forest_kernel():
    """The tree kernel for several conversations: conversation b's nodes follow its own parents
    (``par`` [B, M]) from its own state, all conversations side by side on the grid."""
    source = """
        auto z_idx = thread_position_in_grid.z;
        auto b_idx = z_idx / Hv;
        auto hv_idx = z_idx % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in: [B, Hv, Dv, Dk]
        auto i_state = state_in + ((b_idx * Hv + hv_idx) * Dv + dv_idx) * Dk;
        auto par_ = par + b_idx * M;
        float kept[M * n_per_t];
        float state[n_per_t];

        for (int n = 0; n < M; ++n) {
          int p = par_[n];
          for (int i = 0; i < n_per_t; ++i) {
            state[i] = p < 0 ? static_cast<float>(i_state[n_per_t * dk_idx + i]) : kept[p * n_per_t + i];
          }
          // q, k: [B, M, Hk, Dk]; v, y: [B, M, Hv, Dv]; g, beta: [B, M, Hv]
          auto row = b_idx * M + n;
          auto q_ = q + (row * Hk + hk_idx) * Dk;
          auto k_ = k + (row * Hk + hk_idx) * Dk;
          auto v_ = v + (row * Hv + hv_idx) * Dv;
          auto gate = g[row * Hv + hv_idx];

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * gate;
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem += simd_shuffle_xor(kv_mem, 1);
          kv_mem += simd_shuffle_xor(kv_mem, 2);
          kv_mem += simd_shuffle_xor(kv_mem, 4);
          kv_mem += simd_shuffle_xor(kv_mem, 8);
          kv_mem += simd_shuffle_xor(kv_mem, 16);

          auto delta = (v_[dv_idx] - kv_mem) * beta[row * Hv + hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
            kept[n * n_per_t + i] = state[i];
          }
          out += simd_shuffle_xor(out, 1);
          out += simd_shuffle_xor(out, 2);
          out += simd_shuffle_xor(out, 4);
          out += simd_shuffle_xor(out, 8);
          out += simd_shuffle_xor(out, 16);
          if (thread_index_in_simdgroup == 0) {
            y[(row * Hv + hv_idx) * Dv + dv_idx] = static_cast<InT>(out);
          }
        }
    """
    return mx.fast.metal_kernel(
        name="dspine_gated_delta_forest",
        input_names=["q", "k", "v", "g", "beta", "state_in", "par"],
        output_names=["y"],
        source=source,
    )


def _forest_delta_ops(q, k, v, g, beta, state0, parents):
    """Reference: the one-conversation tree walk, conversation by conversation."""
    return mx.concatenate([
        _tree_delta_ops(q[b : b + 1], k[b : b + 1], v[b : b + 1], g[b : b + 1], beta[b : b + 1],
                        state0[b : b + 1], par)
        for b, par in enumerate(parents)], axis=0)


def forest_delta(q, k, v, a, b, A_log, dt_bias, state0, tree, *, use_kernel: bool = True) -> mx.array:
    """Gated-delta outputs [B, M, Hv, Dv] for B conversations' trees (``tree``: a ``BatchTree``)."""
    global _forest_kernel, _forest_kernel_ok
    beta = mx.sigmoid(b)
    g = compute_g(A_log, a, dt_bias)
    B, M, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if state0 is None:
        state0 = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    kernel = (use_kernel and _forest_kernel_ok is not False and mx.default_device() == mx.gpu
              and mx.metal.is_available() and Dk >= 32 and Dk % 32 == 0)

    def one_by_one():        # the one-conversation tree walk (its own kernel on Metal) per conversation
        return mx.concatenate([
            tree_delta(q[i : i + 1], k[i : i + 1], v[i : i + 1], a[i : i + 1], b[i : i + 1], A_log, dt_bias,
                       state0[i : i + 1], spec, use_kernel=use_kernel)
            for i, spec in enumerate(tree.specs)], axis=0)

    if not kernel:
        return one_by_one()

    def run():
        return _forest_kernel(
            inputs=[q, k, v, g, beta, state0, tree.par_arr],
            template=[("InT", q.dtype), ("StT", state0.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("M", M)],
            grid=(32, Dv, Hv * B),
            threadgroup=(32, 4, 1),
            output_shapes=[(B, M, Hv, Dv)],
            output_dtypes=[q.dtype],
        )[0]

    if _forest_kernel_ok is None:            # first use: the kernel must agree with the ops reference
        try:
            _forest_kernel = _make_forest_kernel()
            y = run()
            ref = _forest_delta_ops(q, k, v, g, beta, state0, tree.parents)
            err = mx.max(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32))).item()
            scale = mx.max(mx.abs(ref.astype(mx.float32))).item()
            _forest_kernel_ok = bool(err <= 2e-2 * max(scale, 1e-3))
            if not _forest_kernel_ok:
                warnings.warn(f"dspine forest kernel differs from the reference (max error {err:.3g}); not used")
        except Exception as e:  # noqa: BLE001 - any Metal compile or run error: use the reference
            _forest_kernel_ok = False
            warnings.warn(f"dspine forest kernel not available ({type(e).__name__}: {e}); using the ops path")
        if not _forest_kernel_ok:
            return one_by_one()
        return y
    return run()


class TreeSpec:
    """Shape of a draft tree: row 0 is the pending token, ``parents[i] < i`` is row i's parent.

    A node sits at absolute position ``cache offset + depth``. Attention rows see the cached
    context plus their own ancestors; GatedDeltaNet layers run once per root-to-leaf path.
    """

    def __init__(self, parents):
        parents = [int(p) for p in parents]
        m = len(parents)
        if m < 1 or parents[0] != -1 or any(not 0 <= parents[i] < i for i in range(1, m)):
            raise ValueError("parents must start with -1 and point to earlier rows")
        self.parents = parents
        depth = [0] * m
        has_child = [False] * m
        for i in range(1, m):
            depth[i] = depth[parents[i]] + 1
            has_child[parents[i]] = True
        self.depth = depth
        paths = []
        path_of = [-1] * m
        for leaf in range(m):
            if has_child[leaf]:
                continue
            rows, r = [], leaf
            while r >= 0:
                rows.append(r)
                r = parents[r]
            rows.reverse()
            for r in rows:
                if path_of[r] < 0:
                    path_of[r] = len(paths)
            paths.append(rows)
        self.paths = paths
        self.path_of = path_of
        lmax = max(len(p) for p in paths)
        self.pidx = mx.array([p + [0] * (lmax - len(p)) for p in paths], dtype=mx.int32)
        self.path_arr = mx.array(path_of, dtype=mx.int32)
        self.depth_arr = mx.array(depth, dtype=mx.int32)
        # nodes laid out as [slot, depth] so one RoPE call rotates every node by its depth
        slots: list[list[int]] = [[] for _ in range(lmax)]
        slot_of = [0] * m
        for i in range(m):
            slot_of[i] = len(slots[depth[i]])
            slots[depth[i]].append(i)
        width = max(len(x) for x in slots)
        self.grid = mx.array([[slots[d][w] if w < len(slots[d]) else 0 for d in range(lmax)]
                              for w in range(width)], dtype=mx.int32)
        self.slot_arr = mx.array(slot_of, dtype=mx.int32)
        anc = [[False] * m for _ in range(m)]
        for i in range(m):
            r = i
            while r >= 0:
                anc[i][r] = True
                r = parents[r]
        self.ancestors = mx.array(anc)
        self.anc_bits = None          # set by the target when the long-context kernel is on
        self.is_chain = all(parents[i] == i - 1 for i in range(m))
        self.par_arr = mx.array(parents, dtype=mx.int32)
        self._windows: dict[int, mx.array] = {}

    def windows(self, k: int) -> mx.array:
        """[M, k] rows of ``concat(conv state (k-1 rows), node inputs)``: every node's conv window,
        its own input last and its ancestors (then the cached state) before it."""
        if k not in self._windows:
            idx = []
            for i in range(len(self.parents)):
                chain, r = [], i
                while r >= 0 and len(chain) < k:
                    chain.append(r + k - 1)
                    r = self.parents[r]
                chain.reverse()
                idx.append(list(range(len(chain) - 1, k - 1)) + chain)
            self._windows[k] = mx.array(idx, dtype=mx.int32)
        return self._windows[k]

    def rope(self, rope, x: mx.array, offset: int) -> mx.array:
        """``x`` [1, H, M, D] rotated so that row i sits at position offset + depth[i]."""
        g = x[0][:, self.grid, :].transpose(1, 0, 2, 3)      # [W, H, depth, D]
        g = rope(g, offset=offset)
        return g[self.slot_arr, :, self.depth_arr, :].transpose(1, 0, 2)[None]

    def mask(self, n_ctx: int) -> mx.array:
        m = len(self.parents)
        if n_ctx == 0:
            return self.ancestors
        return mx.concatenate([mx.ones((m, n_ctx), dtype=mx.bool_), self.ancestors], axis=1)


class TreeAttention(_BASE_ATTN):
    """mlx-lm ``Qwen3NextAttention`` that can also check a draft tree in one pass.

    Outside a tree verify it is the upstream method. During one, the cache is read but not
    written: the tree rows' keys and values are kept aside and ``commit_tree`` appends the
    accepted path afterwards.
    """

    _box: "_RecordBox"

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        if type(cache).__name__ == "RaggedKV":      # several conversations in one check
            from .batch_target import batch_attention

            return batch_attention(self, x, cache)
        tree = self._box.tree
        if tree is None:
            a8 = self._box.a8            # int8 prefill: wide chunks only, never a check or a step
            if a8 is None:
                return super().__call__(x, mask, cache)
            return self._a8_call(x, mask, cache, a8)
        B, L, D = x.shape
        if B != 1 or cache is None or hasattr(cache, "bits"):
            raise NotImplementedError("tree verify supports batch-1 decode with a plain KV cache")

        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)
        keys, values = self.k_proj(x), self.v_proj(x)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        offset = int(cache.offset)
        queries = tree.rope(self.rope, queries, offset)
        keys = tree.rope(self.rope, keys, offset)
        self._box.tree_record = _Record(keys=keys, values=values)
        # write the tree rows after the context in the cache buffer and step the offset back: no
        # copy of the whole context per layer (at 32k that was 130 MB each); commit_tree overwrites
        # these slots with the accepted path
        all_keys, all_values = cache.update_and_fetch(keys, values)
        cache.offset = offset
        output = None
        cfg = getattr(self._box, "long_attn", None)
        if cfg is not None:
            from . import long_attn

            output = long_attn.tree_attention(cfg, queries, all_keys, all_values, self.scale, tree.anc_bits)
        if output is None:
            output = mx.fast.scaled_dot_product_attention(
                queries, all_keys, all_values, scale=self.scale, mask=tree.mask(offset)
            )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output * mx.sigmoid(gate))

    def _a8_call(self, x, mask, cache, a8):
        """mlx-lm 0.32.2 ``qwen3_5.Attention.__call__`` with the four projections through int8."""
        from mlx_lm.models.base import scaled_dot_product_attention

        from . import a8_prefill

        front = a8_prefill.project((self.q_proj, self.k_proj, self.v_proj), x, a8)
        if front is None:
            return super().__call__(x, mask, cache)
        B, L, D = x.shape
        q_proj_output, keys, values = front
        queries, gate = mx.split(q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        output = (output.transpose(0, 2, 1, 3).reshape(B, L, -1) * mx.sigmoid(gate)).astype(x.dtype)
        last = a8_prefill.project((self.o_proj,), output, a8)
        return last[0] if last is not None else self.o_proj(output)

    def commit_tree(self, cache, rows: mx.array) -> None:
        rec = self._box.tree_record
        if rec is None:
            raise RuntimeError("no tree verify to commit")
        cache.update_and_fetch(rec.keys[:, :, rows, :], rec.values[:, :, rows, :])
        self._box.tree_record = None


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
        if self._box.tree is not None:
            return self._tree_call(inputs, cache, self._box.tree)

        a8 = self._box.a8                # int8 prefill: wide chunks only, never a check or a step
        front = None
        if a8 is not None:
            from . import a8_prefill

            front = a8_prefill.project((self.in_proj_qkv, self.in_proj_z), inputs, a8)
        if front is not None:
            qkv, z = front[0], front[1].reshape(B, S, self.num_v_heads, self.head_v_dim)
        else:
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
                raise NotImplementedError("rollback supports unpadded decode")
            self._box.record = _Record(
                conv_input=conv_input, ssm0=state, q=q, k=k, v=v, a=a, b=b, steps=S
            )

        fast = None
        if self._box.gdn is not None and mask is None and not self.training:
            from . import gdn_prefill

            fast = gdn_prefill.update(q, k, v, a, b, self.A_log, self.dt_bias, state, self._box.gdn)
        if fast is not None:
            out, state = fast
        else:
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
        out = out.reshape(B, S, -1)
        last = a8_prefill.project((self.out_proj,), out, a8) if a8 is not None else None
        return last[0] if last is not None else self.out_proj(out)

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

    def _tree_call(self, inputs: mx.array, cache, tree: "TreeSpec") -> mx.array:
        """The recurrence over a draft tree, from the cached state; the cache is not written.

        By node (default): every node is stepped once from its parent's state, so a tree costs
        what a chain of as many rows costs. ``DSPINE_TREE_PATHS=1`` keeps the older layout, one
        full recurrence per root-to-leaf path (a comb of 16 rows then runs 72 steps for 16 nodes).
        """
        B, S, _ = inputs.shape
        if getattr(tree, "specs", None) is not None:        # several conversations (BatchTree)
            return self._tree_call_batch(inputs, cache, tree)
        if B != 1 or cache is None or cache.lengths is not None:
            raise NotImplementedError("tree verify supports unpadded batch-1 decode")
        if TREE_BY_NODE:
            return self._tree_call_nodes(inputs, cache, tree)
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        P, L = tree.pidx.shape
        if cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros((1, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)
        conv_input = mx.concatenate([mx.repeat(conv_state, P, axis=0), qkv[0][tree.pidx]], axis=1)
        conv_out = nn.silu(self.conv1d(conv_input))
        q, k, v = [
            t.reshape(P, L, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        a = a[0][tree.pidx]
        b = b[0][tree.pidx]
        state0 = cache[1]
        state = None if state0 is None else mx.repeat(state0, P, axis=0)
        q, k = normalize_qk(q, k, inv_scale=self.head_k_dim**-0.5, eps=1e-6)
        self._box.tree_record = _Record(conv_input=conv_input, ssm0=state0, q=q, k=k, v=v, a=a, b=b)
        out, _ = gated_delta_update(
            q, k, v, a, b, self.A_log, self.dt_bias, state, None, use_kernel=not self.training
        )
        out = out[tree.path_arr, tree.depth_arr][None]       # back to one row per node
        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))

    def _tree_call_nodes(self, inputs: mx.array, cache, tree: "TreeSpec") -> mx.array:
        B, S, _ = inputs.shape
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        K = self.conv_kernel_size
        if cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros((1, K - 1, self.conv_dim), dtype=inputs.dtype)
        src = mx.concatenate([conv_state[0], qkv[0]], axis=0)
        conv_input = src[tree.windows(K)]                    # [S, K, C]: one window per node
        conv_out = nn.silu(self.conv1d(conv_input))          # [S, 1, C]
        q, k, v = [
            t.reshape(1, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        state0 = cache[1]
        q, k = normalize_qk(q, k, inv_scale=self.head_k_dim**-0.5, eps=1e-6)
        self._box.tree_record = _Record(conv_input=conv_input, ssm0=state0, q=q, k=k, v=v, a=a, b=b, nodes=True)
        out = tree_delta(q, k, v, a, b, self.A_log, self.dt_bias, state0, tree, use_kernel=not self.training)
        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))

    def _tree_call_batch(self, inputs: mx.array, cache, tree) -> mx.array:
        """``_tree_call_nodes`` for B conversations, each with its own tree of S rows."""
        B, S, _ = inputs.shape
        if cache is None or cache[0] is None or cache.lengths is not None:
            raise NotImplementedError("a batched tree verify needs the joined caches")
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        K = self.conv_kernel_size
        W = K - 1 + S
        src = mx.concatenate([cache[0], qkv], axis=1)        # [B, W, C]
        idx = tree.windows(K) + (mx.arange(B) * W)[:, None, None]
        conv_input = src.reshape(B * W, -1)[idx.reshape(B * S, K)]   # [B * S, K, C]: one window per node
        conv_out = nn.silu(self.conv1d(conv_input))
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        state0 = cache[1]
        q, k = normalize_qk(q, k, inv_scale=self.head_k_dim**-0.5, eps=1e-6)
        self._box.tree_record = _Record(conv_input=conv_input.reshape(B, S, K, -1), ssm0=state0,
                                        q=q, k=k, v=v, a=a, b=b)
        out = forest_delta(q, k, v, a, b, self.A_log, self.dt_bias, state0, tree, use_kernel=not self.training)
        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))

    def commit_tree_batch(self, cache: ArraysCache, paths: list) -> None:
        """Set ``cache`` [B, ...] to the state after each conversation's ``paths[b]`` (tree rows from
        row 0 downwards; empty keeps the state as before the check)."""
        rec = self._box.tree_record
        if rec is None:
            raise RuntimeError("no tree verify to commit")
        self._box.tree_record = None
        counts = [len(p) for p in paths]
        top = max(counts)
        if top == 0:
            return
        B = len(paths)
        conv = rec.conv_input[mx.arange(B), mx.array([p[-1] if p else 0 for p in paths])][:, 1:, :]
        if min(counts) == 0:
            conv = mx.where(mx.array(counts)[:, None, None] > 0, conv, cache[0])
        cache[0] = conv
        idx = mx.array([p + [0] * (top - len(p)) for p in paths])
        i3, i4 = idx[:, :, None], idx[:, :, None, None]
        mask = None if min(counts) == top else mx.arange(top)[None, :] < mx.array(counts)[:, None]
        _, state = gated_delta_update(
            mx.take_along_axis(rec.q, i4, axis=1), mx.take_along_axis(rec.k, i4, axis=1),
            mx.take_along_axis(rec.v, i4, axis=1), mx.take_along_axis(rec.a, i3, axis=1),
            mx.take_along_axis(rec.b, i3, axis=1),
            self.A_log, self.dt_bias, rec.ssm0, mask, use_kernel=not self.training)
        cache[1] = state

    def commit_tree(self, cache: ArraysCache, path: int, accepted: int, rows: Optional[mx.array] = None,
                    last: int = -1) -> None:
        """Set ``cache`` to the state after the first ``accepted`` rows of recorded path ``path``
        (``rows``: the same path as tree rows, ``last`` its final row as a plain int: reading it
        from the array would be one GPU round trip per layer, 10 ms a round on the 27B)."""
        rec = self._box.tree_record
        if rec is None:
            raise RuntimeError("no tree verify to commit")
        n_keep = self.conv_kernel_size - 1
        if getattr(rec, "nodes", False):
            if rows is None or last < 0:
                raise ValueError("a by-node tree commit needs the path rows")
            cache[0] = mx.contiguous(rec.conv_input[last : last + 1, 1:, :])
            _, state = gated_delta_update(
                rec.q[:, rows], rec.k[:, rows], rec.v[:, rows], rec.a[:, rows], rec.b[:, rows],
                self.A_log, self.dt_bias, rec.ssm0, None, use_kernel=not self.training,
            )
            cache[1] = state
            return
        p = slice(path, path + 1)
        cache[0] = mx.contiguous(rec.conv_input[p, accepted : accepted + n_keep, :])
        sl = slice(0, accepted)
        _, state = gated_delta_update(
            rec.q[p, sl], rec.k[p, sl], rec.v[p, sl], rec.a[p, sl], rec.b[p, sl],
            self.A_log, self.dt_bias, rec.ssm0, None, use_kernel=not self.training,
        )
        cache[1] = state
        cache.advance(accepted)
        self._box.tree_record = None


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
        if type(mod) not in (_BASE_GDN, RecordingGatedDeltaNet, _UPSTREAM_GDN):
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


def install_tree_attention(model) -> list:
    """Per-cache-slot list of full-attention modules (None for GatedDeltaNet layers)."""
    out = []
    for layer in _inner_layers(model):
        layer = getattr(layer, "_layer", layer)
        mod = getattr(layer, "self_attn", None)
        if mod is None:
            out.append(None)
            continue
        if type(mod) not in (_BASE_ATTN, TreeAttention):
            raise TypeError(f"unsupported attention module {type(mod).__name__}")
        mod.__class__ = TreeAttention
        if not isinstance(getattr(mod, "_box", None), _RecordBox):
            object.__setattr__(mod, "_box", _RecordBox())
        out.append(mod)
    return out


class _FlushHook:
    """Submits the graph built so far every few layers of a verify forward.

    A 64-layer verify is a few thousand lazy ops; without this the GPU only starts once Python
    has built all of them (idea from oMLX's verify layer flush).
    """

    def __init__(self, layer, index: int, box: "_RecordBox"):
        self._layer = layer
        self._index = index
        self._fbox = box

    def __call__(self, *args, **kwargs):
        out = self._layer(*args, **kwargs)
        every = self._fbox.flush
        if every and (self._index + 1) % every == 0:
            mx.async_eval(out)
        return out

    def __getattr__(self, name):
        return getattr(self._layer, name)


class HybridTargetAdapter(TargetAdapter):
    """``TargetAdapter`` for qwen3_5 hybrid targets (e.g. Qwen3.8-27B)."""

    def __init__(self, model, tokenizer, target_layer_ids):
        super().__init__(model, tokenizer, target_layer_ids)
        self._linear = install_recording(model)
        self._attn = install_tree_attention(model)
        self._tree = None
        self.fast_verify = False      # route verify forwards through oMLX's verify kernels
        self.long_attn = None         # multi-row attention kernel gate for long-context checks
        self.a8_prefill = None        # int8 prefill settings, see a8_prefill.enable
        self.gdn_prefill = None       # pipelined prefill recurrence, see enable_gdn_prefill
        self.flush_every = 0          # submit the verify graph every N layers while it is built
        self._flush_box = None
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

    def _prefill_fresh(self, tokens: mx.array, capture: bool = True):
        """Chunked prefill that returns only the last position's logits.

        A one-shot prefill materialises logits for every prompt position: at 32k
        tokens and a 248k vocabulary that is about 16 GB, held for the whole
        decode and computed for nothing. Chunks whose logits are never evaluated
        skip the LM head entirely (MLX is lazy). Hidden features are kept for
        every position, as the drafter needs them.
        """
        self.cache = make_prompt_cache(self.model)
        n = int(tokens.shape[1])
        hs = []
        logits = hidden = None
        for s in range(0, n, PREFILL_CHUNK):
            logits, hidden = self._forward(tokens[:, s : s + PREFILL_CHUNK], capture=capture)
            if capture:
                hs.append(hidden)
            if s + PREFILL_CHUNK < n:
                mx.eval([c.state for c in self.cache] + hs[-1:])
        if capture and len(hs) > 1:
            hidden = mx.concatenate(hs, axis=1)
        return logits[:, -1:], hidden

    def _prefill_chunks(self, tokens: mx.array):
        """Feed ``tokens`` on top of the current cache; last-position logits, all hidden rows."""
        n = int(tokens.shape[1])
        hs = []
        logits = None
        for s in range(0, n, PREFILL_CHUNK):
            logits, hidden = self._forward(tokens[:, s : s + PREFILL_CHUNK], capture=True)
            hs.append(hidden)
            if s + PREFILL_CHUNK < n:
                mx.eval([c.state for c in self.cache] + hs[-1:])
        hidden = hs[0] if len(hs) == 1 else mx.concatenate(hs, axis=1)
        return logits[:, -1:], hidden

    # ---- prompt cache across turns ---------------------------------------------------------
    #
    # With the prompt cache on, ``reset()`` keeps the target cache, and ``prefill(prompt)``
    # reuses the longest part of it that ``prompt`` starts with:
    #   * everything the cache holds (previous prompt + committed answer), when the new prompt
    #     continues it, as in a chat or agent loop that sends the whole history each turn;
    #   * otherwise the previous prompt, from a snapshot taken at the end of its prefill
    #     (GatedDeltaNet state can't be rewound, so restore points have to be saved);
    #   * or a mark: a restore point inside the prompt. One is saved where a prompt stopped
    #     matching the one before it (the end of a shared system prompt), and one where the
    #     caller says a stable prefix ends (``mark_hint``, e.g. the system prompt's length), so a
    #     new conversation with the same long system prompt prefills only what follows it.
    # Anything else falls back to a full prefill. The drafter is fed the last ``ctx_rows``
    # target features, which is all its sliding window can see, so a fresh draft cache
    # gets the same context as after a full prefill.

    prompt_cache = False
    mark_min = 256        # shortest shared prefix worth a mark, in tokens
    mark_max = 3          # marks kept (each holds the GatedDeltaNet states and drafter rows)
    mark_hint = None      # set before prefill: a prefix length to mark in this prompt

    def enable_prompt_cache(self, on: bool = True, ctx_rows: int = 8192) -> None:
        self.prompt_cache = bool(on)
        self._ctx_rows = int(ctx_rows)
        self._drop_session()

    def _drop_session(self) -> None:
        self._seq = None          # token ids the cache holds, committed
        self._hid = []            # hidden chunks for the newest committed positions
        self._pending = None      # (token ids, hidden) of a verify not yet trimmed
        self._snap = None         # (n_tokens, layer states, hidden rows, logits) after the last prefill
        self._marks = []          # (n_tokens, layer states, hidden rows, None) inside the last prompt
        self._shared = 0          # tokens the last prompt shared with the cache before it
        self._settled_short = False
        self.last_reused = 0

    def reset(self) -> None:
        if self.prompt_cache and self.cache is not None and getattr(self, "_seq", None) is not None:
            return
        super().reset()
        self._drop_session()

    def step(self, token: int) -> mx.array:
        if self.prompt_cache:
            self._seq = None      # plain decoding bypasses the bookkeeping
        return super().step(token)

    def _hid_rows(self) -> mx.array | None:
        if not self._hid:
            return None
        h = self._hid[0] if len(self._hid) == 1 else mx.concatenate(self._hid, axis=1)
        h = h[:, -self._ctx_rows :]
        self._hid = [h]
        return h

    def _commit(self, ids: list[int], hidden: mx.array | None) -> None:
        self._seq.extend(ids)
        if hidden is not None:
            self._hid.append(hidden)
            if sum(int(x.shape[1]) for x in self._hid) > 2 * self._ctx_rows:
                self._hid_rows()

    def _layer_state(self):
        out = []
        for c, mod in zip(self.cache, self._linear):
            out.append(int(c.offset) if mod is None else (c[0], c[1]))
        return out

    def _restore(self, snap) -> None:
        n, states, hid, _ = snap
        for c, mod, st in zip(self.cache, self._linear, states):
            if mod is None:
                c.trim(int(c.offset) - st)
            else:
                c[0], c[1] = st
                mod._box.record = None
        self._seq = self._seq[:n]
        self._marks = [m for m in self._marks if m[0] <= n]
        self._hid = [hid] if hid is not None else []
        self._pending = None
        self._verify_steps = 0

    def _settle(self, ids: list[int]) -> None:
        """Commit the part of an untrimmed last verify that the new prompt repeats."""
        if self._pending is None:
            return
        toks, hidden = self._pending
        toks = [int(t) for t in toks.tolist()]
        base = len(self._seq)
        m = 0
        while m < len(toks) and base + m < len(ids) and ids[base + m] == toks[m]:
            m += 1
        if m == 0:
            self._settled_short = True   # the cache is ahead of the prompt; only the snapshot helps
            self._pending = None
            return
        self.trim(len(toks) - m)

    def _resume(self, ids: list[int]) -> int:
        """Bring the cache to a prefix of ``ids``; return its length (0 = start over)."""
        self._shared = 0
        if self.cache is None or self._seq is None:
            return 0
        self._settle(ids)
        seq = self._seq
        c = 0
        lim = min(len(seq), len(ids))
        while c < lim and seq[c] == ids[c]:
            c += 1
        self._shared = c
        if c == len(seq) and c < len(ids) and not self._settled_short:
            return c
        best = None
        for snap in [self._snap] + self._marks:
            # a mark has no logits, so it can't stand for the whole prompt
            if snap is None or snap[0] > c or (snap[3] is None and snap[0] >= len(ids)):
                continue
            if best is None or snap[0] > best[0]:
                best = snap
        if best is not None:
            self._restore(best)
            self._settled_short = False
            return best[0]
        return 0

    def rewind_to_prompt(self) -> bool:
        """After a generation cut short: go back to the end of its prompt. False = nothing kept."""
        snap = getattr(self, "_snap", None)
        if not self.prompt_cache or self.cache is None or self._seq is None or snap is None:
            return False
        self._restore(snap)
        self._tree = None
        self._settled_short = False
        return True

    def _mark_points(self, start: int, n: int) -> list[int]:
        """Where to save marks while prefilling ids[start:n]."""
        pts = {m[0] for m in self._marks}
        if self._shared >= self.mark_min:
            pts.add(self._shared)
        if self.mark_hint is not None and int(self.mark_hint) >= self.mark_min:
            pts.add(int(self.mark_hint))
        return sorted(p for p in pts if start < p < n and p <= max(self._shared, int(self.mark_hint or 0)))

    def prefill(self, tokens: mx.array, capture: bool = True):
        if not (self.prompt_cache and capture):
            if self.prompt_cache:
                out = self._prefill_fresh(tokens, capture)
                self._drop_session()
                return out
            return self._prefill_fresh(tokens, capture)
        ids = [int(t) for t in tokens[0].tolist()]
        start = self._resume(ids)
        if start == len(ids):        # same prompt again: the snapshot holds its logits
            logits, hidden = self._snap[3], self._snap[2]
            self._settled_short = False
            self.last_reused = start
            return logits, hidden
        points = self._mark_points(start, len(ids))
        self.mark_hint = None
        if start == 0:
            self._drop_session()
            self.cache = make_prompt_cache(self.model)
            self._seq = []
            prev = None
        else:
            prev = self._hid_rows()
        parts = [] if prev is None else [prev]
        pos = start
        for p in points + [len(ids)]:
            logits, new = self._prefill_chunks(tokens[:, pos:p])
            self._commit(ids[pos:p], new)
            parts.append(new)
            pos = p
            if p < len(ids):
                rows = self._hid_rows()
                mx.eval([c.state for c in self.cache], rows)
                self._marks.append((p, self._layer_state(), rows, None))
        self._marks = self._marks[-self.mark_max:]
        hidden = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
        rows = self._hid_rows()
        mx.eval(logits, rows)
        self._snap = (len(ids), self._layer_state(), rows, logits)
        self._settled_short = False
        self.last_reused = start
        return logits, hidden

    def verify(self, tokens: mx.array, capture: bool = True):
        from . import fast_verify, long_attn

        self._set_recording(True)
        self._set_flush(True)
        try:
            with fast_verify.armed(self.fast_verify), long_attn.active(self.long_attn):
                out = self._forward(tokens, capture=capture)
        finally:
            self._set_recording(False)
            self._set_flush(False)
        self._verify_steps = int(tokens.shape[1])
        if self.prompt_cache and self._seq is not None:
            self._pending = (tokens[0], out[1])
        return out

    def trim(self, n: int) -> None:
        if self.prompt_cache and self._pending is not None:
            toks, hidden = self._pending
            keep = int(toks.shape[0]) - max(int(n), 0)
            self._pending = None
            self._commit([int(t) for t in toks[:keep].tolist()], None if hidden is None else hidden[:, :keep])
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

    # ---- draft trees -------------------------------------------------------------------------

    def _set_tree(self, tree) -> None:
        for mod in self._linear + self._attn:
            if mod is not None:
                mod._box.tree = tree

    def verify_tree(self, tokens, parents, capture: bool = True):
        """Check a draft tree in one forward. Row 0 is the pending token (parent -1).

        Returns (logits, hidden) with one row per node; each row equals what a chain verify of
        that node's root path would give. The cache is unchanged until ``commit_tree``.
        """
        tree = TreeSpec(parents)
        tokens = [int(t) for t in tokens]
        if len(tokens) != len(tree.parents):
            raise ValueError("one token per tree row")
        if len(self.cache) != len(self._linear):
            raise RuntimeError("cache / layer count mismatch")
        from . import fast_verify, long_attn

        if self.long_attn is not None and long_attn.TREE:
            tree.anc_bits = long_attn.tree_bits(tree.ancestors)
        self._set_tree(tree)
        self._set_flush(True)
        try:
            with fast_verify.armed(self.fast_verify):
                out = self._forward(mx.array([tokens], dtype=mx.int32), capture=capture)
        finally:
            self._set_flush(False)
            self._set_tree(None)
        self._tree = (tree, tokens, out[1])
        self._pending = None
        self._verify_steps = 0
        return out

    def commit_tree(self, rows) -> None:
        """Advance the cache over ``rows``, a path of the last tree from row 0 downwards."""
        if self._tree is None:
            raise RuntimeError("commit_tree needs a preceding verify_tree")
        tree, tokens, hidden = self._tree
        self._tree = None
        rows = [int(r) for r in rows]
        if not rows or rows[0] != 0 or any(tree.parents[rows[i]] != rows[i - 1] for i in range(1, len(rows))):
            raise ValueError("rows must be a path from row 0")
        arr = mx.array(rows, dtype=mx.int32)
        path = tree.path_of[rows[-1]]
        for c, mod, att in zip(self.cache, self._linear, self._attn):
            if mod is None:
                att.commit_tree(c, arr)
            else:
                mod.commit_tree(c, path, len(rows), arr, rows[-1])
        if self.prompt_cache and self._seq is not None:
            self._commit([tokens[r] for r in rows], None if hidden is None else hidden[:, arr])

    def enable_fast_verify(self, on: bool = True) -> bool:
        """Use oMLX's verify-shape kernels for verify forwards, if oMLX is installed."""
        from . import fast_verify

        self.fast_verify = bool(on) and fast_verify.install()
        head = self.head
        if self.fast_verify and getattr(head, "quantized", False) and "matmul" not in vars(head):
            stock = head.matmul

            def matmul(x, _stock=stock, _head=head, _self=self):
                # the drafter's pass-1 logits: route the head through the same kernels
                if not _self.fast_verify or x.ndim != 2:
                    return _stock(x)
                with fast_verify.armed():
                    return _head.module(x[None])[0]

            head.matmul = matmul
        return self.fast_verify

    def plain_layers(self) -> list:
        """The target's decoder layers without the kit's hooks (hidden-state taps, flush hooks).

        The hooks are plain objects, so ``model.named_modules()`` does not reach the layers behind
        them; code that walks the model (third-party patches, measurements) must use this list."""
        return [getattr(layer, "_layer", layer) for layer in self.layers]

    def enable_a8_prefill(self, on: bool = True, min_tokens: int = 128, parts=("mlp", "gdn")) -> bool:
        """Int8-activation prefill through oMLX's native kernels, if they are built (see a8_prefill)."""
        from . import a8_prefill

        return a8_prefill.enable(self, on, min_tokens, parts)

    def enable_gdn_prefill(self, on: bool = True, min_tokens: int = 64) -> bool:
        """Pipelined GatedDeltaNet recurrence for prefill chunks (oMLX's kernel, see gdn_prefill)."""
        from . import gdn_prefill

        cfg = gdn_prefill.Config(min_tokens) if on and gdn_prefill.kernels() is not None else None
        self.gdn_prefill = cfg
        for mod in self._linear:
            if mod is not None:
                mod._box.gdn = cfg
        return cfg is not None

    def enable_long_attention(self, on: bool = True, window=None) -> bool:
        """Use the multi-row attention kernel for checks of several rows at long context (Metal)."""
        from . import long_attn

        attn = next((a for a in self._attn if a is not None), None)
        self.long_attn = long_attn.config(attn, window) if on else None
        for a in self._attn:
            if a is not None:
                a._box.long_attn = self.long_attn
        return self.long_attn is not None

    def enable_flush(self, every: int = 8) -> None:
        """Submit the verify graph to the GPU every ``every`` layers while Python builds the rest."""
        if every and self._flush_box is None:
            self._flush_box = _RecordBox()
            layers = self.layers
            for i in range(len(layers)):
                if not hasattr(layers[i], "_storage"):      # leave the hidden-state taps as they are
                    layers[i] = _FlushHook(layers[i], i, self._flush_box)
        self.flush_every = int(every)

    def _set_flush(self, on: bool) -> None:
        if self._flush_box is not None:
            self._flush_box.flush = self.flush_every if on else 0
