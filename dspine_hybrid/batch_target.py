"""Several conversations checked in one target forward.

A ``BatchTarget`` holds the caches of B conversations side by side and runs one verify pass
for all of them: tokens [B, M] in, logits and hidden features [B, M, ...] out. Each
conversation then keeps as many of its M rows as its own draft got accepted.

Nothing is copied per round:

* Attention layers keep one key/value buffer [B, heads, capacity, dim] and a length per
  conversation. A check writes each conversation's M new rows at its own end, and row i of
  conversation b sees positions <= len_b + i. After the check the length grows by the rows
  kept; the rest is overwritten next round.
* GatedDeltaNet layers record the round as for one conversation (``RecordingGatedDeltaNet``)
  and are rolled back per conversation: the recurrence is replayed over the kept rows, masked
  beyond each conversation's count.

A conversation joins with the cache of an ordinary prefill (``HybridTargetAdapter``) and
leaves by index, both between rounds.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import gated_delta_update


class BatchTree:
    """One draft tree of M rows per conversation (``parents[b]``: as for ``TreeSpec``, row 0 is the
    conversation's pending token)."""

    def __init__(self, parents: list):
        from .hybrid_target import TreeSpec

        self.specs = [TreeSpec(p) for p in parents]
        if len({len(s.parents) for s in self.specs}) != 1:
            raise ValueError("every conversation needs the same number of tree rows")
        self.parents = [s.parents for s in self.specs]
        self.rows = len(self.parents[0])
        self.ancestors = mx.stack([s.ancestors for s in self.specs])      # [B, M, M]
        self.par_arr = mx.stack([s.par_arr for s in self.specs])          # [B, M]
        self._windows: dict[int, mx.array] = {}

    def windows(self, k: int) -> mx.array:
        if k not in self._windows:
            self._windows[k] = mx.stack([s.windows(k) for s in self.specs])
        return self._windows[k]

    def check_path(self, b: int, path: list) -> None:
        par = self.parents[b]
        if path and (path[0] != 0 or any(par[path[i]] != path[i - 1] for i in range(1, len(path)))):
            raise ValueError(f"conversation {b}: rows {path} are not a path from row 0")


class RaggedKV:
    """Keys and values of B conversations with different lengths, in one buffer."""

    step = 256

    def __init__(self):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.lens: list[int] = []
        self.pending = 0  # rows written by the last check and not yet kept or dropped
        self.tree: Optional[BatchTree] = None  # set for the duration of a tree check

    def make_mask(self, *args, **kwargs):  # the batched attention builds its own mask
        return None

    @property
    def state(self):
        return self.keys, self.values

    def _grow(self, need: int) -> None:
        cap = int(self.keys.shape[2])
        if need <= cap:
            return
        extra = ((need - cap + self.step - 1) // self.step) * self.step
        B, H, _, Dk = self.keys.shape
        self.keys = mx.concatenate([self.keys, mx.zeros((B, H, extra, Dk), self.keys.dtype)], axis=2)
        self.values = mx.concatenate(
            [self.values, mx.zeros((B, H, extra, self.values.shape[3]), self.values.dtype)], axis=2)

    def add(self, keys: mx.array, values: mx.array) -> None:
        """Append one conversation: ``keys`` / ``values`` [1, H, n, D] (exactly its n cached rows)."""
        n = int(keys.shape[2])
        if self.keys is None:
            cap = ((n + self.step - 1) // self.step + 1) * self.step
            pad = cap - n
            self.keys = mx.concatenate([keys, mx.zeros((1, keys.shape[1], pad, keys.shape[3]), keys.dtype)], axis=2)
            self.values = mx.concatenate(
                [values, mx.zeros((1, values.shape[1], pad, values.shape[3]), values.dtype)], axis=2)
        else:
            self._grow(n)
            pad = int(self.keys.shape[2]) - n
            k = mx.concatenate([keys, mx.zeros((1, keys.shape[1], pad, keys.shape[3]), keys.dtype)], axis=2)
            v = mx.concatenate([values, mx.zeros((1, values.shape[1], pad, values.shape[3]), values.dtype)], axis=2)
            self.keys = mx.concatenate([self.keys, k], axis=0)
            self.values = mx.concatenate([self.values, v], axis=0)
        self.lens.append(n)

    def remove(self, index: int) -> None:
        keep = [i for i in range(len(self.lens)) if i != index]
        del self.lens[index]
        if not keep:
            self.keys = self.values = None
            return
        idx = mx.array(keep)
        self.keys, self.values = self.keys[idx], self.values[idx]

    def write(self, keys: mx.array, values: mx.array):
        """Put each conversation's new rows [B, H, M, D] at its own end; return the buffers up to
        the longest end and the mask [B, 1, M, total]."""
        M = int(keys.shape[2])
        total = max(self.lens) + M
        self._grow(total)
        for b, n in enumerate(self.lens):
            self.keys[b : b + 1, :, n : n + M, :] = keys[b : b + 1]
            self.values[b : b + 1, :, n : n + M, :] = values[b : b + 1]
        self.pending = M
        pos = mx.arange(total)[None, None, :]
        if self.tree is None:
            ends = mx.array(self.lens)[:, None, None] + mx.arange(M)[None, :, None]
            mask = pos <= ends
        else:              # a row sees its conversation's context and its own ancestors in the tree
            rel = pos - mx.array(self.lens)[:, None, None]                 # [B, 1, total]
            col = mx.broadcast_to(mx.clip(rel, 0, M - 1), (len(self.lens), M, total))
            mask = (rel < 0) | ((rel < M) & mx.take_along_axis(self.tree.ancestors, col, axis=2))
        return self.keys[:, :, :total, :], self.values[:, :, :total, :], mask[:, None]

    def keep(self, rows: list[int]) -> None:
        if any(not 0 <= r <= self.pending for r in rows):
            raise ValueError(f"rows kept {rows} outside [0, {self.pending}]")
        self.lens = [n + r for n, r in zip(self.lens, rows)]
        self.pending = 0


    def keep_paths(self, paths: list) -> None:
        """After a tree check: keep each conversation's ``paths[b]`` (tree rows, in order)."""
        for b, (n, rows) in enumerate(zip(self.lens, paths)):
            k = len(rows)
            if k > self.pending:
                raise ValueError(f"path {rows} longer than the {self.pending} rows checked")
            if rows != list(range(k)):
                idx = mx.array([n + r for r in rows])
                self.keys[b : b + 1, :, n : n + k, :] = mx.take(self.keys[b : b + 1], idx, axis=2)
                self.values[b : b + 1, :, n : n + k, :] = mx.take(self.values[b : b + 1], idx, axis=2)
        self.lens = [n + len(rows) for n, rows in zip(self.lens, paths)]
        self.pending = 0


def batch_attention(attn, x: mx.array, cache: RaggedKV) -> mx.array:
    """mlx-lm ``qwen3_5.Attention.__call__`` for a ``RaggedKV`` cache (positions per conversation)."""
    B, L, _ = x.shape
    if B != len(cache.lens):
        raise ValueError(f"batch of {B} rows for {len(cache.lens)} conversations")
    q_proj_output = attn.q_proj(x)
    queries, gate = mx.split(q_proj_output.reshape(B, L, attn.num_attention_heads, -1), 2, axis=-1)
    gate = gate.reshape(B, L, -1)
    keys, values = attn.k_proj(x), attn.v_proj(x)
    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    keys = attn.k_norm(keys.reshape(B, L, attn.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, attn.num_key_value_heads, -1).transpose(0, 2, 1, 3)
    tree = cache.tree
    if tree is None:
        rope = [lambda t, n=n: attn.rope(t, offset=n) for n in cache.lens]
    else:                  # a tree row sits at its conversation's length + its depth
        rope = [lambda t, n=n, spec=spec: spec.rope(attn.rope, t, n) for n, spec in zip(cache.lens, tree.specs)]
    queries = mx.concatenate([r(queries[b : b + 1]) for b, r in enumerate(rope)], axis=0)
    keys = mx.concatenate([r(keys[b : b + 1]) for b, r in enumerate(rope)], axis=0)
    all_keys, all_values, mask = cache.write(keys, values)
    out = mx.fast.scaled_dot_product_attention(queries, all_keys, all_values, scale=attn.scale, mask=mask)
    out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return attn.o_proj(out * mx.sigmoid(gate))


def rollback_rows(mod, cache: ArraysCache, rows: list[int]) -> None:
    """Set a GatedDeltaNet ``cache`` [B, ...] to the state after ``rows[b]`` recorded steps of each
    conversation (0 = as before the check)."""
    rec = mod._box.record
    if rec is None:
        raise RuntimeError("no recorded verify pass to roll back")
    steps = int(rec.steps)
    if any(not 0 <= r <= steps for r in rows):
        raise ValueError(f"rows kept {rows} outside [0, {steps}]")
    mod._box.record = None
    if all(r == steps for r in rows):
        return
    n_keep = mod.conv_kernel_size - 1
    top = max(rows)
    pos = (mx.array(rows)[:, None] + mx.arange(n_keep))[..., None]
    cache[0] = mx.take_along_axis(rec.conv_input, pos, axis=1)
    if top == 0:
        cache[1] = rec.ssm0
        return
    sl = slice(0, top)
    mask = None if all(r == top for r in rows) else mx.arange(top)[None, :] < mx.array(rows)[:, None]
    _, state = gated_delta_update(
        rec.q[:, sl], rec.k[:, sl], rec.v[:, sl], rec.a[:, sl], rec.b[:, sl],
        mod.A_log, mod.dt_bias, rec.ssm0, mask, use_kernel=not mod.training)
    cache[1] = state


class BatchTarget:
    """The caches of several conversations on one ``HybridTargetAdapter``'s model."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.cache: Optional[list] = None
        self.size = 0
        self._rows = 0
        self._tree: Optional[BatchTree] = None

    def join(self, cache: Optional[list] = None) -> int:
        """Add a conversation from an ordinary single-conversation cache (default: the adapter's
        current one, which the adapter then gives up). Returns its index."""
        ad = self.adapter
        if self._rows:
            raise RuntimeError("join between rounds: keep() the last check first")
        if cache is None:
            cache, ad.cache = ad.cache, None
            if getattr(ad, "prompt_cache", False):
                ad._drop_session()
        if cache is None or len(cache) != len(ad._linear):
            raise ValueError("join needs the cache of a finished prefill")
        if self.cache is None:
            self.cache = [ArraysCache(2) if mod is not None else RaggedKV() for mod in ad._linear]
        for mine, c, mod in zip(self.cache, cache, ad._linear):
            if mod is None:
                n = int(c.offset)
                mine.add(c.keys[..., :n, :], c.values[..., :n, :])
            else:
                for i in (0, 1):
                    mine[i] = c[i] if mine[i] is None else mx.concatenate([mine[i], c[i]], axis=0)
        mx.eval([c.state for c in self.cache])
        self.size += 1
        return self.size - 1

    def leave(self, index: int) -> None:
        if self._rows:
            raise RuntimeError("leave between rounds: keep() the last check first")
        if not 0 <= index < self.size:
            raise IndexError(index)
        keep = [i for i in range(self.size) if i != index]
        for c, mod in zip(self.cache, self.adapter._linear):
            if mod is None:
                c.remove(index)
            elif keep:
                idx = mx.array(keep)
                c[0], c[1] = c[0][idx], c[1][idx]
        self.size -= 1
        if not self.size:
            self.cache = None

    def export(self, index: int, spare: bool = False) -> list:
        """An ordinary single-conversation cache holding conversation ``index`` as it is now (the
        inverse of ``join``); the conversation stays in the batch until ``leave``. ``spare``: hand
        over the buffer with its free rows, so a cache that decodes on does not copy itself to grow."""
        from mlx_lm.models.cache import make_prompt_cache

        if self._rows:
            raise RuntimeError("export between rounds: keep() the last check first")
        if not 0 <= index < self.size:
            raise IndexError(index)
        ad = self.adapter
        out = make_prompt_cache(ad.model)
        for one, c, mod in zip(out, self.cache, ad._linear):
            if mod is None:
                n = c.lens[index]
                cut = slice(None) if spare else slice(0, n)
                one.keys, one.values = c.keys[index : index + 1, :, cut, :], c.values[index : index + 1, :, cut, :]
                one.offset = n
            else:
                one[0], one[1] = c[0][index : index + 1], c[1][index : index + 1]
        mx.eval([c.state for c in out])
        return out

    def lengths(self) -> list[int]:
        """Tokens cached per conversation."""
        kv = next(c for c in self.cache if isinstance(c, RaggedKV))
        return list(kv.lens)

    def verify(self, tokens: mx.array, capture: bool = True):
        """One forward for all conversations: ``tokens`` [B, M] -> (logits [B, M, V], hidden)."""
        from . import fast_verify

        ad = self.adapter
        if int(tokens.shape[0]) != self.size:
            raise ValueError(f"{tokens.shape[0]} rows of tokens for {self.size} conversations")
        if self._rows:
            raise RuntimeError("keep() the last check before the next one")
        held, ad.cache = ad.cache, self.cache
        ad._set_recording(True)
        ad._set_flush(True)
        try:
            with fast_verify.armed(ad.fast_verify):
                out = ad._forward(tokens, capture=capture)
        finally:
            ad._set_recording(False)
            ad._set_flush(False)
            ad.cache = held
        self._rows = int(tokens.shape[1])
        return out

    def verify_tree(self, tokens: mx.array, parents: list, capture: bool = True):
        """One forward for all conversations, each checking its own draft tree: ``tokens`` [B, M],
        ``parents[b]`` the parent row of every row of conversation b (row 0, the pending token, has
        -1). Row i gives what a chain check of its root path would give. ``keep_paths`` follows."""
        from . import fast_verify

        ad = self.adapter
        if int(tokens.shape[0]) != self.size or len(parents) != self.size:
            raise ValueError(f"{tokens.shape[0]} rows of tokens for {self.size} conversations")
        if self._rows:
            raise RuntimeError("keep the last check before the next one")
        tree = BatchTree(parents)
        if tree.rows != int(tokens.shape[1]):
            raise ValueError("one parent per tree row")
        held, ad.cache = ad.cache, self.cache
        for c, mod in zip(self.cache, ad._linear):
            if mod is None:
                c.tree = tree
            else:
                mod._box.tree = tree
        ad._set_flush(True)
        try:
            with fast_verify.armed(ad.fast_verify):
                out = ad._forward(tokens, capture=capture)
        finally:
            ad._set_flush(False)
            ad.cache = held
            for c, mod in zip(self.cache, ad._linear):
                if mod is None:
                    c.tree = None
                else:
                    mod._box.tree = None
        self._rows, self._tree = tree.rows, tree
        return out

    def keep_paths(self, paths: list) -> None:
        """Keep conversation b's ``paths[b]`` of the last tree check: rows from row 0 downwards, each
        the child of the one before (empty keeps none)."""
        tree = self._tree
        if tree is None:
            raise RuntimeError("keep_paths() needs a preceding verify_tree()")
        if len(paths) != self.size:
            raise ValueError(f"{len(paths)} paths for {self.size} conversations")
        paths = [[int(r) for r in p] for p in paths]
        for b, p in enumerate(paths):
            tree.check_path(b, p)
        for c, mod in zip(self.cache, self.adapter._linear):
            if mod is None:
                c.keep_paths(paths)
            else:
                mod.commit_tree_batch(c, paths)
        self._rows, self._tree = 0, None

    def keep(self, rows: list[int]) -> None:
        """Keep the first ``rows[b]`` rows of the last check for conversation b (0 keeps none)."""
        if len(rows) != self.size:
            raise ValueError(f"{len(rows)} counts for {self.size} conversations")
        if not self._rows:
            raise RuntimeError("keep() needs a preceding verify()")
        rows = [int(r) for r in rows]
        if self._tree is not None:
            return self.keep_paths([list(range(r)) for r in rows])
        for c, mod in zip(self.cache, self.adapter._linear):
            if mod is None:
                c.keep(rows)
            else:
                rollback_rows(mod, c, rows)
        self._rows = 0
