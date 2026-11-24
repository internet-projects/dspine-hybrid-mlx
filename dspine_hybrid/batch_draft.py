"""One DSpine drafter pass for several conversations.

The drafter is small, but run once per conversation it is a quarter of a four-request round.
Here the block rows of all conversations go through the drafter together: the projections, the
MLPs, the code bridge and the head are plain row-wise work, and the attention sees each
conversation's own context through a mask (contexts are padded on the left to the longest).
Every conversation keeps its own context cache, so joining and leaving cost nothing.

The result for a conversation is what ``DSpineDraftModel.__call__`` and the refinement tail give
for it alone, up to float rounding of the padded attention.
"""

from __future__ import annotations

import mlx.core as mx


def _attention(attn, x, ctxs, rope, caches):
    """``DraftAttention.__call__`` for B conversations with their own caches and new context rows."""
    B, L, _ = x.shape
    q = attn.q_norm(attn.q_proj(x).reshape(B, L, attn.n_heads, attn.head_dim)).transpose(0, 2, 1, 3)
    pk = attn.k_norm(attn.k_proj(x).reshape(B, L, attn.n_kv_heads, attn.head_dim)).transpose(0, 2, 1, 3)
    pv = attn.v_proj(x).reshape(B, L, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)

    # the new context rows of all conversations through the projections at once
    new = [c for c in ctxs if c is not None and c.shape[1] > 0]
    if new:
        cat = mx.concatenate(new, axis=1)
        S = cat.shape[1]
        ck = attn.k_norm(attn.k_proj(cat).reshape(1, S, attn.n_kv_heads, attn.head_dim)).transpose(0, 2, 1, 3)
        cv = attn.v_proj(cat).reshape(1, S, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
    at = 0
    W = attn.window
    qs, pks, keys, values, lens, starts = [], [], [], [], [], []
    for b in range(B):
        cache, ctx = caches[b], ctxs[b]
        s = 0 if ctx is None else int(ctx.shape[1])
        offset = cache.offset
        qs.append(rope(q[b : b + 1], offset=offset + s))
        pks.append(rope(pk[b : b + 1], offset=offset + s))
        if s:
            k, v = cache.update_and_fetch(rope(ck[:, :, at : at + s], offset=offset), cv[:, :, at : at + s])
            at += s
        elif cache.keys is None:
            k = mx.zeros((1, attn.n_kv_heads, 0, attn.head_dim), dtype=pk.dtype)
            v = mx.zeros((1, attn.n_kv_heads, 0, attn.head_dim), dtype=pv.dtype)
        else:
            k, v = cache.keys[:, :, :offset, :], cache.values[:, :, :offset, :]
        n = int(k.shape[2])
        start = 0 if W is None or n + L <= W else max(0, n - (W - 1))   # keys older than the window are never seen
        keys.append(k[:, :, start:, :])
        values.append(v[:, :, start:, :])
        lens.append(n)
        starts.append(start)
    kept = [n - s for n, s in zip(lens, starts)]
    C = max(kept)
    plain = all(c == C for c in kept) and (W is None or all(n + L <= W for n in lens))
    if not plain:      # pad on the left, so every block sits at the end
        keys = [k if c == C else mx.concatenate([mx.zeros((1, k.shape[1], C - c, k.shape[3]), dtype=k.dtype), k], axis=2)
                for k, c in zip(keys, kept)]
        values = [v if c == C else mx.concatenate([mx.zeros((1, v.shape[1], C - c, v.shape[3]), dtype=v.dtype), v], axis=2)
                  for v, c in zip(values, kept)]
    queries = mx.concatenate(qs, axis=0)
    all_k = mx.concatenate([mx.concatenate(keys, axis=0), mx.concatenate(pks, axis=0)], axis=2)
    all_v = mx.concatenate([mx.concatenate(values, axis=0), pv], axis=2)
    if plain:
        mask = "causal"
    else:
        col = mx.arange(C)[None, None, :]
        pad = mx.array([C - c for c in kept])[:, None, None]
        ok = col >= pad                                                   # [B, 1, C]
        ok = mx.broadcast_to(ok, (B, L, C))
        if W is not None:
            kpos = mx.array(starts)[:, None, None] + col - pad
            qpos = mx.array(lens)[:, None, None] + mx.arange(L)[None, :, None]
            ok = ok & (kpos > qpos - W)
        gap = mx.arange(L)[:, None] - mx.arange(L)[None, :]               # query row - key row
        block = mx.broadcast_to(((gap >= 0) if W is None else (gap >= 0) & (gap < W))[None], (B, L, L))
        mask = mx.concatenate([ok, block], axis=-1)[:, None]              # [B, 1, L, C + L]
    out = mx.fast.scaled_dot_product_attention(queries, all_k, all_v, scale=attn.scale, mask=mask)
    return attn.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


_armed = [False]


def draft_forward(draft, hiddens: list, block_ids: mx.array, caches: list):
    """``DSpineDraftModel.__call__`` for B conversations. ``hiddens[b]``: the target features
    [1, rows, H] conversation b's cache has not seen (or None), ``caches[b]`` its own cache list.
    Returns (normed block hidden [B, T, H], stash with B * T rows)."""
    if getattr(draft, "fast_kernels", False) and not _armed[0]:
        from . import fast_verify

        _armed[0] = True
        try:
            with fast_verify.armed():      # the drafter's quantized projections on the verify kernels
                return draft_forward(draft, hiddens, block_ids, caches)
        finally:
            _armed[0] = False
    B, T = block_ids.shape
    W = getattr(draft, "window", draft.config.sliding_window)
    hiddens = list(hiddens)
    for b, h in enumerate(hiddens):
        if W is not None and h is not None and h.shape[1] > W and all(c.offset == 0 for c in caches[b]):
            hiddens[b] = h[:, -W:]
    ctxs = [None] * B
    have = [b for b, h in enumerate(hiddens) if h is not None and h.shape[1] > 0]
    if have:
        cat = draft.hidden_norm(draft.fc(mx.concatenate([hiddens[b] for b in have], axis=1)))
        at = 0
        for b in have:
            s = int(hiddens[b].shape[1])
            ctxs[b] = cat[:, at : at + s]
            at += s
    x = draft._embed_tokens(block_ids)
    residual = x
    x_in = draft.layers[0].input_layernorm(x)
    last = len(draft.layers) - 1
    stash_out = None
    for i, layer in enumerate(draft.layers):
        h = residual + _attention(layer.self_attn, x_in, ctxs, draft.rope, [c[i] for c in caches])
        out = h + layer.mlp(layer.post_attention_layernorm(h))
        next_norm = draft.layers[i + 1].input_layernorm if i < last else draft.norm
        stash = {} if i == last else None
        x_in, residual = draft.selected_prefix_bridge(out, None, block_ids, i, next_norm, stash)
        if stash is not None:
            stash_out = stash
    return x_in, stash_out


def stash_of(stash: dict, b: int, T: int) -> dict:
    """Conversation b's rows of a batched stash."""
    return {k: v[b * T : (b + 1) * T] for k, v in stash.items()}


def score_tables(tail, out: mx.array, stash: dict, anchors: list):
    """``adaptive.score_table`` for B conversations in one pass: candidates [B, n, K], score
    [B, n, K senders, K] and write scale [B, n]."""
    from dspine_mlx.propose import topk_ids

    B, T, H = out.shape
    n, K, R = T - 1, tail.k, tail.rank
    rows = out[:, 1:].reshape(B * n, H)
    raw = stash["raw"].reshape(B, T, H)[:, 1:].reshape(B * n, H).astype(mx.float32)
    rms = stash["rms"].reshape(B, T)[:, 1:].reshape(B * n).astype(mx.float32)
    q = stash["packed"].reshape(B, T, -1)[:, 1:, :R].reshape(B * n, R).astype(mx.float32)
    cand = topk_ids(tail.head.matmul(rows), K)                               # [B * n, K]
    head_rows = tail.head.rows(cand).astype(mx.float32)
    base = mx.sum((raw[:, None, :] * tail.norm_w) * head_rows, axis=-1)
    cand_b = cand.reshape(B, n, K)
    first = mx.broadcast_to(mx.array(anchors, dtype=mx.int32)[:, None, None], (B, 1, K)).astype(cand.dtype)
    sid = (mx.concatenate([first, cand_b[:, :-1]], axis=1) if n > 1 else first).reshape(B * n, K)
    tsv_sid = tail.tsv[sid].astype(mx.float32)
    m = mx.sigmoid(q[:, None, :] + tsv_sid[..., :R] + tail.gate_b) * tsv_sid[..., R:]
    s = mx.matmul(m, tail.tg[cand].astype(mx.float32).swapaxes(1, 2))
    score = base[:, None, :] + rms[:, None, None] * s
    return cand_b, score.reshape(B, n, K, K), rms.reshape(B, n)
