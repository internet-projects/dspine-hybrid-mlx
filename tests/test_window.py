import mlx.core as mx
import pytest

from dspine_hybrid import draft as dmod
from dspine_hybrid.draft import parse_window
from dspine_mlx.model import DraftAttention
from tiny import HIDDEN, TARGET_IDS, draft_config, make_draft, make_target


def _eq(a, b):
    return a.shape == b.shape and bool(mx.array_equal(a, b).item())


def _run(draft, ctx_len, rounds=3, seed=0):
    """Prefill-like first block with ctx_len context, then block-only rounds."""
    mx.random.seed(seed)
    caches = draft.make_cache()
    outs = []
    hid = (mx.random.normal((1, ctx_len, HIDDEN * len(TARGET_IDS))) ).astype(mx.bfloat16)
    for r in range(rounds):
        block = mx.random.randint(0, 400, (1, 16))
        out, _ = draft(hid, block, caches)
        outs.append(out)
        hid = mx.random.normal((1, 5, HIDDEN * len(TARGET_IDS))).astype(mx.bfloat16)
    return outs


def test_window_below_limit_is_upstream_bit_identical():
    target = make_target(0)
    d_win, _, w = make_draft(3, window=64)
    assert w == 64
    d_win.bind(target)
    a = _run(d_win, 30)  # 30 + 3*16 rounds stays <= 64 keys? (ctx grows by 5)
    for layer in d_win.layers:
        layer.self_attn.__class__ = DraftAttention
    b = _run(d_win, 30)
    for x, y in zip(a, b):
        assert _eq(x, y)


def test_window_mask_semantics():
    """Each query at absolute p must see exactly keys p-W+1..p."""
    W = 8
    target = make_target(0)
    d, _, _ = make_draft(4, window=W)
    d.bind(target)
    seen = []
    real = mx.fast.scaled_dot_product_attention

    def spy(q, k, v, scale, mask):
        out = real(q, k, v, scale=scale, mask=mask)
        if isinstance(mask, str):
            return out
        L, Ks = q.shape[2], k.shape[2]
        assert Ks == L + W - 1
        kk = mx.repeat(k, q.shape[1] // k.shape[1], axis=1).astype(mx.float32)
        vv = mx.repeat(v, q.shape[1] // v.shape[1], axis=1).astype(mx.float32)
        qq = q.astype(mx.float32)
        rows = []
        for i in range(L):
            s = (qq[:, :, i : i + 1] @ kk[:, :, i : i + W].transpose(0, 1, 3, 2)) * scale
            rows.append(mx.softmax(s, axis=-1) @ vv[:, :, i : i + W])
        ref = mx.concatenate(rows, axis=2)
        seen.append(float(mx.max(mx.abs(ref - out.astype(mx.float32))).item()))
        return out

    mx.fast.scaled_dot_product_attention = spy
    try:
        _run(d, 40, rounds=2)
    finally:
        mx.fast.scaled_dot_product_attention = real
    assert seen and max(seen) < 2e-2  # bf16 inputs, fp32 reference


def test_parse_window_rejects_partial():
    raw = draft_config(2, window=None)
    assert parse_window(raw) is None
    raw = draft_config(2, window=8192)
    assert parse_window(raw) == 8192
    raw["layer_types"] = ["sliding_attention", "full_attention"]
    with pytest.raises(ValueError):
        parse_window(raw)
    raw = draft_config(2, window=4096)
    with pytest.raises(ValueError):
        parse_window(raw)


def test_first_call_context_crop_matches_full_context():
    """A fresh cache gets only the last W context features; drafts match the full-context run."""
    W = 16
    target = make_target(0)
    d, _, _ = make_draft(5, window=W)
    d.bind(target)
    assert d.window == W
    a = _run(d, 50, rounds=4, seed=3)
    d.window = None  # layers keep the window; only the crop is switched off
    b = _run(d, 50, rounds=4, seed=3)
    d.window = W
    for x, y in zip(a, b):
        assert x.shape == y.shape
        diff = mx.max(mx.abs(x.astype(mx.float32) - y.astype(mx.float32))).item()
        scale = mx.max(mx.abs(y.astype(mx.float32))).item()
        assert diff <= 2e-2 * max(scale, 1.0), (diff, scale)


def test_crop_skipped_when_cache_not_empty_or_short():
    W = 16
    target = make_target(0)
    d, _, _ = make_draft(6, window=W)
    d.bind(target)
    seen = []
    orig = dmod.DSpineDraftModel.__call__

    def spy(self, th, block, caches=None):
        seen.append(None if th is None else th.shape[1])
        return orig(self, th, block, caches)

    dmod.DSpineDraftModel.__call__ = spy
    try:
        _run(d, 50, rounds=2)
        _run(d, 10, rounds=1)
    finally:
        dmod.DSpineDraftModel.__call__ = orig
    assert seen == [W, 5, 10]
