"""fast_verify: optional, inert unless armed, and harmless where the kernels cannot run."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from dspine_hybrid import fast_verify
from dspine_hybrid.hybrid_target import HybridTargetAdapter
from tiny import TARGET_IDS, VOCAB, FakeTokenizer, make_target


def _quantized_adapter():
    m = make_target(0)
    nn.quantize(m, group_size=64, bits=4)
    mx.eval(m.parameters())
    return HybridTargetAdapter(m, FakeTokenizer(), TARGET_IDS)


def test_off_by_default_and_noop_without_install():
    ad = _quantized_adapter()
    assert ad.fast_verify is False
    with fast_verify.armed(False) as on:
        assert on is False


def test_armed_verify_matches_stock_within_rounding():
    pytest.importorskip("omlx.patches.qwen35_verify_qmm")
    ad = _quantized_adapter()
    prompt = mx.random.randint(0, VOCAB, (1, 15))
    block = mx.random.randint(0, VOCAB, (1, 9))
    ad.reset()
    ad.prefill(prompt)
    ref, href = ad.verify(block)
    mx.eval(ref, href)
    assert ad.enable_fast_verify(True) and fast_verify.available()
    from omlx.patches import qwen35_verify_qmm as q

    assert not q._is_armed()                       # armed only inside a verify
    ad.reset()
    ad.prefill(prompt)
    got, hgot = ad.verify(block)
    mx.eval(got, hgot)
    assert not q._is_armed()
    # tiny layers are below the kernels' size gates on any device, and without Metal a routed
    # call falls back to stock: either way the result must agree with the stock pass
    spread = float((mx.max(ref) - mx.min(ref)).item())
    assert float(mx.max(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32))).item()) <= 0.05 * spread
    ad.trim(4)
    ad.enable_fast_verify(False)
    assert ad.fast_verify is False


def test_routes_reports_shapes():
    pytest.importorskip("omlx.patches.qwen35_verify_qmm")
    fast_verify.install()
    lin = nn.QuantizedLinear(5120, 17408, bias=False, group_size=64, bits=4)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = lin

    assert fast_verify.routes(M(), 1, mx.bfloat16) == {"stock": 1}
    assert fast_verify.routes(M(), 12, mx.bfloat16).get("stock", 0) == 0


def test_flush_changes_nothing():
    from test_hybrid_tiny import _states

    m = make_target(1)
    ad = HybridTargetAdapter(m, FakeTokenizer(), TARGET_IDS)
    prompt = mx.random.randint(0, VOCAB, (1, 15))
    block = mx.random.randint(0, VOCAB, (1, 9))
    ad.reset()
    ad.prefill(prompt)
    ref, href = ad.verify(block)
    mx.eval(ref, href)
    ad.trim(3)
    sref = _states(ad.cache)
    ad.enable_flush(2)
    ad.reset()
    ad.prefill(prompt)
    got, hgot = ad.verify(block)
    assert mx.array_equal(ref, got).item() and mx.array_equal(href, hgot).item()
    ad.trim(3)
    for a, b in zip(sref, _states(ad.cache)):
        assert mx.array_equal(a[1], b[1]).item() and mx.array_equal(a[2], b[2]).item()
    lt, _ = ad.verify_tree(block[0].tolist(), [-1, 0, 0, 1, 1, 3, 0, 6, 7])
    mx.eval(lt)
    ad.commit_tree([0, 1, 3])
    assert ad._flush_box.flush == 0


def test_fast_head_matches_stock_head():
    pytest.importorskip("omlx.patches.qwen35_verify_qmm")
    ad = _quantized_adapter()
    x = mx.random.normal((7, 128)).astype(mx.bfloat16)
    ref = ad.head.matmul(x)
    assert ad.enable_fast_verify(True)
    got = ad.head.matmul(x)
    assert got.shape == ref.shape
    assert float(mx.max(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32))).item()) <= 1e-2 * float(
        mx.max(mx.abs(ref.astype(mx.float32))).item())
    ad.enable_fast_verify(False)
    assert mx.array_equal(ad.head.matmul(x), ref).item()


def test_long_attn_tree_bits():
    from dspine_hybrid import long_attn
    from dspine_hybrid.hybrid_target import TreeSpec

    tree = TreeSpec([-1, 0, 0, 1, 3])
    bits = [int(b) for b in long_attn.tree_bits(tree.ancestors).tolist()]
    assert bits == [0b00001, 0b00011, 0b00101, 0b01011, 0b11011]
    assert long_attn.tree_bits(mx.ones((33, 33), dtype=mx.bool_)) is None


def test_long_attn_is_off_without_metal_or_by_env(monkeypatch):
    from types import SimpleNamespace

    from dspine_hybrid import long_attn

    attn = SimpleNamespace(num_attention_heads=24, num_key_value_heads=4, head_dim=256)
    monkeypatch.setenv("DSPINE_LONG_ATTN", "0")
    assert long_attn.config(attn) is None
    monkeypatch.delenv("DSPINE_LONG_ATTN")
    if not mx.metal.is_available():
        assert long_attn.config(attn) is None
    else:
        cfg = long_attn.config(attn)
        w = cfg.window(24, 4, 256)
        assert (w.min_q, w.max_q, w.min_kv) == long_attn.DEFAULT_WINDOW
    with long_attn.active(None) as on:
        assert on is False


@pytest.mark.skipif(not (mx.metal.is_available() and mx.default_device() == mx.gpu), reason="Metal only")
@pytest.mark.parametrize("rows", [6, 12])
def test_long_attn_kernel_matches_stock_on_metal(rows):
    from dspine_hybrid import long_attn
    from dspine_hybrid import multirow_attn as mr
    from dspine_hybrid.hybrid_target import TreeSpec

    H, KV, D, L = 24, 4, 256, 16384
    mx.random.seed(0)
    q = mx.random.normal((1, H, rows, D)).astype(mx.bfloat16)
    k = mx.random.normal((1, KV, L, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, KV, L, D)).astype(mx.bfloat16)
    f32 = [x.astype(mx.float32) for x in (q, k, v)]
    sdpa = mr._ORIG_SDPA

    def err(x, ref):
        return float(mx.abs(x.astype(mx.float32) - ref).max())

    ref = sdpa(*f32, scale=D ** -0.5, mask="causal")
    stock = err(sdpa(q, k, v, scale=D ** -0.5, mask="causal"), ref)
    assert err(mr.multirow_sdpa(q, k, v, scale=D ** -0.5, causal=True), ref) <= 2 * stock + 2e-3

    tree = TreeSpec([-1] + [max(0, i - 2) for i in range(1, rows)])
    mask = tree.mask(L - rows)
    ref = sdpa(*f32, scale=D ** -0.5, mask=mask)
    stock = err(sdpa(q, k, v, scale=D ** -0.5, mask=mask), ref)
    cfg = mr.MultirowConfig((mr.Window(H, KV, D, 2, 15, 1024),))
    out = long_attn.tree_attention(cfg, q, k, v, D ** -0.5, long_attn.tree_bits(tree.ancestors))
    assert out is not None and err(out, ref) <= 2 * stock + 2e-3


def test_a8_prefill_routes_wide_chunks_only(monkeypatch):
    """With stand-in kernels (exact products) the routed prefill equals stock, checks are untouched."""
    import dataclasses

    from dspine_hybrid import a8_prefill

    @dataclasses.dataclass(frozen=True)
    class Plan:
        act_mode: int = 0
        layout: int = 0

    class FakeOq:
        seen = []

        @staticmethod
        def classify_linear(m):
            return Plan()

        @staticmethod
        def stage_a(x, mode, layout):
            FakeOq.seen.append((int(x.shape[-2]), mode))
            return x

        @staticmethod
        def apply_plan(m, stage, plan):
            assert plan.act_mode == a8_prefill.G64
            return m(stage)

    target = HybridTargetAdapter(make_target(0), FakeTokenizer(), TARGET_IDS)
    target.enable_flush(8)                       # the layer hooks that hid the model in ds-46
    ids = mx.array([[3 + (i * 7) % 300 for i in range(200)]])
    ref, _ = target.prefill(ids, capture=False)
    ref_v, _ = target.verify(mx.array([[5, 6, 7]]))
    mx.eval(ref, ref_v)

    monkeypatch.setitem(a8_prefill._OQ, "mod", FakeOq)
    monkeypatch.setitem(a8_prefill._OQ, "tried", True)
    assert target.enable_a8_prefill(True, min_tokens=128, parts=("mlp", "gdn", "attn"))
    a8_prefill.stats.update(int8=0, stock=0)
    target.reset()
    out, _ = target.prefill(ids, capture=False)
    out_v, _ = target.verify(mx.array([[5, 6, 7]]))
    mx.eval(out, out_v)
    assert a8_prefill.stats["int8"] > 0 and all(rows >= 128 for rows, _ in FakeOq.seen)
    assert {mode for _, mode in FakeOq.seen} == {a8_prefill.G64}
    assert mx.array_equal(mx.argmax(out[0, -1]), mx.argmax(ref[0, -1]))
    assert float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max()) < 1e-2
    assert float(mx.abs(out_v.astype(mx.float32) - ref_v.astype(mx.float32)).max()) < 1e-2

    for parts in (("mlp",), ("gdn",), ("attn",)):          # each part routes on its own
        assert target.enable_a8_prefill(True, min_tokens=128, parts=parts)
        a8_prefill.stats.update(int8=0, stock=0)
        target.reset()
        mx.eval(target.prefill(ids, capture=False)[0])
        assert a8_prefill.stats["int8"] > 0, parts

    n = a8_prefill.stats["int8"]
    assert target.enable_a8_prefill(False) is False
    target.reset()
    mx.eval(target.prefill(ids, capture=False)[0])
    assert a8_prefill.stats["int8"] == n


def test_a8_prefill_without_kernels_changes_nothing():
    from dspine_hybrid import a8_prefill

    target = HybridTargetAdapter(make_target(0), FakeTokenizer(), TARGET_IDS)
    if a8_prefill.kernels() is None:
        assert target.enable_a8_prefill(True) is False and target.a8_prefill is None


def test_gdn_prefill_routes_wide_chunks_only(monkeypatch):
    """With a stand-in kernel (the reference recurrence) the routed prefill equals stock; checks keep the stock kernel."""
    from mlx_lm.models.gated_delta import gated_delta_ops

    from dspine_hybrid import gdn_prefill

    class FakeKernels:
        seen = []

        @staticmethod
        def gated_delta_pipelined_supported(q, k, v):
            return True

        @staticmethod
        def gated_delta_pipelined(q, k, v, g, beta, state=None):
            FakeKernels.seen.append(int(q.shape[1]))
            if state is None:
                state = mx.zeros((q.shape[0], v.shape[2], v.shape[3], q.shape[3]), dtype=mx.float32)
            return gated_delta_ops(q, k, v, g, beta, state, None)

    target = HybridTargetAdapter(make_target(0), FakeTokenizer(), TARGET_IDS)
    target.enable_flush(8)
    ids = mx.array([[3 + (i * 7) % 300 for i in range(200)]])
    ref, _ = target.prefill(ids, capture=False)
    ref_v, _ = target.verify(mx.array([[5, 6, 7]]))
    mx.eval(ref, ref_v)

    monkeypatch.setitem(gdn_prefill._K, "mod", FakeKernels)
    monkeypatch.setitem(gdn_prefill._K, "tried", True)
    assert target.enable_gdn_prefill(True, min_tokens=64)
    gdn_prefill.stats.update(pipelined=0, stock=0)
    target.reset()
    out, _ = target.prefill(ids, capture=False)
    out_v, _ = target.verify(mx.array([[5, 6, 7]]))
    mx.eval(out, out_v)
    assert FakeKernels.seen and all(rows >= 64 for rows in FakeKernels.seen)
    assert gdn_prefill.stats["pipelined"] == len(FakeKernels.seen)
    # on Metal the stock route is mlx-lm's kernel and the stand-in is the reference recurrence: bf16 rounding apart
    assert mx.allclose(out.astype(mx.float32), ref.astype(mx.float32), atol=5e-2)
    assert mx.allclose(out_v.astype(mx.float32), ref_v.astype(mx.float32), atol=5e-2)

    assert target.enable_gdn_prefill(False) is False and target.gdn_prefill is None
    n = len(FakeKernels.seen)
    target.reset()
    mx.eval(target.prefill(ids, capture=False)[0])
    assert len(FakeKernels.seen) == n


def test_gdn_prefill_without_kernels_changes_nothing():
    from dspine_hybrid import gdn_prefill

    target = HybridTargetAdapter(make_target(0), FakeTokenizer(), TARGET_IDS)
    if gdn_prefill.kernels() is None:
        assert target.enable_gdn_prefill(True) is False and target.gdn_prefill is None
