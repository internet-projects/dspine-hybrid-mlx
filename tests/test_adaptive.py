"""Adaptive verify length: greedy output must equal fixed-block DSpine and plain greedy."""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from dspine_hybrid.adaptive import AdaptiveDSpineGenerator, choose_len, greedy_tail_with_conf, measure_cost
from dspine_hybrid.hybrid_target import HybridTargetAdapter
from dspine_mlx.generate import DSpineGenerator, baseline_generate
from test_hybrid_tiny import _Fp32HiddenAdapter, _divergence, _noise_floor
from tiny import TARGET_IDS, VOCAB, FakeTokenizer, make_draft, make_target


def test_tail_matches_upstream():
    from dspine_hybrid.hybrid_target import HybridTargetAdapter

    m = make_target(0)
    ad = HybridTargetAdapter(m, FakeTokenizer(), TARGET_IDS)
    d, _, _ = make_draft(10)
    d.bind(m)
    g = DSpineGenerator(ad, d)
    ad.reset()
    lg, h = ad.prefill(mx.array([list(range(3, 15))]))
    anchor = int(mx.argmax(lg[0, -1]).item())
    out, stash = d(h, mx.array([[anchor] + [d.mask_token_id] * 15], dtype=mx.int32), d.make_cache())
    a, ca = g.tail.greedy(out[0, 1:], stash, anchor)
    b, cb, conf = greedy_tail_with_conf(g.tail, out[0, 1:], stash, anchor)
    assert mx.array_equal(a, b).item() and mx.array_equal(ca, cb).item()
    c = conf.tolist()
    assert all(0.0 <= x <= 1.0 for x in c)


def test_choose_len():
    cost = [0.0] + [0.031, 0.032, 0.035, 0.037, 0.041, 0.054, 0.06, 0.064, 0.07, 0.08, 0.085, 0.09, 0.09,
                    0.09, 0.09, 0.088, 0.09]
    assert choose_len([0.999] * 15, cost, 1.0, 15) == 15
    assert choose_len([0.99] * 3 + [0.05] * 12, cost, 1.0, 15) == 3
    assert choose_len([0.9] * 15, cost, 1.0, 4) <= 4


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_adaptive_greedy_is_exact(seed):
    m = make_target(seed)
    m.load_weights([(k, v.astype(mx.bfloat16) if "embed_tokens" in k else v.astype(mx.float32))
                    for k, v in tree_flatten(m.parameters())])
    ad = _Fp32HiddenAdapter(m, FakeTokenizer(), TARGET_IDS)
    d, _, _ = make_draft(seed + 10)
    d.bind(m)
    mx.random.seed(seed)
    prompt = mx.random.randint(0, VOCAB - 4, (1, 12)).tolist()[0]
    fixed, _ = DSpineGenerator(ad, d).generate(prompt, max_tokens=60)
    base, _ = baseline_generate(ad, prompt, max_tokens=60)
    cost = measure_cost(ad, prompt, max_T=16, reps=1)
    noise = _noise_floor(ad, prompt, base)
    for alpha in (0.5, 2.0):
        out, st = AdaptiveDSpineGenerator(ad, d, cost=cost, alpha=alpha).generate(prompt, max_tokens=60)
        assert st.rounds > 5
        if mx.default_device() == mx.cpu:
            assert out == fixed == base
            continue
        # Metal: block and single-token passes round differently, so only near-ties may flip
        # (same criterion as test_dspine_greedy_fp32_target).
        dv = _divergence(ad, prompt, out, base, seed)
        if dv is not None:
            margin, spread, msg = dv
            assert margin <= max(4 * noise, 1e-4 * spread), "adaptive divergence above noise: " + msg


def test_adaptive_sampling_uses_fixed_block():
    m = make_target(4)
    ad = HybridTargetAdapter(m, FakeTokenizer(), TARGET_IDS)
    d, _, _ = make_draft(14)
    d.bind(m)
    cost = [0.0] + [0.03 + 0.001 * T for T in range(1, 18)]
    g = AdaptiveDSpineGenerator(ad, d, cost=cost, block_size=16, fixed_block_size=5)
    prompt = list(range(3, 15))
    out, st = g.generate(prompt, max_tokens=30, temperature=0.8)
    assert g._sampler_gen is not None and g._sampler_gen.block_size == 5
    assert len(out) == 30 and st.rounds > 0
    out2, _ = g.generate(prompt, max_tokens=20)  # greedy stays adaptive
    assert sum(g.len_hist.values()) > 0 and len(out2) == 20
