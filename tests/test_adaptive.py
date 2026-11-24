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
    for alpha, conf in ((0.5, "vocab"), (2.0, "vocab"), (1.0, "calibrated")):
        g = AdaptiveDSpineGenerator(ad, d, cost=cost, alpha=alpha, confidence=conf)
        out, st = g.generate(prompt, max_tokens=60)
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


def test_calibration_map():
    import json
    import math
    import os

    from dspine_hybrid import calibration as cal

    raw = mx.array([0.05, 0.5, 0.9, 0.999, 1.0] * 3)
    p = cal.apply(raw).tolist()
    for d, (r, q) in enumerate(zip(raw.tolist(), p)):
        r = min(max(r, 1e-6), 1 - 1e-6)
        want = 1 / (1 + math.exp(-(cal.GREEDY["scale"] * math.log(r / (1 - r)) + cal.GREEDY["offset"][d])))
        assert abs(q - want) < 1e-5
    # monotonic within a depth, deeper rows a little less trusted
    a = cal.apply(mx.array([0.3, 0.6, 0.95])).tolist()
    assert a[0] < 0.6 and a[2] < 0.95
    up = os.environ.get("DSPINE_TENCENT", "/tmp/claude-0/tc/DSpine") + "/sglang_dspine/verification/calibration.json"
    if os.path.exists(up):
        assert cal.load(up) == {**cal.GREEDY, "eps": 1e-6}


def test_calibrated_feature_is_tencent_greedy_confidence():
    """raw confidence = max softmax(score / write scale) at the chain's predecessor."""
    from dspine_hybrid import adaptive as A

    m = make_target(2)
    ad = HybridTargetAdapter(m, FakeTokenizer(), TARGET_IDS)
    d, _, _ = make_draft(12)
    d.bind(m)
    g = DSpineGenerator(ad, d)
    ad.reset()
    _, hidden = ad.prefill(mx.array([list(range(5, 25))]))
    block = mx.array([[7] + [d.mask_token_id] * (g.block_size - 1)])
    out, stash = d(hidden, block, d.make_cache())
    seen = {}
    real = A._chain_with_conf

    def spy(idx, cand, score):
        seen["score"], seen["idx"] = score, idx
        return real(idx, cand, score)

    A._chain_with_conf = spy
    try:
        drafts, _, raw_vocab = A.greedy_tail_with_conf(g.tail, out[0, 1:], stash, 7, confidence="vocab")
        drafts2, _, cal_p = A.greedy_tail_with_conf(g.tail, out[0, 1:], stash, 7, confidence="calibrated")
    finally:
        A._chain_with_conf = real
    assert drafts.tolist() == drafts2.tolist()
    sc, idx = seen["score"], seen["idx"]
    prev, raws = 0, []
    for i in range(sc.shape[0]):
        row = mx.softmax(sc[i, prev], axis=-1)
        raws.append(mx.max(row).item())
        prev = int(idx[i, prev].item())
    from dspine_hybrid import calibration as cal

    want = cal.apply(mx.array(raws)).tolist()
    assert max(abs(a - b) for a, b in zip(want, cal_p.tolist())) < 1e-5


def test_sample_chain_with_conf_matches_upstream():
    from dspine_mlx.propose import serial_sample_chain

    from dspine_hybrid.adaptive import sample_chain_with_conf

    mx.random.seed(3)
    table = mx.random.normal((7, 16, 16)) * 3
    cand = mx.argsort(mx.random.uniform(shape=(7, 500)), axis=-1)[:, :16].astype(mx.int32)  # distinct ids
    u = mx.random.uniform(shape=(7,))
    kw = dict(inv_temp=1 / 0.7, top_k=8, top_p=0.9, min_p=0.0, uniforms=u)
    t1, q1 = serial_sample_chain(table, cand, **kw)
    t2, q2, conf = sample_chain_with_conf(table, cand, **kw)
    assert t1.tolist() == t2.tolist() and bool(mx.array_equal(q1, q2).item())
    # conf = max of the unfiltered temperature softmax at the sampled predecessor
    sender = 0
    for i in range(7):
        p = mx.softmax(table[i, sender] / 0.7)
        assert abs(mx.max(p).item() - conf[i].item()) < 1e-6
        sender = cand[i].tolist().index(t2[i].item()) if i < 6 else 0


def test_adaptive_sampling_keeps_target_distribution(monkeypatch):
    """With random verify lengths, the 2nd token still follows the target's filtered distribution."""
    import random

    from dspine_hybrid import adaptive as A
    from dspine_mlx.propose import filter_target_probs

    m = make_target(4)
    ad = HybridTargetAdapter(m, FakeTokenizer(), TARGET_IDS)
    d, _, _ = make_draft(14)
    d.bind(m)
    g = A.AdaptiveDSpineGenerator(ad, d, cost=[0.0] + [1.0] * 17, block_size=16, confidence="calibrated")
    rng = random.Random(0)
    monkeypatch.setattr(A, "choose_len", lambda conf, cost, alpha, max_len: rng.randint(1, max_len))
    prompt = list(range(3, 15))
    T, TOPK, N = 1.0, 6, 900

    def filt(logits):
        p = mx.softmax(logits.astype(mx.float32) / T, axis=-1)
        return filter_target_probs(p[None], TOPK, 1.0, 0.0)[0]

    ad.reset()
    l0, _ = ad.prefill(mx.array([prompt]), capture=False)
    p0 = filt(l0[0, -1])
    exact = mx.zeros((VOCAB,))
    for a in [i for i, v in enumerate(p0.tolist()) if v > 0]:
        ad.reset()
        la, _ = ad.prefill(mx.array([prompt + [a]]), capture=False)
        exact = exact + p0[a] * filt(la[0, -1])
    counts = [0] * VOCAB
    mx.random.seed(1)
    for _ in range(N):
        out, _ = g.generate(prompt, max_tokens=5, temperature=T, top_k=TOPK)  # first round verifies 1..4
        counts[out[1]] += 1
    ex = exact.tolist()
    support = [i for i, v in enumerate(ex) if v > 1e-9]
    assert sum(counts[i] for i in support) == N
    chi2 = sum((counts[i] - N * ex[i]) ** 2 / (N * ex[i]) for i in support if N * ex[i] >= 1)
    df = len(support) - 1
    assert chi2 < df + 5 * (2 * df) ** 0.5, (chi2, df)
    assert len(g.len_hist) > 3
