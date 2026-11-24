import mlx.core as mx
import pytest
from dspine_mlx.propose import filter_target_probs as upstream

from dspine_hybrid.fast_sampling import filter_target_probs as fast


def _probs(seed, rows=6, V=4096, scale=4.0):
    mx.random.seed(seed)
    return mx.softmax(mx.random.normal((rows, V)) * scale, axis=-1)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("top_k,top_p,min_p", [
    (20, 0.8, 0.0), (20, 1.0, 0.0), (1, 1.0, 0.0), (50, 0.95, 0.05), (0, 0.8, 0.0),
    (0, 0.95, 0.02), (0, 1.0, 0.0), (0, 1.0, 0.1), (4096, 0.9, 0.0),
])
def test_same_distribution_as_upstream(seed, top_k, top_p, min_p):
    for scale in (1.0, 4.0, 12.0):
        p = _probs(seed, scale=scale)
        a = upstream(p, 4096 if top_k <= 0 else top_k, top_p, min_p)
        b = fast(p, top_k, top_p, min_p)
        assert a.shape == b.shape
        if top_p < 1.0 or 0 < top_k < 4096:
            assert bool(mx.array_equal(a > 0, b > 0).item()), (scale, top_k, top_p, min_p)
        else:
            # top_p = 1: upstream's prefix sums pass 1.0 by rounding and drop a few
            # near-zero ids; we keep them, which is the exact filter
            assert bool(mx.all((a > 0) <= (b > 0)).item())
            keep = p >= min_p * mx.max(p, axis=-1, keepdims=True)
            ideal = mx.where(keep, p, 0.0)
            a = ideal / mx.sum(ideal, axis=-1, keepdims=True)
        assert mx.max(mx.abs(a - b)).item() <= 1e-6


def test_flat_top_p_falls_back_to_full_rank():
    p = mx.softmax(mx.zeros((2, 4096)), axis=-1)  # 1024 ids hold only 25% of the mass
    a = upstream(p, 4096, 0.9, 0.0)
    b = fast(p, 0, 0.9, 0.0)
    assert bool(mx.array_equal(a, b).item())


def test_install_patches_sampler():
    from dspine_hybrid import fast_sampling
    import dspine_mlx.sampler as s
    import dspine_mlx.generate as g

    fast_sampling.install()
    assert s.filter_target_probs is fast and g.filter_target_probs is fast


def test_sampled_generation_same_tokens_with_fast_filter():
    """Same seed, same sampled tokens with the upstream and the small-support filter."""
    import dspine_mlx.generate as g
    import dspine_mlx.sampler as s
    from dspine_mlx.generate import DSpineGenerator
    from dspine_hybrid.hybrid_target import HybridTargetAdapter
    from tiny import TARGET_IDS, VOCAB, FakeTokenizer, make_draft, make_target

    outs = []
    for f in (upstream, fast):
        s.filter_target_probs = g.filter_target_probs = f
        model = make_target(4)
        adapter = HybridTargetAdapter(model, FakeTokenizer(), TARGET_IDS)
        draft, _, _ = make_draft(14)
        draft.bind(model)
        gen = DSpineGenerator(adapter, draft)
        mx.random.seed(9)
        prompt = list(range(3, 30))
        toks, _ = gen.generate(prompt, max_tokens=60, temperature=0.7, top_k=20, top_p=0.8)
        outs.append(toks)
    s.filter_target_probs = g.filter_target_probs = fast
    assert outs[0] == outs[1]
