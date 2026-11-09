import mlx.core as mx
import pytest
from mlx_lm.models import qwen3_5

from dspine_hybrid.hybrid_target import HybridTargetAdapter, RecordingGatedDeltaNet
from dspine_mlx.generate import DSpineGenerator, baseline_generate
from tiny import TARGET_IDS, VOCAB, FakeTokenizer, make_draft, make_target


def _eq(a, b):
    return a.shape == b.shape and bool(mx.array_equal(a, b).item())


def _states(cache):
    out = []
    from mlx_lm.models.cache import ArraysCache

    for c in cache:
        if isinstance(c, ArraysCache):
            out.append(("ssm", c[0], c[1]))
        else:
            out.append(("kv", c.keys[:, :, : c.offset], c.values[:, :, : c.offset]))
    return out


def test_recording_layer_is_bit_identical():
    ref = make_target(0)
    rec = make_target(0)
    adapter = HybridTargetAdapter(rec, FakeTokenizer(), TARGET_IDS)
    toks = mx.random.randint(0, VOCAB, (1, 23))
    from mlx_lm.models.cache import make_prompt_cache

    c1 = make_prompt_cache(ref)
    l1 = ref(toks, c1)
    adapter.reset()
    l2, _ = adapter.prefill(toks)
    assert _eq(l1, l2)
    nxt = mx.random.randint(0, VOCAB, (1, 9))
    l1 = ref(nxt, c1)
    l2, _ = adapter.verify(nxt)
    assert _eq(l1, l2)
    n_lin = sum(isinstance(getattr(l, "linear_attn", None), RecordingGatedDeltaNet)
                for l in (getattr(x, "_layer", x) for x in rec.language_model.model.layers))
    assert n_lin == 6


@pytest.mark.parametrize("accepted", [1, 2, 5, 8, 9])
def test_rollback_matches_direct_prefix(accepted):
    """verify(T) + trim(T-n) must leave the cache exactly as feeding n tokens."""
    model = make_target(0)
    adapter = HybridTargetAdapter(model, FakeTokenizer(), TARGET_IDS)
    prompt = mx.random.randint(0, VOCAB, (1, 17))
    block = mx.random.randint(0, VOCAB, (1, 9))

    adapter.reset()
    adapter.prefill(prompt)
    adapter.verify(block)
    adapter.trim(9 - accepted)
    rolled = _states(adapter.cache)

    adapter.reset()
    adapter.prefill(prompt)
    adapter.verify(block[:, :accepted])
    adapter.trim(0)
    direct = _states(adapter.cache)

    for (kind, a0, a1), (_, b0, b1) in zip(rolled, direct):
        assert _eq(a0, b0) and _eq(a1, b1), kind

    # and the next step's logits agree
    nxt = mx.array([[7]])
    adapter2 = adapter  # direct state
    l_direct, _ = adapter2.verify(nxt)
    adapter.reset()
    adapter.prefill(prompt)
    adapter.verify(block)
    adapter.trim(9 - accepted)
    l_rolled, _ = adapter.verify(nxt)
    assert _eq(l_direct, l_rolled)


def test_trim_needs_verify():
    model = make_target(0)
    adapter = HybridTargetAdapter(model, FakeTokenizer(), TARGET_IDS)
    adapter.reset()
    adapter.prefill(mx.array([[1, 2, 3]]))
    with pytest.raises(RuntimeError):
        adapter.trim(1)


def _divergence(adapter, prompt, toks, base, seed):
    """None if token-exact, else (margin, spread, message) at the first divergence."""
    if toks == base:
        return None
    i = next((j for j in range(min(len(toks), len(base))) if toks[j] != base[j]), None)
    assert i is not None, f"length mismatch {len(toks)} vs {len(base)}"
    adapter.reset()
    logits, _ = adapter.prefill(mx.array([prompt + base[:i]]), capture=False)
    row = logits[0, -1].astype(mx.float32)
    la, lb = float(row[toks[i]].item()), float(row[base[i]].item())
    spread = float((mx.max(row) - mx.min(row)).item())
    msg = (f"seed {seed}: first divergence at token {i}/{len(base)}: dspine {toks[i]} ({la:.6f}) "
           f"vs plain {base[i]} ({lb:.6f}), margin {abs(la - lb):.6f}, logit spread {spread:.3f}, "
           f"device {mx.default_device()}")
    return abs(la - lb), spread, msg


def _noise_floor(adapter, prompt, toks):
    """Max |logit| difference between one T-token pass and T one-token passes (no rollback)."""
    seq = toks
    adapter.reset()
    adapter.prefill(mx.array([prompt]), capture=False)
    block, _ = adapter._forward(mx.array([seq]), capture=False)
    adapter.reset()
    adapter.prefill(mx.array([prompt]), capture=False)
    single = mx.concatenate([adapter._forward(mx.array([[t]]), capture=False)[0] for t in seq], axis=1)
    return float(mx.max(mx.abs(block.astype(mx.float32) - single.astype(mx.float32))).item())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dspine_greedy_equals_plain_greedy(seed):
    """bf16 target: equal to plain greedy except at genuine near-ties."""
    model = make_target(seed)
    adapter = HybridTargetAdapter(model, FakeTokenizer(), TARGET_IDS)
    draft, cfg, _ = make_draft(seed + 10)
    draft.bind(model)
    gen = DSpineGenerator(adapter, draft)
    mx.random.seed(seed)
    prompt = mx.random.randint(0, VOCAB - 4, (1, 12)).tolist()[0]
    toks, stats = gen.generate(prompt, max_tokens=80)
    base, _ = baseline_generate(adapter, prompt, max_tokens=80)
    assert stats.rounds > 5
    noise = _noise_floor(adapter, prompt, base)
    d = _divergence(adapter, prompt, toks, base, seed)
    if d is None:
        print(f"seed {seed}: token-exact over {len(toks)} tokens, {stats.rounds} rounds; noise floor {noise:.6f}")
        return
    margin, spread, msg = d
    print(msg + f"; block-vs-single noise floor {noise:.6f}")
    assert margin <= max(4 * noise, 0.02 * spread), "not a near-tie: " + msg


class _Fp32HiddenAdapter(HybridTargetAdapter):
    """fp32 target whose captured features are handed to the bf16 draft as bf16."""

    def _forward(self, tokens, capture=True):
        logits, hidden = super()._forward(tokens, capture=capture)
        return logits, (None if hidden is None else hidden.astype(mx.bfloat16))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_dspine_greedy_fp32_target(seed):
    """fp32 target: any divergence must be within the device's block-vs-single noise.

    A wrong rollback changes the state outright and shows up as a margin far
    above that noise.
    """
    from mlx.utils import tree_flatten

    model = make_target(seed)
    model.load_weights([
        (k, v.astype(mx.bfloat16) if "embed_tokens" in k else v.astype(mx.float32))
        for k, v in tree_flatten(model.parameters())
    ])
    adapter = _Fp32HiddenAdapter(model, FakeTokenizer(), TARGET_IDS)
    draft, _, _ = make_draft(seed + 10)
    draft.bind(model)
    gen = DSpineGenerator(adapter, draft)
    mx.random.seed(seed)
    prompt = mx.random.randint(0, VOCAB - 4, (1, 12)).tolist()[0]
    toks, stats = gen.generate(prompt, max_tokens=80)
    base, _ = baseline_generate(adapter, prompt, max_tokens=80)
    assert stats.rounds > 5
    noise = _noise_floor(adapter, prompt, base)
    d = _divergence(adapter, prompt, toks, base, seed)
    if d is None:
        print(f"fp32 seed {seed}: token-exact over {len(toks)} tokens; noise floor {noise:.2e}")
        return
    margin, spread, msg = d
    print("fp32 " + msg + f"; block-vs-single noise floor {noise:.2e}")
    assert margin <= max(4 * noise, 1e-4 * spread), "fp32 divergence above numerical noise: " + msg
