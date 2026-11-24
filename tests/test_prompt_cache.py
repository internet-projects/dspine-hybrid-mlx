"""Prompt cache across turns: a reused cache must give what a full prefill gives."""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from dspine_hybrid.adaptive import AdaptiveDSpineGenerator
from dspine_hybrid.hybrid_target import HybridTargetAdapter
from dspine_mlx.generate import DSpineGenerator
from tiny import TARGET_IDS, VOCAB, FakeTokenizer, make_draft, make_target


def _fp32_target(seed):
    model = make_target(seed)
    model.load_weights([(k, v.astype(mx.bfloat16) if "embed_tokens" in k else v.astype(mx.float32))
                        for k, v in tree_flatten(model.parameters())])
    return model


class Adapter(HybridTargetAdapter):
    """fp32 target whose features reach the bf16 draft as bf16 (as in test_hybrid_tiny)."""

    def __init__(self, model):
        super().__init__(model, FakeTokenizer(), TARGET_IDS)

    def _forward(self, tokens, capture=True):
        logits, hidden = super()._forward(tokens, capture=capture)
        return logits, (None if hidden is None else hidden.astype(mx.bfloat16))


def _states(cache):
    from mlx_lm.models.cache import ArraysCache

    out = []
    for c in cache:
        if isinstance(c, ArraysCache):
            out.append((c[0], c[1]))
        else:
            out.append((c.keys[:, :, : c.offset], c.values[:, :, : c.offset]))
    return out


def _rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    if a.shape != b.shape:
        return float("inf")
    return mx.max(mx.abs(a - b)).item() / max(mx.max(mx.abs(b)).item(), 1.0)


def _close(a, b, tol=2e-4):
    return _rel(a, b) <= tol


def _diffs(logits, hidden, cache, rl, rh, rcache):
    n = int(hidden.shape[1])
    st = max([max(_rel(a0, b0), _rel(a1, b1)) for (a0, a1), (b0, b1) in zip(_states(cache), _states(rcache))])
    return _rel(logits[0, -1], rl[0, -1]), _rel(hidden, rh[:, -n:]), st


def _same_as_fresh(seed, adapter, ids, logits, hidden, blocked_from=None):
    """logits, every cache state and the last hidden rows equal a fresh full prefill.

    blocked_from: the reused cache was built by verify blocks after this many tokens. Metal
    rounds block passes differently from a one-shot prefill (see test_hybrid_tiny's noise
    floor), so the tolerance grows to 8x the difference a plain block-wise pass shows on this
    device; a wrong state is off by far more. On CPU that difference is ~0.
    """
    ref = Adapter(_fp32_target(seed))  # same weights, own model: one adapter per model
    ref.reset()
    rl, rh = ref.prefill(mx.array([ids]))
    got = _diffs(logits, hidden, adapter.cache, rl, rh, ref.cache)
    tol = [2e-4] * 3
    if blocked_from is not None:
        blk = Adapter(_fp32_target(seed))
        blk.reset()
        _, h0 = blk.prefill(mx.array([ids[:blocked_from]]))
        hs = [h0]
        end = len(ids) - 1
        for s in range(blocked_from, end, 5):
            _, h = blk.verify(mx.array([ids[s : min(s + 5, end)]]))
            blk.trim(0)
            hs.append(h)
        bl, h = blk.verify(mx.array([ids[end:]]))
        hs.append(h)
        noise = _diffs(bl, mx.concatenate(hs, axis=1)[:, -int(hidden.shape[1]):], blk.cache, rl, rh, ref.cache)
        tol = [max(t, 8 * x) for t, x in zip(tol, noise)]
        print(f"block-pass noise (logits, hidden, states): {noise}; got {got}")
    for name, g, t in zip(("logits", "hidden", "states"), got, tol):
        assert g <= t, f"{name}: {g:.2e} > {t:.2e}"


def _setup(seed, adaptive=False):
    model = _fp32_target(seed)
    adapter = Adapter(model)
    adapter.enable_prompt_cache(True)
    draft, _, _ = make_draft(seed + 10)
    draft.bind(model)
    if adaptive:
        gen = AdaptiveDSpineGenerator(adapter, draft, cost=[0.0] + [1.0 + 0.05 * t for t in range(1, 18)])
    else:
        gen = DSpineGenerator(adapter, draft)
    mx.random.seed(seed)
    p1 = mx.random.randint(0, VOCAB - 4, (40,)).tolist()
    return model, adapter, gen, p1


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("max_tokens", [30, 31, 37])
def test_continued_chat_reuses_everything(seed, max_tokens):
    """turn 2 = turn 1 prompt + answer + new text: only the new part is prefilled."""
    model, adapter, gen, p1 = _setup(seed)
    toks, _ = gen.generate(p1, max_tokens=max_tokens)
    assert adapter.last_reused == 0
    user = mx.random.randint(0, VOCAB - 4, (11,)).tolist()
    p2 = p1 + toks + user
    adapter.reset()
    logits, hidden = adapter.prefill(mx.array([p2]))
    # the last generated token was never fed, so at least that one is new
    assert len(p1) + len(toks) - 1 - 16 <= adapter.last_reused <= len(p1) + len(toks) - 1
    assert adapter.last_reused > len(p1)
    assert hidden.shape[1] == len(p2)
    _same_as_fresh(seed, adapter, p2, logits, hidden, blocked_from=len(p1))


@pytest.mark.parametrize("seed", [0, 1])
def test_edited_history_restores_prompt_snapshot(seed):
    """turn 2 drops part of the answer (e.g. stripped thinking): restore the prompt snapshot."""
    model, adapter, gen, p1 = _setup(seed)
    toks, _ = gen.generate(p1, max_tokens=24)
    p2 = p1 + toks[:3] + [VOCAB - 2] + toks[5:9]
    adapter.reset()
    logits, hidden = adapter.prefill(mx.array([p2]))
    assert adapter.last_reused == len(p1)
    _same_as_fresh(seed, adapter, p2, logits, hidden)


def test_same_prompt_again_and_new_prompt():
    model, adapter, gen, p1 = _setup(3)
    a, _ = gen.generate(p1, max_tokens=20)
    b, _ = gen.generate(p1, max_tokens=20)
    assert adapter.last_reused == len(p1) and a == b
    other = [(t + 1) % (VOCAB - 4) for t in p1]
    adapter.reset()
    logits, hidden = adapter.prefill(mx.array([other]))
    assert adapter.last_reused == 0
    _same_as_fresh(3, adapter, other, logits, hidden)


@pytest.mark.parametrize("adaptive", [False, True])
def test_multi_turn_greedy_matches_fresh_runs(adaptive):
    """Three turns with the cache give the same tokens as three fresh runs."""
    model, adapter, gen, p1 = _setup(5, adaptive)
    model2 = _fp32_target(5)
    draft2, _, _ = make_draft(15)
    draft2.bind(model2)
    fresh_adapter = Adapter(model2)
    fresh = (AdaptiveDSpineGenerator(fresh_adapter, draft2, cost=gen.cost) if adaptive
             else DSpineGenerator(fresh_adapter, draft2))
    hist = list(p1)
    for turn in range(3):
        toks, _ = gen.generate(hist, max_tokens=25)
        ref, _ = fresh.generate(hist, max_tokens=25)
        assert toks == ref, f"turn {turn}"
        if turn:
            assert adapter.last_reused > len(p1)
        hist = hist + toks + mx.random.randint(0, VOCAB - 4, (7,)).tolist()


def test_off_by_default_and_capture_false_drops_session():
    model, adapter, gen, p1 = _setup(2)
    gen.generate(p1, max_tokens=12)
    adapter.reset()
    adapter.prefill(mx.array([p1]), capture=False)
    adapter.reset()
    adapter.prefill(mx.array([p1 + [1, 2]]))
    assert adapter.last_reused == 0
    assert Adapter(_fp32_target(2)).prompt_cache is False


@pytest.mark.parametrize("seed", [0, 1])
def test_new_conversation_with_the_same_system_prompt_uses_a_mark(seed):
    """Same long start, different question: the third prompt restarts from the shared start."""
    model, adapter, gen, p1 = _setup(seed)
    adapter.mark_min = 8
    system = p1[:30]
    q = [mx.random.randint(0, VOCAB - 4, (9,)).tolist() for _ in range(4)]
    gen.generate(system + q[0], max_tokens=20)
    assert adapter.last_reused == 0
    gen.generate(system + q[1], max_tokens=20)          # learns where the two prompts part
    assert adapter.last_reused == 0 and [m[0] for m in adapter._marks] == [30]
    toks, _ = gen.generate(system + q[2], max_tokens=20)
    assert adapter.last_reused == 30
    fresh_model, fresh, fresh_gen, _ = _setup(seed)
    ref, _ = fresh_gen.generate(system + q[2], max_tokens=20)
    assert toks == ref
    adapter.reset()
    logits, hidden = adapter.prefill(mx.array([system + q[3]]))
    assert adapter.last_reused == 30 and hidden.shape[1] == 39
    _same_as_fresh(seed, adapter, system + q[3], logits, hidden)


def test_mark_hint_helps_from_the_second_prompt_and_survives_a_rewind():
    model, adapter, gen, p1 = _setup(2)
    adapter.mark_min = 8
    system = p1[:25]
    q = [mx.random.randint(0, VOCAB - 4, (7,)).tolist() for _ in range(3)]
    adapter.mark_hint = 25
    gen.generate(system + q[0], max_tokens=12)
    assert adapter.mark_hint is None and [m[0] for m in adapter._marks] == [25]
    adapter.mark_hint = 25
    gen.generate(system + q[1], max_tokens=12)
    assert adapter.last_reused == 25
    assert adapter.rewind_to_prompt()                    # as after a stop word or a client that left
    adapter.reset()
    logits, hidden = adapter.prefill(mx.array([system + q[2]]))
    assert adapter.last_reused == 25
    _same_as_fresh(2, adapter, system + q[2], logits, hidden)
    # a prompt that is exactly the marked start has no stored logits: it is prefilled again
    adapter.reset()
    logits, hidden = adapter.prefill(mx.array([system]))
    _same_as_fresh(2, adapter, system, logits, hidden)
