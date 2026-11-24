"""Several conversations in one check give what each conversation gives alone."""

import random

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.cache import ArraysCache

from dspine_hybrid.batch_target import BatchTarget, RaggedKV
from dspine_hybrid.hybrid_target import HybridTargetAdapter
from tiny import TARGET_IDS, VOCAB, FakeTokenizer, make_target


def _fp32(seed):
    model = make_target(seed)
    model.load_weights([(k, v.astype(mx.bfloat16) if "embed_tokens" in k else v.astype(mx.float32))
                        for k, v in tree_flatten(model.parameters())])
    return HybridTargetAdapter(model, FakeTokenizer(), TARGET_IDS)


def _rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    assert a.shape == b.shape, (a.shape, b.shape)
    return mx.max(mx.abs(a - b)).item() / max(mx.max(mx.abs(b)).item(), 1.0)


def _single_states(cache):
    out = []
    for c in cache:
        if isinstance(c, ArraysCache):
            out.append((c[0][0], c[1][0]))
        else:
            out.append((c.keys[0, :, : c.offset], c.values[0, :, : c.offset]))
    return out


def _batch_states(cache, b):
    out = []
    for c in cache:
        if isinstance(c, RaggedKV):
            out.append((c.keys[b, :, : c.lens[b]], c.values[b, :, : c.lens[b]]))
        else:
            out.append((c[0][b], c[1][b]))
    return out


# Metal rounds a check of several conversations differently from one alone (seen: 3e-2 on the
# tiny model); a wrong mask or state is off by far more
TOL = 2e-4 if mx.default_device() == mx.cpu else 6e-2


@pytest.mark.parametrize("seed", [0, 1])
def test_batched_rounds_match_single_conversations(seed):
    rng = random.Random(seed)
    mx.random.seed(seed)
    lens = [9, 23, 14]
    prompts = [mx.random.randint(0, VOCAB - 4, (1, n)) for n in lens]
    main = _fp32(seed)
    singles = [_fp32(seed) for _ in lens]          # same weights, one adapter per conversation
    batch = BatchTarget(main)
    for p, s in zip(prompts, singles):
        main.reset()
        main.prefill(p)
        batch.join()
        s.reset()
        s.prefill(p)
    assert batch.lengths() == lens
    for rnd in range(6):
        M = rng.choice([1, 4, 9])
        toks = mx.random.randint(0, VOCAB - 4, (len(lens), M))
        rows = [M] * len(lens) if rnd == 3 else [rng.randint(1, M) for _ in lens]
        logits, hidden = batch.verify(toks)
        mx.eval(logits, hidden)
        batch.keep(rows)
        for b, s in enumerate(singles):
            sl, sh = s.verify(toks[b : b + 1])
            assert _rel(logits[b], sl[0]) <= TOL and _rel(hidden[b], sh[0]) <= TOL, (rnd, b)
            s.trim(M - rows[b])
            lens[b] += rows[b]
        assert batch.lengths() == lens
        for b, s in enumerate(singles):
            for (a0, a1), (b0, b1) in zip(_batch_states(batch.cache, b), _single_states(s.cache)):
                assert _rel(a0, b0) <= TOL and _rel(a1, b1) <= TOL, (rnd, b)


def test_dropped_check_join_and_leave():
    mx.random.seed(5)
    main = _fp32(2)
    batch = BatchTarget(main)
    prompts = [mx.random.randint(0, VOCAB - 4, (1, n)) for n in (12, 300, 7)]
    for p in prompts[:2]:
        main.reset()
        main.prefill(p)
        batch.join()
    toks = mx.random.randint(0, VOCAB - 4, (2, 5))
    first, _ = batch.verify(toks)
    mx.eval(first)
    batch.keep([0, 0])                              # nothing kept: the same check gives the same rows
    again, _ = batch.verify(toks)
    assert _rel(again, first) <= TOL
    batch.keep([2, 5])
    with pytest.raises(RuntimeError):
        batch.keep([1, 1])
    # a third conversation joins, the first leaves; the others are untouched
    main.reset()
    main.prefill(prompts[2])
    assert batch.join() == 2 and batch.lengths() == [14, 305, 7]
    before = _batch_states(batch.cache, 1)
    batch.leave(0)
    assert batch.lengths() == [305, 7]
    for (a0, a1), (b0, b1) in zip(_batch_states(batch.cache, 0), before):
        assert _rel(a0, b0) == 0 and _rel(a1, b1) == 0
    ref = _fp32(2)
    ref.reset()
    ref.prefill(prompts[2])
    nxt = mx.random.randint(0, VOCAB - 4, (2, 3))
    logits, _ = batch.verify(nxt)
    assert _rel(logits[1], ref.verify(nxt[1:2])[0][0]) <= TOL
    batch.keep([3, 3])
    batch.leave(1)
    batch.leave(0)
    assert batch.size == 0 and batch.cache is None


def _random_tree(rng, m):
    return [-1] + [rng.randint(max(0, i - 3), i - 1) for i in range(1, m)]


def _random_path(rng, parents):
    """A path from row 0 that takes a random child until a leaf or a coin says stop."""
    path = [0]
    while True:
        kids = [i for i, p in enumerate(parents) if p == path[-1]]
        if not kids or rng.random() < 0.25:
            return path
        path.append(rng.choice(kids))


@pytest.mark.parametrize("seed", [0, 1])
def test_batched_trees_match_single_conversations(seed):
    rng = random.Random(seed)
    mx.random.seed(seed)
    lens = [9, 23, 14]
    prompts = [mx.random.randint(0, VOCAB - 4, (1, n)) for n in lens]
    main = _fp32(seed)
    singles = [_fp32(seed) for _ in lens]
    batch = BatchTarget(main)
    for p, s in zip(prompts, singles):
        main.reset()
        main.prefill(p)
        batch.join()
        s.reset()
        s.prefill(p)
    for rnd in range(6):
        M = rng.choice([1, 5, 9])
        toks = mx.random.randint(0, VOCAB - 4, (len(lens), M))
        # conversation 0 always checks a chain, the others a tree of their own
        parents = [list(range(-1, M - 1))] + [_random_tree(rng, M) for _ in lens[1:]]
        paths = [_random_path(rng, p) for p in parents]
        if rnd == 2:
            paths[1] = []                              # one conversation keeps nothing
        logits, hidden = batch.verify_tree(toks, parents)
        mx.eval(logits, hidden)
        batch.keep_paths(paths)
        for b, s in enumerate(singles):
            sl, sh = s.verify_tree(toks[b].tolist(), parents[b])
            assert _rel(logits[b], sl[0]) <= TOL and _rel(hidden[b], sh[0]) <= TOL, (rnd, b)
            if paths[b]:
                s.commit_tree(paths[b])
            else:
                s._tree = None
                for mod in s._linear + s._attn:
                    if mod is not None:
                        mod._box.tree_record = None
            lens[b] += len(paths[b])
        assert batch.lengths() == lens
        for b, s in enumerate(singles):
            for (a0, a1), (b0, b1) in zip(_batch_states(batch.cache, b), _single_states(s.cache)):
                assert _rel(a0, b0) <= TOL and _rel(a1, b1) <= TOL, (rnd, b)
    # a chain check after the trees still works on the same caches
    toks = mx.random.randint(0, VOCAB - 4, (len(lens), 4))
    logits, _ = batch.verify(toks)
    for b, s in enumerate(singles):
        assert _rel(logits[b], s.verify(toks[b : b + 1])[0][0]) <= TOL
    batch.keep([4, 4, 4])
    with pytest.raises(ValueError):
        batch.verify_tree(toks, [[-1, 0, 1, 2]] * 3)
        batch.keep_paths([[0, 2], [0], [0]])
