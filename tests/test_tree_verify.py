"""Tree verify: every node's logits equal a chain verify of its root path; commit = feeding the path."""

import mlx.core as mx
import pytest

from dspine_hybrid.hybrid_target import HybridTargetAdapter, TreeSpec
from test_hybrid_tiny import _states
from tiny import TARGET_IDS, VOCAB, FakeTokenizer, make_target

#        0
#      / | \
#     1  2  6
#    / \     \
#   3   4     7
#   |
#   5
PARENTS = [-1, 0, 0, 1, 1, 3, 0, 6]


def _path(parents, r):
    out = []
    while r >= 0:
        out.append(r)
        r = parents[r]
    return out[::-1]


def _close(a, b, tol=2e-2):
    return a.shape == b.shape and float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()) <= tol


def _setup(seed=0, n_prompt=19):
    model = make_target(seed)
    ad = HybridTargetAdapter(model, FakeTokenizer(), TARGET_IDS)
    mx.random.seed(seed)
    prompt = mx.random.randint(0, VOCAB, (1, n_prompt))
    toks = mx.random.randint(0, VOCAB, (len(PARENTS),)).tolist()
    return ad, prompt, toks


def test_spec():
    t = TreeSpec(PARENTS)
    assert t.depth == [0, 1, 1, 2, 2, 3, 1, 2]
    assert sorted(map(tuple, t.paths)) == [(0, 1, 3, 5), (0, 1, 4), (0, 2), (0, 6, 7)]
    assert not t.is_chain and TreeSpec([-1, 0, 1]).is_chain
    with pytest.raises(ValueError):
        TreeSpec([-1, 1, 0])


def test_chain_tree_equals_verify():
    ad, prompt, toks = _setup()
    ad.reset()
    ad.prefill(prompt)
    l1, h1 = ad.verify(mx.array([toks]))
    mx.eval(l1, h1)
    ad.trim(len(toks) - 3)
    ref = _states(ad.cache)
    ad.reset()
    ad.prefill(prompt)
    l2, h2 = ad.verify_tree(toks, [-1] + list(range(len(toks) - 1)))
    assert _close(l1, l2) and _close(h1, h2)
    if mx.default_device() == mx.cpu:
        assert mx.array_equal(mx.argmax(l1, -1), mx.argmax(l2, -1)).item()
    ad.commit_tree([0, 1, 2])
    for a, b in zip(ref, _states(ad.cache)):
        assert a[0] == b[0] and _close(a[1], b[1], 1e-3) and _close(a[2], b[2], 1e-3)


@pytest.mark.parametrize("seed", [0, 1])
def test_tree_rows_equal_path_verify(seed):
    ad, prompt, toks = _setup(seed)
    ad.reset()
    ad.prefill(prompt)
    lt, ht = ad.verify_tree(toks, PARENTS)
    mx.eval(lt, ht)
    for leaf in (5, 4, 2, 7):
        rows = _path(PARENTS, leaf)
        ad.reset()
        ad.prefill(prompt)
        lc, hc = ad.verify(mx.array([[toks[r] for r in rows]]))
        idx = mx.array(rows)
        assert _close(lc[0], lt[0][idx]), leaf
        assert _close(hc[0], ht[0][idx]), leaf
        if mx.default_device() == mx.cpu:
            assert mx.array_equal(mx.argmax(lc[0], -1), mx.argmax(lt[0][idx], -1)).item()


@pytest.mark.parametrize("rows", [[0], [0, 1], [0, 1, 3, 5], [0, 1, 4], [0, 6, 7], [0, 2]])
def test_commit_equals_feeding_the_path(rows):
    ad, prompt, toks = _setup(2)
    nxt = mx.random.randint(0, VOCAB, (1, 6))
    ad.reset()
    ad.prefill(prompt)
    ad.verify(mx.array([[toks[r] for r in rows]]))
    ad.trim(0)
    ref = _states(ad.cache)
    lref, href = ad.verify(nxt)
    mx.eval(lref, href)

    ad.reset()
    ad.prefill(prompt)
    ad.verify_tree(toks, PARENTS)
    ad.commit_tree(rows)
    got = _states(ad.cache)
    for a, b in zip(ref, got):
        assert a[0] == b[0] and _close(a[1], b[1], 1e-3) and _close(a[2], b[2], 1e-3)
    l2, h2 = ad.verify(nxt)
    assert _close(lref, l2) and _close(href, h2)
    if mx.default_device() == mx.cpu:
        assert mx.array_equal(mx.argmax(lref, -1), mx.argmax(l2, -1)).item()


def test_commit_rejects_non_paths():
    ad, prompt, toks = _setup()
    ad.reset()
    ad.prefill(prompt)
    ad.verify_tree(toks, PARENTS)
    with pytest.raises(ValueError):
        ad.commit_tree([0, 3])
