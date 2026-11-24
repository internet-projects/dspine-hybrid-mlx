"""Copy drafts and the draft tree: same output as plain decoding, and the lanes actually fire."""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from dspine_hybrid.adaptive import AdaptiveDSpineGenerator, measure_cost
from dspine_hybrid.hybrid_target import HybridTargetAdapter
from dspine_hybrid.lanes import CopyIndex, LaneDSpineGenerator, choose_nodes, grow_tree, node_chances
from dspine_mlx.generate import DSpineGenerator, baseline_generate
from test_hybrid_tiny import _divergence, _Fp32HiddenAdapter, _noise_floor
from tiny import TARGET_IDS, VOCAB, FakeTokenizer, make_draft, make_target

COST = [0.0] + [0.030 + 0.001 * T + (0.012 if T > 5 else 0.0) for T in range(1, 18)]


def _fp32(seed):
    m = make_target(seed)
    m.load_weights([(k, v.astype(mx.bfloat16) if "embed_tokens" in k else v.astype(mx.float32))
                    for k, v in tree_flatten(m.parameters())])
    ad = _Fp32HiddenAdapter(m, FakeTokenizer(), TARGET_IDS)
    d, _, _ = make_draft(seed + 10)
    d.bind(m)
    return ad, d


def _same(ad, prompt, out, base) -> bool:
    """True if token-exact. On Metal a tree, a block and single steps round differently, so a
    near-tie may flip (same criterion as test_dspine_greedy_fp32_target); then the caller stops."""
    if out == base:
        return True
    assert mx.default_device() != mx.cpu, "output differs from plain greedy decoding"
    was = ad.prompt_cache
    ad.enable_prompt_cache(False)
    noise = _noise_floor(ad, prompt, base[:16])
    margin, spread, msg = _divergence(ad, prompt, out, base, 0)
    ad.enable_prompt_cache(was)
    assert margin <= max(4 * noise, 1e-4 * spread), "divergence above noise: " + msg
    return False


def _bf16(seed):
    m = make_target(seed)
    ad = HybridTargetAdapter(m, FakeTokenizer(), TARGET_IDS)
    d, _, _ = make_draft(seed + 10)
    d.bind(m)
    return ad, d


def test_copy_index():
    ix = CopyIndex(3)
    ix.extend([1, 2, 3, 4, 5, 6, 9, 1, 2, 3])
    assert ix.lookup(2) == [4, 5]
    assert ix.lookup(50) == [4, 5, 6, 9, 1, 2, 3]
    ix.extend([4])
    assert ix.lookup(3) == [5, 6, 9]
    ix.extend([7])
    assert ix.lookup(3) == []
    ix.extend([1, 2, 3])
    assert ix.lookup(1) == [4]          # the latest earlier occurrence, which went on with 4
    short = CopyIndex(3)
    short.extend([5, 5, 5])
    assert short.lookup(4) == []
    short.extend([5])
    assert short.lookup(4) == [5]       # overlapping repeat


def test_grow_tree_is_best_first_and_keeps_the_chain():
    cand = [[10, 11, 12], [20, 21, 22], [30, 31, 32]]
    row0 = [0.6, 0.3, 0.1]
    chance = [[row0] * 3,
              [[0.9, 0.05, 0.05], [0.2, 0.7, 0.1], [0.4, 0.3, 0.3]],
              [[0.5, 0.4, 0.1]] * 3]
    tok, par, path = grow_tree(cand, chance, 6, branch=3)
    assert tok[:2] == [10, 20] and par[:2] == [-1, 0]
    assert path == sorted(path, reverse=True)
    assert all(p < i for i, p in enumerate(par))
    assert (11, -1) in list(zip(tok, par))                 # the root's second choice is in the tree
    i11 = tok.index(11)
    assert (21, i11) in list(zip(tok, par))                # and continues with ITS best follower
    assert abs(path[tok.index(21)] - 0.3 * 0.7) < 1e-9
    chain_tok, chain_par, _ = grow_tree(cand, chance, 3, branch=1)
    assert chain_tok == [10, 20, 30] and chain_par == [-1, 0, 1]
    assert choose_nodes([0.9, 0.8, 0.7, 0.01, 0.01], COST, 5) == 3
    assert choose_nodes([0.99] * 15, COST, 15) == 15


def test_node_chances_calibrated_is_a_distribution_with_the_same_winner():
    mx.random.seed(0)
    score = mx.random.normal((5, 16, 16)) * 3
    rms = mx.random.uniform(0.5, 2.0, (5,))
    from dspine_hybrid import calibration as cal

    raw = mx.array(node_chances(score, rms, None, 0.6))
    fit = mx.array(node_chances(score, rms, cal.GREEDY, 0.6))
    assert mx.array_equal(mx.argmax(raw, -1), mx.argmax(fit, -1)).item()
    assert float(mx.max(mx.sum(fit, -1)).item()) <= 1.0 + 1e-5
    assert float(mx.min(fit).item()) >= 0.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lanes_off_reproduces_the_measured_generators(seed):
    ad, d = _fp32(seed)
    prompt = list(range(3 + seed, 17 + seed))
    kw = dict(max_tokens=40)
    for temp in (0.0, 0.8):
        mx.random.seed(seed)
        ref, _ = DSpineGenerator(ad, d, block_size=5).generate(prompt, temperature=temp, **kw)
        mx.random.seed(seed)
        got, _ = LaneDSpineGenerator(ad, d, block_size=5, copy=False).generate(prompt, temperature=temp, **kw)
        assert got == ref
    for conf in ("vocab", "calibrated"):
        for temp in (0.0, 0.8) if conf == "calibrated" else (0.0,):
            mx.random.seed(seed)
            ref, _ = AdaptiveDSpineGenerator(ad, d, cost=COST, confidence=conf, block_size=16,
                                             fixed_block_size=5).generate(prompt, temperature=temp, **kw)
            mx.random.seed(seed)
            got, _ = LaneDSpineGenerator(ad, d, block_size=5, adaptive=True, cost=COST, confidence=conf,
                                         copy=False).generate(prompt, temperature=temp, **kw)
            assert got == ref, (conf, temp)


def _repeating_prompt(ad, seed, n=40):
    """A prompt whose greedy continuation the prompt already contains: plain output, twice."""
    mx.random.seed(seed)
    head = mx.random.randint(0, VOCAB - 4, (1, 10)).tolist()[0]
    first, _ = baseline_generate(ad, head, max_tokens=n)
    return head + first + head


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_copy_greedy_is_exact(seed, mode):
    ad, d = _fp32(seed)
    prompt = _repeating_prompt(ad, seed)
    base, _ = baseline_generate(ad, prompt, max_tokens=70)
    kw = dict(block_size=5, copy=True, copy_min=1)
    if mode == "adaptive":
        kw.update(adaptive=True, cost=COST)
    g = LaneDSpineGenerator(ad, d, **kw)
    out, st = g.generate(prompt, max_tokens=70)
    if not _same(ad, prompt, out, base):
        return
    ls = g.lane_stats
    assert ls["copy_rounds"] > 0 and ls["draft_rounds"] > 0
    assert ls["copy_accepted"] <= ls["copy_offered"]
    assert st.generated_tokens == 70


def test_copy_hits_on_a_copying_target():
    """A target that repeats its prompt: copy rounds must carry most of the tokens."""
    ad, d = _fp32(3)
    mx.random.seed(3)
    head = mx.random.randint(0, VOCAB - 4, (1, 10)).tolist()[0]
    first, _ = baseline_generate(ad, head, max_tokens=60)
    # after `head + first[:30]` again, plain greedy decoding must go on with first[30:] only if
    # the recurrent state agrees; so measure against the baseline rather than assume it
    prompt = head + first + head + first[:30]
    base, _ = baseline_generate(ad, prompt, max_tokens=40)
    g = LaneDSpineGenerator(ad, d, block_size=5, copy=True, copy_min=6)
    out, st = g.generate(prompt, max_tokens=40)
    if not _same(ad, prompt, out, base):
        return
    ls = g.lane_stats
    same = sum(a == b for a, b in zip(base, first[30:]))
    if same >= 20:       # the target really copies: then copy drafts must be doing the work
        assert ls["copy_accepted"] >= same // 2
        assert st.rounds < len(out) / 2


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("mode", ["fixed", "adaptive", "calibrated"])
def test_tree_greedy_is_exact(seed, mode):
    ad, d = _fp32(seed)
    mx.random.seed(seed)
    prompt = mx.random.randint(0, VOCAB - 4, (1, 14)).tolist()[0]
    base, _ = baseline_generate(ad, prompt, max_tokens=60)
    kw = dict(block_size=6, copy=False, tree=True)
    if mode != "fixed":
        kw.update(adaptive=True, cost=COST, confidence="calibrated" if mode == "calibrated" else "vocab")
    g = LaneDSpineGenerator(ad, d, **kw)
    out, st = g.generate(prompt, max_tokens=60)
    if not _same(ad, prompt, out, base):
        return
    assert g.lane_stats["tree_rounds"] > 0
    assert g.lane_stats["tree_nodes"] >= g.lane_stats["tree_rounds"]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_tree_with_copy_branch_is_exact(seed, mode):
    """A copy guess checked as a branch of the tree: same output, and the branch is offered."""
    ad, d = _fp32(seed)
    prompt = _repeating_prompt(ad, seed)
    base, _ = baseline_generate(ad, prompt, max_tokens=40)
    kw = dict(block_size=6, copy=False, tree=True, tree_copy=4, tree_copy_min=1)
    if mode != "fixed":
        kw.update(adaptive=True, cost=COST, confidence="vocab")
    g = LaneDSpineGenerator(ad, d, **kw)
    out, _ = g.generate(prompt, max_tokens=40)
    if not _same(ad, prompt, out, base):
        return
    assert g.lane_stats["graft_nodes"] > 0
    assert g.lane_stats["graft_tokens"] <= g.lane_stats["graft_nodes"]


@pytest.mark.parametrize("seed", [0, 1])
def test_live_cost_is_exact(seed):
    """Lengths chosen from costs with the context's attention time added: same output."""
    from dspine_hybrid.adaptive import context_cost

    ad, d = _fp32(seed)
    mx.random.seed(seed)
    prompt = mx.random.randint(0, VOCAB - 4, (1, 2100)).tolist()[0]
    base, _ = baseline_generate(ad, prompt, max_tokens=30)
    g = LaneDSpineGenerator(ad, d, block_size=6, copy=False, adaptive=True, cost=COST, live_cost=True)
    out, st = g.generate(prompt, max_tokens=30)
    assert g.cost == COST                                  # the measured table itself is not changed
    assert len(g.cost_used) == len(COST) and all(u >= c for u, c in zip(g.cost_used, COST))
    assert context_cost(ad, COST, 100) is COST             # short prompts keep the table
    if not _same(ad, prompt, out, base):
        return


@pytest.mark.parametrize("seed", [0, 1])
def test_tree_and_copy_together_with_prompt_cache(seed):
    ad, d = _fp32(seed)
    ad.enable_prompt_cache(True)
    prompt = _repeating_prompt(ad, seed)
    ad.enable_prompt_cache(True)
    g = LaneDSpineGenerator(ad, d, block_size=5, copy=True, copy_min=1, tree=True)
    out1, _ = g.generate(prompt, max_tokens=30)
    turn2 = prompt + out1 + [7, 8, 9]
    out2, _ = g.generate(turn2, max_tokens=30)
    reused = ad.last_reused
    ad.enable_prompt_cache(False)
    b1, _ = baseline_generate(ad, prompt, max_tokens=30)
    b2, _ = baseline_generate(ad, turn2, max_tokens=30)
    _same(ad, prompt, out1, b1)
    _same(ad, turn2, out2, b2)
    assert reused >= len(prompt)


def test_copy_sampled_follows_the_target_distribution():
    """A copied token checked as a one-point proposal: the committed token is a draw from the target."""
    from dspine_mlx.sampler import sampled_verify

    V, n = 24, 6000
    mx.random.seed(0)
    logits = mx.random.normal((2, V)) * 1.5
    p = mx.softmax(logits[0], axis=-1).tolist()
    for tok in (int(mx.argmax(logits[0]).item()), int(mx.argmin(logits[0]).item())):
        drafts = mx.array([tok], dtype=mx.int32)
        counts = [0] * V
        for i in range(n):
            acc, bonus = sampled_verify(
                logits, drafts, drafts[:, None], mx.ones((1, 1)), inv_temp=1.0, top_k=V, top_p=1.0,
                min_p=0.0, u_accept=mx.random.uniform(shape=(1,)), u_final=mx.random.uniform(),
                u_res=mx.random.uniform())
            counts[tok if int(acc.item()) == 1 else int(bonus.item())] += 1
        tv = 0.5 * sum(abs(c / n - q) for c, q in zip(counts, p))
        assert tv < 0.04, (tok, tv)


def test_copy_sampled_runs_and_counts():
    ad, d = _bf16(6)
    prompt = _repeating_prompt(ad, 6)
    for kw in (dict(), dict(adaptive=True, cost=COST, confidence="calibrated")):
        g = LaneDSpineGenerator(ad, d, block_size=5, copy=True, copy_min=1, **kw)
        mx.random.seed(0)
        out, st = g.generate(prompt, max_tokens=50, temperature=0.3, top_p=0.95)
        assert len(out) == 50
        assert g.lane_stats["copy_offered"] >= g.lane_stats["copy_accepted"]


# The tiny drafter is random, so its drafts are never accepted. The tests below hand the generator
# proposals that are (mostly) right, to exercise long accepts, side branches and their commits.

class _Oracle:
    """Feeds the true continuation, with every ``wrong_every``-th proposed token corrupted."""

    def __init__(self, base, wrong_every=7):
        self.base, self.pos, self.k, self.wrong_every = base, 1, 0, wrong_every

    def on_tokens(self, new):
        self.pos += len(new)

    def take(self, n):
        out = []
        for t in self.base[self.pos : self.pos + n]:
            self.k += 1
            out.append((t + 1) % (VOCAB - 4) if self.k % self.wrong_every == 0 else t)
        return out


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("temp", [0.0])
def test_copy_long_accepts_are_exact(seed, temp, monkeypatch):
    ad, d = _fp32(seed)
    ad.enable_prompt_cache(True)
    mx.random.seed(seed)
    prompt = mx.random.randint(0, VOCAB - 4, (1, 14)).tolist()[0]
    ad.enable_prompt_cache(False)
    base, _ = baseline_generate(ad, prompt, max_tokens=90)
    ad.enable_prompt_cache(True)
    orc = _Oracle(base)
    monkeypatch.setattr(CopyIndex, "lookup", lambda self, n: orc.take(n))
    g = LaneDSpineGenerator(ad, d, block_size=5, copy=True)
    out, st = g.generate(prompt, max_tokens=90, on_tokens=orc.on_tokens)
    if not _same(ad, prompt, out, base):
        return
    ls = g.lane_stats
    assert ls["copy_accepted"] >= 45 and st.rounds < 40
    assert ls["copy_offered"] > ls["copy_accepted"]          # the corrupted ones were refused
    # second turn on the cached state the copy rounds left behind
    turn2 = prompt + out + [5, 6, 7]
    ad.enable_prompt_cache(False)
    base2, _ = baseline_generate(ad, turn2, max_tokens=20)
    ad.enable_prompt_cache(True)
    # rebuild the state the first turn left (the baseline above dropped it)
    orc.__init__(base)
    g.generate(prompt, max_tokens=90, on_tokens=orc.on_tokens)
    orc2 = _Oracle(base2, wrong_every=5)
    monkeypatch.setattr(CopyIndex, "lookup", lambda self, n: orc2.take(n))
    out2, _ = g.generate(turn2, max_tokens=20, on_tokens=orc2.on_tokens)
    reused = ad.last_reused
    _same(ad, turn2, out2, base2)
    assert reused >= len(prompt)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_tree_side_branches_are_exact(seed, mode, monkeypatch):
    """Trees whose first choice is wrong and whose side branch is right, at varying depths."""
    import dspine_hybrid.lanes as lanes

    ad, d = _fp32(seed)
    mx.random.seed(seed)
    prompt = mx.random.randint(0, VOCAB - 4, (1, 14)).tolist()[0]
    base, _ = baseline_generate(ad, prompt, max_tokens=80)
    state = dict(pos=1, k=0)

    def bad(t):
        return (t + 3) % (VOCAB - 4)

    def oracle_tree(cand, chance, max_nodes, branch=4, twins=None):
        true = base[state["pos"] : state["pos"] + 4]
        state["k"] += 1
        shape = state["k"] % 4
        if len(true) < 3 or shape == 0:              # all wrong
            tok, par = [bad(base[state["pos"] - 1]), 1, 2], [-1, 0, -1]
        elif shape == 1:                             # wrong first choice, right sibling, then on
            tok, par = [bad(true[0]), true[0], true[1], bad(true[1]), true[2]], [-1, -1, 1, 1, 2]
        elif shape == 2:                             # right, then a right second choice two deep
            tok, par = [true[0], bad(true[1]), true[1], true[2], bad(true[0])], [-1, 0, 0, 2, -1]
        else:                                        # a chain that is right all the way
            tok, par = list(true[:3]), [-1, 0, 1]
        tok, par = tok[:max_nodes], par[:max_nodes]
        return tok, par, [0.9 ** (i + 1) for i in range(len(tok))]

    def on_tokens(new):
        state["pos"] += len(new)

    monkeypatch.setattr(lanes, "grow_tree", oracle_tree)
    kw = dict(block_size=6, copy=False, tree=True)
    if mode == "adaptive":
        kw.update(adaptive=True, cost=[0.0] + [0.03] * 17)
    g = LaneDSpineGenerator(ad, d, **kw)
    ad.enable_prompt_cache(True)
    out, st = g.generate(prompt, max_tokens=80, on_tokens=on_tokens)
    if not _same(ad, prompt, out, base):
        return
    ls = g.lane_stats
    assert ls["tree_side_tokens"] >= 10 and st.rounds < 50
    assert st.accepted_drafts >= 40
    # the cache the tree commits left must serve a second turn
    turn2 = prompt + out + [9, 10]
    monkeypatch.setattr(lanes, "grow_tree", grow_tree)
    out2, _ = g.generate(turn2, max_tokens=12)
    reused = ad.last_reused
    ad.enable_prompt_cache(False)
    base2, _ = baseline_generate(ad, turn2, max_tokens=12)
    _same(ad, turn2, out2, base2)
    assert reused >= len(prompt)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("tree", [False, True])
def test_auto_rows_is_exact_and_shrinks_the_draft(seed, tree, monkeypatch):
    ad, d = _fp32(seed)
    mx.random.seed(seed)
    prompt = mx.random.randint(0, VOCAB - 4, (1, 14)).tolist()[0]
    base, _ = baseline_generate(ad, prompt, max_tokens=50)
    seen = []
    real = type(d).__call__

    def spy(self, hidden, block, caches=None):
        seen.append(int(block.shape[1]))
        return real(self, hidden, block, caches)

    monkeypatch.setattr(type(d), "__call__", spy)
    g = LaneDSpineGenerator(ad, d, block_size=5, copy=False, tree=tree, adaptive=True, cost=COST,
                            confidence="calibrated", auto_rows=True)
    out, _ = g.generate(prompt, max_tokens=50)
    if not _same(ad, prompt, out, base):
        return
    assert seen[0] == 16 and seen[1] < 16         # the random drafter is rarely accepted, so the draft shrinks


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tree_by_node_matches_tree_by_path(seed, monkeypatch):
    """The by-node recurrence (one step per node) gives the logits and the committed state of the
    older by-path layout, and of a plain chain along the accepted path."""
    import dspine_hybrid.hybrid_target as ht

    mx.random.seed(seed)
    m = make_target(seed)
    m.set_dtype(mx.float32)
    ad = HybridTargetAdapter(m, FakeTokenizer(), TARGET_IDS)
    prompt = mx.random.randint(0, VOCAB, (1, 15))
    toks = mx.random.randint(0, VOCAB, (9,)).tolist()
    parents = [-1, 0, 0, 1, 1, 3, 0, 6, 7]
    path = [0, 1, 3, 5]
    nxt = mx.random.randint(0, VOCAB, (1, 3))

    def run(by_node):
        monkeypatch.setattr(ht, "TREE_BY_NODE", by_node)
        ad.reset()
        ad.prefill(prompt)
        lg, _ = ad.verify_tree(toks, parents)
        mx.eval(lg)
        ad.commit_tree(path)
        after, _ = ad.verify(nxt)
        mx.eval(after)
        return lg, after

    lg_p, after_p = run(False)
    lg_n, after_n = run(True)
    tol = 1e-3 * float(mx.max(mx.abs(lg_p)).item())
    assert float(mx.max(mx.abs(lg_p - lg_n)).item()) <= tol
    assert float(mx.max(mx.abs(after_p - after_n)).item()) <= tol
    ad.reset()
    ad.prefill(prompt)
    chain, _ = ad.verify(mx.array([[toks[r] for r in path]]))
    assert float(mx.max(mx.abs(chain[0] - lg_n[0][mx.array(path)])).item()) <= tol
    if mx.metal.is_available() and mx.default_device() == mx.gpu:
        assert ht._tree_kernel_ok is not False               # on Metal the kernel passed its own check


@pytest.mark.parametrize("dk,hk,hv", [(128, 2, 4), (64, 3, 3)])
def test_tree_kernel_matches_ops(dk, hk, hv):
    """The Metal tree kernel against one ops step per node (runs only on Metal)."""
    import dspine_hybrid.hybrid_target as ht

    if not (mx.metal.is_available() and mx.default_device() == mx.gpu):
        pytest.skip("needs Metal")
    mx.random.seed(dk)
    parents = [-1, 0, 0, 1, 1, 3, 0, 6, 7, 5, 9, 9, 2]
    M, dv = len(parents), 24
    tree = ht.TreeSpec(parents)
    q = mx.random.normal((1, M, hk, dk)) * 0.1
    k = mx.random.normal((1, M, hk, dk)) * 0.1
    v = mx.random.normal((1, M, hv, dv))
    a = mx.random.normal((1, M, hv))
    b = mx.random.normal((1, M, hv))
    A_log, dt_bias = mx.random.normal((hv,)) * 0.1, mx.random.normal((hv,)) * 0.1
    state = mx.random.normal((1, hv, dv, dk)) * 0.1
    ht._tree_kernel_ok = None
    got = ht.tree_delta(q, k, v, a, b, A_log, dt_bias, state, tree)
    assert ht._tree_kernel_ok is True
    got = ht.tree_delta(q, k, v, a, b, A_log, dt_bias, state, tree)      # second call: kernel only
    ref = ht._tree_delta_ops(q, k, v, ht.compute_g(A_log, a, dt_bias), mx.sigmoid(b), state, parents)
    assert float(mx.max(mx.abs(got - ref)).item()) <= 1e-4 * float(mx.max(mx.abs(ref)).item()) + 1e-6


def test_grow_tree_takes_a_twin_subtree():
    cand = [[10, 11], [20, 21]]
    chance = [[[0.6, 0.4]] * 2, [[0.9, 0.1], [0.9, 0.1]]]
    twin_cand = [[10, 11], [30, 31]]
    twin_chance = [[[0.6, 0.4]] * 2, [[0.5, 0.5], [0.2, 0.8]]]
    tok, par, _ = grow_tree(cand, chance, 4, branch=2, twins={11: (twin_cand, twin_chance)})
    assert tok[:2] == [10, 20] and par[:2] == [-1, 0]           # the plain subtree under the first choice
    i = tok.index(11)
    kids = [t for t, p in zip(tok, par) if p == i]
    assert kids and kids[0] == 31                                # under 11: the twin table, sender = 11's slot


@pytest.mark.parametrize("twin", [1, 2])
def test_twin_pass_keeps_the_output(twin):
    """The twin drafter pass changes what is proposed, never what is committed."""
    from dspine_hybrid import twin as tw

    ad, d = _fp32(1)
    mx.random.seed(3)
    prompt = mx.random.randint(0, VOCAB - 4, (1, 14)).tolist()[0]
    base, _ = baseline_generate(ad, prompt, max_tokens=60)
    g = LaneDSpineGenerator(ad, d, block_size=5, adaptive=True, cost=COST, tree=True, copy=False, twin=twin)
    out, _ = g.generate(prompt, max_tokens=60)
    assert g.lane_stats["twin_rounds"] > 0
    assert tw._FORCE["ids"] is None
    _same(ad, prompt, out, base)


def test_twin_bridge_without_force_is_the_stock_bridge():
    from dspine_hybrid import twin as tw

    ad, d = _fp32(2)
    ctx = mx.random.normal((1, 6, d.fc.weight.shape[1])).astype(mx.bfloat16)
    block = mx.array([[5] + [d.mask_token_id] * 4], dtype=mx.int32)
    ref, ref_stash = d(ctx, block, d.make_cache())
    tw.install(d)
    caches = d.make_cache()
    out, stash = d(ctx, block, caches)
    assert mx.array_equal(out, ref) and mx.array_equal(stash["packed"], ref_stash["packed"])
    # forcing row 1's code leaves rows 0 and 1 alone and moves the rows after it
    (rows_a, _), (rows_b, _) = tw.passes(d, 5, [7, 8], 4, caches)
    assert mx.allclose(rows_a[0].astype(mx.float32), out[0, 1].astype(mx.float32), atol=1e-2)
    assert mx.allclose(rows_a[0].astype(mx.float32), rows_b[0].astype(mx.float32), atol=1e-2)
    assert not mx.allclose(rows_a[1:].astype(mx.float32), rows_b[1:].astype(mx.float32), atol=1e-3)
