"""Conversations decoded together give the tokens each gives alone."""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from dspine_hybrid.batch_gen import BatchGenerator, Conversation, DSpineProposer
from dspine_hybrid.hybrid_target import HybridTargetAdapter
from dspine_mlx.generate import DSpineGenerator, baseline_generate
from tiny import TARGET_IDS, VOCAB, FakeTokenizer, make_draft, make_target


class Adapter(HybridTargetAdapter):
    """fp32 target whose features reach the bf16 draft as bf16 (as in test_hybrid_tiny)."""

    def _forward(self, tokens, capture=True):
        logits, hidden = super()._forward(tokens, capture=capture)
        return logits, (None if hidden is None else hidden.astype(mx.bfloat16))


def _setup(seed):
    model = make_target(seed)
    model.load_weights([(k, v.astype(mx.bfloat16) if "embed_tokens" in k else v.astype(mx.float32))
                        for k, v in tree_flatten(model.parameters())])
    ad = Adapter(model, FakeTokenizer(), TARGET_IDS)
    draft, _, _ = make_draft(seed + 10)
    draft.bind(model)
    return ad, DSpineGenerator(ad, draft)


def _prompts(seed, lens):
    mx.random.seed(seed)
    return [mx.random.randint(0, VOCAB - 4, (1, n)).tolist()[0] for n in lens]


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("cost", [None, [0.0] + [0.03 + 0.002 * t for t in range(1, 17)],
                                  {B: [0.0] + [0.03 + 0.004 * B * t for t in range(1, 17)] for B in (1, 2, 3)}])
def test_greedy_batch_equals_alone(seed, cost):
    ad, gen = _setup(seed)
    prompts = _prompts(seed, [12, 31, 7, 20, 15])
    lens = [40, 25, 60, 33, 18]
    alone = [baseline_generate(ad, p, max_tokens=n)[0] for p, n in zip(prompts, lens)]
    seen = [[] for _ in prompts]
    bg = BatchGenerator(ad, DSpineProposer(gen), max_batch=3, cost=cost)   # five conversations, three slots
    convs = bg.run([Conversation(p, max_tokens=n, on_tokens=seen[i].extend)
                    for i, (p, n) in enumerate(zip(prompts, lens))])
    assert mx.default_device() != mx.cpu or [c.tokens for c in convs] == alone
    for c, s, n in zip(convs, seen, lens):
        assert c.done and c.tokens == s and len(c.tokens) == n and c.stats.rounds > 0
    assert bg.batch.size == 0 and not bg.active and not bg.waiting


def test_stop_ids_and_cancel():
    ad, gen = _setup(3)
    prompts = _prompts(3, [10, 14, 9])
    alone = [baseline_generate(ad, p, max_tokens=50)[0] for p in prompts]
    stop = alone[0][17]
    bg = BatchGenerator(ad, DSpineProposer(gen))
    convs = [Conversation(prompts[0], max_tokens=50, stop_ids=(stop,)),
             Conversation(prompts[1], max_tokens=50),
             Conversation(prompts[2], max_tokens=50)]
    convs[2].on_tokens = lambda new: len(convs[2].tokens) >= 8 and convs[2].cancel()
    bg.run(convs)
    assert 8 <= len(convs[2].tokens) < 50 and convs[0].tokens[-1] == stop or len(convs[0].tokens) == 50
    if mx.default_device() == mx.cpu:      # on Metal a rounding tie can turn a greedy token
        assert convs[0].tokens == alone[0][: alone[0].index(stop) + 1]
        assert convs[1].tokens == alone[1]
        assert convs[2].tokens == alone[2][: len(convs[2].tokens)]


def test_sampled_batch_runs_and_mixes_with_greedy():
    ad, gen = _setup(4)
    prompts = _prompts(4, [12, 12])
    alone = baseline_generate(ad, prompts[0], max_tokens=40)[0]
    mx.random.seed(7)
    bg = BatchGenerator(ad, DSpineProposer(gen), cost=[0.0] + [0.03] * 16)
    convs = bg.run([Conversation(prompts[0], max_tokens=40),
                    Conversation(prompts[1], max_tokens=40, temperature=0.8, top_k=20, top_p=0.9)])
    assert convs[0].tokens == alone                      # the greedy one is not disturbed
    assert len(convs[1].tokens) == 40 and all(0 <= t < VOCAB for t in convs[1].tokens)


def test_measure_batch_cost_and_joint_length():
    from dspine_hybrid.batch_gen import measure_batch_cost

    ad, gen = _setup(5)
    table = measure_batch_cost(ad, list(range(20, 40)), max_batch=2, max_T=6, reps=1)
    assert sorted(table) == [1, 2] and all(len(c) == 7 and min(c[1:]) > 0 for c in table.values())
    trees = measure_batch_cost(ad, list(range(20, 40)), max_batch=2, max_T=6, reps=1, tree=True)
    assert sorted(trees) == [1, 2] and all(len(c) == 7 and min(c[1:]) > 0 for c in trees.values())
    # rows that cost a lot make the round short for everyone; cheap rows use the whole block
    prompts = _prompts(5, [10, 12])
    for step, want in ((1.0, "short"), (0.0, "long")):
        cost = {2: [0.0] + [0.01 + step * t for t in range(1, 17)]}
        bg = BatchGenerator(ad, DSpineProposer(gen), max_batch=2, cost=cost)
        bg.run([Conversation(p, max_tokens=30) for p in prompts])
        widest = max(m for (b, m) in bg.row_hist if b == 2)
        assert (widest <= 3) if want == "short" else (widest >= 8), (want, bg.row_hist)


def test_kept_conversations_serve_follow_ups_and_shared_system_prompts():
    ad, gen = _setup(6)
    ad.enable_prompt_cache(True)
    ad.mark_min = 8
    system = _prompts(6, [30])[0]
    qs = _prompts(7, [9, 11, 8, 10])
    bg = BatchGenerator(ad, DSpineProposer(gen), max_batch=2, keep_sessions=3)
    first = bg.run([Conversation(system + q, max_tokens=20, mark_hint=30) for q in qs[:2]])
    assert [c.cached_tokens for c in first] == [0, 30]          # the second starts from the running first one's system prompt
    # follow-up turns of both conversations, together with a new one on the same system prompt
    follow = [system + q + c.tokens + extra for q, c, extra in zip(qs[:2], first, _prompts(8, [6, 7]))]
    prompts = follow + [system + qs[2]]
    second = bg.run([Conversation(p, max_tokens=20, mark_hint=30) for p in prompts])
    assert second[0].cached_tokens >= len(system + qs[0]) + 18
    assert second[1].cached_tokens >= 30 and second[2].cached_tokens >= 30
    assert len(bg.sessions) <= 3
    ad2, gen2 = _setup(6)
    alone = [baseline_generate(ad2, p, max_tokens=20)[0] for p in prompts]
    assert mx.default_device() != mx.cpu or [c.tokens for c in second] == alone
    # a stop in the middle of a round leaves a cache that the follow-up continues
    stop = alone[2][7]
    c = bg.run([Conversation(system + qs[3], max_tokens=20, stop_ids=(stop,), mark_hint=30)])[0]
    nxt = system + qs[3] + c.tokens + [5, 6, 7]
    d = bg.run([Conversation(nxt, max_tokens=12)])[0]
    assert d.cached_tokens >= len(system + qs[3]) + len(c.tokens) - 1
    assert mx.default_device() != mx.cpu or d.tokens == baseline_generate(ad2, nxt, max_tokens=12)[0]


DICT_COST = {B: [0.0] + [0.03 + 0.004 * B * t for t in range(1, 17)] for B in (1, 2, 3)}


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("cost", [None, DICT_COST])
def test_tree_rounds_equal_alone_and_use_side_branches(seed, cost):
    ad, gen = _setup(seed)
    prompts = _prompts(seed, [12, 31, 7, 20, 15])
    lens = [40, 25, 60, 33, 18]
    alone = [baseline_generate(ad, p, max_tokens=n)[0] for p, n in zip(prompts, lens)]
    stop = alone[1][9]
    bg = BatchGenerator(ad, DSpineProposer(gen), max_batch=3, cost=cost, tree=True, tree_from=1)
    convs = [Conversation(p, max_tokens=n) for p, n in zip(prompts, lens)]
    convs[1].stop_ids = (stop,)
    convs.append(Conversation(prompts[0], max_tokens=20, temperature=0.8, top_k=10))   # sampled: a chain
    bg.run(convs)
    alone[1] = alone[1][: alone[1].index(stop) + 1]
    assert mx.default_device() != mx.cpu or [c.tokens for c in convs[:5]] == alone
    assert all(c.done for c in convs) and len(convs[5].tokens) == 20
    if cost is None:       # whole trees; with the steep test cost the rounds are too short to branch
        assert bg.tree_stats["rounds"] > 0 and bg.tree_stats["side_tokens"] > 0
    assert bg.batch.size == 0 and not bg.active


def test_tree_table_starts_with_the_chain():
    """Following each sender's best candidate through the table gives the drafter's own chain."""
    ad, gen = _setup(2)
    prop = DSpineProposer(gen)
    prompt = mx.array([_prompts(2, [14])[0]])
    ad.reset()
    logits, hidden = ad.prefill(prompt)
    anchor = int(mx.argmax(logits[0, -1]).item())
    chain = prop.propose(prop.start(), hidden, anchor, 8, None).tokens.tolist()
    cand, chance = prop.propose_tree(prop.start(), hidden, anchor, 8)
    cand, chance = cand.tolist(), chance.tolist()
    walk, sender = [], 0
    for row in range(8):
        sender = max(range(len(chance[row][sender])), key=lambda j: chance[row][sender][j])
        walk.append(cand[row][sender])
    assert walk == chain


def test_tree_rounds_keep_sessions():
    ad, gen = _setup(4)
    ad.enable_prompt_cache(True)
    system = _prompts(4, [300])[0]
    first = system + _prompts(5, [9])[0]
    bg = BatchGenerator(ad, DSpineProposer(gen), max_batch=2, keep_sessions=2, tree=True, tree_from=1)
    a = bg.run([Conversation(first, max_tokens=24)])[0]
    follow = first + a.tokens + _prompts(6, [5])[0]
    b = bg.run([Conversation(follow, max_tokens=16)])[0]
    assert b.cached_tokens == len(first) + len(a.tokens) - 1     # the last token given out was never checked
    ref = baseline_generate(ad, follow, max_tokens=16)[0]
    assert mx.default_device() != mx.cpu or b.tokens == ref


@pytest.mark.parametrize("tree", [False, True])
def test_copy_drafts_are_lossless_and_used(tree):
    ad, gen = _setup(6)
    prompts = _prompts(6, [300, 15, 260])       # long random prompts: most tokens occurred before
    alone = [baseline_generate(ad, p, max_tokens=40)[0] for p in prompts]
    bg = BatchGenerator(ad, DSpineProposer(gen), max_batch=3, cost=DICT_COST, tree=tree, tree_from=1,
                        copy=True, copy_min=1)
    convs = [Conversation(p, max_tokens=40) for p in prompts]
    convs.append(Conversation(prompts[0], max_tokens=20, temperature=0.8, top_k=10))
    bg.run(convs)
    assert mx.default_device() != mx.cpu or [c.tokens for c in convs[:3]] == alone
    assert all(c.done for c in convs) and len(convs[3].tokens) == 20
    assert bg.copy_stats["rounds"] > 0


@pytest.mark.parametrize("window", [None, 12])
def test_one_drafter_pass_gives_each_conversation_its_own_tables(window):
    """Batched drafter tables equal the ones each conversation gets alone, over two rounds, with
    contexts of different lengths (and a sliding window that only some conversations have passed)."""
    from dspine_hybrid.adaptive import score_table
    from dspine_hybrid.batch_draft import draft_forward, score_tables

    ad, gen = _setup(3)
    if window is not None:
        draft, _, _ = make_draft(13, window=window)
        draft.bind(ad.model)
        gen = DSpineGenerator(ad, draft)
    prop = DSpineProposer(gen)
    d, tail = gen.draft, gen.tail
    mx.random.seed(4)
    H = d.config.concat_size
    lens, n = [5, 17, 9], prop.max_block - 1
    single, many = [d.make_cache() for _ in lens], [d.make_cache() for _ in lens]
    for rnd in range(2):
        hid = [mx.random.normal((1, s if rnd == 0 else 1 + b, H)).astype(mx.bfloat16) for b, s in enumerate(lens)]
        anchors = [3 + b + rnd for b in range(len(lens))]
        block = mx.array([[a] + [d.mask_token_id] * n for a in anchors], dtype=mx.int32)
        out, stash = draft_forward(d, hid, block, many)
        cand, score, rms = score_tables(tail, out, stash, anchors)
        for b in range(len(lens)):
            o1, s1 = d(hid[b], block[b : b + 1], single[b])
            _, c1, sc1, r1 = score_table(tail, o1[0, 1:], s1, anchors[b])
            assert cand[b].tolist() == c1.tolist(), (rnd, b)
            assert mx.max(mx.abs(score[b] - sc1)).item() < 2e-2 * max(mx.max(mx.abs(sc1)).item(), 1.0), (rnd, b)
            assert mx.max(mx.abs(rms[b] - r1)).item() < 2e-2 * max(mx.max(mx.abs(r1)).item(), 1.0)
            assert [c.offset for c in many[b]] == [c.offset for c in single[b]]


def test_kept_conversations_stay_inside_a_token_budget():
    """A long finished prompt is not kept next to the others: the newest stays, older ones go."""
    ad, gen = _setup(9)
    ad.enable_prompt_cache(True)
    prompts = _prompts(9, [20, 150, 25, 18])
    bg = BatchGenerator(ad, DSpineProposer(gen), max_batch=1, keep_sessions=4, keep_tokens=120)
    got = []
    for p in prompts:
        bg.run([Conversation(p, max_tokens=10)])
        got.append([len(s["seq"]) for s in bg.sessions])
    assert got[0] == [29] and got[1] == [159]            # the long one alone, although it is over the budget
    assert got[2] == [34] and got[3] == [34, 27]
    follow = bg.run([Conversation(prompts[3] + bg.sessions[-1]["seq"][18:] + [5, 6], max_tokens=6)])[0]
    assert follow.cached_tokens >= 26


@pytest.mark.parametrize("tree", [False, True])
@pytest.mark.parametrize("after", [1, 3])
def test_a_request_begun_alone_carries_on_in_the_rounds(tree, after):
    """The one-request generator stops between two rounds and the rounds take the conversation over:
    the answer is the one it gives alone, another conversation beside it too, and both are kept."""
    from dspine_hybrid.lanes import LaneDSpineGenerator

    ad, base = _setup(5)
    ad.enable_prompt_cache(True)
    gen = LaneDSpineGenerator(ad, base.draft, block_size=6, tree=tree, copy=False)
    p1, p2 = _prompts(5, [22, 13])
    alone = [baseline_generate(ad, p, max_tokens=40)[0] for p in (p1, p2)]
    ad.cache = None
    ad._drop_session()
    bg = BatchGenerator(ad, DSpineProposer(gen), max_batch=2, keep_sessions=2, tree=tree)

    rounds = []
    gen.yield_when = lambda: rounds.append(1) or len(rounds) >= after
    seen = []
    bg._load_session(p1)
    out, st = gen.generate(p1, max_tokens=40, on_tokens=seen.extend)
    gen.yield_when = None
    hand = gen.handoff
    assert hand is not None and 1 < len(out) < 40 and out == alone[0][: len(out)]
    conv = Conversation(p1, max_tokens=40, on_tokens=seen.extend)
    conv.tokens, conv.stats = list(out), st
    bg.adopt(conv, hand["anchor"], hand["hidden"], hand["state"], hand["started"])
    other = bg.submit(Conversation(p2, max_tokens=40))
    while bg.step():
        pass
    on_cpu = mx.default_device() == mx.cpu
    assert conv.done and len(conv.tokens) == 40 and (not on_cpu or conv.tokens == alone[0])
    assert seen == conv.tokens[1:]                     # the generator reports all but the first token
    assert not on_cpu or other.tokens == alone[1]
    assert sorted(len(s["seq"]) for s in bg.sessions) == [13 + 39, 22 + 39]

    # a request that ends alone is kept too, and a follow-up of it reads only what is new
    p3 = _prompts(6, [17])[0]
    bg._load_session(p3)
    out3, _ = gen.generate(p3, max_tokens=12)
    assert gen.handoff is None
    bg.keep_finished(p3, out3)
    assert ad.cache is None and len(bg.sessions[-1]["seq"]) in (17 + 11, 17 + 12)
    follow = p3 + out3 + [5, 6, 7]
    bg._load_session(follow)
    out4, _ = gen.generate(follow, max_tokens=8)
    assert ad.last_reused >= 17 + 11
    ad.cache = None
    ad._drop_session()
    assert not on_cpu or out4 == baseline_generate(ad, follow, max_tokens=8)[0]


def test_cost_table_fills_the_sizes_it_did_not_measure():
    from dspine_hybrid.batch_gen import measure_batch_cost

    ad, _ = _setup(2)
    cost = measure_batch_cost(ad, list(range(20, 40)), max_batch=5, max_T=4, reps=1, sizes=(1, 2, 4, 8))
    assert sorted(cost) == [1, 2, 3, 4, 5] and all(len(c) == 5 for c in cost.values())
    assert all(abs(cost[3][t] - (cost[2][t] + cost[4][t]) / 2) < 1e-9 for t in range(5))
    assert cost[5][1] > 0


def test_a_new_long_prompt_makes_room_before_it_is_read():
    """Kept prompts go before a new prompt is read when both would not fit the budget; one that the
    new prompt continues stays."""
    ad, gen = _setup(9)
    ad.enable_prompt_cache(True)
    a, b, c = _prompts(11, [100, 30, 90])
    bg = BatchGenerator(ad, DSpineProposer(gen), max_batch=1, keep_sessions=4, keep_tokens=160)
    bg.run([Conversation(a, max_tokens=6)])
    bg.run([Conversation(b, max_tokens=6)])
    assert [len(s["seq"]) for s in bg.sessions] == [105, 35]
    seen = []
    load = bg._load_session
    bg._load_session = lambda p: (load(p), seen.append([len(s["seq"]) for s in bg.sessions]))
    bg.run([Conversation(c, max_tokens=6)])                 # 90 new + 105 + 35 kept: the oldest goes first
    assert seen == [[35]]
    follow = bg.sessions[-1]["seq"] + [5, 6, 7]             # 98 tokens that continue a kept prompt
    out = bg.run([Conversation(follow, max_tokens=6)])[0]
    assert out.cached_tokens >= 94 and seen[-1] == [35]


@pytest.mark.parametrize("tree", [False, True])
def test_the_last_conversation_leaves_the_rounds_and_ends_alone(tree):
    """The reverse of the handoff: when one conversation is left it decodes on with the one-request
    generator, gives the answer it gives alone, and can join the rounds again."""
    from dspine_hybrid.lanes import LaneDSpineGenerator

    ad, base = _setup(7)
    ad.enable_prompt_cache(True)
    gen = LaneDSpineGenerator(ad, base.draft, block_size=6, tree=tree, copy=False)
    p1, p2, p3 = _prompts(7, [19, 11, 14])
    alone = [baseline_generate(ad, p, max_tokens=n)[0] for p, n in ((p1, 60), (p2, 12), (p3, 30))]
    ad.cache = None
    ad._drop_session()
    bg = BatchGenerator(ad, DSpineProposer(gen), max_batch=2, keep_sessions=3, tree=tree)
    seen = []
    long = bg.submit(Conversation(p1, max_tokens=60, on_tokens=seen.extend))
    bg.submit(Conversation(p2, max_tokens=12))
    while len(bg.active) != 1 or bg.waiting:
        assert bg.step()
    conv, resume = bg.release()
    assert conv is long and not bg.active and bg.batch.size == 0 and 1 < len(resume["tokens"]) < 60
    # after two rounds alone another request arrives: back into the rounds, then alone again
    rounds = []
    gen.yield_when = lambda: rounds.append(1) or len(rounds) >= 2
    out, st = gen.generate(p1, max_tokens=60, on_tokens=seen.extend, resume=resume)
    gen.yield_when = None
    hand = gen.handoff
    assert hand is not None and len(resume["tokens"]) < len(out) < 60
    conv.tokens, conv.done = list(out), False
    bg.adopt(conv, hand["anchor"], hand["hidden"], hand["state"], hand["started"])
    other = bg.submit(Conversation(p3, max_tokens=30))
    while len(bg.active) != 1 or bg.waiting:
        assert bg.step()
    last, resume = bg.release()
    assert last is conv and other.done
    out, _ = gen.generate(p1, max_tokens=60, on_tokens=seen.extend, resume=resume)
    assert gen.handoff is None
    bg.keep_finished(p1, out)
    on_cpu = mx.default_device() == mx.cpu
    assert len(out) == 60 and seen == out and (not on_cpu or out == alone[0])
    assert not on_cpu or other.tokens == alone[2]
    assert sorted(len(s["seq"]) for s in bg.sessions) == [11 + 11, 14 + 29, 19 + 59]
