"""Rounds for several conversations at once.

Every round each active conversation drafts its own block, then ONE target forward checks all
of them (``BatchTarget``) and each conversation keeps what its own draft got accepted. A
conversation gives the same tokens as it would alone: the check of one conversation does not
see the others. Conversations join after their prefill and leave when they finish, between
rounds.

The drafter sits behind ``Proposer``: anything that turns the target's hidden features and
the pending token into a block of proposed tokens (plus, for sampling, their proposal
distribution) can drive the rounds. ``DSpineProposer`` wraps this kit's drafter and tail.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import mlx.core as mx

from dspine_mlx.generate import GenStats, _sample_target
from dspine_mlx.sampler import greedy_verify, sampled_verify

from .batch_target import BatchTarget


@dataclass
class Proposal:
    tokens: mx.array                       # [n] drafted token ids
    accept: Optional[mx.array] = None      # [n] chance each is accepted given the earlier ones, or None
    cand: Optional[mx.array] = None        # sampling: candidate ids [n, K]
    q_rows: Optional[mx.array] = None      # sampling: proposal probabilities over cand [n, K]
    raw: Optional[mx.array] = None         # greedy: the drafter's own confidence [n] before any fitted map


class Proposer:
    """What a drafter must offer to drive the rounds (one state per conversation)."""

    max_block: int = 16

    def start(self):
        """A fresh per-conversation state."""
        raise NotImplementedError

    def propose(self, state, hidden: mx.array, anchor: int, n: int, sampling: Optional[dict]) -> Proposal:
        """Draft ``n`` tokens after ``anchor``. ``hidden``: the target features [1, rows, H] the
        drafter has not seen yet (the prompt's on the first call, then the rows kept by the last check)."""
        raise NotImplementedError

    def propose_tree(self, state, hidden: mx.array, anchor: int, n: int):
        """Greedy rounds with a draft tree: the drafter's table for ``n`` positions as two arrays,
        candidates [n, K] and chance [n, K senders, K] (candidate j of a row given that the row
        before took its candidate s; row 0's senders all stand for ``anchor``). Following the best
        candidate from each sender must give ``propose``'s tokens. None: this drafter has no table."""
        return None

    def propose_many(self, items: list) -> list:
        """``propose`` for several conversations of one round; ``items``: (state, hidden, anchor,
        n, sampling) each. A drafter that can run them in one pass overrides this."""
        return [self.propose(*it) for it in items]

    def propose_tree_many(self, items: list) -> list:
        """``propose_tree`` for several conversations; ``items``: (state, hidden, anchor, n) each."""
        return [self.propose_tree(*it) for it in items]

    def observe(self, state, checked: int, accepted: int) -> None:
        """Told after every check how many of the ``checked`` drafted tokens were accepted (a tree
        round reports its first-choice chain only)."""


class DSpineProposer(Proposer):
    """The DSpine drafter with its refinement tail (``gen``: any of this kit's generators)."""

    def __init__(self, gen, calibration: Optional[dict] = None, sampled_calibration: Optional[dict] = None,
                 tree_side: float = 0.6):
        from . import calibration as cal

        self.tree_side = float(tree_side)
        self.batched = True          # one drafter pass for the greedy conversations of a round

        self.draft, self.tail = gen.draft, gen.tail
        self.max_block = int(gen.block_size)
        self.calibration = calibration or cal.GREEDY
        self.sampled_calibration = sampled_calibration or cal.SAMPLED

    def start(self):
        return self.draft.make_cache()

    def propose(self, state, hidden, anchor, n, sampling):
        from . import calibration as cal
        from .adaptive import greedy_tail_with_conf, sample_chain_with_conf

        block = mx.array([[int(anchor)] + [self.draft.mask_token_id] * n], dtype=mx.int32)
        out, stash = self.draft(hidden, block, state)
        rows = out[0, 1:]
        if sampling is None:
            drafts, _, raw = greedy_tail_with_conf(self.tail, rows, stash, anchor, confidence="calibrated",
                                                   calibration=cal.IDENTITY)
            return Proposal(drafts, cal.apply(raw, self.calibration), raw=raw)
        table, cand = self.tail.sampled_table(rows, stash, anchor)
        K = int(self.draft.head_topk)
        top_k = int(sampling["top_k"])
        drafts, q_rows, raw = sample_chain_with_conf(
            table, cand, inv_temp=sampling["inv_temp"], top_k=K if top_k <= 0 else min(max(top_k, 1), K),
            top_p=sampling["top_p"], min_p=sampling["min_p"], uniforms=mx.random.uniform(shape=(n,)))
        return Proposal(drafts, cal.apply(raw, self.sampled_calibration), cand, q_rows)

    def propose_tree(self, state, hidden, anchor, n):
        from .adaptive import score_table
        from .lanes import node_chances

        block = mx.array([[int(anchor)] + [self.draft.mask_token_id] * n], dtype=mx.int32)
        out, stash = self.draft(hidden, block, state)
        _, cand, score, rms = score_table(self.tail, out[0, 1:], stash, anchor)
        return cand, node_chances(score, rms, self.calibration, self.tree_side, as_list=False)

    def _many(self, items: list):
        """One drafter pass for several conversations: their tables, cut to each one's length."""
        from .batch_draft import draft_forward, score_tables

        N = max(it[3] for it in items)
        block = mx.array([[int(it[2])] + [self.draft.mask_token_id] * N for it in items], dtype=mx.int32)
        out, stash = draft_forward(self.draft, [it[1] for it in items], block, [it[0] for it in items])
        cand, score, rms = score_tables(self.tail, out, stash, [int(it[2]) for it in items])
        return [(cand[b, : it[3]], score[b, : it[3]], rms[b, : it[3]]) for b, it in enumerate(items)]

    def propose_many(self, items: list) -> list:
        from . import calibration as cal
        from .adaptive import _chain_with_conf

        greedy = [i for i, it in enumerate(items) if it[4] is None]
        if not self.batched or len(greedy) < 2:
            return [self.propose(*it) for it in items]
        out = [None if it[4] is None else self.propose(*it) for it in items]
        for i, (cand, score, rms) in zip(greedy, self._many([items[i] for i in greedy])):
            idx = mx.argmax(score, axis=-1).astype(mx.int32)
            drafts, conf = _chain_with_conf(idx, cand, score / mx.maximum(rms, 1e-6)[:, None, None])
            raw = cal.apply(conf, cal.IDENTITY)
            out[i] = Proposal(drafts, cal.apply(raw, self.calibration), raw=raw)
        return out

    def propose_tree_many(self, items: list) -> list:
        from .lanes import node_chances

        if not self.batched or len(items) < 2:
            return [self.propose_tree(*it) for it in items]
        return [(cand, node_chances(score, rms, self.calibration, self.tree_side, as_list=False))
                for cand, score, rms in self._many(items)]


@dataclass
class Conversation:
    prompt: list
    max_tokens: int = 256
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    stop_ids: tuple = ()
    on_tokens: Optional[Callable] = None   # called with every new list of tokens, the first one too
    on_done: Optional[Callable] = None     # called with the conversation when it ends
    tokens: list = field(default_factory=list)
    stats: GenStats = field(default_factory=GenStats)
    done: bool = False
    stopped: bool = False                  # set by cancel()
    # round state
    mark_hint: Optional[int] = None        # a stable prefix length of the prompt (system prompt), see prefill
    cached_tokens: int = 0                 # prompt tokens taken from a kept conversation
    _anchor: int = 0
    _fed: Optional[list] = None            # token ids the target cache holds for this conversation
    _hid: Optional[list] = None            # their newest hidden rows, in chunks
    _snap: object = None
    _marks: Optional[list] = None
    _hidden: Optional[mx.array] = None     # target features the drafter has not seen yet
    _index: object = None                  # copy drafts: where each recent run of tokens occurred before
    _copy_len: int = 0
    _copy_rest: bool = False
    _copied: int = 0                       # tokens of this round's proposal that were copied (0: drafted)
    _state: object = None
    _t0: float = 0.0

    def cancel(self) -> None:
        """Stop after the current round (a client that went away)."""
        self.stopped = True

    @property
    def sampling(self) -> Optional[dict]:
        if self.temperature <= 0:
            return None
        return dict(inv_temp=1.0 / max(float(self.temperature), 1e-5), top_k=int(self.top_k),
                    top_p=float(self.top_p), min_p=float(self.min_p))


def measure_batch_cost(target, prompt_ids, max_batch: int = 4, max_T: int = 17, reps: int = 2,
                       tree: bool = False, sizes=None) -> dict:
    """Seconds of one check of B conversations x T rows: {B: [0, cost(1), ..., cost(max_T)]}.

    The check for several conversations is one forward of B x T rows, and what a row costs depends
    on B (kernels have row limits, attention is per conversation), so the length choice of a
    batched round needs the table for its own B. ``tree``: time tree checks (rows in a binary tree)
    instead of chains, for rounds with draft trees. ``sizes``: measure only these B (1 and
    ``max_batch`` always) and fill the others in between by straight lines, for a shorter start.
    """
    from .batch_target import BatchTarget

    ids = mx.array([list(prompt_ids)])
    tok = int(prompt_ids[-1])
    out = {}
    measured = sorted(set(range(1, max_batch + 1)) if sizes is None
                      else {1, int(max_batch)} | {int(x) for x in sizes if 1 <= int(x) <= max_batch})
    for B in measured:
        bt = BatchTarget(target)
        for _ in range(B):
            target.reset()
            target.prefill(ids, capture=False)
            bt.join()
        cost = [0.0]
        for T in range(1, max_T + 1):
            ts = []
            for r in range(reps + 1):
                t0 = time.perf_counter()
                grid = mx.full((B, T), tok, dtype=mx.int32)
                if tree and T > 2:
                    lg, h = bt.verify_tree(grid, [[-1] + [(i - 1) // 2 for i in range(1, T)]] * B)
                else:
                    lg, h = bt.verify(grid)
                mx.eval(lg, h)
                if r:
                    ts.append(time.perf_counter() - t0)
                bt.keep([0] * B)
            cost.append(sorted(ts)[len(ts) // 2])
        out[B] = cost
        del bt
        mx.clear_cache()
    target.reset()
    for B in range(1, max_batch + 1):
        if B not in out:
            lo, hi = max(k for k in measured if k < B), min(k for k in measured if k > B)
            w = (B - lo) / (hi - lo)
            out[B] = [(1 - w) * x + w * y for x, y in zip(out[lo], out[hi])]
    return out


class BatchGenerator:
    """Runs up to ``max_batch`` conversations together; more wait in a queue.

    ``cost``: ``measure_batch_cost``'s table {conversations: seconds per T rows}: one check length is
    chosen per round for all conversations together. Or the older form,
    seconds of a T-token target forward (``adaptive.measure_cost``); with it every
    conversation sizes its own check from the proposer's acceptance chances. Without it the
    whole block is checked.
    """

    def __init__(self, target, proposer: Proposer, *, max_batch: int = 4, cost: Optional[list] = None,
                 keep_sessions: int = 0, keep_tokens: int = 40000, tree: bool = False, tree_branch: int = 4, tree_from: int = 2, tree_nodes: int = 15,
                 copy: bool = False, copy_min: int = 12, copy_start: int = 4, copy_max: int = 15):
        self.adapter = target
        self.batch = BatchTarget(target)
        self.proposer = proposer
        self.max_batch = int(max_batch)
        self.cost = cost
        # greedy conversations spend their rows of the check on the most likely nodes of the
        # drafter's tree instead of its best chain (drafters with ``propose_tree``)
        self.tree = bool(tree)
        self.tree_branch = int(tree_branch)
        # trees from this many conversations in a round: alone, a conversation has rows to spare
        # and the chain was faster (Mac job ds-67: code 148 against 126 tok/s)
        self.tree_from = int(tree_from)
        # most nodes of one conversation's tree (a drafter with a short block can still fill the rows)
        self.tree_nodes = int(tree_nodes)
        self.tree_stats = dict(rounds=0, nodes=0, side_tokens=0)
        self.tree_or_chain = True                 # a tree round may check plain chains when they pay more
        self._check_ratio = dict(chain=1.0, tree=1.0)   # real check time over the cost table, per kind
        self._ratio_age = dict(chain=0, tree=0)
        self._kept = dict(chain=1.0, tree=1.0)          # tokens really kept per token promised, per kind
        self._kept_sum = dict(chain=[8.0, 8.0], tree=[8.0, 8.0])
        # how often a side branch really caught the target when the first choice missed, against
        # what the drafter's table promised: the share given to side branches follows it
        self._side_base = getattr(proposer, "tree_side", None)
        self._side_seen = self._side_told = 5.0
        self.side_scale = 1.0
        self.phase_seconds = dict(draft=0.0, pick=0.0, check=0.0, land=0.0)
        # copy drafts (see lanes.py): when the last ``copy_min`` tokens occurred earlier in the prompt
        # or the answer, the tokens that followed them there are the proposal and the drafter rests
        self.copy = bool(copy)
        self.copy_min, self.copy_start, self.copy_max = int(copy_min), max(1, int(copy_start)), int(copy_max)
        self.copy_stats = dict(rounds=0, offered=0, accepted=0)
        # set to a list to collect (raw confidences, tokens checked, tokens accepted) of every drafted
        # greedy chain round, for fitting an acceptance map (scripts/fit_acceptance.py)
        self.accept_log: Optional[list] = None
        self._copy_tried, self._copy_ok = 10.0, 9.0      # how often a copied token is accepted so far
        self.active: list[Conversation] = []
        self.waiting: list[Conversation] = []
        self.rounds = 0
        # Prompt cache between conversations: the caches of the last ``keep_sessions`` finished
        # conversations are kept, and a new prompt that starts like one of them (a follow-up turn,
        # or the same system prompt) prefills only what is new. Needs the target's prompt cache on.
        self.keep_sessions = int(keep_sessions) if getattr(target, "prompt_cache", False) else 0
        self.sessions: list = []      # newest last: dict(cache, seq, hid, snap, marks)
        # ... and at most this many tokens of them together (the newest is always kept): a kept 32K
        # prompt holds about 10 GB of a 27B target, and eight of them pushed a server past its
        # memory limit, which slowed every later request (engine bench eb-13)
        self.keep_tokens = int(keep_tokens)
        self.round_seconds = 0.0      # time spent in rounds (drafts, check, accept), prefill excluded
        self.prefill_seconds = 0.0
        self.row_hist: dict = {}      # rows per check -> rounds

    # ---- joining and leaving ---------------------------------------------------------------

    def submit(self, conv: Conversation) -> Conversation:
        self.waiting.append(conv)
        return conv

    def _emit(self, conv: Conversation, new: list) -> None:
        conv.tokens.extend(new)
        if conv._index is not None:
            conv._index.extend(new)
        if conv.on_tokens is not None:
            conv.on_tokens(new)

    def _finish(self, conv: Conversation) -> None:
        conv.done = True
        conv.stats.generated_tokens = len(conv.tokens)
        conv.stats.decode_seconds = time.perf_counter() - conv._t0
        conv._hidden = conv._state = conv._index = None
        if conv.on_done is not None:
            conv.on_done(conv)

    def _admit(self, conv: Conversation) -> None:
        ad = self.adapter
        prompt = mx.array([list(conv.prompt)], dtype=mx.int32)
        conv.stats = GenStats(prompt_tokens=int(prompt.shape[1]), block_size=self.proposer.max_block)
        t0 = time.perf_counter()
        if self.keep_sessions:
            self._load_session(list(conv.prompt))
            ad.mark_hint = conv.mark_hint
        ad.reset()
        logits, hidden = ad.prefill(prompt)
        mx.eval(logits, hidden)
        if self.keep_sessions:
            conv.cached_tokens = int(ad.last_reused)
            conv._fed, conv._snap, conv._marks = list(ad._seq), ad._snap, list(ad._marks)
            rows = ad._hid_rows()
            conv._hid = [] if rows is None else [rows]
        conv.stats.prefill_seconds = time.perf_counter() - t0
        self.prefill_seconds += conv.stats.prefill_seconds
        conv._t0 = time.perf_counter()
        s = conv.sampling
        if s is None:
            anchor = int(mx.argmax(logits[0, -1]).item())
        else:
            top_k = ad.vocab_size if s["top_k"] <= 0 else s["top_k"]
            anchor = _sample_target(logits[0, -1], s["inv_temp"], top_k, s["top_p"], s["min_p"])
        conv._anchor, conv._hidden, conv._state = anchor, hidden, self.proposer.start()
        if self.copy:
            from .lanes import CopyIndex

            conv._index = CopyIndex(self.copy_min)
            conv._index.extend(list(conv.prompt))
            conv._copy_len, conv._copy_rest = self.copy_start, False
        self._emit(conv, [anchor])
        if anchor in conv.stop_ids or len(conv.tokens) >= conv.max_tokens or conv.stopped:
            if self.keep_sessions:
                cache, ad.cache = ad.cache, None
                ad._drop_session()
                self._store(conv, cache)
            self._finish(conv)
            return
        self.batch.join()
        self.active.append(conv)

    def adopt(self, conv: Conversation, anchor: int, hidden: list, state, started: float) -> None:
        """Take over a conversation that the one-request generator began (``lanes`` handoff): the
        adapter holds its settled cache, ``conv.tokens`` what it gave out so far, ``anchor`` its last
        token (not in the cache yet), ``hidden`` the feature chunks the drafter has not seen and
        ``state`` the drafter's cache. It decodes on in the rounds from the next ``step``."""
        ad = self.adapter
        conv._anchor, conv._state, conv._t0 = int(anchor), state, started
        conv._hidden = hidden[0] if len(hidden) == 1 else mx.concatenate(hidden, axis=1)
        if self.keep_sessions:
            conv.cached_tokens = int(ad.last_reused)
            conv._fed, conv._snap, conv._marks = list(ad._seq), ad._snap, list(ad._marks)
            rows = ad._hid_rows()
            conv._hid = [] if rows is None else [rows]
        if self.copy:
            from .lanes import CopyIndex

            conv._index = CopyIndex(self.copy_min)
            conv._index.extend(list(conv.prompt) + list(conv.tokens))
            conv._copy_len, conv._copy_rest = self.copy_start, False
        self.batch.join()
        self.active.append(conv)

    def release(self):
        """The reverse of ``adopt``: the one conversation still decoding leaves the rounds and the
        adapter gets its cache. Returns (conversation, resume) with ``resume`` for the one-request
        generator's ``generate(resume=...)``."""
        if len(self.active) != 1 or self.waiting:
            raise RuntimeError("release needs exactly one conversation decoding and none waiting")
        ad, conv = self.adapter, self.active[0]
        ad.cache = None
        ad._drop_session()
        cache = self.batch.export(0, spare=True)
        self.batch.leave(0)
        del self.active[0]
        ad.cache = cache
        if conv._fed is not None:
            ad._seq, ad._hid = list(conv._fed), list(conv._hid or [])
            ad._snap, ad._marks = conv._snap, list(conv._marks or [])
        ad._verify_steps = 0
        ad.last_reused = conv.cached_tokens
        resume = dict(tokens=list(conv.tokens), anchor=conv._anchor, hidden=[conv._hidden], state=conv._state)
        conv._hidden = conv._state = conv._index = None
        conv._fed = conv._hid = conv._snap = conv._marks = None
        return conv, resume

    def keep_finished(self, prompt: list, tokens: list) -> None:
        """Keep the adapter's cache of a request the one-request generator finished, for a follow-up
        turn, as ``_retire`` does for a conversation of the rounds."""
        ad = self.adapter
        if not self.keep_sessions or ad.cache is None or ad._seq is None:
            ad.cache = None
            ad._drop_session()
            return
        ad._settle(list(prompt) + list(tokens))      # an unsettled last check: keep the rows given out
        conv = Conversation(list(prompt))
        conv._fed, conv._snap, conv._marks = list(ad._seq), ad._snap, list(ad._marks)
        rows = ad._hid_rows()
        conv._hid = [] if rows is None else [rows]
        cache, ad.cache = ad.cache, None
        ad._drop_session()
        self._store(conv, cache)

    def _load_session(self, prompt: list) -> None:
        """Give the target the kept conversation that shares the longest usable start with ``prompt``."""
        ad = self.adapter
        ad.cache = None
        ad._drop_session()
        best, best_n = None, 0
        # conversations still running count too: a new one can start from their system prompt
        running = [dict(seq=c._fed, snap=c._snap, marks=c._marks or [], live=i)
                   for i, c in enumerate(self.active) if c._fed is not None]
        for s in self.sessions + running:
            seq = s["seq"]
            c, lim = 0, min(len(seq), len(prompt))
            while c < lim and seq[c] == prompt[c]:
                c += 1
            n = c if c == len(seq) and c < len(prompt) and "live" not in s else 0
            for snap in [s["snap"]] + s["marks"]:
                if snap is not None and n < snap[0] <= c and (snap[3] is not None or snap[0] < len(prompt)):
                    n = snap[0]
            if n > best_n:
                best, best_n = s, n
        # room for the new prompt before it is read: kept prompts and the ones in work share one
        # budget, oldest kept first out (eb-17: a kept 32K prompt beside a new 32K one, 36 GB peak)
        work = len(prompt) + sum(len(c._fed or c.prompt) for c in self.active)
        dropped = False
        while self.sessions and work + sum(len(x["seq"] or ()) for x in self.sessions if x is not best) > self.keep_tokens:
            victim = next((x for x in self.sessions if x is not best), None)
            if victim is None:
                break
            self.sessions.remove(victim)
            dropped = True
        if dropped:
            del victim
            mx.clear_cache()
        if best is None:
            return
        if "live" in best:       # a copy of its cache; the restore point brings its own hidden rows
            ad.cache, ad._hid = self.batch.export(best["live"]), []
        else:
            self.sessions.remove(best)
            ad.cache = best["cache"]
            ad._hid = [] if best["hid"] is None else [best["hid"]]
        ad._seq, ad._snap, ad._marks = list(best["seq"]), best["snap"], list(best["marks"])
        ad._verify_steps = 0

    def _store(self, conv: Conversation, cache: list) -> None:
        hid = None
        if conv._hid:
            hid = conv._hid[0] if len(conv._hid) == 1 else mx.concatenate(conv._hid, axis=1)
            hid = hid[:, -self.adapter._ctx_rows:]
            mx.eval(hid)
        self.sessions.append(dict(cache=cache, seq=conv._fed, hid=hid, snap=conv._snap, marks=conv._marks or []))
        dropped = len(self.sessions) > self.keep_sessions
        del self.sessions[: -self.keep_sessions]
        while len(self.sessions) > 1 and sum(len(x["seq"] or ()) for x in self.sessions) > self.keep_tokens:
            del self.sessions[0]
            dropped = True
        if dropped:
            mx.clear_cache()
        conv._fed = conv._hid = conv._snap = conv._marks = None

    def _retire(self) -> None:
        for i in reversed(range(len(self.active))):
            if self.active[i].done:
                if self.keep_sessions:
                    self._store(self.active[i], self.batch.export(i))
                self.batch.leave(i)
                del self.active[i]

    # ---- one round -------------------------------------------------------------------------

    def _joint_len(self, props) -> list[int]:
        """One check length for the round: the rows are shared (every conversation is padded to the
        longest), so pick the length with the best expected tokens of all conversations per second
        of a check for this many conversations. Each conversation then uses every row it has."""
        B = len(props)
        cost = self._batch_cost(B)
        acc = [p.accept.tolist() for p in props]
        ns = [int(p.tokens.shape[0]) for p in props]
        run = [1.0] * B                    # chance that conversation b is still accepting
        gain = float(B)                    # every conversation gets one token whatever happens
        best, best_l = -1.0, 1
        for L in range(1, min(max(ns), len(cost) - 2) + 1):
            for b in range(B):
                if L <= ns[b]:
                    run[b] *= max(acc[b][L - 1], 1e-6)
                    gain += run[b]
            rate = gain / cost[L + 1]
            if rate > best:
                best, best_l = rate, L
        return [min(n, best_l) for n in ns]

    def _batch_cost(self, B: int) -> list:
        return self.cost.get(B) or self.cost[max(k for k in self.cost if k <= B)]

    def _land(self, b: int, conv: Conversation, accepted: list, bonus_i: int, vhidden: mx.array,
              rows: Optional[list] = None) -> int:
        """Give conversation b the tokens its check accepted; returns how many checked rows it
        keeps (``rows``: the accepted path of a tree check, row 0 first; None for a chain)."""
        accept_i = len(accepted)
        new = (accepted + [bonus_i])[: conv.max_tokens - len(conv.tokens)]
        stop_at = next((i for i, t in enumerate(new) if t in conv.stop_ids), None)
        if stop_at is not None:
            new = new[: stop_at + 1]
        ended = stop_at is not None or len(conv.tokens) + len(new) >= conv.max_tokens or conv.stopped
        # a conversation that ends here keeps only the rows of tokens it gave out, so its cache
        # is the start of a follow-up prompt
        rows_kept = 1 + min(accept_i, len(new)) if ended else accept_i + 1

        def hidden(k):
            if rows is None:
                return vhidden[b : b + 1, :k, :]
            return vhidden[b : b + 1, mx.array(rows[:k]), :]

        if conv._fed is not None:
            conv._fed.extend([conv._anchor] + accepted[: rows_kept - 1])
            conv._hid.append(hidden(rows_kept))
            if len(conv._hid) > 64:
                conv._hid = [mx.concatenate(conv._hid, axis=1)[:, -self.adapter._ctx_rows:]]
        conv.stats.rounds += 1
        conv.stats.accepted_drafts += min(accept_i, len(new))
        conv.stats.committed_tokens += len(new)
        self._emit(conv, new)
        seen = hidden(accept_i + 1)       # a round without the drafter leaves its rows for the next draft
        conv._hidden = seen if conv._hidden is None else mx.concatenate([conv._hidden, seen], axis=1)
        conv._anchor = new[-1]
        if ended:
            self._finish(conv)
        return rows_kept

    def _copy_proposal(self, conv: Conversation, n: int) -> Optional[Proposal]:
        """A proposal copied from earlier in the conversation, if its last tokens occurred before."""
        conv._copied = 0
        if conv._index is None:
            return None
        rest, conv._copy_rest = conv._copy_rest, False
        copied = [] if rest else conv._index.lookup(min(conv._copy_len, self.copy_max, n))
        if not copied:
            return None
        conv._copied = len(copied)
        tokens = mx.array(copied, dtype=mx.int32)
        accept = mx.full((len(copied),), self._copy_ok / self._copy_tried)
        if conv.sampling is None:
            return Proposal(tokens, accept)
        # a proposal with all its mass on the copied token: ordinary rejection sampling
        return Proposal(tokens, accept, cand=tokens[:, None], q_rows=mx.ones((len(copied), 1), dtype=mx.float32))

    def _draft(self, conv: Conversation, n: int) -> Proposal:
        hidden, conv._hidden = conv._hidden, None
        return self.proposer.propose(conv._state, hidden, conv._anchor, n, conv.sampling)

    def _take_hidden(self, conv: Conversation):
        hidden, conv._hidden = conv._hidden, None
        return hidden

    def _observe(self, conv: Conversation, checked: int, accepted: int) -> None:
        """After a check of ``checked`` proposed tokens of which the first ``accepted`` held."""
        if not conv._copied:
            self.proposer.observe(conv._state, checked, accepted)
            return
        cs = self.copy_stats
        cs["rounds"] += 1
        cs["offered"] += checked
        cs["accepted"] += accepted
        self._copy_tried = 0.98 * self._copy_tried + min(checked, accepted + 1)
        self._copy_ok = 0.98 * self._copy_ok + accepted
        conv._copy_len = min(conv._copy_len * 2, self.copy_max) if accepted == checked else self.copy_start
        conv._copy_rest = accepted == 0

    def _tree_round(self) -> Optional[int]:
        """A round in which greedy conversations check a draft tree. Returns the rows per
        conversation, or None when no drafter table is on offer (the caller runs a chain round)."""
        from .lanes import grow_tree

        convs = self.active
        B = len(convs)
        ph, t_a = self.phase_seconds, time.perf_counter()
        room = [min(self.proposer.max_block - 1, c.max_tokens - len(c.tokens)) for c in convs]
        if all(c.sampling is not None for c in convs) or type(self.proposer).propose_tree is Proposer.propose_tree:
            return None
        props = [self._copy_proposal(c, n) for c, n in zip(convs, room)]
        tabs = [None] * B
        want = [b for b, (c, p) in enumerate(zip(convs, props)) if c.sampling is None and p is None]
        for b, t in zip(want, self.proposer.propose_tree_many(
                [(convs[b]._state, self._take_hidden(convs[b]), convs[b]._anchor, room[b]) for b in want])):
            tabs[b] = t
        props = [p if p is not None or t is not None else self._draft(c, n)
                 for c, n, t, p in zip(convs, room, tabs, props)]
        mx.eval([x for t in tabs if t is not None for x in t] + [p.tokens for p in props if p is not None]
                + [x for p in props if p is not None for x in (p.accept, p.q_rows) if x is not None])
        t_b = time.perf_counter()
        ph["draft"] += t_b - t_a
        cost = self._batch_cost(B) if isinstance(self.cost, dict) else None
        limit = 10**9 if cost is None else len(cost) - 2
        most = min(max(room), limit)
        def grow(branch):
            """Per conversation: node tokens, parents (-1: the pending token), path chance."""
            toks, pars, paths = [], [], []
            for t, p in zip(tabs, props):
                if t is not None:
                    tok, par, path = grow_tree(t[0], t[1], min(self.tree_nodes, limit), branch)
                else:
                    tok = p.tokens.tolist()[:most]
                    par = list(range(-1, len(tok) - 1))
                    path, run = [], 1.0
                    for a in (p.accept.tolist() if p.accept is not None else [1.0] * len(tok))[: len(tok)]:
                        run *= max(a, 1e-6)
                        path.append(run)
                toks.append(tok)
                pars.append(par)
                paths.append(path)
            return toks, pars, paths

        def best_count(paths, scale, kept=1.0):
            """One node count for the round (the check is B x M rows whatever each conversation
            uses): the count with the most expected tokens per second, and that rate."""
            count = max(len(path) for path in paths)
            if cost is None:
                return count, 0.0
            gain, best = 0.0, -1.0
            for n in range(1, count + 1):
                gain += sum(path[n - 1] for path in paths if n <= len(path))
                rate = (B + kept * gain) / (cost[n + 1] * scale)     # ``kept``: how much of such promises held so far
                if rate > best:
                    best, count = rate, n
            return count, best

        tabs = [None if t is None else (t[0].tolist(), t[1].tolist()) for t in tabs]
        ratio = self._check_ratio
        toks, pars, paths = grow(self.tree_branch)
        both = cost is not None and self.tree_or_chain
        count, rate = best_count(paths, ratio["tree"], self._kept["tree"] if both else 1.0)
        if both:
            # the same tables as plain chains: taken when they promise more tokens per second, with
            # each kind of check priced at what its rounds have really cost against the table
            for kind in ratio:
                self._ratio_age[kind] += 1
                if self._ratio_age[kind] > 48:        # not tried for a while: price it like the other
                    ratio[kind], self._ratio_age[kind] = min(ratio.values()), 0
            c_toks, c_pars, c_paths = grow(1)
            c_count, c_rate = best_count(c_paths, ratio["chain"], self._kept["chain"])
            if c_rate >= rate:
                toks, pars, paths, count = c_toks, c_pars, c_paths, c_count
        counts = [min(count, len(t)) for t in toks]
        promised = sum(sum(path[:n]) for path, n, c in zip(paths, counts, convs) if c.sampling is None)
        M = 1 + max(counts)
        for b in range(B):
            toks[b], pars[b] = toks[b][: counts[b]], pars[b][: counts[b]]
        chain = all(p == i - 1 for par in pars for i, p in enumerate(par))
        grid = mx.array([[c._anchor] + tok + [0] * (M - 1 - len(tok)) for c, tok in zip(convs, toks)], dtype=mx.int32)
        t_c = time.perf_counter()
        ph["pick"] += t_c - t_b
        if chain:
            vlogits, vhidden = self.batch.verify(grid)
        else:              # filler rows hang under the row before them: checked for nothing, never kept
            vlogits, vhidden = self.batch.verify_tree(
                grid, [[-1] + [p + 1 for p in par] + list(range(len(par), M - 1)) for par in pars])
            self.tree_stats["rounds"] += 1
        want = mx.argmax(vlogits, axis=-1)
        sampled = {}
        for b, (c, p, n) in enumerate(zip(convs, props, counts)):
            s = c.sampling
            if s is not None:
                top_k = self.adapter.vocab_size if s["top_k"] <= 0 else s["top_k"]
                sampled[b] = sampled_verify(
                    vlogits[b, : n + 1], p.tokens[:n], p.cand[:n], p.q_rows[:n], inv_temp=s["inv_temp"],
                    top_k=top_k, top_p=s["top_p"], min_p=s["min_p"], u_accept=mx.random.uniform(shape=(n,)),
                    u_final=mx.random.uniform(), u_res=mx.random.uniform())
        mx.eval([want, vhidden] + [x for r in sampled.values() for x in r])
        want = want.tolist()
        t_d = time.perf_counter()
        ph["check"] += t_d - t_c
        if cost is not None:
            kind = "chain" if chain else "tree"
            self._check_ratio[kind] = 0.9 * self._check_ratio[kind] + 0.1 * (t_d - t_c) / cost[M]
            self._ratio_age[kind] = 0
        kept = []
        for b, (c, tok, par, n) in enumerate(zip(convs, toks, pars, counts)):
            if b in sampled:
                accept_i, bonus_i = int(sampled[b][0].item()), int(sampled[b][1].item())
                rows, main, depth = list(range(accept_i + 1)), accept_i, n
            else:
                kids, first, side = {}, {}, {}
                path = paths[b]
                for i, p in enumerate(par):
                    kids[(p + 1, tok[i])] = i + 1
                    if first.setdefault(p + 1, i + 1) != i + 1:      # a side branch: its promised chance
                        up = path[p] if p >= 0 else 1.0
                        miss = max(1.0 - path[first[p + 1] - 1] / max(up, 1e-9), 1e-3)
                        side[p + 1] = side.get(p + 1, 0.0) + path[i] / max(up, 1e-9) / miss
                rows, main, on_main = [0], 0, True
                while True:
                    r = rows[-1]
                    nxt = kids.get((r, want[b][r]))
                    if r in side and nxt != first[r]:                # the first choice missed here
                        self._side_told = 0.98 * self._side_told + min(side[r], 1.0) / self.side_scale
                        self._side_seen = 0.98 * self._side_seen + (nxt is not None)
                    if nxt is None:
                        break
                    if first[r] == nxt:
                        main += on_main
                    else:
                        on_main = False
                        self.tree_stats["side_tokens"] += 1
                    rows.append(nxt)
                bonus_i = want[b][rows[-1]]
                self.tree_stats["nodes"] += n
                # the drafter learns from its first-choice chain: how deep it went in this check,
                # and how far the target followed it
                depth, r = 0, 0
                while r in first:
                    r, depth = first[r], depth + 1
            self._observe(c, depth, main)
            k = self._land(b, c, [tok[r - 1] for r in rows[1:]], bonus_i, vhidden, None if chain else rows)
            kept.append(k if chain else rows[:k])
        if cost is not None and promised > 0:      # tokens the greedy conversations really got per promised token
            got = sum(len(k) - 1 if not chain else k - 1 for k, c in zip(kept, convs) if c.sampling is None)
            kind = "chain" if chain else "tree"
            self._kept_sum[kind] = [0.97 * self._kept_sum[kind][0] + max(got, 0), 0.97 * self._kept_sum[kind][1] + promised]
            self._kept[kind] = min(max(self._kept_sum[kind][0] / self._kept_sum[kind][1], 0.3), 1.5)
        if chain:
            self.batch.keep(kept)
        else:
            self.batch.keep_paths(kept)
        if self._side_base is not None:
            self.side_scale = min(max(self._side_seen / self._side_told, 0.2), 1.0 / max(self._side_base, 1e-6))
            self.proposer.tree_side = self._side_base * self.side_scale
        ph["land"] += time.perf_counter() - t_d
        return M

    def step(self) -> bool:
        """Admit waiting conversations, run one round for the active ones. False when idle."""
        while self.waiting and len(self.active) < self.max_batch:
            self._admit(self.waiting.pop(0))
        if not self.active:
            return bool(self.waiting)
        from .adaptive import choose_len

        t_round = time.perf_counter()
        M = self._tree_round() if self.tree and len(self.active) >= self.tree_from else None
        if M is None:
            M = self._chain_round(choose_len)
        self.rounds += 1
        self.row_hist[(len(self.active), M)] = self.row_hist.get((len(self.active), M), 0) + 1
        self.round_seconds += time.perf_counter() - t_round
        if self.rounds % 64 == 0:
            mx.clear_cache()
        self._retire()
        return bool(self.active or self.waiting)

    def _chain_round(self, choose_len) -> int:
        ph, t_a = self.phase_seconds, time.perf_counter()
        props, counts = [], []
        room = [min(self.proposer.max_block - 1, conv.max_tokens - len(conv.tokens)) for conv in self.active]
        props = [self._copy_proposal(conv, n) for conv, n in zip(self.active, room)]
        want = [b for b, p in enumerate(props) if p is None]
        for b, p in zip(want, self.proposer.propose_many(
                [(self.active[b]._state, self._take_hidden(self.active[b]), self.active[b]._anchor, room[b],
                  self.active[b].sampling) for b in want])):
            props[b] = p
        mx.eval([p.tokens for p in props] + [p.accept for p in props if p.accept is not None]
                + [p.q_rows for p in props if p.q_rows is not None])
        t_b = time.perf_counter()
        ph["draft"] += t_b - t_a
        if isinstance(self.cost, dict) and all(p.accept is not None for p in props):
            counts = self._joint_len(props)
        else:
            for conv, p in zip(self.active, props):
                n = int(p.tokens.shape[0])
                if self.cost is not None and not isinstance(self.cost, dict) and p.accept is not None:
                    n = choose_len(p.accept.tolist(), self.cost, 1.0, min(n, len(self.cost) - 2))
                counts.append(n)
        M = 1 + max(counts)
        rows = []
        for conv, p, n in zip(self.active, props, counts):
            row = mx.concatenate([mx.array([conv._anchor], dtype=mx.int32), p.tokens[:n]])
            if n + 1 < M:      # filler rows: checked for nothing, never kept
                row = mx.concatenate([row, mx.zeros((M - n - 1,), dtype=mx.int32)])
            rows.append(row)
        t_c = time.perf_counter()
        ph["pick"] += t_c - t_b
        vlogits, vhidden = self.batch.verify(mx.stack(rows))
        results = []
        for b, (conv, p, n) in enumerate(zip(self.active, props, counts)):
            drafts = p.tokens[:n]
            s = conv.sampling
            if s is None:
                results.append(greedy_verify(vlogits[b, : n + 1], drafts))
            else:
                top_k = self.adapter.vocab_size if s["top_k"] <= 0 else s["top_k"]
                results.append(sampled_verify(
                    vlogits[b, : n + 1], drafts, p.cand[:n], p.q_rows[:n], inv_temp=s["inv_temp"], top_k=top_k,
                    top_p=s["top_p"], min_p=s["min_p"], u_accept=mx.random.uniform(shape=(n,)),
                    u_final=mx.random.uniform(), u_res=mx.random.uniform()))
        mx.eval([x for r in results for x in r] + [vhidden])
        if self.accept_log is not None:
            for conv, p, n, (accept, _) in zip(self.active, props, counts, results):
                if p.raw is not None and not conv._copied:
                    self.accept_log.append((p.raw[:n].tolist(), n, int(accept.item())))
        t_d = time.perf_counter()
        ph["check"] += t_d - t_c
        kept = []
        for b, (conv, p, n, (accept, bonus)) in enumerate(zip(self.active, props, counts, results)):
            accept_i, bonus_i = int(accept.item()), int(bonus.item())
            self._observe(conv, n, accept_i)
            kept.append(self._land(b, conv, p.tokens[:n].tolist()[:accept_i], bonus_i, vhidden))
        self.batch.keep(kept)
        ph["land"] += time.perf_counter() - t_d
        return M

    def run(self, convs: list) -> list:
        """Run ``convs`` to the end; returns them (tokens in ``conv.tokens``)."""
        for c in convs:
            self.submit(c)
        while self.step():
            pass
        return convs
