"""Copy drafts and draft trees on top of DSpine (both still checked by the target).

Two ideas taken from engines that serve DFlash-family drafters (TensorFold's context copy and
best-first draft tree, Apache-2.0; re-implemented here, no code copied):

* Copy drafts. When the last ``copy_min`` tokens of the text so far also occur earlier in the
  prompt or the answer, the tokens that followed them there are proposed as the draft and the
  drafter is not run for that round. Quoting, editing code and repeating tool output are mostly
  copying, so these rounds are long and cheap. The proposal starts short and doubles while the
  target keeps accepting all of it.

* Draft tree (greedy only). The drafter's refinement table holds, for every drafted position,
  16 candidates scored against each candidate of the position before. Instead of only the best
  chain, the rows of a verify pass are spent on the most likely nodes of that tree, so a round
  whose first choice misses at position i can still continue through a sibling. The target
  checks the whole tree in one forward (``HybridTargetAdapter.verify_tree``).

Neither changes the output. A greedy round commits exactly the target's argmax tokens, and a
sampled copy round is ordinary rejection sampling against a one-point proposal, so committed
tokens follow the target's filtered distribution. With both switched off this generator runs the
same rounds, with the same random draws, as ``DSpineGenerator`` / ``AdaptiveDSpineGenerator``.
"""

from __future__ import annotations

import heapq
import math
import time

import mlx.core as mx

from dspine_mlx.generate import DSpineGenerator, GenStats, _sample_target
from dspine_mlx.sampler import greedy_verify, sampled_verify

from . import calibration as _cal
from .adaptive import CONFIDENCE, choose_len, context_cost, greedy_tail_with_conf, sample_chain_with_conf, score_table


class CopyIndex:
    """Latest earlier occurrence of every ``n``-token window of a growing token list."""

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("copy window must be at least 1 token")
        self.n = int(n)
        self.ids: list[int] = []
        self._ends: dict[tuple, int] = {}
        self._indexed = self.n - 1      # windows ending before this index are in the map

    def extend(self, tokens) -> None:
        self.ids.extend(int(t) for t in tokens)

    def lookup(self, max_len: int) -> list[int]:
        """Tokens that followed the latest earlier occurrence of the last ``n`` tokens."""
        ids, n = self.ids, self.n
        last = len(ids) - 1
        if last < n or max_len < 1:
            return []
        ends = self._ends
        for e in range(self._indexed, last):
            ends[tuple(ids[e - n + 1 : e + 1])] = e
        self._indexed = max(self._indexed, last)
        j = ends.get(tuple(ids[last - n + 1 :]))
        if j is None:
            return []
        return ids[j + 1 : j + 1 + max_len]


def spread_chances(raw: mx.array, p_top: mx.array, side: float) -> mx.array:
    """``raw`` [n, senders, K] chances with each row's best candidate set to ``p_top`` [n, senders]
    and the others sharing ``side`` of what is left in proportion to their raw chance."""
    top = mx.max(raw, axis=-1)
    rest = float(side) * (1.0 - p_top) / mx.maximum(1.0 - top, 1e-6)
    return mx.where(raw >= top[..., None], p_top[..., None], raw * rest[..., None])


def node_chances(score: mx.array, rms: mx.array, calibration: dict | None, side: float, as_list: bool = True):
    """Chance of every candidate given its sender, as nested lists [row][sender][candidate].

    The raw chance is the softmax of score / write scale over the K candidates (Tencent's
    confidence feature). With ``calibration`` the row's best candidate gets the fitted
    acceptance probability for its depth, and the others share ``side`` of what is left in
    proportion to their raw chance (the rest is the target picking something outside the K).
    """
    raw = mx.softmax(score / mx.maximum(rms, 1e-6)[:, None, None], axis=-1)
    if calibration is not None:
        n = int(raw.shape[0])
        top = mx.max(raw, axis=-1)                                   # [n, K]
        off = calibration["offset"]
        offset = mx.array([off[min(d, len(off) - 1)] for d in range(n)], dtype=mx.float32)[:, None]
        eps = float(calibration["eps"])
        upper = math.log1p(-eps)
        log_p = mx.minimum(mx.log(mx.maximum(top, eps)), upper)
        log_q = mx.minimum(mx.log(mx.maximum(1.0 - top, eps)), upper)
        p_top = mx.sigmoid((log_p - log_q) * float(calibration["scale"]) + offset)
        raw = spread_chances(raw, p_top, side)
    return raw.tolist() if as_list else raw


def grow_tree(cand: list, chance: list, max_nodes: int, branch: int = 4, twins: dict | None = None):
    """Best-first tree over the drafter's table.

    cand[row][j] is a token id, chance[row][sender][j] its chance after the sender. Returns
    (tokens, parents, senders, path chance) per node in the order taken; ``parents`` index
    earlier nodes, -1 is the pending token. Every prefix of the result is the most likely set of
    that many nodes, and following first children reproduces the drafter's greedy chain.

    ``twins`` maps a first token to its own (cand, chance) tables from a twin pass: the subtree
    under that first token is grown from them instead (rows 1 and later).
    """
    rows = len(cand)
    tokens: list[int] = []
    parents: list[int] = []
    path: list[float] = []
    open_: list = []
    tick = 0
    tables = [(cand, chance)]

    def expand(parent: int, row: int, sender: int, p: float, tab: int):
        nonlocal tick
        if row >= rows:
            return
        ch = tables[tab][1][row][sender]
        order = sorted(range(len(ch)), key=lambda j: (-ch[j], j))[:branch]
        for rank, j in enumerate(order):
            if rank and ch[j] <= 0.0:
                break
            # the row's best child first among equals, so the greedy chain is never displaced
            heapq.heappush(open_, (-(p * ch[j]), rank, tick, parent, row, j, tab))
            tick += 1

    expand(-1, 0, 0, 1.0, 0)
    while open_ and len(tokens) < max_nodes:
        neg, _, _, parent, row, j, tab = heapq.heappop(open_)
        tok = int(tables[tab][0][row][j])
        tokens.append(tok)
        parents.append(parent)
        path.append(-neg)
        if row == 0 and twins and tok in twins:
            t_cand, t_chance = twins[tok]
            if rows > 1 and tok in t_cand[0]:
                tables.append((t_cand, t_chance))
                tab, j = len(tables) - 1, t_cand[0].index(tok)
        expand(len(tokens) - 1, row + 1, j, -neg, tab)
    return tokens, parents, path


def choose_nodes(path: list[float], cost: list[float], max_nodes: int) -> int:
    """How many tree nodes to verify: the count with the best expected tokens per second."""
    best, best_n, gain = -1.0, 1, 1.0
    for n in range(1, min(max_nodes, len(path), len(cost) - 2) + 1):
        gain += path[n - 1]
        rate = gain / cost[n + 1]
        if rate > best:
            best, best_n = rate, n
    return best_n


class LaneDSpineGenerator(DSpineGenerator):
    """DSpine rounds with optional copy drafts and a draft tree.

    block_size: verify block when not adaptive (the drafter proposes block_size - 1 tokens).
    adaptive + cost: the drafter proposes its full block (``draft_block``) and the number of
      rows to verify is chosen from its confidence and the measured target cost; this covers
      greedy decoding, and sampling when ``confidence == "calibrated"``.
    copy: propose context copies after ``copy_min`` matching tokens (greedy and sampled).
    tree: verify a best-first tree instead of a chain (greedy; at most ``tree_nodes`` nodes).
    """

    def __init__(self, target, draft, *, block_size: int | None = None, adaptive: bool = False,
                 cost: list[float] | None = None, alpha: float = 1.0, draft_block: int = 16,
                 confidence: str = "vocab", calibration: dict | None = None,
                 sampled_calibration: dict | None = None,
                 copy: bool = True, copy_min: int = 12, copy_start: int = 4, copy_max: int = 15,
                 tree: bool = False, tree_nodes: int | None = None, tree_branch: int = 4,
                 tree_side: float = 0.6, tree_copy: int = 0, tree_copy_min: int = 4,
                 auto_rows: bool = False, live_cost: bool = False,
                 full_from: int | None = None, twin: int = 0, twin_below: float = 1.0, **kw):
        super().__init__(target, draft, block_size=block_size, **kw)
        if adaptive and cost is None:
            raise ValueError("adaptive needs the measured target cost")
        if confidence not in CONFIDENCE:
            raise ValueError(f"confidence must be one of {CONFIDENCE}")
        if tree and not hasattr(target, "verify_tree"):
            raise ValueError("the draft tree needs HybridTargetAdapter.verify_tree")
        self.fixed_block = self.block_size
        self.adaptive = bool(adaptive)
        self.cost = cost
        self.alpha = float(alpha)
        self.draft_block = int(draft_block)
        self.confidence = confidence
        self.calibration = calibration
        self.sampled_calibration = sampled_calibration
        self.copy = bool(copy)
        self.copy_min = int(copy_min)
        self.copy_start = max(1, int(copy_start))
        self.copy_max = max(self.copy_start, int(copy_max))
        self.tree = bool(tree)
        self.tree_nodes = tree_nodes
        self.tree_branch = int(tree_branch)
        self.tree_side = float(tree_side)
        # tree rounds: also check up to tree_copy tokens that followed the latest earlier occurrence
        # of the last tree_copy_min tokens, as a branch of the same tree (0 = off)
        self.tree_copy = int(tree_copy)
        self.tree_copy_min = int(tree_copy_min)
        # adaptive rounds: draft only about twice as many rows as recent rounds chose to verify
        self.auto_rows = bool(auto_rows)
        # with a flat cost above a few rows (verify kernels), a chosen length of full_from or more
        # becomes the whole block: the extra rows are nearly free
        self.full_from = full_from
        # adaptive rounds: add what attention costs at this prompt's length to the cost table (a
        # many-row check is much dearer at 32k than on the short prompt the table was measured on)
        self.live_cost = bool(live_cost)
        # tree rounds: a second drafter pass that knows the first token, for the ``twin`` most likely
        # first tokens, when the best first token's chance is below ``twin_below`` (see twin.py)
        self.twin = int(twin) if tree else 0
        self.twin_below = float(twin_below)
        if self.twin:
            from . import twin as _twin

            _twin.install(draft)
        # A server sets ``yield_when`` to stop a generation between two rounds (another request
        # arrived): ``generate`` then returns what it has, and ``handoff`` holds what the batched
        # rounds need to carry on from the same point (pending token, features the drafter has
        # not seen, drafter state). The target cache is settled at that point.
        self.yield_when = None
        self.handoff: dict | None = None
        self.len_hist: dict[int, int] = {}
        self.lane_stats: dict[str, int] = {}
        self.accept_hist: dict[int, int] = {}   # drafter rounds by number of accepted drafts

    def _set_draft_block(self, block: int) -> None:
        if int(self.draft.block_size) != int(block):
            self.draft.set_block_size(block)

    def generate(self, prompt_tokens, *, max_tokens: int = 256, temperature: float = 0.0, top_k: int = 0,
                 top_p: float = 1.0, min_p: float = 0.0, stop_ids=(), on_tokens=None,
                 resume: dict | None = None) -> tuple[list[int], GenStats]:
        """``resume``: carry on a conversation instead of reading the prompt (the reverse of
        ``handoff``): dict(tokens=the answer so far, anchor=its last token, hidden=feature chunks the
        drafter has not seen, state=the drafter's cache); the target cache holds the prompt and the
        answer up to the token before ``anchor``. Returns the whole answer; ``on_tokens`` gets the new part."""
        prompt = prompt_tokens if isinstance(prompt_tokens, mx.array) else mx.array(
            [list(prompt_tokens)], dtype=mx.int32)
        if prompt.ndim == 1:
            prompt = prompt[None]
        stop = set(int(x) for x in stop_ids)
        greedy = temperature <= 0
        inv_temp = 1.0 / max(float(temperature), 1e-5)
        K = int(self.draft.head_topk)
        chain_top_k = K if top_k <= 0 else min(max(int(top_k), 1), K)
        target_top_k = self.target.vocab_size if top_k <= 0 else int(top_k)
        calibrated = self.confidence == "calibrated"
        # which rounds pick their own length; the rest use the fixed block
        adaptive = self.adaptive and (greedy or calibrated)
        tree = self.tree and greedy
        block_T = self.draft_block if adaptive else self.fixed_block
        self._set_draft_block(block_T)
        sampled_cal = self.sampled_calibration or _cal.SAMPLED
        tree_cal = (self.calibration or _cal.GREEDY) if calibrated else None

        self.accept_hist = {}
        self.handoff = None
        # seconds per phase; "draft" is only separate in rounds that read the drafts before the check
        # (adaptive and tree rounds), otherwise it is part of "verify"
        ph = self.phase_seconds = dict(draft=0.0, pick=0.0, verify=0.0, other=0.0)
        ls = self.lane_stats = dict(draft_rounds=0, copy_rounds=0, copy_offered=0, copy_accepted=0,
                                    tree_rounds=0, tree_nodes=0, tree_side_tokens=0,
                                    graft_nodes=0, graft_tokens=0, twin_rounds=0)
        stats = GenStats(prompt_tokens=int(prompt.shape[1]), block_size=block_T)
        t0 = time.perf_counter()
        if resume is None:
            self.target.reset()
            logits, hidden = self.target.prefill(prompt)
            mx.eval(logits, hidden)
            stats.prefill_seconds = time.perf_counter() - t0

            t0 = time.perf_counter()
            if greedy:
                anchor = int(mx.argmax(logits[0, -1]).item())
            else:
                anchor = _sample_target(logits[0, -1], inv_temp, target_top_k, top_p, min_p)
            tokens = [anchor]
        else:
            anchor, tokens = int(resume["anchor"]), list(resume["tokens"])
        done = anchor in stop or len(tokens) >= max_tokens
        index = None
        if self.copy:
            index = CopyIndex(self.copy_min)
            index.extend(prompt[0].tolist())
            index.extend(tokens)
        graft_index = None
        if tree and self.tree_copy > 0:
            graft_index = CopyIndex(self.tree_copy_min)
            graft_index.extend(prompt[0].tolist())
            graft_index.extend(tokens)
        copy_len = self.copy_start
        cost_now = self.cost
        if self.live_cost and adaptive and cost_now is not None:
            cost_now = context_cost(self.target, cost_now, int(prompt.shape[1]) + len(tokens) - 1)
        self.cost_used = cost_now
        draft_rows = block_T - 1          # rows the drafter proposes; shrinks with auto_rows
        copy_rest = False                 # a copy that missed at once: let the drafter take a round
        if resume is None:
            pend = [hidden]               # target features the drafter has not seen yet
            draft_caches = self.draft.make_cache()
        else:
            pend, draft_caches = list(resume["hidden"]), resume["state"]

        t_end = time.perf_counter()
        while not done:
            t_a = t_end
            remaining = max_tokens - len(tokens)
            round_T = min(block_T, remaining + 1, draft_rows + 1)
            if round_T < 2:
                break
            copied = []
            if index is not None and not copy_rest:
                copied = index.lookup(min(copy_len, self.copy_max, remaining))
            copy_rest = False
            rows = None                   # set by a tree round: the accepted path
            t_tree = None                 # a tree round: (drafts ready, tree chosen, tree checked)
            tree_n = None                 # nodes a tree round chose to verify
            q_rows = cand = None
            if copied:
                L = len(copied)
                drafts = mx.array(copied, dtype=mx.int32)
                if not greedy:            # a proposal that puts all its mass on the copied token
                    cand = drafts[:, None]
                    q_rows = mx.ones((L, 1), dtype=mx.float32)
            else:
                ctx = pend[0] if len(pend) == 1 else mx.concatenate(pend, axis=1)
                pend = []
                block = mx.array([[anchor] + [self.draft.mask_token_id] * (round_T - 1)], dtype=mx.int32)
                out, stash = self.draft(ctx, block, draft_caches)
                drows = out[0, 1:]
                L = round_T - 1
                ls["draft_rounds"] += 1
                if tree:
                    _, cand_t, score, rms = score_table(self.tail, drows, stash, anchor)
                    chance = node_chances(score, rms, tree_cal, self.tree_side)
                    cand_l = cand_t.tolist()
                    twins = None
                    if self.twin and L >= 2:
                        first = chance[0][0]
                        order = sorted(range(len(first)), key=lambda j: (-first[j], j))[: self.twin]
                        if first[order[0]] < self.twin_below:
                            from . import twin as _twin

                            ftoks = [int(cand_l[0][j]) for j in order]
                            twins = {}
                            for tok, (rows_b, stash_b) in zip(ftoks, _twin.passes(self.draft, anchor, ftoks, L, draft_caches)):
                                _, cand_b, score_b, rms_b = score_table(self.tail, rows_b, stash_b, anchor)
                                twins[tok] = (cand_b.tolist(), node_chances(score_b, rms_b, tree_cal, self.tree_side))
                            ls["twin_rounds"] += 1
                    t_1 = time.perf_counter()
                    most = min(self.tree_nodes or (block_T - 1), remaining, len(cost_now) - 2 if self.adaptive else 10**9)
                    t_tok, t_par, t_path = grow_tree(cand_l, chance, most, self.tree_branch, twins)
                    n = choose_nodes(t_path, cost_now, most) if self.adaptive else min(len(t_tok), self.fixed_block - 1, most)
                    t_tok, t_par = t_tok[:n], t_par[:n]
                    graft: set[int] = set()
                    if graft_index is not None:
                        room = (len(cost_now) - 2 - n) if self.adaptive else self.tree_copy
                        child = {(p, t): i for i, (t, p) in enumerate(zip(t_tok, t_par))}
                        par = -1
                        for tok in graft_index.lookup(min(self.tree_copy, remaining)):
                            i = child.get((par, tok))
                            if i is None:
                                if room <= 0:
                                    break
                                t_tok.append(tok)
                                t_par.append(par)
                                i = len(t_tok) - 1
                                graft.add(i)
                                room -= 1
                            par = i
                        n = len(t_tok)
                        ls["graft_nodes"] += len(graft)
                    tree_n = n
                    self.len_hist[n] = self.len_hist.get(n, 0) + 1
                    t_2 = time.perf_counter()
                    ph["pick"] += t_2 - t_1
                    if all(p == i - 1 for i, p in enumerate(t_par)):
                        L = n
                        drafts = mx.array(t_tok, dtype=mx.int32)
                    else:
                        ls["tree_rounds"] += 1
                        ls["tree_nodes"] += n
                        vlogits, vhidden = self.target.verify_tree([anchor] + t_tok, [-1] + [p + 1 for p in t_par])
                        want = mx.argmax(vlogits[0], axis=-1)
                        mx.eval(want, vhidden)
                        want = want.tolist()
                        t_tree = time.perf_counter()
                        kids: dict[tuple[int, int], int] = {}
                        first: dict[int, int] = {}
                        for i, p in enumerate(t_par):
                            kids[(p + 1, t_tok[i])] = i + 1
                            first.setdefault(p + 1, i + 1)
                        rows = [0]
                        while (nxt := kids.get((rows[-1], want[rows[-1]]))) is not None:
                            if first[rows[-1]] != nxt:
                                ls["tree_side_tokens"] += 1
                            if nxt - 1 in graft:
                                ls["graft_tokens"] += 1
                            rows.append(nxt)
                        accept_i = len(rows) - 1
                        bonus_i = int(want[rows[-1]])
                        accepted = [t_tok[r - 1] for r in rows[1:]]
                elif greedy and adaptive:
                    drafts, _, conf = greedy_tail_with_conf(self.tail, drows, stash, anchor,
                                                          confidence=self.confidence, calibration=self.calibration)
                    mx.eval(drafts, conf)
                    L = choose_len(conf.tolist(), cost_now, self.alpha, round_T - 1)
                    if self.full_from and L >= self.full_from:
                        L = round_T - 1
                    self.len_hist[L] = self.len_hist.get(L, 0) + 1
                    drafts = drafts[:L]
                elif greedy:
                    drafts, cand = self.tail.greedy(drows, stash, anchor)
                    mx.async_eval(drafts)
                elif adaptive:
                    table, cand = self.tail.sampled_table(drows, stash, anchor)
                    drafts, q_rows, raw = sample_chain_with_conf(
                        table, cand, inv_temp=inv_temp, top_k=chain_top_k, top_p=top_p, min_p=min_p,
                        uniforms=mx.random.uniform(shape=(round_T - 1,)))
                    conf = _cal.apply(raw, sampled_cal)
                    mx.eval(drafts, q_rows, conf)
                    L = choose_len(conf.tolist(), cost_now, 1.0, round_T - 1)
                    if self.full_from and L >= self.full_from:
                        L = round_T - 1
                    self.len_hist[L] = self.len_hist.get(L, 0) + 1
                    drafts, q_rows, cand = drafts[:L], q_rows[:L], cand[:L]
                else:
                    drafts, q_rows, cand = self.tail.sampled(
                        drows, stash, anchor, inv_temp=inv_temp, top_k=chain_top_k, top_p=top_p,
                        min_p=min_p, uniforms=mx.random.uniform(shape=(round_T - 1,)))
                    mx.async_eval(drafts, q_rows, cand)

            t_b = time.perf_counter()
            if rows is None:              # a chain: the usual verify
                verify_tokens = mx.concatenate([mx.array([[anchor]], dtype=mx.int32), drafts[None]], axis=1)
                vlogits, vhidden = self.target.verify(verify_tokens)
                if greedy:
                    accept, bonus = greedy_verify(vlogits[0], drafts)
                else:
                    accept, bonus = sampled_verify(
                        vlogits[0], drafts, cand, q_rows, inv_temp=inv_temp, top_k=target_top_k,
                        top_p=top_p, min_p=min_p, u_accept=mx.random.uniform(shape=(L,)),
                        u_final=mx.random.uniform(), u_res=mx.random.uniform())
                mx.eval(accept, bonus, drafts, vhidden)
                accept_i = int(accept.item())
                bonus_i = int(bonus.item())
                accepted = drafts.tolist()[:accept_i]
            t_c = time.perf_counter()
            if not copied:
                self.accept_hist[accept_i] = self.accept_hist.get(accept_i, 0) + 1
            if self.auto_rows and adaptive and not copied:
                used = L if tree_n is None else tree_n               # rows this round chose to verify
                offered = round_T - 1
                if used >= offered - 1:
                    draft_rows = min(block_T - 1, max(offered * 2, 4))
                else:
                    draft_rows = max(4, min(offered, 2 * used + 2))
            if copied:
                ls["copy_rounds"] += 1
                ls["copy_offered"] += L
                ls["copy_accepted"] += accept_i
                copy_len = min(copy_len * 2, self.copy_max) if accept_i == L else self.copy_start
                copy_rest = accept_i == 0

            new_tokens = (accepted + [bonus_i])[: max_tokens - len(tokens)]
            eos_at = next((i for i, t in enumerate(new_tokens) if t in stop), None)
            if eos_at is not None:
                new_tokens = new_tokens[: eos_at + 1]
                done = True
            if rows is not None:          # the tree left the cache untouched: add the accepted path
                self.target.commit_tree(rows)
            if not new_tokens:
                break
            tokens.extend(new_tokens)
            if index is not None:
                index.extend(new_tokens)
            if graft_index is not None:
                graft_index.extend(new_tokens)
            stats.rounds += 1
            stats.accepted_drafts += min(accept_i, len(new_tokens))
            stats.committed_tokens += len(new_tokens)
            if on_tokens is not None:
                on_tokens(new_tokens)
            if len(tokens) >= 256 and len(tokens) % 256 < len(new_tokens):
                mx.clear_cache()
            if len(tokens) >= max_tokens:
                break
            if rows is not None:
                pend.append(vhidden[:, mx.array(rows, dtype=mx.int32), :])
            else:
                self.target.trim(L - accept_i)
                pend.append(vhidden[:, : accept_i + 1, :])
            anchor = new_tokens[-1]
            t_end = time.perf_counter()
            if t_tree is not None:        # the tree was checked inside the draft branch
                ph["draft"] += t_2 - t_a - (t_2 - t_1)
                ph["verify"] += t_tree - t_2
            else:
                ph["draft"] += t_b - t_a - ((t_2 - t_1) if tree and not copied else 0.0)
                ph["verify"] += t_c - t_b
            ph["other"] += t_end - t_c
            if not done and self.yield_when is not None and self.yield_when():
                self.handoff = dict(anchor=anchor, hidden=pend, state=draft_caches, started=t0)
                break

        stats.generated_tokens = len(tokens)
        stats.decode_seconds = time.perf_counter() - t0
        stats.peak_memory_gb = mx.get_peak_memory() / 1e9
        return tokens, stats
