"""Greedy DSpine with a verify length chosen every round.

The drafter always proposes its full block (16). Before the target checks it,
we pick how many drafted tokens to verify, L, by maximising

    (1 + sum_{j<=L} prod_{i<=j} c_i ** alpha) / cost(L + 1)

where c_i is the drafter's confidence in drafted token i (softmax of the
refinement scores over its K candidates, at the chosen chain state) and
cost(T) is the measured time of a T-token target forward on this machine.
Greedy verification of any prefix is still exact, so the output is unchanged;
only how much work each round does changes.
"""

from __future__ import annotations

import time

import mlx.core as mx

from dspine_mlx.generate import DSpineGenerator, GenStats
from dspine_mlx.sampler import greedy_verify


def _chain_with_conf(idx: mx.array, cand: mx.array, score: mx.array):
    """Like dspine_mlx.propose._chain, also returning the chosen entry's softmax prob."""
    n = cand.shape[0]
    prob = mx.softmax(score, axis=-1)  # [n,K,K]
    j = idx[0, 0]
    out, conf = [], []
    prev = mx.array(0)
    for i in range(n):
        out.append(mx.take(cand[i], j))
        conf.append(prob[i, prev, j] if i else prob[0, 0, j])
        if i + 1 < n:
            prev = j
            j = mx.take(idx[i + 1], j)
    return mx.stack(out).astype(mx.int32), mx.stack(conf)


def greedy_tail_with_conf(tail, rows: mx.array, stash: dict, anchor: int, full_vocab: bool = True):
    """dspine_mlx RefinementTail.greedy, plus per-token confidence.

    full_vocab=True: confidence = the drafter head's full-vocabulary softmax probability of the
    drafted token (same logits the candidate search uses, so no extra matmul). False: softmax of
    the refinement scores over the K candidates only (tends to sit near 1, i.e. uninformative).
    """
    from dspine_mlx.propose import topk_ids

    n = rows.shape[0]
    raw, rms, q = tail._split_stash(stash, n)
    logits = tail.head.matmul(rows)
    cand = topk_ids(logits, tail.k)
    head_rows = tail.head.rows(cand).astype(mx.float32)
    base = mx.sum((raw[:, None, :] * tail.norm_w) * head_rows, axis=-1)
    anchor_arr = mx.full((1, tail.k), int(anchor), dtype=mx.int32)
    sid = mx.concatenate([anchor_arr, cand[:-1]], axis=0) if n > 1 else anchor_arr
    tsv_sid = tail.tsv[sid].astype(mx.float32)
    tvg = tsv_sid[..., : tail.rank]
    tvv = tsv_sid[..., tail.rank:]
    m = mx.sigmoid(q[:, None, :] + tvg + tail.gate_b) * tvv
    tg_cand = tail.tg[cand].astype(mx.float32)
    s = mx.matmul(m, tg_cand.swapaxes(1, 2))
    score = base[:, None, :] + rms[:, None, None] * s  # [n, prev K, K]
    idx = mx.argmax(score, axis=-1).astype(mx.int32)
    drafts, conf = _chain_with_conf(idx, cand, score)
    if full_vocab:
        lf = logits.astype(mx.float32)
        picked = mx.take_along_axis(lf, drafts[:, None], axis=1)[:, 0]
        conf = mx.exp(picked - mx.logsumexp(lf, axis=-1))
    return drafts, cand, conf


def measure_cost(target, prompt_ids, max_T: int = 17, reps: int = 3) -> list[float]:
    """Seconds per T-token target forward (T = 1..max_T), median of reps."""
    target.reset()
    target.prefill(mx.array([list(prompt_ids)]), capture=False)
    tok = int(prompt_ids[-1])
    cost = [0.0]
    for T in range(1, max_T + 1):
        ts = []
        for r in range(reps + 1):
            t0 = time.perf_counter()
            lg, h = target.verify(mx.array([[tok] * T]))
            mx.eval(lg, h)
            if r:
                ts.append(time.perf_counter() - t0)
            target.trim(T - 1)
        cost.append(sorted(ts)[len(ts) // 2])
    target.reset()
    mx.clear_cache()
    return cost


def choose_len(conf: list[float], cost: list[float], alpha: float, max_len: int) -> int:
    best, best_l, p, gain = -1.0, 1, 1.0, 1.0
    for L in range(1, min(max_len, len(conf)) + 1):
        p *= max(conf[L - 1], 1e-6) ** alpha
        gain += p
        rate = gain / cost[L + 1]
        if rate > best:
            best, best_l = rate, L
    return best_l


class AdaptiveDSpineGenerator(DSpineGenerator):
    """DSpineGenerator whose greedy rounds verify a confidence- and cost-chosen prefix."""

    def __init__(self, target, draft, *, cost: list[float], alpha: float = 1.0, max_len: int | None = None,
                 full_vocab: bool = True, fixed_block_size: int | None = None, **kw):
        super().__init__(target, draft, **kw)
        # Sampling (temperature > 0) is not adaptive; it uses a plain generator with this block.
        self.fixed_block_size = fixed_block_size
        self._sampler_gen = None
        self.full_vocab = bool(full_vocab)
        self.cost = cost
        self.alpha = float(alpha)
        self.max_len = int(max_len or (self.block_size - 1))
        self.len_hist: dict[int, int] = {}

    def generate(self, prompt_tokens, *, max_tokens: int = 256, temperature: float = 0.0, stop_ids=(),
                 on_tokens=None, **kw) -> tuple[list[int], GenStats]:
        if temperature > 0:
            if self.fixed_block_size and self.fixed_block_size != self.block_size:
                if self._sampler_gen is None:
                    self._sampler_gen = DSpineGenerator(self.target, self.draft, block_size=self.fixed_block_size)
                return self._sampler_gen.generate(prompt_tokens, max_tokens=max_tokens, temperature=temperature,
                                                  stop_ids=stop_ids, on_tokens=on_tokens, **kw)
            return super().generate(prompt_tokens, max_tokens=max_tokens, temperature=temperature,
                                    stop_ids=stop_ids, on_tokens=on_tokens, **kw)
        prompt = prompt_tokens if isinstance(prompt_tokens, mx.array) else mx.array(
            [list(prompt_tokens)], dtype=mx.int32)
        if prompt.ndim == 1:
            prompt = prompt[None]
        stop = set(int(x) for x in stop_ids)
        stats = GenStats(prompt_tokens=int(prompt.shape[1]), block_size=self.block_size)
        t0 = time.perf_counter()
        self.target.reset()
        logits, hidden = self.target.prefill(prompt)
        mx.eval(logits, hidden)
        stats.prefill_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        anchor = int(mx.argmax(logits[0, -1]).item())
        tokens = [anchor]
        done = anchor in stop or len(tokens) >= max_tokens
        draft_caches = self.draft.make_cache()
        while not done:
            remaining = max_tokens - len(tokens)
            round_T = min(self.block_size, remaining + 1)
            if round_T < 2:
                break
            block = mx.array([[anchor] + [self.draft.mask_token_id] * (round_T - 1)], dtype=mx.int32)
            out, stash = self.draft(hidden, block, draft_caches)
            drafts, _, conf = greedy_tail_with_conf(self.tail, out[0, 1:], stash, anchor, self.full_vocab)
            mx.eval(drafts, conf)
            L = choose_len(conf.tolist(), self.cost, self.alpha, min(self.max_len, round_T - 1))
            self.len_hist[L] = self.len_hist.get(L, 0) + 1
            drafts = drafts[:L]
            verify_tokens = mx.concatenate([mx.array([[anchor]], dtype=mx.int32), drafts[None]], axis=1)
            vlogits, vhidden = self.target.verify(verify_tokens)
            accept, bonus = greedy_verify(vlogits[0], drafts)
            mx.eval(accept, bonus, vhidden)
            accept_i = int(accept.item())
            bonus_i = int(bonus.item())
            new_tokens = (drafts.tolist()[:accept_i] + [bonus_i])[: max_tokens - len(tokens)]
            eos_at = next((i for i, t in enumerate(new_tokens) if t in stop), None)
            if eos_at is not None:
                new_tokens = new_tokens[: eos_at + 1]
                done = True
            if not new_tokens:
                break
            tokens.extend(new_tokens)
            stats.rounds += 1
            stats.accepted_drafts += min(accept_i, len(new_tokens))
            stats.committed_tokens += len(new_tokens)
            if on_tokens is not None:
                on_tokens(new_tokens)
            if len(tokens) >= 256 and len(tokens) % 256 < len(new_tokens):
                mx.clear_cache()
            if len(tokens) >= max_tokens:
                break
            self.target.trim(L + 1 - accept_i - 1)
            hidden = vhidden[:, : accept_i + 1, :]
            anchor = new_tokens[-1]
        stats.generated_tokens = len(tokens)
        stats.decode_seconds = time.perf_counter() - t0
        stats.peak_memory_gb = mx.get_peak_memory() / 1e9
        return tokens, stats
