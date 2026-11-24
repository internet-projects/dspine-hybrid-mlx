"""Greedy DSpine with a verify length chosen every round.

The drafter always proposes its full block (16). Before the target checks it,
we pick how many drafted tokens to verify, L, by maximising

    (1 + sum_{j<=L} prod_{i<=j} c_i ** alpha) / cost(L + 1)

where c_i is the drafter's confidence in drafted token i (softmax of the
refinement scores over its K candidates, at the chosen chain state) and
cost(T) is the measured time of a T-token target forward on this machine.
Greedy verification of any prefix is still exact, so the output is unchanged;
only how much work each round does changes.

With confidence="calibrated", sampled decoding is adaptive too: the drafter samples its
chain, c_i is Tencent's calibrated sampled-chain confidence, and the target checks the
first L drafts with the usual chain rejection sampling (L depends only on draft-side
quantities, so the output distribution is unchanged).
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


CONFIDENCE = ("vocab", "calibrated", "candidates")


def score_table(tail, rows: mx.array, stash: dict, anchor: int):
    """The refinement tail's table for a drafted block.

    Returns (head logits [n, V], candidates [n, K], score [n, K senders, K], write scale [n]).
    ``score[i, s, j]`` is candidate j of row i given that row i-1 took its candidate s (row 0:
    every sender row holds the anchor's scores).
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
    return logits, cand, score, rms


def greedy_tail_with_conf(tail, rows: mx.array, stash: dict, anchor: int, full_vocab: bool = True,
                          confidence: str | None = None, calibration: dict | None = None):
    """dspine_mlx RefinementTail.greedy, plus per-token confidence.

    confidence (default from full_vocab):
      "vocab": the drafter head's full-vocabulary softmax probability of the drafted token
        (same logits the candidate search uses, so no extra matmul);
      "calibrated": Tencent's greedy feature (max softmax of score / write scale over the K
        candidates at the chain's predecessor) mapped per depth to an acceptance probability
        (dspine_hybrid.calibration);
      "candidates": softmax of the raw refinement scores over the K candidates (tends to sit
        near 1, i.e. uninformative).
    """
    logits, cand, score, rms = score_table(tail, rows, stash, anchor)
    idx = mx.argmax(score, axis=-1).astype(mx.int32)
    if confidence is None:
        confidence = "vocab" if full_vocab else "candidates"
    if confidence == "calibrated":
        from . import calibration as _cal

        drafts, conf = _chain_with_conf(idx, cand, score / mx.maximum(rms, 1e-6)[:, None, None])
        return drafts, cand, _cal.apply(conf, calibration or _cal.GREEDY)
    drafts, conf = _chain_with_conf(idx, cand, score)
    if confidence == "vocab":
        lf = logits.astype(mx.float32)
        picked = mx.take_along_axis(lf, drafts[:, None], axis=1)[:, 0]
        conf = mx.exp(picked - mx.logsumexp(lf, axis=-1))
    return drafts, cand, conf


def sample_chain_with_conf(table: mx.array, cand: mx.array, *, inv_temp: float, top_k: int, top_p: float,
                           min_p: float, uniforms: mx.array):
    """dspine_mlx.propose.serial_sample_chain, plus Tencent's sampled confidence per row.

    Same tokens, q rows and random draws; conf_i = max temperature-softmax probability of row i
    at the actually sampled predecessor, before top-k / top-p / min-p filtering.
    """
    from dspine_mlx.propose import filter_probs

    n, K, _ = table.shape
    lane = mx.arange(K)
    sender = mx.array(0, dtype=mx.int32)
    toks, qs, conf = [], [], []
    for i in range(n):
        row = mx.take(table[i], sender, axis=0)
        p = mx.softmax(row * float(inv_temp), axis=-1)
        conf.append(mx.max(p))
        q, keep = filter_probs(p, top_k, top_p, min_p)
        cdf = mx.cumsum(q, axis=-1)
        threshold = uniforms[i] * cdf[-1]
        choice = mx.sum((cdf <= threshold).astype(mx.int32))
        last = mx.max(mx.where(keep, lane, 0))
        choice = mx.minimum(choice, last)
        toks.append(mx.take(cand[i], choice))
        qs.append(q)
        sender = choice
    return mx.stack(toks).astype(mx.int32), mx.stack(qs), mx.stack(conf)


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


_CTX_COST: dict = {}


def context_cost(target, cost: list[float], ctx: int, reps: int = 3) -> list[float]:
    """``cost`` (measured on a short prompt) plus what attention adds with ``ctx`` cached tokens.

    Only attention grows with the context (measured on the M5 Max: recurrent and MLP time stay
    flat), and not evenly: at 32k keys a 16-row call costs half of a 15-row one. So the attention
    product is timed here on arrays of this context's shape, once per 1024 tokens of context
    (about 0.1 s), and the difference to a short context is added for every attention layer.
    """
    attn = [a for a in getattr(target, "_attn", []) if a is not None]
    if ctx < 2048 or not attn:
        return cost
    from . import long_attn

    key = (id(target), len(cost), ctx // 1024, getattr(target, "long_attn", None) is not None)
    if key not in _CTX_COST:
        a = attn[0]
        H, KV, hd = a.num_attention_heads, a.num_key_value_heads, a.head_dim
        kc = next((c for c, m in zip(target.cache, target._attn) if m is not None), None)
        dtype = kc.keys.dtype if getattr(kc, "keys", None) is not None else mx.float32

        def table(n):
            keys = mx.random.normal((1, KV, n, hd)).astype(dtype)
            vals = mx.random.normal((1, KV, n, hd)).astype(dtype)
            mx.eval(keys, vals)
            out = [0.0]
            for T in range(1, len(cost)):
                q = mx.random.normal((1, H, T, hd)).astype(dtype)
                mx.eval(q)
                ts = []
                for r in range(reps + 1):
                    t0 = time.perf_counter()
                    with long_attn.active(getattr(target, "long_attn", None)):
                        y = mx.fast.scaled_dot_product_attention(
                            q, keys, vals, scale=hd ** -0.5, mask="causal" if T > 1 else None)
                    mx.eval(y)
                    if r:
                        ts.append(time.perf_counter() - t0)
                out.append(sorted(ts)[len(ts) // 2])
            return out

        short, long = table(256), table(int(ctx))
        _CTX_COST[key] = [len(attn) * max(0.0, b - s) for s, b in zip(short, long)]
        mx.clear_cache()
    return [c + e for c, e in zip(cost, _CTX_COST[key])]


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
                 full_vocab: bool = True, fixed_block_size: int | None = None,
                 confidence: str | None = None, calibration: dict | None = None,
                 sampled_calibration: dict | None = None, **kw):
        super().__init__(target, draft, **kw)
        # Sampling (temperature > 0) is adaptive only with confidence="calibrated" (Tencent's sampled
        # map); otherwise it uses a plain generator with this block.
        self.fixed_block_size = fixed_block_size
        self._sampler_gen = None
        self.full_vocab = bool(full_vocab)
        self.confidence = confidence or ("vocab" if full_vocab else "candidates")
        if self.confidence not in CONFIDENCE:
            raise ValueError(f"confidence must be one of {CONFIDENCE}")
        self.calibration = calibration
        self.sampled_calibration = sampled_calibration
        self.cost = cost
        self.alpha = float(alpha)
        self.max_len = int(max_len or (self.block_size - 1))
        self.len_hist: dict[int, int] = {}

    def generate(self, prompt_tokens, *, max_tokens: int = 256, temperature: float = 0.0, stop_ids=(),
                 on_tokens=None, **kw) -> tuple[list[int], GenStats]:
        if temperature > 0 and self.confidence == "calibrated":
            return self._generate_sampled(prompt_tokens, max_tokens=max_tokens, temperature=temperature,
                                          stop_ids=stop_ids, on_tokens=on_tokens, **kw)
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
            drafts, _, conf = greedy_tail_with_conf(self.tail, out[0, 1:], stash, anchor,
                                                  confidence=self.confidence, calibration=self.calibration)
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

    def _generate_sampled(self, prompt_tokens, *, max_tokens: int, temperature: float, top_k: int = 0,
                          top_p: float = 1.0, min_p: float = 0.0, stop_ids=(), on_tokens=None):
        """Sampled DSpine with a verify length chosen from Tencent's sampled-chain confidence.

        The drafter samples its whole chain; L is chosen from draft-side quantities only, and
        the target checks the first L drafts with the usual chain rejection sampling, so the
        committed tokens still follow the target's filtered distribution.
        """
        from dspine_mlx.generate import _sample_target
        from dspine_mlx.sampler import sampled_verify

        from . import calibration as _cal

        cal = self.sampled_calibration or _cal.SAMPLED
        prompt = prompt_tokens if isinstance(prompt_tokens, mx.array) else mx.array(
            [list(prompt_tokens)], dtype=mx.int32)
        if prompt.ndim == 1:
            prompt = prompt[None]
        stop = set(int(x) for x in stop_ids)
        inv_temp = 1.0 / max(float(temperature), 1e-5)
        K = int(self.draft.head_topk)
        chain_top_k = K if top_k <= 0 else min(max(int(top_k), 1), K)
        target_top_k = self.target.vocab_size if top_k <= 0 else int(top_k)
        stats = GenStats(prompt_tokens=int(prompt.shape[1]), block_size=self.block_size)
        t0 = time.perf_counter()
        self.target.reset()
        logits, hidden = self.target.prefill(prompt)
        mx.eval(logits, hidden)
        stats.prefill_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        anchor = _sample_target(logits[0, -1], inv_temp, target_top_k, top_p, min_p)
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
            table, cand = self.tail.sampled_table(out[0, 1:], stash, anchor)
            drafts, q_rows, raw = sample_chain_with_conf(
                table, cand, inv_temp=inv_temp, top_k=chain_top_k, top_p=top_p, min_p=min_p,
                uniforms=mx.random.uniform(shape=(round_T - 1,)))
            conf = _cal.apply(raw, cal)
            mx.eval(drafts, q_rows, conf)
            L = choose_len(conf.tolist(), self.cost, 1.0, min(self.max_len, round_T - 1))
            self.len_hist[L] = self.len_hist.get(L, 0) + 1
            drafts, q_rows, cand = drafts[:L], q_rows[:L], cand[:L]
            verify_tokens = mx.concatenate([mx.array([[anchor]], dtype=mx.int32), drafts[None]], axis=1)
            vlogits, vhidden = self.target.verify(verify_tokens)
            accept, bonus = sampled_verify(
                vlogits[0], drafts, cand, q_rows, inv_temp=inv_temp, top_k=target_top_k, top_p=top_p,
                min_p=min_p, u_accept=mx.random.uniform(shape=(L,)), u_final=mx.random.uniform(),
                u_res=mx.random.uniform())
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
