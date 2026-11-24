"""Copy drafts and the draft tree against the measured v0.3 rounds, on one target.

Greedy decoding on three prompt sets: code (HumanEval), prose (MT-Bench) and edit (the answer
mostly repeats text from the prompt, which is where copy drafts apply). Every row also checks
that the tokens are the same as the plain fixed-block run. A last part measures what the drafter
accepts at its full block of 16 and what that would give on an engine with a cheap wide check.
Takes about fifteen minutes.

Run on an otherwise idle machine:
    python scripts/bench_lanes.py --target <4-bit build> --draft <drafter> --out lanes.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402

from bench_accept import prompts  # noqa: E402
from dspine_hybrid import calibration as cal  # noqa: E402
from dspine_hybrid.adaptive import measure_cost  # noqa: E402
from dspine_hybrid.lanes import LaneDSpineGenerator  # noqa: E402
from dspine_hybrid.run import apply_memory_rules, load  # noqa: E402
from dspine_mlx.generate import _load_tokenizer_eos  # noqa: E402

CODE = '''import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Job:
    name: str
    command: list[str]
    retries: int = 3
    timeout: float = 600.0
    tags: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return f"{self.name} ({' '.join(self.command)}) retries={self.retries}"


class JobQueue:
    def __init__(self, path: Path):
        self.path = path
        self.jobs: list[Job] = []
        self.done: dict[str, float] = {}

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text())
        self.jobs = [Job(**row) for row in data.get("jobs", [])]
        self.done = dict(data.get("done", {}))

    def save(self) -> None:
        data = {"jobs": [job.__dict__ for job in self.jobs], "done": self.done}
        self.path.write_text(json.dumps(data, indent=2))

    def add(self, job: Job) -> None:
        if any(j.name == job.name for j in self.jobs):
            raise ValueError(f"duplicate job {job.name}")
        self.jobs.append(job)
        self.save()

    def pending(self) -> list[Job]:
        return [job for job in self.jobs if job.name not in self.done]

    def mark_done(self, job: Job) -> None:
        self.done[job.name] = time.time()
        self.save()
'''

TEXT = (
    "The harbour office opens at seven in the morning from April to October and at eight for the rest "
    "of the year. Visiting boats report to the office before they tie up, and the duty officer gives "
    "them a berth on the east quay if they are under twelve metres and on the outer wall if they are "
    "longer. Water and power are included in the nightly fee, which is collected in cash or by card "
    "when the boat leaves. Fuel is sold from the pontoon beside the crane between nine and five, and "
    "the pump is locked outside those hours. Boats that stay more than seven nights pay a weekly rate "
    "and receive a key to the shower block, which must be returned on departure."
)

EDIT = [
    "Add a method `remove(self, name: str) -> None` to JobQueue that deletes the job with that name and "
    "saves. Return the complete file.\n\n```python\n" + CODE + "```",
    "Rename the field `retries` to `max_retries` everywhere. Return the complete file.\n\n```python\n" + CODE + "```",
    "Copy the following text exactly, changing only the opening time in the first sentence from seven "
    "to six:\n\n" + TEXT,
]


def chat_ids(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                   enable_thinking=False)


def run_set(gen, tok, stop, texts, max_tokens):
    n = s = rounds = 0.0
    outs, lanes = [], {}
    for text in texts:
        out, st = gen.generate(chat_ids(tok, text), max_tokens=max_tokens, stop_ids=stop)
        n += st.generated_tokens
        s += st.decode_seconds
        rounds += st.rounds
        outs.append(out)
        for k, v in gen.lane_stats.items():
            lanes[k] = lanes.get(k, 0) + v
        mx.clear_cache()
    return dict(tps=round(n / s, 1), tokens=int(n), tokens_per_round=round(n / max(rounds, 1), 2), **lanes), outs


def tree_cost(target, reps=3):
    """Seconds for a chain verify and for trees of the same number of rows (2 and 4 branches)."""
    def branches(rows, k):
        """rows - 1 drafted nodes in k equal chains hanging off the pending token."""
        parents, size = [-1], max(1, (rows - 1 + k - 1) // k)
        for i in range(1, rows):
            parents.append(0 if (i - 1) % size == 0 else i - 1)
        return parents

    shapes = {rows: {"chain_tree": branches(rows, 1), "two_branches": branches(rows, 2),
                     "four_branches": branches(rows, 4)} for rows in (5, 9, 16)}
    def fresh():
        target.reset()
        target.prefill(mx.array([list(range(1000, 1256))]), capture=False)
        mx.eval([c.state for c in target.cache])

    target.enable_prompt_cache(False)
    fresh()
    out = {}
    for rows, group in shapes.items():
        row = {}
        ts = []
        for r in range(reps + 1):
            t0 = time.perf_counter()
            lg, h = target.verify(mx.array([[1000 + i for i in range(rows)]]))
            mx.eval(lg, h)
            if r:
                ts.append(time.perf_counter() - t0)
            fresh()                     # every timing starts from the same cache
        row["chain_ms"] = round(1000 * sorted(ts)[len(ts) // 2], 1)
        for name, parents in group.items():
            ts = []
            for r in range(reps + 1):
                t0 = time.perf_counter()
                lg, h = target.verify_tree([1000 + i for i in range(rows)], parents)
                mx.eval(lg, h)
                if r:
                    ts.append(time.perf_counter() - t0)
            row[name + "_ms"] = round(1000 * sorted(ts)[len(ts) // 2], 1)
        out[rows] = row
    target.reset()
    mx.clear_cache()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--tokens", type=int, default=384)
    p.add_argument("--memory-limit-gb", type=float, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    apply_memory_rules(a.memory_limit_gb)
    base, target, _, _ = load(a.target, a.draft)
    target.enable_prompt_cache(False)
    tok = target.tokenizer
    stop = _load_tokenizer_eos(tok)
    raw = prompts(a.n)
    sets = {"code": raw["humaneval"][: a.n], "prose": raw["mtbench"][: a.n], "edit": EDIT[: a.n]}
    res = {"target": a.target, "tokens": a.tokens, "block": base.block_size}

    res["tree_cost"] = tree_cost(target)
    print("[lanes] tree cost", res["tree_cost"], flush=True)
    cost = measure_cost(target, list(range(1000, 1256)))
    res["cost_ms"] = [round(1000 * c, 1) for c in cost]

    def make(adaptive, copy, tree):
        return LaneDSpineGenerator(
            target, base.draft, block_size=base.block_size, adaptive=adaptive, cost=cost if adaptive else None,
            confidence="calibrated" if adaptive else "vocab", calibration=cal.load(None) if adaptive else None,
            sampled_calibration=cal.load(None, "sampled") if adaptive else None, copy=copy, tree=tree)

    rows = {
        "fixed": (False, False, False), "fixed+copy": (False, True, False), "fixed+tree": (False, False, True),
        "fixed+copy+tree": (False, True, True),
        "adaptive": (True, False, False), "adaptive+copy": (True, True, False), "adaptive+tree": (True, False, True),
        "adaptive+copy+tree": (True, True, True),
    }
    make(False, False, False).generate(chat_ids(tok, "Say hello."), max_tokens=16, stop_ids=stop)
    res["greedy"] = {}
    for bench, texts in sets.items():
        table, ref = {}, None
        for name, flags in rows.items():
            row, outs = run_set(make(*flags), tok, stop, texts, a.tokens)
            if ref is None:
                ref = outs
            row["same_as_fixed"] = outs == ref
            table[name] = row
            print("[lanes]", bench, name, row, flush=True)
        res["greedy"][bench] = table

    # ceiling: what DSpine accepts when it may verify its full block of 16, and what that would give
    # on an engine whose 16-row check costs about what one row costs here
    res["ceiling"] = {}
    for bench in ("code", "prose"):
        table = {}
        for name, tree in (("chain16", False), ("tree16", True)):
            gen = LaneDSpineGenerator(target, base.draft, block_size=16, copy=False, tree=tree)
            n = s = rounds = 0.0
            hist = {}
            for text in sets[bench]:
                _, st = gen.generate(chat_ids(tok, text), max_tokens=a.tokens, stop_ids=stop)
                n += st.generated_tokens
                s += st.decode_seconds
                rounds += st.rounds
                for k, v in gen.accept_hist.items():
                    hist[k] = hist.get(k, 0) + v
                mx.clear_cache()
            total = max(sum(hist.values()), 1)
            reach = [round(sum(v for k, v in hist.items() if k >= d) / total, 3) for d in range(1, 16)]
            per_round = n / max(rounds, 1)
            other_ms = max(1000 * s / max(rounds, 1) - 1000 * cost[16], 0.0)   # drafter + bookkeeping per round
            table[name] = dict(
                tps=round(n / s, 1), tokens_per_round=round(per_round, 2), reach_depth=reach,
                drafter_ms_per_round=round(other_ms, 1),
                tps_if_16_rows_cost_like_1=round(per_round / ((cost[1] * 1000 + other_ms) / 1000), 1),
                tps_if_16_rows_cost_like_5=round(per_round / ((cost[5] * 1000 + other_ms) / 1000), 1))
            print("[lanes] ceiling", bench, name, table[name], flush=True)
        res["ceiling"][bench] = table
    base.draft.set_block_size(base.block_size)

    # sampled decoding (Qwen settings): copy drafts only; the tree is greedy-only
    kw = dict(temperature=0.7, top_k=20, top_p=0.8)
    res["sampled"] = {}
    for bench in ("code", "edit"):
        table = {}
        for name, flags in (("fixed", (False, False, False)), ("fixed+copy", (False, True, False))):
            gen = make(*flags)
            n = s = 0.0
            lanes = {}
            for text in sets[bench]:
                mx.random.seed(0)
                _, st = gen.generate(chat_ids(tok, text), max_tokens=a.tokens, stop_ids=stop, **kw)
                n += st.generated_tokens
                s += st.decode_seconds
                for k, v in gen.lane_stats.items():
                    lanes[k] = lanes.get(k, 0) + v
                mx.clear_cache()
            table[name] = dict(tps=round(n / s, 1), **lanes)
            print("[lanes] sampled", bench, name, table[name], flush=True)
        res["sampled"][bench] = table
    res["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
