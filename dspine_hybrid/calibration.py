"""Calibrated drafter confidence for ``--adaptive`` (greedy and sampled).

Tencent fitted, for Qwen3.8-27B with this DSpine drafter, a per-depth map from
the drafter's greedy confidence to the probability that the target accepts the
drafted token given that all earlier ones were accepted
(``sglang_dspine/verification/calibration.json``, Apache-2.0, greedy family,
kind "logit"):

    raw_d = max softmax of the refinement scores / write scale over the K
            candidates, at the predecessor the chain actually took
            ("greedy_chain_top1_top16_softmax")
    p_d   = sigmoid(scale * logit(raw_d) + offset[d]),   d = 0 .. 14

The expected number of accepted tokens for a verify of L drafts is then
sum_j prod_(d<j) p_d, which ``choose_len`` divides by the measured cost.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx

SOURCE = "greedy_chain_top1_top16_softmax"
SAMPLED_SOURCE = "pre_token_max_probability_actual_sampled_chain"
TENCENT_COMMIT = "6b2dcc3"
GREEDY = dict(
    scale=0.7610845308685312,
    offset=[
        0.2488127532209798,
        -0.10274277185428483,
        -0.3901669912553938,
        -0.5430951316358162,
        -0.38269754561817604,
        -0.4206673125804542,
        -0.6678220582659957,
        -0.7646444376663902,
        -0.4534658822895494,
        -0.6137670740234292,
        -0.3530263983161284,
        -0.8674443447178067,
        -1.0123771435808258,
        -0.6415821745075616,
        -0.5582114334222054,
    ],
    eps=1e-6,
)

# Sampled decoding: raw = max temperature-softmax probability of the drafter's candidate row at
# the actually sampled predecessor, before top-k / top-p / min-p filtering.
SAMPLED = dict(
    scale=0.36094624475961257,
    offset=[
        0.7631799175067309,
        0.4115029662312046,
        0.15266770036889118,
        0.19675929162419648,
        -0.05244197733290056,
        0.05518194486468104,
        -0.07226094731963359,
        -0.024693575673442245,
        -0.15799952180858753,
        -0.09363578475578148,
        -0.13441028129554053,
        -0.2566141939099428,
        -0.11757540428973136,
        0.27655560354849495,
        -0.10284620357200044,
    ],
    eps=1e-6,
)


FIT_FORMAT = "dspine-hybrid-acceptance-fit-v1"
# raw confidence passed through unchanged: sigmoid(1 * logit(raw) + 0)
IDENTITY = dict(scale=1.0, offset=[0.0], eps=1e-6)


def fit(samples: list, depths: int, steps: int = 3000, lr: float = 0.5, eps: float = 1e-6) -> dict:
    """Fit ``p = sigmoid(scale * logit(raw) + offset[depth])`` to what the target did.

    ``samples``: (depth, raw confidence, accepted 0/1) for every drafted token the target reached
    (all tokens before it were accepted). Plain gradient descent on the log loss; returns the map
    in ``apply``'s form plus ``loss`` / ``base_loss`` (the loss of the raw confidence itself) and
    the sample count per depth.
    """
    if not samples:
        raise ValueError("no samples to fit")
    d = mx.array([min(int(a), depths - 1) for a, _, _ in samples])
    raw = mx.clip(mx.array([float(b) for _, b, _ in samples]), eps, 1.0 - eps)
    y = mx.array([float(c) for _, _, c in samples])
    x = mx.log(raw) - mx.log(1.0 - raw)
    onehot = (d[:, None] == mx.arange(depths)[None, :]).astype(mx.float32)
    count = mx.maximum(mx.sum(onehot, axis=0), 1.0)
    scale, offset = mx.array(1.0), mx.zeros((depths,))

    def loss(p):
        p = mx.clip(p, 1e-7, 1.0 - 1e-7)
        return -mx.mean(y * mx.log(p) + (1.0 - y) * mx.log(1.0 - p)).item()

    base = loss(raw)
    for _ in range(steps):
        p = mx.sigmoid(scale * x + mx.take(offset, d))
        g = p - y
        scale = scale - lr * mx.mean(g * x) / mx.maximum(mx.mean(x * x), 1e-6) * 0.25
        offset = offset - lr * mx.sum(g[:, None] * onehot, axis=0) / count
        mx.eval(scale, offset)
    fitted = loss(mx.sigmoid(scale * x + mx.take(offset, d)))
    return dict(scale=float(scale.item()), offset=[float(v) for v in offset.tolist()], eps=eps,
                loss=fitted, base_loss=base, samples=[int(v) for v in mx.sum(onehot, axis=0).tolist()])


def load(path: str | Path | None = None, mode: str = "greedy") -> dict:
    """The logit map for ``mode`` ("greedy" or "sampled"): Tencent's fit, or one from a
    calibration JSON in their format."""
    if mode not in ("greedy", "sampled"):
        raise ValueError("mode must be greedy or sampled")
    if path is None:
        return GREEDY if mode == "greedy" else SAMPLED
    art = json.loads(Path(path).read_text())
    if art.get("format") == FIT_FORMAT:              # our own fit (``fit`` below)
        m = art[mode]
        return dict(scale=float(m["scale"]), offset=[float(x) for x in m["offset"]], eps=float(m.get("eps", 1e-6)))
    if art.get("format") != "dspine-confidence-calibration-v1":
        raise ValueError("not a DSpine confidence calibration artifact")
    want = SOURCE if mode == "greedy" else SAMPLED_SOURCE
    src = art.get("confidence_sources", {}).get(mode)
    if src not in (None, want):
        raise ValueError(f"{mode} confidence source {src!r} is not {want!r}")
    m = art["families"][mode]["model"]
    if m.get("kind") != "logit":
        raise ValueError(f"only the 'logit' {mode} map is supported")
    scale = m["scale"]
    if len(set(scale)) != 1:
        raise ValueError("per-depth scales are not supported; expected one shared scale")
    return dict(scale=float(scale[0]), offset=[float(x) for x in m["offset"]], eps=float(m.get("eps", 1e-6)))


def apply(raw: mx.array, model: dict = GREEDY) -> mx.array:
    """Map raw confidences [n] (depth 0 first) to acceptance probabilities [n]."""
    n = int(raw.shape[0])
    off = model["offset"]
    offset = mx.array([off[min(d, len(off) - 1)] for d in range(n)], dtype=mx.float32)
    eps = float(model["eps"])
    upper = math.log1p(-eps)
    raw = mx.clip(raw.astype(mx.float32), 0.0, 1.0)
    log_p = mx.minimum(mx.log(mx.maximum(raw, eps)), upper)
    log_q = mx.minimum(mx.log(mx.maximum(1.0 - raw, eps)), upper)
    return mx.sigmoid((log_p - log_q) * float(model["scale"]) + offset)
