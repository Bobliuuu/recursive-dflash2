"""Phase 8: content-adaptive block length (verify width K) for a block-causal parallel drafter (DSpine chain).

Pure Python, no torch / SGLang imports, so the serving path can call it per round and the tests run on CPU.

Conventions
- K is the verify width in tokens, including the anchor: a round at width K verifies K-1 draft tokens.
- `accepted` is the number of accepted draft tokens in a round (0..K-1). Emitted tokens = accepted + 1 (bonus).
  An observation with accepted == K-1 is right-censored: the drafter might have accepted more at a wider K.
- Because DSpine's drafter is block-causal, a width-K verify accepts exactly min(a, K-1) drafts of a wider block
  that would have accepted a. So acceptance is non-decreasing in K and a smaller K only saves round time.

Policies (one instance per request; `choose()` -> K for the next round, `observe(accepted, k)` after it):
- FixedK: constant K (baseline).
- EmaThresholdK: EMA of emitted tokens per round, pick the smallest K >= ratio * EMA; a full-block round moves up one step.
- HazardK: per-position acceptance hazards (censoring-aware, Kaplan-Meier style) with exponential forgetting and a
  prior; picks the K that maximises expected emitted tokens / round time (one step wider after a full block).
`allocate_batch` picks per-request widths for one batched verify when the step cost depends on the total token count.
"""
from __future__ import annotations

import bisect
import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence

# Measured A100-80GB / Qwen3-8B / DSpine chain round time (ms) at batch 1, mean over the 5 datasets of makespan / rounds
# (bench/results/p2_blk{16,24,32,48,64}_*_T0_c1.json). 8 and 12 are 16 minus the CUDA-graph verify increments of
# bench/results/p2_verify_cost_8b_ctx1536.json (no batch-1 block-8/12 engine run exists). See docs/phase8-adaptive-block.md.
A100_8B_C1_ROUND_MS: Dict[int, float] = {8: 14.56, 12: 14.88, 16: 15.18, 24: 15.85, 32: 16.04, 48: 16.31, 64: 16.62}

# Prior per-position conditional acceptance P(position i accepted | i-1 accepted), i = 1..47, DSpine-b16 weights.
# 1..15 from docs/step0-findings.md (pooled, b16 run); 16..30 from docs/phase2-findings.md (b32 rollouts); 31+ held at 0.65.
DSPINE_PRIOR_HAZARD: List[float] = (
    [0.88, 0.87, 0.87, 0.88, 0.89, 0.90, 0.91, 0.91, 0.91, 0.91, 0.91, 0.92, 0.92, 0.91, 0.92]
    + [0.90, 0.90, 0.90, 0.88, 0.88, 0.88, 0.85, 0.85, 0.85, 0.78, 0.78, 0.78, 0.78, 0.70, 0.68]
    + [0.65] * 17
)


class CostCurve:
    """Round (or batched step) time as a piecewise-linear function of tokens, clamped-slope extrapolation."""

    def __init__(self, points: Dict[int, float]):
        if len(points) < 2:
            raise ValueError("need at least two points")
        self.xs = sorted(points)
        self.ys = [float(points[x]) for x in self.xs]

    def __call__(self, k: float) -> float:
        xs, ys = self.xs, self.ys
        if k <= xs[0]:
            i = 0
        elif k >= xs[-1]:
            i = len(xs) - 2
        else:
            i = bisect.bisect_right(xs, k) - 1
        x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
        return max(1e-9, y0 + (y1 - y0) * (k - x0) / (x1 - x0))


def _check_menu(menu: Iterable[int]) -> List[int]:
    m = sorted(set(int(k) for k in menu))
    if not m or m[0] < 2:
        raise ValueError("menu must contain widths >= 2")
    return m


class FixedK:
    def __init__(self, k: int):
        self.k = int(k)

    def reset(self) -> None:
        pass

    def choose(self) -> int:
        return self.k

    def observe(self, accepted: int, k: int) -> None:
        pass


class EmaThresholdK:
    """K = smallest menu width >= ratio * EMA(emitted per round); after a full-block round, one step up (probe)."""

    def __init__(self, menu: Sequence[int] = (8, 16, 32), alpha: float = 0.3, ratio: float = 2.0, init_k: Optional[int] = None):
        self.menu = _check_menu(menu)
        self.alpha, self.ratio = float(alpha), float(ratio)
        self.init_k = int(init_k) if init_k is not None else self.menu[len(self.menu) // 2]
        self.reset()

    def reset(self) -> None:
        self.ema: Optional[float] = None
        self.k = self.init_k
        self._full = False

    def choose(self) -> int:
        if self.ema is None:
            return self.k
        want = self.ratio * self.ema
        i = bisect.bisect_left(self.menu, want)
        i = min(i, len(self.menu) - 1)
        if self._full:  # censored round: never step below the width that just filled, and probe one wider
            j = bisect.bisect_left(self.menu, self.k)
            i = max(i, min(j + 1, len(self.menu) - 1))
        self.k = self.menu[i]
        return self.k

    def observe(self, accepted: int, k: int) -> None:
        if not 0 <= accepted <= k - 1:
            raise ValueError(f"accepted={accepted} outside 0..{k - 1}")
        e = accepted + 1
        self.ema = e if self.ema is None else (1 - self.alpha) * self.ema + self.alpha * e
        self._full = accepted == k - 1
        self.k = k


class HazardK:
    """Censoring-aware per-position acceptance estimate with forgetting; maximises E[emitted(K)] / cost(K).

    For each draft position i (1-based) keeps decayed counts at_risk[i], accepted[i]. A round with `accepted`=a at
    width k contributes: positions 1..a reached and accepted; position a+1 reached and rejected if a < k-1 (else the
    round is censored and nothing is known beyond k-1). hazard_i = (acc_i + w*prior_i) / (risk_i + w).
    """

    def __init__(self, menu: Sequence[int] = (8, 16, 32), cost: Optional[Callable[[float], float]] = None,
                 prior: Sequence[float] = DSPINE_PRIOR_HAZARD, prior_weight: float = 4.0, decay: float = 0.9):
        self.menu = _check_menu(menu)
        self.cost = cost or CostCurve(A100_8B_C1_ROUND_MS)
        n = self.menu[-1] - 1
        p = list(prior)
        if len(p) < n:
            p = p + [p[-1] if p else 0.8] * (n - len(p))
        self.prior = [min(max(float(x), 1e-6), 1 - 1e-6) for x in p[:n]]
        self.w, self.decay = float(prior_weight), float(decay)
        self.reset()

    def reset(self) -> None:
        n = self.menu[-1] - 1
        self.risk = [0.0] * n
        self.acc = [0.0] * n
        self._censored_at: Optional[int] = None

    def hazards(self) -> List[float]:
        return [(a + self.w * p) / (r + self.w) for a, r, p in zip(self.acc, self.risk, self.prior)]

    def expected_emitted(self) -> Dict[int, float]:
        h = self.hazards()
        out, s, tot = {}, 1.0, 1.0
        cum = [1.0]
        for x in h:
            s *= x
            tot += s
            cum.append(tot)
        for k in self.menu:
            out[k] = cum[k - 1]
        return out

    def choose(self) -> int:
        ee = self.expected_emitted()
        k = max(self.menu, key=lambda k: (ee[k] / self.cost(k), -k))
        if self._censored_at is not None:  # positions past a full block are never observed: probe one step wider
            j = bisect.bisect_left(self.menu, self._censored_at)
            k = max(k, self.menu[min(j + 1, len(self.menu) - 1)])
        return k

    def observe(self, accepted: int, k: int) -> None:
        if not 0 <= accepted <= k - 1:
            raise ValueError(f"accepted={accepted} outside 0..{k - 1}")
        d = self.decay
        self.risk = [x * d for x in self.risk]
        self.acc = [x * d for x in self.acc]
        n = len(self.risk)
        for i in range(min(accepted, n)):
            self.risk[i] += 1.0
            self.acc[i] += 1.0
        if accepted < k - 1 and accepted < n:
            self.risk[accepted] += 1.0
        self._censored_at = k if accepted == k - 1 else None


def allocate_batch(expected: Sequence[Dict[int, float]], menu: Sequence[int], step_cost: Callable[[float], float]) -> List[int]:
    """Per-request widths for one batched verify whose time is step_cost(sum of widths).

    Greedy: start everyone at the smallest width, repeatedly apply the single one-step widening that most increases
    (sum of expected emitted) / step_cost(total), stop when none does. `expected[r][k]` = expected emitted tokens.
    """
    menu = _check_menu(menu)
    idx = [0] * len(expected)

    def rate(ix):
        return sum(e[menu[i]] for e, i in zip(expected, ix)) / step_cost(sum(menu[i] for i in ix))

    best = rate(idx) if expected else 0.0
    while True:
        cand = None
        for r in range(len(idx)):
            if idx[r] + 1 < len(menu):
                trial = idx[:]
                trial[r] += 1
                v = rate(trial)
                if v > best + 1e-12:
                    best, cand = v, trial
        if cand is None:
            return [menu[i] for i in idx]
        idx = cand


class AdaptiveKController:
    """Per-request policy registry for a serving loop. `next_k(rid)` before a round, `record(rid, accepted, k)` after."""

    def __init__(self, factory: Callable[[], object]):
        self.factory = factory
        self.state: Dict[object, object] = {}

    def _get(self, rid):
        p = self.state.get(rid)
        if p is None:
            p = self.state[rid] = self.factory()
        return p

    def next_k(self, rid) -> int:
        return self._get(rid).choose()

    def record(self, rid, accepted: int, k: int) -> None:
        self._get(rid).observe(int(accepted), int(k))

    def finish(self, rid) -> None:
        self.state.pop(rid, None)


def make_policy(spec: str, menu: Sequence[int] = (8, 16, 32), cost: Optional[Callable[[float], float]] = None):
    """Factory from a short spec string (used by env vars): 'fixed:16', 'ema', 'ema:2.0', 'hazard', 'hazard:0.9'."""
    name, _, arg = spec.partition(":")
    if name == "fixed":
        return FixedK(int(arg or 16))
    if name == "ema":
        return EmaThresholdK(menu, ratio=float(arg or 2.0))
    if name == "hazard":
        return HazardK(menu, cost=cost, decay=float(arg or 0.9))
    raise ValueError(f"unknown policy spec {spec!r}")


def parse_menu(s: str) -> List[int]:
    return _check_menu(int(x) for x in s.replace(",", " ").split())


class ShadowLogger:
    """Serving-side shadow mode (used by bench/patches/apply_adaptive_k_hook.py): after each bs=1 verify round, feed the
    observed accepted-draft count to a policy and append {seq, k, accepted, next_k} to a JSONL file. It never changes
    the served width; it records the per-round acceptance order (missing from the committed results, which only keep
    per-request histograms) and what the policy would have chosen. A new request is detected when the prefix length
    does not grow."""

    def __init__(self, path: str, spec: str = "ema:3.0", menu: Sequence[int] = (8, 16, 32)):
        self.path, self.spec, self.menu = path, spec, list(menu)
        self.policy = make_policy(spec, self.menu)
        self.last_seq: Optional[int] = None
        self.f = None

    def step(self, seq: int, k: int, accepted: int) -> int:
        import json
        if self.last_seq is None or seq <= self.last_seq:
            self.policy.reset()
        self.last_seq = seq
        a = max(0, min(int(accepted), int(k) - 1))
        kk = int(k)
        if kk <= self.menu[-1]:
            self.policy.observe(a, kk)
        nxt = self.policy.choose()
        if self.f is None:
            self.f = open(self.path, "a", buffering=1)
        self.f.write(json.dumps({"seq": int(seq), "k": kk, "accepted": int(accepted), "next_k": nxt}) + "\n")
        return nxt
