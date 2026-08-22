"""UCB1 bandit over mutation operators (and models, when several are offered).

ShinkaEvolve showed that a large share of sample efficiency comes from
LEARNING which mutation operator pays off, instead of always asking for the
same kind of edit. Each experiment pulls an arm = (operator, model); the
reward is how much the attempt actually moved the branch, discounted by what
it cost. Deterministic given the run's RNG; no state survives the run (the
cross-run archive is the long-term memory, the bandit is per-run tactics).
"""
from __future__ import annotations

import math
import threading

# operator name -> instruction injected into the Experimenter prompt
OPERATORS: dict[str, str] = {
    "refine": (
        "OPERATOR: REFINE. Make a targeted incremental improvement to the base "
        "program. Keep its overall structure; change the one thing the evidence "
        "says is weakest."),
    "rewrite": (
        "OPERATOR: REWRITE. Do NOT edit the base program. Re-implement the "
        "hypothesis from scratch with a substantially different algorithmic "
        "structure. Reusing the base program's structure counts as failure."),
    "recombine": (
        "OPERATOR: RECOMBINE. Take the strongest distinct components of the "
        "programs you are shown (base + inspirations) and combine them into "
        "one coherent solver. Name in your approach which piece came from "
        "where and why the combination should beat every source."),
    "explore": (
        "OPERATOR: EXPLORE. Attack a DIFFERENT bottleneck than the base program "
        "does — change the memory access pattern, the tiling scheme, the "
        "precision or accumulation strategy, the launch configuration, or move "
        "between torch / Triton / inline CUDA. Aim to land somewhere the "
        "population has not been yet, even at the risk of a worse time; a "
        "correct kernel that behaves differently is worth more than another "
        "marginal tweak."),
}


class OperatorBandit:
    """UCB1 over (operator, model) arms."""

    def __init__(self, operators: list[str], models: list[str], rng):
        self.arms = [(op, m) for op in operators for m in models]
        self.rng = rng
        self._n = {a: 0 for a in self.arms}
        self._sum = {a: 0.0 for a in self.arms}
        self._t = 0
        self._lock = threading.Lock()

    def pick(self, allowed_ops: list[str] | None = None) -> tuple[str, str]:
        with self._lock:
            arms = [a for a in self.arms
                    if not allowed_ops or a[0] in allowed_ops]
            if not arms:
                arms = self.arms
            self._t += 1
            unexplored = [a for a in arms if self._n[a] == 0]
            if unexplored:
                return self.rng.choice(unexplored)
            def ucb(a):
                mean = self._sum[a] / self._n[a]
                return mean + math.sqrt(2.0 * math.log(self._t) / self._n[a])
            return max(arms, key=ucb)

    def reward(self, arm: tuple[str, str], improved: bool, valid: bool,
               new_niche: bool, cost_usd: float) -> float:
        """Bounded reward: real improvement pays most, opening a brand-new
        behavior niche also pays (novelty is a first-class outcome), validity
        alone pays a little. Slightly discounted by spend."""
        r = 0.0
        if improved:
            r += 0.7
        if new_niche:
            r += 0.5
        if valid and not improved and not new_niche:
            r += 0.15
        r = min(1.0, r) / (1.0 + 4.0 * max(0.0, cost_usd))
        with self._lock:
            if arm in self._n:
                self._n[arm] += 1
                self._sum[arm] += r
        return r

    def stats(self) -> list[dict]:
        with self._lock:
            return [{"operator": op, "model": m, "pulls": self._n[(op, m)],
                     "mean_reward": round(self._sum[(op, m)] / self._n[(op, m)], 4)
                     if self._n[(op, m)] else None}
                    for op, m in self.arms]
