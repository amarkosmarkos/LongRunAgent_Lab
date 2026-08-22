"""The run's program population: an open evolving pool, not per-branch lanes.

Three ideas from the 2025-26 program-evolution literature, combined:

 - Darwin-Gödel-Machine parent selection: any valid program in the DAG can
   parent the next mutation, with probability proportional to its quality and
   INVERSELY proportional to how many children it already has — strong but
   unexplored lineages get their shot, and dead branches remain genetic
   material instead of garbage.
 - MAP-Elites over BEHAVIOR: each program is binned by what its solver
   actually does (behavior descriptor computed by the Problem from the real
   tour/runtime), falling back to technique tags. A genuinely new algorithm
   lands in a fresh niche automatically, whether or not we can name it.
 - AlphaEvolve-style inspiration sampling: prompts show a handful of elites
   from DIFFERENT niches plus one deliberately odd non-elite, because diverse
   context is the strongest known lever on diverse output.

Everything here is deterministic given the run's RNG and API-free.
"""
from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass, field


@dataclass
class Program:
    id: str                      # git sha when available, else a uid
    code: str
    branch_id: str | None
    round: int
    score: float | None = None   # lower is better (engine convention)
    valid: bool = False
    behavior: dict | None = None  # problem-computed behavior descriptor
    tags: list[str] = field(default_factory=list)
    operator: str | None = None
    parent_id: str | None = None
    children: int = 0
    origin: str = "run"          # run | archive | baseline
    name: str | None = None

    def public(self) -> dict:
        return {"id": self.id, "branch_id": self.branch_id, "round": self.round,
                "score": self.score, "valid": self.valid,
                "behavior": self.behavior, "tags": self.tags,
                "operator": self.operator, "parent_id": self.parent_id,
                "children": self.children, "origin": self.origin,
                "name": self.name}


def _log_bin(value: float, per_doubling: int, cap: int) -> int:
    """Bin a MULTIPLICATIVE quantity (a speedup) on a log2 scale.

    A linear bin saturates exactly where kernels get interesting: with a 0.25
    width every result past 3x collapsed into the top bucket, so a 3x and a 12x
    kernel shared a niche and the axis stopped separating anything. On a log
    scale each doubling gets `per_doubling` buckets, so 1x / 2x / 4x / 16x stay
    distinct all the way up.
    """
    if value is None or value <= 0:
        return 0
    return min(cap, max(0, int(round(math.log2(value) * per_doubling)) + cap // 2))


# descriptor key -> (short label, buckets per doubling, max bin). Problems
# return whatever descriptor makes sense for them; only the keys listed here
# become niche axes, so adding a problem means adding its axes, not touching
# the population logic. These are
# speedups, so they are binned logarithmically: 2 buckets per doubling keeps
# 1x, 1.4x, 2x, 2.8x, 4x … apart across the whole useful range.
_NICHE_AXES = {
    # GPU kernels: HOW a kernel wins — on small shapes, on large ones, or evenly
    "small_speedup": ("s", 2, 16),
    "large_speedup": ("l", 2, 16),
    "scaling": ("sc", 2, 12),
}


def niche_key(p: Program) -> str:
    """Behavior bins when a descriptor exists, technique tags otherwise."""
    b = p.behavior or {}
    parts = [f"{label}{_log_bin(b[key], per_doubling, cap)}"
             for key, (label, per_doubling, cap) in _NICHE_AXES.items() if key in b]
    if parts:
        return "|".join(parts)
    return "+".join(p.tags) if p.tags else "unclassified"


class Population:
    def __init__(self, baseline_score: float, rng: random.Random):
        self.baseline_score = baseline_score
        self.rng = rng
        self._programs: dict[str, Program] = {}
        self._elites: dict[str, str] = {}  # niche key -> program id
        self._lock = threading.Lock()

    # --------------------------------------------------------------- write
    def add(self, p: Program) -> dict:
        """Register a program; returns niche outcome for the event stream."""
        with self._lock:
            self._programs[p.id] = p
            parent = self._programs.get(p.parent_id) if p.parent_id else None
            if parent:
                parent.children += 1
            outcome = {"niche": None, "became_elite": False,
                       "new_niche": False, "population": len(self._programs)}
            if p.valid and p.score is not None:
                key = niche_key(p)
                outcome["niche"] = key
                incumbent = self._programs.get(self._elites.get(key, ""))
                if incumbent is None:
                    self._elites[key] = p.id
                    outcome.update(became_elite=True, new_niche=True)
                elif p.score < (incumbent.score or math.inf):
                    self._elites[key] = p.id
                    outcome["became_elite"] = True
            return outcome

    # -------------------------------------------------------------- select
    def _quality(self, p: Program) -> float:
        """Map score to a positive weight via improvement over baseline."""
        if p.score is None or self.baseline_score <= 0:
            return 0.05
        imp = (self.baseline_score - p.score) / self.baseline_score  # fraction
        # sigmoid keeps weights bounded; k=12 spreads typical 0-30% improvements
        return 1.0 / (1.0 + math.exp(-12.0 * imp))

    def select_parent(self, exclude_ids: set[str] | None = None) -> Program | None:
        """DGM rule: weight ∝ quality × 1/(1+children). Valid programs only."""
        with self._lock:
            pool = [p for p in self._programs.values()
                    if p.valid and p.code and
                    (not exclude_ids or p.id not in exclude_ids)]
            if not pool:
                return None
            weights = [self._quality(p) / (1.0 + p.children) for p in pool]
            total = sum(weights)
            if total <= 0:
                return self.rng.choice(pool)
            r = self.rng.random() * total
            acc = 0.0
            for p, w in zip(pool, weights):
                acc += w
                if r <= acc:
                    return p
            return pool[-1]

    def select_inspirations(self, k: int,
                            exclude_ids: set[str] | None = None) -> list[Program]:
        """k-1 best elites from DISTINCT niches + 1 random non-elite oddball.

        With no non-elite available the remaining slot goes to another elite,
        so callers always get k programs when k exist.
        """
        exclude = exclude_ids or set()
        with self._lock:
            elite_ids = set(self._elites.values())
            elites = [self._programs[i] for i in elite_ids
                      if i in self._programs and i not in exclude
                      and self._programs[i].code]
            others = [p for p in self._programs.values()
                      if p.valid and p.code and p.id not in elite_ids
                      and p.id not in exclude]
        elites.sort(key=lambda p: p.score if p.score is not None else math.inf)
        picked = elites[:max(0, k - 1)]
        if others:
            picked.append(self.rng.choice(others))
        elif len(picked) < k:
            # Early in a run every program is an elite (each one opens a fresh
            # niche), so there is no oddball to add. Backfill with the next
            # elites instead of silently handing back fewer examples than
            # asked for, precisely when examples are scarcest.
            picked += [p for p in elites[len(picked):k]]
        return picked[:k]

    # --------------------------------------------------------------- reads
    def get(self, pid: str) -> Program | None:
        with self._lock:
            return self._programs.get(pid)

    def stats(self) -> dict:
        with self._lock:
            valid = [p for p in self._programs.values() if p.valid]
            return {"programs": len(self._programs), "valid": len(valid),
                    "niches": len(self._elites)}

    def elites_public(self) -> list[dict]:
        with self._lock:
            return [{**self._programs[pid].public(), "niche": key}
                    for key, pid in sorted(self._elites.items())
                    if pid in self._programs]
