"""Euclidean TSP problems.

- TSP: random uniform instance, nearest-neighbor baseline (easy mode).
- TSPBenchmark: TSPLIB95 suite with known optima, nearest-neighbor + 2-opt
  baseline, scored as mean gap %% over a dev set, with held-out verification
  of the winning solver at the end of the run (guards against overfitting).
"""
from __future__ import annotations

import math
import random
from functools import lru_cache

from .base import Problem
from .tsplib import (available_instances, gap_pct, lkh_tour, load_instance,
                     nn_2opt_tour, tour_length_euc2d)


@lru_cache(maxsize=None)
def lkh_ref(name: str) -> dict:
    """Cached LKH (state-of-the-art) reference for a named TSPLIB instance:
    its near-optimal tour, length and gap. Cached across runs in-process."""
    inst = load_instance(name)
    tour = lkh_tour(inst["cities"])
    length = tour_length_euc2d(inst["cities"], tour)
    return {"tour": tour, "length": length,
            "gap_pct": gap_pct(length, inst["optimum"])}


def tour_length(cities: list[list[float]], tour: list[int]) -> float:
    total = 0.0
    n = len(tour)
    for i in range(n):
        x1, y1 = cities[tour[i]]
        x2, y2 = cities[tour[(i + 1) % n]]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


class TSP(Problem):
    name = "tsp"
    description = (
        "Euclidean Travelling Salesman Problem: given N cities as (x, y) points, "
        "find the shortest closed tour visiting every city exactly once. "
        "Score = total tour length (lower is better)."
    )

    def generate_instance(self, params: dict) -> dict:
        n = int(params.get("n_cities", 60))
        seed = int(params.get("seed", 42))
        rng = random.Random(seed)
        cities = [[round(rng.uniform(0, 1000), 2), round(rng.uniform(0, 1000), 2)]
                  for _ in range(n)]
        return {"cities": cities, "n": n, "seed": seed}

    def baseline(self, instance: dict):
        cities = instance["cities"]
        n = len(cities)
        unvisited = set(range(1, n))
        tour = [0]
        while unvisited:
            cx, cy = cities[tour[-1]]
            nxt = min(unvisited,
                      key=lambda j: (cities[j][0] - cx) ** 2 + (cities[j][1] - cy) ** 2)
            unvisited.remove(nxt)
            tour.append(nxt)
        return tour, tour_length(cities, tour), "nearest-neighbor (from city 0)"

    def validate(self, instance: dict, solution) -> str | None:
        n = instance["n"]
        if not isinstance(solution, list):
            return "solution is not a list"
        if len(solution) != n:
            return f"tour has {len(solution)} entries, expected {n}"
        if sorted(solution) != list(range(n)):
            return "tour is not a permutation of all city indices"
        return None

    def evaluate(self, instance: dict, solution) -> float:
        return round(tour_length(instance["cities"], solution), 3)

    def instance_stats(self, instance: dict) -> str:
        return (f"{instance['n']} cities, uniform random in [0,1000]^2, "
                f"seed={instance['seed']}")

    def solver_contract(self) -> str:
        return (
            "Write pure-Python (stdlib only) defining exactly one entry point:\n"
            "    def solve(cities: list[list[float]]) -> list[int]\n"
            "`cities` is a list of [x, y] coordinates. Return a tour: a permutation of "
            "all city indices 0..N-1 (closed tour implied; do not repeat the start). "
            "The code must be COMPLETE and self-contained (define every helper you "
            "call; no placeholders). It must be TIME-BOUNDED: capture t0=time.time() "
            "and keep any improvement loop inside `while time.time()-t0 < budget`, "
            "returning the best tour found so far before the limit — a solver that can "
            "time out scores nothing. No imports beyond the standard library; no I/O; "
            "no threads."
        )


# Harder default sets: mid/large instances where a heuristic written in ~10s
# does NOT trivially reach the optimum, so the gap is informative.
DEFAULT_DEV = ["kroA100", "kroA200", "a280", "lin318", "rd400"]
DEFAULT_HOLDOUT = ["pr299", "pr439", "pcb442", "u574", "rat575"]


class TSPBenchmark(Problem):
    name = "tsp_benchmark"
    description = (
        "Euclidean TSP on the TSPLIB95 benchmark (known optima). The solver is "
        "run on several dev instances; score = mean gap %% above the known "
        "optimum (lower is better, 0 = optimal on every instance). Distances "
        "use the TSPLIB metric: each edge rounded to the nearest integer. "
        "The winning solver is re-evaluated on held-out instances at the end."
    )

    # ------------------------------------------------------------ instance
    def generate_instance(self, params: dict) -> dict:
        dev = params.get("dev") or DEFAULT_DEV
        holdout = params.get("holdout") or DEFAULT_HOLDOUT
        overlap = set(dev) & set(holdout)
        if overlap:
            raise ValueError(f"instances in both dev and holdout: {sorted(overlap)}")
        instances = {n: load_instance(n) for n in [*dev, *holdout]}
        # state-of-the-art reference (LKH) per dev instance, for the tour map
        # and the "how close to SOTA are the agents" comparison
        lkh = {n: lkh_ref(n) for n in dev}
        return {"benchmark": True, "dev": list(dev), "holdout": list(holdout),
                "instances": instances, "lkh": lkh}

    def _dev_items(self, instance: dict):
        return [(n, instance["instances"][n]) for n in instance["dev"]]

    def baseline(self, instance: dict):
        tours, gaps = {}, []
        for name, inst in self._dev_items(instance):
            tour = nn_2opt_tour(inst["cities"])
            tours[name] = tour
            gaps.append(gap_pct(tour_length_euc2d(inst["cities"], tour),
                                inst["optimum"]))
        score = round(sum(gaps) / len(gaps), 3)
        return tours, score, "nearest-neighbor + 2-opt (mean gap %% vs optimum)"

    # ----------------------------------------------------------- scoring
    def validate(self, instance: dict, solution) -> str | None:
        if not isinstance(solution, dict):
            return "solution is not a per-instance dict"
        for name, inst in self._dev_items(instance):
            tour = solution.get(name)
            n = len(inst["cities"])
            if not isinstance(tour, list):
                return f"{name}: no tour produced"
            if len(tour) != n or sorted(tour) != list(range(n)):
                return f"{name}: tour is not a permutation of 0..{n - 1}"
        return None

    def evaluate(self, instance: dict, solution) -> float:
        gaps = [gap_pct(tour_length_euc2d(inst["cities"], solution[name]),
                        inst["optimum"])
                for name, inst in self._dev_items(instance)]
        return round(sum(gaps) / len(gaps), 3)

    def instance_stats(self, instance: dict) -> str:
        sizes = ", ".join(
            f"{n} ({len(instance['instances'][n]['cities'])} cities)"
            for n in instance["dev"])
        return (f"TSPLIB95 dev set: {sizes}. Optima are known; score is the "
                f"mean gap %% above optimum across all dev instances "
                f"(TSPLIB rounded-integer metric). "
                f"{len(instance['holdout'])} held-out instances are kept "
                f"hidden for final verification.")

    def solver_contract(self) -> str:
        return (
            "Write pure-Python (stdlib only) defining exactly one entry point:\n"
            "    def solve(cities: list[list[float]]) -> list[int]\n"
            "`cities` is a list of [x, y] coordinates. Return a tour: a "
            "permutation of all city indices 0..N-1 (closed tour implied). "
            "Your solve() is executed SEPARATELY on each benchmark instance "
            "(100-575 cities each — large enough that reaching the optimum in the "
            "time budget is hard). The code must be COMPLETE and self-contained "
            "(define every helper; no placeholders) and TIME-BOUNDED: capture "
            "t0=time.time() and keep improvement loops inside "
            "`while time.time()-t0 < budget`, returning the best tour found so far "
            "before the per-instance limit — a solver that times out on any instance "
            "scores nothing for the whole run. Edge lengths are rounded to the "
            "nearest integer (TSPLIB EUC_2D), and your tour competes against the "
            "known optimum. The baseline is already nearest-neighbor + 2-opt, so go "
            "beyond plain 2-opt (Or-opt moves, candidate lists, don't-look bits, "
            "perturbation/restarts within the time budget). For reference, the "
            "state-of-the-art solver (LKH) reaches ~0% gap on these; aim as close "
            "to that as the time budget allows. Generalize: the final solver is "
            "re-tested on hidden instances. No imports beyond the standard library; "
            "no I/O; no threads."
        )

    # ---------------------------------------------------------- execution
    def execute(self, code: str, instance: dict, timeout_s: int) -> dict:
        from ..sandbox import run_solver
        solutions, detail, total = {}, {}, 0.0
        for name, inst in self._dev_items(instance):
            out = run_solver(code, {"cities": inst["cities"]}, timeout_s)
            total += out["exec_time"]
            if out["error"]:
                return {"solution": None,
                        "error": f"{name}: {out['error']}",
                        "exec_time": round(total, 3), "detail": detail or None}
            solutions[name] = out["solution"]
            length = tour_length_euc2d(inst["cities"], out["solution"])
            detail[name] = {"length": length, "optimum": inst["optimum"],
                            "gap_pct": gap_pct(length, inst["optimum"]),
                            "lkh_gap": instance.get("lkh", {}).get(name, {}).get("gap_pct"),
                            "time_s": out["exec_time"]}
        return {"solution": solutions, "error": None,
                "exec_time": round(total, 3), "detail": detail}

    # ------------------------------------------------------- held-out set
    def holdout_eval(self, code: str, instance: dict, timeout_s: int) -> dict | None:
        """Run the winning solver on instances it has never seen, against the
        same nearest-neighbor + 2-opt baseline. The orchestrator attaches the
        report to run.completed."""
        from ..sandbox import run_solver
        import time as _time
        per_instance = []
        improved = worsened = unchanged = failed = 0
        b_gaps, w_gaps = [], []
        for name in instance["holdout"]:
            inst = instance["instances"][name]
            t0 = _time.time()
            b_tour = nn_2opt_tour(inst["cities"])
            b_time = round(_time.time() - t0, 3)
            b_gap = gap_pct(tour_length_euc2d(inst["cities"], b_tour),
                            inst["optimum"])
            lk = lkh_ref(name)
            row = {"name": name, "n_cities": len(inst["cities"]),
                   "optimum": inst["optimum"],
                   "baseline_gap": b_gap, "baseline_time": b_time,
                   "lkh_gap": lk["gap_pct"]}
            out = run_solver(code, {"cities": inst["cities"]}, timeout_s)
            tour = out["solution"]
            n = len(inst["cities"])
            valid = (isinstance(tour, list) and len(tour) == n
                     and sorted(tour) == list(range(n)))
            if out["error"] or not valid:
                failed += 1
                row.update({"winner_gap": None, "winner_time": out["exec_time"],
                            "error": out["error"] or "invalid tour",
                            "outcome": "failed"})
            else:
                w_gap = gap_pct(tour_length_euc2d(inst["cities"], tour),
                                inst["optimum"])
                outcome = ("improved" if w_gap < b_gap
                           else "worsened" if w_gap > b_gap else "unchanged")
                improved += outcome == "improved"
                worsened += outcome == "worsened"
                unchanged += outcome == "unchanged"
                b_gaps.append(b_gap)
                w_gaps.append(w_gap)
                row.update({"winner_gap": w_gap, "winner_time": out["exec_time"],
                            "error": None, "outcome": outcome})
            per_instance.append(row)
        n_scored = len(w_gaps)
        lkh_gaps = [r["lkh_gap"] for r in per_instance if r.get("lkh_gap") is not None]
        return {
            "instances": per_instance,
            "summary": {
                "mean_baseline_gap": round(sum(b_gaps) / n_scored, 3) if n_scored else None,
                "mean_winner_gap": round(sum(w_gaps) / n_scored, 3) if n_scored else None,
                "mean_lkh_gap": round(sum(lkh_gaps) / len(lkh_gaps), 3) if lkh_gaps else None,
                "improved": improved, "worsened": worsened,
                "unchanged": unchanged, "failed": failed,
                "generalizes": failed == 0 and n_scored > 0 and
                               sum(w_gaps) / n_scored < sum(b_gaps) / n_scored,
            },
        }

PROBLEMS = {"tsp": TSP(), "tsp_benchmark": TSPBenchmark()}


def tsplib_catalog() -> list[dict]:
    """For the API: available TSPLIB instances with sizes and optima."""
    out = []
    for n in available_instances():
        inst = load_instance(n)
        out.append({"name": n, "n_cities": len(inst["cities"]),
                    "optimum": inst["optimum"]})
    return out
