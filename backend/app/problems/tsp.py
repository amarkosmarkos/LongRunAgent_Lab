"""Euclidean TSP. Baseline: nearest-neighbor construction from city 0."""
from __future__ import annotations

import math
import random

from .base import Problem


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
            "It must finish within the time limit — prefer fast heuristics over "
            "exhaustive search. No imports beyond the standard library; no I/O; "
            "no threads."
        )


PROBLEMS = {"tsp": TSP()}
