"""TSPLIB95 instance loading (EUC_2D only) and known optima.

Instances live in backend/data/tsplib/*.tsp (Reinelt's TSPLIB95, public domain).
Optimal tour lengths use the TSPLIB metric: each edge rounded to the nearest
integer (nint). Gaps computed against these optima must use the same metric.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

TSPLIB_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "tsplib"

# Known optimal tour lengths (TSPLIB95 official, EUC_2D rounded metric).
OPTIMA = {
    "eil51": 426, "berlin52": 7542, "st70": 675, "eil76": 538,
    "pr76": 108159, "rat99": 1211, "kroA100": 21282, "kroB100": 22141,
    "rd100": 7910, "eil101": 629, "lin105": 14379, "ch130": 6110,
    "ch150": 6528, "kroA150": 26524, "rat195": 2323, "d198": 15780,
    "kroA200": 29368,
}


def available_instances() -> list[str]:
    return sorted(n for n in OPTIMA if (TSPLIB_DIR / f"{n}.tsp").exists())


@lru_cache(maxsize=None)
def load_instance(name: str) -> dict:
    """Parse a EUC_2D .tsp file -> {"name", "cities", "optimum"}."""
    if name not in OPTIMA:
        raise ValueError(f"unknown TSPLIB instance: {name}")
    path = TSPLIB_DIR / f"{name}.tsp"
    if not path.exists():
        raise FileNotFoundError(f"missing TSPLIB file: {path}")
    cities = []
    in_coords = False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("EDGE_WEIGHT_TYPE") and "EUC_2D" not in line:
            raise ValueError(f"{name}: only EUC_2D instances are supported")
        if line == "NODE_COORD_SECTION":
            in_coords = True
            continue
        if line in ("EOF", "") or not in_coords:
            continue
        parts = line.split()
        if len(parts) >= 3:
            cities.append([float(parts[1]), float(parts[2])])
    if not cities:
        raise ValueError(f"{name}: no coordinates parsed")
    return {"name": name, "cities": cities, "optimum": OPTIMA[name]}


def euc2d(a: list[float], b: list[float]) -> int:
    """TSPLIB EUC_2D distance: Euclidean rounded to nearest integer."""
    return int(math.hypot(a[0] - b[0], a[1] - b[1]) + 0.5)


def tour_length_euc2d(cities: list[list[float]], tour: list[int]) -> int:
    n = len(tour)
    return sum(euc2d(cities[tour[i]], cities[tour[(i + 1) % n]]) for i in range(n))


def gap_pct(length: float, optimum: float) -> float:
    return round((length - optimum) / optimum * 100, 3)


def nn_2opt_tour(cities: list[list[float]]) -> list[int]:
    """Strong classical baseline: nearest-neighbor construction followed by
    2-opt local search to a local optimum (first-improvement passes), all on
    the TSPLIB rounded metric. Typical gap on TSPLIB: ~4-6% above optimum."""
    n = len(cities)
    # nearest-neighbor from city 0
    unvisited = set(range(1, n))
    tour = [0]
    while unvisited:
        cx, cy = cities[tour[-1]]
        nxt = min(unvisited,
                  key=lambda j: (cities[j][0] - cx) ** 2 + (cities[j][1] - cy) ** 2)
        unvisited.remove(nxt)
        tour.append(nxt)

    # 2-opt until no improving move is found
    d = lambda i, j: euc2d(cities[i], cities[j])
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            a, b = tour[i], tour[i + 1]
            d_ab = d(a, b)
            for j in range(i + 2, n):
                c, e = tour[j], tour[(j + 1) % n]
                if a == e:
                    continue
                if d_ab + d(c, e) > d(a, c) + d(b, e):
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True
                    a, b = tour[i], tour[i + 1]
                    d_ab = d(a, b)
    return tour
