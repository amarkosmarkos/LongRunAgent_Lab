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
    # harder, larger instances — out of reach of a plain heuristic in seconds
    "a280": 2579, "pr299": 48191, "lin318": 42029, "rd400": 15281,
    "pr439": 107217, "pcb442": 50778, "u574": 36905, "rat575": 6773,
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


def lkh_tour(cities: list[list[float]]) -> list[int]:
    """State-of-the-art reference: Lin-Kernighan-Helsgaun (LKH) via elkai.
    Reaches the proven optimum on essentially every TSPLIB instance this size.
    Uses the same rounded-integer (EUC_2D) metric as scoring."""
    import elkai
    n = len(cities)
    D = [[euc2d(cities[i], cities[j]) for j in range(n)] for i in range(n)]
    tour = list(elkai.DistanceMatrix(D).solve_tsp())
    if len(tour) > n and tour[-1] == tour[0]:
        tour = tour[:-1]
    return tour


def nn_2opt_tour(cities: list[list[float]], time_limit: float = 8.0) -> list[int]:
    """Classical baseline: nearest-neighbor construction followed by 2-opt local
    search (first-improvement passes), on the TSPLIB rounded metric. Typical gap:
    ~4-8% above optimum. `time_limit` caps the 2-opt phase so large instances
    don't stall the run (it returns the best tour found so far)."""
    import time as _time
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

    # 2-opt until no improving move is found (or the time cap is hit)
    deadline = _time.time() + time_limit
    d = lambda i, j: euc2d(cities[i], cities[j])
    improved = True
    while improved and _time.time() < deadline:
        improved = False
        for i in range(n - 1):
            if _time.time() > deadline:
                break
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
