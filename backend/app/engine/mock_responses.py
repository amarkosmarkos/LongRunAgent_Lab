"""Deterministic mock LLM: scripts the canonical demo arc.

The mock replaces only the *reasoning* (LLM text). All solver code below is real,
is really executed in the sandbox, and is really scored against the baseline —
so mock-mode results are objectively verified, not faked.

Arc: 4 hypotheses -> random-restarts collapses (round 2) -> 2-opt and greedy/or-opt
merge into a hybrid (round 3) -> hybrid iterated-local-search branch wins.
"""
from __future__ import annotations

import json

# --------------------------------------------------------------- solver code

CODE_TWO_OPT_1 = '''
import math, time

def solve(cities):
    n = len(cities)
    def d(a, b):
        return math.hypot(cities[a][0]-cities[b][0], cities[a][1]-cities[b][1])
    # nearest-neighbor seed
    unvisited = set(range(1, n)); tour = [0]
    while unvisited:
        c = tour[-1]
        nxt = min(unvisited, key=lambda j: d(c, j))
        unvisited.remove(nxt); tour.append(nxt)
    # full 2-opt passes until no improvement
    deadline = time.time() + 2.5
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = tour[i-1], tour[i]
                c, e = tour[j], tour[(j+1) % n]
                if d(a, c) + d(b, e) < d(a, b) + d(c, e) - 1e-9:
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
            if time.time() > deadline:
                break
    return tour
'''

CODE_TWO_OPT_2 = '''
import math, time

def solve(cities):
    n = len(cities)
    def d(a, b):
        return math.hypot(cities[a][0]-cities[b][0], cities[a][1]-cities[b][1])
    unvisited = set(range(1, n)); tour = [0]
    while unvisited:
        c = tour[-1]
        nxt = min(unvisited, key=lambda j: d(c, j))
        unvisited.remove(nxt); tour.append(nxt)
    # 2-opt restricted to k-nearest neighbor candidate lists (faster convergence)
    K = min(12, n - 1)
    neigh = [sorted(range(n), key=lambda j, i=i: d(i, j))[1:K+1] for i in range(n)]
    pos = {c: i for i, c in enumerate(tour)}
    deadline = time.time() + 3.0
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for a in range(n):
            i = pos[a]; b = tour[(i+1) % n]
            for c in neigh[a]:
                j = pos[c]; e = tour[(j+1) % n]
                if a == c or b == c or e == a:
                    continue
                if d(a, c) + d(b, e) < d(a, b) + d(c, e) - 1e-9:
                    lo, hi = min(i, j), max(i, j)
                    tour[lo+1:hi+1] = reversed(tour[lo+1:hi+1])
                    pos = {city: idx for idx, city in enumerate(tour)}
                    improved = True
                    break
        if time.time() > deadline:
            break
    return tour
'''

CODE_SA_1 = '''
import math, random, time

def solve(cities):
    n = len(cities)
    def d(a, b):
        return math.hypot(cities[a][0]-cities[b][0], cities[a][1]-cities[b][1])
    def length(t):
        return sum(d(t[i], t[(i+1) % n]) for i in range(n))
    rng = random.Random(7)
    tour = list(range(n)); rng.shuffle(tour)   # random start (mistake: ignores NN seed)
    cur = length(tour)
    T = 10000.0
    deadline = time.time() + 2.5
    while T > 1 and time.time() < deadline:
        i, j = sorted(rng.sample(range(n), 2))
        cand = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
        cl = length(cand)                       # O(n) eval each step: very slow
        if cl < cur or rng.random() < math.exp((cur - cl) / T):
            tour, cur = cand, cl
        T *= 0.999
    return tour
'''

CODE_SA_2 = '''
import math, random, time

def solve(cities):
    n = len(cities)
    def d(a, b):
        return math.hypot(cities[a][0]-cities[b][0], cities[a][1]-cities[b][1])
    rng = random.Random(7)
    # start from nearest-neighbor (insight from lab knowledge)
    unvisited = set(range(1, n)); tour = [0]
    while unvisited:
        c = tour[-1]
        nxt = min(unvisited, key=lambda j: d(c, j))
        unvisited.remove(nxt); tour.append(nxt)
    cur = sum(d(tour[i], tour[(i+1) % n]) for i in range(n))
    T = 80.0
    deadline = time.time() + 3.0
    while time.time() < deadline:
        i, j = sorted(rng.sample(range(1, n), 2))
        a, b = tour[i-1], tour[i]
        c, e = tour[j], tour[(j+1) % n]
        delta = d(a, c) + d(b, e) - d(a, b) - d(c, e)   # O(1) delta evaluation
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
            tour[i:j+1] = reversed(tour[i:j+1])
            cur += delta
        T *= 0.99995
    return tour
'''

CODE_GREEDY_1 = '''
import math

def solve(cities):
    n = len(cities)
    def d(a, b):
        return math.hypot(cities[a][0]-cities[b][0], cities[a][1]-cities[b][1])
    # greedy edge construction with union-find (no subtours, degree <= 2)
    edges = sorted(((d(i, j), i, j) for i in range(n) for j in range(i+1, n)))
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    deg = [0] * n
    adj = [[] for _ in range(n)]
    count = 0
    for w, i, j in edges:
        if count == n - 1:
            break
        if deg[i] < 2 and deg[j] < 2 and find(i) != find(j):
            parent[find(i)] = find(j)
            adj[i].append(j); adj[j].append(i)
            deg[i] += 1; deg[j] += 1; count += 1
    ends = [i for i in range(n) if deg[i] < 2]
    adj[ends[0]].append(ends[1]); adj[ends[1]].append(ends[0])
    tour, prev, cur = [0], None, 0
    while len(tour) < n:
        nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
        tour.append(nxt); prev, cur = cur, nxt
    return tour
'''

CODE_GREEDY_2 = '''
import math, time

def solve(cities):
    n = len(cities)
    def d(a, b):
        return math.hypot(cities[a][0]-cities[b][0], cities[a][1]-cities[b][1])
    edges = sorted(((d(i, j), i, j) for i in range(n) for j in range(i+1, n)))
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    deg = [0] * n
    adj = [[] for _ in range(n)]
    count = 0
    for w, i, j in edges:
        if count == n - 1:
            break
        if deg[i] < 2 and deg[j] < 2 and find(i) != find(j):
            parent[find(i)] = find(j)
            adj[i].append(j); adj[j].append(i)
            deg[i] += 1; deg[j] += 1; count += 1
    ends = [i for i in range(n) if deg[i] < 2]
    adj[ends[0]].append(ends[1]); adj[ends[1]].append(ends[0])
    tour, prev, cur = [0], None, 0
    while len(tour) < n:
        nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
        tour.append(nxt); prev, cur = cur, nxt
    # or-opt: relocate segments of length 1-3
    deadline = time.time() + 2.5
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for seg in (1, 2, 3):
            for i in range(n - seg):
                a = tour[i-1]; b = tour[i]; c = tour[i+seg-1]; e = tour[(i+seg) % n]
                removed = d(a, b) + d(c, e) - d(a, e)
                for k in range(n):
                    if i - 1 <= k <= i + seg - 1:
                        continue
                    p, q = tour[k], tour[(k+1) % n]
                    gain = removed - (d(p, b) + d(c, q) - d(p, q))
                    if gain > 1e-9:
                        segment = tour[i:i+seg]
                        rest = tour[:i] + tour[i+seg:]
                        kk = rest.index(p)
                        tour = rest[:kk+1] + segment + rest[kk+1:]
                        improved = True
                        break
                if improved:
                    break
            if improved or time.time() > deadline:
                break
    return tour
'''

CODE_RANDOM_1 = '''
import math, random, time

def solve(cities):
    n = len(cities)
    def length(t):
        return sum(math.hypot(cities[t[i]][0]-cities[t[(i+1)%n]][0],
                              cities[t[i]][1]-cities[t[(i+1)%n]][1]) for i in range(n))
    rng = random.Random(1)
    best, best_len = None, float("inf")
    deadline = time.time() + 2.0
    while time.time() < deadline:
        t = list(range(n)); rng.shuffle(t)
        l = length(t)
        if l < best_len:
            best, best_len = t, l
    return best
'''

CODE_RANDOM_2 = '''
import math, random, time

def solve(cities):
    n = len(cities)
    def length(t):
        return sum(math.hypot(cities[t[i]][0]-cities[t[(i+1)%n]][0],
                              cities[t[i]][1]-cities[t[(i+1)%n]][1]) for i in range(n))
    rng = random.Random(2)
    best, best_len = None, float("inf")
    deadline = time.time() + 2.0
    while time.time() < deadline:
        t = list(range(n)); rng.shuffle(t)
        # one greedy swap pass (still fundamentally random sampling)
        for _ in range(50):
            i, j = rng.sample(range(n), 2)
            t[i], t[j] = t[j], t[i]
        l = length(t)
        if l < best_len:
            best, best_len = t, l
    return best
'''

CODE_HYBRID_1 = '''
import math, random, time

def solve(cities):
    n = len(cities)
    def d(a, b):
        return math.hypot(cities[a][0]-cities[b][0], cities[a][1]-cities[b][1])
    def length(t):
        return sum(d(t[i], t[(i+1) % n]) for i in range(n))
    rng = random.Random(11)

    def nn_tour():
        unvisited = set(range(1, n)); t = [0]
        while unvisited:
            c = t[-1]
            nxt = min(unvisited, key=lambda j: d(c, j))
            unvisited.remove(nxt); t.append(nxt)
        return t

    def two_opt(t, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    a, b = t[i-1], t[i]
                    c, e = t[j], t[(j+1) % n]
                    if d(a, c) + d(b, e) < d(a, b) + d(c, e) - 1e-9:
                        t[i:j+1] = reversed(t[i:j+1])
                        improved = True
                if time.time() > deadline:
                    break
        return t

    def or_opt(t, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for seg in (1, 2, 3):
                for i in range(1, n - seg):
                    a, b = t[i-1], t[i]
                    c, e = t[i+seg-1], t[(i+seg) % n]
                    removed = d(a, b) + d(c, e) - d(a, e)
                    if removed <= 1e-9:
                        continue
                    for k in range(n - 1):
                        if i - 1 <= k <= i + seg - 1:
                            continue
                        p, q = t[k], t[k+1]
                        if removed - (d(p, b) + d(c, q) - d(p, q)) > 1e-9:
                            segment = t[i:i+seg]
                            rest = t[:i] + t[i+seg:]
                            kk = rest.index(p)
                            t = rest[:kk+1] + segment + rest[kk+1:]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return t

    def double_bridge(t):
        a, b, c = sorted(rng.sample(range(1, n), 3))
        return t[:a] + t[c:] + t[b:c] + t[a:b]

    deadline = time.time() + 5.0
    best = two_opt(nn_tour(), time.time() + 1.5)
    best = or_opt(best, time.time() + 1.0)
    best_len = length(best)
    # iterated local search: perturb (double bridge) + re-optimize
    while time.time() < deadline:
        cand = double_bridge(best[:])
        cand = two_opt(cand, min(deadline, time.time() + 0.8))
        cand = or_opt(cand, min(deadline, time.time() + 0.4))
        cl = length(cand)
        if cl < best_len:
            best, best_len = cand, cl
    return best
'''

CODE_HYBRID_2 = CODE_HYBRID_1.replace("time.time() + 5.0", "time.time() + 7.0") \
                             .replace("random.Random(11)", "random.Random(13)")

# ------------------------------------------------------------ scripted text

_HYPOTHESES = [
    {"name": "2-opt local search", "strategy": "two-opt",
     "hypothesis": "A nearest-neighbor tour contains crossing edges; iteratively reversing segments (2-opt) will remove them and shorten the tour well below baseline.",
     "risk": "2-opt converges to a local optimum and may stall."},
    {"name": "Simulated annealing", "strategy": "simulated-annealing",
     "hypothesis": "Accepting some worse moves with decreasing probability escapes local optima that trap greedy local search, yielding better tours than pure descent.",
     "risk": "Highly sensitive to temperature schedule; may waste the time budget."},
    {"name": "Greedy edge + or-opt", "strategy": "greedy-construction",
     "hypothesis": "Building the tour from globally shortest edges (instead of a greedy walk) gives a stronger starting structure, improvable by relocating short segments (or-opt).",
     "risk": "Greedy matching can create long 'closing' edges that local moves cannot fix."},
    {"name": "Random restarts", "strategy": "random-restarts",
     "hypothesis": "Sampling many random tours and keeping the best will eventually find good structure through sheer volume.",
     "risk": "The space of tours is factorially large; sampling may never reach baseline quality."},
]

_EXPERIMENTS = {
    ("two-opt", 1): ("Seed with nearest-neighbor, then run full 2-opt passes until convergence.",
                     CODE_TWO_OPT_1),
    ("two-opt", 2): ("Restrict 2-opt to 12-nearest-neighbor candidate lists for faster convergence within the time cap.",
                     CODE_TWO_OPT_2),
    ("simulated-annealing", 1): ("Classic SA from a random tour, geometric cooling from T=10000, full-length re-evaluation per move.",
                                 CODE_SA_1),
    ("simulated-annealing", 2): ("Adopt lab insight: start from NN tour, O(1) delta evaluation for 2-opt moves, low initial temperature.",
                                 CODE_SA_2),
    ("greedy-construction", 1): ("Greedy edge construction with union-find to avoid subtours and degree>2.",
                                 CODE_GREEDY_1),
    ("greedy-construction", 2): ("Add or-opt: relocate segments of 1-3 cities to repair long edges left by greedy matching.",
                                 CODE_GREEDY_2),
    ("random-restarts", 1): ("Sample random permutations for 2 seconds, keep the best.",
                             CODE_RANDOM_1),
    ("random-restarts", 2): ("Add a random swap pass per sample; still volume-based search.",
                             CODE_RANDOM_2),
    ("hybrid-merge", 1): ("Combine merged insights: NN seed -> 2-opt -> or-opt, then iterated local search with double-bridge perturbation.",
                          CODE_HYBRID_1),
    ("hybrid-merge", 2): ("Extend ILS time budget and reseed perturbation RNG; keep best-of-all-restarts.",
                          CODE_HYBRID_2),
}

_INSIGHTS = {
    ("two-opt", 1): "2-opt removes edge crossings fast but stalls in a local optimum; perturbation (e.g. double-bridge) plus re-optimization should escape it.",
    ("two-opt", 2): None,
    ("simulated-annealing", 1): "Random starting tours and O(n) move evaluation waste the entire time budget; always seed with a constructive heuristic and use O(1) delta evaluation.",
    ("simulated-annealing", 2): None,
    ("greedy-construction", 1): "Greedy edge construction beats greedy-walk construction, but leaves a few long edges that 2-opt-style reversals cannot fix.",
    ("greedy-construction", 2): "Or-opt segment relocation (1-3 cities) repairs exactly the long-edge defects that 2-opt misses; the two moves are complementary.",
    ("random-restarts", 1): None,
    ("random-restarts", 2): None,
    ("hybrid-merge", 1): "Alternating 2-opt and or-opt inside an iterated local search compounds both moves' strengths.",
    ("hybrid-merge", 2): None,
}


def mock_call(role: str, ctx: dict) -> str:
    if role == "researcher":
        return ("State-of-the-art for Euclidean TSP at this size:\n"
                "- Lin-Kernighan-Helsgaun (LKH) reaches ~0% gap in seconds; the "
                "practical gold standard.\n"
                "- Or-opt + 2-opt inside an iterated local search with double-bridge "
                "perturbation typically lands within ~1-3% of optimum in a few seconds.\n"
                "- Candidate lists (k-nearest) and don't-look bits are essential for "
                "speed on larger instances.\n"
                "- Pitfall: unbounded local search times out; always keep a wall-clock "
                "budget and return the best-so-far.")

    if role == "planner" and ctx.get("review"):
        # planner reviews the round; the mock keeps the demo arc stable by not
        # spawning extra branches (real mode generates new hypotheses here)
        rnd = ctx.get("round", 1)
        return json.dumps({
            "new_hypotheses": [],
            "continue": True,
            "reasoning": f"Round {rnd}: branches are differentiating; continue and let the supervisor prune/merge."})

    if role == "planner":
        cfg = ctx.get("config", {})
        return json.dumps({
            "initial_hypotheses": cfg.get("num_hypotheses", 4),
            "objective": "Find a closed tour at least {:.0f}% shorter than the nearest-neighbor baseline on this instance.".format(cfg.get("target_improvement_pct", 18)),
            "success_criteria": {
                "target_improvement_pct": cfg.get("target_improvement_pct", 18.0),
                "rationale": "Nearest-neighbor is typically 20-25% above optimal on uniform instances. A target near {:.0f}% is close to the optimality gap: no single basic heuristic should reach it, so it forces combining discoveries.".format(cfg.get("target_improvement_pct", 18.0))},
            "constraints": [
                f"Each experiment must finish within {cfg.get('experiment_timeout_s', 10)}s",
                "Solvers must be pure Python (stdlib only), single-threaded, no I/O",
                "Every reported score is re-verified by the engine, never trusted from agents"],
            "stop_conditions": [
                f"Budget of ${cfg.get('budget_usd', 2.0)} USD exhausted",
                f"{cfg.get('max_rounds', 5)} rounds completed",
                "Target improvement reached and confirmed",
                "All branches collapsed"],
            "reasoning": "With ~60 cities, exact methods are infeasible in seconds, but strong heuristics (local search, metaheuristics, better construction) routinely beat nearest-neighbor by 10-20%. A staged exploration of diverse strategies with knowledge sharing should reach the target.",
        })

    if role == "strategist":
        k = ctx.get("k", 4)
        return json.dumps({"hypotheses": _HYPOTHESES[:k]})

    if role == "experimenter":
        key = (ctx.get("strategy"), min(ctx.get("attempt", 1), 2))
        approach, code = _EXPERIMENTS.get(key, _EXPERIMENTS[("two-opt", 1)])
        return json.dumps({"approach": approach,
                           "expectation": "Lower tour length than this branch's previous best."}) \
            + "\n```python\n" + code + "\n```"

    if role == "critic":
        strategy = ctx.get("strategy")
        attempt = min(ctx.get("attempt", 1), 2)
        error = ctx.get("error")
        improved = ctx.get("improved", False)
        beats_baseline = ctx.get("beats_baseline", False)
        if error:
            verdict, analysis = "failed", f"The experiment failed: {error}. The implementation, not the hypothesis, is at fault; fix and retry."
        elif improved and beats_baseline:
            verdict, analysis = "improved", "The change produced a measurably shorter tour than both the branch's previous best and the baseline, supporting the hypothesis."
        elif improved:
            verdict, analysis = "improved", "Improvement over the branch's previous attempt, but still above the baseline; the direction works yet needs more strength."
        else:
            verdict, analysis = "no_improvement", "No measurable gain. The search either wastes its time budget or explores unstructured regions of the solution space."
        insight = _INSIGHTS.get((strategy, attempt))
        return json.dumps({
            "verdict": verdict, "analysis": analysis, "insight": insight,
            "suggestion": "Incorporate shared lab insights and tighten the inner-loop time budget." })

    if role == "supervisor":
        rnd = ctx.get("round", 1)
        branches = ctx.get("branches", [])
        by_strategy = {b["strategy"]: b for b in branches if b["status"] == "active"}
        decision = {"collapse": [], "merge": None,
                    "reasoning": "Branches are still differentiating; everyone continues."}
        if rnd >= 2 and "random-restarts" in by_strategy:
            b = by_strategy["random-restarts"]
            decision["collapse"].append({
                "branch_id": b["id"],
                "reason": "Two rounds without approaching baseline: random sampling cannot compete in a factorial space. Evidence: best score far above baseline while every other branch improved."})
            decision["reasoning"] = "Random restarts is clearly dominated and is collapsed to stop spending budget on it."
        if rnd >= 3 and "two-opt" in by_strategy and "greedy-construction" in by_strategy:
            a, b = by_strategy["two-opt"], by_strategy["greedy-construction"]
            decision["merge"] = {
                "source_ids": [a["id"], b["id"]],
                "name": "Hybrid ILS (2-opt + or-opt)",
                "hypothesis": "2-opt and or-opt fix complementary defects (crossings vs misplaced segments); alternating them inside an iterated local search with double-bridge perturbation escapes the local optima where each stalls alone.",
                "strategy": "hybrid-merge",
                "reason": "Branch insights are complementary: 2-opt stalls at local optima needing perturbation; or-opt repairs exactly the defects 2-opt cannot. Merging concentrates the remaining budget on the combination."}
            decision["reasoning"] = "The two strongest branches discovered complementary moves; merging them is the highest-expected-value use of remaining budget."
        return json.dumps(decision)

    return "{}"
