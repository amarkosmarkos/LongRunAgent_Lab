"""Offline originality scorer.

Takes the winning solver of each completed run and asks an Anthropic judge —
armed with server-side web search — whether the algorithmic idea already exists
on the internet or is something the lab created. Cross the verdict with the
score the run already produced and you get a simple map of where real, original
knowledge is being created versus where we are just reheating textbook methods.

The same judge runs automatically at the end of every live run (see
app.engine.orchestrator); this script re-scores stored runs in bulk, including
older ones that finished before the judge existed.

Run from the backend directory:

    python -m app.scripts.originality                 # score every completed run
    python -m app.scripts.originality 20260621-120000-ab12cd   # one run by id
    python -m app.scripts.originality --limit 5       # newest 5 only

Requires ANTHROPIC_API_KEY in the environment (or backend/.env).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ..config import ANTHROPIC_API_KEY, DATA_DIR
from ..originality import judge, quadrant

OUT_DIR = DATA_DIR.parent / "originality"

# ---------------------------------------------------------------------------
# Reading stored runs
# ---------------------------------------------------------------------------


def _load_run(run_dir: Path) -> dict | None:
    """Return {id, problem, results} for a completed run, or None if it has no
    winning solver to judge."""
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return None
    problem_name = "unknown"
    results: dict | None = None
    with open(events_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "run.created":
                problem_name = ev.get("payload", {}).get("problem", {}).get(
                    "name", problem_name)
            elif ev.get("type") == "run.completed":
                results = ev.get("payload", {}).get("results")
    if not results or not results.get("winner_code"):
        return None
    return {"id": run_dir.name, "problem": problem_name, "results": results}


def _iter_runs(run_ids: list[str] | None):
    dirs = sorted(DATA_DIR.iterdir()) if DATA_DIR.exists() else []
    for d in dirs:
        if not d.is_dir():
            continue
        if run_ids and d.name not in run_ids:
            continue
        run = _load_run(d)
        if run is not None:
            yield run


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_table(rows: list[dict]) -> None:
    headers = ["run", "problem", "improv%", "orig", "online", "nearest", "quadrant"]
    widths = [24, 8, 8, 5, 7, 28, 14]

    def fmt(values):
        cells = []
        for val, w in zip(values, widths):
            s = "" if val is None else str(val)
            if len(s) > w:
                s = s[: w - 1] + "…"
            cells.append(s.ljust(w))
        return "  ".join(cells)

    print(fmt(headers))
    print(fmt(["-" * w for w in widths]))
    for r in rows:
        v = r.get("verdict", {})
        improv = r["results"].get("improvement_pct")
        print(fmt([
            r["id"],
            r["problem"],
            "n/a" if improv is None else f"{improv:.1f}",
            v.get("originality"),
            v.get("exists_online"),
            v.get("nearest_known_technique"),
            r["quadrant"],
        ]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="*", help="specific run ids to score")
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the newest N runs")
    args = parser.parse_args(argv)

    if not ANTHROPIC_API_KEY:
        print("ANTHROPIC_API_KEY is not set; the judge needs the real API.",
              file=sys.stderr)
        return 2

    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    runs = list(_iter_runs(args.run_ids or None))
    runs.sort(key=lambda r: r["id"], reverse=True)
    if args.limit is not None:
        runs = runs[: args.limit]

    if not runs:
        print("No completed runs with a winning solver were found.")
        return 0

    rows: list[dict] = []
    for i, run in enumerate(runs, 1):
        results = run["results"]
        print(f"[{i}/{len(runs)}] judging {run['id']} ...", file=sys.stderr)
        verdict, _in, _out = judge(client, results["winner_code"])
        target_pct = results.get("target_improvement_pct")
        run["verdict"] = verdict
        run["quadrant"] = quadrant(verdict.get("originality"),
                                   results.get("improvement_pct"), target_pct)
        rows.append(run)

    print()
    _print_table(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / (time.strftime("%Y%m%d-%H%M%S") + ".json")
    report = [{
        "run_id": r["id"],
        "problem": r["problem"],
        "best_score": r["results"].get("best_score"),
        "baseline_score": r["results"].get("baseline_score"),
        "improvement_pct": r["results"].get("improvement_pct"),
        "quadrant": r["quadrant"],
        "verdict": r["verdict"],
    } for r in rows]
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nFull report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
