"""Freeze real runs into static JSON the frontend can replay with no backend.

The live app reads runs from the FastAPI backend. To publish a self-contained
demo (e.g. GitHub Pages) there is no backend, so this script snapshots chosen
runs' event logs into `frontend/public/demo/`:

  demo/runs.json        -> the runs-list index (id, config, status, ...)
  demo/<run_id>.json    -> {"status": ..., "events": [...]} for one run

The frontend's demo API layer (see frontend/src/api.js, VITE_DEMO=1) serves
these files instead of calling /api, so the whole experience — branch graph,
story, replay, originality, lab memory — works fully static.

Usage (from backend/):
  python -m app.scripts.export_demo                 # every run on disk
  python -m app.scripts.export_demo <run_id> ...    # only the given runs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ..config import DATA_DIR

DEMO_DIR = (Path(__file__).resolve().parents[3] / "frontend" / "public" / "demo")

def _all_runs() -> list[str]:
    """Every run directory on disk that has an event log, so the published demo
    mirrors the full experiment history exactly like the live runs list."""
    if not DATA_DIR.exists():
        return []
    return sorted(d.name for d in DATA_DIR.iterdir()
                  if (d / "events.jsonl").exists() and (d / "run.json").exists())


def _load(run_id: str) -> tuple[dict, list[dict]] | None:
    run_dir = DATA_DIR / run_id
    meta_path, ev_path = run_dir / "run.json", run_dir / "events.jsonl"
    if not meta_path.exists() or not ev_path.exists():
        print(f"  ! skip {run_id}: missing run.json or events.jsonl")
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    events = [json.loads(line) for line in ev_path.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    return meta, events


def main(argv: list[str]) -> int:
    run_ids = argv or _all_runs()
    # a fresh export shouldn't leave stale run files behind
    if not argv:
        for old in DEMO_DIR.glob("*.json"):
            old.unlink()
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for run_id in run_ids:
        loaded = _load(run_id)
        if loaded is None:
            continue
        meta, events = loaded
        (DEMO_DIR / f"{run_id}.json").write_text(
            json.dumps({"status": meta["status"], "events": events},
                       ensure_ascii=False),
            encoding="utf-8")
        index.append({
            "id": meta["id"],
            "config": meta["config"],
            "status": meta["status"],
            "created_at": meta.get("created_at", 0),
            "num_events": len(events),
        })
        print(f"  + {run_id}: {len(events)} events, status={meta['status']}")
    # newest first, mirroring the live RunStore.list() ordering
    index.sort(key=lambda r: r["created_at"], reverse=True)
    (DEMO_DIR / "runs.json").write_text(
        json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(index)} run(s) to {DEMO_DIR}")
    return 0 if index else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
