"""FastAPI app: start runs, read/stream events, stop runs."""
from __future__ import annotations

import asyncio
import json
import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import DEFAULT_RUN_CONFIG, LLM_MOCK
from .engine.orchestrator import Orchestrator
from .problems import PROBLEMS
from .store import STORE

app = FastAPI(title="Long Run Agent Lab")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True, "mock_mode": LLM_MOCK}


@app.get("/api/problems")
def problems():
    from .problems.tsp import DEFAULT_DEV, DEFAULT_HOLDOUT, tsplib_catalog
    return {
        "problems": [{"name": p.name, "description": p.description}
                     for p in PROBLEMS.values()],
        "tsplib": {"catalog": tsplib_catalog(),
                   "default_dev": DEFAULT_DEV,
                   "default_holdout": DEFAULT_HOLDOUT},
    }


@app.get("/api/runs")
def list_runs():
    return STORE.list()


@app.post("/api/runs")
def create_run(config: dict | None = None):
    cfg = {**DEFAULT_RUN_CONFIG, **(config or {})}
    if cfg["problem"] not in PROBLEMS:
        raise HTTPException(400, f"unknown problem: {cfg['problem']}")
    run = STORE.create(cfg)
    orch = Orchestrator(run)
    threading.Thread(target=orch.execute, daemon=True,
                     name=f"run-{run.id}").start()
    return run.public()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = STORE.get(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return run.public()


@app.get("/api/runs/{run_id}/events")
def get_events(run_id: str, since: int = 0):
    run = STORE.get(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return {"status": run.status, "events": run.events_since(since)}


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str):
    run = STORE.get(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    run.stop_requested = True
    return {"ok": True}


@app.get("/api/runs/{run_id}/stream")
async def stream(run_id: str, since: int = 0):
    run = STORE.get(run_id)
    if not run:
        raise HTTPException(404, "run not found")

    async def gen():
        seq = since
        while True:
            events = run.events_since(seq)
            for ev in events:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                seq = ev["seq"] + 1
            if run.status in ("completed", "failed", "stopped", "budget_exceeded") \
                    and not events:
                yield f"data: {json.dumps({'type': 'stream.end', 'status': run.status})}\n\n"
                return
            await asyncio.sleep(0.4)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
