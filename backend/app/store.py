"""Event-sourced run storage: JSONL on disk + in-memory index for live reads."""
from __future__ import annotations

import json
import threading
import time
import uuid

from .config import DATA_DIR
from .models import make_event


class Run:
    def __init__(self, run_id: str, config: dict):
        self.id = run_id
        self.config = config
        self.status = "created"
        self.created_at = time.time()
        self.events: list[dict] = []
        self.stop_requested = False
        self._lock = threading.Lock()
        self._dir = DATA_DIR / run_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._save_meta()

    # ---- events ----
    def emit(self, type_: str, agent: str | None = None,
             branch_id: str | None = None, payload: dict | None = None) -> dict:
        with self._lock:
            ev = make_event(len(self.events), type_, agent, branch_id, payload)
            self.events.append(ev)
            with open(self._dir / "events.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        return ev

    def events_since(self, seq: int) -> list[dict]:
        with self._lock:
            return self.events[seq:]

    # ---- meta ----
    def set_status(self, status: str):
        self.status = status
        self._save_meta()

    def _save_meta(self):
        meta = {"id": self.id, "config": self.config, "status": self.status,
                "created_at": self.created_at}
        with open(self._dir / "run.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def public(self) -> dict:
        return {"id": self.id, "config": self.config, "status": self.status,
                "created_at": self.created_at, "num_events": len(self.events)}


class RunStore:
    def __init__(self):
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()
        self._load_existing()

    def _load_existing(self):
        for d in sorted(DATA_DIR.iterdir()) if DATA_DIR.exists() else []:
            meta_path = d / "run.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                run = Run.__new__(Run)
                run.id = meta["id"]
                run.config = meta["config"]
                run.status = meta["status"]
                run.created_at = meta.get("created_at", 0)
                run.stop_requested = False
                run._lock = threading.Lock()
                run._dir = d
                run.events = []
                ev_path = d / "events.jsonl"
                if ev_path.exists():
                    with open(ev_path, encoding="utf-8") as f:
                        run.events = [json.loads(line) for line in f if line.strip()]
                # a run interrupted by server restart can't resume
                if run.status in ("created", "scoping", "running"):
                    run.status = "failed"
                self._runs[run.id] = run
            except Exception:
                continue

    def create(self, config: dict) -> Run:
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        run = Run(run_id, config)
        with self._lock:
            self._runs[run_id] = run
        return run

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def list(self) -> list[dict]:
        return sorted((r.public() for r in self._runs.values()),
                      key=lambda r: r["created_at"], reverse=True)


STORE = RunStore()
