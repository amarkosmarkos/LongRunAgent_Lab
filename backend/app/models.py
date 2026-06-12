"""Core domain objects. Plain dicts on the wire; light classes in the engine."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Branch:
    id: str
    name: str
    hypothesis: str
    strategy: str
    parent_ids: list[str] = field(default_factory=list)
    status: str = "active"  # active | collapsed | merged | winner
    best_score: float | None = None
    best_code: str | None = None
    best_solution: list | None = None
    last_error: str | None = None
    rounds_without_improvement: int = 0
    failures_in_a_row: int = 0
    experiments: int = 0
    cost_usd: float = 0.0

    def public(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "hypothesis": self.hypothesis,
            "strategy": self.strategy,
            "parent_ids": self.parent_ids,
            "status": self.status,
            "best_score": self.best_score,
            "experiments": self.experiments,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class Insight:
    id: str
    branch_id: str
    round: int
    text: str

    def public(self) -> dict:
        return {"id": self.id, "branch_id": self.branch_id, "round": self.round, "text": self.text}


def make_event(seq: int, type_: str, agent: str | None = None,
               branch_id: str | None = None, payload: dict | None = None) -> dict:
    return {
        "seq": seq,
        "ts": time.time(),
        "type": type_,
        "agent": agent,
        "branch_id": branch_id,
        "payload": payload or {},
    }
