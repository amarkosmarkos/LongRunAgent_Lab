"""Persistent lab memory: a cross-run knowledge archive (MAP-Elites style).

Every run used to start from zero: the web research was redone, winning solvers
and hard-won insights were written to that run's event log and never read again.
This module gives the lab long-term memory:

 - An ARCHIVE of winning solvers kept per technique niche (the best solver for
   each combination of algorithmic techniques survives — quality-diversity, not
   just a single global best).
 - A pool of transferable insights accumulated across runs.
 - Lexical relevance ranking (IDF-weighted token overlap) so callers retrieve
   the top-k relevant items instead of dumping everything into a prompt.
   No embeddings API is required, so this works identically in mock mode.

The orchestrator recalls from the archive before planning (so hypotheses build
on what previous runs learned) and ingests the run's outcome at conclusion.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading

from .config import DATA_DIR

ARCHIVE_DIR = DATA_DIR.parent / "knowledge"
ARCHIVE_PATH = ARCHIVE_DIR / "archive.json"

MAX_INSIGHTS = 300          # FIFO cap on the cross-run insight pool
RECALL_SOLVERS = 5          # elites surfaced to the planner/strategist
RECALL_INSIGHTS = 8         # insights surfaced to the planner/strategist

# keyword -> technique tag; a solver's niche is the SET of techniques it uses,
# so "2-opt alone" and "2-opt + or-opt + annealing" occupy different niches
_TECHNIQUE_RULES = [
    (r"christofides", "christofides"),
    (r"lin[-_ ]?kernighan|lk[-_ ]?move|\blkh\b", "lin-kernighan"),
    (r"or[-_ ]?opt", "or-opt"),
    (r"3[-_ ]?opt", "3-opt"),
    (r"2[-_ ]?opt", "2-opt"),
    (r"anneal|temperature|cooling", "simulated-annealing"),
    (r"pheromone|ant[-_ ]colony|\baco\b", "ant-colony"),
    (r"genetic|crossover|mutation|population", "genetic"),
    (r"tabu", "tabu-search"),
    (r"convex[-_ ]?hull", "convex-hull"),
    (r"insertion", "insertion"),
    (r"greedy[-_ ]?edge|greedy", "greedy"),
    (r"nearest[-_ ]?neigh|nearest", "nearest-neighbour"),
    (r"segment[-_ ]?revers|perturb|restart|kick", "perturbation"),
    (r"candidate[-_ ]?list|k[-_ ]?nearest|neighbor[-_ ]?list", "candidate-lists"),
]
_TECHNIQUE_RES = [(re.compile(pat, re.IGNORECASE), tag) for pat, tag in _TECHNIQUE_RULES]

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]+")
_STOP = frozenset(
    "the a an and or of to in for with is are be this that it its on by from "
    "as at not no we you your our will can could should must into over under "
    "then than when while each per all any some more most very".split())


def technique_tags(code: str | None, *extra_texts: str | None) -> list[str]:
    """Name the algorithmic techniques present in a solver (code + hypothesis)."""
    blob = " ".join(t for t in (code, *extra_texts) if t)
    tags = [tag for rx, tag in _TECHNIQUE_RES if rx.search(blob)]
    # dedupe preserving rule order (most specific rules come first)
    return list(dict.fromkeys(tags))


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP]


def rank(query: str, docs: list[str], k: int) -> list[int]:
    """Return the indices of the k docs most relevant to the query, using
    IDF-weighted token overlap. Pure lexical — deterministic and API-free."""
    if not docs:
        return []
    q = set(_tokenize(query))
    if not q:
        return list(range(min(k, len(docs))))
    doc_tokens = [set(_tokenize(d)) for d in docs]
    n = len(docs)
    idf = {t: math.log(1 + n / (1 + sum(1 for dt in doc_tokens if t in dt)))
           for t in q}
    scored = sorted(
        ((sum(idf[t] for t in q & dt), i) for i, dt in enumerate(doc_tokens)),
        key=lambda si: (-si[0], si[1]))
    return [i for score, i in scored[:k] if score > 0]


class KnowledgeArchive:
    """JSON-backed archive of elite solvers (best per technique niche) and a
    cross-run pool of insights. Thread-safe; writes are atomic."""

    def __init__(self, path=ARCHIVE_PATH):
        self._path = path
        self._lock = threading.Lock()
        self._data = {"solvers": {}, "insights": [], "ingested_runs": []}
        self._load()
        self._backfill()

    # ------------------------------------------------------------- storage
    def _load(self):
        try:
            self._data.update(json.loads(self._path.read_text(encoding="utf-8")))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass

    def _save(self):
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, self._path)

    def _backfill(self):
        """Ingest completed runs recorded before the archive existed, so the
        lab's memory starts full instead of empty."""
        if not DATA_DIR.exists():
            return
        for run_dir in sorted(DATA_DIR.iterdir()):
            run_id = run_dir.name
            if run_id in self._data["ingested_runs"]:
                continue
            ev_path = run_dir / "events.jsonl"
            if not ev_path.exists():
                continue
            problem, results, insights = "", None, []
            try:
                with open(ev_path, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        ev = json.loads(line)
                        p = ev.get("payload") or {}
                        if ev.get("type") == "run.created":
                            problem = (p.get("problem") or {}).get("name", "")
                        elif ev.get("type") == "insight.added":
                            text = (p.get("insight") or {}).get("text")
                            if text:
                                insights.append(text)
                        elif ev.get("type") == "run.completed":
                            results = p.get("results")
            except (json.JSONDecodeError, OSError):
                continue
            if results is not None:
                self.ingest_run(run_id, problem, results, insights, save=False)
        self._save()

    # -------------------------------------------------------------- ingest
    def ingest_run(self, run_id: str, problem: str, results: dict,
                   insights: list[str], save: bool = True) -> dict:
        """Archive a concluded run: the winning solver competes for its niche,
        insights join the shared pool. Returns what actually changed."""
        code = results.get("winner_code")
        imp = results.get("improvement_pct")
        tags = technique_tags(code, results.get("winner_branch_name"))
        niche = f"{problem}::" + ("+".join(tags) if tags else "unclassified")
        outcome = {"solver_added": False, "niche": niche, "insights_added": 0}
        with self._lock:
            if run_id in self._data["ingested_runs"]:
                return outcome
            self._data["ingested_runs"].append(run_id)
            if code and imp is not None:
                incumbent = self._data["solvers"].get(niche)
                # MAP-Elites rule: the niche keeps only its best solver
                if incumbent is None or imp > incumbent.get("improvement_pct", -1e9):
                    orig = ((results.get("originality") or {}).get("verdict")
                            or {})
                    self._data["solvers"][niche] = {
                        "run_id": run_id,
                        "problem": problem,
                        "name": results.get("winner_branch_name"),
                        "techniques": tags,
                        "score": results.get("best_score"),
                        "improvement_pct": imp,
                        "holdout_generalizes": ((results.get("holdout") or {})
                                                .get("summary") or {}).get("generalizes"),
                        "originality": orig.get("originality"),
                        "mechanism": orig.get("mechanism"),
                        "quadrant": (results.get("originality") or {}).get("quadrant"),
                        "code": code,
                    }
                    outcome["solver_added"] = True
            known = {i["text"] for i in self._data["insights"]}
            for text in insights:
                if text and text not in known:
                    self._data["insights"].append({"text": text, "run_id": run_id})
                    known.add(text)
                    outcome["insights_added"] += 1
            self._data["insights"] = self._data["insights"][-MAX_INSIGHTS:]
            if save:
                self._save()
        outcome["archive_size"] = self.size()
        return outcome

    # -------------------------------------------------------------- recall
    def size(self) -> dict:
        return {"solvers": len(self._data["solvers"]),
                "insights": len(self._data["insights"]),
                "runs": len(self._data["ingested_runs"])}

    def recall(self, problem: str, query: str,
               k_solvers: int = RECALL_SOLVERS,
               k_insights: int = RECALL_INSIGHTS) -> dict | None:
        """Retrieve the archive knowledge most relevant to a new run: the
        strongest elites for this problem plus the top-k relevant insights.
        Returns None when the archive has nothing to offer."""
        with self._lock:
            elites = [s for s in self._data["solvers"].values()
                      if not problem or s.get("problem") in ("", problem)]
            insights = list(self._data["insights"])
        elites.sort(key=lambda s: -(s.get("improvement_pct") or 0))
        elites = elites[:k_solvers]
        idx = rank(query, [i["text"] for i in insights], k_insights)
        picked = [insights[i]["text"] for i in idx]
        if not elites and not picked:
            return None
        return {
            "solvers": [{k: v for k, v in s.items() if k != "code"}
                        for s in elites],
            "insights": picked,
            "archive_size": self.size(),
        }

    @staticmethod
    def as_prompt(recall: dict) -> str:
        """Render a recall digest as a prompt section for planner/strategist."""
        lines = ["LAB MEMORY (what previous runs on this problem already learned):"]
        for s in recall.get("solvers", []):
            bits = [f"- \"{s.get('name')}\" [{'+'.join(s.get('techniques') or []) or 'unclassified'}]",
                    f"improved {s.get('improvement_pct')}% over baseline"]
            if s.get("originality") is not None:
                bits.append(f"originality {s['originality']}/10")
            if s.get("holdout_generalizes") is not None:
                bits.append("generalizes" if s["holdout_generalizes"]
                            else "does NOT generalize on held-out instances")
            if s.get("mechanism"):
                bits.append(f"mechanism: {s['mechanism']}")
            lines.append("; ".join(bits))
        if recall.get("insights"):
            lines.append("Transferable insights from past runs:")
            lines.extend(f"- {t}" for t in recall["insights"])
        lines.append("Build on this memory: prefer directions it says work, avoid "
                     "repeating what it says fails, and aim for niches it has not "
                     "explored yet.")
        return "\n".join(lines)


ARCHIVE = KnowledgeArchive()
