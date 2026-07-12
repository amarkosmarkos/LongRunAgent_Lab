"""Originality judging: is a winning solver a genuinely novel idea, or a
reheated textbook method that already lives on the internet?

A judge model reads the solver, names its search mechanism, searches the web for
that mechanism, and rules on whether it is published. The verdict is crossed
with the score the run already produced: originality without quality is noise,
quality without originality is a rehash — the prize is the "novel AND it wins"
quadrant.

This module is shared by the live run flow (app.engine.orchestrator, via
app.llm.LLMClient.judge_originality) and the offline scorer
(app.scripts.originality). Keep it free of run/orchestrator state.
"""
from __future__ import annotations

import json
import re

# The judge uses the most capable model: it reads the solver, names the search
# mechanism, searches the web, and rules on whether that mechanism is published.
JUDGE_MODEL = "claude-opus-4-8"
JUDGE_MAX_TOKENS = 8000

SYSTEM = (
    "You are a rigorous judge of algorithmic originality. You are given a Python "
    "solver written by an autonomous research agent for a combinatorial "
    "optimization problem (Traveling Salesman). Your job is to decide whether the "
    "underlying ALGORITHMIC IDEA already exists in the public literature / on the "
    "internet, or whether it is a genuinely non-standard construction. Judge the "
    "mechanism, not the surface syntax: renamed variables or reformatted code do "
    "not make a known algorithm original. A novel COMBINATION of known pieces can "
    "still be original — say so explicitly when that is what you see."
)

PROMPT_TEMPLATE = """Analyze this solver in three steps.

1. Describe the search MECHANISM in one or two sentences (what it actually does:
   construction heuristic, local search moves, metaheuristic, exact method, etc.)
   — not a line-by-line code summary.
2. Use web search to look up that mechanism and find the closest known technique.
3. Decide how original it is.

Return ONLY a JSON object inside a ```json fenced block, with exactly these keys:
{{
  "mechanism": "<one or two sentences>",
  "originality": <integer 0-10, 0 = textbook copy, 10 = no public precedent>,
  "exists_online": <true|false>,
  "nearest_known_technique": "<name of the closest published technique>",
  "source_url": "<a URL backing your verdict, or empty string>",
  "justification": "<2-4 sentences explaining the score>"
}}

SOLVER CODE:
```python
{code}
```
"""

# server-side tools: the API runs the search loop itself and may pause_turn
JUDGE_TOOLS = [
    {"type": "web_search_20260209", "name": "web_search"},
    {"type": "web_fetch_20260209", "name": "web_fetch"},
]

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_verdict(text: str) -> dict | None:
    """Pull the verdict object out of the judge's reply, tolerating prose and
    fences around it."""
    match = _JSON_BLOCK.search(text)
    candidate = match.group(1) if match else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            candidate = text[start:end + 1]
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def judge(client, code: str) -> tuple[dict, int, int]:
    """Run the web-search-backed judge over one solver.

    Returns (verdict, input_tokens, output_tokens). Structured outputs are
    intentionally not used: they 400 alongside the citations web search
    produces, so we parse a fenced JSON block from the reply instead.
    """
    messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(code=code)}]
    texts: list[str] = []
    in_tok = out_tok = 0
    for _ in range(6):  # allow server-tool continuation (pause_turn)
        msg = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=JUDGE_MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=SYSTEM,
            tools=JUDGE_TOOLS,
            messages=messages,
        )
        in_tok += msg.usage.input_tokens
        out_tok += msg.usage.output_tokens
        texts.append("".join(b.text for b in msg.content if b.type == "text"))
        if msg.stop_reason == "pause_turn":
            messages = messages + [{"role": "assistant", "content": msg.content}]
            continue
        break
    verdict = extract_verdict("\n".join(t for t in texts if t))
    if verdict is None:
        verdict = {"error": "judge returned no parseable verdict"}
    return verdict, in_tok, out_tok


# Keyword -> (technique, originality, exists_online) used only in mock mode, so
# the demo still surfaces a plausible originality panel without the real API.
_MOCK_RULES = [
    ("christofides", "Christofides algorithm", 2),
    ("lin", "Lin-Kernighan", 2),
    ("kernighan", "Lin-Kernighan", 2),
    ("or_opt", "Or-opt local search", 3),
    ("or-opt", "Or-opt local search", 3),
    ("anneal", "Simulated annealing", 3),
    ("temperature", "Simulated annealing", 3),
    ("pheromone", "Ant colony optimization", 3),
    ("ant", "Ant colony optimization", 3),
    ("2-opt", "2-opt local search", 2),
    ("2opt", "2-opt local search", 2),
    ("greedy", "Greedy edge construction", 2),
    ("nearest", "Nearest-neighbour construction", 1),
]


def mock_verdict(code: str) -> dict:
    """Deterministic, API-free verdict for mock/demo runs."""
    low = (code or "").lower()
    hits = [(name, orig) for kw, name, orig in _MOCK_RULES if kw in low]
    if hits:
        name, orig = hits[0]
        # combining several known techniques nudges originality up a little
        if len({n for n, _ in hits}) >= 3:
            orig += 2
        return {
            "mechanism": f"Detected a {name.lower()} based search (mock analysis).",
            "originality": min(orig, 8),
            "exists_online": True,
            "nearest_known_technique": name,
            "source_url": "",
            "justification": ("Mock-mode heuristic verdict (no web search): the "
                              "solver matches well-known TSP techniques."),
            "mock": True,
        }
    return {
        "mechanism": "Unrecognised search mechanism (mock analysis).",
        "originality": 5,
        "exists_online": False,
        "nearest_known_technique": "unknown",
        "source_url": "",
        "justification": ("Mock-mode heuristic verdict (no web search): no "
                          "familiar TSP technique was detected in the code."),
        "mock": True,
    }


def quadrant(originality, improvement_pct, target_pct) -> str:
    """Place a result on the novelty x quality map. The interesting corner is
    'novel AND it wins' — that is where the lab creates original knowledge."""
    if originality is None or improvement_pct is None:
        return "?"
    wins = improvement_pct >= (target_pct or 0)
    novel = originality >= 6
    if novel and wins:
        return "original+wins"   # the knowledge we actually want
    if not novel and wins:
        return "rehash"          # works, but already known
    if novel and not wins:
        return "novel-weak"      # new idea, doesn't beat the bar yet
    return "noise"
