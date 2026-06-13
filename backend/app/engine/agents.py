"""Agent roles: prompt construction + response parsing.

Each agent returns structured data. All prompts demand a single JSON object
(plus a python code fence for the experimenter), parsed defensively.
"""
from __future__ import annotations

import json
import re

CODE_RE = re.compile(r"```[ \t]*[Pp]ython[ \t]*\n?(.*?)```", re.DOTALL)
GENERIC_FENCE_RE = re.compile(r"```[ \t]*\n(.*?)```", re.DOTALL)


def parse_json(text: str) -> dict:
    """Extract the first balanced JSON object (greedy regex would swallow
    code-fence braces that follow the JSON in experimenter responses)."""
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in response")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON object in response")


def parse_code(text: str) -> str | None:
    m = CODE_RE.search(text)
    if m:
        return m.group(1).strip()
    # fallback: a plain ``` fence whose body is clearly the solver (the model
    # forgot the `python` language tag) — better than rejecting valid code
    for body in GENERIC_FENCE_RE.findall(text):
        if "def solve" in body:
            return body.strip()
    return None


# ---------------------------------------------------------------- planner
PLANNER_SYSTEM = (
    "You are the Planner of an autonomous research lab. You define the scope of an "
    "optimization run: objective, baseline, success criteria, constraints, stop "
    "conditions. Be precise and realistic. Respond with a single JSON object only."
)


def planner_prompt(problem_desc: str, stats: str, baseline_alg: str,
                   baseline_score: float, config: dict) -> str:
    return f"""Define the scope for this optimization run.

PROBLEM: {problem_desc}
INSTANCE: {stats}
BASELINE: {baseline_alg} scored {baseline_score} (lower is better).
HARD LIMITS set by the operator: max {config['max_rounds']} rounds, budget ${config['budget_usd']} USD, {config['experiment_timeout_s']}s per experiment, suggested target improvement {config['target_improvement_pct']}%.

Return JSON:
{{
  "objective": "...one sentence...",
  "success_criteria": {{"target_improvement_pct": <number>, "rationale": "..."}},
  "constraints": ["..."],
  "stop_conditions": ["..."],
  "reasoning": "...why these targets are appropriate for this instance size..."
}}"""


# ------------------------------------------------------------- strategist
STRATEGIST_SYSTEM = (
    "You are the Strategist of an autonomous research lab. You propose diverse, "
    "testable algorithmic hypotheses. Each must be meaningfully different from the "
    "others. Respond with a single JSON object only."
)


def strategist_prompt(problem_desc: str, stats: str, scope: dict, k: int) -> str:
    return f"""Propose {k} distinct strategies to beat the baseline.

PROBLEM: {problem_desc}
INSTANCE: {stats}
SCOPE: {json.dumps(scope)}

Each strategy must be implementable in pure Python within the time limit.
Return JSON:
{{
  "hypotheses": [
    {{"name": "<short name>", "strategy": "<algorithm family>",
      "hypothesis": "<falsifiable claim: doing X will improve over baseline because Y>",
      "risk": "<main reason this could fail>"}}
  ]
}}"""


# ----------------------------------------------------------- experimenter
EXPERIMENTER_SYSTEM = (
    "You are an Experimenter in an autonomous research lab. You write correct, fast, "
    "pure-Python solver code to test a hypothesis.\n"
    "Your reply MUST have exactly two parts, in this order and nothing else:\n"
    "  1. a single-line JSON object: "
    '{\"approach\": \"...\", \"expectation\": \"...\"}\n'
    "  2. exactly one fenced code block opened with a line containing only ```python "
    "and closed with a line containing only ``` — inside it, the COMPLETE solver "
    "defining `def solve(cities): ...` and returning a tour (a list of int).\n"
    "Do not add prose before, between, or after these two parts. Do not use any "
    "other code fence. If you omit the ```python block the experiment is a total "
    "failure, so never describe code in words — always emit the runnable block.\n"
    "The code MUST be complete and self-contained: define every function and name "
    "you use, no '...' placeholders, no TODOs, no references to earlier messages. "
    "It MUST be time-bounded: capture t0 = time.time() at the start of solve() and "
    "stop improving once the time budget is reached, returning the best tour found "
    "so far — a solver that can exceed the time limit is a failed solver. "
    "Keep it compact enough to fit well within the output limit (trim comments)."
)


def experimenter_prompt(contract: str, stats: str, branch: dict, round_: int,
                        last_result: dict | None, critique: str | None,
                        insights: list[dict], time_limit_s: int,
                        retry_feedback: str | None = None) -> str:
    parts = [f"""You work on branch "{branch['name']}".
HYPOTHESIS: {branch['hypothesis']}
STRATEGY: {branch['strategy']}
ROUND: {round_}. TIME LIMIT: your solve() must finish well under {time_limit_s}s.

{contract}
INSTANCE: {stats}"""]
    if branch.get("best_code"):
        parts.append(f"YOUR CURRENT BEST CODE (score {branch.get('best_score')}):\n"
                     f"```python\n{branch['best_code']}\n```")
    if last_result:
        parts.append(f"LAST EXPERIMENT RESULT: {json.dumps(last_result)}")
    if critique:
        parts.append(f"CRITIC FEEDBACK: {critique}")
    if insights:
        lines = "\n".join(f"- [{i['branch_id']}] {i['text']}" for i in insights)
        parts.append(f"SHARED LAB KNOWLEDGE (from all branches):\n{lines}")
    if retry_feedback:
        # the previous attempt this same round failed before producing a score;
        # tell the model exactly what broke so it can fix it immediately
        parts.append("YOUR PREVIOUS ATTEMPT THIS ROUND FAILED — fix it now.\n"
                     f"{retry_feedback}")
    budget = max(1, time_limit_s - 1)
    parts.append(f"""Reply with exactly these two parts and nothing else. The code must be
COMPLETE (define everything, no placeholders) and TIME-BOUNDED (stop before {budget}s):
{{"approach": "<one sentence: what you changed and why>", "expectation": "<expected effect>"}}
```python
import time

def solve(cities):
    t0 = time.time()
    budget = {budget}            # seconds — must return before the {time_limit_s}s hard limit
    n = len(cities)
    best = list(range(n))        # replace with a real construction + improvement loop
    while time.time() - t0 < budget:
        ...                      # improve `best`; break/return when time runs out
    return best                  # a permutation of range(n)
```""")
    return "\n\n".join(parts)


# ----------------------------------------------------------------- critic
CRITIC_SYSTEM = (
    "You are the Critic of an autonomous research lab. You analyze experiment results "
    "honestly, diagnose failures, and extract transferable insights other branches can "
    "reuse. Respond with a single JSON object only."
)


def critic_prompt(branch: dict, round_: int, result: dict,
                  baseline_score: float, best_overall: float | None) -> str:
    return f"""Analyze this experiment.

BRANCH: {branch['name']} — {branch['hypothesis']}
ROUND: {round_}
RESULT: {json.dumps(result)}
BASELINE: {baseline_score}. BEST SCORE ACROSS ALL BRANCHES: {best_overall}.

Return JSON:
{{
  "verdict": "improved" | "no_improvement" | "failed",
  "analysis": "<2-3 sentences: why did this happen>",
  "insight": "<one transferable, concrete insight other branches could reuse, or null>",
  "suggestion": "<concrete next step for this branch>"
}}"""


# ------------------------------------------------------------- supervisor
SUPERVISOR_SYSTEM = (
    "You are the Supervisor of an autonomous research lab. After each round you "
    "decide which branches to collapse (clearly weak), which pairs to merge "
    "(complementary discoveries), and whether to continue. You are decisive but "
    "evidence-driven. Respond with a single JSON object only."
)


def supervisor_prompt(round_: int, max_rounds: int, branches: list[dict],
                      insights: list[dict], baseline_score: float,
                      budget_spent: float, budget_usd: float,
                      stagnation_rounds: int) -> str:
    return f"""End of round {round_}/{max_rounds}. Review the lab.

BASELINE: {baseline_score} (lower is better)
BUDGET: ${budget_spent:.4f} spent of ${budget_usd}
BRANCHES: {json.dumps(branches)}
("rounds_without_improvement" >= {stagnation_rounds} means stagnant)
SHARED INSIGHTS: {json.dumps(insights)}

Rules:
- Collapse a branch only with clear evidence: repeated failures or stagnation while others improve.
- Merge at most one pair per round, only when their discoveries are complementary; describe the combined hypothesis.
- Keep at least one branch active unless the run should end.

Return JSON:
{{
  "collapse": [{{"branch_id": "...", "reason": "..."}}],
  "merge": {{"source_ids": ["...", "..."], "name": "<short name>",
             "hypothesis": "<combined hypothesis>", "strategy": "<combined strategy>",
             "reason": "..."}} | null,
  "reasoning": "<overall assessment of the round>"
}}"""
