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
    "optimization run (objective, success criteria, constraints, stop conditions) "
    "AND you decide how many parallel hypotheses to explore — that number is YOUR "
    "call, scaled to the difficulty of the problem, not a fixed value. Be precise "
    "and realistic. Respond with a single JSON object only."
)


def planner_prompt(problem_desc: str, stats: str, baseline_alg: str,
                   baseline_score: float, config: dict,
                   research: str | None = None,
                   memory: str | None = None) -> str:
    parts = [f"""Define the scope for this optimization run.

PROBLEM: {problem_desc}
INSTANCE: {stats}
BASELINE: {baseline_alg} scored {baseline_score} (lower is better).
HARD LIMITS set by the operator: max {config['max_rounds']} rounds, budget ${config['budget_usd']} USD, {config['experiment_timeout_s']}s per experiment, at most {config.get('max_branches', 12)} concurrent hypotheses."""]
    if research:
        parts.append(f"WEB RESEARCH (state-of-the-art approaches found online):\n{research}")
    if memory:
        parts.append(memory)
    parts.append("""Return JSON:
{
  "objective": "...one sentence...",
  "success_criteria": {"target_improvement_pct": <number>, "rationale": "..."},
  "initial_hypotheses": <integer: how many DISTINCT strategies to explore in parallel this run — your decision, scaled to difficulty, within the concurrent cap>,
  "constraints": ["..."],
  "stop_conditions": ["..."],
  "reasoning": "...why these targets AND this number of hypotheses fit this instance..."
}""")
    return "\n\n".join(parts)


# --------------------------------------------------------------- researcher
RESEARCHER_SYSTEM = (
    "You are the Research agent of an autonomous optimization lab. You search the "
    "web for the current state-of-the-art and best practical approaches for the "
    "given problem, and summarize concrete, implementable techniques (algorithm "
    "families, known strong heuristics, typical optimality gaps, pitfalls). Cite "
    "what you find. Be concise and practical — this feeds the Planner and "
    "Strategist. Plain text, no code."
)


def researcher_prompt(problem_desc: str, stats: str) -> str:
    return f"""Research the best-known practical approaches for this problem.

PROBLEM: {problem_desc}
INSTANCE: {stats}

Search the web and report, in a few short bullet points:
- The strongest practical algorithm families for this size/type of instance.
- Typical optimality gaps each achieves within a few seconds.
- Concrete implementation tips and common pitfalls.
Keep it actionable — the lab will turn this into testable hypotheses."""


# --------------------------------------------------------- planner review
PLANNER_REVIEW_SYSTEM = (
    "You are the Planner of an autonomous research lab, reviewing progress at the "
    "end of a round. You look at every branch's results and the shared insights, "
    "then steer what happens next. The lab improves in several ways and you control "
    "two of them:\n"
    " - Every active branch is ALREADY refined in place each round by its "
    "Experimenter (it iterates on its own best code). You don't need to ask for "
    "that — it happens automatically.\n"
    " - You may EVOLVE an existing branch: fork it (keeping its current best code) "
    "and push it in a specific new direction — the right move when a path is "
    "promising and you want a variation without losing its progress.\n"
    " - You may open brand-NEW hypotheses from scratch — for genuinely different "
    "directions the evidence now suggests.\n"
    "(The Supervisor separately collapses weak branches and merges complementary "
    "pairs.) You are not limited to a fixed number of branches. Respond with a "
    "single JSON object only."
)


def planner_review_prompt(round_: int, max_rounds: int, branches: list[dict],
                          insights: list[dict], baseline_score: float,
                          best_score: float | None, target_pct: float,
                          budget_spent: float, budget_usd: float,
                          active_count: int, max_branches: int) -> str:
    return f"""End of round {round_}/{max_rounds}. Review the lab and plan next steps.

BASELINE: {baseline_score} (lower is better). BEST SO FAR: {best_score}. TARGET: {target_pct}% improvement.
BUDGET: ${budget_spent:.4f} of ${budget_usd}. ACTIVE BRANCHES: {active_count} (hard cap {max_branches}).
BRANCHES (with results, including their id and best_score): {json.dumps(branches)}
SHARED INSIGHTS: {json.dumps(insights)}

Prefer EVOLVING a promising existing branch over starting from scratch when the
evidence says a path is working and just needs a variation. Open new-from-scratch
hypotheses only for genuinely different directions. Add nothing if nothing is
justified. Never exceed the active-branch cap.
Return JSON:
{{
  "evolve": [
    {{"parent_id": "<id of an existing branch to fork (keeps its code)>",
      "name": "<short name>", "strategy": "<algorithm family>",
      "hypothesis": "<the specific variation to try, grounded in that branch's results>",
      "risk": "<main reason this could fail>"}}
  ],
  "new_hypotheses": [
    {{"name": "<short name>", "strategy": "<algorithm family>",
      "hypothesis": "<falsifiable claim grounded in the evidence above>",
      "risk": "<main reason this could fail>"}}
  ],
  "continue": <true|false: should the run keep going?>,
  "reasoning": "<what the round showed and why these directions>"
}}"""


# ------------------------------------------------------------- strategist
STRATEGIST_SYSTEM = (
    "You are the Strategist of an autonomous research lab. You propose diverse, "
    "testable algorithmic hypotheses. Each must be meaningfully different from the "
    "others. Respond with a single JSON object only."
)


def strategist_prompt(problem_desc: str, stats: str, scope: dict, k: int,
                      research: str | None = None,
                      memory: str | None = None) -> str:
    parts = [f"""Propose {k} distinct strategies to beat the baseline.

PROBLEM: {problem_desc}
INSTANCE: {stats}
SCOPE: {json.dumps(scope)}"""]
    if research:
        parts.append(f"WEB RESEARCH (use these state-of-the-art ideas):\n{research}")
    if memory:
        parts.append(memory)
    parts.append("""Each strategy must be implementable in pure Python within the time limit.
Return JSON:
{
  "hypotheses": [
    {"name": "<short name>", "strategy": "<algorithm family>",
      "hypothesis": "<falsifiable claim: doing X will improve over baseline because Y>",
      "risk": "<main reason this could fail>"}
  ]
}""")
    return "\n\n".join(parts)


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
    "It MUST be time-bounded with MARGIN: derive the budget from the instance size "
    "(budget = 0.04 * len(cities) seconds), capture t0 = time.time(), and check the "
    "clock INSIDE long loops — not only between sweeps — returning the best tour so "
    "far before the budget. Timeouts are the most common failure; a solver that can "
    "exceed the time limit is a failed solver. "
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
    parts.append("""Reply with exactly these two parts and nothing else. The code must be
COMPLETE (define everything, no placeholders) and strictly TIME-BOUNDED.
TIMEOUTS ARE THE #1 FAILURE — avoid them with margin:
- Derive your budget from the instance size: `budget = 0.04 * len(cities)` seconds.
  You are given MORE wall time than that, so staying under it guarantees no timeout.
- Check `time.time() - t0` INSIDE every long loop (e.g. inside a 2-opt/Or-opt sweep),
  not only between sweeps — a single full sweep on a few hundred cities can take >1s.
  Never START a sweep or restart you cannot finish before the budget.
- For large n, use k-nearest candidate lists; do NOT build a full n*n matrix or use
  O(n^3) moves.
{"approach": "<one sentence: what you changed and why>", "expectation": "<expected effect>"}
```python
import time

def solve(cities):
    t0 = time.time()
    n = len(cities)
    budget = 0.04 * n            # seconds; you get more wall than this — stay well under it
    best = list(range(n))        # replace with NN/greedy construction + local search
    while time.time() - t0 < budget:
        ...                      # one improvement step; check time.time()-t0 inside long sweeps too
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
