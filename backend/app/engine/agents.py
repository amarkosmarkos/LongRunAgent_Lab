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
    # fallback: a plain ``` fence whose body is clearly the submission (the model
    # forgot the `python` language tag) — better than rejecting valid code
    for body in GENERIC_FENCE_RE.findall(text):
        if "def custom_kernel" in body:
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
    "You are the Research agent of an autonomous GPU-performance lab. You search "
    "the web for the current state of the art in GPU kernel optimisation for the "
    "given operation and hardware, and summarize concrete, implementable "
    "techniques (tiling and blocking schemes, shared-memory and register usage, "
    "vectorised/coalesced loads, warp-level primitives, tensor cores, kernel "
    "fusion, launch configuration, Triton and CUDA specifics). Cite what you "
    "find. Be concise and practical — this feeds the Planner and Strategist. "
    "Plain text, no code."
)


def researcher_prompt(problem_desc: str, stats: str) -> str:
    return f"""Research how to make this GPU kernel fast.

PROBLEM: {problem_desc}
TARGET: {stats}

Search the web and report, in a few short bullet points:
- The techniques that actually move the needle for this operation, dtype and
  shape range on this class of GPU, roughly in order of expected payoff.
- What a well-optimised implementation typically achieves relative to a naive
  torch eager baseline, and where the hardware roofline sits.
- Concrete implementation tips and the common pitfalls that cause wrong results
  or blown compile budgets.
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
      "name": "<short name>", "strategy": "<optimisation technique, e.g. triton-tiled / tensor-cores / fused-epilogue>",
      "hypothesis": "<the specific variation to try, grounded in that branch's results>",
      "risk": "<main reason this could fail>"}}
  ],
  "new_hypotheses": [
    {{"name": "<short name>", "strategy": "<optimisation technique, e.g. triton-tiled / tensor-cores / fused-epilogue>",
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
    parts.append("""Each strategy must be a distinct OPTIMISATION TECHNIQUE implementable in one
self-contained submission.py (torch / Triton / inline CUDA) within the harness
timeout. Well-known techniques are welcome — the goal is measured speedup, not
novelty. Prefer strategies that attack different bottlenecks from one another.
Return JSON:
{
  "hypotheses": [
    {"name": "<short name>", "strategy": "<optimisation technique, e.g. triton-tiled / tensor-cores / fused-epilogue>",
      "hypothesis": "<falsifiable claim: doing X will improve over baseline because Y>",
      "risk": "<main reason this could fail>"}
  ]
}""")
    return "\n\n".join(parts)


# ----------------------------------------------------------- experimenter
EXPERIMENTER_SYSTEM = (
    "You are an Experimenter in an autonomous GPU-performance lab. You write "
    "correct, fast GPU kernel code to test a hypothesis.\n"
    "Your reply MUST have exactly two parts, in this order and nothing else:\n"
    "  1. a single-line JSON object: "
    '{\"approach\": \"...\", \"expectation\": \"...\"}\n'
    "  2. exactly one fenced code block opened with a line containing only ```python "
    "and closed with a line containing only ``` — inside it, the COMPLETE contents "
    "of `submission.py`, defining `def custom_kernel(data): ...`.\n"
    "Do not add prose before, between, or after these two parts. Do not use any "
    "other code fence. If you omit the ```python block the experiment is a total "
    "failure, so never describe code in words — always emit the runnable block.\n"
    "The code MUST be complete and self-contained: every import, helper, kernel "
    "and constant you use must be defined in that one file. No '...' placeholders, "
    "no TODOs, no references to earlier messages.\n"
    "CORRECTNESS IS A HARD GATE: the harness compares your output against the "
    "reference on every test shape before it times anything. A kernel that is "
    "fast but wrong scores nothing at all, so never trade accuracy for speed.\n"
    "Compilation and autotuning happen at benchmark time and count against the "
    "clock, so keep JIT work bounded: a huge Triton autotune grid or a slow "
    "cpp_extension build can blow the harness timeout and lose the experiment. "
    "Standard optimisations are exactly what is wanted here — tiling, shared "
    "memory, coalesced and vectorised loads, warp-level primitives, kernel "
    "fusion, better launch configurations, tensor cores, shape specialisation. "
    "Reusing a well-known technique is a success, not a weakness."
)


def experimenter_prompt(contract: str, stats: str, branch: dict, round_: int,
                        last_result: dict | None, critique: str | None,
                        insights: list[dict], time_limit_s: int,
                        retry_feedback: str | None = None,
                        operator_instruction: str | None = None,
                        base: dict | None = None,
                        inspirations: list[dict] | None = None,
                        crossover: dict | None = None) -> str:
    parts = [f"""You work on branch "{branch['name']}".
HYPOTHESIS: {branch['hypothesis']}
STRATEGY: {branch['strategy']}
ROUND: {round_}. TIME LIMIT: your solve() must finish well under {time_limit_s}s.

{contract}
INSTANCE: {stats}"""]
    if operator_instruction:
        parts.append(operator_instruction)
    if crossover:
        parts.append(
            "CROSSOVER TASK — this branch is a MERGE of two lineages. Combine "
            "them into one kernel that beats both.\n"
            f"WHY THEY WERE MERGED: {crossover.get('reflection')}\n"
            f"PARENT A \"{crossover.get('a_name')}\" (score {crossover.get('a_score')}):\n"
            f"```python\n{crossover.get('a_code')}\n```\n"
            f"PARENT B \"{crossover.get('b_name')}\" (score {crossover.get('b_score')}):\n"
            f"```python\n{crossover.get('b_code')}\n```")
    elif base and base.get("code"):
        parts.append(f"BASE PROGRAM to mutate — {base.get('label', 'your current best')} "
                     f"(score {base.get('score')}):\n"
                     f"```python\n{base['code']}\n```")
    if inspirations:
        blocks = []
        for insp in inspirations:
            blocks.append(
                f"- \"{insp.get('name') or insp.get('id')}\" "
                f"[niche: {insp.get('niche') or '+'.join(insp.get('tags') or []) or '?'}] "
                f"score {insp.get('score')}:\n```python\n{insp['code']}\n```")
        parts.append(
            "INSPIRATION PROGRAMS from OTHER niches of the population (do not "
            "copy any of them — steal their best distinct idea if it helps, or "
            "deliberately avoid their whole family of approaches):\n"
            + "\n".join(blocks))
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
the COMPLETE submission.py (define everything, no placeholders).
THE TWO WAYS EXPERIMENTS ARE LOST — avoid both:
- WRONG OUTPUT. The harness checks your result against the reference on every test
  shape before timing anything, and any mismatch beyond tolerance scores nothing.
  Watch dtypes, accumulation precision, and non-square / non-power-of-two shapes.
- TIMEOUT. Compilation and autotuning run inside the measured window. Keep any
  Triton autotune space small and any cpp_extension build simple, and cache
  compiled artifacts at module scope so the cost is paid once, not per call.
Handle every benchmark shape, not just the largest: specialising is allowed, but a
shape you forget to handle is a correctness failure.
{"approach": "<one sentence: what you changed and why>", "expectation": "<expected effect>"}
```python
import torch
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    ...            # your optimised implementation; must match the reference exactly
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
