# Long Run Agent Lab — Architecture

A research lab for long-running autonomous agent experiments on verifiable problems.

## Core ideas

1. **Event-sourced runs.** Every meaningful action is an immutable event appended to
   `backend/data/runs/<run_id>/events.jsonl`. The UI (live view *and* replay) is a pure
   reduction of that event stream. Replayability is free by construction.
2. **Problem-agnostic engine.** The orchestrator only knows the `Problem` interface
   (`generate_instance`, `baseline`, `evaluate`, `validate`, `render_payload`).
   TSP is the first implementation; any benchmarkable problem can plug in.
3. **Branches as first-class objects.** A hypothesis becomes a branch. Branches are a DAG:
   merges have two parents. Branch states: `active → collapsed | merged | winner`
   (terminal), with `failed`/`stagnant` as observable conditions that drive supervisor decisions.
4. **Objective evaluation.** Agent-produced solver code is executed in a subprocess with a
   timeout; the *engine* (never the agent) validates the solution and computes the score
   against the baseline.
5. **Cost awareness.** Every LLM call emits `llm.called` with token counts and USD cost,
   attributed to an agent role and branch. The orchestrator checks the budget before each
   call and stops the run gracefully when exceeded.

## Run lifecycle

```
created ──► scoping ──► running ──► completed
                              ├──► budget_exceeded
                              ├──► stopped (user)
                              └──► failed (engine error)
```

### Phase 1 — Scope definition
The **Planner** agent receives the problem description, instance statistics and the
baseline score (computed deterministically by the engine), and emits a scope:
objective, baseline, success criteria (target improvement %), constraints
(time limit per experiment), and stop conditions (max rounds, budget, target reached).

### Phase 2 — Hypothesis branching
The **Strategist** proposes K distinct strategies. Each becomes a branch with its own
hypothesis and rationale.

### Phase 3 — Iteration rounds
Per round, for every active branch:

- **Experimenter** writes/improves solver code (`solve(cities) -> tour`), given the
  hypothesis, previous code, last result, critic feedback, and shared knowledge.
- Engine executes the code (subprocess, timeout), validates, scores. Emits `experiment.completed`.
- **Critic** analyzes the result; may emit an `insight.added` into the shared knowledge base.

After each round the **Supervisor** reviews all branches and decides:
collapse (repeated failure / stagnation), merge (combine complementary discoveries into a
new branch with two parents), spawn (new hypothesis from accumulated knowledge), or continue.

### Phase 4 — Conclusion
When a stop condition fires, the best branch is declared `winner`, the winning solution is
re-verified, and `run.completed` carries the final results (baseline vs best, improvement %,
solution, costs).

## Knowledge reuse

Insights are short, attributed statements (`branch`, `round`, `text`). They live in a shared
store and are injected into every Experimenter prompt and into Supervisor merge decisions.
A merged branch's hypothesis explicitly references the insights it combines.

## Event types

| Event | Payload (key fields) |
|---|---|
| `run.created` | config |
| `scope.defined` | scope object, planner reasoning |
| `hypotheses.proposed` | list of strategies + rationale |
| `branch.created` | branch_id, parent_ids, hypothesis, strategy |
| `experiment.started` | branch_id, round, approach |
| `experiment.completed` | branch_id, round, score, valid, improved, error, code, exec_time |
| `critique.added` | branch_id, verdict, analysis |
| `insight.added` | insight {id, branch_id, round, text} |
| `supervisor.decision` | decisions + reasoning |
| `branch.collapsed` | branch_id, reason |
| `branch.merged` | source_ids, new_branch_id, reason |
| `branch.winner` | branch_id, score |
| `llm.called` | agent, model, input_tokens, output_tokens, cost_usd, branch_id |
| `run.completed` / `run.stopped` / `run.failed` | results / reason |

Every event: `{seq, ts, type, agent, branch_id, payload}`. `seq` is the replay cursor.

## Backend layout

```
backend/app/
  config.py        settings, model pricing
  models.py        dataclasses for events/branches/scope
  store.py         RunStore: JSONL log + in-memory index + SSE polling
  llm.py           LLMClient (Anthropic API or deterministic mock), cost accounting
  sandbox.py       subprocess execution of agent solver code
  problems/        base.py (interface), tsp.py
  engine/          agents.py (prompts/parsing), mock_responses.py, orchestrator.py
  main.py          FastAPI: runs CRUD, event paging, SSE stream, stop
```

## Frontend layout

```
frontend/src/
  api.js           REST + SSE client
  replay.js        pure reducer: events[0..seq] -> view state
  agents.js        visual identity (color/initials) per agent role
  narrative.js     events -> human-readable story lines
  App.jsx          run list + new-run form
  RunView.jsx      header (phase/score/budget), replay controls, layout, live tour map
  components/
    BranchGraph.jsx  git-style lane graph (deterministic SVG layout, hover tooltips)
    TourCanvas.jsx   baseline vs best tour rendering (pinned under the graph, live)
    Panels.jsx       Story / Branches / Detail / Scope / Knowledge / Costs / Results / Events
```

The branch graph uses fixed lanes per branch (x) and event order (y) — a git-log style
layout that stays readable regardless of run size. Merges draw two in-edges; collapses
terminate a lane with a ⊘ node; the winner gets a ★ node.

## Benchmark problems (tsp_benchmark)

`TSPBenchmark` plugs into the same `Problem` interface but evaluates over a SET of
TSPLIB95 instances: `generate_instance` loads dev + held-out instances,
`execute` runs the agent's `solve()` separately per dev instance (one subprocess
each, per-instance timeout) and attaches a per-instance `detail` to
`experiment.completed`, and `evaluate` returns the mean gap %% vs the known optima.
`holdout_eval` (called by the orchestrator at `run.completed` with the winner's
code) re-runs the solver on the held-out instances against the same
nearest-neighbor + 2-opt baseline and reports improved/worsened counts plus a
`generalizes` verdict. Sandbox temp dirs live in `backend/data/tmp` (project-local).

## Mock mode

`LLM_MOCK=1` (or leaving `ANTHROPIC_API_KEY` unset) replaces the LLM with a deterministic
script that follows the canonical demo arc — 4 hypotheses, one failing branch collapsed,
insights discovered, two branches merged, the merged branch wins — while **all experiment
execution, validation, and scoring remain real**. Useful for demos, development, and CI.
