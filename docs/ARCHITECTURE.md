# Long Run Agent Lab — Architecture

A research lab for long-running autonomous agent experiments on verifiable problems.

## Core ideas

1. **Event-sourced runs.** Every meaningful action is an immutable event appended to
   `backend/data/runs/<run_id>/events.jsonl`. The UI (live view *and* replay) is a pure
   reduction of that event stream. Replayability is free by construction.
2. **Problem-agnostic engine.** The orchestrator only knows the `Problem` interface
   (`generate_instance`, `baseline`, `evaluate`, `validate`, `execute`, and optionally
   `holdout_eval` / `behavior_descriptor`). GPU kernel optimization against the GPU MODE
   reference-kernels benchmark is the current implementation; any problem with a
   verifiable score and a baseline can plug in. (The lab previously ran on TSP; the
   migration touched only the `Problem` implementation and the execution backend.)
3. **Branches as first-class objects.** A hypothesis becomes a branch. Branches are a DAG:
   merges have two parents. Branch states: `active → collapsed | merged | winner`
   (terminal), with `failed`/`stagnant` as observable conditions that drive supervisor decisions.
4. **Objective evaluation.** Agent-produced `submission.py` is executed by the
   *official upstream harness* (`problems/pmpp/eval.py`, vendored unmodified), which
   checks the output against the reference on every test shape before timing anything.
   The agent never reports its own result: correctness is a hard gate, and the score
   is computed by the engine from the harness's per-shape measurements.
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

- **Experimenter** writes/improves a kernel (`submission.py` defining `custom_kernel`), given the
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
  kernels/         GPU MODE integration
    spec.py        load a vendored reference-kernel problem (task.yml shapes, sources)
    runner.py      run the official harness: "local" subprocess or "modal" GPU backend
    modal_app.py   the Modal app that runs the harness on a real GPU
  problems/        base.py (interface), kernel.py (KernelBenchmark)
  novelty.py       AST-fingerprint duplicate gate (disabled by default for kernels)
  population.py    DGM parent selection + MAP-Elites over behavior descriptors
  bandit.py        UCB1 over mutation operators
  gitrepo.py       the run's real git DAG
  scripts/
    popcorn_submit.py  offline validation against the official leaderboard
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
  RunView.jsx      header (phase/timing/budget), replay controls, layout
  components/
    BranchGraph.jsx  git-style lane graph (deterministic SVG layout, hover tooltips)
    Panels.jsx       Story / Branches / Detail / Scope / Knowledge / Evolution /
                     Costs / Results / Events
```

The branch graph uses fixed lanes per branch (x) and event order (y) — a git-log style
layout that stays readable regardless of run size. Merges draw two in-edges; collapses
terminate a lane with a ⊘ node; the winner gets a ★ node.

## The kernel benchmark (gpu_kernel)

`KernelBenchmark` plugs into the same `Problem` interface and evaluates a submission
against one vendored GPU MODE reference-kernel (`backend/data/kernels/<problem>/`,
upstream files unchanged):

- `generate_instance` reads `task.yml` and splits the benchmark shapes into **dev**
  (what the run optimises against) and **held-out** (final verification only).
- `execute` runs the official harness twice: `test` mode first (correctness on every
  test shape — cheap, fails fast) and only then `benchmark` mode for the timings. It
  attaches per-shape results as `detail` on `experiment.completed`.
- `validate` gates on the harness's `check: pass` plus a timing for every dev shape.
- `evaluate` returns the **geometric mean of the per-shape mean runtimes in
  nanoseconds** — lower is better, so the engine's existing improvement maths reads as
  the aggregate speedup over the reference kernel.
- `holdout_eval` re-times both the reference and the winner on the held-out shapes and
  reports per-shape speedups plus a `generalizes` verdict — this is what exposes a
  kernel tuned only to the shapes it saw.
- `behavior_descriptor` reports `small_speedup` / `large_speedup` / `scaling`, so
  MAP-Elites niches capture *how* a kernel wins (small shapes vs large ones) rather
  than what its source text says.

### Execution backends

`app/kernels/runner.py` assembles an identical working directory for both backends and
parses the identical harness output, so the two are interchangeable:

| backend | correctness | timings | needs |
|---|---|---|---|
| `local` | real | real, but **CPU timings** unless this box has CUDA | nothing |
| `modal` | real | real GPU | a Modal account + `modal deploy app/kernels/modal_app.py` |

The harness writes `key: value` lines to the fd named by `POPCORN_FD`; the runner points
that at stdout, because passing a private fd is POSIX-only and would break on Windows.
On a CUDA-less box the local backend rewrites `device='cuda'` in `reference.py` and
no-ops `torch.cuda.synchronize` via an injected `sitecustomize.py` — the only source
rewriting that ever happens, and it is tagged in the result so CPU numbers are never
mistaken for GPU numbers.

`popcorn-cli` is deliberately **not** in the loop (queue latency and rate limits); use
`python -m app.scripts.popcorn_submit <run_id>` to validate a finished winner against
the official leaderboard.

## Mock mode

`LLM_MOCK=1` (or leaving `ANTHROPIC_API_KEY` unset) replaces the LLM with a deterministic
script that follows the canonical demo arc — 4 hypotheses, the low-precision branch
rejected by the correctness gate and collapsed, insights discovered, two branches merged
into a shape-specialised kernel that wins — while **all kernel execution, correctness
checking, and timing remain real**. The mock cannot invent a speedup the harness did not
measure. Useful for demos, development, and CI.
