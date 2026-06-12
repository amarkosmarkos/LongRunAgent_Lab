# ⚗ Long Run Agent Lab

A research lab for **long-running autonomous agent experiments on verifiable problems**.

Agents define a scope, propose hypotheses, branch into experiment paths, test ideas
against an objective baseline, critique failures, share discoveries, collapse weak
branches, merge complementary ones — and the entire run is observable, replayable,
cost-tracked, and objectively verified.

First problem: **Euclidean TSP** (baseline: nearest-neighbor). The engine is
problem-agnostic — see `backend/app/problems/base.py` to add another benchmark.

## Quick start

### 1. Backend (Python 3.10+)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on mac/linux)
pip install -r requirements.txt
copy .env.example .env          # optional: add your ANTHROPIC_API_KEY
uvicorn app.main:app --port 8000
```

- **No API key?** The lab runs in **mock mode**: agent reasoning is scripted along the
  canonical demo arc, but all solver code is *really executed and really scored* —
  results stay objective. Perfect for a free 5-minute demo.
- **With `ANTHROPIC_API_KEY`** in `backend/.env`: the five agents (Planner, Strategist,
  Experimenter, Critic, Supervisor) run on real Claude models. Default budget: $2/run.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173, click **Start run**, and watch live. When it finishes,
press **▶ Replay** to scrub through the whole run from event 0.

## The 5-minute demo

1. **Start a run** — the Planner defines scope: objective, baseline score, target
   improvement, constraints, stop conditions (Scope tab).
2. **Hypotheses branch** — the Strategist proposes 4 distinct strategies; each becomes
   a lane in the branch graph.
3. **Experiments run** — green nodes improved, gray didn't, red failed. Click any node
   to see the approach, the engine-verified result, the critic's verdict, and the exact
   code that was executed.
4. **A weak branch collapses** (⊘) — with the supervisor's evidence-based reason.
5. **Insights accumulate** (Knowledge tab) and flow into every branch's prompts.
6. **Two branches merge** (purple edges) into a combined hypothesis.
7. **The merged branch wins** (★) — Results tab shows baseline vs best tour drawn on
   canvas, improvement %, target met, and an independent re-verification of the score.
8. **Costs** tab: spend per agent, per branch, against budget. **Replay** to relive it.

## How it works

```
run starts
  └─ Planner  ──► scope.defined (objective, baseline, success criteria, stop conditions)
  └─ Strategist ─► N hypotheses ──► N branches
  └─ per round, per active branch:
        Experimenter ─► solver code ─► sandboxed execution ─► engine validates + scores
        Critic ─► verdict + transferable insight ─► shared knowledge base
     Supervisor ─► collapse weak / merge complementary / continue
  └─ stop condition fires ─► winner verified ─► run.completed
```

- **Event-sourced**: every action is an event in `backend/data/runs/<id>/events.jsonl`.
  Live view and replay are the same pure reduction of that stream.
- **Objective evaluation**: agent code runs in a subprocess with a timeout; the engine
  (never the agent) validates the solution and computes the score. The winner is
  re-verified at the end.
- **Cost-aware**: every LLM call emits tokens + USD, attributed to agent and branch.
  The run stops gracefully when the budget is hit.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design and event schema.

## Configuration

Per run (UI or `POST /api/runs`): `n_cities`, `seed`, `num_hypotheses`, `max_rounds`,
`budget_usd`. Defaults in `backend/app/config.py`, models per agent role in
`backend/.env` (`MODEL_PLANNER`, `MODEL_EXPERIMENTER`, …). Pricing table in
`config.py` — keep it in sync with current Anthropic pricing.

## ⚠ Security note

Experimenter agents write Python that is executed on your machine (subprocess +
timeout — process isolation, **not** a security sandbox). Run it locally for research,
inspect generated code in the UI, and don't expose the backend publicly.

## TSPLIB benchmark mode

Select **TSPLIB benchmark** in the new-run form (problem `tsp_benchmark`) to run
against real TSPLIB95 instances with known optima (files in `backend/data/tsplib/`):

- **Score = mean gap %** above the known optimum across the dev instances
  (TSPLIB rounded-integer metric), so 0 means optimal on every instance.
- **Strong baseline**: nearest-neighbor + 2-opt to local optimum (~6.5% mean gap),
  so agents must invent something beyond plain 2-opt.
- **Held-out verification**: when the run ends, the winning solver code is
  re-executed on instances the agents never saw. The Results tab reports
  per-instance gaps, improved/worsened counts, and a generalizes / does-not-generalize
  verdict — improvements that only work on the dev set are exposed.

Dev and held-out sets are configurable per run; defaults in
`backend/app/problems/tsp.py` (`DEFAULT_DEV`, `DEFAULT_HOLDOUT`).

## Adding a new problem

Implement `Problem` (`generate_instance`, `baseline`, `validate`, `evaluate`,
`instance_stats`, `solver_contract`) in `backend/app/problems/`, register it in
`PROBLEMS`. The engine, agents, UI graph, replay, and cost tracking all come for free;
only the result visualization (e.g. `TourCanvas`) is TSP-specific.
