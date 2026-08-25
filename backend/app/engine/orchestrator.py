"""The run engine: scope -> hypotheses -> branching iteration -> conclusion.

Problem-agnostic: everything domain-specific lives behind the Problem interface.
Every decision, experiment, insight and dollar is emitted as an event.
"""
from __future__ import annotations

import random
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from .. import knowledge
from ..bandit import OPERATORS, OperatorBandit
from ..config import (AGENT_MODELS, DATA_DIR, EXPERIMENTER_MODELS,
                      ROLE_BUDGET_SHARE)
from ..gitrepo import RunRepo
from ..llm import LLMClient
from ..models import Branch, Insight
from ..novelty import NoveltyIndex
from ..population import Population, Program, niche_key
from ..problems import PROBLEMS
from ..store import Run
from . import agents


class StopRun(Exception):
    def __init__(self, reason: str):
        self.reason = reason


class Orchestrator:
    def __init__(self, run: Run):
        self.run = run
        self.cfg = run.config
        self.problem = PROBLEMS[self.cfg["problem"]]
        self.llm = LLMClient()
        self.instance = self.problem.generate_instance(self.cfg.get("problem_params", {}))
        self.scope: dict = {}
        self.branches: dict[str, Branch] = {}
        self.insights: list[Insight] = []
        self.total_cost = 0.0
        self.cost_by_agent: dict[str, float] = {}
        self.baseline_solution: list = []
        self.baseline_score: float = 0.0
        self.research: str | None = None
        self.memory: str | None = None  # prompt digest recalled from the archive
        self.round = 0
        self._lock = threading.Lock()  # guards cost + insights across parallel branches
        # ---- evolution substrate: git DAG + population + novelty + bandit ----
        self.rng = random.Random(run.id)  # deterministic per run
        self.repo: RunRepo | None = None
        self.population: Population | None = None
        self.novelty = NoveltyIndex()
        self.bandit: OperatorBandit | None = None

    # ------------------------------------------------------------- helpers
    def _check_interrupts(self):
        if self.run.stop_requested:
            raise StopRun("stopped by user")
        if self.total_cost >= self.cfg["budget_usd"]:
            raise StopRun("budget exceeded")

    # short human label per role, shown while the (slow) call is in flight
    _THINKING = {
        "planner": "reading the problem and setting the objective",
        "strategist": "designing hypotheses to explore",
        "experimenter": "writing solver code",
        "critic": "analysing the result",
        "supervisor": "reviewing branches and deciding what to keep",
        "researcher": "searching the web for state-of-the-art approaches",
    }

    def _call(self, role: str, system: str, prompt: str,
              context: dict | None = None, branch_id: str | None = None,
              action: str | None = None, model: str | None = None):
        self._check_interrupts()
        # announce intent BEFORE the API call so the UI can show the agent
        # "thinking" live, instead of nodes only appearing once work is done
        self.run.emit("agent.thinking", agent=role, branch_id=branch_id, payload={
            "action": action or self._THINKING.get(role, "thinking"),
            "round": self.round or None,
        })
        budget_left = self._budget_for(role)
        res = self.llm.call(role, system, prompt, context, model=model,
                            budget_left_usd=budget_left)
        with self._lock:  # branches run in parallel — cost mutation must be atomic
            self.total_cost += res.cost_usd
            self.cost_by_agent[role] = self.cost_by_agent.get(role, 0.0) + res.cost_usd
            if branch_id and branch_id in self.branches:
                self.branches[branch_id].cost_usd += res.cost_usd
        if getattr(res, "over_budget", False):
            # the call's own tool loop was cut short to protect the run
            self.run.emit("budget.capped", agent=role, branch_id=branch_id, payload={
                "cost_usd": round(res.cost_usd, 6),
                "budget_left_before_call": round(budget_left, 6),
                "total_cost_usd": round(self.total_cost, 6),
                "note": "the agent's web-search loop was stopped early because "
                        "this single call had already spent the run's remaining "
                        "budget; its answer may be incomplete",
            })
        self.run.emit("llm.called", agent=role, branch_id=branch_id, payload={
            "model": res.model, "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens, "cost_usd": round(res.cost_usd, 6),
            "total_cost_usd": round(self.total_cost, 6),
            # full context for decision-making transparency: exactly what the
            # agent saw (accumulated insights, critiques, previous code) and
            # exactly what it answered before any parsing
            "system_prompt": system,
            "user_prompt": prompt,
            "raw_response": res.text,
        })
        return res

    def _budget_for(self, role: str) -> float:
        """What this call may spend: the run's remaining budget, further
        capped by the role's own share of it.

        Preliminary phases are nice-to-have and must not be able to starve the
        work the run exists to do. Without a share, a single researcher turn
        could take the whole budget before one hypothesis had been proposed.
        """
        with self._lock:
            remaining = self.cfg["budget_usd"] - self.total_cost
            share = ROLE_BUDGET_SHARE.get(role)
            if share is None:
                return remaining
            allowance = share * self.cfg["budget_usd"] - self.cost_by_agent.get(role, 0.0)
        return max(0.0, min(remaining, allowance))

    def _active(self) -> list[Branch]:
        return [b for b in self.branches.values() if b.status == "active"]

    def _relevant_insights(self, query: str, k: int = 8) -> list[dict]:
        """Top-k insights relevant to a branch's hypothesis instead of dumping
        the whole pool into the prompt — keeps context focused as insights grow."""
        with self._lock:
            pool = list(self.insights)
        if len(pool) <= k:
            return [i.public() for i in pool]
        idx = knowledge.rank(query, [i.text for i in pool], k)
        picked = idx or range(k)
        return [pool[i].public() for i in picked]

    def _best_overall(self) -> tuple[Branch | None, float | None]:
        best_b, best_s = None, None
        for b in self.branches.values():
            if b.best_score is not None and (best_s is None or b.best_score < best_s):
                best_b, best_s = b, b.best_score
        return best_b, best_s

    def _improvement_pct(self, score: float | None) -> float | None:
        if score is None:
            return None
        return round((self.baseline_score - score) / self.baseline_score * 100, 3)

    def _run_attempt(self, code: str | None) -> tuple[dict, dict | None]:
        """Execute one solver attempt -> (result, per-instance detail)."""
        if not code:
            return ({"score": None, "valid": False,
                     "error": "no python code produced (missing ```python fence)",
                     "exec_time": 0.0}, None)
        out = self.problem.execute(code, self.instance,
                                   self.cfg["experiment_timeout_s"])
        detail = out.get("detail")
        if out["error"]:
            return ({"score": None, "valid": False, "error": out["error"],
                     "exec_time": out["exec_time"]}, detail)
        err = self.problem.validate(self.instance, out["solution"])
        if err:
            return ({"score": None, "valid": False,
                     "error": f"invalid solution: {err}",
                     "exec_time": out["exec_time"]}, detail)
        score = self.problem.evaluate(self.instance, out["solution"])
        return ({"score": score, "valid": True, "error": None,
                 "exec_time": out["exec_time"], "solution": out["solution"]}, detail)

    @staticmethod
    def _retry_feedback(code: str | None, result: dict, truncated: bool = False) -> str:
        if truncated:
            return ("Your previous reply was CUT OFF at the output token limit before "
                    "the ```python block closed, so submission.py was incomplete. "
                    "Write a SHORTER, fully self-contained submission: trim comments "
                    "and dead code, and make sure the closing ``` is reached.")
        if not code:
            return ("Your reply contained no ```python code block, so nothing ran. "
                    "Emit the one-line JSON followed by exactly one ```python fence "
                    "containing the whole submission.py, defining custom_kernel(data).")
        error = result.get("error") or ""
        if "incorrect kernel" in error or "mismatch" in error:
            return (f"Your kernel was REJECTED BY THE CORRECTNESS GATE: {error}\n"
                    "It was never timed, so it scored nothing. Match the reference "
                    "exactly: keep at least the reference's accumulation precision "
                    "(lowering it is the most common cause), and make sure every "
                    "test shape is handled — including non-square and small ones. "
                    "Fix correctness first; optimise only once it passes.")
        if "timed out" in error or "timeout" in error:
            return (f"The harness timed out: {error}. Compilation and autotuning "
                    "count against that clock, so shrink any Triton autotune "
                    "space, simplify or drop any cpp_extension build, and cache "
                    "compiled artifacts at module scope so the cost is paid once.")
        return (f"The submission you returned failed: {error}. "
                "Return a corrected, complete submission.py in one ```python fence.")

    # --------------------------------------------------------------- phases
    def execute(self):
        try:
            self.run.set_status("scoping")
            self._setup()
            self._recall_phase()
            self._research_phase()
            self._scope_phase()
            self.run.set_status("running")
            self._hypothesis_phase()
            ended = self._iterate()
            self._conclude(ended)
            self.run.set_status("completed")
        except StopRun as e:
            try:
                self._conclude(e.reason)
            finally:
                self.run.set_status(
                    "budget_exceeded" if e.reason == "budget exceeded" else "stopped")
        except Exception as e:  # engine error
            self.run.emit("run.failed", payload={"error": f"{type(e).__name__}: {e}"})
            self.run.set_status("failed")

    def _setup(self):
        self.baseline_solution, self.baseline_score, baseline_alg = \
            self.problem.baseline(self.instance)
        self.baseline_alg = baseline_alg
        self.run.emit("run.created", payload={
            "config": self.cfg,
            "problem": {"name": self.problem.name,
                        "description": self.problem.description,
                        "stats": self.problem.instance_stats(self.instance)},
            "instance": self.instance,
            "baseline": {"solution": self.baseline_solution,
                         "score": self.baseline_score, "algorithm": baseline_alg},
        })
        self._setup_evolution()

    def _setup_evolution(self):
        """Boot the evolution substrate: the run's git repo (every attempt will
        be a real commit), the program population (DGM parent selection +
        behavior-niche elites), archive seeding, and the operator bandit.
        Each piece degrades gracefully — a run never fails because of them."""
        if self.cfg.get("enable_git_repo", True):
            repo = RunRepo(DATA_DIR / self.run.id, self.run.id)
            root = repo.commit_root(
                f"# baseline: {self.baseline_alg}\n"
                f"# score: {self.baseline_score} (lower is better)\n"
                "# Every solver attempt in this run is a commit descending "
                "from this root.\n",
                {"baseline": True, "algorithm": self.baseline_alg,
                 "score": self.baseline_score}) if repo.available else None
            self.repo = repo if root else None
            self.run.emit("git.initialized", payload={
                "available": self.repo is not None, "root_sha": root,
                "path": str(repo.path) if self.repo else None})
        self.population = Population(self.baseline_score, self.rng)
        # seed the gene pool with past winners: elites from the cross-run
        # archive re-enter as first-class parents, not just prompt digests
        seeded = 0
        if self.cfg.get("enable_knowledge_archive", True):
            try:
                seeds = knowledge.ARCHIVE.recall_seed_programs(self.problem.name)
            except Exception:
                seeds = []
            for i, s in enumerate(seeds):
                imp = s.get("improvement_pct")
                est_score = (self.baseline_score * (1 - imp / 100.0)
                             if imp is not None else None)
                pid = f"arch-{i}"
                self.population.add(Program(
                    id=pid, code=s["code"], branch_id=None, round=0,
                    score=est_score, valid=True, tags=s.get("techniques") or [],
                    origin="archive", name=s.get("name")))
                self.novelty.add(pid, s["code"])
                seeded += 1
        if self.cfg.get("enable_operator_bandit", True):
            models = EXPERIMENTER_MODELS or [AGENT_MODELS["experimenter"]]
            self.bandit = OperatorBandit(list(OPERATORS), models, self.rng)
        self.run.emit("population.seeded", agent="archivist", payload={
            "archive_seeds": seeded,
            "operators": list(OPERATORS),
            "bandit_models": (EXPERIMENTER_MODELS
                              or [AGENT_MODELS["experimenter"]])
                             if self.bandit else None})

    def _recall_phase(self):
        """Long-term memory: recall what previous runs on this problem already
        learned (elite solvers per technique niche + transferable insights) so
        the Planner and Strategist build on it instead of starting from zero.
        Pure retrieval — no LLM call, no cost."""
        if not self.cfg.get("enable_knowledge_archive", True):
            return
        try:
            query = f"{self.problem.description} {self.problem.instance_stats(self.instance)}"
            recall = knowledge.ARCHIVE.recall(self.problem.name, query)
            if recall:
                self.memory = knowledge.KnowledgeArchive.as_prompt(recall)
                self.run.emit("knowledge.recalled", agent="archivist",
                              payload=recall)
            else:
                self.run.emit("knowledge.recalled", agent="archivist",
                              payload={"empty": True,
                                       "archive_size": knowledge.ARCHIVE.size()})
        except Exception as e:  # memory must never break a run
            self.memory = None
            self.run.emit("knowledge.recalled", agent="archivist",
                          payload={"error": f"{type(e).__name__}: {e}"})

    def _research_phase(self):
        """Optional: a web-research agent surveys the state of the art and feeds
        the Planner and Strategist. Degrades gracefully if web search is off or
        unavailable."""
        if not self.cfg.get("enable_web_research", True):
            return
        try:
            res = self._call(
                "researcher", agents.RESEARCHER_SYSTEM,
                agents.researcher_prompt(self.problem.description,
                                         self.problem.instance_stats(self.instance)))
            self.research = (res.text or "").strip() or None
            self.run.emit("research.findings", agent="researcher",
                          payload={"findings": self.research})
        except Exception as e:
            self.research = None
            self.run.emit("research.findings", agent="researcher",
                          payload={"findings": None,
                                   "error": f"{type(e).__name__}: {e}"})

    def _scope_phase(self):
        res = self._call(
            "planner", agents.PLANNER_SYSTEM,
            agents.planner_prompt(self.problem.description,
                                  self.problem.instance_stats(self.instance),
                                  self.baseline_alg, self.baseline_score, self.cfg,
                                  research=self.research, memory=self.memory),
            context={"config": self.cfg, "baseline_score": self.baseline_score})
        self.scope = agents.parse_json(res.text)
        self.run.emit("scope.defined", agent="planner", payload={"scope": self.scope})

    def _hypothesis_phase(self):
        # the Planner decides the count; config value is only a fallback / cap
        k = self.scope.get("initial_hypotheses") or self.cfg["num_hypotheses"]
        try:
            k = int(k)
        except (TypeError, ValueError):
            k = self.cfg["num_hypotheses"]
        k = max(1, min(k, self.cfg["max_branches"]))
        res = self._call(
            "strategist", agents.STRATEGIST_SYSTEM,
            agents.strategist_prompt(self.problem.description,
                                     self.problem.instance_stats(self.instance),
                                     self.scope, k, research=self.research,
                                     memory=self.memory),
            context={"k": k})
        hyps = agents.parse_json(res.text).get("hypotheses", [])
        self.run.emit("hypotheses.proposed", agent="strategist",
                      payload={"hypotheses": hyps})
        for h in hyps[:k]:
            self._create_branch(h.get("name", "unnamed"), h.get("hypothesis", ""),
                                h.get("strategy", "unknown"), [],
                                extra={"risk": h.get("risk")})

    def _create_branch(self, name: str, hypothesis: str, strategy: str,
                       parent_ids: list[str], extra: dict | None = None,
                       git_from: str | None = None) -> Branch:
        b = Branch(id="b-" + uuid.uuid4().hex[:6], name=name,
                   hypothesis=hypothesis, strategy=strategy, parent_ids=parent_ids)
        self.branches[b.id] = b
        git_base = self.repo.create_branch(b.id, git_from) if self.repo else None
        self.run.emit("branch.created", branch_id=b.id, payload={
            "branch": b.public(), "git_base": git_base, **(extra or {})})
        return b

    def _iterate(self) -> str:
        for rnd in range(1, self.cfg["max_rounds"] + 1):
            self.round = rnd
            active = self._active()
            if not active:
                return "all branches closed"
            self._run_round(active, rnd)           # experiments IN PARALLEL (barrier)
            _, best = self._best_overall()
            target = self.scope.get("success_criteria", {}).get(
                "target_improvement_pct", self.cfg["target_improvement_pct"])
            imp = self._improvement_pct(best)
            self._supervise(rnd)                   # prune: collapse / merge
            if imp is not None and imp >= target:
                return f"target improvement reached ({imp}% >= {target}%)"
            # the Planner reviews the round's output and designs NEW hypotheses
            if not self._planner_review(rnd, best, target):
                return "planner concluded the run"
        return "max rounds completed"

    def _run_round(self, active: list[Branch], rnd: int):
        """Experiment every active branch concurrently, then wait for all (a
        barrier) before the round's review — exactly the parallel-then-sync model."""
        workers = min(len(active), 8)
        stop = None
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self._experiment, b, rnd): b for b in active}
            for f in as_completed(futs):
                try:
                    f.result()
                except StopRun as e:
                    stop = stop or e
                except Exception as e:  # one branch crashing must not kill the run
                    b = futs[f]
                    self.run.emit("experiment.completed", agent="experimenter",
                                  branch_id=b.id, payload={
                                      "round": rnd, "valid": False, "score": None,
                                      "error": f"engine error: {type(e).__name__}: {e}",
                                      "exec_time": 0.0, "improved": False,
                                      "beats_baseline": False, "code": None,
                                      "baseline_score": self.baseline_score,
                                      "branch_best_score": b.best_score,
                                      "improvement_pct": None, "detail": None,
                                      "retries": 0})
        if stop:
            raise stop

    def _planner_review(self, rnd: int, best: float | None, target: float) -> bool:
        """The Planner sees every branch's output + insights and may spawn new
        hypotheses (new directions) — the branch count is not fixed. Returns
        whether the run should continue."""
        publics = [{**b.public(),
                    "rounds_without_improvement": b.rounds_without_improvement,
                    "failures_in_a_row": b.failures_in_a_row,
                    "last_error": b.last_error}
                   for b in self.branches.values()]
        res = self._call(
            "planner", agents.PLANNER_REVIEW_SYSTEM,
            agents.planner_review_prompt(
                rnd, self.cfg["max_rounds"], publics,
                [i.public() for i in self.insights], self.baseline_score, best,
                target, self.total_cost, self.cfg["budget_usd"],
                len(self._active()), self.cfg["max_branches"]),
            context={"review": True, "round": rnd},
            action="reviewing results and designing new hypotheses")
        try:
            dec = agents.parse_json(res.text)
        except Exception:
            dec = {"new_hypotheses": [], "evolve": [], "continue": True,
                   "reasoning": res.text[:300]}
        spawned, evolved = [], []
        # EVOLVE: fork an existing branch, carrying over its best code so the new
        # branch builds on established progress instead of starting from zero
        for e in (dec.get("evolve") or []):
            if len(self._active()) >= self.cfg["max_branches"]:
                break
            parent = self.branches.get(e.get("parent_id"))
            if not parent:
                continue
            b = self._create_branch(e.get("name", "evolved"), e.get("hypothesis", ""),
                                    e.get("strategy", parent.strategy), [parent.id],
                                    extra={"risk": e.get("risk"), "planner_round": rnd,
                                           "evolved_from": parent.id},
                                    git_from=self.repo.head(parent.id)
                                    if self.repo else None)
            b.best_code = parent.best_code  # seed with the parent's best solver
            b.best_score = parent.best_score
            b.best_solution = parent.best_solution
            b._last_pid = getattr(parent, "_last_pid", None)  # lineage continues
            evolved.append(b.id)
        # NEW: brand-new directions from scratch
        for h in (dec.get("new_hypotheses") or []):
            if len(self._active()) >= self.cfg["max_branches"]:
                break
            b = self._create_branch(h.get("name", "unnamed"), h.get("hypothesis", ""),
                                    h.get("strategy", "unknown"), [],
                                    extra={"risk": h.get("risk"), "planner_round": rnd})
            spawned.append(b.id)
        self.run.emit("planner.review", agent="planner", payload={
            "round": rnd, "reasoning": dec.get("reasoning"),
            "new_branch_ids": spawned, "evolved_branch_ids": evolved,
            "continue": bool(dec.get("continue", True))})
        return bool(dec.get("continue", True))

    def _pick_operator(self, b: Branch, crossover: dict | None
                       ) -> tuple[tuple | None, str | None, str | None, str | None]:
        """(bandit arm, operator name, prompt instruction, model override)."""
        if crossover:
            # a merged branch's first experiment is always a recombination
            return None, "recombine", OPERATORS["recombine"], None
        if not self.bandit:
            return None, None, None, None
        # nothing to refine yet -> restrict to from-scratch operators
        allowed = None if b.best_code else ["rewrite", "explore"]
        arm = self.bandit.pick(allowed)
        operator, model = arm
        model_override = model if model != AGENT_MODELS["experimenter"] else None
        return arm, operator, OPERATORS[operator], model_override

    def _pick_base(self, b: Branch, operator: str | None
                   ) -> tuple[dict | None, str | None, Program | None]:
        """The program this attempt mutates: usually the branch's own best,
        but with population_parent_prob a DGM-sampled parent from the WHOLE
        population (other branches, dead lineages, past runs).
        Returns (base prompt section, parent program id, population Program)."""
        own_pid = getattr(b, "_last_pid", None)
        if (self.population and operator != "rewrite" and
                self.rng.random() < self.cfg.get("population_parent_prob", 0.35)):
            p = self.population.select_parent()
            if p is not None and p.code and p.code != b.best_code:
                label = (f"a parent sampled from the lab population "
                         f"(origin: {p.origin}"
                         + (f', "{p.name}"' if p.name else "") + ")")
                return ({"code": p.code, "score": p.score, "label": label},
                        p.id, p)
        if b.best_code:
            return ({"code": b.best_code, "score": b.best_score,
                     "label": "your current best"}, own_pid, None)
        return None, own_pid, None

    def _experiment(self, b: Branch, rnd: int):
        self._check_interrupts()
        attempt = b.experiments + 1
        last = getattr(b, "_last_result", None)
        critique = getattr(b, "_last_critique", None)
        max_attempts = self.cfg.get("experiment_max_attempts", 3)
        crossover = getattr(b, "_crossover", None)

        arm, operator, op_instruction, model_override = \
            self._pick_operator(b, crossover)
        base, parent_pid, pool_parent = self._pick_base(b, operator)
        if crossover:
            base = None  # the two merge parents replace the single base

        inspirations = []
        if self.population and not crossover:
            exclude = {parent_pid} if parent_pid else set()
            progs = self.population.select_inspirations(
                self.cfg.get("inspiration_count", 3), exclude)
            inspirations = [
                {"id": p.id, "name": p.name, "score": p.score, "tags": p.tags,
                 "niche": niche_key(p), "code": p.code}
                for p in progs if not base or p.code != base.get("code")]

        # novelty gate thresholds: exploratory operators are held to a
        # stricter bar; a refine may stay close to its OWN parent but must
        # not be a near-copy of any OTHER program
        gate_on = self.cfg.get("enable_novelty_gate", True)
        thresh = self.cfg.get("novelty_threshold", 0.92)
        if operator in ("rewrite", "explore"):
            thresh = min(thresh, self.cfg.get("novelty_threshold_explore", 0.85))

        # Retry loop: a malformed reply (no code), a runtime/validation error or
        # a novelty-gate rejection is recoverable — re-ask the experimenter
        # immediately with the exact reason, up to max_attempts, before this
        # round counts as a failure. Each retry is emitted so the UI can draw
        # the loop.
        meta: dict = {"approach": "(unparseable)", "expectation": ""}
        code = None
        result: dict = {"score": None, "valid": False, "error": "not run", "exec_time": 0.0}
        detail = None
        retry_feedback = None
        attempt_cost = 0.0
        novelty_sim, novelty_near = 0.0, None
        novelty_rejections = 0
        last_rejected_code = None
        try_i = 1
        for try_i in range(1, max_attempts + 1):
            action = (f"writing solver code (round {rnd}, attempt {attempt})"
                      if try_i == 1 else
                      f"fixing error (round {rnd}, retry {try_i}/{max_attempts})")
            res = self._call(
                "experimenter", agents.EXPERIMENTER_SYSTEM,
                agents.experimenter_prompt(
                    self.problem.solver_contract(),
                    self.problem.instance_stats(self.instance),
                    b.public(), rnd, last, critique,
                    self._relevant_insights(f"{b.hypothesis} {b.strategy}"),
                    self.cfg["experiment_timeout_s"], retry_feedback=retry_feedback,
                    operator_instruction=op_instruction, base=base,
                    inspirations=inspirations, crossover=crossover),
                context={"strategy": b.strategy, "attempt": attempt}, branch_id=b.id,
                action=action, model=model_override)
            attempt_cost += res.cost_usd
            try:
                meta = agents.parse_json(res.text)
            except Exception:
                meta = {"approach": "(unparseable)", "expectation": ""}
            code = agents.parse_code(res.text)
            if try_i == 1:
                self.run.emit("experiment.started", agent="experimenter", branch_id=b.id,
                              payload={"round": rnd, "attempt": attempt,
                                       "operator": operator,
                                       "approach": meta.get("approach"),
                                       "expectation": meta.get("expectation")})

            # ---- novelty gate: refuse to evaluate near-copies of programs the
            # lab has already tried (ShinkaEvolve-style, checked BEFORE any
            # sandbox time). Soft: one improvement demand, then run it anyway.
            if code and gate_on and self.novelty.size():
                exclude = {parent_pid} if parent_pid else None
                novelty_sim, novelty_near = self.novelty.nearest(code, exclude)
                own_sim = (self.novelty.similarity_to(code, parent_pid)
                           if parent_pid else 0.0)
                is_copy = novelty_sim >= thresh or own_sim >= 0.995
                # if the model resubmits the same code after a rejection, let
                # it through (soft gate) — never wedge a run on stubbornness
                if (is_copy and novelty_rejections == 0 and try_i < max_attempts
                        and code != last_rejected_code):
                    novelty_rejections += 1
                    last_rejected_code = code
                    worst = max(novelty_sim, own_sim)
                    near = novelty_near if novelty_sim >= own_sim else parent_pid
                    self.run.emit("novelty.rejected", agent="experimenter",
                                  branch_id=b.id, payload={
                                      "round": rnd, "attempt": attempt,
                                      "similarity": round(worst, 3),
                                      "nearest_program": near,
                                      "threshold": thresh})
                    self.run.emit("experiment.retry", agent="experimenter",
                                  branch_id=b.id, payload={
                                      "round": rnd, "attempt": attempt,
                                      "retry": try_i, "max_attempts": max_attempts,
                                      "reason": f"novelty gate: {worst:.0%} similar "
                                                "to an already-evaluated program"})
                    retry_feedback = (
                        "REJECTED BY THE NOVELTY GATE before execution: your "
                        f"program is {worst:.0%} structurally similar to a program "
                        "the lab has ALREADY evaluated (renaming variables does "
                        "not help — the comparison is on canonicalized AST "
                        "structure). Produce a MEANINGFULLY DIFFERENT algorithm: "
                        "change the construction, the neighborhood, the "
                        "acceptance rule, or the overall search strategy — not "
                        "the surface details.")
                    continue

            result, detail = self._run_attempt(code)
            if code is None and getattr(res, "truncated", False):
                # not "no code" — we cut it off at the token cap; say so honestly
                result["error"] = "reply truncated at the output token limit (code cut off)"
            if result["valid"] or try_i == max_attempts:
                break
            retry_feedback = self._retry_feedback(
                code, result, truncated=getattr(res, "truncated", False))
            self.run.emit("experiment.retry", agent="experimenter", branch_id=b.id,
                          payload={"round": rnd, "attempt": attempt, "retry": try_i,
                                   "max_attempts": max_attempts,
                                   "reason": result["error"]})

        retries = try_i - 1
        improved = (result["valid"] and
                    (b.best_score is None or result["score"] < b.best_score))
        beats_baseline = result["valid"] and result["score"] < self.baseline_score
        b.experiments += 1
        if result["valid"]:
            b.failures_in_a_row = 0
            if improved:
                b.best_score = result["score"]
                b.best_code = code
                b.best_solution = result["solution"]
                b.rounds_without_improvement = 0
            else:
                b.rounds_without_improvement += 1
        else:
            b.failures_in_a_row += 1
            b.rounds_without_improvement += 1
            b.last_error = result["error"]

        # ---- register the attempt in the evolution substrate --------------
        behavior = None
        if result["valid"]:
            try:
                behavior = self.problem.behavior_descriptor(
                    self.instance, result["solution"], result["exec_time"],
                    self.cfg["experiment_timeout_s"])
            except Exception:
                behavior = None
        tags = knowledge.technique_tags(code, b.hypothesis, meta.get("approach"))

        sha = None
        if self.repo and code:
            # a population parent from another lineage makes this a true
            # merge commit — the git DAG records the cross-lineage gene flow
            # (a crossover branch needs none: its merge commit already
            # recorded both parents, so the weaker head is an ancestor)
            extra_parents = ([pool_parent.id] if not crossover
                             and pool_parent is not None
                             and pool_parent.origin == "run" else None)
            sha = self.repo.commit_attempt(b.id, code, {
                "round": rnd, "attempt": attempt, "score": result["score"],
                "valid": result["valid"], "error": result["error"],
                "operator": operator, "behavior": behavior, "tags": tags,
                "branch": b.name,
                "summary": f"{b.name} r{rnd}: score={result['score']} "
                           f"op={operator or '-'} valid={result['valid']}",
            }, extra_parents=extra_parents)
        pid = sha or ("p-" + uuid.uuid4().hex[:8])

        if code:
            self.novelty.add(pid, code)
        niche_out = None
        if result["valid"] and self.population is not None:
            niche_out = self.population.add(Program(
                id=pid, code=code, branch_id=b.id, round=rnd,
                score=result["score"], valid=True, behavior=behavior,
                tags=tags, operator=operator, parent_id=parent_pid,
                origin="run", name=f"{b.name} r{rnd}"))
            b._last_pid = pid
        if crossover:
            b._crossover = None  # consumed by this experiment

        if self.bandit and arm:
            self.bandit.reward(arm, improved, result["valid"],
                               bool(niche_out and niche_out.get("new_niche")),
                               attempt_cost)

        if sha:
            self.run.emit("git.committed", agent="experimenter", branch_id=b.id,
                          payload={"sha": sha, "round": rnd,
                                   "score": result["score"],
                                   "valid": result["valid"],
                                   "operator": operator,
                                   "parent_program": parent_pid,
                                   "niche": (niche_out or {}).get("niche")})

        ev_payload = {
            "round": rnd, "attempt": attempt, "score": result["score"],
            "valid": result["valid"], "error": result["error"],
            "exec_time": result["exec_time"], "improved": improved,
            "beats_baseline": beats_baseline, "code": code,
            "baseline_score": self.baseline_score,
            "branch_best_score": b.best_score,
            "improvement_pct": self._improvement_pct(result["score"]) if result["valid"] else None,
            "detail": detail,
            "retries": retries,
            "operator": operator,
            "git_sha": sha,
            "parent_program": parent_pid,
            "behavior": behavior,
            "niche": niche_out,
            "novelty": {"max_similarity": round(novelty_sim, 3),
                        "nearest_program": novelty_near,
                        "rejections": novelty_rejections},
        }
        if result["valid"]:
            ev_payload["solution"] = result["solution"]
        self.run.emit("experiment.completed", agent="experimenter",
                      branch_id=b.id, payload=ev_payload)
        b._last_result = {k: result.get(k) for k in ("score", "valid", "error", "exec_time")}

        # critic
        _, best_overall = self._best_overall()
        cres = self._call(
            "critic", agents.CRITIC_SYSTEM,
            agents.critic_prompt(b.public(), rnd, b._last_result,
                                 self.baseline_score, best_overall),
            context={"strategy": b.strategy, "attempt": attempt,
                     "error": result["error"], "improved": improved,
                     "beats_baseline": beats_baseline},
            branch_id=b.id)
        try:
            crit = agents.parse_json(cres.text)
        except Exception:
            crit = {"verdict": "unknown", "analysis": cres.text[:500],
                    "insight": None, "suggestion": None}
        self.run.emit("critique.added", agent="critic", branch_id=b.id, payload={
            "round": rnd, "verdict": crit.get("verdict"),
            "analysis": crit.get("analysis"), "suggestion": crit.get("suggestion")})
        b._last_critique = crit.get("analysis")

        ins_text = crit.get("insight")
        if ins_text:
            with self._lock:  # shared knowledge pool — parallel branches append here
                if not any(i.text == ins_text for i in self.insights):
                    ins = Insight(id="k-" + uuid.uuid4().hex[:6], branch_id=b.id,
                                  round=rnd, text=ins_text)
                    self.insights.append(ins)
                    self.run.emit("insight.added", agent="critic", branch_id=b.id,
                                  payload={"insight": ins.public()})

    def _supervise(self, rnd: int):
        publics = [{**b.public(),
                    "rounds_without_improvement": b.rounds_without_improvement,
                    "failures_in_a_row": b.failures_in_a_row,
                    "last_error": b.last_error}
                   for b in self.branches.values()]
        res = self._call(
            "supervisor", agents.SUPERVISOR_SYSTEM,
            agents.supervisor_prompt(rnd, self.cfg["max_rounds"], publics,
                                     [i.public() for i in self.insights],
                                     self.baseline_score, self.total_cost,
                                     self.cfg["budget_usd"],
                                     self.cfg["stagnation_rounds"]),
            context={"round": rnd, "branches": publics})
        try:
            dec = agents.parse_json(res.text)
        except Exception:
            dec = {"collapse": [], "merge": None, "reasoning": res.text[:500]}
        self.run.emit("supervisor.decision", agent="supervisor", payload={
            "round": rnd, "decisions": dec, "reasoning": dec.get("reasoning")})

        for c in dec.get("collapse") or []:
            b = self.branches.get(c.get("branch_id"))
            if b and b.status == "active":
                b.status = "collapsed"
                self.run.emit("branch.collapsed", agent="supervisor",
                              branch_id=b.id,
                              payload={"reason": c.get("reason"),
                                       "final_score": b.best_score})

        m = dec.get("merge")
        if m and isinstance(m, dict):
            sources = [self.branches.get(sid) for sid in m.get("source_ids", [])]
            sources = [s for s in sources if s and s.status == "active"]
            if len(sources) == 2:
                # crossover: the merge is a REAL two-parent git merge, and the
                # merged branch's first experiment sees BOTH parents' code plus
                # the supervisor's reflection on why combining should win
                # (ReEvo-style verbal gradient), instead of silently inheriting
                # only the stronger parent.
                stronger = min(sources,
                               key=lambda s: s.best_score if s.best_score is not None
                               else float("inf"))
                weaker = next(s for s in sources if s is not stronger)
                nb = self._create_branch(
                    m.get("name", "merged"), m.get("hypothesis", ""),
                    m.get("strategy", "merged"), [s.id for s in sources],
                    extra={"merge_reason": m.get("reason")},
                    git_from=self.repo.head(stronger.id) if self.repo else None)
                nb.best_code = stronger.best_code
                merge_sha = None
                if self.repo and stronger.best_code:
                    merge_sha = self.repo.commit_attempt(
                        nb.id, stronger.best_code, {
                            "merge": True, "round": rnd,
                            "sources": [s.id for s in sources],
                            "reason": m.get("reason"),
                            "summary": f"merge {stronger.name} + {weaker.name}",
                        }, extra_parents=[self.repo.head(weaker.id)])
                    nb._last_pid = merge_sha
                if stronger.best_code and weaker.best_code:
                    nb._crossover = {
                        "reflection": m.get("reason"),
                        "a_name": stronger.name, "a_score": stronger.best_score,
                        "a_code": stronger.best_code,
                        "b_name": weaker.name, "b_score": weaker.best_score,
                        "b_code": weaker.best_code,
                        "b_sha": self.repo.head(weaker.id) if self.repo else None,
                    }
                for s in sources:
                    s.status = "merged"
                self.run.emit("branch.merged", agent="supervisor",
                              branch_id=nb.id,
                              payload={"source_ids": [s.id for s in sources],
                                       "new_branch_id": nb.id,
                                       "git_sha": merge_sha,
                                       "reason": m.get("reason")})

    def _conclude(self, ended_reason: str):
        winner, best = self._best_overall()
        results = {
            "ended_reason": ended_reason,
            "rounds_completed": self.round,
            "baseline_score": self.baseline_score,
            "best_score": best,
            "improvement_pct": self._improvement_pct(best),
            "target_improvement_pct": self.scope.get("success_criteria", {}).get(
                "target_improvement_pct", self.cfg["target_improvement_pct"]),
            "total_cost_usd": round(self.total_cost, 6),
            "cost_by_agent": {k: round(v, 6) for k, v in self.cost_by_agent.items()},
            "cost_by_branch": {b.id: round(b.cost_usd, 6)
                               for b in self.branches.values()},
        }
        # evolution substrate summary: what the population looked like, which
        # behavior niches were discovered, what the bandit learned, and the
        # run's full git DAG
        try:
            results["evolution"] = {
                "population": self.population.stats() if self.population else None,
                "niche_elites": (self.population.elites_public()
                                 if self.population else []),
                "bandit": self.bandit.stats() if self.bandit else None,
                "novelty_index_size": self.novelty.size(),
                "git_log": self.repo.log_graph() if self.repo else None,
            }
        except Exception:
            pass
        if winner is not None and winner.best_solution is not None:
            # independent re-verification of the winning solution
            err = self.problem.validate(self.instance, winner.best_solution)
            verified_score = (None if err
                              else self.problem.evaluate(self.instance,
                                                         winner.best_solution))
            results.update({
                "winner_branch_id": winner.id,
                "winner_branch_name": winner.name,
                "best_solution": winner.best_solution,
                "winner_code": winner.best_code,
                "verified": err is None and verified_score == winner.best_score,
                "verified_score": verified_score,
                "target_met": (results["improvement_pct"] or 0) >=
                              results["target_improvement_pct"],
            })
            # held-out verification: does the winning solver generalize to
            # instances it never saw during the run?
            if winner.best_code:
                try:
                    holdout = self.problem.holdout_eval(
                        winner.best_code, self.instance,
                        self.cfg["experiment_timeout_s"])
                except Exception as e:
                    holdout = {"error": f"{type(e).__name__}: {e}"}
                if holdout is not None:
                    results["holdout"] = holdout
            winner.status = "winner"
            self.run.emit("branch.winner", agent="supervisor", branch_id=winner.id,
                          payload={"score": winner.best_score,
                                   "improvement_pct": results["improvement_pct"]})
        # long-term memory: archive this run's outcome so future runs start
        # from what it learned (elite solver per niche + insights)
        if self.cfg.get("enable_knowledge_archive", True):
            try:
                outcome = knowledge.ARCHIVE.ingest_run(
                    self.run.id, self.problem.name, results,
                    [i.text for i in self.insights])
                self.run.emit("knowledge.archived", agent="archivist",
                              payload=outcome)
            except Exception as e:  # memory must never break concluding
                self.run.emit("knowledge.archived", agent="archivist",
                              payload={"error": f"{type(e).__name__}: {e}"})
        self.run.emit("run.completed", payload={"results": results})
