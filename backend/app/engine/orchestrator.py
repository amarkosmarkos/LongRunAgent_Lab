"""The run engine: scope -> hypotheses -> branching iteration -> conclusion.

Problem-agnostic: everything domain-specific lives behind the Problem interface.
Every decision, experiment, insight and dollar is emitted as an event.
"""
from __future__ import annotations

import uuid

from ..llm import LLMClient
from ..models import Branch, Insight
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
        self.round = 0

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
    }

    def _call(self, role: str, system: str, prompt: str,
              context: dict | None = None, branch_id: str | None = None,
              action: str | None = None):
        self._check_interrupts()
        # announce intent BEFORE the API call so the UI can show the agent
        # "thinking" live, instead of nodes only appearing once work is done
        self.run.emit("agent.thinking", agent=role, branch_id=branch_id, payload={
            "action": action or self._THINKING.get(role, "thinking"),
            "round": self.round or None,
        })
        res = self.llm.call(role, system, prompt, context)
        self.total_cost += res.cost_usd
        self.cost_by_agent[role] = self.cost_by_agent.get(role, 0.0) + res.cost_usd
        if branch_id and branch_id in self.branches:
            self.branches[branch_id].cost_usd += res.cost_usd
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

    def _active(self) -> list[Branch]:
        return [b for b in self.branches.values() if b.status == "active"]

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
                    "the ```python block closed, so the code was incomplete. Write a "
                    "SHORTER, fully self-contained solver: trim comments and dead code, "
                    "and make sure the closing ``` is reached.")
        if not code:
            return ("Your reply contained no ```python code block, so nothing ran. "
                    "Emit the one-line JSON followed by exactly one ```python fence "
                    "that defines solve(cities) and returns a tour.")
        if "timeout" in (result.get("error") or ""):
            return (f"The solver was too slow: {result['error']}. Make solve() "
                    "time-bounded — record t0 = time.time() at the start and stop "
                    "improving once time.time() - t0 exceeds the budget, returning "
                    "the best tour found so far. Never run an unbounded loop.")
        return (f"The solver you returned failed: {result['error']}. "
                "Return a corrected, complete solver in one ```python fence.")

    # --------------------------------------------------------------- phases
    def execute(self):
        try:
            self.run.set_status("scoping")
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

    def _scope_phase(self):
        self.baseline_solution, self.baseline_score, baseline_alg = \
            self.problem.baseline(self.instance)
        self.run.emit("run.created", payload={
            "config": self.cfg,
            "problem": {"name": self.problem.name,
                        "description": self.problem.description,
                        "stats": self.problem.instance_stats(self.instance)},
            "instance": self.instance,
            "baseline": {"solution": self.baseline_solution,
                         "score": self.baseline_score, "algorithm": baseline_alg},
        })
        res = self._call(
            "planner", agents.PLANNER_SYSTEM,
            agents.planner_prompt(self.problem.description,
                                  self.problem.instance_stats(self.instance),
                                  baseline_alg, self.baseline_score, self.cfg),
            context={"config": self.cfg, "baseline_score": self.baseline_score})
        self.scope = agents.parse_json(res.text)
        self.run.emit("scope.defined", agent="planner", payload={"scope": self.scope})

    def _hypothesis_phase(self):
        res = self._call(
            "strategist", agents.STRATEGIST_SYSTEM,
            agents.strategist_prompt(self.problem.description,
                                     self.problem.instance_stats(self.instance),
                                     self.scope, self.cfg["num_hypotheses"]),
            context={"k": self.cfg["num_hypotheses"]})
        hyps = agents.parse_json(res.text).get("hypotheses", [])
        self.run.emit("hypotheses.proposed", agent="strategist",
                      payload={"hypotheses": hyps})
        for h in hyps[: self.cfg["num_hypotheses"]]:
            self._create_branch(h.get("name", "unnamed"), h.get("hypothesis", ""),
                                h.get("strategy", "unknown"), [],
                                extra={"risk": h.get("risk")})

    def _create_branch(self, name: str, hypothesis: str, strategy: str,
                       parent_ids: list[str], extra: dict | None = None) -> Branch:
        b = Branch(id="b-" + uuid.uuid4().hex[:6], name=name,
                   hypothesis=hypothesis, strategy=strategy, parent_ids=parent_ids)
        self.branches[b.id] = b
        self.run.emit("branch.created", branch_id=b.id, payload={
            "branch": b.public(), **(extra or {})})
        return b

    def _iterate(self) -> str:
        for rnd in range(1, self.cfg["max_rounds"] + 1):
            self.round = rnd
            if not self._active():
                return "all branches closed"
            for b in list(self._active()):
                self._experiment(b, rnd)
            _, best = self._best_overall()
            target = self.scope.get("success_criteria", {}).get(
                "target_improvement_pct", self.cfg["target_improvement_pct"])
            imp = self._improvement_pct(best)
            self._supervise(rnd)
            if imp is not None and imp >= target:
                return f"target improvement reached ({imp}% >= {target}%)"
        return "max rounds completed"

    def _experiment(self, b: Branch, rnd: int):
        self._check_interrupts()
        attempt = b.experiments + 1
        last = getattr(b, "_last_result", None)
        critique = getattr(b, "_last_critique", None)
        max_attempts = self.cfg.get("experiment_max_attempts", 3)

        # Retry loop: a malformed reply (no code) or a runtime/validation error is
        # recoverable — re-ask the experimenter immediately with the exact error,
        # up to max_attempts, before this round counts as a failure. Each retry is
        # emitted so the UI can draw the loop.
        meta: dict = {"approach": "(unparseable)", "expectation": ""}
        code = None
        result: dict = {"score": None, "valid": False, "error": "not run", "exec_time": 0.0}
        detail = None
        retry_feedback = None
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
                    {**b.public(), "best_code": b.best_code}, rnd, last, critique,
                    [i.public() for i in self.insights],
                    self.cfg["experiment_timeout_s"], retry_feedback=retry_feedback),
                context={"strategy": b.strategy, "attempt": attempt}, branch_id=b.id,
                action=action)
            try:
                meta = agents.parse_json(res.text)
            except Exception:
                meta = {"approach": "(unparseable)", "expectation": ""}
            code = agents.parse_code(res.text)
            if try_i == 1:
                self.run.emit("experiment.started", agent="experimenter", branch_id=b.id,
                              payload={"round": rnd, "attempt": attempt,
                                       "approach": meta.get("approach"),
                                       "expectation": meta.get("expectation")})
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
        if ins_text and not any(i.text == ins_text for i in self.insights):
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
                nb = self._create_branch(
                    m.get("name", "merged"), m.get("hypothesis", ""),
                    m.get("strategy", "merged"), [s.id for s in sources],
                    extra={"merge_reason": m.get("reason")})
                # seed the merged branch with the stronger parent's discoveries
                stronger = min(sources,
                               key=lambda s: s.best_score if s.best_score is not None
                               else float("inf"))
                nb.best_code = stronger.best_code
                for s in sources:
                    s.status = "merged"
                self.run.emit("branch.merged", agent="supervisor",
                              branch_id=nb.id,
                              payload={"source_ids": [s.id for s in sources],
                                       "new_branch_id": nb.id,
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
        self.run.emit("run.completed", payload={"results": results})
