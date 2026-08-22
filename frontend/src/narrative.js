import { fmtScore } from "./format.js";

// Turns raw events into human-readable story lines attributed to an agent.
// Returns null for noise events (llm.called, experiment.started) — those stay
// available in the raw Events tab.

const DEFAULT_AGENT = {
  "research.findings": "researcher",
  "planner.review": "planner",
  "scope.defined": "planner",
  "hypotheses.proposed": "strategist",
  "experiment.completed": "experimenter",
  "experiment.retry": "experimenter",
  "critique.added": "critic",
  "insight.added": "critic",
  "supervisor.decision": "supervisor",
  "branch.collapsed": "supervisor",
  "branch.merged": "supervisor",
  "branch.winner": "supervisor",
  "knowledge.recalled": "archivist",
  "knowledge.archived": "archivist",
  "novelty.rejected": "experimenter",
  "population.seeded": "archivist",
  "budget.capped": "planner",
};

export function narrate(ev, state) {
  const p = ev.payload || {};
  const bname = (id) => state.branches[id]?.name || id;
  const agent = ev.agent || DEFAULT_AGENT[ev.type] || null;

  switch (ev.type) {
    case "run.created": {
      const c = p.config || {};
      return {
        agent: null,
        // problem.stats already names the kernel and the backend
        text: `Run created — ${p.problem?.stats || c.problem || "problem"} ` +
          `${c.num_hypotheses} hypotheses, budget $${c.budget_usd}. ` +
          `Baseline (${p.baseline?.algorithm}): ${fmtScore(p.baseline?.score)} ns.`,
      };
    }
    case "research.findings":
      return p.findings
        ? { agent, text: `Searched the web for state-of-the-art approaches and shared the findings with the lab.` }
        : { agent, text: `Web research unavailable${p.error ? ` (${p.error})` : ""}; proceeding without it.` };
    case "scope.defined":
      return {
        agent,
        text: `Set the objective: ${p.scope?.objective} ` +
          `Target: at least ${p.scope?.success_criteria?.target_improvement_pct}% over baseline` +
          `${p.scope?.initial_hypotheses ? `, with ${p.scope.initial_hypotheses} hypotheses to explore` : ""}.`,
      };
    case "planner.review": {
      const n = (p.new_branch_ids || []).length;
      const ev = (p.evolved_branch_ids || []).length;
      if (!p.continue) return { agent, text: `Reviewed the round and concluded the run. ${p.reasoning || ""}` };
      const bits = [];
      if (ev) bits.push(`evolved ${ev} promising branch${ev === 1 ? "" : "es"}`);
      if (n) bits.push(`opened ${n} new direction${n === 1 ? "" : "s"}`);
      return {
        agent,
        text: bits.length
          ? `Reviewed the round's results and ${bits.join(" and ")}. ${p.reasoning || ""}`
          : `Reviewed the round's results — branches continue refining in place. ${p.reasoning || ""}`,
      };
    }
    case "hypotheses.proposed":
      return {
        agent,
        text: `Proposed ${(p.hypotheses || []).length} hypotheses to explore in parallel branches.`,
      };
    case "branch.created": {
      const b = p.branch || {};
      if (p.evolved_from)
        return {
          agent: "planner",
          text: `Evolved "${bname(p.evolved_from)}" into "${b.name}" (keeps its code) — ${b.hypothesis}`,
        };
      if (b.parent_ids?.length)
        return {
          agent: "supervisor",
          text: `Merged ${b.parent_ids.map(bname).join(" + ")} into "${b.name}"` +
            `${p.merge_reason ? ` — ${p.merge_reason}` : "."}`,
        };
      if (p.planner_round)
        return { agent: "planner", text: `Designed a new hypothesis from round ${p.planner_round}'s evidence: "${b.name}" — ${b.hypothesis}` };
      return { agent: "strategist", text: `Opened branch "${b.name}" — ${b.hypothesis}` };
    }
    case "experiment.retry":
      if ((p.reason || "").startsWith("novelty gate")) return null; // narrated by novelty.rejected
      return {
        agent,
        text: `Round ${p.round} on "${bname(ev.branch_id)}" hit an error ` +
          `(${p.reason}) — auto-retrying (${p.retry}/${(p.max_attempts || 1) - 1}) ` +
          `instead of abandoning the branch.`,
      };
    case "novelty.rejected":
      return {
        agent,
        text: `Proposed a near-duplicate on "${bname(ev.branch_id)}" — ` +
          `${Math.round((p.similarity || 0) * 100)}% structurally similar to a ` +
          `program the lab already evaluated. The novelty gate rejected it ` +
          `before spending any budget and demanded a genuinely different algorithm.`,
      };
    case "population.seeded": {
      const n = p.archive_seeds || 0;
      const bits = [];
      if (n) bits.push(`${n} past elite kernel${n === 1 ? "" : "s"} joined the gene pool as parents`);
      if (p.bandit_models) bits.push("the operator bandit will learn which mutation pays off");
      if (!bits.length) return null;
      return { agent, text: `Prepared the evolving population: ${bits.join("; ")}.` };
    }
    case "git.initialized":
      return p.available
        ? { agent: null, text: "Git substrate ready — every kernel submission in this run becomes a real commit in the run's repository." }
        : null;
    case "experiment.completed": {
      const name = bname(ev.branch_id);
      if (!p.valid)
        return {
          agent,
          text: `Round ${p.round} on "${name}" failed: ${p.error || "the kernel did not pass correctness"}.`,
        };
      if (p.improved)
        return {
          agent,
          text: `Round ${p.round} on "${name}": ${fmtScore(p.score)} ns` +
            `${p.improvement_pct != null ? ` — ${p.improvement_pct}% faster than the reference` : ""}.`,
        };
      return { agent, text: `Round ${p.round} on "${name}": ${fmtScore(p.score)} ns, no improvement.` };
    }
    case "critique.added":
      return {
        agent,
        text: `Verdict on "${bname(ev.branch_id)}": ${p.verdict} — ${p.analysis}`,
      };
    case "insight.added":
      return { agent, text: `Shared an insight with all branches: "${p.insight?.text}"` };
    case "supervisor.decision":
      return p.reasoning ? { agent, text: `End-of-round review: ${p.reasoning}` } : null;
    case "branch.collapsed":
      return { agent, text: `Collapsed "${bname(ev.branch_id)}" — ${p.reason}` };
    case "branch.winner":
      return {
        agent,
        text: `Declared "${bname(ev.branch_id)}" the winner` +
          `${p.score != null ? ` at ${fmtScore(p.score)} ns` : ""}.`,
      };
    case "knowledge.recalled": {
      if (p.error) return { agent, text: `Lab memory unavailable (${p.error}).` };
      if (p.empty)
        return { agent, text: "Checked the lab's long-term memory — nothing archived yet; this run starts from scratch." };
      const ns = (p.solvers || []).length;
      const ni = (p.insights || []).length;
      return {
        agent,
        text: `Recalled the lab's long-term memory: ${ns} elite kernel${ns === 1 ? "" : "s"} ` +
          `and ${ni} insight${ni === 1 ? "" : "s"} from ${p.archive_size?.runs ?? "?"} past runs — ` +
          `handed to the Planner and Strategist.`,
      };
    }
    case "knowledge.archived": {
      if (p.error) return { agent, text: `Could not archive this run's knowledge (${p.error}).` };
      const bits = [];
      if (p.solver_added) bits.push(`the winning kernel now holds the "${p.niche}" niche`);
      if (p.insights_added) bits.push(`${p.insights_added} new insight${p.insights_added === 1 ? "" : "s"} joined the pool`);
      return {
        agent,
        text: bits.length
          ? `Archived this run into the lab's long-term memory: ${bits.join("; ")}.`
          : "Archived this run — no new knowledge beat what the memory already holds.",
      };
    }
    case "budget.capped":
      return {
        agent,
        text: `Budget guard: this single call had already spent the run's ` +
          `remaining budget ($${(p.cost_usd ?? 0).toFixed(2)}), so its ` +
          `web-search loop was stopped early. The answer may be incomplete.`,
      };
    case "run.completed": {
      const r = p.results || {};
      // a run can conclude without ever producing a kernel (budget, stop, error)
      if (r.improvement_pct == null)
        return {
          agent: null,
          text: `Run ended without a working kernel — ${r.ended_reason || "unknown reason"}. ` +
            `Spent $${(r.total_cost_usd ?? 0).toFixed(2)}` +
            `${r.rounds_completed ? ` over ${r.rounds_completed} round(s)` : " before any experiment ran"}.`,
        };
      let text = `Run completed — ${r.improvement_pct}% faster than the reference, ` +
        `target ${r.target_met ? "met" : "not met"}, ` +
        `total cost $${(r.total_cost_usd ?? 0).toFixed(2)}.`;
      const h = r.holdout?.summary;
      if (h)
        text += ` Held-out shapes: ${h.improved} faster, ${h.worsened} slower` +
          `${h.failed ? `, ${h.failed} failed` : ""} — the winning kernel ` +
          `${h.generalizes ? "GENERALIZES" : "does NOT generalize"}` +
          `${h.mean_speedup ? ` (mean ${h.mean_speedup}x)` : ""}.`;
      return { agent: null, text };
    }
    case "run.stopped":
      return { agent: null, text: "Run stopped by user." };
    case "run.failed":
      return { agent: null, text: `Run failed: ${p.error}` };
    default:
      return null;
  }
}
