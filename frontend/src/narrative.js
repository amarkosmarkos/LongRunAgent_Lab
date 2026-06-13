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
        text: `Run created — ${c.problem || "problem"} with ` +
          `${c.problem_params?.n_cities ?? "?"} cities, ${c.num_hypotheses} hypotheses, ` +
          `budget $${c.budget_usd}. Baseline (${p.baseline?.algorithm}): ${fmtScore(p.baseline?.score)}.`,
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
      if (!p.continue) return { agent, text: `Reviewed the round and concluded the run. ${p.reasoning || ""}` };
      return {
        agent,
        text: n > 0
          ? `Reviewed the round's results and opened ${n} new hypothes${n === 1 ? "is" : "es"} to explore next. ${p.reasoning || ""}`
          : `Reviewed the round's results — no new directions warranted. ${p.reasoning || ""}`,
      };
    }
    case "hypotheses.proposed":
      return {
        agent,
        text: `Proposed ${(p.hypotheses || []).length} hypotheses to explore in parallel branches.`,
      };
    case "branch.created": {
      const b = p.branch || {};
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
      return {
        agent,
        text: `Round ${p.round} on "${bname(ev.branch_id)}" hit an error ` +
          `(${p.reason}) — auto-retrying (${p.retry}/${(p.max_attempts || 1) - 1}) ` +
          `instead of abandoning the branch.`,
      };
    case "experiment.completed": {
      const name = bname(ev.branch_id);
      if (!p.valid)
        return {
          agent,
          text: `Round ${p.round} on "${name}" failed: ${p.error || "invalid solution"}.`,
        };
      if (p.improved)
        return {
          agent,
          text: `Round ${p.round} on "${name}": score ${fmtScore(p.score)}` +
            `${p.improvement_pct != null ? ` — ${p.improvement_pct}% vs baseline` : ""}.`,
        };
      return { agent, text: `Round ${p.round} on "${name}": score ${fmtScore(p.score)}, no improvement.` };
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
          `${p.score != null ? ` with score ${fmtScore(p.score)}` : ""}.`,
      };
    case "run.completed": {
      const r = p.results || {};
      let text = `Run completed — ${r.improvement_pct}% improvement, ` +
        `target ${r.target_met ? "met" : "not met"}, ` +
        `total cost $${(r.total_cost_usd ?? 0).toFixed(2)}.`;
      const h = r.holdout?.summary;
      if (h)
        text += ` Held-out check: ${h.improved} improved, ${h.worsened} worsened` +
          `${h.failed ? `, ${h.failed} failed` : ""} — the winner ` +
          `${h.generalizes ? "GENERALIZES" : "does NOT generalize"} ` +
          `(mean gap ${fmtScore(h.mean_baseline_gap)}% → ${fmtScore(h.mean_winner_gap)}%).`;
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
