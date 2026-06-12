import React from "react";
import TourCanvas from "./TourCanvas.jsx";
import { agentMeta } from "../agents.js";
import { narrate } from "../narrative.js";
import { fmtScore } from "../format.js";

const usd = (v) => `$${(v || 0).toFixed(4)}`;

export function AgentBadge({ name, small }) {
  const m = agentMeta(name);
  const size = small ? 20 : 26;
  return (
    <span className="agentbadge">
      <span className="avatar"
        style={{ background: m.color, width: size, height: size, fontSize: small ? 9 : 10 }}>
        {m.initials}
      </span>
      {!small && m.label}
    </span>
  );
}

// ------------------------------------------------------------- Branches
export function BranchesPanel({ state, onSelect }) {
  const { branchOrder, branches, baseline } = state;
  if (!branchOrder.length)
    return (
      <div className="empty">
        No branches yet — the Strategist proposes hypotheses first.
      </div>
    );
  return (
    <div>
      <div className="sub" style={{ marginBottom: 10 }}>
        Each hypothesis becomes a branch. Click a card to inspect it in the graph.
      </div>
      {branchOrder.map((id) => {
        const b = branches[id];
        const ended = state.results != null || state.endedStatus != null;
        const status = b.status || (ended ? "ended" : "active");
        const imp = b.best_score != null && baseline
          ? Math.round(((baseline.score - b.best_score) / baseline.score) * 1000) / 10
          : null;
        return (
          <div key={id} className={`branchcard ${status}`}
            onClick={() => onSelect(b.createdSeq)}>
            <div className="head">
              <span className="name">{b.name}</span>
              <span className={`badge ${status}`}>{status}</span>
            </div>
            {b.strategy && <div className="sub">{b.strategy}</div>}
            <div className="hyp">{b.hypothesis}</div>
            <div className="meta">
              <span>best: <b>{fmtScore(b.best_score)}</b></span>
              <span>vs baseline:{" "}
                <b style={{ color: imp > 0 ? "var(--green)" : "inherit" }}>
                  {imp != null ? `${imp}%` : "—"}
                </b>
              </span>
              <span>{b.experiments || 0} experiment{(b.experiments || 0) === 1 ? "" : "s"}</span>
            </div>
            {b.parent_ids?.length > 0 && (
              <div className="sub" style={{ marginTop: 4, color: "var(--purple)" }}>
                merged from {b.parent_ids.map((pid) => branches[pid]?.name || pid).join(" + ")}
                {b.mergeReason ? ` — ${b.mergeReason}` : ""}
              </div>
            )}
            {b.collapseReason && (
              <div className="sub" style={{ marginTop: 4, color: "var(--red)" }}>
                collapsed: {b.collapseReason}
              </div>
            )}
            {b.risk && <div className="sub" style={{ marginTop: 4 }}>risk: {b.risk}</div>}
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------- Story
export function StoryPanel({ events, state, selectedSeq, onSelect }) {
  const rows = events
    .map((ev) => ({ ev, line: narrate(ev, state) }))
    .filter((r) => r.line);
  if (!rows.length) return <div className="empty">Nothing has happened yet.</div>;
  return (
    <div>
      {rows.map(({ ev, line }) => {
        const m = agentMeta(line.agent);
        return (
          <div key={ev.seq}
            className={`storyrow ${selectedSeq === ev.seq ? "sel" : ""}`}
            onClick={() => onSelect(ev.seq)}>
            <span className="avatar" style={{ background: m.color }}>{m.initials}</span>
            <span className="txt">
              <span className="who" style={{ color: m.color }}>{m.label}</span>
              {line.text}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------- Scope
export function ScopePanel({ state }) {
  const { scope, problem, baseline, config } = state;
  if (!scope) return <div className="empty">Scope not defined yet.</div>;
  return (
    <div>
      <div className="panelcard">
        <h4>Problem</h4>
        <div>{problem?.description}</div>
        <div className="sub" style={{ marginTop: 4 }}>{problem?.stats}</div>
      </div>
      <div className="panelcard">
        <h4>Objective</h4>
        <div>{scope.objective}</div>
      </div>
      <div className="panelcard">
        <h4>Baseline</h4>
        <div className="kv">
          <span className="k">algorithm</span><span className="v">{baseline?.algorithm}</span>
          <span className="k">score</span><span className="v">{fmtScore(baseline?.score)}</span>
        </div>
      </div>
      <div className="panelcard">
        <h4>Success criteria</h4>
        <div>≥ {scope.success_criteria?.target_improvement_pct}% improvement over baseline</div>
        <div className="sub" style={{ marginTop: 4 }}>{scope.success_criteria?.rationale}</div>
      </div>
      <div className="panelcard">
        <h4>Constraints</h4>
        <ul>{(scope.constraints || []).map((c, i) => <li key={i}>{c}</li>)}</ul>
      </div>
      <div className="panelcard">
        <h4>Stop conditions</h4>
        <ul>{(scope.stop_conditions || []).map((c, i) => <li key={i}>{c}</li>)}</ul>
      </div>
      <div className="panelcard">
        <h4><AgentBadge name="planner" small /> Planner reasoning</h4>
        <div className="sub">{scope.reasoning}</div>
      </div>
      {config && (
        <div className="panelcard">
          <h4>Run config</h4>
          <div className="kv">
            <span className="k">max rounds</span><span className="v">{config.max_rounds}</span>
            <span className="k">budget</span><span className="v">${config.budget_usd}</span>
            <span className="k">hypotheses</span><span className="v">{config.num_hypotheses}</span>
            <span className="k">experiment timeout</span><span className="v">{config.experiment_timeout_s}s</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------ Knowledge
export function KnowledgePanel({ state }) {
  const { insights, branches } = state;
  if (!insights.length)
    return <div className="empty">No shared insights discovered yet.</div>;
  return (
    <div>
      <div className="sub" style={{ marginBottom: 10 }}>
        Insights extracted by the Critic, shared with every branch's Experimenter
        and used by the Supervisor for merge decisions.
      </div>
      {insights.map((ins) => (
        <div className="insight" key={ins.id}>
          {ins.text}
          <div className="src">
            from {branches[ins.branch_id]?.name || ins.branch_id} · round {ins.round}
          </div>
        </div>
      ))}
    </div>
  );
}

// ----------------------------------------------------------------- Cost
function Bars({ entries, max, withAgents }) {
  return entries.map(([name, v]) => (
    <div className="costrow" key={name}>
      <span className="name">
        {withAgents && <AgentBadge name={name} small />}
        {withAgents ? agentMeta(name).label : name}
      </span>
      <span className="bar"><div style={{ width: `${max ? (v / max) * 100 : 0}%` }} /></span>
      <span className="amt">{usd(v)}</span>
    </div>
  ));
}

export function CostPanel({ state }) {
  const { costs, branches, config } = state;
  const agents = Object.entries(costs.byAgent).sort((a, b) => b[1] - a[1]);
  const brs = Object.entries(costs.byBranch)
    .map(([id, v]) => [branches[id]?.name || id, v])
    .sort((a, b) => b[1] - a[1]);
  const maxA = Math.max(...agents.map(([, v]) => v), 0);
  const maxB = Math.max(...brs.map(([, v]) => v), 0);
  const budget = config?.budget_usd || 0;
  const pct = budget ? Math.min(100, (costs.total / budget) * 100) : 0;
  return (
    <div>
      <div className="panelcard">
        <h4>Budget</h4>
        <div className="kv">
          <span className="k">spent</span><span className="v">{usd(costs.total)}</span>
          <span className="k">budget</span><span className="v">${budget}</span>
          <span className="k">LLM calls</span><span className="v">{costs.calls}</span>
        </div>
        <div className="budgetbar" style={{ width: "100%", marginTop: 8 }}>
          <div className={pct > 80 ? "hot" : ""} style={{ width: `${pct}%` }} />
        </div>
      </div>
      <div className="panelcard">
        <h4>Cost by agent</h4>
        {agents.length
          ? <Bars entries={agents} max={maxA} withAgents />
          : <span className="sub">—</span>}
      </div>
      <div className="panelcard">
        <h4>Cost by branch</h4>
        {brs.length ? <Bars entries={brs} max={maxB} /> : <span className="sub">—</span>}
        <div className="sub" style={{ marginTop: 6 }}>
          Planner/strategist/supervisor calls are run-level and not attributed to branches.
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------- Held-out
function HoldoutReport({ holdout }) {
  if (!holdout) return null;
  if (holdout.error)
    return (
      <div className="panelcard">
        <h4>Held-out verification</h4>
        <div style={{ color: "var(--red)" }}>{holdout.error}</div>
      </div>
    );
  const s = holdout.summary || {};
  return (
    <div className="panelcard">
      <h4>Held-out verification (instances the agents never saw)</h4>
      <div style={{ marginBottom: 8 }}>
        <span className={`badge ${s.generalizes ? "winner" : "collapsed"}`}
          style={{ marginLeft: 0 }}>
          {s.generalizes ? "generalizes" : "does not generalize"}
        </span>
        <span className="sub" style={{ marginLeft: 10 }}>
          mean gap {fmtScore(s.mean_baseline_gap)}% → {fmtScore(s.mean_winner_gap)}% ·
          {" "}{s.improved} improved · {s.worsened} worsened
          {s.failed ? ` · ${s.failed} failed` : ""}
        </span>
      </div>
      <table className="htable">
        <thead>
          <tr><th>instance</th><th>cities</th><th>optimum</th>
            <th>baseline gap</th><th>winner gap</th><th>outcome</th></tr>
        </thead>
        <tbody>
          {(holdout.instances || []).map((r) => (
            <tr key={r.name}>
              <td>{r.name}</td>
              <td>{r.n_cities}</td>
              <td>{r.optimum}</td>
              <td>{fmtScore(r.baseline_gap)}%</td>
              <td>{r.winner_gap != null ? `${fmtScore(r.winner_gap)}%` : (r.error || "—")}</td>
              <td style={{
                color: r.outcome === "improved" ? "var(--green)" :
                  r.outcome === "worsened" || r.outcome === "failed"
                    ? "var(--red)" : "var(--muted)",
              }}>{r.outcome}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="sub" style={{ marginTop: 6 }}>
        A modification only counts if it beats the baseline on instances that
        were not used during development.
      </div>
    </div>
  );
}

// -------------------------------------------------------------- Results
export function ResultsPanel({ state }) {
  const { results, baseline, instance, bestSolution, bestScore } = state;
  const imp = results?.improvement_pct ??
    (bestScore != null && baseline
      ? Math.round(((baseline.score - bestScore) / baseline.score) * 1000) / 10
      : null);
  return (
    <div>
      <div className="resultbig">
        <div className="stat">
          <span className="k">baseline</span>
          <span className="v">{fmtScore(baseline?.score)}</span>
        </div>
        <div className="stat">
          <span className="k">best found</span>
          <span className="v good">{fmtScore(bestScore)}</span>
        </div>
        <div className="stat">
          <span className="k">improvement</span>
          <span className="v gold">{imp != null ? `${imp}%` : "—"}</span>
        </div>
      </div>
      {results && (
        <div className="panelcard">
          <h4>Final results</h4>
          <div className="kv">
            <span className="k">winner</span>
            <span className="v">{results.winner_branch_name || "—"}</span>
            <span className="k">target</span>
            <span className="v">{results.target_improvement_pct}%</span>
            <span className="k">target met</span>
            <span className="v" style={{ color: results.target_met ? "var(--green)" : "var(--red)" }}>
              {results.target_met ? "yes" : "no"}
            </span>
            <span className="k">re-verified</span>
            <span className="v" style={{ color: results.verified ? "var(--green)" : "var(--red)" }}>
              {results.verified ? `yes (score ${results.verified_score})` : "no"}
            </span>
            <span className="k">ended because</span>
            <span className="v">{results.ended_reason}</span>
            <span className="k">rounds</span>
            <span className="v">{results.rounds_completed}</span>
            <span className="k">total cost</span>
            <span className="v">{usd(results.total_cost_usd)}</span>
          </div>
        </div>
      )}
      {results?.holdout && <HoldoutReport holdout={results.holdout} />}
      <TourCanvas instance={instance} baseline={baseline}
        bestSolution={results?.best_solution || bestSolution} bestScore={bestScore} />
    </div>
  );
}

// ---------------------------------------------------------- Node detail
function findLlmCall(events, ev) {
  // The llm.called event emitted just before this agent output: same agent
  // (and branch, when set), highest seq below the selected event.
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    if (e.seq >= ev.seq || e.type !== "llm.called") continue;
    if (ev.agent && e.agent !== ev.agent) continue;
    if (ev.branch_id && e.branch_id !== ev.branch_id) continue;
    return e;
  }
  return null;
}

function PromptBlock({ title, text, open }) {
  if (!text) return null;
  return (
    <details className="promptblock" open={open}>
      <summary>{title} <span className="sub">({text.length.toLocaleString()} chars)</span></summary>
      <pre className="code prompt">{text}</pre>
    </details>
  );
}

export function DetailPanel({ state, events, selectedSeq, onSelect }) {
  if (selectedSeq == null)
    return <div className="empty">Select a node in the graph or an entry in the story.</div>;
  const ev = events.find((e) => e.seq === selectedSeq);
  if (!ev) return <div className="empty">Event not yet loaded.</div>;
  const p = ev.payload || {};
  const branch = ev.branch_id ? state.branches[ev.branch_id] : null;
  const llmCall = ev.type === "llm.called" ? null : findLlmCall(events, ev);

  const common = (
    <div className="panelcard">
      <h4>{ev.type}</h4>
      <div className="kv">
        <span className="k">seq</span><span className="v">#{ev.seq}</span>
        {ev.agent && (<><span className="k">agent</span>
          <span className="v"><AgentBadge name={ev.agent} /></span></>)}
        {branch && (<><span className="k">branch</span><span className="v">{branch.name}</span></>)}
      </div>
      {llmCall?.payload?.user_prompt && onSelect && (
        <button className="link" style={{ marginTop: 6, paddingLeft: 0 }}
          onClick={() => onSelect(llmCall.seq)}>
          → view the LLM call behind this (full prompt + raw response)
        </button>
      )}
    </div>
  );

  if (ev.type === "llm.called") {
    return (
      <div>
        {common}
        <div className="panelcard">
          <h4>LLM call</h4>
          <div className="kv">
            <span className="k">model</span><span className="v">{p.model}</span>
            <span className="k">input tokens</span><span className="v">{p.input_tokens}</span>
            <span className="k">output tokens</span><span className="v">{p.output_tokens}</span>
            <span className="k">cost</span><span className="v">{usd(p.cost_usd)}</span>
          </div>
        </div>
        {p.user_prompt ? (
          <div className="panelcard">
            <h4>What the agent saw and answered</h4>
            <PromptBlock title="System prompt (role definition)" text={p.system_prompt} />
            <PromptBlock title="User prompt (full accumulated context)" text={p.user_prompt} open />
            <PromptBlock title="Raw response (before parsing)" text={p.raw_response} />
          </div>
        ) : (
          <div className="panelcard">
            <div className="sub">
              This run predates prompt capture — newer runs record the full
              prompt and raw response of every call.
            </div>
          </div>
        )}
      </div>
    );
  }

  if (ev.type === "experiment.completed") {
    const started = [...events].reverse().find(
      (e) => e.type === "experiment.started" && e.branch_id === ev.branch_id &&
        e.seq < ev.seq);
    const critique = events.find(
      (e) => e.type === "critique.added" && e.branch_id === ev.branch_id &&
        e.seq > ev.seq && e.seq < ev.seq + 6);
    return (
      <div>
        {common}
        {started && (
          <div className="panelcard">
            <h4>Approach (why this was tried)</h4>
            <div>{started.payload.approach}</div>
            {started.payload.expectation && (
              <div className="sub" style={{ marginTop: 4 }}>
                Expected: {started.payload.expectation}</div>)}
          </div>
        )}
        <div className="panelcard">
          <h4>Result (engine-verified)</h4>
          <div className="kv">
            <span className="k">valid</span><span className="v">{String(p.valid)}</span>
            <span className="k">score</span><span className="v">{fmtScore(p.score)}</span>
            <span className="k">baseline</span><span className="v">{fmtScore(p.baseline_score)}</span>
            <span className="k">vs baseline</span>
            <span className="v" style={{ color: p.beats_baseline ? "var(--green)" : "var(--muted)" }}>
              {p.improvement_pct != null ? `${p.improvement_pct}%` : "—"}</span>
            <span className="k">improved branch</span><span className="v">{String(p.improved)}</span>
            <span className="k">exec time</span><span className="v">{p.exec_time}s</span>
            {p.error && (<><span className="k">error</span>
              <span className="v" style={{ color: "var(--red)" }}>{p.error}</span></>)}
          </div>
        </div>
        {p.detail && (
          <div className="panelcard">
            <h4>Per-instance results</h4>
            <table className="htable">
              <thead>
                <tr><th>instance</th><th>length</th><th>optimum</th>
                  <th>gap</th><th>time</th></tr>
              </thead>
              <tbody>
                {Object.entries(p.detail).map(([name, d]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>{d.length}</td>
                    <td>{d.optimum}</td>
                    <td>{fmtScore(d.gap_pct)}%</td>
                    <td>{d.time_s}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {critique && (
          <div className="panelcard">
            <h4><AgentBadge name="critic" small /> Critic verdict</h4>
            <div><span className={`verdict ${critique.payload.verdict}`}>
              {critique.payload.verdict}</span> — {critique.payload.analysis}</div>
            {critique.payload.suggestion && (
              <div className="sub" style={{ marginTop: 4 }}>
                Next: {critique.payload.suggestion}</div>)}
          </div>
        )}
        {p.code && (
          <div className="panelcard">
            <h4>Solver code (as executed)</h4>
            <pre className="code">{p.code}</pre>
          </div>
        )}
      </div>
    );
  }

  if (ev.type === "branch.created") {
    const b = p.branch || {};
    return (
      <div>
        {common}
        <div className="panelcard">
          <h4>{b.name}</h4>
          <div className="kv">
            <span className="k">strategy</span><span className="v">{b.strategy}</span>
            {b.parent_ids?.length > 0 && (<><span className="k">parents</span>
              <span className="v">{b.parent_ids.map((id) =>
                state.branches[id]?.name || id).join(" + ")}</span></>)}
          </div>
          <div style={{ marginTop: 8 }}><b>Hypothesis:</b> {b.hypothesis}</div>
          {p.risk && <div className="sub" style={{ marginTop: 4 }}>Risk: {p.risk}</div>}
          {p.merge_reason && (
            <div className="sub" style={{ marginTop: 4 }}>Merge rationale: {p.merge_reason}</div>)}
        </div>
      </div>
    );
  }

  if (ev.type === "branch.collapsed") {
    return (
      <div>
        {common}
        <div className="panelcard">
          <h4>Why it was collapsed</h4>
          <div>{p.reason}</div>
          <div className="sub" style={{ marginTop: 6 }}>
            Final score: {p.final_score ?? "never produced a valid solution"}</div>
        </div>
      </div>
    );
  }

  if (ev.type === "scope.defined")
    return (<div>{common}<ScopePanel state={state} /></div>);

  return (
    <div>
      {common}
      <div className="panelcard">
        <h4>Payload</h4>
        <pre className="code">{JSON.stringify(p, null, 2)}</pre>
      </div>
    </div>
  );
}

// ------------------------------------------------------------ Event feed
export function EventFeed({ events, selectedSeq, onSelect, state }) {
  if (!events.length) return <div className="empty">No events yet.</div>;
  return (
    <div>
      {events.map((ev) => (
        <div key={ev.seq}
          className={`eventrow ${selectedSeq === ev.seq ? "sel" : ""}`}
          onClick={() => onSelect(ev.seq)}>
          <span className="seq">#{ev.seq}</span>
          <span className="type" style={{ color: typeColor(ev.type) }}>{ev.type}</span>
          <span className="who">
            {ev.agent || ""}{ev.branch_id ? ` · ${state.branches[ev.branch_id]?.name || ev.branch_id}` : ""}
          </span>
        </div>
      ))}
    </div>
  );
}

function typeColor(t) {
  if (t.startsWith("experiment")) return "var(--text)";
  if (t === "insight.added") return "var(--purple)";
  if (t === "branch.collapsed" || t === "run.failed") return "var(--red)";
  if (t === "branch.winner" || t === "run.completed") return "var(--gold)";
  if (t === "branch.merged" || t === "branch.created") return "var(--accent)";
  return "var(--muted)";
}
