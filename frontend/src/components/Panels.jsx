import React from "react";
import TourCanvas from "./TourCanvas.jsx";

const usd = (v) => `$${(v || 0).toFixed(4)}`;

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
          <span className="k">score</span><span className="v">{baseline?.score}</span>
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
        <h4>Planner reasoning</h4>
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
      <div className="sub" style={{ color: "#8b96a8", marginBottom: 10 }}>
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
function Bars({ entries, max }) {
  return entries.map(([name, v]) => (
    <div className="costrow" key={name}>
      <span className="name">{name}</span>
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
        {agents.length ? <Bars entries={agents} max={maxA} /> : <span className="sub">—</span>}
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

// -------------------------------------------------------------- Results
export function ResultsPanel({ state }) {
  const { results, baseline, instance, bestSolution, bestScore, branches } = state;
  const imp = results?.improvement_pct ??
    (bestScore != null && baseline
      ? Math.round(((baseline.score - bestScore) / baseline.score) * 1000) / 10
      : null);
  return (
    <div>
      <div className="resultbig">
        <div className="stat">
          <span className="k">baseline</span>
          <span className="v">{baseline?.score ?? "—"}</span>
        </div>
        <div className="stat">
          <span className="k">best found</span>
          <span className="v good">{bestScore ?? "—"}</span>
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
            <span className="v" style={{ color: results.target_met ? "#3fb950" : "#f85149" }}>
              {results.target_met ? "yes" : "no"}
            </span>
            <span className="k">re-verified</span>
            <span className="v" style={{ color: results.verified ? "#3fb950" : "#f85149" }}>
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
      <TourCanvas instance={instance} baseline={baseline}
        bestSolution={results?.best_solution || bestSolution} bestScore={bestScore} />
    </div>
  );
}

// ---------------------------------------------------------- Node detail
export function DetailPanel({ state, events, selectedSeq }) {
  if (selectedSeq == null)
    return <div className="empty">Select a node in the graph or an event in the feed.</div>;
  const ev = events.find((e) => e.seq === selectedSeq);
  if (!ev) return <div className="empty">Event not yet loaded.</div>;
  const p = ev.payload || {};
  const branch = ev.branch_id ? state.branches[ev.branch_id] : null;

  const common = (
    <div className="panelcard">
      <h4>{ev.type}</h4>
      <div className="kv">
        <span className="k">seq</span><span className="v">#{ev.seq}</span>
        {ev.agent && (<><span className="k">agent</span><span className="v">{ev.agent}</span></>)}
        {branch && (<><span className="k">branch</span><span className="v">{branch.name}</span></>)}
      </div>
    </div>
  );

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
            <span className="k">score</span><span className="v">{p.score ?? "—"}</span>
            <span className="k">baseline</span><span className="v">{p.baseline_score}</span>
            <span className="k">vs baseline</span>
            <span className="v" style={{ color: p.beats_baseline ? "#3fb950" : "#8b96a8" }}>
              {p.improvement_pct != null ? `${p.improvement_pct}%` : "—"}</span>
            <span className="k">improved branch</span><span className="v">{String(p.improved)}</span>
            <span className="k">exec time</span><span className="v">{p.exec_time}s</span>
            {p.error && (<><span className="k">error</span>
              <span className="v" style={{ color: "#f85149" }}>{p.error}</span></>)}
          </div>
        </div>
        {critique && (
          <div className="panelcard">
            <h4>Critic verdict</h4>
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
  if (t.startsWith("experiment")) return "#dde3ec";
  if (t === "insight.added") return "#bc8cff";
  if (t === "branch.collapsed" || t === "run.failed") return "#f85149";
  if (t === "branch.winner" || t === "run.completed") return "#e3b341";
  if (t === "branch.merged" || t === "branch.created") return "#58a6ff";
  if (t === "llm.called") return "#8b96a8";
  return "#8b96a8";
}
