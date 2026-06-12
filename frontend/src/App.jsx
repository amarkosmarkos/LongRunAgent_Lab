import React, { useEffect, useState } from "react";
import { api } from "./api.js";
import RunView from "./RunView.jsx";

export default function App() {
  const [view, setView] = useState({ name: "list" });
  const [runs, setRuns] = useState([]);
  const [health, setHealth] = useState(null);
  const [cfg, setCfg] = useState({
    n_cities: 60, seed: 42, num_hypotheses: 4, max_rounds: 5, budget_usd: 2.0,
  });
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    if (view.name !== "list") return;
    const load = () => api.listRuns().then(setRuns).catch(() => {});
    load();
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, [view]);

  const startRun = async () => {
    setStarting(true);
    try {
      const run = await api.createRun({
        problem: "tsp",
        problem_params: { n_cities: Number(cfg.n_cities), seed: Number(cfg.seed) },
        num_hypotheses: Number(cfg.num_hypotheses),
        max_rounds: Number(cfg.max_rounds),
        budget_usd: Number(cfg.budget_usd),
      });
      setView({ name: "run", id: run.id });
    } finally {
      setStarting(false);
    }
  };

  const field = (key, label, step) => (
    <label key={key}>{label}
      <input type="number" step={step || 1} value={cfg[key]}
        onChange={(e) => setCfg({ ...cfg, [key]: e.target.value })} />
    </label>
  );

  return (
    <>
      <div className="topbar">
        <h1><span className="flask">⚗</span> Long Run Agent Lab</h1>
        {health && (
          <span className="chip" title="Set ANTHROPIC_API_KEY in backend/.env for real agents">
            {health.mock_mode ? "mock agents (no API key)" : "live Claude agents"}
          </span>
        )}
        {health === null && <span className="chip failed">backend offline</span>}
      </div>

      {view.name === "run" ? (
        <RunView runId={view.id} onBack={() => setView({ name: "list" })} />
      ) : (
        <div className="runlist">
          <div className="newrun">
            <h2 style={{ margin: "0 0 4px" }}>New experiment run</h2>
            <div className="sub" style={{ color: "#8b96a8" }}>
              Travelling Salesman Problem — agents must beat the nearest-neighbor baseline.
            </div>
            <div className="fields">
              {field("n_cities", "cities")}
              {field("seed", "seed")}
              {field("num_hypotheses", "hypotheses")}
              {field("max_rounds", "max rounds")}
              {field("budget_usd", "budget (USD)", 0.5)}
            </div>
            <button className="primary" onClick={startRun} disabled={starting || !health}>
              {starting ? "Starting…" : "▶ Start run"}
            </button>
          </div>

          <h2>Runs</h2>
          {runs.length === 0 && <div className="empty">No runs yet. Start one above.</div>}
          {runs.map((r) => (
            <div className="runrow" key={r.id}
              onClick={() => setView({ name: "run", id: r.id })}>
              <span className="rid">{r.id}</span>
              <span className={`chip ${r.status}`}>{r.status}</span>
              <span className="meta">
                {r.config?.problem} · {r.config?.problem_params?.n_cities} cities ·
                {" "}{r.num_events} events ·
                {" "}{new Date(r.created_at * 1000).toLocaleString()}
              </span>
            </div>
          ))}
        </div>
      )}
    </>
  );
}
