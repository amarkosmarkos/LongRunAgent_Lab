import React, { useEffect, useState } from "react";
import { api } from "./api.js";
import RunView from "./RunView.jsx";

const splitNames = (s) =>
  s.split(",").map((x) => x.trim()).filter(Boolean);

export default function App() {
  const [view, setView] = useState({ name: "list" });
  const [runs, setRuns] = useState([]);
  const [health, setHealth] = useState(null);
  const [meta, setMeta] = useState(null); // /api/problems
  const [cfg, setCfg] = useState({
    problem: "tsp", n_cities: 60, seed: 42,
    num_hypotheses: 4, max_rounds: 5, budget_usd: 2.0,
    dev: "", holdout: "",
  });
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    api.problems().then((m) => {
      setMeta(m);
      if (m.tsplib) {
        setCfg((c) => ({
          ...c,
          dev: m.tsplib.default_dev.join(", "),
          holdout: m.tsplib.default_holdout.join(", "),
        }));
      }
    }).catch(() => {});
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
      const problem_params = cfg.problem === "tsp_benchmark"
        ? { dev: splitNames(cfg.dev), holdout: splitNames(cfg.holdout) }
        : { n_cities: Number(cfg.n_cities), seed: Number(cfg.seed) };
      const run = await api.createRun({
        problem: cfg.problem,
        problem_params,
        num_hypotheses: Number(cfg.num_hypotheses),
        max_rounds: Number(cfg.max_rounds),
        budget_usd: Number(cfg.budget_usd),
      });
      setView({ name: "run", id: run.id });
    } finally {
      setStarting(false);
    }
  };

  const field = (key, label, step, width) => (
    <label key={key}>{label}
      <input type="number" step={step || 1} value={cfg[key]}
        style={width ? { width } : undefined}
        onChange={(e) => setCfg({ ...cfg, [key]: e.target.value })} />
    </label>
  );
  const textField = (key, label) => (
    <label key={key} style={{ flex: 1, minWidth: 260 }}>{label}
      <input type="text" value={cfg[key]}
        onChange={(e) => setCfg({ ...cfg, [key]: e.target.value })} />
    </label>
  );

  const isBench = cfg.problem === "tsp_benchmark";
  const catalog = meta?.tsplib?.catalog || [];

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
            <div className="sub">
              {isBench
                ? "TSPLIB95 benchmark with known optima — agents must beat a nearest-neighbor + 2-opt baseline; the winner is re-tested on held-out instances."
                : "Random Euclidean TSP — agents must beat the nearest-neighbor baseline."}
            </div>
            <div className="fields">
              <label>problem
                <select value={cfg.problem}
                  onChange={(e) => setCfg({ ...cfg, problem: e.target.value })}>
                  <option value="tsp">Random Euclidean</option>
                  <option value="tsp_benchmark">TSPLIB benchmark</option>
                </select>
              </label>
              {!isBench && field("n_cities", "cities")}
              {!isBench && field("seed", "seed")}
              {field("num_hypotheses", "hypotheses")}
              {field("max_rounds", "max rounds")}
              {field("budget_usd", "budget (USD)", 0.5)}
            </div>
            {isBench && (
              <div className="fields">
                {textField("dev", "dev instances (agents iterate on these)")}
                {textField("holdout", "held-out instances (final verification only)")}
              </div>
            )}
            {isBench && catalog.length > 0 && (
              <div className="sub" style={{ marginBottom: 10 }}>
                available: {catalog.map((c) => `${c.name} (${c.n_cities})`).join(", ")}
              </div>
            )}
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
                {r.config?.problem}
                {r.config?.problem === "tsp_benchmark"
                  ? ` · ${(r.config?.problem_params?.dev || []).length} dev + ${(r.config?.problem_params?.holdout || []).length} held-out`
                  : ` · ${r.config?.problem_params?.n_cities} cities`}
                {" "}· {r.num_events} events ·
                {" "}{new Date(r.created_at * 1000).toLocaleString()}
              </span>
            </div>
          ))}
        </div>
      )}
    </>
  );
}
