import React, { useEffect, useState } from "react";
import { api, IS_DEMO } from "./api.js";
import RunView from "./RunView.jsx";

export default function App() {
  const [view, setView] = useState({ name: "list" });
  const [runs, setRuns] = useState([]);
  const [health, setHealth] = useState(null);
  const [meta, setMeta] = useState(null); // /api/problems
  const [cfg, setCfg] = useState({
    kernel_problem: "grayscale_py", backend: "local", gpu: "T4",
    num_hypotheses: 4, max_rounds: 5, budget_usd: 2.0,
  });
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    api.problems().then(setMeta).catch(() => {});
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
        problem: "gpu_kernel",
        problem_params: {
          kernel_problem: cfg.kernel_problem,
          backend: cfg.backend,
          gpu: cfg.gpu,
        },
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

  const kernels = meta?.kernels || [];
  const localInfo = meta?.backends?.local;
  const modalReady = meta?.backends?.modal?.available;

  return (
    <>
      <div className="topbar">
        <h1><span className="flask">⚗</span> Long Run Agent Lab</h1>
        {health && (
          <span className={`chip llmtag ${health.mock_mode ? "mock" : "live"}`}
            title={health.mock_mode
              ? "No ANTHROPIC_API_KEY set — new runs use a scripted mock LLM (solver code still really runs). Add the key to backend/.env for real agents."
              : "ANTHROPIC_API_KEY detected — new runs call the real Claude API."}>
            {health.mock_mode ? "◌ mock mode (no API key)" : "● live mode · real Claude agents"}
          </span>
        )}
        {health === null && !IS_DEMO && <span className="chip failed">backend offline</span>}
        {IS_DEMO && (
          <span className="chip llmtag mock"
            title="Static demo — real frozen runs replayed with no backend. Clone the repo to run live experiments.">
            ◍ demo · frozen runs (read-only)
          </span>
        )}
      </div>

      {view.name === "run" ? (
        <RunView runId={view.id} onBack={() => setView({ name: "list" })} />
      ) : (
        <div className="runlist">
          <div className="newrun">
            <h2 style={{ margin: "0 0 4px" }}>
              {IS_DEMO ? "Explore the frozen runs below" : "New experiment run"}
            </h2>
            <div className="sub">
              {IS_DEMO
                ? "This is a static demo: real runs are replayed from frozen event logs with no backend. Pick a run below to explore its branch graph, story, replay and cross-run lab memory. To launch live experiments, clone the repo and run the backend."
                : "GPU MODE reference-kernels: agents write submission.py and are scored by the official harness — correctness first, then runtime. The winner is re-timed on held-out benchmark shapes."}
            </div>
            {!IS_DEMO && (<>
            <div className="fields">
              <label>kernel
                <select value={cfg.kernel_problem}
                  onChange={(e) => setCfg({ ...cfg, kernel_problem: e.target.value })}>
                  {(kernels.length ? kernels.map((k) => k.name)
                    : ["grayscale_py"]).map((n) => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
              </label>
              <label>eval backend
                <select value={cfg.backend}
                  onChange={(e) => setCfg({ ...cfg, backend: e.target.value })}>
                  <option value="local">local ({localInfo?.gpu || "cpu"})</option>
                  <option value="modal" disabled={!modalReady}>
                    Modal GPU{modalReady ? "" : " (not installed)"}
                  </option>
                </select>
              </label>
              {cfg.backend === "modal" && (
                <label>GPU
                  <select value={cfg.gpu}
                    onChange={(e) => setCfg({ ...cfg, gpu: e.target.value })}>
                    {(meta?.backends?.modal?.gpus || ["T4"]).map((g) => (
                      <option key={g} value={g}>{g}</option>
                    ))}
                  </select>
                </label>
              )}
              {field("num_hypotheses", "hypotheses")}
              {field("max_rounds", "max rounds")}
              {field("budget_usd", "budget (USD)", 0.5)}
            </div>
            {cfg.backend === "local" && localInfo && !localInfo.timings_are_gpu && (
              <div className="sub" style={{ marginBottom: 10, color: "var(--amber)" }}>
                ⚠ No CUDA device here, so the local backend runs the official
                harness on CPU: correctness is real, but the timings are CPU
                timings and are not a GPU benchmark. Use the Modal backend for
                numbers that mean something.
              </div>
            )}
            <button className="primary" onClick={startRun} disabled={starting || !health}>
              {starting ? "Starting…" : "▶ Start run"}
            </button>
            </>)}
          </div>

          <h2>Runs</h2>
          {runs.length === 0 && <div className="empty">No runs yet. Start one above.</div>}
          {runs.map((r) => (
            <div className="runrow" key={r.id}
              onClick={() => setView({ name: "run", id: r.id })}>
              <span className="rid">{r.id}</span>
              <span className={`chip ${r.status}`}>{r.status}</span>
              <span className="meta">
                {r.config?.problem_params?.kernel_problem || r.config?.problem}
                {r.config?.problem_params?.backend
                  ? ` · ${r.config.problem_params.backend === "modal"
                      ? r.config.problem_params.gpu : "local"}`
                  : ""}
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
