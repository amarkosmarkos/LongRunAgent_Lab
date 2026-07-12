const json = (r) => {
  if (!r.ok) throw new Error(`API ${r.status}`);
  return r.json();
};

// Demo mode: the site is published static (e.g. GitHub Pages) with no backend.
// Real runs were frozen into public/demo/*.json by app.scripts.export_demo, and
// we serve those instead of hitting /api. Everything read-only still works —
// runs list, branch graph, story, replay, originality, lab memory — while the
// write paths (start/stop) are disabled.
const DEMO = import.meta.env.VITE_DEMO === "1";
const DEMO_BASE = `${import.meta.env.BASE_URL}demo`;

const demoRuns = () => fetch(`${DEMO_BASE}/runs.json`).then(json);
const demoEvents = (id) => fetch(`${DEMO_BASE}/${id}.json`).then(json);

const demoApi = {
  health: () => Promise.resolve({ ok: true, mock_mode: true, demo: true }),
  problems: () => Promise.resolve({ problems: [], tsplib: null }),
  listRuns: () => demoRuns(),
  createRun: () => Promise.reject(new Error("demo mode: starting runs is disabled")),
  getRun: (id) =>
    demoRuns().then((runs) => {
      const r = runs.find((x) => x.id === id);
      if (!r) throw new Error(`run ${id} not found`);
      return r;
    }),
  getEvents: (id, since = 0) =>
    demoEvents(id).then((d) => ({ status: d.status, events: d.events.slice(since) })),
  stopRun: () => Promise.resolve({ ok: true }),
  // frozen runs are terminal, so there is nothing to stream — end immediately
  stream: (id, since, onEvent, onEnd) => {
    demoEvents(id).then((d) => onEnd?.(d.status)).catch(() => onEnd?.(null));
    return () => {};
  },
};

const liveApi = {
  health: () => fetch("/api/health").then(json),
  problems: () => fetch("/api/problems").then(json),
  listRuns: () => fetch("/api/runs").then(json),
  createRun: (config) =>
    fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    }).then(json),
  getRun: (id) => fetch(`/api/runs/${id}`).then(json),
  getEvents: (id, since = 0) =>
    fetch(`/api/runs/${id}/events?since=${since}`).then(json),
  stopRun: (id) => fetch(`/api/runs/${id}/stop`, { method: "POST" }).then(json),
  stream: (id, since, onEvent, onEnd) => {
    const es = new EventSource(`/api/runs/${id}/stream?since=${since}`);
    es.onmessage = (m) => {
      const ev = JSON.parse(m.data);
      if (ev.type === "stream.end") {
        es.close();
        onEnd?.(ev.status);
      } else {
        onEvent(ev);
      }
    };
    es.onerror = () => {
      es.close();
      onEnd?.(null); // caller may re-poll
    };
    return () => es.close();
  },
};

export const IS_DEMO = DEMO;
export const api = DEMO ? demoApi : liveApi;
