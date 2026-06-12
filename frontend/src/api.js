const json = (r) => {
  if (!r.ok) throw new Error(`API ${r.status}`);
  return r.json();
};

export const api = {
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
