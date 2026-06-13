import React, { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api.js";
import { fmtScore } from "./format.js";
import { buildGraph, reduceEvents } from "./replay.js";
import BranchGraph, { GraphLegend } from "./components/BranchGraph.jsx";
import TourCanvas from "./components/TourCanvas.jsx";
import { agentMeta } from "./agents.js";
import {
  BranchesPanel, CostPanel, DetailPanel, EventFeed, KnowledgePanel,
  ResultsPanel, ScopePanel, StoryPanel,
} from "./components/Panels.jsx";

const TERMINAL = ["completed", "failed", "stopped", "budget_exceeded"];

function phaseLabel(state, status) {
  if (TERMINAL.includes(status)) return "4 · concluded";
  if (!state.scope) return "1 · scoping";
  if (!state.branchOrder.length) return "2 · branching";
  return "3 · experimenting";
}

export default function RunView({ runId, onBack }) {
  const [events, setEvents] = useState([]);
  const [status, setStatus] = useState("…");
  const [cursor, setCursor] = useState(null); // null = live (all events)
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(4);
  const [tab, setTab] = useState("story");
  const [selectedSeq, setSelectedSeq] = useState(null);
  const [mapOpen, setMapOpen] = useState(true);
  const eventsRef = useRef(events);
  eventsRef.current = events;

  // ---- data: SSE with polling fallback
  useEffect(() => {
    let closed = false;
    let cleanup = () => {};
    const start = async () => {
      const init = await api.getEvents(runId, 0);
      if (closed) return;
      setEvents(init.events);
      setStatus(init.status);
      if (TERMINAL.includes(init.status)) return;
      cleanup = api.stream(
        runId, init.events.length,
        (ev) => setEvents((prev) =>
          ev.seq >= prev.length ? [...prev, ev] : prev),
        (endStatus) => {
          if (endStatus) setStatus(endStatus);
          else if (!closed) poll();
        });
    };
    const poll = () => {
      const t = setInterval(async () => {
        try {
          const r = await api.getEvents(runId, eventsRef.current.length);
          if (r.events.length)
            setEvents((prev) => [...prev, ...r.events]);
          setStatus(r.status);
          if (TERMINAL.includes(r.status)) clearInterval(t);
        } catch { /* keep trying */ }
      }, 1000);
      cleanup = () => clearInterval(t);
    };
    start();
    return () => { closed = true; cleanup(); };
  }, [runId]);

  // refresh status while running
  useEffect(() => {
    if (TERMINAL.includes(status)) return;
    const t = setInterval(async () => {
      try { setStatus((await api.getRun(runId)).status); } catch { /* ignore */ }
    }, 1500);
    return () => clearInterval(t);
  }, [runId, status]);

  // ---- replay playback
  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => {
      setCursor((c) => {
        const next = (c == null ? 0 : c) + 1;
        if (next >= eventsRef.current.length) {
          setPlaying(false);
          return null; // reached live
        }
        return next;
      });
    }, 1000 / speed);
    return () => clearInterval(t);
  }, [playing, speed]);

  const state = useMemo(() => reduceEvents(events, cursor), [events, cursor]);
  const graph = useMemo(() => buildGraph(events, cursor), [events, cursor]);
  const visibleEvents = useMemo(
    () => (cursor == null ? events : events.slice(0, cursor)),
    [events, cursor]);

  const imp = state.bestScore != null && state.baseline
    ? Math.round(((state.baseline.score - state.bestScore) / state.baseline.score) * 1000) / 10
    : null;
  const budget = state.config?.budget_usd || 0;
  const budgetPct = budget ? Math.min(100, (state.costs.total / budget) * 100) : 0;

  const select = (seq) => { setSelectedSeq(seq); setTab("detail"); };
  const live = cursor == null;
  const running = !TERMINAL.includes(status);
  const thinking = Object.values(state.activity || {});

  return (
    <div className="runview">
      <div className="runhead">
        <button onClick={onBack}>← Runs</button>
        <span className="chip" style={{ fontFamily: "monospace" }}>{runId}</span>
        <span className={`chip ${status}`}>{status}</span>
        <span className="chip phase">{phaseLabel(state, status)}</span>
        {state.mockMode != null && (
          <span className={`chip llmtag ${state.mockMode ? "mock" : "live"}`}
            title={state.mockMode
              ? "Scripted mock LLM — solver code still really runs and is verified"
              : `Real Claude API calls: ${state.models.join(", ")}`}>
            {state.mockMode
              ? "◌ mock LLM (demo)"
              : `● real agents · ${state.models.map((m) => m.replace("claude-", "").replace(/-\d+$/, "")).join(" + ")}`}
          </span>
        )}
        <div className="stat"><span className="k">baseline</span>
          <span className="v">{fmtScore(state.baseline?.score)}</span></div>
        <div className="stat"><span className="k">best</span>
          <span className="v good">{fmtScore(state.bestScore)}</span></div>
        <div className="stat"><span className="k">improvement</span>
          <span className="v gold">{imp != null ? `${imp}%` : "—"}</span></div>
        <div className="stat"><span className="k">cost</span>
          <span className="v">${state.costs.total.toFixed(4)}</span></div>
        <div className="stat"><span className="k">budget {budget ? `$${budget}` : ""}</span>
          <span className="budgetbar"><div className={budgetPct > 80 ? "hot" : ""}
            style={{ width: `${budgetPct}%` }} /></span></div>
        {!TERMINAL.includes(status) && (
          <button className="danger" style={{ marginLeft: "auto" }}
            onClick={() => api.stopRun(runId)}>■ Stop run</button>
        )}
      </div>

      <div className="replaybar">
        <button onClick={() => { setCursor(0); setPlaying(false); }}>⏮</button>
        <button className="primary" onClick={() => {
          if (playing) { setPlaying(false); }
          else { if (cursor == null) setCursor(0); setPlaying(true); }
        }}>{playing ? "⏸ Pause" : "▶ Replay"}</button>
        <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
          <option value={1}>1×</option><option value={2}>2×</option>
          <option value={4}>4×</option><option value={10}>10×</option>
        </select>
        <input type="range" min={0} max={events.length}
          value={cursor == null ? events.length : cursor}
          onChange={(e) => {
            const v = Number(e.target.value);
            setPlaying(false);
            setCursor(v >= events.length ? null : v);
          }} />
        <span className="cursorinfo">
          {live ? `live · ${events.length} events` : `event ${cursor}/${events.length}`}
        </span>
        {!live && <button onClick={() => { setCursor(null); setPlaying(false); }}>Go live</button>}
      </div>

      {live && running && (
        <div className="activitybar">
          {thinking.length ? thinking.map((g) => {
            const m = agentMeta(g.agent);
            return (
              <span className="thinking" key={g.agent + (g.branch_id || "")}>
                <span className="avatar pulse" style={{ background: m.color }}>{m.initials}</span>
                <span className="who" style={{ color: m.color }}>{m.label}</span>
                <span className="act">{g.action}
                  {g.branch_id && state.branches[g.branch_id]
                    ? ` · ${state.branches[g.branch_id].name}` : ""}
                  <span className="dots"><i/><i/><i/></span>
                </span>
              </span>
            );
          }) : (
            <span className="thinking idle">
              <span className="avatar pulse" style={{ background: "var(--faint)" }}>··</span>
              <span className="act">agents are between steps — preparing the next move…</span>
            </span>
          )}
        </div>
      )}

      <div className="runbody">
        <div className="graphcol">
          <GraphLegend />
          <div className="graphwrap">
            <BranchGraph graph={graph} branches={state.branches}
              activity={live ? state.activity : {}}
              selectedSeq={selectedSeq} onSelect={select} />
          </div>
          {state.instance && (
            <div className="mappanel">
              <div className="maphead">
                <span>Tour map</span>
                <span className="sub">updates live as experiments improve the tour</span>
                <button className="link" onClick={() => setMapOpen((o) => !o)}>
                  {mapOpen ? "hide ▾" : "show ▸"}
                </button>
              </div>
              {mapOpen && (
                <TourCanvas instance={state.instance} baseline={state.baseline}
                  bestSolution={state.results?.best_solution || state.bestSolution}
                  bestScore={state.bestScore} width={520} height={230} />
              )}
            </div>
          )}
        </div>
        <div className="sidepanel">
          <div className="tabs">
            {[["story", "Story"], ["branches", "Branches"], ["detail", "Detail"],
              ["scope", "Scope"], ["knowledge", "Knowledge"], ["costs", "Costs"],
              ["results", "Results"], ["events", "Events"]]
              .map(([k, label]) => (
                <button key={k} className={tab === k ? "active" : ""}
                  onClick={() => setTab(k)}>
                  {label}
                  {k === "knowledge" && state.insights.length > 0 &&
                    ` (${state.insights.length})`}
                  {k === "branches" && state.branchOrder.length > 0 &&
                    ` (${state.branchOrder.length})`}
                </button>
              ))}
          </div>
          <div className="tabbody">
            {tab === "story" && (
              <StoryPanel events={visibleEvents} state={state}
                selectedSeq={selectedSeq} onSelect={select} />)}
            {tab === "branches" && <BranchesPanel state={state} onSelect={select} />}
            {tab === "detail" && (
              <DetailPanel state={state} events={visibleEvents}
                selectedSeq={selectedSeq} onSelect={select} />)}
            {tab === "scope" && <ScopePanel state={state} />}
            {tab === "knowledge" && <KnowledgePanel state={state} />}
            {tab === "costs" && <CostPanel state={state} />}
            {tab === "results" && <ResultsPanel state={state} />}
            {tab === "events" && (
              <EventFeed events={visibleEvents} selectedSeq={selectedSeq}
                onSelect={select} state={state} />)}
          </div>
        </div>
      </div>
    </div>
  );
}
