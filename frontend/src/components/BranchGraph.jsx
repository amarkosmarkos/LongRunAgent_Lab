// Deterministic git-style lane graph. x = branch lane, y = event order.
// No physics, no overlap: stays readable for any run size.
// Hovering a node shows a tooltip; clicking opens it in the Detail tab.
import React, { useState } from "react";
import { fmtScore } from "../format.js";

const COLORS = {
  scope: "#2667d6",
  created: "#2667d6",
  improved: "#4f8a1d",
  neutral: "#a3a094",
  failed: "#c93b3b",
  collapsed: "#c93b3b",
  winner: "#a8780f",
};
const EDGE = "#d8d4c8";
const MERGE = "#6f63d2";
const GUIDE = "#eeebe1";
const TEXT_MUTED = "#737167";
const TEXT_FAINT = "#a3a094";

const LANE_W = 140;
const ROW_H = 46;
const PAD_X = 96;
const PAD_Y = 60;

function nodeXY(n) {
  const lane = n.lane < 0 ? -0.6 : n.lane;
  return [PAD_X + lane * LANE_W, PAD_Y + n.row * ROW_H];
}

function tooltipFor(n, branches) {
  const b = n.branch_id ? branches[n.branch_id] : null;
  switch (n.kind) {
    case "scope":
      return {
        title: "Scope defined",
        lines: ["The Planner set the objective, baseline and stop conditions.",
          "Click to read the full scope."],
      };
    case "created": {
      const lines = [];
      if (b?.parent_ids?.length)
        lines.push(`Merge of ${b.parent_ids.map((id) => branches[id]?.name || id).join(" + ")}`);
      else if (b?.strategy) lines.push(`Strategy: ${b.strategy}`);
      if (b?.hypothesis) lines.push(b.hypothesis);
      return { title: `Branch: ${b?.name || "?"}`, lines };
    }
    case "improved":
      return {
        title: `Round ${n.round} — score ${fmtScore(n.score)}`,
        lines: [n.beats ? "Beats the baseline" : "Improved this branch (still above baseline)",
          "Click to see the code and the critic's verdict."],
      };
    case "neutral":
      return {
        title: `Round ${n.round} — score ${fmtScore(n.score)}`,
        lines: ["No improvement this round.", "Click to see what was tried."],
      };
    case "failed":
      return {
        title: `Round ${n.round} — experiment failed`,
        lines: ["The solver crashed, timed out or returned an invalid tour.",
          "Click for the error and the code."],
      };
    case "collapsed":
      return {
        title: `Branch collapsed: ${b?.name || ""}`,
        lines: [b?.collapseReason || "The Supervisor abandoned this line of work."],
      };
    case "winner":
      return {
        title: `Winner: ${b?.name || ""}`,
        lines: ["Best verified solution of the run."],
      };
    default:
      return null;
  }
}

export default function BranchGraph({ graph, branches, selectedSeq, onSelect }) {
  const { nodes, edges, lanes, branchMeta } = graph;
  const [tip, setTip] = useState(null); // {x, y, title, lines}
  if (!nodes.length)
    return <div className="empty">Waiting for the run to produce events…</div>;

  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const numLanes = Math.max(1, Object.keys(lanes).length);
  const width = PAD_X + numLanes * LANE_W + 60;
  const height = PAD_Y + graph.rows * ROW_H + 40;

  const hover = (n) => {
    const [x, y] = nodeXY(n);
    const t = tooltipFor(n, branches);
    if (t) setTip({ x: x + 16, y: y + 10, ...t });
  };

  return (
    <div style={{ position: "relative", width, minHeight: height }}>
      <svg width={width} height={height} style={{ display: "block" }}>
        {/* lane guides + labels */}
        {Object.entries(lanes).map(([bid, lane]) => {
          const x = PAD_X + lane * LANE_W;
          const meta = branchMeta[bid] || {};
          const st = branches[bid]?.status;
          const color =
            st === "winner" ? COLORS.winner :
            st === "collapsed" ? COLORS.failed :
            st === "merged" ? MERGE : TEXT_MUTED;
          return (
            <g key={bid}>
              <line x1={x} y1={PAD_Y - 24} x2={x} y2={height - 20}
                stroke={GUIDE} strokeWidth="1" />
              <text x={x} y={PAD_Y - 36} textAnchor="middle" fill={color}
                fontSize="11.5" fontWeight="600">
                {(meta.name || bid).slice(0, 20)}
              </text>
              {st && st !== "active" && (
                <text x={x} y={PAD_Y - 24} textAnchor="middle" fill={color} fontSize="9.5">
                  {st}
                </text>
              )}
            </g>
          );
        })}

        {/* edges */}
        {edges.map((e, i) => {
          const a = byId[e.from], b = byId[e.to];
          if (!a || !b) return null;
          const [x1, y1] = nodeXY(a);
          const [x2, y2] = nodeXY(b);
          const midY = (y1 + y2) / 2;
          const path = x1 === x2
            ? `M ${x1} ${y1} L ${x2} ${y2}`
            : `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`;
          const color = e.kind === "merge" ? MERGE : EDGE;
          return (
            <path key={i} d={path} fill="none" stroke={color}
              strokeWidth={e.kind === "merge" ? 2 : 1.6}
              strokeDasharray={e.kind === "spawn" ? "4 3" : "none"} />
          );
        })}

        {/* nodes */}
        {nodes.map((n) => {
          const [x, y] = nodeXY(n);
          const color = COLORS[n.kind] || TEXT_FAINT;
          const sel = selectedSeq === n.seq;
          const isMerge = n.kind === "created" &&
            branches[n.branch_id]?.parent_ids?.length > 0;
          return (
            <g key={n.id} onClick={() => onSelect(n.seq)}
              onMouseOver={() => hover(n)} onMouseOut={() => setTip(null)}
              style={{ cursor: "pointer" }}>
              {sel && <circle cx={x} cy={y} r={13} fill="none"
                stroke="#2667d6" strokeWidth="1.5" opacity="0.8" />}
              {/* round number gutter for experiment rows */}
              {n.round != null && (
                <text x={26} y={y + 4} fontSize="10" fill={TEXT_FAINT}
                  fontFamily="monospace">r{n.round}</text>
              )}
              {n.kind === "winner" ? (
                <g>
                  <circle cx={x} cy={y} r={11} fill="#faf0da" stroke={color} strokeWidth="2" />
                  <text x={x} y={y + 5} textAnchor="middle" fontSize="13" fill={color}>★</text>
                </g>
              ) : n.kind === "collapsed" ? (
                <g>
                  <circle cx={x} cy={y} r={8} fill="#fbecec" stroke={color} strokeWidth="2" />
                  <line x1={x - 4} y1={y - 4} x2={x + 4} y2={y + 4}
                    stroke={color} strokeWidth="2" />
                </g>
              ) : n.kind === "scope" || n.kind === "created" ? (
                <circle cx={x} cy={y} r={7} fill="#fff"
                  stroke={isMerge ? MERGE : color} strokeWidth="2.5" />
              ) : (
                <circle cx={x} cy={y} r={7} fill={color} />
              )}
              {(n.kind === "improved" || n.kind === "neutral") && (
                <text x={x + 13} y={y + 4} fontSize="10.5" fill={
                  n.beats ? COLORS.improved : TEXT_MUTED} fontFamily="monospace">
                  {n.label}
                </text>
              )}
              {n.kind === "failed" && (
                <text x={x + 13} y={y + 4} fontSize="10.5" fill={COLORS.failed}
                  fontFamily="monospace">✗ {n.label}</text>
              )}
              {n.kind === "scope" && (
                <text x={x + 14} y={y + 4} fontSize="11" fill={color}>Scope defined</text>
              )}
              {isMerge && (
                <text x={x + 13} y={y + 4} fontSize="10" fill={MERGE}>merge</text>
              )}
              {n.kind === "collapsed" && (
                <text x={x + 14} y={y + 4} fontSize="10" fill={color}>collapsed</text>
              )}
            </g>
          );
        })}
      </svg>
      {tip && (
        <div className="gtooltip" style={{ left: tip.x, top: tip.y }}>
          <div className="tt">{tip.title}</div>
          {tip.lines.map((l, i) => <div className="tl" key={i}>{l}</div>)}
        </div>
      )}
    </div>
  );
}

export function GraphLegend() {
  return (
    <div className="legend">
      <span><span className="dot" style={{ border: "2px solid var(--accent)" }} /> branch start</span>
      <span><span className="dot" style={{ background: "var(--green)" }} /> improved</span>
      <span><span className="dot" style={{ background: "var(--faint)" }} /> no gain</span>
      <span><span className="dot" style={{ background: "var(--red)" }} /> failed</span>
      <span style={{ color: "var(--red)" }}>⊘ collapsed</span>
      <span style={{ color: "var(--purple)" }}>— merge</span>
      <span style={{ color: "var(--gold)" }}>★ winner</span>
      <span className="hint">hover a node for details · click to inspect</span>
    </div>
  );
}
