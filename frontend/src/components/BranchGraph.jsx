// Deterministic git-style lane graph. x = branch lane, y = event order.
// No physics, no overlap: stays readable for any run size.
import React from "react";

const COLORS = {
  scope: "#58a6ff",
  created: "#58a6ff",
  improved: "#3fb950",
  neutral: "#8b96a8",
  failed: "#f85149",
  collapsed: "#f85149",
  winner: "#e3b341",
};

const LANE_W = 130;
const ROW_H = 46;
const PAD_X = 90;
const PAD_Y = 56;

function nodeXY(n) {
  const lane = n.lane < 0 ? -0.6 : n.lane;
  return [PAD_X + lane * LANE_W, PAD_Y + n.row * ROW_H];
}

export default function BranchGraph({ graph, branches, selectedSeq, onSelect }) {
  const { nodes, edges, lanes, branchMeta } = graph;
  if (!nodes.length)
    return <div className="empty">Waiting for the run to produce events…</div>;

  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const numLanes = Math.max(1, Object.keys(lanes).length);
  const width = PAD_X + numLanes * LANE_W + 60;
  const height = PAD_Y + graph.rows * ROW_H + 40;

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      {/* lane guides + labels */}
      {Object.entries(lanes).map(([bid, lane]) => {
        const x = PAD_X + lane * LANE_W;
        const meta = branchMeta[bid] || {};
        const st = branches[bid]?.status;
        return (
          <g key={bid}>
            <line x1={x} y1={PAD_Y - 24} x2={x} y2={height - 20}
              stroke="#1c2330" strokeWidth="1" />
            <text x={x} y={PAD_Y - 34} textAnchor="middle" fill={
              st === "winner" ? COLORS.winner :
              st === "collapsed" ? COLORS.failed : "#8b96a8"}
              fontSize="11" fontFamily="monospace">
              {(meta.name || bid).slice(0, 18)}
            </text>
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
        const color = e.kind === "merge" ? "#bc8cff" : "#2d3646";
        return (
          <path key={i} d={path} fill="none" stroke={color}
            strokeWidth={e.kind === "merge" ? 2 : 1.6}
            strokeDasharray={e.kind === "spawn" ? "4 3" : "none"} />
        );
      })}

      {/* nodes */}
      {nodes.map((n) => {
        const [x, y] = nodeXY(n);
        const color = COLORS[n.kind] || "#8b96a8";
        const sel = selectedSeq === n.seq;
        return (
          <g key={n.id} onClick={() => onSelect(n.seq)} style={{ cursor: "pointer" }}>
            {sel && <circle cx={x} cy={y} r={13} fill="none"
              stroke="#58a6ff" strokeWidth="1.5" opacity="0.8" />}
            {n.kind === "winner" ? (
              <text x={x} y={y + 6} textAnchor="middle" fontSize="18">★</text>
            ) : n.kind === "collapsed" ? (
              <g>
                <circle cx={x} cy={y} r={8} fill="#0d1117" stroke={color} strokeWidth="2" />
                <line x1={x - 4} y1={y - 4} x2={x + 4} y2={y + 4}
                  stroke={color} strokeWidth="2" />
              </g>
            ) : n.kind === "scope" || n.kind === "created" ? (
              <circle cx={x} cy={y} r={7} fill="#0d1117" stroke={color} strokeWidth="2.5" />
            ) : (
              <circle cx={x} cy={y} r={7} fill={color} />
            )}
            {(n.kind === "improved" || n.kind === "neutral") && (
              <text x={x + 13} y={y + 4} fontSize="10.5" fill={
                n.beats ? COLORS.improved : "#8b96a8"} fontFamily="monospace">
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
          </g>
        );
      })}
    </svg>
  );
}

export function GraphLegend() {
  return (
    <div className="legend">
      <span><span className="dot" style={{ border: "2px solid #58a6ff" }} /> branch created</span>
      <span><span className="dot" style={{ background: "#3fb950" }} /> improved</span>
      <span><span className="dot" style={{ background: "#8b96a8" }} /> no gain</span>
      <span><span className="dot" style={{ background: "#f85149" }} /> failed</span>
      <span style={{ color: "#f85149" }}>⊘ collapsed</span>
      <span style={{ color: "#bc8cff" }}>— merge edge</span>
      <span style={{ color: "#e3b341" }}>★ winner</span>
    </div>
  );
}
