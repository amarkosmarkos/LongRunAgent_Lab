// Renders TSP cities + a tour (baseline gray, best green) on a canvas.
// Default export handles both problem flavors: a single random instance, or a
// TSPLIB benchmark set (adds a per-instance selector).
import React, { useEffect, useRef, useState } from "react";
import { fmtScore } from "../format.js";

// TSPLIB EUC_2D metric: each edge rounded to the nearest integer (matches backend)
function tourLenEuc2d(cities, tour) {
  if (!tour || !tour.length) return null;
  let total = 0;
  for (let i = 0; i < tour.length; i++) {
    const [x1, y1] = cities[tour[i]];
    const [x2, y2] = cities[tour[(i + 1) % tour.length]];
    total += Math.round(Math.hypot(x2 - x1, y2 - y1));
  }
  return total;
}
const gapPct = (len, opt) => Math.round(((len - opt) / opt) * 1000) / 10;

function Canvas({ cities, baselineTour, bestTour, lkhTour, baselineLabel, width, height }) {
  const ref = useRef(null);
  const [show, setShow] = useState("both"); // baseline | best | both

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !cities) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height, M = 18;
    ctx.clearRect(0, 0, W, H);

    const xs = cities.map((p) => p[0]), ys = cities.map((p) => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const sx = (x) => M + ((x - minX) / (maxX - minX || 1)) * (W - 2 * M);
    const sy = (y) => M + ((y - minY) / (maxY - minY || 1)) * (H - 2 * M);

    const drawTour = (tour, color, lw) => {
      if (!tour || !tour.length) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.beginPath();
      tour.forEach((c, i) => {
        const [x, y] = cities[c];
        i === 0 ? ctx.moveTo(sx(x), sy(y)) : ctx.lineTo(sx(x), sy(y));
      });
      const [x0, y0] = cities[tour[0]];
      ctx.lineTo(sx(x0), sy(y0));
      ctx.stroke();
    };

    if (show !== "best") drawTour(baselineTour, "#cfcbbf", 1.3);
    if (lkhTour && show === "both") drawTour(lkhTour, "#a8780f", 1.2);
    if (show !== "baseline") drawTour(bestTour, "#2b8a4f", 1.9);

    ctx.fillStyle = "#44423c";
    cities.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(sx(x), sy(y), 2.4, 0, Math.PI * 2);
      ctx.fill();
    });
  }, [cities, baselineTour, bestTour, lkhTour, show]);

  const blbl = baselineLabel || "baseline";
  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 8, alignItems: "center", flexWrap: "wrap" }}>
        <select value={show} onChange={(e) => setShow(e.target.value)}>
          <option value="both">both tours</option>
          <option value="baseline">starting tour only</option>
          <option value="best">agents' best only</option>
        </select>
        <span className="sub">
          <b style={{ color: "#9b968a" }}>gray</b> = starting tour ({blbl}) ·{" "}
          <b style={{ color: "#2b8a4f" }}>green</b> = best the agents found
          {lkhTour ? <> · <b style={{ color: "#a8780f" }}>gold</b> = LKH (state-of-the-art)</> : null}
        </span>
      </div>
      <canvas ref={ref} width={width} height={height}
        style={{
          background: "#fff", borderRadius: 8,
          border: "1px solid var(--border)", maxWidth: "100%",
        }} />
    </div>
  );
}

export default function TourCanvas({
  instance, baseline, bestSolution, bestScore, width = 460, height = 400,
}) {
  const [sel, setSel] = useState(null);
  if (!instance) return <div className="empty">No instance yet.</div>;

  if (!instance.benchmark) {
    const imp = bestScore != null && baseline?.score
      ? Math.round(((baseline.score - bestScore) / baseline.score) * 1000) / 10
      : null;
    return (
      <div>
        <div className="sub" style={{ marginBottom: 4 }}>
          starting tour {fmtScore(baseline?.score)}
          {bestScore != null ? ` → agents' best ${fmtScore(bestScore)}` : ""}
          {imp != null ? ` (${imp}% shorter)` : ""}
        </div>
        <Canvas cities={instance.cities} baselineTour={baseline?.solution}
          bestTour={bestSolution} baselineLabel={baseline?.algorithm}
          width={width} height={height} />
        <div className="sub" style={{ marginTop: 6, lineHeight: 1.5 }}>
          Only two tours exist here: the <b>starting tour</b> ({baseline?.algorithm}) and
          the <b>best tour the agents found</b>. This is a random instance, so there is
          <b> no known optimum</b> to compare against — for that, run the
          <b> TSPLIB benchmark</b>, where every instance has a proven optimal length.
        </div>
      </div>
    );
  }

  const names = instance.dev || [];
  const name = sel && names.includes(sel) ? sel : names[0];
  const sub = instance.instances?.[name];
  if (!sub) return <div className="empty">No benchmark instances.</div>;

  const baseTour = baseline?.solution?.[name];
  const bestTour = bestSolution?.[name];
  const baseLen = tourLenEuc2d(sub.cities, baseTour);
  const bestLen = tourLenEuc2d(sub.cities, bestTour);
  const baseGap = baseLen != null ? gapPct(baseLen, sub.optimum) : null;
  const bestGap = bestLen != null ? gapPct(bestLen, sub.optimum) : null;
  const lk = instance.lkh?.[name]; // state-of-the-art (LKH) reference
  const reached = bestGap != null && bestGap <= 0.05;
  // how much of the baseline's gap-to-optimum the agents closed
  const gapClosed = baseGap && bestGap != null && baseGap > 0
    ? Math.round(((baseGap - bestGap) / baseGap) * 1000) / 10 : null;

  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 6, alignItems: "center" }}>
        <select value={name} onChange={(e) => setSel(e.target.value)}>
          {names.map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
        <span className="sub">{sub.cities.length} cities</span>
      </div>

      <div className="optboard">
        <div className="optrow">
          <span className="lab gold">optimum (proven best)</span>
          <span className="val">{sub.optimum}</span>
          <span className="gap">0%</span>
        </div>
        {lk && (
          <div className="optrow">
            <span className="lab lkh">LKH · state-of-the-art solver</span>
            <span className="val">{lk.length}</span>
            <span className="gap">{lk.gap_pct === 0 ? "+0% (optimal)" : `+${lk.gap_pct}%`}</span>
          </div>
        )}
        <div className="optrow">
          <span className="lab gray">baseline · nearest-neighbor + 2-opt</span>
          <span className="val">{baseLen ?? "—"}</span>
          <span className="gap">{baseGap != null ? `+${baseGap}%` : "—"}</span>
        </div>
        <div className="optrow">
          <span className="lab green">agents' best</span>
          <span className="val">{bestLen ?? "—"}</span>
          <span className={`gap ${reached ? "good" : ""}`}>
            {bestGap != null ? (reached ? "+0% ✓ optimal" : `+${bestGap}%`) : "—"}
          </span>
        </div>
      </div>
      {gapClosed != null && (
        <div className="sub" style={{ margin: "2px 0 8px" }}>
          The agents closed <b>{gapClosed}%</b> of the gap between the baseline and
          the proven optimum{reached ? " — they reached the optimal tour." : "."}
        </div>
      )}

      <Canvas cities={sub.cities} baselineTour={baseTour} bestTour={bestTour}
        lkhTour={lk?.tour} baselineLabel="nearest-neighbor + 2-opt"
        width={width} height={height} />
      <div className="sub" style={{ marginTop: 6, lineHeight: 1.5 }}>
        The optimum ({sub.optimum}) is a <b>proven length</b>, not a drawn tour.
        {reached
          ? " Since the agents reached it, the green tour above IS an optimal tour."
          : " The agents are scored by how close the green tour gets to it (gap %)."}
      </div>
    </div>
  );
}
