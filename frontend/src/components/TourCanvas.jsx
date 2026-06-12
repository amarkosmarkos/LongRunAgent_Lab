// Renders TSP cities + a tour (baseline gray, best green) on a canvas.
// Default export handles both problem flavors: a single random instance, or a
// TSPLIB benchmark set (adds a per-instance selector).
import React, { useEffect, useRef, useState } from "react";
import { fmtScore } from "../format.js";

function Canvas({ cities, baselineTour, bestTour, width, height }) {
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
    if (show !== "baseline") drawTour(bestTour, "#2b8a4f", 1.9);

    ctx.fillStyle = "#44423c";
    cities.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(sx(x), sy(y), 2.4, 0, Math.PI * 2);
      ctx.fill();
    });
  }, [cities, baselineTour, bestTour, show]);

  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 8, alignItems: "center" }}>
        <select value={show} onChange={(e) => setShow(e.target.value)}>
          <option value="both">baseline + best</option>
          <option value="baseline">baseline only</option>
          <option value="best">best only</option>
        </select>
        <span className="sub">gray = baseline tour · green = best tour</span>
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
    return (
      <div>
        <div className="sub" style={{ marginBottom: 4 }}>
          baseline score {fmtScore(baseline?.score)}
          {bestScore != null ? ` · best ${fmtScore(bestScore)}` : ""}
        </div>
        <Canvas cities={instance.cities} baselineTour={baseline?.solution}
          bestTour={bestSolution} width={width} height={height} />
      </div>
    );
  }

  const names = instance.dev || [];
  const name = sel && names.includes(sel) ? sel : names[0];
  const sub = instance.instances?.[name];
  if (!sub) return <div className="empty">No benchmark instances.</div>;
  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 6, alignItems: "center" }}>
        <select value={name} onChange={(e) => setSel(e.target.value)}>
          {names.map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
        <span className="sub">
          {sub.cities.length} cities · known optimum {sub.optimum}
        </span>
      </div>
      <Canvas cities={sub.cities} baselineTour={baseline?.solution?.[name]}
        bestTour={bestSolution?.[name]} width={width} height={height} />
    </div>
  );
}
