// Renders TSP cities + a tour (baseline gray, best green) on a canvas.
import React, { useEffect, useRef, useState } from "react";
import { fmtScore } from "../format.js";

export default function TourCanvas({
  instance, baseline, bestSolution, bestScore, width = 460, height = 400,
}) {
  const ref = useRef(null);
  const [show, setShow] = useState("both"); // baseline | best | both

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !instance) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height, M = 18;
    ctx.clearRect(0, 0, W, H);

    const pts = instance.cities;
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const sx = (x) => M + ((x - minX) / (maxX - minX || 1)) * (W - 2 * M);
    const sy = (y) => M + ((y - minY) / (maxY - minY || 1)) * (H - 2 * M);

    const drawTour = (tour, color, width) => {
      if (!tour || !tour.length) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      tour.forEach((c, i) => {
        const [x, y] = pts[c];
        i === 0 ? ctx.moveTo(sx(x), sy(y)) : ctx.lineTo(sx(x), sy(y));
      });
      const [x0, y0] = pts[tour[0]];
      ctx.lineTo(sx(x0), sy(y0));
      ctx.stroke();
    };

    if (show !== "best") drawTour(baseline?.solution, "#cfcbbf", 1.3);
    if (show !== "baseline") drawTour(bestSolution, "#2b8a4f", 1.9);

    ctx.fillStyle = "#44423c";
    pts.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(sx(x), sy(y), 2.4, 0, Math.PI * 2);
      ctx.fill();
    });
  }, [instance, baseline, bestSolution, show]);

  if (!instance) return <div className="empty">No instance yet.</div>;

  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 8, alignItems: "center" }}>
        <select value={show} onChange={(e) => setShow(e.target.value)}>
          <option value="both">baseline + best</option>
          <option value="baseline">baseline only</option>
          <option value="best">best only</option>
        </select>
        <span className="sub">
          gray = baseline ({fmtScore(baseline?.score)}) · green = best{bestScore != null ? ` (${fmtScore(bestScore)})` : ""}
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
