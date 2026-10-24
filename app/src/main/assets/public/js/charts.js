/*
 * charts.js — Gráficos de línea mínimos sobre <canvas> (sin dependencias,
 * para que la app funcione sin internet). Pensados para pantalla de celular:
 * nitidez en pantallas de alta densidad y colores tomados del tema CSS.
 */
(function (global) {
  "use strict";

  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  /** "Bonitos" valores de marca para un eje entre a y b. */
  function ticks(a, b, n = 4) {
    const span = b - a || 1, step0 = span / n;
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  function fmt(v) {
    const a = Math.abs(v);
    if (a >= 1000) return (v / 1000).toFixed(a >= 10000 ? 0 : 1) + "k";
    if (a >= 10 || v === 0) return String(Math.round(v));
    return v.toFixed(a >= 1 ? 1 : 2);
  }

  /**
   * Dibuja series (x, y) en el canvas.
   * opts: { series:[{x,y,color,width,dash,minmax}], xMin,xMax,yMin,yMax,
   *         xLabel, vlines:[{x,label,color}], bands:[{x0,x1,color}],
   *         hlines:[{y,label}] }
   * `minmax: true` reduce la serie a mín/máx por píxel (formas de onda).
   */
  function lineChart(canvas, opts) {
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth, H = canvas.clientHeight;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);

    const pad = { l: 34, r: 10, t: 12, b: opts.xLabel ? 32 : 20 };
    const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    const { xMin, xMax, yMin, yMax } = opts;
    const X = (v) => pad.l + ((v - xMin) / (xMax - xMin)) * pw;
    const Y = (v) => pad.t + (1 - (v - yMin) / (yMax - yMin)) * ph;
    const muted = css("--muted"), grid = css("--grid");

    // Rejilla y ejes
    g.font = "10px system-ui, sans-serif";
    g.lineWidth = 1;
    g.fillStyle = muted;
    g.textAlign = "right"; g.textBaseline = "middle";
    for (const v of ticks(yMin, yMax, 3)) {
      g.strokeStyle = grid; g.beginPath(); g.moveTo(pad.l, Y(v)); g.lineTo(W - pad.r, Y(v)); g.stroke();
      g.fillText(fmt(v), pad.l - 5, Y(v));
    }
    g.textAlign = "center"; g.textBaseline = "top";
    for (const v of ticks(xMin, xMax, Math.max(2, Math.floor(pw / 70)))) {
      g.strokeStyle = grid; g.beginPath(); g.moveTo(X(v), pad.t); g.lineTo(X(v), pad.t + ph); g.stroke();
      g.fillText(fmt(v), X(v), pad.t + ph + 4);
    }
    if (opts.xLabel) { g.fillText(opts.xLabel, pad.l + pw / 2, H - 12); }

    g.save();
    g.beginPath(); g.rect(pad.l, pad.t, pw, ph); g.clip();

    for (const b of opts.bands || []) {
      g.fillStyle = b.color; g.fillRect(X(b.x0), pad.t, X(b.x1) - X(b.x0), ph);
    }
    for (const h of opts.hlines || []) {
      g.strokeStyle = muted; g.setLineDash([4, 4]);
      g.beginPath(); g.moveTo(pad.l, Y(h.y)); g.lineTo(W - pad.r, Y(h.y)); g.stroke();
      g.setLineDash([]);
    }

    for (const s of opts.series) {
      g.strokeStyle = s.color; g.lineWidth = s.width || 1.5;
      g.setLineDash(s.dash || []);
      g.lineJoin = "round";
      g.beginPath();
      if (s.minmax) {
        // Una barra vertical mín–máx por columna de píxel: fiel y rápido.
        const cols = Math.max(1, Math.floor(pw));
        const lo = new Float64Array(cols).fill(Infinity), hi = new Float64Array(cols).fill(-Infinity);
        for (let i = 0; i < s.x.length; i++) {
          const c = Math.floor(((s.x[i] - xMin) / (xMax - xMin)) * (cols - 1));
          if (c < 0 || c >= cols) continue;
          if (s.y[i] < lo[c]) lo[c] = s.y[i];
          if (s.y[i] > hi[c]) hi[c] = s.y[i];
        }
        for (let c = 0; c < cols; c++) {
          if (lo[c] === Infinity) continue;
          const px = pad.l + c + 0.5;
          g.moveTo(px, Y(hi[c])); g.lineTo(px, Y(lo[c]) + 0.5);
        }
      } else {
        const step = Math.max(1, Math.floor(s.x.length / (pw * 2)));
        let first = true;
        for (let i = 0; i < s.x.length; i += step) {
          if (s.x[i] < xMin || s.x[i] > xMax) continue;
          if (first) { g.moveTo(X(s.x[i]), Y(s.y[i])); first = false; }
          else g.lineTo(X(s.x[i]), Y(s.y[i]));
        }
      }
      g.stroke();
      g.setLineDash([]);
    }
    g.restore();

    for (const v of opts.vlines || []) {
      if (v.x < xMin || v.x > xMax) continue;
      g.strokeStyle = v.color; g.lineWidth = 1.5; g.setLineDash([3, 3]);
      g.beginPath(); g.moveTo(X(v.x), pad.t); g.lineTo(X(v.x), pad.t + ph); g.stroke();
      g.setLineDash([]);
      if (v.label) {
        g.font = "600 11px system-ui, sans-serif";
        g.fillStyle = v.color;
        const right = X(v.x) + 4 + g.measureText(v.label).width < W - pad.r;
        g.textAlign = right ? "left" : "right"; g.textBaseline = "top";
        g.fillText(v.label, X(v.x) + (right ? 4 : -4), pad.t + 2 + (v.row || 0) * 14);
        g.font = "10px system-ui, sans-serif";
      }
    }
  }

  global.PapayaCharts = { lineChart };
})(window);
