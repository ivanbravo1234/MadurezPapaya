/*
 * estadios.js — Escala de madurez E1–E5 del proyecto (Tabla 2) y
 * descriptores de la cáscara por zona.
 *
 * Trabaja sobre la MISMA máscara que calcula bimodal.js (no la cambia):
 *   - % de superficie amarilla o anaranjada (criterio de la Tabla 2):
 *     cada píxel de la cáscara cuenta como amarillo si su ángulo de tono
 *     h° = atan2(b*, a*) es menor que H_AMARILLO y tiene color (C* ≥ CROMA_MIN).
 *   - Tres zonas (peduncular, ecuatorial y apical) a lo largo del eje mayor
 *     del fruto, con L*, a*, b*, C* y h° por zona (Chung et al., 2023).
 *   - Desviación de h° (color mezclado) y % de manchas oscuras (el informe
 *     excluye frutos con manchas > 10 % de la superficie).
 *   - Control del fondo blanco: dominante de color de la luz.
 *
 * Funciona en el navegador (window.PapayaEstadios) y en Node (module.exports).
 */
(function (global) {
  "use strict";

  // =========================================================================
  // Tabla 2 del informe: escala de estadios de madurez
  // =========================================================================
  const ESTADIOS = [
    { codigo: "E1", clase: "verde", nombre: "Verde", hasta: 0,
      cascara: "Cáscara completamente verde, sin vetas amarillas",
      uso: "Madurez fisiológica; apto para mercados distantes" },
    { codigo: "E2", clase: "pinton", nombre: "Pintón", hasta: 25,
      cascara: "Verde con inicio de vetas amarillas (hasta 25 % de la superficie)",
      uso: "Cosecha para transporte prolongado" },
    { codigo: "E3", clase: "semi", nombre: "Semimaduro", hasta: 50,
      cascara: "Entre 26 % y 50 % de la superficie amarilla",
      uso: "Cosecha para mercados regionales" },
    { codigo: "E4", clase: "maduro", nombre: "Maduro", hasta: 75,
      cascara: "Entre 51 % y 75 % de la superficie amarilla o anaranjada",
      uso: "Cosecha para mercado local" },
    { codigo: "E5", clase: "muy_maduro", nombre: "Muy maduro", hasta: 100,
      cascara: "Más del 75 % de la superficie amarilla o anaranjada",
      uso: "Madurez de consumo" },
  ];

  // Parámetros por defecto. Se pueden sobrescribir con el bloque "tabla2"
  // de umbrales_app.json (p. ej. tras calibrarlos con el colorímetro).
  const PARAMS = {
    h_amarillo: 100,    // h° < 100° = amarillo/anaranjado (verde ≈ 120°, amarillo ≈ 90°, naranja ≈ 65°)
    croma_min: 15,      // por debajo el píxel es casi gris y su tono no es fiable
    tolerancia_e1: 2,   // % de píxeles "amarillos" que se acepta como ruido en E1
    manchas_delta_L: 25, // mancha = L* más de 25 unidades bajo la mediana del fruto
  };

  /** Estadio de la Tabla 2 según el % de superficie amarilla. */
  function estadioPorAmarillo(pct, tolE1 = PARAMS.tolerancia_e1) {
    if (pct <= tolE1) return ESTADIOS[0];
    for (let i = 1; i < ESTADIOS.length; i++) if (pct <= ESTADIOS[i].hasta) return ESTADIOS[i];
    return ESTADIOS[ESTADIOS.length - 1];
  }

  /** Estadio E1–E5 de una clase del clasificador (verde…muy_maduro o E1…E5).
   *  Si la clase no se reconoce se usa su posición en la lista de clases. */
  function estadioDeClase(clase, clases) {
    const e = ESTADIOS.find((x) => x.clase === clase || x.codigo === clase);
    if (e) return e;
    const i = clases ? clases.indexOf(clase) : -1;
    return i >= 0 && clases.length === ESTADIOS.length ? ESTADIOS[i] : null;
  }

  // =========================================================================
  // Color: mismas constantes que bimodal.js (RGB -> XYZ D65 -> CIELAB)
  // =========================================================================
  const MATRIZ_XYZ = {
    "srgb": [0.412453, 0.357580, 0.180423, 0.212671, 0.715160, 0.072169, 0.019334, 0.119193, 0.950227],
    "display-p3": [0.4865709, 0.2656677, 0.1982173, 0.2289746, 0.6917385, 0.0792869, 0.0, 0.0451134, 1.0439444],
  };
  const LIN = new Float64Array(256);
  for (let i = 0; i < 256; i++) {
    const c = i / 255;
    LIN[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  }
  const fLab = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const GRADOS = 180 / Math.PI;

  function labDeLineal(M, R, G, B) {
    const X = (M[0] * R + M[1] * G + M[2] * B) / 0.950456;
    const Y = M[3] * R + M[4] * G + M[5] * B;
    const Z = (M[6] * R + M[7] * G + M[8] * B) / 1.088754;
    const fy = fLab(Y);
    return [Y > 0.008856 ? 116 * fy - 16 : 903.3 * Y, 500 * (fLab(X) - fy), 200 * (fy - fLab(Z))];
  }

  /** h° en [0, 360) y C* a partir de a*, b*. */
  const tono = (a, b) => { const h = Math.atan2(b, a) * GRADOS; return h < 0 ? h + 360 : h; };
  const croma = (a, b) => Math.hypot(a, b);

  /** CIELAB -> color CSS sRGB (para pintar muestras de color). */
  function labACss(L, a, b) {
    const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
    const inv = (t) => (t * t * t > 0.008856 ? t * t * t : (t - 16 / 116) / 7.787);
    const X = 0.950456 * inv(fx), Y = inv(fy), Z = 1.088754 * inv(fz);
    const lin = [
      3.240479 * X - 1.53715 * Y - 0.498535 * Z,
      -0.969256 * X + 1.875992 * Y + 0.041556 * Z,
      0.055648 * X - 0.204043 * Y + 1.057311 * Z,
    ];
    const g = lin.map((c) => {
      c = Math.min(1, Math.max(0, c));
      return Math.round(255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055));
    });
    return `rgb(${g[0]}, ${g[1]}, ${g[2]})`;
  }

  // =========================================================================
  // Análisis de la superficie
  // =========================================================================
  /**
   * rgba: ImageData.data · mascara: Uint8Array 0/1 (de PapayaBimodal) ·
   * espacio: "srgb" o "display-p3" · params: sobrescribe PARAMS.
   */
  function analizarSuperficie(rgba, mascara, w, h, espacio = "srgb", params = {}) {
    const p = Object.assign({}, PARAMS, params || {});
    const M = MATRIZ_XYZ[espacio] || MATRIZ_XYZ.srgb;

    // 1. Eje mayor del fruto (momentos de la máscara).
    let n = 0, sx = 0, sy = 0;
    for (let y = 0, i = 0; y < h; y++) for (let x = 0; x < w; x++, i++) if (mascara[i]) { n++; sx += x; sy += y; }
    if (!n) throw new Error("La máscara de la papaya quedó vacía.");
    const cx = sx / n, cy = sy / n;
    let sxx = 0, syy = 0, sxy = 0;
    for (let y = 0, i = 0; y < h; y++) for (let x = 0; x < w; x++, i++) {
      if (!mascara[i]) continue;
      const dx = x - cx, dy = y - cy;
      sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }
    const ang = 0.5 * Math.atan2(2 * sxy, sxx - syy);
    const ux = Math.cos(ang), uy = Math.sin(ang);
    let tMin = Infinity, tMax = -Infinity;
    for (let y = 0, i = 0; y < h; y++) for (let x = 0; x < w; x++, i++) {
      if (!mascara[i]) continue;
      const t = (x - cx) * ux + (y - cy) * uy;
      if (t < tMin) tMin = t;
      if (t > tMax) tMax = t;
    }
    const largo = Math.max(1e-9, tMax - tMin);
    const cortes = [tMin + largo / 3, tMin + (2 * largo) / 3];

    // 2. CIELAB por píxel, acumulado por zona (0 y 2 son los extremos).
    const Z = [0, 1, 2].map(() => ({ n: 0, L: 0, a: 0, b: 0, amar: 0, x: 0, y: 0 }));
    const histL = new Uint32Array(101), Ls = new Float32Array(n);
    let sumH = 0, sumH2 = 0, amar = 0, k = 0;
    const extremos = cortes.map(() => ({ sMin: Infinity, sMax: -Infinity }));
    for (let y = 0, i = 0; y < h; y++) for (let x = 0; x < w; x++, i++) {
      if (!mascara[i]) continue;
      const j = 4 * i;
      const [L, a, b] = labDeLineal(M, LIN[rgba[j]], LIN[rgba[j + 1]], LIN[rgba[j + 2]]);
      const hh = tono(a, b), esAmarillo = hh < p.h_amarillo && croma(a, b) >= p.croma_min;
      const t = (x - cx) * ux + (y - cy) * uy, s = -(x - cx) * uy + (y - cy) * ux;
      const z = t < cortes[0] ? 0 : t < cortes[1] ? 1 : 2;
      const zz = Z[z];
      zz.n++; zz.L += L; zz.a += a; zz.b += b; zz.x += x; zz.y += y;
      if (esAmarillo) { zz.amar++; amar++; }
      sumH += hh; sumH2 += hh * hh;
      histL[Math.max(0, Math.min(100, Math.round(L)))]++;
      Ls[k++] = L;
      for (let c = 0; c < cortes.length; c++)
        if (Math.abs(t - cortes[c]) < 1.5) {
          if (s < extremos[c].sMin) extremos[c].sMin = s;
          if (s > extremos[c].sMax) extremos[c].sMax = s;
        }
    }

    // 3. Manchas oscuras: L* muy por debajo de la mediana del fruto.
    let acum = 0, mediana = 50;
    for (let v = 0; v <= 100; v++) { acum += histL[v]; if (acum >= n / 2) { mediana = v; break; } }
    let oscuros = 0;
    for (let i = 0; i < n; i++) if (Ls[i] < mediana - p.manchas_delta_L) oscuros++;

    // 4. Resumen por zona. El extremo con menor h° (más amarillo) se toma
    //    como apical: la papaya empieza a amarillear por el ápice.
    const resumen = Z.map((zz) => {
      const L = zz.n ? zz.L / zz.n : NaN, a = zz.n ? zz.a / zz.n : NaN, b = zz.n ? zz.b / zz.n : NaN;
      return {
        n: zz.n, L, a, b, C: croma(a, b), h: tono(a, b),
        pct_amarilla: zz.n ? (100 * zz.amar) / zz.n : 0,
        x: zz.n ? zz.x / zz.n : cx, y: zz.n ? zz.y / zz.n : cy,
      };
    });
    const apicalAlFinal = !(resumen[0].h < resumen[2].h);
    const orden = apicalAlFinal ? [0, 1, 2] : [2, 1, 0];
    const NOMBRES = ["peduncular", "ecuatorial", "apical"], CORTO = ["Ped", "Ecu", "Api"];
    const zonas = orden.map((iz, r) => Object.assign({ nombre: NOMBRES[r], corto: CORTO[r] }, resumen[iz]));
    const dif = Math.abs(resumen[0].h - resumen[2].h);

    // Promedio de las tres zonas (Tabla 1: "promedio de tres zonas del fruto").
    const prom = { L: 0, a: 0, b: 0 };
    for (const zn of zonas) { prom.L += zn.L / 3; prom.a += zn.a / 3; prom.b += zn.b / 3; }
    prom.C = croma(prom.a, prom.b); prom.h = tono(prom.a, prom.b);

    const pct = (100 * amar) / n;
    const mediaH = sumH / n;
    return {
      pct_amarilla: pct,
      estadio: estadioPorAmarillo(pct, p.tolerancia_e1),
      zonas, promedio: prom,
      h_desv: Math.sqrt(Math.max(0, sumH2 / n - mediaH * mediaH)),
      manchas: (100 * oscuros) / n,
      orientacion_clara: dif >= 2, // si los extremos casi no difieren, apical/peduncular es incierto
      params: p,
      // Para dibujar las divisiones de zona sobre la foto.
      lineas: cortes.map((tc, c) => {
        const e = extremos[c];
        if (!Number.isFinite(e.sMin)) return null;
        const px = cx + ux * tc, py = cy + uy * tc;
        return { x1: px - uy * e.sMin, y1: py + ux * e.sMin, x2: px - uy * e.sMax, y2: py + ux * e.sMax };
      }).filter(Boolean),
    };
  }

  /** Color del fondo blanco (fuera del fruto): si la luz tiñe el papel, el
   *  color de la cáscara también se desvía. Devuelve null si no hay fondo. */
  function controlFondo(rgba, mascara, w, h, espacio = "srgb") {
    const M = MATRIZ_XYZ[espacio] || MATRIZ_XYZ.srgb;
    let n = 0, R = 0, G = 0, B = 0;
    for (let i = 0, j = 0; i < w * h; i++, j += 4) {
      if (mascara[i]) continue;
      const r = rgba[j], g = rgba[j + 1], b = rgba[j + 2];
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      if (mx < 170 || mx - mn > 0.25 * mx) continue; // solo zonas claras y poco saturadas
      R += LIN[r]; G += LIN[g]; B += LIN[b]; n++;
    }
    if (n < 0.02 * w * h) return null;
    const [L, a, b] = labDeLineal(M, R / n, G / n, B / n);
    return { L, a, b, dominante: croma(a, b), calido: b > 0 };
  }

  const api = {
    ESTADIOS, PARAMS, estadioPorAmarillo, estadioDeClase,
    analizarSuperficie, controlFondo, labACss, tono, croma,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else global.PapayaEstadios = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
