/*
 * bimodal.js — Color de la cáscara (CIELAB) + frecuencia del golpe + regla
 * de umbrales. Es el port a JavaScript de:
 *   bimodal/vision_color.py     -> colorDesdeImagen
 *   bimodal/acustica_fft.py     -> acusticaDominante
 *   bimodal/calcular_umbrales.py (clasificar) -> clasificar, indiceMadurez
 * Mismos parámetros y pasos; verificado contra Python con
 * mobile/tests/compare_bimodal.mjs.
 *
 * Funciona en el navegador (window.PapayaBimodal) y en Node (module.exports).
 * Depende de PapayaPipeline.fft y PapayaPipeline.pyRound.
 */
(function (global) {
  "use strict";
  const P = typeof module !== "undefined" && module.exports
    ? require("./pipeline.js") : global.PapayaPipeline;

  // =========================================================================
  // Visión: parámetros (idénticos a vision_color.py)
  // =========================================================================
  const ANCHO_TRABAJO = 1024;
  const FACTOR_UMBRAL = 0.5;
  const FRAC_RUIDO = 0.01, FRAC_TALLO = 0.15, FRAC_EROSION = 0.01;
  const BRILLO_V_MIN = 235, BRILLO_S_MAX = 60;

  /** S y V de HSV (0–255) como OpenCV: V = max, S = 255·(V − min)/V. */
  function canalesSV(rgba, n) {
    const S = new Uint8Array(n), V = new Uint8Array(n);
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      const r = rgba[j], g = rgba[j + 1], b = rgba[j + 2];
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      V[i] = mx;
      S[i] = mx === 0 ? 0 : Math.round((255 * (mx - mn)) / mx);
    }
    return { S, V };
  }

  /** GaussianBlur 7×7 con sigma automático: OpenCV usa la tabla fija
   *  [1, 3.5, 7, 9, 7, 3.5, 1]/32 y borde BORDER_REFLECT_101. */
  function gauss7(src, w, h) {
    const k = [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125];
    const refl = (i, n) => (i < 0 ? -i : i >= n ? 2 * n - 2 - i : i);
    const tmp = new Float32Array(w * h), out = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      const o = y * w;
      for (let x = 0; x < w; x++) {
        let s = 0;
        for (let t = -3; t <= 3; t++) s += k[t + 3] * src[o + refl(x + t, w)];
        tmp[o + x] = s;
      }
    }
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        let s = 0;
        for (let t = -3; t <= 3; t++) s += k[t + 3] * tmp[refl(y + t, h) * w + x];
        out[y * w + x] = Math.round(s);
      }
    return out;
  }

  /** Umbral de Otsu (mismo recorrido que cv::threshold con THRESH_OTSU). */
  function otsu(img) {
    const hist = new Float64Array(256), n = img.length;
    for (let i = 0; i < n; i++) hist[img[i]]++;
    let mu = 0;
    for (let i = 0; i < 256; i++) mu += i * (hist[i] /= n);
    let q1 = 0, mu1 = 0, maxSigma = 0, maxVal = 0;
    const EPS = 1.1920929e-7;
    for (let i = 0; i < 256; i++) {
      const p = hist[i];
      mu1 *= q1; q1 += p;
      const q2 = 1 - q1;
      if (Math.min(q1, q2) < EPS || Math.max(q1, q2) > 1 - EPS) continue;
      mu1 = (mu1 + i * p) / q1;
      const mu2 = (mu - q1 * mu1) / q2;
      const sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
      if (sigma > maxSigma) { maxSigma = sigma; maxVal = i; }
    }
    return maxVal;
  }

  /** Semiancho horizontal de cada fila de una elipse k×k
   *  (cv::getStructuringElement(MORPH_ELLIPSE)). */
  function filasElipse(k) {
    const r = Math.floor(k / 2), c = r, filas = [];
    for (let dy = -r; dy <= r; dy++) {
      const dx = r ? Math.round(c * Math.sqrt((r * r - dy * dy) / (r * r))) : 0;
      filas.push([dy, dx]);
    }
    return filas;
  }

  /** Erosión (valor=0) o dilatación (valor=1) binaria con elipse k×k.
   *  Para cada fila del kernel se cuenta, con sumas acumuladas por fila,
   *  si hay algún píxel "valor" en el tramo horizontal. Los píxeles fuera
   *  de la imagen no cuentan (borde neutro, como OpenCV). */
  function morfo(m, w, h, k, valor) {
    const pref = new Int32Array((w + 1) * h);
    for (let y = 0; y < h; y++) {
      const o = y * (w + 1);
      for (let x = 0; x < w; x++) pref[o + x + 1] = pref[o + x] + (m[y * w + x] === valor ? 1 : 0);
    }
    const filas = filasElipse(k), out = new Uint8Array(w * h);
    const enc = valor === 0 ? 0 : 1; // erosión: si hay un 0 -> 0; dilatación: si hay un 1 -> 1
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        let hay = false;
        for (let f = 0; f < filas.length && !hay; f++) {
          const yy = y + filas[f][0];
          if (yy < 0 || yy >= h) continue;
          const dx = filas[f][1], x0 = Math.max(0, x - dx), x1 = Math.min(w - 1, x + dx);
          const o = yy * (w + 1);
          if (pref[o + x1 + 1] - pref[o + x0] > 0) hay = true;
        }
        out[y * w + x] = hay ? enc : 1 - enc;
      }
    return out;
  }
  const erosion = (m, w, h, k) => morfo(m, w, h, k, 0);
  const dilatacion = (m, w, h, k) => morfo(m, w, h, k, 1);
  const apertura = (m, w, h, k) => dilatacion(erosion(m, w, h, k), w, h, k);

  /** Componente conexa (8-vecinos) de mayor área. */
  function mayorComponente(m, w, h) {
    const lab = new Int32Array(w * h), pila = new Int32Array(w * h);
    let mejor = 0, mejorArea = 0, id = 0;
    for (let s = 0; s < w * h; s++) {
      if (!m[s] || lab[s]) continue;
      id++; let top = 0, area = 0;
      pila[top++] = s; lab[s] = id;
      while (top) {
        const p = pila[--top]; area++;
        const px = p % w, py = (p - px) / w;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const x = px + dx, y = py + dy;
          if (x < 0 || y < 0 || x >= w || y >= h) continue;
          const q = y * w + x;
          if (m[q] && !lab[q]) { lab[q] = id; pila[top++] = q; }
        }
      }
      if (area > mejorArea) { mejorArea = area; mejor = id; }
    }
    if (!mejor) throw new Error("No se encontró la papaya en la foto. Usa un fondo blanco y que el fruto ocupe buena parte de la imagen.");
    const out = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) out[i] = lab[i] === mejor ? 1 : 0;
    return out;
  }

  /** Relleno de huecos: todo lo que el fondo no alcanza desde el borde
   *  (4-vecinos) pertenece al fruto. Equivale a dibujar el contorno externo lleno. */
  function rellenar(m, w, h) {
    const fuera = new Uint8Array(w * h), pila = new Int32Array(w * h);
    let top = 0;
    const sembrar = (q) => { if (!m[q] && !fuera[q]) { fuera[q] = 1; pila[top++] = q; } };
    for (let x = 0; x < w; x++) { sembrar(x); sembrar((h - 1) * w + x); }
    for (let y = 0; y < h; y++) { sembrar(y * w); sembrar(y * w + w - 1); }
    while (top) {
      const p = pila[--top], x = p % w;
      if (x > 0) sembrar(p - 1);
      if (x < w - 1) sembrar(p + 1);
      if (p >= w) sembrar(p - w);
      if (p < w * (h - 1)) sembrar(p + w);
    }
    const out = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) out[i] = fuera[i] ? 0 : 1;
    return out;
  }

  const kernelProp = (w, h, frac) => Math.max(3, P.pyRound(Math.min(w, h) * frac) | 1);

  /** Máscara de la cáscara (Uint8Array 0/1). Mismos pasos a–f que
   *  vision_color.segmentar_papaya. */
  function segmentar(rgba, w, h) {
    const n = w * h, { S, V } = canalesSV(rgba, n);
    const s = gauss7(S, w, h);
    const umbral = FACTOR_UMBRAL * otsu(s);
    let m = new Uint8Array(n);
    for (let i = 0; i < n; i++) m[i] = s[i] > umbral ? 1 : 0;

    m = mayorComponente(apertura(m, w, h, kernelProp(w, h, FRAC_RUIDO)), w, h);
    m = rellenar(m, w, h);

    let area = 0; for (let i = 0; i < n; i++) area += m[i];
    const diametro = Math.sqrt((4 * area) / Math.PI);
    const kTallo = Math.max(3, Math.trunc(diametro * FRAC_TALLO) | 1);
    m = mayorComponente(apertura(m, w, h, kTallo), w, h);
    m = erosion(m, w, h, kernelProp(w, h, FRAC_EROSION));

    for (let i = 0; i < n; i++) if (V[i] >= BRILLO_V_MIN && S[i] <= BRILLO_S_MAX) m[i] = 0;
    return m;
  }

  // RGB -> CIELAB (D65), igual que vision_color.bgr_a_lab: curva sRGB y
  // matriz RGB lineal -> XYZ según el espacio de la imagen (sRGB o Display P3).
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

  function promedioLab(rgba, m, espacio = "srgb") {
    const M = MATRIZ_XYZ[espacio] || MATRIZ_XYZ.srgb;
    let sL = 0, sa = 0, sb = 0, n = 0;
    for (let i = 0, j = 0; i < m.length; i++, j += 4) {
      if (!m[i]) continue;
      const R = LIN[rgba[j]], G = LIN[rgba[j + 1]], B = LIN[rgba[j + 2]];
      const X = (M[0] * R + M[1] * G + M[2] * B) / 0.950456;
      const Y = M[3] * R + M[4] * G + M[5] * B;
      const Z = (M[6] * R + M[7] * G + M[8] * B) / 1.088754;
      const fy = fLab(Y);
      sL += Y > 0.008856 ? 116 * fy - 16 : 903.3 * Y;
      sa += 500 * (fLab(X) - fy);
      sb += 200 * (fy - fLab(Z));
      n++;
    }
    if (!n) throw new Error("La máscara de la papaya quedó vacía.");
    return { L: sL / n, a: sa / n, b: sb / n, fraccion: n / m.length };
  }

  /** rgba: datos RGBA (ImageData.data) ya reducidos a ~1024 px de ancho.
   *  espacio: "display-p3" o "srgb" (ImageData.colorSpace). */
  function colorDesdeImagen(rgba, w, h, espacio = "srgb") {
    const mascara = segmentar(rgba, w, h);
    return Object.assign(promedioLab(rgba, mascara, espacio), { mascara, w, h, espacio });
  }

  // =========================================================================
  // Acústica: parámetros (idénticos a acustica_fft.py)
  // =========================================================================
  const CORTE_PASA_ALTOS = 100, ORDEN_FILTRO = 4;
  const PRE_IMPACTO = 0.002, POST_IMPACTO = 0.040;
  const N_FFT = 32768, SUAVIZADO_HZ = 20, FREQ_MIN = 100, FREQ_MAX = 3000;

  /** Butterworth pasa-altos digital (transformación bilineal con
   *  pre-distorsión, como scipy.signal.butter) en secciones de 2º orden:
   *  [b0, b1, b2, a1, a2] con ganancia 1 en Nyquist. */
  function butterPasaAltos(orden, fc, fs) {
    const Wa = 2 * fs * Math.tan((Math.PI * fc) / fs), K = 2 * fs, sos = [];
    for (let k = 0; k < orden / 2; k++) {
      // Polo analógico del prototipo y su versión pasa-altos: p_hp = Wa / p.
      const th = (Math.PI * (2 * k + orden + 1)) / (2 * orden);
      const pr = Math.cos(th), pi = Math.sin(th), d = pr * pr + pi * pi;
      const hr = (Wa * pr) / d, hi = (-Wa * pi) / d;
      // Bilineal: z = (K + s) / (K − s).
      const nr = K + hr, ni = hi, dr = K - hr, di = -hi, dd = dr * dr + di * di;
      const zr = (nr * dr + ni * di) / dd, zi = (ni * dr - nr * di) / dd;
      const a1 = -2 * zr, a2 = zr * zr + zi * zi;
      const g = (1 - a1 + a2) / 4; // |H(z = −1)| = 1
      sos.push([g, -2 * g, g, a1, a2]);
    }
    return sos;
  }

  function sosfilt(sos, x, zi) {
    let y = Float64Array.from(x);
    sos.forEach(([b0, b1, b2, a1, a2], s) => {
      let z0 = zi[s][0], z1 = zi[s][1];
      for (let n = 0; n < y.length; n++) {
        const xn = y[n], yn = b0 * xn + z0;
        z0 = b1 * xn - a1 * yn + z1;
        z1 = b2 * xn - a2 * yn;
        y[n] = yn;
      }
    });
    return y;
  }

  /** scipy.signal.sosfilt_zi: estado estacionario ante un escalón. */
  function sosfiltZi(sos) {
    let escala = 1;
    return sos.map(([b0, b1, b2, a1, a2]) => {
      const B0 = b1 - a1 * b0, B1 = b2 - a2 * b0;
      const z0 = (B0 + B1) / (1 + a1 + a2), z1 = (1 + a1) * z0 - B0;
      const zi = [escala * z0, escala * z1];
      escala *= (b0 + b1 + b2) / (1 + a1 + a2);
      return zi;
    });
  }

  /** scipy.signal.sosfiltfilt (padtype "odd", padlen por defecto). */
  function sosfiltfilt(sos, x) {
    const pad = 3 * (2 * sos.length + 1), n = x.length;
    if (n <= pad) throw new Error("El audio es demasiado corto.");
    const ext = new Float64Array(n + 2 * pad);
    for (let i = 0; i < pad; i++) ext[i] = 2 * x[0] - x[pad - i];
    ext.set(x, pad);
    for (let i = 0; i < pad; i++) ext[pad + n + i] = 2 * x[n - 1] - x[n - 2 - i];
    const zi = sosfiltZi(sos);
    const escalar = (v) => zi.map(([a, b]) => [a * v, b * v]);
    let y = sosfilt(sos, ext, escalar(ext[0]));
    y.reverse();
    y = sosfilt(sos, y, escalar(y[0]));
    y.reverse();
    return y.subarray(pad, pad + n);
  }

  /** scipy.ndimage.uniform_filter1d, modo "reflect". */
  function promedioMovil(x, size) {
    const n = x.length, izq = Math.floor(size / 2), der = size - izq - 1;
    const at = (i) => (i < 0 ? x[-i - 1] : i >= n ? x[2 * n - i - 1] : x[i]);
    const out = new Float64Array(n);
    let s = 0;
    for (let i = -izq; i <= der; i++) s += at(i);
    for (let i = 0; i < n; i++) {
      out[i] = s / size;
      s += at(i + der + 1) - at(i - izq);
    }
    return out;
  }

  /** Frecuencia dominante y amplitud del golpe (acustica_fft.procesar_audio).
   *  `signal` en la escala original del audio, sin normalizar. */
  function acusticaDominante(signal, fs) {
    const xf = sosfiltfilt(butterPasaAltos(ORDEN_FILTRO, CORTE_PASA_ALTOS, fs), signal);
    let k = 0;
    for (let i = 1; i < xf.length; i++) if (Math.abs(xf[i]) > Math.abs(xf[k])) k = i;
    const ini = Math.max(0, k - Math.trunc(PRE_IMPACTO * fs));
    const fin = Math.min(xf.length, k + Math.trunc(POST_IMPACTO * fs));
    const n = fin - ini, m = Math.floor(n / 2), M = 2 * (n - m);

    const re = new Float64Array(N_FFT), im = new Float64Array(N_FFT);
    let sumaW = 0;
    for (let i = 0; i < n; i++) {
      const w = i < m ? 1 : 0.5 - 0.5 * Math.cos((2 * Math.PI * (i - m + (n - m))) / (M - 1));
      re[i] = xf[ini + i] * w; sumaW += w;
    }
    const F = P.fft(re, im);
    const nb = N_FFT / 2 + 1, X = new Float64Array(nb);
    for (let i = 0; i < nb; i++) X[i] = Math.hypot(F.re[i], F.im[i]);

    const df = fs / N_FFT;
    const Xs = promedioMovil(X, Math.max(1, P.pyRound(SUAVIZADO_HZ / df)));
    let best = -1;
    for (let i = 0; i < nb; i++) {
      const f = i * df;
      if (f < FREQ_MIN || f > FREQ_MAX) continue;
      if (best < 0 || Xs[i] > Xs[best]) best = i;
    }
    const y0 = Xs[best - 1], y1 = Xs[best], y2 = Xs[best + 1], den = y0 - 2 * y1 + y2;
    const delta = den !== 0 ? (0.5 * (y0 - y2)) / den : 0;
    return { frecuencia_hz: best * df + delta * df, amplitud: (2 * X[best]) / sumaW };
  }

  // =========================================================================
  // Regla de decisión (calcular_umbrales.clasificar)
  // =========================================================================
  function umbralesValidos(u) {
    return u && Array.isArray(u.clases) && u.centroides && u.sigma && u.pesos && Array.isArray(u.variables);
  }

  /** D_c = Σ w_v ((x_v − μ_cv)/σ_v)²; confianza = softmax(−D/2).
   *  Se usan solo las variables presentes en x (p. ej. sin audio, solo color). */
  function clasificar(u, x) {
    const vars = u.variables.filter((v) => Number.isFinite(x[v]));
    if (!vars.length) throw new Error("No hay mediciones para clasificar.");
    const D = {};
    for (const c of u.clases) {
      const mu = u.centroides[c];
      D[c] = vars.reduce((s, v) => { const z = (x[v] - mu[v]) / u.sigma[v]; return s + u.pesos[v] * z * z; }, 0);
    }
    const dMin = Math.min(...Object.values(D));
    let total = 0; const e = {};
    for (const c of u.clases) { e[c] = Math.exp(-(D[c] - dMin) / 2); total += e[c]; }
    const proba = {};
    for (const c of u.clases) proba[c] = e[c] / total;
    const label = u.clases.reduce((m, c) => (D[c] < D[m] ? c : m));
    return { label, proba, distancias: D, variables: vars };
  }

  /** Índice continuo de madurez: β0 + β1·a*, acotado al rango del dataset. */
  function indiceMadurez(u, x) {
    const ind = u.indice;
    if (!ind || !Number.isFinite(x[ind.variable])) return null;
    const v = ind.intercepto + ind.pendiente * x[ind.variable];
    return Math.min(ind.maximo, Math.max(ind.minimo, v));
  }

  const api = {
    ANCHO_TRABAJO, colorDesdeImagen, segmentar, promedioLab, otsu, gauss7,
    acusticaDominante, butterPasaAltos, sosfiltfilt, promedioMovil,
    umbralesValidos, clasificar, indiceMadurez,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else global.PapayaBimodal = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
