/*
 * pipeline.js
 * ===========
 * Port a JavaScript del pipeline Python de resonancia acústica
 * (audio_io.py + preprocessing.py + features.py) y de la predicción del
 * Random Forest exportado con export_model.py.
 *
 * Objetivo: que el celular calcule EXACTAMENTE lo mismo que la laptop.
 * Cada función replica a su equivalente de numpy/scipy/sklearn; donde hay
 * un detalle sutil (redondeo, empates, float32) se indica en el comentario.
 * La equivalencia se verifica con mobile/tests/compare.mjs.
 *
 * Funciona en el navegador (window.PapayaPipeline) y en Node (module.exports).
 */
(function (global) {
  "use strict";

  // =========================================================================
  // FFT (radix-2 + Bluestein para longitudes arbitrarias)
  // =========================================================================
  // scipy.fft usa la longitud exacta pedida (p. ej. 15655, que no es potencia
  // de 2). Para obtener los mismos bins de frecuencia se usa el algoritmo de
  // Bluestein, que expresa una DFT de cualquier N como una convolución que
  // se calcula con FFTs radix-2.

  function isPow2(n) { return (n & (n - 1)) === 0; }

  /** FFT compleja in-place radix-2 (n potencia de 2). inverse sin escalar. */
  function fftRadix2(re, im, inverse) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    const sign = inverse ? 1 : -1;
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (sign * 2 * Math.PI) / len;
      const half = len >> 1;
      // Tabla de twiddles por etapa (más precisa que multiplicar en cadena).
      const wr = new Float64Array(half), wi = new Float64Array(half);
      for (let k = 0; k < half; k++) { wr[k] = Math.cos(ang * k); wi[k] = Math.sin(ang * k); }
      for (let i = 0; i < n; i += len) {
        for (let k = 0; k < half; k++) {
          const a = i + k, b = a + half;
          const xr = re[b] * wr[k] - im[b] * wi[k];
          const xi = re[b] * wi[k] + im[b] * wr[k];
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr; im[a] += xi;
        }
      }
    }
  }

  /** DFT compleja de longitud arbitraria. Devuelve {re, im} nuevos. */
  function fft(reIn, imIn, inverse) {
    const n = reIn.length;
    const re = Float64Array.from(reIn);
    const im = imIn ? Float64Array.from(imIn) : new Float64Array(n);
    if (n <= 1) return { re, im };
    if (isPow2(n)) { fftRadix2(re, im, inverse); return { re, im }; }

    // Bluestein: X[k] = w[k] · Σ (x[j]·w[j]) · conj(w[k-j]),  w[j] = e^{∓iπ j²/N}
    const sign = inverse ? 1 : -1;
    let m = 1; while (m < 2 * n - 1) m <<= 1;
    const wr = new Float64Array(n), wi = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      const j2 = (j * j) % (2 * n); // reduce el ángulo para no perder precisión
      const a = (sign * Math.PI * j2) / n;
      wr[j] = Math.cos(a); wi[j] = Math.sin(a);
    }
    const ar = new Float64Array(m), ai = new Float64Array(m);
    const br = new Float64Array(m), bi = new Float64Array(m);
    for (let j = 0; j < n; j++) {
      ar[j] = re[j] * wr[j] - im[j] * wi[j];
      ai[j] = re[j] * wi[j] + im[j] * wr[j];
    }
    br[0] = wr[0]; bi[0] = -wi[0];
    for (let j = 1; j < n; j++) {
      br[j] = br[m - j] = wr[j];
      bi[j] = bi[m - j] = -wi[j];
    }
    fftRadix2(ar, ai, false);
    fftRadix2(br, bi, false);
    for (let k = 0; k < m; k++) {
      const r = ar[k] * br[k] - ai[k] * bi[k];
      const i = ar[k] * bi[k] + ai[k] * br[k];
      ar[k] = r; ai[k] = i;
    }
    fftRadix2(ar, ai, true);
    for (let k = 0; k < n; k++) {
      const cr = ar[k] / m, ci = ai[k] / m;
      re[k] = cr * wr[k] - ci * wi[k];
      im[k] = cr * wi[k] + ci * wr[k];
    }
    return { re, im };
  }

  // =========================================================================
  // Utilidades numéricas (equivalentes de numpy)
  // =========================================================================
  function argmax(a, from = 0, to = a.length) {
    let idx = from, best = a[from];
    for (let i = from + 1; i < to; i++) if (a[i] > best) { best = a[i]; idx = i; }
    return idx;
  }

  function median(a) {
    const s = Float64Array.from(a).sort();
    const n = s.length, h = n >> 1;
    return n % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
  }

  /** round() de Python 3: redondeo "del banquero" (0.5 -> par). */
  function pyRound(x) {
    const f = Math.floor(x), d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  }

  // =========================================================================
  // audio_io.py
  // =========================================================================
  class AudioValidationError extends Error {}

  /** Lee un .wav (ArrayBuffer) igual que scipy.io.wavfile + _to_float_mono. */
  function parseWav(buffer) {
    const dv = new DataView(buffer);
    const tag = (o) => String.fromCharCode(dv.getUint8(o), dv.getUint8(o + 1),
                                           dv.getUint8(o + 2), dv.getUint8(o + 3));
    if (buffer.byteLength < 12 || tag(0) !== "RIFF" || tag(8) !== "WAVE")
      throw new AudioValidationError("El archivo no es un .wav válido (falta la cabecera RIFF/WAVE).");
    let fmt = null, data = null, off = 12;
    while (off + 8 <= buffer.byteLength) {
      const id = tag(off), size = dv.getUint32(off + 4, true), body = off + 8;
      if (id === "fmt ") {
        let format = dv.getUint16(body, true);
        const bits = dv.getUint16(body + 14, true);
        if (format === 0xfffe && size >= 26) format = dv.getUint16(body + 24, true);
        fmt = { format, channels: dv.getUint16(body + 2, true),
                sr: dv.getUint32(body + 4, true), bits };
      } else if (id === "data") {
        data = { off: body, size: Math.min(size, buffer.byteLength - body) };
      }
      off = body + size + (size % 2);
    }
    if (!fmt || !data) throw new AudioValidationError("El .wav no tiene bloques 'fmt' y 'data'.");
    const { format, channels, sr, bits } = fmt;
    const bps = bits / 8, frames = Math.floor(data.size / (bps * channels));
    const out = new Float64Array(frames);
    let read;
    if (format === 1 && bits === 16) read = (o) => dv.getInt16(o, true) / 32767;
    else if (format === 1 && bits === 8) read = (o) => (dv.getUint8(o) - 128) / 128;
    else if (format === 1 && bits === 24)
      // scipy lee 24 bits como int32 desplazado (<<8) y se divide por 2^31-1.
      read = (o) => ((dv.getUint8(o) << 8) | (dv.getUint8(o + 1) << 16) | (dv.getUint8(o + 2) << 24)) / 2147483647;
    else if (format === 1 && bits === 32) read = (o) => dv.getInt32(o, true) / 2147483647;
    else if (format === 3 && bits === 32) read = (o) => dv.getFloat32(o, true);
    else if (format === 3 && bits === 64) read = (o) => dv.getFloat64(o, true);
    else throw new AudioValidationError(`Formato .wav no soportado (código ${format}, ${bits} bits).`);
    for (let i = 0; i < frames; i++) {
      let acc = 0;
      for (let c = 0; c < channels; c++) acc += read(data.off + (i * channels + c) * bps);
      out[i] = channels === 1 ? acc : acc / channels; // estéreo -> promedio (mono)
    }
    return { signal: out, sr };
  }

  // --- scipy.signal.resample_poly (ventana Kaiser β=5, igual que scipy) ---
  function gcd(a, b) { while (b) [a, b] = [b, a % b]; return a; }

  function limitDenominator(num, den, maxDen) { // fractions.Fraction.limit_denominator
    const g = gcd(num, den); num /= g; den /= g;
    if (den <= maxDen) return [num, den];
    let p0 = 0, q0 = 1, p1 = 1, q1 = 0, n = num, d = den;
    for (;;) {
      const a = Math.floor(n / d), q2 = q0 + a * q1;
      if (q2 > maxDen) break;
      [p0, q0, p1, q1] = [p1, q1, p0 + a * p1, q2];
      [n, d] = [d, n - a * d];
    }
    const k = Math.floor((maxDen - q0) / q1);
    return 2 * d * (q0 + k * q1) <= den ? [p1, q1] : [p0 + k * p1, q0 + k * q1];
  }

  function besselI0(x) {
    let sum = 1, term = 1, k = 1;
    const q = (x * x) / 4;
    do { term *= q / (k * k); sum += term; k++; } while (term > sum * 1e-17);
    return sum;
  }

  function firwinKaiser(numtaps, cutoff, beta) {
    const alpha = 0.5 * (numtaps - 1), h = new Float64Array(numtaps);
    const i0b = besselI0(beta);
    let s = 0;
    for (let i = 0; i < numtaps; i++) {
      const m = i - alpha, x = cutoff * m;
      const sinc = x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
      const r = (2 * i) / (numtaps - 1) - 1;
      const w = besselI0(beta * Math.sqrt(Math.max(0, 1 - r * r))) / i0b;
      h[i] = cutoff * sinc * w;
      s += h[i];
    }
    for (let i = 0; i < numtaps; i++) h[i] /= s; // escala: ganancia 1 en DC
    return h;
  }

  function resamplePoly(x, up, down) {
    const g = gcd(up, down); up /= g; down /= g;
    if (up === 1 && down === 1) return Float64Array.from(x);
    const nIn = x.length;
    const nOut = Math.floor((nIn * up) / down) + ((nIn * up) % down ? 1 : 0);
    const maxRate = Math.max(up, down), halfLen = 10 * maxRate;
    let h = firwinKaiser(2 * halfLen + 1, 1 / maxRate, 5.0).map((v) => v * up);
    const nPrePad = down - (halfLen % down);
    const nPreRemove = Math.floor((halfLen + nPrePad) / down);
    const outLen = (lh) => Math.floor(((nIn - 1) * up + lh - 1) / down) + 1;
    let nPostPad = 0;
    while (outLen(h.length + nPrePad + nPostPad) < nOut + nPreRemove) nPostPad++;
    const hp = new Float64Array(nPrePad + h.length + nPostPad);
    hp.set(h, nPrePad); h = hp;
    // upfirdn: y[k] = Σ_j h[j] · x_up[k·down − j]  (x_up = x con ceros intercalados)
    const y = new Float64Array(nOut);
    for (let o = 0; o < nOut; o++) {
      const m0 = (o + nPreRemove) * down;
      let j = m0 % up;
      const jMin = m0 - (nIn - 1) * up;
      if (j < jMin) j += Math.ceil((jMin - j) / up) * up;
      let acc = 0;
      for (; j < h.length && j <= m0; j += up) acc += h[j] * x[(m0 - j) / up];
      y[o] = acc;
    }
    return y;
  }

  function resample(signal, srIn, srOut) {
    if (srIn === srOut) return signal;
    const [up, down] = limitDenominator(srOut, srIn, 1000);
    return resamplePoly(signal, up, down);
  }

  function validateSignal(signal, name = "audio", silenceRms = 1e-4, clipFraction = 0.01) {
    if (signal.length === 0) throw new AudioValidationError(`'${name}': la señal está vacía.`);
    let sq = 0, clipped = 0;
    for (let i = 0; i < signal.length; i++) {
      const v = signal[i];
      if (!Number.isFinite(v)) throw new AudioValidationError(`'${name}': la señal contiene NaN o Inf.`);
      sq += v * v;
      if (Math.abs(v) >= 0.999) clipped++;
    }
    const rms = Math.sqrt(sq / signal.length);
    if (rms < silenceRms)
      throw new AudioValidationError(`'${name}': parece silencio (RMS=${rms.toExponential(2)}). ¿El golpe quedó fuera de la grabación?`);
    const frac = clipped / signal.length;
    if (frac > clipFraction)
      throw new AudioValidationError(`'${name}': audio saturado (${(frac * 100).toFixed(1)}% de muestras al tope). Aleja el micrófono o golpea más suave.`);
  }

  // =========================================================================
  // preprocessing.py
  // =========================================================================
  function shortTimeEnergy(x, frame, hop) {
    if (x.length < frame) { let s = 0; for (const v of x) s += v * v; return Float64Array.of(s); }
    const nFrames = 1 + Math.floor((x.length - frame) / hop);
    const e = new Float64Array(nFrames);
    for (let m = 0; m < nFrames; m++) {
      let s = 0; const st = m * hop;
      for (let i = st; i < st + frame; i++) s += x[i] * x[i];
      e[m] = s;
    }
    return e;
  }

  function detectOnset(x, cfg) {
    const ste = shortTimeEnergy(x, cfg.FRAME_LENGTH, cfg.HOP_LENGTH);
    if (ste.length === 1) return 0;
    const med = median(ste);
    let mad = median(ste.map((v) => Math.abs(v - med)));
    if (mad < 1e-12) { let s = 0; for (const v of ste) s += v; mad = (s / ste.length) * 0.1 + 1e-12; }
    const thr = med + cfg.ONSET_K * mad;
    for (let m = 0; m < ste.length; m++) if (ste[m] > thr) return m * cfg.HOP_LENGTH;
    throw new Error("No se detectó ningún impulso: la señal no supera el umbral adaptativo. ¿La grabación es silencio o ruido de fondo plano?");
  }

  function preprocess(x, sr, cfg) {
    const onset = detectOnset(x, cfg);
    const start = Math.max(0, onset - Math.trunc(cfg.PRE_ONSET_PAD * sr));
    const end = Math.min(x.length, onset + Math.trunc(cfg.ANALYSIS_WINDOW * sr));
    const imp = x.slice(start, end);
    if (imp.length === 0) throw new Error("La ventana de impulso resultó vacía tras el recorte.");
    let peak = 0; for (const v of imp) peak = Math.max(peak, Math.abs(v));
    if (peak > 0) for (let i = 0; i < imp.length; i++) imp[i] /= peak;
    return { impulse: imp, onset, start };
  }

  // =========================================================================
  // features.py
  // =========================================================================
  function computeSpectrum(signal, sr, fftSize) {
    const M = signal.length, n = Math.max(fftSize, M);
    const re = new Float64Array(n);
    for (let k = 0; k < M; k++) { // np.hanning(M)
      const w = M === 1 ? 1 : 0.5 + 0.5 * Math.cos((Math.PI * (1 - M + 2 * k)) / (M - 1));
      re[k] = signal[k] * w;
    }
    const X = fft(re, null, false);
    const nb = Math.floor(n / 2) + 1, val = 1.0 / (n * (1.0 / sr));
    const freqs = new Float64Array(nb), mag = new Float64Array(nb);
    for (let k = 0; k < nb; k++) { freqs[k] = k * val; mag[k] = Math.hypot(X.re[k], X.im[k]); }
    return { freqs, mag };
  }

  /** scipy.signal.find_peaks(x, distance, prominence) — mismo orden de filtros. */
  function findPeaks(x, distance, minProminence) {
    // 1) máximos locales (con mesetas: se toma el punto medio)
    let peaks = [];
    for (let i = 1, iMax = x.length - 1; i < iMax; i++) {
      if (x[i - 1] < x[i]) {
        let ia = i + 1;
        while (ia < iMax && x[ia] === x[i]) ia++;
        if (x[ia] < x[i]) { peaks.push(Math.floor((i + ia - 1) / 2)); i = ia; }
      }
    }
    // 2) distancia mínima: se conservan primero los picos más altos
    if (distance != null && peaks.length) {
      const d = Math.ceil(distance), keep = peaks.map(() => true);
      const order = peaks.map((_, i) => i).sort((a, b) => x[peaks[a]] - x[peaks[b]]);
      for (let t = peaks.length - 1; t >= 0; t--) {
        const j = order[t];
        if (!keep[j]) continue;
        for (let k = j - 1; k >= 0 && peaks[j] - peaks[k] < d; k--) keep[k] = false;
        for (let k = j + 1; k < peaks.length && peaks[k] - peaks[j] < d; k++) keep[k] = false;
      }
      peaks = peaks.filter((_, i) => keep[i]);
    }
    // 3) prominencia
    const prom = peaks.map((p) => {
      let lmin = x[p], rmin = x[p];
      for (let i = p; i >= 0 && x[i] <= x[p]; i--) if (x[i] < lmin) lmin = x[i];
      for (let i = p; i < x.length && x[i] <= x[p]; i++) if (x[i] < rmin) rmin = x[i];
      return x[p] - Math.max(lmin, rmin);
    });
    const outP = [], outProm = [];
    peaks.forEach((p, i) => { if (prom[i] >= minProminence) { outP.push(p); outProm.push(prom[i]); } });
    return { peaks: outP, prominences: outProm };
  }

  function findResonances(freqs, mag, cfg) {
    const fB = [], mB = [];
    for (let i = 0; i < freqs.length; i++)
      if (freqs[i] >= cfg.FREQ_MIN && freqs[i] <= cfg.FREQ_MAX) { fB.push(freqs[i]); mB.push(mag[i]); }
    if (fB.length < 2) return { peakFreqs: [], peakMags: [] };
    const df = fB[1] - fB[0];
    const distance = Math.max(1, pyRound(cfg.MIN_PEAK_DISTANCE_HZ / df));
    let mMax = -Infinity; for (const v of mB) mMax = Math.max(mMax, v);
    const { peaks, prominences } = findPeaks(mB, distance, mMax * 0.05);
    if (peaks.length === 0) { const i = argmax(mB); return { peakFreqs: [fB[i]], peakMags: [mB[i]] }; }
    const byProm = peaks.map((_, i) => i).sort((a, b) => prominences[b] - prominences[a])
      .slice(0, cfg.N_RESONANCES).map((i) => peaks[i]).sort((a, b) => fB[a] - fB[b]);
    return { peakFreqs: byProm.map((i) => fB[i]), peakMags: byProm.map((i) => mB[i]) };
  }

  /** |scipy.signal.hilbert(x)| */
  function hilbertEnvelope(x) {
    const N = x.length;
    if (N === 0) throw new Error("La señal está vacía; no se puede calcular la envolvente.");
    const X = fft(x, null, false);
    const h = new Float64Array(N);
    if (N % 2 === 0) { h[0] = h[N / 2] = 1; for (let i = 1; i < N / 2; i++) h[i] = 2; }
    else { h[0] = 1; for (let i = 1; i < (N + 1) / 2; i++) h[i] = 2; }
    for (let i = 0; i < N; i++) { X.re[i] *= h[i]; X.im[i] *= h[i]; }
    const z = fft(X.re, X.im, true);
    const env = new Float64Array(N);
    for (let i = 0; i < N; i++) env[i] = Math.hypot(z.re[i], z.im[i]) / N;
    return env;
  }

  function zeroCrossingRate(x) {
    if (x.length < 2) return 0;
    let c = 0, prev = x[0] < 0 ? -1 : 1; // np.sign con 0 -> +1
    for (let i = 1; i < x.length; i++) {
      const s = x[i] < 0 ? -1 : 1;
      if (s !== prev) c++;
      prev = s;
    }
    return c / (x.length - 1);
  }

  function shortTimeEnergyMean(x, frame, hop) {
    const e = shortTimeEnergy(x, frame, hop);
    if (x.length < frame) return e[0];
    let s = 0; for (const v of e) s += v;
    return s / e.length;
  }

  function dampingLogDecrement(env, f1, sr) {
    if (!(f1 > 0)) return NaN;
    const p = argmax(env), pv = env[p];
    if (pv <= 0) return NaN;
    const floor = 0.05 * pv;
    let end = env.length - p;
    for (let i = p; i < env.length; i++) if (env[i] < floor) { end = i - p; break; }
    if (end < 4) return NaN;
    // Ajuste lineal de ln(env) vs t (equivale a np.polyfit grado 1).
    let st = 0, sl = 0;
    const t = new Float64Array(end), l = new Float64Array(end);
    for (let i = 0; i < end; i++) { t[i] = i / sr; l[i] = Math.log(Math.max(env[p + i], 1e-12)); st += t[i]; sl += l[i]; }
    const tm = st / end, lm = sl / end;
    let num = 0, den = 0;
    for (let i = 0; i < end; i++) { num += (t[i] - tm) * (l[i] - lm); den += (t[i] - tm) ** 2; }
    return -(num / den) / (2 * Math.PI * f1);
  }

  function dampingBandwidth(freqs, mag, f1, cfg) {
    const fB = [], mB = [];
    for (let i = 0; i < freqs.length; i++)
      if (freqs[i] >= cfg.FREQ_MIN && freqs[i] <= cfg.FREQ_MAX) { fB.push(freqs[i]); mB.push(mag[i]); }
    if (fB.length < 3) return NaN;
    let pi = 0, best = Infinity;
    for (let i = 0; i < fB.length; i++) { const d = Math.abs(fB[i] - f1); if (d < best) { best = d; pi = i; } }
    const half = mB[pi] / Math.sqrt(2);
    let lo = pi; while (lo > 0 && mB[lo] > half) lo--;
    let hi = pi; while (hi < mB.length - 1 && mB[hi] > half) hi++;
    if (lo === hi) return NaN;
    const interp = (i, j) => (mB[j] === mB[i] ? fB[i] : fB[i] + ((half - mB[i]) / (mB[j] - mB[i])) * (fB[j] - fB[i]));
    const bw = interp(hi - 1, hi) - interp(lo, lo + 1);
    return bw <= 0 ? NaN : bw / (2 * f1);
  }

  function extractAllFeatures(impulse, massG, sr, cfg) {
    const { freqs, mag } = computeSpectrum(impulse, sr, cfg.FFT_SIZE);
    const { peakFreqs, peakMags } = findResonances(freqs, mag, cfg);
    const f1 = peakFreqs.length >= 1 ? peakFreqs[0] : NaN;
    const f2 = peakFreqs.length >= 2 ? peakFreqs[1] : NaN;
    const env = hilbertEnvelope(impulse);
    const m23 = Math.pow(massG, 2 / 3);
    const feats = {
      f1, f2,
      damping_logdec: Number.isFinite(f1) ? dampingLogDecrement(env, f1, sr) : NaN,
      damping_bw: Number.isFinite(f1) ? dampingBandwidth(freqs, mag, f1, cfg) : NaN,
      S: Number.isFinite(f1) ? f1 ** 2 * m23 : NaN,
      EI: Number.isFinite(f2) ? f2 ** 2 * m23 : NaN,
      zcr: zeroCrossingRate(impulse),
      ste: shortTimeEnergyMean(impulse, cfg.FRAME_LENGTH, cfg.HOP_LENGTH),
    };
    return { feats, freqs, mag, env, peakFreqs, peakMags };
  }

  // =========================================================================
  // Predicción (Pipeline sklearn: SimpleImputer(median) -> RandomForest)
  // =========================================================================
  function predict(model, feats, massG) {
    const row = Object.assign({ masa_g: massG }, feats);
    // sklearn convierte X a float32 antes de recorrer los árboles.
    const x = model.feature_cols.map((c, i) => {
      const v = row[c];
      return Math.fround(Number.isFinite(v) ? v : model.imputer_medians[i]);
    });
    const proba = new Float64Array(model.classes.length);
    for (const t of model.trees) {
      let n = 0;
      while (t.left[n] !== -1) n = x[t.feature[n]] <= t.threshold[n] ? t.left[n] : t.right[n];
      const v = t.value[n];
      for (let c = 0; c < v.length; c++) proba[c] += v[c];
    }
    for (let c = 0; c < proba.length; c++) proba[c] /= model.trees.length;
    const best = argmax(proba);
    const out = {};
    model.classes.forEach((c, i) => (out[c] = proba[i]));
    return { label: model.classes[best], proba: out };
  }

  // =========================================================================
  // Orquestador
  // =========================================================================
  /**
   * Analiza una señal cruda: resamplea, valida, recorta el impulso, extrae
   * features y (si hay modelo) predice. Lanza errores con mensajes claros.
   */
  function analyze(signal, srIn, massG, model, name = "audio") {
    const cfg = model.config;
    const sr = cfg.SAMPLE_RATE;
    const x = resample(signal, srIn, sr);
    validateSignal(x, name);
    const pre = preprocess(x, sr, cfg);
    const r = extractAllFeatures(pre.impulse, massG, sr, cfg);
    return Object.assign(r, {
      signal: x, sr, onset: pre.onset, start: pre.start, impulse: pre.impulse,
      prediction: model.trees ? predict(model, r.feats, massG) : null,
    });
  }

  const api = {
    AudioValidationError, parseWav, resample, resamplePoly, validateSignal,
    shortTimeEnergy, detectOnset, preprocess, computeSpectrum, findPeaks,
    findResonances, hilbertEnvelope, zeroCrossingRate, shortTimeEnergyMean,
    dampingLogDecrement, dampingBandwidth, extractAllFeatures, predict, analyze,
    fft, pyRound,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else global.PapayaPipeline = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
