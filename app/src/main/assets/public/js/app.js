/*
 * app.js — Interfaz de la app móvil (sistema bimodal).
 *
 * Flujo:
 *   1. Foto de la cáscara (hasta 2 caras) -> PapayaBimodal.colorDesdeImagen -> L*, a*, b*
 *      + PapayaEstadios.analizarSuperficie -> % superficie amarilla (Tabla 2),
 *        color por zona (peduncular, ecuatorial, apical), C*, h°, manchas
 *   2. Golpe (hasta 3, micrófono o .wav) -> PapayaBimodal.acusticaDominante -> frecuencia, amplitud
 *      (+ PapayaPipeline.analyze para ζ, S, EI y los gráficos de detalle)
 *   3. Regla de umbrales (umbrales_app.json, calculada con
 *      bimodal/calcular_umbrales.py sobre el dataset de campo) -> estadio
 *      E1–E5, confianza e índice de madurez 1–10. Con varias caras o golpes
 *      se usa el promedio.
 *   4. Clima de la localidad (PapayaClima) -> días estimados hasta E5.
 *   5. Cada fruto se guarda en el historial local (PapayaHistorial).
 * Basta con una de las dos mediciones; con ambas se combinan.
 */
(function () {
  "use strict";
  const P = window.PapayaPipeline, B = window.PapayaBimodal, C = window.PapayaCharts;
  const E = window.PapayaEstadios, Cl = window.PapayaClima, H = window.PapayaHistorial;
  const $ = (id) => document.getElementById(id);

  const DURACION_MS = 2500; // duración de cada grabación
  const MAX_CARAS = 2, MAX_GOLPES = 3, META_CICLO_S = 3;
  const CLASES = {
    verde: { nombre: "Verde", css: "--verde" },
    pinton: { nombre: "Pintón", css: "--pinton" },
    semi: { nombre: "Semimaduro", css: "--semi" },
    maduro: { nombre: "Maduro", css: "--maduro" },
    muy_maduro: { nombre: "Muy maduro", css: "--muy-maduro" },
  };
  const ORDEN = ["verde", "pinton", "semi", "maduro", "muy_maduro"];
  const NOMBRE_VAR = { L: "L*", a: "a*", b: "b*", frecuencia_hz: "frecuencia", amplitud: "amplitud" };
  const LS_UMBRALES = "papaya.umbrales", LS_MASA = "papaya.masa";
  const SR_GRABACION = 44100;

  let umbrales = null, umbralesImportados = false;
  let modeloAcustico = null;  // model.json: solo su 'config' para los gráficos de detalle
  let tamModelo = 0;          // bytes de model.json (rendimiento)
  let caras = [];             // [{ L, a, b, fraccion, sup, fondo, ms, vista }]
  let golpes = [];            // [{ frecuencia_hz, amplitud, analisis, nombre, ms, ciclo_s }]
  let medColor = null;        // promedio de las caras: { L, a, b, fraccion }
  let medAudio = null;        // promedio de los golpes: { frecuencia_hz, amplitud, analisis, nombre, cv }
  let audioUrl = null;        // audio para "Escuchar"
  let grabando = false;
  let clima = null;           // último clima (PapayaClima.obtener)
  let sesion = null;          // fruto en curso: { id, codigoAuto }

  const color = (clase) => getComputedStyle(document.documentElement)
    .getPropertyValue(CLASES[clase] ? CLASES[clase].css : "--muted").trim();

  // Almacenamiento local protegido (puede no estar disponible).
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); return true; } catch { return false; } },
    del(k) { try { localStorage.removeItem(k); } catch { /* sin almacenamiento */ } },
  };

  const media = (v) => v.reduce((s, x) => s + x, 0) / v.length;
  const desv = (v) => { const m = media(v); return Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / (v.length - 1)); };
  const coma = (v, d) => (Number.isFinite(v) ? v.toFixed(d).replace(".", ",") : "—");
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  /** Estadio E1–E5 (Tabla 2) de una clase del clasificador. */
  const estadoDe = (clase) => E.estadioDeClase(clase, umbrales && umbrales.clases);
  const nombreClase = (clase) => (CLASES[clase] ? CLASES[clase].nombre : (estadoDe(clase) || {}).nombre || clase);
  const colorClase = (clase) => { const e = estadoDe(clase); return color(CLASES[clase] ? clase : e ? e.clase : clase); };
  const colorEstadio = (codigo) => { const e = E.ESTADIOS.find((x) => x.codigo === codigo); return e ? color(e.clase) : color(""); };

  // =======================================================================
  // Umbrales (regla de decisión)
  // =======================================================================
  async function cargarUmbrales() {
    const guardado = store.get(LS_UMBRALES);
    if (guardado) {
      try {
        const u = JSON.parse(guardado);
        if (B.umbralesValidos(u)) { umbrales = u; umbralesImportados = true; return; }
      } catch { /* se ignora y se usan los incluidos */ }
    }
    umbrales = await (await fetch("umbrales_app.json")).json();
    umbralesImportados = false;
  }

  const pct = (v) => `${Math.round(v * 100)} %`;
  const kb = (bytes) => `${coma(bytes / 1024, 1)} KB`;

  function pintarInfo() {
    const u = umbrales, ind = u.indice;
    const pesos = [...u.variables].sort((x, y) => u.pesos[y] - u.pesos[x])
      .map((v) => `${NOMBRE_VAR[v] || v} ${(u.pesos[v] * 100).toFixed(1)} %`).join(" · ");
    const filas = [
      ["Método", "Centroide más cercano ponderado (sin red neuronal); fusión intermedia de color + golpe"],
      ["Dataset", `${u.n_papayas} papayas · ${u.clases.map((c) => { const e = estadoDe(c); return `${e ? e.codigo + " " : ""}${nombreClase(c).toLowerCase()}`; }).join(", ")}`],
      ["Peso por variable", pesos],
      ["Exactitud", u.exactitud_deja_una_fuera != null ? `${pct(u.exactitud_deja_una_fuera)} (deja una papaya fuera)` : "—"],
      ["Índice 1–10", ind ? `${ind.intercepto.toFixed(2)} ${ind.pendiente >= 0 ? "+" : "−"} ${Math.abs(ind.pendiente).toFixed(3)}·a* (R² ${ind.r2.toFixed(2)}, ±${ind.mae_deja_una_fuera})` : "—"],
      ["Origen", umbralesImportados ? "importados en este celular" : "incluidos en la app"],
    ];
    $("info-modelo").innerHTML = filas.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
    $("btn-restaurar").hidden = !umbralesImportados;

    // Estado del entrenamiento frente a lo que pide el proyecto (OE1: 350 frutos).
    const n = u.n_papayas || 0;
    $("info-estado").textContent = n >= 350
      ? `Regla calculada con ${n} frutos de campo.`
      : `Resultado preliminar: la regla se calculó con ${n} papayas de campo` +
        (u.exactitud_deja_una_fuera != null ? ` (exactitud ${pct(u.exactitud_deja_una_fuera)})` : "") +
        ". El proyecto exige 350 frutos (70 por estadio) etiquetados por el panel y verificados con firmeza y °Brix. " +
        "Cuando tengas ese conjunto, recalcula umbrales_app.json en la laptop e impórtalo aquí. Los gráficos acústicos de detalle usan un modelo entrenado con audios sintéticos.";

    const t2 = parametrosTabla2();
    $("info-tabla2").innerHTML = E.ESTADIOS.map((e) => `
      <div class="t2-fila">
        <span class="t2-cod" style="background:${color(e.clase)}">${e.codigo}</span>
        <div><p>${e.cascara}</p><small>${e.uso}</small></div>
      </div>`).join("") +
      `<p class="hint">Un píxel de la cáscara cuenta como amarillo o anaranjado si h° &lt; ${t2.h_amarillo}° y C* ≥ ${t2.croma_min}. E1 tolera hasta ${t2.tolerancia_e1} % por ruido. Estos valores se pueden ajustar con el bloque <code>tabla2</code> de umbrales_app.json.</p>`;

    const mem = performance && performance.memory ? performance.memory.usedJSHeapSize : null;
    const ultCara = caras[caras.length - 1], ultGolpe = golpes[golpes.length - 1];
    const nHist = H.leer().length;
    const rend = [
      ["Regla (umbrales_app.json)", kb(JSON.stringify(u).length)],
      ["Modelo acústico (model.json)", tamModelo ? kb(tamModelo) : "—"],
      ["Memoria JavaScript en uso", mem ? `${coma(mem / 1048576, 1)} MB` : "no disponible"],
      ["Última foto", ultCara ? `${Math.round(ultCara.ms)} ms` : "—"],
      ["Último golpe", ultGolpe ? `${Math.round(ultGolpe.ms)} ms de proceso${ultGolpe.ciclo_s != null ? ` · ciclo ${coma(ultGolpe.ciclo_s, 2)} s` : ""}` : "—"],
      ["Meta del ciclo captura–inferencia", `≤ ${META_CICLO_S} s`],
      ["Historial", `${nHist} fruto${nHist === 1 ? "" : "s"} · ${kb((store.get("papaya.historial") || "").length)}`],
    ];
    $("info-rendimiento").innerHTML = rend.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");

    const m = Cl.MODELO;
    $("info-clima").textContent =
      `Datos de Open-Meteo (gratuito, sin clave), se actualizan cada 30 min; sin conexión se usa el último dato guardado. ` +
      `La papaya es climatérica: madura más rápido con calor. Velocidad relativa r(T) = Q10^((T − ${m.T_REF})/10), con Q10 = ${coma(m.Q10, 1)} y T acotada a ${m.T_MIN}–${m.T_MAX} °C. ` +
      `Días de referencia a ${m.T_REF} °C hasta E5: ${Object.entries(m.DIAS_A_E5).filter(([, d]) => d > 0).map(([k, d]) => `${k} ${d}`).join(" · ")}. ` +
      `Son valores orientativos: ajústalos con el seguimiento de frutos en Tarapoto.`;
  }

  // =======================================================================
  // Clima de la localidad
  // =======================================================================
  const hoyISO = () => new Date().toLocaleDateString("en-CA");

  function hace(ts) {
    const min = Math.round((Date.now() - ts) / 60000);
    if (min < 1) return "actualizado ahora";
    if (min < 60) return `actualizado hace ${min} min`;
    const h = Math.round(min / 60);
    return h < 48 ? `actualizado hace ${h} h` : `actualizado hace ${Math.round(h / 24)} días`;
  }

  function pintarClima() {
    const u = Cl.ubicacion();
    $("clima-lugar").textContent = (clima && clima.lugar) || u.lugar;
    $("btn-tarapoto").hidden = !u.gps;
    if (!clima) {
      $("clima-t").textContent = "—";
      $("clima-desc").textContent = "";
      $("clima-factor").textContent = "—";
      $("clima-detalle").textContent = "Sin datos de clima (sin conexión). La medición funciona igual.";
      $("clima-dias").hidden = true;
      return;
    }
    $("clima-t").textContent = `${Math.round(clima.temp)} °C`;
    $("clima-desc").textContent = Cl.describir(clima.codigo);
    const hoy = clima.dias.find((d) => d.fecha === hoyISO()) || clima.dias[0];
    $("clima-detalle").textContent =
      `Humedad ${Math.round(clima.hr)} %` +
      (hoy ? ` · hoy ${Math.round(hoy.tmin)}–${Math.round(hoy.tmax)} °C` : "") +
      ` · ${hace(clima.ts)}${clima.sin_conexion ? " (sin conexión)" : ""}`;
    const f = Cl.factorHoy(clima);
    $("clima-factor").textContent = f != null ? `×${coma(f, 1)}` : "—";
    $("clima-factor").style.color = f == null ? "" : f >= 1.15 ? "var(--maduro)" : f <= 0.9 ? "var(--blue)" : "var(--verde)";

    const futuros = clima.dias.filter((d) => d.fecha >= hoyISO()).slice(0, 7);
    $("clima-dias").innerHTML = futuros.map((d) => {
      const fm = Cl.factor((d.tmax + d.tmin) / 2);
      const dia = new Date(`${d.fecha}T12:00:00`).toLocaleDateString("es-PE", { weekday: "short" }).replace(".", "");
      return `<div class="dia">
        <span class="dia-n">${d.fecha === hoyISO() ? "hoy" : dia}</span>
        <b>${Math.round(d.tmax)}°</b><small>${Math.round(d.tmin)}°</small>
        <i style="opacity:${Math.min(1, Math.max(0.25, (fm - 0.6) / 0.9))}"></i>
      </div>`;
    }).join("");
    $("clima-dias").hidden = !futuros.length;
  }

  async function cargarClima(forzar = false) {
    if (!clima) $("clima-detalle").textContent = "Cargando el clima…";
    clima = await Cl.obtener(forzar);
    pintarClima();
    if (!$("resultado").hidden) actualizar({ desplazar: false });
  }

  // =======================================================================
  // 1. Foto de la cáscara
  // =======================================================================
  function cargarImagen(file) {
    return new Promise((ok, mal) => {
      const url = URL.createObjectURL(file), img = new Image();
      img.onload = () => { URL.revokeObjectURL(url); ok(img); };
      img.onerror = () => { URL.revokeObjectURL(url); mal(new Error("No se pudo abrir la imagen.")); };
      img.src = url;
    });
  }

  const parametrosTabla2 = () => Object.assign({}, E.PARAMS, (umbrales && umbrales.tabla2) || {});

  async function desdeFoto(file) {
    ocultarError();
    $("foto-estado").textContent = "Analizando la foto…";
    await new Promise((r) => setTimeout(r, 30)); // deja pintar "Analizando…"
    const t0 = performance.now();
    try {
      const img = await cargarImagen(file);
      // Igual que vision_color.cargar_imagen: se reduce a 1024 px de ancho.
      const esc = Math.min(1, B.ANCHO_TRABAJO / img.naturalWidth);
      const w = Math.round(img.naturalWidth * esc), h = Math.round(img.naturalHeight * esc);
      // Canvas en Display P3: las fotos del celular vienen en P3 y así se
      // leen sin recortar su gama; una foto sRGB se convierte a P3 sin pérdida.
      // Si el WebView no soporta P3, se trabaja en sRGB.
      const cv = document.createElement("canvas");
      cv.width = w; cv.height = h;
      let ctx = null;
      try { ctx = cv.getContext("2d", { colorSpace: "display-p3", willReadFrequently: true }); } catch { /* sin P3 */ }
      if (!ctx) ctx = cv.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(img, 0, 0, w, h);
      let datos;
      try { datos = ctx.getImageData(0, 0, w, h, { colorSpace: "display-p3" }); }
      catch { datos = ctx.getImageData(0, 0, w, h); }
      const espacio = datos.colorSpace || "srgb";
      const c = B.colorDesdeImagen(datos.data, w, h, espacio);
      const sup = E.analizarSuperficie(datos.data, c.mascara, w, h, espacio, parametrosTabla2());
      const fondo = E.controlFondo(datos.data, c.mascara, w, h, espacio);
      const ms = performance.now() - t0;

      if (caras.length >= MAX_CARAS) caras.shift(); // se reemplaza la cara más antigua
      caras.push({ L: c.L, a: c.a, b: c.b, fraccion: c.fraccion, sup, fondo, ms, vista: { datos, mascara: c.mascara } });
      recalcular();
      pintarFoto(datos, c.mascara, sup);
      pintarCaras();
      $("foto-estado").textContent =
        `Cara ${caras.length}: L* ${c.L.toFixed(1)} · a* ${c.a.toFixed(1)} · b* ${c.b.toFixed(1)} · ${coma(sup.pct_amarilla, 0)} % amarilla — zona clara = cáscara analizada` +
        (caras.length < MAX_CARAS ? ". Gira la papaya y fotografía la otra cara." : ".");
      pintarAvisoFondo(fondo);
      actualizar();
    } catch (e) {
      $("foto-estado").textContent = "Sobre papel blanco, con el fruto ocupando buena parte de la imagen y sin flash. Fotografía las dos caras del fruto.";
      mostrarError(e.message || String(e));
    }
  }

  function pintarAvisoFondo(fondo) {
    const nota = $("foto-fondo");
    if (!fondo) {
      nota.textContent = "No se detectó el fondo blanco: coloca la papaya sobre papel blanco para que el color sea comparable.";
      nota.hidden = false;
    } else if (fondo.dominante > 6) {
      nota.textContent = `El papel blanco se ve ${fondo.calido ? "amarillento" : "azulado"} (desvío de color ${coma(fondo.dominante, 1)}): la luz altera el color medido. Usa luz blanca pareja, sin sol directo ni focos cálidos.`;
      nota.hidden = false;
    } else nota.hidden = true;
  }

  /** Vista previa: la zona fuera de la máscara se aclara para que se vea
   *  qué píxeles se promediaron; encima, las divisiones de las 3 zonas. */
  function pintarFoto(datos, mascara, sup) {
    const { width: w, height: h } = datos;
    let out;
    try { out = new ImageData(new Uint8ClampedArray(datos.data), w, h, { colorSpace: datos.colorSpace || "srgb" }); }
    catch { out = new ImageData(new Uint8ClampedArray(datos.data), w, h); }
    const d = out.data;
    for (let i = 0; i < mascara.length; i++) {
      if (mascara[i]) continue;
      const j = 4 * i;
      d[j] = 0.35 * d[j] + 165; d[j + 1] = 0.35 * d[j + 1] + 165; d[j + 2] = 0.35 * d[j + 2] + 165;
    }
    const cv = $("foto-canvas");
    cv.width = w; cv.height = h;
    let ctx = null;
    try { ctx = cv.getContext("2d", { colorSpace: datos.colorSpace || "srgb" }); } catch { /* sin P3 */ }
    ctx = ctx || cv.getContext("2d");
    ctx.putImageData(out, 0, 0);
    if (sup) dibujarZonas(ctx, sup, w);
    $("foto-vista").hidden = false;
  }

  function dibujarZonas(ctx, sup, w) {
    const lw = Math.max(2, Math.round(w / 280));
    ctx.save();
    ctx.lineCap = "round";
    for (const l of sup.lineas) {
      ctx.beginPath(); ctx.moveTo(l.x1, l.y1); ctx.lineTo(l.x2, l.y2);
      ctx.strokeStyle = "rgba(0,0,0,.35)"; ctx.lineWidth = lw * 2.2; ctx.stroke();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = lw; ctx.stroke();
    }
    ctx.font = `700 ${Math.round(w / 24)}px system-ui, sans-serif`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.lineJoin = "round"; ctx.lineWidth = lw * 2; ctx.strokeStyle = "rgba(0,0,0,.6)"; ctx.fillStyle = "#fff";
    for (const z of sup.zonas) { ctx.strokeText(z.corto, z.x, z.y); ctx.fillText(z.corto, z.x, z.y); }
    ctx.restore();
  }

  function pintarCaras() {
    $("caras").innerHTML = caras.map((c, i) => `
      <span class="chip">
        <span class="dot" style="background:${E.labACss(c.L, c.a, c.b)}"></span>
        Cara ${i + 1} · ${coma(c.sup.pct_amarilla, 0)} % amarilla
        <button data-quitar-cara="${i}" aria-label="Quitar cara ${i + 1}">✕</button>
      </span>`).join("");
    $("txt-camara").textContent = !caras.length ? "Tomar foto" : caras.length < MAX_CARAS ? "Foto de la otra cara" : "Repetir foto";
    resumenFruto();
  }

  function quitarCara(i) {
    caras.splice(i, 1);
    recalcular(); pintarCaras();
    const ult = caras[caras.length - 1];
    if (ult) { pintarFoto(ult.vista.datos, ult.vista.mascara, ult.sup); pintarAvisoFondo(ult.fondo); }
    else { $("foto-vista").hidden = true; $("foto-fondo").hidden = true; }
    actualizar({ desplazar: false });
  }

  // =======================================================================
  // 2. Golpe: masa, grabación y archivo
  // =======================================================================
  /** Masa opcional: solo interviene en S y EI del detalle acústico. */
  function leerMasa() {
    const v = parseFloat(String($("masa").value).replace(",", "."));
    if (!Number.isFinite(v) || v <= 0 || v > 5000) return NaN;
    store.set(LS_MASA, String(v));
    return v;
  }

  async function grabar() {
    if (grabando) return;
    ocultarError();
    const t0 = performance.now(); // inicio del ciclo captura–inferencia

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)
      return mostrarError("Este dispositivo no permite grabar audio desde la app.");

    let stream;
    try {
      // Sin cancelación de eco, supresión de ruido ni control automático de
      // ganancia: esos filtros deforman el golpe.
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
      });
    } catch (e) {
      return mostrarError("No hay permiso para usar el micrófono. Actívalo en Ajustes → Aplicaciones → Madurez de papaya → Permisos.");
    }

    let ctx;
    try { ctx = new AudioContext({ sampleRate: SR_GRABACION }); }
    catch { ctx = new AudioContext(); }

    const chunks = [];
    let nodo;
    const alRecibir = (buf) => {
      chunks.push(buf);
      let pico = 0; for (let i = 0; i < buf.length; i++) pico = Math.max(pico, Math.abs(buf[i]));
      $("nivel").style.width = Math.min(100, pico * 140) + "%";
      $("nivel").style.background = pico >= 0.99 ? "var(--maduro)" : "var(--verde)";
    };
    const fuente = ctx.createMediaStreamSource(stream);
    const silencio = ctx.createGain(); silencio.gain.value = 0;
    try {
      await ctx.audioWorklet.addModule("js/recorder-worklet.js");
      nodo = new AudioWorkletNode(ctx, "recorder-processor");
      nodo.port.onmessage = (e) => alRecibir(e.data);
    } catch {
      nodo = ctx.createScriptProcessor(4096, 1, 1); // respaldo para WebViews antiguos
      nodo.onaudioprocess = (e) => alRecibir(e.inputBuffer.getChannelData(0).slice(0));
    }
    fuente.connect(nodo); nodo.connect(silencio); silencio.connect(ctx.destination);

    grabando = true;
    const btn = $("btn-grabar");
    btn.classList.add("recording");
    $("estado").textContent = "Escuchando… ¡golpea la papaya ahora!";
    const tRec = performance.now();
    const tick = () => {
      if (!grabando) return;
      const f = Math.min(1, (performance.now() - tRec) / DURACION_MS);
      btn.style.setProperty("--progress", String(352 * (1 - f)));
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);

    await new Promise((r) => setTimeout(r, DURACION_MS));

    grabando = false;
    stream.getTracks().forEach((t) => t.stop());
    fuente.disconnect(); nodo.disconnect();
    const sr = ctx.sampleRate;
    await ctx.close();
    btn.classList.remove("recording");
    btn.style.removeProperty("--progress");
    $("nivel").style.width = "0";
    $("estado").textContent = "Analizando…";

    const n = chunks.reduce((a, c) => a + c.length, 0);
    const senal = new Float64Array(n);
    let o = 0; for (const c of chunks) { senal.set(c, o); o += c.length; }

    fijarAudio(new Blob([codificarWav(senal, sr)], { type: "audio/wav" }));
    setTimeout(() => analizarAudio(senal, sr, "grabación", t0), 30);
  }

  /** Float -> WAV PCM 16 bits (para poder escuchar la grabación). */
  function codificarWav(x, sr) {
    const buf = new ArrayBuffer(44 + x.length * 2), dv = new DataView(buf);
    const w = (o, s) => [...s].forEach((ch, i) => dv.setUint8(o + i, ch.charCodeAt(0)));
    w(0, "RIFF"); dv.setUint32(4, 36 + x.length * 2, true); w(8, "WAVE");
    w(12, "fmt "); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
    dv.setUint32(24, sr, true); dv.setUint32(28, sr * 2, true); dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
    w(36, "data"); dv.setUint32(40, x.length * 2, true);
    for (let i = 0; i < x.length; i++) dv.setInt16(44 + i * 2, Math.max(-1, Math.min(1, x[i])) * 32767, true);
    return buf;
  }

  function fijarAudio(blob) {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    audioUrl = URL.createObjectURL(blob);
  }

  async function desdeArchivo(file) {
    ocultarError();
    $("estado").textContent = "Analizando…";
    try {
      const { signal, sr } = P.parseWav(await file.arrayBuffer());
      fijarAudio(file);
      setTimeout(() => analizarAudio(signal, sr, file.name, null), 30);
    } catch (e) { mostrarError(e.message); $("estado").textContent = "Toca para grabar y golpea la papaya"; }
  }

  function analizarAudio(senal, sr, nombre, tCiclo) {
    try {
      const t0 = performance.now();
      const ac = B.acusticaDominante(senal, sr);
      // Detalle (ζ, f1, gráficos): si falla, la regla sigue funcionando.
      let analisis = null;
      try { analisis = P.analyze(senal, sr, leerMasa(), modeloAcustico, nombre); } catch { /* sin detalle */ }
      const ms = performance.now() - t0;
      if (golpes.length >= MAX_GOLPES) golpes.shift(); // se reemplaza el golpe más antiguo
      golpes.push({
        frecuencia_hz: ac.frecuencia_hz, amplitud: ac.amplitud, analisis, nombre, ms,
        ciclo_s: tCiclo != null ? (performance.now() - tCiclo) / 1000 : null,
      });
      recalcular();
      pintarGolpes();
      const k = golpes.length;
      $("estado").textContent = `Golpe ${k}: ${ac.frecuencia_hz.toFixed(0)} Hz. ` +
        (k < MAX_GOLPES ? `Toca para el golpe ${k + 1} de ${MAX_GOLPES}` : "Toca para repetir uno");
      actualizar();
    } catch (e) {
      $("estado").textContent = "Toca para grabar y golpea la papaya";
      mostrarError(e.message || String(e));
    }
  }

  function pintarGolpes() {
    $("golpes").innerHTML = golpes.map((g, i) => `
      <span class="chip">
        Golpe ${i + 1} · ${g.frecuencia_hz.toFixed(0)} Hz
        <button data-quitar-golpe="${i}" aria-label="Quitar golpe ${i + 1}">✕</button>
      </span>`).join("");
    const nota = $("golpes-nota");
    if (medAudio && medAudio.cv != null) {
      nota.textContent = `Frecuencia media ${medAudio.frecuencia_hz.toFixed(0)} Hz · variación entre golpes ${coma(medAudio.cv, 1)} % ` +
        (medAudio.cv <= 5 ? "(repetible)" : "(alta: revisa la altura del péndulo y la posición del fruto)");
      nota.hidden = false;
    } else nota.hidden = true;
    resumenFruto();
  }

  function quitarGolpe(i) {
    golpes.splice(i, 1);
    recalcular(); pintarGolpes();
    if (!golpes.length) $("estado").textContent = "Toca para grabar y golpea la papaya";
    actualizar({ desplazar: false });
  }

  /** Promedio de caras (color) y de golpes (acústica): entrada de la regla.
   *  Con una sola cara o un solo golpe es exactamente esa medición. */
  function recalcular() {
    medColor = caras.length
      ? { L: media(caras.map((c) => c.L)), a: media(caras.map((c) => c.a)), b: media(caras.map((c) => c.b)), fraccion: media(caras.map((c) => c.fraccion)) }
      : null;
    if (golpes.length) {
      const f = golpes.map((g) => g.frecuencia_hz), m = media(f), ult = golpes[golpes.length - 1];
      medAudio = {
        frecuencia_hz: m, amplitud: media(golpes.map((g) => g.amplitud)),
        analisis: ult.analisis, nombre: ult.nombre,
        cv: golpes.length > 1 ? (100 * desv(f)) / m : null,
      };
    } else medAudio = null;
  }

  /** Tabla 2 promediada sobre las caras fotografiadas. */
  function tabla2() {
    if (!caras.length) return null;
    const s = caras.map((c) => c.sup), p = parametrosTabla2();
    const pctA = media(s.map((x) => x.pct_amarilla));
    const zonas = [0, 1, 2].map((i) => {
      const z = s.map((x) => x.zonas[i]);
      const L = media(z.map((q) => q.L)), a = media(z.map((q) => q.a)), b = media(z.map((q) => q.b));
      return { nombre: z[0].nombre, L, a, b, C: E.croma(a, b), h: E.tono(a, b), pct_amarilla: media(z.map((q) => q.pct_amarilla)) };
    });
    const fondos = caras.map((c) => c.fondo).filter(Boolean);
    return {
      pct_amarilla: pctA, estadio: E.estadioPorAmarillo(pctA, p.tolerancia_e1), zonas,
      h_desv: media(s.map((x) => x.h_desv)), manchas: media(s.map((x) => x.manchas)),
      orientacion_clara: s.every((x) => x.orientacion_clara), caras: caras.length,
      fondo_dominante: fondos.length ? Math.max(...fondos.map((f) => f.dominante)) : null,
    };
  }

  function resumenFruto() {
    $("fruto-resumen").textContent =
      `${caras.length} cara${caras.length === 1 ? "" : "s"} · ${golpes.length} golpe${golpes.length === 1 ? "" : "s"}`;
  }

  // =======================================================================
  // 3. Regla de decisión y resultado
  // =======================================================================
  function actualizar({ desplazar = true } = {}) {
    if (!medColor && !medAudio) { $("resultado").hidden = true; return; }
    const x = {};
    if (medColor) Object.assign(x, { L: medColor.L, a: medColor.a, b: medColor.b });
    if (medAudio) Object.assign(x, { frecuencia_hz: medAudio.frecuencia_hz, amplitud: medAudio.amplitud });

    const r = B.clasificar(umbrales, x);
    const indice = B.indiceMadurez(umbrales, x);
    const t2 = tabla2();
    pintarResultado(r, indice, t2);
    pintarTabla2(t2, r);
    const proy = pintarProyeccion(r);
    guardarEnHistorial(r, indice, t2, proy);
    $("resultado").hidden = false;
    requestAnimationFrame(() => {
      pintarGraficos();
      if (desplazar) $("resultado").scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  const fmt = (v, d) => (Number.isFinite(v) ? v.toFixed(d) : "—");
  const fmtSci = (v) => (Number.isFinite(v) ? v.toExponential(2).replace("e+", "·10^") : "—");
  const usaTexto = () => (medColor && medAudio ? "color + golpe" : medColor ? "solo color" : "solo golpe");

  function pintarResultado(r, indice, t2) {
    const clase = r.label, est = estadoDe(clase);
    $("clase").textContent = est ? est.codigo : nombreClase(clase);
    $("clase").style.color = colorClase(clase);
    $("clase-nombre").textContent = est ? nombreClase(clase) : "";
    $("clase-desc").innerHTML = est ? `${est.cascara}<br><b>${est.uso}</b>` : "";
    $("indice").hidden = indice == null;
    if (indice != null) $("indice").textContent = `Índice de madurez ${indice.toFixed(1).replace(".", ",")} / 10`;
    $("confianza").textContent = `${Math.round(r.proba[clase] * 100)} % de confianza`;
    const orden = ORDEN.filter((c) => c in r.proba).concat(Object.keys(r.proba).filter((c) => !ORDEN.includes(c)));
    $("barras").innerHTML = orden.map((c) => {
      const e = estadoDe(c);
      return `
      <div class="bar-row ${c === clase ? "win" : ""}">
        <span>${e ? `<b class="cod">${e.codigo}</b> ` : ""}${nombreClase(c)}</span>
        <div class="track"><div class="fill" style="width:${(r.proba[c] * 100).toFixed(1)}%;background:${colorClase(c)}"></div></div>
        <span class="pct">${Math.round(r.proba[c] * 100)}%</span>
      </div>`;
    }).join("");

    $("origen").textContent = `Regla de umbrales · ${umbrales.n_papayas} papayas de campo · ${usaTexto()}` +
      ` · ${caras.length} cara${caras.length === 1 ? "" : "s"}, ${golpes.length} golpe${golpes.length === 1 ? "" : "s"}`;

    const nota = $("nota-resultado");
    if (!medColor) {
      const pesoAudio = (umbrales.pesos.frecuencia_hz || 0) + (umbrales.pesos.amplitud || 0);
      nota.textContent = `Solo con el golpe el resultado es poco confiable: en el dataset de campo el sonido aporta ${(pesoAudio * 100).toFixed(1)} % de la decisión. Toma también la foto.`;
      nota.hidden = false;
    } else if (!medAudio) {
      nota.textContent = "Solo con la foto: es la madurez externa. Graba el golpe para la estimación bimodal (color + firmeza).";
      nota.hidden = false;
    } else nota.hidden = true;

    const tarjetas = [];
    if (medColor) {
      const h = (Math.atan2(medColor.b, medColor.a) * 180) / Math.PI;
      tarjetas.push(
        ["a* (verde → rojo)", fmt(medColor.a, 1), "", "Variable clave: sube al madurar", true],
        ["b* (azul → amarillo)", fmt(medColor.b, 1), "", "Sube al aparecer el amarillo"],
        ["L* (luminosidad)", fmt(medColor.L, 1), "", "0 negro – 100 blanco"],
        ["Tono h°", fmt(h, 0), "°", "≈ 120° verde, 90° amarillo, 60° naranja"],
        ["Croma C*", fmt(E.croma(medColor.a, medColor.b), 1), "", "Saturación del color"],
      );
      if (t2) tarjetas.push(
        ["Superficie amarilla", fmt(t2.pct_amarilla, 0), "%", "Criterio de la Tabla 2 (E1–E5)", true],
        ["Desviación de h°", fmt(t2.h_desv, 1), "°", "Alta = colores mezclados"],
        ["Manchas", fmt(t2.manchas, 1), "%", "Zonas oscuras; el estudio excluye > 10 %"],
      );
    }
    if (medAudio) {
      const an = medAudio.analisis, f = an ? an.feats : {};
      const m = leerMasa(), EI = Number.isFinite(m) ? medAudio.frecuencia_hz ** 2 * Math.pow(m, 2 / 3) : NaN;
      tarjetas.push(
        ["Frecuencia dominante", fmt(medAudio.frecuencia_hz, 0), "Hz", golpes.length > 1 ? `Promedio de ${golpes.length} golpes (FFT, 100–3000 Hz)` : "Pico de la FFT del golpe (100–3000 Hz)"],
        ["Amplitud", fmt(20 * Math.log10(medAudio.amplitud), 1), "dBFS", "Intensidad del pico dominante"],
        ["Índice de elasticidad EI", fmtSci(EI), "", "EI = f²·m^(2/3); baja al ablandarse"],
        ["Amortiguación ζ", fmt(f.damping_logdec, 4), "", "Más maduro → se apaga antes"],
        ["Rigidez S", fmtSci(f.S), "", "S = f1²·m^(2/3) (requiere masa)"],
        ["Repetibilidad", medAudio.cv != null ? fmt(medAudio.cv, 1) : "—", medAudio.cv != null ? "%" : "", medAudio.cv != null ? "Variación de f entre golpes" : `Graba hasta ${MAX_GOLPES} golpes`],
      );
    }
    $("features").innerHTML = tarjetas.map(([n, v, u, d, star]) => `
      <div class="feat ${star ? "star" : ""}">
        <div class="name">${n}</div>
        <div class="value">${v.replace("·10^", "·10<sup>") + (v.includes("·10^") ? "</sup>" : "")}${u ? `<small>${u}</small>` : ""}</div>
        <div class="desc">${d}</div>
      </div>`).join("");

    // Rendimiento algorítmico (OE4): proceso y ciclo captura–inferencia.
    const ultCara = caras[caras.length - 1], ultGolpe = golpes[golpes.length - 1], partes = [];
    if (ultCara) partes.push(`Última foto: procesada en ${Math.round(ultCara.ms)} ms`);
    if (ultGolpe) {
      partes.push(`último golpe: procesado en ${Math.round(ultGolpe.ms)} ms`);
      if (ultGolpe.ciclo_s != null)
        partes.push(`ciclo captura–inferencia ${coma(ultGolpe.ciclo_s, 2)} s ${ultGolpe.ciclo_s <= META_CICLO_S ? "✓" : `(meta ≤ ${META_CICLO_S} s)`}`);
    }
    $("rendimiento").textContent = partes.join(" · ");

    $("btn-escuchar").hidden = !medAudio;
    $("detalle-audio").hidden = !(medAudio && medAudio.analisis);
  }

  /** Madurez externa por % de superficie amarilla (Tabla 2) y color por zona. */
  function pintarTabla2(t2, r) {
    $("bloque-tabla2").hidden = !t2;
    if (!t2) return;
    const e = t2.estadio, p = parametrosTabla2();
    const tramos = [[0, p.tolerancia_e1], [p.tolerancia_e1, 25], [25, 50], [50, 75], [75, 100]];
    const fusion = estadoDe(r.label);
    const notas = [];
    if (t2.caras < MAX_CARAS) notas.push("Fotografía también la otra cara: el informe mide las dos caras del fruto.");
    if (!t2.orientacion_clara) notas.push("Los extremos tienen un color parecido: peduncular y apical podrían estar intercambiados.");
    if (t2.manchas > 10) notas.push(`Manchas en ${coma(t2.manchas, 1)} % de la superficie: el protocolo excluye frutos con manchas > 10 %.`);
    if (fusion && medAudio && fusion.codigo !== e.codigo)
      notas.push(`La fusión color + golpe estima ${fusion.codigo}; por superficie amarilla corresponde a ${e.codigo}. Si difieren mucho, revisa la luz de la foto y repite el golpe.`);

    $("tabla2").innerHTML = `
      <div class="t2-head">
        <span class="t2-badge" style="background:${colorEstadio(e.codigo)}">${e.codigo}</span>
        <div>
          <p class="t2-pct">${coma(t2.pct_amarilla, 0)} % de superficie amarilla</p>
          <p class="t2-desc">${e.cascara}</p>
          <p class="t2-uso">${e.uso}</p>
        </div>
      </div>
      <div class="escala" aria-hidden="true">
        <div class="escala-barra">${tramos.map(([a, b], i) => `<span style="width:${b - a}%;background:${color(E.ESTADIOS[i].clase)}"></span>`).join("")}</div>
        <span class="escala-marca" style="left:${Math.min(100, Math.max(0, t2.pct_amarilla))}%"></span>
        <div class="escala-ticks"><span>0</span><span>25</span><span>50</span><span>75</span><span>100 %</span></div>
      </div>
      <div class="zonas">${t2.zonas.map((z) => `
        <div class="zona">
          <div class="swatch" style="background:${E.labACss(z.L, z.a, z.b)}"></div>
          <b>${z.nombre[0].toUpperCase() + z.nombre.slice(1)}</b>
          <small>h° ${coma(z.h, 0)} · C* ${coma(z.C, 0)}</small>
          <small>${coma(z.pct_amarilla, 0)} % amarilla</small>
        </div>`).join("")}
      </div>
      <p class="hint">${t2.caras} cara${t2.caras === 1 ? "" : "s"} · la papaya madura del ápice al pedúnculo, por eso el extremo más amarillo se toma como apical.</p>
      ${notas.map((n) => `<p class="result-note">${n}</p>`).join("")}`;
  }

  /** Días estimados hasta la madurez de consumo con el clima previsto. */
  function pintarProyeccion(r) {
    const est = estadoDe(r.label);
    const p = est && clima ? Cl.proyectar(est.codigo, clima) : null;
    $("bloque-proyeccion").hidden = !p;
    if (!p) return null;
    const fechaLarga = (d) => d.toLocaleDateString("es-PE", { weekday: "long", day: "numeric", month: "long" });
    const vence = new Date(p.fecha.getTime() + p.vida * 86400000);
    const lugar = esc(clima.lugar);
    const principal = est.codigo === "E5"
      ? `<p class="proy-big">Madurez de consumo</p>
         <p class="proy-txt">Con ${coma(p.tmedia, 1)} °C de media prevista en ${lugar}, se conserva bien ~${coma(p.vida, 1)} días (hasta el ${fechaLarga(vence)}).</p>`
      : `<p class="proy-big">~${coma(p.dias, 1)} días <small>hasta madurez de consumo (E5)</small></p>
         <p class="proy-txt">Hacia el ${fechaLarga(p.fecha)}, con ${coma(p.tmedia, 1)} °C de media prevista en ${lugar}: madura ×${coma(p.factor, 1)} respecto a 25 °C. Después se conserva ~${coma(p.vida, 1)} días.</p>`;
    $("proyeccion").innerHTML = principal +
      `<p class="proy-uso">Uso sugerido ahora (Tabla 2): <b>${est.uso}</b></p>` +
      p.alertas.map((a) => `<p class="result-note">${a}</p>`).join("") +
      `<p class="hint">Estimación orientativa: modelo Q10 con días de referencia a 25 °C y el pronóstico de 7 días${clima.sin_conexion ? ` (clima guardado, ${hace(clima.ts)})` : ""}.</p>`;
    return p;
  }

  // =======================================================================
  // 4. Historial
  // =======================================================================
  function nuevaSesion() {
    sesion = { id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`, codigoAuto: null };
    $("codigo").value = "";
  }

  function codigoActual() {
    const escrito = $("codigo").value.trim();
    if (escrito) return escrito;
    if (!sesion.codigoAuto) sesion.codigoAuto = H.nuevoCodigo();
    $("codigo").value = sesion.codigoAuto;
    return sesion.codigoAuto;
  }

  function guardarEnHistorial(r, indice, t2, proy) {
    const est = estadoDe(r.label);
    const ultCara = caras[caras.length - 1], ultGolpe = golpes[golpes.length - 1];
    const masa = leerMasa();
    const reg = {
      id: sesion.id, codigo: codigoActual(), fecha: new Date().toLocaleString("sv-SE").slice(0, 16),
      estadio: est ? est.codigo : null, clase: r.label, confianza: r.proba[r.label], proba: r.proba,
      origen: usaTexto(), indice,
      color: medColor && { L: medColor.L, a: medColor.a, b: medColor.b, C: E.croma(medColor.a, medColor.b), h: E.tono(medColor.a, medColor.b) },
      tabla2: t2 && {
        estadio: t2.estadio.codigo, pct_amarilla: t2.pct_amarilla, h_desv: t2.h_desv, manchas: t2.manchas,
        caras: t2.caras, fondo_dominante: t2.fondo_dominante,
        zonas: t2.zonas.map((z) => ({ nombre: z.nombre, L: z.L, a: z.a, b: z.b, C: z.C, h: z.h, pct_amarilla: z.pct_amarilla })),
      },
      audio: medAudio && {
        golpes: golpes.length, frecuencia_hz: medAudio.frecuencia_hz, cv_frecuencia: medAudio.cv,
        amplitud_dbfs: 20 * Math.log10(medAudio.amplitud),
        damping: medAudio.analisis ? medAudio.analisis.feats.damping_logdec : null,
        EI: Number.isFinite(masa) ? medAudio.frecuencia_hz ** 2 * Math.pow(masa, 2 / 3) : null,
        S: medAudio.analisis ? medAudio.analisis.feats.S : null,
      },
      masa_g: medAudio && Number.isFinite(masa) ? masa : null,
      clima: clima && {
        lugar: clima.lugar, lat: clima.lat, lon: clima.lon, temp: clima.temp, hr: clima.hr,
        tmedia: proy ? proy.tmedia : null, factor: proy ? proy.factor : null,
        dias_a_E5: proy ? proy.dias : null, fecha_E5: proy ? proy.fecha.toLocaleDateString("en-CA") : null,
        sin_conexion: !!clima.sin_conexion,
      },
      rendimiento: {
        foto_ms: ultCara ? ultCara.ms : null, golpe_ms: ultGolpe ? ultGolpe.ms : null,
        ciclo_golpe_s: ultGolpe ? ultGolpe.ciclo_s : null,
      },
      version: 2,
    };
    const ok = H.guardar(reg);
    const g = $("guardado");
    g.textContent = ok ? `✓ Guardado en el historial como ${reg.codigo}` : "No se pudo guardar en el historial (almacenamiento lleno o bloqueado).";
    g.classList.toggle("bad", !ok);
    g.hidden = false;
    pintarContador();
  }

  function pintarContador() {
    const n = H.leer().length, b = $("hist-contador");
    b.textContent = n > 99 ? "99+" : String(n);
    b.hidden = !n;
  }

  function pintarHistorial() {
    const lista = H.leer();
    const porEstadio = E.ESTADIOS.map((e) => [e, lista.filter((r) => r.estadio === e.codigo).length]);
    $("hist-resumen").innerHTML = lista.length
      ? `<b>${lista.length} fruto${lista.length === 1 ? "" : "s"}</b>` + porEstadio.map(([e, n]) =>
        `<span class="mini-chip"><i style="background:${color(e.clase)}"></i>${e.codigo} ${n}</span>`).join("")
      : "";
    $("hist-lista").innerHTML = lista.length ? lista.map((r) => {
      const partes = [];
      if (r.tabla2) partes.push(`${coma(r.tabla2.pct_amarilla, 0)} % amarilla`);
      if (r.audio) partes.push(`${Math.round(r.audio.frecuencia_hz)} Hz`);
      if (r.clima && Number.isFinite(r.clima.temp)) partes.push(`${Math.round(r.clima.temp)} °C`);
      if (r.clima && Number.isFinite(r.clima.dias_a_E5) && r.clima.dias_a_E5 > 0) partes.push(`E5 en ~${coma(r.clima.dias_a_E5, 1)} d`);
      const sw = r.color ? E.labACss(r.color.L, r.color.a, r.color.b) : "var(--surface-2)";
      return `<div class="hist-item">
        <span class="hist-sw" style="background:${sw}"></span>
        <div class="hist-info">
          <p><b>${esc(r.codigo)}</b> <span class="muted">· ${esc(r.fecha)}</span></p>
          <p class="hist-det">${partes.join(" · ") || esc(r.origen)}</p>
        </div>
        <span class="hist-est" style="background:${colorEstadio(r.estadio)}">${esc(r.estadio || "—")}</span>
        <button class="hist-del" data-borrar="${esc(r.id)}" aria-label="Borrar ${esc(r.codigo)}">✕</button>
      </div>`;
    }).join("") : `<p class="hint vacio">Aún no hay mediciones guardadas. Cada fruto se guarda solo al obtener su resultado.</p>`;
    $("btn-exportar").disabled = !lista.length;
    $("btn-borrar-todo").disabled = !lista.length;
  }

  // =======================================================================
  // Gráficos acústicos
  // =======================================================================
  function pintarGraficos() {
    const r = medAudio && medAudio.analisis;
    if (!r || !modeloAcustico) return;
    const cfg = modeloAcustico.config, css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
    const naranja = css("--accent"), azul = css("--blue"), gris = css("--muted");

    // --- Espectro ---
    const fMax = cfg.FREQ_MAX + 500;
    let mMax = 0;
    for (let i = 0; i < r.freqs.length && r.freqs[i] <= fMax; i++) mMax = Math.max(mMax, r.mag[i]);
    const magN = r.mag.map((v) => v / (mMax || 1));
    const f1 = r.feats.f1, zbw = r.feats.damping_bw;
    const bandas = [{ x0: cfg.FREQ_MIN, x1: cfg.FREQ_MAX, color: "rgba(28,126,214,.05)" }];
    const hl = [];
    if (Number.isFinite(f1) && Number.isFinite(zbw)) {
      const bw = 2 * zbw * f1;
      bandas.push({ x0: f1 - bw / 2, x1: f1 + bw / 2, color: "rgba(232,89,12,.22)" });
      let k = 0, best = Infinity;
      r.freqs.forEach((f, i) => { const d = Math.abs(f - f1); if (d < best) { best = d; k = i; } });
      hl.push({ y: magN[k] / Math.SQRT2 });
    }
    C.lineChart($("ch-espectro"), {
      series: [{ x: r.freqs, y: magN, color: azul, width: 1.6 }],
      xMin: 0, xMax: fMax, yMin: 0, yMax: 1.05, xLabel: "frecuencia (Hz)",
      bands: bandas, hlines: hl,
      vlines: r.peakFreqs.slice(0, 2).map((f, i) => ({ x: f, label: `f${i + 1} ${Math.round(f)} Hz`, color: naranja, row: i })),
    });

    // --- Decaimiento: impulso + envolvente + ajuste exponencial ---
    const n = r.impulse.length, t = new Float64Array(n);
    for (let i = 0; i < n; i++) t[i] = i / r.sr;
    const series = [
      { x: t, y: r.impulse, color: gris, width: 1, minmax: true },
      { x: t, y: r.env, color: azul, width: 2 },
    ];
    const z = r.feats.damping_logdec;
    if (Number.isFinite(z) && Number.isFinite(f1)) {
      let k = 0; for (let i = 1; i < n; i++) if (r.env[i] > r.env[k]) k = i;
      const tt = [], yy = [];
      for (let i = k; i < n; i += 4) { tt.push(t[i]); yy.push(r.env[k] * Math.exp(-z * 2 * Math.PI * f1 * (t[i] - t[k]))); }
      series.push({ x: tt, y: yy, color: naranja, width: 2, dash: [6, 4] });
    }
    // Zoom al tramo útil: hasta que la envolvente cae al 2 % del pico (+60 %).
    let kp = 0; for (let i = 1; i < n; i++) if (r.env[i] > r.env[kp]) kp = i;
    let fin = n - 1;
    for (let i = kp; i < n; i++) if (r.env[i] < 0.02 * r.env[kp]) { fin = i; break; }
    const tFin = Math.min(n / r.sr, Math.max(0.02, (kp + (fin - kp) * 1.6) / r.sr));
    C.lineChart($("ch-decaimiento"), {
      series, xMin: 0, xMax: tFin, yMin: -1.05, yMax: 1.05, xLabel: "tiempo desde el golpe (s)",
    });

    // --- Grabación completa ---
    const N = r.signal.length, ts = new Float64Array(N);
    let a = 0; for (let i = 0; i < N; i++) { ts[i] = i / r.sr; a = Math.max(a, Math.abs(r.signal[i])); }
    a = a || 1;
    C.lineChart($("ch-grabacion"), {
      series: [{ x: ts, y: r.signal, color: gris, width: 1, minmax: true }],
      xMin: 0, xMax: N / r.sr, yMin: -a * 1.05, yMax: a * 1.05, xLabel: "tiempo (s)",
      bands: [{ x0: r.start / r.sr, x1: (r.start + n) / r.sr, color: "rgba(232,89,12,.16)" }],
      vlines: [{ x: r.onset / r.sr, label: "golpe", color: naranja }],
    });
  }

  function mostrarError(msg) {
    $("error-texto").textContent = msg;
    $("error").hidden = false;
    $("error").scrollIntoView({ behavior: "smooth", block: "center" });
  }
  function ocultarError() { $("error").hidden = true; }

  // =======================================================================
  // Eventos
  // =======================================================================
  function enlazar() {
    const masaGuardada = store.get(LS_MASA);
    if (masaGuardada) $("masa").value = masaGuardada;

    document.querySelectorAll(".step").forEach((b) => b.addEventListener("click", () => {
      const v = parseFloat($("masa").value) || 0;
      $("masa").value = Math.max(1, Math.round(v + Number(b.dataset.step)));
    }));
    $("btn-grabar").addEventListener("click", grabar);
    const alElegir = (fn) => (e) => {
      const f = e.target.files[0]; e.target.value = "";
      if (f) fn(f);
    };
    $("archivo").addEventListener("change", alElegir(desdeArchivo));
    $("foto-camara").addEventListener("change", alElegir(desdeFoto));
    $("foto-galeria").addEventListener("change", alElegir(desdeFoto));
    $("caras").addEventListener("click", (e) => {
      const b = e.target.closest("[data-quitar-cara]");
      if (b) quitarCara(Number(b.dataset.quitarCara));
    });
    $("golpes").addEventListener("click", (e) => {
      const b = e.target.closest("[data-quitar-golpe]");
      if (b) quitarGolpe(Number(b.dataset.quitarGolpe));
    });
    // Cambiar el código o la masa con el resultado a la vista actualiza el registro.
    const reGuardar = () => { if (!$("resultado").hidden) actualizar({ desplazar: false }); };
    $("codigo").addEventListener("change", reGuardar);
    $("masa").addEventListener("change", reGuardar);

    $("btn-nueva").addEventListener("click", () => {
      medColor = null; medAudio = null; caras = []; golpes = [];
      nuevaSesion();
      pintarCaras(); pintarGolpes();
      $("resultado").hidden = true; $("foto-vista").hidden = true; $("foto-fondo").hidden = true; ocultarError();
      $("foto-estado").textContent = "Sobre papel blanco, con el fruto ocupando buena parte de la imagen y sin flash. Fotografía las dos caras del fruto.";
      $("estado").textContent = "Toca para grabar y golpea la papaya";
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
    $("btn-escuchar").addEventListener("click", () => { if (audioUrl) new Audio(audioUrl).play(); });

    const hoja = $("hoja");
    $("btn-info").addEventListener("click", () => { pintarInfo(); hoja.hidden = false; });
    $("btn-cerrar").addEventListener("click", () => (hoja.hidden = true));
    hoja.addEventListener("click", (e) => { if (e.target === hoja) hoja.hidden = true; });

    // Historial
    const hojaH = $("hoja-historial");
    $("btn-historial").addEventListener("click", () => { pintarHistorial(); hojaH.hidden = false; });
    $("btn-cerrar-hist").addEventListener("click", () => (hojaH.hidden = true));
    hojaH.addEventListener("click", (e) => { if (e.target === hojaH) hojaH.hidden = true; });
    $("hist-lista").addEventListener("click", (e) => {
      const b = e.target.closest("[data-borrar]");
      if (!b) return;
      const r = H.leer().find((x) => x.id === b.dataset.borrar);
      if (r && confirm(`¿Borrar la medición ${r.codigo}?`)) { H.borrar(r.id); pintarHistorial(); pintarContador(); }
    });
    $("btn-borrar-todo").addEventListener("click", () => {
      const n = H.leer().length;
      if (n && confirm(`¿Borrar las ${n} mediciones del historial? Exporta el CSV antes si las necesitas.`)) {
        H.limpiar(); pintarHistorial(); pintarContador();
      }
    });
    $("btn-exportar").addEventListener("click", async () => {
      try { await H.exportar(); }
      catch (err) { alert(err && err.message ? err.message : "No se pudo exportar el historial."); }
    });

    // Clima
    $("btn-clima").addEventListener("click", () => cargarClima(true));
    $("btn-ubicacion").addEventListener("click", async () => {
      const b = $("btn-ubicacion");
      b.disabled = true; $("clima-detalle").textContent = "Buscando tu ubicación…";
      try { await Cl.usarMiUbicacion(); await cargarClima(true); }
      catch (err) { pintarClima(); $("clima-detalle").textContent = err.message; }
      finally { b.disabled = false; }
    });
    $("btn-tarapoto").addEventListener("click", () => { Cl.usarTarapoto(); cargarClima(true); });
    document.addEventListener("visibilitychange", () => { if (!document.hidden) cargarClima(false); });

    $("importar").addEventListener("change", async (e) => {
      const f = e.target.files[0]; e.target.value = "";
      if (!f) return;
      try {
        const u = JSON.parse(await f.text());
        if (!B.umbralesValidos(u)) throw new Error();
        if (!store.set(LS_UMBRALES, JSON.stringify(u)))
          throw new Error("No se pudieron guardar los umbrales en el celular.");
        umbrales = u; umbralesImportados = true;
        pintarInfo(); if (medColor || medAudio) actualizar({ desplazar: false });
        alert(`Umbrales importados: ${u.n_papayas} papayas, clases ${u.clases.join(", ")}.`);
      } catch (err) {
        alert(err.message || "El archivo no es un umbrales_app.json válido (genéralo con python bimodal/calcular_umbrales.py).");
      }
    });
    $("btn-restaurar").addEventListener("click", async () => {
      store.del(LS_UMBRALES);
      await cargarUmbrales();
      pintarInfo(); if (medColor || medAudio) actualizar({ desplazar: false });
    });

    let t;
    window.addEventListener("resize", () => {
      clearTimeout(t);
      t = setTimeout(() => { if (!$("resultado").hidden) pintarGraficos(); }, 150);
    });
  }

  (async function iniciar() {
    enlazar();
    nuevaSesion();
    pintarContador();
    pintarClima();
    cargarClima(false);
    try {
      await cargarUmbrales();
    } catch (e) {
      mostrarError("No se pudo cargar la regla de decisión (umbrales_app.json).");
    }
    try {
      const txt = await (await fetch("model.json")).text();
      tamModelo = txt.length;
      modeloAcustico = JSON.parse(txt);
    } catch { /* sin gráficos de detalle acústico */ }
  })();
})();
