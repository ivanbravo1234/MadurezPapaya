/*
 * clima.js — Clima de la localidad y velocidad de maduración.
 *
 * Datos: Open-Meteo (https://open-meteo.com), gratuito y sin clave.
 * Temperatura y humedad actuales + pronóstico de 7 días. Se guarda en el
 * celular para seguir mostrando el último dato sin conexión.
 * Ubicación por defecto: Tarapoto (ámbito del estudio); el botón
 * "Usar mi ubicación" usa el GPS del celular.
 *
 * Velocidad de maduración (modelo Q10, orientativo):
 *   La papaya es climatérica: su respiración y su maduración se aceleran con
 *   la temperatura. La velocidad relativa a 25 °C es
 *       r(T) = Q10 ^ ((T − 25) / 10)
 *   con T acotada a 12–32 °C (debajo de ~12 °C hay daño por frío y la
 *   maduración se detiene o es anormal; sobre ~32 °C la pulpa se ablanda
 *   sin colorear bien). Los días que faltan para la madurez de consumo (E5)
 *   se calculan sumando r(T) de cada día del pronóstico hasta completar los
 *   días que tomaría a 25 °C. Los valores de DIAS_A_E5 son referenciales y
 *   deben reemplazarse con el seguimiento propio de frutos en Tarapoto.
 */
(function (global) {
  "use strict";

  const LS_CLIMA = "papaya.clima", LS_UBIC = "papaya.ubicacion";
  const TARAPOTO = { lat: -6.4833, lon: -76.3667, lugar: "Tarapoto, San Martín", gps: false };
  const VIGENCIA_MS = 30 * 60 * 1000;

  const MODELO = {
    T_REF: 25, Q10: 2.5, T_MIN: 12, T_MAX: 32,
    // Días aproximados a 25 °C desde cada estadio hasta E5 (madurez de consumo).
    DIAS_A_E5: { E1: 9, E2: 6, E3: 4, E4: 2, E5: 0 },
    VIDA_E5: 3, // días de consumo aceptable una vez en E5, a 25 °C
  };

  const store = {
    get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* sin almacenamiento */ } },
  };

  function ubicacion() {
    const u = store.get(LS_UBIC);
    return u && Number.isFinite(u.lat) && Number.isFinite(u.lon) ? u : TARAPOTO;
  }

  function usarTarapoto() { store.set(LS_UBIC, TARAPOTO); return TARAPOTO; }

  async function conTiempoLimite(url, ms) {
    const ctl = typeof AbortController !== "undefined" ? new AbortController() : null;
    const t = setTimeout(() => ctl && ctl.abort(), ms);
    try {
      const r = await fetch(url, ctl ? { signal: ctl.signal } : undefined);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } finally { clearTimeout(t); }
  }

  /** Pide el GPS, busca el nombre del lugar y lo guarda como ubicación. */
  async function usarMiUbicacion() {
    if (!navigator.geolocation) throw new Error("Este dispositivo no da acceso a la ubicación.");
    const pos = await new Promise((ok, mal) => navigator.geolocation.getCurrentPosition(ok, mal, {
      enableHighAccuracy: false, timeout: 15000, maximumAge: 10 * 60 * 1000,
    })).catch((e) => {
      throw new Error(e && e.code === 1
        ? "No hay permiso de ubicación. Actívalo en Ajustes → Aplicaciones → Madurez de papaya → Permisos."
        : "No se pudo obtener la ubicación. Revisa que el GPS esté encendido.");
    });
    const lat = pos.coords.latitude, lon = pos.coords.longitude;
    let lugar = `${lat.toFixed(3)}, ${lon.toFixed(3)}`;
    try {
      const g = await conTiempoLimite(
        `https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${lon}&localityLanguage=es`, 6000);
      const ciudad = g.city || g.locality, region = g.principalSubdivision;
      if (ciudad || region) lugar = [ciudad, region].filter(Boolean).join(", ");
    } catch { /* sin nombre: quedan las coordenadas */ }
    const u = { lat, lon, lugar, gps: true };
    store.set(LS_UBIC, u);
    return u;
  }

  /** Clima actual + 7 días. Si no hay red devuelve el último guardado
   *  (con sin_conexion = true) o null. */
  async function obtener(forzar = false) {
    const u = ubicacion(), previo = store.get(LS_CLIMA);
    const mismoLugar = previo && Math.abs(previo.lat - u.lat) < 0.01 && Math.abs(previo.lon - u.lon) < 0.01;
    if (!forzar && mismoLugar && Date.now() - previo.ts < VIGENCIA_MS) return previo;
    const url = "https://api.open-meteo.com/v1/forecast"
      + `?latitude=${u.lat}&longitude=${u.lon}`
      + "&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code"
      + "&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,weather_code"
      + "&timezone=auto&forecast_days=7";
    try {
      const d = await conTiempoLimite(url, 10000);
      const c = {
        ts: Date.now(), lat: u.lat, lon: u.lon, lugar: u.lugar, gps: !!u.gps,
        temp: d.current.temperature_2m, sensacion: d.current.apparent_temperature,
        hr: d.current.relative_humidity_2m, codigo: d.current.weather_code,
        dias: d.daily.time.map((fecha, i) => ({
          fecha,
          tmax: d.daily.temperature_2m_max[i], tmin: d.daily.temperature_2m_min[i],
          hr: d.daily.relative_humidity_2m_mean ? d.daily.relative_humidity_2m_mean[i] : null,
          codigo: d.daily.weather_code ? d.daily.weather_code[i] : null,
        })),
      };
      store.set(LS_CLIMA, c);
      return c;
    } catch {
      return previo ? Object.assign({}, previo, { sin_conexion: true }) : null;
    }
  }

  // =========================================================================
  // Velocidad de maduración
  // =========================================================================
  const factor = (T) => Math.pow(MODELO.Q10, (Math.min(MODELO.T_MAX, Math.max(MODELO.T_MIN, T)) - MODELO.T_REF) / 10);
  const media = (d) => (d.tmax + d.tmin) / 2;
  const hoyISO = () => new Date().toLocaleDateString("en-CA"); // AAAA-MM-DD en hora local

  /** Temperaturas medias de hoy en adelante (pronóstico). */
  function temperaturasFuturas(c) {
    if (!c || !c.dias) return [];
    const hoy = hoyISO();
    const fut = c.dias.filter((d) => d.fecha >= hoy && Number.isFinite(d.tmax) && Number.isFinite(d.tmin)).map(media);
    return fut.length ? fut : c.dias.filter((d) => Number.isFinite(d.tmax)).map(media);
  }

  /** Factor de hoy (o de la temperatura actual si no hay pronóstico). */
  function factorHoy(c) {
    const t = temperaturasFuturas(c);
    return t.length ? factor(t[0]) : c && Number.isFinite(c.temp) ? factor(c.temp) : null;
  }

  /** Días estimados hasta E5 desde el estadio dado, con el pronóstico. */
  function proyectar(codigoEstadio, c) {
    const temps = temperaturasFuturas(c);
    if (!temps.length) return null;
    const prom = temps.reduce((s, t) => s + t, 0) / temps.length;
    const tDia = (i) => (i < temps.length ? temps[i] : prom); // tras el pronóstico, el promedio
    const avanzar = (dias25) => {
      let acum = 0;
      for (let i = 0; i < 90; i++) {
        const f = factor(tDia(i));
        if (acum + f >= dias25) return i + (dias25 - acum) / f;
        acum += f;
      }
      return 90;
    };
    const restante = MODELO.DIAS_A_E5[codigoEstadio];
    if (restante == null) return null;
    const dias = restante > 0 ? avanzar(restante) : 0;
    const vida = avanzar(restante + MODELO.VIDA_E5) - dias;
    const usados = temps.slice(0, Math.max(1, Math.ceil(dias + vida)));
    const tmedia = usados.reduce((s, t) => s + t, 0) / usados.length;
    const fecha = new Date(Date.now() + dias * 86400000);
    const alertas = [];
    if (temps.some((t) => t > MODELO.T_MAX)) alertas.push(`Hay días sobre ${MODELO.T_MAX} °C: la maduración puede ser irregular (pulpa blanda con cáscara verde).`);
    if (temps.some((t) => t < MODELO.T_MIN)) alertas.push(`Hay días bajo ${MODELO.T_MIN} °C: riesgo de daño por frío; la maduración se detiene.`);
    return { dias, vida, fecha, tmedia, factor: factor(tmedia), alertas };
  }

  /** Texto corto para el código WMO del clima. */
  function describir(codigo) {
    if (codigo == null) return "";
    if (codigo === 0) return "Despejado";
    if (codigo <= 2) return "Parcialmente nublado";
    if (codigo === 3) return "Nublado";
    if (codigo === 45 || codigo === 48) return "Niebla";
    if (codigo >= 51 && codigo <= 57) return "Llovizna";
    if (codigo >= 61 && codigo <= 67) return "Lluvia";
    if (codigo >= 71 && codigo <= 77) return "Nieve";
    if (codigo >= 80 && codigo <= 82) return "Chubascos";
    if (codigo >= 95) return "Tormenta";
    return "";
  }

  const api = {
    MODELO, TARAPOTO, ubicacion, usarMiUbicacion, usarTarapoto, obtener,
    factor, factorHoy, temperaturasFuturas, proyectar, describir,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else global.PapayaClima = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
