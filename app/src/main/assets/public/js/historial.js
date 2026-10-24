/*
 * historial.js — Registro local de mediciones (sin internet) y exportación.
 *
 * Cada fruto medido es un registro con su código, fecha, estadio, color por
 * zona, descriptores acústicos, clima del momento y tiempos de proceso. Se
 * guarda en el almacenamiento del celular (localStorage del WebView), que se
 * conserva al cerrar la app y solo se borra si se borran los datos de la app.
 *
 * La exportación genera un CSV (una fila por fruto) con las mismas columnas
 * que el conjunto de datos del proyecto, para abrirlo en Excel, Python o R.
 * En el celular se envía al menú Compartir de Android (Drive, correo,
 * WhatsApp, Archivos) mediante el plugin nativo "Exportar" (MainActivity).
 */
(function (global) {
  "use strict";

  const LS = "papaya.historial", LS_CONTADOR = "papaya.contador";

  function leer() {
    try {
      const v = JSON.parse(localStorage.getItem(LS) || "[]");
      return Array.isArray(v) ? v : [];
    } catch { return []; }
  }
  function escribir(lista) {
    try { localStorage.setItem(LS, JSON.stringify(lista)); return true; } catch { return false; }
  }

  /** Inserta o actualiza (mismo id) y deja el más reciente primero. */
  function guardar(reg) {
    const lista = leer().filter((r) => r.id !== reg.id);
    lista.unshift(reg);
    return escribir(lista);
  }
  const borrar = (id) => escribir(leer().filter((r) => r.id !== id));
  const limpiar = () => escribir([]);

  /** Código correlativo P-0001, P-0002… (no se repite aunque se borren). */
  function nuevoCodigo() {
    let n = 0;
    try { n = parseInt(localStorage.getItem(LS_CONTADOR) || "0", 10) || 0; } catch { /* sin almacenamiento */ }
    n++;
    try { localStorage.setItem(LS_CONTADOR, String(n)); } catch { /* sin almacenamiento */ }
    return `P-${String(n).padStart(4, "0")}`;
  }

  // =========================================================================
  // CSV
  // =========================================================================
  const num = (v, d) => (Number.isFinite(v) ? Number(v.toFixed(d)) : "");
  const zona = (r, i, k, d) => (r.tabla2 && r.tabla2.zonas && r.tabla2.zonas[i] ? num(r.tabla2.zonas[i][k], d) : "");

  const COLUMNAS = [
    ["id", (r) => r.id],
    ["codigo", (r) => r.codigo],
    ["fecha", (r) => r.fecha],
    ["estadio_fusion", (r) => r.estadio],
    ["clase_fusion", (r) => r.clase],
    ["confianza", (r) => num(r.confianza, 3)],
    ["origen", (r) => r.origen],
    ["indice_1a10", (r) => num(r.indice, 2)],
    ["estadio_tabla2", (r) => r.tabla2 && r.tabla2.estadio],
    ["pct_amarilla", (r) => r.tabla2 && num(r.tabla2.pct_amarilla, 1)],
    ["caras", (r) => r.tabla2 && r.tabla2.caras],
    ["L", (r) => r.color && num(r.color.L, 2)],
    ["a", (r) => r.color && num(r.color.a, 2)],
    ["b", (r) => r.color && num(r.color.b, 2)],
    ["C", (r) => r.color && num(r.color.C, 2)],
    ["h", (r) => r.color && num(r.color.h, 2)],
    ["h_desv", (r) => r.tabla2 && num(r.tabla2.h_desv, 2)],
    ["manchas_pct", (r) => r.tabla2 && num(r.tabla2.manchas, 2)],
    ["ped_L", (r) => zona(r, 0, "L", 2)], ["ped_a", (r) => zona(r, 0, "a", 2)], ["ped_b", (r) => zona(r, 0, "b", 2)],
    ["ped_h", (r) => zona(r, 0, "h", 2)], ["ped_pct_amarilla", (r) => zona(r, 0, "pct_amarilla", 1)],
    ["ecu_L", (r) => zona(r, 1, "L", 2)], ["ecu_a", (r) => zona(r, 1, "a", 2)], ["ecu_b", (r) => zona(r, 1, "b", 2)],
    ["ecu_h", (r) => zona(r, 1, "h", 2)], ["ecu_pct_amarilla", (r) => zona(r, 1, "pct_amarilla", 1)],
    ["api_L", (r) => zona(r, 2, "L", 2)], ["api_a", (r) => zona(r, 2, "a", 2)], ["api_b", (r) => zona(r, 2, "b", 2)],
    ["api_h", (r) => zona(r, 2, "h", 2)], ["api_pct_amarilla", (r) => zona(r, 2, "pct_amarilla", 1)],
    ["fondo_dominante", (r) => r.tabla2 && num(r.tabla2.fondo_dominante, 1)],
    ["golpes", (r) => r.audio && r.audio.golpes],
    ["frecuencia_hz", (r) => r.audio && num(r.audio.frecuencia_hz, 2)],
    ["cv_frecuencia_pct", (r) => r.audio && num(r.audio.cv_frecuencia, 2)],
    ["amplitud_dbfs", (r) => r.audio && num(r.audio.amplitud_dbfs, 2)],
    ["damping", (r) => r.audio && num(r.audio.damping, 5)],
    ["masa_g", (r) => num(r.masa_g, 1)],
    ["EI", (r) => r.audio && num(r.audio.EI, 0)],
    ["S", (r) => r.audio && num(r.audio.S, 0)],
    ["lugar", (r) => r.clima && r.clima.lugar],
    ["lat", (r) => r.clima && num(r.clima.lat, 4)],
    ["lon", (r) => r.clima && num(r.clima.lon, 4)],
    ["temp_c", (r) => r.clima && num(r.clima.temp, 1)],
    ["hr_pct", (r) => r.clima && num(r.clima.hr, 0)],
    ["tmedia_prevista_c", (r) => r.clima && num(r.clima.tmedia, 1)],
    ["factor_maduracion", (r) => r.clima && num(r.clima.factor, 2)],
    ["dias_a_E5", (r) => r.clima && num(r.clima.dias_a_E5, 1)],
    ["fecha_E5", (r) => r.clima && r.clima.fecha_E5],
    ["foto_ms", (r) => r.rendimiento && num(r.rendimiento.foto_ms, 0)],
    ["golpe_ms", (r) => r.rendimiento && num(r.rendimiento.golpe_ms, 0)],
    ["ciclo_golpe_s", (r) => r.rendimiento && num(r.rendimiento.ciclo_golpe_s, 2)],
  ];

  function celda(v) {
    if (v == null) return "";
    const s = String(v);
    return /[",\n;]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  }

  function aCSV(lista = leer()) {
    const filas = [COLUMNAS.map((c) => c[0]).join(",")];
    for (const r of [...lista].reverse()) filas.push(COLUMNAS.map(([, f]) => celda(f(r))).join(","));
    return filas.join("\r\n") + "\r\n";
  }

  /** Comparte el CSV (app Android) o lo descarga (navegador). */
  async function exportar(lista = leer()) {
    const nombre = `historial_papaya_${new Date().toLocaleDateString("en-CA")}.csv`;
    const texto = aCSV(lista);
    const cap = global.Capacitor;
    if (cap && cap.isNativePlatform && cap.isNativePlatform() && cap.nativePromise) {
      await cap.nativePromise("Exportar", "compartir", { nombre, contenido: texto, tipo: "text/csv" });
      return nombre;
    }
    const url = URL.createObjectURL(new Blob(["﻿" + texto], { type: "text/csv;charset=utf-8" }));
    const a = document.createElement("a");
    a.href = url; a.download = nombre;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
    return nombre;
  }

  const api = { leer, guardar, borrar, limpiar, nuevoCodigo, aCSV, exportar, COLUMNAS };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else global.PapayaHistorial = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
