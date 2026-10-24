"""
app.py
======
Interfaz gráfica (Streamlit) del módulo de resonancia acústica para papaya.

No reimplementa nada del pipeline: llama directamente a los módulos
`audio_io`, `preprocessing`, `features`, `build_dataset` y `train`, de modo
que la interfaz y la consola SIEMPRE dan los mismos números.

Pestañas:
    1. Analizar golpe : onda + impulso con ajuste exponencial + espectro,
                        tarjetas de features y predicción de madurez.
    2. Comparar       : dos golpes superpuestos (p. ej. verde vs maduro).
    3. Dataset        : subir grabaciones, editar metadata.csv, pasar de
                        audios sintéticos a reales y actualizar dataset +
                        modelo (el reentreno es automático).

Uso:
    streamlit run app.py
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # train.py importa pyplot; sin backend GUI en un servidor

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold)

import config
import audio_io
import preprocessing
import features
import build_dataset
import export_model
import train

# ---------------------------------------------------------------------------
# Constantes de la interfaz
# ---------------------------------------------------------------------------
CLASS_COLORS = {"verde": "#2f9e44", "semi": "#e0a800", "maduro": "#e8590c"}
CLASS_ORDER = train.CLASS_ORDER
AUDIO_DIRS = {"Sintéticos (data/synthetic)": config.SYNTHETIC_DIR,
              "Grabaciones reales (data/raw)": config.RAW_DIR}
MODEL_FILES = {"RandomForest": config.MODELS_DIR / "randomforest.joblib",
               "SVM": config.MODELS_DIR / "svm.joblib"}
# Orden de columnas con que train.py entrena (features.csv sin id/etiqueta).
DEFAULT_FEATURE_COLS = ["masa_g", "f1", "f2", "damping_logdec", "damping_bw",
                        "S", "EI", "zcr", "ste"]
# Frecuencia máxima que se muestra en los espectros (un poco más que la banda).
SPECTRUM_FMAX = config.FREQ_MAX + 500.0
# Nº máximo de puntos por traza en las gráficas de tiempo (fluidez del navegador).
MAX_PLOT_POINTS = 6000

st.set_page_config(page_title="Resonancia acústica · Papaya",
                   page_icon="🥭", layout="wide")


# ===========================================================================
# LÓGICA (envoltorios delgados sobre el pipeline)
# ===========================================================================
def analizar(path: Path, masa_g: float) -> dict:
    """Corre el pipeline completo sobre un .wav y devuelve todo lo graficable.

    Mismo flujo que build_dataset.py:
        load_wav -> preprocess (onset + recorte + normalización) ->
        extract_all_features
    más los vectores intermedios (espectro, envolvente) para graficar.
    """
    signal, sr = audio_io.load_wav(path)
    return analizar_senal(signal, sr, masa_g, path.stem)


def analizar_senal(signal: np.ndarray, sr: int, masa_g: float,
                   nombre: str) -> dict:
    """Igual que `analizar`, pero sobre una señal ya cargada en memoria."""
    impulse, onset = preprocessing.preprocess(signal, sr)
    freqs, mag = features.compute_spectrum(impulse, sr)
    peak_freqs, peak_mags = features.find_resonances(freqs, mag)
    env = features.hilbert_envelope(impulse)
    feats = features.extract_all_features(impulse, mass_g=masa_g, sr=sr)
    return {
        "nombre": nombre, "signal": signal, "sr": sr, "onset": onset,
        "impulse": impulse, "env": env, "freqs": freqs, "mag": mag,
        "peak_freqs": peak_freqs, "peak_mags": peak_mags, "feats": feats,
        "masa_g": masa_g,
    }


# ---------------------------------------------------------------------------
# Con qué datos está entrenado el modelo: audios sintéticos o grabaciones reales
# ---------------------------------------------------------------------------
# Al principio el modelo viene entrenado con los audios sintéticos. En cuanto
# hay suficientes grabaciones reales y se pulsa «Actualizar dataset y modelo»,
# pasa a entrenarse SOLO con ellas. Se recuerda en un pequeño JSON.
FUENTE_JSON = config.OUTPUTS_DIR / "fuente_datos.json"
# Las filas sintéticas de metadata.csv se apartan aquí al pasar a reales.
META_SINT_BAK = config.DATA_DIR / "metadata_sinteticos.csv"
FUENTES = {"sinteticos": "Sintéticos (data/synthetic)",
           "reales": "Grabaciones reales (data/raw)"}
# Requisitos del entrenamiento automático.
TEST_SIZE, FOLDS = 0.3, 5


def fuente_activa() -> str:
    try:
        return json.loads(FUENTE_JSON.read_text(encoding="utf-8"))["fuente"]
    except Exception:
        return "sinteticos"


def apartar_filas_sinteticas() -> None:
    """Saca de metadata.csv las filas de audios sintéticos (a un respaldo)."""
    meta = leer_metadata()
    ids_sint = {p.stem for p in config.SYNTHETIC_DIR.glob("*.wav")}
    es_sint = meta["id"].astype(str).isin(ids_sint)
    if not es_sint.any():
        return
    previas = (pd.read_csv(META_SINT_BAK) if META_SINT_BAK.exists()
               else meta.iloc[0:0])
    pd.concat([previas, meta[es_sint]]).drop_duplicates(
        subset="id", keep="last").to_csv(META_SINT_BAK, index=False)
    meta[~es_sint].to_csv(config.METADATA_CSV, index=False)


def problema_para_entrenar(y: np.ndarray, test_size: float,
                           folds: int) -> str | None:
    """Explica por qué no se puede entrenar con estas etiquetas (o None)."""
    clases, conteos = np.unique(y, return_counts=True)
    if clases.size < 2:
        return (f"hace falta al menos 2 clases distintas; hay "
                f"{clases.size} ({', '.join(clases) or 'ninguna'}).")
    minimo = max(2, folds)
    pocas = [f"{c}: {n}" for c, n in zip(clases, conteos) if n < minimo]
    if pocas:
        return (f"cada clase necesita al menos {minimo} grabaciones. "
                f"Ahora hay {', '.join(pocas)}.")
    if int(np.ceil(y.size * test_size)) < clases.size:
        return "hay muy pocas grabaciones para evaluar el modelo."
    return None


def actualizar_con_reales() -> list[tuple[str, str]]:
    """Reconstruye el dataset con las grabaciones REALES y reentrena.

    Todo se prepara primero en archivos temporales: si las grabaciones aún
    no alcanzan para entrenar, el dataset y el modelo actuales NO se tocan.
    Si alcanzan, reemplazan a los sintéticos. Devuelve mensajes (tipo, texto).
    """
    meta = leer_metadata()
    ids_raw = {p.stem for p in config.RAW_DIR.glob("*.wav")}
    reales = meta.loc[meta["id"].astype(str).isin(ids_raw)]
    if reales.empty:
        return [("warning", "No hay grabaciones reales con masa y etiqueta. "
                            "Súbelas y regístralas en los pasos 0 y 1.")]

    tmp = Path(tempfile.mkdtemp(prefix="papaya_"))
    tmp_meta, tmp_feats = tmp / "metadata.csv", tmp / "features.csv"
    reales.to_csv(tmp_meta, index=False)
    msgs, log = [], io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            df = build_dataset.build(config.RAW_DIR, tmp_meta, tmp_feats)
    except Exception as e:
        return [("warning", f"No se pudo construir el dataset: {e}"),
                ("log", log.getvalue())]
    if "[SKIP]" in log.getvalue():
        msgs.append(("log", log.getvalue()))

    problema = problema_para_entrenar(df["etiqueta"].astype(str).to_numpy(),
                                      TEST_SIZE, FOLDS)
    if problema:
        actual = ("grabaciones reales" if fuente_activa() == "reales"
                  else "audios sintéticos")
        msgs.append(("warning", f"Se leyeron {len(df)} grabaciones reales, "
                     f"pero todavía no alcanzan para entrenar: {problema} "
                     f"El modelo actual (entrenado con {actual}) se mantiene "
                     f"sin cambios."))
        return msgs

    # Alcanzan: las grabaciones reales reemplazan a los sintéticos.
    shutil.move(str(tmp_feats), str(config.FEATURES_CSV))
    res = entrenar(TEST_SIZE, FOLDS)
    FUENTE_JSON.parent.mkdir(parents=True, exist_ok=True)
    FUENTE_JSON.write_text(json.dumps({"fuente": "reales"}), encoding="utf-8")
    apartar_filas_sinteticas()
    st.session_state["reset_carpetas"] = True  # selectores -> data/raw
    cargar_modelos.clear()
    cv = {n: m["cv"].mean() * 100 for n, m in res["modelos"].items()}
    msgs.append(("success", f"Modelo reentrenado con {len(df)} grabaciones "
                 f"reales. Precisión estimada (validación cruzada "
                 f"{FOLDS}-fold): " +
                 ", ".join(f"{n} {v:.1f}%" for n, v in cv.items()) + "."))
    # Versión del modelo para la app móvil (se importa desde el celular).
    try:
        export_model.exportar(export_model.DEFAULT_OUT)
        msgs.append(("info", "Modelo para la app móvil actualizado: descárgalo "
                             "abajo y cárgalo en el celular (ⓘ → Importar modelo)."))
    except Exception as e:
        msgs.append(("warning", f"No se pudo exportar el modelo móvil: {e}"))
    return msgs


def mostrar_mensajes(msgs: list[tuple[str, str]]) -> None:
    for tipo, texto in msgs:
        if tipo == "log":
            if texto.strip():
                with st.expander("Detalle: grabaciones que no se pudieron usar"):
                    st.code(texto)
        else:
            getattr(st, tipo)(texto)


def leer_metadata() -> pd.DataFrame:
    """metadata.csv como DataFrame (vacío con las columnas si no existe)."""
    if config.METADATA_CSV.exists():
        return pd.read_csv(config.METADATA_CSV)
    return pd.DataFrame(columns=["id", "masa_g", "etiqueta"])


def masa_de_metadata(sample_id: str) -> float | None:
    """Masa registrada en metadata.csv para un id, o None si no está."""
    meta = leer_metadata()
    fila = meta.loc[meta["id"].astype(str) == sample_id]
    return float(fila["masa_g"].iloc[0]) if not fila.empty else None


def etiqueta_de_metadata(sample_id: str) -> str | None:
    meta = leer_metadata()
    fila = meta.loc[meta["id"].astype(str) == sample_id]
    return str(fila["etiqueta"].iloc[0]) if not fila.empty else None


@st.cache_resource
def cargar_modelos() -> dict:
    """Carga los modelos .joblib entrenados (los que existan)."""
    return {n: joblib.load(p) for n, p in MODEL_FILES.items() if p.exists()}


def columnas_features() -> list[str]:
    """Columnas de features en el MISMO orden con que se entrenó el modelo."""
    if config.FEATURES_CSV.exists():
        cols = pd.read_csv(config.FEATURES_CSV, nrows=0).columns
        return [c for c in cols if c not in train.NON_FEATURE_COLS]
    return DEFAULT_FEATURE_COLS


def predecir(feats: dict, masa_g: float) -> dict:
    """Predicción de cada modelo disponible (y probabilidades si las hay)."""
    fila = {**feats, "masa_g": masa_g}
    x = np.array([[fila.get(c, np.nan) for c in columnas_features()]],
                 dtype=float)
    out = {}
    for nombre, modelo in cargar_modelos().items():
        try:
            pred = str(modelo.predict(x)[0])
            proba = None
            if hasattr(modelo, "predict_proba"):
                try:
                    p = modelo.predict_proba(x)[0]
                    proba = dict(zip(modelo.classes_, p))
                except AttributeError:
                    pass  # SVC sin probability=True
            out[nombre] = {"pred": pred, "proba": proba}
        except Exception as e:  # modelo entrenado con otras columnas, etc.
            out[nombre] = {"error": str(e)}
    return out


def _submuestrear(x: np.ndarray, *ys: np.ndarray):
    """Reduce puntos para graficar sin saturar el navegador."""
    paso = max(1, x.size // MAX_PLOT_POINTS)
    return (x[::paso],) + tuple(y[::paso] for y in ys)


# ===========================================================================
# GRÁFICAS
# ===========================================================================
def _layout(fig: go.Figure, titulo: str, xt: str, yt: str,
            height: int = 320) -> go.Figure:
    fig.update_layout(title=titulo, xaxis_title=xt, yaxis_title=yt,
                      height=height, margin=dict(l=10, r=10, t=45, b=10),
                      legend=dict(orientation="h", y=1.02, x=1,
                                  xanchor="right", yanchor="bottom"),
                      hovermode="x unified")
    return fig


def fig_grabacion(r: dict) -> go.Figure:
    """Grabación completa con el onset y la ventana de análisis marcados."""
    sr, sig, onset = r["sr"], r["signal"], r["onset"]
    t = np.arange(sig.size) / sr
    t, s = _submuestrear(t, sig)
    fig = go.Figure(go.Scatter(x=t, y=s, name="señal", mode="lines",
                               line=dict(width=1, color="#868e96")))
    t0 = max(0, onset - int(config.PRE_ONSET_PAD * sr)) / sr
    t1 = t0 + r["impulse"].size / sr
    fig.add_vrect(x0=t0, x1=t1, fillcolor="#e8590c", opacity=0.12,
                  line_width=0, annotation_text="ventana de análisis",
                  annotation_position="top left")
    fig.add_vline(x=onset / sr, line_dash="dash", line_color="#e8590c",
                  annotation_text=f"onset {onset / sr:.3f} s")
    return _layout(fig, "Grabación completa", "tiempo [s]", "amplitud")


def fig_impulso(r: dict) -> go.Figure:
    """Impulso recortado + envolvente de Hilbert + ajuste A0·e^(−ζ·ω·t)."""
    sr, imp, env = r["sr"], r["impulse"], r["env"]
    t = np.arange(imp.size) / sr
    ts, imps, envs = _submuestrear(t, imp, env)
    fig = go.Figure()
    fig.add_scatter(x=ts, y=imps, name="impulso", mode="lines",
                    line=dict(width=1, color="#adb5bd"))
    fig.add_scatter(x=ts, y=envs, name="envolvente (Hilbert)", mode="lines",
                    line=dict(width=2, color="#1c7ed6"))

    # Curva teórica con el ζ medido: permite juzgar a ojo si el ajuste es bueno.
    zeta, f1 = r["feats"]["damping_logdec"], r["feats"]["f1"]
    if np.isfinite(zeta) and np.isfinite(f1):
        k = int(np.argmax(env))
        tt = t[k:]
        ajuste = env[k] * np.exp(-zeta * 2 * np.pi * f1 * (tt - tt[0]))
        tt, ajuste = _submuestrear(tt, ajuste)
        fig.add_scatter(x=tt, y=ajuste, mode="lines",
                        name=f"ajuste e^(−ζωt), ζ={zeta:.4f}",
                        line=dict(width=2, dash="dash", color="#e8590c"))
    return _layout(fig, "Impulso recortado y decaimiento", "tiempo desde el "
                   "recorte [s]", "amplitud normalizada")


def fig_espectro(r: dict) -> go.Figure:
    """Espectro FFT con f1/f2 y el ancho de banda a −3 dB de f1."""
    freqs, mag = r["freqs"], r["mag"]
    sel = freqs <= SPECTRUM_FMAX
    f, m = freqs[sel], mag[sel] / (mag[sel].max() or 1.0)
    fig = go.Figure(go.Scatter(x=f, y=m, name="|X(f)|", mode="lines",
                               line=dict(width=1.5, color="#1c7ed6")))
    fig.add_vrect(x0=config.FREQ_MIN, x1=config.FREQ_MAX, fillcolor="#1c7ed6",
                  opacity=0.04, line_width=0)
    for i, pf in enumerate(r["peak_freqs"][:2]):
        fig.add_vline(x=pf, line_dash="dot", line_color="#e8590c",
                      annotation_text=f"f{i + 1} = {pf:.0f} Hz",
                      annotation_position="top right")
    # Ancho de banda −3 dB alrededor de f1 (lo que usa damping_bandwidth).
    f1, zbw = r["feats"]["f1"], r["feats"]["damping_bw"]
    if np.isfinite(f1) and np.isfinite(zbw):
        bw = 2 * zbw * f1
        fig.add_vrect(x0=f1 - bw / 2, x1=f1 + bw / 2, fillcolor="#e8590c",
                      opacity=0.15, line_width=0)
        fig.add_hline(y=m[np.argmin(np.abs(f - f1))] / np.sqrt(2),
                      line_dash="dash", line_color="#adb5bd",
                      annotation_text="−3 dB", annotation_position="bottom left")
    return _layout(fig, "Espectro FFT (ventana Hann)", "frecuencia [Hz]",
                   "magnitud normalizada")


def fig_comparar_envolventes(ra: dict, rb: dict, ca: str, cb: str) -> go.Figure:
    fig = go.Figure()
    for r, c in ((ra, ca), (rb, cb)):
        t = np.arange(r["env"].size) / r["sr"]
        t, e = _submuestrear(t, r["env"])
        fig.add_scatter(x=t, y=e, name=r["nombre"], mode="lines",
                        line=dict(width=2, color=c))
    return _layout(fig, "Envolventes (decaimiento)", "tiempo [s]",
                   "amplitud normalizada", height=360)


def fig_comparar_espectros(ra: dict, rb: dict, ca: str, cb: str) -> go.Figure:
    fig = go.Figure()
    for r, c in ((ra, ca), (rb, cb)):
        sel = r["freqs"] <= SPECTRUM_FMAX
        m = r["mag"][sel] / (r["mag"][sel].max() or 1.0)
        fig.add_scatter(x=r["freqs"][sel], y=m, name=r["nombre"],
                        mode="lines", line=dict(width=1.8, color=c))
        for pf in r["peak_freqs"][:2]:
            fig.add_vline(x=pf, line_dash="dot", line_color=c, opacity=0.6)
    return _layout(fig, "Espectros superpuestos", "frecuencia [Hz]",
                   "magnitud normalizada", height=360)


# ===========================================================================
# COMPONENTES DE UI
# ===========================================================================
def selector_audio(key: str,
                   default_idx: int = 0) -> tuple[Path | None, str]:
    """Elige un .wav del proyecto o sube uno. Devuelve (ruta, id).

    Un archivo subido aquí es solo para consultar: se analiza desde una
    carpeta TEMPORAL y nunca se agrega al proyecto ni al dataset (eso se
    hace únicamente en la pestaña Dataset).
    """
    origen = st.radio("Origen", ["Carpeta del proyecto", "Subir .wav"],
                      horizontal=True, key=f"{key}_origen",
                      label_visibility="collapsed")
    if origen == "Subir .wav":
        up = st.file_uploader("Archivo .wav", type=["wav"], key=f"{key}_up")
        st.caption("Solo para consultar: este audio no se guarda en el "
                   "proyecto ni entra al dataset.")
        if up is None:
            return None, ""
        # audio_io.load_wav trabaja sobre rutas: se vuelca a un temporal.
        tmp = Path(tempfile.gettempdir()) / f"papaya_{key}_{up.name}"
        tmp.write_bytes(up.getvalue())
        return tmp, Path(up.name).stem

    carpeta = st.selectbox("Carpeta", list(AUDIO_DIRS), key=f"{key}_dir",
                           index=list(AUDIO_DIRS).index(FUENTES[fuente_activa()]))
    wavs = sorted(AUDIO_DIRS[carpeta].glob("*.wav"))
    if not wavs:
        st.info(f"No hay .wav en `{AUDIO_DIRS[carpeta].relative_to(config.BASE_DIR)}`.")
        return None, ""
    nombres = [p.name for p in wavs]
    elegido = st.selectbox("Archivo", nombres, key=f"{key}_file",
                           index=min(default_idx, len(nombres) - 1))
    p = AUDIO_DIRS[carpeta] / elegido
    return p, p.stem


def input_masa(sample_id: str, key: str) -> float:
    """Masa en gramos: por defecto la de metadata.csv si el id está registrado."""
    registrada = masa_de_metadata(sample_id) if sample_id else None
    masa = st.number_input(
        "Masa del fruto [g]", min_value=1.0, max_value=5000.0, step=1.0,
        value=registrada if registrada is not None else 1200.0,
        key=f"{key}_masa_{sample_id}",
        help="Se usa en S = f1²·m^(2/3) y EI = f2²·m^(2/3). "
             "Se toma de metadata.csv cuando el id está registrado.")
    if registrada is None and sample_id:
        st.caption("Este id no está en metadata.csv: ingresa la masa medida.")
    return masa


def tarjetas_features(feats: dict) -> None:
    """Muestra las features principales como métricas."""
    def fmt(v, spec):
        return format(v, spec) if np.isfinite(v) else "—"

    c = st.columns(4)
    c[0].metric("f1 (dominante)", f"{fmt(feats['f1'], '.1f')} Hz")
    c[1].metric("f2", f"{fmt(feats['f2'], '.1f')} Hz")
    c[2].metric("ζ log-decrement", fmt(feats["damping_logdec"], ".4f"),
                help="Valor físico: pendiente de ln(envolvente) / (2π·f1).")
    c[3].metric("ζ ancho de banda −3 dB", fmt(feats["damping_bw"], ".4f"),
                help="Δf / (2·f1). Subestima el valor absoluto, pero confirma "
                     "la tendencia.")
    c = st.columns(4)
    c[0].metric("S = f1²·m^(2/3)", fmt(feats["S"], ".3e"))
    c[1].metric("EI = f2²·m^(2/3)", fmt(feats["EI"], ".3e"))
    c[2].metric("ZCR", fmt(feats["zcr"], ".4f"))
    c[3].metric("STE media", fmt(feats["ste"], ".3f"))


def mostrar_error(e: Exception) -> None:
    """Muestra los errores de validación del pipeline de forma legible."""
    if isinstance(e, (audio_io.AudioValidationError, FileNotFoundError,
                      ValueError)):
        st.error(f"**No se pudo analizar el audio.** {e}")
    else:
        st.exception(e)


# ===========================================================================
# PESTAÑAS
# ===========================================================================
def tab_analizar() -> None:
    col_sel, col_res = st.columns([1, 3], gap="large")
    with col_sel:
        st.subheader("Golpe")
        path, sid = selector_audio("an")
        if path is None:
            return
        masa = input_masa(sid, "an")
        st.audio(str(path))
        real = etiqueta_de_metadata(sid)
        if real:
            st.caption(f"Etiqueta registrada en metadata.csv: **{real}**")

    with col_res:
        try:
            r = analizar(path, masa)
        except Exception as e:
            mostrar_error(e)
            return

        preds = predecir(r["feats"], masa)
        if preds:
            cols = st.columns(len(preds))
            for col, (nombre, p) in zip(cols, preds.items()):
                if "error" in p:
                    col.warning(f"{nombre}: {p['error']}")
                    continue
                color = CLASS_COLORS.get(p["pred"], "#495057")
                extra = ""
                if p["proba"]:
                    extra = f" · {p['proba'][p['pred']] * 100:.0f}%"
                col.markdown(
                    f"<div style='padding:10px 14px;border-radius:8px;"
                    f"border-left:6px solid {color};background:{color}1a'>"
                    f"<small>{nombre}</small><br><b style='font-size:1.4em;"
                    f"color:{color}'>{p['pred'].upper()}</b>{extra}</div>",
                    unsafe_allow_html=True)
            if fuente_activa() == "sinteticos":
                st.caption("⚠️ Estos modelos aprendieron de audios "
                           "SINTÉTICOS: con una papaya real la predicción no es "
                           "confiable. Cuando tengas tus grabaciones, regístralas "
                           "en 📊 Dataset y pulsa «Actualizar dataset y modelo».")
        else:
            st.info("Aún no hay un modelo entrenado. En **📊 Dataset** registra "
                    "tus grabaciones y pulsa «Actualizar dataset y modelo».")

        st.write("")
        tarjetas_features(r["feats"])
        st.plotly_chart(fig_espectro(r), width="stretch")
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_impulso(r), width="stretch")
        c2.plotly_chart(fig_grabacion(r), width="stretch")


def tab_comparar() -> None:
    ca, cb = st.columns(2, gap="large")
    resultados = []
    for col, key, idx, color in ((ca, "A", 0, "#2f9e44"),
                                 (cb, "B", 24, "#e8590c")):
        with col:
            st.subheader(f"Golpe {key}")
            path, sid = selector_audio(f"cmp{key}", default_idx=idx)
            if path is None:
                resultados.append(None)
                continue
            masa = input_masa(sid, f"cmp{key}")
            try:
                resultados.append((analizar(path, masa), color))
            except Exception as e:
                mostrar_error(e)
                resultados.append(None)

    if None in resultados:
        return
    (ra, col_a), (rb, col_b) = resultados

    nombres = [ra["nombre"], rb["nombre"]]
    if nombres[0] == nombres[1]:
        nombres = [f"{nombres[0]} (A)", f"{nombres[1]} (B)"]
    tabla = pd.DataFrame([ra["feats"], rb["feats"]], index=nombres).T
    c1, c2 = st.columns([3, 1])
    with c1:
        st.plotly_chart(fig_comparar_espectros(ra, rb, col_a, col_b),
                        width="stretch")
        st.plotly_chart(fig_comparar_envolventes(ra, rb, col_a, col_b),
                        width="stretch")
    with c2:
        st.markdown("**Features**")
        st.dataframe(tabla.style.format("{:.4g}"), width="stretch")


def tab_dataset() -> None:
    st.subheader("0 · Subir grabaciones a data/raw")
    ups = st.file_uploader("Arrastra aquí uno o varios .wav", type=["wav"],
                           accept_multiple_files=True, key="ds_upload")
    if ups and st.button(f"💾 Guardar {len(ups)} archivo(s) en data/raw"):
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        for up in ups:
            (config.RAW_DIR / up.name).write_bytes(up.getvalue())
        st.success(f"{len(ups)} archivo(s) guardados. Completa su masa y "
                   f"etiqueta en la tabla de abajo y pulsa Guardar.")

    st.subheader("1 · Metadatos")
    st.caption("`id` = nombre del .wav sin extensión · `masa_g` en gramos · "
               "`etiqueta` ∈ verde / semi / maduro")
    meta = leer_metadata()
    # Grabaciones de data/raw que aún no tienen fila: se añaden vacías para
    # que el usuario solo tenga que completar masa y etiqueta.
    registrados = set(meta["id"].astype(str))
    faltantes = [p.stem for p in sorted(config.RAW_DIR.glob("*.wav"))
                 if p.stem not in registrados]
    if faltantes:
        st.warning(f"{len(faltantes)} grabación(es) de data/raw sin masa ni "
                   f"etiqueta (al final de la tabla): {', '.join(faltantes)}")
        meta = pd.concat([meta, pd.DataFrame({"id": faltantes})],
                         ignore_index=True)
    meta = st.data_editor(
        meta, num_rows="dynamic", width="stretch",
        height=250, key="meta_editor",
        column_config={"etiqueta": st.column_config.SelectboxColumn(
            options=CLASS_ORDER)})
    if st.button("💾 Guardar metadata.csv"):
        completas = meta.dropna(subset=["id", "masa_g", "etiqueta"])
        completas.to_csv(config.METADATA_CSV, index=False)
        st.success(f"Guardado en {config.METADATA_CSV}")
        if len(completas) < len(meta):
            st.info(f"{len(meta) - len(completas)} fila(s) sin masa o "
                    f"etiqueta no se guardaron todavía.")

    st.subheader("2 · Actualizar dataset y modelo")
    st.caption(f"Extrae las features de tus grabaciones reales (data/raw) "
               f"registradas arriba y reentrena el modelo solo con ellas, "
               f"reemplazando a los audios sintéticos. Se necesitan al menos "
               f"{FOLDS} grabaciones por clase; mientras no las haya, el modelo "
               f"actual se mantiene. Hazlo cada vez que agregues grabaciones.")
    if st.button("🔄 Actualizar dataset y modelo", type="primary"):
        with st.spinner("Extrayendo features y reentrenando..."):
            st.session_state["msgs_ds"] = actualizar_con_reales()
        st.rerun()  # que todas las pestañas reflejen el modelo nuevo
    mostrar_mensajes(st.session_state.pop("msgs_ds", []))

    if export_model.DEFAULT_OUT.exists():
        st.download_button(
            "📱 Descargar modelo para la app móvil",
            export_model.DEFAULT_OUT.read_bytes(), "model_movil.json",
            "application/json",
            help="Cópialo al celular y cárgalo en la app: botón ⓘ → "
                 "Importar modelo. Así no hace falta recompilar el APK.")

    if not config.FEATURES_CSV.exists():
        st.info("Todavía no hay dataset construido.")
        return

    df = pd.read_csv(config.FEATURES_CSV)
    st.subheader("3 · Explorar")
    conteo = df["etiqueta"].value_counts()
    cols = st.columns(len(conteo) + 1)
    cols[0].metric("Muestras", len(df))
    for c, (clase, n) in zip(cols[1:], conteo.items()):
        c.metric(clase, n)

    num_cols = [c for c in df.columns if c not in ("id", "etiqueta")]
    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("Eje X", num_cols, index=num_cols.index("f1"))
    y = c2.selectbox("Eje Y", num_cols, index=num_cols.index("damping_logdec"))
    box = c3.selectbox("Distribución de", num_cols,
                       index=num_cols.index("damping_logdec"))
    g1, g2 = st.columns([3, 2])
    g1.plotly_chart(
        px.scatter(df, x=x, y=y, color="etiqueta", hover_name="id",
                   color_discrete_map=CLASS_COLORS,
                   category_orders={"etiqueta": CLASS_ORDER},
                   title=f"{y} vs {x}").update_layout(height=420),
        width="stretch")
    g2.plotly_chart(
        px.box(df, x="etiqueta", y=box, color="etiqueta", points="all",
               color_discrete_map=CLASS_COLORS,
               category_orders={"etiqueta": CLASS_ORDER},
               title=f"{box} por clase").update_layout(height=420,
                                                       showlegend=False),
        width="stretch")
    st.dataframe(df, width="stretch", height=300)
    st.download_button("⬇️ Descargar features.csv",
                       config.FEATURES_CSV.read_bytes(), "features.csv",
                       "text/csv")


def entrenar(test_size: float, folds: int) -> dict:
    """Misma evaluación que train.py, pero devolviendo resultados en vez de
    imprimirlos: hold-out estratificado, CV k-fold y reentreno final en todo
    el dataset (el modelo final se guarda en outputs/models/)."""
    X, y, feature_cols = train.load_dataset(config.FEATURES_CSV)
    labels = [c for c in CLASS_ORDER if c in np.unique(y)]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=train.RANDOM_STATE)
    skf = StratifiedKFold(n_splits=folds, shuffle=True,
                          random_state=train.RANDOM_STATE)

    res = {"labels": labels, "feature_cols": feature_cols, "modelos": {}}
    config.ensure_dirs()
    for nombre, modelo in train.build_models().items():
        modelo.fit(X_tr, y_tr)
        y_pred = modelo.predict(X_te)
        cv = cross_val_score(modelo, X, y, cv=skf, scoring="accuracy")
        reporte = classification_report(y_te, y_pred, labels=labels,
                                        output_dict=True, zero_division=0)
        modelo.fit(X, y)
        joblib.dump(modelo, MODEL_FILES[nombre])
        res["modelos"][nombre] = {
            "acc": accuracy_score(y_te, y_pred),
            "cm": confusion_matrix(y_te, y_pred, labels=labels),
            "cv": cv, "reporte": pd.DataFrame(reporte).T,
            "importancias": (modelo.named_steps["clf"].feature_importances_
                             if nombre == "RandomForest" else None),
        }
    return res


# ===========================================================================
# BARRA LATERAL Y MAIN
# ===========================================================================
def sidebar() -> None:
    with st.sidebar:
        st.markdown("## Papaya")
        st.caption("Módulo de resonancia acústica · método impulso-respuesta")
        st.markdown("**Parámetros (config.py)**")
        st.markdown(
            f"- Sample rate: `{config.SAMPLE_RATE} Hz`\n"
            f"- Banda de resonancias: `{config.FREQ_MIN:.0f}–"
            f"{config.FREQ_MAX:.0f} Hz`\n"
            f"- FFT: `{config.FFT_SIZE}` puntos\n"
            f"- Ventana de análisis: `{config.ANALYSIS_WINDOW * 1000:.0f} ms`\n"
            f"- Onset: mediana + `{config.ONSET_K}`·MAD")
        st.markdown("**Modelo entrenado con:** " +
                    ("🎙️ grabaciones reales" if fuente_activa() == "reales"
                     else "🧪 audios sintéticos"))
        modelos = cargar_modelos()
        st.markdown("**Modelos entrenados:** " +
                    (", ".join(modelos) if modelos else "ninguno"))


def main() -> None:
    config.ensure_dirs()
    if st.session_state.pop("reset_carpetas", False):
        for k in ("an_dir", "cmpA_dir", "cmpB_dir"):
            st.session_state.pop(k, None)
    sidebar()
    st.title("Estimación de madurez por resonancia acústica")
    t1, t2, t3 = st.tabs(["🔍 Analizar golpe", "⚖️ Comparar", "📊 Dataset"])
    with t1:
        tab_analizar()
    with t2:
        tab_comparar()
    with t3:
        tab_dataset()


main()
