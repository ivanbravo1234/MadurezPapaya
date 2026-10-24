"""
calcular_umbrales.py
====================
Convierte dataset_papayas.csv en los parámetros de una regla de decisión
liviana (sin red neuronal) para la app móvil: data/papayas/umbrales_app.json

Requiere saber el estado de madurez REAL de cada papaya (verdad de campo:
escala visual de la norma, °Brix, firmeza con penetrómetro o días desde
cosecha). Se escribe en data/papayas/etiquetas_papayas.csv:

    id_papaya,indice_madurez,estado
    1,1,verde
    2,2,verde
    ...

En el muestreo de campo las papayas se ordenaron de la más verde (1) a la
más madura (10): indice_madurez es ese orden y 'estado' lo agrupa en cinco
estadios de 2 papayas (1–2 verde, 3–4 pintón, 5–6 semi, 7–8 maduro,
9–10 muy maduro). Los cortes se cambian editando
el CSV.

Regla calculada (clasificador de centroide más cercano ponderado):

    1. Centroide de cada clase c para cada variable v:   μ_cv
    2. Dispersión dentro de clase (agrupada) por variable: σ_v
    3. Peso de cada variable = razón de Fisher (cuánto separa las clases
       respecto a su ruido interno):
           F_v = var_entre_clases(μ_cv) / σ_v²       w_v = F_v / Σ F
       Una variable que no discrimina (p. ej. una frecuencia dominada por el
       montaje y no por el fruto) recibe peso ≈ 0 de forma automática.
    4. En la app, para una medición x:
           D_c = Σ_v w_v · ((x_v − μ_cv) / σ_v)²        clase = argmin_c D_c
           confianza_c = exp(−D_c/2) / Σ_k exp(−D_k/2)

Índice continuo de madurez (si el CSV trae indice_madurez): recta de
mínimos cuadrados sobre el promedio de a* de cada papaya,
    indice = β0 + β1 · a*
que la app muestra como "madurez 6,3 / 10".

También se reportan umbrales simples sobre a* (punto de corte entre clases
vecinas ponderado por su dispersión) para una regla de "si/entonces".

Validación: deja-una-papaya-fuera (se calculan los centroides sin la
papaya i y se clasifica la papaya i). Con 10 frutos es la única estimación
honesta: validar por fila mezclaría repeticiones de la misma papaya entre
entrenamiento y prueba.

Uso:
    python bimodal/calcular_umbrales.py [etiquetas.csv]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CARPETA = RAIZ / "data" / "papayas"
DATASET_CSV = CARPETA / "dataset_papayas.csv"
ETIQUETAS_CSV = CARPETA / "etiquetas_papayas.csv"
SALIDA_JSON = CARPETA / "umbrales_app.json"

VARIABLES = ["L", "a", "b", "frecuencia_hz", "amplitud"]
ORDEN_CLASES = ["verde", "pinton", "semi", "maduro", "muy_maduro"]   # orden de madurez creciente


# ---------------------------------------------------------------------------
# Entrenamiento (= calcular promedios, no hay optimización iterativa)
# ---------------------------------------------------------------------------
def ajustar(df: pd.DataFrame) -> dict:
    """Calcula centroides, σ agrupada y pesos de Fisher a partir de filas
    etiquetadas (columna 'estado')."""
    clases = [c for c in ORDEN_CLASES if c in set(df["estado"])]
    clases += sorted(set(df["estado"]) - set(clases))   # clases extra, si hay

    centroides = df.groupby("estado")[VARIABLES].mean().loc[clases]

    # σ agrupada: raíz del promedio de varianzas internas de cada clase.
    var_interna = df.groupby("estado")[VARIABLES].var(ddof=1).loc[clases].mean()
    sigma = np.sqrt(var_interna).replace(0, 1e-9)

    fisher = centroides.var(ddof=0) / var_interna
    pesos = fisher / fisher.sum()

    return {
        "variables": VARIABLES,
        "clases": clases,
        "centroides": {c: centroides.loc[c].round(4).to_dict() for c in clases},
        "sigma": sigma.round(4).to_dict(),
        "pesos": pesos.round(4).to_dict(),
        "fisher": fisher.round(3).to_dict(),
    }


def clasificar(modelo: dict, x: dict) -> tuple[str, dict]:
    """Aplica la regla de distancia ponderada a una medición x (dict)."""
    D = {}
    for c in modelo["clases"]:
        mu = modelo["centroides"][c]
        D[c] = sum(modelo["pesos"][v] * ((x[v] - mu[v]) / modelo["sigma"][v]) ** 2
                   for v in modelo["variables"])
    d_min = min(D.values())
    e = {c: np.exp(-(d - d_min) / 2) for c, d in D.items()}   # estable numéricamente
    total = sum(e.values())
    confianza = {c: round(float(v / total), 3) for c, v in e.items()}
    return min(D, key=D.get), confianza


def umbrales_a(modelo: dict, df: pd.DataFrame) -> list[dict]:
    """Cortes sobre a* entre clases consecutivas:
           t = (μ1·σ2 + μ2·σ1) / (σ1 + σ2)
    (punto donde ambas clases quedan a igual número de desviaciones)."""
    stats = df.groupby("estado")["a"].agg(["mean", "std"])
    cortes = []
    clases = modelo["clases"]
    for c1, c2 in zip(clases, clases[1:]):
        m1, s1 = stats.loc[c1]
        m2, s2 = stats.loc[c2]
        t = (m1 * s2 + m2 * s1) / (s1 + s2)
        cortes.append({"entre": [c1, c2], "a_corte": round(float(t), 2)})
    return cortes


def ajustar_indice(df: pd.DataFrame, variable: str = "a") -> dict:
    """Recta indice_madurez = β0 + β1·variable sobre los promedios por papaya,
    con su R² y el error absoluto medio deja-una-papaya-fuera."""
    medias = df.groupby("id_papaya")[[variable, "indice_madurez"]].mean()
    x, y = medias[variable].to_numpy(), medias["indice_madurez"].to_numpy()
    b1, b0 = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (b0 + b1 * x)) ** 2) / np.sum((y - y.mean()) ** 2)
    errores = []
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        c1, c0 = np.polyfit(x[m], y[m], 1)
        errores.append(abs(c0 + c1 * x[i] - y[i]))
    return {"variable": variable, "intercepto": round(float(b0), 4),
            "pendiente": round(float(b1), 4), "minimo": float(y.min()),
            "maximo": float(y.max()), "r2": round(float(r2), 3),
            "mae_deja_una_fuera": round(float(np.mean(errores)), 2)}


# ---------------------------------------------------------------------------
# Validación deja-una-papaya-fuera
# ---------------------------------------------------------------------------
def validar(df: pd.DataFrame) -> pd.DataFrame:
    """Clasifica cada papaya (promedio de sus 9 registros) con un modelo
    ajustado sin ella."""
    filas = []
    for pid, prueba in df.groupby("id_papaya"):
        entreno = df[df["id_papaya"] != pid]
        if prueba["estado"].iloc[0] not in set(entreno["estado"]):
            continue   # su clase no existe sin ella: no se puede evaluar
        modelo = ajustar(entreno)
        x = prueba[VARIABLES].mean().to_dict()
        pred, conf = clasificar(modelo, x)
        filas.append({"id_papaya": pid, "real": prueba["estado"].iloc[0],
                      "predicho": pred, "confianza": conf[pred]})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
def cargar_etiquetas(ruta: Path) -> pd.DataFrame:
    if not ruta.exists():
        plantilla = pd.DataFrame({"id_papaya": range(1, 11),
                                  "indice_madurez": [""] * 10, "estado": [""] * 10})
        plantilla.to_csv(ruta, index=False)
        raise SystemExit(f"Se creó la plantilla {ruta}.\n"
                         f"Escribe el estado real de cada papaya ({', '.join(ORDEN_CLASES)}) "
                         "y vuelve a ejecutar.")
    et = pd.read_csv(ruta, dtype={"estado": str})
    et["estado"] = et["estado"].str.strip().str.lower()
    if et["estado"].isna().any() or (et["estado"] == "").any():
        raise SystemExit(f"Faltan estados en {ruta}. Complétalos y vuelve a ejecutar.")
    return et


if __name__ == "__main__":
    ruta_et = Path(sys.argv[1]) if len(sys.argv) > 1 else ETIQUETAS_CSV
    df = pd.read_csv(DATASET_CSV).merge(cargar_etiquetas(ruta_et), on="id_papaya")

    modelo = ajustar(df)
    modelo["umbrales_a"] = umbrales_a(modelo, df)
    modelo["n_papayas"] = int(df["id_papaya"].nunique())

    val = validar(df)
    if "indice_madurez" in df.columns:
        modelo["indice"] = ajustar_indice(df)
    if len(val):
        modelo["exactitud_deja_una_fuera"] = round(float((val.real == val.predicho).mean()), 3)

    SALIDA_JSON.write_text(json.dumps(modelo, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Peso de cada variable (razón de Fisher normalizada):")
    for v in VARIABLES:
        print(f"   {v:14s} F = {modelo['fisher'][v]:8.3f}   peso = {modelo['pesos'][v]:.3f}")
    print("\nCentroides por clase:")
    print(pd.DataFrame(modelo["centroides"]).T.round(1).to_string())
    print("\nUmbrales simples sobre a*:", modelo["umbrales_a"])
    if len(val):
        print("\nValidación deja-una-papaya-fuera:")
        print(val.to_string(index=False))
        print(f"Exactitud: {modelo['exactitud_deja_una_fuera']:.0%}")
    if "indice" in modelo:
        ind = modelo["indice"]
        print(f"\nÍndice de madurez = {ind['intercepto']:.3f} + {ind['pendiente']:.4f}·a*"
              f"   R² = {ind['r2']:.3f}   error medio (deja-una-fuera) = ±{ind['mae_deja_una_fuera']} puntos")
    print(f"\nParámetros para la app → {SALIDA_JSON}")
