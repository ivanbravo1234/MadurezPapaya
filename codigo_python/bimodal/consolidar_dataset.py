"""
consolidar_dataset.py
=====================
Une los resultados de visión y acústica en un único dataset_papayas.csv.

Flujo:
    vision_color.procesar_carpeta()  → id_papaya, serie, repeticion, L, a, b
    acustica_fft.procesar_carpeta()  → id_papaya, serie, repeticion, frecuencia_hz, amplitud
    merge por (id_papaya, serie, repeticion)
    → data/papayas/dataset_papayas.csv
      columnas: id_papaya, serie, repeticion, L, a, b, frecuencia_hz, amplitud

Además exporta data/papayas/resumen_por_papaya.csv con la media y la
desviación estándar de cada variable por papaya (los 9 registros de cada
fruto: 3 series × 3 repeticiones). Ese resumen es la base para calcular
los umbrales de la regla de decisión de la app.

Uso:
    python bimodal/consolidar_dataset.py            # procesa todo
    python bimodal/consolidar_dataset.py --depurar  # + guarda las máscaras
                                                    #   en outputs/segmentacion
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import acustica_fft
import vision_color
from nomenclatura import LLAVES

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "data" / "papayas"
DATASET_CSV = SALIDA / "dataset_papayas.csv"
RESUMEN_CSV = SALIDA / "resumen_por_papaya.csv"

COLUMNAS = LLAVES + ["L", "a", "b", "frecuencia_hz", "amplitud"]
VARIABLES = ["L", "a", "b", "frecuencia_hz", "amplitud"]


def consolidar(depurar: bool = False) -> pd.DataFrame:
    df_color = vision_color.procesar_carpeta(depurar=depurar)
    df_audio = acustica_fft.procesar_carpeta()

    # how="outer" + indicator: si a alguna muestra le falta la imagen o el
    # audio, aparece igual en el CSV (con vacíos) y se avisa por consola,
    # en vez de perderse en silencio.
    df = df_color.merge(df_audio, on=LLAVES, how="outer", indicator=True)
    huerfanas = df[df["_merge"] != "both"]
    if len(huerfanas):
        print(f"[aviso] {len(huerfanas)} muestra(s) sin pareja imagen/audio:")
        print(huerfanas[LLAVES + ["_merge"]].to_string(index=False))

    df = df[COLUMNAS].sort_values(LLAVES).reset_index(drop=True)
    return df


def resumir(df: pd.DataFrame) -> pd.DataFrame:
    """Media y desviación estándar por papaya."""
    res = df.groupby("id_papaya")[VARIABLES].agg(["mean", "std"])
    res.columns = [f"{v}_{e}" for v, e in res.columns]   # L_mean, L_std, ...
    return res.reset_index()


if __name__ == "__main__":
    SALIDA.mkdir(parents=True, exist_ok=True)
    dataset = consolidar(depurar="--depurar" in sys.argv)
    dataset.to_csv(DATASET_CSV, index=False, float_format="%.6g")
    resumen = resumir(dataset)
    resumen.to_csv(RESUMEN_CSV, index=False, float_format="%.6g")

    print(f"\nDataset ({len(dataset)} filas) → {DATASET_CSV}")
    print(f"Resumen por papaya          → {RESUMEN_CSV}\n")
    print(resumen.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
