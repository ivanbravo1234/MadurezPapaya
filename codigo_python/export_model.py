"""
export_model.py
===============
Exporta el modelo entrenado (Random Forest) y los parámetros del pipeline a
un archivo JSON que la app móvil (carpeta `mobile/`) puede leer SIN Python.

Qué se exporta:
    - Parámetros de config.py que usa el pipeline de señal (sample rate,
      tramas STE, umbral de onset, ventana, FFT, banda, distancia de picos).
    - El imputador (medianas por feature) que va antes del Random Forest.
    - Cada árbol del bosque: hijos izq/der, feature, umbral y la fracción de
      cada clase en cada hoja. La app predice igual que sklearn:
          proba = promedio de las fracciones de la hoja alcanzada en cada árbol
          (en cada nodo: x[feature] <= umbral -> izquierda).
    - Información para mostrar: con qué datos se entrenó, cuántas muestras
      y precisión estimada por validación cruzada.

Solo se exporta el Random Forest: es el mejor modelo en la comparación y
además entrega probabilidades (el SVM no).

Uso:
    python export_model.py                       # -> outputs/model_movil.json
    python export_model.py --out ruta/model.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

import config
import train

RF_PATH = config.MODELS_DIR / "randomforest.joblib"
DEFAULT_OUT = config.OUTPUTS_DIR / "model_movil.json"
FUENTE_JSON = config.OUTPUTS_DIR / "fuente_datos.json"


def _fuente() -> str:
    try:
        return json.loads(FUENTE_JSON.read_text(encoding="utf-8"))["fuente"]
    except Exception:
        return "sinteticos"


def _exportar_arbol(estimator) -> dict:
    """Arrays planos del árbol (compactos para JSON)."""
    t = estimator.tree_
    hoja = t.children_left == -1
    # Fracción de clases por nodo (en sklearn >= 1.4 ya viene normalizada;
    # se normaliza de nuevo por robustez frente a versiones anteriores).
    v = t.value[:, 0, :]
    v = v / v.sum(axis=1, keepdims=True)
    return {
        "left": t.children_left.tolist(),
        "right": t.children_right.tolist(),
        "feature": np.where(hoja, -1, t.feature).tolist(),
        "threshold": np.where(hoja, 0.0, t.threshold).tolist(),
        # Solo las hojas necesitan la distribución de clases.
        "value": [np.round(v[i], 6).tolist() if hoja[i] else None
                  for i in range(t.node_count)],
    }


def exportar(out_path: Path = DEFAULT_OUT, calcular_cv: bool = True) -> dict:
    if not RF_PATH.exists():
        raise FileNotFoundError(
            f"No existe {RF_PATH}. Entrena primero el modelo.")
    modelo = joblib.load(RF_PATH)
    imputer = modelo.named_steps["imputer"]
    rf = modelo.named_steps["clf"]

    X, y, feature_cols = train.load_dataset(config.FEATURES_CSV)
    if X.shape[1] != rf.n_features_in_:
        raise ValueError("features.csv no coincide con el modelo guardado; "
                         "reentrena antes de exportar.")

    cv = None
    if calcular_cv:
        _, conteos = np.unique(y, return_counts=True)
        folds = int(min(5, conteos.min()))
        if folds >= 2:
            skf = StratifiedKFold(n_splits=folds, shuffle=True,
                                  random_state=train.RANDOM_STATE)
            cv = float(cross_val_score(train.build_models()["RandomForest"],
                                       X, y, cv=skf).mean())

    clases, conteos = np.unique(y, return_counts=True)
    data = {
        "formato": "papaya-rf-v1",
        "info": {
            "fuente": _fuente(),
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "n_muestras": int(len(y)),
            "por_clase": {str(c): int(n) for c, n in zip(clases, conteos)},
            "cv_accuracy": cv,
        },
        "config": {
            "SAMPLE_RATE": config.SAMPLE_RATE,
            "FRAME_LENGTH": config.FRAME_LENGTH,
            "HOP_LENGTH": config.HOP_LENGTH,
            "ONSET_K": config.ONSET_K,
            "PRE_ONSET_PAD": config.PRE_ONSET_PAD,
            "ANALYSIS_WINDOW": config.ANALYSIS_WINDOW,
            "FFT_SIZE": config.FFT_SIZE,
            "FREQ_MIN": config.FREQ_MIN,
            "FREQ_MAX": config.FREQ_MAX,
            "MIN_PEAK_DISTANCE_HZ": config.MIN_PEAK_DISTANCE_HZ,
            "N_RESONANCES": config.N_RESONANCES,
        },
        "feature_cols": feature_cols,
        "imputer_medians": imputer.statistics_.tolist(),
        "classes": [str(c) for c in rf.classes_],
        "trees": [_exportar_arbol(e) for e in rf.estimators_],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, separators=(",", ":")),
                        encoding="utf-8")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exporta el Random Forest a JSON para la app móvil.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    data = exportar(Path(args.out))
    info = data["info"]
    print(f"Modelo exportado a: {args.out}")
    print(f"  Entrenado con: {info['fuente']}  ({info['n_muestras']} muestras)")
    if info["cv_accuracy"] is not None:
        print(f"  Precisión estimada (CV): {info['cv_accuracy'] * 100:.1f}%")
    print(f"  Árboles: {len(data['trees'])}")


if __name__ == "__main__":
    main()
