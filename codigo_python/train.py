"""
train.py
========
Entrena y COMPARA dos clasificadores ligeros que discriminan la madurez
de la papaya en 3 clases: verde / semi / maduro.

Modelos:
    - SVM (SVC con kernel RBF) sobre features estandarizadas.
    - Random Forest.

Para cada modelo reporta:
    - Accuracy en un conjunto de prueba (hold-out estratificado).
    - Matriz de confusión (texto + PNG en outputs/figures).
    - Reporte por clase (precision / recall / f1).
    - Validación cruzada estratificada (k folds) sobre TODO el dataset,
      que es más honesta que un solo split cuando hay pocas muestras.

Además:
    - Muestra la importancia de features del Random Forest (qué parámetro
      pesa más: se espera que el damping domine en pulpa blanda).
    - Guarda ambos modelos entrenados en outputs/models/*.joblib.

Uso:
    python train.py
    python train.py --features outputs/features.csv --test-size 0.3 --folds 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)

import config

# Columnas que NO son features (identificador y objetivo).
NON_FEATURE_COLS = {"id", "etiqueta"}

# Orden natural de madurez, para que la matriz de confusión se lea bien.
CLASS_ORDER = ["verde", "semi", "maduro"]

RANDOM_STATE = 42


def load_dataset(features_csv: Path):
    """Carga el CSV de features y separa X (features), y (etiquetas).

    Las columnas de features son todas menos `id` y `etiqueta` (incluye
    masa_g y los índices S/EI). Valores faltantes se dejan para que el
    imputador del pipeline los resuelva.
    """
    if not features_csv.exists():
        raise FileNotFoundError(
            f"No existe {features_csv}. Genera el dataset con "
            f"'python build_dataset.py' primero.")

    df = pd.read_csv(features_csv)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    X = df[feature_cols].to_numpy(dtype=float)
    y = df["etiqueta"].astype(str).to_numpy()
    return X, y, feature_cols


def build_models() -> dict[str, Pipeline]:
    """Define los dos pipelines a comparar.

    Ambos empiezan con imputación por mediana (robusta a NaN de features
    que no se pudieron calcular). El SVM además estandariza (media 0,
    varianza 1), imprescindible porque S y EI tienen escalas gigantes
    frente a ζ o ZCR; Random Forest no necesita escalado.
    """
    svm = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10.0, gamma="scale",
                    random_state=RANDOM_STATE)),
    ])

    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=300,
                                       random_state=RANDOM_STATE)),
    ])

    return {"SVM": svm, "RandomForest": rf}


def evaluate(name: str, model: Pipeline, X, y,
             test_size: float, folds: int) -> None:
    """Entrena, evalúa en hold-out y por validación cruzada, y reporta."""
    print("\n" + "=" * 60)
    print(f"MODELO: {name}")
    print("=" * 60)

    # --- Hold-out estratificado ---
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    print(f"\nAccuracy (hold-out, {int(test_size*100)}% test): {acc:.3f}")

    print("\nMatriz de confusión (filas=real, cols=predicho):")
    labels = [c for c in CLASS_ORDER if c in np.unique(y)]
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    print("           " + "  ".join(f"{l:>7}" for l in labels))
    for i, l in enumerate(labels):
        print(f"    {l:>7} " + "  ".join(f"{v:>7d}" for v in cm[i]))

    print("\nReporte por clase:")
    print(classification_report(y_te, y_pred, labels=labels, zero_division=0))

    # --- Validación cruzada sobre TODO el dataset ---
    skf = StratifiedKFold(n_splits=folds, shuffle=True,
                          random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    print(f"Validación cruzada ({folds}-fold): "
          f"accuracy = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"   folds: {np.round(cv_scores, 3)}")

    # --- Reentrenar en TODO el dataset y guardar el modelo final ---
    model.fit(X, y)
    config.ensure_dirs()
    model_path = config.MODELS_DIR / f"{name.lower()}.joblib"
    joblib.dump(model, model_path)
    print(f"Modelo guardado en: {model_path}")

    # --- Guardar matriz de confusión como PNG ---
    _save_confusion_png(name, y_te, y_pred, labels)


def _save_confusion_png(name, y_true, y_pred, labels) -> None:
    """Guarda la matriz de confusión del hold-out como imagen."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, ax=ax, cmap="Oranges",
        colorbar=False)
    ax.set_title(f"Matriz de confusión — {name}")
    fig.tight_layout()
    out = config.FIGURES_DIR / f"confusion_{name.lower()}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Matriz de confusión guardada en: {out}")


def report_feature_importance(model: Pipeline, feature_cols) -> None:
    """Importancia de features del Random Forest (ordenada)."""
    rf = model.named_steps["clf"]
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    print("\n" + "=" * 60)
    print("IMPORTANCIA DE FEATURES (Random Forest)")
    print("=" * 60)
    for i in order:
        bar = "#" * int(round(importances[i] * 50))
        print(f"  {feature_cols[i]:>16}: {importances[i]:.3f}  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena y compara SVM vs Random Forest de madurez.")
    parser.add_argument("--features", default=str(config.FEATURES_CSV),
                        help="CSV de features (default: outputs/features.csv)")
    parser.add_argument("--test-size", type=float, default=0.3,
                        help="Fracción para el conjunto de prueba (default 0.3)")
    parser.add_argument("--folds", type=int, default=5,
                        help="Nº de folds para validación cruzada (default 5)")
    args = parser.parse_args()

    X, y, feature_cols = load_dataset(Path(args.features))
    print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} features.")
    print(f"Features: {feature_cols}")

    models = build_models()
    for name, model in models.items():
        evaluate(name, model, X, y, args.test_size, args.folds)

    # Importancia de features del RF ya entrenado sobre todo el dataset.
    report_feature_importance(models["RandomForest"], feature_cols)


if __name__ == "__main__":
    main()
