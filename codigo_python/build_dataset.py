"""
build_dataset.py
================
Consolida un LOTE de audios en un único CSV de características, una fila
por muestra, listo para entrenar el clasificador (train.py).

Flujo:
    1. Lee data/metadata.csv (columnas: id, masa_g, etiqueta).
    2. Para cada `id`, busca <audio_dir>/<id>.wav.
    3. Carga (audio_io) -> recorta el impulso (preprocessing) ->
       extrae features (features.extract_all_features).
    4. Escribe outputs/features.csv con:
         id, masa_g, etiqueta, f1, f2, damping_logdec, damping_bw,
         S, EI, zcr, ste
    5. Reporta cuántas muestras se procesaron y cuáles fallaron
       (audio inválido, sin impulso, etc.), sin abortar todo el lote.

Uso:
    # Lote sintético (por defecto):
    python build_dataset.py

    # Lote de grabaciones reales en otra carpeta:
    python build_dataset.py --audio-dir data/raw

    # Otro CSV de metadatos o de salida:
    python build_dataset.py --metadata data/raw_meta.csv --out outputs/reales.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config
import audio_io
import preprocessing
import features


def build(audio_dir: Path, metadata_csv: Path, out_csv: Path) -> pd.DataFrame:
    """Procesa el lote y devuelve (además de escribir) el DataFrame."""
    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"No existe el CSV de metadatos: {metadata_csv}\n"
            f"Genera datos con 'python generate_synthetic.py' o crea el CSV "
            f"con columnas: id, masa_g, etiqueta")

    meta = pd.read_csv(metadata_csv)
    required = {"id", "masa_g", "etiqueta"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(
            f"El metadata.csv no tiene las columnas requeridas: {missing}")

    rows = []
    fallidos = []

    for _, r in meta.iterrows():
        sample_id = str(r["id"])
        masa = float(r["masa_g"])
        etiqueta = str(r["etiqueta"])
        wav_path = audio_dir / f"{sample_id}.wav"

        try:
            signal, sr = audio_io.load_wav(wav_path)
            impulse, _ = preprocessing.preprocess(signal, sr)
            feats = features.extract_all_features(impulse, mass_g=masa, sr=sr)

            row = {"id": sample_id, "masa_g": masa, "etiqueta": etiqueta}
            row.update(feats)
            rows.append(row)
        except (FileNotFoundError, ValueError,
                audio_io.AudioValidationError) as e:
            fallidos.append((sample_id, str(e)))
            print(f"  [SKIP] {sample_id}: {e}")

    if not rows:
        raise RuntimeError(
            "No se procesó ninguna muestra. Revisa las rutas y el metadata.")

    df = pd.DataFrame(rows)

    # Orden de columnas estable y legible.
    cols = ["id", "masa_g", "etiqueta", "f1", "f2",
            "damping_logdec", "damping_bw", "S", "EI", "zcr", "ste"]
    df = df[cols]

    config.ensure_dirs()
    df.to_csv(out_csv, index=False)

    print(f"\nDataset escrito en: {out_csv}")
    print(f"  Muestras OK:      {len(df)}")
    print(f"  Muestras fallidas: {len(fallidos)}")
    print(f"  Distribución de clases:\n{df['etiqueta'].value_counts().to_string()}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolida un lote de audios en un CSV de features.")
    parser.add_argument("--audio-dir", default=str(config.SYNTHETIC_DIR),
                        help="Carpeta con los .wav (default: data/synthetic)")
    parser.add_argument("--metadata", default=str(config.METADATA_CSV),
                        help="CSV de metadatos (default: data/metadata.csv)")
    parser.add_argument("--out", default=str(config.FEATURES_CSV),
                        help="CSV de salida (default: outputs/features.csv)")
    args = parser.parse_args()

    build(Path(args.audio_dir), Path(args.metadata), Path(args.out))


if __name__ == "__main__":
    main()
