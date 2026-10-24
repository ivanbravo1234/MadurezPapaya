"""
generate_synthetic.py
======================
Genera audios .wav SINTÉTICOS de golpes a papaya para probar todo el
pipeline ANTES de tener grabaciones reales.

Modelo físico de cada golpe (respuesta a impulso de un sistema con
varios modos amortiguados):

        x(t) = Σ_i  A_i · exp(-ζ_i · ω_i · t) · sin(ω_i · t)  +  ruido

    donde:
        ω_i = 2π · f_i     frecuencia angular del modo i (rad/s)
        f_i                frecuencia natural del modo i (Hz)
        ζ_i                razón de amortiguación del modo i (adimensional)
        A_i                amplitud inicial del modo i

Interpretación física (nuestro "ground truth"):
    - VERDE  : pulpa firme  -> f1 alta, ζ bajo  (vibra rápido y decae lento).
    - MADURO : pulpa blanda -> f1 baja, ζ alto  (vibra lento y decae rápido).
    - SEMI   : intermedio.

Cada .wav lleva un tramo de silencio+ruido ANTES del impulso, para poder
probar la detección de onset de la Tanda 2.

Salida:
    - data/synthetic/<id>.wav  (uno por muestra)
    - data/metadata.csv        (columnas: id, masa_g, etiqueta)

Uso:
    python generate_synthetic.py
"""

from __future__ import annotations

import csv

import numpy as np
from scipy.io import wavfile

import config


def _damped_impulse(f1: float, f2: float, zeta: float,
                    duration: float, sr: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Construye un impulso amortiguado de dos modos (f1, f2).

    El segundo modo (f2) se genera con la misma ζ y algo menos de
    amplitud, imitando que las resonancias superiores suelen ser más
    débiles y decaen parecido.
    """
    t = np.arange(int(duration * sr)) / sr

    modes = [
        # (frecuencia, zeta, amplitud)
        (f1, zeta, 1.00),
        (f2, zeta * 1.1, 0.45),
    ]

    x = np.zeros_like(t)
    for f, z, amp in modes:
        omega = 2.0 * np.pi * f
        x += amp * np.exp(-z * omega * t) * np.sin(omega * t)

    # Normalizar el impulso a amplitud pico ~0.9 (deja headroom).
    peak = np.max(np.abs(x))
    if peak > 0:
        x = 0.9 * x / peak
    return x


def _assemble_recording(impulse: np.ndarray, sr: int,
                        pre_silence: float, noise_level: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Antepone silencio con ruido y añade ruido de fondo global.

    Resultado: [ silencio+ruido | impulso+ruido ], como una grabación
    real de celular donde el micrófono capta algo antes del golpe.
    """
    n_pre = int(pre_silence * sr)
    pre = np.zeros(n_pre)

    signal = np.concatenate([pre, impulse])
    # Ruido blanco gaussiano en toda la grabación.
    noise = noise_level * rng.standard_normal(signal.size)
    return signal + noise


def _to_int16(signal: np.ndarray) -> np.ndarray:
    """Convierte float [-1, 1] a PCM int16, con recorte defensivo."""
    clipped = np.clip(signal, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def generate(classes: dict | None = None,
             n_per_class: int = config.SYNTHETIC_N_PER_CLASS,
             freq_jitter: float = config.SYNTHETIC_FREQ_JITTER,
             zeta_jitter: float = config.SYNTHETIC_ZETA_JITTER,
             mass_jitter: float = config.SYNTHETIC_MASS_JITTER,
             noise_level: float = config.SYNTHETIC_NOISE_LEVEL,
             seed: int = config.RANDOM_SEED) -> list[dict]:
    """Genera todos los .wav sintéticos y actualiza metadata.csv.

    Por defecto usa los valores de config.py; la interfaz (app.py) puede
    pasar otros sin tocar ese archivo. `classes` tiene la forma de
    config.SYNTHETIC_CLASSES: {etiqueta: {f1, f2, zeta, masa_media}}.

    Antes de generar se borran los .wav sintéticos anteriores (para no dejar
    huérfanos si ahora se generan menos) y sus filas en metadata.csv. Las
    filas de grabaciones REALES se conservan.

    Devuelve la lista de muestras generadas con su ground truth.
    """
    classes = classes or config.SYNTHETIC_CLASSES
    config.ensure_dirs()
    rng = np.random.default_rng(seed)
    sr = config.SAMPLE_RATE

    # --- Limpiar la generación anterior (solo lo sintético) ---
    old_ids = set()
    for old in config.SYNTHETIC_DIR.glob("*.wav"):
        old_ids.add(old.stem)
        old.unlink()
    kept = []
    if config.METADATA_CSV.exists():
        with open(config.METADATA_CSV, newline="", encoding="utf-8") as f:
            kept = [r for r in csv.DictReader(f) if r["id"] not in old_ids]

    rows = []       # filas para metadata.csv
    generated = []  # ground truth de cada muestra

    for etiqueta, params in classes.items():
        for k in range(n_per_class):
            # --- Jitter por muestra: las clases se solapan un poco ---
            fj = 1.0 + rng.uniform(-freq_jitter, freq_jitter)
            f1 = params["f1"] * fj
            f2 = params["f2"] * fj  # f2 escala junto a f1 (mismo fruto)

            zj = 1.0 + rng.uniform(-zeta_jitter, zeta_jitter)
            zeta = params["zeta"] * zj

            masa = params["masa_media"] + rng.uniform(-mass_jitter,
                                                      mass_jitter)

            # --- Construir la grabación ---
            impulse = _damped_impulse(f1, f2, zeta,
                                      config.SYNTHETIC_DURATION, sr, rng)
            recording = _assemble_recording(
                impulse, sr, config.SYNTHETIC_PRE_SILENCE, noise_level, rng)

            sample_id = f"{etiqueta}_{k+1:02d}"
            out_path = config.SYNTHETIC_DIR / f"{sample_id}.wav"
            wavfile.write(str(out_path), sr, _to_int16(recording))

            rows.append({
                "id": sample_id,
                "masa_g": round(masa, 1),
                "etiqueta": etiqueta,
            })
            generated.append({"id": sample_id, "etiqueta": etiqueta,
                              "f1": f1, "f2": f2, "zeta": zeta,
                              "masa_g": round(masa, 1)})
            print(f"  [{etiqueta:>6}] {sample_id}.wav  "
                  f"f1={f1:6.1f} Hz  f2={f2:6.1f} Hz  "
                  f"zeta={zeta:.4f}  masa={masa:5.1f} g")

    # --- Escribir metadata.csv (id = nombre del .wav sin extensión) ---
    # Primero las filas sintéticas nuevas, luego las reales que ya existían.
    new_ids = {r["id"] for r in rows}
    kept = [r for r in kept if r["id"] not in new_ids]
    with open(config.METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "masa_g", "etiqueta"])
        writer.writeheader()
        writer.writerows(rows + [{k: r[k] for k in ("id", "masa_g",
                                                    "etiqueta")}
                                 for r in kept])

    print(f"\nGenerados {len(rows)} audios en: {config.SYNTHETIC_DIR}")
    print(f"Metadatos escritos en:       {config.METADATA_CSV}")
    if kept:
        print(f"Se conservaron {len(kept)} filas de grabaciones reales.")
    return generated


if __name__ == "__main__":
    print("Generando audios sintéticos de papaya (verde / semi / maduro)...\n")
    generate()
