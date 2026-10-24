"""
audio_io.py
===========
Carga y validación de archivos .wav de golpes a papaya.

Responsabilidades:
    - Leer un .wav (PCM entero o float).
    - Convertir a float mono en el rango [-1, 1].
    - Resamplear al sample rate objetivo (config.SAMPLE_RATE) si difiere.
    - Validar la señal y dar mensajes de error CLAROS ante entradas malas
      (archivo inexistente, vacío, todo silencio, saturado/clipping).

El resampling usa `scipy.signal.resample_poly` (filtro polifásico), que no
requiere librosa/numba y funciona sin fricción en Python 3.13.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Union

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

import config


class AudioValidationError(Exception):
    """Error de validación de una grabación de audio."""


def _to_float_mono(data: np.ndarray) -> np.ndarray:
    """Convierte datos crudos de wav a float64 mono en [-1, 1].

    - Enteros (int16/int32/uint8) se escalan por su rango máximo.
    - Float se asume ya en [-1, 1].
    - Estéreo se promedia a mono (el análisis modal no usa estéreo).
    """
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        if info.min == 0:
            # PCM sin signo (uint8): centrar en 0 antes de normalizar.
            data = data.astype(np.float64) - (info.max + 1) / 2.0
            data = data / ((info.max + 1) / 2.0)
        else:
            data = data.astype(np.float64) / info.max
    else:
        data = data.astype(np.float64)

    if data.ndim == 2 and config.FORCE_MONO:
        data = data.mean(axis=1)

    return data


def _resample(signal: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resamplea `signal` de sr_in a sr_out con filtro polifásico.

    Se aproxima la razón sr_out/sr_in por una fracción racional up/down
    con denominador acotado, para que resample_poly sea eficiente.
    """
    if sr_in == sr_out:
        return signal
    ratio = Fraction(sr_out, sr_in).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    return resample_poly(signal, up, down)


def load_wav(path: Union[str, Path],
             target_sr: int = config.SAMPLE_RATE,
             validate: bool = True) -> tuple[np.ndarray, int]:
    """Carga un .wav y lo devuelve como (señal float mono, sample_rate).

    Parámetros
    ----------
    path : str | Path
        Ruta al archivo .wav.
    target_sr : int
        Sample rate objetivo; si el archivo difiere, se resamplea.
    validate : bool
        Si True, aplica `validate_signal` y lanza AudioValidationError
        ante señales inservibles.

    Devuelve
    --------
    signal : np.ndarray
        Señal float64 mono en [-1, 1], resampleada a target_sr.
    sr : int
        Sample rate resultante (== target_sr).

    Lanza
    -----
    FileNotFoundError
        Si el archivo no existe.
    AudioValidationError
        Si el contenido no es utilizable (vacío, silencio, saturado).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de audio: {path}")
    if path.suffix.lower() != ".wav":
        raise AudioValidationError(
            f"Formato no soportado: '{path.suffix}'. Se espera .wav")

    try:
        file_sr, data = wavfile.read(str(path))
    except Exception as exc:  # archivo corrupto o no-wav
        raise AudioValidationError(
            f"No se pudo leer '{path.name}' como .wav: {exc}") from exc

    signal = _to_float_mono(data)
    signal = _resample(signal, file_sr, target_sr)

    if validate:
        validate_signal(signal, name=path.name)

    return signal, target_sr


def validate_signal(signal: np.ndarray, name: str = "audio",
                    silence_rms: float = 1e-4,
                    clip_fraction: float = 0.01) -> None:
    """Valida una señal cargada y lanza AudioValidationError si es mala.

    Comprobaciones:
        - No vacía y sin NaN/Inf.
        - No es prácticamente silencio (RMS por encima de `silence_rms`).
        - No está saturada: menos de `clip_fraction` de muestras al tope
          (|x| >= 0.999). El clipping destruye la forma del transitorio
          y arruina el cálculo de damping.
    """
    if signal.size == 0:
        raise AudioValidationError(f"'{name}': la señal está vacía.")

    if not np.all(np.isfinite(signal)):
        raise AudioValidationError(f"'{name}': la señal contiene NaN o Inf.")

    rms = float(np.sqrt(np.mean(signal ** 2)))
    if rms < silence_rms:
        raise AudioValidationError(
            f"'{name}': parece silencio (RMS={rms:.2e}). "
            f"¿El golpe quedó fuera de la grabación?")

    clipped = np.mean(np.abs(signal) >= 0.999)
    if clipped > clip_fraction:
        raise AudioValidationError(
            f"'{name}': audio saturado ({clipped*100:.1f}% de muestras al "
            f"tope). Reduce el volumen de grabación o aleja el micrófono.")


if __name__ == "__main__":
    # Prueba rápida: carga y valida todos los sintéticos.
    import sys
    from glob import glob

    paths = sys.argv[1:] or sorted(glob(str(config.SYNTHETIC_DIR / "*.wav")))
    if not paths:
        print("No hay .wav para probar. Corre antes generate_synthetic.py")
        sys.exit(1)

    for p in paths:
        try:
            sig, sr = load_wav(p)
            print(f"OK  {Path(p).name:16}  n={sig.size:6d}  sr={sr}  "
                  f"RMS={np.sqrt(np.mean(sig**2)):.4f}")
        except (AudioValidationError, FileNotFoundError) as e:
            print(f"ERR {Path(p).name:16}  {e}")
