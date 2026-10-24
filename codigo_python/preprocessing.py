"""
preprocessing.py
================
Aislamiento del impulso dentro de una grabación: detección de onset,
recorte de la ventana de análisis y normalización.

Idea general de una grabación de golpe a papaya:

    [ silencio / ruido de fondo ] [ ¡GOLPE! transitorio que decae ] [ cola ]
                                  ^
                                  onset

El onset es el instante donde empieza el transitorio. Detectarlo bien es
crítico: si se corta tarde, se pierde el ataque (y el damping sale mal);
si se corta pronto, entra ruido de fondo en la FFT.

Método de detección: energía de corto plazo (STE) con umbral ADAPTATIVO
    umbral = mediana(STE) + k · MAD(STE)
donde MAD es la desviación absoluta mediana. Usar mediana+MAD (robustos a
outliers) en vez de un umbral fijo tolera grabaciones de celulares con
niveles muy distintos.
"""

from __future__ import annotations

import numpy as np

import config


def short_time_energy(signal: np.ndarray,
                      frame_length: int = config.FRAME_LENGTH,
                      hop_length: int = config.HOP_LENGTH) -> np.ndarray:
    """Energía de corto plazo (STE) por tramas.

    STE[m] = Σ_{n en trama m} x[n]²

    Es un detector barato de "dónde hay señal fuerte", ideal para móvil.
    Devuelve un vector con un valor por trama.
    """
    if signal.size < frame_length:
        # Señal más corta que una trama: una sola energía global.
        return np.array([float(np.sum(signal ** 2))])

    n_frames = 1 + (signal.size - frame_length) // hop_length
    energies = np.empty(n_frames)
    for m in range(n_frames):
        start = m * hop_length
        frame = signal[start:start + frame_length]
        energies[m] = np.sum(frame ** 2)
    return energies


def detect_onset(signal: np.ndarray,
                 sr: int = config.SAMPLE_RATE,
                 frame_length: int = config.FRAME_LENGTH,
                 hop_length: int = config.HOP_LENGTH,
                 k: float = config.ONSET_K) -> int:
    """Detecta el índice de muestra donde empieza el impulso.

    Estrategia:
        1. Calcular STE por tramas.
        2. Umbral adaptativo = mediana + k·MAD sobre las tramas.
        3. Primera trama que supera el umbral -> onset.

    Devuelve
    --------
    onset_sample : int
        Índice de muestra (no de trama) del inicio del impulso.

    Lanza
    -----
    ValueError
        Si no se detecta ningún impulso claro (toda la señal por debajo
        del umbral): probablemente la grabación es silencio o ruido plano.
    """
    ste = short_time_energy(signal, frame_length, hop_length)

    if ste.size == 1:
        return 0  # señal muy corta: el impulso es toda la señal

    median = np.median(ste)
    mad = np.median(np.abs(ste - median))
    # Si MAD es ~0 (señal muy uniforme), usar un piso relativo para no
    # dividir contra cero ni disparar el umbral al infinito.
    if mad < 1e-12:
        mad = np.mean(ste) * 0.1 + 1e-12

    threshold = median + k * mad
    above = np.where(ste > threshold)[0]

    if above.size == 0:
        raise ValueError(
            "No se detectó ningún impulso: la señal no supera el umbral "
            "adaptativo. ¿La grabación es silencio o ruido de fondo plano?")

    onset_frame = int(above[0])
    onset_sample = onset_frame * hop_length
    return onset_sample


def extract_impulse(signal: np.ndarray,
                    sr: int = config.SAMPLE_RATE,
                    pre_pad: float = config.PRE_ONSET_PAD,
                    window: float = config.ANALYSIS_WINDOW) -> tuple[np.ndarray, int]:
    """Recorta la ventana de análisis alrededor del impulso.

    Toma desde (onset - pre_pad) hasta (onset + window). El pre-pad
    conserva el ataque; `window` acota el decaimiento útil.

    Devuelve
    --------
    impulse : np.ndarray
        Fragmento recortado que contiene el transitorio.
    onset_sample : int
        Índice del onset en la señal ORIGINAL (útil para graficar).
    """
    onset = detect_onset(signal, sr)

    start = max(0, onset - int(pre_pad * sr))
    end = min(signal.size, onset + int(window * sr))
    impulse = signal[start:end]

    if impulse.size == 0:
        raise ValueError("La ventana de impulso resultó vacía tras el recorte.")

    return impulse, onset


def normalize(signal: np.ndarray, method: str = "peak") -> np.ndarray:
    """Normaliza la señal.

    method="peak" -> divide por el máximo absoluto (pico a ±1).
    method="rms"  -> lleva el RMS a 1 (útil para comparar energías).

    La normalización por pico es la adecuada antes de medir damping,
    porque no altera la FORMA del decaimiento, solo la escala.
    """
    if signal.size == 0:
        raise ValueError("No se puede normalizar una señal vacía.")

    if method == "peak":
        peak = np.max(np.abs(signal))
        return signal / peak if peak > 0 else signal
    elif method == "rms":
        rms = np.sqrt(np.mean(signal ** 2))
        return signal / rms if rms > 0 else signal
    else:
        raise ValueError(f"Método de normalización desconocido: '{method}'")


def preprocess(signal: np.ndarray,
               sr: int = config.SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Pipeline estándar: recorta el impulso y lo normaliza por pico.

    Es el punto de entrada que usarán `features.py` y `build_dataset.py`.

    Devuelve (impulso_normalizado, onset_sample_en_original).
    """
    impulse, onset = extract_impulse(signal, sr)
    impulse = normalize(impulse, method="peak")
    return impulse, onset


if __name__ == "__main__":
    # Prueba: detectar onset en un sintético y reportar el tiempo.
    import audio_io

    p = config.SYNTHETIC_DIR / "verde_01.wav"
    sig, sr = audio_io.load_wav(p)
    impulse, onset = preprocess(sig, sr)
    print(f"{p.name}: onset en muestra {onset} (t={onset/sr:.3f} s), "
          f"impulso recortado a {impulse.size} muestras "
          f"({impulse.size/sr:.3f} s).")
