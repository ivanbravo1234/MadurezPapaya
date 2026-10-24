"""
acustica_fft.py
===============
Módulo de resonancia acústica: frecuencia dominante y amplitud por golpe.

Para cada audio `papaya{N}_serie{S}_rep{R}.wav`:
    1. Carga el WAV y lo pasa a mono.
    2. Filtro pasa-altos Butterworth (100 Hz): las grabaciones de campo traen
       mucho "rumble" < 100 Hz (manipulación del celular, viento, mesa) que
       por energía dominaría la FFT y no tiene que ver con el fruto.
    3. Localiza el impacto (máximo de |x| tras el filtro) y recorta una
       ventana corta: 2 ms antes y 40 ms después. En papaya la respuesta se
       amortigua en ~20–30 ms; una ventana más larga solo suma ruido.
    4. Aplica una ventana con caída coseno al final (media Hann) para evitar
       la fuga espectral del corte, sin atenuar el ataque del golpe.
    5. FFT con relleno de ceros: X(f) = Σ x[n]·w[n]·e^(−j2πfn/fs).
    6. Suaviza |X(f)| (~20 Hz) y busca el máximo en 100–3000 Hz. Afina la
       frecuencia con interpolación parabólica entre bins vecinos.

Salidas por golpe:
    frecuencia_hz : frecuencia del pico espectral dominante (Hz).
    amplitud      : amplitud del componente sinusoidal en ese pico, como
                    fracción de la escala completa del audio (1 = 0 dBFS):
                    A = 2·|X(f_dom)| / Σ w[n].
                    Depende de la fuerza del golpe y de la distancia del
                    micrófono: solo es comparable si el péndulo y la posición
                    del celular son siempre los mismos.

Nota física: un fruto más firme (verde) vibra con mayor frecuencia y menor
amortiguamiento; al madurar la pulpa se ablanda, la frecuencia baja y el
golpe "se apaga" antes.

Uso independiente:
    python bimodal/acustica_fft.py  [carpeta_audios]
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from nomenclatura import listar_muestras

# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
CARPETA_AUDIOS = RAIZ / "data" / "papayas" / "audios"

CORTE_PASA_ALTOS = 100.0     # Hz
ORDEN_FILTRO = 4
PRE_IMPACTO = 0.002          # s antes del pico
POST_IMPACTO = 0.040         # s después del pico
N_FFT = 2 ** 15              # 32768 → resolución fs/N ≈ 1,5 Hz a 48 kHz
SUAVIZADO_HZ = 20.0          # ancho del promedio móvil sobre |X(f)|
FREQ_MIN, FREQ_MAX = 100.0, 3000.0


# ---------------------------------------------------------------------------
# 1. Carga
# ---------------------------------------------------------------------------
def cargar_audio(ruta: Path) -> tuple[np.ndarray, int]:
    """Lee un WAV y devuelve (señal float64 mono, frecuencia de muestreo).

    Los WAV PCM enteros se escalan a [−1, 1] (fracción de la escala
    completa, igual que el micrófono de la app). No se normaliza al pico:
    así `amplitud` conserva cuán fuerte sonó el golpe.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)  # chunks de metadatos
        fs, x = wavfile.read(ruta)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if len(x) == 0 or not np.any(x):
        raise ValueError("audio vacío o en silencio")
    return x, fs


# ---------------------------------------------------------------------------
# 2-4. Preprocesamiento
# ---------------------------------------------------------------------------
def filtrar_pasa_altos(x: np.ndarray, fs: int, corte: float = CORTE_PASA_ALTOS) -> np.ndarray:
    """Butterworth pasa-altos de fase cero (sosfiltfilt filtra ida y vuelta,
    así el impacto no se desplaza en el tiempo)."""
    sos = butter(ORDEN_FILTRO, corte, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def recortar_impacto(x: np.ndarray, fs: int) -> np.ndarray:
    """Recorta la ventana [pico − 2 ms, pico + 40 ms] alrededor del golpe."""
    k = int(np.argmax(np.abs(x)))
    ini = max(0, k - int(PRE_IMPACTO * fs))
    fin = min(len(x), k + int(POST_IMPACTO * fs))
    return x[ini:fin]


def ventana_caida_coseno(n: int) -> np.ndarray:
    """Ventana plana en la primera mitad y media Hann descendente en la segunda.

    Una Hann completa anularía el inicio, justo donde está el ataque del golpe.
    """
    w = np.ones(n)
    m = n // 2
    w[m:] = np.hanning(2 * (n - m))[n - m:]
    return w


# ---------------------------------------------------------------------------
# 5-6. FFT y pico dominante
# ---------------------------------------------------------------------------
def espectro(segmento: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray, float]:
    """FFT de un solo lado. Devuelve (frecuencias, |X|, Σw)."""
    w = ventana_caida_coseno(len(segmento))
    X = np.abs(np.fft.rfft(segmento * w, n=N_FFT))
    f = np.fft.rfftfreq(N_FFT, d=1.0 / fs)
    return f, X, float(w.sum())


def pico_dominante(f: np.ndarray, X: np.ndarray, suma_ventana: float) -> tuple[float, float]:
    """Frecuencia y amplitud del pico máximo en [FREQ_MIN, FREQ_MAX].

    - El suavizado evita que un bin aislado de ruido gane la búsqueda (el
      espectro de un golpe muy amortiguado es ancho, no una línea fina).
    - La interpolación parabólica sobre 3 bins (y−1, y0, y+1) da el vértice:
          δ = ½·(y−1 − y+1) / (y−1 − 2·y0 + y+1),   f* = f0 + δ·Δf
    """
    df = f[1] - f[0]
    Xs = uniform_filter1d(X, size=max(1, int(round(SUAVIZADO_HZ / df))))
    banda = np.where((f >= FREQ_MIN) & (f <= FREQ_MAX))[0]
    i = banda[int(np.argmax(Xs[banda]))]

    y0, y1, y2 = Xs[i - 1], Xs[i], Xs[i + 1]
    den = y0 - 2 * y1 + y2
    delta = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    frecuencia = f[i] + delta * df

    # Amplitud de una sinusoide: A = 2·|X| / Σw (se usa |X| sin suavizar).
    amplitud = 2.0 * X[i] / suma_ventana
    return float(frecuencia), float(amplitud)


# ---------------------------------------------------------------------------
# Procesamiento de la carpeta
# ---------------------------------------------------------------------------
def procesar_audio(ruta: Path) -> dict:
    """Pipeline completo para un golpe."""
    x, fs = cargar_audio(ruta)
    segmento = recortar_impacto(filtrar_pasa_altos(x, fs), fs)
    f, X, sw = espectro(segmento, fs)
    frecuencia, amplitud = pico_dominante(f, X, sw)
    return {"frecuencia_hz": frecuencia, "amplitud": amplitud}


def procesar_carpeta(carpeta: Path = CARPETA_AUDIOS) -> pd.DataFrame:
    """Itera todos los WAV con nombre estándar y devuelve un DataFrame con
    columnas id_papaya, serie, repeticion, frecuencia_hz, amplitud."""
    filas = []
    muestras = listar_muestras(carpeta, (".wav",))
    for i, (llave, ruta) in enumerate(muestras, 1):
        try:
            filas.append({**llave, **procesar_audio(ruta)})
        except ValueError as e:
            print(f"[error] {ruta.name}: {e}")
            filas.append({**llave, "frecuencia_hz": np.nan, "amplitud": np.nan})
        print(f"\r[acústica] {i}/{len(muestras)} {ruta.name}      ", end="")
    print()
    return pd.DataFrame(filas)


if __name__ == "__main__":
    carpeta = Path(sys.argv[1]) if len(sys.argv) > 1 else CARPETA_AUDIOS
    print(procesar_carpeta(carpeta).round(2).to_string(index=False))
