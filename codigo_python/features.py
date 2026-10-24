"""
features.py
===========
Extracción de características espectrales y temporales de un impulso acústico
para estimar la madurez de una papaya.

Contenido:
    Espectrales:
        - compute_spectrum : FFT de magnitud (ventana Hann).
        - find_resonances  : primeras resonancias f1, f2.
        - dominant_frequency
    Temporales:
        - hilbert_envelope : envolvente de amplitud.
        - zero_crossing_rate (ZCR)
        - short_time_energy_mean (STE)
    Amortiguación (parámetro estrella en pulpa blanda):
        - damping_log_decrement : por decaimiento de la envolvente.
        - damping_bandwidth     : por ancho de banda a -3 dB.
    Índices de firmeza:
        - stiffness_index  S  = f1² · m^(2/3)
        - elasticity_index EI = f2² · m^(2/3)
    Orquestador:
        - extract_all_features : devuelve un dict con todo (lo usa
          build_dataset.py).

Notación:
    x[n]  : señal discreta, muestreada a `sr` Hz.
    ω     : frecuencia angular = 2π·f.
    ζ     : razón de amortiguación (adimensional).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import hilbert, find_peaks

import config


# ===========================================================================
# ESPECTRALES
# ===========================================================================
def compute_spectrum(signal: np.ndarray, sr: int = config.SAMPLE_RATE,
                     fft_size: int = config.FFT_SIZE):
    """Espectro de magnitud (FFT real unilateral) con ventana de Hann.

    Se usa `rfft` porque el audio es real (basta la mitad positiva del
    espectro). La ventana de Hann reduce el "spectral leakage" que
    ensancharía los picos de resonancia. El zero-padding hasta `fft_size`
    interpola el espectro para picos visualmente más suaves.

    Devuelve (freqs, mag) del mismo largo.
    """
    if signal.size == 0:
        raise ValueError("La señal está vacía; no se puede calcular la FFT.")

    n = max(fft_size, signal.size)
    window = np.hanning(signal.size)
    spectrum = rfft(signal * window, n=n)
    mag = np.abs(spectrum)
    freqs = rfftfreq(n, d=1.0 / sr)
    return freqs, mag


def find_resonances(freqs: np.ndarray, mag: np.ndarray,
                    n_peaks: int = config.N_RESONANCES,
                    fmin: float = config.FREQ_MIN,
                    fmax: float = config.FREQ_MAX,
                    min_distance_hz: float = config.MIN_PEAK_DISTANCE_HZ):
    """Primeras `n_peaks` resonancias del espectro, en banda [fmin, fmax].

    Los picos se seleccionan por PROMINENCIA (no altura absoluta) para no
    confundir ruido de fondo con resonancias, exigiendo separación mínima
    entre picos. Se devuelven ordenados por frecuencia ascendente.

    Devuelve (peak_freqs, peak_mags) como listas (posiblemente < n_peaks).
    """
    band = (freqs >= fmin) & (freqs <= fmax)
    f_band = freqs[band]
    m_band = mag[band]

    if f_band.size < 2:
        return [], []

    df = f_band[1] - f_band[0]
    distance = max(1, int(round(min_distance_hz / df)))

    peaks, props = find_peaks(m_band, distance=distance,
                              prominence=np.max(m_band) * 0.05)

    if peaks.size == 0:
        idx = int(np.argmax(m_band))
        return [float(f_band[idx])], [float(m_band[idx])]

    order = np.argsort(props["prominences"])[::-1][:n_peaks]
    selected = peaks[order]
    selected = selected[np.argsort(f_band[selected])]

    peak_freqs = [float(f_band[i]) for i in selected]
    peak_mags = [float(m_band[i]) for i in selected]
    return peak_freqs, peak_mags


def dominant_frequency(freqs: np.ndarray, mag: np.ndarray,
                       fmin: float = config.FREQ_MIN,
                       fmax: float = config.FREQ_MAX) -> float:
    """Frecuencia dominante f1 = frecuencia del máximo espectral en banda."""
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        raise ValueError("No hay bins de FFT dentro de la banda [fmin, fmax].")
    f_band = freqs[band]
    m_band = mag[band]
    return float(f_band[int(np.argmax(m_band))])


# ===========================================================================
# TEMPORALES
# ===========================================================================
def hilbert_envelope(signal: np.ndarray) -> np.ndarray:
    """Envolvente de amplitud instantánea por transformada de Hilbert.

    La señal analítica es z[n] = x[n] + j·H{x[n]}; su módulo |z[n]| es la
    envolvente que "abraza" las oscilaciones y sobre la que se mide el
    decaimiento exponencial (damping).
    """
    if signal.size == 0:
        raise ValueError("La señal está vacía; no se puede calcular la envolvente.")
    return np.abs(hilbert(signal))


def zero_crossing_rate(signal: np.ndarray) -> float:
    """Zero-crossing rate (ZCR): fracción de cambios de signo.

    ZCR = (nº de cruces por cero) / (nº de muestras - 1)

    Feature barata muy usada en audio móvil: correlaciona con el contenido
    de alta frecuencia. Un impulso más "agudo" (verde) tiende a mayor ZCR.
    """
    if signal.size < 2:
        return 0.0
    signs = np.sign(signal)
    signs[signs == 0] = 1  # tratar ceros exactos como positivos
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(crossings) / (signal.size - 1)


def short_time_energy_mean(signal: np.ndarray,
                           frame_length: int = config.FRAME_LENGTH,
                           hop_length: int = config.HOP_LENGTH) -> float:
    """Short-time energy (STE) promedio de la señal.

    STE por trama = Σ x[n]²; se promedia sobre todas las tramas para
    obtener un escalar. Resume cuánta energía tiene el transitorio.
    """
    if signal.size < frame_length:
        return float(np.sum(signal ** 2))

    n_frames = 1 + (signal.size - frame_length) // hop_length
    total = 0.0
    for m in range(n_frames):
        start = m * hop_length
        frame = signal[start:start + frame_length]
        total += np.sum(frame ** 2)
    return total / n_frames


# ===========================================================================
# AMORTIGUACIÓN (DAMPING)
# ===========================================================================
def damping_log_decrement(signal: np.ndarray, f1: float,
                          sr: int = config.SAMPLE_RATE,
                          env: Optional[np.ndarray] = None) -> float:
    """Razón de amortiguación ζ por decaimiento de la envolvente.

    Modelo de vibración libre amortiguada:
        A(t) = A0 · exp(-ζ · ω_n · t),   con ω_n = 2π·f1

    Tomando logaritmo:
        ln A(t) = ln A0 - (ζ · ω_n) · t

    es una recta cuya pendiente es  m = -ζ·ω_n. Ajustando la recta a
    ln(envolvente) en la zona de decaimiento se despeja:

        ζ = -m / ω_n = -m / (2π·f1)

    (Aproximación de amortiguación ligera: ω_amortiguada ≈ ω_n, válida para
     ζ pequeño, que es el caso de la papaya.)

    El ajuste se hace desde el pico de la envolvente hasta que esta cae a
    ~5 % del pico (o fin de señal), evitando la cola de ruido donde el
    logaritmo se vuelve inestable.

    Devuelve ζ (adimensional), o np.nan si no se puede ajustar.
    """
    if f1 <= 0:
        return float("nan")
    if env is None:
        env = hilbert_envelope(signal)

    peak_idx = int(np.argmax(env))
    peak_val = env[peak_idx]
    if peak_val <= 0:
        return float("nan")

    # Zona de decaimiento: del pico hasta caer al 5 % del pico.
    floor = 0.05 * peak_val
    decay = env[peak_idx:]
    below = np.where(decay < floor)[0]
    end = below[0] if below.size > 0 else decay.size
    if end < 4:
        return float("nan")  # muy pocos puntos para un ajuste fiable

    seg = decay[:end]
    t = np.arange(seg.size) / sr

    # Ajuste lineal de ln(env) vs t. Se protege el log con un piso.
    log_env = np.log(np.maximum(seg, 1e-12))
    slope, _ = np.polyfit(t, log_env, 1)

    omega_n = 2.0 * np.pi * f1
    zeta = -slope / omega_n
    return float(zeta)


def damping_bandwidth(freqs: np.ndarray, mag: np.ndarray, f1: float,
                      fmin: float = config.FREQ_MIN,
                      fmax: float = config.FREQ_MAX) -> float:
    """Razón de amortiguación ζ por ancho de banda a -3 dB (half-power).

    En el pico de resonancia f1, los puntos de media potencia son donde la
    magnitud cae a  1/√2  del pico (−3 dB en potencia). Si Δf = f_alta −
    f_baja es ese ancho de banda, entonces:

        Q = f1 / Δf        (factor de calidad)
        ζ = Δf / (2·f1) = 1 / (2Q)

    Método sensible al ruido y a la resolución espectral; por eso se
    reporta JUNTO al log-decrement para contrastar.

    Devuelve ζ (adimensional), o np.nan si no se hallan los cruces.
    """
    band = (freqs >= fmin) & (freqs <= fmax)
    f_band = freqs[band]
    m_band = mag[band]
    if f_band.size < 3:
        return float("nan")

    # Localizar el bin del pico más cercano a f1.
    peak_idx = int(np.argmin(np.abs(f_band - f1)))
    peak_val = m_band[peak_idx]
    half_power = peak_val / np.sqrt(2.0)

    # Buscar f_baja: bajando en frecuencia desde el pico.
    lo = peak_idx
    while lo > 0 and m_band[lo] > half_power:
        lo -= 1
    # Buscar f_alta: subiendo en frecuencia desde el pico.
    hi = peak_idx
    while hi < m_band.size - 1 and m_band[hi] > half_power:
        hi += 1

    if lo == hi:
        return float("nan")

    f_low = _interp_crossing(f_band, m_band, lo, lo + 1, half_power)
    f_high = _interp_crossing(f_band, m_band, hi - 1, hi, half_power)

    bandwidth = f_high - f_low
    if bandwidth <= 0:
        return float("nan")
    return float(bandwidth / (2.0 * f1))


def _interp_crossing(f: np.ndarray, m: np.ndarray, i: int, j: int,
                     level: float) -> float:
    """Interpola linealmente la frecuencia donde la magnitud cruza `level`.

    Mejora la resolución del ancho de banda por debajo del tamaño de bin.
    """
    m_i, m_j = m[i], m[j]
    if m_j == m_i:
        return float(f[i])
    frac = (level - m_i) / (m_j - m_i)
    return float(f[i] + frac * (f[j] - f[i]))


# ===========================================================================
# ÍNDICES DE FIRMEZA
# ===========================================================================
def stiffness_index(f1: float, mass_g: float) -> float:
    """Índice de rigidez  S = f1² · m^(2/3).

    Deriva del modelo de vibración de un cuerpo elástico: la frecuencia
    natural cumple f ∝ √(k/m) para un modo, y normalizando por la masa
    elevada a 2/3 (proporcional a la sección transversal ∝ V^(2/3)) se
    obtiene un índice ∝ a la rigidez del tejido, poco dependiente del
    tamaño del fruto. (Cooke & Rand; Abbott et al.)

    m en gramos.
    """
    return float(f1 ** 2 * mass_g ** (2.0 / 3.0))


def elasticity_index(f2: float, mass_g: float) -> float:
    """Índice de elasticidad  EI = f2² · m^(2/3).

    Igual construcción que S pero usando la segunda resonancia f2, que
    capta un modo de vibración distinto y aporta información elástica
    complementaria. m en gramos.
    """
    return float(f2 ** 2 * mass_g ** (2.0 / 3.0))


# ===========================================================================
# ORQUESTADOR
# ===========================================================================
def extract_all_features(impulse: np.ndarray, mass_g: float,
                         sr: int = config.SAMPLE_RATE) -> dict:
    """Extrae TODAS las features de un impulso ya recortado y normalizado.

    Pensado para recibir la salida de `preprocessing.preprocess`.
    Devuelve un dict listo para volcarse como fila de un CSV.

    Claves devueltas:
        f1, f2, damping_logdec, damping_bw, S, EI, zcr, ste
    Los valores no calculables salen como np.nan (nunca rompe el lote).
    """
    freqs, mag = compute_spectrum(impulse, sr)
    peak_freqs, _ = find_resonances(freqs, mag)

    f1 = peak_freqs[0] if len(peak_freqs) >= 1 else float("nan")
    f2 = peak_freqs[1] if len(peak_freqs) >= 2 else float("nan")

    env = hilbert_envelope(impulse)

    zeta_log = damping_log_decrement(impulse, f1, sr, env=env) \
        if np.isfinite(f1) else float("nan")
    zeta_bw = damping_bandwidth(freqs, mag, f1) \
        if np.isfinite(f1) else float("nan")

    S = stiffness_index(f1, mass_g) if np.isfinite(f1) else float("nan")
    EI = elasticity_index(f2, mass_g) if np.isfinite(f2) else float("nan")

    return {
        "f1": f1,
        "f2": f2,
        "damping_logdec": zeta_log,
        "damping_bw": zeta_bw,
        "S": S,
        "EI": EI,
        "zcr": zero_crossing_rate(impulse),
        "ste": short_time_energy_mean(impulse),
    }


if __name__ == "__main__":
    # Prueba de extremo a extremo sobre un sintético con ground truth.
    import audio_io
    import preprocessing

    for name in ("verde_01", "semi_01", "maduro_01"):
        p = config.SYNTHETIC_DIR / f"{name}.wav"
        sig, sr = audio_io.load_wav(p)
        impulse, _ = preprocessing.preprocess(sig, sr)
        # masa desde metadata no se lee aquí; se usa una masa fija de prueba.
        feats = extract_all_features(impulse, mass_g=300.0, sr=sr)
        print(f"\n{name}:")
        for k, v in feats.items():
            print(f"   {k:16} = {v:.4f}")
