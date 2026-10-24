"""
config.py
=========
Parámetros centrales del pipeline de resonancia acústica para papaya.

Toda constante ajustable del proyecto vive aquí para que los módulos
(`audio_io`, `preprocessing`, `features`, `visualize`, ...) no tengan
"números mágicos" dispersos. Cambiar un umbral o el sample rate se hace
en un solo sitio.

Convención de unidades:
    - Frecuencias en Hz.
    - Tiempos en segundos.
    - Masa del fruto en gramos (g).
    - Niveles de audio normalizados en el rango [-1, 1].
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
# BASE_DIR = carpeta donde vive este archivo (raíz del proyecto).
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"                # grabaciones reales de papaya (.wav)
SYNTHETIC_DIR = DATA_DIR / "synthetic"    # audios sintéticos de prueba (.wav)
METADATA_CSV = DATA_DIR / "metadata.csv"  # id, masa_g, etiqueta

OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"     # PNG de onda/espectro
FEATURES_CSV = OUTPUTS_DIR / "features.csv"
MODELS_DIR = OUTPUTS_DIR / "models"


def ensure_dirs() -> None:
    """Crea todas las carpetas de trabajo si no existen.

    Se llama al inicio de los scripts para que escribir en `data/` u
    `outputs/` no falle en una instalación recién clonada.
    """
    for d in (RAW_DIR, SYNTHETIC_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Parámetros de audio
# ---------------------------------------------------------------------------
# Sample rate objetivo. Las grabaciones de celular suelen ser 44100 Hz.
# Todo el pipeline resamplea a este valor para ser consistente.
SAMPLE_RATE = 44100

# Fuerza la conversión a mono (el análisis modal no necesita estéreo).
FORCE_MONO = True


# ---------------------------------------------------------------------------
# Detección de onset (usado en preprocessing.py, Tanda 2)
# ---------------------------------------------------------------------------
# Tamaño de trama para energía de corto plazo (STE), en muestras.
FRAME_LENGTH = 1024
HOP_LENGTH = 256

# Umbral adaptativo: onset donde la energía supera
#   mediana + ONSET_K * MAD   (MAD = desviación absoluta mediana).
# Usar mediana + MAD en vez de un umbral fijo tolera grabaciones con
# niveles muy distintos entre celulares.
ONSET_K = 6.0

# Segundos de "guarda" que se conservan ANTES del onset detectado,
# para no cortar el ataque del impulso.
PRE_ONSET_PAD = 0.005  # 5 ms

# Duración máxima de la ventana de análisis tras el onset (segundos).
# El decaimiento útil de un golpe a papaya rara vez supera esto.
ANALYSIS_WINDOW = 0.35


# ---------------------------------------------------------------------------
# Parámetros de FFT / espectro
# ---------------------------------------------------------------------------
# Tamaño de la FFT. Potencia de 2 para eficiencia; con zero-padding
# mejora la resolución visual del espectro.
FFT_SIZE = 8192

# Rango de frecuencias de interés para picos de resonancia (Hz).
# Fuera de esto casi todo es ruido o componentes no estructurales.
FREQ_MIN = 80.0
FREQ_MAX = 3000.0

# Separación mínima entre picos detectados (Hz), para no contar como
# "dos resonancias" a dos bins vecinos del mismo lóbulo.
MIN_PEAK_DISTANCE_HZ = 60.0

# Cuántas resonancias extraer (decisión de diseño: f1 y f2 bastan para
# audio de celular en pulpa blanda; f3+ suele ser ruido).
N_RESONANCES = 2


# ---------------------------------------------------------------------------
# Generación de audio sintético (generate_synthetic.py)
# ---------------------------------------------------------------------------
# Modelo de cada golpe: suma de sinusoides amortiguadas
#     x(t) = Σ A_i · exp(-ζ_i · ω_i · t) · sin(ω_i · t) + ruido
# donde ω_i = 2π·f_i.
#
# Física esperada (sirve de "ground truth" para validar el pipeline):
#   - Papaya VERDE  -> pulpa firme  -> mayor rigidez -> f1 ALTA, ζ BAJO.
#   - Papaya MADURA -> pulpa blanda -> menor rigidez -> f1 BAJA, ζ ALTO.
#   - SEMI-maduro  -> valores intermedios.
#
# Cada clase define medias; el generador añade "jitter" aleatorio por
# muestra para que las clases se solapen un poco (realista) y el
# clasificador de la Tanda 3 no tenga una tarea trivial.
#
# Escala papaya: el fruto pesa ~1-1.5 kg (vs ~0.3 kg del mango original).
# Para un mismo material, la frecuencia de resonancia escala con el tamaño
# como f ∝ m^(-1/3), así que se conservaron los índices de firmeza
# S = f1²·m^(2/3) de cada clase y se bajaron f1 y f2 en el factor
# (1250/320)^(-1/3) ≈ 0.635. ζ no depende del tamaño y se mantiene.
# Son valores PROVISIONALES: se reemplazan al grabar papayas reales.
SYNTHETIC_CLASSES = {
    "verde": {
        "f1": 520.0,   # Hz, frecuencia dominante (modo 1)
        "f2": 985.0,   # Hz, segunda resonancia (modo 2)
        "zeta": 0.020,  # razón de amortiguación (adimensional)
        "masa_media": 1200.0,  # g
    },
    "semi": {
        "f1": 380.0,
        "f2": 730.0,
        "zeta": 0.045,
        "masa_media": 1250.0,
    },
    "maduro": {
        "f1": 275.0,
        "f2": 520.0,
        "zeta": 0.080,
        "masa_media": 1300.0,
    },
}

# Cuántas muestras generar por clase.
SYNTHETIC_N_PER_CLASS = 12

# Jitter relativo (±) aplicado a f1, f2 y zeta por muestra.
SYNTHETIC_FREQ_JITTER = 0.06   # ±6 %
SYNTHETIC_ZETA_JITTER = 0.15   # ±15 %
SYNTHETIC_MASS_JITTER = 100.0  # ±100 g (absoluto)

# Estructura temporal de cada .wav sintético.
SYNTHETIC_PRE_SILENCE = 0.10   # s de silencio+ruido antes del impulso
SYNTHETIC_DURATION = 0.60      # s de duración del impulso amortiguado
SYNTHETIC_NOISE_LEVEL = 0.010  # amplitud RMS aprox. del ruido blanco

# Semilla para reproducibilidad de los datos sintéticos.
RANDOM_SEED = 42
