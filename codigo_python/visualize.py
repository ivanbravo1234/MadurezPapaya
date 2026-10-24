"""
visualize.py
============
Inspección visual de un golpe a papaya: forma de onda + envolvente y
espectro FFT con las resonancias (f1, f2) marcadas.

Este es el entregable clave de la primera etapa: sirve para VER a ojo la
diferencia entre una papaya verde y una madura y así validar tus primeras
grabaciones ANTES de montar todo el clasificador.

Qué se grafica:
    Panel superior  -> señal en el tiempo + envolvente de Hilbert.
                       (una papaya firme "suena" más tiempo -> envolvente
                        decae lento; una papaya madura decae rápido).
    Panel inferior  -> espectro de magnitud con líneas verticales en f1/f2.
                       (verde -> picos a frecuencia más alta;
                        maduro -> picos a frecuencia más baja).

Uso:
    # Graficar un archivo concreto (lo muestra en pantalla):
    python visualize.py data/synthetic/verde_01.wav

    # Guardar la figura en outputs/figures en vez de mostrarla:
    python visualize.py data/synthetic/verde_01.wav --save

    # Comparar dos audios (verde vs maduro) en una sola figura:
    python visualize.py --compare data/synthetic/verde_01.wav data/synthetic/maduro_01.wav

    # Sin argumentos: elige automáticamente un verde y un maduro de los
    # datos sintéticos y los compara (demo rápida tras generar datos).
    python visualize.py

NOTA: la carga de audio aquí es una versión mínima. En la Tanda 2 se
reemplaza por el módulo `audio_io.py` (validación más estricta).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import config
import features


# ---------------------------------------------------------------------------
# Carga de audio (temporal — se moverá a audio_io.py en la Tanda 2)
# ---------------------------------------------------------------------------
def _load_audio(path: Path, sr: int = config.SAMPLE_RATE):
    """Carga un .wav como float mono normalizado a [-1, 1].

    Usa scipy para leer PCM y convierte a float. Si el audio es estéreo,
    promedia los canales (el análisis modal no necesita estéreo).
    """
    from scipy.io import wavfile

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de audio: {path}")

    file_sr, data = wavfile.read(str(path))

    # Convertir a float en [-1, 1] según el tipo entero original.
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float64) / max_val
    else:
        data = data.astype(np.float64)

    # Estéreo -> mono por promedio de canales.
    if data.ndim == 2 and config.FORCE_MONO:
        data = data.mean(axis=1)

    if file_sr != sr:
        # Aviso claro: en Tanda 1 no resampleamos, solo advertimos.
        print(f"[AVISO] {path.name} está a {file_sr} Hz, se esperaba {sr} Hz. "
              f"El eje de frecuencias usará {file_sr} Hz.")
        sr = file_sr

    if data.size == 0:
        raise ValueError(f"El archivo {path.name} no contiene muestras de audio.")

    return data, sr


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def plot_single(path: Path, save: bool = False) -> None:
    """Grafica onda+envolvente y espectro de un solo audio."""
    signal, sr = _load_audio(path)

    env = features.hilbert_envelope(signal)
    freqs, mag = features.compute_spectrum(signal, sr)
    peak_freqs, peak_mags = features.find_resonances(freqs, mag)

    t = np.arange(signal.size) / sr

    fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(11, 7))
    fig.suptitle(f"Resonancia acústica — {path.name}", fontsize=13)

    # --- Panel 1: forma de onda + envolvente ---
    ax_wave.plot(t, signal, color="#9ecae1", linewidth=0.6, label="Señal")
    ax_wave.plot(t, env, color="#e6550d", linewidth=1.4, label="Envolvente (Hilbert)")
    ax_wave.plot(t, -env, color="#e6550d", linewidth=1.4)
    ax_wave.set_xlabel("Tiempo (s)")
    ax_wave.set_ylabel("Amplitud")
    ax_wave.set_title("Forma de onda y envolvente de amplitud")
    ax_wave.legend(loc="upper right", fontsize=9)
    ax_wave.grid(True, alpha=0.3)

    # --- Panel 2: espectro con f1/f2 marcados ---
    band = (freqs >= config.FREQ_MIN) & (freqs <= config.FREQ_MAX)
    ax_spec.plot(freqs[band], mag[band], color="#3182bd", linewidth=1.0)
    _annotate_peaks(ax_spec, peak_freqs, peak_mags)
    ax_spec.set_xlabel("Frecuencia (Hz)")
    ax_spec.set_ylabel("|X(f)|")
    ax_spec.set_title("Espectro FFT y resonancias")
    ax_spec.set_xlim(config.FREQ_MIN, config.FREQ_MAX)
    ax_spec.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _finish(fig, save, path.stem)


def plot_compare(path_a: Path, path_b: Path, save: bool = False) -> None:
    """Compara dos audios (p. ej. verde vs maduro) en una figura.

    Superpone las envolventes normalizadas y los espectros, que es la
    forma más directa de "ver" la diferencia de madurez.
    """
    sig_a, sr_a = _load_audio(path_a)
    sig_b, sr_b = _load_audio(path_b)

    env_a = features.hilbert_envelope(sig_a)
    env_b = features.hilbert_envelope(sig_b)
    fa, ma = features.compute_spectrum(sig_a, sr_a)
    fb, mb = features.compute_spectrum(sig_b, sr_b)

    ta = np.arange(sig_a.size) / sr_a
    tb = np.arange(sig_b.size) / sr_b

    fig, (ax_env, ax_spec) = plt.subplots(2, 1, figsize=(11, 7))
    fig.suptitle(f"Comparación — {path_a.stem}  vs  {path_b.stem}", fontsize=13)

    # --- Panel 1: envolventes normalizadas ---
    ax_env.plot(ta, env_a / np.max(env_a), color="#31a354",
                linewidth=1.4, label=path_a.stem)
    ax_env.plot(tb, env_b / np.max(env_b), color="#e6550d",
                linewidth=1.4, label=path_b.stem)
    ax_env.set_xlabel("Tiempo (s)")
    ax_env.set_ylabel("Envolvente (norm.)")
    ax_env.set_title("Envolventes: decaimiento (proxy visual del damping)")
    ax_env.legend(loc="upper right", fontsize=9)
    ax_env.grid(True, alpha=0.3)

    # --- Panel 2: espectros normalizados ---
    band_a = (fa >= config.FREQ_MIN) & (fa <= config.FREQ_MAX)
    band_b = (fb >= config.FREQ_MIN) & (fb <= config.FREQ_MAX)
    ax_spec.plot(fa[band_a], ma[band_a] / np.max(ma[band_a]),
                 color="#31a354", linewidth=1.1, label=path_a.stem)
    ax_spec.plot(fb[band_b], mb[band_b] / np.max(mb[band_b]),
                 color="#e6550d", linewidth=1.1, label=path_b.stem)
    ax_spec.set_xlabel("Frecuencia (Hz)")
    ax_spec.set_ylabel("|X(f)| (norm.)")
    ax_spec.set_title("Espectros: desplazamiento de las resonancias")
    ax_spec.set_xlim(config.FREQ_MIN, config.FREQ_MAX)
    ax_spec.legend(loc="upper right", fontsize=9)
    ax_spec.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _finish(fig, save, f"compare_{path_a.stem}_{path_b.stem}")


def _annotate_peaks(ax, peak_freqs, peak_mags) -> None:
    """Dibuja líneas verticales y etiquetas f1, f2 en el espectro."""
    for i, (f, m) in enumerate(zip(peak_freqs, peak_mags), start=1):
        ax.axvline(f, color="#e6550d", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.annotate(f"f{i} = {f:.0f} Hz",
                    xy=(f, m), xytext=(f, m * 1.05),
                    fontsize=9, color="#a63603",
                    ha="left", va="bottom")


def _finish(fig, save: bool, stem: str) -> None:
    """Muestra o guarda la figura según la bandera `save`."""
    if save:
        config.ensure_dirs()
        out = config.FIGURES_DIR / f"{stem}.png"
        fig.savefig(out, dpi=130)
        print(f"Figura guardada en: {out}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _auto_demo(save: bool) -> None:
    """Sin argumentos: compara un verde y un maduro de los sintéticos."""
    verde = config.SYNTHETIC_DIR / "verde_01.wav"
    maduro = config.SYNTHETIC_DIR / "maduro_01.wav"
    if not (verde.exists() and maduro.exists()):
        print("No hay audios sintéticos. Genera primero con:\n"
              "    python generate_synthetic.py")
        sys.exit(1)
    print("Demo: comparando verde_01 vs maduro_01 ...")
    plot_compare(verde, maduro, save=save)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualiza onda y espectro de un golpe a papaya.")
    parser.add_argument("audio", nargs="?", help="Ruta a un archivo .wav")
    parser.add_argument("--compare", nargs=2, metavar=("WAV_A", "WAV_B"),
                        help="Compara dos audios en una sola figura")
    parser.add_argument("--save", action="store_true",
                        help="Guarda la figura en outputs/figures en vez de mostrarla")
    args = parser.parse_args()

    if args.compare:
        plot_compare(Path(args.compare[0]), Path(args.compare[1]), save=args.save)
    elif args.audio:
        plot_single(Path(args.audio), save=args.save)
    else:
        _auto_demo(save=args.save)


if __name__ == "__main__":
    main()
