"""
organizar_archivos.py
=====================
Paso 0: copia los archivos de campo a la nomenclatura estándar
`papaya{N}_serie{S}_rep{R}` SIN modificar ni borrar los originales.

Los archivos originales salieron del celular con nombres automáticos:
    - Imágenes: IMG_20261018_181625119.jpg  (fecha_hora en milisegundos)
    - Audios:   Mi grabación 1.wav ... Mi grabación 90.wav

Supuesto de muestreo (orden de captura en campo):
    papaya 1 → serie 1 → rep 1, 2, 3 → serie 2 → rep 1, 2, 3 → serie 3 ...
    papaya 2 → ...
Es decir, el k-ésimo archivo en orden cronológico (imágenes) o numérico
(audios) corresponde a:
    papaya = k // 9 + 1,  serie = (k % 9) // 3 + 1,  rep = k % 3 + 1

Se generan:
    data/papayas/imagenes/papaya1_serie1_rep1.jpg ...
    data/papayas/audios/papaya1_serie1_rep1.wav   ...
    data/papayas/mapeo_archivos.csv   (trazabilidad: nombre original → estándar)

Uso:
    python bimodal/organizar_archivos.py
"""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

from nomenclatura import nombre_estandar

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
ORIGEN_IMAGENES = RAIZ / "imagenes papaya"
ORIGEN_AUDIOS = RAIZ / "AUDIOS DE PAPAYA"

DESTINO = RAIZ / "data" / "papayas"
DESTINO_IMAGENES = DESTINO / "imagenes"
DESTINO_AUDIOS = DESTINO / "audios"
MAPEO_CSV = DESTINO / "mapeo_archivos.csv"

N_PAPAYAS, N_SERIES, N_REPS = 10, 3, 3
N_ESPERADO = N_PAPAYAS * N_SERIES * N_REPS  # 90

# Fotos sobrantes que NO forman parte del diseño 10×3×3.
# En la carpeta hay 91 fotos: la serie 2 de la papaya 8 tiene 4 tomas casi
# idénticas (18:29:11 a 18:29:19). Se descarta la primera, que es la menos
# nítida (varianza del Laplaciano 255 frente a ~310 de las otras tres).
IMAGENES_DESCARTADAS = {"IMG_20261018_182911859.jpg"}


def _ordenar_imagenes(carpeta: Path) -> list[Path]:
    """Imágenes en orden cronológico (el nombre IMG_fecha_hora ya lo es).

    Se ignoran los archivos '.trashed-*' que deja la papelera de Android.
    """
    fotos = [p for p in carpeta.glob("*.jpg")
             if not p.name.startswith(".") and p.name not in IMAGENES_DESCARTADAS]
    return sorted(fotos, key=lambda p: p.name)


def _ordenar_audios(carpeta: Path) -> list[Path]:
    """Audios por el número de 'Mi grabación N' (orden numérico, no alfabético:
    alfabéticamente 'Mi grabación 10' quedaría antes que 'Mi grabación 2')."""
    def numero(p: Path) -> int:
        m = re.search(r"(\d+)\s*$", p.stem)
        return int(m.group(1)) if m else 10**9
    return sorted(carpeta.glob("*.wav"), key=numero)


def _llave_por_indice(k: int) -> tuple[int, int, int]:
    """Índice 0..89 → (id_papaya, serie, repeticion) según el orden de captura."""
    return (k // (N_SERIES * N_REPS) + 1,
            (k % (N_SERIES * N_REPS)) // N_REPS + 1,
            k % N_REPS + 1)


def organizar() -> None:
    imagenes = _ordenar_imagenes(ORIGEN_IMAGENES)
    audios = _ordenar_audios(ORIGEN_AUDIOS)

    # Validación estricta: si no hay exactamente 90 de cada uno, el mapeo por
    # orden quedaría corrido y mezclaría papayas. Mejor detenerse.
    for nombre, lista in (("imágenes", imagenes), ("audios", audios)):
        if len(lista) != N_ESPERADO:
            raise SystemExit(f"Se esperaban {N_ESPERADO} {nombre} y hay {len(lista)}. "
                             "Revisa la carpeta o IMAGENES_DESCARTADAS.")

    DESTINO_IMAGENES.mkdir(parents=True, exist_ok=True)
    DESTINO_AUDIOS.mkdir(parents=True, exist_ok=True)

    filas = []
    for k, (img, aud) in enumerate(zip(imagenes, audios)):
        n, s, r = _llave_por_indice(k)
        base = nombre_estandar(n, s, r)
        shutil.copy2(img, DESTINO_IMAGENES / f"{base}{img.suffix.lower()}")
        shutil.copy2(aud, DESTINO_AUDIOS / f"{base}.wav")
        filas.append({"id_papaya": n, "serie": s, "repeticion": r,
                      "nombre_estandar": base,
                      "imagen_original": img.name, "audio_original": aud.name})

    with open(MAPEO_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    print(f"Copiadas {len(filas)} imágenes → {DESTINO_IMAGENES}")
    print(f"Copiados {len(filas)} audios   → {DESTINO_AUDIOS}")
    print(f"Mapeo de trazabilidad        → {MAPEO_CSV}")
    if IMAGENES_DESCARTADAS:
        print(f"Imágenes descartadas: {sorted(IMAGENES_DESCARTADAS)}")


if __name__ == "__main__":
    organizar()
