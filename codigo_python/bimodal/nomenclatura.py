"""
nomenclatura.py
===============
Convención de nombres del dataset bimodal de papaya (imagen + audio).

Cada muestra se identifica por tres números:
    papaya{N}_serie{S}_rep{R}.{ext}     p. ej. papaya1_serie1_rep1.wav

    - N : id de la papaya (1..10)
    - S : serie = cara del fruto que se fotografió / golpeó (1..3)
    - R : repetición dentro de la serie (1..3)

Tanto `vision_color.py` como `acustica_fft.py` usan este módulo para leer el
nombre de archivo, y así el cruce imagen ↔ audio se hace por la misma llave
(id_papaya, serie, repeticion).
"""

from __future__ import annotations

import re
from pathlib import Path

# Acepta mayúsculas/minúsculas y separadores '_' o '-'.
_PATRON = re.compile(
    r"papaya[_-]?(\d+)[_-]serie[_-]?(\d+)[_-]rep[_-]?(\d+)", re.IGNORECASE
)

# Columnas llave compartidas por todos los CSV del dataset.
LLAVES = ["id_papaya", "serie", "repeticion"]


def nombre_estandar(id_papaya: int, serie: int, repeticion: int) -> str:
    """Devuelve el nombre base (sin extensión) de una muestra."""
    return f"papaya{id_papaya}_serie{serie}_rep{repeticion}"


def parsear_nombre(ruta: str | Path) -> dict | None:
    """Extrae id_papaya, serie y repeticion del nombre de un archivo.

    Devuelve None si el nombre no sigue la convención, para que el script
    que itera la carpeta pueda avisar y saltar ese archivo en vez de fallar.
    """
    m = _PATRON.search(Path(ruta).stem)
    if m is None:
        return None
    n, s, r = (int(g) for g in m.groups())
    return {"id_papaya": n, "serie": s, "repeticion": r}


def listar_muestras(carpeta: str | Path, extensiones: tuple[str, ...]) -> list[tuple[dict, Path]]:
    """Lista los archivos de `carpeta` con nombre estándar, ordenados por llave.

    Los archivos que no cumplen la convención se reportan por consola.
    """
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {carpeta}")

    muestras, ignorados = [], []
    for ruta in carpeta.iterdir():
        if ruta.suffix.lower() not in extensiones:
            continue
        llave = parsear_nombre(ruta)
        if llave is None:
            ignorados.append(ruta.name)
        else:
            muestras.append((llave, ruta))

    if ignorados:
        print(f"[aviso] {len(ignorados)} archivo(s) sin nombre estándar en "
              f"{carpeta.name}: {ignorados[:5]}{' ...' if len(ignorados) > 5 else ''}")

    muestras.sort(key=lambda t: (t[0]["id_papaya"], t[0]["serie"], t[0]["repeticion"]))
    return muestras
