"""
vision_color.py
===============
Módulo de visión computacional: color de la cáscara de papaya en CIELAB.

Para cada imagen `papaya{N}_serie{S}_rep{R}.jpg`:
    1. Carga y reduce la imagen (4096×3072 → ~1024 px de ancho; el promedio
       de color no cambia y el proceso es ~16 veces más rápido).
    2. Segmenta el fruto y aísla el fondo neutro (papel blanco + sombras).
    3. Convierte de RGB a CIELAB (iluminante D65) según el espacio de color
       de la foto: los celulares actuales guardan en Display P3 (más amplio
       que sRGB). Se lee el perfil ICC del JPEG y se usa la matriz RGB→XYZ
       que corresponde; tratar P3 como sRGB desatura el color (|a*|, |b*|
       salen más bajos de lo real).
    4. Promedia L*, a*, b* solo sobre los píxeles de la cáscara.

¿Por qué segmentar por SATURACIÓN (canal S de HSV)?
    El fondo es blanco/gris: su color es "neutro" y tiene saturación ≈ 0,
    tanto en las zonas iluminadas como en las sombras. La cáscara (verde,
    amarilla o naranja) es cromática y tiene saturación alta. El umbral se
    deriva de Otsu en cada foto, así que no hay que fijar un número a mano
    aunque la iluminación cambie de foto a foto.

Interpretación de CIELAB para la madurez:
    L* : luminosidad (0 negro – 100 blanco)
    a* : eje verde (−) ↔ rojo (+)  → sube al madurar (verde → naranja)
    b* : eje azul (−) ↔ amarillo (+) → sube al aparecer el amarillo
    Índices derivados útiles: tono h° = atan2(b*, a*) y croma C* = √(a*² + b*²).

Uso independiente:
    python bimodal/vision_color.py  [carpeta_imagenes]  [--depurar]
"""

from __future__ import annotations

import sys
from pathlib import Path

import io

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageCms

from nomenclatura import listar_muestras

# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
CARPETA_IMAGENES = RAIZ / "data" / "papayas" / "imagenes"
CARPETA_DEPURACION = RAIZ / "outputs" / "segmentacion"   # máscaras de control

EXTENSIONES = (".jpg", ".jpeg", ".png")
ANCHO_TRABAJO = 1024        # px; se reduce la imagen a este ancho

# Umbral de histéresis sobre la saturación: Otsu da el umbral "fuerte"; los
# bordes del fruto (luz rasante, verde oscuro) quedan por debajo, así que se
# acepta todo píxel con S > FACTOR_UMBRAL · Otsu conectado al fruto.
FACTOR_UMBRAL = 0.5

# Tamaños de los elementos morfológicos, en proporción (no en píxeles) para
# no depender de la resolución.
FRAC_RUIDO = 0.01           # × lado menor de la imagen: limpia motas
FRAC_TALLO = 0.15           # × diámetro equivalente del fruto: corta el pedúnculo
FRAC_EROSION = 0.01         # × lado menor: aleja el borde (sombra/mezcla)

# Brillos especulares (reflejos de la luz sobre la cáscara): son casi blancos
# y sesgarían L* hacia arriba y a*, b* hacia 0. Se excluyen del promedio.
BRILLO_V_MIN = 235          # V (0–255) por encima de esto...
BRILLO_S_MAX = 60           # ...y S por debajo de esto = reflejo


# Matrices RGB lineal → XYZ (iluminante D65). Ambos espacios usan la misma
# curva de transferencia (gamma sRGB) y el mismo blanco D65.
MATRIZ_XYZ = {
    "srgb": np.array([[0.412453, 0.357580, 0.180423],
                      [0.212671, 0.715160, 0.072169],
                      [0.019334, 0.119193, 0.950227]]),
    "display-p3": np.array([[0.4865709, 0.2656677, 0.1982173],
                            [0.2289746, 0.6917385, 0.0792869],
                            [0.0000000, 0.0451134, 1.0439444]]),
}
BLANCO_D65 = np.array([0.950456, 1.0, 1.088754])


# ---------------------------------------------------------------------------
# 1. Carga
# ---------------------------------------------------------------------------
def cargar_imagen(ruta: Path, ancho: int = ANCHO_TRABAJO) -> np.ndarray:
    """Lee una imagen BGR reducida a `ancho` píxeles de ancho.

    Se usa np.fromfile + cv2.imdecode en lugar de cv2.imread porque
    cv2.imread falla en Windows con rutas que contienen tildes o 'ñ'.
    """
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    img = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta}")
    escala = ancho / img.shape[1]
    if escala < 1:
        img = cv2.resize(img, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    return img


def espacio_color(ruta: Path) -> str:
    """'display-p3' si el perfil ICC embebido es P3; si no, 'srgb'
    (un JPEG sin perfil se interpreta como sRGB por convención)."""
    try:
        icc = Image.open(ruta).info.get("icc_profile")
        if icc:
            desc = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc)))
            if "P3" in desc:
                return "display-p3"
    except (OSError, ImageCms.PyCMSError):
        pass
    return "srgb"


# ---------------------------------------------------------------------------
# 2. Segmentación (aislar el fondo neutro)
# ---------------------------------------------------------------------------
def _kernel(img: np.ndarray, fraccion: float) -> np.ndarray:
    """Elemento estructurante elíptico proporcional al tamaño de la imagen."""
    k = max(3, int(round(min(img.shape[:2]) * fraccion)) | 1)  # impar
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _mayor_componente(mascara: np.ndarray) -> np.ndarray:
    """Conserva solo la componente conexa de mayor área (la papaya)."""
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    if n <= 1:
        raise ValueError("No se encontró ningún objeto saturado (¿no hay papaya en la foto?)")
    mayor = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(etiquetas == mayor, 255, 0).astype(np.uint8)


def segmentar_papaya(bgr: np.ndarray) -> np.ndarray:
    """Devuelve una máscara binaria (uint8, 0/255) con los píxeles de cáscara.

    Pasos:
        a) Saturación HSV suavizada. Umbral = FACTOR_UMBRAL × Otsu: Otsu
           separa fondo neutro y fruto, pero corta los bordes oscuros del
           fruto; la mitad de ese umbral sigue muy por encima del papel
           (S ≈ 0–25) y recupera el contorno completo.
        b) Apertura pequeña + componente conexa mayor: quita motas y objetos
           sueltos (manchas del papel, borde de la mesa).
        c) Relleno de huecos: manchas claras/oscuras y reflejos dentro del
           fruto quedan dentro del contorno.
        d) Apertura grande (≈15 % del diámetro del fruto): el pedúnculo es
           delgado y desaparece; el cuerpo del fruto, redondeado, se conserva.
        e) Erosión: se descarta un anillo en el borde, donde el píxel mezcla
           fruto con sombra y fondo.
        f) Se quitan los reflejos especulares (V muy alto y S bajo), que
           sesgarían L* hacia arriba y a*, b* hacia 0.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = cv2.GaussianBlur(hsv[:, :, 1], (7, 7), 0)

    # a) Umbral relativo al de Otsu.
    otsu, _ = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mascara = np.where(s > FACTOR_UMBRAL * otsu, 255, 0).astype(np.uint8)

    # b) Limpieza y objeto principal.
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, _kernel(bgr, FRAC_RUIDO))
    mascara = _mayor_componente(mascara)

    # c) Relleno de huecos dibujando el contorno externo lleno.
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascara = np.zeros_like(mascara)
    cv2.drawContours(mascara, contornos, -1, 255, thickness=cv2.FILLED)

    # d) Eliminación del pedúnculo. Diámetro equivalente: d = √(4·área/π).
    diametro = np.sqrt(4.0 * np.count_nonzero(mascara) / np.pi)
    k = max(3, int(diametro * FRAC_TALLO) | 1)
    mascara = cv2.morphologyEx(
        mascara, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    mascara = _mayor_componente(mascara)

    # e) Erosión del borde.
    mascara = cv2.erode(mascara, _kernel(bgr, FRAC_EROSION))

    # f) Exclusión de reflejos especulares.
    reflejo = (hsv[:, :, 2] >= BRILLO_V_MIN) & (hsv[:, :, 1] <= BRILLO_S_MAX)
    mascara[reflejo] = 0
    return mascara


# ---------------------------------------------------------------------------
# 3-4. Conversión a CIELAB y promedio
# ---------------------------------------------------------------------------
def bgr_a_lab(bgr: np.ndarray, espacio: str = "srgb") -> np.ndarray:
    """Convierte BGR (uint8) a CIELAB (L* ∈ [0, 100], a*, b* ≈ [−128, 127]).

        1. Linealiza la curva sRGB:  c ≤ 0.04045 → c/12.92,
                                     si no → ((c + 0.055)/1.055)^2.4
        2. XYZ = M · RGB_lineal  (M según el espacio: sRGB o Display P3)
        3. f(t) = t^(1/3) si t > 0.008856, si no 7.787·t + 16/116
           L* = 116·f(Y/Yn) − 16,  a* = 500·(f(X/Xn) − f(Y/Yn)),
           b* = 200·(f(Y/Yn) − f(Z/Zn))
    Para 'srgb' coincide con cv2.cvtColor(float, COLOR_BGR2Lab).
    """
    c = bgr[..., ::-1].astype(np.float64) / 255.0           # BGR → RGB
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ MATRIZ_XYZ[espacio].T / BLANCO_D65
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    Y = xyz[..., 1]
    L = np.where(Y > 0.008856, 116 * f[..., 1] - 16, 903.3 * Y)
    return np.stack([L, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def extraer_lab(bgr: np.ndarray, mascara: np.ndarray, espacio: str = "srgb") -> dict:
    """Promedio de L*, a*, b* sobre los píxeles de la máscara."""
    lab = bgr_a_lab(bgr, espacio)
    pix = lab[mascara > 0]
    if len(pix) == 0:
        raise ValueError("La máscara quedó vacía")
    L, a, b = pix.mean(axis=0)
    return {"L": float(L), "a": float(a), "b": float(b),
            "fraccion_fruto": len(pix) / mascara.size}   # control de calidad


def guardar_depuracion(bgr: np.ndarray, mascara: np.ndarray, destino: Path) -> None:
    """Guarda la foto con el contorno de la máscara en verde, para verificar
    visualmente que la segmentación tomó la cáscara y no el fondo."""
    vista = bgr.copy()
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vista, contornos, -1, (0, 255, 0), 3)
    destino.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", vista)
    if ok:
        buf.tofile(str(destino))


# ---------------------------------------------------------------------------
# Procesamiento de la carpeta
# ---------------------------------------------------------------------------
def procesar_imagen(ruta: Path, depurar: bool = False) -> dict:
    """Pipeline completo para una imagen."""
    bgr = cargar_imagen(ruta)
    mascara = segmentar_papaya(bgr)
    if depurar:
        guardar_depuracion(bgr, mascara, CARPETA_DEPURACION / f"{ruta.stem}.jpg")
    return extraer_lab(bgr, mascara, espacio_color(ruta))


def procesar_carpeta(carpeta: Path = CARPETA_IMAGENES, depurar: bool = False) -> pd.DataFrame:
    """Itera todas las imágenes con nombre estándar y devuelve un DataFrame
    con columnas id_papaya, serie, repeticion, L, a, b, fraccion_fruto."""
    filas = []
    muestras = listar_muestras(carpeta, EXTENSIONES)
    for i, (llave, ruta) in enumerate(muestras, 1):
        try:
            filas.append({**llave, **procesar_imagen(ruta, depurar)})
        except ValueError as e:
            print(f"[error] {ruta.name}: {e}")
            filas.append({**llave, "L": np.nan, "a": np.nan, "b": np.nan})
        print(f"\r[visión] {i}/{len(muestras)} {ruta.name}      ", end="")
    print()
    return pd.DataFrame(filas)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    carpeta = Path(args[0]) if args else CARPETA_IMAGENES
    df = procesar_carpeta(carpeta, depurar="--depurar" in sys.argv)
    print(df.round(2).to_string(index=False))
