# Módulo de resonancia acústica — Estimación de madurez en papaya

Pipeline en Python para estimar el grado de madurez fisiológica de papayas
mediante **análisis acústico de impulso-respuesta**: se golpea suavemente el
fruto, se graba el sonido, y del audio se extraen parámetros (sobre todo la
**razón de amortiguación / damping**) que correlacionan con la firmeza de la
pulpa.

Esta es la etapa de **prototipado y validación en laptop** (previa a la app
Android). El objetivo es confirmar que el fenómeno es medible antes de
invertir en el desarrollo móvil.

---

## 1. Fundamento físico

Un golpe excita modos de vibración del fruto. La respuesta es una suma de
sinusoides amortiguadas:

```
x(t) = Σ_i  A_i · exp(-ζ_i · ω_i · t) · sin(ω_i · t) + ruido
```

- **ζ (damping ratio)**: cuán rápido decae la vibración. En papaya (pulpa
  blanda) es el parámetro más informativo. Maduro → ζ **alto** (decae rápido).
- **f (frecuencia dominante)**: sube con la rigidez. Verde → f **alta**.
- **Índices de firmeza** (normalizan por el tamaño del fruto):
  - Rigidez:    `S  = f1² · m^(2/3)`
  - Elasticidad: `EI = f2² · m^(2/3)`  (m = masa en gramos)

El **damping se calcula por dos métodos** que se contrastan entre sí:
1. **Log-decrement** sobre la envolvente temporal (valor físico correcto).
2. **Ancho de banda a −3 dB** del pico espectral (subestima en valor
   absoluto, pero confirma la tendencia).

---

## 2. Instalación

Requiere **Python 3.10+**.

```bash
pip install -r requirements.txt
```

Dependencias: `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `pandas`.
(La carga y el resampling de audio usan `scipy`, sin necesidad de `librosa`.)

---

## 3. Estructura del proyecto

```
Proyecto_mango/
├── config.py             # Parámetros: sample rate, ventanas, umbrales, rutas
├── audio_io.py           # Carga y validación de .wav (mono, resample, clipping)
├── preprocessing.py      # Onset (umbral adaptativo), recorte del impulso, normalización
├── features.py           # FFT, f1/f2, damping (×2), S, EI, ZCR, STE
├── visualize.py          # Gráficos de onda + espectro (inspección visual)
├── generate_synthetic.py # Genera audios de prueba (verde/semi/maduro)
├── build_dataset.py      # Carpeta de audios + metadata.csv -> features.csv
├── train.py              # Entrena y compara SVM vs Random Forest
├── app.py                # Interfaz gráfica (Streamlit) de todo el pipeline
├── export_model.py       # Exporta el modelo a JSON para la app móvil
├── bimodal/              # Datos reales: color CIELAB (foto) + FFT (audio) → regla de 5 estadios
├── mobile/               # App móvil Android (Capacitor) + pruebas JS vs Python
├── MadurezPapaya.apk      # APK listo para instalar
├── requirements.txt
├── data/
│   ├── synthetic/        # .wav sintéticos (generados)
│   ├── raw/              # tus grabaciones reales de papaya
│   ├── papayas/          # dataset bimodal (imagenes/, audios/, CSV y umbrales_app.json)
│   └── metadata.csv      # id, masa_g, etiqueta
└── outputs/
    ├── figures/          # PNG de espectro/onda/matrices de confusión
    ├── features.csv      # dataset consolidado
    └── models/           # modelos entrenados (.joblib)
```

---

## 4. Uso paso a paso

### Opción rápida — Interfaz gráfica

```powershell
streamlit run app.py
```

Se abre en el navegador (http://localhost:8501), o con doble clic en
`abrir_app.bat`. Tiene tres pestañas:

| Pestaña | Qué hace |
|---|---|
| **Analizar golpe** | Elige o sube un `.wav` (y guárdalo en `data/raw`): espectro con f1/f2 y banda −3 dB, impulso con envolvente y ajuste e^(−ζωt), onset, todas las features y la **predicción de madurez**. |
| **Comparar** | Dos golpes superpuestos (espectros y envolventes) con tabla de features lado a lado. |
| **Dataset** | Sube grabaciones, registra masa y etiqueta en `metadata.csv` y pulsa «Actualizar dataset y modelo»: reconstruye `features.csv` con las **grabaciones reales** y reentrena el modelo, reemplazando a los audios sintéticos (se necesitan al menos 5 grabaciones por clase; mientras tanto el modelo actual se mantiene). |

La interfaz llama a los mismos módulos que los scripts, así que da exactamente
los mismos números que la consola. Los pasos por consola siguen disponibles:


### App móvil (Android)

El APK listo para instalar está en **`MadurezPapaya.apk`** (raíz del proyecto).
Funciona sin internet y sin la laptop: graba el golpe con el micrófono del
celular (o analiza un `.wav`), calcula las mismas features que el pipeline
Python y predice la madurez con el Random Forest exportado.

- `mobile/www/` — la app (HTML/CSS/JS). `js/pipeline.js` es el port del
  pipeline Python; `model.json` es el modelo exportado.
- `export_model.py` — exporta el Random Forest + parámetros de `config.py` a JSON.
- `mobile/tests/` — verificación: `python mobile/tests/make_reference.py` y
  `node mobile/tests/compare.mjs` comparan el JavaScript contra Python
  (resultado actual: 39/39 audios idénticos, error relativo < 1e-14).
- `mobile/build_apk.ps1` — recompila el APK con el modelo más reciente.

**Actualizar el modelo del celular sin recompilar:** al pulsar «Actualizar
dataset y modelo» en la interfaz de la laptop se genera
`outputs/model_movil.json` (también se descarga desde la pestaña Dataset).
Cópialo al celular y cárgalo en la app con el botón ⓘ → *Importar modelo*.

### Paso 0 — Generar datos sintéticos de prueba (sin papayas reales)

Permite validar TODO el pipeline antes de grabar. Crea 36 audios (12 por
clase) con ζ y frecuencias conocidas, más `data/metadata.csv`.

```bash
python generate_synthetic.py
```

### Paso 1 — Inspección visual (el uso más importante al inicio)

Ver a ojo la diferencia entre verde y maduro:

```bash
# Demo automática: compara un verde y un maduro
python visualize.py

# Un archivo concreto (muestra en pantalla)
python visualize.py data/synthetic/verde_01.wav

# Guardar la figura en outputs/figures en vez de mostrarla
python visualize.py data/synthetic/verde_01.wav --save

# Comparar dos audios cualesquiera
python visualize.py --compare data/synthetic/verde_01.wav data/synthetic/maduro_01.wav
```

### Paso 2 — Consolidar el lote en un CSV de features

```bash
# Lote sintético (por defecto)
python build_dataset.py

# Lote de grabaciones reales
python build_dataset.py --audio-dir data/raw --metadata data/metadata.csv
```

Genera `outputs/features.csv` (una fila por muestra).

### Paso 3 — Entrenar y evaluar el clasificador

```bash
python train.py
```

Reporta, para SVM y Random Forest:
- Accuracy en hold-out + matriz de confusión (texto y PNG).
- Reporte por clase (precision / recall / f1).
- Validación cruzada estratificada (5-fold).
- Importancia de features (Random Forest).

Los modelos se guardan en `outputs/models/*.joblib`.

---

## 5. Usar tus propias grabaciones

1. Graba a **44100 Hz** (configurable en `config.py`). Un golpe por archivo,
   con un poco de silencio antes del impacto.
2. Guarda los `.wav` en `data/raw/`.
3. Crea/edita `data/metadata.csv`. El **`id` debe coincidir con el nombre del
   archivo** (sin `.wav`):

   ```csv
   id,masa_g,etiqueta
   papaya_01,1180,verde
   papaya_02,1260,semi
   papaya_03,1310,maduro
   ```

4. Ejecuta:
   ```bash
   python build_dataset.py --audio-dir data/raw
   python train.py
   ```

**Consejo:** antes de entrenar, corre `visualize.py` sobre tus grabaciones.
Si no ves un pico de resonancia claro ni un decaimiento limpio en la
envolvente, el problema es de captura (micrófono, ruido, golpe) y ningún
clasificador lo arreglará.

---

## 6. Parámetros ajustables (`config.py`)

| Parámetro | Qué controla |
|---|---|
| `SAMPLE_RATE` | Frecuencia de muestreo objetivo (Hz) |
| `ONSET_K` | Sensibilidad de la detección de onset (mediana + k·MAD) |
| `ANALYSIS_WINDOW` | Duración de la ventana de análisis tras el onset (s) |
| `FREQ_MIN`, `FREQ_MAX` | Banda de búsqueda de resonancias (Hz) |
| `N_RESONANCES` | Cuántas resonancias extraer (f1, f2) |
| `FFT_SIZE` | Resolución del espectro |

---

## 7. Notas de validación

Sobre los datos sintéticos (ζ conocido inyectado), el pipeline recupera:

| Clase | ζ inyectado | ζ log-decrement | f1 real | f1 medido |
|---|---|---|---|---|
| verde | 0.020 | ~0.020 | 847 Hz | 845 Hz |
| semi | 0.045 | ~0.049 | 599 Hz | 597 Hz |
| maduro | 0.080 | ~0.080 | 423 Hz | 420 Hz |

La feature más discriminante resulta ser **`damping_logdec`**, coherente con
la literatura para frutos de pulpa blanda.

---

## 8. Módulo bimodal (datos reales de papaya)

Con las 10 papayas reales (3 series × 3 repeticiones = 90 fotos + 90 audios)
el proyecto pasó a un enfoque **bimodal**: color de la cáscara (foto) +
resonancia acústica (golpe). Los scripts están en `bimodal/` y se ejecutan en
este orden desde la raíz del proyecto:

```powershell
python bimodal/organizar_archivos.py   # Paso 0: renombra a papaya{N}_serie{S}_rep{R}
python bimodal/consolidar_dataset.py   # Color L*a*b* + FFT → data/papayas/dataset_papayas.csv
python bimodal/calcular_umbrales.py    # Regla de decisión → data/papayas/umbrales_app.json
```

| Script | Qué hace |
|---|---|
| `nomenclatura.py` | Convención de nombres y llave (papaya, serie, repetición) |
| `organizar_archivos.py` | Copia fotos (`IMG_*.jpg`) y audios (`Mi grabación N.wav`) a la nomenclatura estándar |
| `vision_color.py` | Segmenta la cáscara (saturación + Otsu) y calcula L*, a*, b* (respeta Display P3) |
| `acustica_fft.py` | Pasa-altos 100 Hz, ventana del impacto, FFT → frecuencia dominante y amplitud |
| `consolidar_dataset.py` | Une visión y acústica; genera `dataset_papayas.csv` y `resumen_por_papaya.csv` |
| `calcular_umbrales.py` | Centroide más cercano ponderado por Fisher → 5 estadios (verde, pintón, semimaduro, maduro, muy maduro) |

La verdad de campo está en `data/papayas/etiquetas_papayas.csv` (orden de
madurez 1–10, 2 papayas por estadio). La app móvil lee `umbrales_app.json`;
`mobile/www/js/bimodal.js` es el port a JavaScript de `vision_color.py` y se
verifica con `python mobile/tests/make_reference_bimodal.py` +
`node mobile/tests/compare_bimodal.mjs`.

> Las fotos y audios originales (`imagenes papaya/`, `AUDIOS DE PAPAYA/`,
> ~250 MB) no van en el paquete de código; se comparten aparte.

