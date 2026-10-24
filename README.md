# Madurez de papaya — proyecto para Android Studio

Abrir: **File → Open** → elegir esta carpeta (`MadurezPapaya`). Esperar el Gradle Sync.
Para ver todos los archivos, en el panel izquierdo cambiar la vista de **Android** a **Project**.

| Carpeta | Contenido |
|---|---|
| `app/src/main/assets/public/js/pipeline.js` | Análisis acústico del golpe (onset, FFT, f1/f2, ζ, S, EI) |
| `app/src/main/assets/public/js/bimodal.js` | Color de la cáscara (CIELAB) y regla de 5 estadios |
| `app/src/main/assets/public/js/estadios.js` | Escala E1–E5 de la Tabla 2: % de superficie amarilla, color por zona (peduncular, ecuatorial, apical), C*, h°, manchas y control del fondo blanco |
| `app/src/main/assets/public/js/clima.js` | Clima de la localidad (Open-Meteo, sin clave; Tarapoto por defecto o GPS) y días estimados hasta E5 con un modelo Q10 |
| `app/src/main/assets/public/js/historial.js` | Historial local de mediciones (un registro por fruto) y exportación a CSV |
| `app/src/main/java/.../ExportarPlugin.java` | Plugin nativo que envía el CSV al menú Compartir de Android |
| `app/src/main/assets/public/js/app.js` | Interfaz: grabación (hasta 3 golpes), cámara (2 caras), resultados, clima e historial |
| `app/src/main/assets/public/model.json`, `umbrales_app.json` | Modelo y umbrales exportados desde Python |
| `app/src/main/java/`, `AndroidManifest.xml` | Arranque Android (Capacitor) y permisos |
| `codigo_python/` | Código Python original del que salen el modelo y los umbrales (solo para consulta; Gradle no lo compila) |

La app es HTML/JS dentro de un WebView (Capacitor): la lógica está en los `.js`, no en Java/Kotlin.
