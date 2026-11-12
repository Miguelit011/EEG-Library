# EEG Pipeline — TFG

## Descripción del proyecto
Librería Python para procesar señales EEG del casco Emotiv Epoc X.
Objetivo: facilitar el análisis de fatiga mental a investigadores
de la Facultad de Ciencias del Deporte.

## Stack técnico
- Python 3.10+
- MNE-Python (procesamiento EEG, lectura EDF)
- MNE-ICALabel + ONNX Runtime (clasificación automática de componentes ICA, Fase 6)
- NumPy / SciPy (análisis espectral)
- Pandas (metadata de origen por época — sujeto/condición/sesión/archivo, Fase 7)
- Pytest (tests)

## Pipeline de procesamiento

> Orden corregido (revisado según el algoritmo FASTER original de Nolan et al.,
> 2010, y el pipeline de referencia de Makoto Miyakoshi/EEGLAB). El orden
> anterior evaluaba la calidad de canal y aplicaba CAR **antes** de recortar
> márgenes y filtrar, lo que contaminaba las métricas de calidad (varianza,
> z-score robusto) con artefactos de colocación/retirada del casco y con
> deriva de baja frecuencia sin filtrar.

1. Lectura de archivos EDF (prioridad), EDF+, CSV, BDF+
2. Recorte de márgenes: eliminar 5 s al inicio y 5 s al final del registro
   (artefactos de colocación y retirada del casco)
3. Filtro notch a 50 Hz (ruido red eléctrica) + filtro bandpass FIR 1–30 Hz
4. Evaluación de calidad de canales (criterios CQ + señal plana absoluta +
   z-score robusto FASTER para amplitud excesiva) — sobre señal ya recortada
   y filtrada
5. Re-referenciación CAR (Common Average Reference) condicional: ≥8 canales
   buenos → aplicar; 6–7 → aplicar con advertencia; <6 → omitir
6. Eliminación de artefactos mediante ICA (algoritmo Infomax Extended) —
   alcance limitado a **componentes oculares**, dado que el Emotiv Epoc X no
   tiene canales EOG/ECG dedicados y el presupuesto del TFG es de 300h.
   Clasificación automática de componentes con **ICLabel** (MNE-ICALabel) en
   vez de `find_bads_eog` con un canal frontal como proxy de EOG, para evitar
   la circularidad de usar como referencia "esto es ojo" un canal que también
   contiene señal cerebral de interés (justificación completa en docs/ESTADO.md,
   Fase 6).
7. Epoching (segmentación en ventanas): ventanas fijas de 4 s (1024 muestras
   a 256 Hz) con solapamiento del 50 % (paso de 2 s) entre ventanas
   consecutivas; el resto final de cada archivo que no completa una ventana
   entera se descarta. Este epocado es compartido por las Fases 9-10
   (extracción de features para Random Forest) y por la ruta CNN (ventanas
   crudas normalizadas), por lo que no aplica ninguna transformación
   específica de ninguna de las dos rutas. Cada época conserva metadata de
   origen (sujeto, condición, sesión, archivo fuente), imprescindible para
   la partición train/test por sujeto (evitar fuga de datos) en fases
   posteriores.
   - Duración de 4 s: respaldada por la literatura de detección de
     fatiga/carga mental con EEG tanto para modelos clásicos (features
     espectrales estables en ese horizonte) como para CNN.
   - Solapamiento del 50 %: compensa la corta duración de cada archivo
     (~3 min) sin generar redundancia excesiva entre ventanas.
8. Análisis espectral: densidad espectral de potencia (PSD) mediante el
   método de Welch, por canal individual y por época (sin promediar por
   banda ni por región — eso es la Fase 9). Sub-ventanas internas de 1 s
   (256 muestras a 256 Hz) con 50 % de solapamiento y ventana de Hann,
   resolución de 1 Hz; rango de cálculo limitado a 1–30 Hz, ya que el
   filtro de la Fase 3 eliminó el resto (justificación completa en
   docs/ESTADO.md, Fase 8).
9. Extracción de potencia por bandas: delta, theta, alpha, beta + índice
   theta/alpha. El índice **no es la etiqueta**: se calcula porque es el
   biomarcador de referencia de la literatura y el TFG tiene que reportarlo.
   Medido, no sigue a la fatiga percibida en esta cohorte (ρ = +0.004,
   p = 0.93 dentro de persona) — resultado negativo que es en sí un hallazgo
10. *(retirada)* — el auto-etiquetado de fatiga por umbrales sobre el índice
    theta/alpha se eliminó del pipeline el 2026-08-23, al disponer de una
    etiqueta externa real (el VAS, Fase 12). Nunca llegó a implementarse. El
    hueco en la numeración es deliberado: renumerar invalidaría las
    referencias del código, los tests y el historial. Ver docs/ESTADO.md,
    "Fase 10 — retirada"
11. Procesamiento por lotes (múltiples archivos simultáneamente)
12. Dataset supervisado: enlace del VAS de fatiga percibida con las
    grabaciones. La etiqueta
    es **por sesión** (sujeto × momento), no por grabación ni por ventana.
    El enlace se hace por `(sujeto_id, medicion)` y la agrupación para
    splits y bootstrap por `persona_uid` — nunca por `sujeto_id`, porque seis
    de ellos contienen dos personas distintas (pulsera Fitbit reutilizada).
    Detalle completo, correcciones de metadata y recuentos en docs/ESTADO.md,
    Fase 12
13. Matriz de features del Random Forest: agregación por grabación y **por
    región** (frontal / frontocentral / temporal / posterior), restringida a
    los canales válidos de la Fase 11.5; potencia absoluta en log₁₀, media y
    desviación a lo largo de las ventanas. 4543 × 77 features.
    `columnas_de_features()` es la **única** definición de X del proyecto:
    lo que no devuelve, no entra al `fit` — en particular `medicion`

> Para el estado de implementación por fase (qué está hecho, decisiones de
> diseño detalladas, tests), ver docs/ESTADO.md — cubre las Fases 1-13 (pipeline y
> matriz de features) y está congelado en ese punto. El entrenamiento del
> modelo, su evaluación y la ruta CNN (Fase 14 en adelante) viven en
> docs/MODELADO.md.

## Convenciones de código
- Comentarios y docstrings en español
- Cada función debe tener su test correspondiente en /tests
- Frecuencia de muestreo mínima: 256 Hz (validar al cargar)
- Los datos reales están fuera del repo (ver "Ubicación del dataset" más abajo)

## Contexto científico

> Cifras medidas sobre el dataset ya organizado (2026-08-10). Sustituyen a la
> estimación previa que figuraba aquí (160 sujetos, 960 archivos, 6 momentos,
> 5 condiciones, 30–37 h), que era anterior a la organización. Desglose
> completo en docs/ESTADO.md, "El dataset final".

- Ubicación del dataset (fuera del repo y fuera de OneDrive): `C:\Users\migue\Desktop\TFG\EEG-Data\` —
  `dataset_final/` con los EDF, `metadata/` con `manifest_canonical.csv`,
  `splits.csv`, `VAS_100mm_preFE_filtrado.csv` y la salida de la Fase 12
  (`labels.csv`, `labels_grabaciones.csv`), y `processed/` + `ventanas/` con
  la salida de la Fase 11.
  Está fuera de OneDrive a propósito: las ventanas crudas para la CNN ocupan
  16.1 GB y no deben sincronizarse
- Sujetos: 178, procedentes de un estudio de la Facultad de Ciencias del
  Deporte (edad y otras características demográficas no confirmadas —
  no asumir)
- Registros: 4574 archivos EEG válidos = 178 sujetos × 5 mediciones (M2–M6)
  × 7 condiciones, con ausencias (no todos los sujetos completan las 5×7)
- Mediciones (información del equipo del estudio, 2026-10-21): tomadas al
  inicio y al final de cada trimestre de un curso escolar. M1 = inicio 1.er
  trimestre, M2 = final 1.º, M3 = inicio 2.º, M4 = final 2.º, M5 = inicio 3.º,
  M6 = final 3.º. El dataset EEG solo contiene M2–M6: M1 se descartó tras
  auditarla (solo 5 condiciones, sin MSIT ni PVSAT; ~29 % de las grabaciones
  sin identidad de sujeto resuelta). Tabla y motivos en docs/ESTADO.md
- Condiciones: MSIT, PASAT, PVSAT, PVTB, STROOP (carga cognitiva) más OA y
  OC (reposo con ojos abiertos y cerrados, que sirven de línea base)
- Orden del protocolo dentro de cada sesión, fijo:
  OC → OA → STROOP → PASAT → PVTB → MSIT → PVSAT (~15.5 min)
- Duración por grabación: 25–379 s (mediana 127 s)
- Volumen total del dataset: 172.8 horas de señal EEG bruta (antes de
  filtrado/recorte)
- Partición: por sujeto y estratificada por centro (8 centros, C1–C8) —
  125 sujetos train / 25 val / 28 test, definida en `splits.csv`
- Variable de interés: **fatiga mental percibida**, medida con VAS de 100 mm
  respondido por el participante al inicio de cada sesión, inmediatamente
  antes del EEG. Regresión sobre el continuo 0–100; clasificación en tres
  clases como análisis secundario. Subconjunto etiquetado real: 4574
  grabaciones del manifest → **4453 grabaciones / 653 sesiones / 174 personas
  / 274 718 ventanas**. El n efectivo son las **653 sesiones**, no las
  ventanas: todas las de una sesión comparten el mismo valor de VAS. Línea
  base a batir: R² = 0.260 (media por persona, leave-one-out); ICC(1) = 0.457
- Casco: Emotiv Epoc X (14 electrodos + 2 referencia CMS/DRL). Los EDF traen
  60, 67 o 71 canales según la versión de EmotivPro: solo 14 son EEG, el
  resto son calidad de contacto (CQ), calidad de señal (EQ), telemetría,
  marcadores y movimiento

## Líneas de análisis en carpeta y nombres

Después de la Fase 21, cada análisis vive en su propia carpeta dentro de
`analisis/` (índice en `analisis/README.md`): `normalizacion_vas/`,
`fatiga_fg_fc/` y `tiempo_en_tarea/` (con su sublínea `deep_learning/`).

- **La línea `tiempo_en_tarea` se llamó «fatiga aguda» hasta el 2026-11-05**
  (carpeta `analisis/fatiga_aguda/`, API `*_fatiga_aguda`, modelos `fa_*`). Se
  renombró tras la revisión de los tutores: lo confirmado en test es un cambio
  del EEG con el tiempo en la tarea, no que se deba a la fatiga. En el código, la
  documentación y la memoria se dice «tiempo en tarea»; «fatiga aguda» solo se
  usa para describir la hipótesis («compatible con fatiga aguda»).
- **Conservan el nombre antiguo a propósito y no se renombran:**
  - los guiones y configuraciones con SHA-256 registrado en test:
    `06_test/evaluacion_test.py`, los cinco guiones y `configuracion_dl.json`
    de `deep_learning/04_test_eegnet/`, y `03_clasificador/configuracion_elegida.json`
    y `medida_elegida.json`. Editarlos, aunque sea un comentario, rompe la
    verificación de que el código del repositorio es el que se evaluó;
  - los logs y los `resultados_*.json`, que son el registro de lo ejecutado;
  - el texto de los pre-registros (los cambios van en su sección 11);
  - las carpetas de EEG-Data `fatiga_aguda_test_20261027-190547` y
    `fatiga_aguda_dl_test_20261031-224217`, que son de solo lectura.
