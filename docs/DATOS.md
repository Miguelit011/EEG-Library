# Datos del estudio (`EEG_DATA`)

El dataset vive fuera del repositorio, en la carpeta que indica la variable de entorno `EEG_DATA` (por defecto, `EEG-Data` junto al repositorio; ver el [README](../README.md#datos)). Contiene datos pseudonimizados de participantes menores de edad, cedidos por el equipo del estudio de la Facultad de Ciencias del Deporte, y nunca se versiona.

La carpeta se trata como de **solo lectura**. Los únicos que escriben en ella son el procesado por lotes (Fase 11), la construcción de etiquetas y matriz (Fases 12 y 13) y los guiones de análisis, cada uno en su archivo o carpeta de salida. A mano no se edita nada.

---

## Estructura

```
<EEG_DATA>/                                   36,1 GB
├── BD_COMPLETA_corregida.xlsx                base del estudio (cuestionarios, conducta), versión vigente
├── BD_COMPLETA_sincalificaciones.xlsx        versión anterior de la base
├── BD_COMPLETA_ConCalificaciones.xlsx        primera versión recibida de la base
├── dataset_final/                            4574 EDF                                        18,6 GB
├── metadata/                                 manifest, partición, VAS y etiquetas
├── processed/
│   ├── features/                             4543 parquet, uno por grabación (Fase 11)        0,6 GB
│   ├── features_all.parquet                  features por época, todas las grabaciones
│   ├── matriz_modelado.parquet               ídem + validez de canal (Fase 11.5)
│   ├── matriz_rf.parquet                     matriz del Random Forest (Fase 13)
│   ├── procesamiento_log.csv / _resumen.json registro del lote
│   ├── *.json                                cifras de los análisis de las Fases 14-21
│   └── <ejecución>_<fecha>/                  artefactos de las evaluaciones sobre val y test
└── ventanas/                                 4543 .npy de ventanas crudas (Fase 11)          16,1 GB
```

---

## Qué es cada archivo

Tipo: **fuente** (recibido del estudio o resultado de su organización), **derivado** (lo genera la librería y se puede regenerar) o **registro** (salida de una ejecución que no se repite; algunos tienen su SHA-256 registrado).

### Raíz: base del estudio

| Archivo | Tipo | Contenido y uso |
|---|---|---|
| `BD_COMPLETA_corregida.xlsx` | fuente | Base de cuestionarios del estudio (PedsQL, VAS, rendimiento en las tareas, antropometría, actividad de la pulsera). Versión vigente: corrige la codificación de los ítems PedsQL de M5 y M6 (hoja `AUDIT_CORRECCION_PEDS`). La leen `analisis/fatiga_fg_fc/` (hoja `CUESTIONARIOS`) y los pasos `05_entre_sesiones` y `07_redaccion` de `analisis/tiempo_en_tarea/`. |
| `BD_COMPLETA_sincalificaciones.xlsx` | fuente | Versión anterior, con la codificación de M5 y M6 sin corregir. Solo la lee el punto 0 de `fatiga_fg_fc/01_descriptivo`, que verifica la corrección celda a celda. |
| `BD_COMPLETA_ConCalificaciones.xlsx` | fuente | Primera versión recibida, que además incluye calificaciones escolares. Ningún guion la lee. Se conserva porque sirvió de prueba en una de las correcciones de centro de `fatiga_fg_fc/datos_fg_fc.py`. |

Para la PedsQL solo vale la versión corregida. Detalle en [`analisis/fatiga_fg_fc/README.md`](../analisis/fatiga_fg_fc/README.md).

### `dataset_final/`

| Archivo | Tipo | Contenido |
|---|---|---|
| `sub-<sujeto>_ses-<medición>_task-<condición>_eeg.edf` | fuente | Las 4574 grabaciones válidas tras auditar y organizar los EDF originales: 178 sujetos, mediciones M2-M6 y las 7 condiciones (OA, OC, STROOP, PASAT, PVTB, MSIT y PVSAT). |

La nomenclatura está inspirada en BIDS (`sub-`, `ses-`, `task-`), pero la carpeta no es un dataset BIDS: no tiene `dataset_description.json`, `participants.tsv` ni archivos laterales. Los metadatos de cada grabación están en el manifest.

### `metadata/`

| Archivo | Tipo | Contenido | Lo genera / lo usa |
|---|---|---|---|
| `manifest_canonical.csv` | fuente | Una fila por EDF: sujeto, centro, medición, condición, casco, fecha y hora, duración, canales, tamaño y **MD5** | Organización del dataset; lo lee `cargar_tareas` (Fase 11) |
| `splits.csv` | fuente | Partición por sujeto, estratificada por centro: 125 train, 25 val, 28 test | Única definición de la partición |
| `VAS_100mm_preFE_filtrado.csv` | fuente | Respuestas al VAS de fatiga (M1-M6). El nombre es el del archivo original | Fase 12 y `analisis/normalizacion_vas/` |
| `labels.csv` | derivado | Una fila por sesión (sujeto × medición): VAS, `persona_uid`, partición, recuentos | `construir_etiquetas` (Fase 12) |
| `labels_grabaciones.csv` | derivado | Una fila por grabación, con la etiqueta de su sesión | `construir_etiquetas` (Fase 12) |

### `processed/`: salida del pipeline

| Archivo | Tipo | Contenido | Lo genera |
|---|---|---|---|
| `features/<grabación>.parquet` | derivado | Features por época de una grabación | `procesar_lote` (Fase 11) |
| `features_all.parquet` | derivado | Concatenación de las anteriores: 280 339 épocas × 124 columnas | `consolidar_features` (Fase 11) |
| `matriz_modelado.parquet` | derivado | `features_all` más la validez de cada canal: 139 columnas | Fase 11.5 |
| `matriz_rf.parquet` | derivado | Una fila por grabación procesada (4543), agregada por región | `construir_matriz_rf` (Fase 13) |
| `procesamiento_log.csv` | registro | Una fila por EDF: estado, canales excluidos, ICA, épocas, error si lo hubo | `procesar_lote` |
| `procesamiento_resumen.json` | registro | Parámetros y agregados de la última ejecución del lote | `procesar_lote` |

### `processed/*.json`: cifras de las Fases 14-21

| Archivo | Guion |
|---|---|
| `fase14_resultados.json` | `analisis/analisis_fase14.py` |
| `fase14b_clases.json` | `analisis/analisis_fase14b_clases.py` |
| `control_visita.json` | `analisis/analisis_control_visita.py` |
| `potencia_test.json` | `analisis/analisis_potencia_test.py` |
| `regresion_lineal.json` | `analisis/analisis_regresion_lineal.py` |
| `pca_clustering.json` | `analisis/analisis_pca_clustering.py` |
| `oa_oc.json` | `analisis/analisis_oa_oc.py` |
| `reposo_tarea.json` | `analisis/analisis_reposo_tarea.py` |
| `tareas_ejecutivas.json` | `analisis/analisis_tareas_ejecutivas.py` |

Las líneas de análisis posteriores (`normalizacion_vas/`, `fatiga_fg_fc/`, `tiempo_en_tarea/`) guardan sus resultados dentro del repositorio, en la carpeta de cada paso. Solo las evaluaciones sobre test escriben en `EEG_DATA`.

### `processed/<ejecución>_<fecha>/`: evaluaciones sobre val y test

Cada carpeta es la salida completa de una ejecución: modelos, réplicas de bootstrap y permutaciones, predicciones, `resultados.json` y registro. Son **registros**: no se modifican ni se renombran.

| Carpeta | Qué es | Guion y documentación |
|---|---|---|
| `ensayo_val_20261006-231323` | Primer ensayo general (train → val) con una versión previa del guion, que guardaba las matrices en `matrices/`. Registro idéntico al siguiente | `analisis/analisis_test.py --ensayo` |
| `ensayo_val_20261006-231756` | Ensayo general con la versión definitiva del guion (mismo SHA-256 que la ejecución sobre test) | `analisis/analisis_test.py --ensayo` |
| `test_20261007-133358` | Análisis confirmatorio sobre test, ejecutado una sola vez | [`PREREGISTRO_TEST.md`](../PREREGISTRO_TEST.md); [`MODELADO.md`](MODELADO.md) |
| `fatiga_aguda_test_20261027-190547` | Evaluación en test de la logística de tiempo en tarea | `analisis/tiempo_en_tarea/06_test/` |
| `fatiga_aguda_dl_test_20261031-224217` | Evaluación en test de EEGNet (tiempo en tarea) | `analisis/tiempo_en_tarea/deep_learning/04_test_eegnet/` |

Las dos últimas conservan el nombre que tenía la línea `tiempo_en_tarea` cuando se ejecutaron («fatiga aguda», ver [`CLAUDE.md`](../CLAUDE.md)). Sus resultados y los SHA-256 registrados las citan con ese nombre.

Los modelos que usa EEG-Platform (`src/eeg_library/modelos/`) son copias exactas de los de `test_20261007-133358/modelos/` y `fatiga_aguda_test_20261027-190547/modelos/`, hechas por `analisis/empaquetar_modelos*.py`, que compara los SHA-256. Los ONNX de EEGNet se exportaron desde los pesos de `fatiga_aguda_dl_test_20261031-224217/modelos/`.

### `ventanas/`

| Archivo | Tipo | Contenido |
|---|---|---|
| `<grabación>.npy` | derivado | Ventanas crudas de 4 s de una grabación, `(n_épocas, 14, 1024)` en `float32`. La fila `i` corresponde a la fila `i` de su parquet de `features/`. Las usa la sublínea `tiempo_en_tarea/deep_learning/` |

---

## Recuentos

Verificados sobre los archivos el 2026-11-12. Son los que citan esta documentación y la memoria.

| Concepto | Valor |
|---|---|
| Grabaciones (EDF) | 4574: 178 sujetos, 8 centros, M2-M6, 7 condiciones |
| Señal | 172,8 h; de 25 a 379 s por grabación (mediana 127 s) |
| Grabaciones procesadas | 4543 (31 con error en el lote, registradas en `procesamiento_log.csv`) |
| Épocas | 280 339 |
| Partición | 125 / 25 / 28 sujetos (train / val / test) |
| Sesiones con VAS | 656; con ventanas, **653**, de 174 personas |
| Grabaciones etiquetadas | 4453, con 274 718 ventanas |

---

## Notas de uso

- **Partición e identidad.** La partición se define por sujeto en `splits.csv`. `labels*.csv` y `matriz_rf.parquet` la aplican y añaden `persona_uid`, la unidad de agrupación (algunos `sujeto_id` corresponden a dos personas por reutilización de la pulsera; ver `ESTADO.md`, Fase 12). Las columnas `sujeto` y `split` de `features_all.parquet` y `matriz_modelado.parquet` son las del lote, anteriores a las correcciones de metadatos de la Fase 12: difieren en las 42 grabaciones corregidas, y 14 de ellas (3 personas) cambian de partición. La partición y la identidad se toman siempre de `labels*.csv` o de `matriz_rf.parquet`.
- **Rutas históricas.** `procesamiento_resumen.json`, las columnas `ruta_absoluta` y `ruta_destino` del manifest y los registros de ejecución conservan las rutas del equipo donde se generaron. Documentan el origen; para localizar archivos se usa `EEG_DATA`.
- **Ubicación de los Excel.** Están en la raíz porque así los esperan los guiones de las líneas que los usan. No se mueven a `metadata/` para no cambiar rutas de guiones y registros ya ejecutados.

---

## Integridad

- **EDF:** el manifest guarda el MD5 y el tamaño de cada uno. El 2026-11-12 coincidían los 4574, sin archivos de más ni de menos.
- **Derivados y registros:** los guiones de análisis guardan en su `resultados.json` el SHA-256 de los archivos de `EEG_DATA` que leen, y las fichas de los modelos, el de su modelo. El 2026-11-12 coincidían todos.

Los derivados se pueden regenerar desde los EDF con la librería (Fases 11-13), pero los archivos actuales son la referencia: son los que tienen la huella registrada en los resultados.
