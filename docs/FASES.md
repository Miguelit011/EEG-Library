# Concordancia de fases

Índice único de las fases del proyecto: dónde está documentada cada una, dónde
se implementa y cómo se numeraba antes. Los huecos de numeración (10 y 15) son
deliberados: renumerar habría invalidado referencias del código, los tests y el
historial.

## Pipeline (Fases 1-13) — documentadas en [`ESTADO.md`](ESTADO.md), congelado

| Fase | Contenido | Código (`src/eeg_library/`) |
|---|---|---|
| 1 | Lectura de EDF | `preprocesado/reader.py` |
| 2 | Recorte de márgenes | `preprocesado/filtering.py` |
| 3 | Filtros notch + bandpass | `preprocesado/filtering.py` |
| 4 | Calidad de canales | `preprocesado/quality.py` |
| 5 | Re-referenciación CAR | `preprocesado/referencing.py` |
| 6 | Artefactos oculares (ICA + ICLabel) | `preprocesado/artifacts.py` |
| 7 | Epocado | `procesado/epoching.py` |
| 8 | PSD (Welch) | `procesado/spectral.py` |
| 9 | Potencia por bandas e índice theta/alfa | `procesado/bands.py` |
| 10 | **Retirada** (2026-08-23): auto-etiquetado por umbrales; nunca se implementó | — |
| 11 | Procesamiento por lotes | `procesado/batch.py` |
| 11.5 | Máscara de validez de canal | `procesado/features.py` |
| 12 | Etiqueta VAS por sesión | `procesado/labels.py` |
| 13 | Matriz de features del Random Forest | `procesado/matrix.py` |

### Numeración antigua

Los commits anteriores a la «Nota de corrección de orden» (2026-07-28) usan
otra numeración: *Fase 2* = calidad de canal, *Fase 3* = CAR, *Fase 4* =
filtros y recorte. Equivalen a las Fases 4, 5 y 2-3 actuales. Los mensajes de
commit no se reescriben.

## Modelado (Fase 14 en adelante) — documentadas en [`MODELADO.md`](MODELADO.md)

| Fase | Contenido | Guion (`analisis/`) | Informe |
|---|---|---|---|
| 14 | Random Forest sobre el VAS (regresión) | `analisis_fase14.py` | `MODELADO.md` |
| 14b | VAS en tres niveles | `analisis_fase14b_clases.py` | `MODELADO.md` |
| — | Control de la deriva por visita | `analisis_control_visita.py` | `MODELADO.md` |
| — | Potencia de la prueba principal | `analisis_potencia_test.py` | `MODELADO.md` |
| — | VAS ~ theta/alfa (Fase 9) | `analisis_regresion_lineal.py` | [`analisis/ANALISIS_REGRESION_LINEAL.md`](analisis/ANALISIS_REGRESION_LINEAL.md) |
| 15 | **Retirada** (2026-09-15); su código no llegó a commitearse | — | [`archivo/fase15_retirada_2026-08-26/`](../archivo/fase15_retirada_2026-08-26/README.md) |
| 16 | PCA + clustering de los grupos extremos | `analisis_pca_clustering.py` | [`analisis/ANALISIS_PCA_CLUSTERING.md`](analisis/ANALISIS_PCA_CLUSTERING.md) |
| 17 | Ojos abiertos frente a cerrados | `analisis_oa_oc.py` | [`analisis/ANALISIS_OA_OC.md`](analisis/ANALISIS_OA_OC.md) |
| 18 | Reposo frente a tarea | `analisis_reposo_tarea.py` | [`analisis/ANALISIS_REPOSO_TAREA.md`](analisis/ANALISIS_REPOSO_TAREA.md) |
| 19 | Las cinco tareas ejecutivas | `analisis_tareas_ejecutivas.py` | [`analisis/ANALISIS_TAREAS_EJECUTIVAS.md`](analisis/ANALISIS_TAREAS_EJECUTIVAS.md) |
| — | Análisis confirmatorio sobre el test (ejecutado el 2026-10-07) | `analisis_test.py` | [`../PREREGISTRO_TEST.md`](../PREREGISTRO_TEST.md), `MODELADO.md` |
