# Registro de cambios

Una entrada por versión de la librería (`__version__` en `src/eeg_library/__init__.py` y
`version` en `pyproject.toml`), sacada de los commits que la cambiaron. Las fases siguen la
numeración actual de `CLAUDE.md`; los commits más antiguos usan la de entonces, anterior a la
corrección del orden del pipeline.

- Las versiones 0.8, 0.10 y 0.12 no existen: se saltaron.
- `docs/ESTADO.md` se congeló en la 0.11.0 (Fases 1-13); las versiones que cita son las de entonces.
- Qué entrega cada versión a EEG-Platform, desde la 0.14.0: [`docs/CONTRATO.md`](docs/CONTRATO.md).

## Sin versión todavía

Cambios de la revisión de los repositorios (2026-11-11 y 2026-11-12). No cambian ningún cálculo.

- Extra `[analisis]` (`matplotlib`, `openpyxl`), `requirements-validacion.txt` con las versiones
  exactas de validación y `requirements-dl.txt` con el entorno aparte de PyTorch.
- `EEG_DATA` toma por defecto la carpeta `EEG-Data` junto al repositorio, en vez de la ruta del
  equipo de desarrollo (salvo en los dos guiones con SHA-256 registrado).
- `.gitignore`: también `*.parquet`, `*.xlsx`, `*.xls` y las carpetas `tablas/`.
- README actualizado (tests, Python, estructura, líneas de análisis y EEG-Platform).
- `columnas_de_features_reposo`, `verificar_sin_fuga`, `COLUMNA_FUGA` y `COLUMNAS_PROCESAMIENTO`
  pasan de `modelado/reposo.py` a `procesado/matrix.py`, para que `analizar_grabacion` no dependa
  de un módulo de análisis. `modelado/reposo.py` los reexporta con los mismos nombres.
- La carpeta de los modelos y su SHA-256 se calculan en un solo sitio (`modelado/prediccion.py`);
  `tiempo_en_tarea_eegnet.py` toma `CONDICIONES`, `MIN_VENTANAS` y la carpeta de
  `tiempo_en_tarea.py`. Constante `UMBRAL_OCULAR` (0.8) en `preprocesado/artifacts.py`.
- Docstrings corregidos: orden de filtrado y CAR, criterio z de la calidad (solo por arriba) y
  referencias a la Fase 10, retirada.

## 0.17.0 — 2026-11-05

Renombrado de «fatiga aguda» a «tiempo en tarea»: módulos `modelado/tiempo_en_tarea*.py`,
funciones `*_tiempo_en_tarea`, carpeta `modelos/tiempo_en_tarea/` e identificadores `tt_*`. No
cambia ningún cálculo, peso ni cifra.

## 0.16.0 — 2026-11-01

EEGNet exportada a ONNX, con paridad exacta con PyTorch, y puntuación por ventana al procesar
una grabación (`puntuar_eegnet`).

## 0.15.0 — 2026-11-01

Potencia por ventana (`por_ventana`) en `analizar_grabacion` y modelos logísticos de tiempo en
tarea para PASAT y PVTB, empaquetados con su ficha.

## 0.14.0 — 2026-10-15

Contrato con EEG-Platform: `analizar_grabacion` con parámetros cerrados, `prediccion.py` y los
modelos c1, c2 y c3 empaquetados con su ficha.

## 0.13.0 — 2026-08-25

Fases 14 y 14b: banco de evaluación supervisada del Random Forest (`modelado/modeling.py`), con
las reglas que dan sentido a su resultado, escrito y probado antes de entrenar.

## 0.11.0 — 2026-08-23

Fase 12, enlace del VAS con las grabaciones (por centro, pulsera y medición, con agrupación por
`persona_uid`), y Fase 13, matriz de features del Random Forest.

## 0.9.0 — 2026-08-10

Fase 11, procesamiento por lotes, y Fase 11.5, máscara de validez de canal.

## 0.7.2 — 2026-08-10

Corrección: los canales de mala calidad se excluyen del promedio del índice theta/alfa.

## 0.7.1 — 2026-08-10

Corrección del tipado de los canales EEG y del rescalado de la calidad de contacto (CQ) con el
dataset real.

## 0.7.0 — 2026-08-02

Fase 9: potencia por bandas e índice theta/alfa.

## 0.6.0 — 2026-08-01

Fase 8: densidad espectral de potencia por el método de Welch, y test de integración del pipeline.

## 0.5.0 — 2026-07-31

Fase 7: epocado en ventanas fijas de 4 s con 50 % de solapamiento.

## 0.4.0 — 2026-07-29

Fase 6: eliminación de artefactos oculares con ICA.

## 0.3.0 — 2026-05-18

Fases 2 y 3: recorte de márgenes y filtros (notch a 50 Hz y paso banda de 1 a 30 Hz).

## 0.2.0 — 2026-03-09

Fases 4 y 5: evaluación de la calidad de canal y re-referenciación CAR.

## 0.1.0 — 2026-03-03

Fase 1: estructura del proyecto y lectura de archivos EDF.
