# EEG-Library

Librería en Python para el procesamiento automatizado y el análisis de señales EEG del casco **Emotiv Epoc X**, orientada al estudio de la **fatiga mental percibida**. Trabajo Fin de Grado desarrollado con datos de la Facultad de Ciencias del Deporte.

<img width="1024" height="480" alt="EEG-Library" src="https://github.com/user-attachments/assets/fccc9847-7664-4eff-96a4-abebb1e61f63" />

---

## Qué hay en el repositorio

| Componente | Dónde | Qué es |
|---|---|---|
| **Librería** | `src/eeg_library/` | Pipeline de 13 fases: del EDF crudo a la matriz de features, más las reglas de evaluación del modelado. |
| **Tests** | `tests/` | 504 pruebas con `pytest`, organizadas como la librería. |
| **Análisis** | `analisis/` | Guiones reproducibles que generan las cifras citadas en la documentación y en la memoria. |
| **Documentación** | `docs/` | Estado del pipeline, cuaderno de modelado, concordancia de fases e informe de cada análisis. |
| **Pre-registro** | `PREREGISTRO_TEST.md` | Protocolo del análisis confirmatorio sobre el test, fijado antes de abrirlo. No se modifica. |
| **Resultados** | `resultados/` | Figuras y registros de ejecución vigentes. |
| **Archivo** | `archivo/` | Material de fases retiradas, conservado como registro. |

El dataset **no está en el repositorio** (172,8 h de señal, 16,2 GB de ventanas, datos pseudonimizados). Se localiza con la variable de entorno `EEG_DATA`.

---

## Estructura

```
EEG-Library/
├── src/eeg_library/
│   ├── preprocesado/     Fases 1-6: lectura, recorte, filtros, calidad, CAR, ICA
│   ├── procesado/        Fases 7-13: épocas, PSD, bandas, lotes, etiquetas VAS, matriz del RF
│   └── modelado/         Fase 14 en adelante: evaluación y análisis de condición
├── tests/
│   ├── preprocesado/  procesado/  modelado/
│   └── test_pipeline_integracion.py
├── analisis/             guiones de análisis (+ README con el detalle de cada uno)
├── docs/
│   ├── ESTADO.md         pipeline Fases 1-13 (congelado)
│   ├── MODELADO.md       cuaderno de modelado y resultados (Fase 14 en adelante)
│   ├── FASES.md          concordancia de fases, código y documentos
│   ├── RUTAS_ORIGINALES.md
│   └── analisis/         un informe por análisis
├── resultados/
│   ├── figuras/
│   └── logs/
├── archivo/              Fase 15 (retirada)
├── PREREGISTRO_TEST.md
├── pyproject.toml
└── LICENSE
```

---

## Instalación

Requiere **Python 3.10 o superior** y Git. Los resultados documentados se generaron con Python 3.10.11 (versiones completas en `analisis/README.md`).

**Windows (PowerShell)**

```powershell
git clone https://github.com/Miguelit011/EEG-Library
cd EEG-Library
python -m venv .venv
.venv\Scripts\Activate
pip install --upgrade pip setuptools
pip install -e ".[dev]"
```

**Mac / Linux**

```bash
git clone https://github.com/Miguelit011/EEG-Library
cd EEG-Library
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools
pip install -e ".[dev]"
```

La instalación se hace una sola vez; después solo hay que activar el entorno en cada terminal nueva. Para comprobarla:

```bash
pytest        # 504 pruebas
```

---

## Datos

Los guiones esperan esta estructura bajo `EEG_DATA`:

```
<EEG_DATA>/
├── dataset_final/    EDF (178 sujetos)
├── metadata/         manifest_canonical.csv, splits.csv, labels*.csv, VAS_100mm_preFE_filtrado.csv
├── processed/        matriz_rf.parquet, matriz_modelado.parquet y los JSON de resultados
└── ventanas/         ventanas crudas para la ruta CNN
```

Si no está en la ruta por defecto (`C:\Users\migue\Desktop\TFG\EEG-Data`):

```powershell
$env:EEG_DATA = "D:\otra\ruta"      # PowerShell
```
```bash
export EEG_DATA=/ruta/al/dataset    # bash
```

`metadata/splits.csv` es la única fuente de verdad de las particiones (125 sujetos train, 25 val, 28 test).

---

## Ejecutar el pipeline

Cada fase es una función independiente, exportada desde `eeg_library`:

```python
from eeg_library import cargar_eeg, recortar_registro, aplicar_filtros, evaluar_calidad

raw, meta = cargar_eeg(r"C:\ruta\al\sub-C1F081_ses-M2_task-MSIT_eeg.edf")   # Fase 1
raw, _ = recortar_registro(raw)            # Fase 2: 5 s al inicio y al final
raw, _ = aplicar_filtros(raw)              # Fase 3: notch 50 Hz + bandpass 1-30 Hz
buenos, excluidos, metricas = evaluar_calidad(raw)   # Fase 4
```

El orden de las fases importa: la calidad de canal se evalúa sobre la señal ya recortada y filtrada (explicación en `docs/ESTADO.md`, «Nota de corrección de orden»). Para procesar el dataset completo están `cargar_tareas`, `procesar_lote` y `consolidar_features` (Fase 11), y `construir_matriz_rf` genera la matriz de la Fase 13. Cada fase, su firma y sus decisiones de diseño están en `docs/ESTADO.md`; `docs/FASES.md` indica en qué módulo vive cada una.

---

## Ejecutar los análisis

Desde la raíz del repositorio y con el entorno activado:

```bash
python analisis/analisis_regresion_lineal.py
python analisis/analisis_oa_oc.py > resultados/logs/oa_oc.log 2>&1
```

| Guion | Pregunta | Duración | Informe |
|---|---|---|---|
| `analisis_fase14.py` | ¿Predice el RF el VAS (regresión)? | ~4 min | `docs/MODELADO.md` |
| `analisis_fase14b_clases.py` | ¿Y como tres niveles de fatiga? | ~6 min | `docs/MODELADO.md` |
| `analisis_control_visita.py` | Control de la deriva por número de visita | ~1 min | `docs/MODELADO.md` |
| `analisis_potencia_test.py` | Potencia de la prueba principal | ~1,5 min | `docs/MODELADO.md` |
| `analisis_regresion_lineal.py` | VAS ~ índice theta/alfa | ~20 s | `docs/analisis/ANALISIS_REGRESION_LINEAL.md` |
| `analisis_pca_clustering.py` | ¿Se separan solos los grupos extremos de VAS? | ~40 s | `docs/analisis/ANALISIS_PCA_CLUSTERING.md` |
| `analisis_oa_oc.py` | Ojos abiertos frente a cerrados (control positivo) | ~7,5 min | `docs/analisis/ANALISIS_OA_OC.md` |
| `analisis_reposo_tarea.py` | Reposo frente a tarea ejecutiva | ~15 min | `docs/analisis/ANALISIS_REPOSO_TAREA.md` |
| `analisis_tareas_ejecutivas.py` | Las cinco tareas ejecutivas entre sí | ~26 min | `docs/analisis/ANALISIS_TAREAS_EJECUTIVAS.md` |

Son independientes y pueden ejecutarse en cualquier orden. Los de reposo y tareas aceptan `--rapido` (parámetros mínimos, salida temporal; sus cifras no son resultados). Todos usan semilla fija (`SEMILLA = 42`). Detalle de cada uno, requisitos y garantías metodológicas: [`analisis/README.md`](analisis/README.md).

> **El conjunto de test no se reejecuta.** `analisis_test.py --confirmatorio` se ejecutó una sola vez el 2026-10-07, según [`PREREGISTRO_TEST.md`](PREREGISTRO_TEST.md). Los demás guiones trabajan solo con `train` + `val`.

---

## Dónde están los resultados

| Qué | Dónde |
|---|---|
| Cifras (JSON) | `<EEG_DATA>/processed/*.json` |
| Resultado del test | `<EEG_DATA>/processed/test_20261007-133358/resultados.json` y la sección final de `docs/MODELADO.md` |
| Interpretación | `docs/MODELADO.md` y `docs/analisis/` |
| Figuras | `resultados/figuras/` |
| Registros de ejecución | `resultados/logs/` |
| Fases retiradas (10 y 15) | `docs/FASES.md` y `archivo/` |

---

## Versiones citables

| Tag | Qué fija |
|---|---|
| `preregistro-test-v1` | Commit del pre-registro del test (`bb55aee`), con los SHA-256 de `PREREGISTRO_TEST.md` y `analisis/analisis_test.py` |
| `pre-reorganizacion` | Estado del repositorio antes de reorganizar carpetas y documentos |

Los hashes de los ficheros congelados y la correspondencia entre rutas antiguas y nuevas están en [`docs/RUTAS_ORIGINALES.md`](docs/RUTAS_ORIGINALES.md).

---

## Licencia

MIT. Ver [`LICENSE`](LICENSE).
