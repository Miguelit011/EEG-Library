# EEG-Library

Librería en Python para el procesamiento automatizado y el análisis de señales EEG del casco **Emotiv Epoc X**, orientada al estudio de la **fatiga mental percibida**. Trabajo Fin de Grado desarrollado con datos de la Facultad de Ciencias del Deporte.

<img width="1024" height="480" alt="EEG-Library" src="https://github.com/user-attachments/assets/fccc9847-7664-4eff-96a4-abebb1e61f63" />

---

## Qué hay en el repositorio

| Componente | Dónde | Qué es |
|---|---|---|
| **Librería** | `src/eeg_library/` | Pipeline de 13 fases: del EDF crudo a la matriz de features, más las reglas de evaluación del modelado, los modelos empaquetados (c1, c2, c3 y tiempo en tarea) y `prediccion.py`, que los aplica a una grabación nueva. |
| **Tests** | `tests/` | 589 pruebas con `pytest` (más 7 que necesitan el dataset), organizadas como la librería. |
| **Análisis** | `analisis/` | Guiones reproducibles que generan las cifras citadas en la documentación y en la memoria. |
| **Documentación** | `docs/` | Estado del pipeline, cuaderno de modelado, concordancia de fases e informe de cada análisis. |
| **Pre-registro** | `PREREGISTRO_TEST.md` | Protocolo del análisis confirmatorio sobre el test, fijado antes de abrirlo. No se modifica. |
| **Resultados** | `resultados/` | Figuras y registros de ejecución vigentes. |
| **Archivo** | `archivo/` | Material de fases retiradas, conservado como registro. |
| **Entornos** | `requirements-validacion.txt`, `requirements-dl.txt` | Versiones exactas del entorno con que se generaron los resultados y del entorno aparte de deep learning. |

El dataset **no está en el repositorio** (172,8 h de señal, 16,1 GB de ventanas, datos pseudonimizados). Se localiza con la variable de entorno `EEG_DATA`.

---

## Estructura

```
EEG-Library/
├── src/eeg_library/
│   ├── preprocesado/     Fases 1-6: lectura, recorte, filtros, calidad, CAR, ICA
│   ├── procesado/        Fases 7-13: épocas, PSD, bandas, lotes, etiquetas VAS, matriz del RF
│   ├── modelado/         Fase 14 en adelante: evaluación, análisis de condición, tiempo en tarea
│   │                     y prediccion.py (aplicar los modelos a una grabación nueva)
│   └── modelos/          modelos empaquetados c1-c3 y, en tiempo_en_tarea/, logística y EEGNet (ONNX)
├── tests/
│   ├── preprocesado/  procesado/  modelado/  datos/
│   └── test_pipeline_integracion.py
├── analisis/
│   ├── *.py              guiones de las Fases 14-21 y de empaquetado (+ README con el detalle)
│   ├── normalizacion_vas/    línea de análisis en carpeta
│   ├── fatiga_fg_fc/         línea de análisis en carpeta
│   └── tiempo_en_tarea/      línea de análisis en carpeta (con deep_learning/)
├── docs/
│   ├── README.md         índice de la documentación
│   ├── ESTADO.md         pipeline Fases 1-13 (congelado)
│   ├── MODELADO.md       cuaderno de modelado y resultados (Fase 14 en adelante)
│   ├── FASES.md          concordancia de fases, código y documentos
│   ├── CONTRATO.md       qué entrega la librería a EEG-Platform
│   ├── DATOS.md          qué contiene la carpeta de datos (EEG_DATA)
│   ├── ejemplos/         salida de ejemplo de analizar_grabacion (JSON)
│   ├── RUTAS_ORIGINALES.md
│   └── analisis/         un informe por análisis de las Fases 14-21
├── resultados/
│   ├── figuras/
│   ├── logs/
│   └── test_20261007-133358/  resultado del test confirmatorio (copia de EEG_DATA)
├── archivo/              Fase 15 (retirada)
├── PREREGISTRO_TEST.md
├── CHANGELOG.md          cambios de cada versión de la librería
├── CLAUDE.md             instrucciones para el asistente de IA usado en el desarrollo
├── pyproject.toml
├── requirements-validacion.txt
├── requirements-dl.txt
└── LICENSE
```

---

## Instalación

Requiere **Python 3.10** (no 3.11 ni posteriores: los modelos empaquetados se guardaron con 3.10 y scikit-learn 1.7.2) y Git. Los resultados documentados se generaron con Python 3.10.11.

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
pytest        # 589 pruebas; las 7 que necesitan el dataset se lanzan con: pytest -m datos
```

**Para ejecutar los análisis**, con las versiones exactas con que se generaron los resultados (incluye `matplotlib` y `openpyxl`, del extra `[analisis]`):

```bash
pip install -r requirements-validacion.txt
```

**Deep learning** (`analisis/tiempo_en_tarea/deep_learning/`): PyTorch va en un entorno aparte, para no alterar las versiones validadas. Cómo crearlo: [`requirements-dl.txt`](requirements-dl.txt).

---

## Datos

Los guiones esperan esta estructura bajo `EEG_DATA`:

```
<EEG_DATA>/
├── BD_COMPLETA_*.xlsx  base de cuestionarios del estudio (tres versiones)
├── dataset_final/      EDF (178 sujetos)
├── metadata/           manifest_canonical.csv, splits.csv, labels*.csv, VAS_100mm_preFE_filtrado.csv
├── processed/          features del lote, matriz_rf.parquet, JSON de resultados y carpetas de las evaluaciones
└── ventanas/           ventanas crudas para la ruta CNN
```

Qué es cada archivo, qué guion lo genera o lo lee, los recuentos y cómo se comprueba su integridad: [`docs/DATOS.md`](docs/DATOS.md).

Por defecto, los guiones y los tests buscan la carpeta `EEG-Data` **junto al repositorio**:

```
TFG\
├── EEG-Library\
└── EEG-Data\
```

Si está en otro sitio:

```powershell
$env:EEG_DATA = "D:\otra\ruta"      # PowerShell
```
```bash
export EEG_DATA=/ruta/al/dataset    # bash
```

Hay dos excepciones: `analisis/analisis_test.py` y `analisis/tiempo_en_tarea/deep_learning/04_test_eegnet/entrenar.py` conservan como valor por defecto la ruta del equipo de desarrollo, porque tienen su SHA-256 registrado y no se editan. En otro equipo, para ejecutarlos hay que definir `EEG_DATA`.

`metadata/splits.csv` es la única fuente de verdad de las particiones (125 sujetos train, 25 val, 28 test).

---

## Uso rápido

Para procesar una grabación completa basta una llamada, la misma que usa EEG-Platform:

```python
from eeg_library import analizar_grabacion, listar_modelos, predecir

resultado = analizar_grabacion(r"C:\ruta\a\grabacion.edf")   # dict serializable a JSON
print(resultado["calidad"])                                   # estado de cada electrodo

for id_modelo in listar_modelos():                            # modelos c1-c3 empaquetados
    p = predecir(id_modelo, resultado["caracteristicas"], resultado["parametros_aplicados"])
    print(id_modelo, p.get("clase"))
```

`resultado` contiene `metadatos`, `parametros_aplicados`, `calidad`, `potencia`, `caracteristicas` y `por_ventana`. Qué es cada campo, qué modelo conviene a cada condición y cómo se puntúa el tiempo en tarea: [`docs/CONTRATO.md`](docs/CONTRATO.md). Hay dos salidas de ejemplo en [`docs/ejemplos/`](docs/ejemplos/).

---

## Ejecutar el pipeline fase a fase

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

Los análisis están organizados de dos formas, según cuándo se hicieron:

- **Fases 14-21:** guiones sueltos en `analisis/*.py`. Sus figuras y registros van a `resultados/figuras/` y `resultados/logs/`, y su interpretación, a `docs/MODELADO.md` y `docs/analisis/`.
- **Líneas posteriores:** una carpeta por línea (`analisis/<línea>/<paso>/`), que reúne en cada paso el guion, sus tablas, sus figuras, `resultados.json` e `INFORME.md`. Cada línea tiene su `README.md`.

No se reorganizan en una sola forma: varios ficheros tienen su SHA-256 registrado antes de evaluar el test, y sus rutas están citadas en la memoria.

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
| `analisis_test.py` | Análisis confirmatorio sobre el test | **Ejecutado una vez; no se reejecuta** | `docs/MODELADO.md` |
| `empaquetar_modelos.py`, `empaquetar_modelos_tiempo_en_tarea.py` | No es un análisis: empaqueta los modelos que usa EEG-Platform | — | `docs/CONTRATO.md` |
| `generar_ejemplo.py` | No es un análisis: genera los JSON de `docs/ejemplos/` | — | `docs/CONTRATO.md` |
| `normalizacion_vas/` | ¿Da una etiqueta más útil el VAS respecto a cada persona? | por pasos | [`README`](analisis/normalizacion_vas/README.md) |
| `fatiga_fg_fc/` | ¿Guardan relación con el EEG la fatiga general y la cognitiva del PedsQL? | por pasos | [`README`](analisis/fatiga_fg_fc/README.md) |
| `tiempo_en_tarea/` | ¿Cambia el EEG a lo largo de una misma tarea, en personas no vistas? | por pasos | [`README`](analisis/tiempo_en_tarea/README.md) |

Son independientes y pueden ejecutarse en cualquier orden. Los de reposo y tareas aceptan `--rapido` (parámetros mínimos, salida temporal; sus cifras no son resultados). Todos usan semilla fija (`SEMILLA = 42`). Detalle de cada uno, requisitos y garantías metodológicas: [`analisis/README.md`](analisis/README.md).

> **El conjunto de test no se reejecuta.** `analisis_test.py --confirmatorio` se ejecutó una sola vez el 2026-10-07, según [`PREREGISTRO_TEST.md`](PREREGISTRO_TEST.md). Los demás guiones trabajan solo con `train` + `val`.

---

## Dónde están los resultados

| Qué | Dónde |
|---|---|
| Cifras (JSON) | `<EEG_DATA>/processed/*.json` |
| Resultado del test | `resultados/test_20261007-133358/` (`resultados.json` y `ejecucion.log`, copias de `<EEG_DATA>/processed/test_20261007-133358/`) y la sección final de `docs/MODELADO.md` |
| Interpretación | `docs/MODELADO.md` y `docs/analisis/` |
| Figuras | `resultados/figuras/` |
| Registros de ejecución | `resultados/logs/` |
| Fases retiradas (10 y 15) | `docs/FASES.md` y `archivo/` |

---

## Uso desde EEG-Platform

[EEG-Platform](https://github.com/Miguelit011/EEG-Platform) es la aplicación web local que permite a los investigadores procesar sus grabaciones sin programar. No copia código: instala esta librería en modo editable (`pip install -e ../EEG-Library`), llama a sus funciones y muestra lo que devuelven. Qué funciones usa y qué devuelve cada una está fijado en [`docs/CONTRATO.md`](docs/CONTRATO.md).

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
