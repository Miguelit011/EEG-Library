# Scripts de análisis

Guiones reproducibles que generan **las cifras citadas en la documentación del
proyecto y en la memoria del TFG**.

No son parte de la librería: la librería (`src/eeg_library/`) contiene las
reglas de evaluación y los modelos; estos guiones las orquestan sobre el
dataset real y producen los resultados. La distinción importa:

| | `probar.py`, `probar_ica.py` (raíz) | `analisis/*.py` |
|---|---|---|
| Qué son | Exploración manual | Registro reproducible de resultados |
| Si cambian | No pasa nada | Cambian cifras ya citadas |

**Por eso estos guiones no deben modificarse a la ligera.** Si un resultado
publicado se genera aquí, cualquier cambio invalida la correspondencia entre el
código y lo escrito. Los cambios van acompañados de una nueva ejecución y de la
actualización de la documentación que los cita.

---

## Requisitos

```bash
pip install -e ".[dev]"
```

El dataset **no está en el repositorio** (son 16,2 GB, principalmente las
ventanas crudas para la CNN, y contiene datos personales pseudonimizados). Los
guiones lo localizan mediante la variable de entorno `EEG_DATA`:

| | |
|---|---|
| Valor por defecto | `C:\Users\migue\Desktop\TFG\EEG-Data` |
| Estructura esperada | `<EEG_DATA>/processed/matriz_rf.parquet` |
| Salidas | `<EEG_DATA>/processed/*.json` |

Si el dataset está en la ruta por defecto no hace falta configurar nada:

```bash
# PowerShell, solo si la ruta es distinta
$env:EEG_DATA = "D:\otra\ruta"
```

```bash
# bash
export EEG_DATA=/ruta/al/dataset
```

---

## Guiones

Se ejecutan **desde la raíz del repositorio**.

### `analisis_fase14.py` — Regresión sobre el VAS

```bash
python analisis/analisis_fase14.py > analisis/salidas/fase14.log 2>&1
```

Entrena y evalúa el Random Forest como problema de regresión sobre la escala
0–100 del VAS. Ocho secciones: datos, líneas base, control de identidad,
Tarea A (participante nuevo), Tarea B (participante conocido), selección de
variables anidada, importancias y bootstrap.

- **Salida:** `<EEG_DATA>/processed/fase14_resultados.json`
- **Registro de ejecución:** `analisis/salidas/fase14.log`
- **Duración:** ~4 min

### `analisis_fase14b_clases.py` — Clasificación ordinal en tres niveles

```bash
python analisis/analisis_fase14b_clases.py > analisis/salidas/fase14b_clases.log 2>&1
```

Reformula el problema como clasificación en tres niveles (poca / media / mucha
fatiga) y determina si aporta sobre la regresión. Cuatro secciones: datos,
Tarea A, Tarea B y el subconjunto de variables `_rel_sd`.

- **Salida:** `<EEG_DATA>/processed/fase14b_clases.json`
- **Registro de ejecución:** `analisis/salidas/fase14b_clases.log`
- **Duración:** ~6 min

### `analisis_regresion_lineal.py` — VAS ~ índice theta/alfa

```bash
python analisis/analisis_regresion_lineal.py
```

Regresión lineal simple entre el VAS y el biomarcador de referencia de la
literatura, con su gráfica. No entrena ningún modelo: es el análisis descriptivo
de la Fase 9 que la memoria tiene que reportar. Siete secciones: datos, líneas
base, regresión, comprobaciones de forma (log₁₀ y recorte de colas), regresión
intrapersona, figuras y veredicto frente al óptimo declarado *antes* de ejecutar.

- **Salida:** `<EEG_DATA>/processed/regresion_lineal.json`,
  `figures/regresion_vas_theta_alfa*.png`
- **Duración:** ~20 s
- **Documentado en:** `REGRESION_LINEAL.md`

### `analisis_pca_clustering.py` — PCA + clustering de los grupos extremos

```bash
python analisis/analisis_pca_clustering.py
```

Exploratorio **no supervisado**: ¿se separan solas las grabaciones de sesiones
con mucha fatiga (VAS ≥ 80) de las de poca (VAS ≤ 20) en el espacio de features?
Trabaja **por grabación**, centrando cada feature por su condición para que la
tarea no domine el clustering. Siete secciones: datos, confusión entre identidad
y grupo, análisis principal, referencia sin ajustar, robustez en reposo, figuras
y resumen.

- **Salida:** `<EEG_DATA>/processed/pca_clustering.json`,
  `figures/pca_clustering_vas_extremos*.png`
- **Duración:** ~40 s
- **Documentado en:** `MODELADO.md`, Fase 16

### `analisis_oa_oc.py` — Ojos abiertos frente a ojos cerrados

```bash
python analisis/analisis_oa_oc.py
```

Control positivo del pipeline: ¿conservan las features de la Fase 13 el efecto
Berger? Clasificación binaria OA/OC por **grabación** (la condición es una
propiedad de cada registro, no de la sesión) sobre las 1072 grabaciones de reposo
de train + val, con validación cruzada agrupada por `persona_uid`. Compara
regresión logística y Random Forest contra dos baselines triviales y una
referencia de una sola variable, y comprueba si lo que pesa en el modelo es
neurofisiológicamente coherente (alfa posterior). Diez secciones: datos,
baselines, regresión logística, Random Forest, referencia de una variable,
coherencia neurofisiológica, controles y sensibilidad, resolución por canal,
figuras y resumen.

- **Salida:** `<EEG_DATA>/processed/oa_oc.json`, `figures/oa_oc_*.png`
- **Duración:** ~7.5 min
- **Requiere además** (solo el suplemento por canal):
  `<EEG_DATA>/processed/matriz_modelado.parquet` y
  `<EEG_DATA>/metadata/labels_grabaciones.csv`
- **Verificado en:** `tests/test_reposo.py` (78 pruebas sobre
  `src/eeg_library/modelado/reposo.py`)
- **Documentado en:** `ANALISIS_OA_OC.md` y `MODELADO.md`, Fase 17

### `analisis_reposo_tarea.py` — Reposo frente a tarea ejecutiva

```bash
python analisis/analisis_reposo_tarea.py
```

Clasificación binaria de las grabaciones de reposo (OA y OC) frente a las de tarea
ejecutiva (STROOP, PASAT, PVTB, MSIT, PVSAT): ¿conservan las features de la Fase 13
una señal de carga cognitiva? Por **grabación** sobre las 3728 de train + val, con
validación cruzada agrupada por `persona_uid` y las clases desbalanceadas (2.48 : 1)
tratadas con `class_weight="balanced"` y la exactitud balanceada como métrica
principal. Reutiliza el módulo y las exclusiones de X del análisis OA/OC. Compara
regresión logística y Random Forest contra dos baselines triviales y una referencia
de una sola variable (theta frontal), comprueba la coherencia neurofisiológica (theta
frontal hacia tarea, alfa posterior hacia reposo), desglosa los errores por condición
original y mide dos amenazas que el diseño no puede eliminar: OC dentro del reposo y
el orden fijo del protocolo. Once secciones: datos, baselines, regresión logística,
Random Forest, referencia de una variable, coherencia neurofisiológica, errores por
condición, controles y sensibilidad, resolución por canal, figuras y resumen.

- **Salida:** `<EEG_DATA>/processed/reposo_tarea.json`, `figures/reposo_tarea_*.png`
- **Duración:** ~15 min. `--rapido` ejecuta el mismo código con parámetros mínimos
  (~40 s) y escribe en un directorio temporal: **sus cifras no son resultados**
- **Requiere además** (solo el suplemento por canal):
  `<EEG_DATA>/processed/matriz_modelado.parquet` y
  `<EEG_DATA>/metadata/labels_grabaciones.csv`
- **Verificado en:** `tests/test_reposo_tarea.py` (73 pruebas sobre
  `src/eeg_library/modelado/reposo_tarea.py`)
- **Documentado en:** `ANALISIS_REPOSO_TAREA.md` y `MODELADO.md`, Fase 18

### `analisis_tareas_ejecutivas.py` — Las cinco tareas ejecutivas entre sí

```bash
python analisis/analisis_tareas_ejecutivas.py
```

Clasificación multiclase (5 clases) de las tareas ejecutivas del protocolo
(STROOP, PASAT, PVTB, MSIT, PVSAT; OA y OC quedan fuera): ¿distinguen las
features de la Fase 13 entre tareas, y qué tareas se confunden entre sí? Por
**grabación** sobre las 2656 de train + val, casi perfectamente equilibradas
entre clases (azar = 0.200), con validación cruzada agrupada por
`persona_uid`. Reutiliza el módulo y las exclusiones de X de OA/OC y
reposo/tarea. Compara regresión logística multinomial y Random Forest contra
tres baselines triviales, comprueba 5 criterios de coherencia neurofisiológica
declarados antes de ejecutar (incluida la dirección del efecto por clase con
coeficientes multinomiales), analiza qué pares de tareas se confunden y si es
por parecido cognitivo, distancia en el protocolo o duración de la grabación
(con un suplemento de 10 clasificadores binarios uno-contra-uno), y audita
ese último confusor directamente: los 6 contadores de procesamiento, solos,
igualan al modelo espectral completo. Once secciones: datos, baselines,
regresión logística, Random Forest, métricas por clase y matriz de confusión,
coherencia neurofisiológica, qué tareas se confunden, controles y
sensibilidad, resolución por canal, figuras y resumen.

- **Salida:** `<EEG_DATA>/processed/tareas_ejecutivas.json`,
  `figures/tareas_ejecutivas_*.png`
- **Duración:** ~26 min. `--rapido` ejecuta el mismo código con parámetros
  mínimos (~1 min) y escribe en un directorio temporal: **sus cifras de
  bootstrap/permutación no son resultados**
- **Requiere además** (solo el suplemento por canal):
  `<EEG_DATA>/processed/matriz_modelado.parquet` y
  `<EEG_DATA>/metadata/labels_grabaciones.csv`
- **Verificado en:** `tests/test_tareas_ejecutivas.py` (77 pruebas sobre
  `src/eeg_library/modelado/tareas_ejecutivas.py`)
- **Documentado en:** `ANALISIS_TAREAS_EJECUTIVAS.md` y `MODELADO.md`, Fase 19

Son independientes entre sí y pueden ejecutarse en cualquier orden.

---

## Garantías metodológicas

Estos guiones respetan los invariantes definidos y verificados en
`src/eeg_library/modelado/modeling.py` (41 pruebas en `tests/test_modeling.py`):

- La unidad de evaluación es la **sesión**, no la grabación ni la ventana:
  hay 653 etiquetas independientes, no 274 718.
- La partición es por **`persona_uid`**, nunca por sesión — estas variables
  identifican al participante 33 veces por encima del azar.
- Toda media de referencia por persona se estima **leave-one-out**.
- Toda transformación dependiente de los datos (imputación, tipificación,
  selección de variables, puntos de corte) se ajusta **solo con la partición de
  entrenamiento**.
- Los intervalos de confianza se obtienen remuestreando **personas**, no
  sesiones.

> ### El conjunto de test
>
> **Ninguno de estos guiones toca el conjunto de test.** Todos lo apartan
> explícitamente en su primer paso y operan solo sobre `train` + `val`.
>
> El test (106 sesiones, 28 participantes) se reserva para un único análisis
> confirmatorio, **pre-registrado por escrito antes de ejecutarse**. Añadir aquí
> código que lea el split `test` invalidaría ese diseño.

---

## Semilla y reproducibilidad

Semilla fija (`SEMILLA = 42`) en todos los procedimientos estocásticos. Dos
ejecuciones sobre la misma matriz producen cifras idénticas.

Versiones con las que se generaron los resultados documentados: Python 3.10.11,
scikit-learn 1.7.2, numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, matplotlib 3.10.9 y
mne 1.12.1. Es el entorno de `.venv` (fuera del repositorio); los resultados se
ejecutan siempre con su intérprete, `.venv/Scripts/python.exe`.

**Reproducción comprobada el 2026-10-06** en otra máquina, regenerando desde
cero las etiquetas, `matriz_rf.parquet` y los siete guiones con estas
versiones: `matriz_rf.parquet` es idéntica valor a valor a la original, los
logs coinciden línea a línea salvo las rutas, las figuras son idénticas byte a
byte y `fase14b_clases.json` no tiene ninguna diferencia. La única discrepancia
es una cifra no citada de `fase14_resultados.json` (el ρ de Spearman del RF con
hoja≥1 en la Tarea B: 0.107994 frente a 0.107980, con R² idéntico), atribuible
al orden de los empates en otra CPU.
