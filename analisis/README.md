# Scripts de análisis

Guiones reproducibles que generan **las cifras citadas en la documentación del
proyecto y en la memoria del TFG**.

No son parte de la librería: la librería (`src/eeg_library/`) contiene las
reglas de evaluación y los modelos; estos guiones las orquestan sobre el
dataset real y producen los resultados. La distinción importa: los guiones son el registro reproducible de los resultados, y si cambian, cambian cifras ya citadas.

**Por eso estos guiones no deben modificarse a la ligera.** Si un resultado
publicado se genera aquí, cualquier cambio invalida la correspondencia entre el
código y lo escrito. Los cambios van acompañados de una nueva ejecución y de la
actualización de la documentación que los cita.

---

## Requisitos

```bash
pip install -r requirements-validacion.txt
```

Instala la librería con los extras `[dev,analisis]` y las versiones exactas con
que se generaron los resultados. Los guiones de PyTorch de
`tiempo_en_tarea/deep_learning/` usan un entorno aparte: `requirements-dl.txt`.

El dataset **no está en el repositorio** (son 36 GB, principalmente los EDF y las
ventanas crudas para la CNN, y contiene datos personales pseudonimizados). Los
guiones lo localizan mediante la variable de entorno `EEG_DATA`:

| | |
|---|---|
| Valor por defecto | la carpeta `EEG-Data` junto al repositorio (`TFG/EEG-Data`) |
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

`analisis_test.py` y `tiempo_en_tarea/deep_learning/04_test_eegnet/entrenar.py`
conservan como valor por defecto la ruta del equipo de desarrollo
(`C:\Users\migue\Desktop\TFG\EEG-Data`): tienen su SHA-256 registrado y no se
editan. En otro equipo, para ejecutarlos hay que definir `EEG_DATA`.

---

## Líneas de análisis en carpeta

Los análisis posteriores a la Fase 21 viven cada uno en su carpeta. Cada carpeta
tiene un `README.md` con sus reglas y una subcarpeta por paso, con su script,
sus tablas, sus figuras y un `INFORME.md`.

| Carpeta | Pregunta | Estado |
|---|---|---|
| [`normalizacion_vas/`](normalizacion_vas/README.md) | ¿Da una etiqueta más útil el VAS expresado respecto a cada persona? (propuesta del tutor) | Cerrada el 2026-10-21: no mejora. Solo `train` + `val` |
| [`fatiga_fg_fc/`](fatiga_fg_fc/README.md) | ¿Guardan relación con el EEG la fatiga general y la cognitiva del PedsQL? (petición de los tutores) | Cerrada el 2026-10-22: nada robusto; la relación de la fatiga general dentro de persona desaparece al controlar la visita. Solo `train` + `val` |
| [`tiempo_en_tarea/`](tiempo_en_tarea/README.md) | ¿Cambia el EEG de forma sistemática a lo largo de una misma tarea, en personas no vistas? Incluye `deep_learning/` (EEGNet y ShallowConvNet) | Cerrada: evaluada en test con pre-registro el 2026-10-27 (logística) y el 2026-10-31 (EEGNet). Hasta el 2026-11-05 se llamó `fatiga_aguda/` (ver su README) |

---

## Guiones

Se ejecutan **desde la raíz del repositorio**. Los logs van a `resultados/logs/` y las figuras a `resultados/figuras/`.

### `analisis_fase14.py` — Regresión sobre el VAS

```bash
python analisis/analisis_fase14.py > resultados/logs/fase14.log 2>&1
```

Entrena y evalúa el Random Forest como problema de regresión sobre la escala
0–100 del VAS. Ocho secciones: datos, líneas base, control de identidad,
Tarea A (participante nuevo), Tarea B (participante conocido), selección de
variables anidada, importancias y bootstrap.

- **Salida:** `<EEG_DATA>/processed/fase14_resultados.json`
- **Registro de ejecución:** `resultados/logs/fase14.log`
- **Duración:** ~4 min

### `analisis_fase14b_clases.py` — Clasificación ordinal en tres niveles

```bash
python analisis/analisis_fase14b_clases.py > resultados/logs/fase14b_clases.log 2>&1
```

Reformula el problema como clasificación en tres niveles (poca / media / mucha
fatiga) y determina si aporta sobre la regresión. Cuatro secciones: datos,
Tarea A, Tarea B y el subconjunto de variables `_rel_sd`.

- **Salida:** `<EEG_DATA>/processed/fase14b_clases.json`
- **Registro de ejecución:** `resultados/logs/fase14b_clases.log`
- **Duración:** ~6 min

### `analisis_control_visita.py` — Control de la deriva por número de visita

```bash
python analisis/analisis_control_visita.py > resultados/logs/control_visita.log 2>&1
```

Recalcula, con y sin una recta de visita (M2…M6) ajustada dentro de cada
persona, las tres cifras intra-persona del VAS que entran en la familia
confirmatoria del test: la prueba principal (Fase 14b, Tarea B), la pendiente
intrapersona VAS ~ theta/alfa y el compuesto `_rel_sd`. Las cifras sin control
reproducen las ya documentadas.

- **Salida:** `<EEG_DATA>/processed/control_visita.json`
- **Registro de ejecución:** `resultados/logs/control_visita.log`
- **Duración:** ~1 min
- **Documentado en:** `docs/MODELADO.md`, "Control de la deriva por número de visita"

### `analisis_test.py` — Análisis confirmatorio sobre el test

```bash
python analisis/analisis_test.py --ensayo         # train -> val, ensayo general
python analisis/analisis_test.py --confirmatorio  # train+val -> test, UNA sola vez
```

Ejecuta exactamente lo que fija `PREREGISTRO_TEST.md`: ajusta los modelos
finales con el conjunto de entrenamiento, predice una sola vez el evaluado y
calcula la familia VAS (prueba principal con control de visita, 5 secundarias
con Holm y lo descriptivo) y la familia de condición (OA/OC, reposo/tarea y
las cinco tareas, con Holm y los criterios de coherencia). Guarda los datos en
bruto (predicciones, matrices evaluadas, modelos, parámetros, réplicas
bootstrap y de permutación, importancias, inventario) y los metadatos (commit,
hashes, entorno, CPU).

- **`--ensayo`**: entrena con `train` y evalúa `val`. Sus cifras **no son
  resultados** (`val` ya se usó en la validación cruzada); solo comprueba que
  el guion corre y escribe todos los artefactos.
- **`--confirmatorio`**: entrena con `train` + `val` y evalúa `test`. Se niega
  si el árbol de git no está limpio o si ya existe `processed/test_*`.
- **`--rapido`** (con `--ensayo`): réplicas mínimas y salida a un directorio
  temporal, para depurar.
- **Salida:** `<EEG_DATA>/processed/ensayo_val_<fecha>/` o
  `<EEG_DATA>/processed/test_<fecha>/`, con su `ejecucion.log`
- **Verificado en:** `tests/modelado/test_confirmatorio.py` (piezas de
  `src/eeg_library/modelado/confirmatorio.py`)
- **Documentado en:** `PREREGISTRO_TEST.md`

### `analisis_potencia_test.py` — Potencia de la prueba principal en test

```bash
python analisis/analisis_potencia_test.py > resultados/logs/potencia_test.log 2>&1
```

Simula 5000 tests de 28 personas remuestreando las predicciones out-of-fold de
la prueba principal (con y sin control de visita) y aplica la regla
pre-registrada (p unilateral por permutación dentro de persona). Incluye un
control de error de tipo I con predicciones barajadas.

- **Salida:** `<EEG_DATA>/processed/potencia_test.json`
- **Registro de ejecución:** `resultados/logs/potencia_test.log`
- **Duración:** ~1.5 min
- **Documentado en:** `docs/MODELADO.md`, "Potencia de la prueba principal en test"

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
  `resultados/figuras/regresion_vas_theta_alfa*.png`
- **Duración:** ~20 s
- **Documentado en:** `docs/analisis/ANALISIS_REGRESION_LINEAL.md`

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
  `resultados/figuras/pca_clustering_vas_extremos*.png`
- **Duración:** ~40 s
- **Documentado en:** `docs/MODELADO.md`, Fase 16

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

- **Salida:** `<EEG_DATA>/processed/oa_oc.json`, `resultados/figuras/oa_oc_*.png`
- **Duración:** ~7.5 min
- **Requiere además** (solo el suplemento por canal):
  `<EEG_DATA>/processed/matriz_modelado.parquet` y
  `<EEG_DATA>/metadata/labels_grabaciones.csv`
- **Verificado en:** `tests/modelado/test_reposo.py` (78 pruebas sobre
  `src/eeg_library/modelado/reposo.py`)
- **Documentado en:** `docs/analisis/ANALISIS_OA_OC.md` y `docs/MODELADO.md`, Fase 17

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

- **Salida:** `<EEG_DATA>/processed/reposo_tarea.json`, `resultados/figuras/reposo_tarea_*.png`
- **Duración:** ~15 min. `--rapido` ejecuta el mismo código con parámetros mínimos
  (~40 s) y escribe en un directorio temporal: **sus cifras no son resultados**
- **Requiere además** (solo el suplemento por canal):
  `<EEG_DATA>/processed/matriz_modelado.parquet` y
  `<EEG_DATA>/metadata/labels_grabaciones.csv`
- **Verificado en:** `tests/modelado/test_reposo_tarea.py` (73 pruebas sobre
  `src/eeg_library/modelado/reposo_tarea.py`)
- **Documentado en:** `docs/analisis/ANALISIS_REPOSO_TAREA.md` y `docs/MODELADO.md`, Fase 18

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
  `resultados/figuras/tareas_ejecutivas_*.png`
- **Duración:** ~26 min. `--rapido` ejecuta el mismo código con parámetros
  mínimos (~1 min) y escribe en un directorio temporal: **sus cifras de
  bootstrap/permutación no son resultados**
- **Requiere además** (solo el suplemento por canal):
  `<EEG_DATA>/processed/matriz_modelado.parquet` y
  `<EEG_DATA>/metadata/labels_grabaciones.csv`
- **Verificado en:** `tests/modelado/test_tareas_ejecutivas.py` (77 pruebas sobre
  `src/eeg_library/modelado/tareas_ejecutivas.py`)
- **Documentado en:** `docs/analisis/ANALISIS_TAREAS_EJECUTIVAS.md` y `docs/MODELADO.md`, Fase 19

Son independientes entre sí y pueden ejecutarse en cualquier orden.

### Guiones de empaquetado (plataforma web)

No son análisis: preparan lo que la librería entrega a EEG-Platform. El
contrato está en `docs/CONTRATO.md`.

```bash
python analisis/empaquetar_modelos.py   # modelos c1-c3 + fichas -> src/eeg_library/modelos/
python analisis/empaquetar_modelos_tiempo_en_tarea.py   # logística PASAT y PVTB -> modelos/tiempo_en_tarea/
python analisis/generar_ejemplo.py      # JSON de ejemplo -> docs/ejemplos/
```

- `empaquetar_modelos.py` copia sin modificar los `.joblib` de la regresión
  logística del análisis confirmatorio
  (`<EEG_DATA>/processed/test_20261007-133358/modelos/`) y escribe su ficha
  con las cifras de `resultados.json`. Exige que el SHA-256 de
  `matriz_rf.parquet` sea el registrado en ese análisis. Es idempotente.
- `empaquetar_modelos_tiempo_en_tarea.py` copia sin modificar los `.joblib` de
  la logística de tiempo en tarea
  (`<EEG_DATA>/processed/fatiga_aguda_test_20261027-190547/modelos/`; la carpeta
  conserva el nombre antiguo) y escribe sus fichas `tt_*.json` con las cifras de
  `tiempo_en_tarea/06_test/resultados_test.json`. Los modelos de EEGNet (`.onnx`
  y `tt_eegnet.json`) los escribe `tiempo_en_tarea/deep_learning/05_onnx/exportar.py`.
- `generar_ejemplo.py` analiza una grabación de train y anonimiza su nombre.
- **Verificado en:** `tests/modelado/test_prediccion.py`,
  `tests/procesado/test_grabacion.py` y `tests/datos/test_fase0_datos.py`
  (este último con `pytest -m datos`: matriz idéntica, paridad y clases).

---

## Garantías metodológicas

Estos guiones respetan los invariantes definidos y verificados en
`src/eeg_library/modelado/modeling.py` (49 pruebas en `tests/modelado/test_modeling.py`):

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
> **Ninguno de estos guiones toca el conjunto de test, salvo
> `analisis_test.py --confirmatorio`.** Todos los demás lo apartan
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
