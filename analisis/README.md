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
| Valor por defecto | `C:\EEG-Data` |
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
python analisis/analisis_fase14.py
```

Entrena y evalúa el Random Forest como problema de regresión sobre la escala
0–100 del VAS. Ocho secciones: datos, líneas base, control de identidad,
Tarea A (participante nuevo), Tarea B (participante conocido), selección de
variables anidada, importancias y bootstrap.

- **Salida:** `<EEG_DATA>/processed/fase14_resultados.json`
- **Duración:** ~4 min

### `analisis_fase14b_clases.py` — Clasificación ordinal en tres niveles

```bash
python analisis/analisis_fase14b_clases.py
```

Reformula el problema como clasificación en tres niveles (poca / media / mucha
fatiga) y determina si aporta sobre la regresión. Cuatro secciones: datos,
Tarea A, Tarea B y el subconjunto de variables `_rel_sd`.

- **Salida:** `<EEG_DATA>/processed/fase14b_clases.json`
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

Son independientes entre sí y pueden ejecutarse en cualquier orden.

---

## Garantías metodológicas

Ambos guiones respetan los invariantes definidos y verificados en
`src/eeg_library/modelado/modeling.py` (40 pruebas en `tests/test_modeling.py`):

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
> **Ninguno de estos guiones toca el conjunto de test.** Ambos lo apartan
> explícitamente en su primer paso y operan solo sobre `train` + `val`.
>
> El test (106 sesiones, 28 participantes) se reserva para un único análisis
> confirmatorio, **pre-registrado por escrito antes de ejecutarse**. Añadir aquí
> código que lea el split `test` invalidaría ese diseño.

---

## Semilla y reproducibilidad

Semilla fija (`SEMILLA = 42`) en todos los procedimientos estocásticos. Dos
ejecuciones sobre la misma matriz producen cifras idénticas.

Versiones con las que se generaron los resultados documentados: Python 3.11,
scikit-learn 1.7.2, numpy 2.2.6, pandas 2.3.3, scipy 1.15.3.
