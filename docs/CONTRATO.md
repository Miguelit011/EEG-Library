# Contrato con EEG-Platform

La plataforma web no calcula nada: llama a funciones de la librería, guarda lo
que devuelven y lo muestra. Este documento fija qué devuelven. Versión de la
librería: **0.17.0** (0.15.0 añadió `por_ventana` y los modelos de tiempo en tarea; 0.16.0, EEGNet;
0.17.0 renombró «fatiga aguda» a «tiempo en tarea»: módulos `modelado/tiempo_en_tarea*.py`,
funciones `*_tiempo_en_tarea`, carpeta `modelos/tiempo_en_tarea/` e identificadores `tt_*`, sin
cambiar ningún cálculo, peso ni cifra).

```python
from eeg_library import (analizar_grabacion, listar_modelos, cargar_modelo, predecir,
                         puntuar_tiempo_en_tarea)

resultado = analizar_grabacion(ruta_edf, parametros)        # dict -> JSON
for id_modelo in listar_modelos():
    p = predecir(id_modelo, resultado["caracteristicas"],
                 resultado["parametros_aplicados"])

tt = puntuar_tiempo_en_tarea(resultado, "PVTB")              # solo PASAT y PVTB
```

Hay dos ejemplos reales (anonimizados) en `docs/ejemplos/`, generados con
`analisis/generar_ejemplo.py`: `analisis_grabacion.json` (ojos abiertos) y
`analisis_grabacion_pasat.json` (PASAT, para la pestaña Tiempo en tarea; elegido
para que el cambio se vea, no representativo).

## Parámetros de procesado

Son opciones cerradas (`eeg_library.procesado.parametros`):

| Clave | Opciones | Estudio |
|---|---|---|
| `referencia` | `car_condicional`, `original` | `car_condicional` |
| `banda_hz` | `[1, 30]`, `[3, 30]` | `[1, 30]` |
| `notch_hz` | `50`, `None` (sin notch) | `50` |

- `car_condicional`: CAR sobre los canales buenos si hay al menos 6; con menos
  se omite. ICA usa `n_buenos − 1` componentes en ambos casos, como en el
  estudio.
- `original`: se conserva la referencia de hardware del Emotiv (CMS/DRL). Sin
  CAR no hay pérdida de rango, así que ICA usa `n_buenos` componentes.
- Las claves que faltan toman el valor del estudio. Un valor fuera de las
  opciones lanza `ValueError`.
- Con la banda 3–30 Hz la potencia delta (1–4 Hz) queda casi eliminada por el
  propio filtro. Es lo esperable, pero conviene avisarlo en la interfaz.

El recorte (5 s + 5 s), ICA con ICLabel, el epocado (4 s, solape 2 s) y Welch
(1–30 Hz) no son elegibles.

## `analizar_grabacion(ruta_edf, parametros=None)`

Devuelve un `dict` serializable con `json.dumps(..., allow_nan=False)`: los NaN
salen como `null`. Si falla una fase del pipeline lanza `ErrorDeFase`, con la
fase en `.fase`.

| Sección | Contenido |
|---|---|
| `metadatos` | `archivo`, `duracion_s` (antes del recorte), `sfreq_original_hz`, `sfreq_hz`, `resampleado`, `canales_eeg`, `n_canales_buenos`, `n_epocas`, `versiones` (eeg_library, python, mne, scikit_learn, numpy) y `fecha_analisis` |
| `parametros_aplicados` | `referencia`, `banda_hz`, `notch_hz` (los tres elegibles, tal como se aplicaron), `car_aplicada` (bool), `recorte_s`, `ica` (`n_componentes`, `componentes_eliminadas`, `criterio`, `varianza_eliminada`, `random_state`), `epocado` y `psd` |
| `calidad` | Por canal: `region`, `estado` (`bueno` o `descartado`; la librería no tiene estado intermedio), `motivo` (texto de `evaluar_calidad`, `null` si es bueno) y `metricas` (`cq_medio`, `pct_muestras_malas`, `varianza`, `z_robusto`) |
| `potencia` | `bandas_hz`; `por_canal[canal]`: `valido` y, por banda, `absoluta_v2` y `relativa` (media entre épocas); `por_region[region]`: `n_canales_validos` y, por banda, `log10_absoluta_v2` y `relativa`, solo con canales válidos (el mismo agregado que la matriz) |
| `caracteristicas` | Las 70 entradas de los modelos, por nombre |
| `por_ventana` | `instante_s` (inicio de cada ventana de 4 s, en segundos desde el registro recortado; paso de 2 s), `canal_valido[canal]` (bool, la misma máscara que `potencia.por_canal[canal].valido`), `pot_rel[canal][banda]`: una lista con la potencia relativa de cada ventana, sin redondear, y `eegnet` (desde 0.16.0): la puntuación de EEGNet de cada ventana, media de los logits de las tres semillas, o `null` si los parámetros no son los del estudio o EEGNet falló |

Notas:

- La potencia de un canal descartado se calcula igual, pero sobre señal que no
  pasó por CAR ni por ICA: `valido = false` y no entra en las regiones.
- Las características salen del **mismo código** que construyó
  `matriz_rf.parquet` (`ejecutar_fases` → `anotar_validez_de_canales` →
  `resumir_por_grabacion`). Ninguna depende de otras grabaciones.
- `por_ventana` ocupa unos 100–200 KB en las grabaciones más largas. Es la
  entrada de los modelos de tiempo en tarea; las grabaciones analizadas con 0.14.0
  o antes no lo tienen.
- Alias neutros: `indice_fatiga_media` y `indice_fatiga_sd` de la matriz salen
  como `indice_theta_alfa_media` e `indice_theta_alfa_sd`. `predecir` los
  traduce de vuelta. Ninguna clave del JSON contiene «fatiga».

## Modelos y fichas

Cada modelo es un `.joblib` (Pipeline de scikit-learn 1.7.2) con una ficha
`.json` del mismo nombre. Los del estudio están en `src/eeg_library/modelos/` y
los genera `analisis/empaquetar_modelos.py`.

| Clave de la ficha | Contenido |
|---|---|
| `id`, `titulo`, `archivo`, `sha256_modelo` | Identidad. La carga falla si el SHA-256 del `.joblib` no coincide |
| `que_predice` | Texto para la interfaz |
| `clases` | `[{codigo, nombre}]`, en el orden de `predict_proba` |
| `condiciones_aplicables` | Condiciones del protocolo para las que tiene sentido |
| `condiciones_por_clase` | Opcional. `{clase: [condiciones]}`: qué condiciones corresponden a cada clase (en c2, `reposo` = OA y OC). La plataforma lo usa para comparar la predicción con la condición conocida de la grabación; sin él, supone que la clase se llama como la condición |
| `parametros_validos` | `referencia`, `banda_hz` y `notch_hz` con que se entrenó |
| `columnas` | `feature_names_in_` del Pipeline, con los nombres de la matriz |
| `resultado_test` | `metrica`, `valor`, `ic95`, `azar`, `n_grabaciones`, `n_personas` y `fuente` |
| `entrenamiento` | `fecha`, `datos`, `variante`, `sha256_matriz_rf`, `commit_eeg_library`, `python` y `scikit_learn` |
| `avisos` | Textos que la interfaz muestra siempre |

| Modelo | Clases | Test (exactitud balanceada) | Azar |
|---|---|---|---|
| `c1_oa_oc_logistica` | OA, OC | 0.855 [0.794–0.912] | 0.5 |
| `c2_reposo_tarea_logistica` | reposo, tarea | 0.851 [0.813–0.885] | 0.5 |
| `c3_tareas_logistica` | STROOP, PASAT, PVTB, MSIT, PVSAT | 0.488 [0.441–0.531] | 0.2 |

Un modelo externo se carga con la ruta de su ficha: `cargar_modelo("ruta/x.json")`,
con el `.joblib` al lado.

## `predecir(modelo, caracteristicas, parametros)`

- `modelo`: un `Modelo` de `cargar_modelo`, o su identificador o ruta.
- `parametros`: se comparan solo las tres claves elegibles; las demás se ignoran,
  así que se puede pasar `parametros_aplicados` tal cual.
- Si coinciden con `parametros_validos`:
  `{"disponible": true, "modelo", "clase", "codigo", "probabilidades": {nombre: p}}`.
- Si no: `{"disponible": false, "modelo", "motivo", "parametros_requeridos",
  "parametros_aplicados"}`.
- Si falta una columna lanza `ValueError`. Un `null` lo imputa el Pipeline.

`predecir` no mira la condición de la grabación: elegir qué modelos mostrar
según la condición conocida o estimada es cosa de la plataforma (Fase 7).

## Tiempo en tarea: `puntuar_tiempo_en_tarea(resultado, condicion)`

Modelos de la línea `analisis/tiempo_en_tarea/` (evaluación única en test del
2026-10-27): una regresión logística por tarea, entrenada con `train` + `val`
para distinguir el primer tercio de la grabación del último. Miden el cambio
del EEG con el tiempo en tarea, no el nivel de fatiga de la persona. Viven en
`src/eeg_library/modelos/tiempo_en_tarea/` (los genera
`analisis/empaquetar_modelos_tiempo_en_tarea.py`) y no aparecen en
`listar_modelos()`: sus entradas son por ventana, no las 70 de la grabación.

| Modelo | Condición | Acierto pareado de tercios en test | ρ media en test | Azar |
|---|---|---|---|---|
| `tt_pasat_logistica` | PASAT | 0.735 [0.657–0.816], 102 grabaciones | +0.162 [+0.111, +0.217] | 0.5 / 0 |
| `tt_pvtb_logistica` | PVTB | 0.835 [0.757–0.909], 103 grabaciones | +0.180 [+0.141, +0.220] | 0.5 / 0 |

- **Ficha:** las claves de la de c1–c3 (con `clases` = primer tercio / último
  tercio y `condiciones_aplicables` = su tarea) más `canales` (orden de las 56
  entradas: canal × banda) y `resultado_test_rho`. `resultado_test` es el
  acierto pareado de tercios.
- **Devuelve**, si se puede: `{"disponible": true, "modelo", "condicion",
  "n_ventanas", "instante_s", "puntuacion", "rho", "tercios": {"p_primero",
  "p_ultimo", "final_mayor"}, "pesos": {canal: {banda: peso}}}`.
  - `puntuacion`: una por ventana (`decision_function`); positiva = se parece
    al último tercio. Es el cálculo de `rho_ventanas` del test.
  - `rho`: Spearman entre la puntuación y el instante de la ventana.
  - `tercios`: probabilidad de «último tercio» de la media del primer tercio y
    de la del último; `final_mayor` es `true` si el último puntúa más (`null`
    si empatan). Es el cálculo del acierto pareado del test.
  - `pesos`: coeficiente de la logística dividido por la escala del
    `StandardScaler`, por canal y banda. Descriptivo: las entradas están
    correlacionadas.
- **No disponible** (`{"disponible": false, "condicion", "modelo",
  "motivo"}`) si la condición no es PASAT ni PVTB, si los parámetros no son
  los del entrenamiento, si falta `por_ventana` (hay que reprocesar) o si hay
  menos de 10 ventanas.

## Tiempo en tarea con EEGNet: `puntuar_eegnet(resultado, condicion)` (desde 0.16.0)

La red neuronal de la línea (EEGNet, paso 4 de `deep_learning/`), exportada a
ONNX (`tt_eegnet_s0/1/2.onnx` y la ficha `tt_eegnet.json` en
`src/eeg_library/modelos/tiempo_en_tarea/`). Trabaja con la señal de cada ventana,
así que se aplica **al procesar** (`por_ventana["eegnet"]`); `puntuar_eegnet`
solo resume: `{"disponible": true, "modelo", "condicion", "n_ventanas",
"instante_s", "puntuacion", "rho", "tercios": {"media_primero", "media_ultimo",
"final_mayor"}}`, o «no disponible» con el motivo (otra condición, otros
parámetros, menos de 10 ventanas o grabación sin `eegnet`, que hay que
reprocesar).

| Modelo | Condición | Acierto pareado de tercios en test | ρ media en test | Frente a la logística |
|---|---|---|---|---|
| `tt_eegnet` | PASAT y PVTB | 0.737 [0.683–0.787] | +0.160 [+0.130, +0.190] | Δρ −0.011 [−0.047, +0.028]: no se distinguen |

En el test cada semilla se evaluó por separado y se promediaron las cifras; aquí
se combinan las tres en una puntuación por ventana (media de los logits). Se
declara en la ficha.

## Verificación

- `pytest`: tests unitarios del contrato con un EDF sintético
  (`tests/procesado/test_grabacion.py`, `tests/modelado/test_prediccion.py`).
- `pytest -m datos` (necesita `EEG_DATA`, ~1 min, solo train/val):
  - la matriz reconstruida es idéntica a `matriz_rf.parquet`
    (`check_exact=True`);
  - paridad exacta de las 70 características en 13 grabaciones (las 7
    condiciones, los exports de 59, 63, 67 y 71 canales, la grabación con CAR
    omitida y regiones a NaN);
  - con los nombres de clase de la ficha, la exactitud balanceada en train/val
    es 0.868 (c1), 0.871 (c2) y 0.512 (c3).
- `pytest -m datos tests/datos/test_tiempo_en_tarea_datos.py` (~2 min, solo
  train/val): en 20 grabaciones PASAT y PVTB (19 con algún canal excluido), la
  `pot_rel` por ventana coincide exactamente con `features_all.parquet` y la
  puntuación, la ρ y el veredicto de tercios coinciden con los del análisis
  (`medidas.ventanas`, `comun.tercios`; diferencia máxima 1.4·10⁻¹⁴).
- `pytest -m datos tests/datos/test_tiempo_en_tarea_eegnet_datos.py` (~2 min, solo
  train/val): en 20 grabaciones PASAT y PVTB, la entrada de EEGNet que prepara
  la librería y la puntuación que guarda `analizar_grabacion` son idénticas a
  las del análisis (diferencia 0). ONNX = PyTorch se comprobó al exportar
  (`analisis/tiempo_en_tarea/deep_learning/05_onnx/`).
