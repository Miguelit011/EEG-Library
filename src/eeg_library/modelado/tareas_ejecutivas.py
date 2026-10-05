"""
Clasificación multiclase de las cinco tareas ejecutivas del protocolo.

¿Con qué rendimiento identifica el pipeline cuál de las cinco tareas ejecutivas
(STROOP, PASAT, PVTB, MSIT, PVSAT) es una grabación, y qué tareas se confunden
entre sí? Es el escalón siguiente a `reposo_tarea.py`: aquella pregunta era «¿hay
carga cognitiva?» (binaria); esta es «¿de qué tipo?» (multiclase, 5 clases). OA y
OC quedan fuera —ya están cubiertas en `reposo.py` y `reposo_tarea.py`—, así que
este módulo no depende de ninguno de los dos, solo reutiliza su arnés.

Este módulo **no reimplementa la evaluación**. El arnés de `modeling.py` y
`reposo.py` (`GroupKFold` por persona, bootstrap por persona, importancia por
permutación out-of-fold, d de Cohen) es agnóstico al número de clases y se
reutiliza tal cual. Lo que sí es propio de este contraste, y vive aquí:

- **Qué grabaciones entran y cómo se codifica la etiqueta**
  (`preparar_tareas_ejecutivas`, `etiqueta_multiclase`).
- **Las métricas y la matriz de confusión multiclase** (`metricas_multiclase`,
  `matriz_confusion_normalizada`): con 5 clases casi equilibradas (ver el
  análisis de datos) el azar de la exactitud balanceada es 1/5 = 0.200, no 0.5.
- **Qué tareas se confunden entre sí** (`confusion_por_pares`,
  `separabilidad_por_pares`, `estructura_de_la_confusion`): el patrón de
  confusión, comparado con tres estructuras candidatas — parecido cognitivo
  declarado antes de ejecutar, distancia en el protocolo y distancia de
  duración de la grabación.
- **Los coeficientes de un modelo multinomial**
  (`coeficientes_por_clase_y_pliegue`, `coeficiente_neto_por_clase`): con más de
  dos clases `coef_` tiene una fila por clase (softmax, uno-contra-el-resto en
  la parametrización lineal), no un solo vector, así que
  `reposo.coeficientes_por_pliegue` (pensado para el caso binario) no sirve
  aquí.

Decisiones de diseño (justificación completa en ANALISIS_TAREAS_EJECUTIVAS.md):

- **X es la misma que en OA/OC y reposo/tarea**: `columnas_de_features_reposo`,
  que parte de `columnas_de_features()` (sin tocar) y le resta `condicion_orden`
  y los seis contadores de procesamiento. Aquí `condicion_orden` no separa un
  solo corte —hacen falta cuatro, uno entre cada par de tareas consecutivas—,
  pero sigue siendo la posición en el protocolo y su exclusión no es discutible.
- **El diseño tiene una disociación aprovechable**: los dos pares
  cognitivamente más parecidos (PASAT-PVSAT, STROOP-MSIT) están ambos a
  distancia 3 en el protocolo, mientras que los pares contiguos (PASAT-PVTB,
  PVTB-MSIT) son cognitivamente distintos. Similitud cognitiva y proximidad
  temporal apuntan en direcciones opuestas, así que el patrón de confusión
  puede discriminar entre ellas. La duración no se disocia igual de bien
  (STROOP y MSIT duran lo mismo; PASAT y PVSAT también), y por eso se audita
  aparte con `distancias_de_duracion`, medida sobre los propios datos.
- **El conjunto de test no se abre.** `preparar_tareas_ejecutivas` se niega a
  incluir `test` salvo que se pida con `permitir_test=True`.

Lo que este módulo **no** hace: no modifica el pipeline congelado, no recalcula
features y no toca `reposo.py` ni `reposo_tarea.py`. Todo sale de
`matriz_rf.parquet`.
"""

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_library.modelado.modeling import COLUMNA_GRUPO, predecir_cv_agrupada
from eeg_library.modelado.reposo import (
    COLUMNA_CONDICION,
    COLUMNA_SPLIT,
    SPLITS_ANALISIS,
    bootstrap_por_persona,
    columnas_de_features_reposo,
    evaluar_reposo,
    metricas_binarias,
    pliegue_por_fila,
    tamanos_de_efecto,
)
from eeg_library.modelado.reposo_tarea import columnas_media_del_bloque
from eeg_library.procesado.matrix import ORDEN_PROTOCOLO

CONDICIONES_EJECUTIVAS = ("STROOP", "PASAT", "PVTB", "MSIT", "PVSAT")
CODIGO_DE_CLASE = {c: i for i, c in enumerate(CONDICIONES_EJECUTIVAS)}
SEMILLA = 42

# Similitud cognitiva declarada a priori (3 alta, 2 media, 1 baja), simétrica y
# completa: los 10 pares de las 5 tareas. Se fija ANTES de ejecutar y es uno de
# los predictores que se compara con el patrón de confusión observado en
# `estructura_de_la_confusion`.
#   3: PASAT-PVSAT (la misma tarea de suma serial, auditiva frente a visual) y
#      STROOP-MSIT (las dos de interferencia/conflicto).
#   2: interferencia (STROOP, MSIT) frente a memoria de trabajo serial (PASAT,
#      PVSAT).
#   1: PVTB con cualquiera de las otras cuatro — es vigilancia sostenida, sin
#      carga de memoria de trabajo ni conflicto de respuesta, la más distinta.
SIMILITUD_COGNITIVA = {
    frozenset({"PASAT", "PVSAT"}): 3,
    frozenset({"STROOP", "MSIT"}): 3,
    frozenset({"STROOP", "PASAT"}): 2,
    frozenset({"STROOP", "PVSAT"}): 2,
    frozenset({"PASAT", "MSIT"}): 2,
    frozenset({"MSIT", "PVSAT"}): 2,
    frozenset({"STROOP", "PVTB"}): 1,
    frozenset({"PASAT", "PVTB"}): 1,
    frozenset({"PVTB", "MSIT"}): 1,
    frozenset({"PVTB", "PVSAT"}): 1,
}


def _validar_columnas(df, columnas, nombre):
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise ValueError(f"A '{nombre}' le faltan columnas necesarias: {faltan}.")


def _validar_longitudes(**arrays):
    longitudes = {k: len(v) for k, v in arrays.items()}
    if len(set(longitudes.values())) > 1:
        raise ValueError(
            "Los argumentos deben tener la misma longitud: "
            + ", ".join(f"{k}={n}" for k, n in longitudes.items()) + "."
        )


def similitud_cognitiva(a, b):
    """Similitud cognitiva a priori entre dos tareas ejecutivas (1 baja, 3 alta)."""
    if a == b:
        raise ValueError("Hay que comparar dos condiciones distintas.")
    try:
        return SIMILITUD_COGNITIVA[frozenset({a, b})]
    except KeyError:
        raise ValueError(f"No hay similitud cognitiva declarada entre '{a}' y '{b}'.")


# --------------------------------------------------------------------------
# Definición de X, preparación de los datos y la etiqueta
# --------------------------------------------------------------------------

def columnas_de_features_ejecutivas(matriz, incluir_procesamiento=False):
    """
    Definición de X para las tareas ejecutivas.

    Es exactamente la de OA/OC y reposo/tarea (`columnas_de_features_reposo`):
    resta `condicion_orden` **siempre** y los seis contadores de procesamiento
    salvo que se pidan con `incluir_procesamiento=True` (análisis de
    sensibilidad). Existe con nombre propio para que este guion no dependa de
    un nombre que habla de otro contraste; un test comprueba que coinciden.
    Sobre la matriz de la Fase 13 (77 features) devuelve 70.
    """
    return columnas_de_features_reposo(
        matriz, incluir_procesamiento=incluir_procesamiento)


def preparar_tareas_ejecutivas(matriz, splits=SPLITS_ANALISIS, permitir_test=False,
                               condiciones=CONDICIONES_EJECUTIVAS):
    """
    Deja la matriz de la Fase 13 lista para la clasificación multiclase.

    Se queda con las grabaciones de las `condiciones` pedidas (por defecto las
    cinco tareas ejecutivas; OA y OC quedan siempre fuera de este módulo) de
    los `splits` pedidos, y descarta las que no tienen `persona_uid` (sin él no
    se puede agrupar por persona).

    Parámetros
    ----------
    matriz : pandas.DataFrame
        `matriz_rf.parquet`, una fila por grabación.
    splits : iterable de str
        Particiones que entran al análisis. Por defecto `train` y `val`.
    permitir_test : bool
        `test` se reserva para un único análisis confirmatorio pre-registrado:
        pedirlo sin activar esto es un error, no un aviso.
    condiciones : iterable de str
        Por defecto las cinco tareas. Restringirlas da controles por
        subconjunto (p. ej. un solo par, para un contraste binario). Cada
        condición debe ser una tarea ejecutiva; no admite OA ni OC.

    Devuelve
    --------
    datos : pandas.DataFrame
        Grabaciones de análisis, con el índice reiniciado.
    metadata : dict
        Recuentos: grabaciones, personas, sesiones, por condición y por split,
        clase mayoritaria/minoritaria, desbalance, y lo que quedó apartado o
        descartado.
    """
    _validar_columnas(matriz, (COLUMNA_CONDICION, COLUMNA_SPLIT, COLUMNA_GRUPO),
                      "matriz")
    splits = tuple(splits)
    if "test" in splits and not permitir_test:
        raise ValueError(
            "El conjunto de test no se abre en este análisis. Si de verdad hace "
            "falta, hay que pedirlo con `permitir_test=True` y documentarlo."
        )
    condiciones = tuple(condiciones)
    if len(condiciones) < 2:
        raise ValueError("Hacen falta al menos 2 condiciones para clasificar.")
    ajenas = sorted(set(condiciones) - set(CONDICIONES_EJECUTIVAS))
    if ajenas:
        raise ValueError(
            f"'condiciones' solo admite tareas ejecutivas de "
            f"{list(CONDICIONES_EJECUTIVAS)}; llegaron {ajenas}."
        )

    elegidas = matriz[matriz[COLUMNA_CONDICION].isin(condiciones)]
    en_splits = elegidas[elegidas[COLUMNA_SPLIT].isin(splits)]
    apartadas = elegidas[~elegidas[COLUMNA_SPLIT].isin(splits)]
    sin_persona = en_splits[COLUMNA_GRUPO].isna()
    datos = en_splits[~sin_persona].reset_index(drop=True)

    recuentos = {k: int(v) for k, v in
                datos[COLUMNA_CONDICION].value_counts().items()}
    faltan_clases = set(condiciones) - set(recuentos)
    if faltan_clases:
        raise ValueError(
            f"Tras filtrar no quedan grabaciones de: {sorted(faltan_clases)}."
        )
    mayor = max(recuentos, key=recuentos.get)
    menor = min(recuentos, key=recuentos.get)

    metadata = {
        "splits": list(splits),
        "condiciones": list(condiciones),
        "n_clases": len(condiciones),
        "n_grabaciones": int(len(datos)),
        "n_personas": int(datos[COLUMNA_GRUPO].nunique()),
        "n_sesiones": (int(datos["sesion_uid"].nunique())
                       if "sesion_uid" in datos.columns else None),
        "por_condicion": recuentos,
        "clase_mayoritaria": mayor,
        "clase_minoritaria": menor,
        "ratio_desbalance": float(recuentos[mayor] / recuentos[menor]),
        "por_split": {k: int(v) for k, v in
                      datos[COLUMNA_SPLIT].value_counts().items()},
        "n_apartadas": int(len(apartadas)),
        "n_personas_apartadas": int(apartadas[COLUMNA_GRUPO].nunique()),
        "por_split_apartado": {k: int(v) for k, v in
                               apartadas[COLUMNA_SPLIT].value_counts().items()},
        "n_sin_persona_uid": int(sin_persona.sum()),
    }
    return datos, metadata


def etiqueta_multiclase(datos, condiciones=CONDICIONES_EJECUTIVAS):
    """`condicion` -> código entero 0..k-1, en el orden de `condiciones`."""
    condicion = datos[COLUMNA_CONDICION]
    condiciones = tuple(condiciones)
    ajenas = sorted(set(condicion.unique()) - set(condiciones))
    if ajenas:
        raise ValueError(
            f"Solo hay etiqueta para {list(condiciones)}; hay también: {ajenas}."
        )
    codigo = {c: i for i, c in enumerate(condiciones)}
    return condicion.map(codigo).to_numpy(dtype=int)


def nombres_de_clase(codigos, condiciones=CONDICIONES_EJECUTIVAS):
    """Códigos enteros 0..k-1 -> nombres de condición."""
    codigos = np.asarray(codigos)
    condiciones = tuple(condiciones)
    fuera_de_rango = sorted(set(codigos.tolist()) - set(range(len(condiciones))))
    if fuera_de_rango:
        raise ValueError(
            f"Códigos fuera de rango para {len(condiciones)} clases: {fuera_de_rango}."
        )
    return np.array([condiciones[int(c)] for c in codigos])


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------

def modelo_logistico_multinomial():
    """
    Regresión logística multinomial: imputación por mediana -> tipificación ->
    logística.

    Con más de dos clases, `LogisticRegression` con el solver por defecto
    (`lbfgs`) ajusta automáticamente softmax (multinomial) desde
    scikit-learn >= 1.5; no se pasa `multi_class`, que está deprecado.
    `class_weight="balanced"` compensa cualquier desbalance residual entre las
    cinco tareas en el entrenamiento, igual que en OA/OC y reposo/tarea. Sus
    coeficientes (uno por clase, forma (5, 70)) tienen signo y permiten ver la
    dirección del efecto de cada tarea frente a las demás
    (`coeficientes_por_clase_y_pliegue`). Imputación y tipificación van dentro
    del `Pipeline`: se ajustan solo con el pliegue de entrenamiento.
    """
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"))


def modelo_bosque_multiclase(n_estimators=500, semilla=SEMILLA, n_jobs=-1):
    """
    Random Forest multiclase: imputación por mediana -> 500 árboles, hoja >= 5.

    `RandomForestClassifier` admite más de dos clases sin ningún cambio. Misma
    configuración que las Fases 14, 14b, 17 y 18, para que la cifra sea
    comparable con todo lo anterior del proyecto. Una sola configuración, sin
    ajustar hiperparámetros.
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=5,
                               class_weight="balanced", random_state=semilla,
                               n_jobs=n_jobs))


# --------------------------------------------------------------------------
# Evaluación y métricas multiclase
# --------------------------------------------------------------------------

def evaluar_multiclase(X, y, grupos, modelo, nombre=None, n_splits=5):
    """
    Validación cruzada agrupada por persona y métricas multiclase en un solo paso.

    Es el punto de entrada de la evaluación: `predecir_cv_agrupada` (el mismo
    arnés de la Fase 14, agnóstico al número de clases) para las predicciones
    out-of-fold y `metricas_multiclase_por_pliegue` para puntuarlas. A
    diferencia de `evaluar_reposo`, aquí no hace falta envolver el modelo para
    obtener una probabilidad: la predicción de clase (0..k-1) ya es la
    etiqueta que se puntúa.

    Devuelve
    --------
    dict con `y_true`, `y_pred`, `pliegues`, `detalle` (el de
    `predecir_cv_agrupada`) y `metricas` (ver `metricas_multiclase_por_pliegue`).
    """
    X = pd.DataFrame(X)
    y = np.asarray(y)
    grupos = np.asarray(grupos)
    pred_float, detalle = predecir_cv_agrupada(X, y, grupos, modelo,
                                               n_splits=n_splits)
    pred = np.rint(pred_float).astype(int)
    if not np.allclose(pred_float, pred):
        raise ValueError(
            "Las predicciones out-of-fold no son códigos de clase enteros: "
            "¿el modelo es un clasificador?"
        )
    pliegues = pliegue_por_fila(grupos, n_splits)
    return {
        "y_true": y.astype(int), "y_pred": pred, "pliegues": pliegues,
        "detalle": detalle,
        "metricas": metricas_multiclase_por_pliegue(y.astype(int), pred, pliegues,
                                                     nombre=nombre),
    }


def metricas_multiclase(y_true, y_pred, condiciones=CONDICIONES_EJECUTIVAS,
                        nombre=None):
    """
    Métricas de la clasificación multiclase de tareas ejecutivas.

    - `exactitud_balanceada`: media de la exhaustividad de cada clase. Es la
      métrica principal porque su valor de azar es 1/k exacto (0.200 con las 5
      tareas) pase lo que pase con el reparto de clases; la exactitud simple
      no lo garantiza si las clases no están perfectamente equilibradas.
    - `f1_macro`: media no ponderada del F1 de cada clase.
    - `kappa`: coeficiente kappa de Cohen (0 en el azar, corrige por el azar
      esperado dado el reparto de clases observado; a diferencia de la
      exactitud balanceada, depende de ese reparto).
    - `por_clase`: precisión, exhaustividad, F1 y soporte de cada tarea.

    Las etiquetas deben ser códigos `0..k-1` (ver `etiqueta_multiclase`).
    """
    _validar_longitudes(y_true=y_true, y_pred=y_pred)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    condiciones = tuple(condiciones)
    k = len(condiciones)
    ajenos = (set(y_true.tolist()) | set(y_pred.tolist())) - set(range(k))
    if ajenos:
        raise ValueError(
            f"Las etiquetas deben ser códigos 0-{k - 1}; llegaron {sorted(ajenos)}."
        )
    if len(set(y_true.tolist())) < 2:
        raise ValueError(
            "y_true tiene una sola clase: la exactitud balanceada no está definida."
        )

    precision, exhaustividad, f1, soporte = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(k)), zero_division=0)
    por_clase = {
        condiciones[i]: {
            "precision": float(precision[i]),
            "exhaustividad": float(exhaustividad[i]),
            "f1": float(f1[i]),
            "soporte": int(soporte[i]),
        } for i in range(k)
    }
    salida = {
        "n": int(len(y_true)),
        "exactitud": float(accuracy_score(y_true, y_pred)),
        "exactitud_balanceada": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "por_clase": por_clase,
    }
    if nombre is not None:
        salida["modelo"] = nombre
    return salida


def metricas_multiclase_por_pliegue(y_true, y_pred, pliegues,
                                    condiciones=CONDICIONES_EJECUTIVAS,
                                    nombre=None):
    """
    Métricas agregadas **y** dentro de cada pliegue.

    Igual que `reposo.metricas_binarias_por_pliegue`: las claves de
    `metricas_multiclase` son las de las predicciones out-of-fold amontonadas;
    las que llevan `_pliegues`, `_media` o `_sd` se calculan dentro de cada
    pliegue y se reportan las dos para comprobar que coinciden en vez de
    asumirlo.
    """
    _validar_longitudes(y_true=y_true, y_pred=y_pred, pliegues=pliegues)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    pliegues = np.asarray(pliegues)

    salida = metricas_multiclase(y_true, y_pred, condiciones, nombre)
    por_pliegue = []
    for p in sorted(set(pliegues.tolist())):
        m = pliegues == p
        fila = {
            "pliegue": int(p), "n": int(m.sum()),
            "exactitud_balanceada": metricas_multiclase(
                y_true[m], y_pred[m], condiciones)["exactitud_balanceada"],
        }
        por_pliegue.append(fila)

    ba = np.array([f["exactitud_balanceada"] for f in por_pliegue])
    salida["n_pliegues"] = len(por_pliegue)
    salida["por_pliegue"] = por_pliegue
    salida["exactitud_balanceada_pliegues"] = ba.tolist()
    salida["exactitud_balanceada_media"] = float(ba.mean())
    salida["exactitud_balanceada_sd"] = float(ba.std(ddof=1)) if len(ba) > 1 else 0.0
    return salida


def matriz_confusion_normalizada(y_true, y_pred, condiciones=CONDICIONES_EJECUTIVAS,
                                 normalizar="real"):
    """
    Matriz de confusión k×k con nombres de condición.

    `normalizar="real"` (por defecto): cada fila (clase real) suma 1; la
    diagonal es la exhaustividad de cada clase. `normalizar="predicha"`: cada
    columna suma 1. `normalizar=None`: recuentos brutos.
    """
    _validar_longitudes(y_true=y_true, y_pred=y_pred)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    condiciones = tuple(condiciones)
    k = len(condiciones)
    cuentas = confusion_matrix(y_true, y_pred, labels=list(range(k))).astype(float)
    # Una clase sin ninguna grabación real o sin ninguna predicción da una fila
    # o columna de ceros (0/0 = NaN, que es lo correcto: no hay tasa que
    # calcular), no un error de numpy.
    if normalizar not in ("real", "predicha", None):
        raise ValueError(
            f"'normalizar' debe ser 'real', 'predicha' o None; llegó {normalizar!r}."
        )
    with np.errstate(invalid="ignore"):
        if normalizar == "real":
            tabla = cuentas / cuentas.sum(axis=1, keepdims=True)
        elif normalizar == "predicha":
            tabla = cuentas / cuentas.sum(axis=0, keepdims=True)
        else:
            tabla = cuentas
    return pd.DataFrame(tabla, index=list(condiciones), columns=list(condiciones))


# --------------------------------------------------------------------------
# Qué tareas se confunden
# --------------------------------------------------------------------------

def confusion_por_pares(y_true, y_pred, condiciones=CONDICIONES_EJECUTIVAS):
    """
    Confusión de cada uno de los 10 pares de tareas, a partir de la matriz de
    confusión de recuentos.

    `confusion_simetrica` = (n(a predicha como b) + n(b predicha como a)) /
    (n(a) + n(b)): la fracción de las grabaciones de las dos condiciones que el
    modelo intercambia entre ellas. `asimetria` = (n(a→b) − n(b→a)) sobre el
    mismo denominador: 0 si el error va igual en las dos direcciones.

    Añade `similitud_cognitiva` (`SIMILITUD_COGNITIVA`, declarada antes de
    ejecutar) y `distancia_protocolo` (diferencia de posición en
    `ORDEN_PROTOCOLO`), las estructuras candidatas que compara
    `estructura_de_la_confusion`.
    """
    _validar_longitudes(y_true=y_true, y_pred=y_pred)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    condiciones = tuple(condiciones)
    k = len(condiciones)
    cuentas = confusion_matrix(y_true, y_pred, labels=list(range(k)))
    posicion = {c: ORDEN_PROTOCOLO.index(c) for c in condiciones}

    filas = []
    for i, j in itertools.combinations(range(k), 2):
        a, b = condiciones[i], condiciones[j]
        n_a_b, n_b_a = int(cuentas[i, j]), int(cuentas[j, i])
        n_a, n_b = int(cuentas[i].sum()), int(cuentas[j].sum())
        denominador = n_a + n_b
        filas.append({
            "par": f"{a}-{b}", "a": a, "b": b,
            "n_a_b": n_a_b, "n_b_a": n_b_a, "n_a": n_a, "n_b": n_b,
            "confusion_simetrica":
                (n_a_b + n_b_a) / denominador if denominador else float("nan"),
            "asimetria":
                (n_a_b - n_b_a) / denominador if denominador else float("nan"),
            "similitud_cognitiva": similitud_cognitiva(a, b),
            "distancia_protocolo": abs(posicion[a] - posicion[b]),
        })
    return pd.DataFrame(filas).set_index("par")


def distancias_de_duracion(datos, condiciones=CONDICIONES_EJECUTIVAS,
                           columna="n_ventanas"):
    """
    Distancia de duración de cada par de tareas.

    `|mediana(columna) de A − mediana(columna) de B|`, medida **sobre las
    propias grabaciones del análisis** (no se codifican a mano las duraciones
    de ESTADO.md, que son de todo el dataset y no de este subconjunto de
    train+val). `columna` sigue fuera de X (`n_ventanas` es uno de los seis
    contadores de procesamiento excluidos); aquí se usa solo como descriptor,
    para comparar con el patrón de confusión en `estructura_de_la_confusion`.
    """
    _validar_columnas(datos, (COLUMNA_CONDICION, columna), "datos")
    condiciones = tuple(condiciones)
    medianas = datos.groupby(COLUMNA_CONDICION)[columna].median()
    faltan = set(condiciones) - set(medianas.index)
    if faltan:
        raise ValueError(f"Faltan grabaciones de: {sorted(faltan)}.")

    filas = []
    for a, b in itertools.combinations(condiciones, 2):
        filas.append({
            "par": f"{a}-{b}", "a": a, "b": b,
            "distancia_duracion": float(abs(medianas[a] - medianas[b])),
        })
    return pd.DataFrame(filas).set_index("par")


def estructura_de_la_confusion(pares, duraciones=None):
    """
    Compara el patrón de confusión observado con las estructuras candidatas.

    Correlación de Spearman de `confusion_simetrica` (columna de `pares`,
    salida de `confusion_por_pares`) contra la similitud cognitiva y la
    distancia en el protocolo (con el signo invertido: más distancia debería
    significar MENOS confusión), y con la distancia de duración si se pasa
    `duraciones` (salida de `distancias_de_duracion`, alineada por el mismo
    índice de par). También las correlaciones de las estructuras candidatas
    entre sí, para dejar explícito cuánto se disocian.

    Con 5 tareas hay exactamente 10 pares: estas correlaciones son
    descriptivas, no un contraste con potencia estadística.
    """
    _validar_columnas(pares, ("confusion_simetrica", "similitud_cognitiva",
                              "distancia_protocolo"), "pares")
    confusion = pares["confusion_simetrica"].to_numpy(dtype=float)
    similitud = pares["similitud_cognitiva"].to_numpy(dtype=float)
    dist_protocolo = pares["distancia_protocolo"].to_numpy(dtype=float)

    def rho(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            raise ValueError("Una de las variables es constante: Spearman no definido.")
        return float(stats.spearmanr(a, b).statistic)

    salida = {
        "n_pares": int(len(pares)),
        "confusion_vs_similitud_cognitiva": rho(confusion, similitud),
        "confusion_vs_distancia_protocolo": rho(confusion, -dist_protocolo),
        "similitud_vs_distancia_protocolo": rho(similitud, -dist_protocolo),
    }
    if duraciones is not None:
        _validar_columnas(duraciones, ("distancia_duracion",), "duraciones")
        alineadas = duraciones.reindex(pares.index)
        if alineadas["distancia_duracion"].isna().any():
            raise ValueError("`duraciones` no cubre los mismos pares que `pares`.")
        dist_duracion = alineadas["distancia_duracion"].to_numpy(dtype=float)
        salida["confusion_vs_distancia_duracion"] = rho(confusion, -dist_duracion)
        salida["similitud_vs_distancia_duracion"] = rho(similitud, -dist_duracion)
        salida["distancia_protocolo_vs_distancia_duracion"] = rho(
            dist_protocolo, dist_duracion)
    return salida


def _exactitud_balanceada_binaria(y_true, y_pred):
    return metricas_binarias(y_true, y_pred)["exactitud_balanceada"]


def separabilidad_por_pares(X, y, grupos, modelo_fabrica,
                            condiciones=CONDICIONES_EJECUTIVAS, n_splits=5,
                            n_replicas=2000, semilla=42):
    """
    Los 10 clasificadores binarios uno-contra-uno.

    La matriz de confusión mezcla la separabilidad de cada par con la
    dificultad global del problema de 5 clases. Aquí, para cada par de tareas,
    se filtra a esas dos clases y se evalúa un binario con `evaluar_reposo`
    (agrupado por persona, con su bootstrap por persona) reutilizado tal cual:
    dentro de un par, el contraste vuelve a ser binario. `modelo_fabrica` es
    una función sin argumentos que devuelve un estimador nuevo (se llama una
    vez por par, para no compartir estado entre ellos); dentro del par, la
    clase 1 es la segunda condición en el orden de `condiciones`.

    Devuelve
    --------
    pandas.DataFrame indexado por par, con `n`, `n_a`, `n_b`, `n_personas`,
    `exactitud_balanceada` e `ic95`.
    """
    X = pd.DataFrame(X)
    y = np.asarray(y)
    grupos = np.asarray(grupos)
    _validar_longitudes(X=X, y=y, grupos=grupos)
    condiciones = tuple(condiciones)
    codigo = {c: i for i, c in enumerate(condiciones)}

    filas = []
    for a, b in itertools.combinations(condiciones, 2):
        m = np.isin(y, [codigo[a], codigo[b]])
        y_par = (y[m] == codigo[b]).astype(int)
        r = evaluar_reposo(X[m], y_par, grupos[m], modelo_fabrica(), n_splits=n_splits)
        boot = bootstrap_por_persona(y_par, r["y_pred"], grupos[m],
                                     _exactitud_balanceada_binaria,
                                     n_replicas=n_replicas, semilla=semilla)
        filas.append({
            "par": f"{a}-{b}", "a": a, "b": b, "n": int(m.sum()),
            "n_a": int((y[m] == codigo[a]).sum()),
            "n_b": int((y[m] == codigo[b]).sum()),
            "n_personas": int(len(set(grupos[m]))),
            "exactitud_balanceada": r["metricas"]["exactitud_balanceada_media"],
            "ic95": boot["ic95"],
        })
    return pd.DataFrame(filas).set_index("par")


# --------------------------------------------------------------------------
# Dirección del efecto
# --------------------------------------------------------------------------

def coeficientes_por_clase_y_pliegue(X, y, grupos, modelo,
                                     condiciones=CONDICIONES_EJECUTIVAS, n_splits=5):
    """
    Coeficientes de un modelo lineal multinomial, por clase y por pliegue.

    `reposo.coeficientes_por_pliegue` asume un coeficiente por columna
    (clasificación binaria, `coef_` de forma `(1, n_features)`); un modelo
    multinomial tiene `coef_` de forma `(k, n_features)`, una fila por clase.
    Devuelve un dict `{condición: DataFrame features × pliegues}`: el signo
    estable entre pliegues de cada clase es lo que permite comprobar la
    dirección de su efecto frente a las demás.
    """
    X = pd.DataFrame(X)
    y = np.asarray(y, dtype=float)
    grupos = np.asarray(grupos)
    _validar_longitudes(X=X, y=y, grupos=grupos)
    condiciones = tuple(condiciones)
    k = len(condiciones)
    pliegues = pliegue_por_fila(grupos, n_splits)

    columnas = {c: {} for c in condiciones}
    for p in range(1, n_splits + 1):
        entrenamiento = pliegues != p
        ajustado = clone(modelo).fit(X[entrenamiento], y[entrenamiento])
        final = ajustado[-1] if hasattr(ajustado, "steps") else ajustado
        coef = np.asarray(final.coef_, dtype=float)
        if coef.shape != (k, X.shape[1]):
            raise ValueError(
                f"El modelo devolvió coeficientes de forma {coef.shape}; se "
                f"esperaba ({k}, {X.shape[1]}) para {k} clases. ¿Es un modelo "
                f"multinomial ajustado con estas mismas clases?"
            )
        for j, c in enumerate(condiciones):
            columnas[c][f"pliegue_{p}"] = coef[j]
    return {c: pd.DataFrame(columnas[c], index=X.columns) for c in condiciones}


def coeficiente_neto_por_clase(coeficientes, region, banda):
    """
    Suma, por pliegue, de los coeficientes de las features `_media` de un
    bloque, para cada clase.

    `coeficientes` es el dict que devuelve `coeficientes_por_clase_y_pliegue`.
    Con softmax el coeficiente de una clase se lee frente a la combinación
    lineal de las demás (no hay una clase de referencia fija): la comparación
    con sentido es entre clases, no el signo absoluto de una sola. Devuelve un
    DataFrame clases × pliegues.
    """
    filas = {}
    for clase, coef in coeficientes.items():
        columnas = columnas_media_del_bloque(list(coef.index), region, banda)
        if not columnas:
            raise ValueError(
                f"No hay features `_media` de {region}/{banda} entre los coeficientes."
            )
        filas[clase] = coef.loc[columnas].sum()
    return pd.DataFrame(filas).T


def tamanos_de_efecto_una_contra_resto(X, y, condiciones=CONDICIONES_EJECUTIVAS):
    """
    d de Cohen de cada columna, una clase contra el resto, para cada clase.

    Reutiliza `reposo.tamanos_de_efecto` con una etiqueta binaria distinta por
    clase (1 = esa clase, 0 = las otras cuatro juntas). Devuelve un dict
    `{condición: Serie de d de Cohen}`.
    """
    X = pd.DataFrame(X)
    y = np.asarray(y)
    _validar_longitudes(X=X, y=y)
    condiciones = tuple(condiciones)
    return {c: tamanos_de_efecto(X, (y == i).astype(int))
           for i, c in enumerate(condiciones)}
