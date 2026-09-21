"""
Clasificación binaria de reposo (OA, OC) frente a tarea ejecutiva.

Contraste de **carga cognitiva**: ¿distinguen las features de la Fase 13 una
grabación de reposo de una de tarea (STROOP, PASAT, PVTB, MSIT, PVSAT)? Es el
escalón siguiente al control OA/OC (`reposo.py`). Allí el efecto esperado era el
más grande y localizado del EEG de superficie (efecto Berger); aquí es un efecto
de carga ejecutiva, más pequeño y más cercano en naturaleza a la fatiga mental,
que es la variable de interés del proyecto.

Este módulo **no reimplementa la evaluación**. `reposo.py` es casi por completo
agnóstico a qué es la clase 1 (métricas, validación cruzada agrupada, bootstrap
por persona, importancias por bloque, umbral de una variable, control
negativo) y el guion de análisis lo reutiliza tal cual. Lo que sí es propio de
este contraste, y vive aquí:

- **Qué grabaciones entran y cómo se etiquetan** (`preparar_reposo_tarea`,
  `etiqueta_tarea`). `preparar_reposo` solo admite OA/OC y `etiqueta_binaria`
  falla con cualquier otra condición.
- **El desbalance de clases**: hay unas 2.5 grabaciones de tarea por cada una de
  reposo. En el entrenamiento se compensa con `class_weight="balanced"`
  (`modelo_logistico`, `modelo_bosque`). En la evaluación, la métrica principal
  es la exactitud balanceada (su valor de azar es 0.5 pase lo que pase con el
  reparto) y `precision_media` puntúa la clase minoritaria, que es la que la
  exactitud balanceada no deja ver.
- **El desglose de errores por condición original** (`desglose_por_condicion`,
  `comparar_condiciones`): un acierto global esconde qué tareas se confunden
  con reposo y si OA y OC fallan por igual.
- **La dirección del efecto** (`coeficiente_neto_por_pliegue`,
  `signos_umbral_por_pliegue`): comprobar que theta frontal empuja hacia tarea
  y alfa posterior hacia reposo, que es lo que la neurofisiología predice.

Decisiones de diseño (justificación completa en ANALISIS_REPOSO_TAREA.md):

- **La clase 1 es «tarea».** Así `tamanos_de_efecto` da d positivo para lo que
  sube con la carga (theta frontal) y negativo para lo que baja (alfa
  posterior), y los signos se leen como en la literatura.
- **X es la misma que en OA/OC**: `columnas_de_features_reposo`, que parte de
  `columnas_de_features()` (la única definición de X del proyecto, sin tocar) y
  le resta `condicion_orden` y los seis contadores de procesamiento.
  `condicion_orden` es la etiqueta con otro nombre: vale 1-2 en el reposo y 3-7
  en las tareas, así que un solo corte la separa sin error. Es una fuga, no una
  decisión discutible.
- **Dos amenazas a la interpretación que el diseño no puede eliminar y por eso
  se miden**: el reposo incluye OC, que tiene el alfa disparado por el efecto
  Berger, y el orden del protocolo es fijo (el reposo ocupa siempre las
  posiciones 1-2 y las tareas las 3-7), de modo que cualquier deriva dentro de
  la sesión queda confundida con la condición.
- **El conjunto de test no se abre.** `preparar_reposo_tarea` se niega a incluir
  `test` salvo que se pida con `permitir_test=True`.

Lo que este módulo **no** hace: no modifica el pipeline congelado, no recalcula
features y no toca `reposo.py`. Todo sale de `matriz_rf.parquet`.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_library.modelado.modeling import COLUMNA_GRUPO
from eeg_library.modelado.reposo import (
    COLUMNA_CONDICION,
    COLUMNA_SPLIT,
    CONDICIONES_REPOSO,
    SPLITS_ANALISIS,
    UmbralUnaVariable,
    bootstrap_por_persona,
    columnas_de_features_reposo,
    columnas_del_bloque,
    pliegue_por_fila,
)
from eeg_library.procesado.matrix import ORDEN_PROTOCOLO

CONDICIONES_TAREA = ("STROOP", "PASAT", "PVTB", "MSIT", "PVSAT")
CLASE_POSITIVA = "tarea"    # 1 = tarea ejecutiva
CLASE_NEGATIVA = "reposo"   # 0 = reposo (OA u OC)
SEMILLA = 42


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


def _validar_condiciones(condiciones_reposo, condiciones_tarea):
    """
    Cada conjunto debe ser un subconjunto no vacío del suyo.

    Es estricto a propósito: aceptar `STROOP` como «reposo» invertiría la
    etiqueta sin dar ningún error, y una etiqueta invertida da resultados que
    parecen razonables.
    """
    for nombre, elegidas, permitidas in (
            ("condiciones_reposo", condiciones_reposo, CONDICIONES_REPOSO),
            ("condiciones_tarea", condiciones_tarea, CONDICIONES_TAREA)):
        if not elegidas:
            raise ValueError(f"'{nombre}' no puede estar vacío.")
        ajenas = sorted(set(elegidas) - set(permitidas))
        if ajenas:
            raise ValueError(
                f"'{nombre}' solo admite condiciones de {list(permitidas)}; "
                f"llegaron {ajenas}."
            )


def _media_de_valores(_, valores):
    return float(np.mean(valores))


# --------------------------------------------------------------------------
# Definición de X y preparación de los datos
# --------------------------------------------------------------------------

def columnas_de_features_tarea(matriz, incluir_procesamiento=False):
    """
    Definición de X para reposo frente a tarea.

    Es exactamente la de OA/OC (`columnas_de_features_reposo`): resta
    `condicion_orden` **siempre** y los seis contadores de procesamiento salvo
    que se pidan con `incluir_procesamiento=True` (análisis de sensibilidad).
    Existe con nombre propio para que el guion de este análisis no dependa de
    un nombre que habla de OA/OC, y un test comprueba que coinciden. Sobre la
    matriz de la Fase 13 (77 features) devuelve 70.

    Aquí `condicion_orden` es una fuga todavía más grosera que en OA/OC: vale
    1-2 en el reposo y 3-7 en las tareas.
    """
    return columnas_de_features_reposo(
        matriz, incluir_procesamiento=incluir_procesamiento)


def preparar_reposo_tarea(matriz, splits=SPLITS_ANALISIS, permitir_test=False,
                          condiciones_reposo=CONDICIONES_REPOSO,
                          condiciones_tarea=CONDICIONES_TAREA):
    """
    Deja la matriz de la Fase 13 lista para el contraste reposo/tarea.

    Se queda con las grabaciones de las condiciones pedidas y de los `splits`
    pedidos, y descarta las que no tienen `persona_uid` (la Fase 12 lo deja a
    NaN en las sesiones sin etiqueta de VAS; sin él no se puede agrupar por
    persona).

    Parámetros
    ----------
    matriz : pandas.DataFrame
        `matriz_rf.parquet`, una fila por grabación.
    splits : iterable de str
        Particiones que entran al análisis. Por defecto `train` y `val`.
    permitir_test : bool
        `test` se reserva para un único análisis confirmatorio pre-registrado:
        pedirlo sin activar esto es un error, no un aviso.
    condiciones_reposo, condiciones_tarea : iterable de str
        Por defecto todas. Restringirlas da los controles: `("OA",)` deja fuera
        OC (¿es solo el efecto Berger?) y una sola tarea da un contraste por
        pares. Cada una solo admite condiciones de su propia familia.

    Devuelve
    --------
    datos : pandas.DataFrame
        Grabaciones de análisis, con el índice reiniciado.
    metadata : dict
        Recuentos: grabaciones, personas, sesiones, reposo y tarea, desbalance,
        reparto por condición y por split, y lo que quedó apartado o
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
    condiciones_reposo = tuple(condiciones_reposo)
    condiciones_tarea = tuple(condiciones_tarea)
    _validar_condiciones(condiciones_reposo, condiciones_tarea)

    elegidas = matriz[matriz[COLUMNA_CONDICION].isin(
        condiciones_reposo + condiciones_tarea)]
    en_splits = elegidas[elegidas[COLUMNA_SPLIT].isin(splits)]
    apartadas = elegidas[~elegidas[COLUMNA_SPLIT].isin(splits)]
    sin_persona = en_splits[COLUMNA_GRUPO].isna()
    datos = en_splits[~sin_persona].reset_index(drop=True)

    n_reposo = int(datos[COLUMNA_CONDICION].isin(condiciones_reposo).sum())
    n_tarea = int(len(datos) - n_reposo)
    if n_reposo == 0 or n_tarea == 0:
        raise ValueError(
            f"Tras filtrar quedan {n_reposo} grabación(es) de reposo y "
            f"{n_tarea} de tarea: hacen falta de las dos clases."
        )

    metadata = {
        "splits": list(splits),
        "condiciones_reposo": list(condiciones_reposo),
        "condiciones_tarea": list(condiciones_tarea),
        "n_grabaciones": int(len(datos)),
        "n_reposo": n_reposo,
        "n_tarea": n_tarea,
        "clase_mayoritaria": CLASE_POSITIVA if n_tarea > n_reposo else CLASE_NEGATIVA,
        "ratio_desbalance": float(max(n_reposo, n_tarea) / min(n_reposo, n_tarea)),
        "n_personas": int(datos[COLUMNA_GRUPO].nunique()),
        "n_sesiones": (int(datos["sesion_uid"].nunique())
                       if "sesion_uid" in datos.columns else None),
        "por_condicion": {k: int(v) for k, v in
                          datos[COLUMNA_CONDICION].value_counts().items()},
        "por_split": {k: int(v) for k, v in
                      datos[COLUMNA_SPLIT].value_counts().items()},
        "n_apartadas": int(len(apartadas)),
        "n_personas_apartadas": int(apartadas[COLUMNA_GRUPO].nunique()),
        "por_split_apartado": {k: int(v) for k, v in
                               apartadas[COLUMNA_SPLIT].value_counts().items()},
        "n_sin_persona_uid": int(sin_persona.sum()),
    }
    return datos, metadata


def etiqueta_tarea(datos, condiciones_reposo=CONDICIONES_REPOSO,
                   condiciones_tarea=CONDICIONES_TAREA):
    """`condicion` -> 1 para una tarea ejecutiva, 0 para reposo (OA u OC)."""
    condicion = datos[COLUMNA_CONDICION]
    ajenas = sorted(set(condicion.unique())
                    - set(condiciones_reposo) - set(condiciones_tarea))
    if ajenas:
        raise ValueError(
            f"Solo hay etiqueta para {list(condiciones_reposo)} (0) y "
            f"{list(condiciones_tarea)} (1); hay también: {ajenas}."
        )
    return condicion.isin(condiciones_tarea).to_numpy(dtype=int)


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------

def modelo_logistico():
    """
    Regresión logística: imputación por mediana -> tipificación -> logística.

    `class_weight="balanced"` es la forma de tener en cuenta el desbalance
    **en el entrenamiento**: cada clase pesa en la función de coste como si
    tuviera el mismo número de grabaciones, así que la frontera no se desplaza
    hacia la mayoritaria y `P(tarea) > 0.5` da el mismo coste a cada tipo de
    error, que es lo que mide la exactitud balanceada (no se ajusta ningún
    umbral de decisión). Sus coeficientes tienen signo, que es lo que permite
    comprobar la dirección del efecto. Imputación y tipificación van dentro
    del `Pipeline`: se ajustan solo con el pliegue de entrenamiento.
    """
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"))


def modelo_bosque(n_estimators=500, semilla=SEMILLA, n_jobs=-1):
    """
    Random Forest: imputación por mediana -> 500 árboles, hoja >= 5.

    Misma configuración que las Fases 14, 14b y 17, para que la cifra sea
    comparable con todo lo anterior del proyecto. `class_weight="balanced"`
    compensa el desbalance en el entrenamiento, igual que en la logística.
    Una sola configuración, sin ajustar hiperparámetros: elegir la mejor de
    varias sobre estos datos es justo el sesgo que el arnés de evaluación existe
    para evitar.
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=5,
                               class_weight="balanced", random_state=semilla,
                               n_jobs=n_jobs))


# --------------------------------------------------------------------------
# Métricas para clases desbalanceadas
# --------------------------------------------------------------------------

def precision_media(y_true, puntuaciones, clase_positiva=0):
    """
    Precisión media (área bajo la curva precisión-exhaustividad) de una clase.

    `puntuaciones` es la probabilidad de la clase 1 (tarea). Por defecto la
    clase puntuada es la **0 (reposo)**, que es la minoritaria: con ~2.5
    grabaciones de tarea por cada una de reposo, es la que la exactitud
    balanceada y el AUC no dejan ver. Su valor de azar no es 0.5: es la
    prevalencia de la clase puntuada.

    Con `clase_positiva=0` se puntúa `1 - p`. Las filas con NaN en cualquiera
    de los dos vectores se descartan.
    """
    _validar_longitudes(y_true=y_true, puntuaciones=puntuaciones)
    if clase_positiva not in (0, 1):
        raise ValueError(f"`clase_positiva` debe ser 0 o 1; llegó {clase_positiva}.")
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(puntuaciones, dtype=float)
    validas = ~(np.isnan(y) | np.isnan(s))
    y, s = y[validas], s[validas]
    if len(np.unique(y)) < 2:
        raise ValueError(
            "y_true tiene una sola clase: la precisión media no está definida."
        )
    objetivo = (y == clase_positiva).astype(int)
    puntuacion = s if clase_positiva == 1 else 1.0 - s
    return float(average_precision_score(objetivo, puntuacion))


# --------------------------------------------------------------------------
# Errores por condición original
# --------------------------------------------------------------------------

def _preparar_predicciones(datos, y_true, y_pred, grupos=None):
    """Valida y devuelve `(condiciones, y_true, y_pred, grupos)` como arrays."""
    datos = pd.DataFrame(datos)
    _validar_columnas(datos, (COLUMNA_CONDICION,), "datos")
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _validar_longitudes(datos=datos, y_true=y_true, y_pred=y_pred)
    if grupos is not None:
        grupos = np.asarray(grupos)
        _validar_longitudes(datos=datos, grupos=grupos)
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError(
            "y_true e y_pred no pueden contener NaN: el desglose por condición "
            "no descarta filas en silencio."
        )
    ajenos = (set(np.unique(y_true)) | set(np.unique(y_pred))) - {0.0, 1.0}
    if ajenos:
        raise ValueError(f"Las etiquetas deben ser 0 o 1; llegaron {sorted(ajenos)}.")
    condiciones = datos[COLUMNA_CONDICION].to_numpy()
    desconocidas = sorted(set(condiciones) - set(ORDEN_PROTOCOLO))
    if desconocidas:
        raise ValueError(
            f"Condiciones que no están en el protocolo: {desconocidas}."
        )
    return condiciones, y_true, y_pred, grupos


def desglose_por_condicion(datos, y_true, y_pred, grupos=None, n_replicas=2000,
                           semilla=42):
    """
    Tasa de acierto de cada condición original.

    Un acierto global esconde qué falla. Aquí cada una de las condiciones
    presentes (en el orden del protocolo) tiene su propio recuento, y con
    `grupos` un IC al 95 % remuestreando **personas**: las grabaciones de una
    misma persona no son independientes.

    Parámetros
    ----------
    datos : pandas.DataFrame
        Con la columna `condicion`, una fila por grabación.
    y_true, y_pred : array
        0 (reposo) o 1 (tarea), una por fila de `datos`. La etiqueta tiene que
        ser función de la condición: si una condición mezcla las dos clases,
        es un error (con etiquetas barajadas este desglose no tiene sentido).
    grupos : array o None
        `persona_uid` por fila. Sin él no hay intervalos.

    Devuelve
    --------
    pandas.DataFrame indexado por condición, con `clase_real`, `n`, `aciertos`,
    `tasa_acierto`, `frac_predicha_tarea` (la fracción de la condición que el
    modelo lee como tarea: para una tarea es su acierto y para el reposo es su
    error) y, con `grupos`, `n_personas`, `ic95_inf` e `ic95_sup`.
    """
    condiciones, y_true, y_pred, grupos = _preparar_predicciones(
        datos, y_true, y_pred, grupos)

    filas = {}
    for condicion in (c for c in ORDEN_PROTOCOLO if c in set(condiciones)):
        m = condiciones == condicion
        clases = np.unique(y_true[m])
        if len(clases) != 1:
            raise ValueError(
                f"La condición '{condicion}' tiene las dos clases en y_true: "
                f"la etiqueta no es función de la condición."
            )
        acierta = (y_true[m] == y_pred[m]).astype(float)
        fila = {
            "clase_real": int(clases[0]),
            "n": int(m.sum()),
            "aciertos": int(acierta.sum()),
            "tasa_acierto": float(acierta.mean()),
            "frac_predicha_tarea": float(y_pred[m].mean()),
        }
        if grupos is not None:
            boot = bootstrap_por_persona(acierta, acierta, grupos[m],
                                         _media_de_valores,
                                         n_replicas=n_replicas, semilla=semilla)
            fila["n_personas"] = int(len(pd.unique(grupos[m])))
            fila["ic95_inf"], fila["ic95_sup"] = boot["ic95"]
        filas[condicion] = fila
    return pd.DataFrame.from_dict(filas, orient="index")


def comparar_condiciones(datos, y_true, y_pred, grupos, condicion_a, condicion_b,
                         n_replicas=2000, semilla=42):
    """
    Diferencia de tasa de acierto `a - b` entre dos condiciones, con IC95.

    Responde a preguntas como «¿falla más OA que OC?». Remuestrea **personas**
    y evalúa las dos condiciones sobre la misma remuestra en cada réplica, de
    modo que la dificultad propia de cada persona (que aporta grabaciones a las
    dos) se cancela en la diferencia. Las réplicas donde alguna de las dos
    condiciones queda sin grabaciones se descartan y se cuentan aparte.

    Devuelve un dict con `diferencia` (sobre todos los datos), `ic95`,
    `prop_a_mayor` (fracción de réplicas con `a > b`), `tasa_a`, `tasa_b`,
    `n_a`, `n_b`, `n_replicas` y `n_descartadas`.
    """
    condiciones, y_true, y_pred, grupos = _preparar_predicciones(
        datos, y_true, y_pred, grupos)
    if grupos is None:
        raise ValueError("Comparar condiciones necesita `grupos` para el bootstrap.")
    if condicion_a == condicion_b:
        raise ValueError("Hay que comparar dos condiciones distintas.")
    for c in (condicion_a, condicion_b):
        if c not in set(condiciones):
            raise ValueError(f"La condición '{c}' no está en los datos.")

    m = np.isin(condiciones, (condicion_a, condicion_b))
    cond = condiciones[m]
    acierta = (y_true[m] == y_pred[m]).astype(float)

    def diferencia(cond_filas, acierta_filas):
        a = acierta_filas[cond_filas == condicion_a]
        b = acierta_filas[cond_filas == condicion_b]
        if len(a) == 0 or len(b) == 0:
            raise ValueError("La remuestra se quedó sin una de las dos condiciones.")
        return float(a.mean() - b.mean())

    boot = bootstrap_por_persona(cond, acierta, grupos[m], diferencia,
                                 n_replicas=n_replicas, semilla=semilla)
    return {
        "condicion_a": condicion_a, "condicion_b": condicion_b,
        "diferencia": diferencia(cond, acierta),
        "ic95": boot["ic95"],
        "prop_a_mayor": float(np.mean(boot["replicas"] > 0)),
        "tasa_a": float(acierta[cond == condicion_a].mean()),
        "tasa_b": float(acierta[cond == condicion_b].mean()),
        "n_a": int((cond == condicion_a).sum()),
        "n_b": int((cond == condicion_b).sum()),
        "n_replicas": boot["n_replicas"],
        "n_descartadas": boot["n_descartadas"],
    }


# --------------------------------------------------------------------------
# Dirección del efecto
# --------------------------------------------------------------------------

def columnas_media_del_bloque(columnas, region, banda):
    """
    Las features `_media` de una región × banda.

    Son las que entran al coeficiente neto de `coeficiente_neto_por_pliegue`.
    Incluyen los alias de los componentes del índice de fatiga
    (`theta_frontal_log_media`, `alfa_posterior_log_media`), que son copias
    exactas de otra columna y por tanto cuentan dos veces.
    """
    return [c for c in columnas_del_bloque(columnas, region=region, banda=banda)
            if c.endswith("_media")]


def coeficiente_neto_por_pliegue(coef, region, banda):
    """
    Suma, por pliegue, de los coeficientes de las features `_media` de un bloque.

    `coef` es el DataFrame features × pliegues de `coeficientes_por_pliegue`
    de una logística con tipificación, así que los coeficientes están en
    unidades de desviación típica y se pueden sumar. Un signo estable entre los
    pliegues es lo que permite afirmar la **dirección** del efecto: con la clase
    1 = tarea, un valor positivo empuja hacia tarea y uno negativo hacia
    reposo.

    Devuelve una `pandas.Series` indexada por pliegue.
    """
    columnas = columnas_media_del_bloque(list(coef.index), region, banda)
    if not columnas:
        raise ValueError(
            f"No hay features `_media` de {region}/{banda} entre los coeficientes."
        )
    return coef.loc[columnas].sum()


def signos_umbral_por_pliegue(X, y, grupos, columna, n_splits=5):
    """
    Umbral y signo que aprende un corte sobre una columna en cada pliegue.

    Ajusta `UmbralUnaVariable` con el entrenamiento de cada pliegue de la CV
    agrupada (el mismo reparto que `predecir_cv_agrupada`, vía
    `pliegue_por_fila`). Con la clase 1 = tarea, `signo = +1` significa «tarea
    si la variable supera el corte» y `-1` «tarea si queda por debajo». El signo
    se aprende, no se impone: que salga el que predice la teoría es una
    comprobación neurofisiológica.

    Devuelve una lista de dicts `{pliegue, umbral, signo}`.
    """
    X = pd.DataFrame(X)
    y = np.asarray(y)
    grupos = np.asarray(grupos)
    _validar_longitudes(X=X, y=y, grupos=grupos)
    _validar_columnas(X, (columna,), "X")
    pliegues = pliegue_por_fila(grupos, n_splits)
    salida = []
    for p in range(1, n_splits + 1):
        entrenamiento = pliegues != p
        ajustado = UmbralUnaVariable(columna).fit(X[entrenamiento], y[entrenamiento])
        salida.append({"pliegue": p, "umbral": ajustado.umbral_,
                       "signo": ajustado.signo_})
    return salida
