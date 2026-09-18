"""
Módulo de evaluación y modelado supervisado (Fase 14).

Convierte la matriz por grabación de la Fase 13 en el banco de pruebas con el
que se entrena y se juzga el Random Forest. No contiene el modelo: contiene
las reglas que hacen que su resultado signifique algo.

Decisiones de diseño (justificación completa en MODELADO.md, Fase 14):

- **La unidad de evaluación es la sesión, no la grabación.** El VAS se responde
  una vez por sesión y las 7 grabaciones de esa sesión comparten su valor. Medir
  a nivel de grabación multiplica por 7 el n aparente sin añadir ni una etiqueta
  independiente, y todas las barras de error salen falsamente estrechas.
  `agregar_a_sesion` colapsa la matriz **antes** de modelar, para que el n del
  `fit` sea el n real del estudio.
- **La partición es por persona (`persona_uid`), nunca por sesión ni por
  ventana.** Estas features identifican a la persona con acierto 0.383 sobre 129
  clases (Fase 13). Si dos sesiones de la misma persona caen a ambos lados de la
  partición, el modelo puede reconocerla y devolver su nivel habitual de VAS: el
  resultado mediría reidentificación, no fatiga.
- **Hay dos tareas distintas, con dos líneas base distintas, y no se pueden
  comparar entre sí.** Es la decisión central de esta fase:
  - *Persona nueva* (`predecir_cv_agrupada`): el modelo nunca ha visto a quien
    evalúa. Su rival es la media global, R² = 0. Es el escenario de un
    investigador que pone el casco a alguien por primera vez.
  - *Persona conocida* (`centrar_por_persona_loo`): se conocen otras sesiones de
    esa persona, así que la pregunta es si el EEG explica su **desviación**
    respecto a su propio nivel habitual. Su rival es la media por persona
    leave-one-out, R² = 0.260. Es el escenario de seguimiento a lo largo del
    curso.
  Confundirlas es la forma más fácil de reportar un R² de 0.26 "conseguido" por
  el EEG que en realidad sale de saber a quién se mide.
- **Toda media que se use como referencia se calcula leave-one-out.** La media
  de una persona sobre *sus propias* sesiones evaluadas es circular: en estos
  datos infla el R² de 0.260 a 0.600. `medias_loo_por_grupo` excluye siempre la
  fila que se está prediciendo.
- **`condicion_orden` no sobrevive a la agregación.** Promediada sobre las 7
  grabaciones de una sesión vale ~4 en todas, y en las sesiones incompletas se
  convierte en un indicador de qué grabaciones faltan. Deja de ser una feature y
  pasa a ser ruido con estructura, así que se descarta explícitamente.

Lo que este módulo **no** hace: no toca `test`. Todas las funciones están
pensadas para correr sobre `train`+`val`; gastar el conjunto de test es una
decisión manual que se toma una sola vez y se documenta.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from eeg_library.procesado.matrix import COLUMNAS_IDENTIDAD, columnas_de_features

COLUMNA_GRUPO = "persona_uid"
COLUMNA_SESION = "sesion_uid"
COLUMNA_ETIQUETA = "vas"

# Features de la Fase 13 que pierden su significado al promediarse sobre las
# grabaciones de una sesión (ver la nota del módulo).
COLUMNAS_NO_AGREGABLES = ("condicion_orden",)

# Columnas de identidad que sí tienen sentido a nivel de sesión: son constantes
# dentro de una sesión, así que se conservan con `first`.
IDENTIDAD_SESION = ("persona_uid", "sujeto_id", "centro", "medicion", "split")


def _validar_columnas(df, columnas, nombre):
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise ValueError(f"A '{nombre}' le faltan columnas necesarias: {faltan}.")


def columnas_de_features_sesion(sesiones):
    """
    Definición de X a nivel de sesión, análoga a `columnas_de_features`.

    Excluye la identidad, la etiqueta, las columnas no agregables y las
    auxiliares que añade `agregar_a_sesion`.
    """
    excluidas = (set(COLUMNAS_IDENTIDAD) | {COLUMNA_ETIQUETA, COLUMNA_SESION}
                 | set(COLUMNAS_NO_AGREGABLES))
    return [c for c in sesiones.columns if c not in excluidas]


def agregar_a_sesion(matriz, solo_etiquetadas=True):
    """
    Colapsa la matriz por grabación (Fase 13) a una fila por sesión.

    Promedia cada feature sobre las grabaciones de la sesión. El promedio ignora
    NaN, así que una sesión en la que solo algunas grabaciones se quedaron sin
    región temporal conserva el valor de las demás; solo sale NaN si **ninguna**
    grabación de la sesión tiene esa región.

    Parámetros
    ----------
    matriz : pandas.DataFrame
        `matriz_rf.parquet` (Fase 13), una fila por grabación.
    solo_etiquetadas : bool
        Si True (por defecto) descarta las sesiones sin VAS. Esas sesiones
        tienen además `persona_uid` a NaN (Fase 12), así que arrastrarlas
        rompería cualquier agrupación por persona.

    Devuelve
    --------
    sesiones : pandas.DataFrame
        Una fila por `sesion_uid`, con las columnas de `IDENTIDAD_SESION`, la
        etiqueta `vas`, las features promediadas y `n_grabaciones`.
    """
    _validar_columnas(matriz, (COLUMNA_SESION, COLUMNA_ETIQUETA, COLUMNA_GRUPO),
                      "matriz")
    m = matriz
    if solo_etiquetadas:
        m = m[m[COLUMNA_ETIQUETA].notna()]
        if m[COLUMNA_GRUPO].isna().any():
            raise ValueError(
                "Hay grabaciones con VAS pero sin persona_uid. No deberían "
                "existir: revisar la salida de la Fase 12."
            )

    features = [c for c in columnas_de_features(matriz)
                if c not in COLUMNAS_NO_AGREGABLES]
    identidad = [c for c in IDENTIDAD_SESION if c in m.columns]

    agrupado = m.groupby(COLUMNA_SESION, sort=True)
    sesiones = agrupado[features].mean()
    sesiones["n_grabaciones"] = agrupado.size()
    for columna in identidad:
        unicos = agrupado[columna].nunique(dropna=False)
        if (unicos > 1).any():
            culpables = sorted(unicos[unicos > 1].index)[:5]
            raise ValueError(
                f"La columna '{columna}' no es constante dentro de la sesión "
                f"en {int((unicos > 1).sum())} caso(s): {culpables}. No puede "
                f"agregarse con 'first' sin perder información."
            )
        sesiones[columna] = agrupado[columna].first()
    sesiones[COLUMNA_ETIQUETA] = agrupado[COLUMNA_ETIQUETA].first()

    orden = identidad + [COLUMNA_ETIQUETA, "n_grabaciones"]
    resto = [c for c in sesiones.columns if c not in orden]
    return sesiones[orden + resto].reset_index()


def medias_loo_por_grupo(datos, grupos):
    """
    Media **leave-one-out** por grupo: para cada fila, la media de las *otras*
    filas de su mismo grupo.

    Es la pieza que evita la circularidad de estimar la media de una persona
    incluyendo la sesión que se está prediciendo. Una fila cuyo grupo no tiene
    ninguna otra observación devuelve NaN, no un valor inventado: es un caso en
    el que la referencia sencillamente no existe.

    Los NaN del propio dato se ignoran, igual que en `mean()`: una fila con NaN
    recibe la media de todas las demás de su grupo, y una fila con valor recibe
    la media de las demás excluyéndose a sí misma.

    Parámetros
    ----------
    datos : pandas.DataFrame o pandas.Series
        Valores numéricos a promediar.
    grupos : array-like
        Etiqueta de grupo por fila (aquí, `persona_uid`).

    Devuelve
    --------
    loo : mismo tipo que `datos`
        Misma forma e índice que la entrada.
    """
    es_serie = isinstance(datos, pd.Series)
    df = datos.to_frame() if es_serie else datos
    g = pd.Series(np.asarray(grupos), index=df.index)

    agrupado = df.groupby(g)
    suma = agrupado.transform("sum")      # ignora NaN
    cuenta = agrupado.transform("count")  # cuenta no-NaN

    presente = df.notna()
    # Si la fila tiene valor, hay que descontarlo de la suma y del recuento;
    # si es NaN, no entró en ninguno de los dos y la media de "las otras" es la
    # de todo el grupo.
    numerador = suma.where(~presente, suma - df.fillna(0))
    denominador = cuenta.where(~presente, cuenta - 1)
    loo = numerador / denominador.where(denominador > 0)

    return loo.iloc[:, 0].rename(datos.name) if es_serie else loo


def centrar_por_persona_loo(sesiones, columnas, grupo=COLUMNA_GRUPO):
    """
    Resta a cada columna la media leave-one-out de su persona.

    Es la traducción operativa de la tarea *persona conocida*: se le concede al
    modelo el nivel habitual de esa persona (estimado con sus **otras**
    sesiones) y se le pregunta solo por la desviación respecto a él. Al centrar
    a la vez X e y, el eje entre-personas desaparece de ambos lados, así que
    reconocer a la persona deja de servir de nada — que es justamente el atajo
    que hay que cerrar.

    Devuelve
    --------
    centradas : pandas.DataFrame
        Copia de `sesiones` con `columnas` centradas y una columna nueva
        `<columna>_loo` por cada una, con la referencia usada. Las filas de
        personas con una sola sesión salen a NaN.
    """
    _validar_columnas(sesiones, [grupo], "sesiones")
    _validar_columnas(sesiones, columnas, "sesiones")
    referencias = medias_loo_por_grupo(sesiones[list(columnas)], sesiones[grupo])
    centradas = sesiones.copy()
    centradas[list(columnas)] = sesiones[list(columnas)] - referencias
    return pd.concat([centradas, referencias.add_suffix("_loo")], axis=1)


def metricas(y_true, y_pred, nombre=None):
    """
    R², MAE, RMSE y correlación de Spearman entre observado y predicho.

    Spearman se reporta siempre porque el VAS es ordinal de facto (el 76.7 % de
    las respuestas son múltiplos de 10): el orden es más fiable que la distancia
    exacta. Las filas con NaN en cualquiera de los dos vectores se descartan.

    R² se calcula sobre el vector completo de predicciones agrupadas, no
    promediando el R² de cada pliegue: con pliegues de ~30 personas, el promedio
    de R² por pliegue es inestable y depende de cómo se repartan las personas.
    """
    y_true = pd.Series(np.asarray(y_true, dtype=float))
    y_pred = pd.Series(np.asarray(y_pred, dtype=float))
    validas = y_true.notna() & y_pred.notna()
    y_true, y_pred = y_true[validas], y_pred[validas]
    if len(y_true) < 3:
        raise ValueError(
            f"Hacen falta al menos 3 observaciones válidas para calcular "
            f"métricas y hay {len(y_true)}."
        )
    # Una predicción constante (la línea base de la media global lo es siempre)
    # no tiene orden que correlacionar: Spearman queda indefinido y se reporta
    # como NaN en vez de dejar que scipy avise por cada llamada.
    if y_true.nunique() < 2 or y_pred.nunique() < 2:
        rho, p_rho = np.nan, np.nan
    else:
        rho, p_rho = stats.spearmanr(y_true, y_pred)
    salida = {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "spearman": float(rho),
        "p_spearman": float(p_rho),
    }
    if nombre is not None:
        salida["modelo"] = nombre
    return salida


def lineas_base(sesiones, etiqueta=COLUMNA_ETIQUETA, grupo=COLUMNA_GRUPO,
                ejes=("medicion", "centro")):
    """
    Calcula las líneas base contra las que se juzga cualquier modelo.

    Una línea base es una regla de predicción trivial, sin EEG. Un modelo que no
    las bata no ha aprendido nada de la señal, por bueno que parezca su R² en
    abstracto.

    Devuelve
    --------
    base : pandas.DataFrame
        Una fila por línea base, con las métricas de `metricas`. Incluye:
        media global; media por cada eje de `ejes`; media por persona en muestra
        (circular, se reporta solo para enseñar cuánto infla); y media por
        persona leave-one-out, que es la línea base honesta de la tarea
        *persona conocida*.
    """
    _validar_columnas(sesiones, [etiqueta, grupo], "sesiones")
    y = sesiones[etiqueta]
    filas = [metricas(y, np.full(len(y), y.mean()), "media global")]

    for eje in ejes:
        if eje in sesiones.columns:
            pred = sesiones.groupby(eje)[etiqueta].transform("mean")
            filas.append(metricas(y, pred, f"media por {eje} (en muestra)"))

    filas.append(metricas(y, sesiones.groupby(grupo)[etiqueta].transform("mean"),
                          f"media por {grupo} (en muestra, circular)"))

    loo = medias_loo_por_grupo(y, sesiones[grupo])
    # Una persona con una sola sesión no tiene referencia propia; lo que un
    # sistema real haría es recurrir a la media global, así que eso se hace aquí
    # en vez de descartar la sesión y quedarse con una muestra más fácil.
    n_sin_referencia = int(loo.isna().sum())
    filas.append(metricas(y, loo.fillna(y.mean()),
                          f"media por {grupo} (leave-one-out)"))
    base = pd.DataFrame(filas)
    base.attrs["n_sin_referencia_loo"] = n_sin_referencia
    return base


def predecir_cv_agrupada(X, y, grupos, modelo, n_splits=5):
    """
    Validación cruzada agrupada: predicciones *out-of-fold* para cada fila.

    Reparte las **personas** (no las sesiones) en `n_splits` bloques. Para cada
    bloque, entrena con las personas de los otros y predice las sesiones de
    este. Así cada predicción procede de un modelo que nunca vio a esa persona,
    y al final hay una predicción honesta para cada fila.

    El modelo se clona en cada pliegue, de modo que no arrastra nada del
    anterior.

    Devuelve
    --------
    predicciones : numpy.ndarray
        Una predicción out-of-fold por fila de `X`, en su mismo orden.
    detalle : dict
        `n_splits`, personas y filas por pliegue, y el reparto real.
    """
    from sklearn.base import clone

    X = pd.DataFrame(X)
    y = np.asarray(y, dtype=float)
    grupos = np.asarray(grupos)
    if len(X) != len(y) or len(X) != len(grupos):
        raise ValueError(
            f"X, y y grupos deben tener la misma longitud: {len(X)}, {len(y)}, "
            f"{len(grupos)}."
        )
    n_grupos = len(pd.unique(grupos))
    if n_grupos < n_splits:
        raise ValueError(
            f"No se puede hacer una validación cruzada de {n_splits} pliegues "
            f"con {n_grupos} grupo(s): habría pliegues vacíos."
        )

    predicciones = np.full(len(X), np.nan)
    reparto = []
    cv = GroupKFold(n_splits=n_splits)
    for i, (idx_train, idx_val) in enumerate(cv.split(X, y, groups=grupos), start=1):
        ajustado = clone(modelo).fit(X.iloc[idx_train], y[idx_train])
        predicciones[idx_val] = ajustado.predict(X.iloc[idx_val])
        reparto.append({
            "pliegue": i,
            "personas_train": int(len(set(grupos[idx_train]))),
            "personas_val": int(len(set(grupos[idx_val]))),
            "sesiones_train": int(len(idx_train)),
            "sesiones_val": int(len(idx_val)),
        })

    solapan = set()
    for i, (idx_train, idx_val) in enumerate(cv.split(X, y, groups=grupos)):
        solapan |= set(grupos[idx_train]) & set(grupos[idx_val])
    if solapan:
        raise ValueError(
            f"{len(solapan)} grupo(s) aparecen a la vez en train y validación: "
            f"{sorted(solapan)[:5]}. La partición no está agrupando de verdad."
        )

    return predicciones, {"n_splits": n_splits, "n_grupos": int(n_grupos),
                          "reparto": reparto}


# --------------------------------------------------------------------------
# Clasificación ordinal en tres niveles (Fase 14b)
#
# El VAS es ordinal de facto: el 76.7 % de las respuestas son múltiplos de 10,
# así que el participante no está usando una escala continua sino un puñado de
# escalones. Si la señal que hay ordena pero no cuantifica —que es exactamente
# lo que mide la Fase 14: rho ~ +0.12 con R² ~ 0— tratarlo como tres niveles
# puede recuperar algo que la regresión pierde al exigirle el número exacto.
#
# La regla que hace que esto signifique algo: **los cortes de los tercios se
# calculan con el pliegue de entrenamiento y se aplican al de validación.**
# Calcularlos sobre todos los datos filtraría la distribución de la etiqueta de
# las personas evaluadas, que es la misma circularidad que se cierra en el resto
# del módulo. `predecir_clases_cv_agrupada` lo impone; no se puede saltar.
# --------------------------------------------------------------------------

NOMBRES_CLASES = ("poca", "media", "mucha")


def cortes_terciles(y_referencia):
    """
    Puntos de corte de los tercios, estimados sobre un vector de referencia.

    Devuelve los percentiles 33.3 y 66.7 de `y_referencia`. Se separa de
    `a_clases` a propósito: en validación cruzada la referencia es **solo** el
    pliegue de entrenamiento, y el mismo par de cortes se aplica después a las
    personas evaluadas.

    Devuelve
    --------
    cortes : numpy.ndarray de 2 elementos, ordenados.
    """
    y = np.asarray(y_referencia, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) < 3:
        raise ValueError(
            f"Hacen falta al menos 3 valores para estimar los tercios y hay "
            f"{len(y)}."
        )
    cortes = np.percentile(y, [100 / 3, 200 / 3])
    if cortes[0] == cortes[1]:
        raise ValueError(
            "Los dos cortes de los tercios coinciden: la etiqueta de "
            "referencia está demasiado concentrada para partirla en tres "
            "niveles."
        )
    return cortes


def a_clases(y, cortes):
    """
    Convierte una etiqueta continua en 0 (poca) / 1 (media) / 2 (mucha).

    Un valor exactamente igual a un corte cae en la clase superior, que es el
    convenio de `numpy.digitize`. Los NaN se propagan como NaN en vez de
    asignarse a una clase arbitraria.
    """
    y = np.asarray(y, dtype=float)
    cortes = np.asarray(cortes, dtype=float)
    if len(cortes) != 2:
        raise ValueError(f"Se esperaban 2 cortes y llegaron {len(cortes)}.")
    clases = np.digitize(y, cortes, right=False).astype(float)
    clases[np.isnan(y)] = np.nan
    return clases


def metricas_clasificacion(y_true, y_pred, nombre=None):
    """
    Métricas de la clasificación en tres niveles, tratados como **ordenados**.

    - `exactitud_balanceada`: acierto medio por clase. Es la métrica principal
      porque no la infla predecir siempre la clase mayoritaria: su valor de azar
      es 1/3 exacto, pase lo que pase con el reparto de clases.
    - `f1_macro`: media de los F1 por clase, misma lógica.
    - `exactitud_adyacente`: fracción de aciertos permitiendo fallar por un
      escalón. Distingue "confunde poca con media" (leve) de "confunde poca con
      mucha" (grave), que la exactitud normal trata igual.
    - `mae_ordinal`: error medio en escalones. 0 = perfecto, 2 = máximo.
    - `kendall_tau`: concordancia de orden entre la clase real y la predicha.
      Es el análogo de Spearman para pocos niveles con muchos empates.

    Las filas con NaN en cualquiera de los dos vectores se descartan.
    """
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score)

    y_true = pd.Series(np.asarray(y_true, dtype=float))
    y_pred = pd.Series(np.asarray(y_pred, dtype=float))
    validas = y_true.notna() & y_pred.notna()
    y_true, y_pred = y_true[validas], y_pred[validas]
    if len(y_true) < 3:
        raise ValueError(
            f"Hacen falta al menos 3 observaciones válidas para calcular "
            f"métricas y hay {len(y_true)}."
        )
    yt, yp = y_true.to_numpy(int), y_pred.to_numpy(int)

    # Sin variación en alguno de los dos vectores el tau queda indefinido.
    if y_true.nunique() < 2 or y_pred.nunique() < 2:
        tau, p_tau = np.nan, np.nan
    else:
        tau, p_tau = stats.kendalltau(yt, yp, variant="b")

    salida = {
        "n": int(len(yt)),
        "exactitud": float(accuracy_score(yt, yp)),
        "exactitud_balanceada": float(balanced_accuracy_score(yt, yp)),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "exactitud_adyacente": float(np.mean(np.abs(yt - yp) <= 1)),
        "mae_ordinal": float(np.mean(np.abs(yt - yp))),
        "kendall_tau": float(tau),
        "p_kendall": float(p_tau),
    }
    if nombre is not None:
        salida["modelo"] = nombre
    return salida


def predecir_clases_cv_agrupada(X, y_continua, grupos, modelo, n_splits=5,
                                modo="clasificacion"):
    """
    Validación cruzada agrupada para la tarea de tres niveles.

    Reparte **personas** igual que `predecir_cv_agrupada`, y además estima los
    cortes de los tercios dentro de cada pliegue de entrenamiento. La clase
    "verdadera" de una sesión de validación se define con los cortes de *su*
    pliegue de entrenamiento, no con los del conjunto entero: es lo que haría un
    sistema real, que tiene unos umbrales fijados con sus datos y los aplica a
    gente nueva.

    Parámetros
    ----------
    X : pandas.DataFrame
    y_continua : array-like
        Etiqueta **continua** (VAS, o su desviación centrada por persona). La
        discretización ocurre dentro, nunca fuera.
    grupos : array-like
        `persona_uid`.
    modelo : estimador de scikit-learn
        Clasificador si `modo="clasificacion"`; regresor si `modo="regresion"`.
    modo : {"clasificacion", "regresion"}
        `"clasificacion"` entrena sobre las clases del pliegue.
        `"regresion"` entrena sobre el continuo y **discretiza la predicción**
        con los mismos cortes. Sirve para responder si pasar a tres niveles
        aporta algo o si basta con redondear lo que ya da la regresión.

    Devuelve
    --------
    clases_true, clases_pred : numpy.ndarray
        Clase real y predicha por fila, en el orden de `X`. NaN donde no hubo
        predicción (etiqueta ausente).
    detalle : dict
        Cortes por pliegue y reparto de personas.
    """
    from sklearn.base import clone

    if modo not in ("clasificacion", "regresion"):
        raise ValueError(
            f"`modo` debe ser 'clasificacion' o 'regresion', no {modo!r}."
        )
    X = pd.DataFrame(X)
    y = np.asarray(y_continua, dtype=float)
    grupos = np.asarray(grupos)
    if len(X) != len(y) or len(X) != len(grupos):
        raise ValueError(
            f"X, y y grupos deben tener la misma longitud: {len(X)}, {len(y)}, "
            f"{len(grupos)}."
        )
    n_grupos = len(pd.unique(grupos))
    if n_grupos < n_splits:
        raise ValueError(
            f"No se puede hacer una validación cruzada de {n_splits} pliegues "
            f"con {n_grupos} grupo(s): habría pliegues vacíos."
        )

    clases_true = np.full(len(X), np.nan)
    clases_pred = np.full(len(X), np.nan)
    pliegue_por_fila = np.full(len(X), -1, dtype=int)
    reparto = []
    cv = GroupKFold(n_splits=n_splits)
    for i, (idx_tr, idx_va) in enumerate(cv.split(X, y, groups=grupos), start=1):
        if set(grupos[idx_tr]) & set(grupos[idx_va]):
            raise ValueError(
                f"El pliegue {i} comparte personas entre train y validación. "
                f"La partición no está agrupando de verdad."
            )
        # Los cortes salen SOLO del pliegue de entrenamiento.
        cortes = cortes_terciles(y[idx_tr])
        ytr_clase = a_clases(y[idx_tr], cortes)
        clases_true[idx_va] = a_clases(y[idx_va], cortes)
        pliegue_por_fila[idx_va] = i

        usables = ~np.isnan(ytr_clase)
        ajustado = clone(modelo)
        if modo == "clasificacion":
            ajustado.fit(X.iloc[idx_tr][usables], ytr_clase[usables])
            clases_pred[idx_va] = ajustado.predict(X.iloc[idx_va])
        else:
            ajustado.fit(X.iloc[idx_tr][usables], y[idx_tr][usables])
            clases_pred[idx_va] = a_clases(
                ajustado.predict(X.iloc[idx_va]), cortes)

        reparto.append({
            "pliegue": i,
            "cortes": [float(c) for c in cortes],
            "personas_train": int(len(set(grupos[idx_tr]))),
            "personas_val": int(len(set(grupos[idx_va]))),
            "sesiones_val": int(len(idx_va)),
        })

    return clases_true, clases_pred, {
        "n_splits": n_splits, "n_grupos": int(n_grupos), "modo": modo,
        "reparto": reparto, "pliegue_por_fila": pliegue_por_fila}


def metricas_clasificacion_por_pliegue(y_true, y_pred, pliegues, nombre=None):
    """
    Métricas de clasificación calculadas **dentro de cada pliegue** y luego
    promediadas.

    Es la forma correcta de puntuar `predecir_clases_cv_agrupada`, y se separa
    de `metricas_clasificacion` porque aquí no da igual: los cortes de los
    tercios son distintos en cada pliegue, así que la clase 1 de un pliegue no
    es la misma cosa que la clase 1 de otro. Amontonar las predicciones de los
    cinco y puntuarlas de golpe mezcla escalas y crea señal donde no la hay:
    medido sobre estos datos, un `DummyClassifier` sin ninguna información saca
    así 0.407 de exactitud balanceada y un Kendall tau de +0.105 con p = 0.0098.
    Promediando por pliegue, ese mismo clasificador vuelve a su 1/3 exacto.

    Nótese que esto **no** afecta a `metricas` (la de regresión): allí la
    etiqueta es la misma en todos los pliegues y agrupar es legítimo — y de
    hecho preferible, porque el R² por pliegue es inestable con ~30 personas.

    Devuelve
    --------
    resumen : dict
        Media entre pliegues de cada métrica, más `<métrica>_sd`, el número de
        pliegues usados y la lista `por_pliegue` con el detalle.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    pliegues = np.asarray(pliegues)
    if not (len(y_true) == len(y_pred) == len(pliegues)):
        raise ValueError(
            f"y_true, y_pred y pliegues deben tener la misma longitud: "
            f"{len(y_true)}, {len(y_pred)}, {len(pliegues)}."
        )

    por_pliegue = []
    for p in sorted(set(pliegues[pliegues >= 0])):
        sel = pliegues == p
        valido = sel & ~np.isnan(y_true) & ~np.isnan(y_pred)
        if valido.sum() < 3:
            continue
        m = metricas_clasificacion(y_true[valido], y_pred[valido])
        m["pliegue"] = int(p)
        por_pliegue.append(m)
    if not por_pliegue:
        raise ValueError("Ningún pliegue tiene observaciones suficientes.")

    claves = [k for k in por_pliegue[0] if k not in ("pliegue", "n", "modelo")]
    resumen = {"n_pliegues": len(por_pliegue),
               "n": int(sum(m["n"] for m in por_pliegue))}
    for k in claves:
        valores = np.array([m[k] for m in por_pliegue], dtype=float)
        finitos = valores[~np.isnan(valores)]
        # Una métrica indefinida en todos los pliegues (el tau de un
        # clasificador constante, por ejemplo) se reporta como NaN, no como el
        # promedio de un conjunto vacío.
        resumen[k] = float(finitos.mean()) if len(finitos) else float("nan")
        resumen[f"{k}_sd"] = (float(finitos.std(ddof=1)) if len(finitos) > 1
                              else float("nan"))
    resumen["por_pliegue"] = por_pliegue
    if nombre is not None:
        resumen["modelo"] = nombre
    return resumen
