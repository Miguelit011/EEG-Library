"""
Clasificación de reposo: ojos abiertos (OA) frente a ojos cerrados (OC).

Este módulo no construye un modelo útil por sí mismo —nadie necesita adivinar si
alguien tenía los ojos cerrados—: contiene las reglas que convierten esa
pregunta en un **control positivo del pipeline**. El bloqueo del ritmo alfa
posterior al abrir los ojos (efecto Berger) es el fenómeno mejor establecido del
EEG de superficie. Si las features de la Fase 13 lo conservan, los resultados
negativos sobre el VAS (Fases 14, 14b y 16) se atribuyen a la fatiga percibida y
no a que el procesamiento destruya la señal; si no lo conservan, hay un problema
en el pipeline.

Decisiones de diseño (justificación completa en ANALISIS_OA_OC.md):

- **La unidad de análisis es la grabación.** A diferencia del VAS, que se
  responde una vez por sesión y obligó a `agregar_a_sesion` (Fase 14), la
  condición ojos abiertos/cerrados es una propiedad de cada grabación, con su
  propio registro EEG. Cada fila es una etiqueta independiente. La dependencia
  entre las grabaciones de una misma persona se controla agrupando por
  `persona_uid`, tanto al validar como al calcular intervalos.
- **X parte de `columnas_de_features()` y le resta siete columnas.**
  `columnas_de_features()` sigue siendo la única definición de X del proyecto y
  no se toca; este módulo la restringe, de forma explícita y comprobable:
  - `condicion_orden` **es la etiqueta**: codifica la posición en el protocolo
    (OC = 1, OA = 2) sin una sola excepción. Dejarla daría un acierto del 100 %
    por construcción. Es una fuga, no una decisión discutible, y ni siquiera
    `incluir_procesamiento=True` la devuelve.
  - Los seis contadores de procesamiento (`n_ventanas`, `n_canales_buenos`,
    `n_validos_*`) no son señal cerebral, y algunos discriminan OA/OC
    (presumiblemente porque con los ojos cerrados hay menos parpadeos y más
    canales frontales superan el control de calidad de la Fase 4; es una
    hipótesis, no se comprueba aquí). Sea cual sea el mecanismo, es una vía que
    no pasa por el alfa: un acierto apoyado ahí no diría nada sobre si el
    pipeline lo conserva. Quedan fuera del modelo principal;
    `incluir_procesamiento=True` los devuelve para medir cuánto acierto
    aportarían.
- **La validación es la de la Fase 14.** `predecir_cv_agrupada` reparte
  personas (`persona_uid`, nunca `sujeto_id`) y comprueba que ninguna quede a
  ambos lados. Aquí no se reimplementa: se reutiliza tal cual.
- **El conjunto de test no se abre.** `preparar_reposo` se niega a incluir
  `test` salvo que se pida con `permitir_test=True`.

Lo que este módulo **no** hace: no modifica el pipeline congelado ni recalcula
features. Todo sale de `matriz_rf.parquet`.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import GroupKFold

from eeg_library.modelado.modeling import COLUMNA_GRUPO, predecir_cv_agrupada
from eeg_library.procesado.bands import BANDAS_DEFECTO
from eeg_library.procesado.matrix import (COLUMNA_ETIQUETA, COLUMNAS_IDENTIDAD,
                                          REGIONES, columnas_de_features)

CONDICIONES_REPOSO = ("OA", "OC")
CLASE_POSITIVA = "OC"   # 1 = ojos cerrados
CLASE_NEGATIVA = "OA"   # 0 = ojos abiertos
COLUMNA_CONDICION = "condicion"
COLUMNA_SPLIT = "split"
SPLITS_ANALISIS = ("train", "val")

# La etiqueta disfrazada: posición de la condición en ORDEN_PROTOCOLO. Se
# separa de las demás porque su exclusión no se discute, es una fuga.
COLUMNA_FUGA = "condicion_orden"

# Metadatos del procesamiento, no señal cerebral. Algunos discriminan OA/OC por
# una vía que no es el alfa (presumiblemente la calidad de contacto).
COLUMNAS_PROCESAMIENTO = (
    "n_ventanas", "n_canales_buenos", "n_validos_frontal",
    "n_validos_frontocentral", "n_validos_temporal", "n_validos_posterior",
)

# Los dos componentes del índice de fatiga entran a la matriz con nombre propio
# en vez de `<región>_<banda>_...`, pero son potencia de banda en un clúster.
_ALIAS_COMPONENTES = {
    "theta_frontal_log": ("frontal", "theta"),
    "alfa_posterior_log": ("posterior", "alfa"),
}

SIN_ASIGNAR = "sin_asignar"


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


# --------------------------------------------------------------------------
# Definición de X y preparación de los datos
# --------------------------------------------------------------------------

def verificar_sin_fuga(columnas):
    """
    Comprueba que una lista de columnas no arrastra la etiqueta ni la identidad.

    Falla si aparece `condicion_orden` (la etiqueta codificada), `condicion`, la
    etiqueta `vas` o cualquier columna de identidad. Se llama sobre **cada**
    conjunto de features que se pasa a un `fit`, subconjuntos de ablación
    incluidos, porque una fuga se cuela justo al construir uno a mano.

    Devuelve la misma lista para poder encadenarla.
    """
    prohibidas = set(COLUMNAS_IDENTIDAD) | {COLUMNA_ETIQUETA, COLUMNA_FUGA}
    presentes = sorted(set(columnas) & prohibidas)
    if presentes:
        raise ValueError(
            f"Estas columnas no pueden ser features de OA/OC: {presentes}. "
            f"`{COLUMNA_FUGA}` codifica la propia condición."
        )
    return list(columnas)


def columnas_de_features_reposo(matriz, incluir_procesamiento=False):
    """
    Definición de X para OA/OC: `columnas_de_features()` menos las fugas.

    Resta `condicion_orden` **siempre**, y los seis contadores de procesamiento
    salvo que se pidan con `incluir_procesamiento=True` (análisis de
    sensibilidad). Sobre la matriz de la Fase 13 (77 features) devuelve 70.

    `columnas_de_features()` no se modifica: sigue siendo la definición de X del
    resto del proyecto.
    """
    excluidas = {COLUMNA_FUGA}
    if not incluir_procesamiento:
        excluidas |= set(COLUMNAS_PROCESAMIENTO)
    columnas = [c for c in columnas_de_features(matriz) if c not in excluidas]
    return verificar_sin_fuga(columnas)


def preparar_reposo(matriz, splits=SPLITS_ANALISIS, permitir_test=False):
    """
    Deja la matriz de la Fase 13 lista para OA/OC.

    Se queda con las grabaciones de reposo (OA y OC) de los `splits` pedidos y
    descarta las que no tienen `persona_uid` (la Fase 12 lo deja a NaN en las
    sesiones sin etiqueta de VAS; sin él no se puede agrupar por persona).

    Parámetros
    ----------
    matriz : pandas.DataFrame
        `matriz_rf.parquet`, una fila por grabación.
    splits : iterable de str
        Particiones que entran al análisis. Por defecto `train` y `val`.
    permitir_test : bool
        `test` se reserva para un único análisis confirmatorio pre-registrado:
        pedirlo sin activar esto es un error, no un aviso.

    Devuelve
    --------
    datos : pandas.DataFrame
        Grabaciones de análisis, con el índice reiniciado.
    metadata : dict
        Recuentos: grabaciones, personas, sesiones, reparto por condición y por
        split, y lo que quedó apartado o descartado.
    """
    _validar_columnas(matriz, (COLUMNA_CONDICION, COLUMNA_SPLIT, COLUMNA_GRUPO),
                      "matriz")
    splits = tuple(splits)
    if "test" in splits and not permitir_test:
        raise ValueError(
            "El conjunto de test no se abre en este análisis. Si de verdad hace "
            "falta, hay que pedirlo con `permitir_test=True` y documentarlo."
        )

    reposo = matriz[matriz[COLUMNA_CONDICION].isin(CONDICIONES_REPOSO)]
    en_splits = reposo[reposo[COLUMNA_SPLIT].isin(splits)]
    apartadas = reposo[~reposo[COLUMNA_SPLIT].isin(splits)]
    sin_persona = en_splits[COLUMNA_GRUPO].isna()
    datos = en_splits[~sin_persona].reset_index(drop=True)

    if datos[COLUMNA_CONDICION].nunique() < 2:
        raise ValueError(
            f"Tras filtrar quedan {datos[COLUMNA_CONDICION].nunique()} "
            f"condición(es) de reposo: hacen falta OA y OC."
        )

    metadata = {
        "splits": list(splits),
        "n_grabaciones": int(len(datos)),
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


def etiqueta_binaria(datos):
    """`condicion` -> 1 para OC (ojos cerrados), 0 para OA (ojos abiertos)."""
    condicion = datos[COLUMNA_CONDICION]
    ajenas = sorted(set(condicion.unique()) - set(CONDICIONES_REPOSO))
    if ajenas:
        raise ValueError(
            f"Solo OA y OC tienen etiqueta binaria; hay también: {ajenas}."
        )
    return (condicion == CLASE_POSITIVA).to_numpy(dtype=int)


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------

def metricas_binarias(y_true, y_pred, nombre=None):
    """
    Métricas de la clasificación OA (0) / OC (1), con OC como clase positiva.

    - `exactitud_balanceada`: media de sensibilidad y especificidad. Es la
      métrica principal porque su valor de azar es 0.5 exacto, pase lo que pase
      con el reparto de clases; la exactitud simple premia predecir la mayoritaria.
    - `sensibilidad`: fracción de OC bien detectadas. `especificidad`: fracción
      de OA bien detectadas. Separarlas dice en qué clase falla.
    - `matriz_confusion`: `{vn, fp, fn, vp}`, con las filas de la matriz por
      clase real (OA, OC) y las columnas por clase predicha.
    - `mcc`: coeficiente de correlación de Matthews, 0 en el azar.

    Las filas con NaN en cualquiera de los dos vectores se descartan.
    """
    _validar_longitudes(y_true=y_true, y_pred=y_pred)
    y_true = pd.Series(np.asarray(y_true, dtype=float))
    y_pred = pd.Series(np.asarray(y_pred, dtype=float))
    validas = y_true.notna() & y_pred.notna()
    y_true, y_pred = y_true[validas], y_pred[validas]
    if len(y_true) < 3:
        raise ValueError(
            f"Hacen falta al menos 3 observaciones válidas para calcular "
            f"métricas y hay {len(y_true)}."
        )
    ajenos = (set(y_true.unique()) | set(y_pred.unique())) - {0.0, 1.0}
    if ajenos:
        raise ValueError(f"Las etiquetas deben ser 0 o 1; llegaron {sorted(ajenos)}.")
    if y_true.nunique() < 2:
        raise ValueError(
            "y_true tiene una sola clase: la exactitud balanceada no está "
            "definida."
        )
    yt, yp = y_true.to_numpy(int), y_pred.to_numpy(int)

    vn, fp, fn, vp = (int(v) for v in
                      confusion_matrix(yt, yp, labels=[0, 1]).ravel())
    sensibilidad = vp / (vp + fn)
    especificidad = vn / (vn + fp)
    precision = vp / (vp + fp) if (vp + fp) > 0 else float("nan")
    salida = {
        "n": int(len(yt)),
        "exactitud": float((vn + vp) / len(yt)),
        "exactitud_balanceada": float((sensibilidad + especificidad) / 2),
        "sensibilidad": float(sensibilidad),
        "especificidad": float(especificidad),
        "precision": float(precision),
        "f1": float(2 * vp / (2 * vp + fp + fn)),
        "mcc": float(matthews_corrcoef(yt, yp)),
        "matriz_confusion": {"vn": vn, "fp": fp, "fn": fn, "vp": vp},
    }
    if nombre is not None:
        salida["modelo"] = nombre
    return salida


def auc_roc(y_true, puntuaciones):
    """Área bajo la curva ROC de una puntuación continua para la clase OC."""
    _validar_longitudes(y_true=y_true, puntuaciones=puntuaciones)
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(puntuaciones, dtype=float)
    validas = ~(np.isnan(y) | np.isnan(s))
    y, s = y[validas], s[validas]
    if len(np.unique(y)) < 2:
        raise ValueError("y_true tiene una sola clase: el AUC no está definido.")
    return float(roc_auc_score(y, s))


def pliegue_por_fila(grupos, n_splits=5):
    """
    Número de pliegue (1..n_splits) de cada fila en la CV agrupada.

    `GroupKFold` sin barajar depende solo de los grupos, así que esto reproduce
    exactamente el reparto de `predecir_cv_agrupada`: sirve para puntuar por
    pliegue sin tocar esa función.
    """
    grupos = np.asarray(grupos)
    pliegues = np.full(len(grupos), -1, dtype=int)
    cv = GroupKFold(n_splits=n_splits)
    for i, (_, idx_val) in enumerate(
            cv.split(np.zeros((len(grupos), 1)), groups=grupos), start=1):
        pliegues[idx_val] = i
    return pliegues


def metricas_binarias_por_pliegue(y_true, y_pred, pliegues, puntuaciones=None,
                                  nombre=None):
    """
    Métricas agregadas **y** dentro de cada pliegue.

    Las claves de `metricas_binarias` (`exactitud_balanceada`, `matriz_confusion`,
    ...) son las de las predicciones out-of-fold amontonadas. Las que llevan
    `_pliegues`, `_media` o `_sd` se calculan dentro de cada pliegue.

    A diferencia de la Fase 14b, aquí la etiqueta es la misma en todos los
    pliegues (no se deriva de ningún corte estimado con los datos), así que
    amontonar es legítimo y las dos formas deben dar casi lo mismo. Se reportan
    ambas para comprobarlo en vez de asumirlo: un clasificador constante da
    exactamente 0.5 dentro de cada pliegue, y algo distinto al amontonarlos si
    los pliegues no tienen el mismo reparto de clases.

    Si se pasan `puntuaciones` (probabilidad de OC) se añade el AUC agregado y
    por pliegue.
    """
    _validar_longitudes(y_true=y_true, y_pred=y_pred, pliegues=pliegues)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    pliegues = np.asarray(pliegues)
    if puntuaciones is not None:
        puntuaciones = np.asarray(puntuaciones, dtype=float)
        _validar_longitudes(y_true=y_true, puntuaciones=puntuaciones)

    salida = metricas_binarias(y_true, y_pred, nombre)
    por_pliegue = []
    for p in sorted(set(pliegues.tolist())):
        m = pliegues == p
        fila = {"pliegue": int(p), "n": int(m.sum()),
                "exactitud_balanceada":
                    metricas_binarias(y_true[m], y_pred[m])["exactitud_balanceada"]}
        if puntuaciones is not None:
            fila["auc"] = auc_roc(y_true[m], puntuaciones[m])
        por_pliegue.append(fila)

    ba = np.array([f["exactitud_balanceada"] for f in por_pliegue])
    salida["n_pliegues"] = len(por_pliegue)
    salida["por_pliegue"] = por_pliegue
    salida["exactitud_balanceada_pliegues"] = ba.tolist()
    salida["exactitud_balanceada_media"] = float(ba.mean())
    salida["exactitud_balanceada_sd"] = float(ba.std(ddof=1)) if len(ba) > 1 else 0.0
    if puntuaciones is not None:
        auc = np.array([f["auc"] for f in por_pliegue])
        salida["auc_agregado"] = auc_roc(y_true, puntuaciones)
        salida["auc_pliegues"] = auc.tolist()
        salida["auc_media"] = float(auc.mean())
        salida["auc_sd"] = float(auc.std(ddof=1)) if len(auc) > 1 else 0.0
    return salida


# --------------------------------------------------------------------------
# Estimadores auxiliares y validación cruzada
# --------------------------------------------------------------------------

class ProbabilidadComoPrediccion(BaseEstimator):
    """
    Envuelve un clasificador para que `predict` devuelva P(clase 1).

    Permite obtener probabilidades out-of-fold **reutilizando
    `predecir_cv_agrupada` tal cual**, con su comprobación de que ninguna persona
    cae a ambos lados de la partición, en vez de duplicar el bucle de
    `GroupKFold`. No hereda de `ClassifierMixin` a propósito: `predict` no
    devuelve etiquetas, y el mixin haría que sklearn lo tratara como si lo
    hiciera.
    """

    def __init__(self, estimador):
        self.estimador = estimador

    def fit(self, X, y):
        """Ajusta una copia del clasificador y localiza la columna de la clase 1."""
        self.estimador_ = clone(self.estimador).fit(X, np.asarray(y))
        clases = list(self.estimador_.classes_)
        if 1 not in clases:
            raise ValueError(
                f"La clase positiva (1) no está en el entrenamiento: {clases}."
            )
        self.columna_positiva_ = clases.index(1)
        return self

    def predict(self, X):
        """Devuelve P(clase 1) para cada fila de `X`, no una etiqueta."""
        return self.estimador_.predict_proba(X)[:, self.columna_positiva_]


def evaluar_reposo(X, y, grupos, modelo, nombre=None, n_splits=5,
                   con_probabilidad=True):
    """
    Validación cruzada agrupada por persona y métricas de OA/OC en un solo paso.

    Es el punto de entrada de la evaluación: `predecir_cv_agrupada` para las
    predicciones out-of-fold y `metricas_binarias_por_pliegue` para puntuarlas.

    Con `con_probabilidad=True` el modelo necesita `predict_proba`; la etiqueta
    se obtiene como `P(OC) > 0.5`, que coincide con `predict` de un clasificador
    de sklearn salvo en el empate exacto. Con `False` se usa `predict` (para
    estimadores que no dan probabilidad) y no hay AUC.

    Devuelve
    --------
    dict con `y_true`, `y_pred`, `puntuaciones` (o None), `pliegues`,
    `metricas` (ver `metricas_binarias_por_pliegue`) y `detalle` (el de
    `predecir_cv_agrupada`).
    """
    X = pd.DataFrame(X)
    y = np.asarray(y)
    grupos = np.asarray(grupos)
    if con_probabilidad:
        puntuaciones, detalle = predecir_cv_agrupada(
            X, y, grupos, ProbabilidadComoPrediccion(modelo), n_splits=n_splits)
        pred = (puntuaciones > 0.5).astype(int)
    else:
        pred, detalle = predecir_cv_agrupada(X, y, grupos, modelo,
                                             n_splits=n_splits)
        puntuaciones = None
    pliegues = pliegue_por_fila(grupos, n_splits)
    return {
        "y_true": y, "y_pred": pred, "puntuaciones": puntuaciones,
        "pliegues": pliegues, "detalle": detalle,
        "metricas": metricas_binarias_por_pliegue(
            y, pred, pliegues, puntuaciones, nombre),
    }


# --------------------------------------------------------------------------
# Referencia de una sola variable
# --------------------------------------------------------------------------

def umbral_una_variable(x, y):
    """
    Corte sobre una sola variable que maximiza la exactitud balanceada.

    Devuelve `(umbral, signo)`: con `signo = +1` la regla es «OC si x > umbral»
    y con `-1` es «OC si x < umbral». El signo **se aprende**, no se impone:
    que salga +1 con el alfa posterior es una comprobación neurofisiológica, y
    no puede serlo si se fijó de antemano. El umbral es el punto medio entre dos
    valores distintos consecutivos, así que nunca coincide con una observación.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    _validar_longitudes(x=x, y=y)
    validas = ~np.isnan(x)
    x, y = x[validas], (y[validas] == 1).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("Hacen falta las dos clases para estimar el umbral.")

    orden = np.argsort(x, kind="stable")
    xs, ys = x[orden], y[orden]
    cortes = np.flatnonzero(xs[1:] > xs[:-1])
    if len(cortes) == 0:
        raise ValueError("La variable es constante: no hay umbral posible.")

    n_pos = ys.sum()
    n_neg = len(ys) - n_pos
    # Regla «1 si x > t» con t entre xs[i] y xs[i+1]: los positivos con x <= xs[i]
    # son los falsos negativos y los negativos con x <= xs[i] los verdaderos.
    sensibilidad = (n_pos - np.cumsum(ys)[cortes]) / n_pos
    especificidad = np.cumsum(1 - ys)[cortes] / n_neg
    ba = (sensibilidad + especificidad) / 2
    mejor = int(np.argmax(np.abs(ba - 0.5)))
    i = cortes[mejor]
    umbral = (xs[i] + xs[i + 1]) / 2
    return float(umbral), (1 if ba[mejor] >= 0.5 else -1)


def clasificar_por_umbral(x, umbral, signo=1):
    """Aplica la regla de `umbral_una_variable`: 1 (OC) o 0 (OA) por cada `x`."""
    x = np.asarray(x, dtype=float)
    return (signo * (x - umbral) > 0).astype(int)


class UmbralUnaVariable(BaseEstimator):
    """
    Clasificador de un solo corte sobre una columna, ajustado por pliegue.

    Es la referencia neurofisiológica mínima: si una sola feature (el alfa
    posterior) basta, el modelo multivariante no añade nada. Estimador de sklearn
    para que `predecir_cv_agrupada` lo ajuste con el pliegue de entrenamiento y
    lo aplique al de validación, como a cualquier otro modelo. Los NaN se
    sustituyen por la mediana del entrenamiento.
    """

    def __init__(self, columna):
        self.columna = columna

    def _valores(self, X):
        if not isinstance(X, pd.DataFrame) or self.columna not in X.columns:
            raise ValueError(
                f"UmbralUnaVariable necesita un DataFrame con la columna "
                f"'{self.columna}'."
            )
        return X[self.columna].to_numpy(dtype=float)

    def fit(self, X, y):
        """Aprende el corte y su signo, y guarda la mediana para rellenar los NaN."""
        x = self._valores(X)
        self.umbral_, self.signo_ = umbral_una_variable(x, y)
        self.mediana_ = float(np.nanmedian(x))
        return self

    def predict(self, X):
        """Clasifica cada fila con el corte aprendido: 1 (OC) o 0 (OA)."""
        x = self._valores(X)
        x = np.where(np.isnan(x), self.mediana_, x)
        return clasificar_por_umbral(x, self.umbral_, self.signo_)


# --------------------------------------------------------------------------
# Interpretación: qué pesa en el modelo
# --------------------------------------------------------------------------

def bloque_de_feature(nombre, regiones=None, bandas=None):
    """
    `(región, banda)` de una feature, o `(None, None)` si no es de potencia.

    Reconoce `<región>_<banda>_...` y los dos componentes del índice de fatiga
    (`theta_frontal_log_*` es theta frontal; `alfa_posterior_log_*`, alfa
    posterior). `indice_fatiga_*` mezcla dos bandas y dos regiones y queda sin
    asignar, igual que los contadores de procesamiento.

    `regiones` y `bandas` valen por defecto `REGIONES` y `BANDAS_DEFECTO`; pasar
    los nombres de los canales como `regiones` da la versión por canal.
    """
    regiones = tuple(regiones if regiones is not None else REGIONES)
    bandas = tuple(bandas if bandas is not None else BANDAS_DEFECTO)
    partes = nombre.split("_")
    if len(partes) >= 2 and partes[0] in regiones and partes[1] in bandas:
        return partes[0], partes[1]
    for prefijo, (region, banda) in _ALIAS_COMPONENTES.items():
        if (nombre.startswith(prefijo + "_")
                and region in regiones and banda in bandas):
            return region, banda
    return None, None


def columnas_del_bloque(columnas, region=None, banda=None, regiones=None,
                        bandas=None):
    """Columnas de `columnas` de una región y/o banda (`None` = cualquiera)."""
    elegidas = []
    for c in columnas:
        r, b = bloque_de_feature(c, regiones, bandas)
        if region is not None and r != region:
            continue
        if banda is not None and b != banda:
            continue
        elegidas.append(c)
    return elegidas


def importancia_por_bloque(importancias, regiones=None, bandas=None):
    """
    Reparte una importancia por feature entre regiones y bandas.

    Convierte un ranking de 70 números en la afirmación que interesa («el modelo
    se apoya en el alfa posterior»). Las importancias negativas (una permutación
    que mejora el modelo por azar) cuentan como 0. Las fracciones son sobre el
    **total**, incluidas las features sin región/banda, que aparecen como
    `sin_asignar`; así cada reparto suma 1 y no se infla una banda descartando
    lo que no encaja.

    Devuelve
    --------
    dict con `total` y las fracciones `por_region`, `por_banda` y `por_bloque`
    (clave `"<región>/<banda>"`).
    """
    serie = pd.Series(importancias, dtype=float).clip(lower=0.0)
    total = float(serie.sum())
    if total <= 0:
        raise ValueError("Todas las importancias son 0: no hay nada que repartir.")

    por_region, por_banda, por_bloque = {}, {}, {}
    for nombre, valor in serie.items():
        region, banda = bloque_de_feature(nombre, regiones, bandas)
        clave_r = region or SIN_ASIGNAR
        clave_b = banda or SIN_ASIGNAR
        clave = f"{clave_r}/{clave_b}"
        por_region[clave_r] = por_region.get(clave_r, 0.0) + valor / total
        por_banda[clave_b] = por_banda.get(clave_b, 0.0) + valor / total
        por_bloque[clave] = por_bloque.get(clave, 0.0) + valor / total
    return {"total": total, "por_region": por_region, "por_banda": por_banda,
            "por_bloque": por_bloque}


def tamanos_de_efecto(X, y):
    """
    d de Cohen de cada columna entre la clase 1 (OC) y la 0 (OA).

    Positivo = la feature es mayor con los ojos cerrados. Usa la desviación
    típica agrupada ponderada por tamaño y descarta los NaN columna a columna;
    da NaN donde no hay varianza.
    """
    X = pd.DataFrame(X)
    y = np.asarray(y)
    _validar_longitudes(X=X, y=y)
    uno, cero = X[y == 1], X[y == 0]
    if len(uno) < 2 or len(cero) < 2:
        raise ValueError("Hacen falta al menos 2 observaciones de cada clase.")
    n1, n0 = uno.count(), cero.count()
    gl = (n1 + n0 - 2).where(lambda s: s > 0)
    agrupada = np.sqrt(((n1 - 1) * uno.var(ddof=1)
                        + (n0 - 1) * cero.var(ddof=1)) / gl)
    d = (uno.mean() - cero.mean()) / agrupada.where(agrupada > 0)
    d.name = "d_cohen"
    return d


def _dividir_por_persona(X, y, grupos, n_splits):
    """Pliegues de `GroupKFold` con la comprobación de que no comparten personas."""
    cv = GroupKFold(n_splits=n_splits)
    for i, (idx_tr, idx_va) in enumerate(cv.split(X, y, groups=grupos)):
        if set(grupos[idx_tr]) & set(grupos[idx_va]):
            raise ValueError(
                f"El pliegue {i + 1} comparte personas entre train y validación."
            )
        yield i, idx_tr, idx_va


def importancia_permutacion_cv(X, y, grupos, modelo, bloques=None, n_splits=5,
                               n_repeticiones=10, semilla=42):
    """
    Importancia por permutación **out-of-fold**, por columna o por bloque.

    En cada pliegue ajusta el modelo con el entrenamiento y mide cuánto cae la
    exactitud balanceada de *validación* al barajar una columna (o un bloque
    entero de columnas a la vez). Es la importancia de lo que el modelo usa para
    generalizar a gente nueva; la de Gini de un `fit` sobre todos los datos
    sobrevalora las variables con muchos valores distintos y mide lo aprendido
    sobre el propio entrenamiento.

    `bloques` es un dict `{nombre: [columnas]}`. Con `None` cada columna es su
    propio bloque. Un bloque se baraja **con la misma permutación en todas sus
    columnas**, conservando su estructura interna. Esto importa con features tan
    correlacionadas como las de aquí: si dos columnas casi idénticas llevan la
    misma señal, barajar una sola apenas cae porque la otra la cubre, y la
    importancia individual se diluye. Barajar el bloque entero mide lo que se
    pierde de verdad.

    Devuelve
    --------
    pandas.DataFrame indexado por bloque con `caida_media` (puntos de exactitud
    balanceada perdidos, media entre pliegues) y `sd_entre_pliegues`.
    """
    X = pd.DataFrame(X)
    y = np.asarray(y, dtype=float)
    grupos = np.asarray(grupos)
    _validar_longitudes(X=X, y=y, grupos=grupos)
    if bloques is None:
        bloques = {c: [c] for c in X.columns}
    for nombre, columnas in bloques.items():
        faltan = [c for c in columnas if c not in X.columns]
        if faltan or not columnas:
            raise ValueError(f"El bloque '{nombre}' está vacío o usa columnas "
                             f"que no existen: {faltan}.")

    caidas = {nombre: [] for nombre in bloques}
    for i, idx_tr, idx_va in _dividir_por_persona(X, y, grupos, n_splits):
        ajustado = clone(modelo).fit(X.iloc[idx_tr], y[idx_tr])
        X_va, y_va = X.iloc[idx_va], y[idx_va]
        base = balanced_accuracy_score(y_va, ajustado.predict(X_va))
        rng = np.random.default_rng(semilla + i)
        for nombre, columnas in bloques.items():
            valores = X_va[columnas].to_numpy()
            perdido = 0.0
            for _ in range(n_repeticiones):
                barajado = X_va.copy()
                barajado[columnas] = valores[rng.permutation(len(X_va))]
                perdido += base - balanced_accuracy_score(
                    y_va, ajustado.predict(barajado))
            caidas[nombre].append(perdido / n_repeticiones)

    tabla = pd.DataFrame(caidas).T
    return pd.DataFrame({
        "caida_media": tabla.mean(axis=1),
        "sd_entre_pliegues": tabla.std(axis=1, ddof=1),
    })


def coeficientes_por_pliegue(X, y, grupos, modelo, n_splits=5):
    """
    Coeficientes de un modelo lineal, uno por pliegue de la CV agrupada.

    `modelo` es un clasificador lineal o un `Pipeline` cuyo último paso lo es; si
    lleva `StandardScaler` los coeficientes están en unidades de desviación
    típica y son comparables entre features. Devuelve un DataFrame
    features × pliegues: el signo estable entre pliegues es lo que permite
    comprobar la **dirección** del efecto, que una importancia (sin signo) no da.
    """
    X = pd.DataFrame(X)
    y = np.asarray(y, dtype=float)
    grupos = np.asarray(grupos)
    _validar_longitudes(X=X, y=y, grupos=grupos)
    columnas = {}
    for i, idx_tr, _ in _dividir_por_persona(X, y, grupos, n_splits):
        ajustado = clone(modelo).fit(X.iloc[idx_tr], y[idx_tr])
        final = ajustado[-1] if hasattr(ajustado, "steps") else ajustado
        coef = np.asarray(final.coef_, dtype=float).ravel()
        if len(coef) != X.shape[1]:
            raise ValueError(
                f"El modelo devolvió {len(coef)} coeficientes para "
                f"{X.shape[1]} columnas: un paso previo descartó alguna."
            )
        columnas[f"pliegue_{i + 1}"] = coef
    return pd.DataFrame(columnas, index=X.columns)


# --------------------------------------------------------------------------
# Controles e incertidumbre
# --------------------------------------------------------------------------

def permutar_etiquetas_dentro_de_persona(y, grupos, semilla=42):
    """
    Baraja las etiquetas **dentro de cada persona**, conservando cuántas de cada
    clase tiene.

    Es el control negativo: destruye la relación entre las features y OA/OC pero
    deja intacta la estructura de personas y el reparto de clases por persona.
    Una CV agrupada correcta tiene que devolver ~0.5 de exactitud balanceada; si
    devolviera más, habría una fuga en el propio montaje.
    """
    y = np.asarray(y)
    grupos = np.asarray(grupos)
    _validar_longitudes(y=y, grupos=grupos)
    rng = np.random.default_rng(semilla)
    salida = y.copy()
    for g in pd.unique(grupos):
        idx = np.flatnonzero(grupos == g)
        salida[idx] = y[rng.permutation(idx)]
    return salida


def bootstrap_por_persona(y_true, valores, grupos, metrica, n_replicas=2000,
                          semilla=42):
    """
    IC al 95 % de una métrica remuestreando **personas**, no grabaciones.

    Las grabaciones de una misma persona no son independientes (cada una aporta
    ~7), así que remuestrear filas daría un intervalo falsamente estrecho.
    `metrica(y_true, valores)` debe devolver un número; las réplicas donde no
    está definida (`ValueError` o NaN) se saltan y se cuentan aparte.

    Devuelve
    --------
    dict con `ic95` (percentiles 2.5 y 97.5), `n_replicas` (válidas),
    `n_descartadas` y `replicas` (el array, por si se quiere inspeccionar).
    """
    y_true = np.asarray(y_true)
    valores = np.asarray(valores)
    grupos = np.asarray(grupos)
    _validar_longitudes(y_true=y_true, valores=valores, grupos=grupos)
    return _bootstrap_personas(
        grupos, lambda filas: metrica(y_true[filas], valores[filas]),
        n_replicas, semilla)


def bootstrap_diferencia_por_persona(y_true, valores_a, valores_b, grupos,
                                     metrica, n_replicas=2000, semilla=42):
    """
    IC al 95 % de `metrica(A) - metrica(B)` remuestreando **personas**, con las
    mismas personas en las dos ramas de cada réplica.

    Es la comparación **pareada** de dos modelos evaluados sobre las mismas
    grabaciones. Mirar si se solapan los IC de cada uno por separado no sirve:
    ambos comparten las personas (y por tanto su dificultad), y esa parte común
    hace que la diferencia sea mucho más estable que cada cifra. Con la misma
    remuestra en A y en B esa covarianza se cancela, como debe.

    Devuelve un dict con `diferencia` (sobre todos los datos), `ic95`,
    `prop_positivas` (fracción de réplicas con A > B), `n_replicas`,
    `n_descartadas` y `replicas`.
    """
    y_true = np.asarray(y_true)
    valores_a = np.asarray(valores_a)
    valores_b = np.asarray(valores_b)
    grupos = np.asarray(grupos)
    _validar_longitudes(y_true=y_true, valores_a=valores_a, valores_b=valores_b,
                        grupos=grupos)
    todas = np.arange(len(y_true))

    def diferencia(filas):
        return (metrica(y_true[filas], valores_a[filas])
                - metrica(y_true[filas], valores_b[filas]))

    salida = _bootstrap_personas(grupos, diferencia, n_replicas, semilla)
    salida["diferencia"] = float(diferencia(todas))
    salida["prop_positivas"] = float(np.mean(salida["replicas"] > 0))
    return salida


def _bootstrap_personas(grupos, estadistico, n_replicas, semilla):
    """Núcleo común: remuestrea personas y evalúa `estadistico(filas)`."""
    rng = np.random.default_rng(semilla)
    unicos = pd.unique(grupos)
    indices = {g: np.flatnonzero(grupos == g) for g in unicos}

    obtenidas, descartadas = [], 0
    for _ in range(n_replicas):
        elegidas = rng.choice(unicos, size=len(unicos), replace=True)
        filas = np.concatenate([indices[g] for g in elegidas])
        try:
            valor = float(estadistico(filas))
        except ValueError:
            descartadas += 1
            continue
        if np.isnan(valor):
            descartadas += 1
            continue
        obtenidas.append(valor)
    if not obtenidas:
        raise ValueError("Ninguna réplica del bootstrap tuvo la métrica definida.")
    replicas = np.array(obtenidas)
    return {
        "ic95": (float(np.percentile(replicas, 2.5)),
                 float(np.percentile(replicas, 97.5))),
        "n_replicas": int(len(replicas)),
        "n_descartadas": int(descartadas),
        "replicas": replicas,
    }


def columnas_duplicadas(X):
    """
    Pares `(conservada, duplicada)` de columnas **idénticas** valor a valor.

    Dos columnas son duplicadas si coinciden en todas las filas, con los NaN en
    las mismas posiciones. No detecta columnas casi iguales ni correlacionadas,
    solo copias exactas. Importa porque los duplicados no añaden información y sí
    distorsionan lo que se lee de un modelo: reparten la importancia entre las
    copias, reparten el peso de una regresión regularizada y hacen que un
    Random Forest muestree esa señal más veces.
    """
    X = pd.DataFrame(X)
    vistas, pares = {}, []
    for c in X.columns:
        huella = pd.util.hash_pandas_object(X[c], index=False).to_numpy().tobytes()
        if huella in vistas and X[c].equals(X[vistas[huella]]):
            pares.append((vistas[huella], c))
        else:
            vistas.setdefault(huella, c)
    return pares
