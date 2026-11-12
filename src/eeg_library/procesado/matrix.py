"""
Módulo de construcción de la matriz de features del Random Forest (Fase 13).

Convierte la matriz por época de la Fase 11.5 (280 339 filas × 139 columnas) en
una matriz por **grabación** lista para `RandomForestRegressor`, con la
etiqueta VAS de la Fase 12 ya unida.

Decisiones de diseño (justificación completa en ESTADO.md, Fase 13):

- **Se agrega por región, no por canal.** Un canal que `evaluar_calidad`
  descartó tiene sus columnas de potencia calculadas sobre señal que no pasó
  por CAR ni por ICA, y con valores fuera de escala por factores de hasta 7.9x
  (Fase 11.5). Promediando por región y **restringiendo el promedio a los
  canales válidos**, un electrodo seco degrada la precisión de su región en vez
  de meter un valor basura en la matriz. Además baja de 112 columnas de
  potencia a 32, que con 653 etiquetas independientes importa.
- **La potencia absoluta se agrega en escala logarítmica.** Los valores del
  dataset van de 4e-38 a 3e-7 W/Hz: una media aritmética sobre las ventanas de
  una grabación la decidiría la ventana más ruidosa. La media del logaritmo es
  la media geométrica, que es la forma estándar de promediar potencia espectral
  y hace que la desviación típica sea una medida de variabilidad relativa en
  vez de una función del nivel. La potencia **relativa** ya está acotada en
  [0, 1] y se agrega tal cual.
- **Media y desviación típica a lo largo de las ventanas.** La media es el
  nivel; la desviación es cuánto fluctúa la grabación, que es un candidato a
  marcador de fatiga por derecho propio y se perdería al promediar.
- **Una región sin ningún canal válido produce NaN, no un valor inventado.**
  Pasa en el 2.9 % de las grabaciones, siempre en las regiones de dos
  electrodos (temporal 2.3 %, frontocentral 0.6 %); frontal y posterior nunca
  se quedan vacías. `RandomForestRegressor` admite NaN desde scikit-learn 1.4,
  así que imputar sería inventar datos sin necesidad. Cada región lleva además
  su `n_validos_<region>`, para que el modelo pueda ver la degradación.
- **`medicion` no es una feature.** Está en la matriz como columna de
  identidad, para estratificar el reporte y calcular líneas base, pero
  `columnas_de_features` no la devuelve. Medido sobre las 653 sesiones, el
  momento del curso explica el 1.5 % de la varianza del VAS, así que excluirlo
  no cuesta nada; lo que se gana es poder afirmar que el modelo se explica sin
  el calendario.
- **`condicion` sí es una feature**, codificada por su posición en el
  protocolo. Las 7 grabaciones de una sesión comparten el mismo VAS, así que la
  condición está balanceada dentro de cada sesión y no puede filtrar la
  etiqueta; lo que aporta es el contexto para leer la potencia de banda
  relativa a la tarea.
"""

import numpy as np
import pandas as pd

from eeg_library.procesado.bands import BANDAS_DEFECTO
from eeg_library.procesado.features import COLUMNA_N_BUENOS, PREFIJO_VALIDEZ

# Regiones sobre los 14 electrodos del Epoc X. Las dos primeras son los
# clústeres que la Fase 9 ya usa para el índice theta/alfa; las otras dos
# recogen los cuatro electrodos que quedaban fuera.
REGIONES = {
    "frontal": ("AF3", "AF4", "F3", "F4", "F7", "F8"),
    "frontocentral": ("FC5", "FC6"),
    "temporal": ("T7", "T8"),
    "posterior": ("P7", "P8", "O1", "O2"),
}

# Orden fijo del protocolo dentro de cada sesión. Se usa para codificar la
# condición como entero: es la codificación que respeta que PVSAT viene
# siempre después de MSIT, cosa que un one-hot perdería.
ORDEN_PROTOCOLO = ("OC", "OA", "STROOP", "PASAT", "PVTB", "MSIT", "PVSAT")

# Columnas de identidad y de etiqueta. Viajan en la matriz porque hacen falta
# para agrupar, estratificar y evaluar, pero NO son entradas del modelo.
COLUMNAS_IDENTIDAD = (
    "archivo", "sesion_uid", "persona_uid", "sujeto_id", "centro", "medicion",
    "condicion", "split", "etiquetada", "concordancia_fecha", "corregida",
    "ocupante", "procesada",
)
COLUMNA_ETIQUETA = "vas"

NOMBRE_MATRIZ = "matriz_rf.parquet"


def _validar_columnas(df, columnas, nombre):
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise ValueError(f"A '{nombre}' le faltan columnas necesarias: {faltan}.")


def columnas_de_features(matriz):
    """
    Devuelve las columnas que son entrada del modelo, en orden estable.

    Es la única definición de "X" del proyecto: todo lo que no esté aquí no
    entra al `fit`. Existe como función y no como convención mental porque la
    exclusión de `medicion` es una decisión de diseño que hay que poder
    comprobar con un test, no recordar (ver la nota de este módulo).
    """
    excluidas = set(COLUMNAS_IDENTIDAD) | {COLUMNA_ETIQUETA}
    return [c for c in matriz.columns if c not in excluidas]


# La etiqueta disfrazada de los modelos de condición: posición de la condición
# en ORDEN_PROTOCOLO. Se separa de las demás porque su exclusión no se discute,
# es una fuga.
COLUMNA_FUGA = "condicion_orden"

# Metadatos del procesamiento, no señal cerebral. Algunos discriminan OA/OC por
# una vía que no es el alfa (presumiblemente la calidad de contacto).
COLUMNAS_PROCESAMIENTO = (
    "n_ventanas", "n_canales_buenos", "n_validos_frontal",
    "n_validos_frontocentral", "n_validos_temporal", "n_validos_posterior",
)


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

    Es también la X de reposo/tarea, de las tareas ejecutivas y de los modelos
    c1-c3 que usa la plataforma (`analizar_grabacion`). Vive aquí, junto a
    `columnas_de_features()`, para que el procesado de una grabación no
    dependa de los módulos de análisis; `modelado/reposo.py`, donde se
    justifica cada exclusión, la reexporta con el mismo nombre.

    `columnas_de_features()` no se modifica: sigue siendo la definición de X del
    resto del proyecto.
    """
    excluidas = {COLUMNA_FUGA}
    if not incluir_procesamiento:
        excluidas |= set(COLUMNAS_PROCESAMIENTO)
    columnas = [c for c in columnas_de_features(matriz) if c not in excluidas]
    return verificar_sin_fuga(columnas)


def agregar_por_region(features, regiones=None, bandas=None):
    """
    Promedia la potencia por región **sobre los canales válidos** de cada una.

    Trabaja fila a fila (una fila = una época). La máscara de la Fase 11.5 es
    por archivo, así que dentro de una grabación el conjunto de canales que
    entran en cada región es constante; aquí no hace falta saberlo, porque las
    columnas `canal_valido_*` ya vienen replicadas en cada época.

    Parámetros
    ----------
    features : pandas.DataFrame
        Matriz por época con la máscara de validez ya anotada
        (`matriz_modelado.parquet`, salida de la Fase 11.5).
    regiones : dict[str, tuple[str, ...]] o None
        Mapa región -> canales. Por defecto `REGIONES`. Pasar
        `{c: (c,) for c in CANALES_EPOC}` da la variante por canal.
    bandas : iterable[str] o None
        Bandas a agregar. Por defecto las de `BANDAS_DEFECTO`.

    Devuelve
    --------
    agregado : pandas.DataFrame
        Una fila por época, con `archivo_origen`, `log10` de la potencia
        absoluta media de cada región × banda, la potencia relativa media, y
        `n_validos_<region>`. Una región sin canales válidos da NaN.
    """
    regiones = dict(regiones or REGIONES)
    bandas = list(bandas or BANDAS_DEFECTO)
    if "archivo_origen" not in features.columns:
        raise ValueError("'features' necesita la columna 'archivo_origen'.")

    salida = {"archivo_origen": features["archivo_origen"].to_numpy()}
    for region, canales in regiones.items():
        columnas_validez = [f"{PREFIJO_VALIDEZ}{c}" for c in canales]
        _validar_columnas(features, columnas_validez, "features")
        mascara = features[columnas_validez].to_numpy(dtype=float)
        n_validos = mascara.sum(axis=1)
        salida[f"n_validos_{region}"] = n_validos.astype("int8")
        # Sin canales válidos no hay valor: se deja NaN en vez de imputar.
        divisor = np.where(n_validos > 0, n_validos, np.nan)

        for banda in bandas:
            for tipo, prefijo in (("logabs", "pot_abs"), ("rel", "pot_rel")):
                columnas = [f"{prefijo}_{c}_{banda}" for c in canales]
                _validar_columnas(features, columnas, "features")
                valores = features[columnas].to_numpy(dtype=float)
                media = (valores * mascara).sum(axis=1) / divisor
                salida[f"{region}_{banda}_{tipo}"] = (
                    np.log10(media) if tipo == "logabs" else media)
    return pd.DataFrame(salida, index=features.index)


def resumir_por_grabacion(features, regiones=None, bandas=None):
    """
    Resume la matriz por época en una fila por grabación, sin etiquetas.

    Es la parte de `construir_matriz_rf` que solo necesita la señal: no usa
    VAS, condición ni ninguna clave del estudio. Por eso sirve igual para el
    dataset completo que para un EDF suelto (`analizar_grabacion`), y las
    features de una grabación nueva salen del **mismo código** que construyó
    `matriz_rf.parquet`.

    Parámetros
    ----------
    features : pandas.DataFrame
        Matriz por época con la máscara de validez ya anotada (Fase 11.5):
        `archivo_origen`, columnas de potencia por canal, `canal_valido_*` y
        `n_canales_buenos`.
    regiones, bandas
        Ver `agregar_por_region`.

    Devuelve
    --------
    resumen : pandas.DataFrame
        Una fila por `archivo` (en el orden de aparición), con la media y la
        desviación típica por región × banda, las de los componentes del
        índice theta/alfa, `n_ventanas`, `n_validos_<region>` y
        `n_canales_buenos`.
    """
    _validar_columnas(features, ("archivo_origen", COLUMNA_N_BUENOS), "features")

    regionales = agregar_por_region(features, regiones=regiones, bandas=bandas)
    columnas_region = [c for c in regionales.columns
                       if c not in ("archivo_origen",) and not c.startswith("n_validos_")]
    columnas_validos = [c for c in regionales.columns if c.startswith("n_validos_")]

    # Las dos potencias del índice de fatiga son potencias absolutas: se
    # agregan en log, igual que las regionales. El índice en sí se deja en
    # crudo porque es el biomarcador con nombre y su valor tiene lectura
    # directa (>1 = predominio theta).
    extra = pd.DataFrame(index=features.index)
    for columna in ("theta_frontal", "alfa_posterior"):
        if columna in features.columns:
            extra[f"{columna}_log"] = np.log10(features[columna].to_numpy(dtype=float))
    if "indice_fatiga_theta_alfa" in features.columns:
        extra["indice_fatiga"] = features["indice_fatiga_theta_alfa"]

    juntas = pd.concat([regionales, extra], axis=1)
    columnas_continuas = columnas_region + list(extra.columns)

    agrupado = juntas.groupby("archivo_origen", sort=False)
    resumen = agrupado[columnas_continuas].agg(["mean", "std"])
    resumen.columns = [f"{col}_{'media' if est == 'mean' else 'sd'}"
                       for col, est in resumen.columns]
    # `std` con una sola ventana daría NaN; con ddof=0 daría 0, que es lo
    # correcto (no hay variabilidad observable) y evita un NaN espurio.
    una_sola = agrupado.size() == 1
    if una_sola.any():
        for columna in columnas_continuas:
            resumen.loc[una_sola[una_sola].index, f"{columna}_sd"] = 0.0

    resumen["n_ventanas"] = agrupado.size()
    resumen[columnas_validos] = agrupado[columnas_validos].first()
    resumen[COLUMNA_N_BUENOS] = (
        features.groupby("archivo_origen", sort=False)[COLUMNA_N_BUENOS].first())
    return resumen.reset_index().rename(columns={"archivo_origen": "archivo"})


def construir_matriz_rf(features, grabaciones, regiones=None, bandas=None):
    """
    Construye la matriz por grabación para el Random Forest.

    Parámetros
    ----------
    features : pandas.DataFrame
        Matriz por época de la Fase 11.5 (`matriz_modelado.parquet`).
    grabaciones : pandas.DataFrame
        `labels_grabaciones.csv` (Fase 12): etiqueta y claves por grabación.
    regiones, bandas
        Ver `agregar_por_region`.

    Devuelve
    --------
    matriz : pandas.DataFrame
        Una fila por grabación presente en `features`. Columnas de identidad
        (`COLUMNAS_IDENTIDAD`), etiqueta (`vas`) y features: por cada región ×
        banda, la media y la desviación típica a lo largo de las ventanas de la
        grabación, tanto del log de la potencia absoluta como de la relativa;
        más el índice de fatiga y sus dos componentes, los canales válidos por
        región, el número de ventanas y la condición codificada.
    metadata : dict
        Recuentos y composición de la matriz.
    """
    _validar_columnas(features, ("archivo_origen", "condicion", COLUMNA_N_BUENOS),
                      "features")
    _validar_columnas(grabaciones, ("archivo", COLUMNA_ETIQUETA), "grabaciones")

    # La parte que solo depende de la señal es común con `analizar_grabacion`.
    resumen = resumir_por_grabacion(features, regiones=regiones, bandas=bandas)

    condicion = features.groupby("archivo_origen", sort=False)["condicion"].first()
    resumen["condicion"] = resumen["archivo"].map(condicion)
    orden = {c: i + 1 for i, c in enumerate(ORDEN_PROTOCOLO)}
    resumen["condicion_orden"] = resumen["condicion"].map(orden).astype("float")
    desconocidas = sorted(set(resumen.loc[resumen["condicion_orden"].isna(), "condicion"]))
    if desconocidas:
        raise ValueError(
            f"Condiciones que no están en ORDEN_PROTOCOLO: {desconocidas}. Sin "
            f"posición en el protocolo no se pueden codificar."
        )

    n_antes = len(resumen)
    columnas_etiqueta = [c for c in COLUMNAS_IDENTIDAD if c in grabaciones.columns
                         and c != "condicion"] + [COLUMNA_ETIQUETA]
    matriz = resumen.merge(grabaciones[["archivo"] + [c for c in columnas_etiqueta
                                                      if c != "archivo"]],
                           on="archivo", how="left")
    if len(matriz) != n_antes:
        raise ValueError(
            f"La unión con las etiquetas ha cambiado el número de grabaciones "
            f"({n_antes} -> {len(matriz)}); habría 'archivo' repetido."
        )
    sin_etiqueta = matriz[COLUMNA_ETIQUETA].isna().sum()

    identidad = [c for c in COLUMNAS_IDENTIDAD if c in matriz.columns]
    matriz = matriz[identidad + [COLUMNA_ETIQUETA]
                    + [c for c in matriz.columns
                       if c not in identidad and c != COLUMNA_ETIQUETA]]

    features_x = columnas_de_features(matriz)
    metadata = {
        "metodo": "agregacion_por_region_y_grabacion",
        "regiones": {r: list(c) for r, c in (regiones or REGIONES).items()},
        "bandas": list(bandas or BANDAS_DEFECTO),
        "n_grabaciones": len(matriz),
        "n_features": len(features_x),
        "columnas_features": features_x,
        "n_grabaciones_sin_etiqueta": int(sin_etiqueta),
        "n_celdas_nan": int(matriz[features_x].isna().sum().sum()),
        "columnas_con_nan": {c: int(n) for c, n in
                             matriz[features_x].isna().sum().items() if n},
    }
    print(
        f"Matriz RF: {len(matriz)} grabación(es) × {len(features_x)} feature(s). "
        f"{len(matriz) - sin_etiqueta} con etiqueta VAS."
    )
    return matriz, metadata
