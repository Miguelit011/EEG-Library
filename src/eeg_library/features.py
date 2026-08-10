"""
Módulo de anotación de validez de canal (Fase 11.5).

`extraer_bandas` (Fase 9) calcula las columnas de potencia para los 14
electrodos pase lo que pase, para que la matriz de features tenga las mismas
columnas en todos los archivos y el lote (Fase 11) pueda concatenarlas. Eso
implica que las 8 columnas de un canal que `evaluar_calidad` descartó
(4 bandas x absoluta/relativa) están ahí igual que las de un canal bueno,
pero calculadas sobre señal que no pasó por CAR (Fase 5) ni por ICA (Fase 6)
— ambas operan solo sobre los canales buenos.

Este módulo añade a la matriz una **máscara de validez por canal**, para que
esa diferencia deje de ser invisible.

Por qué una máscara y no imputación ni exclusión (medido sobre el dataset
completo, 4543 grabaciones y 280 339 épocas):

- El problema es general pero superficial: el 85 % de los archivos tiene al
  menos un canal excluido (media 1.87) y alcanza al 85.8 % de las épocas,
  pero solo al **13.7 % de las celdas de potencia**.
- Está concentrado en los electrodos de la frente, donde las almohadillas del
  Epoc X se secan antes: AF4 (33 % de los archivos), AF3 (26 %), F8 (24 %) y
  F7 (18 %); el resto no pasa del 13 %.
- Los valores de un canal excluido están fuera de escala por factores de 1.3x
  a 7.9x respecto a la mediana del mismo canal cuando está bueno (AF4 el
  peor). **Esa desviación es en sí misma una señal legible para un árbol de
  decisión**: el modelo puede usar la calidad de contacto como atajo tanto si
  se le da la máscara como si no. Dejarla implícita no elimina el atajo, solo
  lo vuelve inmedible; con la máscara se puede cuantificar su peso en
  `feature_importances_` y entrenar con y sin ella.
- Imputar habría fabricado ese 13.7 % de celdas justo en los canales
  frontales donde vive el biomarcador theta de fatiga.
- Descartar las grabaciones afectadas no era viable: 21 de los 178 sujetos no
  tienen ninguna grabación sin exclusiones.

Advertencia para la Fase 10: la calidad de contacto correlaciona con el
índice de fatiga (Spearman 0.167; el índice mediano por archivo sube de 1.14
con 0 canales excluidos a 1.88 con 5). Esa correlación **sobrevive** a la
restricción de clústeres a canales buenos de la Fase 9, así que no es el
mismo problema: apunta a que en una grabación con mal contacto general
también los canales que sobreviven son más ruidosos. Es un confundido a tener
en cuenta al construir la etiqueta, no algo que resuelva esta máscara. En
cambio **no** hay relación con la posición en el protocolo (Spearman 0.048,
plana de la posición 1 a la 7), así que la etiqueta de protocolo de la Fase
10 no queda contaminada por este efecto.
"""

import pandas as pd

from eeg_library.quality import CANALES_EPOC

PREFIJO_VALIDEZ = "canal_valido_"
COLUMNA_N_BUENOS = "n_canales_buenos"


def columnas_de_validez(canales=None):
    """Nombres de las columnas de máscara, en el orden de `CANALES_EPOC`."""
    canales = canales or CANALES_EPOC
    return [f"{PREFIJO_VALIDEZ}{canal}" for canal in canales]


def anotar_validez_de_canales(features, log, canales=None):
    """
    Añade a la matriz de features la máscara de validez por canal.

    La exclusión de canales la decide `evaluar_calidad` sobre el registro
    completo, así que la máscara es **por archivo**: todas las épocas de una
    misma grabación comparten los mismos canales válidos.

    Parámetros
    ----------
    features : pandas.DataFrame
        Matriz de features con una fila por época y una columna
        `archivo_origen` (salida de `consolidar_features`, Fase 11).
    log : pandas.DataFrame
        Log del lote (`procesamiento_log.csv`), con las columnas `archivo`,
        `canales_excluidos` y `n_canales_buenos`. Las filas con estado
        distinto de "ok" se ignoran: sus archivos no llegaron a producir
        features.
    canales : list[str] o None
        Canales a anotar. Por defecto `CANALES_EPOC`.

    Devuelve
    --------
    anotadas : pandas.DataFrame
        Copia de `features` con `len(canales)` columnas booleanas
        `canal_valido_<canal>` y una columna entera `n_canales_buenos`
        añadidas al final. Las columnas originales no se tocan.

    Raises
    ------
    ValueError
        Si a `features` o a `log` les faltan las columnas necesarias, si
        algún `archivo_origen` no aparece en el log, o si el número de
        canales válidos de la máscara no coincide con el `n_canales_buenos`
        que registró `evaluar_calidad` — esto último delataría una grabación
        a la que le faltaba algún electrodo, caso en el que la máscara lo
        estaría dando por válido.
    """
    canales = list(canales or CANALES_EPOC)

    if "archivo_origen" not in features.columns:
        raise ValueError("'features' necesita la columna 'archivo_origen'.")
    faltan = [c for c in ("archivo", "canales_excluidos", "n_canales_buenos")
              if c not in log.columns]
    if faltan:
        raise ValueError(f"Al log le faltan columnas necesarias: {faltan}.")

    correctos = log[log["estado"] == "ok"] if "estado" in log.columns else log

    excluidos_por_archivo = {}
    buenos_por_archivo = {}
    for archivo, excluidos, n_buenos in zip(
        correctos["archivo"], correctos["canales_excluidos"], correctos["n_canales_buenos"]
    ):
        texto = "" if pd.isna(excluidos) else str(excluidos)
        excluidos_por_archivo[archivo] = {c for c in texto.split("|") if c}
        buenos_por_archivo[archivo] = n_buenos

    archivos = features["archivo_origen"].unique()
    sin_log = sorted(set(archivos) - set(excluidos_por_archivo))
    if sin_log:
        raise ValueError(
            f"{len(sin_log)} archivo(s) de 'features' no aparecen como "
            f"procesados con exito en el log, asi que no se puede saber que "
            f"canales eran validos: {sin_log[:5]}"
            + (" ..." if len(sin_log) > 5 else "")
        )

    # Se construye la máscara por archivo (no por época) y se propaga con un
    # map sobre archivo_origen: la decisión de calidad es del registro entero.
    mascara_por_archivo = {}
    for archivo in archivos:
        excluidos = excluidos_por_archivo[archivo]
        mascara = {c: c not in excluidos for c in canales}
        n_validos = sum(mascara.values())
        n_registrado = buenos_por_archivo[archivo]
        if not pd.isna(n_registrado) and int(n_registrado) != n_validos:
            raise ValueError(
                f"En '{archivo}' la mascara da {n_validos} canal(es) valido(s) "
                f"pero evaluar_calidad registro {int(n_registrado)}. Suele "
                f"significar que a la grabacion le faltaba algun electrodo de "
                f"CANALES_EPOC, que la mascara estaria marcando como valido."
            )
        mascara_por_archivo[archivo] = mascara

    anotadas = features.copy()
    for canal in canales:
        anotadas[f"{PREFIJO_VALIDEZ}{canal}"] = (
            anotadas["archivo_origen"].map(
                {a: m[canal] for a, m in mascara_por_archivo.items()}
            ).astype(bool)
        )
    anotadas[COLUMNA_N_BUENOS] = (
        anotadas[columnas_de_validez(canales)].sum(axis=1).astype("int16")
    )

    n_afectadas = int((anotadas[COLUMNA_N_BUENOS] < len(canales)).sum())
    print(
        f"Validez de canal anotada: {len(canales)} columna(s) "
        f"'{PREFIJO_VALIDEZ}*' + '{COLUMNA_N_BUENOS}'. "
        f"{n_afectadas} de {len(anotadas)} epoca(s) "
        f"({100 * n_afectadas / len(anotadas):.1f} %) tienen algun canal excluido."
    )
    return anotadas
