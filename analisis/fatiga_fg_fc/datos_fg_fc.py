"""
Fatiga general (FG) y fatiga cognitiva (FC) — carga, enlace y centrado.

Módulo común a todos los pasos de la línea. Hace tres cosas:

1. Lee la base de cuestionarios corregida y puntúa FG y FC (suma de 6 ítems
   de 0 a 4).
2. Enlaza las respuestas con las sesiones EEG etiquetadas de la Fase 12, por
   `(centro, pulsera)` → `persona_uid` (con `normalizar_vas`, que aplica la
   corrección F111/F115) y `(persona_uid, medición)` → sesión. Aplica las
   correcciones de `CORRECCIONES_CUESTIONARIO`, cada una con su evidencia
   escrita, y se detiene si algo del enlace no cuadra.
3. Centra una puntuación por medición, con las medias estimadas solo en las
   filas que se indiquen (las del pliegue de entrenamiento).

El `CÓDIGO` (iniciales + fecha de nacimiento de menores) solo se usa aquí
dentro, para comprobar que el enlace es unívoco. Ninguna función lo devuelve.

La ubicación del dataset se toma de la variable de entorno `EEG_DATA`, con
`C:\\Users\\migue\\Desktop\\TFG\\EEG-Data` por defecto (ver `analisis/README.md`).
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_library.procesado.labels import GENERO_A_SEXO, normalizar_vas

EEG_DATA = Path(os.environ.get("EEG_DATA", r"C:\Users\migue\Desktop\TFG\EEG-Data"))
# Base corregida por el equipo del estudio (2026-10-22): restaura los ítems
# PedsQL de M5 y M6, que en las versiones anteriores se habían codificado como
# max(respuesta − 1, 0). Ver la hoja AUDIT_CORRECCION_PEDS y el paso 1.
RUTA_BD = EEG_DATA / "BD_COMPLETA_corregida.xlsx"
HOJA_BD = "CUESTIONARIOS"
RUTA_LABELS = EEG_DATA / "metadata" / "labels.csv"

MEDICIONES = ["M1", "M2", "M3", "M4", "M5", "M6"]
ITEMS_FG = [f"FG_{i}" for i in range(1, 7)]
ITEMS_FC = [f"FC_{i}" for i in range(1, 7)]
ITEMS = {"FG": ITEMS_FG, "FC": ITEMS_FC}
# Tercera subescala de la PedsQL: no se analiza, pero la corrección de M5–M6
# también la afectó y el paso 1 la verifica.
ITEMS_FSD = [f"FS/D_{i}" for i in range(1, 7)]
ITEMS_PEDSQL = ITEMS_FG + ITEMS_FSD + ITEMS_FC

COLUMNAS_BD = ("Marca temporal", "N_CENTRO", "PARTICIPANTES_PROYC", "MEDIDA", "CÓDIGO",
               "FITBIT", "GÉNERO", "CURSO_CLASE", "FATIGA_MENTAL", "VAS_100mm_preFE",
               *ITEMS_PEDSQL)

# Columnas que devuelve `respuestas_participantes` (sin CÓDIGO).
COLUMNAS_RESPUESTA = ["persona_uid", "split", "medicion", "centro", "sexo", "curso",
                      "vas_bd", "FATIGA_MENTAL", "FG", "FC", *ITEMS_FG, *ITEMS_FC]


# --- Correcciones y excepciones ------------------------------------------------
#
# Mismo formato que CORRECCIONES_VAS (Fase 12): cada una lleva su evidencia y el
# número de filas que espera tocar. Si la base cambia y deja de cuadrar, el
# módulo se detiene en vez de aplicarla a medias. Las filas se identifican por
# (centro, pulsera, medida), nunca por CÓDIGO.

CORRECCIONES_CUESTIONARIO = (
    {
        "nombre": "c2_f105_m4_m5_archivadas_en_c4",
        "n_centro": 4, "fitbit": 105, "medidas": (4, 5),
        "n_esperado": 2,
        "n_centro_correcto": 2,
        "evidencia": (
            "Las respuestas de M4 y M5 de la pulsera 105 figuran en el centro 4; "
            "las de M1-M3 y M6, en el centro 2. Tienen el mismo CÓDIGO que las "
            "demás filas de C2/105, en C4 no hay ninguna otra fila con esa "
            "pulsera, y las grabaciones EEG de esas dos sesiones están archivadas "
            "como C2_F105 (Fase 12). En BD_COMPLETA_ConCalificaciones estaban en C2."
        ),
    },
    {
        "nombre": "c2_f116_m3_registrada_en_c7",
        "n_centro": 7, "fitbit": 116, "medidas": (3,),
        "n_esperado": 1,
        "n_centro_correcto": 2,
        "evidencia": (
            "La respuesta de M3 de la pulsera 116 figura en el centro 7 y las de "
            "M1, M2 y M4-M6 en el centro 2, con el mismo CÓDIGO. Es el caso "
            "documentado en docs/ESTADO.md (Fase 12): la misma persona registrada "
            "dos veces."
        ),
    },
)

EXCEPCIONES_DOS_CODIGOS = {
    "C3_F066": (
        "La pulsera 66 de C3 lleva un código en M1, M5 y M6 y otro en M2-M4: un "
        "código corregido solo en algunas mediciones. Misma pulsera, centro, sexo "
        "y curso en las seis, y ninguna medición repetida: es la misma persona."
    ),
}


def _validar_columnas(df, columnas, nombre):
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise ValueError(f"A '{nombre}' le faltan columnas necesarias: {faltan}.")


def sha256(ruta):
    """SHA-256 de un archivo, para dejar constancia de qué base se leyó."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


# --- Lectura y puntuación ------------------------------------------------------

def puntuar(datos, items):
    """
    Suma de los ítems de una subescala (0–24 para FG y FC).

    NaN si falta cualquier ítem. En esta base los ítems vienen completos o
    ausentes enteros, así que no hay nada que imputar; la regla de la PedsQL
    (imputar si falta menos de la mitad) no llega a aplicarse.
    """
    return datos[items].sum(axis=1, min_count=len(items))


def leer_bd(ruta=RUTA_BD, hoja=HOJA_BD):
    """
    Lee una base de cuestionarios, se queda con M1–M6 y puntúa FG y FC.

    Comprueba que todos los ítems de FG y FC estén en {0, 1, 2, 3, 4}. FS/D no
    se valida porque no se analiza (tiene un 5 en M3, ver el paso 1).
    """
    bd = pd.read_excel(ruta, sheet_name=hoja)
    bd.columns = [str(c).strip() for c in bd.columns]
    _validar_columnas(bd, COLUMNAS_BD, hoja)
    bd = bd[bd["MEDIDA"].between(1, 6)].copy()
    valores = pd.unique(bd[ITEMS_FG + ITEMS_FC].to_numpy().ravel())
    raros = [v for v in valores if not pd.isna(v) and v not in (0, 1, 2, 3, 4)]
    if raros:
        raise ValueError(f"Ítems de FG/FC fuera de 0–4: {raros}.")
    for nombre, items in ITEMS.items():
        bd[nombre] = puntuar(bd, items)
    bd["medicion"] = "M" + bd["MEDIDA"].astype(int).astype(str)
    bd["sexo"] = pd.to_numeric(bd["GÉNERO"], errors="coerce").map(GENERO_A_SEXO)
    bd["curso"] = pd.to_numeric(bd["CURSO_CLASE"], errors="coerce")
    return bd


def aplicar_correcciones(bd):
    """
    Aplica `CORRECCIONES_CUESTIONARIO` comprobando el nº de filas y que el
    CÓDIGO de las filas corregidas sea el de la persona de destino.
    """
    v = bd.copy()
    aplicadas = []
    for c in CORRECCIONES_CUESTIONARIO:
        filas = ((v["N_CENTRO"] == c["n_centro"]) & (v["FITBIT"] == c["fitbit"])
                 & v["MEDIDA"].isin(c["medidas"]))
        if int(filas.sum()) != c["n_esperado"]:
            raise ValueError(
                f"La corrección '{c['nombre']}' esperaba {c['n_esperado']} fila(s) y "
                f"encontró {int(filas.sum())}: revisar la corrección antes de seguir.")
        destino = (v["N_CENTRO"] == c["n_centro_correcto"]) & (v["FITBIT"] == c["fitbit"])
        codigos_destino = set(v.loc[destino, "CÓDIGO"].dropna())
        if not set(v.loc[filas, "CÓDIGO"].dropna()) <= codigos_destino:
            raise ValueError(
                f"La corrección '{c['nombre']}' movería filas de otra persona: el "
                f"CÓDIGO no coincide con el de las filas de destino.")
        v.loc[filas, "N_CENTRO"] = c["n_centro_correcto"]
        aplicadas.append({"nombre": c["nombre"], "filas": int(filas.sum())})
    return v, aplicadas


# --- Participantes: respuestas con persona_uid ---------------------------------

def respuestas_participantes(bd, labels):
    """
    Respuestas de los alumnos con pulsera, con su `persona_uid` y su split.

    Las filas con pulsera pasan por `aplicar_correcciones` y por
    `normalizar_vas` (Fase 12), que construye `persona_uid` desde
    `(centro, pulsera)` y aplica F111/F115. No se usa `PARTICIPANTES_PROYC`,
    que no es fiable (`C2_F108` tiene EEG y figura como no participante). Se
    comprueba que no haya dos respuestas por `(persona_uid, medición)`, que
    cada `persona_uid` tenga un solo CÓDIGO (salvo `EXCEPCIONES_DOS_CODIGOS`)
    y cada CÓDIGO un solo `persona_uid`, y que ninguna persona esté en dos
    splits.

    Devuelve
    --------
    respuestas : pandas.DataFrame
        Una fila por `(persona_uid, medicion)` con `COLUMNAS_RESPUESTA`: `vas_bd`
        es el VAS de la base y `split` es NaN para quien no tiene EEG
        etiquetado. Sin CÓDIGO.
    resumen : dict
    """
    con_pulsera = bd[bd["FITBIT"].notna()]
    corregidas, aplicadas = aplicar_correcciones(con_pulsera)
    v = normalizar_vas(corregidas)

    duplicadas = v.duplicated(["persona_uid", "medicion"], keep=False)
    if duplicadas.any():
        raise ValueError(f"{int(duplicadas.sum())} respuestas duplicadas por "
                         f"(persona_uid, medicion).")
    codigos = v.groupby("persona_uid")["CÓDIGO"].nunique()
    raras = sorted(set(codigos[codigos > 1].index) - set(EXCEPCIONES_DOS_CODIGOS))
    if raras:
        raise ValueError(f"persona_uid con más de un CÓDIGO sin excepción documentada: {raras}.")
    sin_uso = sorted(set(EXCEPCIONES_DOS_CODIGOS) - set(codigos[codigos > 1].index))
    if sin_uso:
        raise ValueError(f"Excepciones que ya no hacen falta (revisar): {sin_uso}.")
    personas = v.groupby("CÓDIGO")["persona_uid"].nunique()
    if (personas > 1).any():
        raise ValueError(f"{int((personas > 1).sum())} CÓDIGO(s) con más de un persona_uid.")

    con_persona = labels.dropna(subset=["persona_uid"])
    splits = con_persona.groupby("persona_uid")["split"].unique()
    if (splits.map(len) > 1).any():
        raise ValueError("Hay personas en más de un split: revisar labels.csv.")
    v["split"] = v["persona_uid"].map(splits.map(lambda s: s[0]))
    v = v.rename(columns={"vas": "vas_bd"})
    # Tras las correcciones, el centro de la persona es el de su persona_uid.
    v["centro"] = v["persona_uid"].str.split("_").str[0]

    resumen = {"filas_con_pulsera": int(len(con_pulsera)), "correcciones": aplicadas,
               "respuestas": int(len(v)), "personas": int(v["persona_uid"].nunique()),
               "personas_con_dos_codigos": sorted(EXCEPCIONES_DOS_CODIGOS)}
    return v[COLUMNAS_RESPUESTA].reset_index(drop=True), resumen


def _moda_unica(serie, nombre, persona):
    cuenta = serie.dropna().value_counts()
    if cuenta.empty:
        return np.nan
    if len(cuenta) > 1 and cuenta.iloc[0] == cuenta.iloc[1]:
        raise ValueError(f"{persona}: {nombre} sin moda única ({cuenta.to_dict()}).")
    return cuenta.index[0]


def caracteristicas_persona(respuestas, labels):
    """
    Sexo, curso y centro de cada persona: la moda de sus filas.

    El sexo se contrasta con el de la Fase 12 (`sexo_vas` de `labels.csv`, que a
    su vez se validó contra el nombre de los EDF); si alguno discrepa, se
    detiene. El `GÉNERO` 0 u otros códigos cuentan como desconocido.
    """
    filas = []
    for persona, r in respuestas.groupby("persona_uid"):
        filas.append({"persona_uid": persona, "centro": r["centro"].iloc[0],
                      "sexo": _moda_unica(r["sexo"], "sexo", persona),
                      "curso": _moda_unica(r["curso"], "curso", persona)})
    personas = pd.DataFrame(filas).set_index("persona_uid")
    sexo_fase12 = (labels.dropna(subset=["persona_uid", "sexo_vas"])
                   .groupby("persona_uid")["sexo_vas"].agg(lambda s: s.mode().iloc[0]))
    comun = personas.index.intersection(sexo_fase12.index)
    discrepan = comun[personas.loc[comun, "sexo"] != sexo_fase12.loc[comun]]
    if len(discrepan):
        raise ValueError(f"Sexo distinto del de la Fase 12 en: {list(discrepan)}.")
    return personas


def sesiones_etiquetadas(respuestas, labels):
    """
    Sesiones EEG etiquetadas de la Fase 12 con su FG y FC.

    Une por `(persona_uid, medicion)`. Comprobación de integridad: en toda
    sesión con respuesta en la base, el VAS de la base tiene que ser igual al
    de `labels.csv`; si no, se detiene.

    Devuelve
    --------
    sesiones : pandas.DataFrame
        Una fila por sesión etiquetada (todas las particiones), con `con_fgfc`
        (True si FG y FC están disponibles).
    """
    lab = labels[labels["etiquetada"]].copy()
    s = lab.merge(respuestas.drop(columns=["split", "centro", "sexo", "curso"]),
                  on=["persona_uid", "medicion"], how="left", indicator=True)
    s["con_respuesta"] = s.pop("_merge") == "both"
    discrepa = s["con_respuesta"] & ~(s["vas"] == s["vas_bd"])
    if discrepa.any():
        raise ValueError("VAS de la base distinto del de labels.csv en: "
                         f"{s.loc[discrepa, 'sesion_uid'].tolist()}.")
    s["con_fgfc"] = s["FG"].notna() & s["FC"].notna()
    return s


def cargar_enlace(ruta_bd=RUTA_BD, ruta_labels=RUTA_LABELS):
    """
    Todo el enlace de una vez: base leída, respuestas, características de cada
    persona, sesiones etiquetadas y un resumen con el SHA-256 de la base.

    Devuelve todas las particiones: cada paso aparta el test por su cuenta.
    """
    bd = leer_bd(ruta_bd)
    labels = pd.read_csv(ruta_labels)
    respuestas, resumen = respuestas_participantes(bd, labels)
    personas = caracteristicas_persona(respuestas, labels)
    sesiones = sesiones_etiquetadas(respuestas, labels)
    resumen = {"base": Path(ruta_bd).name, "hoja": HOJA_BD, "sha256_base": sha256(ruta_bd),
               **resumen}
    return bd, respuestas, personas, sesiones, resumen


# --- Centrado por medición -----------------------------------------------------

def medias_por_medicion(datos, columna, ajuste=None):
    """
    Media de `columna` en cada medición, estimada solo con las filas `ajuste`
    (máscara booleana; None = todas). En los pasos 2 y 3, `ajuste` son las
    personas del pliegue de entrenamiento.
    """
    filas = datos if ajuste is None else datos[np.asarray(ajuste, bool)]
    medias = filas.groupby("medicion")[columna].mean()
    faltan = sorted(set(datos["medicion"]) - set(medias.dropna().index))
    if faltan:
        raise ValueError(f"Sin datos de ajuste para '{columna}' en: {faltan}.")
    return medias


def centrar_por_medicion(datos, columna, ajuste=None):
    """
    `columna` menos la media de su medición (estimada con `ajuste`).

    Quita el efecto común de calendario (M1 más baja, subida a lo largo del
    curso) sin tocar las diferencias entre personas ni dentro de persona.
    """
    medias = medias_por_medicion(datos, columna, ajuste)
    return datos[columna] - datos["medicion"].map(medias)
