"""
Fatiga general (FG) y fatiga cognitiva (FC) — carga, enlace y tipificación.

Módulo común a todos los pasos de la línea. Hace cuatro cosas:

1. Lee la base de cuestionarios y puntúa FG y FC (suma de 6 ítems de 0 a 4).
2. Enlaza las respuestas con las sesiones EEG etiquetadas de la Fase 12, por
   `(centro, pulsera)` → `persona_uid` (con `normalizar_vas`, que aplica la
   corrección F111/F115) y `(persona_uid, medición)` → sesión. Aplica las
   correcciones y exclusiones de `CORRECCIONES_CUESTIONARIO` y
   `EXCLUSIONES_SESION`, cada una con su evidencia escrita, y se detiene si
   algo del enlace no cuadra.
3. Construye la referencia para tipificar: los alumnos que no hicieron EEG.
4. Tipifica una puntuación dentro de cada medición frente a esa referencia
   (percentil normalizado o z lineal), con pesos opcionales por raking.

El `CÓDIGO` (iniciales + fecha de nacimiento de menores) solo se usa aquí
dentro, para comprobar que el enlace es unívoco. Ninguna función lo devuelve.

La ubicación del dataset se toma de la variable de entorno `EEG_DATA`, con
`C:\\Users\\migue\\Desktop\\TFG\\EEG-Data` por defecto (ver `analisis/README.md`).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from eeg_library.procesado.labels import GENERO_A_SEXO, normalizar_vas

EEG_DATA = Path(os.environ.get("EEG_DATA", r"C:\Users\migue\Desktop\TFG\EEG-Data"))
RUTA_BD = EEG_DATA / "BD_COMPLETA_sincalificaciones.xlsx"
HOJA_BD = "CUESTIONARIOS"
RUTA_LABELS = EEG_DATA / "metadata" / "labels.csv"

MEDICIONES = ["M1", "M2", "M3", "M4", "M5", "M6"]
ITEMS_FG = [f"FG_{i}" for i in range(1, 7)]
ITEMS_FC = [f"FC_{i}" for i in range(1, 7)]
ITEMS = {"FG": ITEMS_FG, "FC": ITEMS_FC}
# Otros bloques del mismo formulario, solo para describir el salto de M5.
ITEMS_FSD = [f"FS/D_{i}" for i in range(1, 7)]
ITEMS_WS = [f"WS_{i}" for i in range(1, 7)]
ITEMS_AE = [f"AE_{i}" for i in range(1, 5)]

COLUMNAS_BD = ("Marca temporal", "N_CENTRO", "PARTICIPANTES_PROYC", "MEDIDA", "CÓDIGO",
               "FITBIT", "GÉNERO", "CURSO_CLASE", "FATIGA_MENTAL", "VAS_100mm_preFE",
               *ITEMS_FG, *ITEMS_FC, *ITEMS_FSD, *ITEMS_WS, *ITEMS_AE)

# Columnas que devuelven las funciones de respuestas (sin CÓDIGO).
COLUMNAS_RESPUESTA = ["medicion", "centro", "sexo", "curso", "FG", "FC", "FATIGA_MENTAL",
                      *ITEMS_FG, *ITEMS_FC]


# --- Correcciones, exclusiones y excepciones -----------------------------------
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
            "En BD_COMPLETA_sincalificaciones las respuestas de M4 y M5 de la "
            "pulsera 105 figuran en el centro 4; en M1-M3 y M6, en el centro 2. "
            "Tienen el mismo CÓDIGO que las demás filas de C2/105, en C4 no hay "
            "ninguna otra fila con esa pulsera, y las grabaciones EEG de esas dos "
            "sesiones están archivadas como C2_F105 (Fase 12). En la versión "
            "anterior de la base estaban en C2."
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
            "dos veces. Igual en las dos versiones de la base."
        ),
    },
)

EXCLUSIONES_SESION = (
    {
        "sesion_uid": "C3_F063_M6",
        "evidencia": (
            "En BD_COMPLETA_sincalificaciones el VAS (15) y las pruebas cognitivas "
            "de M6 pasan del alumno de la pulsera 63 al de la 2063; en la versión "
            "anterior, y en la Fase 12, eran del de la 63. Los dos son XY, "
            "respondieron el mismo día y usaron las dos pulseras a la vez todo el "
            "curso, así que nada en los datos dice de quién es la grabación EEG "
            "(archivada como F063). Elegir mal importa: su FG sería 6 o 1. "
            "Pendiente de confirmar con el equipo del estudio."
        ),
    },
)

EXCEPCIONES_DOS_CODIGOS = {
    "C3_F066": (
        "En BD_COMPLETA_sincalificaciones la pulsera 66 de C3 lleva un código en "
        "M1, M5 y M6 y otro en M2-M4: un código corregido solo en algunas "
        "mediciones. Misma pulsera, centro, sexo y curso en las seis, y ninguna "
        "medición repetida: es la misma persona."
    ),
}


def _validar_columnas(df, columnas, nombre):
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise ValueError(f"A '{nombre}' le faltan columnas necesarias: {faltan}.")


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
    Lee la base de cuestionarios, se queda con M1–M6 y puntúa FG y FC.

    Comprueba que todos los ítems de FG y FC estén en {0, 1, 2, 3, 4}.
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
    `(centro, pulsera)` y aplica F111/F115. Se comprueba que no haya dos
    respuestas por `(persona_uid, medición)`, que cada `persona_uid` tenga un
    solo CÓDIGO (salvo `EXCEPCIONES_DOS_CODIGOS`) y cada CÓDIGO un solo
    `persona_uid`, y que ninguna persona esté en dos splits.

    Devuelve
    --------
    respuestas : pandas.DataFrame
        Una fila por `(persona_uid, medicion)`, con `vas_bd` (el VAS de la base)
        y `split` (NaN para quien no tiene EEG etiquetado). Sin CÓDIGO.
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
    columnas = ["persona_uid", "split", "vas_bd", *COLUMNAS_RESPUESTA,
                *ITEMS_FSD, *ITEMS_WS, *ITEMS_AE]
    return v[columnas].reset_index(drop=True), resumen


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
    de `labels.csv`; si no, se detiene. Las sesiones de `EXCLUSIONES_SESION`
    salen marcadas y sin FG/FC, y se comprueba que la discrepancia que las
    justifica siga existiendo.

    Devuelve
    --------
    sesiones : pandas.DataFrame
        Una fila por sesión etiquetada (todas las particiones), con `con_fgfc`
        (True si FG y FC están disponibles y la sesión no está excluida).
    """
    lab = labels[labels["etiquetada"]].copy()
    s = lab.merge(respuestas.drop(columns=["split", "centro"]),
                  on=["persona_uid", "medicion"], how="left", indicator=True)
    s["con_respuesta"] = s.pop("_merge") == "both"
    s["excluida"] = s["sesion_uid"].isin([e["sesion_uid"] for e in EXCLUSIONES_SESION])

    discrepa = s["con_respuesta"] & ~(s["vas"] == s["vas_bd"])
    if (discrepa & ~s["excluida"]).any():
        raise ValueError("VAS de la base distinto del de labels.csv en: "
                         f"{s.loc[discrepa & ~s['excluida'], 'sesion_uid'].tolist()}.")
    caducadas = s.loc[s["excluida"] & ~discrepa, "sesion_uid"].tolist()
    if caducadas:
        raise ValueError(f"Exclusiones cuya discrepancia ya no existe (revisar): {caducadas}.")

    s.loc[s["excluida"], ["FG", "FC", *ITEMS_FG, *ITEMS_FC]] = np.nan
    s["con_fgfc"] = s["FG"].notna() & s["FC"].notna()
    return s


# --- Referencia: alumnos sin EEG -----------------------------------------------

def referencia(bd):
    """
    Respuestas de los alumnos que no participan en el estudio EEG.

    Son las filas con `PARTICIPANTES_PROYC` = 0, sin pulsera y cuyo CÓDIGO no
    aparece nunca con pulsera. Las dos últimas condiciones hacen falta porque la
    marca de participación no es fiable (`C2_F108` tiene EEG y figura como no
    participante). Solo filas con FG y FC.
    """
    con_pulsera = set(bd.loc[bd["FITBIT"].notna(), "CÓDIGO"].dropna())
    r = bd[(bd["PARTICIPANTES_PROYC"] == 0) & bd["FITBIT"].isna()
           & ~bd["CÓDIGO"].isin(con_pulsera)].copy()
    r["centro"] = "C" + r["N_CENTRO"].astype(int).astype(str)
    r = r.dropna(subset=["FG", "FC"])
    return r[[*COLUMNAS_RESPUESTA, *ITEMS_FSD, *ITEMS_WS, *ITEMS_AE]].reset_index(drop=True)


# --- Pesos y tipificación ------------------------------------------------------

def pesos_raking(datos, objetivo, variables, max_iter=500, tol=1e-9):
    """
    Pesos por raking (ajuste proporcional iterativo) para que las marginales
    ponderadas de `variables` en `datos` igualen las de `objetivo`.

    Las filas con alguna variable desconocida (NaN) o en una categoría que no
    existe en `objetivo` reciben peso 0. Si `objetivo` tiene una categoría que
    no aparece en `datos`, no hay pesos posibles y se detiene.

    Devuelve los pesos (media 1 sobre las filas con peso > 0) y el tamaño
    efectivo de Kish, (Σw)² / Σw².
    """
    w = np.ones(len(datos))
    for v in variables:
        w[datos[v].isna().to_numpy() | ~datos[v].isin(objetivo[v].dropna().unique()).to_numpy()] = 0
    for v in variables:
        faltan = set(objetivo[v].dropna().unique()) - set(datos.loc[w > 0, v])
        if faltan:
            raise ValueError(f"Categorías de '{v}' sin filas en la referencia: {sorted(faltan)}.")
    metas = {v: objetivo[v].value_counts(normalize=True) for v in variables}
    for _ in range(max_iter):
        desvio = 0.0
        for v in variables:
            actual = pd.Series(w).groupby(datos[v].to_numpy()).sum() / w.sum()
            factor = (metas[v] / actual).reindex(datos[v].to_numpy()).fillna(0).to_numpy()
            w = w * factor
            desvio = max(desvio, float((actual - metas[v]).abs().max()))
        if desvio < tol:
            break
    else:
        raise ValueError("El raking no converge.")
    usados = w > 0
    w = w / w[usados].mean()
    kish = float(w.sum() ** 2 / (w ** 2).sum())
    return w, kish


def percentil_normal(x, ref_valores, ref_pesos=None):
    """
    Posición de cada valor de `x` en la distribución de referencia, en escala z.

    p = (W(<x) + ½·W(=x) + ½) / (W + 1), con W la suma de pesos (escalados para
    sumar el nº de filas usadas), y z = Φ⁻¹(p). El ½ de los empates da el rango
    medio, que importa porque la puntuación es entera (0–24) y con muchos
    empates; el ½ y el +1 del total evitan p = 0 o 1 en los extremos.
    """
    ref = np.asarray(ref_valores, float)
    w = np.ones(len(ref)) if ref_pesos is None else np.asarray(ref_pesos, float)
    usados = w > 0
    ref, w = ref[usados], w[usados]
    w = w * len(ref) / w.sum()
    x = np.asarray(x, float)
    p = np.array([(w[ref < xi].sum() + 0.5 * w[ref == xi].sum() + 0.5) / (w.sum() + 1)
                  if not np.isnan(xi) else np.nan for xi in x])
    return stats.norm.ppf(p)


def z_lineal(x, ref_valores, ref_pesos=None):
    """(x − media de la referencia) / desviación de la referencia, ponderadas."""
    ref = np.asarray(ref_valores, float)
    w = np.ones(len(ref)) if ref_pesos is None else np.asarray(ref_pesos, float)
    media = np.average(ref, weights=w)
    sd = np.sqrt(np.average((ref - media) ** 2, weights=w))
    return (np.asarray(x, float) - media) / sd


def tipificar_por_medicion(datos, ref, columna, metodo="percentil", pesos_ref=None):
    """
    Tipifica `columna` de `datos` dentro de cada medición frente a `ref`.

    `metodo` es "percentil" (`percentil_normal`, la principal) o "lineal"
    (`z_lineal`, sensibilidad). `pesos_ref` (array alineado con `ref`) aplica
    los pesos del raking; None = todos iguales.
    """
    funcion = {"percentil": percentil_normal, "lineal": z_lineal}[metodo]
    salida = pd.Series(np.nan, index=datos.index)
    pesos = np.ones(len(ref)) if pesos_ref is None else np.asarray(pesos_ref, float)
    for m in MEDICIONES:
        en_m = (ref["medicion"] == m).to_numpy()
        filas = datos["medicion"] == m
        if filas.any():
            salida[filas] = funcion(datos.loc[filas, columna], ref.loc[en_m, columna],
                                    pesos[en_m])
    return salida


def pesos_referencia(ref, objetivo, variables=("centro", "sexo", "curso")):
    """
    Pesos de raking de la referencia, medición a medición, hacia la composición
    de `objetivo` (respuestas de los participantes, con las mismas `variables`).

    Devuelve los pesos alineados con `ref` y el tamaño efectivo de Kish por
    medición.
    """
    pesos = np.zeros(len(ref))
    kish = {}
    for m in MEDICIONES:
        en_m = (ref["medicion"] == m).to_numpy()
        obj = objetivo[objetivo["medicion"] == m]
        w, k = pesos_raking(ref.loc[en_m].reset_index(drop=True), obj, list(variables))
        pesos[en_m] = w
        kish[m] = k
    return pesos, kish
