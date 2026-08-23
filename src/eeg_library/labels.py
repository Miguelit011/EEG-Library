"""
Módulo de construcción del dataset supervisado (Fase 12).

Une el VAS de fatiga percibida (`VAS_100mm_preFE_filtrado.csv`, una fila por
persona × momento) con las grabaciones EEG del manifest, y produce la tabla de
etiquetas que consumen el Random Forest y la CNN.

Decisiones de diseño (justificación completa en ESTADO.md, Fase 12):

- **La unidad de la etiqueta es la sesión, no la grabación ni la ventana.** El
  VAS se responde una vez por sesión, inmediatamente antes del EEG, y las 7
  grabaciones de esa sesión comparten su valor al milímetro. Replicarlo hacia
  abajo es correcto (así lo describe el protocolo), pero el n del estudio son
  las sesiones. Por eso `construir_etiquetas` devuelve una fila por sesión y
  `expandir_a_grabaciones` hace la réplica explícita en vez de esconderla en
  un `merge`.
- **El enlace es por `(centro, fitbit, medicion)`, nunca por sujeto.** La
  pulsera Fitbit se reutiliza *dentro de un centro y a lo largo del curso*: en
  10 pulseras hay dos ocupantes distintos, y en 6 de ellas ambos tienen EEG.
  Unir por sujeto asignaría el VAS de una persona a las grabaciones de otra.
  Dentro del rango que cubre el EEG (M2–M6) la clave es única; la única
  colisión de dos personas en la misma clave está en M1 (C8, pulsera 44), que
  el manifest no cubre.
- **`sujeto_id` no identifica a una persona; `persona_uid` sí.** Todo lo que
  agrupe por individuo —splits, validación cruzada, bootstrap— tiene que usar
  `persona_uid`.
- **Las correcciones de metadata son declarativas y verificadas.** Cada una
  vive en una constante con su evidencia y su número esperado de filas, y se
  aplica comprobando que ese número cuadra. Si el manifest o el VAS cambian,
  falla en voz alta en vez de corregir en silencio algo que ya no es.

Lo que este módulo **no** hace: no toca la señal, no reprocesa nada y no
escribe el `CÓDIGO` del participante (iniciales + fecha de nacimiento de un
menor) en ninguna salida.
"""

import numpy as np
import pandas as pd

# Columnas mínimas que se necesitan de cada fuente.
COLUMNAS_VAS = ("N_CENTRO", "MEDIDA", "CÓDIGO", "FITBIT", "GÉNERO", "VAS_100mm_preFE")
COLUMNAS_MANIFEST = ("nombre_nuevo", "sujeto_id", "sujeto_local", "medicion",
                     "centro_efectivo", "condicion_norm", "timestamp_iso")

COLUMNA_VAS = "vas"
NOMBRE_ETIQUETAS = "labels.csv"
NOMBRE_GRABACIONES = "labels_grabaciones.csv"

# El VAS codifica el sexo como 1/2. El mapeo no está documentado en el archivo;
# se dedujo cruzando con `sexo_filename` del manifest, que sale del nombre del
# EDF: 85 de 86 sujetos con GÉNERO=1 son XX y 87 de 88 con GÉNERO=2 son XY.
# Las dos excepciones son justamente F111 y F115 (ver CORRECCIONES_VAS).
# El valor 0 aparece en 7 grabaciones y no significa nada: es un hueco del
# formulario, así que se trata como desconocido.
GENERO_A_SEXO = {1: "XX", 2: "XY"}

# --- Correcciones sobre el manifest ------------------------------------------
# Grabaciones archivadas bajo un centro que no es el suyo. Se detectaron porque
# no encontraban fila en el VAS, y se resolvieron con evidencia convergente
# (fecha de la jornada, sexo, y el hueco exacto que dejan en el centro real).
CORRECCIONES_MANIFEST = (
    {
        "nombre": "c3_a_c4_m5",
        "seleccion": {"centro_efectivo": ("C3",), "medicion": ("M5",),
                      "sujeto_local": ("F125", "F128", "F130", "F131", "F135", "F136")},
        "centro_nuevo": "C4",
        "n_esperado": 41,
        "evidencia": (
            "Las 41 están fechadas el 2025-04-28, que es la jornada M5 de C4 "
            "(la de C3 es el 2025-04-30). Esos seis sujetos existen en C4 con "
            "M3/M4/M6 pero sin M5, y en C3 no tienen ninguna otra medición: "
            "C3_F1xx es un sujeto fantasma que solo existe por este archivado. "
            "El sexo del nombre de archivo coincide con el GÉNERO del VAS en "
            "los seis. Las 41 filas vienen ya marcadas con a_revisar=True y "
            "centro_filename vacío."
        ),
    },
    {
        "nombre": "c3f103_a_c2",
        "seleccion": {"nombre_nuevo": ("sub-C3F103_ses-M3_task-PASAT_eeg.edf",)},
        "centro_nuevo": "C2",
        "n_esperado": 1,
        "evidencia": (
            "C2_F103 tiene la M3 del 2025-01-09 con seis tareas y le falta "
            "exactamente PASAT; esta grabación es un PASAT del 2025-01-09 con "
            "el mismo sexo (XY). Juntas completan el protocolo de 7. La fecha "
            "no discrimina por sí sola (C3 también midió M3 ese día), pero la "
            "tarea que falta sí."
        ),
    },
)

# --- Correcciones sobre el VAS -----------------------------------------------
CORRECCIONES_VAS = (
    {
        "nombre": "swap_f111_f115",
        "centro": "C2",
        "intercambio": (111, 115),
        "n_esperado": 6,
        "evidencia": (
            "Dos fuentes independientes contradicen al CSV del VAS en los seis "
            "momentos de ambos sujetos. (1) Sexo: el VAS da GÉNERO=1 (XX) a la "
            "pulsera 111, pero sus EDF son XY en todas las mediciones, y al "
            "revés con la 115; son los dos únicos sujetos discordantes de 174. "
            "(2) El tracker del estudio RESUMEN DATOS(TOTAL).csv, indexado por "
            "F-code y no por pulsera, da a F111 la serie [60,30,20,15,20,20] y "
            "a F115 la [40,60,40,40,40,65] — exactamente al revés que el CSV."
        ),
    },
)


def _validar_columnas(df, columnas, nombre):
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise ValueError(f"A '{nombre}' le faltan columnas necesarias: {faltan}.")


def normalizar_vas(vas):
    """
    Normaliza el CSV del VAS y aplica `CORRECCIONES_VAS`.

    Deriva la clave de enlace (`centro`, `medicion`, `fitbit_base`) y el
    `persona_uid`, y elimina la única fila exactamente duplicada del archivo
    (C6/M3/pulsera 26, mismo valor en las dos), que si no duplicaría las 7
    grabaciones de esa sesión al unir.

    Parámetros
    ----------
    vas : pandas.DataFrame
        Contenido de `VAS_100mm_preFE_filtrado.csv` (separador `;`,
        codificación `utf-8-sig`).

    Devuelve
    --------
    normalizado : pandas.DataFrame
        Con las columnas originales más `centro`, `medicion`, `fitbit_base`,
        `ocupante`, `persona_uid`, `vas`, `fecha_vas` y `sexo_vas`.
    """
    _validar_columnas(vas, COLUMNAS_VAS, "VAS")
    v = vas.copy()

    for correccion in CORRECCIONES_VAS:
        a, b = correccion["intercambio"]
        centro = int(correccion["centro"].lstrip("C"))
        en_centro = v["N_CENTRO"] == centro
        lado_a = en_centro & (v["FITBIT"] == a)
        lado_b = en_centro & (v["FITBIT"] == b)
        n_a, n_b = int(lado_a.sum()), int(lado_b.sum())
        esperado = correccion["n_esperado"]
        if n_a == 0 and n_b == 0:
            continue  # el archivo no contiene a estos sujetos: no aplica
        if n_a != esperado or n_b != esperado:
            raise ValueError(
                f"La corrección '{correccion['nombre']}' esperaba {esperado} "
                f"fila(s) por lado y encontró {n_a} para {a} y {n_b} para {b}. "
                f"Aplicarla a medias intercambiaría mal los valores: revisar la "
                f"corrección antes de seguir."
            )
        v.loc[lado_a, "FITBIT"] = b
        v.loc[lado_b, "FITBIT"] = a

    v["centro"] = "C" + v["N_CENTRO"].astype(int).astype(str)
    v["medicion"] = "M" + v["MEDIDA"].astype(int).astype(str)
    # El ocupante de una pulsera se codifica en el propio número: 2094 es el
    # segundo ocupante de la 094. Es la convención del tracker del estudio, y
    # se conserva tal cual en `persona_uid` para que el enlace sea trazable.
    fitbit = v["FITBIT"].astype(int)
    reutilizada = fitbit >= 1000
    v["ocupante"] = (fitbit // 1000).where(reutilizada, 1).astype(int)
    v["fitbit_base"] = (fitbit % 1000).where(reutilizada, fitbit).astype(int)
    v["persona_uid"] = v["centro"] + "_F" + fitbit.astype(str).where(
        reutilizada, v["fitbit_base"].astype(str).str.zfill(3))
    v[COLUMNA_VAS] = pd.to_numeric(v["VAS_100mm_preFE"], errors="coerce")
    v["fecha_vas"] = pd.to_datetime(v["Marca temporal"], errors="coerce").dt.date
    v["sexo_vas"] = pd.to_numeric(v["GÉNERO"], errors="coerce").map(GENERO_A_SEXO)

    antes = len(v)
    v = v.drop_duplicates(["centro", "medicion", "fitbit_base", "CÓDIGO"], keep="first")
    if antes != len(v):
        print(f"VAS: {antes - len(v)} fila(s) duplicada(s) exacta(s) eliminada(s).")

    # La clave de enlace no es única en todo el archivo: en C8 la pulsera 44
    # tiene dos ocupantes que respondieron ambos a M1. No es un error del
    # archivo ni impide el enlace, porque el manifest solo cubre M2–M6; quien
    # decide si molesta es `construir_etiquetas`, que sí sabe qué sesiones EEG
    # existen. Aquí solo se avisa.
    choque = v.duplicated(["centro", "medicion", "fitbit_base"], keep=False)
    if choque.any():
        claves = sorted(set(zip(v.loc[choque, "centro"], v.loc[choque, "medicion"],
                                v.loc[choque, "fitbit_base"])))
        print(f"VAS: {len(claves)} clave(s) con dos ocupantes en la misma "
              f"medición: {claves}. Solo importan si tienen EEG.")
    return v


def corregir_manifest(manifest):
    """
    Aplica `CORRECCIONES_MANIFEST`: reasigna al centro correcto las grabaciones
    archivadas bajo otro.

    Cambia `centro_efectivo` y recalcula `sujeto_id`. Añade una columna
    booleana `corregida` para que el efecto sea auditable aguas abajo.

    Raises
    ------
    ValueError
        Si alguna corrección no selecciona exactamente el número de filas que
        declara. Es deliberado: una corrección que ya no cuadra con los datos
        debe parar el proceso, no aplicarse a medias.
    """
    _validar_columnas(manifest, COLUMNAS_MANIFEST, "manifest")
    # Idempotente: si ya viene corregido se devuelve tal cual, para que el
    # flujo normal (corregir una vez y pasar el resultado a `construir_etiquetas`
    # y a `expandir_a_grabaciones`) no intente aplicar dos veces lo mismo — la
    # segunda vez seleccionaría 0 filas y fallaría la comprobación de recuento.
    if {"corregida", "sesion_uid", "fitbit_base"} <= set(manifest.columns):
        return manifest.copy()

    m = manifest.copy()
    m["corregida"] = False

    for correccion in CORRECCIONES_MANIFEST:
        seleccion = pd.Series(True, index=m.index)
        for columna, valores in correccion["seleccion"].items():
            seleccion &= m[columna].isin(valores)
        n = int(seleccion.sum())
        if n == 0:
            continue  # el manifest no contiene estas grabaciones: no aplica
        if n != correccion["n_esperado"]:
            raise ValueError(
                f"La corrección '{correccion['nombre']}' esperaba "
                f"{correccion['n_esperado']} grabación(es) y ha seleccionado "
                f"{n}. Aplicarla a medias movería de centro solo una parte de "
                f"la sesión: revisar la corrección antes de seguir."
            )
        m.loc[seleccion, "centro_efectivo"] = correccion["centro_nuevo"]
        m.loc[seleccion, "corregida"] = True
        print(f"Corrección '{correccion['nombre']}': {n} grabación(es) "
              f"reasignada(s) a {correccion['centro_nuevo']}.")

    m["sujeto_id"] = m["centro_efectivo"] + "_" + m["sujeto_local"]
    m["fitbit_base"] = m["sujeto_local"].str[1:].astype(int)
    if (m["fitbit_base"] >= 1000).any():
        raise ValueError(
            "Hay sujeto_local con número >= 1000 en el manifest. Ese rango "
            "está reservado al segundo ocupante de una pulsera y colisionaría "
            "con la construcción de persona_uid."
        )
    m["sesion_uid"] = m["sujeto_id"] + "_" + m["medicion"]
    return m


def construir_etiquetas(vas, manifest, splits, log=None):
    """
    Construye la tabla de etiquetas a nivel de **sesión**.

    Parámetros
    ----------
    vas : pandas.DataFrame
        `VAS_100mm_preFE_filtrado.csv` sin normalizar.
    manifest : pandas.DataFrame
        `manifest_canonical.csv` sin corregir.
    splits : pandas.DataFrame
        `splits.csv`, con las columnas `sujeto_id` y `split`. El split se toma
        del `sujeto_id` **ya corregido**, así que las entradas de los sujetos
        fantasma que desaparecen con las correcciones quedan sin usar.
    log : pandas.DataFrame o None
        `procesamiento_log.csv` (Fase 11). Si se pasa, añade `n_grabaciones_ok`
        y `n_ventanas` reales por sesión; si no, esas columnas quedan a NaN.

    Devuelve
    --------
    etiquetas : pandas.DataFrame
        Una fila por sesión EEG (`sesion_uid`), **incluidas las que no tienen
        VAS**, con `vas` a NaN y `etiquetada=False`. Se devuelven todas a
        propósito: filtrar es cosa de quien entrena, y así el archivo documenta
        también lo que se pierde.
    metadata : dict
        Recuentos y trazabilidad de lo aplicado.
    """
    v = normalizar_vas(vas)
    m = corregir_manifest(manifest)
    _validar_columnas(splits, ("sujeto_id", "split"), "splits")

    sesiones = (
        m.groupby(["sesion_uid", "sujeto_id", "centro_efectivo", "fitbit_base", "medicion"],
                  as_index=False)
        .agg(n_grabaciones=("nombre_nuevo", "size"),
             n_corregidas=("corregida", "sum"),
             fecha_eeg=("timestamp_iso", lambda s: pd.to_datetime(
                 s, format="ISO8601", utc=True).dt.tz_convert("Europe/Madrid").dt.date.min()))
        .rename(columns={"centro_efectivo": "centro"})
    )

    # Una clave del VAS con dos ocupantes solo es un problema si esa sesión
    # tiene EEG: entonces no habría forma de saber a cuál de los dos pertenece.
    clave = ["centro", "medicion", "fitbit_base"]
    ambiguas = v[v.duplicated(clave, keep=False)]
    con_eeg = ambiguas.merge(sesiones[clave], on=clave, how="inner")
    if len(con_eeg):
        raise ValueError(
            "Hay sesiones EEG cuya clave (centro, medicion, fitbit_base) "
            "corresponde a dos personas distintas en el VAS, así que el enlace "
            f"sería ambiguo: {sorted(set(zip(con_eeg['centro'], con_eeg['medicion'], con_eeg['fitbit_base'])))}."
        )

    # 'CÓDIGO' viaja solo dentro de esta función, para la comprobación de fuga
    # de más abajo, y se elimina antes de devolver nada: son las iniciales y la
    # fecha de nacimiento de un menor y no deben salir del VAS.
    columnas_vas = clave + ["persona_uid", "ocupante", COLUMNA_VAS, "fecha_vas",
                            "sexo_vas", "CÓDIGO"]
    etiquetas = sesiones.merge(v[columnas_vas].drop_duplicates(clave), on=clave, how="left")
    if len(etiquetas) != len(sesiones):
        raise ValueError(
            f"El enlace ha cambiado el número de sesiones ({len(sesiones)} -> "
            f"{len(etiquetas)}). Habría filas del VAS con la clave repetida."
        )

    etiquetas["etiquetada"] = etiquetas[COLUMNA_VAS].notna()
    # `persona_uid` queda a NaN en las sesiones sin VAS, y no se rellena con el
    # `sujeto_id`: en una pulsera con dos ocupantes el número por sí solo no
    # dice quién la llevaba. Ocurre de verdad — C1_F162_M5 tiene EEG pero no
    # VAS, y su sujeto_id es el mismo que el del ocupante 1, que es otra
    # persona. Dejarlo a NaN hace que cualquier groupby por persona las
    # descarte en vez de mezclarlas.
    # Tres estados, no dos: la mayoría de las "discrepancias" son en realidad
    # filas del VAS sin 'Marca temporal'. Mezclarlas con las fechas que de
    # verdad no cuadran daría una tasa de error inflada.
    conocidas = etiquetas["fecha_eeg"].notna() & etiquetas["fecha_vas"].notna()
    etiquetas["concordancia_fecha"] = np.where(
        ~conocidas, "desconocida",
        np.where(etiquetas["fecha_eeg"] == etiquetas["fecha_vas"], "coincide", "discrepa"))
    n_antes = len(etiquetas)
    etiquetas = etiquetas.merge(splits[["sujeto_id", "split"]], on="sujeto_id", how="left")
    if len(etiquetas) != n_antes:
        raise ValueError(
            f"'splits' tiene sujeto_id repetido: la unión ha pasado de "
            f"{n_antes} a {len(etiquetas)} sesiones."
        )

    if log is not None:
        _validar_columnas(log, ("archivo", "estado", "n_epocas"), "log")
        ok = log[log["estado"] == "ok"]
        por_archivo = m.set_index("nombre_nuevo")["sesion_uid"]
        ok = ok.assign(sesion_uid=ok["archivo"].map(por_archivo))
        agregado = ok.groupby("sesion_uid").agg(
            n_grabaciones_ok=("archivo", "size"),
            n_ventanas=("n_epocas", "sum"))
        etiquetas = etiquetas.merge(agregado, on="sesion_uid", how="left")
    else:
        etiquetas["n_grabaciones_ok"] = np.nan
        etiquetas["n_ventanas"] = np.nan

    etiquetas = etiquetas.sort_values(["centro", "fitbit_base", "medicion"]).reset_index(drop=True)

    lab = etiquetas[etiquetas["etiquetada"]]
    # La comprobación se hace sobre 'CÓDIGO' y no sobre `persona_uid`: este
    # último se deriva de (centro, pulsera), así que por construcción no puede
    # abarcar dos splits y comprobarlo no diría nada. 'CÓDIGO' sí identifica a
    # la persona con independencia del centro bajo el que se archivó su
    # grabación, que es justo el fallo que se busca — antes de corregir el
    # manifest, seis participantes de C4 tenían su M5 archivada en C3, con lo
    # que la misma persona vivía en dos splits a la vez.
    cruces = lab.groupby("CÓDIGO")["split"].nunique()
    if (cruces > 1).any():
        culpables = sorted(cruces[cruces > 1].index)
        detalle = (lab[lab["CÓDIGO"].isin(culpables)]
                   .groupby("CÓDIGO")
                   .agg(sujeto_id=("sujeto_id", lambda s: sorted(set(s))),
                        split=("split", lambda s: sorted(set(s))))
                   .to_dict("index"))
        raise ValueError(
            f"{len(culpables)} persona(s) aparecen en más de un split, lo que "
            f"sería fuga de datos entre train y test: {detalle}. Suele "
            f"significar que hay grabaciones archivadas bajo un centro que no "
            f"es el suyo y que falta una corrección en CORRECCIONES_MANIFEST."
        )

    n_personas_reales = int(lab["CÓDIGO"].nunique())
    etiquetas = etiquetas.drop(columns=["CÓDIGO"])
    lab = lab.drop(columns=["CÓDIGO"])

    metadata = {
        "metodo": "enlace_vas_por_centro_fitbit_medicion",
        "correcciones_manifest": [c["nombre"] for c in CORRECCIONES_MANIFEST],
        "correcciones_vas": [c["nombre"] for c in CORRECCIONES_VAS],
        "n_sesiones": len(etiquetas),
        "n_sesiones_etiquetadas": int(etiquetas["etiquetada"].sum()),
        "n_personas": n_personas_reales,
        "n_persona_uid": int(lab["persona_uid"].nunique()),
        "n_sujeto_id": int(lab["sujeto_id"].nunique()),
        "n_sujeto_id_con_dos_personas": int(
            (lab.groupby("sujeto_id")["persona_uid"].nunique() > 1).sum()),
        "n_grabaciones_etiquetadas": int(lab["n_grabaciones"].sum()),
        "n_ventanas_etiquetadas": (
            int(lab["n_ventanas"].sum()) if lab["n_ventanas"].notna().any() else None),
        "concordancia_fecha": lab["concordancia_fecha"].value_counts().to_dict(),
        "sesiones_fecha_discrepante": sorted(
            lab.loc[lab["concordancia_fecha"] == "discrepa", "sesion_uid"]),
        "sesiones_sin_vas": sorted(etiquetas.loc[~etiquetas["etiquetada"], "sesion_uid"]),
    }
    print(
        f"Etiquetas: {metadata['n_sesiones_etiquetadas']} de "
        f"{metadata['n_sesiones']} sesión(es) con VAS, "
        f"{metadata['n_personas']} persona(s), "
        f"{metadata['n_grabaciones_etiquetadas']} grabación(es)."
    )
    return etiquetas, metadata


def expandir_a_grabaciones(etiquetas, manifest, log=None):
    """
    Replica la etiqueta de sesión a cada grabación de esa sesión.

    Es la réplica que el protocolo autoriza (el VAS se responde una vez por
    sesión y las 7 grabaciones son posteriores a esa respuesta), pero se hace
    aquí de forma explícita para que quede claro que **no** añade información:
    el n del estudio sigue siendo el número de sesiones.

    Devuelve
    --------
    grabaciones : pandas.DataFrame
        Una fila por grabación del manifest, con la etiqueta de su sesión, el
        `persona_uid`, el `split` y (si se pasa `log`) `n_ventanas` y si la
        grabación llegó a procesarse.
    """
    m = corregir_manifest(manifest) if "sesion_uid" not in manifest.columns else manifest.copy()
    columnas = ["sesion_uid", "persona_uid", "sujeto_id", "centro", "medicion",
                COLUMNA_VAS, "etiquetada", "split", "ocupante", "concordancia_fecha"]
    grabaciones = m[["nombre_nuevo", "sesion_uid", "condicion_norm", "corregida"]].merge(
        etiquetas[columnas], on="sesion_uid", how="left")
    grabaciones = grabaciones.rename(columns={"nombre_nuevo": "archivo",
                                              "condicion_norm": "condicion"})
    if len(grabaciones) != len(m):
        raise ValueError(
            f"La expansión ha cambiado el número de grabaciones ({len(m)} -> "
            f"{len(grabaciones)}); habría sesion_uid repetido en 'etiquetas'."
        )

    if log is not None:
        _validar_columnas(log, ("archivo", "estado", "n_epocas"), "log")
        estado = log.set_index("archivo")["estado"]
        epocas = log.set_index("archivo")["n_epocas"]
        grabaciones["procesada"] = grabaciones["archivo"].map(estado).eq("ok")
        grabaciones["n_ventanas"] = grabaciones["archivo"].map(epocas).where(
            grabaciones["procesada"])
    else:
        grabaciones["procesada"] = np.nan
        grabaciones["n_ventanas"] = np.nan
    return grabaciones


def unir_etiquetas(features, grabaciones):
    """
    Añade la etiqueta a la matriz de features (una fila por época).

    Une por `archivo_origen`, no por sujeto ni por sesión: es la única clave
    que sobrevive a las correcciones de centro, porque el `sujeto` que lleva la
    matriz de features es el que tenía el manifest **antes** de corregirlo.

    Si la matriz ya trae `centro` o `split` (los añade `procesar_lote` desde el
    manifest sin corregir), se sustituyen por los corregidos informando de
    cuántos valores cambian.

    Raises
    ------
    ValueError
        Si algún `archivo_origen` no aparece en `grabaciones`, o si la unión
        cambiaría el número de filas.
    """
    if "archivo_origen" not in features.columns:
        raise ValueError("'features' necesita la columna 'archivo_origen'.")
    _validar_columnas(grabaciones, ("archivo", COLUMNA_VAS, "sesion_uid", "persona_uid"),
                      "grabaciones")

    sin_etiqueta = sorted(set(features["archivo_origen"]) - set(grabaciones["archivo"]))
    if sin_etiqueta:
        raise ValueError(
            f"{len(sin_etiqueta)} archivo(s) de 'features' no aparecen en la "
            f"tabla de etiquetas: {sin_etiqueta[:5]}"
            + (" ..." if len(sin_etiqueta) > 5 else "")
        )

    columnas = ["archivo", "sesion_uid", "persona_uid", "sujeto_id", "centro",
                "medicion", COLUMNA_VAS, "etiquetada", "split", "corregida"]
    aporte = grabaciones[[c for c in columnas if c in grabaciones.columns]]

    unidas = features.copy()
    for columna in ("centro", "split"):
        if columna in unidas.columns and columna in aporte.columns:
            nuevo = unidas["archivo_origen"].map(
                aporte.set_index("archivo")[columna])
            cambian = int((unidas[columna].astype(str) != nuevo.astype(str)).sum())
            if cambian:
                print(f"unir_etiquetas: '{columna}' cambia en {cambian} fila(s) "
                      f"por las correcciones de centro.")
            unidas = unidas.drop(columns=[columna])

    n_antes = len(unidas)
    unidas = unidas.merge(aporte, left_on="archivo_origen", right_on="archivo", how="left")
    unidas = unidas.drop(columns=["archivo"])
    if len(unidas) != n_antes:
        raise ValueError(
            f"La unión ha cambiado el número de épocas ({n_antes} -> "
            f"{len(unidas)}); habría 'archivo' repetido en 'grabaciones'."
        )

    con = int(unidas["etiquetada"].fillna(False).sum())
    print(f"Etiqueta unida: {con} de {len(unidas)} época(s) "
          f"({100 * con / len(unidas):.1f} %) tienen VAS.")
    return unidas
