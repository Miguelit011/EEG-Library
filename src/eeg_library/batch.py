"""
Módulo de procesamiento por lotes (Fase 11).

Encadena las Fases 1-9 sobre muchos archivos EDF y persiste el resultado en
disco: una tabla de features por época para el Random Forest (Fases 9-10) y,
opcionalmente, las ventanas crudas de señal para la CNN. Ambas salidas nacen
del mismo epocado de la Fase 7, así que la fila `i` del parquet corresponde a
la ventana `i` del `.npy` del mismo archivo.

Decisiones de diseño (justificación completa en ESTADO.md, Fase 11):

- **El origen de la metadata es el manifest, no el nombre del archivo.**
  `manifest_canonical.csv` ya trae sujeto, condición y medición parseados y
  validados, más la frecuencia de muestreo y la duración leídas de la
  cabecera EDF real. Parsear el nombre de archivo sería reimplementar peor
  algo que ya está hecho, y dejaría fuera columnas (centro, timestamp) que
  hacen falta para el enlace con la etiqueta (Fase 12).
- **Un proceso por archivo, no hilos.** El coste está dominado por ICA
  (~60 % del tiempo), que es CPU pura y libera el GIL solo parcialmente.
  Los workers reciben rutas y devuelven registros de log; nunca objetos de
  MNE, que son caros de serializar.
- **Cada worker escribe su propia salida.** Así el resultado no viaja entre
  procesos y reanudar un lote interrumpido es simplemente saltarse los
  archivos cuya salida ya existe.
- **Un fallo no detiene el lote.** Cada archivo se procesa dentro de un
  try/except que registra en qué fase falló; el lote termina y el log dice
  qué hay que revisar.
"""

import datetime
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

# Fases del pipeline en el orden corregido (ver ESTADO.md).
FASES = (
    "carga", "recorte", "filtros", "calidad", "car", "ica", "iclabel",
    "ica_aplicar", "epocado", "psd", "bandas", "escritura",
)


class ErrorDeFase(RuntimeError):
    """
    Fallo al procesar un archivo, conservando en qué fase ocurrió.

    Con 4574 archivos, saber que algo lanzó `ValueError` no sirve de nada; lo
    accionable es si falló al leer el EDF, al epocar una grabación demasiado
    corta o al calcular el índice sobre un clúster entero excluido. La fase
    queda en el log y agrupa los fallos por causa real.

    Arrastra además el `registro` parcial acumulado hasta el momento del
    fallo. Sin él, la fila de log de un archivo que revienta al ajustar ICA no
    diría *por qué* se quedó sin canales, que es justo el dato con el que se
    diagnostica: las fases anteriores sí se completaron y sus métricas
    (canales buenos, canales excluidos y su motivo) ya estaban calculadas.
    """

    def __init__(self, fase, original, registro=None):
        super().__init__(f"[{fase}] {type(original).__name__}: {original}")
        self.fase = fase
        self.original = original
        self.registro = registro or {}

# Columnas del manifest que se usan como origen de metadata por archivo.
COLUMNAS_MANIFEST = ("nombre_nuevo", "sujeto_id", "condicion_norm", "medicion",
                     "centro_efectivo", "timestamp_iso")

NOMBRE_LOG = "procesamiento_log.csv"
NOMBRE_RESUMEN = "procesamiento_resumen.json"
NOMBRE_FEATURES = "features_all.parquet"


def cargar_tareas(ruta_manifest, dir_edf, ruta_splits=None):
    """
    Construye la lista de archivos a procesar a partir del manifest.

    Añade dos columnas que no están en el manifest y que necesitan las fases
    posteriores:

    - `split`: partición train/val/test del sujeto, leída de `splits.csv`. Es
      por sujeto, no por grabación, así que se une por `sujeto_id`.
    - `orden_en_sesion`: posición de la grabación dentro de su sesión (1 = la
      primera), derivada de `timestamp_iso`. El protocolo tiene orden fijo
      (OC → OA → STROOP → PASAT → PVTB → MSIT → PVSAT), así que esta posición
      es un proxy del tiempo acumulado en tarea **independiente de la señal
      EEG**. La Fase 13 usa su equivalente `condicion_orden`, derivado del
      nombre de la condición en vez de las marcas de tiempo, que es estable
      aunque a la sesión le falte alguna grabación.

    Parámetros
    ----------
    ruta_manifest : str
        Ruta a `manifest_canonical.csv`.
    dir_edf : str
        Directorio que contiene los EDF (`dataset_final/`).
    ruta_splits : str o None
        Ruta a `splits.csv`. Si es None, la columna `split` queda a None.

    Devuelve
    --------
    tareas : pandas.DataFrame
        Una fila por archivo, con las columnas: ruta, archivo, sujeto,
        condicion, sesion, centro, split, orden_en_sesion, timestamp_iso.
    """
    manifest = pd.read_csv(ruta_manifest, encoding="utf-8-sig")

    faltan = [c for c in COLUMNAS_MANIFEST if c not in manifest.columns]
    if faltan:
        raise ValueError(
            f"Al manifest '{ruta_manifest}' le faltan columnas necesarias: {faltan}."
        )

    tareas = pd.DataFrame({
        "archivo": manifest["nombre_nuevo"],
        "ruta": [os.path.join(dir_edf, n) for n in manifest["nombre_nuevo"]],
        "sujeto": manifest["sujeto_id"],
        "condicion": manifest["condicion_norm"],
        "sesion": manifest["medicion"],
        "centro": manifest["centro_efectivo"],
        "timestamp_iso": manifest["timestamp_iso"],
    })

    # Orden dentro de la sesión: 1 = primera grabación del sujeto en esa
    # medición. Se ordena por marca de tiempo real, no por condición, para no
    # dar por supuesto el orden del protocolo en las sesiones incompletas.
    #
    # Queda como entero nullable (Int64) a propósito: en el dataset hay una
    # sesión sin marca de tiempo en ninguna de sus 7 grabaciones, y ahí el
    # orden es genuinamente desconocido. Inventarlo (p. ej. ordenando por
    # nombre de condición) metería una posición falsa justo en la variable
    # que sirve de proxy de tiempo en tarea; es preferible que esas filas
    # queden como <NA> y quien las use decida qué hacer con ellas.
    momento = pd.to_datetime(
        tareas["timestamp_iso"], format="ISO8601", utc=True, errors="coerce"
    )
    tareas["orden_en_sesion"] = (
        momento.groupby([tareas["sujeto"], tareas["sesion"]])
        .rank(method="first")
        .astype("Int64")
    )

    if ruta_splits is not None:
        splits = pd.read_csv(ruta_splits, encoding="utf-8-sig")
        tareas = tareas.merge(
            splits[["sujeto_id", "split"]].rename(columns={"sujeto_id": "sujeto"}),
            on="sujeto", how="left",
        )
    else:
        tareas["split"] = None

    return tareas.sort_values(["sujeto", "sesion", "orden_en_sesion"]).reset_index(drop=True)


def procesar_archivo(ruta, sujeto, condicion, sesion, extras=None,
                     devolver_ventanas=False, random_state=42):
    """
    Ejecuta las Fases 1-9 sobre un único archivo EDF.

    Parámetros
    ----------
    ruta : str
        Ruta al archivo EDF.
    sujeto, condicion, sesion : str
        Identificadores de origen, que se propagan a `epochs.metadata` y de
        ahí a la tabla de features.
    extras : dict o None
        Columnas adicionales a añadir a la tabla de features (p. ej. centro,
        split, orden_en_sesion). Se añaden con el mismo valor en todas las
        filas del archivo.
    devolver_ventanas : bool
        Si True, devuelve también las ventanas crudas de los 14 canales EEG
        (para la CNN), alineadas fila a fila con la tabla de features.
    random_state : int
        Semilla de ICA, para que el lote sea reproducible.

    Devuelve
    --------
    features : pandas.DataFrame
        Una fila por época (ver `extraer_bandas`, Fase 9), más las columnas
        de `extras`.
    ventanas : numpy.ndarray o None
        Array (n_epocas, 14, n_muestras) en float32, o None si
        `devolver_ventanas` es False.
    registro : dict
        Traza del procesamiento del archivo: parámetros observados en cada
        fase, tiempos y estado. Es lo que alimenta el log del lote.

    Raises
    ------
    ErrorDeFase
        Si falla cualquier fase, envolviendo la excepción original y
        anotando en cuál ocurrió.
    """
    from eeg_library.reader import cargar_eeg
    from eeg_library.filtering import recortar_registro, aplicar_filtros
    from eeg_library.quality import evaluar_calidad
    from eeg_library.referencing import aplicar_car
    from eeg_library.artifacts import (
        entrenar_ica, identificar_componentes_oculares, eliminar_artefactos_oculares,
    )
    from eeg_library.epoching import crear_epocas
    from eeg_library.spectral import calcular_psd
    from eeg_library.bands import extraer_bandas

    extras = extras or {}
    archivo = os.path.basename(ruta)
    registro = {
        "archivo": archivo, "sujeto": sujeto, "condicion": condicion,
        "sesion": sesion, "estado": "error", "fase_fallo": None, "error": None,
    }
    registro.update(extras)
    t0 = time.time()

    # Una única envoltura para todo el encadenado: `fase` se va actualizando
    # y queda registrada en el ErrorDeFase, que es lo que el log necesita.
    fase = FASES[0]
    try:
        fase = "carga"
        raw, meta_carga = cargar_eeg(ruta)
        registro.update(
            n_canales=meta_carga["n_canales"],
            n_canales_eeg=meta_carga["n_canales_eeg"],
            sfreq_original=meta_carga["original_sfreq"],
            resampleado=meta_carga["resampled"],
            duracion_s=meta_carga["duracion_s"],
        )

        fase = "recorte"
        raw, _ = recortar_registro(raw)

        fase = "filtros"
        raw, _ = aplicar_filtros(raw)

        fase = "calidad"
        good_channels, excluidos, _ = evaluar_calidad(raw)
        registro.update(
            n_canales_buenos=len(good_channels),
            canales_excluidos="|".join(sorted(excluidos)),
        )

        fase = "car"
        raw, meta_car = aplicar_car(raw, good_channels)
        registro["car"] = meta_car["metodo"]

        fase = "ica"
        ica, meta_ica = entrenar_ica(raw, good_channels, random_state=random_state)
        registro["n_componentes_ica"] = meta_ica["n_components"]

        fase = "iclabel"
        componentes, diagnostico, _ = identificar_componentes_oculares(raw, ica)
        registro["n_componentes_oculares"] = len(componentes)
        # Categorías de artefacto detectadas pero no tratadas (ver Fase 6): se
        # registran para poder justificarlo en la memoria sin reprocesar el lote.
        registro["diagnostico_otras"] = "|".join(
            f"{cat}:{len(items)}" for cat, items in sorted(diagnostico.items()) if items
        )

        fase = "ica_aplicar"
        raw, meta_limpieza = eliminar_artefactos_oculares(raw, ica, componentes)
        registro["varianza_eliminada"] = meta_limpieza["varianza_explicada_eliminada"]

        fase = "epocado"
        epochs, meta_epocado = crear_epocas(
            raw, sujeto=sujeto, condicion=condicion, sesion=sesion,
            archivo_origen=archivo,
        )
        registro["n_epocas"] = meta_epocado["n_epocas"]

        fase = "psd"
        spectrum, _ = calcular_psd(epochs)

        fase = "bandas"
        features, meta_bandas = extraer_bandas(spectrum, canales_validos=good_channels)
        registro["cluster_descartados"] = "|".join(
            meta_bandas["cluster_frontal_descartados"]
            + meta_bandas["cluster_posterior_descartados"]
        )

        for clave, valor in extras.items():
            features[clave] = valor

        ventanas = None
        if devolver_ventanas:
            # Mismo objeto Epochs que alimentó la PSD, así que la fila i de
            # `features` y la ventana i de `ventanas` son la misma época.
            ventanas = epochs.get_data(picks="eeg", copy=True).astype(np.float32)
            if ventanas.shape[0] != len(features):
                raise RuntimeError(
                    f"Desalineación entre ventanas ({ventanas.shape[0]}) y "
                    f"features ({len(features)})."
                )
            registro["forma_ventanas"] = "x".join(str(d) for d in ventanas.shape)

    except Exception as exc:
        registro["segundos"] = round(time.time() - t0, 2)
        raise ErrorDeFase(fase, exc, registro) from exc

    registro["estado"] = "ok"
    registro["segundos"] = round(time.time() - t0, 2)
    return features, ventanas, registro


def _procesar_y_guardar(tarea):
    """
    Worker del lote: procesa un archivo y escribe su salida en disco.

    Se define a nivel de módulo y recibe/devuelve solo tipos simples porque
    `ProcessPoolExecutor` en Windows arranca los workers con *spawn*, lo que
    obliga a que tanto la función como sus argumentos sean serializables.
    Devuelve únicamente el registro de log: los datos no viajan entre
    procesos, los escribe el propio worker.
    """
    import warnings
    import mne

    warnings.filterwarnings("ignore")
    mne.set_log_level("ERROR")

    import contextlib
    import io

    ruta_features = tarea["ruta_features"]
    ruta_ventanas = tarea.get("ruta_ventanas")

    try:
        # Las funciones de las Fases 1-9 imprimen un resumen por consola, útil
        # en uso interactivo pero ilegible con varios procesos a la vez. Se
        # captura y se descarta: lo que hay que conservar va al log.
        with contextlib.redirect_stdout(io.StringIO()):
            features, ventanas, registro = procesar_archivo(
                tarea["ruta"], tarea["sujeto"], tarea["condicion"], tarea["sesion"],
                extras=tarea.get("extras"),
                devolver_ventanas=ruta_ventanas is not None,
                random_state=tarea.get("random_state", 42),
            )

        try:
            features.to_parquet(ruta_features, index=False)
            if ruta_ventanas is not None:
                np.save(ruta_ventanas, ventanas)
        except Exception as exc:
            raise ErrorDeFase("escritura", exc, registro) from exc

        registro["ruta_features"] = ruta_features
        registro["ruta_ventanas"] = ruta_ventanas
        return registro

    except ErrorDeFase as exc:
        return _registro_de_fallo(tarea, exc.fase, exc.original, exc.registro)
    except Exception as exc:  # fallo fuera del encadenado de fases
        return _registro_de_fallo(tarea, "desconocida", exc)


def _registro_de_fallo(tarea, fase, exc, registro_parcial=None):
    """
    Fila de log de un archivo que no se pudo procesar.

    Parte del `registro_parcial` acumulado hasta el fallo, de forma que las
    métricas de las fases que sí se completaron (duración, canales buenos y
    excluidos, componentes ICA...) queden en el log y el fallo se pueda
    diagnosticar sin volver a procesar el archivo.
    """
    registro = dict(registro_parcial or {})
    registro.update({
        "archivo": os.path.basename(tarea["ruta"]),
        "sujeto": tarea["sujeto"],
        "condicion": tarea["condicion"],
        "sesion": tarea["sesion"],
        "estado": "error",
        "fase_fallo": fase,
        "error": f"{type(exc).__name__}: {exc}",
        "traza": traceback.format_exc(limit=3),
        **(tarea.get("extras") or {}),
    })
    return registro


def procesar_lote(tareas, dir_salida, dir_ventanas=None, n_procesos=None,
                  reanudar=True, random_state=42, limite=None):
    """
    Procesa un conjunto de archivos en paralelo y persiste features y log.

    Estructura de salida::

        dir_salida/
            features/<archivo>.parquet     una tabla por EDF
            features_all.parquet           concatenación de todas
            procesamiento_log.csv          una fila por archivo, con estado
            procesamiento_resumen.json     parámetros y agregados del lote
        dir_ventanas/
            <archivo>.npy                  ventanas crudas (n_epocas, 14, n)

    Parámetros
    ----------
    tareas : pandas.DataFrame
        Salida de `cargar_tareas`, o cualquier DataFrame con las columnas
        ruta, archivo, sujeto, condicion, sesion.
    dir_salida : str
        Directorio donde escribir features, log y resumen.
    dir_ventanas : str o None
        Directorio donde escribir las ventanas crudas para la CNN. Si es
        None, no se generan. **Debe apuntar a disco local, no a una carpeta
        sincronizada**: son ~16 GB para el dataset completo.
    n_procesos : int o None
        Procesos en paralelo. Por defecto, `os.cpu_count() - 2` (deja margen
        para que la máquina siga usable). Con `n_procesos=1` no se crea
        ningún pool y los archivos se procesan en serie en el proceso actual:
        más lento, pero deja ver los errores con su traza completa y permite
        usar un depurador, que es lo que hace falta cuando hay que averiguar
        por qué falla un archivo concreto.
    reanudar : bool
        Si True, se salta los archivos cuyo parquet ya existe. Permite
        retomar un lote interrumpido sin repetir trabajo.
    random_state : int
        Semilla de ICA, propagada a todos los archivos.
    limite : int o None
        Procesa solo los primeros `limite` archivos. Para pruebas.

    Devuelve
    --------
    log : pandas.DataFrame
        Una fila por archivo procesado en esta ejecución, con estado, métricas
        por fase y motivo de fallo si lo hubo.
    resumen : dict
        Agregados del lote: nº de archivos ok/error, épocas totales, tiempos y
        parámetros usados.
    """
    if n_procesos is None:
        n_procesos = max(1, (os.cpu_count() or 4) - 2)

    dir_features = os.path.join(dir_salida, "features")
    os.makedirs(dir_features, exist_ok=True)
    if dir_ventanas is not None:
        os.makedirs(dir_ventanas, exist_ok=True)

    tareas = tareas.head(limite) if limite else tareas
    columnas_extra = [
        c for c in ("centro", "split", "orden_en_sesion") if c in tareas.columns
    ]

    pendientes, omitidos = [], 0
    for fila in tareas.to_dict("records"):
        base = os.path.splitext(fila["archivo"])[0]
        ruta_features = os.path.join(dir_features, f"{base}.parquet")
        ruta_ventanas = (
            os.path.join(dir_ventanas, f"{base}.npy") if dir_ventanas else None
        )
        if reanudar and os.path.exists(ruta_features) and (
            ruta_ventanas is None or os.path.exists(ruta_ventanas)
        ):
            omitidos += 1
            continue
        pendientes.append({
            "ruta": fila["ruta"], "sujeto": fila["sujeto"],
            "condicion": fila["condicion"], "sesion": fila["sesion"],
            "ruta_features": ruta_features, "ruta_ventanas": ruta_ventanas,
            "random_state": random_state,
            "extras": {c: fila[c] for c in columnas_extra},
        })

    print(
        f"Lote: {len(pendientes)} archivo(s) por procesar"
        + (f", {omitidos} ya existente(s) omitido(s)" if omitidos else "")
        + f" — {n_procesos} proceso(s)."
    )

    t0 = time.time()
    registros = []
    if pendientes:
        if n_procesos == 1:
            resultados = map(_procesar_y_guardar, pendientes)
        else:
            executor = ProcessPoolExecutor(max_workers=n_procesos)
            resultados = executor.map(_procesar_y_guardar, pendientes)
        try:
            for i, registro in enumerate(resultados, start=1):
                registros.append(registro)
                if i % 50 == 0 or i == len(pendientes):
                    n_err = sum(1 for r in registros if r["estado"] != "ok")
                    ritmo = (time.time() - t0) / i
                    restante = ritmo * (len(pendientes) - i)
                    print(
                        f"  {i}/{len(pendientes)} — {n_err} error(es) — "
                        f"{ritmo:.2f} s/archivo — quedan ~{restante/60:.0f} min",
                        flush=True,
                    )
        finally:
            if n_procesos != 1:
                executor.shutdown()

    segundos = time.time() - t0
    log = pd.DataFrame(registros)

    # El log se acumula entre ejecuciones para que reanudar un lote no borre
    # la traza de lo ya procesado: al reanudar, los archivos omitidos no
    # generan registro nuevo y su fila debe seguir siendo la de la ejecución
    # que sí los procesó.
    ruta_log = os.path.join(dir_salida, NOMBRE_LOG)
    previo = pd.read_csv(ruta_log) if os.path.exists(ruta_log) else pd.DataFrame()
    log_total = pd.concat([previo, log], ignore_index=True) if not previo.empty else log
    if not log_total.empty:
        log_total = log_total.drop_duplicates("archivo", keep="last").reset_index(drop=True)
        log_total.to_csv(ruta_log, index=False, encoding="utf-8")

    n_ok = int((log_total["estado"] == "ok").sum()) if not log_total.empty else 0
    n_error = int((log_total["estado"] != "ok").sum()) if not log_total.empty else 0

    # Solo cuentan las épocas de los archivos que llegaron hasta el final: una
    # grabación que falla después del epocado (p. ej. al calcular el índice
    # sobre un clúster entero excluido) deja su `n_epocas` en el registro
    # parcial, pero no escribe ningún parquet, así que sumarla daría un total
    # que no cuadra con `features_all.parquet`.
    if "n_epocas" in log_total and not log_total.empty:
        n_epocas_total = int(log_total.loc[log_total["estado"] == "ok", "n_epocas"].sum())
    else:
        n_epocas_total = 0
    resumen = {
        "metodo": "lote_fases_1_9",
        "n_archivos_solicitados": len(tareas),
        "n_procesados_ahora": len(pendientes),
        "n_omitidos_por_reanudar": omitidos,
        "n_ok": n_ok,
        "n_error": n_error,
        "n_epocas_total": n_epocas_total,
        "n_procesos": n_procesos,
        "random_state": random_state,
        "dir_features": dir_features,
        "dir_ventanas": dir_ventanas,
        "segundos": round(segundos, 1),
        "segundos_por_archivo": round(segundos / len(pendientes), 2) if pendientes else None,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(dir_salida, NOMBRE_RESUMEN), "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, indent=2, ensure_ascii=False)

    print(
        f"\nLote terminado en {segundos/60:.1f} min — "
        f"{n_ok} ok / {n_error} error(es) — "
        f"{resumen['n_epocas_total']} época(s)."
    )
    if n_error:
        print(f"Revisa los fallos en {ruta_log}.")

    return log_total, resumen


def consolidar_features(dir_salida, ruta_destino=None):
    """
    Concatena los parquet por archivo en una única tabla.

    Se hace en un paso aparte y no dentro de `procesar_lote` para que un lote
    interrumpido siga dejando utilizables los parquet ya escritos, y para
    poder reconsolidar tras reanudar sin reprocesar nada.

    Verifica que todos los archivos compartan exactamente las mismas columnas
    y en el mismo orden: si un archivo trajera otro conjunto de canales, la
    concatenación introduciría columnas con NaN en silencio.

    Devuelve
    --------
    features : pandas.DataFrame
        Tabla concatenada, ya escrita en `ruta_destino`.
    """
    dir_features = os.path.join(dir_salida, "features")
    rutas = sorted(
        os.path.join(dir_features, f)
        for f in os.listdir(dir_features) if f.endswith(".parquet")
    )
    if not rutas:
        raise ValueError(f"No hay parquet que consolidar en '{dir_features}'.")

    partes, columnas_ref = [], None
    for ruta in rutas:
        parte = pd.read_parquet(ruta)
        if columnas_ref is None:
            columnas_ref = list(parte.columns)
        elif list(parte.columns) != columnas_ref:
            faltan = set(columnas_ref) - set(parte.columns)
            sobran = set(parte.columns) - set(columnas_ref)
            raise ValueError(
                f"'{os.path.basename(ruta)}' no tiene las mismas columnas que el "
                f"resto. Faltan: {sorted(faltan)}. Sobran: {sorted(sobran)}."
            )
        partes.append(parte)

    features = pd.concat(partes, ignore_index=True)
    ruta_destino = ruta_destino or os.path.join(dir_salida, NOMBRE_FEATURES)
    features.to_parquet(ruta_destino, index=False)

    print(
        f"Consolidado: {len(rutas)} archivo(s) -> {len(features)} época(s) x "
        f"{features.shape[1]} columna(s) en {ruta_destino}"
    )
    return features
