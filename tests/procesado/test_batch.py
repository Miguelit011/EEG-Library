"""
Tests para el módulo de procesamiento por lotes (Fase 11).

A diferencia del resto de módulos, aquí los tests **escriben archivos EDF
reales** en `tmp_path` (vía `mne.export.export_raw`) en vez de mockear
`mne.io.read_raw_edf`. Dos razones: lo que se prueba es precisamente el
recorrido completo desde el archivo en disco hasta el parquet, y el lote
puede ejecutarse en procesos separados, donde un mock aplicado con
`monkeypatch` en el proceso padre no existiría.

Los EDF sintéticos reproducen la estructura del dataset final: los 14
electrodos más canales `CQ.*` y de telemetría, con la nomenclatura de punto.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
import mne

from eeg_library.procesado.batch import (
    ErrorDeFase,
    cargar_tareas,
    procesar_archivo,
    procesar_lote,
    consolidar_features,
    NOMBRE_LOG,
    NOMBRE_RESUMEN,
)
from eeg_library.preprocesado.quality import CANALES_EPOC

# Avisos esperados de ICLabel, que salen también con datos reales: la CAR se
# aplica restando la media a mano (referencing.py), así que MNE no la registra,
# y el pipeline filtra a 1-30 Hz y no a 1-100 Hz (caveat conocido, docs/ESTADO.md,
# Fase 6).
pytestmark = [
    pytest.mark.filterwarnings("ignore:The provided Raw instance does not seem to be referenced:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:The provided Raw instance is not filtered between 1 and 100 Hz:RuntimeWarning"),
]

SFREQ = 256


def _escribir_edf(destino, nombre, duracion_s=30.0, sfreq=SFREQ, cq=4.0, semilla=0):
    """
    Escribe un EDF sintético con la estructura del dataset final.

    Devuelve la ruta del archivo creado.
    """
    rng = np.random.default_rng(semilla)
    n = int(duracion_s * sfreq)

    nombres, filas = [], []
    for i, canal in enumerate(CANALES_EPOC):
        nombres.append(canal)
        # Ruido con una ligera variación de amplitud entre canales, para que
        # el MAD inter-canal de evaluar_calidad sea > 0.
        filas.append(rng.normal(0, 5e-6 * (1 + 0.03 * i), n))
    for canal in CANALES_EPOC:
        nombres.append(f"CQ.{canal}")
        filas.append(np.full(n, cq))
    for canal in ("TimestampS", "Counter", "MOT.AccX"):
        nombres.append(canal)
        filas.append(rng.normal(0, 1.0, n))

    # Los tipos se fijan solo para *escribir*: `export_raw` con
    # physical_range="auto" calcula un único rango físico por tipo de canal, y
    # si los CQ (valor 4.0) compartieran tipo con la señal EEG (±5e-6), la
    # cuantización a int16 sobre ese rango común aplastaría el EEG a cero.
    # Al leer el archivo, MNE vuelve a tipar los 31 canales como 'eeg' —que es
    # justo la entrada que debe recibir `cargar_eeg`— porque el EDF no guarda
    # el tipo de canal.
    tipos = ["eeg" if n in CANALES_EPOC else "misc" for n in nombres]
    info = mne.create_info(nombres, sfreq, ch_types=tipos)
    raw = mne.io.RawArray(np.array(filas), info, verbose=False)

    os.makedirs(destino, exist_ok=True)
    ruta = os.path.join(destino, nombre)
    mne.export.export_raw(ruta, raw, fmt="edf", overwrite=True, verbose=False)
    return ruta


@pytest.fixture
def dir_edf(tmp_path):
    return str(tmp_path / "edf")


def _tareas_de(rutas, sujetos=None, condiciones=None, sesiones=None):
    """DataFrame de tareas mínimo, equivalente a la salida de cargar_tareas."""
    n = len(rutas)
    return pd.DataFrame({
        "ruta": rutas,
        "archivo": [os.path.basename(r) for r in rutas],
        "sujeto": sujetos or [f"C1_F{i:03d}" for i in range(n)],
        "condicion": condiciones or ["PASAT"] * n,
        "sesion": sesiones or ["M4"] * n,
    })


# ---------------------------------------------------------------------------
# cargar_tareas — metadata desde el manifest, no desde el nombre de archivo
# ---------------------------------------------------------------------------

def _escribir_metadata(tmp_path, con_timestamps=True):
    """manifest_canonical.csv y splits.csv mínimos, con dos sujetos."""
    filas = []
    # Protocolo en orden OC -> OA -> STROOP, con marcas de tiempo crecientes.
    for sujeto, centro in (("C1_F001", "C1"), ("C2_F002", "C2")):
        for i, condicion in enumerate(("OC", "OA", "STROOP")):
            filas.append({
                "nombre_nuevo": f"sub-{centro}{sujeto.split('_')[1]}_ses-M4_task-{condicion}_eeg.edf",
                "sujeto_id": sujeto,
                "condicion_norm": condicion,
                "medicion": "M4",
                "centro_efectivo": centro,
                "timestamp_iso": f"2025-04-25T10:{10 + i * 5:02d}:00+02:00"
                if con_timestamps else None,
            })
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(filas).to_csv(manifest, index=False, encoding="utf-8")

    splits = tmp_path / "splits.csv"
    pd.DataFrame({
        "sujeto_id": ["C1_F001", "C2_F002"],
        "split": ["train", "test"],
    }).to_csv(splits, index=False, encoding="utf-8")
    return str(manifest), str(splits)


def test_cargar_tareas_deriva_orden_de_sesion_y_split(tmp_path):
    manifest, splits = _escribir_metadata(tmp_path)

    tareas = cargar_tareas(manifest, "/datos/edf", splits)

    assert len(tareas) == 6
    assert set(tareas.columns) >= {
        "archivo", "ruta", "sujeto", "condicion", "sesion", "centro",
        "split", "orden_en_sesion",
    }
    # El orden sale de la marca de tiempo, no del nombre de la condición.
    primero = tareas[tareas.sujeto == "C1_F001"].sort_values("orden_en_sesion")
    assert list(primero.condicion) == ["OC", "OA", "STROOP"]
    assert list(primero.orden_en_sesion) == [1, 2, 3]
    # El split es por sujeto y se une desde splits.csv.
    assert set(tareas[tareas.sujeto == "C1_F001"].split) == {"train"}
    assert set(tareas[tareas.sujeto == "C2_F002"].split) == {"test"}


def test_cargar_tareas_deja_orden_nulo_si_no_hay_timestamp(tmp_path):
    """
    Una sesión sin marcas de tiempo tiene orden desconocido, y debe quedar
    como <NA> en vez de inventarse una posición: es la variable que sirve de
    proxy de tiempo acumulado en tarea.
    """
    manifest, splits = _escribir_metadata(tmp_path, con_timestamps=False)

    tareas = cargar_tareas(manifest, "/datos/edf", splits)

    assert tareas.orden_en_sesion.isna().all()


def test_cargar_tareas_sin_splits_deja_la_columna_vacia(tmp_path):
    manifest, _ = _escribir_metadata(tmp_path)
    tareas = cargar_tareas(manifest, "/datos/edf")
    assert tareas.split.isna().all()


def test_cargar_tareas_exige_las_columnas_del_manifest(tmp_path):
    incompleto = tmp_path / "malo.csv"
    pd.DataFrame({"nombre_nuevo": ["x.edf"]}).to_csv(incompleto, index=False)
    with pytest.raises(ValueError, match="faltan columnas"):
        cargar_tareas(str(incompleto), "/datos/edf")


# ---------------------------------------------------------------------------
# procesar_archivo — Fases 1-9 sobre un EDF real
# ---------------------------------------------------------------------------

def test_procesar_archivo_devuelve_features_y_registro(dir_edf):
    ruta = _escribir_edf(dir_edf, "sub-C1F001_ses-M4_task-PASAT_eeg.edf")

    features, ventanas, registro = procesar_archivo(ruta, "C1_F001", "PASAT", "M4")

    assert registro["estado"] == "ok"
    assert registro["n_canales_eeg"] == len(CANALES_EPOC)
    assert registro["n_epocas"] == len(features)
    assert ventanas is None

    # Las columnas de potencia son las de los 14 electrodos, sin telemetría.
    columnas_potencia = [
        c for c in features.columns if c.startswith(("pot_abs_", "pot_rel_"))
    ]
    assert len(columnas_potencia) == 112
    assert not any(
        p in c for c in columnas_potencia for p in ("CQ", "MOT", "Timestamp")
    )
    # Metadata de origen propagada desde crear_epocas.
    assert set(features.sujeto) == {"C1_F001"}
    assert set(features.condicion) == {"PASAT"}
    assert set(features.sesion) == {"M4"}


def test_procesar_archivo_ventanas_alineadas_con_features(dir_edf):
    """
    La fila i del parquet y la ventana i del .npy tienen que ser la misma
    época: la CNN y el Random Forest comparten el epocado de la Fase 7, y la
    matriz de la Fase 13 se agrega sobre las features de estas mismas filas.
    """
    ruta = _escribir_edf(dir_edf, "sub-C1F001_ses-M4_task-PASAT_eeg.edf")

    features, ventanas, registro = procesar_archivo(
        ruta, "C1_F001", "PASAT", "M4", devolver_ventanas=True
    )

    assert ventanas.shape[0] == len(features)
    assert ventanas.shape[1] == len(CANALES_EPOC)
    assert ventanas.shape[2] == 4 * SFREQ  # ventanas de 4 s
    assert ventanas.dtype == np.float32
    assert registro["forma_ventanas"] == "x".join(str(d) for d in ventanas.shape)


def test_procesar_archivo_anade_columnas_extra(dir_edf):
    ruta = _escribir_edf(dir_edf, "sub-C1F001_ses-M4_task-PASAT_eeg.edf")

    features, _, registro = procesar_archivo(
        ruta, "C1_F001", "PASAT", "M4",
        extras={"centro": "C1", "split": "train", "orden_en_sesion": 4},
    )

    assert set(features.centro) == {"C1"}
    assert set(features.split) == {"train"}
    assert set(features.orden_en_sesion) == {4}
    assert registro["split"] == "train"


@pytest.mark.parametrize(
    "duracion_s, sfreq, fase_esperada",
    [
        (30.0, 100.0, "carga"),    # por debajo de MIN_SFREQ -> cargar_eeg
        (8.0, SFREQ, "recorte"),   # más corto que los 5+5 s de margen
    ],
)
def test_procesar_archivo_indica_en_que_fase_fallo(dir_edf, duracion_s, sfreq, fase_esperada):
    """
    Con miles de archivos, el tipo de excepción no es accionable; lo que
    agrupa los fallos por causa real es la fase en la que ocurrieron.
    """
    ruta = _escribir_edf(dir_edf, "malo.edf", duracion_s=duracion_s, sfreq=sfreq)

    with pytest.raises(ErrorDeFase) as excinfo:
        procesar_archivo(ruta, "C1_F001", "PASAT", "M4")

    assert excinfo.value.fase == fase_esperada
    assert excinfo.value.original is not None


# ---------------------------------------------------------------------------
# procesar_lote — paralelismo, aislamiento de fallos y reanudación
# ---------------------------------------------------------------------------

def test_procesar_lote_escribe_features_log_y_resumen(dir_edf, tmp_path):
    rutas = [
        _escribir_edf(dir_edf, f"sub-C1F00{i}_ses-M4_task-PASAT_eeg.edf", semilla=i)
        for i in range(2)
    ]
    salida = str(tmp_path / "salida")

    log, resumen = procesar_lote(_tareas_de(rutas), salida, n_procesos=1)

    assert resumen["n_ok"] == 2
    assert resumen["n_error"] == 0
    assert (log.estado == "ok").all()

    for ruta in rutas:
        base = os.path.splitext(os.path.basename(ruta))[0]
        assert os.path.exists(os.path.join(salida, "features", f"{base}.parquet"))
    assert os.path.exists(os.path.join(salida, NOMBRE_LOG))

    with open(os.path.join(salida, NOMBRE_RESUMEN), encoding="utf-8") as fh:
        guardado = json.load(fh)
    assert guardado["n_ok"] == 2
    assert guardado["n_epocas_total"] == int(log.n_epocas.sum())


def test_procesar_lote_aisla_los_fallos(dir_edf, tmp_path):
    """Un archivo ilegible no puede tumbar el lote entero."""
    bueno = _escribir_edf(dir_edf, "bueno.edf")
    malo = _escribir_edf(dir_edf, "malo.edf", duracion_s=8.0)
    salida = str(tmp_path / "salida")

    log, resumen = procesar_lote(_tareas_de([bueno, malo]), salida, n_procesos=1)

    assert resumen["n_ok"] == 1
    assert resumen["n_error"] == 1
    fallo = log[log.estado == "error"].iloc[0]
    assert fallo.archivo == "malo.edf"
    assert fallo.fase_fallo == "recorte"
    assert isinstance(fallo.error, str) and fallo.error

    # El archivo bueno se escribió igualmente.
    assert os.path.exists(os.path.join(salida, "features", "bueno.parquet"))
    assert not os.path.exists(os.path.join(salida, "features", "malo.parquet"))

    # El total de épocas cuenta solo lo que llegó a disco, para que cuadre con
    # features_all.parquet.
    escritas = len(pd.read_parquet(os.path.join(salida, "features", "bueno.parquet")))
    assert resumen["n_epocas_total"] == escritas


def test_n_epocas_total_ignora_las_epocas_de_archivos_fallidos(dir_edf, tmp_path):
    """
    Un archivo puede fallar *después* del epocado y dejar su `n_epocas` en el
    registro parcial sin haber escrito ningún parquet. Sumarlo inflaría el
    total frente a lo que hay realmente en disco.
    """
    ruta = _escribir_edf(dir_edf, "f0.edf")
    salida = str(tmp_path / "salida")
    tareas = _tareas_de([ruta])

    _, resumen = procesar_lote(tareas, salida, n_procesos=1)
    consolidadas = len(consolidar_features(salida))

    assert resumen["n_epocas_total"] == consolidadas


def test_registro_de_fallo_conserva_el_diagnostico_de_las_fases_superadas(dir_edf, tmp_path):
    """
    Un archivo que revienta a mitad del encadenado debe dejar en el log las
    métricas de las fases que sí se completaron. Sin ellas, la fila de un
    fallo en ICA diría que no había canales buenos pero no *por qué*, que es
    justo lo que hace falta para diagnosticarlo sin reprocesar el archivo.
    """
    # CQ = 1.0 en los 14 electrodos: todos quedan excluidos por el criterio 1
    # de evaluar_calidad, ICA se queda sin canales y falla.
    ruta = _escribir_edf(dir_edf, "cq_malo.edf", cq=1.0)
    salida = str(tmp_path / "salida")

    log, _ = procesar_lote(_tareas_de([ruta]), salida, n_procesos=1)

    fila = log.iloc[0]
    assert fila.estado == "error"
    assert fila.fase_fallo == "ica"
    # Diagnóstico de las fases anteriores, que sí se ejecutaron.
    assert fila.n_canales_eeg == len(CANALES_EPOC)
    assert fila.n_canales_buenos == 0
    assert set(fila.canales_excluidos.split("|")) == set(CANALES_EPOC)
    assert fila.duracion_s > 0
    assert fila.segundos > 0


def test_procesar_lote_reanuda_saltando_lo_ya_hecho(dir_edf, tmp_path):
    rutas = [
        _escribir_edf(dir_edf, f"f{i}.edf", semilla=i) for i in range(2)
    ]
    salida = str(tmp_path / "salida")
    tareas = _tareas_de(rutas)

    procesar_lote(tareas, salida, n_procesos=1)
    log2, resumen2 = procesar_lote(tareas, salida, n_procesos=1, reanudar=True)

    assert resumen2["n_procesados_ahora"] == 0
    assert resumen2["n_omitidos_por_reanudar"] == 2
    # El log conserva la traza de la primera ejecución.
    assert resumen2["n_ok"] == 2
    assert len(log2) == 2


def test_procesar_lote_sin_reanudar_reprocesa(dir_edf, tmp_path):
    ruta = _escribir_edf(dir_edf, "f0.edf")
    salida = str(tmp_path / "salida")
    tareas = _tareas_de([ruta])

    procesar_lote(tareas, salida, n_procesos=1)
    _, resumen = procesar_lote(tareas, salida, n_procesos=1, reanudar=False)

    assert resumen["n_procesados_ahora"] == 1
    assert resumen["n_omitidos_por_reanudar"] == 0


def test_procesar_lote_guarda_ventanas_alineadas(dir_edf, tmp_path):
    ruta = _escribir_edf(dir_edf, "f0.edf")
    salida = str(tmp_path / "salida")
    ventanas = str(tmp_path / "ventanas")

    procesar_lote(_tareas_de([ruta]), salida, dir_ventanas=ventanas, n_procesos=1)

    features = pd.read_parquet(os.path.join(salida, "features", "f0.parquet"))
    guardadas = np.load(os.path.join(ventanas, "f0.npy"))
    assert guardadas.shape == (len(features), len(CANALES_EPOC), 4 * SFREQ)
    assert guardadas.dtype == np.float32


def test_procesar_lote_en_paralelo_da_el_mismo_resultado_que_en_serie(dir_edf, tmp_path):
    """
    El lote real corre con varios procesos; el resultado no puede depender de
    eso. Comprueba de paso que los workers arrancan bien bajo el `spawn` de
    Windows, donde la función y sus argumentos deben ser serializables.
    """
    rutas = [_escribir_edf(dir_edf, f"f{i}.edf", semilla=i) for i in range(2)]
    tareas = _tareas_de(rutas)
    serie = str(tmp_path / "serie")
    paralelo = str(tmp_path / "paralelo")

    procesar_lote(tareas, serie, n_procesos=1)
    procesar_lote(tareas, paralelo, n_procesos=2)

    for ruta in rutas:
        base = os.path.splitext(os.path.basename(ruta))[0]
        a = pd.read_parquet(os.path.join(serie, "features", f"{base}.parquet"))
        b = pd.read_parquet(os.path.join(paralelo, "features", f"{base}.parquet"))
        pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# consolidar_features
# ---------------------------------------------------------------------------

def test_consolidar_features_concatena_todos_los_archivos(dir_edf, tmp_path):
    rutas = [_escribir_edf(dir_edf, f"f{i}.edf", semilla=i) for i in range(3)]
    salida = str(tmp_path / "salida")
    log, _ = procesar_lote(_tareas_de(rutas), salida, n_procesos=1)

    features = consolidar_features(salida)

    assert len(features) == int(log.n_epocas.sum())
    assert os.path.exists(os.path.join(salida, "features_all.parquet"))
    # Cada archivo aporta sus épocas y su origen queda trazable.
    assert features.archivo_origen.nunique() == 3


def test_consolidar_features_detecta_columnas_incompatibles(dir_edf, tmp_path):
    """
    Concatenar a ciegas parquet con columnas distintas rellenaría de NaN en
    silencio; es preferible fallar y saber qué archivo se desvía.
    """
    ruta = _escribir_edf(dir_edf, "f0.edf")
    salida = str(tmp_path / "salida")
    procesar_lote(_tareas_de([ruta]), salida, n_procesos=1)

    intruso = pd.DataFrame({"otra_cosa": [1, 2, 3]})
    intruso.to_parquet(os.path.join(salida, "features", "intruso.parquet"), index=False)

    with pytest.raises(ValueError, match="mismas columnas"):
        consolidar_features(salida)


def test_consolidar_features_sin_parquet_lanza_error(tmp_path):
    salida = tmp_path / "salida"
    (salida / "features").mkdir(parents=True)
    with pytest.raises(ValueError, match="No hay parquet"):
        consolidar_features(str(salida))
