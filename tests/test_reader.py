"""
Tests para el módulo de lectura de archivos EEG (Fase 1).

`cargar_eeg` necesita un archivo EDF real en disco (comprobación
`os.path.isfile`), pero no un contenido EDF válido: se mockea
`mne.io.read_raw_edf` para devolver un `RawArray` sintético equivalente,
igual que `test_pipeline_integracion.py` mockea `iclabel_label_components`.
Así se prueba la lógica de armonización de frecuencia de muestreo sin
depender de un archivo real ni de un escritor de EDF (no disponible en
el entorno de test).
"""

import numpy as np
import pytest
import mne

from eeg_library.reader import cargar_eeg, TARGET_SFREQ, MIN_SFREQ
from eeg_library.quality import CANALES_EPOC, evaluar_calidad

# ---------------------------------------------------------------------------
# Esquemas de nombres de canal de EmotivPro
#
# El mismo casco se ha exportado con tres convenciones distintas, tomadas de
# archivos reales: la de 60 canales usa guion bajo y mayúsculas
# (CQ_AF3, RAW_CQ, TIME_STAMP_s) y es la de los archivos de data/samples/ con
# los que se desarrolló la librería; las de 67 y 71 canales usan punto y
# CamelCase (CQ.AF3, RawCq, TimestampS) y son las del dataset final. Entre 67
# y 71 solo cambian cuatro canales de telemetría OrTimestamp*.
#
# Los 14 nombres de electrodo son idénticos en las tres (vienen del sistema
# 10-20), que es justo lo que permite tipar los canales por allowlist.
# ---------------------------------------------------------------------------

CANALES_EXPORT_60 = [
    "TIME_STAMP_s", "TIME_STAMP_ms", "OR_TIME_STAMP_s", "OR_TIME_STAMP_ms",
    "COUNTER", "INTERPOLATED",
    *CANALES_EPOC,
    "RAW_CQ", "BATTERY", "BATTERY_PERCENT",
    "MarkerIndex", "MarkerType", "MarkerValueInt", "MARKER_HARDWARE",
    *[f"CQ_{c}" for c in CANALES_EPOC], "CQ_Overall",
    "EQ_SampleRateQua", "EQ_OVERALL", *[f"EQ_{c}" for c in CANALES_EPOC],
    "CQ_CMS", "CQ_DRL",
]

CANALES_EXPORT_71 = [
    "TimestampS", "TimestampMs", "OrTimestampS", "OrTimestampMs",
    "Counter", "Interpolated",
    *CANALES_EPOC,
    "RawCq", "Battery", "BatteryPercent", "MarkerHardware",
    *[f"CQ.{c}" for c in CANALES_EPOC], "CQ.Overall",
    "EQ.SampleRateQua", "EQ.OVERALL", *[f"EQ.{c}" for c in CANALES_EPOC],
    "MOT.TimestampS", "MOT.TimestampMs", "MOT.OrTimestampS",
    "MOT.OrTimestampM", "MOT.CounterMems", "MOT.Interpolated",
    "MOT.Q0", "MOT.Q1", "MOT.Q2", "MOT.Q3",
    "MOT.AccX", "MOT.AccY", "MOT.AccZ", "MOT.MagX", "MOT.MagY", "MOT.MagZ",
]

CANALES_EXPORT_67 = [
    c for c in CANALES_EXPORT_71
    if c not in ("OrTimestampS", "OrTimestampMs",
                 "MOT.OrTimestampS", "MOT.OrTimestampM")
]

ESQUEMAS_EXPORT = {
    "export_60_guion_bajo": CANALES_EXPORT_60,
    "export_67_punto": CANALES_EXPORT_67,
    "export_71_punto": CANALES_EXPORT_71,
}


def _raw_sintetico(sfreq, duracion_s=10.0, semilla=0):
    """RawArray de 14 canales EPOC con ruido gaussiano a la sfreq indicada."""
    rng = np.random.default_rng(semilla)
    n_muestras = int(duracion_s * sfreq)
    data = rng.normal(0, 5e-6, (len(CANALES_EPOC), n_muestras))
    info = mne.create_info(CANALES_EPOC, sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def _raw_export_completo(nombres, cq_por_canal=None, cq_escalado=False,
                         sfreq=TARGET_SFREQ, duracion_s=10.0, semilla=0):
    """
    RawArray con el juego completo de canales de un export de EmotivPro.

    Reproduce la entrada real de `cargar_eeg`: MNE lee el EDF tipando *todos*
    los canales como 'eeg', incluidos telemetría, CQ, EQ y movimiento.

    nombres : list[str]
        Esquema de nombres a reproducir (uno de ESQUEMAS_EXPORT).
    cq_por_canal : dict[str, float] o None
        Valor CQ constante por electrodo (p. ej. {"AF3": 2.0}). Los que no se
        indiquen valen 4.0 (contacto perfecto).
    cq_escalado : bool
        Si True, los canales CQ se generan multiplicados por 1e-6, simulando
        el export cuyo EDF declara la unidad CQ como voltaje y al que MNE
        aplica el factor de escala estándar.
    """
    rng = np.random.default_rng(semilla)
    n_muestras = int(duracion_s * sfreq)
    cq_por_canal = cq_por_canal or {}
    factor_cq = 1e-6 if cq_escalado else 1.0

    filas = []
    for i, nombre in enumerate(nombres):
        if nombre in CANALES_EPOC:
            # Pequeña variación de std entre canales para que el MAD
            # inter-canal de evaluar_calidad sea > 0.
            filas.append(rng.normal(0, 5e-6 * (1 + 0.02 * i), n_muestras))
        elif nombre.startswith(("CQ_", "CQ.")):
            electrodo = nombre.split("_", 1)[-1].split(".", 1)[-1]
            filas.append(
                np.full(n_muestras, cq_por_canal.get(electrodo, 4.0) * factor_cq)
            )
        else:
            filas.append(rng.normal(0, 1.0, n_muestras))

    info = mne.create_info(nombres, sfreq, ch_types="eeg")
    return mne.io.RawArray(np.array(filas), info, verbose=False)


def _mockear_read_raw_edf(monkeypatch, raw_sintetico):
    """Sustituye mne.io.read_raw_edf para que devuelva `raw_sintetico`."""
    monkeypatch.setattr(
        mne.io, "read_raw_edf",
        lambda ruta, preload=True, verbose=False: raw_sintetico,
    )


@pytest.fixture
def archivo_edf(tmp_path):
    """
    Archivo "EDF" de sustitución: solo necesita existir en disco para pasar
    la comprobación `os.path.isfile` de `cargar_eeg`; su contenido no se lee
    porque `mne.io.read_raw_edf` está mockeado.
    """
    ruta = tmp_path / "sujeto_01.edf"
    ruta.write_bytes(b"")
    return ruta


# ---------------------------------------------------------------------------
# Archivo ya a TARGET_SFREQ — no debe tocarse
# ---------------------------------------------------------------------------

def test_archivo_ya_a_target_sfreq_no_se_resamplea(monkeypatch, archivo_edf):
    raw_original = _raw_sintetico(sfreq=TARGET_SFREQ, duracion_s=10.0)
    n_muestras_original = raw_original.n_times
    _mockear_read_raw_edf(monkeypatch, raw_original)

    raw, metadata = cargar_eeg(str(archivo_edf))

    assert raw.info["sfreq"] == TARGET_SFREQ
    assert raw.n_times == n_muestras_original
    assert metadata["resampled"] is False
    assert metadata["original_sfreq"] == TARGET_SFREQ
    assert metadata["sfreq"] == TARGET_SFREQ


# ---------------------------------------------------------------------------
# Archivo a 128 Hz — debe resamplearse a TARGET_SFREQ
# ---------------------------------------------------------------------------

def test_archivo_a_128hz_se_resamplea_a_target_sfreq(monkeypatch, archivo_edf):
    sfreq_original = 128.0
    duracion_s = 10.0
    raw_original = _raw_sintetico(sfreq=sfreq_original, duracion_s=duracion_s)
    duracion_original = raw_original.times[-1]
    _mockear_read_raw_edf(monkeypatch, raw_original)

    raw, metadata = cargar_eeg(str(archivo_edf))

    # sfreq armonizada; solo cambia el nº de muestras, no la duración.
    assert raw.info["sfreq"] == TARGET_SFREQ
    assert raw.times[-1] == pytest.approx(duracion_original, abs=1.0 / sfreq_original)
    assert raw.n_times == pytest.approx(
        duracion_s * TARGET_SFREQ, abs=1
    )

    assert metadata["resampled"] is True
    assert metadata["original_sfreq"] == sfreq_original
    assert metadata["sfreq"] == TARGET_SFREQ


# ---------------------------------------------------------------------------
# Archivo por debajo de MIN_SFREQ — se rechaza, no se resamplea al alza
# ---------------------------------------------------------------------------

def test_archivo_por_debajo_de_min_sfreq_lanza_valueerror(monkeypatch, archivo_edf):
    raw_original = _raw_sintetico(sfreq=MIN_SFREQ - 1, duracion_s=10.0)
    _mockear_read_raw_edf(monkeypatch, raw_original)

    with pytest.raises(ValueError):
        cargar_eeg(str(archivo_edf))


def test_archivo_muy_por_debajo_de_min_sfreq_lanza_valueerror(monkeypatch, archivo_edf):
    raw_original = _raw_sintetico(sfreq=100.0, duracion_s=10.0)
    _mockear_read_raw_edf(monkeypatch, raw_original)

    with pytest.raises(ValueError):
        cargar_eeg(str(archivo_edf))


def test_resampleo_no_escribe_ni_modifica_el_archivo_en_disco(monkeypatch, archivo_edf):
    """La armonización de sfreq es solo en memoria: el .edf fuente no cambia."""
    contenido_antes = archivo_edf.read_bytes()
    raw_original = _raw_sintetico(sfreq=128.0, duracion_s=5.0)
    _mockear_read_raw_edf(monkeypatch, raw_original)

    cargar_eeg(str(archivo_edf))

    assert archivo_edf.read_bytes() == contenido_antes


# ---------------------------------------------------------------------------
# Metadata general y comportamiento no relacionado con el resampleo
# ---------------------------------------------------------------------------

def test_metadata_tiene_las_claves_esperadas(monkeypatch, archivo_edf):
    raw_original = _raw_sintetico(sfreq=128.0, duracion_s=5.0)
    _mockear_read_raw_edf(monkeypatch, raw_original)

    _, metadata = cargar_eeg(str(archivo_edf))

    claves_esperadas = {
        "metodo", "archivo", "sfreq", "original_sfreq", "resampled",
        "duracion_s", "n_canales", "timestamp",
    }
    assert claves_esperadas <= metadata.keys()
    assert metadata["metodo"] == "carga_edf"
    assert metadata["archivo"] == archivo_edf.name
    assert metadata["n_canales"] == len(CANALES_EPOC)


def test_archivo_inexistente_lanza_filenotfounderror():
    with pytest.raises(FileNotFoundError):
        cargar_eeg("ruta/que/no/existe.edf")


# ---------------------------------------------------------------------------
# Tipado de canales: solo los 14 electrodos deben quedar como 'eeg'
#
# Regresión: la versión anterior identificaba los canales no-EEG por una lista
# de nombres con guion bajo tomada del export de 60 canales, así que sobre el
# dataset final (nombres con punto) no retipaba ninguno y los 71 canales
# quedaban como 'eeg'. Con eso, calcular_psd (Fase 8) calculaba el espectro de
# los 71 y extraer_bandas (Fase 9) devolvía 568 columnas de potencia en vez de
# 112, incluyendo features de acelerómetro y batería.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("esquema", list(ESQUEMAS_EXPORT), ids=list(ESQUEMAS_EXPORT))
def test_solo_los_canales_epoc_quedan_tipados_como_eeg(monkeypatch, archivo_edf, esquema):
    nombres = ESQUEMAS_EXPORT[esquema]
    _mockear_read_raw_edf(monkeypatch, _raw_export_completo(nombres))

    raw, _ = cargar_eeg(str(archivo_edf))

    canales_eeg = raw.copy().pick("eeg").ch_names
    assert canales_eeg == CANALES_EPOC

    # El resto sigue presente en el Raw, pero como 'misc': ninguna fase que
    # seleccione picks="eeg" debe tocarlos.
    tipos = dict(zip(raw.ch_names, raw.get_channel_types()))
    no_eeg = [ch for ch in nombres if ch not in CANALES_EPOC]
    assert all(tipos[ch] == "misc" for ch in no_eeg)
    assert len(raw.ch_names) == len(nombres)


@pytest.mark.parametrize("esquema", list(ESQUEMAS_EXPORT), ids=list(ESQUEMAS_EXPORT))
def test_metadata_distingue_canales_totales_de_canales_eeg(monkeypatch, archivo_edf, esquema):
    nombres = ESQUEMAS_EXPORT[esquema]
    _mockear_read_raw_edf(monkeypatch, _raw_export_completo(nombres))

    _, metadata = cargar_eeg(str(archivo_edf))

    assert metadata["n_canales"] == len(nombres)
    assert metadata["n_canales_eeg"] == len(CANALES_EPOC)


# ---------------------------------------------------------------------------
# Escala de los canales CQ
#
# Según cómo declare el EDF la unidad física del canal CQ, MNE le aplica el
# factor de escala de voltaje o no. El rescalado debe decidirse por el rango
# real de los valores: aplicarlo siempre convierte un CQ correcto de 4.0 en
# 4e6 y deja el criterio "CQ medio < 3.0" inoperante para siempre.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("esquema", list(ESQUEMAS_EXPORT), ids=list(ESQUEMAS_EXPORT))
def test_cq_escalado_por_mne_se_revierte_al_rango_original(monkeypatch, archivo_edf, esquema):
    nombres = ESQUEMAS_EXPORT[esquema]
    raw_mock = _raw_export_completo(nombres, cq_escalado=True)
    _mockear_read_raw_edf(monkeypatch, raw_mock)

    raw, _ = cargar_eeg(str(archivo_edf))

    nombre_cq = next(ch for ch in raw.ch_names if ch in (f"CQ_{CANALES_EPOC[0]}", f"CQ.{CANALES_EPOC[0]}"))
    valores_cq = raw.get_data(picks=[nombre_cq])
    assert valores_cq.max() == pytest.approx(4.0, rel=1e-6)


@pytest.mark.parametrize("esquema", list(ESQUEMAS_EXPORT), ids=list(ESQUEMAS_EXPORT))
def test_cq_sin_escalar_no_se_modifica(monkeypatch, archivo_edf, esquema):
    nombres = ESQUEMAS_EXPORT[esquema]
    raw_mock = _raw_export_completo(nombres, cq_escalado=False)
    _mockear_read_raw_edf(monkeypatch, raw_mock)

    raw, _ = cargar_eeg(str(archivo_edf))

    nombre_cq = next(ch for ch in raw.ch_names if ch in (f"CQ_{CANALES_EPOC[0]}", f"CQ.{CANALES_EPOC[0]}"))
    valores_cq = raw.get_data(picks=[nombre_cq])
    assert valores_cq.max() == pytest.approx(4.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Regresión end-to-end del camino CQ: los criterios 1 y 2 de evaluar_calidad
# (Fase 4) deben poder ejecutarse sobre la salida real de cargar_eeg.
#
# Con la versión anterior no se ejecutaban nunca sobre el dataset final: la
# búsqueda del canal CQ era f"CQ_{canal}" y los archivos traen "CQ.{canal}",
# así que el reporte imprimía N/A en esas columnas y solo actuaban los
# criterios de varianza — en silencio, sin error.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("esquema", list(ESQUEMAS_EXPORT), ids=list(ESQUEMAS_EXPORT))
@pytest.mark.parametrize("cq_escalado", [False, True], ids=["cq_sin_escalar", "cq_escalado"])
def test_cq_bajo_excluye_el_canal_tras_cargar(monkeypatch, archivo_edf, esquema, cq_escalado):
    nombres = ESQUEMAS_EXPORT[esquema]
    raw_mock = _raw_export_completo(
        nombres, cq_por_canal={"AF3": 2.0}, cq_escalado=cq_escalado
    )
    _mockear_read_raw_edf(monkeypatch, raw_mock)

    raw, _ = cargar_eeg(str(archivo_edf))
    good, excluidos, metricas = evaluar_calidad(raw)

    assert metricas["AF3"]["cq_medio"] == pytest.approx(2.0, rel=1e-6)
    assert "AF3" in excluidos
    assert "CQ medio bajo" in excluidos["AF3"]
    assert "AF3" not in good
    # Los demás electrodos tienen CQ 4.0 y no deben excluirse por ese motivo.
    assert all("CQ medio bajo" not in motivo
               for canal, motivo in excluidos.items() if canal != "AF3")
