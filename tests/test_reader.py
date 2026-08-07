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
from eeg_library.quality import CANALES_EPOC


def _raw_sintetico(sfreq, duracion_s=10.0, semilla=0):
    """RawArray de 14 canales EPOC con ruido gaussiano a la sfreq indicada."""
    rng = np.random.default_rng(semilla)
    n_muestras = int(duracion_s * sfreq)
    data = rng.normal(0, 5e-6, (len(CANALES_EPOC), n_muestras))
    info = mne.create_info(CANALES_EPOC, sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


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
