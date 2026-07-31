"""
Tests para el módulo de epocado (Fase 7).
"""

import numpy as np
import pytest
import mne

from eeg_library.epoching import crear_epocas

SFREQ = 256
CANALES = ["AF3", "F7", "F3", "FC5"]


def _raw_de_duracion(segundos, sfreq=SFREQ, canales=None):
    """Crea un RawArray sintético de la duración indicada (en segundos)."""
    canales = canales or CANALES
    rng = np.random.default_rng(0)
    n_muestras = int(round(segundos * sfreq))
    data = rng.normal(0, 5e-6, (len(canales), n_muestras))
    info = mne.create_info(canales, sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


# ---------------------------------------------------------------------------
# Segmentación básica: duración, solape y número de épocas
# ---------------------------------------------------------------------------

def test_numero_de_epocas_con_solape_50_por_ciento():
    # 12 s de señal, ventanas de 4 s con paso de 2 s (solape 2 s) ->
    # inicios en 0, 2, 4, 6, 8 s (8 + 4 = 12, cabe exacto) = 5 épocas.
    raw = _raw_de_duracion(12.0)
    epochs, metadata = crear_epocas(
        raw, sujeto="S01", condicion="C1", sesion=1, archivo_origen="a.edf",
        duracion=4.0, solape=2.0,
    )
    assert len(epochs) == 5
    assert metadata["n_epocas"] == 5
    assert metadata["paso_s"] == pytest.approx(2.0)
    assert metadata["duracion_descartada_s"] == pytest.approx(0.0, abs=1e-6)


def test_cada_epoca_tiene_1024_muestras_a_256hz():
    raw = _raw_de_duracion(10.0)
    epochs, _ = crear_epocas(
        raw, sujeto="S01", condicion="C1", sesion=1, archivo_origen="a.edf",
        duracion=4.0, solape=2.0,
    )
    datos = epochs.get_data()
    assert datos.shape[1] == len(CANALES)
    assert datos.shape[2] == 1024


def test_resto_final_incompleto_se_descarta():
    # 9 s de señal, ventanas de 4 s, sin solape -> inicios en 0, 4 s (2 épocas),
    # sobran 1 s que no completan una ventana entera.
    raw = _raw_de_duracion(9.0)
    epochs, metadata = crear_epocas(
        raw, sujeto="S01", condicion="C1", sesion=1, archivo_origen="a.edf",
        duracion=4.0, solape=0.0,
    )
    assert len(epochs) == 2
    assert metadata["duracion_descartada_s"] == pytest.approx(1.0, abs=1e-6)


def test_registro_mas_corto_que_una_ventana_lanza_error():
    raw = _raw_de_duracion(2.0)
    with pytest.raises(ValueError):
        crear_epocas(
            raw, sujeto="S01", condicion="C1", sesion=1, archivo_origen="a.edf",
            duracion=4.0, solape=2.0,
        )


# ---------------------------------------------------------------------------
# Validaciones de parámetros
# ---------------------------------------------------------------------------

def test_duracion_no_positiva_lanza_error():
    raw = _raw_de_duracion(10.0)
    with pytest.raises(ValueError):
        crear_epocas(
            raw, sujeto="S01", condicion="C1", sesion=1, archivo_origen="a.edf",
            duracion=0.0, solape=0.0,
        )


def test_solape_igual_o_mayor_que_duracion_lanza_error():
    raw = _raw_de_duracion(10.0)
    with pytest.raises(ValueError):
        crear_epocas(
            raw, sujeto="S01", condicion="C1", sesion=1, archivo_origen="a.edf",
            duracion=4.0, solape=4.0,
        )


def test_solape_negativo_lanza_error():
    raw = _raw_de_duracion(10.0)
    with pytest.raises(ValueError):
        crear_epocas(
            raw, sujeto="S01", condicion="C1", sesion=1, archivo_origen="a.edf",
            duracion=4.0, solape=-1.0,
        )


# ---------------------------------------------------------------------------
# Metadata de origen (sujeto, condición, sesión, archivo)
# ---------------------------------------------------------------------------

def test_metadata_por_epoca_contiene_origen():
    raw = _raw_de_duracion(12.0)
    epochs, _ = crear_epocas(
        raw, sujeto="S07", condicion="C3", sesion=4, archivo_origen="M4_XX_F007_C3.edf",
        duracion=4.0, solape=2.0,
    )
    meta = epochs.metadata
    assert len(meta) == len(epochs)
    assert (meta["sujeto"] == "S07").all()
    assert (meta["condicion"] == "C3").all()
    assert (meta["sesion"] == 4).all()
    assert (meta["archivo_origen"] == "M4_XX_F007_C3.edf").all()
    assert list(meta["indice_ventana"]) == list(range(len(epochs)))


def test_tmin_s_refleja_el_paso_entre_ventanas():
    raw = _raw_de_duracion(12.0)
    epochs, metadata = crear_epocas(
        raw, sujeto="S01", condicion="C1", sesion=1, archivo_origen="a.edf",
        duracion=4.0, solape=2.0,
    )
    esperado = [i * metadata["paso_s"] for i in range(len(epochs))]
    assert list(epochs.metadata["tmin_s"]) == pytest.approx(esperado)


# ---------------------------------------------------------------------------
# No mutación del Raw original
# ---------------------------------------------------------------------------

def test_no_muta_el_raw_original():
    raw = _raw_de_duracion(12.0)
    datos_antes = raw.get_data().copy()
    crear_epocas(
        raw, sujeto="S01", condicion="C1", sesion=1, archivo_origen="a.edf",
        duracion=4.0, solape=2.0,
    )
    np.testing.assert_array_equal(raw.get_data(), datos_antes)


# ---------------------------------------------------------------------------
# Metadata general devuelta
# ---------------------------------------------------------------------------

def test_metadata_contiene_campos_esperados():
    raw = _raw_de_duracion(12.0)
    _, metadata = crear_epocas(
        raw, sujeto="S01", condicion="C1", sesion=1, archivo_origen="a.edf",
        duracion=4.0, solape=2.0,
    )
    for clave in (
        "metodo", "duracion_s", "solape_s", "paso_s", "sfreq",
        "n_epocas", "duracion_descartada_s", "timestamp",
    ):
        assert clave in metadata


def test_metadata_timestamp_formato_iso():
    import datetime
    raw = _raw_de_duracion(12.0)
    _, metadata = crear_epocas(
        raw, sujeto="S01", condicion="C1", sesion=1, archivo_origen="a.edf",
        duracion=4.0, solape=2.0,
    )
    datetime.datetime.fromisoformat(metadata["timestamp"])
