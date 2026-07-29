"""
Tests para el módulo de re-referenciación CAR (Fase 5).
"""

import numpy as np
import pytest
import mne

from eeg_library.referencing import aplicar_car
from eeg_library.quality import CANALES_EPOC

SFREQ = 256
N_MUESTRAS = SFREQ * 10


def _raw_con_canales(canales: list[str], amplitud: float = 5e-6) -> mne.io.RawArray:
    """Crea un RawArray con señales aleatorias para los canales indicados."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, amplitud, (len(canales), N_MUESTRAS))
    info = mne.create_info(ch_names=canales, sfreq=SFREQ, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


# ---------------------------------------------------------------------------
# CAR con >= 8 canales
# ---------------------------------------------------------------------------

def test_car_aplicada_con_8_o_mas_canales():
    good = CANALES_EPOC[:8]
    raw = _raw_con_canales(CANALES_EPOC[:8])
    raw_ref, meta = aplicar_car(raw, good)
    assert meta["metodo"] == "CAR"
    assert meta["canales_incluidos"] == good


def test_media_good_channels_es_cero_tras_car():
    """Tras CAR, el promedio de los good_channels debe ser ~0 en cada muestra."""
    good = CANALES_EPOC[:10]
    raw = _raw_con_canales(good)
    raw_ref, _ = aplicar_car(raw, good)

    indices = [raw_ref.ch_names.index(ch) for ch in good]
    promedio_post = np.mean(raw_ref._data[indices], axis=0)
    np.testing.assert_allclose(promedio_post, 0, atol=1e-20)


def test_car_modifica_copia_no_el_original():
    good = CANALES_EPOC[:8]
    raw = _raw_con_canales(good)
    datos_originales = raw.get_data().copy()
    raw_ref, _ = aplicar_car(raw, good)
    # El original no debe haber cambiado
    np.testing.assert_array_equal(raw.get_data(), datos_originales)
    # La copia sí debe ser diferente
    assert not np.allclose(raw_ref.get_data(), datos_originales)


# ---------------------------------------------------------------------------
# CAR con 6–7 canales: se aplica con advertencia
# ---------------------------------------------------------------------------

def test_car_aplicada_con_6_canales(capsys):
    good = CANALES_EPOC[:6]
    raw = _raw_con_canales(CANALES_EPOC[:6])
    raw_ref, meta = aplicar_car(raw, good)
    assert meta["metodo"] == "CAR"
    assert "ADVERTENCIA" in capsys.readouterr().out


def test_car_aplicada_con_7_canales(capsys):
    good = CANALES_EPOC[:7]
    raw = _raw_con_canales(CANALES_EPOC[:7])
    raw_ref, meta = aplicar_car(raw, good)
    assert meta["metodo"] == "CAR"
    assert "ADVERTENCIA" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# CAR omitida con < 6 canales
# ---------------------------------------------------------------------------

def test_car_omitida_con_menos_de_6_canales(capsys):
    good = CANALES_EPOC[:5]
    raw = _raw_con_canales(CANALES_EPOC[:5])
    datos_antes = raw.get_data().copy()
    raw_ref, meta = aplicar_car(raw, good)
    assert meta["metodo"] == "ninguno"
    # Los datos no deben haberse modificado
    np.testing.assert_array_equal(raw_ref.get_data(), datos_antes)
    assert "omitida" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# Canales excluidos no se modifican
# ---------------------------------------------------------------------------

def test_canales_excluidos_no_se_modifican():
    """Los canales que no están en good_channels no deben cambiar."""
    todos = CANALES_EPOC[:10]
    good = CANALES_EPOC[:8]       # primeros 8 son buenos
    excluidos = CANALES_EPOC[8:10]  # los 2 últimos son malos

    raw = _raw_con_canales(todos)
    datos_excluidos_antes = raw.get_data(picks=excluidos).copy()

    raw_ref, _ = aplicar_car(raw, good)

    datos_excluidos_despues = raw_ref.get_data(picks=excluidos)
    np.testing.assert_array_equal(datos_excluidos_despues, datos_excluidos_antes)


# ---------------------------------------------------------------------------
# Metadatos devueltos
# ---------------------------------------------------------------------------

def test_metadata_contiene_campos_esperados():
    good = CANALES_EPOC[:8]
    raw = _raw_con_canales(good)
    _, meta = aplicar_car(raw, good)
    assert "metodo" in meta
    assert "canales_incluidos" in meta
    assert "canales_excluidos" in meta
    assert "timestamp" in meta


def test_metadata_canales_excluidos_correctos():
    todos = CANALES_EPOC[:10]
    good = CANALES_EPOC[:8]
    excluidos_esperados = sorted(CANALES_EPOC[8:10])

    raw = _raw_con_canales(todos)
    _, meta = aplicar_car(raw, good)
    assert sorted(meta["canales_excluidos"]) == excluidos_esperados


def test_metadata_timestamp_formato_iso():
    import datetime
    good = CANALES_EPOC[:8]
    raw = _raw_con_canales(good)
    _, meta = aplicar_car(raw, good)
    # Debe poder parsearse como ISO 8601 sin excepciones
    datetime.datetime.fromisoformat(meta["timestamp"])
