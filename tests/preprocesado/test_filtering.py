"""
Tests para el módulo de filtrado (Fases 2 y 3).
"""

import numpy as np
import pytest
import mne

from eeg_library.preprocesado.filtering import aplicar_filtros, recortar_registro
from eeg_library.preprocesado.quality import CANALES_EPOC

SFREQ = 256
DURACION_S = 20                 # segundos — suficiente para un FIR a 1 Hz
N_MUESTRAS = SFREQ * DURACION_S


def _raw_senoidal(frecuencias, canales=None, amplitud=5e-6, offset=0.0, sfreq=SFREQ):
    """
    Crea un RawArray cuya señal es la suma de senoidales a las frecuencias
    indicadas, replicada idénticamente en todos los canales.
    """
    canales = canales or CANALES_EPOC[:8]
    t = np.arange(N_MUESTRAS) / sfreq
    señal = np.zeros_like(t)
    for f in frecuencias:
        señal += amplitud * np.sin(2 * np.pi * f * t)
    señal += offset

    data = np.tile(señal, (len(canales), 1))
    info = mne.create_info(ch_names=canales, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def _potencia_en(raw, freq, canal_idx=0, ancho=1.0):
    """Devuelve la potencia espectral media en la banda [freq-ancho, freq+ancho] Hz."""
    psd = raw.compute_psd(
        fmin=max(freq - ancho, 0.1),
        fmax=freq + ancho,
        picks=[canal_idx],
        verbose=False,
    )
    return float(psd.get_data().mean())


# ---------------------------------------------------------------------------
# Comportamiento básico
# ---------------------------------------------------------------------------

def test_filtros_devuelve_raw_y_metadata():
    raw = _raw_senoidal([10.0])
    raw_filt, meta = aplicar_filtros(raw)
    assert isinstance(raw_filt, mne.io.BaseRaw)
    assert meta["metodo"] == "notch + bandpass FIR zero-phase"
    assert meta["notch_hz"] == 50.0
    assert meta["banda_hz"] == [1.0, 30.0]


def test_filtros_no_mutan_el_original():
    raw = _raw_senoidal([10.0])
    datos_originales = raw.get_data().copy()
    aplicar_filtros(raw)
    np.testing.assert_array_equal(raw.get_data(), datos_originales)


# ---------------------------------------------------------------------------
# Efecto de cada filtro
# ---------------------------------------------------------------------------

def test_notch_atenua_50hz():
    """Una senoidal a 50 Hz debe quedar fuertemente atenuada tras el notch."""
    raw = _raw_senoidal([50.0])
    pot_antes = _potencia_en(raw, 50.0)
    raw_filt, _ = aplicar_filtros(raw)
    pot_despues = _potencia_en(raw_filt, 50.0)
    assert pot_despues < pot_antes * 1e-3


def test_bandpass_preserva_frecuencias_en_banda():
    """Una senoidal a 10 Hz (dentro de 1–30 Hz) debe sobrevivir al filtrado."""
    raw = _raw_senoidal([10.0])
    pot_antes = _potencia_en(raw, 10.0)
    raw_filt, _ = aplicar_filtros(raw)
    pot_despues = _potencia_en(raw_filt, 10.0)
    # Permitimos hasta 50 % de pérdida por efectos de borde
    assert pot_despues > pot_antes * 0.5


def test_bandpass_atenua_frecuencias_altas():
    """Una senoidal a 45 Hz (fuera del bandpass) debe quedar atenuada."""
    raw = _raw_senoidal([45.0])
    pot_antes = _potencia_en(raw, 45.0)
    raw_filt, _ = aplicar_filtros(raw)
    pot_despues = _potencia_en(raw_filt, 45.0)
    assert pot_despues < pot_antes * 1e-2


def test_bandpass_elimina_dc():
    """Un offset DC debe desaparecer tras el bandpass (media ~0)."""
    raw = _raw_senoidal([10.0], offset=3e-5)
    media_antes = float(np.mean(raw.get_data()))
    raw_filt, _ = aplicar_filtros(raw)
    media_despues = float(np.mean(raw_filt.get_data()))
    assert abs(media_despues) < abs(media_antes) * 1e-3


# ---------------------------------------------------------------------------
# Parámetros personalizados
# ---------------------------------------------------------------------------

def test_banda_personalizada_3_30_hz():
    raw = _raw_senoidal([10.0])
    _, meta = aplicar_filtros(raw, banda=(3.0, 30.0))
    assert meta["banda_hz"] == [3.0, 30.0]


def test_banda_invalida_lanza_error():
    raw = _raw_senoidal([10.0])
    with pytest.raises(ValueError):
        aplicar_filtros(raw, banda=(30.0, 1.0))    # l >= h
    with pytest.raises(ValueError):
        aplicar_filtros(raw, banda=(0.0, 30.0))    # l <= 0


def test_hfreq_por_encima_de_nyquist_lanza_error():
    raw = _raw_senoidal([10.0])
    with pytest.raises(ValueError):
        aplicar_filtros(raw, banda=(1.0, 200.0))   # Nyquist = 128 Hz


# ---------------------------------------------------------------------------
# Canales no-EEG no se filtran
# ---------------------------------------------------------------------------

def test_canales_no_eeg_no_se_modifican():
    """Los canales tipo 'misc' (p. ej. CQ) no deben verse alterados."""
    t = np.arange(N_MUESTRAS) / SFREQ
    eeg = 5e-6 * np.sin(2 * np.pi * 50.0 * t)   # 50 Hz puro
    cq = np.full(N_MUESTRAS, 4.0)               # canal CQ constante a 4

    data = np.vstack([eeg, cq])
    info = mne.create_info(
        ch_names=["AF3", "CQ_AF3"],
        sfreq=SFREQ,
        ch_types=["eeg", "misc"],
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    raw_filt, _ = aplicar_filtros(raw)

    cq_despues = raw_filt.get_data(picks=["CQ_AF3"])[0]
    np.testing.assert_array_equal(cq_despues, cq)


# ---------------------------------------------------------------------------
# Metadatos
# ---------------------------------------------------------------------------

def test_metadata_contiene_campos_esperados():
    raw = _raw_senoidal([10.0])
    _, meta = aplicar_filtros(raw)
    for clave in ("metodo", "notch_hz", "banda_hz",
                  "fir_design", "phase", "timestamp"):
        assert clave in meta


def test_metadata_timestamp_formato_iso():
    import datetime
    raw = _raw_senoidal([10.0])
    _, meta = aplicar_filtros(raw)
    datetime.datetime.fromisoformat(meta["timestamp"])


# ---------------------------------------------------------------------------
# recortar_registro — comportamiento básico
# ---------------------------------------------------------------------------

def test_recortar_devuelve_raw_y_metadata():
    raw = _raw_senoidal([10.0])
    raw_rec, meta = recortar_registro(raw)
    assert isinstance(raw_rec, mne.io.BaseRaw)
    assert meta["metodo"] == "crop"
    assert meta["inicio_s"] == 5.0
    assert meta["fin_s"] == 5.0


def test_recortar_no_muta_el_original():
    raw = _raw_senoidal([10.0])
    n_muestras_original = raw.n_times
    recortar_registro(raw)
    assert raw.n_times == n_muestras_original


def test_recortar_reduce_duracion_en_10_segundos():
    raw = _raw_senoidal([10.0])
    duracion_original = raw.n_times / SFREQ
    raw_rec, meta = recortar_registro(raw, inicio=5.0, fin=5.0)
    duracion_final = raw_rec.n_times / SFREQ
    assert abs(duracion_final - (duracion_original - 10.0)) < 1.0 / SFREQ
    assert abs(meta["duracion_final_s"] - duracion_final) < 1e-6


def test_recortar_parametros_personalizados():
    raw = _raw_senoidal([10.0])
    duracion_original = raw.n_times / SFREQ
    raw_rec, meta = recortar_registro(raw, inicio=2.0, fin=3.0)
    duracion_final = raw_rec.n_times / SFREQ
    assert abs(duracion_final - (duracion_original - 5.0)) < 1.0 / SFREQ
    assert meta["inicio_s"] == 2.0
    assert meta["fin_s"] == 3.0


def test_recortar_sin_margen_no_cambia_duracion():
    raw = _raw_senoidal([10.0])
    raw_rec, _ = recortar_registro(raw, inicio=0.0, fin=0.0)
    assert raw_rec.n_times == raw.n_times


# ---------------------------------------------------------------------------
# recortar_registro — validaciones de entrada
# ---------------------------------------------------------------------------

def test_recortar_margen_mayor_que_duracion_lanza_error():
    raw = _raw_senoidal([10.0])
    with pytest.raises(ValueError):
        recortar_registro(raw, inicio=15.0, fin=10.0)   # 25 s > 20 s


def test_recortar_margen_igual_a_duracion_lanza_error():
    raw = _raw_senoidal([10.0])
    duracion = raw.n_times / SFREQ
    with pytest.raises(ValueError):
        recortar_registro(raw, inicio=duracion / 2, fin=duracion / 2)


def test_recortar_margen_negativo_lanza_error():
    raw = _raw_senoidal([10.0])
    with pytest.raises(ValueError):
        recortar_registro(raw, inicio=-1.0, fin=5.0)
    with pytest.raises(ValueError):
        recortar_registro(raw, inicio=5.0, fin=-1.0)


# ---------------------------------------------------------------------------
# recortar_registro — metadatos
# ---------------------------------------------------------------------------

def test_recortar_metadata_contiene_campos_esperados():
    raw = _raw_senoidal([10.0])
    _, meta = recortar_registro(raw)
    for clave in ("metodo", "inicio_s", "fin_s",
                  "duracion_original_s", "duracion_final_s", "timestamp"):
        assert clave in meta


def test_recortar_metadata_timestamp_formato_iso():
    import datetime
    raw = _raw_senoidal([10.0])
    _, meta = recortar_registro(raw)
    datetime.datetime.fromisoformat(meta["timestamp"])
