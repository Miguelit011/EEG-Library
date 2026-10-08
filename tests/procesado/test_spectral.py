"""
Tests para el módulo de análisis espectral (Fase 8).
"""

import numpy as np
import pytest
import mne
import pandas as pd

from eeg_library.procesado.spectral import calcular_psd

SFREQ = 256
CANALES = ["AF3", "F7", "F3", "FC5"]
N_MUESTRAS_EPOCA = 1024  # 4 s a 256 Hz, como produce crear_epocas (Fase 7)


def _epocas_sinteticas(n_epocas=3, frecuencias=(), canales=None, semilla=0,
                        n_muestras=N_MUESTRAS_EPOCA, con_metadata=True):
    """
    Crea un mne.EpochsArray sintético, opcionalmente con una suma de
    senoidales (para verificar la localización en frecuencia de la PSD).
    """
    canales = canales or CANALES
    rng = np.random.default_rng(semilla)
    t = np.arange(n_muestras) / SFREQ

    data = rng.normal(0, 1e-7, (n_epocas, len(canales), n_muestras))
    for f in frecuencias:
        señal = 3e-5 * np.sin(2 * np.pi * f * t)
        data += señal  # misma senoidal en todos los canales/épocas

    info = mne.create_info(canales, SFREQ, ch_types="eeg")
    epochs = mne.EpochsArray(data, info, verbose=False)
    if con_metadata:
        epochs.metadata = pd.DataFrame(
            {
                "sujeto": ["S07"] * n_epocas,
                "condicion": ["C3"] * n_epocas,
                "sesion": [4] * n_epocas,
                "archivo_origen": ["M4_XX_F007_C3.edf"] * n_epocas,
            }
        )
    return epochs


# ---------------------------------------------------------------------------
# Resolución en frecuencia y sub-segmentación de Welch
# ---------------------------------------------------------------------------

def test_resolucion_en_frecuencia_de_1hz():
    epochs = _epocas_sinteticas()
    spectrum, metadata = calcular_psd(epochs)
    assert metadata["resolucion_hz"] == pytest.approx(1.0)
    assert np.allclose(np.diff(spectrum.freqs), 1.0)


def test_ventana_welch_256_muestras_con_128_de_solape():
    epochs = _epocas_sinteticas()
    _, metadata = calcular_psd(epochs)
    assert metadata["n_per_seg"] == 256
    assert metadata["n_overlap"] == 128
    assert metadata["ventana"] == "hann"


def test_numero_de_segmentos_promediados():
    # Época de 1024 muestras, ventana 256, paso 128 (256-128):
    # inicios en 0,128,256,384,512,640,768 (768+256=1024, cabe exacto) = 7.
    epochs = _epocas_sinteticas()
    _, metadata = calcular_psd(epochs)
    assert metadata["n_segmentos"] == 7


def test_pico_espectral_localizado_en_la_frecuencia_correcta():
    epochs = _epocas_sinteticas(frecuencias=[10.0])
    spectrum, _ = calcular_psd(epochs)
    datos = spectrum.get_data()  # (n_epocas, n_canales, n_frecuencias)
    pico = spectrum.freqs[np.argmax(datos[0, 0])]
    assert pico == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Rango de frecuencias: limitado a 1-30 Hz, sin contenido por encima
# ---------------------------------------------------------------------------

def test_no_devuelve_contenido_por_encima_de_30hz():
    epochs = _epocas_sinteticas()
    spectrum, metadata = calcular_psd(epochs)
    assert spectrum.freqs.max() <= 30.0
    assert spectrum.freqs.min() >= 1.0
    assert metadata["banda_hz"] == [1.0, 30.0]


def test_banda_configurable():
    epochs = _epocas_sinteticas()
    spectrum, metadata = calcular_psd(epochs, banda=(4.0, 13.0))
    assert spectrum.freqs.min() >= 4.0
    assert spectrum.freqs.max() <= 13.0
    assert metadata["banda_hz"] == [4.0, 13.0]


# ---------------------------------------------------------------------------
# Granularidad por canal y por época (sin promediar entre canales)
# ---------------------------------------------------------------------------

def test_forma_de_salida_por_canal_y_por_epoca():
    epochs = _epocas_sinteticas(n_epocas=5)
    spectrum, metadata = calcular_psd(epochs)
    datos = spectrum.get_data()
    assert datos.shape[0] == 5
    assert datos.shape[1] == len(CANALES)
    assert metadata["n_epocas"] == 5
    assert metadata["n_canales"] == len(CANALES)
    assert list(spectrum.ch_names) == CANALES


# ---------------------------------------------------------------------------
# Metadata de origen (sujeto, condición, sesión) preservada desde la Fase 7
# ---------------------------------------------------------------------------

def test_conserva_metadata_de_las_epocas():
    epochs = _epocas_sinteticas(n_epocas=4)
    spectrum, _ = calcular_psd(epochs)
    meta = spectrum.metadata
    assert len(meta) == 4
    assert (meta["sujeto"] == "S07").all()
    assert (meta["condicion"] == "C3").all()
    assert (meta["sesion"] == 4).all()
    assert (meta["archivo_origen"] == "M4_XX_F007_C3.edf").all()


# ---------------------------------------------------------------------------
# No mutación de las épocas de entrada
# ---------------------------------------------------------------------------

def test_no_muta_las_epocas_originales():
    epochs = _epocas_sinteticas()
    datos_antes = epochs.get_data().copy()
    calcular_psd(epochs)
    np.testing.assert_array_equal(epochs.get_data(), datos_antes)


# ---------------------------------------------------------------------------
# Validaciones de parámetros
# ---------------------------------------------------------------------------

def test_banda_invalida_lanza_error():
    epochs = _epocas_sinteticas()
    with pytest.raises(ValueError):
        calcular_psd(epochs, banda=(30.0, 1.0))


def test_fmax_igual_o_mayor_que_nyquist_lanza_error():
    epochs = _epocas_sinteticas()
    with pytest.raises(ValueError):
        calcular_psd(epochs, banda=(1.0, 128.0))


def test_duracion_ventana_no_positiva_lanza_error():
    epochs = _epocas_sinteticas()
    with pytest.raises(ValueError):
        calcular_psd(epochs, duracion_ventana=0.0)


def test_solape_ventana_igual_o_mayor_que_duracion_lanza_error():
    epochs = _epocas_sinteticas()
    with pytest.raises(ValueError):
        calcular_psd(epochs, duracion_ventana=1.0, solape_ventana=1.0)


def test_ventana_de_welch_mas_larga_que_la_epoca_lanza_error():
    epochs = _epocas_sinteticas(n_muestras=128)  # época de solo 0.5 s
    with pytest.raises(ValueError):
        calcular_psd(epochs, duracion_ventana=1.0, solape_ventana=0.5)


# ---------------------------------------------------------------------------
# Metadata general devuelta
# ---------------------------------------------------------------------------

def test_metadata_contiene_campos_esperados():
    epochs = _epocas_sinteticas()
    _, metadata = calcular_psd(epochs)
    for clave in (
        "metodo", "banda_hz", "duracion_ventana_s", "solape_ventana_s",
        "paso_ventana_s", "n_per_seg", "n_overlap", "n_segmentos",
        "ventana", "resolucion_hz", "n_epocas", "n_canales", "sfreq",
        "timestamp",
    ):
        assert clave in metadata


def test_metadata_timestamp_formato_iso():
    import datetime
    epochs = _epocas_sinteticas()
    _, metadata = calcular_psd(epochs)
    datetime.datetime.fromisoformat(metadata["timestamp"])
