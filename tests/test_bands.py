"""
Tests para el módulo de extracción de potencia por banda (Fase 9).
"""

import numpy as np
import pytest
import mne
import pandas as pd

from eeg_library.spectral import calcular_psd
from eeg_library.bands import (
    extraer_bandas,
    BANDAS_DEFECTO,
    CLUSTER_FRONTAL,
    CLUSTER_POSTERIOR,
)

SFREQ = 256
N_MUESTRAS_EPOCA = 1024  # 4 s a 256 Hz, como produce crear_epocas (Fase 7)

# Los 14 canales del Emotiv Epoc X, incluyendo ambos clústeres de interés.
CANALES_14 = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]


def _spectrum_sintetico(n_epocas=4, frecuencias=(), canal_senal=None,
                         canales=None, amplitud=3e-5, semilla=0):
    """
    Crea un EpochsSpectrum sintético (vía calcular_psd) a partir de ruido
    blanco, opcionalmente con una senoidal inyectada en un único canal
    (para verificar que la potencia aparece en la banda correcta).
    """
    canales = canales or CANALES_14
    rng = np.random.default_rng(semilla)
    t = np.arange(N_MUESTRAS_EPOCA) / SFREQ

    data = rng.normal(0, 1e-7, (n_epocas, len(canales), N_MUESTRAS_EPOCA))
    if frecuencias:
        idx_canal = canales.index(canal_senal) if canal_senal else None
        for f in frecuencias:
            señal = amplitud * np.sin(2 * np.pi * f * t)
            if idx_canal is None:
                data += señal
            else:
                data[:, idx_canal, :] += señal

    info = mne.create_info(canales, SFREQ, ch_types="eeg")
    epochs = mne.EpochsArray(data, info, verbose=False)
    epochs.metadata = pd.DataFrame(
        {
            "sujeto": ["S07"] * n_epocas,
            "condicion": ["C3"] * n_epocas,
            "sesion": [4] * n_epocas,
            "archivo_origen": ["M4_XX_F007_C3.edf"] * n_epocas,
        }
    )
    spectrum, _ = calcular_psd(epochs)
    return spectrum


# ---------------------------------------------------------------------------
# Bandas: sin gamma, columnas por canal x banda x representación
# ---------------------------------------------------------------------------

def test_no_calcula_banda_gamma():
    assert "gamma" not in BANDAS_DEFECTO
    spectrum = _spectrum_sintetico()
    features, metadata = extraer_bandas(spectrum)
    assert "gamma" not in metadata["bandas_hz"]
    assert not any("gamma" in col for col in features.columns)


def test_forma_de_columnas_por_canal_y_banda():
    spectrum = _spectrum_sintetico()
    features, metadata = extraer_bandas(spectrum)
    for canal in CANALES_14:
        for banda in BANDAS_DEFECTO:
            assert f"pot_abs_{canal}_{banda}" in features.columns
            assert f"pot_rel_{canal}_{banda}" in features.columns
    assert metadata["n_features_banda"] == len(CANALES_14) * len(BANDAS_DEFECTO) * 2
    assert metadata["n_features_banda"] == 112


def test_una_fila_por_epoca():
    spectrum = _spectrum_sintetico(n_epocas=6)
    features, metadata = extraer_bandas(spectrum)
    assert len(features) == 6
    assert metadata["n_epocas"] == 6


# ---------------------------------------------------------------------------
# Coherencia de la potencia con el contenido espectral de entrada
# ---------------------------------------------------------------------------

def test_potencia_se_localiza_en_la_banda_correcta():
    # Senoidal de 10 Hz (banda alfa) inyectada solo en el canal P7.
    spectrum = _spectrum_sintetico(frecuencias=[10.0], canal_senal="P7")
    features, _ = extraer_bandas(spectrum)
    pot_alfa = features["pot_abs_P7_alfa"]
    pot_delta = features["pot_abs_P7_delta"]
    pot_beta = features["pot_abs_P7_beta"]
    assert (pot_alfa > pot_delta * 10).all()
    assert (pot_alfa > pot_beta * 10).all()
    # Un canal sin la senoidal no debe mostrar el mismo dominio de alfa.
    assert (features["pot_abs_F3_alfa"] < pot_alfa).all()


def test_potencia_relativa_suma_uno_por_canal():
    spectrum = _spectrum_sintetico(frecuencias=[10.0], canal_senal="P7")
    features, _ = extraer_bandas(spectrum)
    for canal in CANALES_14:
        suma = sum(features[f"pot_rel_{canal}_{banda}"] for banda in BANDAS_DEFECTO)
        assert suma.values == pytest.approx(1.0, abs=1e-6)


def test_bandas_particionan_la_potencia_total_sin_solape():
    # La suma de la potencia absoluta de las 4 bandas debe reproducir
    # exactamente la potencia total del espectro (sin huecos ni doble conteo
    # en los límites 4, 8 y 13 Hz).
    spectrum = _spectrum_sintetico()
    features, _ = extraer_bandas(spectrum)
    datos = spectrum.get_data()
    resolucion = float(spectrum.freqs[1] - spectrum.freqs[0])
    potencia_total = datos.sum(axis=2) * resolucion  # (n_epocas, n_canales)
    for i, canal in enumerate(CANALES_14):
        suma_bandas = sum(features[f"pot_abs_{canal}_{banda}"] for banda in BANDAS_DEFECTO)
        np.testing.assert_allclose(suma_bandas.values, potencia_total[:, i], rtol=1e-10)


# ---------------------------------------------------------------------------
# Clústeres frontal/posterior e índice de fatiga theta/alfa
# ---------------------------------------------------------------------------

def test_composicion_cluster_frontal():
    assert CLUSTER_FRONTAL == ["AF3", "AF4", "F3", "F4", "F7", "F8"]


def test_composicion_cluster_posterior():
    assert CLUSTER_POSTERIOR == ["P7", "P8", "O1", "O2"]


def test_theta_frontal_es_la_media_del_cluster():
    spectrum = _spectrum_sintetico()
    features, metadata = extraer_bandas(spectrum)
    esperado = sum(features[f"pot_abs_{c}_theta"] for c in CLUSTER_FRONTAL) / len(CLUSTER_FRONTAL)
    np.testing.assert_allclose(features["theta_frontal"].values, esperado.values, rtol=1e-10)
    assert metadata["cluster_frontal"] == CLUSTER_FRONTAL


def test_alfa_posterior_es_la_media_del_cluster():
    spectrum = _spectrum_sintetico()
    features, metadata = extraer_bandas(spectrum)
    esperado = sum(features[f"pot_abs_{c}_alfa"] for c in CLUSTER_POSTERIOR) / len(CLUSTER_POSTERIOR)
    np.testing.assert_allclose(features["alfa_posterior"].values, esperado.values, rtol=1e-10)
    assert metadata["cluster_posterior"] == CLUSTER_POSTERIOR


def test_indice_de_fatiga_es_el_cociente_theta_alfa():
    spectrum = _spectrum_sintetico()
    features, _ = extraer_bandas(spectrum)
    esperado = features["theta_frontal"] / features["alfa_posterior"]
    np.testing.assert_allclose(
        features["indice_fatiga_theta_alfa"].values, esperado.values, rtol=1e-10
    )


def test_cluster_frontal_o_posterior_ausente_lanza_error():
    # Espectro sin ningún canal del clúster posterior (solo frontales).
    spectrum = _spectrum_sintetico(canales=["AF3", "AF4", "F3", "F4"])
    with pytest.raises(ValueError):
        extraer_bandas(spectrum)


def test_bandas_sin_theta_o_alfa_lanza_error():
    spectrum = _spectrum_sintetico()
    with pytest.raises(ValueError):
        extraer_bandas(spectrum, bandas={"delta": (1.0, 4.0), "beta": (13.0, 30.0)})


# ---------------------------------------------------------------------------
# canales_validos: los canales excluidos por evaluar_calidad (Fase 4) no
# deben entrar en los promedios de clúster del índice de fatiga.
#
# Un canal excluido no pasó por CAR (Fase 5) ni por ICA (Fase 6) —ambas operan
# solo sobre los canales buenos— y suele haberse excluido por amplitud
# excesiva, así que su potencia de banda domina la media del clúster. Medido
# sobre 147 archivos reales: el 64 % tiene algún canal de clúster excluido y
# en ellos el índice se infla por un factor mediano de 1.75 (máximo 47.6).
# ---------------------------------------------------------------------------

def test_canales_validos_none_mantiene_el_comportamiento_anterior():
    spectrum = _spectrum_sintetico()
    features_sin, meta_sin = extraer_bandas(spectrum)
    features_con, meta_con = extraer_bandas(spectrum, canales_validos=CANALES_14)

    np.testing.assert_allclose(
        features_sin["indice_fatiga_theta_alfa"].values,
        features_con["indice_fatiga_theta_alfa"].values,
        rtol=1e-12,
    )
    assert meta_sin["cluster_frontal"] == meta_con["cluster_frontal"]
    assert meta_sin["cluster_frontal_descartados"] == []
    assert meta_sin["cluster_posterior_descartados"] == []


def test_canales_validos_excluye_del_cluster_los_canales_malos():
    # AF3 lleva una senoidal de 6 Hz (banda theta) de gran amplitud, como un
    # canal de mal contacto excluido por amplitud excesiva.
    spectrum = _spectrum_sintetico(frecuencias=(6.0,), canal_senal="AF3", amplitud=1e-3)
    buenos = [c for c in CANALES_14 if c != "AF3"]

    features_todos, _ = extraer_bandas(spectrum)
    features_buenos, metadata = extraer_bandas(spectrum, canales_validos=buenos)

    frontal_bueno = [c for c in CLUSTER_FRONTAL if c != "AF3"]
    assert metadata["cluster_frontal"] == frontal_bueno
    assert metadata["cluster_frontal_descartados"] == ["AF3"]
    assert metadata["cluster_posterior"] == CLUSTER_POSTERIOR
    assert metadata["cluster_posterior_descartados"] == []

    esperado = sum(features_buenos[f"pot_abs_{c}_theta"] for c in frontal_bueno) / len(frontal_bueno)
    np.testing.assert_allclose(
        features_buenos["theta_frontal"].values, esperado.values, rtol=1e-10
    )

    # El canal contaminado inflaba la media del clúster y con ella el índice.
    assert (features_todos["theta_frontal"] > features_buenos["theta_frontal"]).all()
    assert (
        features_todos["indice_fatiga_theta_alfa"]
        > features_buenos["indice_fatiga_theta_alfa"]
    ).all()


def test_canales_validos_no_altera_las_columnas_de_potencia_por_canal():
    """
    La matriz de features debe conservar las mismas columnas y el mismo orden
    aunque cambien los canales válidos: el procesamiento por lotes (Fase 11)
    concatena los DataFrames de todos los archivos y cada uno tiene su propio
    conjunto de canales excluidos.
    """
    spectrum = _spectrum_sintetico()
    features_todos, meta_todos = extraer_bandas(spectrum)
    features_buenos, meta_buenos = extraer_bandas(
        spectrum, canales_validos=[c for c in CANALES_14 if c not in ("AF3", "P8")]
    )

    assert list(features_todos.columns) == list(features_buenos.columns)
    assert meta_todos["n_features_banda"] == meta_buenos["n_features_banda"]
    for canal in CANALES_14:
        for banda in BANDAS_DEFECTO:
            np.testing.assert_allclose(
                features_todos[f"pot_abs_{canal}_{banda}"].values,
                features_buenos[f"pot_abs_{canal}_{banda}"].values,
                rtol=1e-12,
            )


def test_cluster_entero_excluido_por_calidad_lanza_error():
    spectrum = _spectrum_sintetico()
    buenos = [c for c in CANALES_14 if c not in CLUSTER_FRONTAL]

    with pytest.raises(ValueError, match="excluidos por calidad"):
        extraer_bandas(spectrum, canales_validos=buenos)


def test_error_distingue_cluster_ausente_de_cluster_excluido():
    """
    Los dos motivos piden acciones distintas: un clúster ausente del espectro
    apunta a un montaje o unos picks mal puestos, mientras que uno excluido
    entero por calidad es un problema de esa grabación concreta.
    """
    spectrum_sin_posterior = _spectrum_sintetico(canales=["AF3", "AF4", "F3", "F4"])
    with pytest.raises(ValueError, match="esta presente"):
        extraer_bandas(spectrum_sin_posterior)

    spectrum = _spectrum_sintetico()
    with pytest.raises(ValueError, match="excluidos por calidad"):
        extraer_bandas(
            spectrum, canales_validos=[c for c in CANALES_14 if c not in CLUSTER_POSTERIOR]
        )


# ---------------------------------------------------------------------------
# Metadata de origen (sujeto, condición, sesión) preservada desde la Fase 7-8
# ---------------------------------------------------------------------------

def test_conserva_metadata_de_las_epocas():
    spectrum = _spectrum_sintetico(n_epocas=5)
    features, _ = extraer_bandas(spectrum)
    assert len(features) == 5
    assert (features["sujeto"] == "S07").all()
    assert (features["condicion"] == "C3").all()
    assert (features["sesion"] == 4).all()
    assert (features["archivo_origen"] == "M4_XX_F007_C3.edf").all()


# ---------------------------------------------------------------------------
# No mutación del espectro de entrada
# ---------------------------------------------------------------------------

def test_no_muta_el_spectrum_original():
    spectrum = _spectrum_sintetico()
    datos_antes = spectrum.get_data().copy()
    freqs_antes = spectrum.freqs.copy()
    extraer_bandas(spectrum)
    np.testing.assert_array_equal(spectrum.get_data(), datos_antes)
    np.testing.assert_array_equal(spectrum.freqs, freqs_antes)


# ---------------------------------------------------------------------------
# Metadata general devuelta
# ---------------------------------------------------------------------------

def test_metadata_contiene_campos_esperados():
    spectrum = _spectrum_sintetico()
    _, metadata = extraer_bandas(spectrum)
    for clave in (
        "metodo", "bandas_hz", "cluster_frontal", "cluster_posterior",
        "cluster_frontal_descartados", "cluster_posterior_descartados",
        "canales", "resolucion_hz", "n_epocas", "n_canales",
        "n_features_banda", "timestamp",
    ):
        assert clave in metadata


def test_metadata_timestamp_formato_iso():
    import datetime
    spectrum = _spectrum_sintetico()
    _, metadata = extraer_bandas(spectrum)
    datetime.datetime.fromisoformat(metadata["timestamp"])
