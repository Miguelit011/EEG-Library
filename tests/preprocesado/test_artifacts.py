"""
Tests para el módulo de eliminación de artefactos oculares (Fase 6).

ICLabel es una red entrenada sobre EEG real; comprobado manualmente que ni
una inyección deliberada de pulsos frontales lentos en datos sintéticos de
RawArray logra una clasificación fiable de 'eye blink'. Por eso estos tests
mockean `iclabel_label_components` con matrices de probabilidad controladas
y así prueban solo la lógica de selección/diagnóstico de
identificar_componentes_oculares y eliminar_artefactos_oculares — no la
precisión del clasificador. El test de verificación de rango sí ejecuta
ICA real, ya que no depende de ICLabel.
"""

import numpy as np
import pytest
import mne

from eeg_library.preprocesado import artifacts
from eeg_library.preprocesado.artifacts import (
    entrenar_ica,
    identificar_componentes_oculares,
    eliminar_artefactos_oculares,
    CATEGORIAS_ICLABEL,
)
from eeg_library.preprocesado.reader import MONTAJE_1020

SFREQ = 256
DURACION_S = 15
N_MUESTRAS = SFREQ * DURACION_S
CANALES = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]


def _raw_sintetico(canales=None, semilla=0, sfreq=SFREQ, n_muestras=N_MUESTRAS):
    """Raw sintético ya recortado/filtrado/CAR (Fases 2-5) simulado a mano."""
    canales = canales or CANALES
    rng = np.random.default_rng(semilla)
    data = rng.normal(0, 3e-6, (len(canales), n_muestras))
    info = mne.create_info(canales, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage(MONTAJE_1020, on_missing="warn", verbose=False)
    raw.filter(l_freq=1.0, h_freq=30.0, verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    return raw


def _mockear_iclabel(monkeypatch, proba_matrix):
    """Sustituye iclabel_label_components para devolver una matriz fija."""
    monkeypatch.setattr(
        artifacts, "iclabel_label_components", lambda raw, ica: proba_matrix
    )


def _matriz_base(n_componentes):
    """Matriz donde todas las componentes son 'brain' con probabilidad alta."""
    proba = np.zeros((n_componentes, len(CATEGORIAS_ICLABEL)))
    proba[:, CATEGORIAS_ICLABEL.index("brain")] = 0.9
    return proba


@pytest.fixture(scope="module")
def raw_y_ica():
    """Raw sintético + ICA real ajustada, compartidos entre tests que mockean ICLabel."""
    raw = _raw_sintetico()
    ica, _ = entrenar_ica(raw, CANALES, random_state=42)
    return raw, ica


# ---------------------------------------------------------------------------
# Caso 1 — artefacto ocular simulado reconocible: se detecta y se elimina
# ---------------------------------------------------------------------------

def test_componente_ocular_detectada_y_eliminada(raw_y_ica, monkeypatch):
    raw, ica = raw_y_ica
    n = ica.n_components_
    componente_ocular = 0

    proba = _matriz_base(n)
    proba[componente_ocular] = 0.0
    proba[componente_ocular, CATEGORIAS_ICLABEL.index("eye blink")] = 0.95
    _mockear_iclabel(monkeypatch, proba)

    componentes, diagnostico, proba_devuelta = identificar_componentes_oculares(
        raw, ica, umbral_prob=0.8
    )

    assert componentes == [componente_ocular]
    assert proba_devuelta is proba
    assert "eye blink" not in diagnostico  # la categoría ocular no entra en el diagnóstico

    raw_limpio, meta = eliminar_artefactos_oculares(raw, ica, componentes)

    assert meta["n_componentes_eliminadas"] == 1
    assert meta["componentes_eliminadas"] == [componente_ocular]
    assert meta["varianza_explicada_eliminada"] is not None
    assert not np.allclose(raw_limpio.get_data(), raw.get_data())
    assert ica.exclude == []  # eliminar_artefactos_oculares no debe mutar el ica original


# ---------------------------------------------------------------------------
# Caso 2 — sin componente ocular clara: no se fuerza ninguna exclusión
# ---------------------------------------------------------------------------

def test_sin_componente_ocular_no_excluye_nada(raw_y_ica, monkeypatch):
    raw, ica = raw_y_ica
    proba = _matriz_base(ica.n_components_)  # todo 'brain', ninguna 'eye blink' alta
    _mockear_iclabel(monkeypatch, proba)

    componentes, _, _ = identificar_componentes_oculares(raw, ica, umbral_prob=0.8)
    assert componentes == []

    raw_limpio, meta = eliminar_artefactos_oculares(raw, ica, componentes)

    assert meta["n_componentes_eliminadas"] == 0
    assert meta["varianza_explicada_eliminada"] is None
    assert np.allclose(raw_limpio.get_data(), raw.get_data())


# ---------------------------------------------------------------------------
# Caso 3 — otras categorías presentes: se registran pero no se excluyen
# ---------------------------------------------------------------------------

def test_otras_categorias_se_registran_pero_no_se_excluyen(raw_y_ica, monkeypatch):
    raw, ica = raw_y_ica
    n = ica.n_components_
    proba = _matriz_base(n)
    proba[1] = 0.0
    proba[1, CATEGORIAS_ICLABEL.index("muscle artifact")] = 0.85
    proba[2] = 0.0
    proba[2, CATEGORIAS_ICLABEL.index("heart beat")] = 0.90
    proba[3] = 0.0
    proba[3, CATEGORIAS_ICLABEL.index("line noise")] = 0.81
    _mockear_iclabel(monkeypatch, proba)

    componentes, diagnostico, _ = identificar_componentes_oculares(
        raw, ica, umbral_prob=0.8
    )

    assert componentes == []  # nada ocular -> nada se excluye
    assert diagnostico["muscle artifact"] == [
        {"componente": 1, "probabilidad": pytest.approx(0.85)}
    ]
    assert diagnostico["heart beat"] == [
        {"componente": 2, "probabilidad": pytest.approx(0.90)}
    ]
    assert diagnostico["line noise"] == [
        {"componente": 3, "probabilidad": pytest.approx(0.81)}
    ]
    assert "eye blink" not in diagnostico


# ---------------------------------------------------------------------------
# Caso 4 — verificación de rango: n_components = n_good_channels - 1
# ---------------------------------------------------------------------------

def test_n_components_es_n_good_channels_menos_uno():
    raw = _raw_sintetico()
    good_channels = CANALES[:10]

    ica, metadata = entrenar_ica(raw, good_channels, random_state=42)

    assert ica.n_components_ == len(good_channels) - 1
    assert metadata["n_components"] == len(good_channels) - 1
    assert metadata["n_components"] != len(good_channels)


# ---------------------------------------------------------------------------
# Validaciones de entrenar_ica
# ---------------------------------------------------------------------------

def test_entrenar_ica_valida_minimo_dos_canales():
    raw = _raw_sintetico()
    with pytest.raises(ValueError):
        entrenar_ica(raw, ["AF3"])


def test_entrenar_ica_valida_canales_existentes():
    raw = _raw_sintetico()
    with pytest.raises(ValueError):
        entrenar_ica(raw, ["AF3", "CANAL_INEXISTENTE"])


# Aviso esperado: este test entrena la ICA sobre ruido sin filtrar a propósito.
@pytest.mark.filterwarnings("ignore:The data has not been high-pass filtered:RuntimeWarning")
def test_n_components_explicito_para_referencia_original():
    # Sin CAR no hay pérdida de rango: se pueden pedir tantas componentes
    # como canales buenos.
    raw = _raw_sintetico()
    raw_sin_car = mne.io.RawArray(
        np.random.default_rng(1).normal(0, 3e-6, (len(CANALES), N_MUESTRAS)),
        mne.create_info(CANALES, SFREQ, ch_types="eeg"), verbose=False)
    good_channels = CANALES[:10]
    ica, metadata = entrenar_ica(raw_sin_car, good_channels, random_state=42,
                                 n_components=len(good_channels))
    assert ica.n_components_ == len(good_channels)
    assert metadata["n_components"] == len(good_channels)
    with pytest.raises(ValueError, match="fuera de rango"):
        entrenar_ica(raw, good_channels, n_components=len(good_channels) + 1)
