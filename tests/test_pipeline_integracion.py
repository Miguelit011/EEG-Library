"""
Test de integración del pipeline completo (Fases 1-6) en el orden corregido.

Los tests de cada módulo (test_quality.py, test_filtering.py, etc.) prueban
cada función de forma aislada con datos sintéticos y no habrían detectado el
error de orden documentado en docs/ESTADO.md ("Nota de corrección de orden"): que
`evaluar_calidad`/`aplicar_car` se ejecutaran antes de `recortar_registro`/
`aplicar_filtros` contaminaba las métricas de calidad con artefactos de borde
y deriva de baja frecuencia sin filtrar.

Este archivo encadena las funciones reales de Fases 2-6 (recortar, filtrar,
evaluar calidad, CAR, ICA + ICLabel) sobre un Raw sintético que simula esos
dos problemas para demostrar que el orden corregido los evita.

Fase 1 (`cargar_eeg`) no se ejercita aquí porque requiere un archivo EDF real
en disco; se parte de un `RawArray` sintético equivalente a su salida (con
montaje asignado), igual que hace `test_artifacts.py`.
"""

import numpy as np
import pytest
import mne

from eeg_library.preprocesado.filtering import recortar_registro, aplicar_filtros
from eeg_library.preprocesado.quality import evaluar_calidad, CANALES_EPOC
from eeg_library.preprocesado.referencing import aplicar_car
from eeg_library.preprocesado import artifacts
from eeg_library.preprocesado.artifacts import (
    entrenar_ica,
    identificar_componentes_oculares,
    eliminar_artefactos_oculares,
    CATEGORIAS_ICLABEL,
)
from eeg_library.preprocesado.reader import MONTAJE_1020

SFREQ = 256
DURACION_S = 40  # se recortan 5 s al inicio y 5 s al final -> 30 s finales
N_MUESTRAS = SFREQ * DURACION_S
MARGEN_S = 5.0
CANAL_CONTAMINADO = "F3"


def _raw_crudo_con_artefactos_de_orden(semilla=0):
    """
    Raw sintético (14 canales EPOC) equivalente a la salida de cargar_eeg,
    donde un canal está contaminado con los dos problemas que motivaron la
    corrección de orden:

      - Deriva de baja frecuencia (0.05 Hz) en todo el registro, muy por
        debajo de la banda de interés 1-30 Hz -> debe eliminarla el filtro
        (Fase 3), no el recorte.
      - Un transitorio de gran amplitud confinado a los primeros y últimos
        MARGEN_S segundos (colocación/retirada del casco) -> debe
        eliminarlo el recorte (Fase 2), no el filtro.

    El resto de canales llevan solo ruido gaussiano de amplitud típica EEG,
    con una pequeña variación de std entre canales (para que el MAD
    inter-canal de evaluar_calidad sea > 0).
    """
    rng = np.random.default_rng(semilla)
    t = np.arange(N_MUESTRAS) / SFREQ

    data = {}
    for i, canal in enumerate(CANALES_EPOC):
        std = 5e-6 * (1 + 0.05 * i)  # varía ligeramente entre canales
        data[canal] = rng.normal(0, std, N_MUESTRAS)

    deriva = 5e-5 * np.sin(2 * np.pi * 0.05 * t)
    transitorio = np.zeros(N_MUESTRAS)
    n_margen = int(MARGEN_S * SFREQ)
    transitorio[:n_margen] = 3e-4
    transitorio[-n_margen:] = -3e-4
    data[CANAL_CONTAMINADO] = data[CANAL_CONTAMINADO] + deriva + transitorio

    info = mne.create_info(list(data.keys()), SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(np.array(list(data.values())), info, verbose=False)
    raw.set_montage(MONTAJE_1020, on_missing="warn", verbose=False)
    return raw


def _mockear_iclabel_sin_oculares(monkeypatch, n_componentes):
    """Todas las componentes clasificadas como 'brain' -> nada se excluye."""
    proba = np.zeros((n_componentes, len(CATEGORIAS_ICLABEL)))
    proba[:, CATEGORIAS_ICLABEL.index("brain")] = 0.9
    monkeypatch.setattr(
        artifacts, "iclabel_label_components", lambda raw, ica: proba
    )
    return proba


# ---------------------------------------------------------------------------
# El orden incorrecto (evaluar calidad antes de recortar/filtrar) genera
# el falso positivo que documenta la Nota de corrección de docs/ESTADO.md.
# ---------------------------------------------------------------------------

def test_orden_incorrecto_genera_falso_positivo_en_evaluar_calidad():
    raw_crudo = _raw_crudo_con_artefactos_de_orden()

    good_channels, excluidos, _ = evaluar_calidad(raw_crudo)

    assert CANAL_CONTAMINADO not in good_channels
    assert CANAL_CONTAMINADO in excluidos


# ---------------------------------------------------------------------------
# Pipeline completo Fases 2-6 en el orden corregido: recortar -> filtrar ->
# evaluar calidad -> CAR -> ICA -> ICLabel -> eliminar artefactos oculares.
# ---------------------------------------------------------------------------

def test_pipeline_completo_orden_corregido(monkeypatch):
    raw_crudo = _raw_crudo_con_artefactos_de_orden()

    # Fase 2 — recorte de márgenes
    raw_rec, meta_rec = recortar_registro(raw_crudo, inicio=MARGEN_S, fin=MARGEN_S)
    assert meta_rec["metodo"] == "crop"
    assert raw_rec.times[-1] == pytest.approx(DURACION_S - 2 * MARGEN_S, abs=1.0 / SFREQ)

    # Fase 3 — notch + bandpass
    raw_filt, meta_filt = aplicar_filtros(raw_rec)
    assert meta_filt["banda_hz"] == [1.0, 30.0]

    # Fase 4 — evaluación de calidad, ya sobre señal recortada y filtrada:
    # el canal contaminado ya no debe excluirse (a diferencia del test
    # anterior, que lo evalúa directamente sobre el Raw crudo).
    good_channels, excluidos, metricas = evaluar_calidad(raw_filt)
    assert CANAL_CONTAMINADO in good_channels
    assert CANAL_CONTAMINADO not in excluidos
    assert set(good_channels) == set(CANALES_EPOC)

    # Fase 5 — CAR
    raw_ref, meta_car = aplicar_car(raw_filt, good_channels)
    assert meta_car["metodo"] == "CAR"

    # Fase 6 — ICA (Infomax Extended) + ICLabel (mockeado, ver test_artifacts.py)
    ica, meta_ica = entrenar_ica(raw_ref, good_channels, random_state=42)
    assert meta_ica["n_components"] == len(good_channels) - 1

    proba = _mockear_iclabel_sin_oculares(monkeypatch, ica.n_components_)
    componentes, diagnostico, proba_devuelta = identificar_componentes_oculares(
        raw_ref, ica, umbral_prob=0.8
    )
    assert proba_devuelta is proba
    assert componentes == []
    # Ninguna componente ocular; el diagnóstico solo recoge las 'brain'
    # (única categoría con probabilidad >= umbral en esta matriz mockeada).
    assert set(diagnostico.keys()) == {"brain"}
    assert len(diagnostico["brain"]) == ica.n_components_

    raw_limpio, meta_limpieza = eliminar_artefactos_oculares(raw_ref, ica, componentes)
    assert meta_limpieza["n_componentes_eliminadas"] == 0
    assert np.allclose(raw_limpio.get_data(), raw_ref.get_data())


def test_pipeline_completo_elimina_componente_ocular_marcada(monkeypatch):
    raw_crudo = _raw_crudo_con_artefactos_de_orden()
    raw_rec, _ = recortar_registro(raw_crudo, inicio=MARGEN_S, fin=MARGEN_S)
    raw_filt, _ = aplicar_filtros(raw_rec)
    good_channels, _, _ = evaluar_calidad(raw_filt)
    raw_ref, _ = aplicar_car(raw_filt, good_channels)
    ica, _ = entrenar_ica(raw_ref, good_channels, random_state=42)

    proba = np.zeros((ica.n_components_, len(CATEGORIAS_ICLABEL)))
    proba[:, CATEGORIAS_ICLABEL.index("brain")] = 0.9
    proba[0] = 0.0
    proba[0, CATEGORIAS_ICLABEL.index("eye blink")] = 0.95
    monkeypatch.setattr(
        artifacts, "iclabel_label_components", lambda raw, ica: proba
    )

    componentes, _, _ = identificar_componentes_oculares(raw_ref, ica, umbral_prob=0.8)
    assert componentes == [0]

    raw_limpio, meta_limpieza = eliminar_artefactos_oculares(raw_ref, ica, componentes)
    assert meta_limpieza["n_componentes_eliminadas"] == 1
    assert not np.allclose(raw_limpio.get_data(), raw_ref.get_data())


# ---------------------------------------------------------------------------
# Los canales no-EEG no deben llegar a la matriz de features (Fases 1 -> 9).
#
# Regresión del segundo error de integración detectado, hermano del error de
# orden de arriba: los tests por módulo parten de RawArrays que ya contienen
# solo los 14 electrodos, así que ninguno detectaba que `cargar_eeg` dejara
# telemetría y CQ tipados como 'eeg'. Sobre archivos reales del dataset final
# eso hacía que `extraer_bandas` devolviera 568 columnas de potencia en vez de
# 112, con features de acelerómetro, batería y calidad de contacto entre las
# que el Random Forest tendría que elegir.
#
# Se salta la Fase 6 (ICA): lo que se comprueba aquí es la propagación del
# tipado de canales, que no depende de ella.
# ---------------------------------------------------------------------------

def _raw_export_con_canales_no_eeg(semilla=0):
    """
    Raw equivalente a lo que MNE devuelve al leer un EDF del dataset final:
    los 14 electrodos más canales de calidad de contacto, telemetría y
    movimiento, todos tipados como 'eeg' (es `cargar_eeg` quien debe
    reclasificarlos).
    """
    rng = np.random.default_rng(semilla)
    nombres, filas = [], []

    for i, canal in enumerate(CANALES_EPOC):
        nombres.append(canal)
        filas.append(rng.normal(0, 5e-6 * (1 + 0.05 * i), N_MUESTRAS))

    for canal in CANALES_EPOC:
        nombres.append(f"CQ.{canal}")
        filas.append(np.full(N_MUESTRAS, 4.0))

    for canal in ("TimestampS", "Counter", "Battery", "MOT.AccX", "MOT.MagZ"):
        nombres.append(canal)
        filas.append(rng.normal(0, 1.0, N_MUESTRAS))

    info = mne.create_info(nombres, SFREQ, ch_types="eeg")
    return mne.io.RawArray(np.array(filas), info, verbose=False)


def test_features_solo_contienen_los_14_canales_eeg(monkeypatch, tmp_path):
    from eeg_library.preprocesado.reader import cargar_eeg
    from eeg_library.procesado.epoching import crear_epocas
    from eeg_library.procesado.spectral import calcular_psd
    from eeg_library.procesado.bands import extraer_bandas, BANDAS_DEFECTO

    ruta = tmp_path / "sub-C1F081_ses-M4_task-PASAT_eeg.edf"
    ruta.write_bytes(b"")
    raw_mock = _raw_export_con_canales_no_eeg()
    monkeypatch.setattr(
        mne.io, "read_raw_edf",
        lambda ruta, preload=True, verbose=False: raw_mock,
    )

    raw, meta_carga = cargar_eeg(str(ruta))
    assert meta_carga["n_canales_eeg"] == len(CANALES_EPOC)
    assert meta_carga["n_canales"] > len(CANALES_EPOC)

    raw_rec, _ = recortar_registro(raw, inicio=MARGEN_S, fin=MARGEN_S)
    raw_filt, _ = aplicar_filtros(raw_rec)
    good_channels, _, _ = evaluar_calidad(raw_filt)
    raw_ref, _ = aplicar_car(raw_filt, good_channels)

    epochs, _ = crear_epocas(
        raw_ref, sujeto="C1_F081", condicion="PASAT", sesion="M4",
        archivo_origen=ruta.name,
    )
    spectrum, meta_psd = calcular_psd(epochs)
    assert meta_psd["n_canales"] == len(CANALES_EPOC)
    assert set(spectrum.ch_names) == set(CANALES_EPOC)

    # canales_validos se pasa siempre que se dispone de él: restringe los
    # promedios de clúster del índice de fatiga a los canales buenos, sin
    # alterar las columnas de potencia por canal.
    features, meta_bandas = extraer_bandas(spectrum, canales_validos=good_channels)
    assert meta_bandas["cluster_frontal_descartados"] == []

    n_esperado = len(CANALES_EPOC) * len(BANDAS_DEFECTO) * 2  # 14 x 4 x 2 = 112
    columnas_potencia = [
        c for c in features.columns if c.startswith(("pot_abs_", "pot_rel_"))
    ]
    assert meta_bandas["n_features_banda"] == n_esperado
    assert len(columnas_potencia) == n_esperado

    # Ninguna columna puede proceder de un canal que no sea un electrodo.
    for prefijo in ("CQ.", "MOT.", "Timestamp", "Counter", "Battery"):
        assert not any(prefijo in c for c in columnas_potencia)
