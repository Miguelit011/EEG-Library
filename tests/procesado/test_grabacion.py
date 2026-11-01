"""
Tests de `analizar_grabacion`, el contrato con la plataforma web.

Usan el EDF sintético de `test_batch.py` (escrito en disco, con canales CQ y
de telemetría) y el pipeline real, ICLabel incluido. La paridad con
`matriz_rf.parquet` sobre EDF reales está en `tests/datos/`.
"""

import json

import pandas as pd
import pytest

from eeg_library.preprocesado.quality import CANALES_EPOC
from eeg_library.procesado.batch import procesar_archivo
from eeg_library.procesado.grabacion import ALIAS_CARACTERISTICAS, analizar_grabacion
from eeg_library.procesado.matrix import REGIONES
from eeg_library.procesado.parametros import PARAMETROS_ESTUDIO

from test_batch import _escribir_edf


@pytest.fixture(scope="module")
def edf(tmp_path_factory):
    return _escribir_edf(str(tmp_path_factory.mktemp("edf")), "grabacion.edf")


@pytest.fixture(scope="module")
def resultado(edf):
    return analizar_grabacion(edf)


def _claves(objeto):
    """Todas las claves de un JSON anidado."""
    if isinstance(objeto, dict):
        for k, v in objeto.items():
            yield k
            yield from _claves(v)
    elif isinstance(objeto, list):
        for v in objeto:
            yield from _claves(v)


def test_secciones_del_contrato(resultado):
    assert set(resultado) == {"metadatos", "parametros_aplicados", "calidad",
                              "potencia", "caracteristicas", "por_ventana"}


def test_es_json_estricto(resultado):
    # allow_nan=False falla con NaN/inf: deben haberse convertido a null.
    json.dumps(resultado, allow_nan=False)


def test_metadatos(resultado):
    meta = resultado["metadatos"]
    assert meta["archivo"] == "grabacion.edf"
    assert meta["sfreq_hz"] == 256.0
    assert meta["canales_eeg"] == CANALES_EPOC
    assert meta["n_epocas"] > 0
    assert meta["versiones"]["eeg_library"]


def test_parametros_por_defecto_son_los_del_estudio(resultado):
    aplicados = resultado["parametros_aplicados"]
    for clave, valor in PARAMETROS_ESTUDIO.items():
        assert aplicados[clave] == valor
    assert aplicados["car_aplicada"] is True
    # Con CAR, ICA usa n_buenos - 1 componentes (como en el estudio).
    assert aplicados["ica"]["n_componentes"] == resultado["metadatos"]["n_canales_buenos"] - 1


def test_calidad_por_canal(resultado):
    calidad = resultado["calidad"]
    assert list(calidad) == CANALES_EPOC
    for canal, info in calidad.items():
        assert info["estado"] in ("bueno", "descartado")
        assert (info["motivo"] is None) == (info["estado"] == "bueno")
        assert info["region"] in REGIONES
        assert set(info["metricas"]) == {"cq_medio", "pct_muestras_malas",
                                         "varianza", "z_robusto"}


def test_potencia_por_canal_y_region(resultado):
    potencia = resultado["potencia"]
    assert set(potencia["bandas_hz"]) == {"delta", "theta", "alfa", "beta"}
    for canal in CANALES_EPOC:
        relativas = [potencia["por_canal"][canal][b]["relativa"]
                     for b in potencia["bandas_hz"]]
        assert sum(relativas) == pytest.approx(1.0)
    assert set(potencia["por_region"]) == set(REGIONES)
    assert potencia["por_region"]["posterior"]["n_canales_validos"] == 4


def test_setenta_caracteristicas_con_alias_neutro(resultado):
    caracteristicas = resultado["caracteristicas"]
    assert len(caracteristicas) == 70
    for original, alias in ALIAS_CARACTERISTICAS.items():
        assert original not in caracteristicas
        assert alias in caracteristicas


def test_potencia_por_ventana(resultado):
    """Una lista por canal y banda, con un valor por ventana, coherente con la media guardada."""
    pv = resultado["por_ventana"]
    n = resultado["metadatos"]["n_epocas"]
    assert len(pv["instante_s"]) == n
    assert pv["instante_s"][:2] == [0.0, 2.0]          # ventanas de 4 s con paso de 2 s
    assert list(pv["pot_rel"]) == CANALES_EPOC == list(pv["canal_valido"])
    for canal, bandas in pv["pot_rel"].items():
        assert list(bandas) == ["delta", "theta", "alfa", "beta"]
        assert pv["canal_valido"][canal] == resultado["potencia"]["por_canal"][canal]["valido"]
        for banda, valores in bandas.items():
            assert len(valores) == n
            media = resultado["potencia"]["por_canal"][canal][banda]["relativa"]
            assert sum(valores) / n == pytest.approx(media, rel=1e-12)


def test_ninguna_clave_menciona_fatiga(resultado):
    assert not [k for k in _claves(resultado) if "fatiga" in k.lower()]


def test_referencia_original_no_aplica_car(edf):
    r = analizar_grabacion(edf, {"referencia": "original"})
    aplicados = r["parametros_aplicados"]
    assert aplicados["referencia"] == "original"
    assert aplicados["car_aplicada"] is False
    # Sin CAR no hay pérdida de rango: tantas componentes como canales buenos.
    assert aplicados["ica"]["n_componentes"] == r["metadatos"]["n_canales_buenos"]


def test_banda_y_notch_aplicados(edf, resultado):
    r = analizar_grabacion(edf, {"banda_hz": [3, 30], "notch_hz": None})
    aplicados = r["parametros_aplicados"]
    assert aplicados["banda_hz"] == [3.0, 30.0]
    assert aplicados["notch_hz"] is None
    # Con el paso alto a 3 Hz cae la potencia relativa de delta (1-4 Hz).
    delta = lambda res: res["potencia"]["por_canal"]["O1"]["delta"]["relativa"]  # noqa: E731
    assert delta(r) < delta(resultado)


def test_parametros_invalidos(edf):
    with pytest.raises(ValueError, match="no admitido"):
        analizar_grabacion(edf, {"banda_hz": [0.5, 40]})


def test_procesar_archivo_sin_parametros_es_el_estudio(edf):
    a, _, _ = procesar_archivo(edf, "S", "OA", "M2")
    b, _, _ = procesar_archivo(edf, "S", "OA", "M2", parametros=PARAMETROS_ESTUDIO)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
