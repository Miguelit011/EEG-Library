"""
Tests de los modelos de fatiga aguda (modelado/fatiga_aguda.py).

Usan los modelos reales empaquetados y un bloque `por_ventana` sintético. La
paridad con el análisis sobre EDF reales está en
`tests/datos/test_fatiga_aguda_datos.py`; que las fichas copien las cifras
del test, aquí (el `resultados_test.json` viaja con el repositorio).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from eeg_library.modelado.fatiga_aguda import (
    BANDAS,
    CONDICIONES,
    MIN_VENTANAS,
    cargar_modelo_fatiga_aguda,
    leer_ficha_fatiga_aguda,
    puntuar,
)
from eeg_library.preprocesado.quality import CANALES_EPOC
from eeg_library.procesado.parametros import PARAMETROS_ESTUDIO, normalizar_parametros

RESULTADOS_TEST = (Path(__file__).resolve().parents[2] / "analisis" / "tiempo_en_tarea"
                   / "06_test" / "resultados_test.json")


def _resultado(n=60, excluidos=(), semilla=0, tendencia=0.0):
    """Un `analizar_grabacion` mínimo: parámetros del estudio y `por_ventana` sintético."""
    rng = np.random.default_rng(semilla)
    pot = {}
    for c in CANALES_EPOC:
        bruto = rng.uniform(0.5, 1.5, size=(n, len(BANDAS)))
        # Con `tendencia`, la beta sube y la delta baja a lo largo de la grabación.
        rampa = np.linspace(-1, 1, n)
        bruto[:, 3] *= np.exp(tendencia * rampa)
        bruto[:, 0] *= np.exp(-tendencia * rampa)
        rel = bruto / bruto.sum(axis=1, keepdims=True)
        pot[c] = {b: rel[:, j].tolist() for j, b in enumerate(BANDAS)}
    return {
        "parametros_aplicados": {**PARAMETROS_ESTUDIO, "car_aplicada": True},
        "por_ventana": {
            "instante_s": [2.0 * i for i in range(n)],
            "canal_valido": {c: c not in excluidos for c in CANALES_EPOC},
            "pot_rel": pot,
        },
    }


@pytest.mark.parametrize("condicion", CONDICIONES)
def test_modelos_cargan_con_su_ficha(condicion):
    modelo = cargar_modelo_fatiga_aguda(condicion)
    ficha = modelo.ficha
    assert ficha["condiciones_aplicables"] == [condicion]
    assert ficha["canales"] == CANALES_EPOC
    assert len(ficha["columnas"]) == 56 == modelo.pipeline.n_features_in_
    assert ficha["columnas"][:5] == ["pot_rel_AF3_delta", "pot_rel_AF3_theta",
                                     "pot_rel_AF3_alfa", "pot_rel_AF3_beta", "pot_rel_F7_delta"]
    assert ficha["parametros_validos"] == {**PARAMETROS_ESTUDIO, "banda_hz": [1.0, 30.0]} or \
        ficha["parametros_validos"] == dict(PARAMETROS_ESTUDIO)


@pytest.mark.parametrize("condicion", CONDICIONES)
def test_fichas_copian_las_cifras_del_test(condicion):
    resultados = json.loads(RESULTADOS_TEST.read_text(encoding="utf-8"))
    ficha = leer_ficha_fatiga_aguda(condicion)
    tercios = resultados["descriptivos"][f"completa | {condicion} | tercios_pareado"]
    rho = resultados["descriptivos"][f"completa | {condicion} | rho"]
    assert ficha["resultado_test"]["valor"] == tercios["valor"]
    assert ficha["resultado_test"]["ic95"] == tercios["ic95"]
    assert ficha["resultado_test"]["n_grabaciones"] == tercios["grabaciones"]
    assert ficha["resultado_test"]["n_personas"] == 28
    assert ficha["resultado_test"]["azar"] == 0.5
    assert ficha["resultado_test_rho"]["valor"] == rho["valor"]
    assert ficha["resultado_test_rho"]["ic95"] == rho["ic95"]
    assert ficha["entrenamiento"]["commit_eeg_library"] == resultados["commit"]


def test_ficha_no_promete_medir_fatiga():
    for condicion in CONDICIONES:
        avisos = " ".join(leer_ficha_fatiga_aguda(condicion)["avisos"]).lower()
        assert "no el nivel de fatiga" in avisos


@pytest.mark.parametrize("condicion", CONDICIONES)
def test_puntuar_devuelve_todo_y_es_json(condicion):
    r = puntuar(_resultado(), condicion)
    assert r["disponible"] and r["condicion"] == condicion
    assert r["n_ventanas"] == 60 == len(r["puntuacion"]) == len(r["instante_s"])
    assert -1 <= r["rho"] <= 1
    assert 0 < r["tercios"]["p_primero"] < 1 and 0 < r["tercios"]["p_ultimo"] < 1
    assert r["tercios"]["final_mayor"] == (r["tercios"]["p_ultimo"] > r["tercios"]["p_primero"])
    assert set(r["pesos"]) == set(CANALES_EPOC)
    assert all(set(v) == set(BANDAS) for v in r["pesos"].values())
    json.dumps(r, allow_nan=False)


def test_condicion_en_minusculas_o_con_espacios():
    assert puntuar(_resultado(), " pvtb ")["disponible"]


@pytest.mark.parametrize("condicion", ["OA", "STROOP", "", None])
def test_otras_condiciones_no_disponible(condicion):
    r = puntuar(_resultado(), condicion)
    assert not r["disponible"] and "PASAT y PVTB" in r["motivo"]


def test_otros_parametros_no_disponible():
    resultado = _resultado()
    resultado["parametros_aplicados"]["banda_hz"] = [3.0, 30.0]
    r = puntuar(resultado, "PASAT")
    assert not r["disponible"] and "banda_hz" in r["motivo"]


def test_sin_bloque_por_ventana_pide_reprocesar():
    resultado = _resultado()
    del resultado["por_ventana"]
    r = puntuar(resultado, "PASAT")
    assert not r["disponible"] and "volver a procesarla" in r["motivo"]


def test_minimo_de_ventanas():
    assert not puntuar(_resultado(n=MIN_VENTANAS - 1), "PASAT")["disponible"]
    assert puntuar(_resultado(n=MIN_VENTANAS), "PASAT")["disponible"]


def test_un_factor_constante_por_canal_no_cambia_nada():
    # El centrado por grabación quita cualquier diferencia constante entre grabaciones.
    base = _resultado()
    escalada = _resultado()
    for c in CANALES_EPOC[:5]:
        for b in BANDAS:
            escalada["por_ventana"]["pot_rel"][c][b] = [3.0 * v for v in base["por_ventana"]["pot_rel"][c][b]]
    a, b = puntuar(base, "PVTB"), puntuar(escalada, "PVTB")
    np.testing.assert_allclose(a["puntuacion"], b["puntuacion"], atol=1e-10)
    assert a["tercios"] == pytest.approx(b["tercios"])


def test_canal_excluido_no_influye():
    a = _resultado(excluidos=("T7",))
    b = _resultado(excluidos=("T7",))
    for banda in BANDAS:
        b["por_ventana"]["pot_rel"]["T7"][banda] = list(np.linspace(0.01, 0.9, 60))
    ra, rb = puntuar(a, "PASAT"), puntuar(b, "PASAT")
    np.testing.assert_allclose(ra["puntuacion"], rb["puntuacion"], atol=1e-12)
    assert ra["tercios"] == pytest.approx(rb["tercios"])


def test_puntuacion_es_la_del_pipeline():
    """La puntuación es `decision_function` sobre log10 centrado; la ρ, su Spearman con el tiempo."""
    resultado = _resultado()
    modelo = cargar_modelo_fatiga_aguda("PASAT")
    pv = resultado["por_ventana"]
    X = np.log10(np.column_stack([pv["pot_rel"][c][b] for c in CANALES_EPOC for b in BANDAS]))
    X = X - X.mean(axis=0)
    esperada = modelo.pipeline.decision_function(X)
    r = puntuar(resultado, "PASAT")
    np.testing.assert_allclose(r["puntuacion"], esperada, rtol=0, atol=1e-12)
    assert r["rho"] == pytest.approx(stats.spearmanr(esperada, pv["instante_s"])[0])


def test_tendencia_de_beta_se_ve_en_pvtb():
    # En PVTB la firma confirmada es la beta relativa; con una tendencia clara, el
    # sentido de la ρ tiene que coincidir con el del peso medio de la beta.
    pesos = puntuar(_resultado(), "PVTB")["pesos"]
    beta_menos_delta = np.mean([pesos[c]["beta"] - pesos[c]["delta"] for c in CANALES_EPOC])
    r = puntuar(_resultado(tendencia=1.0), "PVTB")
    assert np.sign(r["rho"]) == np.sign(beta_menos_delta)
    assert abs(r["rho"]) > 0.5
