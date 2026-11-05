"""
Tests de EEGNet en la librería (modelado/tiempo_en_tarea_eegnet.py), sin el dataset.

Usan los `.onnx` empaquetados y datos sintéticos. La paridad con el análisis sobre EDF reales
está en `tests/datos/test_tiempo_en_tarea_eegnet_datos.py`; la de ONNX con PyTorch, en
`analisis/tiempo_en_tarea/deep_learning/05_onnx/resultados.json`.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from eeg_library.modelado.tiempo_en_tarea_eegnet import (
    MIN_VENTANAS,
    _dir_modelos,
    leer_ficha_eegnet,
    preparar_entrada,
    puntuar_eegnet,
    puntuar_entrada,
    puntuar_ventanas,
)
from eeg_library.preprocesado.quality import CANALES_EPOC
from eeg_library.procesado.parametros import PARAMETROS_ESTUDIO

DL = Path(__file__).resolve().parents[2] / "analisis" / "tiempo_en_tarea" / "deep_learning"


def _ventanas(n=30, semilla=0):
    return np.random.default_rng(semilla).normal(scale=1e-5, size=(n, 14, 1024)).astype(np.float32)


def _resultado(puntuacion, parametros=None):
    n = len(puntuacion)
    return {"parametros_aplicados": {**PARAMETROS_ESTUDIO, **(parametros or {})},
            "por_ventana": {"instante_s": [2.0 * i for i in range(n)], "eegnet": list(puntuacion)}}


def test_ficha_y_archivos():
    ficha = leer_ficha_eegnet()
    assert ficha["canales"] == CANALES_EPOC and ficha["condiciones_aplicables"] == ["PASAT", "PVTB"]
    assert [s["semilla"] for s in ficha["semillas"]] == [0, 1, 2]
    for s in ficha["semillas"]:
        ruta = _dir_modelos() / s["archivo"]
        assert hashlib.sha256(ruta.read_bytes()).hexdigest() == s["sha256"]
    config = json.loads((DL / "04_test_eegnet" / "configuracion_dl.json").read_text(encoding="utf-8"))
    assert [s["sha256_pt"] for s in ficha["semillas"]] == [config["pesos"][f"s{i}"] for i in range(3)]
    assert ficha["exportacion"]["dif_max_onnx_torch"] < 1e-5
    assert any("por separado" in a for a in ficha["avisos"])     # la diferencia con el test, declarada


def test_ficha_copia_las_cifras_del_test():
    c = json.loads((DL / "04_test_eegnet" / "resultados_test.json").read_text(encoding="utf-8"))["cifras"]
    ficha = leer_ficha_eegnet()
    assert ficha["resultado_test"]["valor"] == c["tercios_cnn_familia"]["valor"]
    assert ficha["resultado_test"]["ic95"] == c["tercios_cnn_familia"]["ic95"]
    assert ficha["resultado_test_rho"]["valor"] == c["cnn_familia"]["valor"]
    assert ficha["comparacion_logistica"]["veredicto"] == c["P1"]["veredicto"] == "NO SE DISTINGUE"


def test_preparar_entrada():
    validos = [c not in ("T7", "O2") for c in CANALES_EPOC]
    X, M = preparar_entrada(_ventanas(), validos)
    assert X.shape == (30, 14, 512) and X.dtype == np.float32
    assert M.shape == (30, 14) and (M == np.array(validos, dtype=np.uint8)).all()
    i, j = CANALES_EPOC.index("T7"), CANALES_EPOC.index("AF3")
    assert (X[:, i] == 0).all()                                          # excluido a 0
    assert abs(X[:, j].mean()) < 1e-6 and X[:, j].std() == pytest.approx(1, abs=1e-5)   # z-score


def test_la_escala_de_la_senal_no_importa():
    """El z-score por grabación quita la escala: multiplicar la señal no cambia la puntuación."""
    v = _ventanas()
    a = puntuar_ventanas(v, [True] * 14)
    b = puntuar_ventanas(v * 7.0, [True] * 14)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_puntuar_entrada_tres_semillas():
    X, M = preparar_entrada(_ventanas(n=700), [True] * 14)      # más de un lote
    p = puntuar_entrada(X, M)
    assert p.shape == (3, 700) and np.isfinite(p).all()
    np.testing.assert_allclose(puntuar_ventanas(_ventanas(n=700), [True] * 14), p.mean(axis=0), atol=1e-6)


def test_puntuar_eegnet_tercios_y_rho():
    s = np.linspace(-1, 1, 60) + np.random.default_rng(1).normal(scale=0.3, size=60)
    r = puntuar_eegnet(_resultado(s), "PVTB")
    assert r["disponible"] and r["n_ventanas"] == 60
    t = np.arange(60) * 2.0
    primero, ultimo = t <= t[-1] / 3, t >= 2 * t[-1] / 3
    assert r["tercios"]["media_primero"] == pytest.approx(s[primero].mean())
    assert r["tercios"]["media_ultimo"] == pytest.approx(s[ultimo].mean())
    assert r["tercios"]["final_mayor"] is True
    assert r["rho"] == pytest.approx(stats.spearmanr(s, t)[0])
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize("condicion", ["OA", "", None])
def test_otras_condiciones(condicion):
    assert not puntuar_eegnet(_resultado(np.zeros(30)), condicion)["disponible"]


def test_otros_parametros():
    r = puntuar_eegnet(_resultado(np.zeros(30), {"banda_hz": [3.0, 30.0]}), "PASAT")
    assert not r["disponible"] and "banda_hz" in r["motivo"]


def test_sin_puntuacion_pide_reprocesar():
    resultado = _resultado(np.zeros(30))
    resultado["por_ventana"]["eegnet"] = None
    assert "volver a procesarla" in puntuar_eegnet(resultado, "PASAT")["motivo"]
    del resultado["por_ventana"]["eegnet"]
    assert "volver a procesarla" in puntuar_eegnet(resultado, "PASAT")["motivo"]


def test_minimo_de_ventanas():
    assert not puntuar_eegnet(_resultado(np.zeros(MIN_VENTANAS - 1)), "PASAT")["disponible"]
    assert puntuar_eegnet(_resultado(np.arange(MIN_VENTANAS, dtype=float)), "PASAT")["disponible"]
