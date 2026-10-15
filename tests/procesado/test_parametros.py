"""
Tests de los parámetros de procesado cerrados (procesado/parametros.py).
"""

import pytest

from eeg_library.procesado.parametros import (
    OPCIONES,
    PARAMETROS_ESTUDIO,
    normalizar_parametros,
)


def test_none_da_los_parametros_del_estudio():
    assert normalizar_parametros(None) == PARAMETROS_ESTUDIO
    assert normalizar_parametros({}) == PARAMETROS_ESTUDIO


def test_parametros_del_estudio_son_los_del_pipeline_original():
    assert PARAMETROS_ESTUDIO == {"referencia": "car_condicional",
                                  "banda_hz": [1.0, 30.0], "notch_hz": 50.0}


def test_completa_las_claves_que_faltan():
    p = normalizar_parametros({"referencia": "original"})
    assert p == {"referencia": "original", "banda_hz": [1.0, 30.0], "notch_hz": 50.0}


def test_forma_canonica_permite_comparar():
    a = normalizar_parametros({"banda_hz": (3, 30), "notch_hz": 50})
    b = normalizar_parametros({"banda_hz": [3.0, 30.0], "notch_hz": 50.0})
    assert a == b
    assert a["banda_hz"] == [3.0, 30.0]


def test_notch_desactivado():
    assert normalizar_parametros({"notch_hz": None})["notch_hz"] is None


@pytest.mark.parametrize("parametros", [
    {"referencia": "promedio"},
    {"banda_hz": [0.5, 40]},
    {"banda_hz": [1, 30, 50]},
    {"notch_hz": 60},
])
def test_rechaza_valores_fuera_de_las_opciones(parametros):
    with pytest.raises(ValueError, match="no admitido"):
        normalizar_parametros(parametros)


def test_rechaza_claves_desconocidas():
    with pytest.raises(ValueError, match="desconocidos"):
        normalizar_parametros({"ica": False})


def test_todas_las_combinaciones_son_validas():
    for ref in OPCIONES["referencia"]:
        for banda in OPCIONES["banda_hz"]:
            for notch in OPCIONES["notch_hz"]:
                normalizar_parametros({"referencia": ref, "banda_hz": banda,
                                       "notch_hz": notch})


def test_no_muta_la_entrada():
    entrada = {"banda_hz": (3, 30)}
    normalizar_parametros(entrada)
    assert entrada == {"banda_hz": (3, 30)}
