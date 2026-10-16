"""
Tests de los modelos empaquetados y de `predecir` (modelado/prediccion.py).

Usan los modelos reales de `eeg_library/modelos/`, que viajan con la
librería, pero no el dataset: las características son sintéticas. Que las
clases tengan el nombre correcto sobre datos reales se comprueba en
`tests/datos/`.
"""

import json
import shutil

import pytest

from eeg_library.modelado.prediccion import (
    Modelo,
    cargar_modelo,
    leer_ficha,
    listar_modelos,
    predecir,
)
from eeg_library.procesado.grabacion import ALIAS_CARACTERISTICAS
from eeg_library.procesado.parametros import PARAMETROS_ESTUDIO, normalizar_parametros

MODELOS = ["c1_oa_oc_logistica", "c2_reposo_tarea_logistica", "c3_tareas_logistica"]


def _caracteristicas(ficha, valor=0.0, con_alias=True):
    nombre = (lambda c: ALIAS_CARACTERISTICAS.get(c, c)) if con_alias else (lambda c: c)
    return {nombre(c): valor for c in ficha["columnas"]}


def test_lista_los_tres_modelos_del_estudio():
    assert listar_modelos() == MODELOS


@pytest.mark.parametrize("id_modelo", MODELOS)
def test_ficha_completa_y_coherente(id_modelo):
    modelo = cargar_modelo(id_modelo)
    ficha = modelo.ficha
    assert isinstance(modelo, Modelo) and ficha["id"] == id_modelo
    assert len(ficha["columnas"]) == 70
    assert normalizar_parametros(ficha["parametros_validos"]) == PARAMETROS_ESTUDIO
    test = ficha["resultado_test"]
    assert test["ic95"][0] < test["valor"] < test["ic95"][1]
    assert test["azar"] == pytest.approx(1 / len(ficha["clases"]))
    assert test["n_personas"] == 28
    assert ficha["entrenamiento"]["scikit_learn"] == "1.7.2"


@pytest.mark.parametrize("id_modelo", MODELOS)
def test_condiciones_por_clase(id_modelo):
    """Cada clase dice a qué condiciones corresponde; juntas, las del modelo."""
    ficha = leer_ficha(id_modelo)[0]
    por_clase = ficha["condiciones_por_clase"]
    assert list(por_clase) == [c["nombre"] for c in ficha["clases"]]
    todas = [cond for conds in por_clase.values() for cond in conds]
    assert sorted(todas) == sorted(ficha["condiciones_aplicables"])
    assert len(todas) == len(set(todas))       # ninguna condición en dos clases


def test_condiciones_de_reposo_y_tarea():
    por_clase = leer_ficha("c2_reposo_tarea_logistica")[0]["condiciones_por_clase"]
    assert por_clase["reposo"] == ["OA", "OC"]
    assert por_clase["tarea"] == ["STROOP", "PASAT", "PVTB", "MSIT", "PVSAT"]


def test_nombres_de_clase():
    nombres = {m: [c["nombre"] for c in leer_ficha(m)[0]["clases"]] for m in MODELOS}
    assert nombres == {
        "c1_oa_oc_logistica": ["OA", "OC"],
        "c2_reposo_tarea_logistica": ["reposo", "tarea"],
        "c3_tareas_logistica": ["STROOP", "PASAT", "PVTB", "MSIT", "PVSAT"],
    }


def test_cifras_del_test_confirmatorio():
    valores = {m: round(leer_ficha(m)[0]["resultado_test"]["valor"], 3) for m in MODELOS}
    assert valores == {"c1_oa_oc_logistica": 0.855, "c2_reposo_tarea_logistica": 0.851,
                       "c3_tareas_logistica": 0.488}


@pytest.mark.parametrize("id_modelo", MODELOS)
def test_predice_con_los_parametros_del_entrenamiento(id_modelo):
    modelo = cargar_modelo(id_modelo)
    r = predecir(modelo, _caracteristicas(modelo.ficha), PARAMETROS_ESTUDIO)
    nombres = [c["nombre"] for c in modelo.ficha["clases"]]
    assert r["disponible"] is True
    assert list(r["probabilidades"]) == nombres
    assert sum(r["probabilidades"].values()) == pytest.approx(1.0)
    assert r["clase"] == max(r["probabilidades"], key=r["probabilidades"].get)


def test_alias_y_nombres_originales_dan_lo_mismo():
    modelo = cargar_modelo("c2_reposo_tarea_logistica")
    a = predecir(modelo, _caracteristicas(modelo.ficha, 0.1, con_alias=True), PARAMETROS_ESTUDIO)
    b = predecir(modelo, _caracteristicas(modelo.ficha, 0.1, con_alias=False), PARAMETROS_ESTUDIO)
    assert a == b


def test_none_lo_imputa_el_pipeline():
    modelo = cargar_modelo("c1_oa_oc_logistica")
    caracteristicas = _caracteristicas(modelo.ficha)
    caracteristicas["temporal_alfa_rel_media"] = None
    assert predecir(modelo, caracteristicas, PARAMETROS_ESTUDIO)["disponible"]


@pytest.mark.parametrize("cambio", [
    {"referencia": "original"}, {"banda_hz": [3, 30]}, {"notch_hz": None},
])
def test_no_disponible_con_otros_parametros(cambio):
    modelo = cargar_modelo("c1_oa_oc_logistica")
    r = predecir(modelo, _caracteristicas(modelo.ficha), {**PARAMETROS_ESTUDIO, **cambio})
    assert r["disponible"] is False
    assert "clase" not in r
    assert r["parametros_requeridos"] == PARAMETROS_ESTUDIO
    assert list(cambio)[0] in r["motivo"]


def test_acepta_parametros_aplicados_de_analizar_grabacion():
    # El bloque `parametros_aplicados` trae más claves (ICA, recorte...).
    aplicados = {**PARAMETROS_ESTUDIO, "car_aplicada": False, "ica": {"n_componentes": 3}}
    modelo = cargar_modelo("c1_oa_oc_logistica")
    assert predecir(modelo, _caracteristicas(modelo.ficha), aplicados)["disponible"]


def test_faltan_caracteristicas():
    modelo = cargar_modelo("c3_tareas_logistica")
    caracteristicas = _caracteristicas(modelo.ficha)
    del caracteristicas["frontal_delta_rel_media"]
    with pytest.raises(ValueError, match="frontal_delta_rel_media"):
        predecir(modelo, caracteristicas, PARAMETROS_ESTUDIO)


def test_predecir_acepta_el_identificador():
    ficha, _ = leer_ficha("c1_oa_oc_logistica")
    assert predecir("c1_oa_oc_logistica", _caracteristicas(ficha), None)["disponible"]


def _copia_externa(tmp_path, id_modelo="c1_oa_oc_logistica"):
    """Un modelo fuera de la librería: su .joblib y su ficha en otra carpeta."""
    ficha, ruta = leer_ficha(id_modelo)
    shutil.copy(ruta, tmp_path / ruta.name)
    shutil.copy(ruta.parent / ficha["archivo"], tmp_path / ficha["archivo"])
    return tmp_path / ruta.name


def test_modelo_externo_por_ruta_y_con_otros_parametros(tmp_path):
    ruta = _copia_externa(tmp_path)
    ficha = json.loads(ruta.read_text(encoding="utf-8"))
    ficha["id"] = "externo"
    ficha["parametros_validos"] = {"referencia": "original", "banda_hz": [3, 30],
                                   "notch_hz": None}
    ruta.write_text(json.dumps(ficha), encoding="utf-8")
    modelo = cargar_modelo(ruta)
    assert modelo.id == "externo"
    assert not predecir(modelo, _caracteristicas(ficha), PARAMETROS_ESTUDIO)["disponible"]
    assert predecir(modelo, _caracteristicas(ficha), ficha["parametros_validos"])["disponible"]


def test_sha_distinto_no_carga(tmp_path):
    ruta = _copia_externa(tmp_path)
    ficha = json.loads(ruta.read_text(encoding="utf-8"))
    ficha["sha256_modelo"] = "0" * 64
    ruta.write_text(json.dumps(ficha), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        cargar_modelo(ruta)


def test_ficha_incompleta(tmp_path):
    ruta = _copia_externa(tmp_path)
    ficha = json.loads(ruta.read_text(encoding="utf-8"))
    del ficha["parametros_validos"]
    ruta.write_text(json.dumps(ficha), encoding="utf-8")
    with pytest.raises(ValueError, match="parametros_validos"):
        cargar_modelo(ruta)


def test_modelo_inexistente():
    with pytest.raises(FileNotFoundError):
        cargar_modelo("c9_no_existe")
