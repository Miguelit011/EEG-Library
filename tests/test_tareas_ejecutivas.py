"""
Tests de `modelado/tareas_ejecutivas.py`: clasificación multiclase de las cinco
tareas ejecutivas (STROOP, PASAT, PVTB, MSIT, PVSAT). OA y OC quedan fuera; sus
tests viven en `test_reposo.py` y `test_reposo_tarea.py`.

Protegen los invariantes que hacen que el resultado de ese análisis signifique
algo:

1. **`condicion_orden` nunca entra a X.** Dentro de las 5 tareas toma un valor
   constante distinto por cada una (3-7): es la etiqueta en otra escala, más
   grosera todavía que en el contraste binario.
2. **El conjunto de test no se abre** sin pedirlo explícitamente.
3. **La exactitud balanceada es la métrica principal** porque su azar es
   exactamente 1/5 = 0.200 con 5 clases, a diferencia de la exactitud simple.
4. **Los coeficientes multinomiales tienen la forma correcta** (una fila por
   clase, no una sola), y el mensaje de error es claro si no la tienen.
5. **El patrón de confusión se puede comparar con estructuras candidatas**
   (similitud cognitiva, distancia en el protocolo, distancia de duración), y
   esa comparación puede dar cualquiera de los dos resultados: el test no
   asume que la similitud cognitiva siempre gana.
6. **La validación cruzada agrupa por persona**, con el mismo control negativo
   que en OA/OC y reposo/tarea, adaptado al azar de 5 clases.

Todo con datos sintéticos en memoria: ningún test toca disco.
"""

import itertools

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

from eeg_library.modelado.reposo import (
    COLUMNA_FUGA,
    COLUMNAS_PROCESAMIENTO,
    columnas_de_features_reposo,
    permutar_etiquetas_dentro_de_persona,
)
from eeg_library.modelado.tareas_ejecutivas import (
    CODIGO_DE_CLASE,
    CONDICIONES_EJECUTIVAS,
    SIMILITUD_COGNITIVA,
    coeficiente_neto_por_clase,
    coeficientes_por_clase_y_pliegue,
    columnas_de_features_ejecutivas,
    confusion_por_pares,
    distancias_de_duracion,
    estructura_de_la_confusion,
    etiqueta_multiclase,
    evaluar_multiclase,
    matriz_confusion_normalizada,
    metricas_multiclase,
    modelo_bosque_multiclase,
    modelo_logistico_multinomial,
    nombres_de_clase,
    preparar_tareas_ejecutivas,
    separabilidad_por_pares,
    similitud_cognitiva,
    tamanos_de_efecto_una_contra_resto,
)
from eeg_library.procesado.bands import BANDAS_DEFECTO
from eeg_library.procesado.matrix import (
    COLUMNAS_IDENTIDAD,
    ORDEN_PROTOCOLO,
    REGIONES,
    columnas_de_features,
)

CAMPOS_IDENTIDAD = {
    "archivo": None, "sesion_uid": None, "persona_uid": None, "sujeto_id": None,
    "centro": "C1", "medicion": "M2", "condicion": "OC", "split": "train",
    "etiquetada": True, "concordancia_fecha": "coincide", "corregida": False,
    "ocupante": 1, "procesada": True, "vas": 50.0,
}

# Duración mediana por condición, reproduciendo el patrón real (ESTADO.md):
# STROOP y MSIT cortas, el resto larga. Sirve para los tests de duración.
_DURACION = {"OA": 110.0, "OC": 110.0, "STROOP": 60.0, "MSIT": 60.0,
            "PASAT": 150.0, "PVTB": 150.0, "PVSAT": 150.0}


def _esquema_fase_13():
    """Los 91 nombres de columna reales de `matriz_rf.parquet`."""
    features = [f"{r}_{b}_{t}_{s}" for r in REGIONES for b in BANDAS_DEFECTO
                for t in ("logabs", "rel") for s in ("media", "sd")]
    features += ["theta_frontal_log_media", "theta_frontal_log_sd",
                 "alfa_posterior_log_media", "alfa_posterior_log_sd",
                 "indice_fatiga_media", "indice_fatiga_sd"]
    features += list(COLUMNAS_PROCESAMIENTO) + [COLUMNA_FUGA]
    return list(COLUMNAS_IDENTIDAD) + ["vas"] + features


def _matriz_ejecutivas(n_personas=25, n_sesiones=4, separacion=3.0, semilla=0):
    """
    Matriz por grabación mínima con la forma de la Fase 13, las 7 condiciones.

    Cada persona tiene un nivel propio en `feat_a` y en `feat_b` (su «firma»,
    sd 0.5). `feat_a` sube `separacion` en STROOP y en MSIT (y solo ahí);
    `feat_b` sube `separacion` en PASAT y en PVSAT (y solo ahí). PVTB no tiene
    ningún salto en ninguna de las dos: es la tarea más distinta y, con estas
    dos features, la que ningún par comparte —igual que en el diseño real,
    donde PVTB es cognitivamente la más distinta de las cinco—. `n_ventanas`
    reproduce el patrón real de duraciones (`_DURACION`). `ruido_1`/`ruido_2`
    no dependen de la condición y `firma` identifica a la persona sin decir
    nada de la condición. Splits por persona: 70 % train, 15 % val, 15 % test.
    """
    rng = np.random.default_rng(semilla)
    filas = []
    for p in range(n_personas):
        fraccion = p / n_personas
        split = "train" if fraccion < 0.7 else ("val" if fraccion < 0.85 else "test")
        nivel_a, nivel_b = rng.normal(0, 0.5), rng.normal(0, 0.5)
        for s in range(n_sesiones):
            for orden, condicion in enumerate(ORDEN_PROTOCOLO, start=1):
                fila = dict(CAMPOS_IDENTIDAD)
                fila.update({
                    "archivo": f"P{p:02d}_M{s}_{condicion}.edf",
                    "sesion_uid": f"P{p:02d}_M{s}",
                    "persona_uid": f"P{p:02d}", "sujeto_id": f"P{p:02d}",
                    "medicion": f"M{s}", "condicion": condicion, "split": split,
                    "feat_a": nivel_a
                        + (separacion if condicion in ("STROOP", "MSIT") else 0.0)
                        + rng.normal(0, 1),
                    "feat_b": nivel_b
                        + (separacion if condicion in ("PASAT", "PVSAT") else 0.0)
                        + rng.normal(0, 1),
                    "ruido_1": rng.normal(), "ruido_2": rng.normal(),
                    "firma": nivel_a * 20 + rng.normal(0, 0.1),
                    "n_ventanas": _DURACION[condicion],
                    "n_canales_buenos": 12.0,
                    "n_validos_frontal": 5, "n_validos_frontocentral": 2,
                    "n_validos_temporal": 2, "n_validos_posterior": 4,
                    COLUMNA_FUGA: float(orden),
                })
                filas.append(fila)
    return pd.DataFrame(filas)


def _solo_tareas(matriz):
    return matriz[matriz["condicion"].isin(CONDICIONES_EJECUTIVAS)].reset_index(drop=True)


def _xyg(matriz, columnas=("feat_a", "feat_b", "ruido_1", "ruido_2")):
    return matriz[list(columnas)], etiqueta_multiclase(matriz), \
        matriz["persona_uid"].to_numpy()


# --------------------------------------------------------------------------
# Las condiciones y las constantes
# --------------------------------------------------------------------------

def test_condiciones_ejecutivas_son_las_cinco_tareas_sin_oa_ni_oc():
    assert CONDICIONES_EJECUTIVAS == ("STROOP", "PASAT", "PVTB", "MSIT", "PVSAT")
    assert not set(CONDICIONES_EJECUTIVAS) & {"OA", "OC"}
    assert set(CONDICIONES_EJECUTIVAS) < set(ORDEN_PROTOCOLO)


def test_codigo_de_clase_asigna_0_a_4_en_el_orden_declarado():
    assert CODIGO_DE_CLASE == {"STROOP": 0, "PASAT": 1, "PVTB": 2, "MSIT": 3,
                               "PVSAT": 4}


def test_similitud_cognitiva_cubre_los_diez_pares_y_es_simetrica():
    pares_esperados = set(itertools.combinations(CONDICIONES_EJECUTIVAS, 2))
    assert len(SIMILITUD_COGNITIVA) == 10
    for a, b in pares_esperados:
        v_ab, v_ba = similitud_cognitiva(a, b), similitud_cognitiva(b, a)
        assert v_ab == v_ba
        assert v_ab in (1, 2, 3)


def test_similitud_cognitiva_de_los_dos_pares_gemelos_es_maxima():
    """Los dos pares declarados a priori como más parecidos: PASAT-PVSAT (misma
    tarea, distinta modalidad) y STROOP-MSIT (las dos de interferencia)."""
    assert similitud_cognitiva("PASAT", "PVSAT") == 3
    assert similitud_cognitiva("STROOP", "MSIT") == 3


def test_similitud_cognitiva_de_pvtb_con_cualquiera_es_minima():
    for otra in ("STROOP", "PASAT", "MSIT", "PVSAT"):
        assert similitud_cognitiva("PVTB", otra) == 1


def test_similitud_cognitiva_falla_al_comparar_una_condicion_consigo_misma():
    with pytest.raises(ValueError, match="distintas"):
        similitud_cognitiva("STROOP", "STROOP")


def test_similitud_cognitiva_falla_con_una_condicion_ajena():
    with pytest.raises(ValueError, match="OA"):
        similitud_cognitiva("STROOP", "OA")


# --------------------------------------------------------------------------
# Guardianes de la fuga: qué entra a X
# --------------------------------------------------------------------------

def test_columnas_de_features_ejecutivas_excluye_condicion_orden():
    matriz = _matriz_ejecutivas()
    assert COLUMNA_FUGA in columnas_de_features(matriz), (
        "el escenario real: `columnas_de_features()` SÍ la devuelve"
    )
    assert COLUMNA_FUGA not in columnas_de_features_ejecutivas(matriz)


def test_un_modelo_con_condicion_orden_acierta_siempre():
    """
    Control positivo de que el peligro es real. Dentro de las 5 tareas
    ejecutivas, `condicion_orden` toma un valor constante DISTINTO en cada una
    (3, 4, 5, 6 y 7): es la etiqueta en otra escala, todavía más grosera que en
    el contraste binario (allí hacía falta un solo corte; aquí ni siquiera
    eso). Metida a mano da acierto perfecto por validación cruzada agrupada.
    """
    matriz = _solo_tareas(_matriz_ejecutivas())
    X, y, g = _xyg(matriz, columnas=(COLUMNA_FUGA,))
    r = evaluar_multiclase(X, y, g, modelo_logistico_multinomial())
    assert r["metricas"]["exactitud_balanceada"] == pytest.approx(1.0)


def test_condicion_orden_sigue_excluida_aunque_se_pidan_los_conteos():
    columnas = columnas_de_features_ejecutivas(_matriz_ejecutivas(),
                                               incluir_procesamiento=True)
    assert COLUMNA_FUGA not in columnas
    assert set(COLUMNAS_PROCESAMIENTO) <= set(columnas)


def test_por_defecto_quedan_fuera_los_seis_contadores_de_procesamiento():
    columnas = set(columnas_de_features_ejecutivas(_matriz_ejecutivas()))
    assert not columnas & set(COLUMNAS_PROCESAMIENTO)
    assert {"feat_a", "feat_b", "ruido_1", "firma"} <= columnas


def test_columnas_de_features_ejecutivas_coincide_exactamente_con_las_de_reposo():
    """Misma exclusión que en OA/OC y en reposo/tarea, y lo dice un test en vez
    de un comentario."""
    matriz = _matriz_ejecutivas()
    for incluir in (False, True):
        assert (columnas_de_features_ejecutivas(matriz, incluir_procesamiento=incluir)
                == columnas_de_features_reposo(matriz, incluir_procesamiento=incluir))


def test_columnas_de_features_ejecutivas_excluye_identidad_y_etiqueta():
    columnas = set(columnas_de_features_ejecutivas(_matriz_ejecutivas()))
    assert not columnas & set(COLUMNAS_IDENTIDAD)
    assert "vas" not in columnas
    assert "condicion" not in columnas


def test_con_el_esquema_de_la_fase_13_quedan_70_features():
    """
    Guardián de la documentación: ANALISIS_TAREAS_EJECUTIVAS.md habla de 77
    features de la Fase 13, 7 excluidas y 70 en el modelo. Si este test falla,
    esas cifras ya no corresponden al código y hay que reejecutar el análisis y
    actualizar el documento.
    """
    columnas = _esquema_fase_13()
    assert len(columnas) == 91
    matriz = pd.DataFrame(np.zeros((2, len(columnas))), columns=columnas)
    assert len(columnas_de_features(matriz)) == 77
    assert len(columnas_de_features_ejecutivas(matriz)) == 70
    assert len(columnas_de_features_ejecutivas(matriz, incluir_procesamiento=True)) == 76


# --------------------------------------------------------------------------
# Preparación de los datos
# --------------------------------------------------------------------------

def test_preparar_se_queda_solo_con_las_cinco_tareas_ejecutivas():
    matriz = _matriz_ejecutivas(n_personas=10)
    datos, meta = preparar_tareas_ejecutivas(matriz)
    assert set(datos["condicion"]) == set(CONDICIONES_EJECUTIVAS)
    assert "OA" not in set(datos["condicion"]) and "OC" not in set(datos["condicion"])
    assert set(meta["por_condicion"]) == set(CONDICIONES_EJECUTIVAS)


def test_preparar_se_niega_a_abrir_test_salvo_permiso():
    matriz = _matriz_ejecutivas()
    with pytest.raises(ValueError, match="test"):
        preparar_tareas_ejecutivas(matriz, splits=("train", "val", "test"))
    datos, meta = preparar_tareas_ejecutivas(matriz, splits=("test",), permitir_test=True)
    assert set(datos["split"]) == {"test"}
    assert meta["n_apartadas"] > 0


def test_preparar_por_defecto_no_deja_pasar_ninguna_fila_de_test():
    matriz = _matriz_ejecutivas()
    datos, meta = preparar_tareas_ejecutivas(matriz)
    assert "test" not in set(datos["split"])
    assert meta["splits"] == ["train", "val"]
    assert set(meta["por_split_apartado"]) == {"test"}
    assert meta["n_apartadas"] == meta["por_split_apartado"]["test"]


def test_preparar_descarta_las_grabaciones_sin_persona_uid_y_las_cuenta():
    matriz = _matriz_ejecutivas(n_personas=10)
    idx_tareas = matriz.index[matriz["condicion"].isin(CONDICIONES_EJECUTIVAS)][:5]
    matriz.loc[idx_tareas, "persona_uid"] = np.nan
    datos, meta = preparar_tareas_ejecutivas(matriz)
    assert datos["persona_uid"].notna().all()
    assert meta["n_sin_persona_uid"] == 5


def test_preparar_da_los_recuentos_y_las_clases_mayoritaria_y_minoritaria():
    matriz = _matriz_ejecutivas(n_personas=20, n_sesiones=3)
    datos, meta = preparar_tareas_ejecutivas(matriz)
    n_personas = 17  # 70 % train + 15 % val de 20
    assert meta["n_personas"] == n_personas
    assert meta["n_clases"] == 5
    for c in CONDICIONES_EJECUTIVAS:
        assert meta["por_condicion"][c] == n_personas * 3
    assert meta["ratio_desbalance"] == pytest.approx(1.0)
    assert meta["clase_mayoritaria"] in CONDICIONES_EJECUTIVAS
    assert meta["clase_minoritaria"] in CONDICIONES_EJECUTIVAS
    assert meta["n_grabaciones"] == n_personas * 3 * 5 == len(datos)


def test_preparar_no_modifica_la_matriz_de_entrada():
    matriz = _matriz_ejecutivas(n_personas=10)
    antes = matriz.copy()
    preparar_tareas_ejecutivas(matriz)
    pd.testing.assert_frame_equal(matriz, antes)


def test_preparar_rechaza_menos_de_dos_condiciones():
    with pytest.raises(ValueError, match="al menos 2"):
        preparar_tareas_ejecutivas(_matriz_ejecutivas(), condiciones=("STROOP",))


def test_preparar_rechaza_condiciones_ajenas_a_las_tareas_ejecutivas():
    """OA y OC no son tareas ejecutivas: aceptarlas aquí confundiría este
    análisis multiclase con el contraste binario de reposo/tarea."""
    with pytest.raises(ValueError, match="solo admite"):
        preparar_tareas_ejecutivas(_matriz_ejecutivas(), condiciones=("OA", "STROOP"))


def test_preparar_con_un_subconjunto_de_tareas_da_un_contraste_por_pares():
    datos, meta = preparar_tareas_ejecutivas(_matriz_ejecutivas(),
                                             condiciones=("STROOP", "MSIT"))
    assert set(datos["condicion"]) == {"STROOP", "MSIT"}
    assert meta["n_clases"] == 2


def test_preparar_falla_si_falta_alguna_clase_pedida():
    sin_pvtb = _matriz_ejecutivas()
    sin_pvtb = sin_pvtb[sin_pvtb["condicion"] != "PVTB"]
    with pytest.raises(ValueError, match="PVTB"):
        preparar_tareas_ejecutivas(sin_pvtb)


def test_preparar_falla_si_faltan_columnas():
    with pytest.raises(ValueError, match="persona_uid"):
        preparar_tareas_ejecutivas(_matriz_ejecutivas().drop(columns="persona_uid"))


# --------------------------------------------------------------------------
# La etiqueta
# --------------------------------------------------------------------------

def test_etiqueta_multiclase_da_codigos_0_a_4_en_el_orden_declarado():
    datos = pd.DataFrame({"condicion": list(CONDICIONES_EJECUTIVAS)})
    y = etiqueta_multiclase(datos)
    assert y.tolist() == [0, 1, 2, 3, 4]
    assert np.issubdtype(y.dtype, np.integer)
    assert len(y) == len(datos)


def test_etiqueta_multiclase_falla_con_oa_o_oc():
    with pytest.raises(ValueError, match="OA"):
        etiqueta_multiclase(pd.DataFrame({"condicion": ["STROOP", "OA"]}))


def test_etiqueta_multiclase_falla_con_una_condicion_ajena():
    with pytest.raises(ValueError, match="OTRA"):
        etiqueta_multiclase(pd.DataFrame({"condicion": ["STROOP", "OTRA"]}))


def test_etiqueta_multiclase_con_un_subconjunto_usa_ese_orden():
    datos = pd.DataFrame({"condicion": ["MSIT", "STROOP", "MSIT"]})
    y = etiqueta_multiclase(datos, condiciones=("MSIT", "STROOP"))
    assert y.tolist() == [0, 1, 0]


def test_nombres_de_clase_deshace_la_etiqueta():
    datos = pd.DataFrame({"condicion": ["MSIT", "STROOP", "PVSAT"]})
    y = etiqueta_multiclase(datos)
    assert nombres_de_clase(y).tolist() == ["MSIT", "STROOP", "PVSAT"]


def test_nombres_de_clase_falla_con_codigos_fuera_de_rango():
    with pytest.raises(ValueError, match="fuera de rango"):
        nombres_de_clase([0, 1, 5])


# --------------------------------------------------------------------------
# Modelos y desbalance
# --------------------------------------------------------------------------

def test_los_dos_modelos_pesan_las_clases():
    logistica = modelo_logistico_multinomial()[-1]
    bosque = modelo_bosque_multiclase()[-1]
    assert logistica.class_weight == "balanced"
    assert bosque.class_weight == "balanced"


def test_el_bosque_es_la_configuracion_de_las_fases_14_14b_17_y_18():
    bosque = modelo_bosque_multiclase()[-1]
    assert bosque.n_estimators == 500
    assert bosque.min_samples_leaf == 5
    assert bosque.random_state == 42


def test_los_modelos_imputan_y_tipifican_dentro_del_pipeline():
    """Imputación y tipificación se ajustan solo con el entrenamiento de cada
    pliegue porque van dentro del `Pipeline`, no antes de la CV."""
    nombres = [n for n, _ in modelo_logistico_multinomial().steps]
    assert nombres[:2] == ["simpleimputer", "standardscaler"]
    assert modelo_logistico_multinomial()[0].strategy == "median"
    assert modelo_bosque_multiclase()[0].strategy == "median"


def test_la_logistica_ajusta_un_coeficiente_por_clase():
    """Con 5 clases, `coef_` tiene una fila por clase (softmax), no una sola:
    es justo lo que exige `coeficientes_por_clase_y_pliegue`."""
    matriz = _solo_tareas(_matriz_ejecutivas(n_personas=10))
    X, y, _ = _xyg(matriz)
    modelo = modelo_logistico_multinomial().fit(X, y)
    assert modelo[-1].coef_.shape == (5, X.shape[1])


def test_los_modelos_admiten_nan_en_x():
    """La Fase 13 deja NaN en las regiones sin canales válidos."""
    matriz = _solo_tareas(_matriz_ejecutivas(n_personas=10))
    X, y, _ = _xyg(matriz)
    X = X.copy()
    X.iloc[::9, 0] = np.nan
    for modelo in (modelo_logistico_multinomial(),
                  modelo_bosque_multiclase(n_estimators=20, n_jobs=1)):
        ajustado = modelo.fit(X, y)
        assert not np.isnan(ajustado.predict(X)).any()


# --------------------------------------------------------------------------
# Métricas multiclase
# --------------------------------------------------------------------------

def test_metricas_de_un_predictor_constante_dan_exactamente_02_de_ba():
    """La razón de que la métrica principal sea la exactitud balanceada: con
    5 clases el azar es 1/5 = 0.2 exacto, pase lo que pase con el reparto."""
    y = np.tile(np.arange(5), 20)
    pred = np.zeros_like(y)
    m = metricas_multiclase(y, pred)
    assert m["exactitud_balanceada"] == pytest.approx(0.2)


def test_metricas_de_una_prediccion_perfecta():
    y = np.tile(np.arange(5), 20)
    m = metricas_multiclase(y, y)
    assert m["exactitud"] == 1.0
    assert m["exactitud_balanceada"] == 1.0
    assert m["f1_macro"] == 1.0
    assert m["kappa"] == 1.0
    for fila in m["por_clase"].values():
        assert fila["precision"] == fila["exhaustividad"] == fila["f1"] == 1.0


def test_por_clase_cuadra_con_el_soporte_real():
    y = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 4])
    m = metricas_multiclase(y, y.copy())
    soportes = {c: m["por_clase"][c]["soporte"] for c in CONDICIONES_EJECUTIVAS}
    assert soportes == {"STROOP": 3, "PASAT": 2, "PVTB": 4, "MSIT": 1, "PVSAT": 1}


def test_metricas_falla_con_etiquetas_fuera_de_rango():
    with pytest.raises(ValueError, match="códigos 0-4"):
        metricas_multiclase(np.array([0, 1, 5]), np.array([0, 1, 2]))


def test_metricas_falla_con_una_sola_clase():
    with pytest.raises(ValueError, match="una sola clase"):
        metricas_multiclase(np.zeros(5, dtype=int), np.zeros(5, dtype=int))


def test_metricas_valida_longitudes():
    with pytest.raises(ValueError, match="misma longitud"):
        metricas_multiclase(np.array([0, 1]), np.array([0]))


def test_metricas_por_pliegue_da_media_y_amontonada_que_casi_coinciden():
    matriz = _solo_tareas(_matriz_ejecutivas(n_personas=30, separacion=3.0))
    X, y, g = _xyg(matriz)
    r = evaluar_multiclase(X, y, g, modelo_logistico_multinomial())
    m = r["metricas"]
    assert m["n_pliegues"] == 5
    assert len(m["exactitud_balanceada_pliegues"]) == 5
    assert m["exactitud_balanceada_media"] == pytest.approx(
        m["exactitud_balanceada"], abs=0.08)


# --------------------------------------------------------------------------
# Matriz de confusión
# --------------------------------------------------------------------------

def test_matriz_de_confusion_normalizada_por_real_suma_1_por_fila():
    y = np.tile(np.arange(5), 20)
    pred = np.random.default_rng(0).integers(0, 5, len(y))
    cm = matriz_confusion_normalizada(y, pred)
    assert np.allclose(cm.sum(axis=1).to_numpy(), 1.0)
    assert list(cm.index) == list(CONDICIONES_EJECUTIVAS)
    assert list(cm.columns) == list(CONDICIONES_EJECUTIVAS)


def test_matriz_de_confusion_diagonal_es_la_exhaustividad():
    y = np.array([0, 0, 0, 1, 1, 2, 3, 4])
    pred = np.array([0, 0, 1, 1, 1, 2, 3, 4])
    cm = matriz_confusion_normalizada(y, pred)
    m = metricas_multiclase(y, pred)
    for i, c in enumerate(CONDICIONES_EJECUTIVAS):
        assert cm.iloc[i, i] == pytest.approx(m["por_clase"][c]["exhaustividad"])


def test_matriz_de_confusion_normalizada_por_predicha_suma_1_por_columna():
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    cm = matriz_confusion_normalizada(y, pred, normalizar="predicha")
    for j in (0, 1, 2):
        assert cm.iloc[:, j].sum() == pytest.approx(1.0)


def test_matriz_de_confusion_sin_normalizar_da_los_recuentos_brutos():
    y = np.array([0, 0, 1])
    pred = np.array([0, 1, 1])
    cm = matriz_confusion_normalizada(y, pred, normalizar=None)
    assert cm.loc["STROOP", "STROOP"] == 1
    assert cm.loc["STROOP", "PASAT"] == 1
    assert cm.loc["PASAT", "PASAT"] == 1


def test_matriz_de_confusion_clases_ausentes_en_y_pred_no_rompen_la_forma():
    y = np.array([0, 1, 2, 3, 4])
    pred = np.zeros(5, dtype=int)
    cm = matriz_confusion_normalizada(y, pred, normalizar=None)
    assert cm.shape == (5, 5)


def test_matriz_de_confusion_falla_con_normalizar_invalido():
    with pytest.raises(ValueError, match="normalizar"):
        matriz_confusion_normalizada(np.array([0, 1]), np.array([0, 1]), normalizar="mal")


# --------------------------------------------------------------------------
# Confusión por pares
# --------------------------------------------------------------------------

def _confusion_a_mano():
    """
    100 grabaciones (20 por clase), con una confusión construida a mano:
    STROOP <-> MSIT se confunden mucho (10 de cada 20, en las dos direcciones);
    PASAT <-> PVSAT algo (4 de cada 20); el resto, nada. Es justo el patrón que
    predice la similitud cognitiva declarada (los dos pares de similitud 3).
    """
    y = np.repeat(np.arange(5), 20)
    pred = y.copy()
    pred[0:10] = 3     # 10 STROOP (0) predichas MSIT (3)
    pred[70:80] = 0    # 10 MSIT (3) predichas STROOP (0)
    pred[20:24] = 4    # 4 PASAT (1) predichas PVSAT (4)
    pred[92:96] = 1    # 4 PVSAT (4) predichas PASAT (1)
    return y, pred


def test_confusion_por_pares_da_los_diez_pares():
    y, pred = _confusion_a_mano()
    pares = confusion_por_pares(y, pred)
    assert len(pares) == 10
    assert set(pares.index) == {
        f"{a}-{b}" for a, b in itertools.combinations(CONDICIONES_EJECUTIVAS, 2)}


def test_confusion_por_pares_da_la_cifra_exacta_del_caso_construido():
    y, pred = _confusion_a_mano()
    pares = confusion_por_pares(y, pred)
    assert pares.loc["STROOP-MSIT", "n_a_b"] == 10
    assert pares.loc["STROOP-MSIT", "n_b_a"] == 10
    assert pares.loc["STROOP-MSIT", "confusion_simetrica"] == pytest.approx(0.5)
    assert pares.loc["STROOP-MSIT", "asimetria"] == pytest.approx(0.0)
    assert pares.loc["PASAT-PVSAT", "confusion_simetrica"] == pytest.approx(0.2)
    assert pares.loc["STROOP-PASAT", "confusion_simetrica"] == pytest.approx(0.0)


def test_confusion_por_pares_lleva_la_similitud_y_la_distancia_declaradas():
    y, pred = _confusion_a_mano()
    pares = confusion_por_pares(y, pred)
    assert pares.loc["PASAT-PVSAT", "similitud_cognitiva"] == 3
    assert pares.loc["STROOP-MSIT", "similitud_cognitiva"] == 3
    assert pares.loc["STROOP-PVTB", "similitud_cognitiva"] == 1
    # Posiciones en el protocolo: STROOP=3, PASAT=4, PVTB=5, MSIT=6, PVSAT=7.
    assert pares.loc["STROOP-PVSAT", "distancia_protocolo"] == 4
    assert pares.loc["PASAT-PVTB", "distancia_protocolo"] == 1


def test_confusion_por_pares_asimetria_positiva_con_error_direccional():
    """Todas las STROOP mal clasificadas van a PASAT y ninguna PASAT a STROOP:
    el error va en una sola dirección."""
    y = np.array([0] * 10 + [1] * 10)
    pred = np.array([1] * 10 + [1] * 10)
    pares = confusion_por_pares(y, pred, condiciones=("STROOP", "PASAT"))
    assert pares.loc["STROOP-PASAT", "n_a_b"] == 10
    assert pares.loc["STROOP-PASAT", "n_b_a"] == 0
    assert pares.loc["STROOP-PASAT", "asimetria"] == pytest.approx(0.5)


def test_confusion_por_pares_valida_longitudes():
    with pytest.raises(ValueError, match="misma longitud"):
        confusion_por_pares(np.array([0, 1]), np.array([0]))


# --------------------------------------------------------------------------
# Distancia de duración
# --------------------------------------------------------------------------

def _datos_duracion():
    return pd.DataFrame({
        "condicion": (["STROOP"] * 3 + ["PASAT"] * 3 + ["PVTB"] * 3
                     + ["MSIT"] * 3 + ["PVSAT"] * 3),
        "n_ventanas": [60, 60, 60, 150, 150, 150, 150, 150, 150,
                      62, 58, 60, 148, 150, 152],
    })


def test_distancias_de_duracion_da_las_diferencias_absolutas_de_la_mediana():
    dur = distancias_de_duracion(_datos_duracion())
    assert dur.loc["STROOP-PASAT", "distancia_duracion"] == pytest.approx(90.0)
    assert dur.loc["PASAT-PVTB", "distancia_duracion"] == pytest.approx(0.0)
    assert dur.loc["STROOP-MSIT", "distancia_duracion"] == pytest.approx(0.0)
    assert dur.loc["PASAT-PVSAT", "distancia_duracion"] == pytest.approx(0.0)


def test_distancias_de_duracion_falla_si_falta_una_condicion():
    datos = pd.DataFrame({"condicion": ["STROOP", "PASAT"], "n_ventanas": [60, 150]})
    with pytest.raises(ValueError, match="PVTB"):
        distancias_de_duracion(datos)


def test_distancias_de_duracion_falla_si_faltan_columnas():
    with pytest.raises(ValueError, match="n_ventanas"):
        distancias_de_duracion(pd.DataFrame({"condicion": ["STROOP"]}))


# --------------------------------------------------------------------------
# Estructura de la confusión: qué predictor gana
# --------------------------------------------------------------------------

def test_estructura_de_la_confusion_detecta_que_sigue_la_similitud_cognitiva():
    """
    Con el caso construido a mano (`_confusion_a_mano`), la confusión sigue el
    parecido cognitivo declarado (los dos pares de similitud máxima son
    justamente los que más se confunden) más de lo que sigue la distancia en
    el protocolo.
    """
    y, pred = _confusion_a_mano()
    pares = confusion_por_pares(y, pred)
    est = estructura_de_la_confusion(pares)
    assert est["confusion_vs_similitud_cognitiva"] > 0.5
    assert (abs(est["confusion_vs_similitud_cognitiva"])
           > abs(est["confusion_vs_distancia_protocolo"]))


def test_estructura_de_la_confusion_puede_seguir_la_distancia_del_protocolo_y_el_criterio_falla():
    """
    Construcción deliberada de un caso donde la confusión es una función
    DECRECIENTE exacta de la distancia en el protocolo, sin relación con el
    parecido cognitivo: sirve para comprobar que el criterio «la confusión
    sigue más al parecido cognitivo que a la distancia en el protocolo» puede
    fallar. El test no está escrito para que ese criterio dé siempre «sí».
    """
    posicion = {c: ORDEN_PROTOCOLO.index(c) for c in CONDICIONES_EJECUTIVAS}
    filas = []
    for a, b in itertools.combinations(CONDICIONES_EJECUTIVAS, 2):
        d = abs(posicion[a] - posicion[b])
        filas.append({"par": f"{a}-{b}", "confusion_simetrica": 1.0 / d,
                      "similitud_cognitiva": similitud_cognitiva(a, b),
                      "distancia_protocolo": d})
    pares = pd.DataFrame(filas).set_index("par")
    est = estructura_de_la_confusion(pares)
    assert est["confusion_vs_distancia_protocolo"] == pytest.approx(1.0)
    assert est["confusion_vs_similitud_cognitiva"] < 0
    assert (abs(est["confusion_vs_similitud_cognitiva"])
           < abs(est["confusion_vs_distancia_protocolo"]))


def test_estructura_de_la_confusion_con_duraciones_incluye_las_correlaciones_extra():
    y, pred = _confusion_a_mano()
    pares = confusion_por_pares(y, pred)
    dur = distancias_de_duracion(_datos_duracion())
    est = estructura_de_la_confusion(pares, dur)
    assert "confusion_vs_distancia_duracion" in est
    assert "similitud_vs_distancia_duracion" in est
    assert "distancia_protocolo_vs_distancia_duracion" in est


def test_estructura_de_la_confusion_falla_si_duraciones_no_cubre_los_mismos_pares():
    y, pred = _confusion_a_mano()
    pares = confusion_por_pares(y, pred)
    incompleta = pd.DataFrame({"distancia_duracion": [1.0]}, index=["STROOP-PASAT"])
    with pytest.raises(ValueError, match="mismos pares"):
        estructura_de_la_confusion(pares, incompleta)


def test_estructura_de_la_confusion_falla_con_una_variable_constante():
    pares = pd.DataFrame({
        "confusion_simetrica": [0.5] * 10,
        "similitud_cognitiva": [2] * 5 + [3] * 5,
        "distancia_protocolo": [1] * 10,
    }, index=[f"p{i}" for i in range(10)])
    with pytest.raises(ValueError, match="constante"):
        estructura_de_la_confusion(pares)


# --------------------------------------------------------------------------
# Separabilidad por pares (uno contra uno)
# --------------------------------------------------------------------------

def test_separabilidad_por_pares_da_las_diez_filas_consistentes():
    matriz = _solo_tareas(_matriz_ejecutivas(n_personas=20, separacion=3.0))
    X, y, g = _xyg(matriz)
    sep = separabilidad_por_pares(X, y, g, modelo_logistico_multinomial,
                                  n_replicas=50)
    assert len(sep) == 10
    assert (sep["n_a"] + sep["n_b"] == sep["n"]).all()
    assert set(sep.index) == {
        f"{a}-{b}" for a, b in itertools.combinations(CONDICIONES_EJECUTIVAS, 2)}


def test_separabilidad_por_pares_da_intervalos_que_contienen_la_estimacion():
    matriz = _solo_tareas(_matriz_ejecutivas(n_personas=20, separacion=3.0))
    X, y, g = _xyg(matriz)
    sep = separabilidad_por_pares(X, y, g, modelo_logistico_multinomial,
                                  n_replicas=100)
    for _, fila in sep.iterrows():
        lo, hi = fila["ic95"]
        assert lo <= fila["exactitud_balanceada"] + 1e-9 <= hi


def test_separabilidad_por_pares_agrupa_por_persona():
    """Con una feature que solo identifica a la persona (`firma`), ningún par
    debería separar por encima del azar en personas nuevas."""
    matriz = _solo_tareas(_matriz_ejecutivas(n_personas=30, n_sesiones=4))
    X, y, g = _xyg(matriz, columnas=("firma",))
    sep = separabilidad_por_pares(X, y, g, modelo_logistico_multinomial,
                                  n_replicas=50)
    assert (sep["exactitud_balanceada"] - 0.5).abs().max() < 0.15


def test_separabilidad_por_pares_distingue_un_par_separable_de_uno_no_separable():
    matriz = _solo_tareas(_matriz_ejecutivas(n_personas=30, separacion=3.0))
    X, y, g = _xyg(matriz)
    sep = separabilidad_por_pares(X, y, g, modelo_logistico_multinomial,
                                  n_replicas=100)
    # STROOP y PASAT saltan en features DISTINTAS (feat_a y feat_b): par fácil.
    # STROOP y MSIT comparten el mismo salto en feat_a: par difícil.
    assert (sep.loc["STROOP-PASAT", "exactitud_balanceada"]
           > sep.loc["STROOP-MSIT", "exactitud_balanceada"])


# --------------------------------------------------------------------------
# Coeficientes por clase y pliegue
# --------------------------------------------------------------------------

def test_coeficientes_por_clase_y_pliegue_da_una_entrada_por_clase_con_forma_correcta():
    matriz = _solo_tareas(_matriz_ejecutivas(n_personas=20, separacion=3.0))
    X, y, g = _xyg(matriz)
    coef = coeficientes_por_clase_y_pliegue(X, y, g, modelo_logistico_multinomial(),
                                            n_splits=5)
    assert set(coef) == set(CONDICIONES_EJECUTIVAS)
    for tabla in coef.values():
        assert tabla.shape == (X.shape[1], 5)
        assert list(tabla.index) == list(X.columns)


def test_coeficientes_por_clase_y_pliegue_falla_con_un_modelo_no_multinomial():
    """Un modelo con un solo vector de coeficientes ajustado sobre 5 clases no
    puede dar la forma esperada; el mensaje debe ser claro."""
    matriz = _solo_tareas(_matriz_ejecutivas(n_personas=10))
    X, y, g = _xyg(matriz)

    class ModeloDeUnSoloCoeficiente(BaseEstimator):
        def fit(self, X, y):
            self.coef_ = np.zeros((1, X.shape[1]))
            return self

    with pytest.raises(ValueError, match="forma"):
        coeficientes_por_clase_y_pliegue(X, y, g, ModeloDeUnSoloCoeficiente(),
                                         n_splits=5)


def test_coeficiente_neto_por_clase_suma_solo_las_media_del_bloque_por_clase():
    coef = {
        "STROOP": pd.DataFrame(
            {"pliegue_1": [1.0, 2.0, 100.0]},
            index=["frontal_theta_rel_media", "frontal_theta_logabs_media",
                  "frontal_theta_rel_sd"]),
        "PASAT": pd.DataFrame(
            {"pliegue_1": [-1.0, 0.5, -100.0]},
            index=["frontal_theta_rel_media", "frontal_theta_logabs_media",
                  "frontal_theta_rel_sd"]),
    }
    neto = coeficiente_neto_por_clase(coef, "frontal", "theta")
    assert neto.loc["STROOP", "pliegue_1"] == pytest.approx(3.0)
    assert neto.loc["PASAT", "pliegue_1"] == pytest.approx(-0.5)


def test_coeficiente_neto_por_clase_falla_si_el_bloque_no_esta():
    coef = {"STROOP": pd.DataFrame({"pliegue_1": [1.0]}, index=["ruido_1"])}
    with pytest.raises(ValueError, match="frontal/theta"):
        coeficiente_neto_por_clase(coef, "frontal", "theta")


# --------------------------------------------------------------------------
# d de Cohen una-contra-resto
# --------------------------------------------------------------------------

def test_tamanos_de_efecto_una_contra_resto_da_una_serie_por_clase():
    matriz = _solo_tareas(_matriz_ejecutivas(n_personas=20, separacion=3.0))
    X, y, g = _xyg(matriz)
    efectos = tamanos_de_efecto_una_contra_resto(X, y)
    assert set(efectos) == set(CONDICIONES_EJECUTIVAS)
    for serie in efectos.values():
        assert list(serie.index) == list(X.columns)


def test_tamanos_de_efecto_una_contra_resto_es_positivo_para_la_feature_de_esa_clase():
    """`feat_a` sube en STROOP y en MSIT: el d de Cohen una-contra-resto de
    `feat_a` debe ser positivo para esas dos clases."""
    matriz = _solo_tareas(_matriz_ejecutivas(n_personas=30, separacion=3.0))
    X, y, g = _xyg(matriz)
    efectos = tamanos_de_efecto_una_contra_resto(X, y)
    assert efectos["STROOP"]["feat_a"] > 0.3
    assert efectos["MSIT"]["feat_a"] > 0.3
    assert efectos["PVTB"]["feat_a"] < 0.1


# --------------------------------------------------------------------------
# Integración: la CV agrupada y el control negativo
# --------------------------------------------------------------------------

def test_con_una_señal_clara_el_modelo_separa_las_tareas_en_personas_nuevas():
    """
    `feat_a` y `feat_b` solo separan 3 grupos (STROOP=MSIT, PASAT=PVSAT, PVTB),
    no las 5 clases entre sí a propósito (para que `separabilidad_por_pares`
    pueda distinguir un par fácil de uno difícil): dentro de cada grupo de dos
    el modelo no tiene con qué distinguirlas y su exhaustividad cae al azar
    condicional. Con todo, 0.2 (azar de 5 clases) queda muy por debajo.
    """
    matriz = _solo_tareas(_matriz_ejecutivas(n_personas=30, separacion=3.0))
    X, y, g = _xyg(matriz)
    r = evaluar_multiclase(X, y, g, modelo_logistico_multinomial())
    assert r["metricas"]["exactitud_balanceada"] > 0.45


def test_el_bosque_tambien_separa_y_admite_la_misma_evaluacion():
    matriz = _solo_tareas(_matriz_ejecutivas(n_personas=30, separacion=3.0))
    X, y, g = _xyg(matriz)
    r = evaluar_multiclase(X, y, g, modelo_bosque_multiclase(n_estimators=60, n_jobs=1))
    assert r["metricas"]["exactitud_balanceada"] > 0.45


def test_con_las_etiquetas_barajadas_dentro_de_persona_la_cv_devuelve_el_azar():
    """
    El control negativo del análisis, con 5 clases: el azar de la exactitud
    balanceada es 1/5 = 0.2, no 0.5. Con las etiquetas verdaderas la BA es muy
    superior; con las etiquetas barajadas dentro de cada persona (conservando
    cuántas grabaciones de cada clase tiene cada una) cae a ~0.2. Si diera más,
    habría una fuga en el propio montaje y ningún resultado posterior valdría.
    """
    matriz = _solo_tareas(_matriz_ejecutivas(n_personas=40, n_sesiones=4, separacion=3.0))
    X, y, g = _xyg(matriz)
    real = evaluar_multiclase(X, y, g, modelo_logistico_multinomial())["metricas"]
    nulas = [
        evaluar_multiclase(
            X, permutar_etiquetas_dentro_de_persona(y, g, semilla=s), g,
            modelo_logistico_multinomial())["metricas"]["exactitud_balanceada"]
        for s in range(5)
    ]
    assert real["exactitud_balanceada"] > 0.4
    assert abs(np.mean(nulas) - 0.2) < 0.08
    assert max(nulas) < 0.35


def test_agrupar_por_persona_impide_que_el_modelo_se_calibre_con_la_propia_persona():
    """
    Réplica multiclase del test que justifica agrupar por persona en OA/OC. La
    feature es un nivel propio de cada persona (muy espaciado entre personas)
    más un salto pequeño por clase: con las otras grabaciones de la persona en
    entrenamiento, un modelo aprende su nivel y separa las cinco clases dentro
    de ella; con personas nuevas ese nivel no ayuda. La diferencia entre ambas
    validaciones es la fuga que `persona_uid` cierra.
    """
    rng = np.random.default_rng(5)
    filas = []
    for p in range(40):
        nivel = rng.uniform(0, 3000)
        for s in range(4):
            for clase in range(5):
                filas.append({"persona_uid": f"P{p:02d}", "y": clase,
                              "x": nivel + 1.0 * clase + rng.normal(0, 0.15)})
    datos = pd.DataFrame(filas)
    X, y, g = datos[["x"]], datos["y"].to_numpy(), datos["persona_uid"].to_numpy()
    bosque = RandomForestClassifier(n_estimators=100, min_samples_leaf=1,
                                    random_state=0)

    agrupada = evaluar_multiclase(X, y, g, bosque)["metricas"]["exactitud_balanceada"]

    pred = np.full(len(y), -1)
    for tr, va in KFold(5, shuffle=True, random_state=0).split(X):
        pred[va] = clone(bosque).fit(X.iloc[tr], y[tr]).predict(X.iloc[va])
    ingenua = balanced_accuracy_score(y, pred)

    assert ingenua > 0.85, "sin agrupar, calibrarse con la propia persona funciona"
    assert agrupada < 0.5, "agrupando, ese nivel personal no sirve"
    assert ingenua - agrupada > 0.25


def test_las_funciones_nuevas_estan_en_la_api_del_paquete():
    import eeg_library
    for nombre in ("columnas_de_features_ejecutivas", "confusion_por_pares",
                   "etiqueta_multiclase", "preparar_tareas_ejecutivas"):
        assert callable(getattr(eeg_library, nombre))
