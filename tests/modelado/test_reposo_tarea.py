"""
Tests de `modelado/reposo_tarea.py`: reposo (OA, OC) frente a tarea ejecutiva.

Protegen los invariantes que hacen que el resultado de ese análisis signifique
algo:

1. **`condicion_orden` nunca entra a X.** Vale 1-2 en el reposo y 3-7 en las
   tareas: con ella el acierto sería del 100 % por construcción.
2. **La etiqueta no se puede invertir por accidente.** Cada conjunto de
   condiciones solo admite las de su familia, y 1 significa siempre «tarea».
3. **El conjunto de test no se abre** sin pedirlo explícitamente.
4. **El desbalance de clases está tratado**: el entrenamiento pesa las clases y
   las métricas no premian predecir siempre la mayoritaria.
5. **El desglose por condición cuadra con el total** y sus intervalos remuestrean
   personas, no filas.
6. **Los controles funcionan**: con las etiquetas barajadas dentro de cada
   persona, la CV devuelve el azar.

Todo con datos sintéticos en memoria: ningún test toca disco.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_library.modelado.reposo import (
    COLUMNA_FUGA,
    COLUMNAS_PROCESAMIENTO,
    CONDICIONES_REPOSO,
    columnas_de_features_reposo,
    evaluar_reposo,
    metricas_binarias,
    permutar_etiquetas_dentro_de_persona,
    pliegue_por_fila,
    umbral_una_variable,
)
from eeg_library.modelado.reposo_tarea import (
    CONDICIONES_TAREA,
    coeficiente_neto_por_pliegue,
    columnas_de_features_tarea,
    columnas_media_del_bloque,
    comparar_condiciones,
    desglose_por_condicion,
    etiqueta_tarea,
    modelo_bosque,
    modelo_logistico,
    precision_media,
    preparar_reposo_tarea,
    signos_umbral_por_pliegue,
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


def _esquema_fase_13():
    """Los 91 nombres de columna reales de `matriz_rf.parquet`."""
    features = [f"{r}_{b}_{t}_{s}" for r in REGIONES for b in BANDAS_DEFECTO
                for t in ("logabs", "rel") for s in ("media", "sd")]
    features += ["theta_frontal_log_media", "theta_frontal_log_sd",
                 "alfa_posterior_log_media", "alfa_posterior_log_sd",
                 "indice_fatiga_media", "indice_fatiga_sd"]
    features += list(COLUMNAS_PROCESAMIENTO) + [COLUMNA_FUGA]
    return list(COLUMNAS_IDENTIDAD) + ["vas"] + features


def _matriz_tarea(n_personas=20, n_sesiones=3, separacion=3.0, semilla=0):
    """
    Matriz por grabación mínima con la forma de la Fase 13, las 7 condiciones.

    Cada persona tiene un nivel propio de `frontal_theta_rel_media` (su
    «firma», sd 0.5) y, en las cinco tareas, un salto de `separacion` sobre el
    reposo. Hay 2 grabaciones de reposo por 5 de tarea en cada sesión (1:2.5,
    como en los datos reales). Las features `ruido_*` no dependen de la
    condición y `firma` identifica a la persona sin decir nada sobre la
    condición. Los splits van por persona: 70 % train, 15 % val, 15 % test.
    """
    rng = np.random.default_rng(semilla)
    filas = []
    for p in range(n_personas):
        fraccion = p / n_personas
        split = "train" if fraccion < 0.7 else ("val" if fraccion < 0.85 else "test")
        nivel = rng.normal(0, 0.5)
        for s in range(n_sesiones):
            for orden, condicion in enumerate(ORDEN_PROTOCOLO, start=1):
                fila = dict(CAMPOS_IDENTIDAD)
                fila.update({
                    "archivo": f"P{p:02d}_M{s}_{condicion}.edf",
                    "sesion_uid": f"P{p:02d}_M{s}",
                    "persona_uid": f"P{p:02d}", "sujeto_id": f"P{p:02d}",
                    "medicion": f"M{s}", "condicion": condicion, "split": split,
                    "frontal_theta_rel_media":
                        nivel + (separacion if condicion in CONDICIONES_TAREA else 0.0)
                        + rng.normal(0, 1),
                    "ruido_1": rng.normal(), "ruido_2": rng.normal(),
                    "firma": nivel * 20 + rng.normal(0, 0.1),
                    "n_ventanas": 55.0, "n_canales_buenos": 12.0,
                    "n_validos_frontal": 5, "n_validos_frontocentral": 2,
                    "n_validos_temporal": 2, "n_validos_posterior": 4,
                    COLUMNA_FUGA: float(orden),
                })
                filas.append(fila)
    return pd.DataFrame(filas)


def _xyg(matriz, columnas=("frontal_theta_rel_media", "ruido_1", "ruido_2")):
    return matriz[list(columnas)], etiqueta_tarea(matriz), \
        matriz["persona_uid"].to_numpy()


# --------------------------------------------------------------------------
# Las condiciones
# --------------------------------------------------------------------------

def test_reposo_y_tarea_particionan_el_protocolo():
    """Si alguien añade una condición al protocolo, este análisis tiene que
    enterarse: quedaría sin etiqueta."""
    assert not set(CONDICIONES_REPOSO) & set(CONDICIONES_TAREA)
    assert set(CONDICIONES_REPOSO) | set(CONDICIONES_TAREA) == set(ORDEN_PROTOCOLO)
    assert set(CONDICIONES_REPOSO) == {"OA", "OC"}
    assert set(CONDICIONES_TAREA) == {"STROOP", "PASAT", "PVTB", "MSIT", "PVSAT"}


def test_el_reposo_ocupa_las_posiciones_1_y_2_y_las_tareas_de_la_3_a_la_7():
    """La razón por la que `condicion_orden` es una fuga total y por la que el
    orden fijo del protocolo confunde la condición con la deriva de sesión."""
    posicion = {c: i for i, c in enumerate(ORDEN_PROTOCOLO, start=1)}
    assert {posicion[c] for c in CONDICIONES_REPOSO} == {1, 2}
    assert {posicion[c] for c in CONDICIONES_TAREA} == {3, 4, 5, 6, 7}


# --------------------------------------------------------------------------
# Guardianes de la fuga: qué entra a X
# --------------------------------------------------------------------------

def test_columnas_de_features_tarea_excluye_condicion_orden():
    """
    El test central. `condicion_orden` separa reposo (1-2) de tarea (3-7) con un
    solo corte: es la etiqueta con otro nombre. Si alguna vez llega a X, el
    acierto se dispara al 100 % y el resultado entero deja de significar algo.
    """
    matriz = _matriz_tarea()
    assert COLUMNA_FUGA in columnas_de_features(matriz), (
        "el escenario real: `columnas_de_features()` SÍ la devuelve"
    )
    assert COLUMNA_FUGA not in columnas_de_features_tarea(matriz)


def test_condicion_orden_sigue_excluida_aunque_se_pidan_los_conteos():
    columnas = columnas_de_features_tarea(_matriz_tarea(),
                                          incluir_procesamiento=True)
    assert COLUMNA_FUGA not in columnas
    assert set(COLUMNAS_PROCESAMIENTO) <= set(columnas)


def test_por_defecto_quedan_fuera_los_seis_contadores_de_procesamiento():
    columnas = set(columnas_de_features_tarea(_matriz_tarea()))
    assert not columnas & set(COLUMNAS_PROCESAMIENTO)
    assert {"frontal_theta_rel_media", "ruido_1", "firma"} <= columnas


def test_columnas_de_features_tarea_coincide_con_la_de_reposo():
    """Misma exclusión que en OA/OC, y lo dice un test en vez de un comentario."""
    matriz = _matriz_tarea()
    for incluir in (False, True):
        assert (columnas_de_features_tarea(matriz, incluir_procesamiento=incluir)
                == columnas_de_features_reposo(matriz, incluir_procesamiento=incluir))


def test_columnas_de_features_tarea_excluye_identidad_y_etiqueta():
    columnas = set(columnas_de_features_tarea(_matriz_tarea()))
    assert not columnas & set(COLUMNAS_IDENTIDAD)
    assert "vas" not in columnas
    assert "condicion" not in columnas


def test_con_el_esquema_de_la_fase_13_quedan_70_features():
    """
    Guardián de la documentación: ANALISIS_REPOSO_TAREA.md habla de 77 features
    de la Fase 13, 7 excluidas y 70 en el modelo. Si este test falla, esas
    cifras ya no corresponden al código y hay que reejecutar el análisis y
    actualizar el documento.
    """
    columnas = _esquema_fase_13()
    assert len(columnas) == 91
    matriz = pd.DataFrame(np.zeros((2, len(columnas))), columns=columnas)
    assert len(columnas_de_features(matriz)) == 77
    assert len(columnas_de_features_tarea(matriz)) == 70
    assert len(columnas_de_features_tarea(matriz, incluir_procesamiento=True)) == 76


def test_un_modelo_con_condicion_orden_acierta_siempre():
    """
    Control positivo de que el peligro es real: metida a mano, la fuga da el 100 %
    por validación cruzada agrupada y todo. Es lo que ocurriría si la exclusión
    se rompiera sin que nadie lo notara.
    """
    matriz = _matriz_tarea()
    X, y, g = _xyg(matriz, columnas=(COLUMNA_FUGA,))
    r = evaluar_reposo(X, y, g, modelo_logistico())
    assert r["metricas"]["exactitud_balanceada"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Preparación de los datos
# --------------------------------------------------------------------------

def test_preparar_se_queda_con_las_siete_condiciones():
    matriz = _matriz_tarea(n_personas=10)
    ajena = matriz.iloc[:4].copy()
    ajena["condicion"] = "OTRA"
    datos, meta = preparar_reposo_tarea(pd.concat([matriz, ajena], ignore_index=True))
    assert set(datos["condicion"]) == set(ORDEN_PROTOCOLO)
    assert set(meta["por_condicion"]) == set(ORDEN_PROTOCOLO)


def test_preparar_se_niega_a_abrir_test_salvo_permiso():
    matriz = _matriz_tarea()
    with pytest.raises(ValueError, match="test"):
        preparar_reposo_tarea(matriz, splits=("train", "val", "test"))
    datos, meta = preparar_reposo_tarea(matriz, splits=("test",), permitir_test=True)
    assert set(datos["split"]) == {"test"}
    assert meta["n_apartadas"] > 0


def test_preparar_por_defecto_no_deja_pasar_ninguna_fila_de_test():
    matriz = _matriz_tarea()
    datos, meta = preparar_reposo_tarea(matriz)
    assert "test" not in set(datos["split"])
    assert meta["splits"] == ["train", "val"]
    assert set(meta["por_split_apartado"]) == {"test"}
    assert meta["n_apartadas"] == meta["por_split_apartado"]["test"]
    assert meta["n_personas_apartadas"] == 3   # 15 % de 20 personas


def test_preparar_descarta_las_grabaciones_sin_persona_uid_y_las_cuenta():
    matriz = _matriz_tarea(n_personas=10)
    matriz.loc[matriz.index[:5], "persona_uid"] = np.nan
    n_train_val = int(matriz["split"].isin(["train", "val"]).sum())
    datos, meta = preparar_reposo_tarea(matriz)
    assert datos["persona_uid"].notna().all()
    assert meta["n_sin_persona_uid"] == 5
    assert meta["n_grabaciones"] == n_train_val - 5


def test_preparar_da_los_recuentos_y_el_desbalance_de_las_clases():
    matriz = _matriz_tarea(n_personas=20, n_sesiones=3)
    datos, meta = preparar_reposo_tarea(matriz)
    n_personas = 17   # 70 % train + 15 % val de 20
    assert meta["n_personas"] == n_personas
    assert meta["n_sesiones"] == n_personas * 3
    assert meta["n_reposo"] == n_personas * 3 * 2
    assert meta["n_tarea"] == n_personas * 3 * 5
    assert meta["n_reposo"] + meta["n_tarea"] == meta["n_grabaciones"] == len(datos)
    assert meta["ratio_desbalance"] == pytest.approx(2.5)
    assert meta["clase_mayoritaria"] == "tarea"
    assert meta["por_condicion"]["OC"] == n_personas * 3
    assert meta["por_split"] == {"train": 14 * 3 * 7, "val": 3 * 3 * 7}


def test_preparar_no_modifica_la_matriz_de_entrada():
    matriz = _matriz_tarea(n_personas=10)
    antes = matriz.copy()
    preparar_reposo_tarea(matriz)
    pd.testing.assert_frame_equal(matriz, antes)


def test_preparar_sin_oc_deja_fuera_oc():
    """El control «solo OA frente a tareas»: si OC se colara, el control no
    separaría el efecto Berger de la carga cognitiva."""
    datos, meta = preparar_reposo_tarea(_matriz_tarea(),
                                        condiciones_reposo=("OA",))
    assert "OC" not in set(datos["condicion"])
    assert set(datos["condicion"]) == {"OA"} | set(CONDICIONES_TAREA)
    assert meta["condiciones_reposo"] == ["OA"]
    assert meta["ratio_desbalance"] == pytest.approx(5.0)   # 1 OA por 5 tareas


def test_preparar_con_una_sola_tarea_da_un_contraste_por_pares_equilibrado():
    """OA (posición 2) frente a STROOP (posición 3): el par adyacente."""
    datos, meta = preparar_reposo_tarea(_matriz_tarea(),
                                        condiciones_reposo=("OA",),
                                        condiciones_tarea=("STROOP",))
    assert set(datos["condicion"]) == {"OA", "STROOP"}
    assert meta["n_reposo"] == meta["n_tarea"]
    assert meta["ratio_desbalance"] == pytest.approx(1.0)


@pytest.mark.parametrize("reposo, tarea", [
    (("STROOP",), CONDICIONES_TAREA),          # una tarea como «reposo»
    (CONDICIONES_REPOSO, ("OA",)),             # el reposo como «tarea»
    (CONDICIONES_REPOSO, ("OA", "STROOP")),
    (("OTRA",), CONDICIONES_TAREA),            # condición inexistente
])
def test_preparar_rechaza_condiciones_de_la_otra_familia(reposo, tarea):
    """Aceptarlas invertiría la etiqueta sin ningún error, y una etiqueta
    invertida da resultados que parecen razonables."""
    with pytest.raises(ValueError, match="solo admite"):
        preparar_reposo_tarea(_matriz_tarea(), condiciones_reposo=reposo,
                              condiciones_tarea=tarea)


def test_preparar_rechaza_conjuntos_vacios():
    with pytest.raises(ValueError, match="vacío"):
        preparar_reposo_tarea(_matriz_tarea(), condiciones_reposo=())
    with pytest.raises(ValueError, match="vacío"):
        preparar_reposo_tarea(_matriz_tarea(), condiciones_tarea=())


def test_preparar_falla_si_una_de_las_clases_queda_vacia():
    solo_tareas = _matriz_tarea()
    solo_tareas = solo_tareas[solo_tareas["condicion"].isin(CONDICIONES_TAREA)]
    with pytest.raises(ValueError, match="hacen falta de las dos clases"):
        preparar_reposo_tarea(solo_tareas)


def test_preparar_falla_si_faltan_columnas():
    with pytest.raises(ValueError, match="persona_uid"):
        preparar_reposo_tarea(_matriz_tarea().drop(columns="persona_uid"))


# --------------------------------------------------------------------------
# La etiqueta
# --------------------------------------------------------------------------

def test_etiqueta_tarea_es_1_para_las_tareas_y_0_para_el_reposo():
    datos = pd.DataFrame({"condicion": list(ORDEN_PROTOCOLO)})
    y = etiqueta_tarea(datos)
    esperado = [0, 0, 1, 1, 1, 1, 1]   # OC, OA, STROOP, PASAT, PVTB, MSIT, PVSAT
    assert y.tolist() == esperado
    assert np.issubdtype(y.dtype, np.integer)
    assert len(y) == len(datos)


def test_etiqueta_tarea_falla_con_una_condicion_ajena():
    with pytest.raises(ValueError, match="OTRA"):
        etiqueta_tarea(pd.DataFrame({"condicion": ["OA", "STROOP", "OTRA"]}))


def test_etiqueta_tarea_falla_con_una_condicion_fuera_del_subconjunto_pedido():
    """Con el control sin OC, una OC que llegara a la etiqueta sería un error
    silencioso; tiene que ser un error a voces."""
    datos = pd.DataFrame({"condicion": ["OA", "OC", "STROOP"]})
    with pytest.raises(ValueError, match="OC"):
        etiqueta_tarea(datos, condiciones_reposo=("OA",))


def test_etiqueta_tarea_con_subconjuntos_da_el_mismo_significado():
    datos = pd.DataFrame({"condicion": ["OA", "STROOP", "OA", "STROOP"]})
    y = etiqueta_tarea(datos, condiciones_reposo=("OA",),
                       condiciones_tarea=("STROOP",))
    assert y.tolist() == [0, 1, 0, 1]


# --------------------------------------------------------------------------
# Modelos: el desbalance en el entrenamiento
# --------------------------------------------------------------------------

def test_los_dos_modelos_pesan_las_clases():
    logistica = modelo_logistico()[-1]
    bosque = modelo_bosque()[-1]
    assert logistica.class_weight == "balanced"
    assert bosque.class_weight == "balanced"


def test_el_bosque_es_la_configuracion_de_las_fases_14_14b_y_17():
    bosque = modelo_bosque()[-1]
    assert bosque.n_estimators == 500
    assert bosque.min_samples_leaf == 5
    assert bosque.random_state == 42


def test_los_modelos_imputan_y_tipifican_dentro_del_pipeline():
    """Imputación y tipificación se ajustan solo con el entrenamiento de cada
    pliegue porque van dentro del `Pipeline`, no antes de la CV."""
    nombres = [n for n, _ in modelo_logistico().steps]
    assert nombres[:2] == ["simpleimputer", "standardscaler"]
    assert modelo_logistico()[0].strategy == "median"
    assert modelo_bosque()[0].strategy == "median"


def test_los_modelos_admiten_nan_en_x():
    """La Fase 13 deja NaN en las regiones sin canales válidos (33-96 filas)."""
    matriz = _matriz_tarea(n_personas=10)
    X, y, _ = _xyg(matriz)
    X = X.copy()
    X.iloc[::9, 0] = np.nan
    for modelo in (modelo_logistico(), modelo_bosque(n_estimators=20, n_jobs=1)):
        ajustado = modelo.fit(X, y)
        assert not np.isnan(ajustado.predict_proba(X)).any()


def test_pesar_las_clases_sube_la_deteccion_de_la_clase_minoritaria():
    """
    Con una señal débil y 1:2.5, una logística sin pesos se recuesta sobre la
    clase mayoritaria (tarea) y detecta poco reposo; con `balanced` reparte los
    errores. Es la razón de pesar las clases en el entrenamiento.
    """
    matriz = _matriz_tarea(n_personas=40, n_sesiones=4, separacion=0.5, semilla=3)
    X, y, g = _xyg(matriz)
    sin_pesos = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(max_iter=2000))
    con = evaluar_reposo(X, y, g, modelo_logistico())["metricas"]
    sin = evaluar_reposo(X, y, g, sin_pesos)["metricas"]
    # especificidad = fracción de reposo (clase 0) bien detectada
    assert con["especificidad"] > sin["especificidad"] + 0.10
    assert con["exactitud_balanceada"] > sin["exactitud_balanceada"]


# --------------------------------------------------------------------------
# Métricas con clases desbalanceadas
# --------------------------------------------------------------------------

def test_el_baseline_mayoritario_da_exactitud_balanceada_05_con_exactitud_alta():
    """
    La razón de que la métrica principal sea la exactitud balanceada: con 1:2.5,
    predecir siempre «tarea» acierta el 71 % y no sabe nada.
    """
    matriz = _matriz_tarea(n_personas=30)
    X, y, g = _xyg(matriz)
    r = evaluar_reposo(X, y, g, DummyClassifier(strategy="most_frequent"))
    m = r["metricas"]
    assert m["exactitud_balanceada"] == pytest.approx(0.5)
    assert m["exactitud"] == pytest.approx(5 / 7)
    assert m["especificidad"] == 0.0      # ningún reposo detectado
    assert m["sensibilidad"] == 1.0       # todas las tareas «detectadas»


def test_precision_media_es_1_con_una_puntuacion_perfecta_para_las_dos_clases():
    y = np.array([0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.95])   # P(tarea)
    assert precision_media(y, p, clase_positiva=0) == pytest.approx(1.0)
    assert precision_media(y, p, clase_positiva=1) == pytest.approx(1.0)


def test_precision_media_por_defecto_puntua_el_reposo():
    y = np.array([0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.95])
    assert precision_media(y, p) == precision_media(y, p, clase_positiva=0)


def test_precision_media_de_una_puntuacion_invertida_es_baja():
    """Si P(tarea) es alta justo en el reposo, la clase reposo queda al final de
    la lista: la precisión media cae muy por debajo de su prevalencia."""
    y = np.array([0, 0, 1, 1, 1, 1])
    p = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.05])
    assert precision_media(y, p, clase_positiva=0) < 0.5 * (2 / 6) + 0.2


def test_precision_media_al_azar_es_la_prevalencia_de_la_clase_puntuada():
    """No es 0.5: por eso hay que compararla con la prevalencia, 2/7 = 0.29."""
    rng = np.random.default_rng(0)
    y = np.tile([0, 0, 1, 1, 1, 1, 1], 400)
    p = rng.random(len(y))
    assert precision_media(y, p, clase_positiva=0) == pytest.approx(2 / 7, abs=0.03)
    assert precision_media(y, p, clase_positiva=1) == pytest.approx(5 / 7, abs=0.03)


def test_precision_media_descarta_los_nan():
    y = np.array([0, 0, 1, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9, np.nan, 0.95, 0.7])
    y_nan = y.astype(float)
    y_nan[1] = np.nan
    assert precision_media(y, p) == precision_media(y[[0, 1, 2, 3, 5, 6]],
                                                    p[[0, 1, 2, 3, 5, 6]])
    assert not np.isnan(precision_media(y_nan, p))


def test_precision_media_falla_con_una_sola_clase_y_con_argumentos_malos():
    with pytest.raises(ValueError, match="una sola clase"):
        precision_media(np.array([1, 1, 1]), np.array([0.1, 0.5, 0.9]))
    with pytest.raises(ValueError, match="misma longitud"):
        precision_media(np.array([0, 1]), np.array([0.1, 0.5, 0.9]))
    with pytest.raises(ValueError, match="0 o 1"):
        precision_media(np.array([0, 1]), np.array([0.1, 0.9]), clase_positiva=2)


# --------------------------------------------------------------------------
# Desglose de errores por condición original
# --------------------------------------------------------------------------

def _caso_a_mano():
    """
    Diez grabaciones con la respuesta conocida:

        OC     2 filas, y=0: predichas 0, 0        -> 2/2 aciertos
        OA     2 filas, y=0: predichas 0, 1        -> 1/2
        STROOP 2 filas, y=1: predichas 1, 0        -> 1/2
        PASAT  2 filas, y=1: predichas 0, 0        -> 0/2
        MSIT   2 filas, y=1: predichas 1, 1        -> 2/2
    """
    datos = pd.DataFrame({"condicion": ["OC", "OC", "OA", "OA", "STROOP", "STROOP",
                                        "PASAT", "PASAT", "MSIT", "MSIT"]})
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    return datos, y_true, y_pred


def test_desglose_da_las_tasas_de_acierto_de_un_caso_a_mano():
    datos, y_true, y_pred = _caso_a_mano()
    d = desglose_por_condicion(datos, y_true, y_pred)
    assert d["n"].to_dict() == {"OC": 2, "OA": 2, "STROOP": 2, "PASAT": 2, "MSIT": 2}
    assert d["aciertos"].to_dict() == {"OC": 2, "OA": 1, "STROOP": 1, "PASAT": 0,
                                       "MSIT": 2}
    assert d["tasa_acierto"].to_dict() == {"OC": 1.0, "OA": 0.5, "STROOP": 0.5,
                                           "PASAT": 0.0, "MSIT": 1.0}
    assert d["clase_real"].to_dict() == {"OC": 0, "OA": 0, "STROOP": 1, "PASAT": 1,
                                         "MSIT": 1}


def test_desglose_frac_predicha_tarea_es_el_error_en_reposo_y_el_acierto_en_tarea():
    datos, y_true, y_pred = _caso_a_mano()
    d = desglose_por_condicion(datos, y_true, y_pred)
    reposo = d[d["clase_real"] == 0]
    tarea = d[d["clase_real"] == 1]
    assert (reposo["frac_predicha_tarea"] == 1 - reposo["tasa_acierto"]).all()
    assert (tarea["frac_predicha_tarea"] == tarea["tasa_acierto"]).all()


def test_desglose_sale_en_el_orden_del_protocolo_y_solo_con_las_presentes():
    datos, y_true, y_pred = _caso_a_mano()
    orden = np.random.default_rng(0).permutation(len(datos))
    d = desglose_por_condicion(datos.iloc[orden].reset_index(drop=True),
                               y_true[orden], y_pred[orden])
    assert list(d.index) == ["OC", "OA", "STROOP", "PASAT", "MSIT"]


def test_desglose_los_aciertos_suman_los_del_total():
    """El desglose no puede inventar ni perder grabaciones."""
    matriz = _matriz_tarea(n_personas=30, separacion=1.0)
    datos, _ = preparar_reposo_tarea(matriz)
    y = etiqueta_tarea(datos)
    X, g = datos[["frontal_theta_rel_media", "ruido_1"]], datos["persona_uid"].to_numpy()
    r = evaluar_reposo(X, y, g, modelo_logistico())
    d = desglose_por_condicion(datos, y, r["y_pred"])
    assert d["n"].sum() == len(datos)
    assert d["aciertos"].sum() == int((y == r["y_pred"]).sum())
    tp = r["metricas"]["matriz_confusion"]
    assert d.loc[d["clase_real"] == 1, "aciertos"].sum() == tp["vp"]
    assert d.loc[d["clase_real"] == 0, "aciertos"].sum() == tp["vn"]


def test_desglose_sin_grupos_no_da_intervalos():
    datos, y_true, y_pred = _caso_a_mano()
    d = desglose_por_condicion(datos, y_true, y_pred)
    assert "ic95_inf" not in d.columns and "ic95_sup" not in d.columns


def test_desglose_con_grupos_da_intervalos_que_contienen_la_tasa():
    matriz = _matriz_tarea(n_personas=30, separacion=1.0)
    datos, _ = preparar_reposo_tarea(matriz)
    y = etiqueta_tarea(datos)
    g = datos["persona_uid"].to_numpy()
    r = evaluar_reposo(datos[["frontal_theta_rel_media", "ruido_1"]], y, g,
                       modelo_logistico())
    d = desglose_por_condicion(datos, y, r["y_pred"], grupos=g, n_replicas=300)
    assert (d["ic95_inf"] <= d["tasa_acierto"] + 1e-9).all()
    assert (d["tasa_acierto"] <= d["ic95_sup"] + 1e-9).all()
    assert (d["ic95_inf"] < d["ic95_sup"]).all()
    assert (d["n_personas"] == datos["persona_uid"].nunique()).all()


def test_desglose_un_acierto_perfecto_da_un_intervalo_degenerado_en_1():
    datos, y_true, _ = _caso_a_mano()
    g = np.array(["A", "B"] * 5)
    d = desglose_por_condicion(datos, y_true, y_true, grupos=g, n_replicas=100)
    assert (d["tasa_acierto"] == 1.0).all()
    assert (d["ic95_inf"] == 1.0).all() and (d["ic95_sup"] == 1.0).all()


def test_desglose_es_reproducible_y_depende_de_la_semilla():
    matriz = _matriz_tarea(n_personas=20, separacion=0.8)
    datos, _ = preparar_reposo_tarea(matriz)
    y = etiqueta_tarea(datos)
    g = datos["persona_uid"].to_numpy()
    pred = np.random.default_rng(1).integers(0, 2, len(y))
    a = desglose_por_condicion(datos, y, pred, g, n_replicas=200, semilla=1)
    b = desglose_por_condicion(datos, y, pred, g, n_replicas=200, semilla=1)
    c = desglose_por_condicion(datos, y, pred, g, n_replicas=200, semilla=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a["ic95_inf"].equals(c["ic95_inf"])


def test_desglose_el_intervalo_remuestrea_personas_no_filas():
    """
    Cinco personas siempre bien y cinco siempre mal, con 40 grabaciones cada
    una. Remuestrear personas da un intervalo enorme (la fracción de personas
    «buenas» oscila mucho con solo 10); remuestrear las 400 filas daría uno
    estrechísimo alrededor de 0.5.
    """
    n_por_persona = 40
    g = np.repeat([f"P{i}" for i in range(10)], n_por_persona)
    datos = pd.DataFrame({"condicion": ["STROOP"] * len(g)})
    y_true = np.ones(len(g), dtype=int)
    y_pred = np.where(np.isin(g, [f"P{i}" for i in range(5)]), 1, 0)
    d = desglose_por_condicion(datos, y_true, y_pred, grupos=g, n_replicas=500)
    ancho = d.loc["STROOP", "ic95_sup"] - d.loc["STROOP", "ic95_inf"]
    assert d.loc["STROOP", "tasa_acierto"] == pytest.approx(0.5)
    assert ancho > 0.4


def test_desglose_falla_si_la_etiqueta_no_es_funcion_de_la_condicion():
    """Con etiquetas barajadas dentro de persona una condición mezcla las dos
    clases y el desglose no tiene sentido: tiene que negarse."""
    datos = pd.DataFrame({"condicion": ["OA", "OA", "STROOP", "STROOP"]})
    with pytest.raises(ValueError, match="las dos clases"):
        desglose_por_condicion(datos, [0, 1, 1, 1], [0, 1, 1, 1])


def test_desglose_valida_sus_entradas():
    datos, y_true, y_pred = _caso_a_mano()
    with pytest.raises(ValueError, match="misma longitud"):
        desglose_por_condicion(datos, y_true[:-1], y_pred)
    with pytest.raises(ValueError, match="misma longitud"):
        desglose_por_condicion(datos, y_true, y_pred, grupos=np.array(["A"]))
    with pytest.raises(ValueError, match="NaN"):
        desglose_por_condicion(datos, y_true, np.where(y_pred == 1, np.nan, y_pred))
    with pytest.raises(ValueError, match="0 o 1"):
        desglose_por_condicion(datos, y_true, np.full(len(y_true), 2))
    with pytest.raises(ValueError, match="condicion"):
        desglose_por_condicion(pd.DataFrame({"otra": [1] * 10}), y_true, y_pred)
    ajena = datos.copy()
    ajena.loc[0, "condicion"] = "OTRA"
    with pytest.raises(ValueError, match="OTRA"):
        desglose_por_condicion(ajena, y_true, y_pred)


# --------------------------------------------------------------------------
# Comparar dos condiciones
# --------------------------------------------------------------------------

def _casi_a_mano(n_personas=12):
    """OA siempre bien, OC bien la mitad de las veces; tres grabaciones por
    persona y condición."""
    filas = []
    for p in range(n_personas):
        for k in range(4):
            filas.append((f"P{p}", "OA", 0, 0))
            filas.append((f"P{p}", "OC", 0, 0 if k % 2 == 0 else 1))
    df = pd.DataFrame(filas, columns=["persona_uid", "condicion", "y", "pred"])
    return df[["condicion"]], df["y"].to_numpy(), df["pred"].to_numpy(), \
        df["persona_uid"].to_numpy()


def test_comparar_condiciones_da_la_diferencia_exacta():
    datos, y, pred, g = _casi_a_mano()
    c = comparar_condiciones(datos, y, pred, g, "OA", "OC", n_replicas=200)
    assert c["tasa_a"] == 1.0 and c["tasa_b"] == 0.5
    assert c["diferencia"] == pytest.approx(0.5)
    assert c["n_a"] == c["n_b"] == 48
    assert c["ic95"][0] <= c["diferencia"] <= c["ic95"][1]


def test_comparar_condiciones_es_antisimetrico():
    datos, y, pred, g = _casi_a_mano()
    ab = comparar_condiciones(datos, y, pred, g, "OA", "OC", n_replicas=200, semilla=5)
    ba = comparar_condiciones(datos, y, pred, g, "OC", "OA", n_replicas=200, semilla=5)
    assert ab["diferencia"] == pytest.approx(-ba["diferencia"])
    assert ab["ic95"][0] == pytest.approx(-ba["ic95"][1])
    assert ab["ic95"][1] == pytest.approx(-ba["ic95"][0])


def test_comparar_condiciones_cuenta_las_replicas_a_favor_de_a():
    datos, y, pred, g = _casi_a_mano()
    c = comparar_condiciones(datos, y, pred, g, "OA", "OC", n_replicas=200)
    assert c["prop_a_mayor"] == 1.0   # OA acierta más en todas las remuestras
    assert c["n_replicas"] + c["n_descartadas"] == 200


def test_comparar_condiciones_sin_diferencia_da_un_intervalo_que_contiene_el_0():
    datos, y, _, g = _casi_a_mano()
    pred = y.copy()   # las dos condiciones perfectas
    c = comparar_condiciones(datos, y, pred, g, "OA", "OC", n_replicas=100)
    assert c["diferencia"] == 0.0
    assert c["ic95"] == (0.0, 0.0)
    assert c["prop_a_mayor"] == 0.0    # ninguna réplica con a > b estricto


def test_comparar_condiciones_remuestrea_personas_no_filas():
    """
    La mitad de las personas aciertan siempre en A y nunca en B, la otra mitad
    al revés: la diferencia por persona es +1 o -1 y la global ronda 0. Con
    personas, el intervalo es ancho; con filas sería casi un punto.
    """
    filas = []
    for p in range(10):
        bueno_en_a = p % 2 == 0
        for _ in range(30):
            filas.append((f"P{p}", "OA", 0, 0 if bueno_en_a else 1))
            filas.append((f"P{p}", "OC", 0, 1 if bueno_en_a else 0))
    df = pd.DataFrame(filas, columns=["persona_uid", "condicion", "y", "pred"])
    c = comparar_condiciones(df[["condicion"]], df["y"].to_numpy(),
                             df["pred"].to_numpy(), df["persona_uid"].to_numpy(),
                             "OA", "OC", n_replicas=500)
    assert c["diferencia"] == pytest.approx(0.0)
    assert c["ic95"][1] - c["ic95"][0] > 0.5


def test_comparar_condiciones_valida_sus_entradas():
    datos, y, pred, g = _casi_a_mano()
    with pytest.raises(ValueError, match="distintas"):
        comparar_condiciones(datos, y, pred, g, "OA", "OA")
    with pytest.raises(ValueError, match="STROOP"):
        comparar_condiciones(datos, y, pred, g, "OA", "STROOP")
    with pytest.raises(ValueError, match="grupos"):
        comparar_condiciones(datos, y, pred, None, "OA", "OC")


# --------------------------------------------------------------------------
# Dirección del efecto
# --------------------------------------------------------------------------

def test_columnas_media_del_bloque_de_theta_frontal_y_alfa_posterior():
    """
    Son las tres features `_media` de cada bloque, incluido el alias del índice
    de fatiga (`theta_frontal_log_media`, `alfa_posterior_log_media`), que es una
    copia exacta de otra y por tanto cuenta dos veces.
    """
    columnas = columnas_de_features_tarea(
        pd.DataFrame(np.zeros((2, 91)), columns=_esquema_fase_13()))
    assert sorted(columnas_media_del_bloque(columnas, "frontal", "theta")) == [
        "frontal_theta_logabs_media", "frontal_theta_rel_media",
        "theta_frontal_log_media"]
    assert sorted(columnas_media_del_bloque(columnas, "posterior", "alfa")) == [
        "alfa_posterior_log_media", "posterior_alfa_logabs_media",
        "posterior_alfa_rel_media"]


def test_columnas_media_del_bloque_no_incluye_las_sd_ni_otros_bloques():
    columnas = columnas_de_features_tarea(
        pd.DataFrame(np.zeros((2, 91)), columns=_esquema_fase_13()))
    elegidas = columnas_media_del_bloque(columnas, "frontal", "theta")
    assert all(c.endswith("_media") for c in elegidas)
    assert not any("_sd" in c for c in elegidas)
    assert not any(c.startswith(("posterior_", "temporal_")) for c in elegidas)


def test_coeficiente_neto_suma_solo_las_media_del_bloque_por_pliegue():
    coef = pd.DataFrame(
        {"pliegue_1": [1.0, 2.0, 4.0, 100.0, 100.0],
         "pliegue_2": [-1.0, 0.5, 0.5, -100.0, 100.0]},
        index=["frontal_theta_rel_media", "frontal_theta_logabs_media",
               "theta_frontal_log_media", "frontal_theta_rel_sd",
               "posterior_alfa_rel_media"])
    neto = coeficiente_neto_por_pliegue(coef, "frontal", "theta")
    assert neto.to_dict() == {"pliegue_1": 7.0, "pliegue_2": 0.0}


def test_coeficiente_neto_falla_si_el_bloque_no_esta_en_los_coeficientes():
    coef = pd.DataFrame({"pliegue_1": [1.0]}, index=["ruido_1"])
    with pytest.raises(ValueError, match="frontal/theta"):
        coeficiente_neto_por_pliegue(coef, "frontal", "theta")


def test_signos_umbral_es_mas_uno_si_la_tarea_tiene_mas_de_la_variable():
    matriz = _matriz_tarea(n_personas=30, separacion=3.0)
    X, y, g = _xyg(matriz)
    salida = signos_umbral_por_pliegue(X, y, g, "frontal_theta_rel_media")
    assert [s["pliegue"] for s in salida] == [1, 2, 3, 4, 5]
    assert all(s["signo"] == 1 for s in salida)


def test_signos_umbral_aprende_el_signo_menos_uno_con_el_efecto_invertido():
    """El signo se aprende: con el efecto al revés (como el alfa, que baja con
    la tarea) sale -1, y no hay que decírselo."""
    matriz = _matriz_tarea(n_personas=30, separacion=3.0)
    X, y, g = _xyg(matriz)
    X = X.assign(frontal_theta_rel_media=-X["frontal_theta_rel_media"])
    salida = signos_umbral_por_pliegue(X, y, g, "frontal_theta_rel_media")
    assert all(s["signo"] == -1 for s in salida)


def test_signos_umbral_de_cada_pliegue_es_el_de_su_entrenamiento():
    """El umbral del pliegue p es exactamente el que se estima con las filas que
    NO están en p, ni una más ni una menos."""
    matriz = _matriz_tarea(n_personas=30, separacion=3.0)
    X, y, g = _xyg(matriz)
    pliegues = pliegue_por_fila(g, 5)
    salida = signos_umbral_por_pliegue(X, y, g, "frontal_theta_rel_media")
    assert len(salida) == 5
    for s in salida:
        entrenamiento = pliegues != s["pliegue"]
        umbral, signo = umbral_una_variable(
            X.loc[entrenamiento, "frontal_theta_rel_media"], y[entrenamiento])
        assert (s["umbral"], s["signo"]) == (umbral, signo)


def test_signos_umbral_no_depende_de_las_filas_de_validacion_de_su_pliegue():
    """
    Corromper las filas de validación de un pliegue no cambia su umbral: es la
    propiedad que impide que el corte «vea» los datos que luego clasifica. (No
    se exige que cambien los de los demás pliegues: el óptimo de un corte puede
    seguir entre los mismos dos valores vecinos aunque cambien los extremos.)
    """
    matriz = _matriz_tarea(n_personas=30, separacion=3.0)
    X, y, g = _xyg(matriz)
    pliegues = pliegue_por_fila(g, 5)
    base = signos_umbral_por_pliegue(X, y, g, "frontal_theta_rel_media")
    for p in range(1, 6):
        corrupta = X.copy()
        corrupta.loc[pliegues == p, "frontal_theta_rel_media"] = 1e6
        tocado = signos_umbral_por_pliegue(corrupta, y, g, "frontal_theta_rel_media")
        assert tocado[p - 1]["umbral"] == base[p - 1]["umbral"]
        assert tocado[p - 1]["signo"] == base[p - 1]["signo"]


def test_signos_umbral_valida_sus_entradas():
    matriz = _matriz_tarea(n_personas=10)
    X, y, g = _xyg(matriz)
    with pytest.raises(ValueError, match="no_existe"):
        signos_umbral_por_pliegue(X, y, g, "no_existe")
    with pytest.raises(ValueError, match="misma longitud"):
        signos_umbral_por_pliegue(X, y[:-1], g, "frontal_theta_rel_media")


# --------------------------------------------------------------------------
# Integración: la CV agrupada y el control negativo
# --------------------------------------------------------------------------

def test_con_una_señal_clara_el_modelo_separa_reposo_de_tarea_en_personas_nuevas():
    matriz = _matriz_tarea(n_personas=30, separacion=3.0)
    datos, _ = preparar_reposo_tarea(matriz)
    X, y, g = _xyg(datos)
    r = evaluar_reposo(X, y, g, modelo_logistico())
    m = r["metricas"]
    assert m["exactitud_balanceada"] > 0.85
    assert m["auc_agregado"] > 0.9
    assert precision_media(y, r["puntuaciones"]) > 2 / 7 + 0.4   # >> prevalencia


def test_el_bosque_tambien_separa_y_admite_la_misma_evaluacion():
    matriz = _matriz_tarea(n_personas=30, separacion=3.0)
    datos, _ = preparar_reposo_tarea(matriz)
    X, y, g = _xyg(datos)
    r = evaluar_reposo(X, y, g, modelo_bosque(n_estimators=40, n_jobs=1))
    assert r["metricas"]["exactitud_balanceada"] > 0.8


def test_con_las_etiquetas_barajadas_dentro_de_persona_la_cv_devuelve_el_azar():
    """
    El control negativo del análisis, con clases desbalanceadas. La misma
    feature con señal real da > 0.8 con las etiquetas verdaderas y ~0.5 con las
    barajadas dentro de cada persona. Si diera más, habría una fuga en el
    propio montaje y ningún resultado posterior valdría.
    """
    matriz = _matriz_tarea(n_personas=40, n_sesiones=4, separacion=2.5)
    X, y, g = _xyg(matriz)
    real = evaluar_reposo(X, y, g, modelo_logistico())["metricas"]
    nulas = [
        evaluar_reposo(X, permutar_etiquetas_dentro_de_persona(y, g, semilla=s), g,
                       modelo_logistico())["metricas"]["exactitud_balanceada"]
        for s in range(5)
    ]
    assert real["exactitud_balanceada"] > 0.8
    assert abs(np.mean(nulas) - 0.5) < 0.05
    assert max(nulas) < 0.62


def test_una_feature_que_solo_identifica_a_la_persona_no_separa_reposo_de_tarea():
    """`firma` identifica a la persona y no dice nada de la condición: validando
    por persona, no puede dar más que el azar."""
    matriz = _matriz_tarea(n_personas=40, n_sesiones=4)
    X, y, g = _xyg(matriz, columnas=("firma",))
    r = evaluar_reposo(X, y, g, modelo_logistico())
    assert abs(r["metricas"]["exactitud_balanceada"] - 0.5) < 0.08


def test_el_desglose_de_un_modelo_con_señal_solo_en_las_tareas_lo_localiza():
    """Si el efecto es de carga (theta sube en TODAS las tareas), OA y OC salen
    igual de bien: el desglose no debería inventar una diferencia entre ellas."""
    matriz = _matriz_tarea(n_personas=40, n_sesiones=4, separacion=3.0)
    datos, _ = preparar_reposo_tarea(matriz)
    X, y, g = _xyg(datos)
    r = evaluar_reposo(X, y, g, modelo_logistico())
    c = comparar_condiciones(datos, y, r["y_pred"], g, "OA", "OC", n_replicas=300)
    assert c["ic95"][0] <= 0.0 <= c["ic95"][1]


def test_las_funciones_nuevas_estan_en_la_api_del_paquete():
    import eeg_library
    for nombre in ("columnas_de_features_tarea", "desglose_por_condicion",
                   "etiqueta_tarea", "preparar_reposo_tarea"):
        assert callable(getattr(eeg_library, nombre))


def test_la_exactitud_balanceada_de_las_metricas_coincide_con_la_del_azar_teorico():
    """Un clasificador constante da 0.5 exacto con cualquier reparto: es lo que
    justifica leer 0.5 como «no sabe nada»."""
    y = np.array([0] * 20 + [1] * 50)
    for constante in (0, 1):
        m = metricas_binarias(y, np.full(len(y), constante))
        assert m["exactitud_balanceada"] == pytest.approx(0.5)
