"""
Tests de `modelado/reposo.py`: clasificación OA (ojos abiertos) frente a OC.

Protegen los invariantes que hacen que el resultado de ese análisis signifique
algo:

1. **`condicion_orden` nunca entra a X.** Es la etiqueta codificada (OC = 1,
   OA = 2): con ella el acierto sería del 100 % por construcción.
2. **Los contadores de procesamiento quedan fuera por defecto**, y solo por
   defecto: `incluir_procesamiento=True` los devuelve, pero jamás devuelve
   `condicion_orden`.
3. **El conjunto de test no se abre** sin pedirlo explícitamente.
4. **La validación es agrupada por persona** y las métricas puntúan lo que dicen
   (exactitud balanceada = 0.5 en el azar).
5. **El control negativo funciona**: con las etiquetas barajadas dentro de cada
   persona, la CV devuelve el azar.

Todo con datos sintéticos en memoria: ningún test toca disco.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_library.modelado.modeling import predecir_cv_agrupada
from eeg_library.modelado.reposo import (
    COLUMNA_FUGA,
    COLUMNAS_PROCESAMIENTO,
    ProbabilidadComoPrediccion,
    UmbralUnaVariable,
    auc_roc,
    bloque_de_feature,
    bootstrap_diferencia_por_persona,
    bootstrap_por_persona,
    clasificar_por_umbral,
    coeficientes_por_pliegue,
    columnas_de_features_reposo,
    columnas_del_bloque,
    columnas_duplicadas,
    etiqueta_binaria,
    evaluar_reposo,
    importancia_permutacion_cv,
    importancia_por_bloque,
    metricas_binarias,
    metricas_binarias_por_pliegue,
    permutar_etiquetas_dentro_de_persona,
    pliegue_por_fila,
    preparar_reposo,
    tamanos_de_efecto,
    umbral_una_variable,
    verificar_sin_fuga,
)
from eeg_library.procesado.bands import BANDAS_DEFECTO
from eeg_library.procesado.matrix import (
    COLUMNAS_IDENTIDAD,
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


def _matriz_reposo(n_personas=20, n_sesiones=3, separacion=3.0, semilla=0):
    """
    Matriz por grabación mínima con la forma de la Fase 13, solo OA y OC.

    Cada persona tiene un nivel propio de `posterior_alfa_rel_media` (su
    «firma», sd 0.5) y, en OC, un salto de `separacion` sobre OA. Las features
    `ruido_*` no dependen de la condición y `firma` identifica a la persona sin
    decir nada sobre OA/OC. Los splits van por persona: 70 % train, 15 % val,
    15 % test.
    """
    rng = np.random.default_rng(semilla)
    filas = []
    for p in range(n_personas):
        fraccion = p / n_personas
        split = "train" if fraccion < 0.7 else ("val" if fraccion < 0.85 else "test")
        nivel = rng.normal(0, 0.5)
        for s in range(n_sesiones):
            for condicion, orden in (("OC", 1.0), ("OA", 2.0)):
                fila = dict(CAMPOS_IDENTIDAD)
                fila.update({
                    "archivo": f"P{p:02d}_M{s}_{condicion}.edf",
                    "sesion_uid": f"P{p:02d}_M{s}",
                    "persona_uid": f"P{p:02d}", "sujeto_id": f"P{p:02d}",
                    "medicion": f"M{s}", "condicion": condicion, "split": split,
                    "posterior_alfa_rel_media":
                        nivel + (separacion if condicion == "OC" else 0.0)
                        + rng.normal(0, 1),
                    "ruido_1": rng.normal(), "ruido_2": rng.normal(),
                    "firma": nivel * 20 + rng.normal(0, 0.1),
                    "n_ventanas": 55.0, "n_canales_buenos": 12.0,
                    "n_validos_frontal": 5, "n_validos_frontocentral": 2,
                    "n_validos_temporal": 2, "n_validos_posterior": 4,
                    COLUMNA_FUGA: orden,
                })
                filas.append(fila)
    return pd.DataFrame(filas)


def _xyg(matriz, columnas=("posterior_alfa_rel_media", "ruido_1", "ruido_2")):
    return matriz[list(columnas)], etiqueta_binaria(matriz), \
        matriz["persona_uid"].to_numpy()


def _logistica():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))


# --------------------------------------------------------------------------
# Guardianes de la fuga: qué entra a X
# --------------------------------------------------------------------------

def test_columnas_de_features_reposo_excluye_condicion_orden():
    """
    El test central. `condicion_orden` vale 1 en OC y 2 en OA sin excepciones: es
    la etiqueta con otro nombre. Si alguna vez llega a X, el acierto de OA/OC se
    dispara al 100 % y el resultado entero deja de significar nada.
    """
    matriz = _matriz_reposo()
    assert COLUMNA_FUGA in columnas_de_features(matriz), (
        "el escenario real: `columnas_de_features()` SÍ la devuelve"
    )
    assert COLUMNA_FUGA not in columnas_de_features_reposo(matriz)


def test_condicion_orden_sigue_excluida_aunque_se_pidan_los_conteos():
    """`incluir_procesamiento=True` devuelve los contadores, nunca la fuga."""
    columnas = columnas_de_features_reposo(_matriz_reposo(),
                                           incluir_procesamiento=True)
    assert COLUMNA_FUGA not in columnas
    assert set(COLUMNAS_PROCESAMIENTO) <= set(columnas)


def test_columnas_de_features_reposo_excluye_identidad_y_etiqueta():
    columnas = set(columnas_de_features_reposo(_matriz_reposo()))
    assert not columnas & set(COLUMNAS_IDENTIDAD)
    assert "vas" not in columnas
    assert "condicion" not in columnas


def test_por_defecto_quedan_fuera_los_seis_contadores_de_procesamiento():
    columnas = set(columnas_de_features_reposo(_matriz_reposo()))
    assert not columnas & set(COLUMNAS_PROCESAMIENTO)
    assert {"posterior_alfa_rel_media", "ruido_1", "firma"} <= columnas


def test_incluir_procesamiento_devuelve_exactamente_las_seis_columnas_mas():
    matriz = _matriz_reposo()
    sin = set(columnas_de_features_reposo(matriz))
    con = set(columnas_de_features_reposo(matriz, incluir_procesamiento=True))
    assert con - sin == set(COLUMNAS_PROCESAMIENTO)
    assert sin <= con


def test_con_el_esquema_de_la_fase_13_quedan_70_features():
    """
    Guardián de la documentación: docs/analisis/ANALISIS_OA_OC.md habla de 77 features de la
    Fase 13, 7 excluidas y 70 en el modelo. Si este test falla, esas cifras ya no
    corresponden al código y hay que reejecutar el análisis y actualizar el
    documento.
    """
    columnas = _esquema_fase_13()
    assert len(columnas) == 91
    matriz = pd.DataFrame(np.zeros((2, len(columnas))), columns=columnas)
    assert len(columnas_de_features(matriz)) == 77
    assert len(columnas_de_features_reposo(matriz)) == 70
    assert len(columnas_de_features_reposo(matriz, incluir_procesamiento=True)) == 76


def test_verificar_sin_fuga_rechaza_condicion_orden_y_la_identidad():
    with pytest.raises(ValueError, match="condicion_orden"):
        verificar_sin_fuga(["posterior_alfa_rel_media", "condicion_orden"])
    with pytest.raises(ValueError, match="persona_uid"):
        verificar_sin_fuga(["persona_uid"])
    with pytest.raises(ValueError, match="condicion"):
        verificar_sin_fuga(["condicion"])
    with pytest.raises(ValueError, match="vas"):
        verificar_sin_fuga(["vas"])


def test_verificar_sin_fuga_devuelve_la_lista_si_esta_limpia():
    limpias = ["posterior_alfa_rel_media", "n_ventanas"]
    assert verificar_sin_fuga(limpias) == limpias


def test_un_modelo_con_condicion_orden_acierta_siempre():
    """
    Control positivo de que el peligro es real: metida a mano, la fuga da el 100 %
    por validación cruzada agrupada y todo. Es lo que ocurriría si la exclusión
    se rompiera sin que nadie lo notara.
    """
    matriz = _matriz_reposo()
    X, y, g = _xyg(matriz, columnas=(COLUMNA_FUGA,))
    r = evaluar_reposo(X, y, g, _logistica())
    assert r["metricas"]["exactitud_balanceada"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Preparación de los datos
# --------------------------------------------------------------------------

def test_preparar_reposo_se_queda_solo_con_oa_y_oc():
    matriz = _matriz_reposo(n_personas=10)
    ajenas = matriz.iloc[:6].copy()
    ajenas["condicion"] = ["STROOP", "PASAT", "PVTB", "MSIT", "PVSAT", "STROOP"]
    datos, meta = preparar_reposo(pd.concat([matriz, ajenas], ignore_index=True))
    assert set(datos["condicion"]) == {"OA", "OC"}
    assert set(meta["por_condicion"]) == {"OA", "OC"}


def test_preparar_reposo_aparta_el_conjunto_de_test():
    matriz = _matriz_reposo(n_personas=20)
    datos, meta = preparar_reposo(matriz)
    assert set(datos["split"]) <= {"train", "val"}
    n_test = int((matriz["split"] == "test").sum())
    assert n_test > 0
    assert meta["n_apartadas"] == n_test
    assert meta["por_split_apartado"] == {"test": n_test}
    assert meta["n_grabaciones"] + n_test == len(matriz)


def test_preparar_reposo_se_niega_a_abrir_test_salvo_permiso():
    """
    `test` se reserva para un único análisis confirmatorio pre-registrado: pedirlo
    por descuido tiene que ser un error, no un aviso que alguien ignore.
    """
    matriz = _matriz_reposo()
    with pytest.raises(ValueError, match="test"):
        preparar_reposo(matriz, splits=("train", "val", "test"))
    datos, _ = preparar_reposo(matriz, splits=("test",), permitir_test=True)
    assert set(datos["split"]) == {"test"}


def test_preparar_reposo_descarta_las_grabaciones_sin_persona_uid():
    """La Fase 12 deja `persona_uid` a NaN en las sesiones sin VAS: sin él no se
    puede agrupar, así que arrastrarlas rompería la CV por persona."""
    matriz = _matriz_reposo(n_personas=10)
    matriz["persona_uid"] = matriz["persona_uid"].astype(object)
    matriz.loc[matriz.index[:4], "persona_uid"] = np.nan
    datos, meta = preparar_reposo(matriz)
    assert datos["persona_uid"].notna().all()
    assert meta["n_sin_persona_uid"] == 4 - int(
        (matriz.loc[matriz.index[:4], "split"] == "test").sum())


def test_preparar_reposo_falla_si_no_quedan_dos_clases():
    matriz = _matriz_reposo(n_personas=10)
    with pytest.raises(ValueError, match="OA y OC"):
        preparar_reposo(matriz[matriz["condicion"] == "OA"])


def test_preparar_reposo_falla_si_faltan_columnas():
    with pytest.raises(ValueError, match="persona_uid"):
        preparar_reposo(_matriz_reposo().drop(columns="persona_uid"))


def test_preparar_reposo_reinicia_el_indice():
    datos, _ = preparar_reposo(_matriz_reposo())
    assert list(datos.index) == list(range(len(datos)))


def test_etiqueta_binaria_codifica_oc_como_uno():
    datos, _ = preparar_reposo(_matriz_reposo())
    y = etiqueta_binaria(datos)
    assert set(np.unique(y)) == {0, 1}
    assert (y[datos["condicion"] == "OC"] == 1).all()
    assert (y[datos["condicion"] == "OA"] == 0).all()


def test_etiqueta_binaria_rechaza_otras_condiciones():
    matriz = _matriz_reposo(n_personas=4)
    matriz.loc[0, "condicion"] = "STROOP"
    with pytest.raises(ValueError, match="STROOP"):
        etiqueta_binaria(matriz)


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------

def test_metricas_binarias_reconocen_el_acierto_perfecto():
    y = np.array([0, 0, 1, 1, 1, 0])
    m = metricas_binarias(y, y)
    assert m["exactitud"] == m["exactitud_balanceada"] == 1.0
    assert m["sensibilidad"] == m["especificidad"] == 1.0
    assert m["mcc"] == pytest.approx(1.0)
    assert m["matriz_confusion"] == {"vn": 3, "fp": 0, "fn": 0, "vp": 3}


def test_la_clase_mayoritaria_da_exactamente_media_balanceada():
    """
    El invariante que hace de la exactitud balanceada la métrica principal:
    predecir siempre la clase más frecuente da 0.5 exacto **sea cual sea** el
    reparto, mientras que la exactitud simple premia el desbalance.
    """
    y = np.array([1] * 90 + [0] * 10)
    m = metricas_binarias(y, np.ones(100))
    assert m["exactitud"] == pytest.approx(0.9)
    assert m["exactitud_balanceada"] == pytest.approx(0.5)
    assert m["mcc"] == 0.0


def test_la_matriz_de_confusion_cuadra_con_los_recuentos():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0])
    m = metricas_binarias(y_true, y_pred)
    c = m["matriz_confusion"]
    assert c == {"vn": 3, "fp": 1, "fn": 2, "vp": 4}
    assert c["vn"] + c["fp"] + c["fn"] + c["vp"] == m["n"] == 10
    assert m["sensibilidad"] == pytest.approx(4 / 6)
    assert m["especificidad"] == pytest.approx(3 / 4)
    assert m["precision"] == pytest.approx(4 / 5)


def test_la_exactitud_balanceada_coincide_con_la_de_sklearn():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, 200)
    y_pred = np.where(rng.random(200) < 0.7, y_true, 1 - y_true)
    assert (metricas_binarias(y_true, y_pred)["exactitud_balanceada"]
            == pytest.approx(balanced_accuracy_score(y_true, y_pred)))


def test_metricas_binarias_descartan_los_nan():
    y_true = np.array([0, 0, 1, 1, np.nan, 1])
    y_pred = np.array([0, 1, 1, 1, 1, np.nan])
    assert metricas_binarias(y_true, y_pred)["n"] == 4


def test_metricas_binarias_exigen_las_dos_clases_y_valores_binarios():
    with pytest.raises(ValueError, match="una sola clase"):
        metricas_binarias(np.ones(10), np.ones(10))
    with pytest.raises(ValueError, match="0 o 1"):
        metricas_binarias(np.array([0, 1, 0, 1]), np.array([0.2, 0.9, 0.1, 0.8]))
    with pytest.raises(ValueError, match="al menos 3"):
        metricas_binarias(np.array([0, 1]), np.array([0, 1]))
    with pytest.raises(ValueError, match="misma longitud"):
        metricas_binarias(np.array([0, 1, 0]), np.array([0, 1]))


def test_auc_roc_de_una_puntuacion_perfecta_y_de_una_constante():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert auc_roc(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) == 1.0
    assert auc_roc(y, np.full(6, 0.5)) == 0.5
    with pytest.raises(ValueError, match="una sola clase"):
        auc_roc(np.ones(4), np.array([0.1, 0.2, 0.3, 0.4]))


def test_un_clasificador_constante_da_exactamente_medio_dentro_de_cada_pliegue():
    """
    Un `DummyClassifier` no usa X: dentro de cada pliegue solo puede acertar una
    clase de dos, así que su exactitud balanceada por pliegue y su AUC son 0.5
    exactos. Es el baseline trivial contra el que se compara todo.
    """
    matriz = _matriz_reposo(n_personas=20)
    X, y, g = _xyg(matriz)
    r = evaluar_reposo(X, y, g, DummyClassifier(strategy="most_frequent"))
    m = r["metricas"]
    assert m["exactitud_balanceada_pliegues"] == pytest.approx([0.5] * 5)
    assert m["exactitud_balanceada_media"] == pytest.approx(0.5)
    assert m["auc_media"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Pliegues y validación cruzada
# --------------------------------------------------------------------------

def test_pliegue_por_fila_cubre_todas_las_filas_una_sola_vez():
    matriz = _matriz_reposo(n_personas=20)
    pliegues = pliegue_por_fila(matriz["persona_uid"].to_numpy())
    assert (pliegues >= 1).all() and (pliegues <= 5).all()
    assert sorted(set(pliegues)) == [1, 2, 3, 4, 5]


def test_ninguna_persona_aparece_en_dos_pliegues():
    matriz = _matriz_reposo(n_personas=20)
    pliegues = pd.Series(pliegue_por_fila(matriz["persona_uid"].to_numpy()))
    por_persona = pliegues.groupby(matriz["persona_uid"].to_numpy()).nunique()
    assert (por_persona == 1).all()


def test_pliegue_por_fila_reproduce_el_reparto_de_la_cv_agrupada():
    """
    `pliegue_por_fila` reconstruye los pliegues en vez de recibirlos de
    `predecir_cv_agrupada`. Este test vigila que no se desincronicen: los
    tamaños por pliegue tienen que coincidir con los que reporta esa función.
    """
    matriz = _matriz_reposo(n_personas=23, n_sesiones=4)
    X, y, g = _xyg(matriz)
    _, detalle = predecir_cv_agrupada(
        X, y, g, DummyClassifier(strategy="most_frequent"), n_splits=5)
    pliegues = pliegue_por_fila(g, n_splits=5)
    for info in detalle["reparto"]:
        en_pliegue = pliegues == info["pliegue"]
        assert int(en_pliegue.sum()) == info["sesiones_val"]
        assert len(set(g[en_pliegue])) == info["personas_val"]


def test_el_envoltorio_devuelve_la_probabilidad_de_la_clase_positiva():
    matriz = _matriz_reposo()
    X, y, _ = _xyg(matriz)
    envuelto = ProbabilidadComoPrediccion(_logistica()).fit(X, y)
    esperado = clone(_logistica()).fit(X, y).predict_proba(X)[:, 1]
    assert envuelto.predict(X) == pytest.approx(esperado)


def test_el_envoltorio_se_puede_clonar():
    """`predecir_cv_agrupada` clona el modelo en cada pliegue."""
    clonado = clone(ProbabilidadComoPrediccion(_logistica()))
    assert isinstance(clonado, ProbabilidadComoPrediccion)


def test_el_envoltorio_falla_si_la_clase_positiva_no_esta_en_el_entrenamiento():
    # Un Dummy acepta una sola clase (LogisticRegression se negaría antes, sin
    # llegar a la guarda del envoltorio).
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="clase positiva"):
        ProbabilidadComoPrediccion(
            DummyClassifier(strategy="most_frequent")).fit(X, np.zeros(4))


def test_las_probabilidades_out_of_fold_son_una_por_fila_entre_0_y_1():
    matriz = _matriz_reposo()
    X, y, g = _xyg(matriz)
    r = evaluar_reposo(X, y, g, _logistica())
    assert len(r["puntuaciones"]) == len(matriz)
    assert not np.isnan(r["puntuaciones"]).any()
    assert r["puntuaciones"].min() >= 0.0 and r["puntuaciones"].max() <= 1.0
    assert set(np.unique(r["y_pred"])) <= {0, 1}


def test_evaluar_reposo_sin_probabilidad_no_da_auc():
    matriz = _matriz_reposo()
    X, y, g = _xyg(matriz)
    r = evaluar_reposo(X, y, g, UmbralUnaVariable("posterior_alfa_rel_media"),
                       con_probabilidad=False)
    assert r["puntuaciones"] is None
    assert "auc_media" not in r["metricas"]
    assert r["metricas"]["exactitud_balanceada"] > 0.7


def test_las_metricas_por_pliegue_exigen_longitudes_coherentes():
    with pytest.raises(ValueError, match="misma longitud"):
        metricas_binarias_por_pliegue(np.array([0, 1, 0, 1]),
                                      np.array([0, 1, 0, 1]), np.array([1, 1, 2]))


def test_la_cv_agrupada_recupera_una_senal_de_condicion_real():
    """Control positivo: una feature realmente ligada a la condición se recupera
    en personas nunca vistas."""
    matriz = _matriz_reposo(n_personas=30, separacion=3.0)
    X, y, g = _xyg(matriz)
    m = evaluar_reposo(X, y, g, _logistica())["metricas"]
    assert m["exactitud_balanceada"] > 0.8
    assert m["auc_media"] > 0.9


def test_la_cv_agrupada_no_recupera_lo_que_es_pura_identidad():
    """
    Control negativo: `firma` identifica a la persona pero no dice nada de OA/OC
    (cada persona tiene las dos condiciones). Reconocer la firma no sirve para
    la etiqueta: tiene que salir el azar.
    """
    matriz = _matriz_reposo(n_personas=40, n_sesiones=4, separacion=0.0)
    X, y, g = _xyg(matriz, columnas=("firma",))
    m = evaluar_reposo(X, y, g, RandomForestClassifier(
        n_estimators=50, random_state=0))["metricas"]
    assert m["exactitud_balanceada"] < 0.6


def test_agrupar_por_persona_impide_que_el_modelo_se_calibre_con_la_propia_persona():
    """
    El motivo de agrupar. La feature es un nivel propio de cada persona (muy
    espaciado entre personas) más un salto pequeño en OC: con las otras
    grabaciones de la persona en entrenamiento, un modelo aprende su nivel y
    separa OA de OC dentro de ella. Con personas nuevas ese nivel no ayuda. La
    diferencia entre ambas validaciones es la fuga que `persona_uid` cierra.
    """
    rng = np.random.default_rng(5)
    filas = []
    for p in range(30):
        nivel = rng.uniform(0, 3000)
        for s in range(4):
            for cond in (0, 1):
                filas.append({"persona_uid": f"P{p:02d}", "y": cond,
                              "x": nivel + 1.0 * cond + rng.normal(0, 0.15)})
    datos = pd.DataFrame(filas)
    X, y, g = datos[["x"]], datos["y"].to_numpy(), datos["persona_uid"].to_numpy()
    bosque = RandomForestClassifier(n_estimators=100, min_samples_leaf=1,
                                    random_state=0)

    agrupada = evaluar_reposo(X, y, g, bosque)["metricas"]["exactitud_balanceada"]

    pred = np.full(len(y), np.nan)
    for tr, va in KFold(5, shuffle=True, random_state=0).split(X):
        pred[va] = clone(bosque).fit(X.iloc[tr], y[tr]).predict(X.iloc[va])
    ingenua = balanced_accuracy_score(y, pred)

    assert ingenua > 0.85, "sin agrupar, calibrarse con la propia persona funciona"
    assert agrupada < 0.65, "agrupando, ese nivel personal no sirve"
    assert ingenua - agrupada > 0.25


# --------------------------------------------------------------------------
# Referencia de una sola variable
# --------------------------------------------------------------------------

def test_el_umbral_separa_dos_nubes_bien_separadas():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0, 1, 100), rng.normal(6, 1, 100)])
    y = np.array([0] * 100 + [1] * 100)
    umbral, signo = umbral_una_variable(x, y)
    assert 2.0 < umbral < 4.0
    assert signo == 1
    assert balanced_accuracy_score(y, clasificar_por_umbral(x, umbral, signo)) > 0.97


def test_el_umbral_aprende_el_signo_no_lo_da_por_supuesto():
    """Con la clase 1 en los valores BAJOS el signo sale -1: si estuviera fijado
    a +1 no se podría usar como comprobación de la dirección del efecto."""
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(6, 1, 100), rng.normal(0, 1, 100)])
    y = np.array([0] * 100 + [1] * 100)
    umbral, signo = umbral_una_variable(x, y)
    assert signo == -1
    assert balanced_accuracy_score(y, clasificar_por_umbral(x, umbral, signo)) > 0.97


def test_el_umbral_cae_entre_dos_valores_distintos_y_no_sobre_uno():
    umbral, _ = umbral_una_variable(np.array([1.0, 2.0, 3.0, 4.0]),
                                    np.array([0, 0, 1, 1]))
    assert umbral == pytest.approx(2.5)


def test_el_umbral_ignora_los_nan_al_ajustar():
    x = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan])
    y = np.array([0, 0, 1, 1, 1, 0])
    limpio, _ = umbral_una_variable(x[~np.isnan(x)], y[~np.isnan(x)])
    assert umbral_una_variable(x, y)[0] == pytest.approx(limpio)


def test_el_umbral_falla_con_una_variable_constante_o_una_sola_clase():
    with pytest.raises(ValueError, match="constante"):
        umbral_una_variable(np.ones(6), np.array([0, 1, 0, 1, 0, 1]))
    with pytest.raises(ValueError, match="dos clases"):
        umbral_una_variable(np.arange(6.0), np.zeros(6))


def test_el_umbral_se_ajusta_solo_con_el_entrenamiento():
    """
    El estimador estima el corte con lo que se le da en `fit`. Los datos de
    validación están desplazados y son más numerosos: si el corte se hubiera
    calculado con ellos también, sería otro (con el mismo tamaño empatarían en
    exactitud balanceada y no se distinguiría).
    """
    rng = np.random.default_rng(2)
    x_tr = np.concatenate([rng.normal(0, 1, 80), rng.normal(4, 1, 80)])
    y_tr = np.array([0] * 80 + [1] * 80)
    x_va = np.concatenate([rng.normal(10, 1, 320), rng.normal(14, 1, 320)])
    y_va = np.array([0] * 320 + [1] * 320)

    modelo = UmbralUnaVariable("x").fit(pd.DataFrame({"x": x_tr}), y_tr)
    assert modelo.umbral_ == pytest.approx(umbral_una_variable(x_tr, y_tr)[0])
    todo, _ = umbral_una_variable(np.concatenate([x_tr, x_va]),
                                  np.concatenate([y_tr, y_va]))
    assert abs(modelo.umbral_ - todo) > 1.0


def test_umbral_una_variable_sustituye_los_nan_por_la_mediana_del_entrenamiento():
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 20.0, 21.0]})
    modelo = UmbralUnaVariable("x").fit(X, np.array([0, 0, 0, 0, 0, 1, 1]))
    assert modelo.mediana_ == pytest.approx(3.0)
    pred = modelo.predict(pd.DataFrame({"x": [np.nan, 3.0, 20.5]}))
    assert list(pred) == [0, 0, 1]
    assert pred[0] == pred[1], "un NaN se trata como la mediana del entrenamiento"


def test_umbral_una_variable_exige_un_dataframe_con_la_columna():
    modelo = UmbralUnaVariable("x")
    with pytest.raises(ValueError, match="'x'"):
        modelo.fit(pd.DataFrame({"otra": [1.0, 2.0, 3.0, 4.0]}),
                   np.array([0, 0, 1, 1]))


# --------------------------------------------------------------------------
# Interpretación: bloques, importancias, coeficientes, tamaños de efecto
# --------------------------------------------------------------------------

def test_bloque_de_feature_reconoce_region_y_banda():
    assert bloque_de_feature("posterior_alfa_rel_media") == ("posterior", "alfa")
    assert bloque_de_feature("frontocentral_delta_logabs_sd") == (
        "frontocentral", "delta")
    assert bloque_de_feature("temporal_beta_rel_media") == ("temporal", "beta")


def test_bloque_de_feature_asigna_los_componentes_del_indice():
    """`alfa_posterior_log_*` es alfa posterior con otro nombre; el índice mezcla
    dos bandas y dos regiones y los contadores no son potencia: sin asignar."""
    assert bloque_de_feature("alfa_posterior_log_media") == ("posterior", "alfa")
    assert bloque_de_feature("theta_frontal_log_sd") == ("frontal", "theta")
    assert bloque_de_feature("indice_fatiga_media") == (None, None)
    assert bloque_de_feature("n_validos_frontal") == (None, None)
    assert bloque_de_feature("n_ventanas") == (None, None)


def test_bloque_de_feature_acepta_canales_como_regiones():
    canales = ("O1", "O2", "F3")
    assert bloque_de_feature("O1_alfa_rel_media", regiones=canales) == ("O1", "alfa")
    assert bloque_de_feature("posterior_alfa_rel_media", regiones=canales) == (
        None, None)
    # Sin la región `posterior` entre las conocidas, el alias tampoco aplica.
    assert bloque_de_feature("alfa_posterior_log_media", regiones=canales) == (
        None, None)


def test_columnas_del_bloque_filtra_por_region_y_banda():
    columnas = ["posterior_alfa_rel_media", "posterior_beta_rel_media",
                "frontal_alfa_rel_media", "alfa_posterior_log_media",
                "indice_fatiga_media"]
    assert columnas_del_bloque(columnas, region="posterior", banda="alfa") == [
        "posterior_alfa_rel_media", "alfa_posterior_log_media"]
    assert columnas_del_bloque(columnas, banda="alfa") == [
        "posterior_alfa_rel_media", "frontal_alfa_rel_media",
        "alfa_posterior_log_media"]
    assert columnas_del_bloque(columnas) == columnas


def test_importancia_por_bloque_reparte_por_region_y_banda():
    imp = pd.Series({
        "posterior_alfa_rel_media": 3.0, "posterior_alfa_logabs_media": 1.0,
        "alfa_posterior_log_media": 1.0, "frontal_delta_rel_sd": 1.0,
        "indice_fatiga_media": 2.0, "n_validos_frontal": 2.0,
    })
    r = importancia_por_bloque(imp)
    assert r["total"] == pytest.approx(10.0)
    assert r["por_banda"]["alfa"] == pytest.approx(0.5)
    assert r["por_banda"]["delta"] == pytest.approx(0.1)
    assert r["por_region"]["posterior"] == pytest.approx(0.5)
    assert r["por_bloque"]["posterior/alfa"] == pytest.approx(0.5)
    for reparto in (r["por_region"], r["por_banda"], r["por_bloque"]):
        assert sum(reparto.values()) == pytest.approx(1.0)


def test_importancia_por_bloque_agrupa_lo_que_no_encaja_como_sin_asignar():
    """No descarta lo que no es región×banda: lo deja como `sin_asignar` dentro
    del total, para que ninguna banda salga inflada al quitar del denominador."""
    imp = pd.Series({"posterior_alfa_rel_media": 1.0, "n_ventanas": 3.0})
    r = importancia_por_bloque(imp)
    assert r["por_banda"]["sin_asignar"] == pytest.approx(0.75)
    assert r["por_banda"]["alfa"] == pytest.approx(0.25)


def test_importancia_por_bloque_cuenta_las_negativas_como_cero():
    imp = pd.Series({"posterior_alfa_rel_media": 2.0, "frontal_delta_rel_sd": -5.0})
    r = importancia_por_bloque(imp)
    assert r["por_banda"]["alfa"] == pytest.approx(1.0)
    assert r["por_banda"]["delta"] == pytest.approx(0.0)


def test_importancia_por_bloque_falla_si_todo_es_cero():
    with pytest.raises(ValueError, match="nada que repartir"):
        importancia_por_bloque(pd.Series({"posterior_alfa_rel_media": 0.0}))


def test_la_importancia_por_permutacion_senala_la_feature_con_senal():
    matriz = _matriz_reposo(n_personas=30, separacion=2.5)
    X, y, g = _xyg(matriz)
    imp = importancia_permutacion_cv(X, y, g, _logistica(), n_repeticiones=5)
    assert imp["caida_media"].idxmax() == "posterior_alfa_rel_media"
    assert imp.loc["posterior_alfa_rel_media", "caida_media"] > 0.15
    assert abs(imp.loc["ruido_1", "caida_media"]) < 0.03
    assert abs(imp.loc["ruido_2", "caida_media"]) < 0.03


def test_permutar_un_bloque_entero_cae_mas_que_una_sola_columna_de_un_par_duplicado():
    """
    La razón de la importancia por bloques. Con dos columnas que llevan la misma
    señal, barajar una sola apenas hace daño porque la otra la cubre: la
    importancia individual se diluye entre las dos. Barajar las dos a la vez
    mide lo que el modelo de verdad pierde. Es exactamente la situación de las
    features de alfa posterior, donde hay varias casi duplicadas.
    """
    matriz = _matriz_reposo(n_personas=30, separacion=2.5)
    rng = np.random.default_rng(0)
    matriz["gemela"] = matriz["posterior_alfa_rel_media"] + rng.normal(0, 0.05, len(matriz))
    columnas = ["posterior_alfa_rel_media", "gemela", "ruido_1"]
    X, y, g = _xyg(matriz, columnas=columnas)
    bloques = {"a": ["posterior_alfa_rel_media"], "b": ["gemela"],
               "ambas": ["posterior_alfa_rel_media", "gemela"]}
    caida = importancia_permutacion_cv(X, y, g, _logistica(), bloques=bloques,
                                       n_repeticiones=10)["caida_media"]
    assert caida["ambas"] > 0.2
    assert caida["ambas"] > 1.5 * max(caida["a"], caida["b"])


def test_la_importancia_por_permutacion_valida_los_bloques():
    matriz = _matriz_reposo(n_personas=10)
    X, y, g = _xyg(matriz)
    with pytest.raises(ValueError, match="no existen"):
        importancia_permutacion_cv(X, y, g, _logistica(), bloques={"a": ["nada"]})
    with pytest.raises(ValueError, match="vacío"):
        importancia_permutacion_cv(X, y, g, _logistica(), bloques={"a": []})


def test_la_importancia_por_permutacion_es_reproducible_con_la_misma_semilla():
    matriz = _matriz_reposo(n_personas=20)
    X, y, g = _xyg(matriz)
    a = importancia_permutacion_cv(X, y, g, _logistica(), n_repeticiones=3, semilla=7)
    b = importancia_permutacion_cv(X, y, g, _logistica(), n_repeticiones=3, semilla=7)
    pd.testing.assert_frame_equal(a, b)


def test_los_coeficientes_por_pliegue_dan_el_signo_del_efecto():
    """La importancia no tiene signo; el coeficiente sí. Con OC más alto en la
    feature, el coeficiente tiene que ser positivo en todos los pliegues."""
    matriz = _matriz_reposo(n_personas=30, separacion=2.0)
    X, y, g = _xyg(matriz)
    coef = coeficientes_por_pliegue(X, y, g, _logistica())
    assert coef.shape == (3, 5)
    assert list(coef.index) == list(X.columns)
    assert (coef.loc["posterior_alfa_rel_media"] > 0).all()


@pytest.mark.filterwarnings("ignore:Skipping features without any observed values")
def test_los_coeficientes_fallan_si_un_paso_previo_descarta_columnas():
    matriz = _matriz_reposo(n_personas=20)
    X, y, g = _xyg(matriz)
    X = X.assign(vacia=np.nan)
    with pytest.raises(ValueError, match="descartó"):
        coeficientes_por_pliegue(X, y, g, _logistica())


def test_tamanos_de_efecto_tienen_el_signo_de_la_clase_positiva():
    base = np.tile([-1.0, 1.0], 50)
    y = np.array([0] * 100 + [1] * 100)
    X = pd.DataFrame({
        "sube": np.concatenate([base, base + 1.0]),
        "baja": np.concatenate([base, base - 1.0]),
    })
    d = tamanos_de_efecto(X, y)
    esperado = 1.0 / np.std(base, ddof=1)
    assert d["sube"] == pytest.approx(esperado)
    assert d["baja"] == pytest.approx(-esperado)


def test_tamanos_de_efecto_ignoran_nan_y_dan_nan_sin_varianza():
    base = np.tile([-1.0, 1.0], 50)
    y = np.array([0] * 100 + [1] * 100)
    limpio = np.concatenate([base, base + 1.0])
    con_nan = limpio.copy()
    con_nan[[3, 150]] = np.nan
    X = pd.DataFrame({"limpio": limpio, "con_nan": con_nan,
                      "constante": np.ones(200)})
    d = tamanos_de_efecto(X, y)
    assert d["con_nan"] == pytest.approx(d["limpio"], rel=0.05)
    assert np.isnan(d["constante"])


def test_tamanos_de_efecto_exigen_dos_observaciones_por_clase():
    with pytest.raises(ValueError, match="al menos 2"):
        tamanos_de_efecto(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), np.array([0, 0, 1]))


# --------------------------------------------------------------------------
# Controles e incertidumbre
# --------------------------------------------------------------------------

def test_permutar_etiquetas_conserva_las_clases_de_cada_persona():
    matriz = _matriz_reposo(n_personas=10)
    _, y, g = _xyg(matriz)
    barajada = permutar_etiquetas_dentro_de_persona(y, g, semilla=1)
    for persona in np.unique(g):
        en = g == persona
        assert barajada[en].sum() == y[en].sum()
    assert not np.array_equal(barajada, y), "tiene que cambiar algo"


def test_permutar_etiquetas_es_reproducible_y_depende_de_la_semilla():
    matriz = _matriz_reposo(n_personas=10)
    _, y, g = _xyg(matriz)
    a = permutar_etiquetas_dentro_de_persona(y, g, semilla=1)
    assert np.array_equal(a, permutar_etiquetas_dentro_de_persona(y, g, semilla=1))
    assert not np.array_equal(a, permutar_etiquetas_dentro_de_persona(y, g, semilla=2))


def test_con_las_etiquetas_barajadas_dentro_de_persona_la_cv_devuelve_el_azar():
    """
    El control negativo del análisis. La misma feature con señal real da > 0.8
    con las etiquetas verdaderas y ~0.5 con las barajadas dentro de cada
    persona. Si diera más, habría una fuga en el propio montaje y ningún
    resultado posterior valdría.
    """
    matriz = _matriz_reposo(n_personas=40, n_sesiones=4, separacion=2.5)
    X, y, g = _xyg(matriz)
    real = evaluar_reposo(X, y, g, _logistica())["metricas"]["exactitud_balanceada"]
    nulas = [
        evaluar_reposo(X, permutar_etiquetas_dentro_de_persona(y, g, semilla=s), g,
                       _logistica())["metricas"]["exactitud_balanceada"]
        for s in range(5)
    ]
    assert real > 0.8
    assert abs(np.mean(nulas) - 0.5) < 0.05
    assert max(nulas) < 0.62


def test_el_bootstrap_remuestrea_personas_no_filas():
    """
    Con dos personas, una que el modelo acierta siempre y otra que falla siempre,
    remuestrear personas solo puede dar tres exactitudes (0, 0.5, 1). Remuestrear
    filas daría un continuo alrededor de 0.5.
    """
    g = np.array(["A"] * 10 + ["B"] * 10)
    y = np.tile([0, 1], 10)
    pred = np.where(g == "A", y, 1 - y)
    exactitud = lambda t, p: float(np.mean(t == p))
    r = bootstrap_por_persona(y, pred, g, exactitud, n_replicas=200, semilla=0)
    assert set(np.round(r["replicas"], 6)) <= {0.0, 0.5, 1.0}
    assert len(set(np.round(r["replicas"], 6))) == 3


def test_el_bootstrap_es_reproducible_y_el_ic_contiene_la_estimacion():
    matriz = _matriz_reposo(n_personas=30, separacion=2.0)
    X, y, g = _xyg(matriz)
    r = evaluar_reposo(X, y, g, _logistica())
    metrica = lambda t, p: metricas_binarias(t, p)["exactitud_balanceada"]
    a = bootstrap_por_persona(y, r["y_pred"], g, metrica, n_replicas=300, semilla=3)
    b = bootstrap_por_persona(y, r["y_pred"], g, metrica, n_replicas=300, semilla=3)
    assert a["ic95"] == b["ic95"]
    estimacion = r["metricas"]["exactitud_balanceada"]
    assert a["ic95"][0] <= estimacion <= a["ic95"][1]
    assert a["n_replicas"] + a["n_descartadas"] == 300


def test_el_bootstrap_cuenta_las_replicas_donde_la_metrica_no_esta_definida():
    """Con una persona por clase, muchas réplicas dejan una sola clase y la
    exactitud balanceada no existe: se cuentan aparte, no se ocultan."""
    g = np.array(["A"] * 6 + ["B"] * 6)
    y = np.array([0] * 6 + [1] * 6)
    metrica = lambda t, p: metricas_binarias(t, p)["exactitud_balanceada"]
    r = bootstrap_por_persona(y, y, g, metrica, n_replicas=100, semilla=0)
    assert r["n_descartadas"] > 0
    assert r["n_replicas"] + r["n_descartadas"] == 100


def test_el_bootstrap_descarta_las_replicas_donde_la_metrica_da_nan():
    """Un NaN no es un valor: si se colara, los percentiles saldrían NaN."""
    # La persona A suma 1 (impar) y la B suma 8 (par): la réplica mixta AB suma
    # 9 y da NaN; AA (2) y BB (16) dan un número.
    g = np.array(["A"] * 4 + ["B"] * 4)
    y = np.array([1, 0, 0, 0, 2, 2, 2, 2])
    metrica = lambda t, v: np.nan if t.sum() % 2 else float(t.mean())
    r = bootstrap_por_persona(y, y, g, metrica, n_replicas=100, semilla=0)
    assert r["n_descartadas"] > 0
    assert not np.isnan(r["replicas"]).any()
    assert not np.isnan(r["ic95"]).any()


def test_el_bootstrap_pareado_remuestrea_las_mismas_personas_en_las_dos_ramas():
    """
    A acierta a la persona A y falla a la B; B hace lo contrario. Con la misma
    remuestra en las dos ramas la diferencia solo puede ser -1, 0 o +1. Si cada
    rama remuestreara por su cuenta saldrían también valores intermedios.
    """
    g = np.array(["A"] * 10 + ["B"] * 10)
    y = np.tile([0, 1], 10)
    pred_a = np.where(g == "A", y, 1 - y)
    pred_b = np.where(g == "A", 1 - y, y)
    exactitud = lambda t, p: float(np.mean(t == p))
    r = bootstrap_diferencia_por_persona(y, pred_a, pred_b, g, exactitud,
                                         n_replicas=300, semilla=0)
    assert set(np.round(r["replicas"], 6)) <= {-1.0, 0.0, 1.0}
    assert r["diferencia"] == pytest.approx(0.0)


def test_el_bootstrap_pareado_da_cero_si_los_dos_modelos_son_el_mismo():
    matriz = _matriz_reposo(n_personas=20)
    X, y, g = _xyg(matriz)
    pred = evaluar_reposo(X, y, g, _logistica())["y_pred"]
    metrica = lambda t, p: metricas_binarias(t, p)["exactitud_balanceada"]
    r = bootstrap_diferencia_por_persona(y, pred, pred, g, metrica,
                                         n_replicas=100, semilla=0)
    assert r["diferencia"] == 0.0
    assert r["ic95"] == (0.0, 0.0)
    assert r["prop_positivas"] == 0.0


def test_el_bootstrap_pareado_detecta_un_modelo_claramente_mejor():
    """La logística con señal frente a un clasificador constante: el IC de la
    diferencia excluye el 0 y casi todas las réplicas favorecen a la logística."""
    matriz = _matriz_reposo(n_personas=30, separacion=3.0)
    X, y, g = _xyg(matriz)
    buena = evaluar_reposo(X, y, g, _logistica())["y_pred"]
    constante = evaluar_reposo(X, y, g, DummyClassifier(strategy="most_frequent"))["y_pred"]
    metrica = lambda t, p: metricas_binarias(t, p)["exactitud_balanceada"]
    r = bootstrap_diferencia_por_persona(y, buena, constante, g, metrica,
                                         n_replicas=300, semilla=1)
    assert r["diferencia"] > 0.25
    assert r["ic95"][0] > 0.0
    assert r["prop_positivas"] > 0.99


def test_el_bootstrap_pareado_valida_las_longitudes():
    with pytest.raises(ValueError, match="misma longitud"):
        bootstrap_diferencia_por_persona(
            np.array([0, 1, 0]), np.array([0, 1, 0]), np.array([0, 1]),
            np.array(["A", "A", "B"]), lambda t, p: 0.0)


def test_columnas_duplicadas_encuentra_las_copias_exactas():
    """En la matriz de la Fase 13 `posterior_alfa_logabs_media` y
    `alfa_posterior_log_media` son la misma columna con dos nombres."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=50)
    X = pd.DataFrame({"a": a, "b": rng.normal(size=50), "a_copia": a.copy(),
                      "otra": rng.normal(size=50)})
    assert columnas_duplicadas(X) == [("a", "a_copia")]


def test_columnas_duplicadas_exige_que_los_nan_coincidan():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    igual = pd.DataFrame({"a": a, "b": a.copy()})
    distinta = pd.DataFrame({"a": a, "b": np.array([1.0, 2.0, 3.0, 4.0])})
    assert columnas_duplicadas(igual) == [("a", "b")]
    assert columnas_duplicadas(distinta) == []


def test_columnas_duplicadas_no_confunde_columnas_casi_iguales():
    a = np.linspace(0, 1, 30)
    X = pd.DataFrame({"a": a, "casi": a + 1e-9, "correlada": a * 2})
    assert columnas_duplicadas(X) == []


def test_columnas_duplicadas_agrupa_varias_copias_bajo_la_primera():
    a = np.arange(10.0)
    X = pd.DataFrame({"a": a, "b": a.copy(), "c": a.copy()})
    assert columnas_duplicadas(X) == [("a", "b"), ("a", "c")]
