"""
Tests del banco de evaluación supervisada (Fase 14).

Aquí no se prueba si el Random Forest acierta —eso es un resultado, no un
invariante— sino las reglas que hacen que su cifra signifique algo. Son cuatro,
y todas protegen contra el mismo tipo de fallo: un R² alto obtenido por un
atajo en vez de por la señal.

1. La media de referencia de una persona nunca incluye la sesión que se está
   prediciendo (`medias_loo_por_grupo`).
2. La validación cruzada reparte **personas**, no sesiones: hay un test que
   monta un dataset donde la etiqueta es pura identidad de la persona y
   comprueba que agrupar la vuelve impredecible mientras que no agrupar la
   regala.
3. Centrar por persona elimina el eje entre-personas de X y de y a la vez.
4. `condicion_orden` no sobrevive a la agregación por sesión.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold

from eeg_library.modeling import (
    COLUMNA_ETIQUETA,
    COLUMNA_GRUPO,
    a_clases,
    agregar_a_sesion,
    centrar_por_persona_loo,
    columnas_de_features_sesion,
    cortes_terciles,
    lineas_base,
    medias_loo_por_grupo,
    metricas,
    metricas_clasificacion,
    metricas_clasificacion_por_pliegue,
    predecir_clases_cv_agrupada,
    predecir_cv_agrupada,
)


def _matriz(filas):
    """
    Matriz por grabación mínima con la forma que deja la Fase 13.

    `filas` es una lista de dicts con al menos sesion_uid, persona_uid, vas y
    las features que interese fijar; el resto se rellena con valores neutros.
    """
    base = {
        "archivo": None, "sesion_uid": None, "persona_uid": None,
        "sujeto_id": None, "centro": "C1", "medicion": "M2", "condicion": "OC",
        "split": "train", "etiquetada": True, "concordancia_fecha": "coincide",
        "corregida": False, "ocupante": 1, "procesada": True,
        "vas": np.nan, "frontal_alfa_rel_sd": 0.0, "frontal_theta_rel_media": 0.0,
        "condicion_orden": 1.0, "n_ventanas": 60.0, "n_canales_buenos": 14.0,
    }
    completas = []
    for i, fila in enumerate(filas):
        completa = dict(base)
        completa.update(fila)
        if completa["archivo"] is None:
            completa["archivo"] = f"rec{i:03d}.edf"
        if completa["sujeto_id"] is None:
            completa["sujeto_id"] = completa["persona_uid"]
        completas.append(completa)
    return pd.DataFrame(completas)


def _sesiones_sinteticas(n_personas=20, n_sesiones=4, senal="persona", semilla=0):
    """
    Dataset a nivel de sesión con una estructura conocida.

    `senal="persona"`: la feature es una **firma** propia de cada persona,
    sorteada con independencia de su nivel de VAS. Identifica a quién se está
    midiendo pero no dice nada sobre cuánta fatiga tiene, que es justo lo que
    ocurre con unas features que reidentifican al participante. Reconocer la
    firma solo sirve si ya se vio a esa persona etiquetada.
    `senal="intra"`: el VAS de cada sesión es el nivel de la persona más una
    desviación que **sí** está en la feature.
    """
    rng = np.random.default_rng(semilla)
    filas = []
    for p in range(n_personas):
        nivel = rng.uniform(10, 90)
        firma = rng.uniform(0, 1000)
        for s in range(n_sesiones):
            desviacion = rng.normal(0, 10)
            if senal == "persona":
                vas = nivel + rng.normal(0, 1)
                x = firma + rng.normal(0, 0.5)
            else:
                vas = nivel + desviacion
                x = desviacion + rng.normal(0, 0.5)
            filas.append({
                "sesion_uid": f"P{p:02d}_M{s}", "persona_uid": f"P{p:02d}",
                "medicion": f"M{s}", "centro": "C1", "split": "train",
                "vas": vas, "x": x, "n_grabaciones": 7,
            })
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# medias_loo_por_grupo — el núcleo anti-circularidad
# --------------------------------------------------------------------------

def test_la_media_loo_excluye_a_la_propia_fila():
    """
    Si la media de una persona incluyese la sesión que se predice, predecir
    sería copiar. Con tres sesiones de valores 0, 30 y 60, la referencia de la
    primera tiene que ser 45 (media de las otras dos), no 30.
    """
    datos = pd.Series([0.0, 30.0, 60.0])
    loo = medias_loo_por_grupo(datos, ["A", "A", "A"])
    assert list(loo) == [45.0, 30.0, 15.0]


def test_la_media_loo_es_nan_sin_otra_observacion():
    """
    Una persona con una sola sesión no tiene nivel habitual estimable. Inventar
    uno (por ejemplo, su propio valor) daría un acierto perfecto y falso.
    """
    loo = medias_loo_por_grupo(pd.Series([50.0, 10.0, 20.0]), ["A", "B", "B"])
    assert pd.isna(loo.iloc[0])
    assert loo.iloc[1] == 20.0 and loo.iloc[2] == 10.0


def test_la_media_loo_no_mezcla_grupos():
    loo = medias_loo_por_grupo(pd.Series([0.0, 100.0, 10.0, 20.0]),
                               ["A", "A", "B", "B"])
    assert list(loo) == [100.0, 0.0, 20.0, 10.0]


def test_la_media_loo_ignora_los_nan_como_mean():
    """
    Una fila con NaN no aporta al promedio pero sí recibe uno: el de todas las
    demás de su grupo. Y una fila con valor no debe verse penalizada porque otra
    del grupo esté a NaN.
    """
    loo = medias_loo_por_grupo(pd.Series([10.0, np.nan, 30.0]), ["A", "A", "A"])
    assert loo.iloc[0] == 30.0          # la otra observación válida
    assert loo.iloc[1] == 20.0          # media de las dos válidas
    assert loo.iloc[2] == 10.0


def test_la_media_loo_acepta_dataframe_columna_a_columna():
    df = pd.DataFrame({"a": [0.0, 30.0, 60.0], "b": [1.0, 2.0, 3.0]})
    loo = medias_loo_por_grupo(df, ["G", "G", "G"])
    assert list(loo["a"]) == [45.0, 30.0, 15.0]
    assert list(loo["b"]) == [2.5, 2.0, 1.5]


# --------------------------------------------------------------------------
# predecir_cv_agrupada — la regresión de fuga de datos
# --------------------------------------------------------------------------

def test_agrupar_por_persona_impide_predecir_lo_que_es_pura_identidad():
    """
    El test central del módulo.

    Se construye un dataset donde el VAS es **solo** el nivel de la persona y la
    feature lo delata: no hay ninguna señal intra-persona que aprender. Un
    reparto aleatorio por sesión deja otras sesiones de la misma persona en
    train, así que el modelo la reconoce y "acierta" con R² alto. Agrupando por
    persona, el mismo modelo con los mismos datos no puede: R² se desploma.

    Es exactamente el escenario real —estas features identifican a la persona
    con acierto 0.383 sobre 129 clases (Fase 13)— y la razón de que toda la
    Fase 14 se evalúe con GroupKFold.
    """
    ses = _sesiones_sinteticas(n_personas=20, n_sesiones=4, senal="persona")
    X, y, g = ses[["x"]], ses["vas"].to_numpy(), ses[COLUMNA_GRUPO].to_numpy()
    modelo = RandomForestRegressor(n_estimators=50, random_state=0)

    pred_agrupada, _ = predecir_cv_agrupada(X, y, g, modelo, n_splits=5)
    r2_agrupada = r2_score(y, pred_agrupada)

    pred_ingenua = np.full(len(y), np.nan)
    for idx_tr, idx_va in KFold(5, shuffle=True, random_state=0).split(X):
        ajustado = RandomForestRegressor(n_estimators=50, random_state=0)
        ajustado.fit(X.iloc[idx_tr], y[idx_tr])
        pred_ingenua[idx_va] = ajustado.predict(X.iloc[idx_va])
    r2_ingenua = r2_score(y, pred_ingenua)

    assert r2_ingenua > 0.9, "sin agrupar, la identidad se filtra y el R² se dispara"
    assert r2_agrupada < 0.5, "agrupando no debería quedar casi nada que predecir"
    assert r2_ingenua - r2_agrupada > 0.4


def test_la_cv_agrupada_recupera_la_senal_intra_persona():
    """
    Contrapunto del anterior: si la señal está de verdad en la feature y no en
    la identidad, agrupar por persona no la destruye. Si no, el test anterior
    podría estar pasando porque la función simplemente no predice nada.
    """
    ses = _sesiones_sinteticas(n_personas=20, n_sesiones=4, senal="intra")
    X, y, g = ses[["x"]], ses["vas"].to_numpy(), ses[COLUMNA_GRUPO].to_numpy()
    centradas = centrar_por_persona_loo(ses, ["x", COLUMNA_ETIQUETA])
    pred, _ = predecir_cv_agrupada(
        centradas[["x"]], centradas[COLUMNA_ETIQUETA].to_numpy(), g,
        RandomForestRegressor(n_estimators=50, random_state=0), n_splits=5)
    assert r2_score(centradas[COLUMNA_ETIQUETA], pred) > 0.3


def test_ninguna_persona_aparece_en_train_y_validacion():
    ses = _sesiones_sinteticas(n_personas=15, n_sesiones=3)
    _, detalle = predecir_cv_agrupada(
        ses[["x"]], ses["vas"].to_numpy(), ses[COLUMNA_GRUPO].to_numpy(),
        RandomForestRegressor(n_estimators=10, random_state=0), n_splits=5)
    assert detalle["n_grupos"] == 15
    assert sum(p["personas_val"] for p in detalle["reparto"]) == 15


def test_todas_las_filas_reciben_prediccion_out_of_fold():
    ses = _sesiones_sinteticas(n_personas=12, n_sesiones=3)
    pred, _ = predecir_cv_agrupada(
        ses[["x"]], ses["vas"].to_numpy(), ses[COLUMNA_GRUPO].to_numpy(),
        RandomForestRegressor(n_estimators=10, random_state=0), n_splits=4)
    assert not np.isnan(pred).any()


def test_error_si_hay_menos_grupos_que_pliegues():
    ses = _sesiones_sinteticas(n_personas=3, n_sesiones=2)
    with pytest.raises(ValueError, match="pliegues"):
        predecir_cv_agrupada(
            ses[["x"]], ses["vas"].to_numpy(), ses[COLUMNA_GRUPO].to_numpy(),
            RandomForestRegressor(n_estimators=10, random_state=0), n_splits=5)


# --------------------------------------------------------------------------
# centrar_por_persona_loo
# --------------------------------------------------------------------------

def test_centrar_elimina_el_eje_entre_personas():
    """
    Tras centrar, dos personas con niveles muy distintos pero la misma
    trayectoria interna tienen que quedar indistinguibles.
    """
    ses = pd.DataFrame({
        "sesion_uid": ["A_1", "A_2", "B_1", "B_2"],
        "persona_uid": ["A", "A", "B", "B"],
        "vas": [10.0, 30.0, 70.0, 90.0],
    })
    centradas = centrar_por_persona_loo(ses, [COLUMNA_ETIQUETA])
    assert list(centradas[COLUMNA_ETIQUETA]) == [-20.0, 20.0, -20.0, 20.0]


def test_centrar_conserva_la_referencia_usada():
    ses = pd.DataFrame({
        "persona_uid": ["A", "A", "A"],
        "vas": [0.0, 30.0, 60.0],
    })
    centradas = centrar_por_persona_loo(ses, [COLUMNA_ETIQUETA])
    assert list(centradas["vas_loo"]) == [45.0, 30.0, 15.0]


def test_centrar_deja_nan_a_quien_tiene_una_sola_sesion():
    ses = pd.DataFrame({"persona_uid": ["A", "B", "B"], "vas": [50.0, 10.0, 20.0]})
    centradas = centrar_por_persona_loo(ses, [COLUMNA_ETIQUETA])
    assert pd.isna(centradas[COLUMNA_ETIQUETA].iloc[0])
    assert not centradas[COLUMNA_ETIQUETA].iloc[1:].isna().any()


# --------------------------------------------------------------------------
# agregar_a_sesion
# --------------------------------------------------------------------------

def test_agregar_a_sesion_promedia_las_grabaciones():
    matriz = _matriz([
        {"sesion_uid": "S1", "persona_uid": "P1", "vas": 40.0, "frontal_alfa_rel_sd": 0.2},
        {"sesion_uid": "S1", "persona_uid": "P1", "vas": 40.0, "frontal_alfa_rel_sd": 0.4},
        {"sesion_uid": "S2", "persona_uid": "P1", "vas": 60.0, "frontal_alfa_rel_sd": 1.0},
    ])
    ses = agregar_a_sesion(matriz)
    assert len(ses) == 2
    s1 = ses[ses["sesion_uid"] == "S1"].iloc[0]
    assert s1["frontal_alfa_rel_sd"] == pytest.approx(0.3)
    assert s1["n_grabaciones"] == 2
    assert s1[COLUMNA_ETIQUETA] == 40.0


def test_agregar_a_sesion_descarta_condicion_orden():
    """
    Promediada sobre las 7 grabaciones vale ~4 en todas las sesiones completas,
    y en las incompletas delata qué grabaciones faltan. Deja de ser feature.
    """
    matriz = _matriz([
        {"sesion_uid": "S1", "persona_uid": "P1", "vas": 40.0, "condicion_orden": 1.0},
        {"sesion_uid": "S1", "persona_uid": "P1", "vas": 40.0, "condicion_orden": 7.0},
    ])
    ses = agregar_a_sesion(matriz)
    assert "condicion_orden" not in ses.columns
    assert "condicion_orden" not in columnas_de_features_sesion(ses)


def test_agregar_a_sesion_ignora_nan_al_promediar():
    """
    Una región sin canales válidos en *una* grabación no debe anular el valor de
    la sesión entera; solo si falta en todas.
    """
    matriz = _matriz([
        {"sesion_uid": "S1", "persona_uid": "P1", "vas": 40.0, "frontal_alfa_rel_sd": 0.2},
        {"sesion_uid": "S1", "persona_uid": "P1", "vas": 40.0, "frontal_alfa_rel_sd": np.nan},
        {"sesion_uid": "S2", "persona_uid": "P1", "vas": 50.0, "frontal_alfa_rel_sd": np.nan},
    ])
    ses = agregar_a_sesion(matriz).set_index("sesion_uid")
    assert ses.loc["S1", "frontal_alfa_rel_sd"] == pytest.approx(0.2)
    assert pd.isna(ses.loc["S2", "frontal_alfa_rel_sd"])


def test_agregar_a_sesion_descarta_las_no_etiquetadas():
    matriz = _matriz([
        {"sesion_uid": "S1", "persona_uid": "P1", "vas": 40.0},
        {"sesion_uid": "S2", "persona_uid": np.nan, "vas": np.nan, "etiquetada": False},
    ])
    ses = agregar_a_sesion(matriz)
    assert list(ses["sesion_uid"]) == ["S1"]


def test_agregar_a_sesion_falla_si_la_identidad_no_es_constante():
    """
    Dos personas distintas bajo la misma sesion_uid significaría que el enlace
    de la Fase 12 está roto; agregarlo con `first` lo escondería.
    """
    matriz = _matriz([
        {"sesion_uid": "S1", "persona_uid": "P1", "vas": 40.0},
        {"sesion_uid": "S1", "persona_uid": "P2", "vas": 40.0},
    ])
    with pytest.raises(ValueError, match="persona_uid"):
        agregar_a_sesion(matriz)


def test_columnas_de_features_sesion_excluye_identidad_y_etiqueta():
    matriz = _matriz([
        {"sesion_uid": "S1", "persona_uid": "P1", "vas": 40.0},
        {"sesion_uid": "S2", "persona_uid": "P1", "vas": 60.0},
    ])
    ses = agregar_a_sesion(matriz)
    features = columnas_de_features_sesion(ses)
    for prohibida in ("vas", "sesion_uid", "persona_uid", "split", "medicion",
                      "centro", "sujeto_id", "condicion_orden"):
        assert prohibida not in features
    assert "frontal_alfa_rel_sd" in features


# --------------------------------------------------------------------------
# lineas_base y metricas
# --------------------------------------------------------------------------

def test_la_media_global_da_r2_cero_por_definicion():
    ses = _sesiones_sinteticas(n_personas=10, n_sesiones=3)
    base = lineas_base(ses).set_index("modelo")
    assert base.loc["media global", "r2"] == pytest.approx(0.0, abs=1e-12)


def test_la_media_por_persona_en_muestra_infla_frente_a_la_loo():
    """
    Es la trampa que documenta la Fase 12: la media en muestra sube el R² de
    0.260 a 0.600 sobre los datos reales. Aquí basta con comprobar el sentido.
    """
    ses = _sesiones_sinteticas(n_personas=15, n_sesiones=3, senal="intra")
    base = lineas_base(ses).set_index("modelo")
    en_muestra = base.loc[f"media por {COLUMNA_GRUPO} (en muestra, circular)", "r2"]
    loo = base.loc[f"media por {COLUMNA_GRUPO} (leave-one-out)", "r2"]
    assert en_muestra > loo


def test_metricas_de_una_prediccion_perfecta():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    m = metricas(y, y)
    assert m["r2"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["spearman"] == pytest.approx(1.0)
    assert m["n"] == 4


def test_metricas_descarta_las_filas_con_nan():
    m = metricas([10.0, 20.0, np.nan, 40.0], [10.0, 20.0, 30.0, 40.0])
    assert m["n"] == 3
    assert m["r2"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Clasificación ordinal en tres niveles (Fase 14b)
#
# El invariante que se protege aquí es el mismo de siempre, aplicado a la
# discretización: los cortes de los tercios no pueden mirar a las personas que
# se van a evaluar. Si se calculan sobre todo el conjunto, la distribución de la
# etiqueta de validación se filtra al entrenamiento por la puerta de atrás.
# --------------------------------------------------------------------------

def test_cortes_terciles_parte_en_tres_grupos_iguales():
    """Con 0..98 los cortes deben caer cerca de 32.7 y 65.3, y repartir 1/3."""
    y = np.arange(99, dtype=float)
    cortes = cortes_terciles(y)
    assert cortes[0] < cortes[1]
    clases = a_clases(y, cortes)
    conteo = pd.Series(clases).value_counts().sort_index().to_numpy()
    assert (np.abs(conteo - 33) <= 1).all(), f"reparto desigual: {conteo}"


def test_cortes_terciles_rechaza_una_etiqueta_constante():
    """Sin variación no hay tres niveles que separar: mejor fallar que inventar."""
    with pytest.raises(ValueError, match="concentrada"):
        cortes_terciles(np.full(50, 42.0))
    with pytest.raises(ValueError, match="al menos 3"):
        cortes_terciles([1.0, 2.0])


def test_cortes_terciles_ignora_los_nan():
    """Un NaN no debe desplazar los percentiles."""
    y = np.arange(99, dtype=float)
    con_nan = np.concatenate([y, [np.nan] * 20])
    assert cortes_terciles(con_nan) == pytest.approx(cortes_terciles(y))


def test_a_clases_asigna_y_propaga_nan():
    """Frontera a la clase superior; NaN sigue siendo NaN, no clase 0."""
    clases = a_clases([0.0, 10.0, 20.0, 30.0, np.nan], cortes=[10.0, 20.0])
    assert clases[0] == 0
    assert clases[1] == 1, "el valor igual al corte sube de clase"
    assert clases[2] == 2
    assert clases[3] == 2
    assert np.isnan(clases[4]), "un NaN no puede convertirse en 'poca fatiga'"


def test_a_clases_exige_dos_cortes():
    with pytest.raises(ValueError, match="2 cortes"):
        a_clases([1.0, 2.0, 3.0], cortes=[1.0, 2.0, 3.0])


def test_metricas_clasificacion_reconoce_el_acierto_perfecto():
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=float)
    m = metricas_clasificacion(y, y)
    assert m["exactitud_balanceada"] == pytest.approx(1.0)
    assert m["mae_ordinal"] == pytest.approx(0.0)
    assert m["kendall_tau"] == pytest.approx(1.0)


def test_metricas_clasificacion_castiga_la_clase_mayoritaria():
    """
    Predecir siempre la clase mayoritaria da una exactitud engañosamente alta;
    la exactitud balanceada la deja en 1/3, que es el azar. Por eso es la
    métrica principal.
    """
    y = np.array([0] * 60 + [1] * 20 + [2] * 20, dtype=float)
    constante = np.zeros(100)
    m = metricas_clasificacion(y, constante)
    assert m["exactitud"] == pytest.approx(0.60)
    assert m["exactitud_balanceada"] == pytest.approx(1 / 3)
    assert np.isnan(m["kendall_tau"]), "sin variación el tau está indefinido"


@pytest.mark.filterwarnings("ignore:y_pred contains classes not in y_true")
def test_metricas_clasificacion_distingue_fallo_leve_de_fallo_grave():
    """Confundir poca con media no es lo mismo que confundir poca con mucha."""
    y = np.array([0, 0, 0, 0], dtype=float)
    leve = metricas_clasificacion(y, np.array([1, 1, 1, 1], dtype=float))
    grave = metricas_clasificacion(y, np.array([2, 2, 2, 2], dtype=float))
    assert leve["exactitud"] == grave["exactitud"] == 0.0
    assert leve["exactitud_adyacente"] == 1.0
    assert grave["exactitud_adyacente"] == 0.0
    assert leve["mae_ordinal"] < grave["mae_ordinal"]


def test_los_cortes_salen_solo_del_pliegue_de_entrenamiento():
    """
    El invariante central de esta parte. Se comprueba comparando los cortes que
    reporta cada pliegue con los percentiles del train de ese pliegue (deben
    coincidir) y con los del conjunto entero (deben diferir, porque hay una
    persona con valores extremos que entra y sale de la referencia).
    """
    from sklearn.dummy import DummyClassifier

    ses = _sesiones_sinteticas(n_personas=20, n_sesiones=4, senal="intra")
    # Una persona con VAS disparado: mueve los percentiles globales, y solo debe
    # moverlos en los pliegues donde está.
    ses.loc[ses[COLUMNA_GRUPO] == "P00", COLUMNA_ETIQUETA] = 500.0
    X, y = ses[["x"]], ses[COLUMNA_ETIQUETA].to_numpy()
    g = ses[COLUMNA_GRUPO].to_numpy()

    _, _, detalle = predecir_clases_cv_agrupada(
        X, y, g, DummyClassifier(strategy="most_frequent"), n_splits=5)

    cortes_globales = cortes_terciles(y)
    cv = GroupKFold(n_splits=5)
    for info, (idx_tr, _) in zip(detalle["reparto"], cv.split(X, y, groups=g)):
        esperados = cortes_terciles(y[idx_tr])
        assert info["cortes"] == pytest.approx(esperados)
    distintos = [i for i, info in enumerate(detalle["reparto"])
                 if info["cortes"] != pytest.approx(cortes_globales)]
    assert distintos, "ningún pliegue difiere del global: no se están recalculando"


def test_las_clases_cv_no_comparten_personas_entre_train_y_validacion():
    from sklearn.dummy import DummyClassifier

    ses = _sesiones_sinteticas(n_personas=20, n_sesiones=4, senal="intra")
    X, y = ses[["x"]], ses[COLUMNA_ETIQUETA].to_numpy()
    g = ses[COLUMNA_GRUPO].to_numpy()
    true, pred, detalle = predecir_clases_cv_agrupada(
        X, y, g, DummyClassifier(strategy="most_frequent"), n_splits=5)

    assert not np.isnan(true).any(), "toda sesión debe recibir clase real"
    assert not np.isnan(pred).any(), "toda sesión debe recibir predicción"
    assert sum(i["personas_val"] for i in detalle["reparto"]) == 20
    assert detalle["n_grupos"] == 20


def test_la_clasificacion_recupera_una_senal_intra_persona_real():
    """
    Con `senal="intra"` la feature *es* la desviación del VAS, así que un
    clasificador honesto debe superar con holgura el azar (1/3). Es el control
    positivo: si esto no pasara, el arnés estaría roto y un resultado negativo
    sobre datos reales no significaría nada.
    """
    from sklearn.ensemble import RandomForestClassifier

    ses = _sesiones_sinteticas(n_personas=40, n_sesiones=4, senal="intra")
    columnas = ["x", COLUMNA_ETIQUETA]
    cen = centrar_por_persona_loo(ses, columnas)
    X, y = cen[["x"]], cen[COLUMNA_ETIQUETA].to_numpy()
    g = cen[COLUMNA_GRUPO].to_numpy()

    true, pred, _ = predecir_clases_cv_agrupada(
        X, y, g, RandomForestClassifier(n_estimators=100, random_state=0))
    m = metricas_clasificacion(true, pred)
    assert m["exactitud_balanceada"] > 0.6, (
        f"control positivo fallido: {m['exactitud_balanceada']:.3f}")


def test_la_clasificacion_no_recupera_lo_que_es_pura_identidad():
    """
    Con `senal="persona"` la feature identifica a la persona pero no dice nada
    de su fatiga. Agrupando, el clasificador no puede pasar del azar.
    """
    from sklearn.ensemble import RandomForestClassifier

    ses = _sesiones_sinteticas(n_personas=40, n_sesiones=4, senal="persona")
    X, y = ses[["x"]], ses[COLUMNA_ETIQUETA].to_numpy()
    g = ses[COLUMNA_GRUPO].to_numpy()
    true, pred, _ = predecir_clases_cv_agrupada(
        X, y, g, RandomForestClassifier(n_estimators=100, random_state=0))
    m = metricas_clasificacion(true, pred)
    assert m["exactitud_balanceada"] < 0.45, (
        f"la identidad no debería predecir el nivel: "
        f"{m['exactitud_balanceada']:.3f}")


def test_modo_regresion_discretiza_con_los_mismos_cortes():
    """
    En modo `regresion` el modelo se entrena sobre el continuo y su predicción
    se corta con los umbrales del pliegue. Sobre una señal intra fuerte tiene
    que funcionar igual de bien que el clasificador.
    """
    from sklearn.ensemble import RandomForestRegressor as RFR

    ses = _sesiones_sinteticas(n_personas=40, n_sesiones=4, senal="intra")
    cen = centrar_por_persona_loo(ses, ["x", COLUMNA_ETIQUETA])
    X, y = cen[["x"]], cen[COLUMNA_ETIQUETA].to_numpy()
    g = cen[COLUMNA_GRUPO].to_numpy()
    true, pred, detalle = predecir_clases_cv_agrupada(
        X, y, g, RFR(n_estimators=100, random_state=0), modo="regresion")
    assert detalle["modo"] == "regresion"
    assert metricas_clasificacion(true, pred)["exactitud_balanceada"] > 0.6


def test_modo_desconocido_falla():
    from sklearn.dummy import DummyClassifier

    ses = _sesiones_sinteticas(n_personas=10, n_sesiones=3)
    with pytest.raises(ValueError, match="clasificacion"):
        predecir_clases_cv_agrupada(
            ses[["x"]], ses[COLUMNA_ETIQUETA].to_numpy(),
            ses[COLUMNA_GRUPO].to_numpy(), DummyClassifier(), modo="ordinal")


# --------------------------------------------------------------------------
# metricas_clasificacion_por_pliegue — el artefacto de amontonar pliegues
# --------------------------------------------------------------------------

def test_puntuar_por_pliegue_devuelve_el_azar_a_un_clasificador_sin_informacion():
    """
    El test que motiva la existencia de esta función. Un `DummyClassifier` no
    usa X para nada, así que su exactitud balanceada tiene que ser exactamente
    1/3. Amontonando los cinco pliegues sale por encima, porque cada pliegue
    define sus tercios con cortes distintos y la clase 1 de uno no es la clase 1
    de otro. Puntuando dentro de cada pliegue, vuelve al 1/3 que le corresponde.
    """
    from sklearn.dummy import DummyClassifier

    ses = _sesiones_sinteticas(n_personas=40, n_sesiones=4, senal="intra")
    X, y = ses[["x"]], ses[COLUMNA_ETIQUETA].to_numpy()
    g = ses[COLUMNA_GRUPO].to_numpy()
    true, pred, detalle = predecir_clases_cv_agrupada(
        X, y, g, DummyClassifier(strategy="most_frequent"))

    por_pliegue = metricas_clasificacion_por_pliegue(
        true, pred, detalle["pliegue_por_fila"])
    assert por_pliegue["exactitud_balanceada"] == pytest.approx(1 / 3), (
        "un clasificador constante dentro del pliegue solo puede acertar una "
        "clase de tres"
    )
    assert por_pliegue["n_pliegues"] == 5
    assert por_pliegue["n"] == len(ses)


def test_pliegue_por_fila_cubre_todas_las_filas_una_sola_vez():
    """Cada sesión se evalúa exactamente en un pliegue, y en ninguno más."""
    from sklearn.dummy import DummyClassifier

    ses = _sesiones_sinteticas(n_personas=20, n_sesiones=4, senal="intra")
    _, _, detalle = predecir_clases_cv_agrupada(
        ses[["x"]], ses[COLUMNA_ETIQUETA].to_numpy(),
        ses[COLUMNA_GRUPO].to_numpy(), DummyClassifier(), n_splits=5)

    pliegues = detalle["pliegue_por_fila"]
    assert (pliegues >= 1).all(), "hay filas sin pliegue asignado"
    assert sorted(set(pliegues)) == [1, 2, 3, 4, 5]
    # una persona no puede repartirse entre dos pliegues de validación
    por_persona = pd.Series(pliegues).groupby(ses[COLUMNA_GRUPO].to_numpy())
    assert (por_persona.nunique() == 1).all()


def test_puntuar_por_pliegue_exige_longitudes_coherentes():
    with pytest.raises(ValueError, match="misma longitud"):
        metricas_clasificacion_por_pliegue([0, 1, 2], [0, 1], [1, 1, 1])
