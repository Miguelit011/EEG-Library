"""
Tests para la construcción de la matriz de features del Random Forest (Fase 13).

Lo que hay que proteger aquí no es la aritmética del `groupby`, sino las cuatro
decisiones que hacen que esta matriz no sea un `mean()` cualquiera:

1. El promedio de región ignora los canales que `evaluar_calidad` descartó.
2. La potencia absoluta se agrega en log, no en crudo.
3. Una región sin canales válidos da NaN en vez de un número inventado.
4. `medicion` no sale nunca en las columnas de features.
"""

import numpy as np
import pandas as pd
import pytest

from eeg_library.procesado.bands import BANDAS_DEFECTO
from eeg_library.procesado.matrix import (
    COLUMNAS_IDENTIDAD,
    ORDEN_PROTOCOLO,
    REGIONES,
    agregar_por_region,
    columnas_de_features,
    construir_matriz_rf,
    resumir_por_grabacion,
)
from eeg_library.preprocesado.quality import CANALES_EPOC


def _features(archivos_condiciones, n_epocas=4, validos=None, semilla=0):
    """
    Matriz por época como la que deja la Fase 11.5.

    `archivos_condiciones` es una lista de (archivo, condicion). `validos` es
    el conjunto de canales marcados como buenos (por defecto, todos).
    """
    rng = np.random.default_rng(semilla)
    validos = set(CANALES_EPOC if validos is None else validos)
    filas = []
    for archivo, condicion in archivos_condiciones:
        for i in range(n_epocas):
            fila = {
                "archivo_origen": archivo, "condicion": condicion,
                "sujeto": "C1_F050", "sesion": "M2", "indice_ventana": i,
                "theta_frontal": 1e-11, "alfa_posterior": 1e-11,
                "indice_fatiga_theta_alfa": 1.0,
                "n_canales_buenos": len(validos),
            }
            for canal in CANALES_EPOC:
                fila[f"canal_valido_{canal}"] = canal in validos
                for banda in BANDAS_DEFECTO:
                    fila[f"pot_abs_{canal}_{banda}"] = float(rng.uniform(1e-12, 1e-10))
                    fila[f"pot_rel_{canal}_{banda}"] = 0.25
            filas.append(fila)
    return pd.DataFrame(filas)


def _grabaciones(archivos_condiciones, vas=40.0):
    return pd.DataFrame([{
        "archivo": archivo, "sesion_uid": "C1_F050_M2", "persona_uid": "C1_F050",
        "sujeto_id": "C1_F050", "centro": "C1", "medicion": "M2",
        "condicion": condicion, "split": "train", "vas": vas, "etiquetada": True,
        "ocupante": 1, "concordancia_fecha": "coincide", "corregida": False,
        "procesada": True,
    } for archivo, condicion in archivos_condiciones])


# --------------------------------------------------------------------------
# agregar_por_region
# --------------------------------------------------------------------------

def test_el_promedio_de_region_ignora_los_canales_invalidos():
    """
    Es la razón de ser del módulo: un canal descartado trae valores fuera de
    escala (hasta 7.9x, Fase 11.5) y no puede entrar en la media de su región.
    """
    features = _features([("a.edf", "OC")], n_epocas=1)
    # Se contamina AF3 con un valor enorme y se marca como inválido.
    for banda in BANDAS_DEFECTO:
        features.loc[:, f"pot_abs_AF3_{banda}"] = 1e-3
    features.loc[:, "canal_valido_AF3"] = False

    agregado = agregar_por_region(features)
    esperado = np.log10(np.mean([
        features.loc[0, f"pot_abs_{c}_theta"] for c in REGIONES["frontal"] if c != "AF3"
    ]))
    assert agregado.loc[0, "frontal_theta_logabs"] == pytest.approx(esperado)
    assert agregado.loc[0, "n_validos_frontal"] == 5


def test_la_potencia_absoluta_se_agrega_en_logaritmo():
    features = _features([("a.edf", "OC")], n_epocas=1)
    agregado = agregar_por_region(features)
    crudo = np.mean([features.loc[0, f"pot_abs_{c}_alfa"] for c in REGIONES["posterior"]])
    assert agregado.loc[0, "posterior_alfa_logabs"] == pytest.approx(np.log10(crudo))
    # La relativa no se transforma.
    assert agregado.loc[0, "posterior_alfa_rel"] == pytest.approx(0.25)


def test_una_region_sin_canales_validos_da_nan():
    """T7 y T8 son los dos únicos canales de la región temporal."""
    validos = [c for c in CANALES_EPOC if c not in ("T7", "T8")]
    features = _features([("a.edf", "OC")], n_epocas=1, validos=validos)
    agregado = agregar_por_region(features)
    assert agregado.loc[0, "n_validos_temporal"] == 0
    assert np.isnan(agregado.loc[0, "temporal_theta_logabs"])
    assert np.isnan(agregado.loc[0, "temporal_theta_rel"])
    # Las regiones que sí tienen canales no se contagian.
    assert not np.isnan(agregado.loc[0, "frontal_theta_logabs"])


def test_agregar_por_region_admite_un_mapa_por_canal():
    features = _features([("a.edf", "OC")], n_epocas=1)
    agregado = agregar_por_region(features, regiones={"AF3": ("AF3",)})
    esperado = np.log10(features.loc[0, "pot_abs_AF3_beta"])
    assert agregado.loc[0, "AF3_beta_logabs"] == pytest.approx(esperado)


def test_agregar_por_region_exige_archivo_origen():
    features = _features([("a.edf", "OC")], n_epocas=1).drop(columns=["archivo_origen"])
    with pytest.raises(ValueError, match="archivo_origen"):
        agregar_por_region(features)


# --------------------------------------------------------------------------
# construir_matriz_rf
# --------------------------------------------------------------------------

def test_una_fila_por_grabacion_con_media_y_desviacion():
    pares = [("a.edf", "OC"), ("b.edf", "OA")]
    matriz, meta = construir_matriz_rf(_features(pares), _grabaciones(pares))
    assert len(matriz) == 2
    assert set(matriz["archivo"]) == {"a.edf", "b.edf"}
    assert matriz["n_ventanas"].tolist() == [4, 4]
    for sufijo in ("_media", "_sd"):
        assert f"frontal_theta_logabs{sufijo}" in matriz.columns
        assert f"posterior_alfa_rel{sufijo}" in matriz.columns
    assert meta["n_grabaciones"] == 2


def test_la_media_y_la_sd_son_las_de_las_ventanas_de_esa_grabacion():
    pares = [("a.edf", "OC")]
    features = _features(pares, n_epocas=5)
    agregado = agregar_por_region(features)
    matriz, _ = construir_matriz_rf(features, _grabaciones(pares))
    columna = "frontal_theta_logabs"
    assert matriz.loc[0, f"{columna}_media"] == pytest.approx(agregado[columna].mean())
    assert matriz.loc[0, f"{columna}_sd"] == pytest.approx(agregado[columna].std())


def test_una_grabacion_de_una_sola_ventana_tiene_sd_cero_no_nan():
    pares = [("a.edf", "OC")]
    matriz, meta = construir_matriz_rf(_features(pares, n_epocas=1), _grabaciones(pares))
    assert matriz.loc[0, "frontal_theta_logabs_sd"] == 0.0
    assert meta["n_celdas_nan"] == 0


def test_la_condicion_se_codifica_por_su_posicion_en_el_protocolo():
    pares = [(f"{c}.edf", c) for c in ORDEN_PROTOCOLO]
    matriz, _ = construir_matriz_rf(_features(pares), _grabaciones(pares))
    orden = matriz.set_index("condicion")["condicion_orden"]
    # La codificación es la posición en el protocolo, no el orden alfabético
    # ni el de aparición: PVSAT va después de MSIT porque así se graba.
    assert orden["OC"] == 1 and orden["OA"] == 2 and orden["MSIT"] == 6
    assert orden["PVSAT"] == 7
    assert sorted(orden.tolist()) == list(range(1, 8))


def test_una_condicion_desconocida_falla_en_vez_de_colarse_como_nan():
    pares = [("a.edf", "SIESTA")]
    with pytest.raises(ValueError, match="ORDEN_PROTOCOLO"):
        construir_matriz_rf(_features(pares), _grabaciones(pares))


def test_la_etiqueta_y_las_claves_llegan_desde_labels_grabaciones():
    pares = [("a.edf", "OC")]
    matriz, _ = construir_matriz_rf(_features(pares), _grabaciones(pares, vas=73.0))
    assert matriz.loc[0, "vas"] == 73.0
    assert matriz.loc[0, "persona_uid"] == "C1_F050"
    assert matriz.loc[0, "split"] == "train"


def test_una_grabacion_sin_etiqueta_se_conserva_con_vas_nan():
    pares = [("a.edf", "OC"), ("b.edf", "OA")]
    grabaciones = _grabaciones(pares)
    grabaciones = grabaciones[grabaciones["archivo"] != "b.edf"]
    matriz, meta = construir_matriz_rf(_features(pares), grabaciones)
    assert len(matriz) == 2
    assert meta["n_grabaciones_sin_etiqueta"] == 1
    assert matriz.set_index("archivo").loc["b.edf", "vas"] is np.nan or np.isnan(
        matriz.set_index("archivo").loc["b.edf", "vas"])


def test_construir_matriz_rf_no_duplica_filas_si_labels_repite_archivo():
    pares = [("a.edf", "OC")]
    grabaciones = pd.concat([_grabaciones(pares), _grabaciones(pares)], ignore_index=True)
    with pytest.raises(ValueError, match="repetido"):
        construir_matriz_rf(_features(pares), grabaciones)


# --------------------------------------------------------------------------
# columnas_de_features — la definición de X
# --------------------------------------------------------------------------

def test_medicion_no_es_una_feature():
    """
    Decisión de la Fase 12: `medicion` viaja en la matriz para estratificar y
    calcular líneas base, pero no entra al modelo. Si alguien la añade a X, el
    modelo recupera el calendario escolar en vez de leer la señal.
    """
    pares = [("a.edf", "OC")]
    matriz, meta = construir_matriz_rf(_features(pares), _grabaciones(pares))
    assert "medicion" in matriz.columns
    assert "medicion" not in columnas_de_features(matriz)
    assert "medicion" not in meta["columnas_features"]


def test_ni_la_etiqueta_ni_las_claves_de_identidad_son_features():
    pares = [("a.edf", "OC")]
    matriz, _ = construir_matriz_rf(_features(pares), _grabaciones(pares))
    x = set(columnas_de_features(matriz))
    assert "vas" not in x
    for columna in COLUMNAS_IDENTIDAD:
        assert columna not in x
    # Y no se ha dejado fuera nada que sí deba entrar.
    assert "condicion_orden" in x
    assert "n_ventanas" in x
    assert "n_validos_temporal" in x


def test_las_features_son_todas_numericas():
    """Si se colara una columna de texto, el `fit` fallaría en tiempo de
    entrenamiento y no aquí, que es mucho peor sitio para enterarse."""
    pares = [(f"{c}.edf", c) for c in ORDEN_PROTOCOLO]
    matriz, _ = construir_matriz_rf(_features(pares), _grabaciones(pares))
    x = matriz[columnas_de_features(matriz)]
    no_numericas = [c for c in x.columns if not pd.api.types.is_numeric_dtype(x[c])]
    assert no_numericas == []


def test_la_matriz_entra_en_un_random_forest_con_nan():
    """
    Comprobación de extremo a extremo: con una región vacía la matriz lleva
    NaN, y `RandomForestRegressor` (scikit-learn >= 1.4) los admite sin
    imputar. Si esto dejara de ser cierto, habría que decidir una imputación.
    """
    from sklearn.ensemble import RandomForestRegressor

    pares = [(f"{c}.edf", c) for c in ORDEN_PROTOCOLO]
    validos = [c for c in CANALES_EPOC if c not in ("T7", "T8")]
    matriz, meta = construir_matriz_rf(_features(pares, validos=validos),
                                       _grabaciones(pares))
    x = matriz[columnas_de_features(matriz)]
    assert meta["n_celdas_nan"] > 0
    RandomForestRegressor(n_estimators=5, random_state=0).fit(x, matriz["vas"])


# --------------------------------------------------------------------------
# resumir_por_grabacion — la parte de la matriz que no necesita etiquetas
# --------------------------------------------------------------------------

def test_resumir_por_grabacion_no_necesita_condicion_ni_vas():
    features = _features([("a.edf", "OC")]).drop(columns=["condicion"])
    resumen = resumir_por_grabacion(features)
    assert resumen["archivo"].tolist() == ["a.edf"]
    assert "condicion" not in resumen.columns and "vas" not in resumen.columns


def test_resumir_por_grabacion_coincide_con_construir_matriz_rf():
    pares = [("a.edf", "OC"), ("b.edf", "PASAT")]
    features = _features(pares, validos=CANALES_EPOC[2:])
    resumen = resumir_por_grabacion(features)
    matriz, _ = construir_matriz_rf(features, _grabaciones(pares))
    pd.testing.assert_frame_equal(
        resumen, matriz[resumen.columns].reset_index(drop=True), check_exact=True)
    # Lo único que añade construir_matriz_rf: identidad, etiqueta y condición.
    añadidas = set(matriz.columns) - set(resumen.columns)
    assert añadidas == (set(COLUMNAS_IDENTIDAD) | {"vas", "condicion_orden"}) - {"archivo"}
