"""
Tests para el módulo de anotación de validez de canal (Fase 11.5).
"""

import numpy as np
import pandas as pd
import pytest

from eeg_library.procesado.features import (
    anotar_validez_de_canales,
    columnas_de_validez,
    PREFIJO_VALIDEZ,
    COLUMNA_N_BUENOS,
)
from eeg_library.preprocesado.quality import CANALES_EPOC


def _features(archivos_y_epocas):
    """Matriz de features mínima: archivo_origen más una columna de potencia."""
    filas = []
    for archivo, n in archivos_y_epocas.items():
        for i in range(n):
            filas.append({
                "archivo_origen": archivo,
                "indice_ventana": i,
                "pot_abs_AF3_theta": 1e-11,
            })
    return pd.DataFrame(filas)


def _log(excluidos_por_archivo, estado="ok"):
    """Log del lote con los canales excluidos de cada archivo."""
    filas = []
    for archivo, excluidos in excluidos_por_archivo.items():
        filas.append({
            "archivo": archivo,
            "estado": estado,
            "canales_excluidos": "|".join(excluidos) if excluidos else np.nan,
            "n_canales_buenos": len(CANALES_EPOC) - len(excluidos),
        })
    return pd.DataFrame(filas)


def test_anade_una_columna_de_validez_por_canal_y_el_recuento():
    features = _features({"a.edf": 3})
    log = _log({"a.edf": []})

    anotadas = anotar_validez_de_canales(features, log)

    for columna in columnas_de_validez():
        assert columna in anotadas.columns
        assert anotadas[columna].all()
    assert (anotadas[COLUMNA_N_BUENOS] == len(CANALES_EPOC)).all()
    assert len(anotadas) == len(features)


def test_marca_como_invalidos_exactamente_los_canales_excluidos():
    features = _features({"a.edf": 4})
    log = _log({"a.edf": ["AF4", "F3"]})

    anotadas = anotar_validez_de_canales(features, log)

    assert not anotadas[f"{PREFIJO_VALIDEZ}AF4"].any()
    assert not anotadas[f"{PREFIJO_VALIDEZ}F3"].any()
    for canal in CANALES_EPOC:
        if canal not in ("AF4", "F3"):
            assert anotadas[f"{PREFIJO_VALIDEZ}{canal}"].all()
    assert (anotadas[COLUMNA_N_BUENOS] == len(CANALES_EPOC) - 2).all()


def test_la_mascara_es_por_archivo_no_por_epoca():
    """
    `evaluar_calidad` decide sobre el registro completo, así que todas las
    épocas de un archivo comparten los mismos canales válidos y archivos
    distintos pueden tener máscaras distintas.
    """
    features = _features({"a.edf": 3, "b.edf": 2})
    log = _log({"a.edf": ["AF4"], "b.edf": ["P7", "O1", "T8"]})

    anotadas = anotar_validez_de_canales(features, log)

    de_a = anotadas[anotadas.archivo_origen == "a.edf"]
    de_b = anotadas[anotadas.archivo_origen == "b.edf"]
    assert (de_a[COLUMNA_N_BUENOS] == 13).all()
    assert (de_b[COLUMNA_N_BUENOS] == 11).all()
    assert not de_a[f"{PREFIJO_VALIDEZ}AF4"].any()
    assert de_b[f"{PREFIJO_VALIDEZ}AF4"].all()


def test_no_altera_las_columnas_originales():
    features = _features({"a.edf": 3})
    log = _log({"a.edf": ["AF4"]})

    anotadas = anotar_validez_de_canales(features, log)

    pd.testing.assert_frame_equal(
        anotadas[list(features.columns)], features, check_dtype=True
    )
    # Las columnas nuevas van al final, sin reordenar las existentes.
    assert list(anotadas.columns)[: len(features.columns)] == list(features.columns)


def test_no_muta_el_dataframe_de_entrada():
    features = _features({"a.edf": 2})
    columnas_antes = list(features.columns)

    anotar_validez_de_canales(features, _log({"a.edf": ["AF4"]}))

    assert list(features.columns) == columnas_antes


def test_ignora_las_filas_de_log_de_archivos_fallidos():
    """Un archivo que falló no produjo features; su fila no debe estorbar."""
    features = _features({"a.edf": 2})
    log = pd.concat([
        _log({"a.edf": ["AF4"]}),
        _log({"fallido.edf": CANALES_EPOC}, estado="error"),
    ], ignore_index=True)

    anotadas = anotar_validez_de_canales(features, log)

    assert set(anotadas.archivo_origen) == {"a.edf"}
    assert (anotadas[COLUMNA_N_BUENOS] == 13).all()


def test_archivo_sin_fila_en_el_log_lanza_error():
    """
    Sin fila de log no se sabe qué canales eran válidos. Rellenar la máscara
    con True daría por buenos canales que quizá no lo eran.
    """
    features = _features({"a.edf": 2, "huerfano.edf": 2})
    log = _log({"a.edf": []})

    with pytest.raises(ValueError, match="no aparecen como procesados"):
        anotar_validez_de_canales(features, log)


def test_archivo_fallido_en_features_lanza_error():
    features = _features({"a.edf": 2})
    log = _log({"a.edf": ["AF4"]}, estado="error")

    with pytest.raises(ValueError, match="no aparecen como procesados"):
        anotar_validez_de_canales(features, log)


def test_desajuste_con_el_recuento_de_evaluar_calidad_lanza_error():
    """
    Si la máscara y `n_canales_buenos` no cuadran, la grabación no tenía los
    14 electrodos y la máscara estaría marcando como válido uno ausente.
    """
    features = _features({"a.edf": 2})
    log = _log({"a.edf": ["AF4"]})
    log.loc[0, "n_canales_buenos"] = 10  # incoherente con un solo excluido

    with pytest.raises(ValueError, match="mascara da"):
        anotar_validez_de_canales(features, log)


def test_exige_las_columnas_necesarias():
    with pytest.raises(ValueError, match="archivo_origen"):
        anotar_validez_de_canales(pd.DataFrame({"otra": [1]}), _log({"a.edf": []}))

    with pytest.raises(ValueError, match="faltan columnas"):
        anotar_validez_de_canales(_features({"a.edf": 1}), pd.DataFrame({"archivo": ["a.edf"]}))
