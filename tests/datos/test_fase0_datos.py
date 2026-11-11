"""
Criterios de aceptación de la Fase 0 de la plataforma, sobre el dataset real.

Necesitan `EEG_DATA` (EDF, `matriz_modelado.parquet`, `matriz_rf.parquet`,
`labels_grabaciones.csv` y `procesamiento_log.csv`) y tardan alrededor de un
minuto, así que van con el marcador `datos` y no entran en el `pytest`
normal:

    .venv/Scripts/python.exe -m pytest -m datos

Nunca usan grabaciones de test: el análisis confirmatorio ya se hizo y el
test no se vuelve a abrir.

- **Matriz idéntica:** reconstruir `matriz_rf.parquet` con el código actual
  da exactamente la misma tabla.
- **Paridad:** las 70 características de `analizar_grabacion` sobre un EDF
  coinciden con su fila de la matriz.
- **Clases:** sobre train/val (sus datos de entrenamiento) los modelos
  aciertan mucho con el nombre de clase de la ficha; con el orden de clases
  cambiado el acierto caería al azar o por debajo.
"""

import contextlib
import io
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score

from eeg_library.modelado.prediccion import cargar_modelo, listar_modelos
from eeg_library.procesado.grabacion import ALIAS_CARACTERISTICAS, analizar_grabacion
from eeg_library.procesado.matrix import construir_matriz_rf

# Por defecto, la carpeta EEG-Data junto al repositorio (TFG/EEG-Data); se cambia con EEG_DATA.
RUTA_DATOS = Path(os.environ.get("EEG_DATA", Path(__file__).resolve().parents[3] / "EEG-Data"))
RUTA_MATRIZ = RUTA_DATOS / "processed" / "matriz_rf.parquet"

pytestmark = [
    pytest.mark.datos,
    pytest.mark.skipif(not RUTA_MATRIZ.is_file(), reason="no hay dataset en EEG_DATA"),
]

SPLITS = ("train", "val")
REPOSO = ("OA", "OC")

# Acierto mínimo sobre los datos de entrenamiento (azar: 0.5, 0.5 y 0.2).
ACIERTO_MINIMO = {
    "c1_oa_oc_logistica": 0.80,
    "c2_reposo_tarea_logistica": 0.80,
    "c3_tareas_logistica": 0.45,
}


@pytest.fixture(scope="module")
def matriz():
    return pd.read_parquet(RUTA_MATRIZ)


def test_matriz_reconstruida_identica(matriz):
    features = pd.read_parquet(RUTA_DATOS / "processed" / "matriz_modelado.parquet")
    grabaciones = pd.read_csv(RUTA_DATOS / "metadata" / "labels_grabaciones.csv")
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        nueva, _ = construir_matriz_rf(features, grabaciones)
    pd.testing.assert_frame_equal(nueva, matriz, check_exact=True)


def _clase_esperada(id_modelo, condicion):
    """Nombre de clase que corresponde a una condición, según la tarea."""
    if id_modelo.startswith("c2_"):
        return "reposo" if condicion in REPOSO else "tarea"
    return condicion


@pytest.mark.parametrize("id_modelo", listar_modelos())
def test_clases_con_nombre_correcto(matriz, id_modelo):
    modelo = cargar_modelo(id_modelo)
    ficha = modelo.ficha
    datos = matriz[matriz["split"].isin(SPLITS)
                   & matriz["condicion"].isin(ficha["condiciones_aplicables"])]
    nombres = {c["codigo"]: c["nombre"] for c in ficha["clases"]}
    predichas = [nombres[int(c)] for c in modelo.pipeline.predict(datos[ficha["columnas"]])]
    esperadas = [_clase_esperada(id_modelo, c) for c in datos["condicion"]]
    acierto = balanced_accuracy_score(esperadas, predichas)
    print(f"\n{id_modelo}: exactitud balanceada en train/val = {acierto:.3f} "
          f"(n = {len(datos)})")
    assert acierto > ACIERTO_MINIMO[id_modelo]


def _muestra_paridad(matriz):
    """
    Unas 13 grabaciones de train/val que cubren los casos que pueden fallar:
    las 7 condiciones, las cuatro variantes de export de EmotivPro (59, 63,
    67 y 71 canales), un EDF resampleado, la grabación con menos de 6
    canales buenos (CAR omitida) y regiones sin canales válidos (NaN).
    """
    log = pd.read_csv(RUTA_DATOS / "processed" / "procesamiento_log.csv")
    datos = matriz[matriz["split"].isin(SPLITS)].merge(
        log[["archivo", "n_canales", "resampleado"]], on="archivo", how="left")
    elegidas = []
    for condicion in ("OC", "OA", "STROOP", "PASAT", "PVTB", "MSIT", "PVSAT"):
        elegidas.append(datos[(datos["condicion"] == condicion)
                              & (datos["n_canales"] == 71)]["archivo"].iloc[0])
    for n in (59, 63, 67):
        elegidas.append(datos[datos["n_canales"] == n]["archivo"].iloc[0])
    for filtro in (datos["resampleado"] == True,  # noqa: E712
                   datos["n_canales_buenos"] < 6,
                   datos["temporal_alfa_rel_media"].isna(),
                   datos["frontocentral_alfa_rel_media"].isna()):
        if filtro.any():
            elegidas.append(datos[filtro]["archivo"].iloc[0])
    return list(dict.fromkeys(elegidas))


def test_paridad_con_la_matriz(matriz):
    import mne
    mne.set_log_level("ERROR")
    inverso = {v: k for k, v in ALIAS_CARACTERISTICAS.items()}
    por_archivo = matriz.set_index("archivo")

    diferencias = {}
    for archivo in _muestra_paridad(matriz):
        fila = por_archivo.loc[archivo]
        assert fila["split"] in SPLITS
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            r = analizar_grabacion(RUTA_DATOS / "dataset_final" / archivo)
        calculadas = {inverso.get(k, k): v for k, v in r["caracteristicas"].items()}
        assert len(calculadas) == 70
        esperadas = fila[list(calculadas)].astype(float).to_numpy()
        obtenidas = np.array([np.nan if v is None else v for v in calculadas.values()])
        np.testing.assert_array_equal(np.isnan(obtenidas), np.isnan(esperadas),
                                      err_msg=f"NaN distintos en {archivo}")
        validas = ~np.isnan(esperadas)
        diferencias[archivo] = float(np.max(
            np.abs(obtenidas[validas] - esperadas[validas])
            / np.maximum(np.abs(esperadas[validas]), 1e-300)))

    print("\nDiferencia relativa máxima por grabación:")
    for archivo, d in diferencias.items():
        print(f"  {archivo:<45s} {d:.1e}")
    # Mismo código, mismas versiones y misma semilla de ICA: igualdad exacta.
    assert max(diferencias.values()) == 0.0
