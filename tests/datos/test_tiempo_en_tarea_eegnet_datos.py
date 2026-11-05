"""
Paridad de EEGNet en la librería con el análisis, sobre EDF reales (Fase 11 de EEG-Platform).

Necesita `EEG_DATA` (EDF, `ventanas/*.npy`, `features_all.parquet`, `matriz_rf.parquet` y
`procesamiento_log.csv`), así que va con el marcador `datos`:

    .venv/Scripts/python.exe -m pytest -m datos tests/datos/test_tiempo_en_tarea_eegnet_datos.py -s

Usa 20 grabaciones PASAT y PVTB de `train` y `val` (nunca de `test`: se filtra por la `split`
antes de abrir nada). La cadena que se comprueba:

1. **Preparación:** `preparar_entrada` sobre las ventanas guardadas del estudio
   (`EEG-Data/ventanas/*.npy`) da exactamente el tensor de `02_piloto_cnn/preparar.py` (`tensor`).
2. **De punta a punta:** la puntuación que `analizar_grabacion` guarda al procesar el EDF
   coincide con la de puntuar ese tensor del análisis con los mismos `.onnx` (< 1e-5).

Que ONNX da lo mismo que PyTorch se comprobó al exportar
(`analisis/tiempo_en_tarea/deep_learning/05_onnx/resultados.json`).
"""

import contextlib
import importlib.util
import io
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eeg_library.modelado.tiempo_en_tarea_eegnet import preparar_entrada, puntuar_entrada
from eeg_library.procesado.grabacion import analizar_grabacion

RUTA_DATOS = Path(os.environ.get("EEG_DATA", r"C:\Users\migue\Desktop\TFG\EEG-Data"))
RUTA_MATRIZ = RUTA_DATOS / "processed" / "matriz_rf.parquet"
DL = Path(__file__).resolve().parents[2] / "analisis" / "tiempo_en_tarea" / "deep_learning"

pytestmark = [
    pytest.mark.datos,
    pytest.mark.skipif(not RUTA_MATRIZ.is_file(), reason="no hay dataset en EEG_DATA"),
]

SPLITS = ("train", "val")
N_POR_CONDICION = 10
SEMILLA = 1


@pytest.fixture(scope="module")
def piloto():
    spec = importlib.util.spec_from_file_location("piloto_preparar", DL / "02_piloto_cnn" / "preparar.py")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["piloto_preparar"] = modulo
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(modulo)
    return modulo


def test_eegnet_igual_que_en_el_analisis(piloto):
    import mne
    mne.set_log_level("ERROR")
    with contextlib.redirect_stdout(io.StringIO()):
        w = piloto.comun.cargar(set(SPLITS))
    assert set(w.split) <= set(SPLITS)
    elegidas = []
    for c in ("PASAT", "PVTB"):
        archivos = sorted(w.archivo_origen[w.condicion == c].unique())
        elegidas += list(np.random.default_rng(SEMILLA).choice(archivos, N_POR_CONDICION, replace=False))

    filas = []
    for archivo in elegidas:
        v = w[w.archivo_origen == archivo].sort_values("indice_ventana").reset_index(drop=True)
        X_ref, M_ref = piloto.tensor(v)

        # 1. Preparación: la de la librería sobre las ventanas guardadas = la del análisis.
        crudas = np.load(piloto.p1.RUTA_VENTANAS / archivo.replace(".edf", ".npy"))[v.indice_ventana.to_numpy()]
        validez = v[piloto.d1.VALIDEZ].iloc[0].to_numpy(dtype=bool)
        X_lib, M_lib = preparar_entrada(crudas, validez)
        dif_tensor = float(np.max(np.abs(X_lib - X_ref)))
        assert (M_lib == M_ref).all()

        # 2. De punta a punta: lo que guarda analizar_grabacion = puntuar el tensor del análisis.
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            r = analizar_grabacion(RUTA_DATOS / "dataset_final" / archivo)
        lib = np.asarray(r["por_ventana"]["eegnet"])
        ref = puntuar_entrada(X_ref, M_ref).mean(axis=0)
        filas.append({"condicion": v.condicion.iloc[0], "ventanas": len(v), "validos": int(validez.sum()),
                      "dif_tensor": dif_tensor, "dif_puntuacion": float(np.max(np.abs(lib - ref)))})

    tabla = pd.DataFrame(filas)
    print("\n" + tabla.to_string(index=False, float_format="%.2e"))
    print(f"\nMáximos: tensor {tabla.dif_tensor.max():.1e} · puntuación {tabla.dif_puntuacion.max():.1e}")
    assert tabla.dif_tensor.max() == 0.0
    assert tabla.dif_puntuacion.max() < 1e-5
