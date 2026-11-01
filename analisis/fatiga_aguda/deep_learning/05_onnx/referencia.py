"""
Fatiga aguda — deep learning, paso 5 (1 de 2): tensor de referencia para comprobar la exportación.

Construye, con la función del piloto (`02_piloto_cnn/preparar.py`, `tensor`), la entrada de
EEGNet de 20 grabaciones PASAT y PVTB de **train** (10 de cada, semilla 0) y la guarda en
`TFG/EEG-DL-cache/onnx/` (fuera de los repositorios, sin git). `exportar.py` la puntúa con
PyTorch y con onnxruntime. No carga `val` ni `test`.

Ejecutar desde la raíz del repositorio con el entorno de la librería:
    .venv/Scripts/python.exe analisis/fatiga_aguda/deep_learning/05_onnx/referencia.py
"""

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np

CARPETA = Path(__file__).resolve().parent
DL = CARPETA.parent


def cargar_modulo(nombre, ruta):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = modulo
    spec.loader.exec_module(modulo)
    return modulo


piloto = cargar_modulo("piloto_preparar", DL / "02_piloto_cnn" / "preparar.py")
comun, d1 = piloto.comun, piloto.d1

CACHE = d1.EEG_DATA.parent / "EEG-DL-cache" / "onnx"
N_POR_CONDICION = 10
SEMILLA = 0


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        w = comun.cargar({"train"})
    assert set(w.split) == {"train"}
    elegidas = []
    for c in ("PASAT", "PVTB"):
        archivos = sorted(w.archivo_origen[w.condicion == c].unique())
        rng = np.random.default_rng(SEMILLA)
        elegidas += list(rng.choice(archivos, N_POR_CONDICION, replace=False))
    v = (w[w.archivo_origen.isin(elegidas)].sort_values(["archivo_origen", "indice_ventana"])
         .reset_index(drop=True))
    X, mascara = piloto.tensor(v)
    assert not np.isnan(X).any()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "X.npy", X)
    np.save(CACHE / "mascara.npy", mascara)
    v[["archivo_origen", "condicion", "indice_ventana"]].to_parquet(CACHE / "ventanas.parquet", index=False)
    resumen = {"grabaciones": len(elegidas), "ventanas": int(len(v)), "forma": list(X.shape),
               "canales_excluidos": float(1 - mascara.mean()),
               "sha256_X": d1.sha256(CACHE / "X.npy")}
    (CACHE / "referencia.json").write_text(json.dumps(resumen, indent=1), encoding="utf-8")
    print(json.dumps(resumen, indent=1))


if __name__ == "__main__":
    main()
