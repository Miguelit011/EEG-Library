"""
Genera el JSON de ejemplo de `analizar_grabacion` a partir de un EDF real.

Sirve de fixture a EEG-Platform: la web se desarrolla y se prueba contra
este archivo sin necesitar el dataset.

- Se usa una grabación de **train** (nunca de test), de ojos abiertos, con
  algún electrodo descartado, para que la pantalla de calidad tenga algo
  que enseñar.
- El archivo va al repositorio y los participantes son menores: el nombre
  del EDF se sustituye por `ejemplo_OA.edf`, de modo que el JSON no lleva
  sujeto, centro ni fecha de la sesión. Solo contiene potencias espectrales
  agregadas.

Uso:
    .venv/Scripts/python.exe analisis/generar_ejemplo.py
"""

import contextlib
import io
import json
import os
import warnings
from pathlib import Path

import mne
import pandas as pd

from eeg_library.procesado.grabacion import analizar_grabacion

RAIZ_REPO = Path(__file__).resolve().parents[1]
RUTA_DATOS = Path(os.environ.get("EEG_DATA", r"C:\Users\migue\Desktop\TFG\EEG-Data"))
DESTINO = RAIZ_REPO / "docs" / "ejemplos" / "analisis_grabacion.json"
NOMBRE_ANONIMO = "ejemplo_OA.edf"


def elegir_grabacion():
    """Primera grabación OA de train, export de 71 canales, con 1-2 descartes."""
    matriz = pd.read_parquet(RUTA_DATOS / "processed" / "matriz_rf.parquet")
    log = pd.read_csv(RUTA_DATOS / "processed" / "procesamiento_log.csv")
    datos = matriz.merge(log[["archivo", "n_canales"]], on="archivo")
    candidatas = datos[(datos["split"] == "train") & (datos["condicion"] == "OA")
                       & (datos["n_canales"] == 71)
                       & datos["n_canales_buenos"].between(12, 13)]
    return candidatas.sort_values("archivo")["archivo"].iloc[0]


def main():
    mne.set_log_level("ERROR")
    archivo = elegir_grabacion()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        resultado = analizar_grabacion(RUTA_DATOS / "dataset_final" / archivo)
    resultado["metadatos"]["archivo"] = NOMBRE_ANONIMO

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    DESTINO.write_text(json.dumps(resultado, ensure_ascii=False, indent=2,
                                  allow_nan=False) + "\n", encoding="utf-8")
    descartados = [c for c, i in resultado["calidad"].items() if i["estado"] == "descartado"]
    print(f"Escrito {DESTINO.relative_to(RAIZ_REPO)} "
          f"({resultado['metadatos']['n_epocas']} épocas, descartados: {descartados}).")


if __name__ == "__main__":
    main()
