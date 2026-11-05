"""
Genera los JSON de ejemplo de `analizar_grabacion` a partir de EDF reales.

Sirven de fixture a EEG-Platform: la web se desarrolla y se prueba contra
estos archivos sin necesitar el dataset.

- `analisis_grabacion.json`: una grabación de **train** (nunca de test), de
  ojos abiertos, con algún electrodo descartado, para que la pantalla de
  calidad tenga algo que enseñar.
- `analisis_grabacion_pasat.json` (desde 0.15.0): una grabación PASAT de
  **train** para la pestaña Tiempo en tarea. Se elige, entre las de 71 canales
  con un electrodo descartado, la primera en orden alfabético en la que el
  modelo de tiempo en tarea ve el cambio con claridad (último tercio por encima
  del primero y ρ ≥ 0,3). **Es un ejemplo para enseñar la pantalla, no una
  muestra representativa:** en test el modelo acierta en el 74 % de las
  grabaciones PASAT.
- Los archivos van al repositorio y los participantes son menores: el nombre
  del EDF se sustituye por `ejemplo_OA.edf` o `ejemplo_PASAT.edf`, de modo que el JSON no lleva
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
DESTINO_PASAT = RAIZ_REPO / "docs" / "ejemplos" / "analisis_grabacion_pasat.json"
NOMBRE_ANONIMO_PASAT = "ejemplo_PASAT.edf"
RHO_MINIMA_PASAT = 0.3


def elegir_grabacion():
    """Primera grabación OA de train, export de 71 canales, con 1-2 descartes."""
    matriz = pd.read_parquet(RUTA_DATOS / "processed" / "matriz_rf.parquet")
    log = pd.read_csv(RUTA_DATOS / "processed" / "procesamiento_log.csv")
    datos = matriz.merge(log[["archivo", "n_canales"]], on="archivo")
    candidatas = datos[(datos["split"] == "train") & (datos["condicion"] == "OA")
                       & (datos["n_canales"] == 71)
                       & datos["n_canales_buenos"].between(12, 13)]
    return candidatas.sort_values("archivo")["archivo"].iloc[0]


def analizar(archivo):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return analizar_grabacion(RUTA_DATOS / "dataset_final" / archivo)


def escribir(resultado, destino, nombre):
    resultado["metadatos"]["archivo"] = nombre
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2,
                                  allow_nan=False) + "\n", encoding="utf-8")
    descartados = [c for c, i in resultado["calidad"].items() if i["estado"] == "descartado"]
    print(f"Escrito {destino.relative_to(RAIZ_REPO)} "
          f"({resultado['metadatos']['n_epocas']} épocas, descartados: {descartados}).")


def ejemplo_pasat():
    """La primera grabación PASAT de train (71 canales, 13 buenos) en la que el cambio se ve claro."""
    from eeg_library.modelado.tiempo_en_tarea import puntuar
    matriz = pd.read_parquet(RUTA_DATOS / "processed" / "matriz_rf.parquet")
    log = pd.read_csv(RUTA_DATOS / "processed" / "procesamiento_log.csv")
    datos = matriz.merge(log[["archivo", "n_canales"]], on="archivo")
    candidatas = datos[(datos["split"] == "train") & (datos["condicion"] == "PASAT")
                       & (datos["n_canales"] == 71) & (datos["n_canales_buenos"] == 13)]
    for archivo in candidatas.sort_values("archivo")["archivo"]:
        resultado = analizar(archivo)
        p = puntuar(resultado, "PASAT")
        if p["disponible"] and p["tercios"]["final_mayor"] and p["rho"] >= RHO_MINIMA_PASAT:
            print(f"PASAT: ρ = {p['rho']:+.3f}, el último tercio puntúa por encima del primero.")
            return resultado
    raise SystemExit("Ninguna grabación PASAT cumple el criterio del ejemplo.")


def main():
    mne.set_log_level("ERROR")
    escribir(analizar(elegir_grabacion()), DESTINO, NOMBRE_ANONIMO)
    escribir(ejemplo_pasat(), DESTINO_PASAT, NOMBRE_ANONIMO_PASAT)


if __name__ == "__main__":
    main()
