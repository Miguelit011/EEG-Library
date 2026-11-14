"""
Figura de la memoria (cap. 5): distribución de las respuestas a la VAS.

Describe la etiqueta del conjunto de datos, no un resultado: las 653 sesiones
con respuesta en la VAS y ventanas procesadas, de las tres particiones (el
mismo conjunto sobre el que `docs/ESTADO.md`, Fase 12, da la media, la
desviación y el ICC). No se entrena ningún modelo.

Solo se leen la respuesta y el recuento de ventanas de `labels.csv`: la figura
no contiene ningún identificador de participante.

Ejecutar (desde la raíz del repositorio):
    python analisis/figura_distribucion_vas.py

Guarda `resultados/figuras/distribucion_vas.pdf` y `.png`. La ubicación del
dataset se toma de la variable de entorno `EEG_DATA`, con `EEG-Data` junto al
repositorio por defecto (ver `analisis/README.md`).
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EEG_DATA = Path(os.environ.get("EEG_DATA", Path(__file__).resolve().parents[2] / "EEG-Data"))
RUTA_LABELS = EEG_DATA / "metadata" / "labels.csv"
CARPETA_FIGURAS = Path(__file__).resolve().parents[1] / "resultados" / "figuras"


def main():
    etiquetas = pd.read_csv(RUTA_LABELS, usecols=["vas", "n_ventanas"])
    vas = etiquetas.loc[etiquetas["vas"].notna() & (etiquetas["n_ventanas"] > 0), "vas"].to_numpy()
    if len(vas) != 653:
        raise ValueError(f"Se esperaban 653 sesiones etiquetadas y hay {len(vas)}.")

    multiplos_10 = np.mean(vas % 10 == 0)
    print(f"Sesiones: {len(vas)}")
    print(f"Media {vas.mean():.1f}, mediana {np.median(vas):.0f}, desviación {vas.std(ddof=1):.1f}")
    print(f"Múltiplos de 10: {multiplos_10:.1%}; de 5: {np.mean(vas % 5 == 0):.1%}")

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, eje = plt.subplots(figsize=(6.0, 2.8))
    bordes = np.arange(-2.5, 105, 5)  # un intervalo por cada múltiplo de 5
    eje.hist(vas, bins=bordes, color="0.65", edgecolor="black", linewidth=0.5)
    eje.axvline(vas.mean(), color="black", linestyle="--", linewidth=0.9)
    eje.text(vas.mean() + 1.5, eje.get_ylim()[1] * 0.92, f"media {vas.mean():.1f}",
             fontsize=9)
    eje.set_xlabel("Respuesta en la VAS (0–100)")
    eje.set_ylabel("Sesiones")
    eje.set_xlim(-3, 103)
    eje.set_xticks(range(0, 101, 10))
    eje.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(CARPETA_FIGURAS / f"distribucion_vas.{extension}", dpi=200)
    print(f"Guardada en {CARPETA_FIGURAS}")


if __name__ == "__main__":
    main()
