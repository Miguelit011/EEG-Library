"""
Figura de la memoria (cap. 5): el procesado sobre una grabación real.

Toma una grabación de ojos abiertos de **entrenamiento** con los 14 electrodos
buenos y al menos un componente ocular, la procesa con las funciones públicas
de la librería y dibuja:

- (a) ocho segundos de seis electrodos antes y después de la ICA;
- (b) el espectro medio de O1 y O2 en la señal leída y en la procesada.

No entrena ni evalúa ningún modelo. La figura no contiene ningún identificador
de participante.

Ejecutar (desde la raíz del repositorio):
    .venv/Scripts/python.exe analisis/figura_pipeline_ejemplo.py

Guarda `resultados/figuras/pipeline_ejemplo.pdf` y `.png`. La ubicación del
dataset se toma de la variable de entorno `EEG_DATA`, con `EEG-Data` junto al
repositorio por defecto (ver `analisis/README.md`).
"""

import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

from eeg_library.preprocesado.artifacts import (eliminar_artefactos_oculares, entrenar_ica,
                                                 identificar_componentes_oculares)
from eeg_library.preprocesado.filtering import aplicar_filtros, recortar_registro
from eeg_library.preprocesado.quality import CANALES_EPOC, evaluar_calidad
from eeg_library.preprocesado.reader import cargar_eeg
from eeg_library.preprocesado.referencing import aplicar_car

RAIZ = Path(__file__).resolve().parents[1]
EEG_DATA = Path(os.environ.get("EEG_DATA", RAIZ.parent / "EEG-Data"))
RUTA_LOG = EEG_DATA / "processed" / "procesamiento_log.csv"
CARPETA_FIGURAS = RAIZ / "resultados" / "figuras"

CANALES = list(CANALES_EPOC)
MOSTRAR = ["AF3", "AF4", "F7", "F8", "O1", "O2"]

# Mismo estilo que el resto de figuras de la memoria.
TEXTO, GRIS, GRIS_CLARO, AMBAR = "#1d2330", "#5b6475", "#9aa1ad", "#b26a00"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9, "pdf.fonttype": 42,
    "text.color": TEXTO, "axes.labelcolor": TEXTO, "axes.edgecolor": GRIS,
    "xtick.color": GRIS, "ytick.color": GRIS,
    "axes.spines.top": False, "axes.spines.right": False,
})


def elegir_grabacion():
    """Una grabación de OA de entrenamiento, completa y con parpadeos que eliminar."""
    log = pd.read_csv(RUTA_LOG)
    candidatas = log[(log.estado == "ok") & (log.split == "train") & (log.condicion == "OA")
                     & (log.n_canales_buenos == 14) & (log.n_componentes_oculares >= 1)
                     & (log.sfreq_original == 256)]
    fila = candidatas.sort_values("varianza_eliminada", ascending=False).iloc[len(candidatas) // 10]
    ruta = EEG_DATA / "dataset_final" / fila.archivo
    return ruta if ruta.exists() else next((EEG_DATA / "dataset_final").rglob(fila.archivo))


def procesar(ruta):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw, _ = cargar_eeg(str(ruta))
        rec, _ = recortar_registro(raw)
        filt, _ = aplicar_filtros(rec)
        buenos, _, _ = evaluar_calidad(filt)
        ref, _ = aplicar_car(filt, buenos)
        ica, _ = entrenar_ica(ref, buenos)
        oculares, _, _ = identificar_componentes_oculares(ref, ica)
        limpio, _ = eliminar_artefactos_oculares(ref, ica, oculares)
    print(f"Componentes oculares eliminados: {len(oculares)}")
    return rec, ref, limpio


def main():
    rec, ref, limpio = procesar(elegir_grabacion())
    sf = rec.info["sfreq"]
    antes = rec.get_data(picks=CANALES) * 1e6
    antes = antes - np.median(antes, axis=1, keepdims=True)
    sin_ica = ref.get_data(picks=CANALES) * 1e6
    despues = limpio.get_data(picks=CANALES) * 1e6
    idx = [CANALES.index(c) for c in MOSTRAR]

    # Ventana de 8 s con mucha señal ocular eliminada y sin otros artefactos grandes
    # que queden después de la ICA (que solo trata los oculares).
    dur = int(8 * sf)
    quitado = np.abs(sin_ica - despues)[[CANALES.index("AF3"), CANALES.index("AF4")]].mean(0)
    mejor, t0 = -1, 0
    for ini in range(0, sin_ica.shape[1] - dur, int(sf / 2)):
        residuo = np.abs(despues[idx, ini:ini + dur]).max()
        if residuo < 60 and quitado[ini:ini + dur].mean() > mejor:
            mejor, t0 = quitado[ini:ini + dur].mean(), ini
    t = np.arange(dur) / sf

    fig = plt.figure(figsize=(6.3, 3.3))
    rejas = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.3)

    # (a) Señal antes y después de la ICA.
    ax = fig.add_subplot(rejas[0, 0])
    sep = 110
    for k, i in enumerate(idx):
        ax.plot(t, sin_ica[i, t0:t0 + dur] - k * sep, color=GRIS_CLARO, lw=0.8)
        ax.plot(t, despues[i, t0:t0 + dur] - k * sep, color=TEXTO, lw=0.6)
    ax.set_yticks([-k * sep for k in range(len(MOSTRAR))], MOSTRAR, fontsize=7)
    ax.set_xlim(0, 8)
    ax.set_ylim(-(len(MOSTRAR) - 0.4) * sep, 1.6 * sep)
    ax.set_title("(a) Seis electrodos, antes y después de la ICA", fontsize=8.5, fontweight="bold", loc="left")
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0, labelsize=7)
    ax.plot([7.8, 7.8], [1.45 * sep, 1.45 * sep - 100], color=TEXTO, lw=1.2)
    ax.text(7.7, 1.45 * sep - 50, "100 µV", ha="right", va="center", fontsize=6.5)
    ax.plot([0.1, 0.5], [1.3 * sep] * 2, color=GRIS_CLARO, lw=1.5)
    ax.text(0.6, 1.3 * sep, "filtrada y referenciada", fontsize=6.5, va="center", color=GRIS)
    ax.plot([3.9, 4.3], [1.3 * sep] * 2, color=TEXTO, lw=1.5)
    ax.text(4.4, 1.3 * sep, "tras la ICA", fontsize=6.5, va="center", color=GRIS)
    ax.set_xlabel("Tiempo (s)", fontsize=8)

    # (b) Espectro de la señal leída y de la procesada.
    ax = fig.add_subplot(rejas[0, 1])
    bandas = [("δ", 1, 4), ("θ", 4, 8), ("α", 8, 13), ("β", 13, 30)]
    tonos = ["#f4f5f7", "#e9ebef", "#f4f5f7", "#e9ebef"]
    for (nombre, a, b), tono in zip(bandas, tonos):
        ax.axvspan(a, b, color=tono, zorder=0)
        ax.text((a + b) / 2, 1.02, nombre, transform=ax.get_xaxis_transform(), ha="center", fontsize=8,
                fontweight="bold", color=GRIS)
    for datos, color, etiqueta in [(antes, GRIS_CLARO, "Leída"), (despues, AMBAR, "Procesada")]:
        f, p = welch(datos[[CANALES.index("O1"), CANALES.index("O2")]], fs=sf, nperseg=int(2 * sf))
        sel = (f >= 0.5) & (f <= 45)
        ax.semilogy(f[sel], p.mean(0)[sel], color=color, lw=1.4, label=etiqueta)
    ax.set_xlim(0.5, 40)
    ax.set_ylim(1e-2, 3e3)
    ax.set_xlabel("Frecuencia (Hz)", fontsize=8)
    ax.set_ylabel("Potencia media en O1 y O2 (µV²/Hz)", fontsize=8)
    ax.set_title("(b) Espectro de la grabación", fontsize=8.5, fontweight="bold", loc="left", pad=14)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.tick_params(length=0, labelsize=7)

    CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(CARPETA_FIGURAS / f"pipeline_ejemplo.{extension}", dpi=220, bbox_inches="tight")
    print(f"Guardada en {CARPETA_FIGURAS}")


if __name__ == "__main__":
    main()
