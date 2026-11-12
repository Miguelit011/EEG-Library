"""
Módulo de epocado EEG (Fase 7).

Segmenta el Raw ya limpio (Fases 1-6) en ventanas de longitud fija con
solapamiento. El resultado es compartido por las dos rutas de análisis
posteriores —extracción de features para Random Forest (Fases 9 y 13) y
ventanas crudas normalizadas para la CNN— por lo que esta función no aplica
ninguna transformación específica de ninguna de las dos.

Justificación científica (breve; referencias completas en ESTADO.md):
ventanas de 4 s están respaldadas por la literatura de detección de fatiga
mental con EEG tanto para modelos clásicos (features espectrales estables
en ese horizonte) como para CNN. El solapamiento del 50 % compensa la corta
duración de cada archivo (~3 min) sin generar redundancia excesiva entre
ventanas consecutivas.
"""

import datetime

import mne
import pandas as pd


def crear_epocas(raw, sujeto, condicion, sesion, archivo_origen, duracion=4.0, solape=2.0):
    """
    Segmenta un Raw en épocas de longitud fija con solapamiento.

    El tramo final que no completa una ventana entera se descarta (ver
    `mne.make_fixed_length_epochs`). No se aplica ninguna transformación a
    la señal: la misma salida alimenta tanto el cálculo de PSD/Welch (Fase
    9) como la preparación de tensores para la CNN.

    Parámetros
    ----------
    raw : mne.io.BaseRaw
        Raw ya recortado, filtrado, evaluado en calidad, re-referenciado y
        limpiado de artefactos oculares (Fases 2-6), con preload=True.
    sujeto : str
        Identificador del sujeto de origen. Se incluye en la metadata de
        cada época (imprescindible para la partición train/test por sujeto
        en fases posteriores, evitando fuga de datos).
    condicion : str
        Identificador de la condición experimental (una de 5 posibles).
    sesion : str o int
        Identificador del momento/sesión de registro (1-6).
    archivo_origen : str
        Ruta o nombre del archivo EEG del que procede este Raw.
    duracion : float
        Duración de cada ventana en segundos. Por defecto 4.0 (1024
        muestras a 256 Hz).
    solape : float
        Solapamiento entre ventanas consecutivas en segundos. Debe cumplir
        0 <= solape < duracion. Por defecto 2.0 (solapamiento del 50 %).

    Devuelve
    --------
    epochs : mne.Epochs
        Épocas de longitud fija, con preload=True y `epochs.metadata`
        (pandas.DataFrame) con una fila por época y las columnas: sujeto,
        condicion, sesion, archivo_origen, indice_ventana, tmin_s (inicio
        de la ventana en segundos relativo al inicio de `raw`).
    metadata : dict
        Diccionario con las claves:
          - metodo               : "ventanas_fijas_solapadas"
          - duracion_s           : duración de cada ventana en segundos
          - solape_s             : solapamiento en segundos
          - paso_s               : paso entre inicios de ventana (duracion - solape)
          - sfreq                : frecuencia de muestreo del Raw
          - n_epocas             : número de épocas generadas
          - duracion_descartada_s: segundos finales descartados por no
                                    completar una ventana entera
          - timestamp            : ISO 8601 del momento de aplicación
    """
    if duracion <= 0:
        raise ValueError(f"duracion debe ser > 0 (recibido: {duracion}).")
    if not (0 <= solape < duracion):
        raise ValueError(
            f"solape debe cumplir 0 <= solape < duracion "
            f"(recibido: solape={solape}, duracion={duracion})."
        )

    duracion_total = raw.times[-1] + 1.0 / raw.info["sfreq"]
    if duracion_total < duracion:
        raise ValueError(
            f"El registro dura {duracion_total:.2f} s, insuficiente para "
            f"crear una ventana de {duracion} s."
        )

    epochs = mne.make_fixed_length_epochs(
        raw, duration=duracion, overlap=solape, preload=True, verbose=False
    )

    n_epocas = len(epochs)
    paso_s = duracion - solape

    epochs.metadata = pd.DataFrame(
        {
            "sujeto": [sujeto] * n_epocas,
            "condicion": [condicion] * n_epocas,
            "sesion": [sesion] * n_epocas,
            "archivo_origen": [archivo_origen] * n_epocas,
            "indice_ventana": range(n_epocas),
            "tmin_s": [i * paso_s for i in range(n_epocas)],
        }
    )

    duracion_usada = (n_epocas - 1) * paso_s + duracion if n_epocas else 0.0
    duracion_descartada = duracion_total - duracion_usada

    metadata = {
        "metodo": "ventanas_fijas_solapadas",
        "duracion_s": float(duracion),
        "solape_s": float(solape),
        "paso_s": float(paso_s),
        "sfreq": float(raw.info["sfreq"]),
        "n_epocas": n_epocas,
        "duracion_descartada_s": float(duracion_descartada),
        "timestamp": datetime.datetime.now().isoformat(),
    }

    print(
        f"Epocado: {n_epocas} ventana(s) de {duracion:g} s "
        f"(paso {paso_s:g} s) — {duracion_descartada:.2f} s descartados al final."
    )

    return epochs, metadata
