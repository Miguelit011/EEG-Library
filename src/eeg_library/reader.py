"""
Módulo de lectura de archivos EEG.

Proporciona funciones para cargar señales EEG desde archivos EDF
registrados con el casco Emotiv Epoc X.
"""

import os

import mne


def cargar_eeg(ruta: str) -> mne.io.BaseRaw:
    """
    Carga un archivo EDF y devuelve el objeto Raw de MNE.

    Realiza una validación básica de la frecuencia de muestreo e imprime
    un resumen del archivo sin aplicar ningún procesamiento a los datos.

    Parámetros
    ----------
    ruta : str
        Ruta al archivo EDF que se desea cargar.

    Devuelve
    --------
    mne.io.BaseRaw
        Objeto Raw de MNE con los datos EEG cargados en memoria.

    Raises
    ------
    FileNotFoundError
        Si el archivo indicado en ``ruta`` no existe.
    ValueError
        Si la frecuencia de muestreo del archivo es inferior a 256 Hz.
    """
    
    if not os.path.isfile(ruta):
        raise FileNotFoundError(
            f"No se encontró el archivo EEG: '{ruta}'. "
            "Comprueba que la ruta sea correcta."
        )

    # preload = True -> Carga los datos en la memoria RAM para permitir el filtrado y procesamiento posterior.
    # verbose = False -> Desactiva los mensajes automáticos en la terminal para una ejecución más limpia.
    raw = mne.io.read_raw_edf(ruta, preload=True, verbose=False)

    # MNE interpreta los canales CQ como voltajes y les aplica el factor de
    # escala estándar (1e-6). Revertimos esa escala multiplicando por 10e6
    # para recuperar los valores originales en el rango [0, 4].
    import numpy as np
    canales_cq = [ch for ch in raw.ch_names if ch.startswith("CQ_")]
    if canales_cq:
        indices_cq = [raw.ch_names.index(ch) for ch in canales_cq]
        raw._data[indices_cq] *= 1e6

    # Asigna las posiciones de electrodo del sistema 10-20 estándar. Los 14
    # canales EEG del Epoc X usan nombres 10-20 y quedan georreferenciados;
    # los canales CQ_* no tienen posición estándar y se omiten con
    # on_missing="warn" (aviso, no error). Necesario para la Fase 6 (ICLabel
    # calcula topografías de scalp a partir de estas posiciones).
    raw.set_montage("standard_1020", on_missing="warn", verbose=False)

    frecuencia_muestreo = raw.info["sfreq"]
    if frecuencia_muestreo < 256:
        raise ValueError(
            f"Frecuencia de muestreo insuficiente: {frecuencia_muestreo} Hz. "
            "Se requieren al menos 256 Hz para el análisis de fatiga mental."
        )

    nombre_archivo = os.path.basename(ruta)
    duracion_segundos = raw.times[-1]
    canales = raw.ch_names

    print(f"Archivo       : {nombre_archivo}")
    print(f"Frec. muestreo: {frecuencia_muestreo} Hz")
    print(f"Duración      : {duracion_segundos:.2f} s")
    print(f"Canales ({len(canales)}): {canales}")

    return raw
