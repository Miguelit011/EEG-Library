"""
Módulo de lectura de archivos EEG.

Proporciona funciones para cargar señales EEG desde archivos EDF
registrados con el casco Emotiv Epoc X.
"""

import datetime
import os

import mne

TARGET_SFREQ = 256
"""Frecuencia de muestreo objetivo (Hz) a la que se armoniza todo el
dataset. La gran mayoría de registros ya se graban a esta frecuencia, pero
una fracción minoritaria del dataset final (~4.2 %, 252 de 5974 grabaciones
válidas — ver ESTADO.md) se grabó a 128 Hz, probablemente por un lote con
otra configuración de EmotivPro. El Nyquist de 128 Hz (64 Hz) sigue muy por
encima de la banda 1-30 Hz que usa el filtro de la Fase 3, así que esas
grabaciones no tienen menos información útil real para este análisis: el
resampleo es puramente para que todas las señales compartan el mismo número
de muestras por segundo antes del epoching (Fase 7) y del Welch (Fase 8),
no para "recuperar" contenido de alta frecuencia que nunca existió."""


def cargar_eeg(ruta: str) -> tuple[mne.io.BaseRaw, dict]:
    """
    Carga un archivo EDF y devuelve el objeto Raw de MNE.

    Realiza una validación básica de la frecuencia de muestreo, armoniza
    esa frecuencia a TARGET_SFREQ si hace falta, e imprime un resumen del
    archivo sin aplicar ningún otro procesamiento a los datos.

    Parámetros
    ----------
    ruta : str
        Ruta al archivo EDF que se desea cargar.

    Devuelve
    --------
    raw : mne.io.BaseRaw
        Objeto Raw de MNE con los datos EEG cargados en memoria, ya a
        TARGET_SFREQ Hz (resampleado si el archivo original no lo estaba).
    metadata : dict
        Diccionario con las claves:
          - metodo         : "carga_edf"
          - archivo        : nombre del archivo cargado
          - sfreq          : frecuencia de muestreo final (Hz, == TARGET_SFREQ)
          - original_sfreq : frecuencia de muestreo del archivo original (Hz)
          - resampled      : True si se aplicó resampleo a TARGET_SFREQ
          - duracion_s     : duración del registro en segundos
          - n_canales      : número de canales
          - timestamp      : ISO 8601 del momento de carga

    Raises
    ------
    FileNotFoundError
        Si el archivo indicado en ``ruta`` no existe.
    ValueError
        Si la frecuencia de muestreo final es inferior a TARGET_SFREQ.
    """

    if not os.path.isfile(ruta):
        raise FileNotFoundError(
            f"No se encontró el archivo EEG: '{ruta}'. "
            "Comprueba que la ruta sea correcta."
        )

    # preload = True -> Carga los datos en la memoria RAM para permitir el filtrado y procesamiento posterior.
    # verbose = False -> Desactiva los mensajes automáticos en la terminal para una ejecución más limpia.
    raw = mne.io.read_raw_edf(ruta, preload=True, verbose=False)

    # Armonización de frecuencia de muestreo — debe ser el primer paso tras
    # la carga, antes de cualquier otra transformación. La Fase 3 diseña el
    # filtro en función de sfreq, la Fase 7 convierte "ventana de N s" a
    # muestras usando sfreq, y la Fase 8 calcula la resolución espectral a
    # partir de sfreq: si esta armonización no queda garantizada aquí, en
    # el punto de entrada, cualquier archivo grabado a una frecuencia
    # distinta de TARGET_SFREQ puede romper esas fases silenciosamente.
    #
    # Solo se resamplea si hace falta, para no reprocesar innecesariamente
    # el ~96 % de archivos que ya están a TARGET_SFREQ. raw.resample()
    # aplica anti-aliasing por FFT internamente (no es una interpolación
    # lineal ingenua); npad="auto" elige el relleno que minimiza artefactos
    # de borde. La transformación es solo en memoria: no se escribe ni
    # modifica ningún .edf en disco.
    original_sfreq = float(raw.info["sfreq"])
    resampled = original_sfreq != TARGET_SFREQ
    if resampled:
        raw.resample(TARGET_SFREQ, npad="auto", verbose=False)

    # MNE tipa todos los canales del EDF como 'eeg' por defecto, incluidos los
    # que no son señal cerebral (calidad de contacto, telemetría, marcadores).
    # Sin corregir esto, aplicar_filtros (picks="eeg") los filtraría también:
    # p. ej. CQ_* es una señal casi constante y un bandpass 1-30 Hz la
    # colapsaría a ~0, invalidando los criterios de calidad de evaluar_calidad.
    canales_no_eeg = [
        ch for ch in raw.ch_names
        if ch.startswith(("CQ_", "EQ_", "TIME_STAMP", "OR_TIME_STAMP"))
        or ch in (
            "COUNTER", "INTERPOLATED", "RAW_CQ", "BATTERY",
            "BATTERY_PERCENT", "MarkerIndex", "MarkerType",
            "MarkerValueInt", "MARKER_HARDWARE",
        )
    ]
    if canales_no_eeg:
        raw.set_channel_types({ch: "misc" for ch in canales_no_eeg}, verbose=False)

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
    # los canales no-EEG no tienen posición estándar y se omiten con
    # on_missing="warn" (aviso, no error). Necesario para la Fase 6 (ICLabel
    # calcula topografías de scalp a partir de estas posiciones).
    raw.set_montage("standard_1020", on_missing="warn", verbose=False)

    frecuencia_muestreo = raw.info["sfreq"]
    if frecuencia_muestreo < TARGET_SFREQ:
        raise ValueError(
            f"Frecuencia de muestreo insuficiente: {frecuencia_muestreo} Hz. "
            f"Se requieren al menos {TARGET_SFREQ} Hz para el análisis de fatiga mental."
        )

    nombre_archivo = os.path.basename(ruta)
    duracion_segundos = raw.times[-1]
    canales = raw.ch_names

    print(f"Archivo       : {nombre_archivo}")
    print(f"Frec. muestreo: {frecuencia_muestreo} Hz")
    if resampled:
        print(f"Resampleado   : {original_sfreq:g} Hz -> {TARGET_SFREQ:g} Hz")
    print(f"Duración      : {duracion_segundos:.2f} s")
    print(f"Canales ({len(canales)}): {canales}")

    metadata = {
        "metodo": "carga_edf",
        "archivo": nombre_archivo,
        "sfreq": float(frecuencia_muestreo),
        "original_sfreq": original_sfreq,
        "resampled": resampled,
        "duracion_s": float(duracion_segundos),
        "n_canales": len(canales),
        "timestamp": datetime.datetime.now().isoformat(),
    }

    return raw, metadata
