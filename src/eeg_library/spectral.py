"""
Módulo de análisis espectral EEG (Fase 8).

Calcula la densidad espectral de potencia (PSD) por canal y por época sobre
las épocas ya generadas en la Fase 7, mediante el método de Welch. No reduce
a bandas de frecuencia (delta/theta/alfa/beta) ni a clústeres de canales —
eso es responsabilidad de la Fase 9, que puede operar directamente sobre el
espectro devuelto aquí (por canal individual o agrupando canales).

Justificación científica (breve; referencias completas en ESTADO.md): la
sub-ventana de Welch de 1 s (256 muestras a 256 Hz) con 50 % de solapamiento
y ventana de Hann es una convención estándar en literatura de análisis
espectral EEG; da una resolución en frecuencia de 1 Hz, suficiente para
separar theta (4-8 Hz) de alfa (8-13 Hz). El rango de cálculo se limita a
1-30 Hz porque el filtro bandpass de la Fase 3 ya eliminó el contenido fuera
de esa banda.
"""

import datetime


def calcular_psd(epochs, banda=(1.0, 30.0), duracion_ventana=1.0, solape_ventana=0.5):
    """
    Calcula la PSD (método de Welch) por canal y por época.

    Sub-segmenta cada época en ventanas de `duracion_ventana` segundos con
    `solape_ventana` segundos de solapamiento (función de ventana Hann) y
    promedia los periodogramas resultantes — el procedimiento estándar de
    Welch para estimar la PSD. Solo se calculan las frecuencias dentro de
    `banda`; no se devuelve contenido por encima del límite superior.

    Parámetros
    ----------
    epochs : mne.Epochs
        Épocas ya generadas por `crear_epocas` (Fase 7), con preload=True y
        `epochs.metadata` (sujeto, condición, sesión, archivo_origen).
    banda : tuple[float, float]
        Tupla (fmin, fmax) en Hz del rango de frecuencias a calcular. Por
        defecto (1.0, 30.0) — coincide con el bandpass de la Fase 3, ya que
        no hay contenido real que calcular fuera de ese rango.
    duracion_ventana : float
        Duración de cada sub-ventana de Welch en segundos. Por defecto 1.0
        (256 muestras a 256 Hz) — da una resolución en frecuencia de 1 Hz.
    solape_ventana : float
        Solapamiento entre sub-ventanas consecutivas en segundos. Debe
        cumplir 0 <= solape_ventana < duracion_ventana. Por defecto 0.5
        (50 %, 128 muestras a 256 Hz).

    Devuelve
    --------
    spectrum : mne.time_frequency.EpochsSpectrum
        Espectro con forma (n_epocas, n_canales, n_frecuencias). Conserva
        `spectrum.metadata` (heredada de `epochs.metadata`) y expone
        `spectrum.freqs`, `spectrum.ch_names` y `spectrum.get_data(...)`
        — la Fase 9 puede filtrar por rango de frecuencia o por canal
        directamente sobre este objeto, sin necesitar una estructura propia.
    metadata : dict
        Diccionario con las claves:
          - metodo             : "welch"
          - banda_hz            : [fmin, fmax]
          - duracion_ventana_s  : duración de cada sub-ventana en segundos
          - solape_ventana_s    : solapamiento en segundos
          - paso_ventana_s      : paso entre sub-ventanas (duracion - solape)
          - n_per_seg           : muestras por sub-ventana
          - n_overlap           : muestras de solapamiento
          - n_segmentos         : nº de sub-ventanas promediadas por Welch
                                    (mismo para todas las épocas, ya que
                                    todas tienen igual longitud)
          - ventana             : "hann"
          - resolucion_hz       : resolución en frecuencia (sfreq / n_per_seg)
          - n_epocas            : número de épocas procesadas
          - n_canales           : número de canales EEG incluidos
          - sfreq               : frecuencia de muestreo
          - timestamp           : ISO 8601 del momento de aplicación
    """
    fmin, fmax = banda
    if fmin <= 0 or fmax <= 0 or fmin >= fmax:
        raise ValueError(
            f"Banda invalida: ({fmin}, {fmax}). Se requiere 0 < fmin < fmax."
        )

    sfreq = epochs.info["sfreq"]
    nyquist = sfreq / 2.0
    if fmax >= nyquist:
        raise ValueError(f"fmax={fmax} Hz >= Nyquist ({nyquist} Hz).")

    if duracion_ventana <= 0:
        raise ValueError(f"duracion_ventana debe ser > 0 (recibido: {duracion_ventana}).")
    if not (0 <= solape_ventana < duracion_ventana):
        raise ValueError(
            f"solape_ventana debe cumplir 0 <= solape_ventana < duracion_ventana "
            f"(recibido: solape_ventana={solape_ventana}, duracion_ventana={duracion_ventana})."
        )

    duracion_epoca = len(epochs.times) / sfreq
    if duracion_epoca < duracion_ventana:
        raise ValueError(
            f"Cada época dura {duracion_epoca:.2f} s, insuficiente para una "
            f"sub-ventana de Welch de {duracion_ventana} s."
        )

    n_per_seg = int(round(duracion_ventana * sfreq))
    n_overlap = int(round(solape_ventana * sfreq))
    paso_muestras = n_per_seg - n_overlap
    n_muestras_epoca = len(epochs.times)
    n_segmentos = 1 + (n_muestras_epoca - n_per_seg) // paso_muestras

    spectrum = epochs.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        n_fft=n_per_seg,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        window="hann",
        picks="eeg",
        verbose=False,
    )

    metadata = {
        "metodo": "welch",
        "banda_hz": [float(fmin), float(fmax)],
        "duracion_ventana_s": float(duracion_ventana),
        "solape_ventana_s": float(solape_ventana),
        "paso_ventana_s": float(duracion_ventana - solape_ventana),
        "n_per_seg": n_per_seg,
        "n_overlap": n_overlap,
        "n_segmentos": n_segmentos,
        "ventana": "hann",
        "resolucion_hz": float(sfreq / n_per_seg),
        "n_epocas": len(spectrum),
        "n_canales": len(spectrum.ch_names),
        "sfreq": float(sfreq),
        "timestamp": datetime.datetime.now().isoformat(),
    }

    print(
        f"PSD (Welch) calculada: {metadata['n_epocas']} epoca(s) x "
        f"{metadata['n_canales']} canal(es), banda {fmin:g}-{fmax:g} Hz, "
        f"resolucion {metadata['resolucion_hz']:g} Hz."
    )

    return spectrum, metadata
