"""
Módulo de filtrado EEG (Fase 4).

Expone dos funciones de preprocesamiento que se aplican en este orden:
  1. recortar_registro — elimina los márgenes inicial y final del registro.
  2. aplicar_filtros   — notch 50 Hz + bandpass FIR zero-phase 1–30 Hz.

Recortar primero es preferible porque los filtros FIR son sensibles a
los bordes de la señal; eliminar los transitorios de colocación/retirada
del casco antes de filtrar reduce los artefactos de borde.

El filtrado solo afecta a canales de tipo EEG. Los canales CQ, markers y
cualquier otro canal no-EEG permanecen intactos.
"""

import datetime

import mne


def recortar_registro(raw, inicio=5.0, fin=5.0):
    """
    Elimina los primeros `inicio` y últimos `fin` segundos del registro.

    Los márgenes suelen contener artefactos de colocación y retirada del
    casco (transitorio de amplificador, movimiento de electrodos). Recortarlos
    antes de filtrar reduce los efectos de borde de los filtros FIR.

    Parámetros
    ----------
    raw : mne.io.BaseRaw
        Objeto Raw con preload=True.
    inicio : float
        Segundos a eliminar al principio. Por defecto 5.0.
    fin : float
        Segundos a eliminar al final. Por defecto 5.0.

    Devuelve
    --------
    raw_rec : mne.io.BaseRaw
        Copia del Raw recortada.
    metadata : dict
        Diccionario con las claves:
          - metodo              : "crop"
          - inicio_s            : segundos eliminados al inicio
          - fin_s               : segundos eliminados al final
          - duracion_original_s : duración original en segundos
          - duracion_final_s    : duración tras el recorte en segundos
          - timestamp           : ISO 8601 del momento de aplicación
    """
    if inicio < 0 or fin < 0:
        raise ValueError("inicio y fin deben ser >= 0.")

    duracion_original = raw.times[-1] + 1.0 / raw.info["sfreq"]

    if inicio + fin >= duracion_original:
        raise ValueError(
            f"El recorte ({inicio} + {fin} s) es mayor o igual a la "
            f"duración del registro ({duracion_original:.1f} s)."
        )

    raw_rec = raw.copy().crop(tmin=inicio, tmax=raw.times[-1] - fin)

    duracion_final = raw_rec.n_times / raw_rec.info["sfreq"]

    metadata = {
        "metodo": "crop",
        "inicio_s": float(inicio),
        "fin_s": float(fin),
        "duracion_original_s": float(duracion_original),
        "duracion_final_s": float(duracion_final),
        "timestamp": datetime.datetime.now().isoformat(),
    }

    print(
        f"Registro recortado: -{inicio:g} s al inicio, -{fin:g} s al final. "
        f"Duración resultante: {duracion_final:.1f} s."
    )

    return raw_rec, metadata


def aplicar_filtros(raw, notch_freq=50.0, banda=(1.0, 30.0)):
    """
    Aplica filtro notch y paso banda FIR zero-phase sobre la señal EEG.

    El notch se aplica primero para eliminar la componente de red antes
    de que el bandpass pueda introducir transitorios. El bandpass usa
    un diseño FIR con ventana (firwin) y fase lineal (phase='zero'),
    lo que preserva las latencias de los eventos sin desfase.

    Solo se filtran los canales de tipo 'eeg'; canales CQ o markers
    se mantienen sin modificar.

    Parámetros
    ----------
    raw : mne.io.BaseRaw
        Objeto Raw con preload=True (tras re-referenciación, idealmente).
    notch_freq : float
        Frecuencia central del filtro notch en Hz. Por defecto 50.0
        (red eléctrica europea).
    banda : tuple[float, float]
        Tupla (l_freq, h_freq) en Hz del paso banda. Por defecto (1.0, 30.0).

    Devuelve
    --------
    raw_filt : mne.io.BaseRaw
        Copia del Raw con los filtros aplicados.
    metadata : dict
        Diccionario con las claves:
          - metodo        : "notch + bandpass FIR zero-phase"
          - notch_hz      : frecuencia del notch aplicada
          - banda_hz      : [l_freq, h_freq] del bandpass
          - fir_design    : "firwin"
          - phase         : "zero"
          - timestamp     : ISO 8601 del momento de aplicación
    """
    l_freq, h_freq = banda

    if l_freq <= 0 or h_freq <= 0 or l_freq >= h_freq:
        raise ValueError(
            f"Banda invalida: ({l_freq}, {h_freq}). "
            "Se requiere 0 < l_freq < h_freq."
        )

    nyquist = raw.info["sfreq"] / 2.0
    if h_freq >= nyquist:
        raise ValueError(
            f"h_freq={h_freq} Hz >= Nyquist ({nyquist} Hz). "
            "Reduzca h_freq o aumente la frecuencia de muestreo."
        )
    if notch_freq >= nyquist:
        raise ValueError(
            f"notch_freq={notch_freq} Hz >= Nyquist ({nyquist} Hz)."
        )

    raw_filt = raw.copy()

    # Filtro notch — elimina la componente de red eléctrica
    raw_filt.notch_filter(
        freqs=notch_freq,
        picks="eeg",
        method="fir",
        fir_design="firwin",
        phase="zero",
        verbose=False,
    )

    # Bandpass FIR zero-phase
    raw_filt.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        picks="eeg",
        method="fir",
        fir_design="firwin",
        phase="zero",
        verbose=False,
    )

    metadata = {
        "metodo": "notch + bandpass FIR zero-phase",
        "notch_hz": float(notch_freq),
        "banda_hz": [float(l_freq), float(h_freq)],
        "fir_design": "firwin",
        "phase": "zero",
        "timestamp": datetime.datetime.now().isoformat(),
    }

    print(
        f"Filtros aplicados: notch {notch_freq:g} Hz + "
        f"bandpass {l_freq:g}-{h_freq:g} Hz (FIR zero-phase)."
    )

    return raw_filt, metadata
