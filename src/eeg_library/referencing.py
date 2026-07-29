"""
Módulo de re-referenciación EEG (Fase 5).

Implementa Common Average Reference (CAR) aplicado únicamente sobre
los canales marcados como buenos por la evaluación de calidad (Fase 4).
"""

import datetime

import numpy as np

from eeg_library.quality import CANALES_EPOC


def aplicar_car(raw, good_channels):
    """
    Aplica Common Average Reference (CAR) usando solo los canales buenos.

    La referencia media se calcula como el promedio de las señales de
    good_channels en cada instante, y se sustrae a cada uno de esos canales:

        señal_car[canal] = señal[canal] - mean(señal[good_channels])

    Los canales excluidos por la Fase 4 no participan en el cálculo ni
    son modificados.

    Lógica según el número de canales buenos:
      >= 8  → CAR estándar.
      6 – 7 → CAR aplicada con advertencia en log.
      < 6   → CAR omitida; raw se devuelve sin re-referenciar.

    Parámetros
    ----------
    raw : mne.io.BaseRaw
        Objeto Raw con preload=True.
    good_channels : list[str]
        Canales aptos devueltos por evaluar_calidad.

    Devuelve
    --------
    raw_ref : mne.io.BaseRaw
        Copia del Raw con la señal re-referenciada (o sin modificar si
        good_channels < 6).
    metadata : dict
        Diccionario con las claves:
          - metodo           : "CAR" o "ninguno"
          - canales_incluidos: lista de canales usados en el promedio
          - canales_excluidos: canales EPOC presentes pero no usados
          - timestamp        : ISO 8601 del momento de aplicación
    """
    n_buenos = len(good_channels)
    canales_excluidos = [
        ch for ch in CANALES_EPOC
        if ch in raw.ch_names and ch not in good_channels
    ]

    metadata = {
        "metodo": None,
        "canales_incluidos": list(good_channels),
        "canales_excluidos": canales_excluidos,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    if n_buenos < 6:
        print(
            f"CAR omitida: {n_buenos} canales buenos (minimo 6). "
            "El objeto Raw se devuelve sin re-referenciar."
        )
        metadata["metodo"] = "ninguno"
        return raw.copy(), metadata

    if n_buenos < 8:
        print(
            f"ADVERTENCIA: aplicando CAR con solo {n_buenos} canales "
            "(se recomiendan al menos 8)."
        )

    raw_ref = raw.copy()
    # raw_ref._data es el array (n_canales, n_muestras) cargado en memoria
    indices = [raw_ref.ch_names.index(ch) for ch in good_channels]
    promedio_car = np.mean(raw_ref._data[indices], axis=0)
    for idx in indices:
        raw_ref._data[idx] -= promedio_car

    metadata["metodo"] = "CAR"
    print(f"CAR aplicada sobre {n_buenos} canales: {good_channels}")

    return raw_ref, metadata
