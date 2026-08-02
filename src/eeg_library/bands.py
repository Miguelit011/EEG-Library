"""
Módulo de extracción de potencia por banda EEG (Fase 9).

Reduce el espectro por canal y por época generado en la Fase 8
(`calcular_psd`) a potencia por banda de frecuencia — por canal individual,
sin promediar por región — y calcula el índice de fatiga theta/alfa a
partir de dos clústeres de canales fijos. No decide el etiquetado de
fatiga (baja/media/alta) ni construye la matriz de features final del
Random Forest — eso es responsabilidad de la Fase 10.

Justificación científica (breve; referencias completas en ESTADO.md):
- Bandas: delta (1-4 Hz), theta (4-8 Hz), alfa (8-13 Hz) y beta (13-30 Hz).
  No se calcula gamma: el filtro bandpass de la Fase 3 ya eliminó ese
  contenido, así que no habría señal real que extraer.
- Índice de fatiga theta/alfa: se calcula sobre theta en el clúster
  frontal (AF3, AF4, F3, F4, F7, F8) y alfa en el clúster posterior
  (P7, P8, O1, O2), no ambos en frontal. Esta separación espacial está
  validada en Emotiv EPOC (mismo hardware de 14 canales que este
  proyecto) por Raufi y Longo (2022, Frontiers in Neuroinformatics) y
  corroborada por un segundo estudio independiente de carga de memoria
  con el mismo hardware.
"""

import datetime

import pandas as pd

# Bandas clásicas de interés para fatiga/carga mental. Gamma se excluye
# porque el bandpass 1-30 Hz de la Fase 3 ya eliminó ese contenido.
BANDAS_DEFECTO = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alfa": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

# Clústeres de canales para el índice de fatiga theta/alfa (Raufi & Longo,
# 2022, sobre el mismo casco Emotiv EPOC de 14 canales): theta se estima
# mejor en frontal, alfa en posterior (parietal/occipital) — no ambos en
# frontal.
CLUSTER_FRONTAL = ["AF3", "AF4", "F3", "F4", "F7", "F8"]
CLUSTER_POSTERIOR = ["P7", "P8", "O1", "O2"]


def extraer_bandas(spectrum, bandas=None, cluster_frontal=None, cluster_posterior=None):
    """
    Extrae potencia por banda (por canal) y el índice de fatiga theta/alfa.

    Para cada banda y cada canal calcula potencia absoluta (suma de la PSD
    en los bins de frecuencia de la banda, ponderada por la resolución en
    frecuencia — aproximación por rectángulos de la integral de la PSD, ya
    que los bins de Welch de la Fase 8 están espaciados uniformemente) y
    potencia relativa (potencia de la banda entre la potencia total del
    espectro calculado). Con las 4 bandas por defecto y 14 canales, produce
    14 x 4 x 2 = 112 columnas de potencia por época.

    Los límites de banda son semiabiertos por la derecha (`[fmin, fmax)`),
    salvo la banda cuyo límite superior alcanza el máximo del espectro
    (`[fmin, fmax]`), para que cada bin de frecuencia pertenezca a
    exactamente una banda — sin solapamiento ni huecos — y las potencias
    relativas de las bandas por defecto sumen 1 por canal.

    El índice de fatiga theta/alfa se calcula como el cociente entre la
    potencia de banda theta promediada sobre `cluster_frontal` y la
    potencia de banda alfa promediada sobre `cluster_posterior`. Requiere
    que `bandas` incluya 'theta' y 'alfa'.

    Parámetros
    ----------
    spectrum : mne.time_frequency.EpochsSpectrum
        Espectro devuelto por `calcular_psd` (Fase 8), con `spectrum.freqs`,
        `spectrum.ch_names`, `spectrum.get_data()` y, opcionalmente,
        `spectrum.metadata` (sujeto, condición, sesión, archivo_origen).
    bandas : dict[str, tuple[float, float]] o None
        Mapeo nombre_banda -> (fmin, fmax) en Hz. Por defecto
        `BANDAS_DEFECTO` (delta, theta, alfa, beta). Debe incluir 'theta'
        y 'alfa' para poder calcular el índice de fatiga.
    cluster_frontal : list[str] o None
        Canales usados para la potencia theta del índice de fatiga. Por
        defecto `CLUSTER_FRONTAL`. Solo se usan los canales presentes en
        `spectrum.ch_names`.
    cluster_posterior : list[str] o None
        Canales usados para la potencia alfa del índice de fatiga. Por
        defecto `CLUSTER_POSTERIOR`. Solo se usan los canales presentes en
        `spectrum.ch_names`.

    Devuelve
    --------
    features : pandas.DataFrame
        Una fila por época. Columnas:
          - metadata heredada de `spectrum.metadata` si existe (sujeto,
            condicion, sesion, archivo_origen, ...), sin modificar.
          - `pot_abs_<canal>_<banda>` y `pot_rel_<canal>_<banda>` por cada
            combinación canal x banda (potencia absoluta y relativa).
          - `theta_frontal`, `alfa_posterior`: potencias de banda
            promediadas sobre cada clúster.
          - `indice_fatiga_theta_alfa`: theta_frontal / alfa_posterior.
    metadata : dict
        Diccionario con las claves:
          - metodo             : "potencia_por_banda_welch"
          - bandas_hz           : dict {nombre_banda: [fmin, fmax]}
          - cluster_frontal     : canales usados en el clúster frontal
          - cluster_posterior   : canales usados en el clúster posterior
          - canales             : lista de canales del espectro (orden usado)
          - resolucion_hz       : resolución en frecuencia del espectro de entrada
          - n_epocas            : número de épocas procesadas
          - n_canales           : número de canales procesados
          - n_features_banda    : nº de columnas de potencia por banda (canales x bandas x 2)
          - timestamp           : ISO 8601 del momento de aplicación
    """
    if bandas is None:
        bandas = dict(BANDAS_DEFECTO)
    if cluster_frontal is None:
        cluster_frontal = list(CLUSTER_FRONTAL)
    if cluster_posterior is None:
        cluster_posterior = list(CLUSTER_POSTERIOR)

    if "theta" not in bandas or "alfa" not in bandas:
        raise ValueError(
            "'bandas' debe incluir 'theta' y 'alfa' para calcular el "
            "indice de fatiga theta/alfa."
        )

    freqs = spectrum.freqs
    ch_names = list(spectrum.ch_names)
    datos = spectrum.get_data()  # (n_epocas, n_canales, n_frecuencias)

    if len(freqs) < 2:
        raise ValueError("El espectro necesita al menos 2 frecuencias para integrar potencia.")
    resolucion_hz = float(freqs[1] - freqs[0])
    fmax_global = float(freqs.max())

    mascaras = _mascaras_de_banda(bandas, freqs, fmax_global)

    potencia_total = datos.sum(axis=2) * resolucion_hz  # (n_epocas, n_canales)

    columnas = {}
    potencias_abs = {}
    for nombre_banda, mascara in mascaras.items():
        pot_abs = datos[:, :, mascara].sum(axis=2) * resolucion_hz  # (n_epocas, n_canales)
        pot_rel = pot_abs / potencia_total
        for i, canal in enumerate(ch_names):
            columnas[f"pot_abs_{canal}_{nombre_banda}"] = pot_abs[:, i]
            columnas[f"pot_rel_{canal}_{nombre_banda}"] = pot_rel[:, i]
        potencias_abs[nombre_banda] = pot_abs

    idx_frontal = _indices_de_cluster(cluster_frontal, ch_names, "frontal")
    idx_posterior = _indices_de_cluster(cluster_posterior, ch_names, "posterior")

    theta_frontal = potencias_abs["theta"][:, idx_frontal].mean(axis=1)
    alfa_posterior = potencias_abs["alfa"][:, idx_posterior].mean(axis=1)

    columnas["theta_frontal"] = theta_frontal
    columnas["alfa_posterior"] = alfa_posterior
    columnas["indice_fatiga_theta_alfa"] = theta_frontal / alfa_posterior

    features = pd.DataFrame(columnas)

    if spectrum.metadata is not None:
        meta = spectrum.metadata.reset_index(drop=True)
        features = pd.concat([meta, features], axis=1)

    n_canales = len(ch_names)
    metadata = {
        "metodo": "potencia_por_banda_welch",
        "bandas_hz": {nombre: [float(fmin), float(fmax)] for nombre, (fmin, fmax) in bandas.items()},
        "cluster_frontal": [c for c in cluster_frontal if c in ch_names],
        "cluster_posterior": [c for c in cluster_posterior if c in ch_names],
        "canales": ch_names,
        "resolucion_hz": resolucion_hz,
        "n_epocas": datos.shape[0],
        "n_canales": n_canales,
        "n_features_banda": len(bandas) * n_canales * 2,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    print(
        f"Potencia por banda calculada: {metadata['n_epocas']} epoca(s) x "
        f"{n_canales} canal(es) x {len(bandas)} banda(s) "
        f"({metadata['n_features_banda']} columnas de potencia)."
    )

    return features, metadata


def _mascaras_de_banda(bandas, freqs, fmax_global):
    """
    Construye, por banda, la máscara booleana de bins de frecuencia que le
    pertenecen. Semiabierta por la derecha salvo la banda que alcanza el
    límite superior del espectro (incluido), para que cada bin pertenezca
    a exactamente una banda.
    """
    mascaras = {}
    for nombre, (fmin, fmax) in bandas.items():
        if fmax >= fmax_global:
            mascara = (freqs >= fmin) & (freqs <= fmax)
        else:
            mascara = (freqs >= fmin) & (freqs < fmax)
        if not mascara.any():
            raise ValueError(
                f"La banda '{nombre}' ({fmin}-{fmax} Hz) no tiene frecuencias "
                f"disponibles en el espectro ({freqs.min():g}-{freqs.max():g} Hz)."
            )
        mascaras[nombre] = mascara
    return mascaras


def _indices_de_cluster(cluster, ch_names, etiqueta):
    """Índices en `ch_names` de los canales de `cluster` presentes en el espectro."""
    indices = [ch_names.index(c) for c in cluster if c in ch_names]
    if not indices:
        raise ValueError(
            f"Ningun canal del cluster {etiqueta} ({cluster}) esta presente "
            f"en el espectro ({ch_names})."
        )
    return indices
