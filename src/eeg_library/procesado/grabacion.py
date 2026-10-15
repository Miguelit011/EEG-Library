"""
Análisis de una grabación suelta: el contrato con la plataforma web.

`analizar_grabacion` lleva un EDF hasta todo lo que la web necesita mostrar y
lo devuelve como un diccionario serializable a JSON. La web no calcula nada:
llama a esta función, guarda el resultado y lo pinta.

Decisiones de diseño:

- **Mismo código que el dataset.** Las fases se ejecutan con
  `ejecutar_fases` (la cadena de `procesar_archivo`), la máscara de validez
  con `anotar_validez_de_canales` y el resumen por grabación con
  `resumir_por_grabacion`, que es la parte de `construir_matriz_rf` que no
  necesita VAS ni condición. Así las características de un EDF nuevo son las
  mismas que tendría su fila en `matriz_rf.parquet`.
- **Sin normalización entre grabaciones.** Ninguna característica depende de
  otras grabaciones, de modo que el resultado de un EDF no cambia según con
  qué otros se procese.
- **Nombres neutros hacia fuera.** Las dos columnas que en la matriz se
  llaman `indice_fatiga_*` salen como `indice_theta_alfa_*`: el TFG muestra
  que el índice no sigue a la fatiga percibida y la web no debe sugerirlo.
  `predecir` deshace el alias antes de llamar al modelo.
- **NaN se escribe como `null`.** Una región sin canales válidos no tiene
  valor; se deja vacía en vez de inventarla.
"""

import datetime
import math
import os
import platform

import numpy as np
import pandas as pd

from eeg_library.procesado.bands import BANDAS_DEFECTO
from eeg_library.procesado.batch import ejecutar_fases
from eeg_library.procesado.features import anotar_validez_de_canales
from eeg_library.procesado.matrix import REGIONES, resumir_por_grabacion

# Nombre en la matriz -> nombre en el JSON. Es general, no de un modelo.
ALIAS_CARACTERISTICAS = {
    "indice_fatiga_media": "indice_theta_alfa_media",
    "indice_fatiga_sd": "indice_theta_alfa_sd",
}

_REGION_DE_CANAL = {c: r for r, canales in REGIONES.items() for c in canales}


def columnas_de_caracteristicas(resumen):
    """
    Columnas de entrada de los modelos, con su nombre en la matriz.

    Son las de `columnas_de_features_reposo`: las de `columnas_de_features`
    (Fase 13) menos `condicion_orden` y los seis contadores de procesamiento.
    Sobre la matriz del estudio son 70.
    """
    from eeg_library.modelado.reposo import columnas_de_features_reposo
    return columnas_de_features_reposo(resumen)


def _a_json(valor):
    """Convierte tipos de numpy/pandas a nativos y NaN/inf a None."""
    if isinstance(valor, dict):
        return {str(k): _a_json(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple, np.ndarray)):
        return [_a_json(v) for v in valor]
    if isinstance(valor, (np.bool_, bool)):
        return bool(valor)
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating, float)):
        valor = float(valor)
        return valor if math.isfinite(valor) else None
    if valor is pd.NA:
        return None
    return valor


def _versiones():
    import mne
    import sklearn
    from eeg_library import __version__
    return {
        "eeg_library": __version__,
        "python": platform.python_version(),
        "mne": mne.__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
    }


def analizar_grabacion(ruta_edf, parametros=None, random_state=42):
    """
    Procesa un EDF y devuelve todo lo que la web muestra de él.

    Parámetros
    ----------
    ruta_edf : str o pathlib.Path
        Ruta al archivo EDF.
    parametros : dict o None
        Parámetros de procesado (ver `procesado/parametros.py`). None usa
        `PARAMETROS_ESTUDIO`, los de entrenamiento de los modelos.
    random_state : int
        Semilla de ICA. 42 es la del dataset.

    Devuelve
    --------
    dict serializable con `json.dumps(..., allow_nan=False)`:
      - `metadatos`: archivo, duración, frecuencias de muestreo, canales,
        épocas y versiones.
      - `parametros_aplicados`: los tres parámetros elegibles tal como se
        aplicaron (`referencia`, `banda_hz`, `notch_hz`), si la CAR se aplicó
        de verdad, recorte, ICA (componentes ajustadas y eliminadas) y
        epocado.
      - `calidad`: por canal, región, estado (`bueno` o `descartado`), motivo
        y métricas de `evaluar_calidad`.
      - `potencia`: por canal y banda, media entre épocas de la potencia
        absoluta (V²) y relativa; por región y banda, log₁₀ de la absoluta y
        la relativa, del mismo agregado que la matriz.
      - `caracteristicas`: entradas de los modelos, con alias neutros.

    Raises
    ------
    ErrorDeFase
        Si falla alguna fase del pipeline (indica cuál).
    """
    ruta_edf = os.fspath(ruta_edf)
    features, _, registro, detalle = ejecutar_fases(
        ruta_edf, sujeto=None, condicion=None, sesion=None,
        random_state=random_state, parametros=parametros,
    )
    archivo = registro["archivo"]

    log = pd.DataFrame([{
        "archivo": archivo, "estado": "ok",
        "canales_excluidos": registro["canales_excluidos"],
        "n_canales_buenos": registro["n_canales_buenos"],
    }])
    anotadas = anotar_validez_de_canales(features, log)
    resumen = resumir_por_grabacion(anotadas).iloc[0]

    calidad = detalle["calidad"]
    bandas = list(BANDAS_DEFECTO)

    por_canal = {}
    for canal in detalle["canales_eeg"]:
        valido = canal not in calidad["excluidos"]
        por_canal[canal] = {
            "valido": valido,
            **{b: {"absoluta_v2": features[f"pot_abs_{canal}_{b}"].mean(),
                   "relativa": features[f"pot_rel_{canal}_{b}"].mean()}
               for b in bandas},
        }
    por_region = {
        region: {
            "n_canales_validos": resumen[f"n_validos_{region}"],
            **{b: {"log10_absoluta_v2": resumen[f"{region}_{b}_logabs_media"],
                   "relativa": resumen[f"{region}_{b}_rel_media"]}
               for b in bandas},
        }
        for region in REGIONES
    }

    caracteristicas = {
        ALIAS_CARACTERISTICAS.get(c, c): resumen[c]
        for c in columnas_de_caracteristicas(resumen.to_frame().T)
    }

    ica = detalle["ica"]
    resultado = {
        "metadatos": {
            "archivo": archivo,
            "duracion_s": detalle["carga"]["duracion_s"],
            "sfreq_original_hz": detalle["carga"]["original_sfreq"],
            "sfreq_hz": detalle["carga"]["sfreq"],
            "resampleado": detalle["carga"]["resampled"],
            "canales_eeg": detalle["canales_eeg"],
            "n_canales_buenos": registro["n_canales_buenos"],
            "n_epocas": detalle["epocado"]["n_epocas"],
            "versiones": _versiones(),
            "fecha_analisis": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "parametros_aplicados": {
            **detalle["parametros"],
            "car_aplicada": detalle["car"]["metodo"] == "CAR",
            "recorte_s": {"inicio": detalle["recorte"]["inicio_s"],
                          "fin": detalle["recorte"]["fin_s"]},
            "ica": {
                "metodo": ica["metodo"],
                "n_componentes": ica["n_components"],
                "componentes_eliminadas": ica["componentes_eliminadas"],
                "criterio": "ICLabel 'eye blink' con probabilidad >= 0.8",
                "varianza_eliminada": ica["varianza_explicada_eliminada"],
                "random_state": ica["random_state"],
            },
            "epocado": {"duracion_s": detalle["epocado"]["duracion_s"],
                        "solape_s": detalle["epocado"]["solape_s"]},
            "psd": {k: detalle["psd"][k] for k in
                    ("metodo", "banda_hz", "duracion_ventana_s", "solape_ventana_s",
                     "ventana", "resolucion_hz")},
        },
        "calidad": {
            canal: {
                "region": _REGION_DE_CANAL.get(canal),
                "estado": "descartado" if canal in calidad["excluidos"] else "bueno",
                "motivo": calidad["excluidos"].get(canal),
                "metricas": calidad["metricas"].get(canal),
            }
            for canal in detalle["canales_eeg"]
        },
        "potencia": {"bandas_hz": {b: list(BANDAS_DEFECTO[b]) for b in bandas},
                     "por_canal": por_canal, "por_region": por_region},
        "caracteristicas": caracteristicas,
    }
    return _a_json(resultado)

