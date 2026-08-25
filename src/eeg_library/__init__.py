"""
eeg_library — Librería para el procesamiento de señales EEG.

Facilita la carga, validación y análisis de registros EEG
obtenidos con el casco Emotiv Epoc X, orientada al estudio
de fatiga mental.
"""

from eeg_library.reader import cargar_eeg
from eeg_library.filtering import recortar_registro, aplicar_filtros
from eeg_library.quality import evaluar_calidad
from eeg_library.referencing import aplicar_car
from eeg_library.artifacts import (
    entrenar_ica,
    identificar_componentes_oculares,
    eliminar_artefactos_oculares,
)
from eeg_library.epoching import crear_epocas
from eeg_library.spectral import calcular_psd
from eeg_library.bands import extraer_bandas
from eeg_library.batch import (
    cargar_tareas,
    procesar_archivo,
    procesar_lote,
    consolidar_features,
)
from eeg_library.features import anotar_validez_de_canales, columnas_de_validez
from eeg_library.labels import (
    construir_etiquetas,
    corregir_manifest,
    expandir_a_grabaciones,
    normalizar_vas,
    unir_etiquetas,
)
from eeg_library.matrix import (
    agregar_por_region,
    columnas_de_features,
    construir_matriz_rf,
    REGIONES,
)
from eeg_library.modeling import (
    a_clases,
    agregar_a_sesion,
    centrar_por_persona_loo,
    columnas_de_features_sesion,
    cortes_terciles,
    lineas_base,
    medias_loo_por_grupo,
    metricas,
    metricas_clasificacion,
    metricas_clasificacion_por_pliegue,
    predecir_clases_cv_agrupada,
    predecir_cv_agrupada,
)

__version__ = "0.13.0"
