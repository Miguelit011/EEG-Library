"""
eeg_library — Librería para el procesamiento de señales EEG.

Facilita la carga, validación y análisis de registros EEG
obtenidos con el casco Emotiv Epoc X, orientada al estudio
de fatiga mental.

Organizada en tres subpaquetes, por etapa del pipeline:
  - preprocesado — Fases 1-6  (EDF crudo -> señal limpia)
  - procesado    — Fases 7-13 (señal limpia -> matriz de features)
  - modelado     — Fase 14 en adelante (evaluación / modelos supervisados)
"""

from eeg_library.preprocesado.reader import cargar_eeg
from eeg_library.preprocesado.filtering import recortar_registro, aplicar_filtros
from eeg_library.preprocesado.quality import evaluar_calidad
from eeg_library.preprocesado.referencing import aplicar_car
from eeg_library.preprocesado.artifacts import (
    entrenar_ica,
    identificar_componentes_oculares,
    eliminar_artefactos_oculares,
)
from eeg_library.procesado.epoching import crear_epocas
from eeg_library.procesado.spectral import calcular_psd
from eeg_library.procesado.bands import extraer_bandas
from eeg_library.procesado.batch import (
    cargar_tareas,
    procesar_archivo,
    procesar_lote,
    consolidar_features,
)
from eeg_library.procesado.features import anotar_validez_de_canales, columnas_de_validez
from eeg_library.procesado.labels import (
    construir_etiquetas,
    corregir_manifest,
    expandir_a_grabaciones,
    normalizar_vas,
    unir_etiquetas,
)
from eeg_library.procesado.matrix import (
    agregar_por_region,
    columnas_de_features,
    construir_matriz_rf,
    REGIONES,
)
from eeg_library.modelado.modeling import (
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
