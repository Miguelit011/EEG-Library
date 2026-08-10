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

__version__ = "0.9.0"
