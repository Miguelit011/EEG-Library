"""
eeg_library — Librería para el procesamiento de señales EEG.

Facilita la carga, validación y análisis de registros EEG
obtenidos con el casco Emotiv Epoc X, orientada al estudio
de fatiga mental.
"""

from eeg_library.reader import cargar_eeg
from eeg_library.quality import evaluar_calidad
from eeg_library.referencing import aplicar_car

__version__ = "0.2.0"
