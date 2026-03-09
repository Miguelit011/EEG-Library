from eeg_library import cargar_eeg, evaluar_calidad, aplicar_car
import matplotlib.pyplot as plt

raw = cargar_eeg("data\samples\M5_OC_F020_C5_XY_EPOCX_236373_2025.04.25T08.33.13+02.00.edf")
good_channels, excluidos, metricas = evaluar_calidad(raw)
raw_ref, meta = aplicar_car(raw, good_channels)


