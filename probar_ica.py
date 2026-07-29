"""
Script manual de verificación visual de la Fase 6 (ICA + ICLabel).

Ejecuta el pipeline completo (Fases 1-6) sobre un EDF real de
data/samples/ y guarda gráficas en figures/ para comprobar a simple
vista que el ICA funciona: componentes detectadas, topografías,
probabilidades de ICLabel y la señal antes/después de eliminar las
componentes oculares.

No es parte de la librería, solo apoyo manual de verificación.
"""

import os

import matplotlib
matplotlib.use("Agg")  # sin pantalla: guarda a fichero en vez de mostrar
import matplotlib.pyplot as plt

from eeg_library import (
    cargar_eeg,
    recortar_registro,
    aplicar_filtros,
    evaluar_calidad,
    aplicar_car,
    entrenar_ica,
    identificar_componentes_oculares,
    eliminar_artefactos_oculares,
)
from eeg_library.artifacts import CATEGORIAS_ICLABEL

RUTA_EDF = r"data\samples\M5_OC_F020_C5_XY_EPOCX_236373_2025.04.25T08.33.13+02.00.edf"
CARPETA_FIGURAS = "figures_oc"


def _separador(titulo):
    print("\n" + "=" * 70)
    print(f"  {titulo}")
    print("=" * 70)


os.makedirs(CARPETA_FIGURAS, exist_ok=True)

# ---------------------------------------------------------------------------
# Fases 1-5: carga, recorte, filtros, calidad, CAR
# ---------------------------------------------------------------------------
_separador("Fases 1-5")

raw = cargar_eeg(RUTA_EDF)
raw_rec, _ = recortar_registro(raw)
raw_filt, _ = aplicar_filtros(raw_rec)
good_channels, excluidos, _ = evaluar_calidad(raw_filt)
raw_car, _ = aplicar_car(raw_filt, good_channels)

print(f"\nCanales buenos ({len(good_channels)}): {good_channels}")
print(f"Excluidos: {excluidos if excluidos else 'ninguno'}")

# ---------------------------------------------------------------------------
# Fase 6: ICA + ICLabel
# ---------------------------------------------------------------------------
_separador("Fase 6 — ICA + ICLabel")

ica, meta_ica = entrenar_ica(raw_car, good_channels)

componentes, diagnostico, proba = identificar_componentes_oculares(raw_car, ica)

print(f"\nComponentes clasificadas como oculares (a excluir): {componentes}")
print("Probabilidades por componente y categoría:")
for i in range(proba.shape[0]):
    fila = ", ".join(
        f"{cat}={proba[i, j]:.2f}" for j, cat in enumerate(CATEGORIAS_ICLABEL)
    )
    marca = " <-- EXCLUIDA (ocular)" if i in componentes else ""
    print(f"  IC{i:02d}: {fila}{marca}")

if diagnostico:
    print("\nDiagnóstico de otras categorías (detectadas, NO excluidas):")
    for categoria, entradas in diagnostico.items():
        print(f"  {categoria}: {entradas}")
else:
    print("\nNinguna otra categoría superó el umbral.")

raw_limpio, meta_limpio = eliminar_artefactos_oculares(raw_car, ica, componentes)
print(f"\nMetadata eliminación: {meta_limpio}")

# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------
_separador("Generando gráficas en " + CARPETA_FIGURAS)

# 1. Topografías de todas las componentes ICA
fig = ica.plot_components(show=False)
figs = fig if isinstance(fig, list) else [fig]
for idx, f in enumerate(figs):
    f.savefig(os.path.join(CARPETA_FIGURAS, f"ica_topografias_{idx}.png"), dpi=120)
plt.close("all")

# 2. Señal cruda (fuentes) de cada componente ICA, con las oculares marcadas
fig_fuentes = ica.plot_sources(raw_car, show=False, start=0, stop=20)
fig_fuentes.savefig(os.path.join(CARPETA_FIGURAS, "ica_fuentes.png"), dpi=120)
plt.close("all")

# 3. Señal en canales frontales antes vs. después de eliminar las componentes
canales_frontales = [ch for ch in ["AF3", "AF4", "F7", "F8"] if ch in raw_car.ch_names]
datos_antes = raw_car.get_data(picks=canales_frontales, start=0, stop=int(20 * raw_car.info["sfreq"]))
datos_despues = raw_limpio.get_data(picks=canales_frontales, start=0, stop=int(20 * raw_car.info["sfreq"]))
t = raw_car.times[: datos_antes.shape[1]]

fig, axes = plt.subplots(len(canales_frontales), 1, figsize=(12, 8), sharex=True)
if len(canales_frontales) == 1:
    axes = [axes]
for i, ch in enumerate(canales_frontales):
    axes[i].plot(t, datos_antes[i] * 1e6, label="Antes (con CAR)", alpha=0.7)
    axes[i].plot(t, datos_despues[i] * 1e6, label="Después (sin componentes oculares)", alpha=0.7)
    axes[i].set_ylabel(f"{ch}\n(µV)")
    axes[i].legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("Tiempo (s)")
fig.suptitle("Canales frontales: antes vs. después de eliminar artefactos oculares")
fig.tight_layout()
fig.savefig(os.path.join(CARPETA_FIGURAS, "antes_despues_frontal.png"), dpi=120)
plt.close("all")

print(f"\nListo. Revisa la carpeta '{CARPETA_FIGURAS}/':")
print("  - ica_topografias_*.png : mapa de scalp de cada componente")
print("  - ica_fuentes.png       : señal temporal de cada componente (20 s)")
print("  - antes_despues_frontal.png : señal frontal antes/después de limpiar")
