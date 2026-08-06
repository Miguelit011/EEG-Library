"""
Script manual de validación del criterio de amplitud excesiva
basado en z-score robusto (FASTER) — Fase 4.

Ejecuta tres escenarios sobre un EDF real de data/samples/:
  1. Sanity check: pipeline normal, comprobar columna z y métricas.
  2. Estrés del criterio 4: inyectar un outlier sintético en un canal.
  3. Caso degenerado: forzar MAD = 0 copiando un canal a todos.

No es parte de la librería, solo apoyo manual de verificación.
"""

from eeg_library import cargar_eeg, evaluar_calidad
from eeg_library.quality import CANALES_EPOC


RUTA_EDF = r"data\samples\M5_OC_F020_C5_XY_EPOCX_236373_2025.04.25T08.33.13+02.00.edf"


def _separador(titulo):
    print("\n" + "=" * 70)
    print(f"  {titulo}")
    print("=" * 70)


def _mostrar_valores_cq(raw, n_muestras=8):
    """Imprime una muestra de los valores de los canales CQ_ presentes."""
    print(f"\nValores CQ_ (primeros {n_muestras} puntos):")
    encontrados = False
    for ch in CANALES_EPOC:
        nombre_cq = f"CQ_{ch}"
        if nombre_cq in raw.ch_names:
            encontrados = True
            idx = raw.ch_names.index(nombre_cq)
            valores = raw._data[idx]
            muestra = [f"{v:.3f}" for v in valores[:n_muestras]]
            print(f"  {nombre_cq:<8}: {', '.join(muestra)}")
    if not encontrados:
        print("  No se encontraron canales CQ_ en este archivo.")


# ---------------------------------------------------------------------------
# Escenario 1 — Sanity check sobre datos reales
# ---------------------------------------------------------------------------
_separador("Escenario 1 — Sanity check (datos reales sin tocar)")

raw, _ = cargar_eeg(RUTA_EDF)
_mostrar_valores_cq(raw)
good, excluidos, metricas = evaluar_calidad(raw)

print(f"\nCanales aptos: {len(good)} / 14")
print(f"Excluidos    : {excluidos if excluidos else 'ninguno'}")
print("\nResumen z-score por canal:")
for ch in CANALES_EPOC:
    if ch in metricas:
        m = metricas[ch]
        z = m["z_robusto"]
        z_str = f"{z:+.2f}" if z is not None else "N/A"
        print(f"  {ch:<5} varianza={m['varianza']:.2e}   z={z_str}")


# ---------------------------------------------------------------------------
# Escenario 2 — Outlier sintético: multiplicar un canal x1000
# ---------------------------------------------------------------------------
_separador("Escenario 2 — Outlier sintético (T7 x1000)")

raw2, _ = cargar_eeg(RUTA_EDF)
_mostrar_valores_cq(raw2)
canal_outlier = "T7"
idx = raw2.ch_names.index(canal_outlier)
raw2._data[idx] *= 1000

good2, excluidos2, metricas2 = evaluar_calidad(raw2)

print(f"\n¿{canal_outlier} en good_channels?  {canal_outlier in good2}")
print(f"Motivo de exclusión: {excluidos2.get(canal_outlier, '(no excluido)')}")
print(f"z_robusto({canal_outlier}): {metricas2[canal_outlier]['z_robusto']:+.2f}")


# ---------------------------------------------------------------------------
# Escenario 3 — Caso degenerado: MAD = 0 (todos los canales iguales)
# ---------------------------------------------------------------------------
_separador("Escenario 3 — Caso degenerado (todos los canales = AF3)")

raw3, _ = cargar_eeg(RUTA_EDF)
_mostrar_valores_cq(raw3)
señal_referencia = raw3._data[raw3.ch_names.index("AF3")].copy()
for ch in CANALES_EPOC:
    if ch in raw3.ch_names:
        raw3._data[raw3.ch_names.index(ch)] = señal_referencia

good3, excluidos3, metricas3 = evaluar_calidad(raw3)

z_values = {ch: metricas3[ch]["z_robusto"] for ch in CANALES_EPOC if ch in metricas3}
todos_none = all(z is None for z in z_values.values())
print(f"\n¿z_robusto = None en todos los canales? {todos_none}")
exclusiones_amplitud = [m for m in excluidos3.values() if "Amplitud excesiva" in m]
print(f"Exclusiones por 'Amplitud excesiva': {len(exclusiones_amplitud)} (debe ser 0)")
