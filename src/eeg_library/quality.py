"""
Módulo de evaluación de calidad de señales EEG (Fase 2).

Evalúa la calidad de contacto de cada canal del Emotiv Epoc X y determina
cuáles son aptos para el análisis basándose en señales CQ y métricas de varianza.
"""

import numpy as np

# Los 14 canales EEG del Emotiv Epoc X en orden estándar
CANALES_EPOC = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]


def evaluar_calidad(raw, umbral_flat=1e-12, umbral_spike=1e-7):
    """
    Evalúa la calidad de cada canal EEG y devuelve los canales aptos.

    Aplica cuatro criterios de exclusión (basta cumplir uno):
      1. CQ medio < 3.0
      2. Más del 20 % de muestras con CQ <= 1
      3. Varianza < umbral_flat  (señal plana)
      4. Varianza > umbral_spike (saturación / alta amplitud)

    Los canales CQ se buscan en el propio Raw con el prefijo "CQ_"
    Si no existen, los criterios 1 y 2 se omiten.

    Parámetros
    ----------
    raw : mne.io.BaseRaw
        Objeto Raw con preload=True procedente de cargar_eeg.
    umbral_flat : float
        Varianza mínima (V²) por debajo de la cual la señal se considera plana.
        Por defecto 1e-12 V² .
    umbral_spike : float
        Varianza máxima (V²) por encima de la cual la señal se considera saturada.
        Por defecto 1e-7 V² .

    Devuelve
    --------
    good_channels : list[str]
        Canales que superan todos los criterios de calidad.
    excluidos : dict[str, str]
        Diccionario {canal: motivo_de_exclusión} para cada canal rechazado.
    metricas : dict[str, dict]
        Métricas calculadas por canal: cq_medio, pct_muestras_malas, varianza.
    """
    nombres = raw.ch_names
    data = raw.get_data()  # (n_canales, n_muestras)

    # Solo procesamos los canales EPOC que estén presentes en el archivo
    canales = [ch for ch in CANALES_EPOC if ch in nombres]

    excluidos = {}
    metricas = {}

    for canal in canales:
        idx = nombres.index(canal)
        señal = data[idx]

        # Métricas de calidad de contacto (CQ) 
        nombre_cq = f"CQ_{canal}"
        if nombre_cq in nombres:
            cq = data[nombres.index(nombre_cq)]
            cq_medio = float(np.mean(cq))
            # Calcula la proporción de tiempo con mala señal (True=1, False=0)
            # Si el promedio es 0.25, significa que el 25% de la grabación tuvo calidad <= 1.
            pct_malo = float(np.mean(cq <= 1))
        else:
            cq_medio = None
            pct_malo = None

        varianza = float(np.var(señal))

        metricas[canal] = {
            "cq_medio": cq_medio,
            "pct_muestras_malas": pct_malo,
            "varianza": varianza,
        }

        # Criterios de exclusión (en orden de prioridad)
        if cq_medio is not None and cq_medio < 3.0:
            excluidos[canal] = f"CQ medio bajo ({cq_medio:.2f} < 3.0)"
        elif pct_malo is not None and pct_malo > 0.20:
            excluidos[canal] = f"Exceso de muestras malas ({pct_malo:.1%} > 20 %)"
        elif varianza < umbral_flat:
            excluidos[canal] = f"Señal plana (varianza {varianza:.2e} < {umbral_flat:.2e})"
        elif varianza > umbral_spike:
            excluidos[canal] = f"Saturación (varianza {varianza:.2e} > {umbral_spike:.2e})"

    good_channels = [ch for ch in canales if ch not in excluidos]
    n_buenos = len(good_channels)

    _imprimir_reporte(canales, good_channels, excluidos, metricas)

    if n_buenos < 6:
        print(
            f"ADVERTENCIA CRITICA: solo {n_buenos} canales buenos. "
            "Resultados no fiables (minimo recomendado: 6)."
        )
    elif n_buenos < 8:
        print(f"ADVERTENCIA: {n_buenos} canales buenos (se recomiendan al menos 8).")

    return good_channels, excluidos, metricas


def _imprimir_reporte(canales, good_channels, excluidos, metricas):
    """Imprime el reporte de calidad por canal en consola."""
    
    print("\n--- Reporte de calidad de canales ---")
    print(f"{'Canal':<8} {'CQ medio':>9} {'% malo':>8} {'Varianza':>12}  Estado")
    print("-" * 60)
    for canal in canales:
        m = metricas[canal]
        cq_str = f"{m['cq_medio']:.2f}" if m["cq_medio"] is not None else "N/A"
        pct_str = f"{m['pct_muestras_malas']:.1%}" if m["pct_muestras_malas"] is not None else "N/A"
        var_str = f"{m['varianza']:.2e}"
        if canal in excluidos:
            estado = f"EXCLUIDO — {excluidos[canal]}"
        else:
            estado = "OK"
        print(f"{canal:<8} {cq_str:>9} {pct_str:>8} {var_str:>12}  {estado}")
    print("-" * 60)
    print(f"Canales aptos: {len(good_channels)} / {len(canales)}\n")
