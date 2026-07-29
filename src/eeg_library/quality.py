"""
Módulo de evaluación de calidad de señales EEG (Fase 4).

Evalúa la calidad de contacto de cada canal del Emotiv Epoc X y determina
cuáles son aptos para el análisis basándose en señales CQ y métricas de varianza.
"""

import numpy as np

# Los 14 canales EEG del Emotiv Epoc X en orden estándar
CANALES_EPOC = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]

# Constante de consistencia para el MAD: bajo distribución normal,
# 1.4826 * MAD coincide en esperanza con la desviación estándar.
_MAD_CONSISTENCIA_NORMAL = 1.4826


def evaluar_calidad(raw, umbral_flat=1e-12, umbral_z=3.0):
    """
    Evalúa la calidad de cada canal EEG y devuelve los canales aptos.

    Aplica cuatro criterios de exclusión (basta cumplir uno):
      1. CQ medio < 3.0
      2. Más del 20 % de muestras con CQ <= 1
      3. Varianza < umbral_flat  (señal plana — umbral absoluto fijo)
      4. Varianza con z-score robusto inter-canal > umbral_z
         (amplitud excesiva — umbral relativo)

    Los criterios 3 y 4 detectan fallos distintos y se evalúan con umbrales
    de naturaleza distinta:

    - Una señal casi plana indica un fallo físico del electrodo (ausencia de
      contacto eléctrico) y no depende del resto del registro, por lo que se
      detecta bien con un umbral absoluto fijo.
    - Una amplitud "excesiva" solo tiene sentido en relación con el resto de
      los canales de la misma sesión: depende del participante, del montaje
      y del estado del equipo. Por eso se aplica el enfoque de FASTER
      (Nolan et al., 2010): se calcula la mediana y el MAD de las varianzas
      entre canales y se excluye el canal cuyo z-score robusto supera el
      umbral estándar de FASTER (|z| > 3).

    Los criterios 1 y 2 solo se evalúan si existe el canal `CQ_<nombre>` en
    el Raw. Si no existe, se omiten y solo se aplican los criterios de
    varianza.

    Caso degenerado: si la dispersión inter-canal es nula (MAD = 0, p. ej.
    cuando todas las varianzas son idénticas o cuando más de la mitad de los
    canales están planos), el z-score robusto no está definido y el criterio
    de amplitud excesiva se omite para evitar exclusiones espurias.

    Parámetros
    ----------
    raw : mne.io.BaseRaw
        Objeto Raw con preload=True procedente de cargar_eeg.
    umbral_flat : float
        Varianza mínima (V²) por debajo de la cual la señal se considera
        plana. Por defecto 1e-12 V². Umbral absoluto porque la señal plana
        responde a un fallo físico (sin contacto), independiente del resto
        del registro.
    umbral_z : float
        Umbral del z-score robusto inter-canal por encima del cual la
        varianza se considera anormalmente alta. Por defecto 3.0 (valor
        estándar en FASTER, Nolan et al., 2010).

    Devuelve
    --------
    good_channels : list[str]
        Canales que superan todos los criterios de calidad.
    excluidos : dict[str, str]
        Diccionario {canal: motivo_de_exclusión} para cada canal rechazado.
    metricas : dict[str, dict]
        Métricas calculadas por canal: cq_medio, pct_muestras_malas,
        varianza y z_robusto (None si MAD = 0).
    """
    nombres = raw.ch_names
    data = raw.get_data()  # (n_canales, n_muestras)

    # Solo procesamos los canales EPOC que estén presentes en el archivo
    canales = [ch for ch in CANALES_EPOC if ch in nombres]

    # Primero calculamos las varianzas de todos los canales: las necesitamos
    # para estimar la dispersión inter-canal antes del bucle de exclusión.
    varianzas = {
        canal: float(np.var(data[nombres.index(canal)])) for canal in canales
    }

    # Estadísticos robustos inter-canal (mediana + MAD): la mediana resiste
    # a varios canales atípicos y el MAD reescalado es comparable con la
    # desviación estándar bajo normalidad.
    if varianzas:
        valores = np.array(list(varianzas.values()))
        mediana_var = float(np.median(valores))
        mad_var = float(np.median(np.abs(valores - mediana_var)))
        mad_escalado = _MAD_CONSISTENCIA_NORMAL * mad_var
    else:
        mediana_var = 0.0
        mad_escalado = 0.0

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

        varianza = varianzas[canal]

        # z_robusto = None si la dispersión inter-canal es nula
        if mad_escalado > 0:
            z_robusto = (varianza - mediana_var) / mad_escalado
        else:
            z_robusto = None

        metricas[canal] = {
            "cq_medio": cq_medio,
            "pct_muestras_malas": pct_malo,
            "varianza": varianza,
            "z_robusto": z_robusto,
        }

        # Criterios de exclusión (en orden de prioridad)
        if cq_medio is not None and cq_medio < 3.0:
            excluidos[canal] = f"CQ medio bajo ({cq_medio:.2f} < 3.0)"
        elif pct_malo is not None and pct_malo > 0.20:
            excluidos[canal] = f"Exceso de muestras malas ({pct_malo:.1%} > 20 %)"
        elif varianza < umbral_flat:
            excluidos[canal] = f"Señal plana (varianza {varianza:.2e} < {umbral_flat:.2e})"
        elif z_robusto is not None and z_robusto > umbral_z:
            excluidos[canal] = f"Amplitud excesiva (z={z_robusto:.2f} > {umbral_z:.2f})"

    good_channels = [ch for ch in canales if ch not in excluidos]
    n_buenos = len(good_channels)

    _imprimir_reporte(canales, good_channels, excluidos, metricas)

    if mad_escalado == 0 and len(canales) > 1:
        print(
            "AVISO: dispersion inter-canal nula (MAD = 0). "
            "Criterio de amplitud excesiva (z-score) no aplicado."
        )

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
    print(f"{'Canal':<8} {'CQ medio':>9} {'% malo':>8} {'Varianza':>12} {'z':>7}  Estado")
    print("-" * 70)
    for canal in canales:
        m = metricas[canal]
        cq_str = f"{m['cq_medio']:.2f}" if m["cq_medio"] is not None else "N/A"
        pct_str = f"{m['pct_muestras_malas']:.1%}" if m["pct_muestras_malas"] is not None else "N/A"
        var_str = f"{m['varianza']:.2e}"
        z_str = f"{m['z_robusto']:+.2f}" if m["z_robusto"] is not None else "N/A"
        if canal in excluidos:
            estado = f"EXCLUIDO — {excluidos[canal]}"
        else:
            estado = "OK"
        print(f"{canal:<8} {cq_str:>9} {pct_str:>8} {var_str:>12} {z_str:>7}  {estado}")
    print("-" * 70)
    print(f"Canales aptos: {len(good_channels)} / {len(canales)}\n")
