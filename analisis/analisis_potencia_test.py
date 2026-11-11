"""
Potencia de la prueba principal del test, estimada antes de abrirlo.

Pregunta: si el efecto real fuera el que mide la validación (Fase 14b, Tarea B,
tres niveles, RF 500 hoja>=5, **con control de la deriva por visita**: 0.3795),
¿con qué probabilidad lo detectaría un test de 28 personas con la regla
pre-registrada (p unilateral < 0.05 por permutación dentro de persona)?

Método: se toman las predicciones out-of-fold de train+val y se remuestrean
28 personas con reposición, como si fueran un test nuevo de la misma población.
En cada simulación se puntúa como se puntuará el test (exactitud balanceada
amontonada, un único juego de cortes) y se calcula el p por permutación de las
clases reales dentro de cada persona frente a las predicciones fijas, con la
definición convencional `(1 + #{nulo >= obs}) / (N + 1)`. El mid-p se calcula
solo como comparación.

Supuestos que hay que leer junto a la cifra:
  - el efecto real es exactamente el estimado en validación (si es menor, la
    potencia también);
  - el test tiene el mismo número de sesiones por persona que train+val;
  - los modelos de validación se entrenaron con ~109 personas y el final con
    146, lo que como mucho haría la cifra algo conservadora.

**Solo usa `train` + `val`.** El test se aparta al leer la matriz.

Ejecutar (desde la raíz del repositorio):
    python analisis/analisis_potencia_test.py > resultados/logs/potencia_test.log 2>&1
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from eeg_library.modelado.modeling import (
    COLUMNA_ETIQUETA,
    COLUMNA_GRUPO,
    agregar_a_sesion,
    centrar_por_persona_loo,
    columnas_de_features_sesion,
    metricas_clasificacion,
    metricas_clasificacion_por_pliegue,
    predecir_clases_cv_agrupada,
    residualizar_por_visita,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEMILLA = 42
N_PERSONAS_TEST = 28
N_SIMULACIONES = 5000
N_PERMUTACIONES = 1000      # por simulación; resolución de p = 0.001
N_BOOTSTRAP = 1000          # por simulación, solo para el IC
N_SIMULACIONES_NULAS = 20000  # control de error de tipo I (±0.0015 a 0.05)
ALFA = 0.05
AZAR = 1 / 3
# Por defecto, la carpeta EEG-Data junto al repositorio (TFG/EEG-Data); se cambia con EEG_DATA.
RUTA_DATOS = Path(os.environ.get("EEG_DATA", Path(__file__).resolve().parents[2] / "EEG-Data"))
RUTA_MATRIZ = RUTA_DATOS / "processed" / "matriz_rf.parquet"
RUTA_RESULTADOS = RUTA_DATOS / "processed" / "potencia_test.json"

PRINCIPAL = RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                   random_state=SEMILLA, n_jobs=-1)


def _titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def exactitud_balanceada(y_true, y_pred, n_clases=3):
    """Exactitud balanceada vectorizada: `y_true` puede ser (n,) o (P, n)."""
    y_true = np.atleast_2d(y_true)
    recalls = []
    for c in range(n_clases):
        reales = y_true == c
        n_c = reales.sum(axis=1)
        aciertos = (reales & (y_pred == c)).sum(axis=1)
        recalls.append(np.where(n_c > 0, aciertos / np.maximum(n_c, 1), np.nan))
    return np.nanmean(np.vstack(recalls), axis=0)


def permutaciones_dentro_de_grupo(grupos, n, rng):
    """Índices (n, filas) que barajan las filas solo dentro de cada grupo."""
    _, codigo = np.unique(grupos, return_inverse=True)
    clave = rng.random((n, len(grupos)))
    return np.lexsort((clave, np.broadcast_to(codigo, clave.shape)), axis=1)


def p_convencional(nulo, obs):
    """
    p unilateral por permutación: el observado cuenta como un valor más del
    conjunto de referencia y los empates cuentan enteros en contra. Garantiza
    un error de tipo I <= alfa por construcción. Es la regla del pre-registro.
    """
    return (1 + np.sum(nulo >= obs)) / (len(nulo) + 1)


def p_mid(nulo, obs):
    """
    Mid-p unilateral, solo como comparación: los empates (incluido el propio
    observado) cuentan la mitad. Se descartó para el pre-registro: con el
    control corregido su error de tipo I y su potencia son los del p
    convencional, y no tiene su garantía de validez (MODELADO.md).
    """
    mayores = np.sum(nulo > obs)
    empates = np.sum(nulo == obs) + 1
    return (mayores + 0.5 * empates) / (len(nulo) + 1)


def simular(y_true, y_pred, grupos, rng, n_sim=N_SIMULACIONES, barajar=False):
    """
    Remuestrea N_PERSONAS_TEST personas con reposición y aplica la regla.

    Con `barajar=True` las predicciones se barajan de nuevo en cada simulación
    (control de error de tipo I: un único barajado reutilizado en todas las
    simulaciones arrastraría su propio sesgo como si fuera un efecto).
    """
    unicos = pd.unique(grupos)
    filas = {u: np.flatnonzero(grupos == u) for u in unicos}
    ba, p, p_med, ic_lo, ic_hi, n_ses = [], [], [], [], [], []
    for _ in range(n_sim):
        elegidas = rng.choice(unicos, size=N_PERSONAS_TEST, replace=True)
        bloques = [filas[u] for u in elegidas]
        idx = np.concatenate(bloques)
        # una persona repetida cuenta como dos personas distintas
        g = np.repeat(np.arange(len(bloques)), [len(b) for b in bloques])
        yt, yp = y_true[idx], y_pred[idx]
        if barajar:
            yp = rng.permutation(yp)
        obs = exactitud_balanceada(yt, yp)[0]
        perm = permutaciones_dentro_de_grupo(g, N_PERMUTACIONES, rng)
        nulo = exactitud_balanceada(yt[perm], yp)
        p.append(p_convencional(nulo, obs))
        p_med.append(p_mid(nulo, obs))
        # IC bootstrap por personas: la exactitud balanceada solo depende de
        # los recuentos por persona (sesiones de cada clase y aciertos), así
        # que remuestrear personas es sumar filas de esas dos tablas.
        n_pc = np.stack([np.bincount(g, weights=(yt == c), minlength=len(bloques))
                         for c in range(3)], axis=1)
        a_pc = np.stack([np.bincount(g, weights=(yt == c) & (yp == c),
                                     minlength=len(bloques)) for c in range(3)], axis=1)
        s = rng.integers(0, len(bloques), (N_BOOTSTRAP, len(bloques)))
        n_b, a_b = n_pc[s].sum(axis=1), a_pc[s].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            boots = np.nanmean(np.where(n_b > 0, a_b / n_b, np.nan), axis=1)
        ic_lo.append(np.nanpercentile(boots, 2.5))
        ic_hi.append(np.nanpercentile(boots, 97.5))
        ba.append(obs)
        n_ses.append(len(idx))
    return {k: np.asarray(v) for k, v in
            dict(ba=ba, p=p, p_mid=p_med, ic_lo=ic_lo, ic_hi=ic_hi, n_sesiones=n_ses).items()}


def resumir(nombre, sim, referencia):
    potencia = float(np.mean(sim["p"] < ALFA))
    potencia_mid = float(np.mean(sim["p_mid"] < ALFA))
    excluye_azar = float(np.mean(sim["ic_lo"] > AZAR))
    contiene_ambos = float(np.mean((sim["ic_lo"] <= AZAR) & (sim["ic_hi"] >= referencia)))
    excluye_ref = float(np.mean(sim["ic_hi"] < referencia))
    ee = float(np.std(sim["ba"], ddof=1))
    print(f"\n--- {nombre} (referencia {referencia:.4f}) ---")
    print(f"  Sesiones por test simulado: mediana {np.median(sim['n_sesiones']):.0f} "
          f"[{np.percentile(sim['n_sesiones'], 5):.0f}, "
          f"{np.percentile(sim['n_sesiones'], 95):.0f}]")
    print(f"  Exactitud balanceada simulada: mediana {np.median(sim['ba']):.4f}, "
          f"EE {ee:.4f}, P5-P95 [{np.percentile(sim['ba'], 5):.4f}, "
          f"{np.percentile(sim['ba'], 95):.4f}]")
    print(f"  POTENCIA  P(p unilateral < {ALFA}) = {potencia:.3f}"
          f"   (con mid-p: {potencia_mid:.3f})")
    print(f"  P(IC95 excluye el azar)               = {excluye_azar:.3f}")
    print(f"  P(IC95 contiene el azar y la referencia) = {contiene_ambos:.3f}")
    print(f"  P(IC95 por debajo de la referencia)   = {excluye_ref:.3f}")
    print(f"  Efecto mínimo detectable al 80 % (aprox. (1.645 + 0.842) × EE): "
          f"+{(1.645 + 0.842) * ee:.3f} sobre el azar -> "
          f"{AZAR + (1.645 + 0.842) * ee:.4f}")
    return {"referencia": referencia, "potencia": potencia,
            "potencia_mid_p": potencia_mid,
            "p_ic_excluye_azar": excluye_azar,
            "p_ic_contiene_azar_y_referencia": contiene_ambos,
            "p_ic_por_debajo_de_referencia": excluye_ref,
            "ee_ba": ee, "ba_mediana": float(np.median(sim["ba"])),
            "ba_p5_p95": [float(np.percentile(sim["ba"], 5)),
                          float(np.percentile(sim["ba"], 95))],
            "efecto_minimo_80": (1.645 + 0.842) * ee,
            "sesiones_mediana": float(np.median(sim["n_sesiones"]))}


def main():
    resultados = {"parametros": {
        "n_personas_test": N_PERSONAS_TEST, "n_simulaciones": N_SIMULACIONES,
        "n_permutaciones": N_PERMUTACIONES, "n_bootstrap": N_BOOTSTRAP,
        "alfa_unilateral": ALFA, "semilla": SEMILLA}}

    _titulo("1. PREDICCIONES OUT-OF-FOLD DE LA PRUEBA PRINCIPAL (train + val)")
    matriz = pd.read_parquet(RUTA_MATRIZ)
    matriz = matriz[matriz["split"].isin(["train", "val"])]
    ses = agregar_a_sesion(matriz)
    assert set(ses["split"]) <= {"train", "val"}
    features = columnas_de_features_sesion(ses)
    cen = centrar_por_persona_loo(ses, features + [COLUMNA_ETIQUETA])
    cen = cen[cen[COLUMNA_ETIQUETA].notna()].reset_index(drop=True)
    g = cen[COLUMNA_GRUPO].to_numpy()

    variantes = {
        "con_control": (
            residualizar_por_visita(cen, features, cen["medicion"], g)[features],
            residualizar_por_visita(cen[[COLUMNA_ETIQUETA]], [COLUMNA_ETIQUETA],
                                    cen["medicion"], g)[COLUMNA_ETIQUETA].to_numpy()),
        "sin_control": (cen[features], cen[COLUMNA_ETIQUETA].to_numpy()),
    }
    oof = {}
    for nombre, (X, y) in variantes.items():
        t, p, d = predecir_clases_cv_agrupada(X, y, g, PRINCIPAL)
        pl = d["pliegue_por_fila"]
        por_pliegue = metricas_clasificacion_por_pliegue(t, p, pl)["exactitud_balanceada"]
        amontonada = metricas_clasificacion(t, p)["exactitud_balanceada"]
        td, pd_, dd = predecir_clases_cv_agrupada(
            X, y, g, DummyClassifier(strategy="most_frequent"))
        dummy = metricas_clasificacion(td, pd_)["exactitud_balanceada"]
        print(f"{nombre:<12s} BA por pliegue {por_pliegue:.4f}  amontonada "
              f"{amontonada:.4f}  | Dummy amontonado {dummy:.4f} (debe ser ~0.3333)")
        print(f"{'':<12s} cortes: " + "  ".join(
            f"[{r['cortes'][0]:+.1f}, {r['cortes'][1]:+.1f}]" for r in d["reparto"]))
        oof[nombre] = (t.astype(int), p.astype(int), por_pliegue)
        resultados.setdefault("oof", {})[nombre] = {
            "ba_por_pliegue": por_pliegue, "ba_amontonada": amontonada,
            "dummy_amontonado": dummy}
    print(f"\n{len(cen)} sesiones, {len(pd.unique(g))} personas elegibles "
          f"(>= 2 sesiones)")

    _titulo(f"2. SIMULACIÓN: {N_SIMULACIONES} tests de {N_PERSONAS_TEST} personas")
    print("Regla de decisión: p unilateral < 0.05 por permutación de las clases")
    print("reales dentro de cada persona, con las predicciones fijas.")
    for nombre in ("con_control", "sin_control"):
        t, p, ref = oof[nombre]
        rng = np.random.default_rng(SEMILLA)
        sim = simular(t, p, g, rng)
        resultados[nombre] = resumir(nombre, sim, ref)

    _titulo("3. CONTROL: con predicciones al azar la tasa de rechazo debe ser ~alfa")
    print(f"{N_SIMULACIONES_NULAS} simulaciones, barajando las predicciones de nuevo")
    print("en cada una.\n")
    t, p, _ = oof["con_control"]
    sim = simular(t, p, g, np.random.default_rng(SEMILLA),
                  n_sim=N_SIMULACIONES_NULAS, barajar=True)
    control = {}
    for nombre, clave in (("p convencional", "p"), ("mid-p", "p_mid")):
        for alfa in (0.05, 0.01):
            tasa = float(np.mean(sim[clave] < alfa))
            ee = float(np.sqrt(tasa * (1 - tasa) / N_SIMULACIONES_NULAS))
            print(f"  {nombre:<15s} alfa = {alfa:.2f}: tasa {tasa:.4f} ± {ee:.4f}")
            control[f"{clave}_{alfa}"] = {"tasa": tasa, "ee_monte_carlo": ee}
    distintas = float(np.mean((sim["p"] < ALFA) != (sim["p_mid"] < ALFA)))
    print(f"  Simulaciones en que los dos deciden distinto a {ALFA}: {distintas:.4f}")
    control["decision_distinta"] = distintas
    resultados["control_error_tipo_1"] = control

    with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n\nResultados guardados en {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
