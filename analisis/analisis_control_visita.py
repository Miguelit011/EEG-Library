"""
Control de la deriva por número de visita en los análisis intra-persona del VAS.

Las sesiones de una persona están ordenadas en el tiempo (M2 ... M6). Si el VAS
y el EEG derivan a la vez a lo largo del curso, aparece una relación
intra-persona entre ambos sin que el EEG lea la fatiga. Este guion quita de X y
de y la recta sobre la visita ajustada dentro de cada persona
(`residualizar_por_visita`) y recalcula, con y sin control, las tres cifras
intra-persona que entran en la familia confirmatoria del test:

  1. la prueba principal: Fase 14b, Tarea B, tres niveles, RF 500 hoja>=5;
  2. la pendiente intrapersona VAS ~ índice theta/alfa (`REGRESION_LINEAL.md`);
  3. el compuesto `_rel_sd` de la Fase 13 (ρ intra-persona con el VAS).

Las cifras "sin control" deben reproducir las ya documentadas; las "con
control" son la referencia de validación del pre-registro del test. El control
se reincorpora desde la Fase 15 retirada (ver MODELADO.md, "Control de la
deriva por visita"), donde la prueba principal ya daba 0.3795.

**Solo usa `train` + `val`.** El test se aparta al leer la matriz, antes de
agregar a sesión, y no vuelve a aparecer.

Ejecutar (desde la raíz del repositorio):
    python analisis/analisis_control_visita.py > resultados/logs/control_visita.log 2>&1

La ubicación del dataset se toma de la variable de entorno `EEG_DATA`, con
`C:\\Users\\migue\\Desktop\\TFG\\EEG-Data` por defecto (ver `analisis/README.md`).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

from eeg_library.modelado.modeling import (
    COLUMNA_ETIQUETA,
    COLUMNA_GRUPO,
    NOMBRES_CLASES,
    agregar_a_sesion,
    centrar_por_persona_loo,
    columnas_de_features_sesion,
    compuesto_rel_sd,
    metricas_clasificacion_por_pliegue,
    numero_de_visita,
    predecir_clases_cv_agrupada,
    residualizar_por_visita,
)

# Se reutilizan tal cual las piezas de los guiones cuyas cifras se reproducen,
# para que "sin control" sea exactamente el mismo cálculo que el documentado.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisis_fase14b_clases import bootstrap_metrica, matriz_confusion  # noqa: E402
from analisis_regresion_lineal import ols  # noqa: E402
from analisis_regresion_lineal import bootstrap_por_persona as bootstrap_pendiente  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEMILLA = 42
N_BOOTSTRAP_CLASES = 2000     # el de la Fase 14b
N_BOOTSTRAP_CONTINUO = 10000  # el de REGRESION_LINEAL.md
RUTA_DATOS = Path(os.environ.get("EEG_DATA", r"C:\Users\migue\Desktop\TFG\EEG-Data"))
RUTA_MATRIZ = RUTA_DATOS / "processed" / "matriz_rf.parquet"
RUTA_RESULTADOS = RUTA_DATOS / "processed" / "control_visita.json"

PRINCIPAL = RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                   random_state=SEMILLA, n_jobs=-1)
PREDICTOR_THETA_ALFA = "indice_fatiga_media"


def _titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def _centrar_por_persona(valores, grupos):
    """Resta la media de la propia persona (en muestra), como REGRESION_LINEAL."""
    s = pd.Series(np.asarray(valores, dtype=float))
    return (s - s.groupby(np.asarray(grupos)).transform("mean")).to_numpy()


def _controlar(valores, medicion, grupos):
    """Centrado por persona y, donde hay recta, residuo sobre la visita."""
    centrado = pd.DataFrame({"v": _centrar_por_persona(valores, grupos)})
    res = residualizar_por_visita(centrado, ["v"], medicion, grupos)
    return res["v"].to_numpy(), res.attrs["personas_residualizadas"]


def _bootstrap_spearman(x, y, grupos, n=N_BOOTSTRAP_CONTINUO, semilla=SEMILLA):
    """IC95 de la ρ de Spearman remuestreando personas."""
    rng = np.random.default_rng(semilla)
    grupos = np.asarray(grupos)
    unicos = pd.unique(grupos)
    indices = {u: np.flatnonzero(grupos == u) for u in unicos}
    valores = []
    for _ in range(n):
        idx = np.concatenate([indices[u] for u in
                              rng.choice(unicos, size=len(unicos), replace=True)])
        r = stats.spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(r):
            valores.append(r)
    return [float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))]


def evaluar_principal(X, y, g, nombre):
    t, p, detalle = predecir_clases_cv_agrupada(X, y, g, PRINCIPAL)
    pl = detalle["pliegue_por_fila"]
    m = metricas_clasificacion_por_pliegue(t, p, pl, nombre)
    lo, hi = bootstrap_metrica(t, p, g, pl, "exactitud_balanceada",
                               n=N_BOOTSTRAP_CLASES)
    print(f"\n--- {nombre} ---")
    print(f"  Exactitud balanceada (media de pliegues): "
          f"{m['exactitud_balanceada']:.4f} (SD {m['exactitud_balanceada_sd']:.3f})"
          f"  IC95 [{lo:.4f}, {hi:.4f}]   (azar 0.3333)")
    print("  Por pliegue: " + "  ".join(
        f"{f['exactitud_balanceada']:.3f}" for f in m["por_pliegue"]))
    print(f"  F1 macro {m['f1_macro']:.4f}  adyacente {m['exactitud_adyacente']:.4f}"
          f"  tau {m['kendall_tau']:+.3f}")
    print("  Cortes por pliegue: " + "  ".join(
        f"[{r['cortes'][0]:+.1f}, {r['cortes'][1]:+.1f}]" for r in detalle["reparto"]))
    print("  Matriz de confusión (amontonada; filas = real):")
    print("    " + matriz_confusion(t, p).to_string().replace("\n", "\n    "))
    return {**m, "ic95": [lo, hi],
            "cortes_por_pliegue": [r["cortes"] for r in detalle["reparto"]]}


def main():
    resultados = {}

    # -- 1. Datos ----------------------------------------------------------
    _titulo("1. DATOS (train + val; test apartado antes de agregar)")
    matriz = pd.read_parquet(RUTA_MATRIZ)
    n_test = int((matriz["split"] == "test").sum())
    matriz = matriz[matriz["split"].isin(["train", "val"])]
    ses = agregar_a_sesion(matriz)
    assert set(ses["split"]) <= {"train", "val"}
    features = columnas_de_features_sesion(ses)
    g = ses[COLUMNA_GRUPO].to_numpy()
    print(f"Grabaciones de test apartadas: {n_test} (no se agregan ni se leen)")
    print(f"Sesiones train+val: {len(ses)} ({ses[COLUMNA_GRUPO].nunique()} personas)")
    sesiones_por_persona = ses.groupby(COLUMNA_GRUPO).size()
    print("Sesiones por persona: " + ", ".join(
        f"{k}: {v}" for k, v in sesiones_por_persona.value_counts().sort_index().items()))
    print(f"Personas con recta de visita (>= 3 sesiones): "
          f"{int((sesiones_por_persona >= 3).sum())}")
    resultados["datos"] = {
        "sesiones": len(ses), "personas": int(ses[COLUMNA_GRUPO].nunique()),
        "features": len(features),
        "personas_con_recta": int((sesiones_por_persona >= 3).sum()),
        "sesiones_por_persona": {int(k): int(v) for k, v in
                                 sesiones_por_persona.value_counts().items()},
    }

    # -- 2. El mecanismo ---------------------------------------------------
    _titulo("2. ¿DERIVA EL VAS CON LA VISITA DENTRO DE LA PERSONA?")
    visita = numero_de_visita(ses["medicion"])
    varias = ses[COLUMNA_GRUPO].map(sesiones_por_persona).to_numpy() > 1
    v_c = _centrar_por_persona(visita, g)[varias]
    y_c = _centrar_por_persona(ses[COLUMNA_ETIQUETA], g)[varias]
    r = stats.spearmanr(v_c, y_c)
    ic = _bootstrap_spearman(v_c, y_c, g[varias])
    print(f"ρ intra-persona (VAS, visita) = {r.statistic:+.4f}  "
          f"p = {r.pvalue:.4f}  IC95 personas [{ic[0]:+.4f}, {ic[1]:+.4f}]  "
          f"n = {int(varias.sum())}")
    resultados["deriva_vas"] = {"spearman": float(r.statistic),
                                "p_spearman": float(r.pvalue), "ic95": ic,
                                "n": int(varias.sum())}

    # -- 3. La prueba principal --------------------------------------------
    _titulo("3. PRUEBA PRINCIPAL — Fase 14b, Tarea B, tres niveles, RF hoja>=5")
    print("Mismo cálculo que la Fase 14b. El control se aplica a X e y ya")
    print("centrados por la media LOO de la persona; los cortes de los tercios")
    print("siguen saliendo de cada pliegue de entrenamiento.")
    centradas = centrar_por_persona_loo(ses, features + [COLUMNA_ETIQUETA])
    cen = centradas[centradas[COLUMNA_ETIQUETA].notna()].reset_index(drop=True)
    gc = cen[COLUMNA_GRUPO].to_numpy()
    yc = cen[COLUMNA_ETIQUETA].to_numpy()
    print(f"\n{len(cen)} sesiones, {len(pd.unique(gc))} personas")

    sin = evaluar_principal(cen[features], yc, gc, "sin control")
    x_res = residualizar_por_visita(cen, features, cen["medicion"], gc)
    y_res = residualizar_por_visita(cen[[COLUMNA_ETIQUETA]], [COLUMNA_ETIQUETA],
                                    cen["medicion"], gc)
    con = evaluar_principal(x_res[features], y_res[COLUMNA_ETIQUETA].to_numpy(),
                            gc, "con control de visita")
    var_antes = float(pd.Series(yc).groupby(gc).var().mean())
    var_despues = float(y_res[COLUMNA_ETIQUETA].groupby(gc).var().mean())
    print(f"\nPersonas residualizadas: {y_res.attrs['personas_residualizadas']} "
          f"de {len(pd.unique(gc))}")
    print(f"Varianza intra-persona media de la desviación del VAS: "
          f"{var_antes:.1f} -> {var_despues:.1f} "
          f"(queda el {100 * var_despues / var_antes:.0f} %)")
    resultados["principal"] = {
        "sin_control": sin, "con_control": con,
        "personas_residualizadas": int(y_res.attrs["personas_residualizadas"]),
        "varianza_intra_vas": [var_antes, var_despues]}

    # -- 4. Índice theta/alfa intrapersona ---------------------------------
    _titulo("4. VAS ~ ÍNDICE THETA/ALFA, INTRAPERSONA")
    x = ses[PREDICTOR_THETA_ALFA].to_numpy(dtype=float)
    y = ses[COLUMNA_ETIQUETA].to_numpy(dtype=float)
    n_personas = int(ses[COLUMNA_GRUPO].nunique())
    # Cada media personal gasta un grado de libertad (como en REGRESION_LINEAL)
    # y cada recta de visita, otro más.
    x_con, con_recta = _controlar(x, ses["medicion"], g)
    y_con, _ = _controlar(y, ses["medicion"], g)
    variantes = {
        "sin_control": (_centrar_por_persona(x, g), _centrar_por_persona(y, g),
                        n_personas),
        "con_control": (x_con, y_con, n_personas + con_recta),
    }
    resultados["theta_alfa"] = {}
    for nombre, (dx, dy, gl_extra) in variantes.items():
        ajuste = ols(dx, dy, gl_extra=gl_extra)
        boot = bootstrap_pendiente(dx, dy, g, n=N_BOOTSTRAP_CONTINUO)
        print(f"\n--- {nombre} ---")
        print(f"  β = {ajuste['pendiente']:+.3f} mm  IC95 t "
              f"[{ajuste['ic95_pendiente'][0]:+.3f}, {ajuste['ic95_pendiente'][1]:+.3f}]"
              f"  IC95 personas [{boot['ic95_pendiente'][0]:+.3f}, "
              f"{boot['ic95_pendiente'][1]:+.3f}]")
        print(f"  p = {ajuste['p']:.3f}  R² = {ajuste['r2']:.5f}  "
              f"ρ = {ajuste['spearman']:+.4f} (p = {ajuste['p_spearman']:.3f})  "
              f"gl = {ajuste['gl']}")
        resultados["theta_alfa"][nombre] = {**ajuste, "bootstrap_personas": boot}

    # -- 5. Compuesto _rel_sd ----------------------------------------------
    _titulo("5. COMPUESTO _rel_sd — ρ INTRA-PERSONA CON EL VAS")
    print("z de cada una de las 16 `_rel_sd` con media y SD de estas sesiones,")
    print("media de las 16, centrado por persona y Spearman. Personas con >= 2")
    print("sesiones (con una sola, el centrado deja 0).")
    comp = compuesto_rel_sd(ses, ses).to_numpy()
    resultados["rel_sd"] = {}
    for nombre, (cx, cy) in {
        "sin_control": (_centrar_por_persona(comp, g), _centrar_por_persona(y, g)),
        "con_control": (_controlar(comp, ses["medicion"], g)[0],
                        _controlar(y, ses["medicion"], g)[0]),
    }.items():
        r = stats.spearmanr(cx[varias], cy[varias])
        ic = _bootstrap_spearman(cx[varias], cy[varias], g[varias])
        print(f"  {nombre:<12s} ρ = {r.statistic:+.4f}  p = {r.pvalue:.5f}  "
              f"IC95 personas [{ic[0]:+.4f}, {ic[1]:+.4f}]  n = {int(varias.sum())}")
        resultados["rel_sd"][nombre] = {"spearman": float(r.statistic),
                                        "p_spearman": float(r.pvalue),
                                        "ic95": ic, "n": int(varias.sum())}

    with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n\nResultados guardados en {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
