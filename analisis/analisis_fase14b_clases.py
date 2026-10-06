"""
Fase 14b — ¿Aporta algo tratar la fatiga como tres niveles en vez de un número?

Paso previo y barato a la decisión de gastar el test. La Fase 14 mide que la
señal disponible **ordena** (rho ~ +0.12, p < 0.01) pero **no cuantifica**
(R² ~ 0). Una señal así puede rendir mejor en una tarea de tres niveles, donde
solo hay que acertar el escalón, que en una regresión que exige el valor. Este
script comprueba si es el caso — sobre `train` + `val` únicamente.

Si la respuesta es que no aporta, la clasificación **no entra** en el paquete
de test: cada análisis extra sobre el test sube el listón de todos los demás.

Reglas metodológicas que hereda del arnés (`eeg_library.modelado.modeling`):
  - la unidad es la sesión, no la grabación ni la ventana;
  - la partición es por `persona_uid`, nunca por sesión;
  - los cortes de los tercios se estiman DENTRO de cada pliegue de
    entrenamiento y se aplican al de validación;
  - toda media de referencia por persona es leave-one-out;
  - y, por lo anterior, **las métricas se calculan dentro de cada pliegue y se
    promedian**. Amontonar los cinco mezcla escalas —la clase "media" de un
    pliegue no está cortada donde la de otro— y regala señal: un
    `DummyClassifier` sin información saca así 0.407 de exactitud balanceada
    y tau = +0.105 (p = 0.0098). Por pliegue vuelve a su 1/3 exacto.

**No toca `test`.** Se aparta en el paso 1 y no vuelve a aparecer.

Ejecutar (desde la raíz del repositorio):
    python analisis/analisis_fase14b_clases.py

La ubicación del dataset se toma de la variable de entorno `EEG_DATA`, con
`C:\\Users\\migue\\Desktop\\TFG\\EEG-Data` por defecto (ver `analisis/README.md`).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from eeg_library.modelado.modeling import (
    COLUMNA_ETIQUETA,
    COLUMNA_GRUPO,
    NOMBRES_CLASES,
    agregar_a_sesion,
    centrar_por_persona_loo,
    columnas_de_features_sesion,
    metricas_clasificacion,
    metricas_clasificacion_por_pliegue,
    predecir_clases_cv_agrupada,
)

# Ver la nota de `analisis_fase14.py`: sin esto, redirigir la salida a un
# fichero en Windows falla con UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEMILLA = 42
# Ver la nota de `analisis_fase14.py`: ruta configurable por entorno.
RUTA_DATOS = Path(os.environ.get("EEG_DATA", r"C:\Users\migue\Desktop\TFG\EEG-Data"))
RUTA_MATRIZ = RUTA_DATOS / "processed" / "matriz_rf.parquet"
RUTA_RESULTADOS = RUTA_DATOS / "processed" / "fase14b_clases.json"

# Mismo abanico de configuraciones que la Fase 14, en versión clasificador. Se
# reportan TODAS: quedarse con la mejor es el sesgo que este arnés existe para
# evitar.
CLASIFICADORES = {
    "RF 500 árboles, hoja>=5": RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5, random_state=SEMILLA, n_jobs=-1),
    "RF 500 árboles, hoja>=1": RandomForestClassifier(
        n_estimators=500, min_samples_leaf=1, random_state=SEMILLA, n_jobs=-1),
    "RF 500 árboles, hoja>=15": RandomForestClassifier(
        n_estimators=500, min_samples_leaf=15, random_state=SEMILLA, n_jobs=-1),
    "RF 500, hoja>=5, balanceado": RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5, class_weight="balanced",
        random_state=SEMILLA, n_jobs=-1),
}
PRINCIPAL = "RF 500 árboles, hoja>=5"
REGRESOR = RandomForestRegressor(
    n_estimators=500, min_samples_leaf=5, random_state=SEMILLA, n_jobs=-1)


def _titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def _fila(m, etiqueta):
    tau = "  n/d " if np.isnan(m["kendall_tau"]) else f"{m['kendall_tau']:+.3f}"
    return (f"  {etiqueta:<32s} bal={m['exactitud_balanceada']:.4f} "
            f"(±{m['exactitud_balanceada_sd']:.3f})  "
            f"F1={m['f1_macro']:.4f}  ady={m['exactitud_adyacente']:.4f}  "
            f"tau={tau}")


def evaluar(X, y, grupos, modelo, nombre, modo="clasificacion"):
    """Predice out-of-fold y puntúa DENTRO de cada pliegue (ver el docstring)."""
    t, p, detalle = predecir_clases_cv_agrupada(X, y, grupos, modelo, modo=modo)
    m = metricas_clasificacion_por_pliegue(
        t, p, detalle["pliegue_por_fila"], nombre)
    return t, p, detalle, m


def bootstrap_metrica(y_true, y_pred, grupos, pliegues, clave,
                      n=2000, semilla=SEMILLA):
    """
    Intervalo de confianza remuestreando **personas**, no sesiones.

    Remuestrear sesiones trataría las 4-5 sesiones de una persona como
    observaciones independientes y daría un intervalo falsamente estrecho: el n
    efectivo son las personas. Como cada persona vive en un único pliegue de
    validación, remuestrearlas conserva la estructura de pliegues y la métrica
    se sigue promediando dentro de cada uno.
    """
    rng = np.random.default_rng(semilla)
    validas = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[validas], y_pred[validas]
    grupos, pliegues = grupos[validas], pliegues[validas]
    unicos = pd.unique(grupos)
    indices = {g: np.flatnonzero(grupos == g) for g in unicos}
    valores = []
    for _ in range(n):
        muestra = rng.choice(unicos, size=len(unicos), replace=True)
        filas = np.concatenate([indices[g] for g in muestra])
        try:
            valores.append(metricas_clasificacion_por_pliegue(
                y_true[filas], y_pred[filas], pliegues[filas])[clave])
        except ValueError:
            continue
    return float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))


def matriz_confusion(y_true, y_pred):
    """Tabla clase real × clase predicha, con los nombres en español."""
    validas = ~(np.isnan(y_true) | np.isnan(y_pred))
    tabla = pd.crosstab(
        pd.Series(y_true[validas].astype(int), name="real"),
        pd.Series(y_pred[validas].astype(int), name="predicha"))
    tabla = tabla.reindex(index=range(3), columns=range(3), fill_value=0)
    tabla.index = [NOMBRES_CLASES[i] for i in tabla.index]
    tabla.columns = [NOMBRES_CLASES[i] for i in tabla.columns]
    return tabla


def main():
    resultados = {}

    # -- 1. Datos ----------------------------------------------------------
    _titulo("1. DATOS (train + val; test apartado y no vuelve a tocarse)")
    matriz = pd.read_parquet(RUTA_MATRIZ)
    todas = agregar_a_sesion(matriz)
    ses = todas[todas["split"].isin(["train", "val"])].reset_index(drop=True)
    apartadas = todas[todas["split"] == "test"]
    features = columnas_de_features_sesion(ses)
    print(f"Sesiones train+val : {len(ses)}  "
          f"({ses[COLUMNA_GRUPO].nunique()} personas)")
    print(f"Sesiones en test   : {len(apartadas)}  "
          f"({apartadas[COLUMNA_GRUPO].nunique()} personas)  <- APARTADAS")
    print(f"Features por sesión: {len(features)}")
    del apartadas, todas
    resultados["datos"] = {"sesiones": len(ses),
                           "personas": int(ses[COLUMNA_GRUPO].nunique()),
                           "features": len(features)}

    X = ses[features]
    y = ses[COLUMNA_ETIQUETA].to_numpy()
    g = ses[COLUMNA_GRUPO].to_numpy()

    # -- 2. Tarea A: persona nueva, tercios absolutos ----------------------
    _titulo("2. TAREA A — persona nueva, tercios ABSOLUTOS de VAS")
    print("Pregunta: a alguien que el modelo no ha visto nunca, ¿le acierta el")
    print("escalón de fatiga (poca / media / mucha)? Azar = 0.3333.\n")

    _, _, _, m_azar = evaluar(
        X, y, g, DummyClassifier(strategy="most_frequent", random_state=SEMILLA),
        "línea base (mayoritaria)")
    print(_fila(m_azar, "línea base (mayoritaria)"))
    resultados["tarea_a"] = {"linea_base": m_azar}

    predicciones_a = {}
    for nombre, modelo in CLASIFICADORES.items():
        t, p, detalle, m = evaluar(X, y, g, modelo, nombre)
        predicciones_a[nombre] = (t, p, detalle["pliegue_por_fila"])
        resultados["tarea_a"][nombre] = m
        print(_fila(m, nombre))
    resultados["tarea_a"]["cortes_por_pliegue"] = [
        r["cortes"] for r in detalle["reparto"]]

    # ¿aporta clasificar, o basta con redondear la regresión?
    _, _, _, m_reg = evaluar(X, y, g, REGRESOR, "regresión discretizada",
                             modo="regresion")
    resultados["tarea_a"]["regresion_discretizada"] = m_reg
    print(_fila(m_reg, "regresión discretizada"))

    print("\nMatriz de confusión del principal (filas = real, cols = predicha):")
    t, p, pl = predicciones_a[PRINCIPAL]
    print(matriz_confusion(t, p).to_string())

    lo, hi = bootstrap_metrica(t, p, g, pl, "exactitud_balanceada")
    print(f"\nExactitud balanceada {PRINCIPAL}: "
          f"{resultados['tarea_a'][PRINCIPAL]['exactitud_balanceada']:.4f}  "
          f"IC 95% [{lo:.4f}, {hi:.4f}]   (azar = 0.3333)")
    resultados["tarea_a"]["ic_principal"] = [lo, hi]

    # -- 3. Tarea B: persona conocida, tercios de la desviación -----------
    _titulo("3. TAREA B — persona conocida, tercios de la DESVIACIÓN propia")
    print("Pregunta: sabiendo el nivel habitual de esta persona (estimado con")
    print("sus otras sesiones), ¿dice el EEG si hoy está por debajo, igual o")
    print("por encima de lo suyo? Se centran X e y a la vez, para que")
    print("reconocer a la persona no sirva de nada. Azar = 0.3333.\n")

    centradas = centrar_por_persona_loo(ses, features + [COLUMNA_ETIQUETA])
    validas = centradas[COLUMNA_ETIQUETA].notna()
    cen = centradas[validas].reset_index(drop=True)
    Xc = cen[features]
    yc = cen[COLUMNA_ETIQUETA].to_numpy()
    gc = cen[COLUMNA_GRUPO].to_numpy()
    print(f"{len(cen)} sesiones con referencia propia "
          f"({int((~validas).sum())} descartadas: persona con una sola sesión)\n")

    resultados["tarea_b"] = {"n_sesiones": len(cen),
                            "n_personas": int(cen[COLUMNA_GRUPO].nunique())}
    _, _, _, m_azar = evaluar(
        Xc, yc, gc, DummyClassifier(strategy="most_frequent", random_state=SEMILLA),
        "línea base (mayoritaria)")
    print(_fila(m_azar, "línea base (mayoritaria)"))
    resultados["tarea_b"]["linea_base"] = m_azar

    predicciones_b = {}
    for nombre, modelo in CLASIFICADORES.items():
        t, p, detalle, m = evaluar(Xc, yc, gc, modelo, nombre)
        predicciones_b[nombre] = (t, p, detalle["pliegue_por_fila"])
        resultados["tarea_b"][nombre] = m
        print(_fila(m, nombre))

    _, _, _, m_reg = evaluar(Xc, yc, gc, REGRESOR, "regresión discretizada",
                             modo="regresion")
    resultados["tarea_b"]["regresion_discretizada"] = m_reg
    print(_fila(m_reg, "regresión discretizada"))

    print("\nMatriz de confusión del principal:")
    t, p, pl = predicciones_b[PRINCIPAL]
    print(matriz_confusion(t, p).to_string())

    lo, hi = bootstrap_metrica(t, p, gc, pl, "exactitud_balanceada")
    print(f"\nExactitud balanceada {PRINCIPAL}: "
          f"{resultados['tarea_b'][PRINCIPAL]['exactitud_balanceada']:.4f}  "
          f"IC 95% [{lo:.4f}, {hi:.4f}]   (azar = 0.3333)")
    resultados["tarea_b"]["ic_principal"] = [lo, hi]

    # -- 4. El compuesto _rel_sd como clasificador -------------------------
    _titulo("4. EL COMPUESTO _rel_sd EN TRES NIVELES (tarea B)")
    print("La hipótesis candidata a gastar el test, pero evaluada como tres")
    print("niveles en vez de como correlación. Selección de features y ajuste")
    print("del compuesto: solo dentro del pliegue de entrenamiento.\n")

    rel_sd = [c for c in features if c.endswith("_rel_sd")]
    print(f"{len(rel_sd)} features _rel_sd: {sorted(rel_sd)}\n")
    Xr = cen[rel_sd]
    for nombre, modelo in (("RF sobre las 16 _rel_sd", CLASIFICADORES[PRINCIPAL]),
                           ("RF balanceado sobre _rel_sd",
                            CLASIFICADORES["RF 500, hoja>=5, balanceado"])):
        t, p, detalle, m = evaluar(Xr, yc, gc, modelo, nombre)
        resultados.setdefault("rel_sd", {})[nombre] = m
        print(_fila(m, nombre))
        lo, hi = bootstrap_metrica(t, p, gc, detalle["pliegue_por_fila"],
                                   "exactitud_balanceada")
        print(f"  {'':<32s} IC 95% [{lo:.4f}, {hi:.4f}]")
        resultados["rel_sd"][f"{nombre} — IC"] = [lo, hi]

    with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n\nResultados guardados en {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
