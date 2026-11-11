"""
Fase 14 — Entrenamiento y evaluación del Random Forest sobre el VAS.

Script de análisis reproducible: no es parte de la librería, pero **sí es la
fuente de todas las cifras** que aparecen en MODELADO.md (Fase 14) y en la
memoria. A diferencia de `probar.py` / `probar_ica.py`, que están en la raíz y
son exploración manual, esto es registro: si cambia, cambian los resultados
citados.

**Solo usa `train` + `val`.** El conjunto de test se aparta en el paso 1 y no
vuelve a tocarse: gastarlo es una decisión que se toma una sola vez, a mano y
documentada, cuando haya una hipótesis concreta que confirmar.

Ejecutar (desde la raíz del repositorio):
    python analisis/analisis_fase14.py

La ubicación del dataset se toma de la variable de entorno `EEG_DATA`, con
`EEG-Data` junto al repositorio por defecto (ver `analisis/README.md`).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_library.modelado.modeling import (
    COLUMNA_ETIQUETA,
    COLUMNA_GRUPO,
    agregar_a_sesion,
    centrar_por_persona_loo,
    columnas_de_features_sesion,
    lineas_base,
    metricas,
    predecir_cv_agrupada,
)

# La salida lleva ρ, ² y acentos. En Windows, redirigir a un fichero usa cp1252
# por defecto y el script muere con UnicodeEncodeError justo cuando se intenta
# archivar el resultado, que es el caso de uso que importa.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEMILLA = 42
# El dataset vive fuera del repositorio y fuera de OneDrive (las ventanas
# crudas de la CNN ocupan 16.2 GB y no deben sincronizarse). La ruta es
# configurable para no atar el script a una máquina concreta.
# Por defecto, la carpeta EEG-Data junto al repositorio (TFG/EEG-Data); se cambia con EEG_DATA.
RUTA_DATOS = Path(os.environ.get("EEG_DATA", Path(__file__).resolve().parents[2] / "EEG-Data"))
RUTA_MATRIZ = RUTA_DATOS / "processed" / "matriz_rf.parquet"
RUTA_RESULTADOS = RUTA_DATOS / "processed" / "fase14_resultados.json"

CONFIGURACIONES = {
    "RF 500 árboles, hoja>=5": RandomForestRegressor(
        n_estimators=500, min_samples_leaf=5, random_state=SEMILLA, n_jobs=-1),
    "RF 500 árboles, hoja>=1": RandomForestRegressor(
        n_estimators=500, min_samples_leaf=1, random_state=SEMILLA, n_jobs=-1),
    "RF 500 árboles, hoja>=15": RandomForestRegressor(
        n_estimators=500, min_samples_leaf=15, random_state=SEMILLA, n_jobs=-1),
    "RF 500, hoja>=5, max_features=0.3": RandomForestRegressor(
        n_estimators=500, min_samples_leaf=5, max_features=0.3,
        random_state=SEMILLA, n_jobs=-1),
    "Ridge (imputado + escalado)": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.0)),
}
PRINCIPAL = "RF 500 árboles, hoja>=5"


def _titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def _tabla(filas):
    columnas = ["modelo", "n", "r2", "mae", "rmse", "spearman", "p_spearman"]
    df = pd.DataFrame(filas)
    print(df[[c for c in columnas if c in df.columns]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))


def bootstrap_r2(y_true, y_pred, grupos, n=2000, semilla=SEMILLA):
    """
    Intervalo de confianza del R² remuestreando **personas**, no sesiones.

    Remuestrear sesiones trataría las 4-5 sesiones de una misma persona como
    observaciones independientes y daría un intervalo demasiado estrecho. La
    unidad que se repite en el estudio es la persona, así que es la que se
    remuestrea.
    """
    rng = np.random.default_rng(semilla)
    unicos = pd.unique(grupos)
    indices = {u: np.flatnonzero(grupos == u) for u in unicos}
    valores = []
    for _ in range(n):
        elegidos = rng.choice(unicos, size=len(unicos), replace=True)
        idx = np.concatenate([indices[u] for u in elegidos])
        yt, yp = y_true[idx], y_pred[idx]
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        if ss_tot > 0:
            valores.append(1 - np.sum((yt - yp) ** 2) / ss_tot)
    return float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))


def seleccion_anidada(X, y, grupos, k, n_splits=5):
    """
    Selecciona las `k` features más correlacionadas y las resume en un índice
    compuesto, **repitiendo la selección dentro de cada pliegue**.

    Es la diferencia entre un resultado honesto y uno circular. Si las features
    se eligen mirando todos los datos y luego se evalúa con validación cruzada,
    la elección ya ha visto el conjunto de validación y el resultado sale
    inflado. Aquí la mediana de imputación, la tipificación, el ranking de
    correlaciones y el ajuste de la recta salen **solo** del pliegue de
    entrenamiento.
    """
    features = list(X.columns)
    pred = np.full(len(y), np.nan)
    elegidas_por_pliegue = []

    for idx_tr, idx_va in GroupKFold(n_splits=n_splits).split(X, y, groups=grupos):
        X_tr, y_tr = X.iloc[idx_tr], y[idx_tr]
        mediana = X_tr.median()
        X_tr_, X_va_ = X_tr.fillna(mediana), X.iloc[idx_va].fillna(mediana)

        rhos = np.nan_to_num(np.array(
            [stats.spearmanr(X_tr_[c], y_tr).statistic for c in features]))
        top = np.argsort(-np.abs(rhos))[:k]
        elegidas = [features[i] for i in top]
        elegidas_por_pliegue.append(elegidas)
        signo = np.sign(rhos[top])

        mu = X_tr_[elegidas].mean()
        sd = X_tr_[elegidas].std().replace(0, 1)
        compuesto_tr = (((X_tr_[elegidas] - mu) / sd) * signo).mean(axis=1)
        compuesto_va = (((X_va_[elegidas] - mu) / sd) * signo).mean(axis=1)

        recta = np.polyfit(compuesto_tr.to_numpy(), y_tr, 1)
        pred[idx_va] = np.polyval(recta, compuesto_va.to_numpy())

    conteo = pd.Series([f for p in elegidas_por_pliegue for f in p]).value_counts()
    return pred, sorted(conteo[conteo == n_splits].index)


def main():
    resultados = {}

    # -- 1. Datos ----------------------------------------------------------
    _titulo("1. DATOS")
    matriz = pd.read_parquet(RUTA_MATRIZ)
    todas = agregar_a_sesion(matriz)
    ses = todas[todas["split"].isin(["train", "val"])].reset_index(drop=True)
    apartado = todas[todas["split"] == "test"]

    features = columnas_de_features_sesion(ses)
    X = ses[features]
    y = ses[COLUMNA_ETIQUETA].to_numpy()
    grupos = ses[COLUMNA_GRUPO].to_numpy()

    print(f"Matriz por grabación : {matriz.shape}")
    print(f"Sesiones etiquetadas : {len(todas)}")
    print(f"train+val            : {len(ses)} sesiones, "
          f"{ses[COLUMNA_GRUPO].nunique()} personas")
    print(f"test (APARTADO)      : {len(apartado)} sesiones, "
          f"{apartado[COLUMNA_GRUPO].nunique()} personas  <- no se toca")
    print(f"Features por sesión  : {len(features)}")
    print(f"NaN en X             : {int(X.isna().sum().sum())} celdas "
          f"({100 * X.isna().sum().sum() / X.size:.2f} %)")
    resultados["datos"] = {
        "n_sesiones": len(ses), "n_personas": int(ses[COLUMNA_GRUPO].nunique()),
        "n_features": len(features),
        "n_sesiones_test_apartado": len(apartado),
    }

    # -- 2. Líneas base ----------------------------------------------------
    _titulo("2. LÍNEAS BASE (train+val, nivel sesión)")
    base = lineas_base(ses)
    _tabla(base.to_dict("records"))
    print(f"\nSesiones sin referencia LOO (persona con 1 sola sesión): "
          f"{base.attrs['n_sin_referencia_loo']}")
    resultados["lineas_base"] = base.to_dict("records")

    # -- 3. Control de identidad ------------------------------------------
    _titulo("3. CONTROL: ¿identifican estas features a la persona?")
    cuenta = ses[COLUMNA_GRUPO].value_counts()
    elegibles = cuenta[cuenta >= 3].index
    sub = ses[ses[COLUMNA_GRUPO].isin(elegibles)]
    acierto = cross_val_score(
        RandomForestClassifier(n_estimators=300, random_state=SEMILLA, n_jobs=-1),
        SimpleImputer(strategy="median").fit_transform(sub[features]),
        sub[COLUMNA_GRUPO].to_numpy(),
        cv=StratifiedKFold(3, shuffle=True, random_state=SEMILLA),
        scoring="accuracy").mean()
    azar = 1.0 / len(elegibles)
    print(f"Personas con >=3 sesiones : {len(elegibles)} ({len(sub)} sesiones)")
    print(f"Acierto identificando     : {acierto:.3f}  (azar = {azar:.4f}) "
          f"-> {acierto / azar:.0f}x el azar")
    resultados["control_identidad"] = {
        "n_personas": int(len(elegibles)), "acierto": float(acierto),
        "azar": float(azar)}

    # -- 4. Tarea A: persona nueva ----------------------------------------
    _titulo("4. TAREA A — PERSONA NUEVA (GroupKFold por persona_uid)")
    print("El modelo nunca ha visto a la persona que evalúa. Rival: media global.\n")
    filas = [metricas(y, np.full(len(y), y.mean()), "-- media global --")]
    predicciones = {}
    for nombre, modelo in CONFIGURACIONES.items():
        pred, detalle = predecir_cv_agrupada(X, y, grupos, modelo, n_splits=5)
        predicciones[nombre] = pred
        filas.append(metricas(y, pred, nombre))
    _tabla(filas)
    print("\nReparto de la validación cruzada:")
    print(pd.DataFrame(detalle["reparto"]).to_string(index=False))
    resultados["tarea_a"] = filas

    # -- 5. Tarea B: persona conocida -------------------------------------
    _titulo("5. TAREA B — PERSONA CONOCIDA (desviación respecto a su nivel)")
    print("Se concede al modelo el nivel habitual de la persona, estimado con SUS")
    print("OTRAS sesiones. La pregunta es si el EEG explica la desviación.\n")
    centradas = centrar_por_persona_loo(ses, features + [COLUMNA_ETIQUETA])
    validas = centradas[COLUMNA_ETIQUETA].notna()
    cen = centradas[validas].reset_index(drop=True)
    X_b = cen[features]
    y_b = cen[COLUMNA_ETIQUETA].to_numpy()
    grupos_b = cen[COLUMNA_GRUPO].to_numpy()
    referencia = cen[f"{COLUMNA_ETIQUETA}_loo"].to_numpy()
    y_original = ses.loc[validas.values, COLUMNA_ETIQUETA].to_numpy()
    print(f"Sesiones utilizables: {len(cen)} (descartadas {int((~validas).sum())} "
          f"de personas con 1 sola sesión) | personas: {cen[COLUMNA_GRUPO].nunique()}\n")

    filas_dev = [metricas(y_b, np.zeros(len(y_b)), "-- sin EEG: desviación 0 --")]
    filas_ori = [metricas(y_original, referencia, "-- sin EEG: media persona LOO --")]
    for nombre, modelo in CONFIGURACIONES.items():
        pred_dev, _ = predecir_cv_agrupada(X_b, y_b, grupos_b, modelo, n_splits=5)
        filas_dev.append(metricas(y_b, pred_dev, nombre))
        filas_ori.append(metricas(y_original, referencia + pred_dev, nombre))
    print("Escala de la DESVIACIÓN (lo que el modelo predice):")
    _tabla(filas_dev)
    print("\nReconstruido en la escala ORIGINAL del VAS:")
    _tabla(filas_ori)
    resultados["tarea_b_desviacion"] = filas_dev
    resultados["tarea_b_original"] = filas_ori

    # -- 6. Selección anidada ---------------------------------------------
    _titulo("6. SELECCIÓN DE FEATURES DENTRO DE LA VALIDACIÓN CRUZADA")
    print("Reduce la dimensión sin la circularidad de elegir mirando todo.\n")
    filas_sel = []
    for k in (3, 5, 10, 16, 25):
        pred_sel, estables = seleccion_anidada(X_b, y_b, grupos_b, k)
        m_dev = metricas(y_b, pred_sel, f"compuesto top-{k}")
        m_ori = metricas(y_original, referencia + pred_sel, f"compuesto top-{k}")
        filas_sel.append({**m_dev, "r2_original": m_ori["r2"],
                          "estables": len(estables)})
        print(f"top-{k:>2}: R²(desv) = {m_dev['r2']:+.4f} | "
              f"ρ = {m_dev['spearman']:+.4f} (p = {m_dev['p_spearman']:.4f}) | "
              f"R²(orig) = {m_ori['r2']:+.4f}")
        print(f"        elegidas en los 5 pliegues ({len(estables)}): {estables}")
    resultados["seleccion_anidada"] = filas_sel

    # -- 7. Importancias ---------------------------------------------------
    _titulo("7. ¿EN QUÉ SE FIJA EL BOSQUE? (tarea A, configuración principal)")
    bosque = RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                                   random_state=SEMILLA, n_jobs=-1)
    bosque.fit(SimpleImputer(strategy="median").fit_transform(X), y)
    imp = pd.Series(bosque.feature_importances_, index=features).sort_values(
        ascending=False)
    print(f"Top 10 (una feature neutra valdría 1/{len(features)} = "
          f"{1 / len(features):.4f}):")
    print(imp.head(10).to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nMáxima {imp.max():.4f} — solo "
          f"{imp.max() * len(features):.2f}x lo uniforme.")
    resultados["importancias_top10"] = imp.head(10).to_dict()

    # -- 8. Intervalos de confianza ---------------------------------------
    _titulo("8. INTERVALOS DE CONFIANZA (bootstrap agrupado por persona)")
    lo_a, hi_a = bootstrap_r2(y, predicciones[PRINCIPAL], grupos)
    pred_dev, _ = predecir_cv_agrupada(
        X_b, y_b, grupos_b, CONFIGURACIONES[PRINCIPAL], n_splits=5)
    lo_b, hi_b = bootstrap_r2(y_original, referencia + pred_dev, grupos_b)
    lo_r, hi_r = bootstrap_r2(y_original, referencia, grupos_b)
    print(f"Tarea A ({PRINCIPAL}):")
    print(f"  R² = {metricas(y, predicciones[PRINCIPAL])['r2']:+.4f}  "
          f"IC 95% [{lo_a:+.4f}, {hi_a:+.4f}]")
    print(f"Tarea B (escala original):")
    print(f"  con EEG = {metricas(y_original, referencia + pred_dev)['r2']:+.4f}  "
          f"IC 95% [{lo_b:+.4f}, {hi_b:+.4f}]")
    print(f"  sin EEG = {metricas(y_original, referencia)['r2']:+.4f}  "
          f"IC 95% [{lo_r:+.4f}, {hi_r:+.4f}]")
    resultados["bootstrap"] = {
        "tarea_a": [lo_a, hi_a], "tarea_b_con_eeg": [lo_b, hi_b],
        "tarea_b_sin_eeg": [lo_r, hi_r]}

    with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n\nResultados guardados en {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
