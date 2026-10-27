"""
Fatiga aguda — paso 2, diagnóstico añadido después de ejecutar: ¿cuánto
depende el acierto pareado de la asignación de personas a los pliegues?

El paso 1 usa `GroupKFold(5)` sin barajar: una sola partición, determinada por
el orden de los datos. Al quitar 40 grabaciones en el paso 2, OA pasó de 0.647
a 0.600 con el mismo clasificador. Este script repite el clasificador del paso
1 (centrado + canales válidos) con 20 particiones por persona barajadas
(`GroupKFold(5, shuffle=True)`, semillas 0–19), con la grabación completa y a
igual tiempo, y da la distribución del acierto pareado y de la diferencia
PASAT + PVTB − OA.

**Solo `train`.** Diagnóstico declarado en `INFORME.md`, no planificado.

Ejecutar (desde la raíz del repositorio):
    .venv/Scripts/python.exe analisis/fatiga_aguda/02_artefactos/estabilidad_pliegues.py > analisis/fatiga_aguda/02_artefactos/ejecucion_estabilidad.log
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent / "01_descriptivo"))
import anexo_igual_tiempo as anexo  # noqa: E402
import descriptivo_fatiga_aguda as d1  # noqa: E402

CONDICIONES = ["PASAT", "PVTB", "PVSAT", "OA"]
FAMILIA = ["PASAT", "PVTB"]
SEMILLAS = range(20)


def aciertos(X, mascara, cv):
    """Acierto por grabación del clasificador del paso 1 con la partición `cv`."""
    F = X[d1.REL] - X[d1.REL].groupby(X.archivo_origen).transform("mean")
    F = F.mask(~mascara, 0.0)
    modelo = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    p = cross_val_predict(modelo, F.to_numpy(), X.seg.to_numpy(), groups=X.persona_uid,
                          cv=cv, method="predict_proba")[:, 1]
    par = X.assign(p=p).pivot_table(index="archivo_origen", columns="seg", values="p").dropna()
    return np.where(par[1] > par[0], 1.0, np.where(par[1] == par[0], 0.5, 0.0))


def main():
    d1._titulo("0. CARGA (solo train)")
    w, _, _ = d1.cargar()
    versiones = {"completa": w, "igual_tiempo": anexo.recortar(w)[0]}

    filas, resumen = [], {}
    for version, wv in versiones.items():
        d1._titulo(f"VERSIÓN {version.upper()}")
        segmentos = {c: d1.segmentos(wv, c) for c in CONDICIONES}
        fija = {c: aciertos(*segmentos[c], GroupKFold(5)) for c in CONDICIONES}
        for s in SEMILLAS:
            cv = GroupKFold(5, shuffle=True, random_state=s)
            a = {c: aciertos(*segmentos[c], cv) for c in CONDICIONES}
            fila = {"version": version, "semilla": s, **{c: a[c].mean() for c in CONDICIONES}}
            fila["familia_menos_oa"] = np.concatenate([a[c] for c in FAMILIA]).mean() - a["OA"].mean()
            filas.append(fila)
        t = pd.DataFrame([f for f in filas if f["version"] == version])
        cols = CONDICIONES + ["familia_menos_oa"]
        fija_fila = {c: fija[c].mean() for c in CONDICIONES}
        fija_fila["familia_menos_oa"] = np.concatenate([fija[c] for c in FAMILIA]).mean() - fija["OA"].mean()
        tabla = pd.DataFrame({"particion_paso_1": pd.Series(fija_fila), "media_20": t[cols].mean(),
                              "sd_20": t[cols].std(), "min_20": t[cols].min(), "max_20": t[cols].max()})
        print(tabla.round(3).to_string())
        resumen[version] = tabla.round(4).to_dict(orient="index")

    (CARPETA / "tablas").mkdir(exist_ok=True)
    pd.DataFrame(filas).to_csv(CARPETA / "tablas" / "estabilidad_pliegues.csv", index=False)
    with open(CARPETA / "resultados_estabilidad.json", "w", encoding="utf-8") as f:
        json.dump({"fecha": "2026-10-27", "semillas": list(SEMILLAS), "resumen": resumen,
                   "miradas": {"val": 0, "test": 0}}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
