"""
Fatiga aguda (tiempo en tarea) — paso 3, reajuste 1: elegir en `train` la
medida más eficiente a igual tiempo.

La configuración del clasificador queda fija (`configuracion_elegida.json`).
Compara tres medidas (`medidas.py`) con la CV repetida de 20 particiones y
elige la de mayor z de la H1 (H1 / error típico bootstrap), con preferencia por
la actual si queda a menos de 0.25. Escribe `medida_elegida.json`.

**Solo `train`.** Plan y regla: `INFORME.md`, «Reajuste 1», escritos antes de
ejecutar.

Ejecutar (desde la raíz del repositorio):
    .venv/Scripts/python.exe analisis/fatiga_aguda/03_clasificador/reajuste_medida.py > analisis/fatiga_aguda/03_clasificador/ejecucion_reajuste.log
"""

import json

import numpy as np
import pandas as pd

import comun
import medidas as md
from comun import FAMILIA, SEMILLAS, d1

CARPETA = comun.CARPETA
CONDICIONES = FAMILIA + ["OA"]
MARGEN_ACTUAL = 0.25


def evaluar(w, medida, feats, mod, pliegues):
    """Valor por grabación de la medida en cada condición, con la predicción media de las particiones."""
    aciertos = {}
    for c in CONDICIONES:
        S, masc = md.segmentos_de_entreno(w, c, medida)
        X = comun.matriz(S, masc, feats)
        y = S.seg.to_numpy()
        if medida == "rho_ventanas":
            filas, Xv = md.ventanas(w, c)
            punt = np.zeros((len(SEMILLAS), len(filas)))
        else:
            punt = np.zeros((len(SEMILLAS), len(S)))
        for i, s in enumerate(SEMILLAS):
            fold = S.persona_uid.map(pliegues[s]).to_numpy()
            fold_v = filas.persona_uid.map(pliegues[s]).to_numpy() if medida == "rho_ventanas" else None
            for k in range(comun.N_PLIEGUES):
                m = comun.modelo(mod).fit(X[fold != k], y[fold != k])
                if medida == "rho_ventanas":
                    punt[i, fold_v == k] = m.decision_function(Xv[fold_v == k])
                else:
                    punt[i, fold == k] = m.predict_proba(X[fold == k])[:, 1]
        if medida == "rho_ventanas":
            aciertos[c] = md.rho_por_grabacion(filas, punt.mean(axis=0))
        else:
            aciertos[c] = comun.aciertos_por_grabacion(S, punt.mean(axis=0))
    return aciertos


def main():
    with open(CARPETA / "configuracion_elegida.json", encoding="utf-8") as f:
        config = json.load(f)
    feats, mod, esq = config["features"], config["modelo"], config["esquema"]
    assert esq == "separado", "El reajuste supone el esquema separado (el elegido)"

    d1._titulo("0. CARGA (solo train) Y CONFIGURACIÓN CONGELADA")
    print(f"Configuración: features = {feats}, modelo = {mod}, esquema = {esq}")
    w = comun.cargar({"train"})
    vers = comun.versiones(w)
    personas = sorted(w.persona_uid.unique())
    pliegues = {s: comun.pliegues_por_persona(personas, s) for s in SEMILLAS}

    filas, todo = [], {}
    for v in ["igual_tiempo", "completa"]:
        d1._titulo(f"1. MEDIDAS — VERSIÓN {v.upper()}")
        for medida in md.MEDIDAS:
            rng = np.random.default_rng(d1.SEMILLA)
            res = md.resumen(evaluar(vers[v], medida, feats, mod, pliegues), medida, rng)
            todo[(v, medida)] = res
            fila = {"version": v, "medida": medida, "H1": res["H1"]["valor"], "H1_ic95": res["H1"]["ic95"],
                    "H1_error_tipico": res["H1"]["error_tipico"], "H1_z": res["H1"]["z"]}
            for c in CONDICIONES + ["familia"]:
                fila[c] = res[c]["valor"]
            for c in FAMILIA:
                fila[f"{c}_z_azar"] = res[c]["z_azar"]
                fila[f"{c}_ic95"] = res[c]["ic95"]
            filas.append(fila)
            print(f"  {medida:16s} PASAT {res['PASAT']['valor']:.3f}  PVTB {res['PVTB']['valor']:.3f}  "
                  f"OA {res['OA']['valor']:.3f}  H1 {res['H1']['valor']:+.3f} "
                  f"[{res['H1']['ic95'][0]:+.3f}, {res['H1']['ic95'][1]:+.3f}]  "
                  f"z(H1) = {res['H1']['z']:.2f}  z(PASAT) = {res['PASAT']['z_azar']:.1f}  "
                  f"z(PVTB) = {res['PVTB']['z_azar']:.1f}")
    tabla = pd.DataFrame(filas)
    (CARPETA / "tablas").mkdir(exist_ok=True)
    tabla.to_csv(CARPETA / "tablas" / "reajuste_medidas.csv", index=False)

    d1._titulo("2. ELECCIÓN (mayor z de la H1 a igual tiempo; la actual si queda a menos de 0.25)")
    t = tabla[tabla.version == "igual_tiempo"].set_index("medida")
    mejor = t.H1_z.idxmax()
    elegida = "tercios_pareado" if t.H1_z["tercios_pareado"] >= t.H1_z[mejor] - MARGEN_ACTUAL else mejor
    print(t[["H1", "H1_error_tipico", "H1_z"]].round(4).to_string())
    print(f"\nMejor z: {mejor} ({t.H1_z[mejor]:.2f}). ELEGIDA: {elegida}")

    with open(CARPETA / "medida_elegida.json", "w", encoding="utf-8") as f:
        json.dump({"fecha": "2026-10-27", "medida": elegida, "azar": md.AZAR[elegida],
                   "regla": f"mayor z(H1) a igual tiempo en train; la actual si queda a menos de {MARGEN_ACTUAL}",
                   "z_train_igual_tiempo": {m: float(t.H1_z[m]) for m in md.MEDIDAS}}, f,
                  ensure_ascii=False, indent=1)
    with open(CARPETA / "resultados_reajuste.json", "w", encoding="utf-8") as f:
        json.dump({"fecha": "2026-10-27", "configuracion": config, "elegida": elegida,
                   "resultados": {f"{v} | {m}": r for (v, m), r in todo.items()},
                   "miradas": {"val": 1, "test": 0}}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
