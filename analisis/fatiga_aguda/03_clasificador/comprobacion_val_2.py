"""
Fatiga aguda (tiempo en tarea) — paso 3, reajuste 1: comprobación en `val`
(mirada 2).

Lee la configuración congelada (`configuracion_elegida.json`) y la medida
elegida en `train` (`medida_elegida.json`). Entrena con todo `train` y evalúa
**solo la medida elegida** en las personas de `val`, a igual tiempo
(principal) y con la grabación completa. Aplica la definición de «bueno»
fijada antes de mirar (`INFORME.md`, «Reajuste 1»). No reajusta nada.

**Carga `train` y `val`; nunca `test`.**

Ejecutar (desde la raíz del repositorio), solo después de `reajuste_medida.py`:
    .venv/Scripts/python.exe analisis/fatiga_aguda/03_clasificador/comprobacion_val_2.py > analisis/fatiga_aguda/03_clasificador/ejecucion_val_2.log
"""

import json

import numpy as np

import comprobacion_val as cv1
import comun
import medidas as md
from comun import FAMILIA, d1

CARPETA = comun.CARPETA
CONDICIONES = ["PASAT", "PVTB", "PVSAT", "OA", "OC"]


def evaluar_val(tr, va, medida, feats, mod):
    aciertos = {}
    for c in CONDICIONES:
        S_tr, m_tr = md.segmentos_de_entreno(tr, c, medida)
        m = comun.modelo(mod).fit(comun.matriz(S_tr, m_tr, feats), S_tr.seg.to_numpy())
        if medida == "rho_ventanas":
            filas, Xv = md.ventanas(va, c)
            aciertos[c] = md.rho_por_grabacion(filas, m.decision_function(Xv))
        else:
            S_va, m_va = md.segmentos_de_entreno(va, c, medida)
            aciertos[c] = comun.aciertos_por_grabacion(S_va, m.predict_proba(comun.matriz(S_va, m_va, feats))[:, 1])
    return aciertos


def main():
    with open(CARPETA / "configuracion_elegida.json", encoding="utf-8") as f:
        config = json.load(f)
    with open(CARPETA / "medida_elegida.json", encoding="utf-8") as f:
        medida = json.load(f)["medida"]
    feats, mod = config["features"], config["modelo"]
    azar = md.AZAR[medida]

    d1._titulo("0. CARGA (train + val), CONFIGURACIÓN Y MEDIDA CONGELADAS")
    print(f"Configuración: {feats}, {mod}, {config['esquema']}. Medida: {medida} (azar = {azar})")
    w = comun.cargar({"train", "val"})
    tr = comun.versiones(w[w.split == "train"].reset_index(drop=True))
    va = comun.versiones(w[w.split == "val"].reset_index(drop=True))
    print(f"Personas: train {w[w.split == 'train'].persona_uid.nunique()}, "
          f"val {w[w.split == 'val'].persona_uid.nunique()}")

    resultados = {}
    for v in ["igual_tiempo", "completa"]:
        d1._titulo(f"1. VAL — {medida} — VERSIÓN {v.upper()}")
        rng = np.random.default_rng(d1.SEMILLA)
        res = md.resumen(evaluar_val(tr[v], va[v], medida, feats, mod), medida, rng)
        resultados[v] = res
        for c in ["PASAT", "PVTB", "familia", "PVSAT", "OA", "OC"]:
            x = res[c]
            extra = f"  ({x['grabaciones']} grab., {x['personas']} pers.)" if "grabaciones" in x else ""
            print(f"  {c:8s} {x['valor']:+.3f} [{x['ic95'][0]:+.3f}, {x['ic95'][1]:+.3f}]{extra}")
        h1 = res["H1"]
        print(f"  H1 (PASAT + PVTB − OA) {h1['valor']:+.3f} [{h1['ic95'][0]:+.3f}, {h1['ic95'][1]:+.3f}]  "
              f"z = {h1['z']:.2f}")

    d1._titulo("2. H3 EN VAL (igual que en la mirada 1; no depende de la medida)")
    res_h3 = cv1.h3(va["igual_tiempo"])
    for c, x in res_h3.items():
        print(f"  {c}: diferencia {x['diferencia']:+.4f} (t = {x['t']:+.2f}, p = {x['p']:.3f})")

    d1._titulo("3. ¿ES BUENO VAL? (definición fijada antes de mirar)")
    r = resultados["igual_tiempo"]
    h1_ok = r["H1"]["valor"] > 0
    h2_ok = {c: r[c]["ic95"][0] > azar for c in FAMILIA}
    print(f"  H1 > 0: {r['H1']['valor']:+.3f} → {'sí' if h1_ok else 'NO'}")
    for c in FAMILIA:
        print(f"  H2 {c}: límite inferior {r[c]['ic95'][0]:+.3f} > {azar} → {'sí' if h2_ok[c] else 'NO'}")
    bueno = bool(h1_ok and all(h2_ok.values()))
    print(f"  VAL ES {'BUENO' if bueno else 'NO BUENO'}")

    with open(CARPETA / "resultados_val_2.json", "w", encoding="utf-8") as f:
        json.dump({"fecha": "2026-10-27", "configuracion": config, "medida": medida, "val": resultados,
                   "H3": res_h3, "bueno": {"H1": h1_ok, "H2": h2_ok, "val_bueno": bueno},
                   "miradas": {"val": 2, "test": 0}}, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
