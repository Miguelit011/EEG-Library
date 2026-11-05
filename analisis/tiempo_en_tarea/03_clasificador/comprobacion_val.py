"""
Tiempo en tarea — paso 3, parte 2: comprobación en `val`
(mirada 1).

Lee la configuración congelada por `seleccion.py` (`configuracion_elegida.json`),
la entrena con todo `train` y la evalúa una vez en las personas de `val`, con
la grabación completa y a igual tiempo. Calcula H1, H1b, H2 y H3 y aplica la
definición de «bueno» fijada antes de mirar (`INFORME.md`). No reajusta nada.

**Carga `train` y `val`; nunca `test`.**

Ejecutar (desde la raíz del repositorio), solo después de `seleccion.py`:
    .venv/Scripts/python.exe analisis/tiempo_en_tarea/03_clasificador/comprobacion_val.py > analisis/tiempo_en_tarea/03_clasificador/ejecucion_val.log
"""

import json

import numpy as np
import pandas as pd
from scipy import stats

import comun
from comun import FAMILIA, anexo, d1

CARPETA = comun.CARPETA
VERSIONES = ["igual_tiempo", "completa"]


def predecir(tr, va, feats, mod, esq):
    """Entrena con `train` y devuelve la probabilidad de cada tercio de `val`, por condición."""
    S_tr = {c: comun.tercios(tr, c) for c in comun.CONDICIONES}
    S_va = {c: comun.tercios(va, c) for c in comun.CONDICIONES}
    X = lambda S, c: comun.matriz(*S[c], feats)  # noqa: E731
    y = lambda c: S_tr[c][0].seg.to_numpy()  # noqa: E731
    probs = {}
    if esq == "conjunto":
        m = comun.modelo(mod).fit(np.vstack([X(S_tr, c) for c in FAMILIA]), np.concatenate([y(c) for c in FAMILIA]))
        for c in FAMILIA:
            probs[c] = m.predict_proba(X(S_va, c))[:, 1]
    for c in comun.CONDICIONES:
        if c not in probs:
            probs[c] = comun.modelo(mod).fit(X(S_tr, c), y(c)).predict_proba(X(S_va, c))[:, 1]
    return {c: comun.aciertos_por_grabacion(S_va[c][0], p) for c, p in probs.items()}


def con_ic(aciertos, rng):
    salida = {}
    for c, x in aciertos.items():
        porp = x.groupby("persona_uid").acierto.agg(["sum", "size"])
        ic = d1.bootstrap_personas(porp["sum"].to_numpy(), porp["size"].to_numpy(), rng)
        salida[c] = {"pareado": x.acierto.mean(), "ic95": [float(i) for i in ic], "grabaciones": len(x),
                     "personas": int(x.persona_uid.nunique())}
    fam = pd.concat([aciertos[c] for c in FAMILIA], ignore_index=True)
    porp = fam.groupby("persona_uid").acierto.agg(["sum", "size"])
    ic = d1.bootstrap_personas(porp["sum"].to_numpy(), porp["size"].to_numpy(), rng)
    salida["familia"] = {"pareado": fam.acierto.mean(), "ic95": [float(i) for i in ic]}
    dif, ic = anexo.diferencia_bootstrap(fam, aciertos["OA"], rng)
    salida["H1"] = {"diferencia": dif, "ic95": [float(i) for i in ic]}
    return salida


def h3(va):
    """Pendiente por minuto de la beta posterior relativa, tarea − OA pareada por persona."""
    nombre, pref, reg, banda = next(m for m in d1.MAGNITUDES_SONDEO if m[0] == "beta_post_rel")
    y = d1.magnitud_regional(va, pref, d1.REGIONES_SONDEO[reg], banda, validos=True)
    d = pd.DataFrame({"a": va.archivo_origen.to_numpy(), "p": va.persona_uid.to_numpy(),
                      "c": va.condicion.to_numpy(), "t": va.t_rec.to_numpy(), "y": y})
    d = d[np.isfinite(d.y)]
    d["tc"] = d.t - d.groupby("a").t.transform("mean")
    d["yc"] = d.y - d.groupby("a").y.transform("mean")
    d["xy"], d["xx"] = d.tc * d.yc, d.tc ** 2
    g = d.groupby(["a", "p", "c"])[["xy", "xx"]].sum().reset_index()
    pp = g.assign(pend=g.xy / g.xx).groupby(["p", "c"]).pend.mean().unstack()
    salida = {}
    for c in FAMILIA:
        dif = (pp[c] - pp["OA"]).dropna()
        t, p = stats.ttest_1samp(dif, 0)
        salida[c] = {"pendiente_tarea": float(pp[c].mean()), "pendiente_oa": float(pp["OA"].mean()),
                     "diferencia": float(dif.mean()), "t": float(t), "p": float(p), "personas": len(dif)}
    return salida


def main():
    with open(CARPETA / "configuracion_elegida.json", encoding="utf-8") as f:
        config = json.load(f)
    feats, mod, esq = config["features"], config["modelo"], config["esquema"]

    d1._titulo("0. CARGA (train + val) Y CONFIGURACIÓN CONGELADA")
    print(f"Configuración: features = {feats}, modelo = {mod}, esquema = {esq}")
    w = comun.cargar({"train", "val"})
    tr = comun.versiones(w[w.split == "train"].reset_index(drop=True))
    va = comun.versiones(w[w.split == "val"].reset_index(drop=True))
    print(f"Personas: train {w[w.split == 'train'].persona_uid.nunique()}, "
          f"val {w[w.split == 'val'].persona_uid.nunique()}")
    assert not set(w[w.split == "train"].persona_uid) & set(w[w.split == "val"].persona_uid)

    rng = np.random.default_rng(d1.SEMILLA)
    resultados = {}
    for v in VERSIONES:
        d1._titulo(f"1. VAL — VERSIÓN {v.upper()}")
        res = con_ic(predecir(tr[v], va[v], feats, mod, esq), rng)
        resultados[v] = res
        for c in ["PASAT", "PVTB", "familia", "PVSAT", "OA", "OC"]:
            x = res[c]
            extra = f"  ({x['grabaciones']} grab., {x['personas']} pers.)" if "grabaciones" in x else ""
            print(f"  {c:8s} {x['pareado']:.3f} [{x['ic95'][0]:.3f}, {x['ic95'][1]:.3f}]{extra}")
        h1 = res["H1"]
        print(f"  H1 (PASAT + PVTB − OA) {h1['diferencia']:+.3f} [{h1['ic95'][0]:+.3f}, {h1['ic95'][1]:+.3f}]")

    d1._titulo("2. H3 EN VAL (beta posterior relativa, igual tiempo, tarea − OA por persona)")
    res_h3 = h3(va["igual_tiempo"])
    for c, x in res_h3.items():
        print(f"  {c}: tarea {x['pendiente_tarea']:+.4f}, OA {x['pendiente_oa']:+.4f}, "
              f"diferencia {x['diferencia']:+.4f} (t = {x['t']:+.2f}, p = {x['p']:.3f}, {x['personas']} pers.)")

    d1._titulo("3. ¿ES BUENO VAL? (definición fijada antes de mirar)")
    r = resultados["igual_tiempo"]
    h1_ok = r["H1"]["diferencia"] > 0
    h2_ok = {c: r[c]["ic95"][0] > 0.5 for c in FAMILIA}
    print(f"  H1 > 0: {r['H1']['diferencia']:+.3f} → {'sí' if h1_ok else 'NO'}")
    for c in FAMILIA:
        print(f"  H2 {c}: límite inferior {r[c]['ic95'][0]:.3f} > 0.5 → {'sí' if h2_ok[c] else 'NO'}")
    bueno = bool(h1_ok and all(h2_ok.values()))
    print(f"  VAL ES {'BUENO' if bueno else 'NO BUENO'}")

    with open(CARPETA / "resultados_val.json", "w", encoding="utf-8") as f:
        json.dump({"fecha": "2026-10-27", "configuracion": config, "val": resultados, "H3": res_h3,
                   "bueno": {"H1": h1_ok, "H2": h2_ok, "val_bueno": bueno},
                   "miradas": {"val": 1, "test": 0}}, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
