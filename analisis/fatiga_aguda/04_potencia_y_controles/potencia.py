"""
Fatiga aguda (tiempo en tarea) — paso 4, parte A: potencia de las hipótesis
en `test` (28 personas).

Simula 2000 veces el `test`: 28 personas de `train` + `val` como pseudo-test y
el resto para entrenar la configuración fijada en el paso 3 (`rel` + logística
C = 0.01, un modelo por condición, medida `rho_ventanas`). Aplica en cada
réplica los contrastes de la sección 8 del `PLAN.md` y estima la potencia.

**Carga `train` y `val`; nunca `test`.** Plan: `INFORME.md`, escrito antes de
ejecutar.

Ejecutar (desde la raíz del repositorio):
    .venv/Scripts/python.exe analisis/fatiga_aguda/04_potencia_y_controles/potencia.py > analisis/fatiga_aguda/04_potencia_y_controles/ejecucion_potencia.log
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent / "03_clasificador"))
import comun  # noqa: E402
import medidas as md  # noqa: E402
from comun import FAMILIA, d1  # noqa: E402

N_REPLICAS = 2000
N_BOOT = 2000
N_TEST = 28
ALFA = 0.05
FACTOR_PRUDENTE = 0.75   # efecto un 25 % menor
CONDICIONES = ["PASAT", "PVTB", "OA"]


def rho_vectorizada(filas, punt):
    """ρ de Spearman por grabación entre la puntuación y el instante (rangos con empates promediados)."""
    d = pd.DataFrame({"a": filas.archivo_origen.to_numpy(), "s": punt, "t": filas.t_rec.to_numpy()})
    g = d.groupby("a")
    d["rs"] = g.s.rank()
    d["rt"] = g.t.rank()
    d["rs"] -= d.groupby("a").rs.transform("mean")
    d["rt"] -= d.groupby("a").rt.transform("mean")
    num = (d.rs * d.rt).groupby(d.a).sum()
    den = np.sqrt((d.rs ** 2).groupby(d.a).sum() * (d.rt ** 2).groupby(d.a).sum())
    return (num / den).dropna()


class Version:
    """Tercios y ventanas de una versión, por condición, para todas las personas."""

    def __init__(self, w):
        self.seg, self.ven = {}, {}
        for c in CONDICIONES:
            S, masc = comun.tercios(w, c)
            self.seg[c] = (S.persona_uid.to_numpy(), comun.matriz(S, masc, "rel"), S.seg.to_numpy())
            filas, X = md.ventanas(w, c)
            persona_de = filas.groupby("archivo_origen").persona_uid.first()
            self.ven[c] = (filas, X, persona_de)

    def rho(self, c, entreno, prueba):
        p, X, y = self.seg[c]
        m = comun.modelo("lr_0.01").fit(X[np.isin(p, entreno)], y[np.isin(p, entreno)])
        filas, Xv, persona_de = self.ven[c]
        sel = filas.persona_uid.isin(prueba).to_numpy()
        r = rho_vectorizada(filas[sel], m.decision_function(Xv[sel]))
        return pd.DataFrame({"persona_uid": persona_de.loc[r.index].to_numpy(), "acierto": r.to_numpy()})


def pendientes_h3(w):
    """Diferencia de pendientes de la beta posterior relativa (tarea − OA) por persona, igual tiempo."""
    nombre, pref, reg, banda = next(m for m in d1.MAGNITUDES_SONDEO if m[0] == "beta_post_rel")
    y = d1.magnitud_regional(w, pref, d1.REGIONES_SONDEO[reg], banda, validos=True)
    d = pd.DataFrame({"a": w.archivo_origen.to_numpy(), "p": w.persona_uid.to_numpy(),
                      "c": w.condicion.to_numpy(), "t": w.t_rec.to_numpy(), "y": y})
    d = d[np.isfinite(d.y) & d.c.isin(CONDICIONES)]
    d["tc"] = d.t - d.groupby("a").t.transform("mean")
    d["yc"] = d.y - d.groupby("a").y.transform("mean")
    d["xy"], d["xx"] = d.tc * d.yc, d.tc ** 2
    g = d.groupby(["a", "p", "c"])[["xy", "xx"]].sum().reset_index()
    pp = g.assign(pend=g.xy / g.xx).groupby(["p", "c"]).pend.mean().unstack()
    return {c: (pp[c] - pp["OA"]).dropna() for c in FAMILIA}


def media_boot(df, personas, rng):
    s = df.groupby("persona_uid").acierto.agg(["sum", "size"]).reindex(personas, fill_value=0)
    idx = rng.integers(len(personas), size=(N_BOOT, len(personas)))
    return s["sum"].to_numpy()[idx].sum(1) / np.maximum(s["size"].to_numpy()[idx].sum(1), 1), idx


def holm_2(p):
    """Rechazos de Holm con dos contrastes."""
    orden = np.argsort(p)
    rech = np.zeros(2, bool)
    if p[orden[0]] <= ALFA / 2:
        rech[orden[0]] = True
        rech[orden[1]] = p[orden[1]] <= ALFA
    return rech


def replica(vers, h3, personas, rng, factor=1.0):
    prueba = rng.choice(personas, N_TEST, replace=False)
    entreno = np.setdiff1d(personas, prueba)
    res = {}
    for v, datos in vers.items():
        r = {c: datos.rho(c, entreno, prueba) for c in CONDICIONES}
        if factor != 1.0:
            # Efecto reducido: se encoge la ρ de las tareas hacia la de OA (la H1 queda × factor).
            base = r["OA"].acierto.mean()
            for c in FAMILIA:
                r[c] = r[c].assign(acierto=base + factor * (r[c].acierto - base))
        fam = pd.concat([r[c] for c in FAMILIA], ignore_index=True)
        pers = np.array(sorted(set(fam.persona_uid) | set(r["OA"].persona_uid)))
        bf, idx = media_boot(fam, pers, rng)
        so = r["OA"].groupby("persona_uid").acierto.agg(["sum", "size"]).reindex(pers, fill_value=0)
        bo = so["sum"].to_numpy()[idx].sum(1) / np.maximum(so["size"].to_numpy()[idx].sum(1), 1)
        h1 = fam.acierto.mean() - r["OA"].acierto.mean()
        res[f"H1_{v}"] = h1
        res[f"H1_{v}_rechaza"] = bool(np.percentile(bf - bo, 5) > 0)
        if v == "igual_tiempo":
            p = []
            for c in FAMILIA:
                b, _ = media_boot(r[c], np.array(sorted(r[c].persona_uid.unique())), rng)
                res[f"rho_{c}"] = r[c].acierto.mean()
                p.append((b <= 0).mean())
            for c, x in zip(FAMILIA, holm_2(np.array(p))):
                res[f"H2_{c}_rechaza"] = bool(x)
    p3 = []
    for c in FAMILIA:
        dif = h3[c][h3[c].index.isin(prueba)]
        p3.append(stats.ttest_1samp(dif, 0, alternative="less").pvalue if len(dif) > 2 else 1.0)
    for c, x in zip(FAMILIA, holm_2(np.array(p3))):
        res[f"H3_{c}_rechaza"] = bool(x)
    return res


def main():
    d1._titulo("0. CARGA (train + val)")
    w = comun.cargar({"train", "val"})
    personas = np.array(sorted(w.persona_uid.unique()))
    print(f"Personas: {len(personas)} (pseudo-test de {N_TEST}, entreno con {len(personas) - N_TEST})")
    vw = comun.versiones(w)
    vers = {v: Version(vw[v]) for v in ["igual_tiempo", "completa"]}
    h3 = pendientes_h3(vw["igual_tiempo"])

    salida = {}
    for etiqueta, factor in (("efecto_estimado", 1.0), ("efecto_25_menor", FACTOR_PRUDENTE)):
        d1._titulo(f"1. SIMULACIÓN — {etiqueta} ({N_REPLICAS} réplicas)")
        rng = np.random.default_rng(d1.SEMILLA)
        t0 = time.time()
        filas = [replica(vers, h3, personas, rng, factor) for _ in range(N_REPLICAS)]
        print(f"  {time.time() - t0:.0f} s")
        t = pd.DataFrame(filas)
        t.to_csv(CARPETA / "tablas" / f"potencia_{etiqueta}.csv", index=False)
        pot = {}
        for col in [c for c in t.columns if c.endswith("_rechaza")]:
            if etiqueta != "efecto_estimado" and col.startswith("H3"):
                continue  # la H3 no depende del factor
            p = t[col].mean()
            pot[col.replace("_rechaza", "")] = {"potencia": p, "error_mc": float(np.sqrt(p * (1 - p) / len(t)))}
            print(f"  {col.replace('_rechaza', ''):18s} potencia = {p:.3f} (± {1.96 * np.sqrt(p * (1 - p) / len(t)):.3f})")
        dist = {c: t[c].describe(percentiles=[.05, .5, .95]).round(4).to_dict()
                for c in t.columns if c.startswith("H1_") and not c.endswith("rechaza")}
        for c, x in dist.items():
            print(f"  {c}: mediana {x['50%']:+.3f}, p5–p95 [{x['5%']:+.3f}, {x['95%']:+.3f}]")
        salida[etiqueta] = {"potencia": pot, "distribucion_H1": dist}

    with open(CARPETA / "resultados_potencia.json", "w", encoding="utf-8") as f:
        json.dump({"fecha": "2026-10-27", "n_replicas": N_REPLICAS, "n_boot": N_BOOT, "n_test": N_TEST,
                   "personas_train_val": len(personas), "resultados": salida,
                   "miradas": {"val": "uso 3 (potencia)", "test": 0}}, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    (CARPETA / "tablas").mkdir(exist_ok=True)
    main()
