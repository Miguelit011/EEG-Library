"""
Fatiga aguda (tiempo en tarea) — paso 4, parte B: tiempo en tarea frente a
tiempo desde la colocación del casco.

El orden del protocolo es fijo (OC → OA → STROOP → PASAT → PVTB → MSIT →
PVSAT), así que el tiempo en la tarea y el tiempo desde la colocación no se
pueden separar del todo. Cuatro comprobaciones parciales:

- B1. Especificidad: matriz de transferencia 7 × 7 con `rho_ventanas`.
- B2. Dientes de sierra: cambio dentro de la tarea frente al salto a la
  siguiente.
- B3. Desaceleración: pendiente por minuto frente al momento de la sesión.
- B4. Modelo de ventanas con efectos fijos de sesión y condición, pendientes
  de `t_rec` por condición y `t_ses` común.

**Solo `train`.** Plan y lectura: `INFORME.md`, escritos antes de ejecutar.

Ejecutar (desde la raíz del repositorio):
    .venv/Scripts/python.exe analisis/fatiga_aguda/04_potencia_y_controles/controles.py > analisis/fatiga_aguda/04_potencia_y_controles/ejecucion_controles.log
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent / "03_clasificador"))
import comun  # noqa: E402
import medidas as md  # noqa: E402
from comun import SEMILLAS, d1  # noqa: E402

PROTOCOLO = ["OC", "OA", "STROOP", "PASAT", "PVTB", "MSIT", "PVSAT"]
MAGNITUDES = ["alfa_post_abs", "theta_front_abs", "beta_post_rel"]
RUTA_MANIFEST = d1.EEG_DATA / "metadata" / "manifest_canonical.csv"
RECORTE_S = 5.0
CARPETA_TABLAS = CARPETA / "tablas"
CARPETA_FIGURAS = CARPETA / "figuras"


def cargar():
    w, _, _ = d1.cargar()
    w = w[w.n_vent >= d1.MIN_VENTANAS].reset_index(drop=True)
    man = pd.read_csv(RUTA_MANIFEST, usecols=["nombre_nuevo", "timestamp_iso", "duracion_sec"])
    man["t0"] = pd.to_datetime(man.timestamp_iso, utc=True, errors="coerce")
    w = w.merge(man[["nombre_nuevo", "t0", "duracion_sec"]], left_on="archivo_origen", right_on="nombre_nuevo",
                how="left").drop(columns="nombre_nuevo")
    w["t0_s"] = (w.t0 - pd.Timestamp("2020-01-01", tz="UTC")).dt.total_seconds()
    w["t_ses"] = (w.t0_s - w.groupby("sesion_uid").t0_s.transform("min") + RECORTE_S + w.tmin_s) / 60
    for nombre, pref, reg, banda in d1.MAGNITUDES_SONDEO:
        if nombre in MAGNITUDES:
            w[nombre] = d1.magnitud_regional(w, pref, d1.REGIONES_SONDEO[reg], banda, validos=True)
    return w


# --- B1 ------------------------------------------------------------------------


def transferencia(w):
    d1._titulo("B1. ESPECIFICIDAD: MATRIZ DE TRANSFERENCIA (rho_ventanas, CV repetida de 20 particiones)")
    personas = sorted(w.persona_uid.unique())
    pliegues = {s: comun.pliegues_por_persona(personas, s) for s in SEMILLAS}
    seg = {c: comun.tercios(w, c) for c in PROTOCOLO}
    Xs = {c: (comun.matriz(*seg[c], "rel"), seg[c][0].seg.to_numpy(), seg[c][0].persona_uid) for c in PROTOCOLO}
    ven = {c: md.ventanas(w, c) for c in PROTOCOLO}
    aciertos = {}
    for a in PROTOCOLO:
        X, y, pers = Xs[a]
        punt = {b: np.zeros(len(ven[b][0])) for b in PROTOCOLO}
        for s in SEMILLAS:
            fold = pers.map(pliegues[s]).to_numpy()
            for k in range(comun.N_PLIEGUES):
                m = comun.modelo("lr_0.01").fit(X[fold != k], y[fold != k])
                for b in PROTOCOLO:
                    fb = ven[b][0].persona_uid.map(pliegues[s]).to_numpy() == k
                    punt[b][fb] += m.decision_function(ven[b][1][fb]) / len(SEMILLAS)
        for b in PROTOCOLO:
            aciertos[(a, b)] = md.rho_por_grabacion(ven[b][0], punt[b])
    matriz = pd.DataFrame([[aciertos[(a, b)].acierto.mean() for b in PROTOCOLO] for a in PROTOCOLO],
                          index=[f"modelo {a}" for a in PROTOCOLO], columns=PROTOCOLO)
    print("ρ media por grabación (filas: condición con la que se entrena; columnas: condición puntuada):")
    print(matriz.round(3).to_string())
    matriz.to_csv(CARPETA_TABLAS / "transferencia.csv")

    rng = np.random.default_rng(d1.SEMILLA)
    filas = []
    print("\nPropia − ajena (pareado por persona), IC95:")
    for a in ["PASAT", "PVTB", "OA", "OC"]:
        for b in PROTOCOLO:
            if b == a:
                continue
            reps = md.bootstrap(aciertos[(a, a)], aciertos[(a, b)], rng=rng)
            dif = aciertos[(a, a)].acierto.mean() - aciertos[(a, b)].acierto.mean()
            ic = np.percentile(reps, [2.5, 97.5])
            filas.append({"modelo": a, "puntuada": b, "propia_menos_ajena": dif, "ic95": ic.tolist()})
        print(f"  modelo {a:6s}: " + "  ".join(f"{f['puntuada']} {f['propia_menos_ajena']:+.3f}"
                                              f"[{f['ic95'][0]:+.3f},{f['ic95'][1]:+.3f}]"
                                              for f in filas if f["modelo"] == a))
    pd.DataFrame(filas).to_csv(CARPETA_TABLAS / "transferencia_propia_menos_ajena.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    lim = np.abs(matriz.to_numpy()).max()
    im = ax.imshow(matriz.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim)
    for i in range(len(PROTOCOLO)):
        for j in range(len(PROTOCOLO)):
            ax.text(j, i, f"{matriz.iat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(matriz.iat[i, j]) > lim * .6 else d1.TINTA)
    ax.set_xticks(range(len(PROTOCOLO)), PROTOCOLO, fontsize=8)
    ax.set_yticks(range(len(PROTOCOLO)), PROTOCOLO, fontsize=8)
    ax.set_xlabel("condición puntuada", color=d1.TINTA_2)
    ax.set_ylabel("condición con la que se entrena", color=d1.TINTA_2)
    fig.colorbar(im, ax=ax, shrink=.8, label="ρ media (ventana ~ instante)")
    ax.set_title("Transferencia entre condiciones (train, CV repetida)", fontsize=10, color=d1.TINTA)
    fig.tight_layout()
    fig.savefig(CARPETA_FIGURAS / "transferencia.png", dpi=150)
    plt.close(fig)
    return matriz, filas


# --- B2 ------------------------------------------------------------------------


def dientes_de_sierra(w):
    d1._titulo("B2. DIENTES DE SIERRA (últimos 30 s de k − primeros 30 s de k; primeros 30 s de k+1 − últimos 30 s de k)")
    g = w[w.t0.notna()]
    tmax = g.groupby("archivo_origen").tmin_s.transform("max")
    ini = g[g.tmin_s <= 28].groupby(["sesion_uid", "persona_uid", "condicion"])[MAGNITUDES].mean()
    fin = g[g.tmin_s >= tmax - 28].groupby(["sesion_uid", "persona_uid", "condicion"])[MAGNITUDES].mean()
    t0 = g.groupby(["sesion_uid", "condicion"]).agg(t0=("t0_s", "first"), dur=("duracion_sec", "first"))
    filas = []
    for k, k1 in zip(PROTOCOLO[:-1], PROTOCOLO[1:]):
        a_ini, a_fin = ini.xs(k, level="condicion"), fin.xs(k, level="condicion")
        b_ini = ini.xs(k1, level="condicion")
        comunes = a_ini.index.intersection(b_ini.index)
        dentro = (a_fin.loc[comunes] - a_ini.loc[comunes]).groupby(level="persona_uid").mean()
        salto = (b_ini.loc[comunes] - a_fin.loc[comunes]).groupby(level="persona_uid").mean()
        ses = comunes.get_level_values("sesion_uid")
        hueco = (t0.xs(k1, level="condicion").loc[ses, "t0"].to_numpy()
                 - t0.xs(k, level="condicion").loc[ses, "t0"].to_numpy()
                 - t0.xs(k, level="condicion").loc[ses, "dur"].to_numpy())
        for m in MAGNITUDES:
            td = stats.ttest_1samp(dentro[m].dropna(), 0)
            ts = stats.ttest_1samp(salto[m].dropna(), 0)
            filas.append({"transicion": f"{k}→{k1}", "magnitud": m, "dentro_k": dentro[m].mean(), "t_dentro": td[0],
                          "salto": salto[m].mean(), "t_salto": ts[0], "personas": int(salto[m].notna().sum()),
                          "hueco_mediano_s": float(np.median(hueco))})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(CARPETA_TABLAS / "dientes_de_sierra.csv", index=False)
    celda = tabla.apply(lambda r: f"{r.dentro_k:+.3f} ({r.t_dentro:+.1f}) | {r.salto:+.3f} ({r.t_salto:+.1f})", axis=1)
    print("Dentro de k (t) | salto a k+1 (t):")
    print(tabla.assign(c=celda).pivot(index="transicion", columns="magnitud", values="c")
          .loc[[f"{k}→{k1}" for k, k1 in zip(PROTOCOLO[:-1], PROTOCOLO[1:])], MAGNITUDES].to_string())
    print("\nHueco mediano entre archivos (s):",
          tabla.drop_duplicates("transicion").set_index("transicion").hueco_mediano_s.round(0).to_dict())
    return tabla


# --- B3 ------------------------------------------------------------------------


def desaceleracion(w):
    d1._titulo("B3. DESACELERACIÓN: PENDIENTE POR MINUTO FRENTE AL MOMENTO DE LA SESIÓN")
    inicio = (w[w.t0.notna()].groupby("archivo_origen").agg(c=("condicion", "first"), t=("t_ses", "min"))
              .groupby("c").t.median())
    filas = []
    for m in MAGNITUDES:
        d = w[["archivo_origen", "persona_uid", "condicion", "t_rec", m]].dropna()
        tc = d.t_rec - d.groupby("archivo_origen").t_rec.transform("mean")
        yc = d[m] - d.groupby("archivo_origen")[m].transform("mean")
        g = (pd.DataFrame({"a": d.archivo_origen, "p": d.persona_uid, "c": d.condicion, "xy": tc * yc,
                           "xx": tc ** 2}).groupby(["a", "p", "c"])[["xy", "xx"]].sum().reset_index())
        pp = g.assign(pend=g.xy / g.xx).groupby(["p", "c"]).pend.mean().unstack()
        for c in PROTOCOLO:
            v = pp[c].dropna()
            filas.append({"magnitud": m, "condicion": c, "inicio_mediano_min": float(inicio[c]),
                          "pendiente": v.mean(), "ic_inf": v.mean() - 1.96 * v.sem(),
                          "ic_sup": v.mean() + 1.96 * v.sem(), "personas": len(v)})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(CARPETA_TABLAS / "desaceleracion.csv", index=False)
    print(tabla.assign(c=tabla.apply(lambda r: f"{r.pendiente:+.4f} [{r.ic_inf:+.4f}, {r.ic_sup:+.4f}]", axis=1))
          .pivot(index="condicion", columns="magnitud", values="c").loc[PROTOCOLO].to_string())
    print("Inicio mediano en la sesión (min):", inicio.loc[PROTOCOLO].round(1).to_dict())

    fig, ejes = plt.subplots(1, len(MAGNITUDES), figsize=(13, 3.8))
    for ax, m in zip(ejes, MAGNITUDES):
        s = tabla[tabla.magnitud == m]
        ax.errorbar(s.inicio_mediano_min, s.pendiente, yerr=[s.pendiente - s.ic_inf, s.ic_sup - s.pendiente],
                    fmt="o", color="#2a78d6", capsize=3)
        for _, r in s.iterrows():
            ax.annotate(r.condicion, (r.inicio_mediano_min, r.pendiente), fontsize=7, xytext=(4, 4),
                        textcoords="offset points", color=d1.TINTA_2)
        ax.axhline(0, color=d1.REJILLA, lw=1)
        ax.set_title(m, fontsize=9, color=d1.TINTA)
        ax.set_xlabel("inicio mediano en la sesión (min)", fontsize=8, color=d1.TINTA_2)
        ax.set_ylabel("pendiente por minuto (log10)", fontsize=8, color=d1.TINTA_2)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    fig.suptitle("Pendiente dentro de la grabación frente al momento de la sesión (train, IC95)", fontsize=10)
    fig.tight_layout()
    fig.savefig(CARPETA_FIGURAS / "desaceleracion.png", dpi=150)
    plt.close(fig)
    return tabla


# --- B4 ------------------------------------------------------------------------


def mco_agrupado(X, y, grupos):
    """MCO con errores típicos robustos agrupados (corrección G/(G−1))."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    xtx_inv = np.linalg.pinv(X.T @ X)
    codigos, idx = np.unique(grupos, return_inverse=True)
    S = np.zeros((X.shape[1], X.shape[1]))
    suma = np.zeros((len(codigos), X.shape[1]))
    np.add.at(suma, idx, X * u[:, None])
    S = suma.T @ suma
    G = len(codigos)
    V = xtx_inv @ S @ xtx_inv * G / (G - 1)
    return beta, np.sqrt(np.diag(V))


def efectos_fijos(w):
    d1._titulo("B4. MODELO DE VENTANAS: EF DE SESIÓN Y CONDICIÓN, t_rec POR CONDICIÓN Y t_ses COMÚN")
    g = w[w.t0.notna()].copy()
    variacion = (g.groupby("archivo_origen").agg(c=("condicion", "first"), t=("t_ses", "min"))
                 .groupby("c").t.std())
    print("Desviación entre sesiones del momento de inicio de cada condición (min):",
          variacion.loc[PROTOCOLO].round(2).to_dict())
    filas = []
    for m in MAGNITUDES:
        d = g[["sesion_uid", "persona_uid", "condicion", "t_rec", "t_ses", m]].dropna().reset_index(drop=True)
        dummies = pd.get_dummies(d.condicion)[PROTOCOLO[1:]].astype(float)  # OC de referencia
        pend = pd.DataFrame({f"t_rec_{c}": d.t_rec * (d.condicion == c) for c in PROTOCOLO})
        for variante, regs in (("pendiente_por_condicion", pd.concat([dummies, pend], axis=1)),
                               ("pendiente_comun", pd.concat([dummies, d[["t_rec"]]], axis=1))):
            R = pd.concat([regs, d[["t_ses"]]], axis=1)
            Rw = R - R.groupby(d.sesion_uid).transform("mean")
            yw = d[m] - d.groupby("sesion_uid")[m].transform("mean")
            beta, se = mco_agrupado(Rw.to_numpy(float), yw.to_numpy(float), d.persona_uid.to_numpy())
            res = pd.DataFrame({"coef": beta, "se": se}, index=R.columns)
            res["t"] = res.coef / res.se
            if variante == "pendiente_comun":
                r = np.corrcoef(Rw.t_ses, Rw.t_rec)[0, 1]
                print(f"\n{m} — correlación t_ses ~ t_rec tras la transformación within: {r:.3f}")
            sel = [c for c in res.index if c.startswith("t_")]
            print(f"{m} — {variante}:")
            print(res.loc[sel].round(4).to_string())
            for c in sel:
                filas.append({"magnitud": m, "variante": variante, "termino": c, **res.loc[c].to_dict()})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(CARPETA_TABLAS / "efectos_fijos.csv", index=False)
    return tabla, variacion


def main():
    CARPETA_TABLAS.mkdir(exist_ok=True)
    CARPETA_FIGURAS.mkdir(exist_ok=True)
    d1._titulo("0. CARGA (solo train, siete condiciones)")
    w = cargar()
    sin_t = w[w.t0.isna()].archivo_origen.nunique()
    print(f"Grabaciones: {w.archivo_origen.nunique()}; personas: {w.persona_uid.nunique()}; "
          f"sin marca de tiempo válida: {sin_t}")

    matriz, propia = transferencia(w)
    sierra = dientes_de_sierra(w)
    desac = desaceleracion(w)
    ef, variacion = efectos_fijos(w)

    with open(CARPETA / "resultados_controles.json", "w", encoding="utf-8") as f:
        json.dump({"fecha": "2026-10-27", "particion": "solo train",
                   "sha256": {p.name: d1.sha256(p) for p in (d1.RUTA_FEATURES, d1.RUTA_MATRIZ, d1.RUTA_LOG,
                                                              RUTA_MANIFEST)},
                   "sin_marca_de_tiempo": int(sin_t),
                   "transferencia": matriz.round(5).to_dict(), "propia_menos_ajena": propia,
                   "dientes_de_sierra": sierra.round(5).to_dict(orient="records"),
                   "desaceleracion": desac.round(5).to_dict(orient="records"),
                   "efectos_fijos": ef.round(5).to_dict(orient="records"),
                   "variacion_inicio_min": variacion.round(3).to_dict(),
                   "miradas": {"val": 0, "test": 0}}, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
