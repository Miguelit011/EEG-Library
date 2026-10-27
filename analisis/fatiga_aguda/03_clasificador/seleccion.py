"""
Fatiga aguda (tiempo en tarea) — paso 3, parte 1: selección del clasificador
en `train` con CV repetida.

Evalúa las 32 configuraciones candidatas (4 familias de features × 4 modelos ×
2 esquemas de entrenamiento) con 20 particiones por persona, elige la de mayor
H1 = PASAT + PVTB − OA a igual tiempo (desempate por simplicidad) y la escribe
en `configuracion_elegida.json`. Con la elegida, añade PVSAT y OC,
transferencia entre tareas, regresión del minuto e importancias.

**Solo `train`.** `val` lo carga únicamente `comprobacion_val.py`, después de
congelar la configuración. Plan, regla de selección y detalles: `INFORME.md`,
escritos antes de ejecutar.

Ejecutar (desde la raíz del repositorio):
    .venv/Scripts/python.exe analisis/fatiga_aguda/03_clasificador/seleccion.py > analisis/fatiga_aguda/03_clasificador/ejecucion_seleccion.log
"""

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold, cross_val_predict

import comun
from comun import FAMILIA, FEATURES, MODELOS, ESQUEMAS, SEMILLAS, anexo, d1

CARPETA = comun.CARPETA
CARPETA_TABLAS = CARPETA / "tablas"
CARPETA_FIGURAS = CARPETA / "figuras"
VERSIONES = ["igual_tiempo", "completa"]
SEMILLAS_REGRESION = SEMILLAS[:5]


class Datos:
    """Tercios por versión y condición, con las matrices de features en caché."""

    def __init__(self, vers, condiciones):
        self.S = {(v, c): comun.tercios(w, c) for v, w in vers.items() for c in condiciones}
        self._X = {}

    def X(self, v, c, feats):
        clave = (v, c, feats)
        if clave not in self._X:
            self._X[clave] = comun.matriz(*self.S[(v, c)], feats)
        return self._X[clave]

    def filas(self, v, c):
        return self.S[(v, c)][0]


def cv(datos, v, entreno, evaluar, feats, mod, pliegue_de):
    """Probabilidades fuera de pliegue: se entrena con `entreno` y se predice `evaluar`."""
    fold = {c: datos.filas(v, c).persona_uid.map(pliegue_de).to_numpy() for c in set(entreno) | set(evaluar)}
    out = {c: np.full(len(fold[c]), np.nan) for c in evaluar}
    for k in range(comun.N_PLIEGUES):
        Xtr = np.vstack([datos.X(v, c, feats)[fold[c] != k] for c in entreno])
        ytr = np.concatenate([datos.filas(v, c).seg.to_numpy()[fold[c] != k] for c in entreno])
        m = comun.modelo(mod).fit(Xtr, ytr)
        for c in evaluar:
            idx = fold[c] == k
            if idx.any():
                out[c][idx] = m.predict_proba(datos.X(v, c, feats)[idx])[:, 1]
    return out


def pareados(datos, v, probs):
    """Acierto pareado por condición, PASAT + PVTB juntas y H1, a partir de probabilidades."""
    a = {c: comun.aciertos_por_grabacion(datos.filas(v, c), p) for c, p in probs.items()}
    res = {c: x.acierto.mean() for c, x in a.items()}
    if all(c in a for c in FAMILIA):
        res["familia"] = pd.concat([a[c] for c in FAMILIA]).acierto.mean()
        if "OA" in a:
            res["H1"] = res["familia"] - res["OA"]
    return res, a


def comprobar_paso_1(datos):
    d1._titulo("0b. COMPROBACIÓN: rel + logística C=0.1 con la partición del paso 1")
    for c, esperado in (("PASAT", 0.773), ("OA", 0.647)):
        S, masc = datos.S[("completa", c)]
        p = cross_val_predict(comun.modelo("lr_0.1"), comun.matriz(S, masc, "rel"), S.seg, groups=S.persona_uid,
                              cv=GroupKFold(5), method="predict_proba")[:, 1]
        obtenido = comun.aciertos_por_grabacion(S, p).acierto.mean()
        print(f"  {c}: {obtenido:.4f} (paso 1: {esperado})")
        assert abs(obtenido - esperado) < 0.001, "No se reproduce el paso 1"


def evaluar_candidatas(datos, pliegues):
    d1._titulo("1. 32 CONFIGURACIONES × 20 PARTICIONES × 2 VERSIONES")
    filas, probs = [], {}
    t0 = time.time()
    for v in VERSIONES:
        for feats in FEATURES:
            for mod in MODELOS:
                for s in SEMILLAS:
                    oa = cv(datos, v, ["OA"], ["OA"], feats, mod, pliegues[s])
                    sep = {c: cv(datos, v, [c], [c], feats, mod, pliegues[s])[c] for c in FAMILIA}
                    conj = cv(datos, v, FAMILIA, FAMILIA, feats, mod, pliegues[s])
                    for esq, p in (("separado", {**sep, **oa}), ("conjunto", {**conj, **oa})):
                        res, _ = pareados(datos, v, p)
                        filas.append({"version": v, "features": feats, "modelo": mod, "esquema": esq,
                                      "semilla": s, **res})
                        probs.setdefault((v, feats, mod, esq), []).append(p)
                print(f"  {v:12s} {feats:8s} {mod:7s} hecho ({time.time() - t0:.0f} s)", flush=True)
    return pd.DataFrame(filas), probs


def resumir(tabla):
    claves = ["version", "features", "modelo", "esquema"]
    g = tabla.groupby(claves)
    res = g[["H1", "familia", "PASAT", "PVTB", "OA"]].mean().add_suffix("_media")
    res["H1_min"], res["H1_max"] = g.H1.min(), g.H1.max()
    res = res.reset_index()
    res["simplicidad"] = list(zip(res.features.map(comun.SIMPLICIDAD_FEATURES),
                                  res.modelo.map(comun.SIMPLICIDAD_MODELO),
                                  res.esquema.map(comun.SIMPLICIDAD_ESQUEMA)))
    return res


def elegir(resumen):
    d1._titulo("2. SELECCIÓN (mayor H1 a igual tiempo; desempate por simplicidad)")
    r = resumen[resumen.version == "igual_tiempo"].sort_values("H1_media", ascending=False)
    mejor = r.H1_media.iloc[0]
    empatadas = r[r.H1_media >= mejor - comun.TOLERANCIA_EMPATE].sort_values("simplicidad")
    cols = ["features", "modelo", "esquema", "H1_media", "H1_min", "H1_max", "familia_media", "PASAT_media",
            "PVTB_media", "OA_media"]
    print("Las 32 configuraciones a igual tiempo, de mayor a menor H1:")
    print(r[cols].round(3).to_string(index=False))
    print(f"\nMejor H1 = {mejor:.4f}. Configuraciones a menos de {comun.TOLERANCIA_EMPATE}:")
    print(empatadas[cols].round(3).to_string(index=False))
    elegida = empatadas.iloc[0]
    print(f"\nELEGIDA: features = {elegida.features}, modelo = {elegida.modelo}, esquema = {elegida.esquema} "
          f"(H1 = {elegida.H1_media:.4f})")
    return elegida


def con_ic(datos, v, probs_semillas, rng):
    """Acierto pareado e IC95 (bootstrap de personas) con la probabilidad media de las repeticiones."""
    media = {c: np.mean([p[c] for p in probs_semillas], axis=0) for c in probs_semillas[0]}
    res, a = pareados(datos, v, media)
    salida = {}
    for c, x in a.items():
        porp = x.groupby("persona_uid").acierto.agg(["sum", "size"])
        ic = d1.bootstrap_personas(porp["sum"].to_numpy(), porp["size"].to_numpy(), rng)
        salida[c] = {"pareado": res[c], "ic95": [float(i) for i in ic], "grabaciones": len(x),
                     "personas": int(x.persona_uid.nunique())}
    if all(c in a for c in FAMILIA):
        fam = pd.concat([a[c] for c in FAMILIA], ignore_index=True)
        porp = fam.groupby("persona_uid").acierto.agg(["sum", "size"])
        ic = d1.bootstrap_personas(porp["sum"].to_numpy(), porp["size"].to_numpy(), rng)
        salida["familia"] = {"pareado": res["familia"], "ic95": [float(i) for i in ic]}
        if "OA" in a:
            dif, ic = anexo.diferencia_bootstrap(fam, a["OA"], rng)
            salida["H1"] = {"diferencia": dif, "ic95": [float(i) for i in ic]}
    return salida


def regresion_minuto(w, feats, mod, pliegues):
    d1._titulo("5. REGRESIÓN DEL MINUTO (grabación completa, bloques de 30 s, semillas 0–4)")
    filas = []
    for c in ["PASAT", "PVTB", "OA"]:
        S, masc = comun.bloques(w, c)
        X = comun.matriz(S, masc, feats)
        y = S.t_bloque.to_numpy()
        pred = np.zeros((len(SEMILLAS_REGRESION), len(S)))
        for i, s in enumerate(SEMILLAS_REGRESION):
            fold = S.persona_uid.map(pliegues[s]).to_numpy()
            for k in range(comun.N_PLIEGUES):
                m = comun.modelo_regresion(mod).fit(X[fold != k], y[fold != k])
                pred[i, fold == k] = m.predict(X[fold == k])
        S = S.assign(pred=pred.mean(axis=0))
        rho = S.groupby(["archivo_origen", "persona_uid"]).apply(
            lambda g: stats.spearmanr(g.pred, g.t_bloque)[0] if len(g) >= 3 else np.nan).dropna()
        pp = rho.groupby("persona_uid").mean()
        t, p = stats.ttest_1samp(pp, 0)
        filas.append({"condicion": c, "rho_medio": pp.mean(), "t": t, "p": p, "personas": len(pp),
                      "grabaciones": len(rho), "bloques_por_grabacion": float(S.groupby("archivo_origen").size().median())})
    tabla = pd.DataFrame(filas)
    print(tabla.round(4).to_string(index=False))
    tabla.to_csv(CARPETA_TABLAS / "regresion_minuto.csv", index=False)
    return tabla


def importancias(datos, feats, mod, esq):
    d1._titulo("6. IMPORTANCIAS (modelo ajustado con todo train, igual tiempo)")
    grupos = [FAMILIA] if esq == "conjunto" else [[c] for c in FAMILIA]
    cols = FEATURES[feats]
    filas = {}
    for g in grupos:
        X = np.vstack([datos.X("igual_tiempo", c, feats) for c in g])
        y = np.concatenate([datos.filas("igual_tiempo", c).seg.to_numpy() for c in g])
        m = comun.modelo(mod).fit(X, y)
        final = m[-1]
        filas["+".join(g)] = final.coef_[0] if mod != "rf" else final.feature_importances_
    tabla = pd.DataFrame(filas, index=cols)
    tabla.to_csv(CARPETA_TABLAS / "importancias.csv")
    orden = tabla.abs().mean(axis=1).sort_values(ascending=False).index[:15]
    print(("Coeficientes tipificados" if mod != "rf" else "Importancias por impureza") + " (15 mayores):")
    print(tabla.loc[orden].round(3).to_string())

    # Mapa: filas = canal o región, columnas = banda (y tipo).
    serie = tabla.mean(axis=1)
    if feats == "regiones":
        mapa = pd.DataFrame({f"{b}_{t}": [serie[f"{r}_{b}_{t}"] for r in comun.REGIONES]
                             for t in ("rel", "abs") for b in d1.BANDAS}, index=list(comun.REGIONES))
    else:
        cols_mapa = {}
        for pref, etiqueta in (("pot_rel", "rel"), ("pot_abs", "abs")):
            if any(c.startswith(pref) for c in cols):
                for b in d1.BANDAS:
                    cols_mapa[f"{b}_{etiqueta}"] = [serie[f"{pref}_{ch}_{b}"] for ch in d1.CANALES_EPOC]
        mapa = pd.DataFrame(cols_mapa, index=d1.CANALES_EPOC)
    fig, ax = plt.subplots(figsize=(1.1 * mapa.shape[1] + 2, 0.4 * mapa.shape[0] + 1.5))
    lim = np.abs(mapa.to_numpy()).max()
    im = ax.imshow(mapa.to_numpy(), cmap="RdBu_r" if mod != "rf" else "Blues",
                   vmin=-lim if mod != "rf" else 0, vmax=lim, aspect="auto")
    ax.set_xticks(range(mapa.shape[1]), mapa.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(mapa.shape[0]), mapa.index, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=.8,
                 label="coeficiente tipificado (+ = más en el último tercio)" if mod != "rf" else "importancia")
    ax.set_title(f"Importancias — {feats}, {mod}, {esq} (train, igual tiempo)", fontsize=9)
    fig.tight_layout()
    fig.savefig(CARPETA_FIGURAS / "importancias.png", dpi=150)
    plt.close(fig)
    return tabla


def figura_configuraciones(resumen, elegida):
    r = resumen[resumen.version == "igual_tiempo"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    marcas = {"regiones": "s", "rel": "o", "abs": "^", "rel+abs": "D"}
    tonos = {"lr_0.01": "#7aa9e3", "lr_0.1": "#2a78d6", "lr_1": "#0d3f80", "rf": "#eb6834"}
    for _, x in r.iterrows():
        ax.scatter(x.familia_media, x.H1_media, marker=marcas[x.features], color=tonos[x.modelo],
                   s=45 if x.esquema == "separado" else 25, alpha=.9,
                   edgecolor="none" if x.esquema == "separado" else tonos[x.modelo],
                   facecolor=tonos[x.modelo] if x.esquema == "separado" else "none")
    ax.scatter(elegida.familia_media, elegida.H1_media, s=180, facecolor="none", edgecolor=d1.TINTA, lw=1.3)
    for f, m in marcas.items():
        ax.scatter([], [], marker=m, color=d1.TINTA_2, label=f)
    for mo, t in tonos.items():
        ax.scatter([], [], marker="o", color=t, label=mo)
    ax.legend(frameon=False, fontsize=7, ncol=2, title="relleno = separado; hueco = conjunto", title_fontsize=7)
    ax.set_xlabel("acierto pareado PASAT + PVTB (media de 20 particiones)", color=d1.TINTA_2)
    ax.set_ylabel("H1 = PASAT + PVTB − OA", color=d1.TINTA_2)
    ax.grid(color=d1.REJILLA, lw=.6)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.set_title("32 configuraciones a igual tiempo (train); círculo = elegida", fontsize=10, color=d1.TINTA)
    fig.tight_layout()
    fig.savefig(CARPETA_FIGURAS / "configuraciones.png", dpi=150)
    plt.close(fig)


def main():
    CARPETA_TABLAS.mkdir(exist_ok=True)
    CARPETA_FIGURAS.mkdir(exist_ok=True)

    d1._titulo("0. CARGA (solo train)")
    w = comun.cargar({"train"})
    vers = comun.versiones(w)
    datos = Datos(vers, comun.CONDICIONES)
    for v in VERSIONES:
        print(f"{v}: grabaciones por condición",
              {c: int(datos.filas(v, c).archivo_origen.nunique()) for c in comun.CONDICIONES})
    personas = sorted(w.persona_uid.unique())
    pliegues = {s: comun.pliegues_por_persona(personas, s) for s in SEMILLAS}
    print(f"Personas de train: {len(personas)}")
    comprobar_paso_1(datos)

    tabla, probs = evaluar_candidatas(datos, pliegues)
    tabla.to_csv(CARPETA_TABLAS / "candidatas_por_semilla.csv", index=False)
    resumen = resumir(tabla)
    resumen.drop(columns="simplicidad").to_csv(CARPETA_TABLAS / "candidatas_resumen.csv", index=False)
    elegida = elegir(resumen)
    figura_configuraciones(resumen, elegida)
    feats, mod, esq = elegida.features, elegida.modelo, elegida.esquema

    d1._titulo("3. LA ELEGIDA EN TRAIN, CON IC (probabilidad media de 20 particiones)")
    rng = np.random.default_rng(d1.SEMILLA)
    elegida_train = {}
    for v in VERSIONES:
        p_sem = probs[(v, feats, mod, esq)]
        extra = [{c: cv(datos, v, [c], [c], feats, mod, pliegues[s])[c] for c in ["PVSAT", "OC"]}
                 for s in SEMILLAS]
        p_sem = [{**a, **b} for a, b in zip(p_sem, extra)]
        elegida_train[v] = con_ic(datos, v, p_sem, rng)
        r = resumen.set_index(["version", "features", "modelo", "esquema"]).loc[(v, feats, mod, esq)]
        elegida_train[v]["H1_rango_20"] = [float(r.H1_min), float(r.H1_max)]
        print(f"\nVersión {v}:")
        for c, x in elegida_train[v].items():
            if isinstance(x, dict) and "pareado" in x:
                print(f"  {c:8s} {x['pareado']:.3f} [{x['ic95'][0]:.3f}, {x['ic95'][1]:.3f}]")
        h1 = elegida_train[v]["H1"]
        print(f"  H1       {h1['diferencia']:+.3f} [{h1['ic95'][0]:+.3f}, {h1['ic95'][1]:+.3f}]; "
              f"rango en 20 particiones [{r.H1_min:+.3f}, {r.H1_max:+.3f}]")

    d1._titulo("4. TRANSFERENCIA ENTRE TAREAS (igual tiempo y completa)")
    transferencia = {}
    for v in VERSIONES:
        for origen, destino in (("PASAT", "PVTB"), ("PVTB", "PASAT")):
            p_sem = [cv(datos, v, [origen], [destino], feats, mod, pliegues[s]) for s in SEMILLAS]
            x = con_ic(datos, v, p_sem, rng)[destino]
            transferencia[f"{v}: {origen} → {destino}"] = x
            print(f"  {v:12s} {origen} → {destino}: {x['pareado']:.3f} [{x['ic95'][0]:.3f}, {x['ic95'][1]:.3f}]")

    tabla_reg = regresion_minuto(vers["completa"], feats, mod, pliegues)
    importancias(datos, feats, mod, esq)

    configuracion = {
        "fecha": "2026-10-27",
        "features": feats, "modelo": mod, "esquema": esq,
        "regla": "mayor H1 = PASAT + PVTB − OA a igual tiempo, media de 20 particiones; desempate por simplicidad "
                 f"a menos de {comun.TOLERANCIA_EMPATE}",
        "H1_media_train_igual_tiempo": float(elegida.H1_media),
        "sha256": {p.name: d1.sha256(p) for p in (d1.RUTA_FEATURES, d1.RUTA_MATRIZ, d1.RUTA_LOG)},
    }
    with open(CARPETA / "configuracion_elegida.json", "w", encoding="utf-8") as f:
        json.dump(configuracion, f, ensure_ascii=False, indent=1)
    resultados = {
        **configuracion,
        "elegida_train": elegida_train,
        "transferencia": transferencia,
        "regresion_minuto": tabla_reg.round(5).to_dict(orient="records"),
        "miradas": {"val": 0, "test": 0},
    }
    with open(CARPETA / "resultados_seleccion.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
