"""
Fatiga aguda (tiempo en tarea) — paso 6: evaluación confirmatoria única en
`test`, según `PREREGISTRO.md`.

Dos modos:

- `--ensayo`: ensayo general. Entrena con `train` y evalúa en `val`. Solo
  comprueba que el guion corre de principio a fin y que reproduce las cifras
  ya vistas en la mirada 2 a `val` (paso 3). No es un resultado ni una mirada
  nueva. Los artefactos van a una carpeta temporal.
- `--test --permitir-test`: la ejecución confirmatoria. Entrena con `train` +
  `val` y evalúa en `test` **una sola vez**. Se niega a correr si el árbol de
  git no está limpio o si la carpeta de salida ya existe.

Todo lo que se calcula está fijado en `PREREGISTRO.md` (configuración del
paso 3, medida `rho_ventanas`, secuencia H1b → H1, secundarias H2–H4 y
sensibilidades). El guion no tiene parámetros de análisis por línea de órdenes.

Ejecutar (desde la raíz del repositorio):
    .venv/Scripts/python.exe analisis/fatiga_aguda/06_test/evaluacion_test.py --ensayo
    .venv/Scripts/python.exe analisis/fatiga_aguda/06_test/evaluacion_test.py --test --permitir-test

El guion escribe su propio registro (no redirigir la salida): en modo test, la
redirección crearía un archivo sin seguimiento antes de comprobar que git está
limpio.
"""

import argparse
import json
import platform
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import stats

CARPETA = Path(__file__).resolve().parent
LINEA = CARPETA.parent
RAIZ_REPO = LINEA.parent.parent
sys.path.insert(0, str(LINEA / "03_clasificador"))
sys.path.insert(0, str(LINEA / "02_artefactos"))
import artefactos as art  # noqa: E402
import comun  # noqa: E402
import medidas as md  # noqa: E402
from comun import FAMILIA, d1  # noqa: E402

from eeg_library.modelado.confirmatorio import (  # noqa: E402
    carpeta_salida_nueva, estado_git, holm, ic_percentil, p_bootstrap, replicas_bootstrap, sha256_archivo,
)
from eeg_library.procesado.features import anotar_validez_de_canales  # noqa: E402

PREREGISTRO = CARPETA / "PREREGISTRO.md"
CONFIG = LINEA / "03_clasificador" / "configuracion_elegida.json"
MEDIDA = LINEA / "03_clasificador" / "medida_elegida.json"

CONDICIONES = ["PASAT", "PVTB", "OA", "PVSAT", "OC"]
VERSIONES = ["completa", "igual_tiempo"]
N_BOOT = 10000
SEMILLA = 42
ALFA = 0.05


class Tee:
    """Escribe a la vez en la consola y en un archivo de registro."""

    def __init__(self, ruta):
        self.archivo = open(ruta, "w", encoding="utf-8")
        self.consola = sys.stdout

    def write(self, texto):
        self.consola.write(texto)
        self.archivo.write(texto)

    def flush(self):
        self.consola.flush()
        self.archivo.flush()


# --- Carga ---------------------------------------------------------------------


def cargar(particiones, permitir_test):
    """Ventanas etiquetadas de las particiones pedidas (como `comun.cargar`, pero puede leer `test`)."""
    particiones = set(particiones)
    if "test" in particiones and not permitir_test:
        raise PermissionError("Leer test exige el modo --test --permitir-test")
    matriz = pd.read_parquet(d1.RUTA_MATRIZ, columns=["archivo", "persona_uid", "sesion_uid", "split"])
    etiq = matriz[matriz.persona_uid.notna() & matriz.split.isin(particiones)]
    archivos = sorted(etiq.archivo)
    columnas = ["condicion", "archivo_origen", "indice_ventana", "tmin_s"] + comun.REL + comun.ABS
    w = pd.read_parquet(d1.RUTA_FEATURES, columns=columnas, filters=[("archivo_origen", "in", archivos)])
    w = w.merge(etiq[["archivo", "persona_uid", "sesion_uid", "split"]], left_on="archivo_origen",
                right_on="archivo").drop(columns="archivo")
    assert set(w.split) <= particiones
    log = pd.read_csv(d1.RUTA_LOG)
    mascara = anotar_validez_de_canales(pd.DataFrame({"archivo_origen": w.archivo_origen.unique()}), log)
    w = w.merge(mascara[["archivo_origen"] + d1.VALIDEZ], on="archivo_origen")
    w["t_rec"] = w.tmin_s / 60
    w["n_vent"] = w.groupby("archivo_origen").tmin_s.transform("size")
    w["dur"] = w.groupby("archivo_origen").t_rec.transform("max")
    w = w[w.condicion.isin(CONDICIONES) & (w.n_vent >= d1.MIN_VENTANAS)].reset_index(drop=True)
    return w, log


def personas_expuestas(evalua):
    """Personas evaluadas cuya `split` antigua (la de `features_all`) era `train`: las vistas en los sondeos."""
    fa = pq_split()
    antigua = fa.set_index("archivo_origen").split
    archivos = evalua.drop_duplicates("archivo_origen").set_index("archivo_origen").persona_uid
    vistas = archivos[antigua.reindex(archivos.index).eq("train").to_numpy()]
    return sorted(vistas.unique())


def pq_split():
    import pyarrow.parquet as pq
    return pq.read_table(d1.RUTA_FEATURES, columns=["archivo_origen", "split"]).to_pandas().drop_duplicates(
        "archivo_origen")


# --- Medidas -------------------------------------------------------------------


def rho_y_tercios(entrena, evalua, feats, mod):
    """Por condición: ρ por grabación (rho_ventanas) y acierto pareado de tercios (descriptivo)."""
    modelos, rho, tercios = {}, {}, {}
    for c in CONDICIONES:
        S_tr, m_tr = comun.tercios(entrena, c)
        m = comun.modelo(mod).fit(comun.matriz(S_tr, m_tr, feats), S_tr.seg.to_numpy())
        modelos[c] = m
        filas, Xv = md.ventanas(evalua, c)
        rho[c] = md.rho_por_grabacion(filas, m.decision_function(Xv)).assign(condicion=c)
        S_ev, m_ev = comun.tercios(evalua, c)
        p = m.predict_proba(comun.matriz(S_ev, m_ev, feats))[:, 1]
        tercios[c] = comun.aciertos_por_grabacion(S_ev, p).assign(condicion=c)
    return modelos, rho, tercios


def pendientes_beta(evalua):
    """Diferencia de pendientes de la beta posterior relativa (tarea − OA) por persona."""
    nombre, pref, reg, banda = next(m for m in d1.MAGNITUDES_SONDEO if m[0] == "beta_post_rel")
    y = d1.magnitud_regional(evalua, pref, d1.REGIONES_SONDEO[reg], banda, validos=True)
    d = pd.DataFrame({"a": evalua.archivo_origen.to_numpy(), "p": evalua.persona_uid.to_numpy(),
                      "c": evalua.condicion.to_numpy(), "t": evalua.t_rec.to_numpy(), "y": y})
    d = d[np.isfinite(d.y) & d.c.isin(FAMILIA + ["OA"])]
    d["tc"] = d.t - d.groupby("a").t.transform("mean")
    d["yc"] = d.y - d.groupby("a").y.transform("mean")
    d["xy"], d["xx"] = d.tc * d.yc, d.tc ** 2
    g = d.groupby(["a", "p", "c"])[["xy", "xx"]].sum().reset_index()
    pp = g.assign(pend=g.xy / g.xx).groupby(["p", "c"]).pend.mean().unstack()
    return pp


def artefactos_por_ventana(w):
    """Medidas de movimiento y contacto por ventana (EDF), solo PASAT y PVTB; sin movimiento → fuera."""
    g = w[w.condicion.isin(FAMILIA)]
    validez = g.drop_duplicates("archivo_origen").set_index("archivo_origen")[d1.VALIDEZ]
    n = g.groupby("archivo_origen").size()
    partes = [art.medidas_grabacion(a, int(n[a]), [c for c, v in zip(d1.CANALES_EPOC, validez.loc[a]) if v])
              for a in n.index]
    m = pd.concat(partes, ignore_index=True)
    g = g.merge(m, on=["archivo_origen", "indice_ventana"], how="left", validate="one_to_one")
    sin_mov = g[g.mov_acc.isna()].archivo_origen.unique()
    return g[g.mov_acc.notna()].reset_index(drop=True), sin_mov


def tercios_artefactos(w, c, S):
    """Medias por tercio de las medidas de artefacto, centradas por grabación, alineadas con las filas de `S`."""
    g = w[w.condicion == c]
    seg = np.where(g.t_rec <= g.dur / 3, 0, np.where(g.t_rec >= 2 * g.dur / 3, 1, -1))
    g = g[seg >= 0].assign(seg=seg[seg >= 0])
    A = g.groupby(["archivo_origen", "persona_uid", "seg"])[art.ART].mean().reset_index()
    assert A[["archivo_origen", "seg"]].equals(S[["archivo_origen", "seg"]]), "tercios desalineados"
    return (A[art.ART] - A[art.ART].groupby(A.archivo_origen).transform("mean")).to_numpy()


def h4_aciertos(entrena_art, evalua_art, feats, mod):
    """Acierto pareado por grabación del EEG (configuración fijada) y de solo artefactos (modelo del paso 2)."""
    filas = []
    for c in FAMILIA:
        S_tr, m_tr = comun.tercios(entrena_art, c)
        S_ev, m_ev = comun.tercios(evalua_art, c)
        p_eeg = (comun.modelo(mod).fit(comun.matriz(S_tr, m_tr, feats), S_tr.seg.to_numpy())
                 .predict_proba(comun.matriz(S_ev, m_ev, feats))[:, 1])
        p_art = (comun.modelo("lr_0.1").fit(tercios_artefactos(entrena_art, c, S_tr), S_tr.seg.to_numpy())
                 .predict_proba(tercios_artefactos(evalua_art, c, S_ev))[:, 1])
        a_eeg = comun.aciertos_por_grabacion(S_ev, p_eeg).set_index(["archivo_origen", "persona_uid"]).acierto
        a_art = comun.aciertos_por_grabacion(S_ev, p_art).set_index(["archivo_origen", "persona_uid"]).acierto
        filas.append(pd.DataFrame({"eeg": a_eeg, "art": a_art}).dropna().reset_index().assign(condicion=c))
    return pd.concat(filas, ignore_index=True)


# --- Contrastes ------------------------------------------------------------------


def bootstrap_h1_h2(rho_v, excluir=()):
    """Réplicas pareadas (mismas personas) de H1b, H1, ρ de PASAT y de PVTB a igual tiempo."""
    filas = pd.concat([rho_v[v][c].assign(version=v) for v in VERSIONES for c in CONDICIONES], ignore_index=True)
    filas = filas[~filas.persona_uid.isin(excluir)].reset_index(drop=True)
    ver, cond, val = filas.version.to_numpy(), filas.condicion.to_numpy(), filas.acierto.to_numpy()

    def media(idx, v, cs):
        sel = idx[(ver[idx] == v) & np.isin(cond[idx], cs)]
        if len(sel) == 0:
            raise ValueError("remuestra sin grabaciones")
        return val[sel].mean()

    def estadistico(idx):
        return np.array([media(idx, "completa", FAMILIA) - media(idx, "completa", ["OA"]),
                         media(idx, "igual_tiempo", FAMILIA) - media(idx, "igual_tiempo", ["OA"]),
                         media(idx, "igual_tiempo", ["PASAT"]), media(idx, "igual_tiempo", ["PVTB"])])

    observado = estadistico(np.arange(len(filas)))
    reps = replicas_bootstrap(filas.persona_uid.to_numpy(), estadistico, n_replicas=N_BOOT, semilla=SEMILLA)
    return observado, reps, filas


def resumen_contraste(observado, reps):
    return {"valor": float(observado), "ic95": ic_percentil(reps), "p_unilateral": p_bootstrap(reps, 0, "menor_o_igual"),
            "replicas_validas": int(np.isfinite(reps).sum())}


def main():
    ap = argparse.ArgumentParser()
    modo = ap.add_mutually_exclusive_group(required=True)
    modo.add_argument("--ensayo", action="store_true")
    modo.add_argument("--test", action="store_true")
    ap.add_argument("--permitir-test", action="store_true")
    args = ap.parse_args()
    if args.test and not args.permitir_test:
        sys.exit("El modo --test exige --permitir-test.")

    inicio = datetime.now()
    git = estado_git(RAIZ_REPO)
    if args.test:
        if not git["limpio"]:
            sys.exit("El árbol de git no está limpio: el pre-registro y el guion deben estar commiteados.\n  "
                     + "\n  ".join(git["cambios"]))
        salida = carpeta_salida_nueva(d1.EEG_DATA / "processed" / f"fatiga_aguda_test_{inicio:%Y%m%d-%H%M%S}")
        entrena_p, evalua_p, nombre = {"train", "val"}, {"test"}, "TEST (confirmatorio)"
    else:
        salida = Path(tempfile.mkdtemp(prefix="fatiga_aguda_ensayo_"))
        entrena_p, evalua_p, nombre = {"train"}, {"val"}, "ENSAYO (train → val)"
    sys.stdout = Tee(salida / "ejecucion.log")
    for sub in ("bootstrap", "modelos"):
        (salida / sub).mkdir()

    with open(CONFIG, encoding="utf-8") as f:
        config = json.load(f)
    with open(MEDIDA, encoding="utf-8") as f:
        medida = json.load(f)
    feats, mod = config["features"], config["modelo"]
    assert (feats, mod, config["esquema"], medida["medida"]) == ("rel", "lr_0.01", "separado", "rho_ventanas")

    d1._titulo(f"0. {nombre}")
    print(f"Inicio: {inicio:%Y-%m-%d %H:%M:%S}. Git {git['commit'][:12]}, limpio = {git['limpio']}")
    print(f"Configuración: {feats}, {mod}, {config['esquema']}; medida: {medida['medida']}")
    print(f"Salida: {salida}")
    w, log = cargar(entrena_p | evalua_p, permitir_test=args.test)
    entrena = w[w.split.isin(entrena_p)].reset_index(drop=True)
    evalua = w[w.split.isin(evalua_p)].reset_index(drop=True)
    assert not set(entrena.persona_uid) & set(evalua.persona_uid)
    expuestas = personas_expuestas(evalua)
    sfreq = log.drop_duplicates("archivo").set_index("archivo").sfreq_original
    a_128 = set(evalua.archivo_origen[evalua.archivo_origen.map(sfreq).eq(128).to_numpy()])
    inventario = (evalua.drop_duplicates("archivo_origen").groupby("condicion")
                  .agg(grabaciones=("archivo_origen", "size"), personas=("persona_uid", "nunique")))
    print(f"Personas: entrena {entrena.persona_uid.nunique()}, evalúa {evalua.persona_uid.nunique()}; "
          f"expuestas en los sondeos: {len(expuestas)}; grabaciones evaluadas a 128 Hz: {len(a_128)}")
    print(inventario.to_string())

    v_ent, v_ev = comun.versiones(entrena), comun.versiones(evalua)
    rho_v, tercios_v = {}, {}
    for v in VERSIONES:
        modelos, rho_v[v], tercios_v[v] = rho_y_tercios(v_ent[v], v_ev[v], feats, mod)
        for c, m in modelos.items():
            joblib.dump(m, salida / "modelos" / f"{v}_{c}.joblib")

    d1._titulo("1. PRINCIPALES EN SECUENCIA FIJA: H1b → H1 (bootstrap de personas, 10 000 réplicas)")
    obs, reps, filas = bootstrap_h1_h2(rho_v)
    np.save(salida / "bootstrap" / "h1b_h1_rhoPASAT_rhoPVTB.npy", reps)
    h1b, h1 = resumen_contraste(obs[0], reps[:, 0]), resumen_contraste(obs[1], reps[:, 1])
    h1b["veredicto"] = "CONFIRMA" if h1b["p_unilateral"] < ALFA else "NO CONCLUYE"
    if h1b["veredicto"] == "CONFIRMA":
        h1["veredicto"] = "CONFIRMA" if h1["p_unilateral"] < ALFA else "NO CONCLUYE"
    else:
        h1["veredicto"] = "NO SE CONTRASTA (H1b no confirmó); se reporta como descriptiva"
    for nombre_h, x in (("H1b (completa)", h1b), ("H1 (igual tiempo)", h1)):
        print(f"  {nombre_h:18s} Δρ = {x['valor']:+.4f}  IC95 [{x['ic95'][0]:+.4f}, {x['ic95'][1]:+.4f}]  "
              f"p = {x['p_unilateral']:.4f}  → {x['veredicto']}")

    d1._titulo("2. SECUNDARIAS")
    h2 = {c: resumen_contraste(obs[2 + i], reps[:, 2 + i]) for i, c in enumerate(FAMILIA)}
    for c, p_aj in zip(FAMILIA, holm([h2[c]["p_unilateral"] for c in FAMILIA])):
        h2[c]["p_holm"] = float(p_aj)
        h2[c]["veredicto"] = "CONFIRMA" if p_aj < ALFA else "NO CONCLUYE"
        print(f"  H2 {c:5s} ρ = {h2[c]['valor']:+.4f}  IC95 [{h2[c]['ic95'][0]:+.4f}, {h2[c]['ic95'][1]:+.4f}]  "
              f"p = {h2[c]['p_unilateral']:.4f}, Holm {p_aj:.4f}  → {h2[c]['veredicto']}")

    pp = pendientes_beta(v_ev["igual_tiempo"])
    h3 = {}
    for c in FAMILIA:
        dif = (pp[c] - pp["OA"]).dropna()
        t = stats.ttest_1samp(dif, 0, alternative="less")
        h3[c] = {"diferencia": float(dif.mean()), "t": float(t.statistic), "p_unilateral": float(t.pvalue),
                 "personas": len(dif)}
    for c, p_aj in zip(FAMILIA, holm([h3[c]["p_unilateral"] for c in FAMILIA])):
        h3[c]["p_holm"] = float(p_aj)
        h3[c]["veredicto"] = "CONFIRMA" if p_aj < ALFA else "NO CONCLUYE (potencia 26–38 %: no indica ausencia)"
        print(f"  H3 {c:5s} Δpendiente = {h3[c]['diferencia']:+.4f}  t = {h3[c]['t']:+.2f}  "
              f"p = {h3[c]['p_unilateral']:.4f}, Holm {p_aj:.4f}  → {h3[c]['veredicto']}")

    ent_art, sin_mov_ent = artefactos_por_ventana(v_ent["igual_tiempo"])
    ev_art, sin_mov_ev = artefactos_por_ventana(v_ev["igual_tiempo"])
    a4 = h4_aciertos(ent_art, ev_art, feats, mod)
    eeg, artv = a4.eeg.to_numpy(), a4.art.to_numpy()
    reps4 = replicas_bootstrap(a4.persona_uid.to_numpy(), lambda i: eeg[i].mean() - artv[i].mean(),
                               n_replicas=N_BOOT, semilla=SEMILLA)
    np.save(salida / "bootstrap" / "h4.npy", reps4)
    h4 = resumen_contraste(eeg.mean() - artv.mean(), reps4)
    h4.update({"eeg": float(eeg.mean()), "art": float(artv.mean()), "grabaciones": len(a4),
               "sin_movimiento_evaluadas": len(sin_mov_ev), "sin_movimiento_entreno": len(sin_mov_ent)})
    h4["veredicto"] = "CONFIRMA" if h4["p_unilateral"] < ALFA else "NO CONCLUYE"
    print(f"  H4 EEG {h4['eeg']:.3f} − artefactos {h4['art']:.3f} = {h4['valor']:+.4f}  "
          f"IC95 [{h4['ic95'][0]:+.4f}, {h4['ic95'][1]:+.4f}]  p = {h4['p_unilateral']:.4f}  → {h4['veredicto']}  "
          f"({h4['grabaciones']} grab.; sin movimiento: {len(sin_mov_ev)})")

    d1._titulo("3. DESCRIPTIVOS (sin veredicto)")
    rng = np.random.default_rng(SEMILLA)
    descriptivos = {}
    for v in VERSIONES:
        for c in CONDICIONES:
            for etiqueta, tabla in (("rho", rho_v[v][c]), ("tercios_pareado", tercios_v[v][c])):
                porp = tabla.groupby("persona_uid").acierto.agg(["sum", "size"])
                ic = d1.bootstrap_personas(porp["sum"].to_numpy(), porp["size"].to_numpy(), rng)
                descriptivos[f"{v} | {c} | {etiqueta}"] = {"valor": float(tabla.acierto.mean()),
                                                         "ic95": [float(x) for x in ic], "grabaciones": len(tabla)}
    for k, x in descriptivos.items():
        print(f"  {k:40s} {x['valor']:+.3f} [{x['ic95'][0]:+.3f}, {x['ic95'][1]:+.3f}]  ({x['grabaciones']} grab.)")

    d1._titulo("4. SENSIBILIDADES (sin veredicto)")
    sensibilidades = {}
    for etiqueta, excluir_p, excluir_a in (("sin_personas_expuestas", set(expuestas), set()),
                                           ("sin_grabaciones_128_hz", set(), a_128)):
        rv = {v: {c: x[~x.archivo_origen.isin(excluir_a)] for c, x in rho_v[v].items()} for v in VERSIONES}
        o, r, _ = bootstrap_h1_h2(rv, excluir=excluir_p)
        sensibilidades[etiqueta] = {n: resumen_contraste(o[i], r[:, i])
                                    for i, n in enumerate(["H1b", "H1", "rho_PASAT", "rho_PVTB"])}
        print(f"  {etiqueta}: " + "  ".join(f"{n} {x['valor']:+.4f} [{x['ic95'][0]:+.4f}, {x['ic95'][1]:+.4f}]"
                                           for n, x in sensibilidades[etiqueta].items()))

    pd.concat([filas, *[t.assign(version=v, medida="tercios_pareado") for v in VERSIONES
                        for t in tercios_v[v].values()]], ignore_index=True).to_parquet(
        salida / "predicciones_por_grabacion.parquet")
    a4.to_parquet(salida / "h4_por_grabacion.parquet")
    resultados = {
        "modo": "test" if args.test else "ensayo", "inicio": f"{inicio:%Y-%m-%d %H:%M:%S}",
        "fin": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "git": git,
        "sha256": {"guion": sha256_archivo(Path(__file__)), "preregistro": sha256_archivo(PREREGISTRO),
                   **{p.name: sha256_archivo(p) for p in (d1.RUTA_FEATURES, d1.RUTA_MATRIZ, d1.RUTA_LOG, CONFIG, MEDIDA)}},
        "entorno": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__,
                    "pandas": pd.__version__, "so": platform.platform(), "cpu": platform.processor()},
        "parametros": {"n_boot": N_BOOT, "semilla": SEMILLA, "alfa": ALFA},
        "inventario": inventario.reset_index().to_dict(orient="records"),
        "personas_expuestas": len(expuestas), "grabaciones_128_hz": len(a_128),
        "H1b": h1b, "H1": h1, "H2": h2, "H3": h3, "H4": h4,
        "descriptivos": descriptivos, "sensibilidades": sensibilidades,
    }
    with open(salida / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    # Copia agregada en el repositorio (sin identificadores de persona).
    destino = CARPETA / ("resultados_test.json" if args.test else "resultados_ensayo.json")
    with open(destino, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in resultados.items() if k != "git"} | {"commit": git["commit"]}, f,
                  ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\nFin: {datetime.now():%Y-%m-%d %H:%M:%S}. Artefactos en {salida}")
    sys.stdout.flush()
    (CARPETA / ("ejecucion_test.log" if args.test else "ejecucion_ensayo.log")).write_text(
        (salida / "ejecucion.log").read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
