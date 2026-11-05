"""
Tiempo en tarea — paso 2: movimiento y calidad de contacto.

¿Puede el cambio del EEG con el tiempo en tarea ser un artefacto del casco?
Lee de cada EDF de PASAT, PVTB, PVSAT y OA el movimiento (acelerómetro y
cuaterniones, `MOT.*`) y la calidad de contacto (`CQ.*`), alineados a las
ventanas del pipeline, y comprueba: si cambian con el tiempo en tarea, qué
acierto alcanza un clasificador de solo artefactos, y si el clasificador EEG
sobrevive a añadirlos como features o a residualizar el EEG por ellos.

**Solo `train`.** Los EDF se leen en modo solo lectura con un lector propio,
sin remuestrear. Las medidas por ventana se calculan en memoria y no se
guardan. Reutiliza la carga y el bootstrap del paso 1 y el recorte a igual
tiempo de su anexo A, que no se modifican. Plan y criterio de continuidad:
`INFORME.md`, escritos antes de ejecutar este script.

Ejecutar (desde la raíz del repositorio):
    .venv/Scripts/python.exe analisis/tiempo_en_tarea/02_artefactos/artefactos.py > analisis/tiempo_en_tarea/02_artefactos/ejecucion.log
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent / "01_descriptivo"))
import anexo_igual_tiempo as anexo  # noqa: E402
import descriptivo_tiempo_en_tarea as d1  # noqa: E402

RUTA_EDF = d1.EEG_DATA / "dataset_final"
CARPETA_TABLAS = CARPETA / "tablas"
CARPETA_FIGURAS = CARPETA / "figuras"

CONDICIONES = ["PASAT", "PVTB", "PVSAT", "OA"]
TAREAS = ["PASAT", "PVTB", "PVSAT"]
FAMILIA = ["PASAT", "PVTB"]

# Alineación ventana ↔ EDF (comprobada en el INFORME): la ventana i cubre
# [5 + 2i, 9 + 2i) s del archivo original.
INICIO_S, PASO_S, DURACION_S = 5.0, 2.0, 4.0

ART = ["mov_acc", "mov_ang", "cq_validos", "cq_overall"]
EQ = ["eq_overall"]
MEDIDAS = ART + EQ
REL = d1.REL

# Modelos: (bloques de features, descripción). P = log10 de la media del
# tercio (como el paso 1); L = media del tercio de log10 por ventana; R y Q =
# residuos de L frente a ART y frente a ART + EQ.
MODELOS = {
    "eeg": (["P"], "56 pot_rel (como el paso 1)"),
    "art": (["A"], "solo artefactos"),
    "eeg+art": (["P", "A"], "EEG + artefactos"),
    "eeg_log_ventana": (["L"], "EEG con log por ventana (referencia del residualizado)"),
    "eeg_res": (["R"], "EEG residualizado por artefactos"),
    "art_eq": (["A", "E"], "solo artefactos + EQ"),
    "eeg+art_eq": (["P", "A", "E"], "EEG + artefactos + EQ"),
    "eeg_res_eq": (["Q"], "EEG residualizado por artefactos + EQ"),
}
BLOQUES_EEG = {"P", "L", "R", "Q"}

TINTA, TINTA_2, REJILLA = d1.TINTA, d1.TINTA_2, d1.REJILLA


# --- Lectura de EDF ----------------------------------------------------------


def leer_edf(ruta, senales):
    """Señales pedidas en unidades físicas y a su frecuencia nativa (sin remuestrear)."""
    with open(ruta, "rb") as f:
        b = f.read()
    ns = int(b[252:256])
    bytes_cabecera = int(b[184:192])
    n_registros = int(b[236:244])
    dur_registro = float(b[244:252])
    cab = b[256:256 + ns * 256]

    def campo(desplazamiento, ancho, i):
        ini = desplazamiento * ns + i * ancho
        return cab[ini:ini + ancho].decode("latin1").strip()

    etiquetas = [campo(0, 16, i) for i in range(ns)]
    pmin, pmax, dmin, dmax = (np.array([float(campo(d, 8, i)) for i in range(ns)])
                              for d in (104, 112, 120, 128))
    muestras = np.array([int(campo(216, 8, i)) for i in range(ns)])
    total = n_registros * muestras.sum()
    if bytes_cabecera + 2 * total > len(b):
        raise ValueError(f"EDF truncado: {ruta.name}")
    datos = np.frombuffer(b, dtype="<i2", count=total, offset=bytes_cabecera).reshape(n_registros, -1)
    desplaz = np.concatenate([[0], np.cumsum(muestras)])
    salida = {}
    for s in senales:
        if s not in etiquetas:
            continue
        i = etiquetas.index(s)
        x = datos[:, desplaz[i]:desplaz[i + 1]].reshape(-1).astype(float)
        salida[s] = ((x - dmin[i]) * (pmax[i] - pmin[i]) / (dmax[i] - dmin[i]) + pmin[i],
                     muestras[i] / dur_registro)
    return salida


def _por_ventana(x, fs, n_vent):
    """Matriz (n_vent, muestras) con las muestras de cada ventana."""
    largo = int(round(DURACION_S * fs))
    ini = np.round((INICIO_S + PASO_S * np.arange(n_vent)) * fs).astype(int)
    if ini[-1] + largo > len(x):
        raise ValueError("La última ventana se sale del EDF: la alineación no cuadra")
    return x[ini[:, None] + np.arange(largo)]


def medidas_grabacion(archivo, n_vent, canales_validos):
    cuat = [f"MOT.Q{i}" for i in range(4)]
    acc = ["MOT.AccX", "MOT.AccY", "MOT.AccZ"]
    cq = [f"CQ.{c}" for c in canales_validos]
    s = leer_edf(RUTA_EDF / archivo, cuat + acc + cq + ["CQ.Overall", "EQ.OVERALL"])
    out = {"indice_ventana": np.arange(n_vent)}
    if all(k in s for k in cuat + acc):
        fs = s["MOT.AccX"][1]
        modulo = np.sqrt(sum(s[k][0] ** 2 for k in acc))
        out["mov_acc"] = np.log10(_por_ventana(modulo, fs, n_vent).std(axis=1) + 1e-4)
        q = np.stack([s[k][0] for k in cuat], axis=1)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        prod = np.clip(np.abs((q[1:] * q[:-1]).sum(axis=1)), 0, 1)
        vel = np.degrees(2 * np.arccos(prod)) * fs
        vel = np.append(vel, vel[-1])  # el par (t, t+1) se asigna a t
        out["mov_ang"] = np.log10(_por_ventana(vel, fs, n_vent).mean(axis=1) + 1e-3)
    else:
        out["mov_acc"] = out["mov_ang"] = np.full(n_vent, np.nan)
    fs_cq = s["CQ.Overall"][1]
    if cq:
        media_cq = np.mean([s[k][0] for k in cq], axis=0)
        out["cq_validos"] = _por_ventana(media_cq, fs_cq, n_vent).mean(axis=1)
    else:
        out["cq_validos"] = np.full(n_vent, np.nan)
    out["cq_overall"] = _por_ventana(s["CQ.Overall"][0], fs_cq, n_vent).mean(axis=1)
    out["eq_overall"] = _por_ventana(s["EQ.OVERALL"][0], s["EQ.OVERALL"][1], n_vent).mean(axis=1)
    return pd.DataFrame(out).assign(archivo_origen=archivo)


def cargar_con_artefactos():
    w, _, _ = d1.cargar()
    w = w[w.condicion.isin(CONDICIONES) & (w.n_vent >= d1.MIN_VENTANAS)].reset_index(drop=True)
    por_grab = w.groupby("archivo_origen").agg(n=("indice_ventana", "size"), ultimo=("indice_ventana", "max"))
    assert (por_grab.n == por_grab.ultimo + 1).all(), "Índices de ventana no consecutivos"
    validez = w.drop_duplicates("archivo_origen").set_index("archivo_origen")[d1.VALIDEZ]
    t0 = time.time()
    partes = []
    for archivo, n in por_grab.n.items():
        validos = [c for c, v in zip(d1.CANALES_EPOC, validez.loc[archivo]) if v]
        partes.append(medidas_grabacion(archivo, n, validos))
    art = pd.concat(partes, ignore_index=True)
    print(f"EDF leídos: {len(partes)} en {time.time() - t0:.0f} s")
    w = w.merge(art, on=["archivo_origen", "indice_ventana"], how="left", validate="one_to_one")
    assert w[["cq_overall", "eq_overall"]].notna().all().all()
    return w, len(partes)


# --- Construcción de features por tercio -------------------------------------


def residualizar(w, columnas_art):
    """log10(pot_rel) por ventana menos su ajuste MCO sobre los artefactos, dentro de cada grabación."""
    Y = np.log10(np.clip(w[REL].to_numpy(float), 1e-6, None))
    Z = w[columnas_art].to_numpy(float)
    res = np.empty_like(Y)
    for idx in w.groupby("archivo_origen").indices.values():
        y = Y[idx] - Y[idx].mean(axis=0)
        z = Z[idx] - Z[idx].mean(axis=0)
        z = z[:, z.std(axis=0) > 1e-12]  # variables constantes en esta grabación
        if z.shape[1]:
            beta = np.linalg.lstsq(z, y, rcond=None)[0]
            y = y - z @ beta
        res[idx] = y
    return res


def preparar(w):
    """Añade a las ventanas los bloques L, R y Q (una columna por pot_rel)."""
    w = w.reset_index(drop=True)
    bloques = {
        "L": np.log10(np.clip(w[REL].to_numpy(float), 1e-6, None)),
        "R": residualizar(w, ART),
        "Q": residualizar(w, ART + EQ),
    }
    extra = pd.DataFrame(np.hstack(list(bloques.values())),
                         columns=[f"{b}_{c}" for b in bloques for c in REL])
    return pd.concat([w, extra], axis=1)


def tercios(w, cond):
    """Una fila por (grabación, tercio) con todos los bloques de features y la máscara de canales."""
    g = w[(w.condicion == cond) & (w.n_vent >= d1.MIN_VENTANAS)]
    seg = np.where(g.t_rec <= g.dur / 3, 0, np.where(g.t_rec >= 2 * g.dur / 3, 1, -1))
    g = g[seg >= 0].assign(seg=seg[seg >= 0])
    claves = ["archivo_origen", "persona_uid", "seg"]
    otras = [f"{b}_{c}" for b in "LRQ" for c in REL] + ART + EQ
    S = g.groupby(claves)[REL + otras].mean().reset_index()
    S[[f"P_{c}" for c in REL]] = np.log10(S[REL].clip(lower=1e-6)).to_numpy()
    validez = g.drop_duplicates("archivo_origen").set_index("archivo_origen")[d1.VALIDEZ]
    mascara = np.repeat(validez.loc[S.archivo_origen].to_numpy(dtype=bool), len(d1.BANDAS), axis=1)
    return S, mascara


def columnas(bloque):
    if bloque == "A":
        return ART
    if bloque == "E":
        return EQ
    return [f"{bloque}_{c}" for c in REL]


def clasificar(S, mascara, bloques):
    """Clasificador del paso 1 (centrado por grabación, canales excluidos a 0) sobre los bloques pedidos."""
    cols = [c for b in bloques for c in columnas(b)]
    F = S[cols] - S[cols].groupby(S.archivo_origen).transform("mean")
    for b in bloques:
        if b in BLOQUES_EEG:
            F[columnas(b)] = F[columnas(b)].mask(~mascara, 0.0)
    modelo = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    p = cross_val_predict(modelo, F.to_numpy(), S.seg.to_numpy(), groups=S.persona_uid,
                          cv=GroupKFold(5), method="predict_proba")[:, 1]
    par = S.assign(p=p).pivot_table(index=["archivo_origen", "persona_uid"], columns="seg",
                                    values="p").dropna().reset_index()
    par["acierto"] = np.where(par[1] > par[0], 1.0, np.where(par[1] == par[0], 0.5, 0.0))
    return {"auc": roc_auc_score(S.seg, p), "pareado": par.acierto.mean(),
            "empates": int((par[1] == par[0]).sum()), "grabaciones": len(par),
            "personas": par.persona_uid.nunique()}, par[["archivo_origen", "persona_uid", "acierto"]]


# --- 1. Tendencias de los artefactos -----------------------------------------


def pendientes_por_persona(w, col):
    d = w[["archivo_origen", "persona_uid", "condicion", "t_rec", col]].dropna()
    tc = d.t_rec - d.groupby("archivo_origen").t_rec.transform("mean")
    yc = d[col] - d.groupby("archivo_origen")[col].transform("mean")
    g = (pd.DataFrame({"a": d.archivo_origen, "p": d.persona_uid, "c": d.condicion, "xy": tc * yc, "xx": tc ** 2})
         .groupby(["a", "p", "c"])[["xy", "xx"]].sum().reset_index())
    g["pend"] = np.where(g.xx > 0, g.xy / g.xx.where(g.xx > 0, 1), np.nan)
    return g.groupby(["p", "c"]).pend.mean().unstack()


def tercio_final_menos_inicial(w, col):
    seg = np.where(w.t_rec <= w.dur / 3, 0, np.where(w.t_rec >= 2 * w.dur / 3, 1, -1))
    g = w.assign(seg=seg)[seg >= 0].groupby(["archivo_origen", "persona_uid", "condicion", "seg"])[col].mean()
    g = g.unstack("seg")
    return (g[1] - g[0]).groupby(["persona_uid", "condicion"]).mean().unstack()


def tendencias(versiones):
    d1._titulo("1. ¿CAMBIAN EL MOVIMIENTO Y EL CONTACTO CON EL TIEMPO EN TAREA?")
    filas = []
    for version, w in versiones.items():
        for col in MEDIDAS:
            pend = pendientes_por_persona(w, col)
            dif = tercio_final_menos_inicial(w, col)
            for cond in CONDICIONES:
                v = pend[cond].dropna()
                t, p = stats.ttest_1samp(v, 0) if v.std() > 0 else (np.nan, np.nan)
                dt = dif[cond].dropna()
                fila = {"version": version, "medida": col, "condicion": cond, "personas": len(v),
                        "pendiente": v.mean(), "t": t, "p": p, "final_menos_inicial": dt.mean()}
                if cond != "OA":
                    pd_ = (pend[cond] - pend["OA"]).dropna()
                    td, ptd = stats.ttest_1samp(pd_, 0) if pd_.std() > 0 else (np.nan, np.nan)
                    fila.update({"pend_menos_oa": pd_.mean(), "t_menos_oa": td, "p_menos_oa": ptd})
                filas.append(fila)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(CARPETA_TABLAS / "tendencias_artefactos.csv", index=False)
    for version in versiones:
        sub = tabla[tabla.version == version]
        print(f"\nVersión {version}: pendiente por minuto (t) y, para las tareas, diferencia con OA (t):")
        celda = sub.apply(lambda r: f"{r.pendiente:+.4f} ({r.t:+.1f})"
                          + ("" if r.condicion == "OA" or pd.isna(r.get("pend_menos_oa"))
                             else f" | {r.pend_menos_oa:+.4f} ({r.t_menos_oa:+.1f})"), axis=1)
        print(sub.assign(c=celda).pivot(index="medida", columns="condicion", values="c")
              .loc[MEDIDAS, CONDICIONES].to_string())
    return tabla


# --- 2-4. Clasificadores ------------------------------------------------------


def clasificadores(versiones):
    d1._titulo("2-4. CLASIFICADORES: SOLO ARTEFACTOS, EEG + ARTEFACTOS, EEG RESIDUALIZADO")
    rng = np.random.default_rng(d1.SEMILLA)
    filas, aciertos = [], {}
    for version, w in versiones.items():
        for cond in CONDICIONES:
            S, mascara = tercios(w, cond)
            for nombre, (bloques, _) in MODELOS.items():
                m, par = clasificar(S, mascara, bloques)
                aciertos[(version, nombre, cond)] = par
                porp = par.groupby("persona_uid").acierto.agg(["sum", "size"])
                ic = d1.bootstrap_personas(porp["sum"].to_numpy(), porp["size"].to_numpy(), rng)
                filas.append({"version": version, "modelo": nombre, "condicion": cond, **m,
                              "pareado_ic95": [float(x) for x in ic]})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(CARPETA_TABLAS / "clasificadores.csv", index=False)
    for version in versiones:
        sub = tabla[tabla.version == version]
        print(f"\nAcierto pareado — versión {version} (grabaciones por condición: "
              f"{sub[sub.modelo == 'eeg'].set_index('condicion').grabaciones.to_dict()}):")
        print(sub.assign(c=sub.pareado.map("{:.3f}".format) + sub.empates.map(lambda e: f" ({e} emp.)" if e else ""))
              .pivot(index="modelo", columns="condicion", values="c").loc[list(MODELOS), CONDICIONES].to_string())
    return tabla, aciertos


def contrastes(aciertos, versiones):
    d1._titulo("5. CONTRASTES (bootstrap de personas)")
    rng = np.random.default_rng(d1.SEMILLA + 1)

    def juntar(version, modelo, conds):
        return pd.concat([aciertos[(version, modelo, c)] for c in conds], ignore_index=True)

    definiciones = [
        ("eeg − art", "eeg", None, "art", None),
        ("eeg − art_eq", "eeg", None, "art_eq", None),
        ("eeg+art − eeg", "eeg+art", None, "eeg", None),
        ("eeg_res − eeg_log_ventana", "eeg_res", None, "eeg_log_ventana", None),
        ("eeg − OA", "eeg", None, "eeg", "OA"),
        ("eeg_res − OA (residualizado)", "eeg_res", None, "eeg_res", "OA"),
        ("eeg_res_eq − OA (residualizado)", "eeg_res_eq", None, "eeg_res_eq", "OA"),
    ]
    filas = []
    for version in versiones:
        print(f"\nVersión {version}:")
        for etiqueta, ma, _, mb, cond_b in definiciones:
            for grupo in [["PASAT"], ["PVTB"], ["PVSAT"], FAMILIA]:
                a = juntar(version, ma, grupo)
                b = aciertos[(version, mb, cond_b)] if cond_b else juntar(version, mb, grupo)
                dif, ic = anexo.diferencia_bootstrap(a, b, rng)
                nombre = "+".join(grupo)
                filas.append({"version": version, "contraste": etiqueta, "tareas": nombre,
                              "diferencia": dif, "ic95": [float(x) for x in ic]})
                print(f"  {etiqueta:32s} {nombre:11s} {dif:+.3f} [{ic[0]:+.3f}, {ic[1]:+.3f}]")
    tabla = pd.DataFrame(filas)
    tabla.to_csv(CARPETA_TABLAS / "contrastes.csv", index=False)
    return tabla


# --- Figuras ------------------------------------------------------------------


def figura_aciertos(tabla):
    modelos = ["eeg", "eeg+art", "eeg_log_ventana", "eeg_res", "art"]
    etiquetas = ["EEG", "EEG + artefactos", "EEG (log por ventana)", "EEG residualizado", "solo artefactos"]
    tonos = ["#2a78d6", "#7aa9e3", "#1f9e77", "#7cc4ab", "#eb6834"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sub = tabla[tabla.version == "igual_tiempo"].set_index(["modelo", "condicion"])
    x = np.arange(len(CONDICIONES))
    ancho = 0.15
    for k, (m, et, tono) in enumerate(zip(modelos, etiquetas, tonos)):
        xs = x + (k - 2) * ancho
        for xi, cond in zip(xs, CONDICIONES):
            r = sub.loc[(m, cond)]
            ax.plot([xi, xi], r.pareado_ic95, color=tono, lw=2, solid_capstyle="round")
            ax.plot(xi, r.pareado, "o", color=tono, ms=5, label=et if cond == CONDICIONES[0] else None)
    ax.axhline(.5, color=TINTA_2, lw=1, ls=":")
    ax.set_xticks(x, CONDICIONES)
    ax.set_ylabel("acierto pareado (IC95)", color=TINTA_2)
    ax.grid(axis="y", color=REJILLA, lw=.6)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    ax.set_ylim(.35, .95)
    ax.set_title("Tercios a igual tiempo: EEG frente a artefactos — train, CV por persona", fontsize=10, color=TINTA)
    fig.tight_layout()
    fig.savefig(CARPETA_FIGURAS / "acierto_por_modelo.png", dpi=150)
    plt.close(fig)


def figura_curvas(w):
    """Medidas de artefacto por minuto, centradas por grabación (grabación completa)."""
    w = w.assign(minuto=(w.tmin_s // 60).astype(int) + 1)
    fig, ejes = plt.subplots(1, len(MEDIDAS), figsize=(15, 3.4))
    for ax, col in zip(ejes, MEDIDAS):
        y = w[col] - w.groupby("archivo_origen")[col].transform("mean")
        g = (w.assign(y=y).groupby(["condicion", "persona_uid", "archivo_origen", "minuto"]).y
             .agg(["mean", "size"]).reset_index())
        g = g[g["size"] >= d1.MIN_VENTANAS_MINUTO]
        pp = g.groupby(["condicion", "minuto", "persona_uid"])["mean"].mean().reset_index()
        for cond in CONDICIONES:
            s = pp[pp.condicion == cond].groupby("minuto")["mean"].agg(["mean", "sem", "size"])
            s = s[s["size"] >= s["size"].max() / 2]
            ax.errorbar(s.index, s["mean"], yerr=1.96 * s["sem"], color=d1.COLORES[cond], marker="o", ms=3,
                        lw=1.5, capsize=2, ls="--" if cond == "OA" else "-", label=cond)
        ax.axhline(0, color=REJILLA, lw=1, zorder=0)
        ax.set_title(col, fontsize=9, color=TINTA)
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("minuto", color=TINTA_2, fontsize=8)
        ax.tick_params(labelsize=7, colors=TINTA_2)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    ejes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Artefactos por minuto, centrados por grabación (media de personas ± IC95 normal) — train",
                 fontsize=10, color=TINTA)
    fig.tight_layout()
    fig.savefig(CARPETA_FIGURAS / "artefactos_por_minuto.png", dpi=150)
    plt.close(fig)


def main():
    CARPETA_TABLAS.mkdir(exist_ok=True)
    CARPETA_FIGURAS.mkdir(exist_ok=True)

    d1._titulo("0. CARGA Y MEDIDAS POR VENTANA (solo train)")
    w, n_edf = cargar_con_artefactos()
    sin_mov = w[w.mov_acc.isna()].groupby("condicion").archivo_origen.nunique()
    print("Grabaciones sin acelerómetro ni cuaterniones (fuera):", sin_mov.to_dict())
    w = w[w.mov_acc.notna()]
    desc = (w.groupby("archivo_origen")[MEDIDAS].agg(["mean", "std"]).describe().loc[["mean", "50%"]])
    print("\nResumen por grabación (media y desviación dentro de grabación):")
    print(desc.round(3).to_string())
    saturacion = {"cq_validos_en_4": float((w.cq_validos == 4).mean()),
                  "cq_overall_en_100": float((w.cq_overall == 100).mean()),
                  "grabaciones_cq_overall_constante":
                      float((w.groupby("archivo_origen").cq_overall.std() == 0).mean())}
    print("Saturación del CQ (fracción de ventanas / grabaciones):", {k: round(v, 3) for k, v in saturacion.items()})

    w_igual, sin_oa = anexo.recortar(w)
    print("Igual tiempo: grabaciones de tarea sin OA en su sesión:", sin_oa.to_dict())
    versiones = {"completa": preparar(w), "igual_tiempo": preparar(w_igual)}

    tabla_tend = tendencias(versiones)
    tabla_clf, aciertos = clasificadores(versiones)
    tabla_con = contrastes(aciertos, versiones)
    figura_aciertos(tabla_clf)
    figura_curvas(versiones["completa"])

    d1._titulo("6. CRITERIO DE CONTINUIDAD (igual tiempo, artefactos principales)")
    c = tabla_con[tabla_con.version == "igual_tiempo"].set_index(["contraste", "tareas"])
    criterio_a = {}
    for tarea in FAMILIA:
        r = c.loc[("eeg − art", tarea)]
        criterio_a[tarea] = bool(r.diferencia > 0 and r.ic95[0] > 0)
        print(f"  (a) {tarea}: EEG − solo artefactos {r.diferencia:+.3f} [{r.ic95[0]:+.3f}, {r.ic95[1]:+.3f}]: "
              f"{'sí' if criterio_a[tarea] else 'NO'}")
    r = c.loc[("eeg_res − OA (residualizado)", "PASAT+PVTB")]
    criterio_b = bool(r.diferencia > 0)
    print(f"  (b) PASAT+PVTB: EEG residualizado − OA {r.diferencia:+.3f} [{r.ic95[0]:+.3f}, {r.ic95[1]:+.3f}]: "
          f"{'sí' if criterio_b else 'NO'}")
    se_cumple = all(criterio_a.values()) and criterio_b
    print(f"  Criterio de continuidad: {'SE CUMPLE' if se_cumple else 'NO SE CUMPLE'}")

    resultados = {
        "fecha": "2026-10-27",
        "particion": "solo train (split de matriz_rf), etiquetadas",
        "sha256": {p.name: d1.sha256(p) for p in (d1.RUTA_FEATURES, d1.RUTA_MATRIZ, d1.RUTA_LOG)},
        "edf_leidos": n_edf,
        "sin_movimiento": {k: int(v) for k, v in sin_mov.items()},
        "igual_tiempo_sin_oa": {k: int(v) for k, v in sin_oa.items()},
        "saturacion_cq": saturacion,
        "tendencias": tabla_tend.round(5).to_dict(orient="records"),
        "clasificadores": tabla_clf.round(5).to_dict(orient="records"),
        "contrastes": tabla_con.round(5).to_dict(orient="records"),
        "criterio_continuidad": {"a": criterio_a, "b": criterio_b, "se_cumple": se_cumple},
        "miradas": {"val": 0, "test": 0},
    }
    with open(CARPETA / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
