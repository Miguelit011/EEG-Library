"""
Tiempo en tarea — paso 1: reproducir los sondeos del
2026-10-26 desde el repositorio y describir los datos.

Rehace las pendientes intra-grabación (`PLAN.md`, 3.2) y el clasificador
primer tercio frente a último tercio (3.3) con los dos defectos de los sondeos
corregidos: la partición y la identidad salen de `matriz_rf` (no de la `split`
de `features_all`, anterior a la Fase 12) y los canales excluidos por calidad
(Fase 11.5) no aportan. Añade las curvas por minuto y el descriptivo de
duraciones y ventanas.

**Solo `train`.** Las ventanas de `val` y `test` no se cargan: el parquet se
lee filtrado por los archivos de `train`. Plan y criterio de continuidad:
`INFORME.md`, escritos antes de ejecutar este script.

Ejecutar (desde la raíz del repositorio):
    .venv/Scripts/python.exe analisis/tiempo_en_tarea/01_descriptivo/descriptivo_tiempo_en_tarea.py > analisis/tiempo_en_tarea/01_descriptivo/ejecucion.log

La ubicación del dataset se toma de la variable de entorno `EEG_DATA`, con
`C:\\Users\\migue\\Desktop\\TFG\\EEG-Data` por defecto (ver `analisis/README.md`).
"""

import hashlib
import json
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_library.preprocesado.quality import CANALES_EPOC
from eeg_library.procesado.features import PREFIJO_VALIDEZ, anotar_validez_de_canales
from eeg_library.procesado.matrix import REGIONES

EEG_DATA = Path(os.environ.get("EEG_DATA", r"C:\Users\migue\Desktop\TFG\EEG-Data"))
RUTA_FEATURES = EEG_DATA / "processed" / "features_all.parquet"
RUTA_MATRIZ = EEG_DATA / "processed" / "matriz_rf.parquet"
RUTA_LOG = EEG_DATA / "processed" / "procesamiento_log.csv"

CARPETA = Path(__file__).resolve().parent
CARPETA_TABLAS = CARPETA / "tablas"
CARPETA_FIGURAS = CARPETA / "figuras"

SEMILLA = 42
N_BOOTSTRAP = 2000
MIN_VENTANAS = 10           # por grabación, para entrar en cualquier análisis
MIN_VENTANAS_MINUTO = 5     # por grabación y minuto, en las curvas
UMBRAL_CONTINUIDAD = 0.70

CONDICIONES = ["OC", "OA", "STROOP", "PASAT", "PVTB", "MSIT", "PVSAT"]
CONDICIONES_TERCIOS = ["PASAT", "PVTB", "PVSAT", "OA"]
TAREAS_CRITERIO = ["PASAT", "PVTB"]
BANDAS = ["delta", "theta", "alfa", "beta"]

# Regiones del sondeo (tabla 3.2). La frontal es más estrecha que la del
# proyecto (`REGIONES`): sin F7 ni F8.
REGIONES_SONDEO = {
    "post": ("P7", "O1", "O2", "P8"),
    "front": ("AF3", "F3", "F4", "AF4"),
}
# Magnitudes de la tabla 3.2: (nombre, prefijo, región del sondeo, banda).
MAGNITUDES_SONDEO = [
    ("alfa_post_abs", "pot_abs", "post", "alfa"),
    ("theta_front_abs", "pot_abs", "front", "theta"),
    ("beta_post_abs", "pot_abs", "post", "beta"),
    ("beta_front_abs", "pot_abs", "front", "beta"),
    ("delta_front_abs", "pot_abs", "front", "delta"),
    ("alfa_post_rel", "pot_rel", "post", "alfa"),
    ("theta_front_rel", "pot_rel", "front", "theta"),
    ("beta_post_rel", "pot_rel", "post", "beta"),
]
# Magnitudes de las curvas por minuto: (nombre, prefijo, banda).
MAGNITUDES_CURVAS = [
    ("alfa_abs", "pot_abs", "alfa"),
    ("theta_abs", "pot_abs", "theta"),
    ("beta_rel", "pot_rel", "beta"),
]

REL = [f"pot_rel_{c}_{b}" for c in CANALES_EPOC for b in BANDAS]
ABS = [f"pot_abs_{c}_{b}" for c in CANALES_EPOC for b in BANDAS]
VALIDEZ = [f"{PREFIJO_VALIDEZ}{c}" for c in CANALES_EPOC]

# Paleta (la de fatiga_fg_fc más dos tonos) y tinta neutra.
COLORES = {"PASAT": "#2a78d6", "PVTB": "#eb6834", "PVSAT": "#1f9e77", "OA": "#8a8984"}
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
REJILLA = "#e4e3df"


def _titulo(texto):
    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _redondear(x, n=4):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


# --- Carga -------------------------------------------------------------------


def cargar():
    """Ventanas de `train` con identidad de la matriz y máscara de validez."""
    matriz = pd.read_parquet(RUTA_MATRIZ, columns=["archivo", "persona_uid", "sesion_uid", "split"])
    etiquetadas = matriz[matriz.persona_uid.notna()]
    archivos_train = sorted(etiquetadas.loc[etiquetadas.split == "train", "archivo"])

    # Incidencia 3.5: solo las dos columnas de partición, sin EEG.
    split_fa = (pq.read_table(RUTA_FEATURES, columns=["archivo_origen", "split"]).to_pandas()
                .drop_duplicates("archivo_origen"))
    cruce = split_fa.merge(etiquetadas[["archivo", "persona_uid", "split"]],
                           left_on="archivo_origen", right_on="archivo", suffixes=("_fa", "_matriz"))
    distintas = cruce[cruce.split_fa != cruce.split_matriz]
    incidencia = (distintas.groupby(["split_fa", "split_matriz"])
                  .agg(grabaciones=("archivo", "size"), personas=("persona_uid", "nunique"))
                  .reset_index())

    columnas = ["condicion", "archivo_origen", "indice_ventana", "tmin_s"] + REL + ABS
    w = pd.read_parquet(RUTA_FEATURES, columns=columnas,
                        filters=[("archivo_origen", "in", archivos_train)])
    assert set(w.archivo_origen) <= set(archivos_train), "Se han colado ventanas fuera de train"
    w = w.merge(etiquetadas[["archivo", "persona_uid", "sesion_uid"]],
                left_on="archivo_origen", right_on="archivo").drop(columns="archivo")

    log = pd.read_csv(RUTA_LOG)
    mascara = anotar_validez_de_canales(pd.DataFrame({"archivo_origen": w.archivo_origen.unique()}), log)
    w = w.merge(mascara[["archivo_origen"] + VALIDEZ], on="archivo_origen")
    w["t_rec"] = w.tmin_s / 60
    w["n_vent"] = w.groupby("archivo_origen").tmin_s.transform("size")
    w["dur"] = w.groupby("archivo_origen").t_rec.transform("max")
    return w, incidencia, len(archivos_train)


def magnitud_regional(w, prefijo, canales, banda, validos):
    """log10 de la potencia media de la región, con o sin los canales excluidos."""
    valores = w[[f"{prefijo}_{c}_{banda}" for c in canales]].to_numpy(dtype=float)
    if validos:
        mascara = w[[f"{PREFIJO_VALIDEZ}{c}" for c in canales]].to_numpy(dtype=bool)
        valores = np.where(mascara, valores, np.nan)
    # Una región sin canales válidos da NaN (aviso de nanmean esperado).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        media = np.nanmean(valores, axis=1) if validos else valores.mean(axis=1)
    return np.log10(np.clip(media, 1e-30, None))


def bootstrap_personas(sumas, cuentas, rng, n=N_BOOTSTRAP):
    """IC95 de sum(sumas)/sum(cuentas) remuestreando personas."""
    idx = rng.integers(len(sumas), size=(n, len(sumas)))
    reps = sumas[idx].sum(axis=1) / cuentas[idx].sum(axis=1)
    return np.percentile(reps, [2.5, 97.5])


# --- 1. Descriptivo ----------------------------------------------------------


def descriptivo(w):
    _titulo("1. DATOS Y DESCRIPTIVO (solo train)")
    por_grab = w.groupby("archivo_origen").agg(
        condicion=("condicion", "first"), persona=("persona_uid", "first"),
        ventanas=("tmin_s", "size"), dur_s=("tmin_s", "max")).reset_index()
    validos = w.drop_duplicates("archivo_origen").set_index("archivo_origen")[VALIDEZ].sum(axis=1)
    por_grab["canales_validos"] = por_grab.archivo_origen.map(validos)

    filas = []
    for cond in CONDICIONES:
        g = por_grab[por_grab.condicion == cond]
        filas.append({
            "condicion": cond, "grabaciones": len(g), "personas": g.persona.nunique(),
            "ventanas": int(g.ventanas.sum()),
            "ventanas_por_grabacion_mediana": float(g.ventanas.median()),
            "dur_util_s_mediana": float(g.dur_s.median()),
            "dur_util_s_p10": float(g.dur_s.quantile(.1)),
            "dur_util_s_p90": float(g.dur_s.quantile(.9)),
            "excluidas_menos_10_ventanas": int((g.ventanas < MIN_VENTANAS).sum()),
            "canales_excluidos_media": round(float(14 - g.canales_validos.mean()), 2),
            "pct_con_algun_excluido": round(float((g.canales_validos < 14).mean() * 100), 1),
            "sin_canales_validos": int((g.canales_validos == 0).sum()),
        })
    tabla = pd.DataFrame(filas)
    print(tabla.to_string(index=False))
    tabla.to_csv(CARPETA_TABLAS / "descriptivo_condiciones.csv", index=False)

    # Regiones vacías por grabación (con canales válidos).
    unicas = w.drop_duplicates("archivo_origen")
    vacias = {}
    for nombre, regiones in (("sondeo", REGIONES_SONDEO), ("proyecto", REGIONES)):
        for region, canales in regiones.items():
            n_val = unicas[[f"{PREFIJO_VALIDEZ}{c}" for c in canales]].sum(axis=1)
            vacias[f"{nombre}_{region}"] = int((n_val == 0).sum())
    print("\nGrabaciones con una región sin canales válidos:", vacias)

    # Ventanas por tercio.
    w_ok = w[w.n_vent >= MIN_VENTANAS]
    filas_t = []
    for cond in CONDICIONES_TERCIOS:
        g = w_ok[w_ok.condicion == cond]
        seg = np.where(g.t_rec <= g.dur / 3, 0, np.where(g.t_rec >= 2 * g.dur / 3, 1, -1))
        n = pd.crosstab(g.archivo_origen, seg)
        filas_t.append({
            "condicion": cond,
            "inicial_mediana": float(n[0].median()), "inicial_min": int(n[0].min()),
            "medio_mediana": float(n[-1].median()),
            "final_mediana": float(n[1].median()), "final_min": int(n[1].min()),
        })
    tercios = pd.DataFrame(filas_t)
    print("\nVentanas por tercio y grabación:")
    print(tercios.to_string(index=False))
    tercios.to_csv(CARPETA_TABLAS / "ventanas_por_tercio.csv", index=False)
    return {"por_condicion": filas, "regiones_vacias": vacias, "ventanas_por_tercio": filas_t}


# --- 2. Pendientes -----------------------------------------------------------


def pendientes(w):
    _titulo("2. PENDIENTES INTRA-GRABACIÓN POR MINUTO (reproduce 3.2)")
    w = w[w.n_vent >= MIN_VENTANAS]
    filas = []
    for validos in (True, False):
        for nombre, pref, reg, banda in MAGNITUDES_SONDEO:
            y = magnitud_regional(w, pref, REGIONES_SONDEO[reg], banda, validos)
            d = pd.DataFrame({"archivo": w.archivo_origen.to_numpy(), "persona": w.persona_uid.to_numpy(),
                              "cond": w.condicion.to_numpy(), "t": w.t_rec.to_numpy(), "y": y})
            d = d[np.isfinite(d.y)]
            # Pendiente MCO por grabación (equivale a polyfit de grado 1).
            d["tc"] = d.t - d.groupby("archivo").t.transform("mean")
            d["yc"] = d.y - d.groupby("archivo").y.transform("mean")
            d["txy"], d["txx"] = d.tc * d.yc, d.tc ** 2
            g = d.groupby(["archivo", "persona", "cond"])[["txy", "txx"]].sum().reset_index()
            g["pend"] = g.txy / g.txx
            for cond in CONDICIONES:
                pp = g[g.cond == cond].groupby("persona").pend.mean()
                t, p = stats.ttest_1samp(pp, 0)
                filas.append({"canales": "validos" if validos else "14", "magnitud": nombre,
                              "condicion": cond, "pendiente": pp.mean(), "t": t, "p": p,
                              "personas": len(pp), "grabaciones": int((g.cond == cond).sum())})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(CARPETA_TABLAS / "pendientes.csv", index=False)
    for canales in ("validos", "14"):
        sub = tabla[tabla.canales == canales]
        celda = sub.assign(c=sub.apply(lambda r: f"{r.pendiente:+.3f} ({r.t:+.1f})", axis=1))
        ancha = celda.pivot(index="condicion", columns="magnitud", values="c")
        print(f"\nPendiente por minuto (t), canales = {canales}:")
        print(ancha.loc[CONDICIONES, [m[0] for m in MAGNITUDES_SONDEO]].to_string())
    return tabla


# --- 3. Clasificador de tercios ---------------------------------------------


def segmentos(w, cond):
    """Media de las 56 pot_rel por (grabación, tercio), en log10, y la máscara de canales."""
    g = w[(w.condicion == cond) & (w.n_vent >= MIN_VENTANAS)].copy()
    g["seg"] = np.where(g.t_rec <= g.dur / 3, 0, np.where(g.t_rec >= 2 * g.dur / 3, 1, -1))
    g = g[g.seg >= 0]
    X = g.groupby(["archivo_origen", "persona_uid", "seg"])[REL].mean().reset_index()
    X[REL] = np.log10(X[REL].clip(lower=1e-6))
    validez = g.drop_duplicates("archivo_origen").set_index("archivo_origen")[VALIDEZ]
    # Una columna por cada pot_rel: el canal de esa columna es válido o no.
    mascara = np.repeat(validez.loc[X.archivo_origen].to_numpy(dtype=bool), len(BANDAS), axis=1)
    return X, mascara


def clasificar(X, mascara, centrar, validos):
    F = X[REL].copy()
    if centrar:
        F = F - F.groupby(X.archivo_origen).transform("mean")
        if validos:
            F = F.mask(~mascara, 0.0)
        modelo = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    else:
        if validos:
            F = F.mask(~mascara)
        modelo = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(),
                               LogisticRegression(C=0.1, max_iter=3000))
    p = cross_val_predict(modelo, F.to_numpy(), X.seg.to_numpy(), groups=X.persona_uid,
                          cv=GroupKFold(5), method="predict_proba")[:, 1]
    par = X.assign(p=p).pivot_table(index=["archivo_origen", "persona_uid"], columns="seg",
                                    values="p").dropna().reset_index()
    acierto = np.where(par[1] > par[0], 1.0, np.where(par[1] == par[0], 0.5, 0.0))
    par["acierto"] = acierto
    return {
        "auc": roc_auc_score(X.seg, p), "ba": balanced_accuracy_score(X.seg, p > .5),
        "pareado": acierto.mean(), "empates": int((par[1] == par[0]).sum()),
        "grabaciones": len(par), "personas": par.persona_uid.nunique(),
    }, par[["archivo_origen", "persona_uid", "acierto"]]


def clasificador(w):
    _titulo("3. CLASIFICADOR PRIMER TERCIO FRENTE A ÚLTIMO TERCIO (reproduce 3.3)")
    rng = np.random.default_rng(SEMILLA)
    filas, aciertos = [], {}
    for cond in CONDICIONES_TERCIOS:
        X, mascara = segmentos(w, cond)
        for centrar in (True, False):
            for validos in (True, False):
                m, par = clasificar(X, mascara, centrar, validos)
                variante = f"{'centrado' if centrar else 'sin_centrar'}+{'validos' if validos else '14'}"
                fila = {"condicion": cond, "variante": variante, **m}
                if variante == "centrado+validos":
                    aciertos[cond] = par
                    porp = par.groupby("persona_uid").acierto.agg(["sum", "size"])
                    fila["pareado_ic95"] = bootstrap_personas(
                        porp["sum"].to_numpy(), porp["size"].to_numpy(), rng).tolist()
                filas.append(fila)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(CARPETA_TABLAS / "clasificador_tercios.csv", index=False)
    for variante in tabla.variante.unique():
        sub = tabla[tabla.variante == variante]
        print(f"\nVariante {variante}:")
        print(sub[["condicion", "grabaciones", "personas", "auc", "ba", "pareado", "empates"]]
              .round(3).to_string(index=False))
    pareadas = pd.concat({c: a for c, a in aciertos.items()}, names=["condicion"]).reset_index(0)
    pareadas.to_csv(CARPETA_TABLAS / "aciertos_por_grabacion.csv", index=False)

    # Diferencia tarea − OA (variante principal), bootstrap de personas.
    print("\nDiferencia de acierto pareado tarea − OA (centrado+validos), IC95 bootstrap de personas:")
    diferencias = {}
    oa = aciertos["OA"]
    for cond in ["PASAT", "PVTB", "PVSAT"]:
        ta = aciertos[cond]
        personas = np.array(sorted(set(ta.persona_uid) | set(oa.persona_uid)))
        def sumas(df):
            s = df.groupby("persona_uid").acierto.agg(["sum", "size"]).reindex(personas, fill_value=0)
            return s["sum"].to_numpy(), s["size"].to_numpy()
        st, nt = sumas(ta)
        so, no = sumas(oa)
        idx = rng.integers(len(personas), size=(N_BOOTSTRAP, len(personas)))
        reps = st[idx].sum(1) / nt[idx].sum(1) - so[idx].sum(1) / no[idx].sum(1)
        d = ta.acierto.mean() - oa.acierto.mean()
        lo, hi = np.percentile(reps, [2.5, 97.5])
        diferencias[cond] = {"diferencia": d, "ic95": [lo, hi]}
        print(f"  {cond:6s} − OA: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return tabla, diferencias


# --- 4. Curvas por minuto ---------------------------------------------------


def curvas(w):
    _titulo("4. CURVAS POR MINUTO (centradas por grabación, canales válidos)")
    rng = np.random.default_rng(SEMILLA + 1)
    w = w[(w.n_vent >= MIN_VENTANAS) & w.condicion.isin(CONDICIONES_TERCIOS)]
    minuto = (w.tmin_s // 60).astype(int).to_numpy()
    filas = []
    for nombre, pref, banda in MAGNITUDES_CURVAS:
        for region, canales in REGIONES.items():
            y = magnitud_regional(w, pref, canales, banda, validos=True)
            d = pd.DataFrame({"archivo": w.archivo_origen.to_numpy(), "persona": w.persona_uid.to_numpy(),
                              "cond": w.condicion.to_numpy(), "minuto": minuto, "y": y})
            d = d[np.isfinite(d.y)]
            d["y"] = d.y - d.groupby("archivo").y.transform("mean")
            g = d.groupby(["cond", "persona", "archivo", "minuto"]).y.agg(["mean", "size"]).reset_index()
            g = g[g["size"] >= MIN_VENTANAS_MINUTO]
            pp = g.groupby(["cond", "minuto", "persona"])["mean"].mean().reset_index()
            for (cond, mi), sub in pp.groupby(["cond", "minuto"]):
                v = sub["mean"].to_numpy()
                reps = v[rng.integers(len(v), size=(N_BOOTSTRAP, len(v)))].mean(1)
                lo, hi = np.percentile(reps, [2.5, 97.5])
                filas.append({"magnitud": nombre, "region": region, "condicion": cond, "minuto": mi + 1,
                              "media": v.mean(), "ic_inf": lo, "ic_sup": hi, "personas": len(v)})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(CARPETA_TABLAS / "curvas_por_minuto.csv", index=False)
    print(tabla.assign(c=tabla.apply(lambda r: f"{r.media:+.3f}", axis=1))
          .pivot_table(index=["magnitud", "region", "condicion"], columns="minuto", values="c", aggfunc="first")
          .to_string())

    # En la figura no se dibujan los minutos de cola (grabaciones más largas que
    # la mediana): con menos de la mitad de las personas de la condición son
    # ruido. Siguen en la tabla, con su número de personas.
    maximo = tabla.groupby("condicion").personas.transform("max")
    dibujar = tabla[tabla.personas >= maximo / 2]
    fig, ejes = plt.subplots(len(MAGNITUDES_CURVAS), len(REGIONES), figsize=(13, 8.5), sharex=True)
    for i, (nombre, _, _) in enumerate(MAGNITUDES_CURVAS):
        for j, region in enumerate(REGIONES):
            ax = ejes[i, j]
            for cond in CONDICIONES_TERCIOS:
                s = dibujar[(dibujar.magnitud == nombre) & (dibujar.region == region) & (dibujar.condicion == cond)]
                es_oa = cond == "OA"
                ax.fill_between(s.minuto, s.ic_inf, s.ic_sup, color=COLORES[cond], alpha=.15, lw=0)
                ax.plot(s.minuto, s.media, marker="o", ms=4, lw=1.8, color=COLORES[cond],
                        ls="--" if es_oa else "-", label=cond)
            ax.axhline(0, color=REJILLA, lw=1, zorder=0)
            ax.grid(color=REJILLA, lw=.6)
            ax.set_xticks([1, 2, 3])
            for lado in ("top", "right"):
                ax.spines[lado].set_visible(False)
            ax.tick_params(colors=TINTA_2, labelsize=8)
            if i == 0:
                ax.set_title(region, color=TINTA, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{nombre}\nΔ log10 (centrado)", color=TINTA_2, fontsize=9)
            if i == len(MAGNITUDES_CURVAS) - 1:
                ax.set_xlabel("minuto de la grabación", color=TINTA_2, fontsize=9)
    ejes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Cambio dentro de la grabación por minuto (media de personas, IC95 bootstrap) — train",
                 color=TINTA, fontsize=11)
    fig.tight_layout()
    fig.savefig(CARPETA_FIGURAS / "curvas_por_minuto.png", dpi=150)
    plt.close(fig)
    return tabla


def figura_pareado(tabla):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    sub = tabla[tabla.variante == "centrado+validos"].set_index("condicion").loc[CONDICIONES_TERCIOS]
    y = np.arange(len(sub))[::-1]
    for yi, (cond, r) in zip(y, sub.iterrows()):
        lo, hi = r.pareado_ic95
        ax.plot([lo, hi], [yi, yi], color=COLORES[cond], lw=2.5, solid_capstyle="round")
        ax.plot(r.pareado, yi, "o", color=COLORES[cond], ms=7)
        ax.text(hi + .01, yi, f"{r.pareado:.3f}", va="center", fontsize=9, color=TINTA)
    ax.axvline(.5, color=TINTA_2, lw=1, ls=":")
    ax.axvline(UMBRAL_CONTINUIDAD, color=REJILLA, lw=1.2)
    ax.set_yticks(y, sub.index)
    ax.set_xlim(.45, 1)
    ax.set_xlabel("acierto pareado (último tercio > primero)", color=TINTA_2)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.set_title("Clasificador de tercios, centrado + canales válidos — train, CV por persona",
                 fontsize=9.5, color=TINTA)
    fig.tight_layout()
    fig.savefig(CARPETA_FIGURAS / "acierto_pareado.png", dpi=150)
    plt.close(fig)


def main():
    CARPETA_TABLAS.mkdir(exist_ok=True)
    CARPETA_FIGURAS.mkdir(exist_ok=True)

    _titulo("0. CARGA")
    w, incidencia, n_archivos_train = cargar()
    print(f"Archivos etiquetados de train (matriz_rf): {n_archivos_train}; con ventanas: "
          f"{w.archivo_origen.nunique()}; ventanas: {len(w)}; personas: {w.persona_uid.nunique()}")
    print("\nIncidencia 3.5 — grabaciones etiquetadas cuya split difiere (features_all → matriz_rf):")
    print(incidencia.to_string(index=False))

    res_desc = descriptivo(w)
    tabla_pend = pendientes(w)
    tabla_clf, diferencias = clasificador(w)
    figura_pareado(tabla_clf)
    tabla_curvas = curvas(w)

    _titulo("5. CRITERIO DE CONTINUIDAD")
    principal = tabla_clf[tabla_clf.variante == "centrado+validos"].set_index("condicion").pareado
    cumple = {c: bool(principal[c] >= UMBRAL_CONTINUIDAD and principal[c] > principal["OA"])
              for c in TAREAS_CRITERIO}
    for c in TAREAS_CRITERIO:
        print(f"  {c}: pareado {principal[c]:.3f} (≥ {UMBRAL_CONTINUIDAD} y > OA {principal['OA']:.3f}): "
              f"{'sí' if cumple[c] else 'NO'}")
    se_cumple = all(cumple.values())
    print(f"  Criterio de continuidad: {'SE CUMPLE' if se_cumple else 'NO SE CUMPLE'}")

    resultados = {
        "fecha": "2026-10-27",
        "particion": "solo train (split de matriz_rf), etiquetadas",
        "sha256": {p.name: sha256(p) for p in (RUTA_FEATURES, RUTA_MATRIZ, RUTA_LOG)},
        "parametros": {"semilla": SEMILLA, "n_bootstrap": N_BOOTSTRAP, "min_ventanas": MIN_VENTANAS,
                       "min_ventanas_minuto": MIN_VENTANAS_MINUTO, "umbral_continuidad": UMBRAL_CONTINUIDAD},
        "datos": {"archivos_train": n_archivos_train, "grabaciones": int(w.archivo_origen.nunique()),
                  "ventanas": int(len(w)), "personas": int(w.persona_uid.nunique())},
        "incidencia_particion": incidencia.to_dict(orient="records"),
        "descriptivo": res_desc,
        "pendientes": tabla_pend.round(5).to_dict(orient="records"),
        "clasificador": [{k: (_redondear(v) if isinstance(v, float) else
                              [_redondear(x) for x in v] if isinstance(v, list) else v)
                          for k, v in fila.items()} for fila in tabla_clf.to_dict(orient="records")],
        "diferencia_con_oa": {c: {"diferencia": _redondear(d["diferencia"]),
                                  "ic95": [_redondear(x) for x in d["ic95"]]} for c, d in diferencias.items()},
        "curvas_por_minuto": tabla_curvas.round(5).to_dict(orient="records"),
        "criterio_continuidad": {"por_tarea": cumple, "se_cumple": se_cumple},
        "miradas": {"val": 0, "test": 0},
    }
    with open(CARPETA / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
