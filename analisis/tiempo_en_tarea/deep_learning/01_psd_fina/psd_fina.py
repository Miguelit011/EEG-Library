"""
Tiempo en tarea — deep learning, paso 1: espectro fino (PSD de 1 Hz) frente a las
cuatro bandas, con la logística del paso 3.

Filtro barato antes de la CNN: si la PSD de 1 Hz (14 × 30) no ordena las
ventanas en el tiempo mejor que las bandas (14 × 4), la vía espectral por la
que una CNN pequeña tiene más opciones de ganar está vacía.

**Solo `train`.** Plan, contrastes y criterio de decisión: `INFORME.md`,
escritos antes de ejecutar. Reutiliza `03_clasificador/comun.py` y
`medidas.py` sin modificarlos.

Ejecutar (desde la raíz del repositorio):
    .venv/Scripts/python.exe analisis/tiempo_en_tarea/deep_learning/01_psd_fina/psd_fina.py > analisis/tiempo_en_tarea/deep_learning/01_psd_fina/ejecucion.log
"""

import contextlib
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parents[1] / "03_clasificador"))
import comun  # noqa: E402
import medidas as md  # noqa: E402
from comun import FAMILIA, SEMILLAS, d1  # noqa: E402

from eeg_library.procesado.bands import BANDAS_DEFECTO, _mascaras_de_banda  # noqa: E402
from eeg_library.procesado.spectral import calcular_psd  # noqa: E402

RUTA_VENTANAS = d1.EEG_DATA / "ventanas"
CONDICIONES = FAMILIA + ["OA", "OC"]
REJILLA_C = [0.001, 0.003, 0.01, 0.03, 0.1]
FRECUENCIAS = list(range(1, 31))
FINA = [f"psd_{c}_{f}" for c in d1.CANALES_EPOC for f in FRECUENCIAS]
UMBRAL_CORRELACION = 0.999
MAX_DESALINEADOS = 0.01
UMBRAL_MARGEN = 0.02
SFREQ = 256.0

mne.set_log_level("ERROR")


# --- Espectro fino desde las ventanas crudas ---------------------------------


def psd_relativa(ventanas):
    """PSD de 1–30 Hz con `calcular_psd` (Fase 8), dividida por la potencia total del canal y la ventana."""
    info = mne.create_info(d1.CANALES_EPOC, SFREQ, ch_types="eeg")
    epochs = mne.EpochsArray(ventanas.astype(np.float64), info, verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        spectrum, _ = calcular_psd(epochs)
    freqs = spectrum.freqs
    assert np.allclose(freqs, FRECUENCIAS), f"Frecuencias inesperadas: {freqs}"
    datos = spectrum.get_data()  # (n, 14, 30)
    return datos / datos.sum(axis=2, keepdims=True), freqs


def anadir_espectro(w):
    """Añade a `w` las 420 columnas de PSD relativa, leyendo un `.npy` por archivo; devuelve integridad."""
    archivos = sorted(w.archivo_origen.unique())
    bloques, desalineados, hashes = [], [], []
    mascaras = None
    for i, (a, g) in enumerate(w.groupby("archivo_origen", sort=True)):
        ruta = RUTA_VENTANAS / a.replace(".edf", ".npy")
        hashes.append(f"{ruta.name}:{d1.sha256(ruta)}")
        x = np.load(ruta)
        if x.shape[0] != g.n_vent.iloc[0] or x.shape[1:] != (14, 1024):
            desalineados.append(a)
            continue
        rel, freqs = psd_relativa(x[g.indice_ventana.to_numpy()])
        if mascaras is None:
            mascaras = _mascaras_de_banda(BANDAS_DEFECTO, freqs, float(freqs.max()))
        bloques.append(pd.DataFrame(rel.reshape(len(g), -1).astype(np.float32), index=g.index, columns=FINA))
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(archivos)} archivos", flush=True)
    F = pd.concat(bloques)
    w = w.loc[F.index].join(F)
    combinado = hashlib.sha256("\n".join(hashes).encode()).hexdigest()
    return w, desalineados, mascaras, combinado, len(archivos)


def reproduccion_de_bandas(w, mascaras):
    """Correlación, por columna, entre la `pot_rel` recalculada desde la PSD fina y la de `features_all`."""
    filas = []
    F = w[FINA].to_numpy(np.float64).reshape(len(w), 14, len(FRECUENCIAS))
    for b in d1.BANDAS:
        recalculada = F[:, :, mascaras[b]].sum(axis=2)
        for j, c in enumerate(d1.CANALES_EPOC):
            original = w[f"pot_rel_{c}_{b}"].to_numpy(float)
            filas.append({"columna": f"pot_rel_{c}_{b}", "r": float(np.corrcoef(recalculada[:, j], original)[0, 1]),
                          "dif_abs_max": float(np.abs(recalculada[:, j] - original).max())})
    return pd.DataFrame(filas)


# --- Features de la representación fina (en paralelo a comun y medidas) -------


def tercios_fina(w, cond):
    """Como `comun.tercios`, con la PSD fina: media lineal por tercio, después log10."""
    g = w[w.condicion == cond]
    seg = np.where(g.t_rec <= g.dur / 3, 0, np.where(g.t_rec >= 2 * g.dur / 3, 1, -1))
    g = g[seg >= 0].assign(seg=seg[seg >= 0])
    claves = ["archivo_origen", "persona_uid", "seg"]
    S = g.groupby(claves)[FINA].mean().reset_index()
    S[FINA] = np.log10(S[FINA].clip(lower=1e-6)).to_numpy()
    validez = g.drop_duplicates("archivo_origen").set_index("archivo_origen")[d1.VALIDEZ]
    mascara = np.repeat(validez.loc[S.archivo_origen].to_numpy(dtype=bool), len(FRECUENCIAS), axis=1)
    return S, mascara


def matriz_fina(S, mascara):
    """Como `comun.matriz`: centrado por grabación y canales excluidos a 0."""
    F = S[FINA] - S[FINA].groupby(S.archivo_origen).transform("mean")
    return F.mask(~mascara, 0.0).fillna(0.0).to_numpy(float)


def ventanas_fina(w, cond):
    """Como `medidas.ventanas`: log10 por ventana, centrado por grabación, canales excluidos a 0."""
    g = w[w.condicion == cond].reset_index(drop=True)
    X = np.log10(np.clip(g[FINA].to_numpy(float), 1e-6, None))
    X = X - pd.DataFrame(X).groupby(g.archivo_origen.to_numpy()).transform("mean").to_numpy()
    mascara = np.repeat(g[d1.VALIDEZ].to_numpy(dtype=bool), len(FRECUENCIAS), axis=1)
    return g[["archivo_origen", "persona_uid", "t_rec"]], np.where(mascara, X, 0.0)


def datos(w, cond, rep):
    """Filas de entreno (tercios), X de entreno, filas de ventana y X de ventana para una representación."""
    if rep == "B":
        S, masc = comun.tercios(w, cond)
        X = comun.matriz(S, masc, "rel")
        filas, Xv = md.ventanas(w, cond)
    else:
        S, masc = tercios_fina(w, cond)
        X = matriz_fina(S, masc)
        filas, Xv = ventanas_fina(w, cond)
    return S, X, filas, Xv


# --- Evaluación --------------------------------------------------------------


def evaluar(w, cond, rep, pliegues):
    """Para cada C: ρ por grabación y acierto pareado, con la puntuación media de las 20 particiones."""
    S, X, filas, Xv = datos(w, cond, rep)
    y = S.seg.to_numpy()
    punt = {c: np.zeros((len(SEMILLAS), len(filas))) for c in REJILLA_C}
    prob = {c: np.zeros((len(SEMILLAS), len(S))) for c in REJILLA_C}
    coefs = {c: [] for c in REJILLA_C}
    for i, s in enumerate(SEMILLAS):
        fold = S.persona_uid.map(pliegues[s]).to_numpy()
        fold_v = filas.persona_uid.map(pliegues[s]).to_numpy()
        for k in range(comun.N_PLIEGUES):
            for c in REJILLA_C:
                m = comun.modelo(f"lr_{c}").fit(X[fold != k], y[fold != k])
                punt[c][i, fold_v == k] = m.decision_function(Xv[fold_v == k])
                prob[c][i, fold == k] = m.predict_proba(X[fold == k])[:, 1]
                if i == 0:
                    coefs[c].append(m[-1].coef_.ravel())
    salida = {}
    for c in REJILLA_C:
        salida[c] = {"rho": md.rho_por_grabacion(filas, punt[c].mean(axis=0)),
                     "pareado": comun.aciertos_por_grabacion(S, prob[c].mean(axis=0)),
                     "coef": np.mean(coefs[c], axis=0)}
    return salida


def ic(reps):
    return [float(v) for v in np.percentile(reps, [2.5, 97.5])]


def diferencia(a, b):
    """Diferencia pareada por grabación a − b (columnas `acierto`)."""
    d = a.merge(b, on=["archivo_origen", "persona_uid"], suffixes=("_a", "_b"))
    return d.assign(acierto=d.acierto_a - d.acierto_b)[["archivo_origen", "persona_uid", "acierto"]]


def main():
    t0 = time.time()
    d1._titulo("0. CARGA (solo train)")
    w = comun.cargar({"train"})
    assert set(w.split) == {"train"}
    w = w[w.condicion.isin(CONDICIONES)].copy()
    print(f"Grabaciones: {w.archivo_origen.nunique()}  personas: {w.persona_uid.nunique()}  ventanas: {len(w)}")

    d1._titulo("1. ESPECTRO FINO DESDE LAS VENTANAS CRUDAS E INTEGRIDAD")
    w, desalineados, mascaras, hash_npy, n_archivos = anadir_espectro(w)
    frac = len(desalineados) / n_archivos
    print(f"Archivos leídos: {n_archivos}. Desalineados (excluidos): {len(desalineados)} ({frac:.2%})")
    assert frac <= MAX_DESALINEADOS, "Más del 1 % de archivos desalineados: se para"
    rep = reproduccion_de_bandas(w, mascaras)
    print(f"Reproducción de pot_rel: r mínima {rep.r.min():.6f}, diferencia absoluta máxima {rep.dif_abs_max.max():.2e}")
    (CARPETA / "tablas").mkdir(exist_ok=True)
    rep.to_csv(CARPETA / "tablas" / "integridad.csv", index=False)
    assert rep.r.min() >= UMBRAL_CORRELACION, "La PSD fina no reproduce las bandas: se para"
    print(f"Tiempo: {time.time() - t0:.0f} s")

    vers = comun.versiones(w)
    personas = sorted(w.persona_uid.unique())
    pliegues = {s: comun.pliegues_por_persona(personas, s) for s in SEMILLAS}

    d1._titulo("2. REJILLA (20 particiones; ρ media por grabación)")
    res, filas = {}, []
    for v in ["completa", "igual_tiempo"]:
        for c in CONDICIONES:
            for r in ["B", "F"]:
                res[(v, c, r)] = evaluar(vers[v], c, r, pliegues)
                for C in REJILLA_C:
                    x = res[(v, c, r)][C]
                    filas.append({"version": v, "condicion": c, "representacion": r, "C": C,
                                  "rho": float(x["rho"].acierto.mean()),
                                  "pareado": float(x["pareado"].acierto.mean()),
                                  "grabaciones": len(x["rho"])})
            print(f"  {v:12s} {c:5s} " + "  ".join(
                f"{r}: " + " ".join(f"{f['rho']:.3f}" for f in filas[-10:] if f["representacion"] == r)
                for r in ["B", "F"]) + f"   ({time.time() - t0:.0f} s)", flush=True)
    rejilla = pd.DataFrame(filas)
    rejilla.to_csv(CARPETA / "tablas" / "rejilla.csv", index=False)

    d1._titulo("3. MEJOR C DE CADA REPRESENTACIÓN (mayor ρ de PASAT + PVTB, grabación completa)")
    mejor = {}
    for r in ["B", "F"]:
        fam = {C: pd.concat([res[("completa", c, r)][C]["rho"] for c in FAMILIA]).acierto.mean() for C in REJILLA_C}
        mejor[r] = max(fam, key=fam.get)
        print(f"  {r}: " + "  ".join(f"C={C}: {fam[C]:.4f}" for C in REJILLA_C) + f"  -> C = {mejor[r]}")

    def rho(v, c, r):
        return res[(v, c, r)][mejor[r]]["rho"]

    d1._titulo("4. CONTRASTES (F − B, pareado por grabación; IC95 por bootstrap de personas)")
    rng = np.random.default_rng(d1.SEMILLA)
    contrastes = []

    def anotar(id_, que, version, dif, b=None):
        reps = md.bootstrap(dif, b, rng=rng)
        valor = dif.acierto.mean() - (b.acierto.mean() if b is not None else 0.0)
        contrastes.append({"id": id_, "que": que, "version": version, "valor": float(valor), "ic95": ic(reps),
                           "grabaciones": len(dif), "personas": int(dif.persona_uid.nunique())})
        print(f"  {id_:3s} {que:30s} {version:12s} {valor:+.4f} [{ic(reps)[0]:+.4f}, {ic(reps)[1]:+.4f}]")

    for v in ["completa", "igual_tiempo"]:
        fam_d = pd.concat([diferencia(rho(v, c, "F"), rho(v, c, "B")) for c in FAMILIA], ignore_index=True)
        anotar("D1" if v == "completa" else "D1'", "Δρ PASAT + PVTB", v, fam_d)
        for c in FAMILIA:
            anotar("D2", f"Δρ {c}", v, diferencia(rho(v, c, "F"), rho(v, c, "B")))
    for c in ["OA", "OC"]:
        anotar("D3", f"Δρ {c}", "completa", diferencia(rho("completa", c, "F"), rho("completa", c, "B")))
    v = "igual_tiempo"
    fam_d = pd.concat([diferencia(rho(v, c, "F"), rho(v, c, "B")) for c in FAMILIA], ignore_index=True)
    anotar("D4", "Δ H1 (tareas − OA), F − B", v, fam_d, diferencia(rho(v, "OA", "F"), rho(v, "OA", "B")))
    tabla = pd.DataFrame(contrastes)
    tabla.to_csv(CARPETA / "tablas" / "contrastes.csv", index=False)

    principal = contrastes[0]
    if principal["valor"] >= UMBRAL_MARGEN and principal["ic95"][0] > 0:
        lectura = "HAY MARGEN"
    elif principal["ic95"][0] > 0:
        lectura = "MEJORA REAL PERO PEQUEÑA"
    else:
        lectura = "SIN MARGEN"
    print(f"\nCriterio (D1): Δρ = {principal['valor']:+.4f}, "
          f"IC95 [{principal['ic95'][0]:+.4f}, {principal['ic95'][1]:+.4f}] -> {lectura}")

    # Descriptivos con la mejor C de cada una: ρ y acierto pareado por condición.
    desc = []
    for v in ["completa", "igual_tiempo"]:
        for c in CONDICIONES:
            for r in ["B", "F"]:
                x = res[(v, c, r)][mejor[r]]
                desc.append({"version": v, "condicion": c, "representacion": r, "C": mejor[r],
                             "rho": float(x["rho"].acierto.mean()),
                             "rho_ic95": ic(md.bootstrap(x["rho"], rng=rng)),
                             "pareado": float(x["pareado"].acierto.mean())})
    desc = pd.DataFrame(desc)
    desc.to_csv(CARPETA / "tablas" / "descriptivos.csv", index=False)
    print()
    print(desc.pivot_table(index=["version", "condicion"], columns="representacion",
                           values=["rho", "pareado"]).round(3).to_string())

    # Perfil de frecuencias: coeficiente medio por Hz (sobre canales y los 5 pliegues de la semilla 0).
    perfil = []
    for c in FAMILIA + ["OA"]:
        coef = res[("completa", c, "F")][mejor["F"]]["coef"].reshape(14, len(FRECUENCIAS))
        for j, f in enumerate(FRECUENCIAS):
            perfil.append({"condicion": c, "hz": f, "coef_medio": float(coef[:, j].mean()),
                           "coef_abs_medio": float(np.abs(coef[:, j]).mean())})
    perfil = pd.DataFrame(perfil)
    perfil.to_csv(CARPETA / "tablas" / "perfil_frecuencias.csv", index=False)
    figuras(tabla, perfil)

    with open(CARPETA / "resultados.json", "w", encoding="utf-8") as f:
        json.dump({
            "fecha": time.strftime("%Y-%m-%d"),
            "particion": "train",
            "mejor_C": mejor,
            "criterio": {"umbral": UMBRAL_MARGEN, "D1": principal, "lectura": lectura},
            "contrastes": contrastes,
            "integridad": {"archivos": n_archivos, "desalineados": desalineados,
                           "r_minima": float(rep.r.min()), "dif_abs_max": float(rep.dif_abs_max.max())},
            "sha256": {p.name: d1.sha256(p) for p in (d1.RUTA_FEATURES, d1.RUTA_MATRIZ, d1.RUTA_LOG)}
                  | {"ventanas_npy_combinado": hash_npy},
            "miradas": {"val": 2, "test": "gastado (2026-10-27); no se carga"},
            "segundos": round(time.time() - t0),
        }, f, ensure_ascii=False, indent=1)
    print(f"\nTiempo total: {time.time() - t0:.0f} s")


def figuras(tabla, perfil):
    (CARPETA / "figuras").mkdir(exist_ok=True)
    t = tabla[tabla.id.isin(["D1", "D2", "D3"])].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    y = np.arange(len(t))[::-1]
    err = np.array([[r.valor - r.ic95[0], r.ic95[1] - r.valor] for r in t.itertuples()]).T
    ax.errorbar(t.valor, y, xerr=err, fmt="o", color="#2a78d6", ecolor="#8a8984", capsize=3)
    ax.axvline(0, color="#8a8984", lw=0.8)
    ax.axvline(UMBRAL_MARGEN, color="#eb6834", lw=0.8, ls="--", label=f"umbral +{UMBRAL_MARGEN}")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.que} ({r.version})" for r in t.itertuples()], fontsize=8)
    ax.set_xlabel("Δρ (espectro fino − bandas), IC95")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CARPETA / "figuras" / "diferencias.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for c in FAMILIA + ["OA"]:
        p = perfil[perfil.condicion == c]
        ax.plot(p.hz, p.coef_medio, marker="o", ms=3, lw=1.2, color=d1.COLORES[c], label=c)
    ax.axhline(0, color="#8a8984", lw=0.8)
    for borde in (4, 8, 13):
        ax.axvline(borde, color="#d0d0d0", lw=0.6, ls=":")
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Coeficiente medio (último tercio +)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CARPETA / "figuras" / "perfil_frecuencias.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
