"""
Tiempo en tarea — deep learning, paso 3 (piloto de ShallowConvNet), parte 2 de 2: evaluar.

Calcula, en las personas de evaluación del pliegue 0, `rho_ventanas` y el AUC
por ventana de cada ejecución de ShallowConvNet, las diferencias pareadas con
la logística F (la principal), la logística B y EEGNet (las 3 semillas del
paso 2), la regla de decisión del `INFORME.md`, los requisitos técnicos y las
figuras. Reutiliza `rho` e `ic` de `02_piloto_cnn/evaluar.py` y `medidas.py`
sin modificarlos, en el entorno validado.

Ejecutar (desde la raíz del repositorio):
    .venv/Scripts/python.exe analisis/tiempo_en_tarea/deep_learning/03_piloto_shallow/evaluar.py > analisis/tiempo_en_tarea/deep_learning/03_piloto_shallow/evaluar.log
"""

import hashlib
import importlib.util
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

CARPETA = Path(__file__).resolve().parent
PASO2 = CARPETA.parent / "02_piloto_cnn"
_spec = importlib.util.spec_from_file_location("piloto2_evaluar", PASO2 / "evaluar.py")
p2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2)
from psd_fina import FAMILIA, d1, md  # noqa: E402  (ruta añadida por el módulo del paso 2)

CACHE = p2.CACHE
SALIDA = CACHE / "ejecuciones_shallow"
MARGEN = 0.01
MAX_EPOCAS = 60
FS = 128
TOLERANCIA = 1e-6  # para reproducir las cifras del paso 2 con la misma caché


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 24), b""):
            h.update(bloque)
    return h.hexdigest()


def media_semillas(rhos, nombres):
    """ρ por grabación promediada sobre las semillas (como la regla del paso 2)."""
    base = rhos[nombres[0]][["archivo_origen", "persona_uid", "condicion"]].copy()
    base["acierto"] = np.mean([rhos[n].set_index("archivo_origen").acierto.loc[base.archivo_origen].to_numpy()
                               for n in nombres], axis=0)
    return base


def main():
    v = pd.read_parquet(CACHE / "ventanas.parquet")
    v = v[v.papel == "evaluacion"].reset_index(drop=True)
    with open(SALIDA / "entrenamiento.json", encoding="utf-8") as f:
        ent = json.load(f)
    with open(CACHE / "entrenamiento.json", encoding="utf-8") as f:
        ent_eegnet = json.load(f)
    with open(PASO2 / "resultados.json", encoding="utf-8") as f:
        res2 = json.load(f)
    rng = np.random.default_rng(d1.SEMILLA)
    terc = v.seg.to_numpy() >= 0

    d1._titulo("1. EJECUCIONES Y REFERENCIAS (personas de evaluación del pliegue 0)")
    print(f"Personas: {v.persona_uid.nunique()}  grabaciones: {v.archivo_origen.nunique()}  ventanas: {len(v)}")
    modelos, filas, curvas = {}, [], []
    for nombre in ent["ejecuciones"]:
        with open(SALIDA / nombre / "resumen.json", encoding="utf-8") as f:
            r = json.load(f)
        punt = np.load(SALIDA / nombre / "puntuacion_evaluacion.npy")
        assert len(punt) == len(v)
        modelos[f"shallow_{nombre}"] = punt
        curvas += [dict(c, ejecucion=f"shallow_{nombre}") for c in r["curva"]]
        filas.append({"nombre": f"shallow_{nombre}", **{k: r[k] for k in (
            "lr", "semilla", "epocas", "mejor_epoca", "mejor_perdida_interna", "auc_interna_mejor",
            "segundos_por_epoca", "nan")}})
    lr2 = ent_eegnet["mejor_lr"]
    eegnet = [n for n in ent_eegnet["ejecuciones"] if n.startswith(f"lr{lr2:g}_")]
    for n in eegnet:
        punt = np.load(CACHE / "ejecuciones" / n / "puntuacion_evaluacion.npy")
        assert len(punt) == len(v)
        modelos[f"eegnet_{n}"] = punt
        filas.append({"nombre": f"eegnet_{n}"})
    for ref in ["F", "B"]:
        modelos[f"logistica_{ref}"] = v[f"punt_{ref}"].to_numpy()
        filas.append({"nombre": f"logistica_{ref}"})
    tabla = pd.DataFrame(filas).set_index("nombre")
    rhos = {}
    for nombre, punt in modelos.items():
        rhos[nombre] = p2.rho(v, punt)
        for c in FAMILIA:
            tabla.loc[nombre, f"rho_{c}"] = rhos[nombre][rhos[nombre].condicion == c].acierto.mean()
        tabla.loc[nombre, "rho_familia"] = rhos[nombre].acierto.mean()
        tabla.loc[nombre, "auc_ventanas"] = roc_auc_score(v.seg[terc], punt[terc])
    print(tabla.round(4).to_string())
    (CARPETA / "tablas").mkdir(exist_ok=True)
    tabla.to_csv(CARPETA / "tablas" / "ejecuciones.csv")
    pd.DataFrame(curvas).to_csv(CARPETA / "tablas" / "curvas.csv", index=False)

    d1._titulo("2. REGLA DE DECISIÓN (media de las 3 semillas de la mejor lr)")
    lr = ent["mejor_lr"]
    semillas = [f"shallow_{n}" for n in ent["ejecuciones"] if n.startswith(f"lr{lr:g}_")]
    shallow = media_semillas(rhos, semillas)
    eeg = media_semillas(rhos, [f"eegnet_{n}" for n in eegnet])
    rho_s = float(shallow.acierto.mean())
    rho_e = float(eeg.acierto.mean())
    rho_f, rho_b = float(tabla.loc["logistica_F", "rho_familia"]), float(tabla.loc["logistica_B", "rho_familia"])
    desv_semillas = float(tabla.loc[semillas, "rho_familia"].std(ddof=1))
    dif = {}
    for ref, otro in [("F", rhos["logistica_F"]), ("B", rhos["logistica_B"]), ("EEGNet", eeg)]:
        d = shallow.merge(otro[["archivo_origen", "persona_uid", "acierto"]], on=["archivo_origen", "persona_uid"],
                          suffixes=("", "_ref"))
        d = d.assign(acierto=d.acierto - d.acierto_ref)
        reps = md.bootstrap(d, rng=rng)
        dif[ref] = {"valor": float(d.acierto.mean()), "ic95": p2.ic(reps)}
        print(f"  ShallowConvNet − {ref}: {dif[ref]['valor']:+.4f} [{dif[ref]['ic95'][0]:+.4f}, "
              f"{dif[ref]['ic95'][1]:+.4f}]")
    print(f"  ρ ShallowConvNet (media de {len(semillas)} semillas, lr = {lr:g}): {rho_s:.4f}  "
          f"(desviación entre semillas {desv_semillas:.4f})")
    print(f"  ρ logística F: {rho_f:.4f}   ρ logística B: {rho_b:.4f}   ρ EEGNet (paso 2): {rho_e:.4f}")
    print(f"  Umbrales: ρ_F − {MARGEN} = {rho_f - MARGEN:.4f}   ρ_B − {MARGEN} = {rho_b - MARGEN:.4f}")
    if rho_s >= rho_f - MARGEN:
        decision = "LLEGA AL NIVEL DE LA LOGÍSTICA F: paso 4"
    elif rho_s >= rho_b - MARGEN:
        decision = "IGUALA A LAS BANDAS PERO NO AL ESPECTRO FINO: se valora"
    else:
        decision = "NO IGUALA A LAS BANDAS: parar"
    print(f"\n  Decisión: {decision}")

    d1._titulo("3. REQUISITOS TÉCNICOS")
    eje = tabla.loc[[f"shallow_{n}" for n in ent["ejecuciones"]]]
    curv = pd.DataFrame(curvas)
    baja = {n: bool(g.perdida_interna.min() < g.perdida_interna.iloc[0]) for n, g in curv.groupby("ejecucion")}
    requisitos = {
        "perdida_interna_baja": all(baja.values()),
        "mejor_epoca_ni_1_ni_max": bool(((eje.mejor_epoca > 1) & (eje.mejor_epoca < MAX_EPOCAS)).all()),
        "sin_nan": bool(not eje.nan.astype(bool).any()),
        "reproduce_F_del_paso_2": bool(abs(rho_f - res2["rho_F"]) < TOLERANCIA),
        "reproduce_B_del_paso_2": bool(abs(rho_b - res2["rho_B"]) < TOLERANCIA),
        "reproduce_EEGNet_del_paso_2": bool(abs(rho_e - res2["rho_cnn"]) < TOLERANCIA),
    }
    for k, ok in requisitos.items():
        print(f"  {k}: {'sí' if ok else 'NO'}")

    d1._titulo("4. FIGURAS Y HUELLAS DE LA CACHÉ")
    figuras(curv, tabla)
    llega = rho_s >= rho_f - MARGEN
    perfil = filtros(lr, ent["ejecuciones"]) if llega else None
    if not llega:
        print("  No llega a F: no se calcula la respuesta en frecuencia de los filtros (según el plan).")
    huellas = {a: sha256(CACHE / a) for a in ["X.npy", "mascara.npy", "ventanas.parquet"]}
    for a, h in huellas.items():
        print(f"  SHA-256 {a}: {h}")
    with open(CARPETA / "resultados.json", "w", encoding="utf-8") as f:
        json.dump({"fecha": time.strftime("%Y-%m-%d"), "particion": "train (pliegue 0 de la semilla 0)",
                   "mejor_lr": lr, "rho_shallow": rho_s, "rho_F": rho_f, "rho_B": rho_b, "rho_eegnet": rho_e,
                   "desviacion_entre_semillas": desv_semillas, "diferencias": dif, "decision": decision,
                   "requisitos": requisitos, "sha256_cache": huellas, "entrenamiento": ent,
                   "perfil_filtros": perfil, "ejecuciones": tabla.reset_index().to_dict(orient="records")},
                  f, ensure_ascii=False, indent=1, default=float)


def figuras(curv, tabla):
    (CARPETA / "figuras").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for n, g in curv.groupby("ejecucion", sort=False):
        ax.plot(g.epoca, g.perdida_interna, lw=1.2, label=n.removeprefix("shallow_"))
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida en el conjunto interno")
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CARPETA / "figuras" / "curvas.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    t = tabla.rho_familia
    colores = ["#2a78d6" if n.startswith("shallow") else "#e08a2b" if n.startswith("eegnet") else "#8a8984"
               for n in t.index]
    ax.barh(range(len(t))[::-1], t.to_numpy(), color=colores)
    ax.set_yticks(range(len(t))[::-1])
    ax.set_yticklabels(t.index, fontsize=8)
    ax.set_xlabel("ρ media por grabación, PASAT + PVTB (pliegue 0)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CARPETA / "figuras" / "rho_por_modelo.png", dpi=150)
    plt.close(fig)


def filtros(lr, ejecuciones):
    """Respuesta en frecuencia (normalizada) de los 40 filtros temporales de las semillas de la mejor lr.

    Solo el mapa de la señal (el de la máscara es constante en el tiempo). Descriptivo: el peso de cada
    filtro en la decisión depende también de las capas siguientes.
    """
    resp = []
    for n in [e for e in ejecuciones if e.startswith(f"lr{lr:g}_")]:
        w = np.load(SALIDA / n / "filtros_temporales.npy")[:, 0, 0, :]
        a = np.abs(np.fft.rfft(w, n=4 * FS, axis=1))
        resp.append(a / a.max(axis=1, keepdims=True))
    resp = np.concatenate(resp)
    f = np.fft.rfftfreq(4 * FS, 1 / FS)
    pico = f[resp.argmax(axis=1)]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2), gridspec_kw={"width_ratios": [2, 1]})
    sel = f <= 40
    ax[0].plot(f[sel], resp[:, sel].T, color="#2a78d6", lw=0.5, alpha=0.25)
    ax[0].plot(f[sel], resp[:, sel].mean(0), color="#1b1b1b", lw=1.5)
    ax[0].set_xlabel("Frecuencia (Hz)")
    ax[0].set_ylabel("Ganancia normalizada")
    ax[1].hist(pico, bins=np.arange(0, 41, 1), color="#2a78d6")
    ax[1].set_xlabel("Frecuencia del pico (Hz)")
    ax[1].set_ylabel("Filtros")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CARPETA / "figuras" / "filtros.png", dpi=150)
    plt.close(fig)
    bandas = {"<4": (0, 4), "4-8": (4, 8), "8-14": (8, 14), "14-20": (14, 20), "20-30": (20, 30), ">=30": (30, 65)}
    reparto = {b: int(((pico >= lo) & (pico < hi)).sum()) for b, (lo, hi) in bandas.items()}
    print(f"  Picos de los filtros por rango (Hz): {reparto}")
    return {"picos_hz": pico.tolist(), "reparto": reparto}


if __name__ == "__main__":
    main()
