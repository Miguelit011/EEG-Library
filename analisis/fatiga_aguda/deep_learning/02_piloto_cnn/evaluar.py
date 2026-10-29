"""
Fatiga aguda — deep learning, paso 2 (piloto), parte 3 de 3: evaluar.

Calcula, en las personas de evaluación del pliegue 0, `rho_ventanas` y el AUC
por ventana de cada ejecución de EEGNet y de las logísticas F y B, la
diferencia pareada EEGNet − F y la regla de decisión del `INFORME.md`. Usa las
funciones de `03_clasificador/medidas.py`, en el entorno validado.

Ejecutar (desde la raíz del repositorio):
    .venv/Scripts/python.exe analisis/fatiga_aguda/deep_learning/02_piloto_cnn/evaluar.py > analisis/fatiga_aguda/deep_learning/02_piloto_cnn/evaluar.log
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA))
sys.path.insert(0, str(CARPETA.parent / "01_psd_fina"))
from preparar import CACHE  # noqa: E402
from psd_fina import FAMILIA, d1, md  # noqa: E402

MARGEN = 0.01
RHO_F_PASO1 = 0.174  # ρ de PASAT + PVTB de F (C = 0.003) en la CV de 20 particiones del paso 1


def rho(v, punt):
    """ρ por grabación (con su condición) a partir de la puntuación de las ventanas."""
    r = md.rho_por_grabacion(v[["archivo_origen", "persona_uid", "t_rec"]], punt)
    return r.merge(v.drop_duplicates("archivo_origen")[["archivo_origen", "condicion"]])


def ic(reps):
    return [float(x) for x in np.percentile(reps, [2.5, 97.5])]


def main():
    v = pd.read_parquet(CACHE / "ventanas.parquet")
    v = v[v.papel == "evaluacion"].reset_index(drop=True)
    with open(CACHE / "entrenamiento.json", encoding="utf-8") as f:
        ent = json.load(f)
    rng = np.random.default_rng(d1.SEMILLA)
    terc = v.seg.to_numpy() >= 0

    d1._titulo("1. EJECUCIONES Y REFERENCIAS (personas de evaluación del pliegue 0)")
    print(f"Personas: {v.persona_uid.nunique()}  grabaciones: {v.archivo_origen.nunique()}  ventanas: {len(v)}")
    modelos, filas, curvas = {}, [], []
    for nombre in ent["ejecuciones"]:
        with open(CACHE / "ejecuciones" / nombre / "resumen.json", encoding="utf-8") as f:
            r = json.load(f)
        punt = np.load(CACHE / "ejecuciones" / nombre / "puntuacion_evaluacion.npy")
        assert len(punt) == len(v)
        modelos[nombre] = punt
        curvas += [dict(c, ejecucion=nombre) for c in r["curva"]]
        filas.append({k: r[k] for k in ("nombre", "lr", "semilla", "epocas", "mejor_epoca", "mejor_perdida_interna",
                                        "auc_interna_mejor", "segundos_por_epoca", "nan")})
    for ref in ["F", "B"]:
        modelos[f"logistica_{ref}"] = v[f"punt_{ref}"].to_numpy()
        filas.append({"nombre": f"logistica_{ref}"})
    tabla = pd.DataFrame(filas).set_index("nombre")
    rhos = {}
    for nombre, punt in modelos.items():
        rhos[nombre] = rho(v, punt)
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
    semillas = [n for n in ent["ejecuciones"] if n.startswith(f"lr{lr:g}_")]
    cnn = rhos[semillas[0]][["archivo_origen", "persona_uid", "condicion"]].copy()
    cnn["acierto"] = np.mean([rhos[n].set_index("archivo_origen").acierto.loc[cnn.archivo_origen].to_numpy()
                              for n in semillas], axis=0)
    rho_cnn = float(cnn.acierto.mean())
    rho_f, rho_b = float(tabla.loc["logistica_F", "rho_familia"]), float(tabla.loc["logistica_B", "rho_familia"])
    desv_semillas = float(tabla.loc[semillas, "rho_familia"].std(ddof=1))
    dif = {}
    for ref in ["F", "B"]:
        d = cnn.merge(rhos[f"logistica_{ref}"], on=["archivo_origen", "persona_uid"], suffixes=("", "_ref"))
        d = d.assign(acierto=d.acierto - d.acierto_ref)
        reps = md.bootstrap(d, rng=rng)
        dif[ref] = {"valor": float(d.acierto.mean()), "ic95": ic(reps)}
        print(f"  EEGNet − logística {ref}: {dif[ref]['valor']:+.4f} [{dif[ref]['ic95'][0]:+.4f}, "
              f"{dif[ref]['ic95'][1]:+.4f}]")
    print(f"  ρ EEGNet (media de {len(semillas)} semillas, lr = {lr:g}): {rho_cnn:.4f}  "
          f"(desviación entre semillas {desv_semillas:.4f})")
    print(f"  ρ logística F: {rho_f:.4f}   ρ logística B: {rho_b:.4f}")
    if rho_cnn >= rho_f - MARGEN:
        decision = "LLEGA AL NIVEL DE LA LOGÍSTICA F: paso 3"
    elif rho_cnn >= rho_b - MARGEN:
        decision = "IGUALA A LAS BANDAS PERO NO AL ESPECTRO FINO: se valora"
    else:
        decision = "NO IGUALA A LAS BANDAS: parar"
    print(f"\n  Decisión: {decision}")

    d1._titulo("3. REQUISITOS TÉCNICOS")
    eje = tabla.loc[[n for n in ent["ejecuciones"]]]
    curv = pd.DataFrame(curvas)
    baja = {n: bool(g.perdida_interna.min() < g.perdida_interna.iloc[0]) for n, g in curv.groupby("ejecucion")}
    requisitos = {
        "perdida_interna_baja": all(baja.values()),
        "mejor_epoca_ni_1_ni_max": bool(((eje.mejor_epoca > 1) & (eje.mejor_epoca < 60)).all()),
        "sin_nan": bool(not eje.nan.astype(bool).any()),
        "logistica_F_cerca_del_paso_1": bool(abs(rho_f - RHO_F_PASO1) < 0.05),
    }
    for k, ok in requisitos.items():
        print(f"  {k}: {'sí' if ok else 'NO'}")

    figuras(curv, tabla, ent["ejecuciones"])
    with open(CARPETA / "resultados.json", "w", encoding="utf-8") as f:
        json.dump({"fecha": time.strftime("%Y-%m-%d"), "particion": "train (pliegue 0 de la semilla 0)",
                   "mejor_lr": lr, "rho_cnn": rho_cnn, "rho_F": rho_f, "rho_B": rho_b,
                   "desviacion_entre_semillas": desv_semillas, "diferencias": dif, "decision": decision,
                   "requisitos": requisitos, "entrenamiento": ent,
                   "ejecuciones": tabla.reset_index().to_dict(orient="records")},
                  f, ensure_ascii=False, indent=1, default=float)


def figuras(curv, tabla, ejecuciones):
    (CARPETA / "figuras").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for n in ejecuciones:
        g = curv[curv.ejecucion == n]
        ax.plot(g.epoca, g.perdida_interna, lw=1.2, label=n)
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida en el conjunto interno")
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CARPETA / "figuras" / "curvas.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    t = tabla.rho_familia
    colores = ["#2a78d6" if n.startswith("lr") else "#8a8984" for n in t.index]
    ax.barh(range(len(t))[::-1], t.to_numpy(), color=colores)
    ax.set_yticks(range(len(t))[::-1])
    ax.set_yticklabels(t.index, fontsize=8)
    ax.set_xlabel("ρ media por grabación, PASAT + PVTB (pliegue 0)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CARPETA / "figuras" / "rho_por_modelo.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
