"""
Tiempo en tarea — paso 7: figuras para la memoria.

Solo lee **agregados** ya calculados por los pasos 1, 2, 4 y 6 (tablas CSV
locales y `06_test/resultados_test.json`). No recalcula ningún análisis ni
carga datos de personas. Escribe cada figura en PDF (vectorial, para LaTeX) y
en PNG en `figuras/`.

Las tablas CSV de los pasos no están en git (`*.csv` en `.gitignore`): si
faltan, se regeneran ejecutando el script de su paso.

Ejecutar (desde la raíz del repositorio):
    .venv/Scripts/python.exe analisis/tiempo_en_tarea/07_redaccion/figuras.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CARPETA = Path(__file__).resolve().parent
LINEA = CARPETA.parent
SALIDA = CARPETA / "figuras"

# Colores de las tareas (los del paso 1; validados como paleta categórica). OA
# es la referencia: gris neutro, con etiqueta directa y marcador propio.
COLOR = {"PASAT": "#2a78d6", "PVTB": "#eb6834", "PVSAT": "#1f9e77", "OA": "#5f5e5a", "OC": "#5f5e5a"}
MARCADOR = {"PASAT": "o", "PVTB": "o", "PVSAT": "o", "OA": "s", "OC": "D"}
TINTA, TINTA_2, REJILLA = "#0b0b0b", "#5f5e5a", "#e4e3df"
SUPERFICIE = "#ffffff"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA,
    "xtick.color": TINTA_2, "ytick.color": TINTA_2, "text.color": TINTA, "axes.grid": True,
    "grid.color": REJILLA, "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "savefig.facecolor": SUPERFICIE,
    "figure.facecolor": SUPERFICIE, "legend.frameon": False,
})


def guardar(fig, nombre):
    for ext in ("pdf", "png"):
        fig.savefig(SALIDA / f"{nombre}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {nombre}.pdf / .png")


def ic(texto):
    return json.loads(texto) if isinstance(texto, str) else [np.nan, np.nan]


def puntos_con_ic(ax, y, valores, ics, relleno, etiqueta, desplaz):
    """Puntos con su IC95 en horizontal: relleno = primera serie, hueco = segunda."""
    v = np.asarray(valores, float)
    lo = v - np.array([i[0] for i in ics])
    hi = np.array([i[1] for i in ics]) - v
    ax.errorbar(v, np.asarray(y) + desplaz, xerr=[lo, hi], fmt="o", ms=6, lw=1.6, capsize=0,
                color=TINTA_2, mfc=TINTA_2 if relleno else SUPERFICIE, mec=TINTA_2, mew=1.6,
                label=etiqueta, zorder=3)


# --- 1. Curvas por minuto ------------------------------------------------------


def curvas():
    t = pd.read_csv(LINEA / "01_descriptivo" / "tablas" / "curvas_por_minuto.csv")
    t = t[t.region == "posterior"]
    # Solo minutos con al menos el 80 % de las personas: la cola (OA minuto 3,
    # tareas minuto 4) la aportan unas pocas grabaciones largas.
    t = t[t.personas >= 0.8 * t.groupby("condicion").personas.transform("max")]
    fig, ejes = plt.subplots(1, 2, figsize=(6.3, 2.6), sharex=True)
    for ax, (mag, titulo) in zip(ejes, (("alfa_abs", "Alfa absoluta, región posterior"),
                                         ("beta_rel", "Beta relativa, región posterior"))):
        for c in ("PASAT", "PVTB", "OA"):
            d = t[(t.magnitud == mag) & (t.condicion == c)].sort_values("minuto")
            ax.fill_between(d.minuto, d.ic_inf, d.ic_sup, color=COLOR[c], alpha=0.12, lw=0)
            ax.plot(d.minuto, d.media, color=COLOR[c], lw=2, marker=MARCADOR[c], ms=5,
                    mec=SUPERFICIE, mew=1.2, label=c, solid_capstyle="round")
        ax.axhline(0, color=TINTA_2, lw=0.8)
        ax.set_title(titulo, loc="left")
        ax.set_xlabel("Minuto de la grabación")
        ax.set_xticks([1, 2, 3])
        ax.set_xlim(0.8, 3.2)
    ejes[0].set_ylabel("Cambio centrado por grabación\n(log$_{10}$ de la potencia)")
    ejes[0].legend(loc="lower left", ncols=3, bbox_to_anchor=(0, 1.12), handlelength=1.6)
    guardar(fig, "f1_curvas_por_minuto")


# --- 2. Acierto pareado frente a OA (completa e igual tiempo) -----------------


def acierto_pareado():
    t = pd.read_csv(LINEA / "01_descriptivo" / "tablas" / "anexo_clasificador_igual_tiempo.csv")
    t = t[t.variante == "centrado+validos"]
    orden = ["PASAT", "PVTB", "PVSAT", "OA"]
    y = np.arange(len(orden))[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    for version, relleno, etiqueta, dz in (("completa", True, "Grabación completa", 0.13),
                                           ("recortada", False, "Igual tiempo que OA", -0.13)):
        d = t[t.version == version].set_index("condicion").reindex(orden)
        ok = d.pareado.notna().to_numpy()
        puntos_con_ic(ax, y[ok], d.pareado[ok], [ic(x) for x in d.pareado_ic95[ok]], relleno, etiqueta, dz)
    ax.axvline(0.5, color=TINTA_2, lw=0.8)
    ax.annotate("azar", (0.5, y[0] + 0.45), xytext=(3, 0), textcoords="offset points", fontsize=8, color=TINTA_2)
    ax.set_yticks(y, orden)
    ax.set_ylim(-0.6, len(orden) - 0.3)
    ax.set_xlim(0.45, 0.85)
    ax.set_xlabel("Acierto pareado de tercios (IC95 por personas)")
    ax.legend(loc="lower left", ncols=2, bbox_to_anchor=(0, 1.0))
    guardar(fig, "f2_acierto_pareado")


# --- 3. Referencia de artefactos ---------------------------------------------


def artefactos():
    t = pd.read_csv(LINEA / "02_artefactos" / "tablas" / "clasificadores.csv")
    t = t[t.version == "igual_tiempo"]
    orden = ["PASAT", "PVTB", "PVSAT", "OA"]
    y = np.arange(len(orden))[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    for modelo, relleno, etiqueta, dz in (("eeg", True, "EEG (56 potencias relativas)", 0.13),
                                          ("art", False, "Solo movimiento y contacto", -0.13)):
        d = t[t.modelo == modelo].set_index("condicion").reindex(orden)
        puntos_con_ic(ax, y, d.pareado, [ic(x) for x in d.pareado_ic95], relleno, etiqueta, dz)
    ax.axvline(0.5, color=TINTA_2, lw=0.8)
    ax.annotate("azar", (0.5, y[0] + 0.45), xytext=(3, 0), textcoords="offset points", fontsize=8, color=TINTA_2)
    ax.set_yticks(y, orden)
    ax.set_ylim(-0.6, len(orden) - 0.3)
    ax.set_xlim(0.45, 0.82)
    ax.set_xlabel("Acierto pareado de tercios, igual tiempo (IC95 por personas)")
    ax.legend(loc="lower left", ncols=2, bbox_to_anchor=(0, 1.0))
    guardar(fig, "f3_referencia_artefactos")


# --- 4. Matriz de transferencia -------------------------------------------------


def transferencia():
    t = pd.read_csv(LINEA / "04_potencia_y_controles" / "tablas" / "transferencia.csv", index_col=0)
    t.index = [i.replace("modelo ", "") for i in t.index]
    t = t.loc[t.columns]
    v = t.to_numpy(float)
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    ax.grid(False)
    im = ax.imshow(np.clip(v, 0, None), cmap="Blues", vmin=0, vmax=0.17)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, f"{v[i, j]:.2f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center",
                    fontsize=7.5, color=SUPERFICIE if v[i, j] > 0.11 else TINTA,
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(len(t.columns)), t.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(t.index)), t.index)
    ax.set_xlabel("Condición cuyas ventanas se puntúan")
    ax.set_ylabel("Condición con la que se entrena el modelo")
    for s in ax.spines.values():
        s.set_visible(False)
    barra = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    barra.set_label("ρ media (rho_ventanas)")
    barra.outline.set_visible(False)
    guardar(fig, "f4_transferencia")


# --- 5. Resultado en test ---------------------------------------------------------


def test():
    r = json.load(open(LINEA / "06_test" / "resultados_test.json", encoding="utf-8"))
    desc = r["descriptivos"]
    orden = ["PASAT", "PVTB", "PVSAT", "OA", "OC"]
    y = np.arange(len(orden))[::-1]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [1.5, 1]})
    for version, relleno, etiqueta, dz in (("completa", True, "Grabación completa", 0.13),
                                           ("igual_tiempo", False, "Igual tiempo que OA", -0.13)):
        d = [desc[f"{version} | {c} | rho"] for c in orden]
        puntos_con_ic(ax, y, [x["valor"] for x in d], [x["ic95"] for x in d], relleno, etiqueta, dz)
    ax.axvline(0, color=TINTA_2, lw=0.8)
    ax.set_yticks(y, [c + (" (control)" if c in ("OA", "OC") else "") for c in orden])
    ax.set_ylim(-0.6, len(orden) - 0.3)
    ax.set_xlabel("ρ media de rho_ventanas (IC95)")
    ax.set_title("Por condición (descriptivo)", loc="left", pad=22)
    ax.legend(loc="lower left", ncols=2, bbox_to_anchor=(0, 1.0), fontsize=7.5)

    pruebas = [("H1b", "H1b: completa"), ("H1", "H1: igual tiempo")]
    y2 = np.array([1, 0])
    for yy, (k, nombre) in zip(y2, pruebas):
        x = r[k]
        ax2.errorbar(x["valor"], yy, xerr=[[x["valor"] - x["ic95"][0]], [x["ic95"][1] - x["valor"]]],
                     fmt="o", ms=6, lw=1.6, color=TINTA_2, mec=TINTA_2, zorder=3)
        ax2.annotate(f"{x['veredicto']}\np = {x['p_unilateral']:.3f}", (x["ic95"][1], yy), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=7.5, color=TINTA)
    ax2.axvline(0, color=TINTA_2, lw=0.8)
    ax2.set_yticks(y2, [n for _, n in pruebas])
    ax2.set_ylim(-0.6, 1.6)
    ax2.set_xlim(-0.06, 0.24)
    ax2.set_xlabel("Δρ = PASAT + PVTB − OA (IC95)")
    ax2.set_title("Pruebas principales (confirmatorio)", loc="left", pad=22)
    fig.tight_layout(w_pad=2.5)
    guardar(fig, "f5_resultado_test")


def main():
    SALIDA.mkdir(exist_ok=True)
    print("Figuras:")
    curvas()
    acierto_pareado()
    artefactos()
    transferencia()
    test()


if __name__ == "__main__":
    main()
