"""
Fatiga general (FG) y fatiga cognitiva (FC) frente al EEG — figura resumen de
la línea, pensada para enseñarla fuera del equipo técnico (tutores).

No calcula nada: lee los `resultados.json` de los pasos 2 y 3 (ejecuciones
completas, no `--rapido`) y dibuja en una sola imagen el ρ principal (Random
Forest) de cada medida en los dos niveles, con su IC95 y su p de Holm, y en el
nivel intra-persona el ρ con control de visita.

Ejecutar (desde la raíz del repositorio):
    python analisis/fatiga_fg_fc/figura_resumen.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

CARPETA = Path(__file__).resolve().parent
RUTA_PASO2 = CARPETA / "02_entre_personas" / "resultados.json"
RUTA_PASO3 = CARPETA / "03_intra_persona" / "resultados.json"
RUTA_FIGURA = CARPETA / "figura_resumen.png"

# Mismos colores que las figuras de los pasos 2 y 3
COLOR_A = "#2a78d6"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
REJILLA = "#e4e3df"
GRIS = "#8a8984"

OBJETIVOS = ["FG", "FC", "VAS"]
NOMBRES = {"FG": "Fatiga general", "FC": "Fatiga cognitiva", "VAS": "VAS\n(referencia)"}


def leer(ruta):
    """Lee un resultados.json y rechaza los de una ejecución rápida."""
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    if datos["parametros"]["rapido"]:
        raise SystemExit(f"{ruta} es de una ejecución --rapido: sus cifras no son resultados")
    return datos


def fila(datos, objetivo, variante):
    """Devuelve la fila de `por_variante` de un objetivo y una variante."""
    for r in datos["por_variante"]:
        if r["objetivo"] == objetivo and r["variante"] == variante:
            return r
    raise KeyError((objetivo, variante))


def holm_de(paso3, objetivo, nivel):
    """p de Holm y veredicto de la familia principal para un objetivo y nivel."""
    for r in paso3["holm"]:
        if r["familia"] == "principal" and r["prueba"] == f"{objetivo}, {nivel}, RF":
            return r["p_holm"], r["significativa"]
    return None, False


def _estilo(ax):
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(TINTA_2)
    ax.tick_params(colors=TINTA_2, labelsize=9, length=0)
    ax.xaxis.grid(True, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)


def panel(ax, datos, paso3, nivel, con_visita, titulo, pregunta, n_texto):
    """Dibuja un nivel: ρ del Random Forest por medida y, si se pide, con control de visita."""
    y0 = {o: 2 - i for i, o in enumerate(OBJETIVOS)}
    ax.axvline(0, color=TINTA, linewidth=1.2)
    desplazamiento = 0.14 if con_visita else 0.0
    for objetivo in OBJETIVOS:
        r = fila(datos, objetivo, "RF")
        y = y0[objetivo] + desplazamiento
        p_holm, signif = holm_de(paso3, objetivo, nivel)
        ax.errorbar(r["rho"], y, xerr=[[r["rho"] - r["ic95_inf"]], [r["ic95_sup"] - r["rho"]]],
                    fmt="o", color=COLOR_A, markersize=10, capsize=3, linewidth=2.2,
                    markerfacecolor=COLOR_A if signif else "white", markeredgewidth=2, zorder=3)
        if p_holm is None:
            texto = f"{r['rho']:+.2f}".replace(".", ",").replace("-", "−")
        else:
            texto = f"{r['rho']:+.2f}   p = {p_holm:.3f}".replace(".", ",").replace("-", "−")
        ax.text(r["ic95_sup"] + 0.015, y, texto, va="center", fontsize=9.5, color=TINTA,
                fontweight="bold" if signif else "normal")
        if con_visita:
            v = fila(datos, objetivo, "RF con control de visita")
            yv = y0[objetivo] - desplazamiento
            ax.errorbar(v["rho"], yv, xerr=[[v["rho"] - v["ic95_inf"]], [v["ic95_sup"] - v["rho"]]],
                        fmt="D", color=GRIS, markersize=7, capsize=2.5, linewidth=1.6,
                        markerfacecolor="white", markeredgewidth=1.8, zorder=3)
            ax.text(v["ic95_sup"] + 0.015, yv, f"{v['rho']:+.2f}".replace(".", ",").replace("-", "−"),
                    va="center", fontsize=9, color=TINTA_2)
    ax.axhspan(-0.5, 0.5, color=REJILLA, alpha=0.45, zorder=0)
    ax.set_yticks([y0[o] for o in OBJETIVOS], [NOMBRES[o] for o in OBJETIVOS], fontsize=10.5,
                  color=TINTA)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlim(-0.25, 0.55)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}".replace(".", ",").replace("-", "−")))
    ax.text(0.0, 1.13, titulo, transform=ax.transAxes, fontsize=12.5, fontweight="bold",
            color=TINTA)
    ax.text(0.0, 1.05, pregunta, transform=ax.transAxes, fontsize=10, color=TINTA_2,
            style="italic")
    ax.text(1.0, 1.13, n_texto, transform=ax.transAxes, fontsize=9, color=TINTA_2, ha="right")
    ax.text(0.003, -0.55, "azar", fontsize=8.5, color=TINTA_2, va="bottom")
    ax.set_xlabel("ρ: cuánto acierta el modelo a ordenar a los alumnos por fatiga",
                  fontsize=9, color=TINTA_2)
    _estilo(ax)


def main():
    paso2, paso3 = leer(RUTA_PASO2), leer(RUTA_PASO3)
    d2, d3 = paso2["datos"], paso3["datos"]

    fig, ejes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    fig.patch.set_facecolor("white")
    panel(ejes[0], paso2, paso3, "entre personas", False,
          "1 · Entre alumnos", "¿Los que suelen estar más fatigados tienen otro EEG?",
          f"{d2['personas']} alumnos")
    panel(ejes[1], paso3, paso3, "intra persona", True,
          "2 · Dentro del mismo alumno", "Cuando está más fatigado de lo normal en él, ¿cambia su EEG?",
          f"{d3['personas']} alumnos · {d3['sesiones']} sesiones")

    leyenda = [
        Line2D([], [], marker="o", color=COLOR_A, markerfacecolor=COLOR_A, markersize=9,
               markeredgewidth=2, linewidth=2.2,
               label="Significativo (p de Holm < 0,05 e intervalo sin el 0)"),
        Line2D([], [], marker="o", color=COLOR_A, markerfacecolor="white", markersize=9,
               markeredgewidth=2, linewidth=2.2, label="No significativo"),
        Line2D([], [], marker="D", color=GRIS, markerfacecolor="white", markersize=7,
               markeredgewidth=1.8, linewidth=1.6,
               label="Quitando la tendencia de cada alumno a lo largo del curso"),
    ]
    fig.legend(handles=leyenda, loc="lower center", ncol=3, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.06))
    fig.text(0.5, 0.015,
             "Punto: ρ de Spearman del Random Forest (0 = azar, 1 = orden perfecto); barra: intervalo "
             "de confianza del 95 %; p: corregida por Holm. El modelo nunca ve al alumno que evalúa. "
             "Solo train + val: exploratorio.",
             ha="center", fontsize=8.5, color=TINTA_2)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.83, bottom=0.27, wspace=0.12)
    fig.savefig(RUTA_FIGURA, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Figura guardada en {RUTA_FIGURA}")


if __name__ == "__main__":
    main()
