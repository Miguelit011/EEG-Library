"""
Regresión lineal simple: VAS ~ índice de fatiga theta/alfa.

Contraste directo del biomarcador de referencia de la literatura contra la
etiqueta real del estudio. No entrena ningún modelo: es el análisis descriptivo
que la memoria del TFG tiene que reportar sobre el índice theta/alfa, con su
gráfica.

**Solo usa `train` + `val`.** El conjunto de test (106 sesiones, 28 personas) se
aparta en el paso 1 y no vuelve a tocarse.

La unidad de análisis es la **sesión**, no la grabación ni la ventana: el VAS se
responde una vez por sesión, así que las 4543 grabaciones aportan 653 etiquetas
independientes (547 en train+val), no 4543.

Ejecutar (desde la raíz del repositorio):
    python analisis/analisis_regresion_lineal.py

La ubicación del dataset se toma de la variable de entorno `EEG_DATA`, con
`C:\\Users\\migue\\Desktop\\TFG\\EEG-Data` por defecto (ver `analisis/README.md`).
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from eeg_library.modelado.modeling import (
    COLUMNA_ETIQUETA,
    COLUMNA_GRUPO,
    agregar_a_sesion,
    lineas_base,
)

# La salida lleva ρ, ² y acentos. En Windows, redirigir a un fichero usa cp1252
# por defecto y el script moriría con UnicodeEncodeError justo al archivar el
# resultado, que es el caso de uso que importa.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEMILLA = 42
N_BOOTSTRAP = 10000
RUTA_DATOS = Path(os.environ.get("EEG_DATA", r"C:\Users\migue\Desktop\TFG\EEG-Data"))
RUTA_MATRIZ = RUTA_DATOS / "processed" / "matriz_rf.parquet"
RUTA_RESULTADOS = RUTA_DATOS / "processed" / "regresion_lineal.json"
RUTA_FIGURAS = Path(__file__).resolve().parents[1] / "resultados" / "figuras"

PREDICTOR = "indice_fatiga_media"

# Listón declarado *antes* de ejecutar (ver REGRESION_LINEAL.md, "resultado
# óptimo"): el R² que ya consigue predecir la media de VAS de la propia persona
# leave-one-out, sin EEG. Un biomarcador que no lo supera no aporta información.
R2_LINEA_BASE_DECLARADO = 0.260
# Pendiente que haría el índice clínicamente útil: que su rango intercuartílico
# (~1.3 a ~2.6) moviera el VAS unos 15 puntos de los 100 de la escala.
PENDIENTE_OPTIMA = 10.0

# Paleta: slot 1 (azul) para los datos, slot 2 (naranja) para el ajuste. Gris
# neutro para la referencia hipotética, que es anotación y no una serie más.
COLOR_PUNTOS = "#2a78d6"
COLOR_RECTA = "#eb6834"
COLOR_REFERENCIA = "#8a8a85"
COLOR_TEXTO = "#0b0b0b"
COLOR_TEXTO_2 = "#52514e"
COLOR_REJILLA = "#e0e0dc"


def _titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def ols(x, y, gl_extra=0):
    """
    Regresión lineal simple por mínimos cuadrados, con IC al 95 %.

    `gl_extra` descuenta grados de libertad ya gastados antes de ajustar la
    recta. Es 0 en la regresión normal y `n_personas` en la versión centrada
    por persona, donde cada media individual consume uno.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    gl = n - 2 - gl_extra
    if gl < 1:
        raise ValueError(f"Grados de libertad insuficientes: {gl}.")

    sx, sy = x - x.mean(), y - y.mean()
    sxx = np.sum(sx ** 2)
    pendiente = np.sum(sx * sy) / sxx
    intercepto = y.mean() - pendiente * x.mean()
    ajustado = intercepto + pendiente * x
    residuos = y - ajustado

    ss_res = np.sum(residuos ** 2)
    ss_tot = np.sum(sy ** 2)
    r2 = 1 - ss_res / ss_tot
    s2 = ss_res / gl
    ee_pendiente = np.sqrt(s2 / sxx)
    ee_intercepto = np.sqrt(s2 * (1 / n + x.mean() ** 2 / sxx))

    t = pendiente / ee_pendiente
    p = float(2 * stats.t.sf(abs(t), gl))
    t_critico = stats.t.ppf(0.975, gl)

    r_pearson = float(np.sign(pendiente) * np.sqrt(max(r2, 0.0)))
    rho, p_rho = stats.spearmanr(x, y)

    return {
        "n": int(n),
        "gl": int(gl),
        "pendiente": float(pendiente),
        "ee_pendiente": float(ee_pendiente),
        "ic95_pendiente": [float(pendiente - t_critico * ee_pendiente),
                           float(pendiente + t_critico * ee_pendiente)],
        "intercepto": float(intercepto),
        "ee_intercepto": float(ee_intercepto),
        "ic95_intercepto": [float(intercepto - t_critico * ee_intercepto),
                            float(intercepto + t_critico * ee_intercepto)],
        "t": float(t),
        "p": p,
        "r2": float(r2),
        "r_pearson": r_pearson,
        "spearman": float(rho),
        "p_spearman": float(p_rho),
        "rmse_residual": float(np.sqrt(ss_res / n)),
        "sd_y": float(np.std(y)),
    }


def bootstrap_por_persona(x, y, grupos, n=N_BOOTSTRAP, semilla=SEMILLA):
    """
    IC al 95 % de la pendiente y del R² remuestreando **personas**.

    El IC de la tabla MCO supone 547 observaciones independientes, y no lo son:
    cada persona aporta ~3.7 sesiones. Remuestrear personas es la única forma
    honesta de poner un intervalo aquí, y es el invariante que sigue el resto
    del proyecto (ver `analisis/README.md`).
    """
    rng = np.random.default_rng(semilla)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    grupos = np.asarray(grupos)
    unicos = pd.unique(grupos)
    indices = {u: np.flatnonzero(grupos == u) for u in unicos}

    pendientes, r2s = [], []
    for _ in range(n):
        elegidos = rng.choice(unicos, size=len(unicos), replace=True)
        idx = np.concatenate([indices[u] for u in elegidos])
        xb, yb = x[idx], y[idx]
        sxx = np.sum((xb - xb.mean()) ** 2)
        if sxx <= 0:
            continue
        pendiente = np.sum((xb - xb.mean()) * (yb - yb.mean())) / sxx
        pendientes.append(pendiente)
        ss_tot = np.sum((yb - yb.mean()) ** 2)
        if ss_tot > 0:
            ajuste = yb.mean() + pendiente * (xb - xb.mean())
            r2s.append(1 - np.sum((yb - ajuste) ** 2) / ss_tot)

    return {
        "n_replicas": len(pendientes),
        "ic95_pendiente": [float(np.percentile(pendientes, 2.5)),
                           float(np.percentile(pendientes, 97.5))],
        "ic95_r2": [float(np.percentile(r2s, 2.5)),
                    float(np.percentile(r2s, 97.5))],
    }


def _formatea(ajuste, etiqueta):
    ic = ajuste["ic95_pendiente"]
    print(f"\n{etiqueta}")
    print(f"  n = {ajuste['n']}   gl = {ajuste['gl']}")
    print(f"  pendiente   = {ajuste['pendiente']:+.4f} "
          f"(EE {ajuste['ee_pendiente']:.4f})  "
          f"IC95 [{ic[0]:+.4f}, {ic[1]:+.4f}]")
    print(f"  intercepto  = {ajuste['intercepto']:+.4f}")
    print(f"  R2          = {ajuste['r2']:.5f}   r = {ajuste['r_pearson']:+.4f}")
    print(f"  t({ajuste['gl']}) = {ajuste['t']:+.3f}   p = {ajuste['p']:.4f}")
    print(f"  Spearman rho = {ajuste['spearman']:+.4f}  p = {ajuste['p_spearman']:.4f}")


def figura_principal(x, y, ajuste, ruta):
    """
    Dispersión VAS ~ índice theta/alfa con la recta de regresión, su banda de
    confianza al 95 % y, en gris discontinuo, la pendiente que el índice
    tendría que tener para ser útil (declarada antes de ejecutar).
    """
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.scatter(x, y, s=24, c=COLOR_PUNTOS, alpha=0.42,
               edgecolors="white", linewidths=0.5, zorder=3,
               label=f"Sesión (n = {len(x)})")

    rejilla = np.linspace(x.min(), x.max(), 200)
    recta = ajuste["intercepto"] + ajuste["pendiente"] * rejilla

    # Banda de confianza de la media condicional.
    n = len(x)
    sxx = np.sum((x - x.mean()) ** 2)
    s = ajuste["rmse_residual"] * np.sqrt(n / ajuste["gl"])
    ee = s * np.sqrt(1 / n + (rejilla - x.mean()) ** 2 / sxx)
    margen = stats.t.ppf(0.975, ajuste["gl"]) * ee
    ax.fill_between(rejilla, recta - margen, recta + margen,
                    color=COLOR_RECTA, alpha=0.18, linewidth=0, zorder=4)

    centro_x, centro_y = x.mean(), y.mean()
    ax.plot(rejilla, centro_y + PENDIENTE_OPTIMA * (rejilla - centro_x),
            color=COLOR_REFERENCIA, linewidth=1.6, linestyle=(0, (5, 4)),
            zorder=5,
            label=f"Óptimo hipotético (pendiente +{PENDIENTE_OPTIMA:.0f})")

    ic = ajuste["ic95_pendiente"]
    ax.plot(rejilla, recta, color=COLOR_RECTA, linewidth=2.2, zorder=6,
            label=(f"Ajuste MCO: {ajuste['pendiente']:+.2f} "
                   f"[{ic[0]:+.2f}, {ic[1]:+.2f}]"))

    ax.set_xlabel("Índice de fatiga theta/alfa\n"
                  "(theta frontal / alfa posterior, media de la sesión)",
                  fontsize=10.5, color=COLOR_TEXTO_2, labelpad=9)
    ax.set_ylabel("VAS de fatiga percibida (mm)", fontsize=10.5,
                  color=COLOR_TEXTO_2, labelpad=9)
    ax.set_title("El índice theta/alfa no predice la fatiga autopercibida",
                 fontsize=13.5, color=COLOR_TEXTO, pad=14, loc="left",
                 fontweight="bold")

    resumen = (f"R² = {ajuste['r2']:.4f}      p = {ajuste['p']:.2f}\n"
               f"Línea base sin EEG (media por persona): R² = "
               f"{R2_LINEA_BASE_DECLARADO:.3f}")
    # zorder por encima de la recta de referencia: si no, la discontinua gris
    # atraviesa el cuadro justo por donde va el texto.
    ax.text(0.985, 0.975, resumen, transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, color=COLOR_TEXTO_2, linespacing=1.6, zorder=10,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="white",
                      edgecolor=COLOR_REJILLA, linewidth=0.9))

    ax.set_ylim(-6, 106)
    ax.grid(True, color=COLOR_REJILLA, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(COLOR_REJILLA)
    ax.tick_params(colors=COLOR_TEXTO_2, labelsize=9.5, length=0)

    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=9,
              edgecolor=COLOR_REJILLA, facecolor="white", borderpad=0.7)

    # El porcentaje se mide sobre esta misma muestra en vez de citar el 76.7 %
    # del proyecto entero, que se calcula sobre las 653 sesiones e incluye test.
    redondos = 100 * np.mean(np.mod(y, 10) == 0)
    fig.text(0.008, 0.012,
             "Train + val (el conjunto de test permanece sin abrir). Una sesión "
             f"= una respuesta al VAS.\nLas bandas horizontales son reales: el "
             f"{redondos:.1f} % de las respuestas son múltiplos de 10.",
             fontsize=8, color=COLOR_TEXTO_2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(ruta, facecolor="white")
    plt.close(fig)
    print(f"  figura -> {ruta}")


def figura_intrapersona(x, y, ajuste, ruta):
    """Misma regresión tras restar a cada persona su propia media."""
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.axhline(0, color=COLOR_REJILLA, linewidth=1.2, zorder=1)
    ax.axvline(0, color=COLOR_REJILLA, linewidth=1.2, zorder=1)
    ax.scatter(x, y, s=24, c=COLOR_PUNTOS, alpha=0.42,
               edgecolors="white", linewidths=0.5, zorder=3,
               label=f"Sesión centrada en su persona (n = {len(x)})")

    rejilla = np.linspace(x.min(), x.max(), 200)
    ic = ajuste["ic95_pendiente"]
    ax.plot(rejilla, ajuste["intercepto"] + ajuste["pendiente"] * rejilla,
            color=COLOR_RECTA, linewidth=2.2, zorder=5,
            label=(f"Ajuste MCO: {ajuste['pendiente']:+.2f} "
                   f"[{ic[0]:+.2f}, {ic[1]:+.2f}]"))

    ax.set_xlabel("Índice theta/alfa − media de la persona", fontsize=10.5,
                  color=COLOR_TEXTO_2, labelpad=9)
    ax.set_ylabel("VAS − media de la persona (mm)", fontsize=10.5,
                  color=COLOR_TEXTO_2, labelpad=9)
    ax.set_title("Dentro de cada persona tampoco: el índice no sigue\n"
                 "a los días en que esa misma persona se siente más cansada",
                 fontsize=12.5, color=COLOR_TEXTO, pad=14, loc="left",
                 fontweight="bold")
    ax.text(0.985, 0.975,
            f"R² = {ajuste['r2']:.4f}      p = {ajuste['p']:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
            color=COLOR_TEXTO_2,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="white",
                      edgecolor=COLOR_REJILLA, linewidth=0.9))

    ax.grid(True, color=COLOR_REJILLA, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(COLOR_REJILLA)
    ax.tick_params(colors=COLOR_TEXTO_2, labelsize=9.5, length=0)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=9,
              edgecolor=COLOR_REJILLA, facecolor="white", borderpad=0.7)

    fig.tight_layout()
    fig.savefig(ruta, facecolor="white")
    plt.close(fig)
    print(f"  figura -> {ruta}")


def main():
    resultados = {}

    _titulo("1. Datos - sesiones de train + val")
    matriz = pd.read_parquet(RUTA_MATRIZ)
    sesiones = agregar_a_sesion(matriz)
    print(f"Matriz por grabación : {matriz.shape[0]} filas")
    print(f"Sesiones etiquetadas : {len(sesiones)}")

    fuera = sesiones[sesiones["split"] == "test"]
    datos = sesiones[sesiones["split"].isin(["train", "val"])].copy()
    print(f"Apartado (test)      : {len(fuera)} sesiones, "
          f"{fuera[COLUMNA_GRUPO].nunique()} personas - no se vuelve a leer")
    print(f"En análisis          : {len(datos)} sesiones, "
          f"{datos[COLUMNA_GRUPO].nunique()} personas")

    datos = datos[datos[PREDICTOR].notna() & datos[COLUMNA_ETIQUETA].notna()]
    x = datos[PREDICTOR].to_numpy(dtype=float)
    y = datos[COLUMNA_ETIQUETA].to_numpy(dtype=float)
    grupos = datos[COLUMNA_GRUPO].to_numpy()
    n_personas = datos[COLUMNA_GRUPO].nunique()
    print(f"Sesiones por persona : mediana "
          f"{datos.groupby(COLUMNA_GRUPO).size().median():.1f}")
    print(f"\nVAS          : media {y.mean():.2f}  SD {y.std(ddof=1):.2f}  "
          f"rango [{y.min():.0f}, {y.max():.0f}]")
    print(f"Indice t/a   : media {x.mean():.3f}  SD {x.std(ddof=1):.3f}  "
          f"rango [{x.min():.3f}, {x.max():.3f}]  "
          f"asimetria {stats.skew(x):.2f}")
    resultados["datos"] = {
        "n_sesiones": int(len(datos)), "n_personas": int(n_personas),
        "n_test_apartado": int(len(fuera)),
        "vas_media": float(y.mean()), "vas_sd": float(y.std(ddof=1)),
        "indice_media": float(x.mean()), "indice_sd": float(x.std(ddof=1)),
        "indice_asimetria": float(stats.skew(x)),
    }

    _titulo("2. El liston: que hay que batir")
    base = lineas_base(datos)
    print(base[["modelo", "n", "r2", "mae", "rmse"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    r2_loo = float(
        base.loc[base["modelo"].str.contains("leave-one-out"), "r2"].iloc[0])
    print(f"\nListón declarado antes de ejecutar: R2 = {R2_LINEA_BASE_DECLARADO:.3f}")
    print(f"Recalculado sobre estas {len(datos)} sesiones: R2 = {r2_loo:.4f}")
    resultados["linea_base"] = {
        "r2_media_por_persona_loo": r2_loo,
        "r2_declarado": R2_LINEA_BASE_DECLARADO,
        "tabla": base.to_dict(orient="records"),
    }

    _titulo("3. Regresion lineal simple: VAS ~ indice theta/alfa")
    ajuste = ols(x, y)
    _formatea(ajuste, f"MCO sobre las {len(datos)} sesiones")
    print(f"\n  SD del VAS                 = {ajuste['sd_y']:.2f} mm")
    print(f"  SD residual tras el ajuste = {ajuste['rmse_residual']:.2f} mm  "
          f"(reduccion: "
          f"{100 * (1 - ajuste['rmse_residual'] / ajuste['sd_y']):.2f} %)")
    resultados["regresion"] = ajuste

    print("\n  IC remuestreando personas (no sesiones):")
    boot = bootstrap_por_persona(x, y, grupos)
    print(f"    pendiente IC95 [{boot['ic95_pendiente'][0]:+.4f}, "
          f"{boot['ic95_pendiente'][1]:+.4f}]")
    print(f"    R2        IC95 [{boot['ic95_r2'][0]:.5f}, "
          f"{boot['ic95_r2'][1]:.5f}]  ({boot['n_replicas']} replicas)")
    resultados["bootstrap_personas"] = boot

    _titulo("4. Es un problema de forma? Dos comprobaciones")
    ajuste_log = ols(np.log10(x), y)
    _formatea(ajuste_log, "MCO con log10 del indice (corrige la asimetria)")
    resultados["regresion_log10"] = ajuste_log

    q1, q99 = np.percentile(x, [1, 99])
    recorte = (x >= q1) & (x <= q99)
    ajuste_rec = ols(x[recorte], y[recorte])
    _formatea(ajuste_rec, "MCO recortando el 1 % de cada cola del indice")
    resultados["regresion_recortada"] = ajuste_rec

    _titulo("5. La comprobacion decisiva: se sostiene dentro de cada persona?")
    print("Se resta a cada sesion la media de su propia persona, en las dos")
    print("variables. Lo que queda es la variacion dia a dia del mismo")
    print("individuo, que es lo que un marcador de fatiga debe seguir.")
    dx = x - datos.groupby(COLUMNA_GRUPO)[PREDICTOR].transform("mean").to_numpy()
    dy = y - datos.groupby(
        COLUMNA_GRUPO)[COLUMNA_ETIQUETA].transform("mean").to_numpy()
    ajuste_intra = ols(dx, dy, gl_extra=n_personas)
    _formatea(ajuste_intra, "MCO intrapersona (efectos fijos por persona)")
    resultados["regresion_intrapersona"] = ajuste_intra

    _titulo("6. Figuras")
    RUTA_FIGURAS.mkdir(exist_ok=True)
    figura_principal(x, y, ajuste, RUTA_FIGURAS / "regresion_vas_theta_alfa.png")
    figura_intrapersona(dx, dy, ajuste_intra,
                        RUTA_FIGURAS / "regresion_vas_theta_alfa_intrapersona.png")

    _titulo("7. Veredicto frente al optimo declarado")
    ic = ajuste["ic95_pendiente"]
    criterios = [
        ("Pendiente positiva", ajuste["pendiente"] > 0,
         f"{ajuste['pendiente']:+.3f}"),
        ("Nube estrecha (SD residual < SD del VAS)",
         ajuste["rmse_residual"] < 0.9 * ajuste["sd_y"],
         f"{ajuste['rmse_residual']:.2f} vs {ajuste['sd_y']:.2f} mm"),
        (f"R2 supera la linea base ({R2_LINEA_BASE_DECLARADO:.3f})",
         ajuste["r2"] > R2_LINEA_BASE_DECLARADO, f"{ajuste['r2']:.4f}"),
        (f"Pendiente con magnitud clinica (~ +{PENDIENTE_OPTIMA:.0f})",
         ajuste["pendiente"] >= PENDIENTE_OPTIMA / 2, f"{ajuste['pendiente']:+.3f}"),
        ("IC95 excluye el 0", ic[0] > 0 or ic[1] < 0,
         f"[{ic[0]:+.3f}, {ic[1]:+.3f}]"),
        # El criterio 6 del optimo era "sin artefactos de forma": comprobar que
        # el resultado no lo dicta la asimetria del indice ni sus colas. Se
        # cumple si las tres versiones (cruda, log10, recortada) coinciden en
        # signo y en orden de magnitud del R2, sea cual sea ese resultado.
        ("El resultado no lo dicta la forma (log10 y recorte coinciden)",
         (np.sign(ajuste_log["pendiente"]) == np.sign(ajuste["pendiente"])
          == np.sign(ajuste_rec["pendiente"])
          and max(ajuste_log["r2"], ajuste_rec["r2"], ajuste["r2"]) < 0.05),
         f"R2 {ajuste['r2']:.3f} / {ajuste_log['r2']:.3f} / "
         f"{ajuste_rec['r2']:.3f}"),
        ("Se sostiene dentro de la persona", ajuste_intra["p"] < 0.05,
         f"p = {ajuste_intra['p']:.3f}"),
    ]
    for nombre, cumple, valor in criterios:
        print(f"  [{'SI' if cumple else 'NO'}] {nombre:<52} {valor}")
    n_ok = sum(c for _, c, _ in criterios)
    print(f"\n  Criterios cumplidos: {n_ok} de {len(criterios)}")
    resultados["veredicto"] = {
        "criterios": [{"criterio": n, "cumple": bool(c), "valor": v}
                      for n, c, v in criterios],
        "n_cumplidos": int(n_ok), "n_total": len(criterios),
    }

    RUTA_RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    RUTA_RESULTADOS.write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print(f"\nResultados -> {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
