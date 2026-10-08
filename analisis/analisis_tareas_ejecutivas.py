"""
Clasificación multiclase: las cinco tareas ejecutivas del protocolo.

¿Con qué rendimiento identifica el pipeline cuál de las cinco tareas ejecutivas
(STROOP, PASAT, PVTB, MSIT, PVSAT) es una grabación, y qué tareas se confunden
entre sí? Es el escalón siguiente a `analisis_reposo_tarea.py`: aquella
pregunta era «¿hay carga cognitiva?» (binaria, sí/no); esta es «¿de qué tipo?»
(multiclase, 5 clases). OA y OC quedan fuera; ya están cubiertas en
`analisis_oa_oc.py` y `analisis_reposo_tarea.py`.

La unidad de análisis es la **grabación**: la condición es una propiedad de
cada registro. Las grabaciones de una misma persona no son independientes, así
que la validación cruzada y los intervalos agrupan por `persona_uid` (nunca
por `sujeto_id`). Las cinco tareas están casi perfectamente equilibradas entre
sí (a diferencia de reposo frente a tarea, que iba 1:2.5): el azar de la
exactitud balanceada es 1/5 = 0.200 exacto.

Lo interesante no es el acierto global —nadie necesita adivinar qué tarea
hacía alguien—, sino el **patrón de confusiones**: si sigue el parecido
cognitivo de las tareas (declarado antes de ejecutar en `SIMILITUD_COGNITIVA`),
es señal neurofisiológica; si sigue la distancia temporal en el protocolo, que
es fijo (STROOP → PASAT → PVTB → MSIT → PVSAT), o la duración de la
grabación, es un artefacto de diseño. Las tres estructuras se miden y se
comparan (sección 7); el diseño tiene una disociación aprovechable, porque los
dos pares cognitivamente más parecidos (PASAT-PVSAT, STROOP-MSIT) están ambos
a distancia 3 en el protocolo, mientras que los pares contiguos son
cognitivamente distintos. La duración no se disocia igual de bien y se audita
aparte (sección 8.2).

**Solo usa `train` + `val`.** El conjunto de test se aparta en el paso 1 y no
vuelve a tocarse. `preparar_tareas_ejecutivas` se niega a incluirlo sin
permiso explícito.

No modifica el pipeline congelado: todo sale de `matriz_rf.parquet` (Fase 13),
salvo el suplemento por canal, que reagrega `matriz_modelado.parquet`
(Fase 11.5) con la propia `construir_matriz_rf` y regiones de un solo canal.

Ejecutar (desde la raíz del repositorio), **una sola vez y nunca en paralelo**:
    python analisis/analisis_tareas_ejecutivas.py > resultados/logs/tareas_ejecutivas.log 2>&1

`--rapido` ejecuta el mismo código con parámetros mínimos y escribe en un
directorio temporal, para comprobar que el guion corre de principio a fin y
revisar las figuras. Sus cifras NO son resultados.

La ubicación del dataset se toma de la variable de entorno `EEG_DATA`, con
`C:\\Users\\migue\\Desktop\\TFG\\EEG-Data` por defecto (ver `analisis/README.md`).
"""

import itertools
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.dummy import DummyClassifier

from eeg_library.modelado.reposo import (
    COLUMNA_FUGA,
    COLUMNAS_PROCESAMIENTO,
    bloque_de_feature,
    bootstrap_diferencia_por_persona,
    bootstrap_por_persona,
    columnas_del_bloque,
    columnas_duplicadas,
    importancia_permutacion_cv,
    importancia_por_bloque,
    permutar_etiquetas_dentro_de_persona,
    tamanos_de_efecto,
    verificar_sin_fuga,
)
from eeg_library.modelado.reposo_tarea import columnas_media_del_bloque
from eeg_library.modelado.tareas_ejecutivas import (
    CONDICIONES_EJECUTIVAS,
    SIMILITUD_COGNITIVA,
    coeficiente_neto_por_clase,
    coeficientes_por_clase_y_pliegue,
    columnas_de_features_ejecutivas,
    confusion_por_pares,
    distancias_de_duracion,
    estructura_de_la_confusion,
    etiqueta_multiclase,
    evaluar_multiclase,
    matriz_confusion_normalizada,
    metricas_multiclase,
    modelo_bosque_multiclase,
    modelo_logistico_multinomial,
    preparar_tareas_ejecutivas,
    separabilidad_por_pares,
    tamanos_de_efecto_una_contra_resto,
)
from eeg_library.preprocesado.quality import CANALES_EPOC
from eeg_library.procesado.bands import BANDAS_DEFECTO
from eeg_library.procesado.matrix import (
    ORDEN_PROTOCOLO,
    REGIONES,
    columnas_de_features,
    construir_matriz_rf,
)

# La salida lleva acentos, ² y signos. En Windows, redirigir a un fichero usa
# cp1252 por defecto y el script moriría con UnicodeEncodeError justo al
# archivar el resultado. `line_buffering` hace que el log vaya apareciendo
# mientras corre (~25-40 min completo).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

RAPIDO = "--rapido" in sys.argv

SEMILLA = 42
N_PLIEGUES = 5
# Mismos parámetros de cómputo que las Fases 17 y 18, para que las cifras sean
# comparables. En modo rápido se reducen al mínimo.
N_BOOTSTRAP = 40 if RAPIDO else 2000
N_BOOTSTRAP_PARES = 40 if RAPIDO else 2000
N_BOOTSTRAP_CANALES = 40 if RAPIDO else 2000
N_REPETICIONES_PERMUTACION = 1 if RAPIDO else 5
N_PERMUTACIONES_ETIQUETAS = 2 if RAPIDO else 20
N_ARBOLES = 30 if RAPIDO else 500

RUTA_DATOS = Path(os.environ.get("EEG_DATA", r"C:\Users\migue\Desktop\TFG\EEG-Data"))
RUTA_MATRIZ = RUTA_DATOS / "processed" / "matriz_rf.parquet"
RUTA_MATRIZ_EPOCAS = RUTA_DATOS / "processed" / "matriz_modelado.parquet"
RUTA_GRABACIONES = RUTA_DATOS / "metadata" / "labels_grabaciones.csv"
if RAPIDO:
    _RUTA_RAPIDO = Path(tempfile.gettempdir()) / "tareas_ejecutivas_rapido"
    RUTA_RESULTADOS = _RUTA_RAPIDO / "tareas_ejecutivas_rapido.json"
    RUTA_FIGURAS = _RUTA_RAPIDO / "figures"
else:
    RUTA_RESULTADOS = RUTA_DATOS / "processed" / "tareas_ejecutivas.json"
    RUTA_FIGURAS = Path(__file__).resolve().parents[1] / "resultados" / "figuras"

REGION_THETA, BANDA_THETA = "frontal", "theta"
REGION_ALFA, BANDA_ALFA = "posterior", "alfa"

# Criterios de coherencia neurofisiológica, declarados ANTES de ejecutar.
#  (a) los dos modelos superan el azar: el límite inferior del IC95 de la
#      exactitud balanceada está por encima de 1/5 = 0.200.
#  (b) los dos pares cognitivamente gemelos -PASAT-PVSAT (misma tarea de suma
#      serial, distinta modalidad) y STROOP-MSIT (las dos de interferencia)-
#      están entre los 3 pares más confundidos de los 10 (regresión logística).
#  (c) PVTB es la clase con mayor exhaustividad: es la tarea más distinta de
#      las otras cuatro (vigilancia sostenida, sin memoria de trabajo ni
#      conflicto de respuesta), y en la Fase 18 fue la única tarea que el
#      modelo confundía a menudo con reposo.
#  (d) dirección: el coeficiente neto de las features `_media` de theta
#      frontal en la logística es MAYOR para PASAT y PVSAT (máxima carga de
#      memoria de trabajo) que para PVTB (vigilancia), en TODOS los pliegues.
#  (e) la confusión sigue el parecido cognitivo más que la posición en el
#      protocolo: rho(confusión, similitud) > 0 y |rho(confusión, similitud)|
#      > |rho(confusión, -distancia en el protocolo)|.
#
# Salvedad declarada junto a ellos: (b) y (e) están parcialmente confundidos
# con la duración de la grabación (STROOP y MSIT duran lo mismo -60 s-, y
# PASAT y PVSAT también -150 s-), que es exactamente lo que mide el control de
# duración (sección 8.2). Se citan siempre juntos.
PARES_GEMELOS = ("PASAT-PVSAT", "STROOP-MSIT")
RANGO_GEMELOS = 3

# Paleta cualitativa de 5 colores, resistente a daltonismo (Okabe & Ito, 2008),
# una por tarea, en el orden del protocolo. Se usa en las figuras de esta
# sección; no se ha pasado por el validador de la guía de visualización (a
# diferencia de las Fases 17/18, que usaban solo 2 colores), así que se declara
# como elección razonable, no como validación formal.
COLOR_POR_CLASE = {
    "STROOP": "#0072B2", "PASAT": "#E69F00", "PVTB": "#009E73",
    "MSIT": "#D55E00", "PVSAT": "#CC79A7",
}
COLOR_SIMILITUD = {3: "#b0413e", 2: "#d99a3d", 1: "#5b7e93"}
COLOR_GRIS = "#c3c2b7"
COLOR_TEXTO = "#0b0b0b"
COLOR_TEXTO_2 = "#52514e"
COLOR_REJILLA = "#e0e0dc"
RAMPA_AZUL = LinearSegmentedColormap.from_list(
    "azul_secuencial", ["#eef4fc", "#2a78d6", "#0f3d80"])

MODELOS = ("Regresión logística", "Random Forest")


def _titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def logistica():
    """Coeficientes con signo por clase (softmax); permite ver dirección."""
    return modelo_logistico_multinomial()


def bosque():
    """Misma configuración que el Random Forest de las Fases 14, 14b, 17 y 18."""
    return modelo_bosque_multiclase(n_estimators=N_ARBOLES, semilla=SEMILLA)


FABRICAS = {"Regresión logística": logistica, "Random Forest": bosque}


def _ba(y_true, y_pred):
    return metricas_multiclase(y_true, y_pred)["exactitud_balanceada"]


def _d_de_cohen(y_binaria, valores):
    return float(tamanos_de_efecto(pd.DataFrame({"x": valores}), y_binaria).iloc[0])


def _a_python(obj):
    """Conversor para `json.dumps`: tipos de numpy y estructuras de pandas."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="index")
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"No serializable: {type(obj)}")


def _volcar(resultados):
    """Guarda el JSON tal como está. Se llama tras cada sección: si algo falla
    a mitad de la ejecución, lo ya calculado no se pierde."""
    RUTA_RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    RUTA_RESULTADOS.write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False, default=_a_python),
        encoding="utf-8")


def evaluar(X, y, g, modelo, nombre, con_ic=True):
    """Valida por persona, calcula el IC bootstrap por persona de la
    exactitud balanceada amontonada y lo devuelve."""
    r = evaluar_multiclase(X, y, g, modelo, nombre, n_splits=N_PLIEGUES)
    if con_ic:
        boot = bootstrap_por_persona(y, r["y_pred"], g, _ba,
                                     n_replicas=N_BOOTSTRAP, semilla=SEMILLA)
        r["ic95_ba"] = boot["ic95"]
    return r


def _informe(r, detalle=True):
    m = r["metricas"]
    print(f"\n--- {m['modelo']} ---")
    print(f"  Exactitud balanceada, media por pliegue : "
          f"{m['exactitud_balanceada_media']:.4f}  (SD {m['exactitud_balanceada_sd']:.4f})")
    print(f"  Exactitud balanceada, amontonada        : {m['exactitud_balanceada']:.4f}")
    if "ic95_ba" in r:
        lo, hi = r["ic95_ba"]
        print(f"    IC95 (bootstrap por persona)          : [{lo:.4f}, {hi:.4f}]")
    print(f"  Exactitud simple                        : {m['exactitud']:.4f}")
    print(f"  F1 macro = {m['f1_macro']:.4f}   Kappa de Cohen = {m['kappa']:.4f}")
    if not detalle:
        return
    print("  Por clase:")
    print(f"    {'tarea':<8}{'precisión':>11}{'exhaustividad':>15}{'f1':>8}{'n':>6}")
    for c in CONDICIONES_EJECUTIVAS:
        f = m["por_clase"][c]
        print(f"    {c:<8}{f['precision']:>11.3f}{f['exhaustividad']:>15.3f}"
              f"{f['f1']:>8.3f}{f['soporte']:>6d}")
    print("  BA por pliegue  : "
          + "  ".join(f"{b:.3f}" for b in m["exactitud_balanceada_pliegues"]))


def _guardar(r):
    """Lo que se archiva de una evaluación (sin los vectores por fila)."""
    salida = {"metricas": dict(r["metricas"])}
    if "ic95_ba" in r:
        salida["ic95_ba"] = list(r["ic95_ba"])
    return salida


def direccion_del_efecto(X, y, g):
    """
    Coeficiente neto de theta frontal y de alfa posterior, por clase y por
    pliegue, con la logística multinomial. Para el criterio (d): el
    coeficiente de theta frontal debe ser mayor en PASAT y en PVSAT que en
    PVTB, en todos los pliegues.
    """
    coef = coeficientes_por_clase_y_pliegue(X, y, g, logistica(), n_splits=N_PLIEGUES)
    neto_theta = coeficiente_neto_por_clase(coef, REGION_THETA, BANDA_THETA)
    neto_alfa = coeficiente_neto_por_clase(coef, REGION_ALFA, BANDA_ALFA)
    cumple_d = bool((neto_theta.loc["PASAT"] > neto_theta.loc["PVTB"]).all()
                    and (neto_theta.loc["PVSAT"] > neto_theta.loc["PVTB"]).all())
    return {"coef": coef, "neto_theta": neto_theta, "neto_alfa": neto_alfa,
           "cumple_d": cumple_d}


def _imprimir_direccion(d):
    print("    Coeficiente neto de theta frontal, por clase (media entre pliegues):")
    for c in CONDICIONES_EJECUTIVAS:
        fila = d["neto_theta"].loc[c]
        por_pliegue = "  ".join(f"{v:+.3f}" for v in fila)
        print(f"      {c:<8} {fila.mean():+.3f}   (por pliegue: {por_pliegue})")
    print(f"    (d) PASAT y PVSAT por encima de PVTB en todos los pliegues: "
          f"{'SI' if d['cumple_d'] else 'NO'}")


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------

def _estilo(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=COLOR_REJILLA, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(COLOR_REJILLA)
    ax.tick_params(colors=COLOR_TEXTO_2, labelsize=9.5, length=0)


def _guardar_figura(fig, ruta):
    fig.savefig(ruta, facecolor="white")
    plt.close(fig)
    print(f"  figura -> {ruta}")


def figura_confusion(modelos, n, ruta):
    """
    Matrices de confusión 5x5 de los dos modelos, normalizadas por fila (clase
    real) y con la misma escala.
    """
    fig, ejes = plt.subplots(1, len(modelos), figsize=(13.6, 6.6), dpi=200)
    fig.patch.set_facecolor("white")
    for ax, (nombre, r) in zip(np.atleast_1d(ejes), modelos):
        cm = matriz_confusion_normalizada(r["y_true"], r["y_pred"])
        valores = cm.to_numpy()
        ax.pcolormesh(np.arange(6) - 0.5, np.arange(6) - 0.5, valores,
                     cmap=RAMPA_AZUL, vmin=0, vmax=1, edgecolors="white", linewidth=4)
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(4.5, -0.5)
        ax.set_aspect("equal")
        for i in range(5):
            for j in range(5):
                v = valores[i, j]
                if v < 0.005:
                    continue
                color = "white" if v > 0.55 else COLOR_TEXTO
                ax.text(j, i, f"{100 * v:.0f} %", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)
        ax.set_xticks(range(5), CONDICIONES_EJECUTIVAS, rotation=30, ha="right")
        ax.set_yticks(range(5), CONDICIONES_EJECUTIVAS)
        ax.grid(False)
        ax.tick_params(which="both", length=0, colors=COLOR_TEXTO_2, labelsize=10.5)
        for lado in ax.spines.values():
            lado.set_visible(False)
        ax.set_xlabel("Clase predicha", fontsize=10.5, color=COLOR_TEXTO_2, labelpad=8)
        ax.set_ylabel("Clase real", fontsize=10.5, color=COLOR_TEXTO_2, labelpad=8)
        m = r["metricas"]
        lo, hi = r["ic95_ba"]
        ax.set_title(nombre, fontsize=12, color=COLOR_TEXTO, loc="left", pad=32,
                     fontweight="bold")
        ax.text(0, 1.055, f"exactitud balanceada {m['exactitud_balanceada_media']:.3f}"
                f" · IC95 [{lo:.3f}, {hi:.3f}]  (azar 0.200)",
                transform=ax.transAxes, fontsize=9.5, color=COLOR_TEXTO_2, va="bottom")
    fig.suptitle("Clasificación de las cinco tareas ejecutivas: matriz de confusión",
                 x=0.012, ha="left", fontsize=13.5, color=COLOR_TEXTO, fontweight="bold")
    fig.text(0.012, 0.012,
             f"Validación cruzada de {N_PLIEGUES} pliegues agrupada por persona; "
             f"n = {n} grabaciones de train + val (el conjunto de test permanece sin abrir).\n"
             "El color es el porcentaje de cada clase real; cada fila suma 100 %.",
             fontsize=8, color=COLOR_TEXTO_2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.11, 1, 0.90))
    _guardar_figura(fig, ruta)


def figura_pares(sep, pares, ruta):
    """
    Separabilidad uno-contra-uno (izquierda) y confusión real dentro del
    problema de 5 clases (derecha), ambas ordenadas de más a menos parecido
    cognitivo declarado antes de ejecutar. Si el parecido cognitivo importa,
    los pares de arriba deberían tener menos separabilidad y más confusión.
    """
    orden = list(pares.sort_values("similitud_cognitiva", ascending=False).index)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 6.4), dpi=200, sharey=True)
    fig.patch.set_facecolor("white")
    for i, par in enumerate(orden):
        sim = int(pares.loc[par, "similitud_cognitiva"])
        color = COLOR_SIMILITUD[sim]
        lo, hi = sep.loc[par, "ic95"]
        ba = sep.loc[par, "exactitud_balanceada"]
        ax1.plot([lo, hi], [i, i], color=COLOR_TEXTO_2, linewidth=1.5,
                 solid_capstyle="round", zorder=4)
        ax1.plot([ba], [i], marker="o", markersize=9.5, color=color,
                 markeredgecolor="white", markeredgewidth=1.8, linestyle="none", zorder=5)
        ax2.barh(i, pares.loc[par, "confusion_simetrica"], color=color, height=0.62,
                 zorder=3)
    _estilo(ax1)
    _estilo(ax2)
    ax2.grid(axis="y", visible=False)
    ax1.axvline(0.5, color=COLOR_TEXTO_2, linewidth=1.0, linestyle="--", zorder=2)
    ax1.set_xlim(0.35, 1.0)
    ax1.set_yticks(range(len(orden)), orden)
    ax1.set_ylim(len(orden) - 0.5, -0.5)
    ax1.tick_params(axis="y", labelsize=10, colors=COLOR_TEXTO)
    ax1.set_xlabel("Separabilidad 1 contra 1 (exactitud balanceada)", fontsize=10,
                   color=COLOR_TEXTO_2)
    ax2.set_xlabel("Confusión dentro del modelo de 5 clases", fontsize=10,
                   color=COLOR_TEXTO_2)
    ax1.set_title("Cuanto más fácil de separar el par...", fontsize=11.5, loc="left",
                  color=COLOR_TEXTO, fontweight="bold")
    ax2.set_title("...menos debería confundirse en la práctica", fontsize=11.5,
                  loc="left", color=COLOR_TEXTO, fontweight="bold")
    manijas = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=9.5,
               markerfacecolor=COLOR_SIMILITUD[s], markeredgecolor="white",
               markeredgewidth=1.8, label=f"similitud cognitiva {s}")
        for s in (3, 2, 1)
    ]
    fig.legend(handles=manijas, loc="upper right", bbox_to_anchor=(0.99, 0.985),
               fontsize=9, frameon=True, framealpha=0.95, edgecolor=COLOR_REJILLA,
               facecolor="white", borderpad=0.7)
    fig.suptitle("Qué pares de tareas se confunden, y si es por parecido cognitivo",
                 x=0.012, ha="left", fontsize=13.5, color=COLOR_TEXTO,
                 fontweight="bold", y=0.985)
    fig.text(0.012, 0.012,
             "Train + val. Izquierda: exactitud balanceada de un clasificador binario logístico "
             "para ese par, agrupado por persona (IC95). Derecha: fracción\nde esas dos condiciones "
             "que el modelo de 5 clases intercambia entre sí. Pares ordenados de mayor a menor "
             "similitud cognitiva declarada antes de ejecutar.",
             fontsize=8, color=COLOR_TEXTO_2, linespacing=1.4)
    fig.subplots_adjust(left=0.16, right=0.97, wspace=0.08, top=0.80, bottom=0.15)
    _guardar_figura(fig, ruta)


def figura_canales(tabla, ruta):
    """
    d de Cohen una-contra-resto de cada una de las 5 tareas, por canal y por
    banda (theta y alfa, potencia relativa). Dos paneles con un mapa de calor
    canales x tareas; colormap divergente centrado en 0.
    """
    orden_canales = [c for r in ("frontal", "frontocentral", "temporal", "posterior")
                     for c in REGIONES[r]]
    fig, ejes = plt.subplots(1, 2, figsize=(11.6, 7.4), dpi=200)
    fig.patch.set_facecolor("white")
    limite = float(tabla["d"].abs().max())
    imagen = None
    for ax, banda in zip(ejes, ("theta", "alfa")):
        sub = tabla[tabla["banda"] == banda].pivot(index="canal", columns="clase",
                                                    values="d")
        sub = sub.reindex(index=orden_canales, columns=list(CONDICIONES_EJECUTIVAS))
        imagen = ax.imshow(sub.to_numpy(), cmap="RdBu_r", vmin=-limite, vmax=limite,
                           aspect="auto")
        ax.set_xticks(range(5), CONDICIONES_EJECUTIVAS, rotation=30, ha="right",
                      fontsize=9.5, color=COLOR_TEXTO)
        ax.set_yticks(range(len(orden_canales)), orden_canales, fontsize=9.5,
                      color=COLOR_TEXTO)
        for i in range(sub.shape[0]):
            for j in range(sub.shape[1]):
                v = sub.iat[i, j]
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.3,
                        color="white" if abs(v) > limite * 0.55 else COLOR_TEXTO)
        ax.set_title(f"Banda {banda}", fontsize=12, color=COLOR_TEXTO, loc="left",
                    fontweight="bold", pad=10)
    # Barra de color en un eje propio, con posición fija (`subplots_adjust` +
    # `add_axes`) en vez de `tight_layout`. Combinar `fig.colorbar(ax=lista)`
    # con `tight_layout` recalculaba la posición de la barra encima del panel
    # de alfa (matplotlib avisa de "Axes not compatible with tight_layout" y
    # la vuelve a colocar mal); con un eje reservado no hay recálculo.
    fig.subplots_adjust(left=0.075, right=0.855, top=0.86, bottom=0.11, wspace=0.35)
    cax = fig.add_axes((0.875, 0.15, 0.02, 0.62))
    cbar = fig.colorbar(imagen, cax=cax)
    cbar.set_label("d de Cohen (esa tarea frente a las otras 4)", fontsize=9.5,
                   color=COLOR_TEXTO_2)
    fig.suptitle("Efecto de cada tarea ejecutiva sobre theta y alfa, por canal",
                 x=0.02, ha="left", fontsize=13.5, color=COLOR_TEXTO, fontweight="bold")
    fig.text(0.02, 0.012,
             "Train + val (el conjunto de test permanece sin abrir). Potencia relativa media de la "
             "grabación. d de Cohen de una clase contra las otras cuatro juntas.",
             fontsize=8, color=COLOR_TEXTO_2)
    _guardar_figura(fig, ruta)


# --------------------------------------------------------------------------
# Suplemento por canal
# --------------------------------------------------------------------------

def analisis_por_canal(datos, y):
    """
    Suplemento: qué canales llevan el efecto de cada tarea.

    Reagrega `matriz_modelado.parquet` con `construir_matriz_rf` y una región
    por canal (la variante que documenta `agregar_por_region`), restringida a
    las mismas grabaciones del análisis principal. Para cada canal y banda
    (theta, alfa) da el d de Cohen una-contra-resto de cada una de las 5
    clases, con IC95 remuestreando personas.
    """
    epocas = pd.read_parquet(
        RUTA_MATRIZ_EPOCAS,
        filters=[("archivo_origen", "in", datos["archivo"].tolist())])
    grabaciones = pd.read_csv(RUTA_GRABACIONES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        por_canal, _ = construir_matriz_rf(
            epocas, grabaciones, regiones={c: (c,) for c in CANALES_EPOC})
    del epocas
    por_canal = por_canal.set_index("archivo")
    faltan = set(datos["archivo"]) - set(por_canal.index)
    if faltan:
        raise ValueError(f"{len(faltan)} grabaciones del análisis no están en la "
                         f"matriz por canal: {sorted(faltan)[:3]}")
    por_canal = por_canal.loc[datos["archivo"]].reset_index(drop=True)
    g = datos["persona_uid"].to_numpy()
    region_de = {c: r for r, canales in REGIONES.items() for c in canales}

    filas = []
    for canal in CANALES_EPOC:
        for banda in ("theta", "alfa"):
            columna = f"{canal}_{banda}_rel_media"
            valores = por_canal[columna].to_numpy(dtype=float)
            for i, clase in enumerate(CONDICIONES_EJECUTIVAS):
                y_bin = (y == i).astype(int)
                d = _d_de_cohen(y_bin, valores)
                ic = bootstrap_por_persona(y_bin, valores, g, _d_de_cohen,
                                           n_replicas=N_BOOTSTRAP_CANALES,
                                           semilla=SEMILLA)["ic95"]
                filas.append({"canal": canal, "region": region_de[canal],
                             "banda": banda, "clase": clase, "d": d,
                             "ic95_inf": ic[0], "ic95_sup": ic[1]})
    return pd.DataFrame(filas)


def main():
    resultados = {"modo_rapido": RAPIDO}
    if RAPIDO:
        print("#" * 78)
        print("# MODO RÁPIDO: parámetros mínimos. Estas cifras NO son resultados.")
        print(f"# Salidas en {RUTA_RESULTADOS.parent}")
        print("#" * 78)

    # ----------------------------------------------------------------------
    _titulo("1. Datos - las cinco tareas ejecutivas de train + val")
    matriz = pd.read_parquet(RUTA_MATRIZ)
    n_fase13 = len(columnas_de_features(matriz))
    print(f"Matriz completa      : {matriz.shape[0]} grabaciones, {n_fase13} features (Fase 13)")

    datos, meta = preparar_tareas_ejecutivas(matriz)
    assert set(datos["split"]) <= {"train", "val"}, "el conjunto de test se ha colado"
    print(f"Apartado (test)      : {meta['n_apartadas']} grabaciones, "
          f"{meta['n_personas_apartadas']} personas - no se vuelve a leer")
    print(f"Sin persona_uid      : {meta['n_sin_persona_uid']} grabaciones descartadas")
    print(f"En análisis          : {meta['n_grabaciones']} grabaciones, "
          f"{meta['n_personas']} personas, {meta['n_sesiones']} sesiones")
    print(f"  por condición      : "
          f"{ {c: meta['por_condicion'][c] for c in CONDICIONES_EJECUTIVAS} }")
    print(f"  desbalance         : {meta['ratio_desbalance']:.3f}:1  "
          f"(mayoritaria {meta['clase_mayoritaria']}, minoritaria {meta['clase_minoritaria']})")
    print(f"  por split          : {meta['por_split']}")
    por_persona = datos.groupby("persona_uid").size()
    print(f"  grabaciones/persona: mediana {por_persona.median():.0f}, "
          f"rango [{por_persona.min()}, {por_persona.max()}]")

    y = etiqueta_multiclase(datos)
    g = datos["persona_uid"].to_numpy()
    columnas = columnas_de_features_ejecutivas(datos)
    X = datos[columnas]

    excluidas = [c for c in columnas_de_features(datos) if c not in columnas]
    print(f"\nFeatures de la Fase 13 : {n_fase13}")
    print(f"Features en X          : {len(columnas)}   (excluidas {len(excluidas)})")
    print(f"  {COLUMNA_FUGA:<26} ES LA ETIQUETA: dentro de las 5 tareas toma un valor "
          f"CONSTANTE distinto por cada una")
    for condicion in CONDICIONES_EJECUTIVAS:
        valores = sorted(datos.loc[datos["condicion"] == condicion, COLUMNA_FUGA]
                         .unique().tolist())
        print(f"    {condicion:<7} -> {valores}")
    for c in COLUMNAS_PROCESAMIENTO:
        print(f"  {c:<26} contador de procesamiento, no señal cerebral")
    verificar_sin_fuga(columnas)
    assert COLUMNA_FUGA not in columnas
    print(f"\nComprobado: `{COLUMNA_FUGA}` no está en las {len(columnas)} features de X.")

    duplicadas = columnas_duplicadas(X)
    print(f"\nColumnas de X que son COPIAS EXACTAS de otra: {len(duplicadas)} "
          f"({len(columnas) - len(duplicadas)} columnas distintas de {len(columnas)})")
    for conservada, copia in duplicadas:
        print(f"    {copia:<32} = {conservada}")

    print("\nReparto de clases por pliegue de la CV agrupada por persona:")
    from eeg_library.modelado.reposo import pliegue_por_fila
    pliegues = pliegue_por_fila(g, N_PLIEGUES)
    tabla_pliegues = pd.crosstab(pliegues, datos["condicion"].to_numpy())
    print(tabla_pliegues.to_string())

    resultados["datos"] = {
        **meta, "n_features_fase13": n_fase13, "n_features_x": len(columnas),
        "columnas_excluidas": excluidas, "columnas_x": columnas,
        "columnas_duplicadas": [list(par) for par in duplicadas],
    }
    _volcar(resultados)

    # ----------------------------------------------------------------------
    _titulo("2. Baselines triviales")
    print("Un clasificador que no usa X. Con 5 clases casi equilibradas, la exactitud")
    print("balanceada de azar es 1/5 = 0.200 exacto.")
    modelos = {}
    for nombre, modelo in (
            ("Baseline: clase mayoritaria",
             DummyClassifier(strategy="most_frequent")),
            ("Baseline: azar estratificado",
             DummyClassifier(strategy="stratified", random_state=SEMILLA)),
            ("Baseline: uniforme",
             DummyClassifier(strategy="uniform", random_state=SEMILLA))):
        modelos[nombre] = evaluar(X, y, g, modelo, nombre)
        _informe(modelos[nombre], detalle=False)
    sanidad = all(abs(r["metricas"]["exactitud_balanceada_media"] - 0.2) < 0.05
                  for r in modelos.values())
    print(f"\n  [{'SI' if sanidad else 'NO'}] Los tres baselines dan ~0.200 de exactitud balanceada")
    resultados["baselines"] = {n: _guardar(r) for n, r in modelos.items()}
    resultados["baselines_sanos"] = bool(sanidad)

    # ----------------------------------------------------------------------
    _titulo("3. Modelo A - regresión logística multinomial")
    print("Imputación por mediana + tipificación + LogisticRegression(class_weight=")
    print("'balanced'), softmax automático con 5 clases. Sus coeficientes (uno por")
    print("clase) permiten comprobar la dirección del efecto de cada tarea.")
    modelos["Regresión logística"] = evaluar(X, y, g, logistica(), "Regresión logística")
    _informe(modelos["Regresión logística"])

    # ----------------------------------------------------------------------
    _titulo("4. Modelo B - Random Forest")
    print(f"RandomForestClassifier({N_ARBOLES} árboles, hoja>=5, class_weight='balanced'): la misma")
    print("configuración que las Fases 14, 14b, 17 y 18, para que la cifra sea comparable.")
    modelos["Random Forest"] = evaluar(X, y, g, bosque(), "Random Forest")
    _informe(modelos["Random Forest"])
    _volcar({**resultados, "modelos": {n: _guardar(r) for n, r in modelos.items()}})

    print("\n4.1 Comparación PAREADA entre los dos modelos (exactitud balanceada)")
    comparaciones = []
    for a, b in (("Regresión logística", "Random Forest"),
                 ("Regresión logística", "Baseline: azar estratificado"),
                 ("Random Forest", "Baseline: azar estratificado")):
        c = bootstrap_diferencia_por_persona(
            y, modelos[a]["y_pred"], modelos[b]["y_pred"], g, _ba,
            n_replicas=N_BOOTSTRAP, semilla=SEMILLA)
        comparaciones.append({"a": a, "b": b, "diferencia": c["diferencia"],
                              "ic95": list(c["ic95"]), "prop_a_mayor": c["prop_positivas"]})
        print(f"    {a} - {b}: {c['diferencia']:+.4f}  IC95 [{c['ic95'][0]:+.4f}, "
              f"{c['ic95'][1]:+.4f}]  (A > B en el {c['prop_positivas']:.1%} de las réplicas)")
    resultados["comparaciones_pareadas"] = comparaciones
    resultados["modelos"] = {n: _guardar(r) for n, r in modelos.items()}

    # Criterio (a): las dos IC95 por encima de 0.200.
    criterio_a = all(modelos[n]["ic95_ba"][0] > 0.2 for n in MODELOS)
    print(f"\n  [{'SI' if criterio_a else 'NO'}] (a) el límite inferior del IC95 de los dos "
          f"modelos supera 0.200")
    _volcar(resultados)

    # ----------------------------------------------------------------------
    _titulo("5. Métricas por clase y matriz de confusión")
    matrices = {}
    for nombre in MODELOS:
        r = modelos[nombre]
        cm = matriz_confusion_normalizada(r["y_true"], r["y_pred"])
        matrices[nombre] = cm
        print(f"\n--- {nombre}: matriz de confusión (filas = real, % de esa fila) ---")
        print((cm * 100).round(1).to_string())
    mejor_exhaustividad = {
        n: max(modelos[n]["metricas"]["por_clase"],
              key=lambda c: modelos[n]["metricas"]["por_clase"][c]["exhaustividad"])
        for n in MODELOS
    }
    print("\nClase con mayor exhaustividad, por modelo:")
    for n, c in mejor_exhaustividad.items():
        print(f"    {n:<22} {c}  "
              f"({modelos[n]['metricas']['por_clase'][c]['exhaustividad']:.3f})")
    criterio_c = mejor_exhaustividad["Regresión logística"] == "PVTB"
    print(f"\n  [{'SI' if criterio_c else 'NO'}] (c) PVTB es la clase con mayor exhaustividad "
          f"(regresión logística)")
    resultados["matrices_confusion"] = {n: m.to_dict(orient="index")
                                        for n, m in matrices.items()}
    _volcar(resultados)

    # ----------------------------------------------------------------------
    _titulo("6. Coherencia neurofisiológica - qué pesa en el modelo")
    print("Criterios declarados antes de ejecutar (ver las constantes del guion):")
    print("  (a) los dos modelos superan el azar (0.200) con el IC95")
    print("  (b) los pares gemelos (PASAT-PVSAT, STROOP-MSIT) están entre los 3 más")
    print("      confundidos de los 10 (sección 7)")
    print("  (c) PVTB es la clase con mayor exhaustividad")
    print("  (d) theta frontal es mayor en PASAT y PVSAT que en PVTB, en todos los pliegues")
    print("  (e) la confusión sigue más el parecido cognitivo que la distancia en el")
    print("      protocolo (sección 7)")

    print("\n6.1 Importancia por permutación del Random Forest, OUT-OF-FOLD")
    print(f"    ({N_REPETICIONES_PERMUTACION} barajados x {N_PLIEGUES} pliegues)")
    imp = importancia_permutacion_cv(X, y, g, bosque(), n_splits=N_PLIEGUES,
                                     n_repeticiones=N_REPETICIONES_PERMUTACION,
                                     semilla=SEMILLA)
    ranking = imp.sort_values("caida_media", ascending=False)
    print(f"\n    {'feature':<34}{'caída BA':>10}{'SD pliegues':>13}")
    for nombre, f in ranking.head(15).iterrows():
        print(f"    {nombre:<34}{f['caida_media']:>+10.4f}{f['sd_entre_pliegues']:>13.4f}")

    print("\n6.2 Importancia agregada por bloque (fracción del total)")
    por_bloque = importancia_por_bloque(imp["caida_media"])
    print("    por banda   : " + "  ".join(
        f"{k} {v:.1%}" for k, v in sorted(por_bloque["por_banda"].items(),
                                          key=lambda kv: -kv[1])))
    print("    por región  : " + "  ".join(
        f"{k} {v:.1%}" for k, v in sorted(por_bloque["por_region"].items(),
                                          key=lambda kv: -kv[1])))

    print("\n6.3 Permutación CONJUNTA de bloques enteros (Random Forest)")
    bloques = {}
    for banda in BANDAS_DEFECTO:
        bloques[f"banda/{banda}"] = columnas_del_bloque(columnas, banda=banda)
    for region in REGIONES:
        bloques[f"region/{region}"] = columnas_del_bloque(columnas, region=region)
    conjunta = importancia_permutacion_cv(X, y, g, bosque(), bloques=bloques,
                                          n_splits=N_PLIEGUES,
                                          n_repeticiones=N_REPETICIONES_PERMUTACION,
                                          semilla=SEMILLA)
    print(f"\n    {'bloque':<24}{'caída BA':>11}{'SD pliegues':>13}")
    for nombre, f in conjunta.sort_values("caida_media", ascending=False).iterrows():
        print(f"    {nombre:<24}{f['caida_media']:>+11.4f}{f['sd_entre_pliegues']:>13.4f}")

    print("\n6.4 Dirección del efecto (coeficientes de la logística multinomial)")
    direccion = direccion_del_efecto(X, y, g)
    _imprimir_direccion(direccion)

    print("\n6.5 d de Cohen una-contra-resto de theta frontal y alfa posterior")
    efectos = tamanos_de_efecto_una_contra_resto(
        X, y, condiciones=CONDICIONES_EJECUTIVAS)
    theta_media = columnas_media_del_bloque(columnas, REGION_THETA, BANDA_THETA)
    alfa_media = columnas_media_del_bloque(columnas, REGION_ALFA, BANDA_ALFA)
    for c in CONDICIONES_EJECUTIVAS:
        vt = efectos[c][theta_media].mean()
        va = efectos[c][alfa_media].mean()
        print(f"    {c:<8} theta frontal {vt:+.3f}   alfa posterior {va:+.3f}")

    print(f"\n6.6 Veredicto PARCIAL frente a (a), (c) y (d) - (b) y (e) van en 8.4")
    for nombre, cumple in (
            ("(a) los dos modelos superan el azar con el IC95", criterio_a),
            ("(c) PVTB es la clase con mayor exhaustividad", criterio_c),
            ("(d) theta frontal mayor en PASAT/PVSAT que en PVTB, todos los pliegues",
             direccion["cumple_d"])):
        print(f"  [{'SI' if cumple else 'NO'}] {nombre}")

    resultados["coherencia"] = {
        "importancia_por_permutacion": ranking.reset_index().rename(
            columns={"index": "feature"}).to_dict(orient="records"),
        "importancia_por_bloque": por_bloque,
        "importancia_conjunta": conjunta.reset_index().rename(
            columns={"index": "bloque"}).to_dict(orient="records"),
        "neto_theta_frontal": direccion["neto_theta"].to_dict(orient="index"),
        "neto_alfa_posterior": direccion["neto_alfa"].to_dict(orient="index"),
        "cumple_d": direccion["cumple_d"],
        "d_cohen_una_contra_resto": {c: efectos[c].to_dict() for c in CONDICIONES_EJECUTIVAS},
        "criterio_a": criterio_a, "criterio_c": criterio_c,
        "mejor_exhaustividad": mejor_exhaustividad,
    }
    _volcar(resultados)

    # ----------------------------------------------------------------------
    _titulo("7. Qué tareas se confunden")
    r_log = modelos["Regresión logística"]
    pares = confusion_por_pares(r_log["y_true"], r_log["y_pred"])
    print("7.1 Confusión por pares (regresión logística), de más a menos confundido:")
    print(f"    {'par':<16}{'confusión':>11}{'asimetría':>11}{'similitud':>11}{'dist. protoc.':>14}")
    orden_confusion = pares.sort_values("confusion_simetrica", ascending=False)
    for par, f in orden_confusion.iterrows():
        print(f"    {par:<16}{f['confusion_simetrica']:>11.3f}{f['asimetria']:>+11.3f}"
              f"{int(f['similitud_cognitiva']):>11d}{int(f['distancia_protocolo']):>14d}")
    top3 = list(orden_confusion.head(RANGO_GEMELOS).index)
    criterio_b = all(par in top3 for par in PARES_GEMELOS)
    print(f"\n    Los {RANGO_GEMELOS} pares más confundidos: {top3}")
    print(f"  [{'SI' if criterio_b else 'NO'}] (b) los pares gemelos {PARES_GEMELOS} están "
          f"entre ellos")

    print("\n7.2 Separabilidad uno-contra-uno (10 logísticas binarias)")
    sep = separabilidad_por_pares(X, y, g, logistica, n_splits=N_PLIEGUES,
                                  n_replicas=N_BOOTSTRAP_PARES, semilla=SEMILLA)
    print(f"    {'par':<16}{'n':>6}{'personas':>10}{'BA':>8}{'IC95':>19}")
    for par, f in sep.sort_values("exactitud_balanceada").iterrows():
        lo, hi = f["ic95"]
        print(f"    {par:<16}{int(f['n']):>6d}{int(f['n_personas']):>10d}"
              f"{f['exactitud_balanceada']:>8.3f}   [{lo:.3f}, {hi:.3f}]")

    print("\n7.3 Estructura de la confusión: qué la predice mejor")
    duraciones = distancias_de_duracion(datos)
    estructura = estructura_de_la_confusion(pares, duraciones)
    for clave, valor in estructura.items():
        if clave == "n_pares":
            continue
        print(f"    {clave:<42} rho = {valor:+.3f}")
    criterio_e = bool(
        estructura["confusion_vs_similitud_cognitiva"] > 0
        and abs(estructura["confusion_vs_similitud_cognitiva"])
        > abs(estructura["confusion_vs_distancia_protocolo"]))
    print(f"\n  [{'SI' if criterio_e else 'NO'}] (e) la confusión sigue más el parecido "
          f"cognitivo que la distancia en el protocolo")
    print("      Limitación de diseño: el orden del protocolo es fijo, así que la tarea")
    print("      coincide con el momento de la grabación; esta comparación no puede separar")
    print("      del todo similitud cognitiva de proximidad temporal (ver sección 12 del")
    print("      documento).")

    resultados["confusion_por_pares"] = pares.reset_index().to_dict(orient="records")
    resultados["separabilidad_por_pares"] = sep.reset_index().to_dict(orient="records")
    resultados["distancias_de_duracion"] = duraciones.reset_index().to_dict(orient="records")
    resultados["estructura_de_la_confusion"] = estructura
    resultados["criterio_b"] = criterio_b
    resultados["criterio_e"] = criterio_e
    _volcar(resultados)

    # ----------------------------------------------------------------------
    _titulo("8. Controles y sensibilidad")
    print("8.1 Control negativo: etiquetas barajadas DENTRO de cada persona")
    print(f"    {N_PERMUTACIONES_ETIQUETAS} barajados por modelo. Tiene que dar ~0.200.")
    nulos = {}
    for nombre in MODELOS:
        valores = []
        for s in range(N_PERMUTACIONES_ETIQUETAS):
            y_perm = permutar_etiquetas_dentro_de_persona(y, g, semilla=SEMILLA + s)
            r = evaluar_multiclase(X, y_perm, g, FABRICAS[nombre](), n_splits=N_PLIEGUES)
            valores.append(r["metricas"]["exactitud_balanceada_media"])
        valores = np.array(valores)
        real = modelos[nombre]["metricas"]["exactitud_balanceada_media"]
        n_supera = int((valores >= real).sum())
        p_emp = (1 + n_supera) / (1 + len(valores))
        sd = float(valores.std(ddof=1)) if len(valores) > 1 else float("nan")
        nulos[nombre] = {
            "media": float(valores.mean()), "sd": sd,
            "min": float(valores.min()), "max": float(valores.max()), "real": float(real),
            "n_nulos_iguales_o_mayores": n_supera, "p_empirico": float(p_emp),
            "valores": valores.tolist()}
        print(f"    {nombre:<22} nulo {valores.mean():.4f} (SD {sd:.4f}, "
              f"rango [{valores.min():.3f}, {valores.max():.3f}])  real {real:.4f}  "
              f"nulos >= real: {n_supera} de {len(valores)}")
    control_ok = all(abs(v["media"] - 0.2) < 0.05 for v in nulos.values())
    print(f"\n  [{'SI' if control_ok else 'NO'}] El nulo está centrado en 0.200")
    resultados["control_etiquetas_barajadas"] = {**nulos, "centrado_en_0_2": bool(control_ok)}
    _volcar(resultados)

    print("\n8.2 Control de duración: los 6 contadores de procesamiento, solos")
    print("    STROOP y MSIT duran ~60 s; PASAT, PVTB y PVSAT ~150 s (ESTADO.md). Si estos")
    print("    6 números, sin una sola feature espectral, ya distinguen las tareas, la")
    print("    duración es un confusor que hay que tener en cuenta al leer las secciones 6 y 7.")
    solo_contadores = {}
    for nombre in MODELOS:
        r = evaluar(datos[list(COLUMNAS_PROCESAMIENTO)], y, g, FABRICAS[nombre](), nombre)
        _informe(r, detalle=False)
        solo_contadores[nombre] = _guardar(r)
    resultados["control_duracion_contadores"] = solo_contadores
    _volcar(resultados)

    print("\n8.3 Sensibilidad: qué features entran a X")
    copias = {copia for _, copia in duplicadas}
    solo_media = [c for c in columnas if c.endswith("_media")]
    solo_sd = [c for c in columnas if c.endswith("_sd")]
    subconjuntos = {
        "Principal": columnas,
        "Principal sin las columnas duplicadas": [c for c in columnas if c not in copias],
        "Principal + 6 contadores de procesamiento":
            columnas_de_features_ejecutivas(datos, incluir_procesamiento=True),
        "Solo las features _media": solo_media,
        "Solo las features _sd": solo_sd,
    }
    for banda in BANDAS_DEFECTO:
        subconjuntos[f"Solo banda {banda}"] = columnas_del_bloque(columnas, banda=banda)
    for region in REGIONES:
        subconjuntos[f"Solo región {region}"] = columnas_del_bloque(columnas, region=region)

    print(f"\n    {'subconjunto':<44}{'n':>4}   {'modelo':<20}{'BA/pliegue':>11}"
          f"{'IC95 BA':>19}{'Δ vs principal (pareado, IC95)':<34}")
    sensibilidad = []
    for etiqueta, cols in subconjuntos.items():
        verificar_sin_fuga(cols)
        for nombre in MODELOS:
            r = evaluar(datos[cols], y, g, FABRICAS[nombre](), nombre, con_ic=False)
            boot = bootstrap_por_persona(y, r["y_pred"], g, _ba, n_replicas=N_BOOTSTRAP,
                                         semilla=SEMILLA)
            m = r["metricas"]
            if etiqueta == "Principal":
                delta, texto_delta = None, "-"
            else:
                delta = bootstrap_diferencia_por_persona(
                    y, r["y_pred"], modelos[nombre]["y_pred"], g, _ba,
                    n_replicas=N_BOOTSTRAP, semilla=SEMILLA)
                texto_delta = (f"{delta['diferencia']:+.4f} "
                               f"[{delta['ic95'][0]:+.3f}, {delta['ic95'][1]:+.3f}]")
            print(f"    {etiqueta:<44}{len(cols):>4d}   {nombre:<20}"
                  f"{m['exactitud_balanceada_media']:>11.4f}"
                  f"   [{boot['ic95'][0]:.3f}, {boot['ic95'][1]:.3f}]   {texto_delta}")
            sensibilidad.append({
                "subconjunto": etiqueta, "modelo": nombre, "n_features": len(cols),
                "ba_media_pliegues": m["exactitud_balanceada_media"],
                "ic95_ba": list(boot["ic95"]),
                "delta_vs_principal": (None if delta is None else {
                    "diferencia": delta["diferencia"], "ic95": list(delta["ic95"])}),
                "columnas": cols})
    resultados["sensibilidad"] = sensibilidad
    _volcar(resultados)

    print("\n8.4 Veredicto COMPLETO frente a los 5 criterios declarados")
    criterios = [
        ("(a) los dos modelos superan el azar con el IC95", criterio_a),
        ("(b) los pares gemelos están entre los 3 más confundidos", criterio_b),
        ("(c) PVTB es la clase con mayor exhaustividad", criterio_c),
        ("(d) theta frontal mayor en PASAT/PVSAT que en PVTB, todos los pliegues",
         direccion["cumple_d"]),
        ("(e) la confusión sigue más el parecido cognitivo que la distancia en el protocolo",
         criterio_e),
    ]
    for nombre, cumple in criterios:
        print(f"  [{'SI' if cumple else 'NO'}] {nombre}")
    n_ok = sum(c for _, c in criterios)
    print(f"\n  Criterios cumplidos: {n_ok} de {len(criterios)}")
    print("  Salvedad: (b) y (e) están parcialmente confundidos con la duración de la")
    print("  grabación (sección 8.2), porque STROOP=MSIT y PASAT=PVSAT también comparten")
    print("  duración.")
    resultados["coherencia"]["veredicto"] = {
        "criterios": [{"criterio": n, "cumple": bool(c)} for n, c in criterios],
        "n_cumplidos": int(n_ok), "n_total": len(criterios)}
    _volcar(resultados)

    # ----------------------------------------------------------------------
    _titulo("9. Resolución por canal (suplemento)")
    print("matriz_rf.parquet agrega por región. Para ver qué canales llevan el efecto se")
    print("reagrega matriz_modelado.parquet con `construir_matriz_rf` y una región por canal.")
    canales = analisis_por_canal(datos, y)
    for banda in ("theta", "alfa"):
        print(f"\n  Banda {banda} (d de Cohen una-contra-resto por canal y clase):")
        tabla = canales[canales["banda"] == banda].pivot(index="canal", columns="clase",
                                                          values="d")
        tabla = tabla.reindex(columns=list(CONDICIONES_EJECUTIVAS))
        print(tabla.round(2).to_string())
    resultados["canales"] = canales.to_dict(orient="records")
    _volcar(resultados)

    # ----------------------------------------------------------------------
    _titulo("10. Figuras")
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    figura_confusion([(n, modelos[n]) for n in MODELOS], len(datos),
                     RUTA_FIGURAS / "tareas_ejecutivas_confusion.png")
    figura_pares(sep, pares, RUTA_FIGURAS / "tareas_ejecutivas_pares.png")
    figura_canales(canales, RUTA_FIGURAS / "tareas_ejecutivas_canales.png")

    # ----------------------------------------------------------------------
    _titulo("11. Resumen")
    filas = []
    for nombre, r in modelos.items():
        m = r["metricas"]
        filas.append({
            "clasificador": nombre,
            "BA/pliegue": f"{m['exactitud_balanceada_media']:.4f}",
            "IC95": (f"[{r['ic95_ba'][0]:.3f}, {r['ic95_ba'][1]:.3f}]"
                    if "ic95_ba" in r else "-"),
            "F1 macro": f"{m['f1_macro']:.4f}", "Kappa": f"{m['kappa']:.4f}"})
    print(pd.DataFrame(filas).to_string(index=False))
    print(f"\n  (azar de la exactitud balanceada con 5 clases = 0.200)")
    print(f"  Coherencia neurofisiológica: {n_ok} de {len(criterios)} criterios cumplidos")
    print(f"  Control de etiquetas barajadas centrado en 0.200: {'sí' if control_ok else 'NO'}")
    print(f"  Par más confundido: {orden_confusion.index[0]} "
          f"({orden_confusion.iloc[0]['confusion_simetrica']:.3f})")

    resultados["modelos"] = {n: _guardar(r) for n, r in modelos.items()}
    _volcar(resultados)
    print(f"\nResultados -> {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
