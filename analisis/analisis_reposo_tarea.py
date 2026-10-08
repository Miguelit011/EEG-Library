"""
Clasificación binaria: reposo (OA, OC) frente a tarea ejecutiva.

¿Distingue el pipeline una grabación de reposo (OA u OC) de una de tarea
ejecutiva (STROOP, PASAT, PVTB, MSIT, PVSAT) y con qué rendimiento? Es el
escalón siguiente al control OA/OC (`analisis_oa_oc.py`): allí el efecto era el
más grande y localizado del EEG de superficie (efecto Berger); aquí es un
efecto de carga cognitiva, más pequeño y más cercano en naturaleza a la fatiga
mental, que es la variable de interés del proyecto.

La unidad de análisis es la **grabación**: la condición es una propiedad de
cada registro. Las grabaciones de una misma persona no son independientes, así
que la validación cruzada y los intervalos agrupan por `persona_uid` (nunca por
`sujeto_id`). Las clases están desbalanceadas (~2.5 grabaciones de tarea por
cada una de reposo): se compensa en el entrenamiento (`class_weight="balanced"`)
y la métrica principal es la exactitud balanceada, cuyo azar es 0.5 exacto.

Dos amenazas condicionan la lectura y el guion las **mide** en vez de
esquivarlas: (1) el reposo incluye OC, con el alfa disparado por el efecto
Berger; (2) el orden del protocolo es fijo (el reposo ocupa siempre las
posiciones 1-2 y las tareas las 3-7), así que cualquier deriva dentro de la
sesión queda confundida con la condición. Los controles de la sección 8 las
acotan; no las eliminan.

**Solo usa `train` + `val`.** El conjunto de test se aparta en el paso 1 y no
vuelve a tocarse. `preparar_reposo_tarea` se niega a incluirlo sin permiso
explícito.

No modifica el pipeline congelado: todo sale de `matriz_rf.parquet` (Fase 13),
salvo el suplemento por canal, que reagrega `matriz_modelado.parquet` (Fase 11.5)
con la propia `construir_matriz_rf` y regiones de un solo canal.

Ejecutar (desde la raíz del repositorio), **una sola vez y nunca en paralelo**:
    python analisis/analisis_reposo_tarea.py > resultados/logs/reposo_tarea.log 2>&1

`--rapido` ejecuta el mismo código con parámetros mínimos y escribe en un
directorio temporal, para comprobar que el guion corre de principio a fin y
revisar las figuras. Sus cifras NO son resultados.

La ubicación del dataset se toma de la variable de entorno `EEG_DATA`, con
`C:\\Users\\migue\\Desktop\\TFG\\EEG-Data` por defecto (ver `analisis/README.md`).
"""

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
from scipy import stats
from sklearn.dummy import DummyClassifier

from eeg_library.modelado.reposo import (
    COLUMNA_FUGA,
    COLUMNAS_PROCESAMIENTO,
    UmbralUnaVariable,
    auc_roc,
    bloque_de_feature,
    bootstrap_diferencia_por_persona,
    bootstrap_por_persona,
    coeficientes_por_pliegue,
    columnas_del_bloque,
    columnas_duplicadas,
    evaluar_reposo,
    importancia_permutacion_cv,
    importancia_por_bloque,
    metricas_binarias,
    permutar_etiquetas_dentro_de_persona,
    tamanos_de_efecto,
    umbral_una_variable,
    verificar_sin_fuga,
)
from eeg_library.modelado.reposo_tarea import (
    coeficiente_neto_por_pliegue,
    columnas_de_features_tarea,
    columnas_media_del_bloque,
    comparar_condiciones,
    desglose_por_condicion,
    etiqueta_tarea,
    modelo_bosque,
    modelo_logistico,
    precision_media,
    preparar_reposo_tarea,
    signos_umbral_por_pliegue,
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
# archivar el resultado, que es el caso de uso que importa. `line_buffering`
# hace que el log vaya apareciendo mientras corre (son ~1 h de ejecución).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

RAPIDO = "--rapido" in sys.argv

SEMILLA = 42
N_PLIEGUES = 5
# Mismos parámetros de cómputo que el análisis OA/OC (Fase 17), para que las
# cifras sean comparables. En modo rápido se reducen al mínimo.
N_BOOTSTRAP = 40 if RAPIDO else 2000
N_BOOTSTRAP_CANALES = 40 if RAPIDO else 2000
N_REPETICIONES_PERMUTACION = 1 if RAPIDO else 5
N_PERMUTACIONES_ETIQUETAS = 2 if RAPIDO else 20
N_ARBOLES = 30 if RAPIDO else 500

RUTA_DATOS = Path(os.environ.get("EEG_DATA", r"C:\Users\migue\Desktop\TFG\EEG-Data"))
RUTA_MATRIZ = RUTA_DATOS / "processed" / "matriz_rf.parquet"
RUTA_MATRIZ_EPOCAS = RUTA_DATOS / "processed" / "matriz_modelado.parquet"
RUTA_GRABACIONES = RUTA_DATOS / "metadata" / "labels_grabaciones.csv"
if RAPIDO:
    _RUTA_RAPIDO = Path(tempfile.gettempdir()) / "reposo_tarea_rapido"
    RUTA_RESULTADOS = _RUTA_RAPIDO / "reposo_tarea_rapido.json"
    RUTA_FIGURAS = _RUTA_RAPIDO / "figures"
else:
    RUTA_RESULTADOS = RUTA_DATOS / "processed" / "reposo_tarea.json"
    RUTA_FIGURAS = Path(__file__).resolve().parents[1] / "resultados" / "figuras"

# La feature de la referencia de una sola variable, elegida por la teoría (el
# theta frontal es el marcador canónico de la carga ejecutiva) y no por los
# datos. El Epoc X no tiene Fz: los seis canales frontales son la región más
# cercana al theta frontal medial.
FEATURE_REFERENCIA = "frontal_theta_rel_media"
REGION_THETA, BANDA_THETA = "frontal", "theta"
REGION_ALFA, BANDA_ALFA = "posterior", "alfa"

# Criterios de coherencia neurofisiológica, declarados ANTES de ejecutar. Son
# más modestos que los del control OA/OC, porque el efecto esperado lo es
# también. Con la clase 1 = tarea, la teoría predice theta frontal SUBE con la
# carga (empuja hacia tarea) y alfa posterior BAJA (empuja hacia reposo).
#  (a) la banda cuya permutación conjunta cuesta más exactitud es theta o alfa
#      (equivale a: max(theta, alfa) > max(delta, beta)). Es un criterio poco
#      exigente por construcción (por azar lo cumpliría la mitad de las
#      veces), y así hay que citarlo.
#  (b) el coeficiente neto de las features `_media` de theta frontal en la
#      logística es POSITIVO en TODOS los pliegues, y el umbral de una sola
#      variable sobre `frontal_theta_rel_media` aprende signo +1 en TODOS.
#  (c) el coeficiente neto de las features `_media` de alfa posterior en la
#      logística es NEGATIVO en TODOS los pliegues.
#  (d) (b) y (c) se siguen cumpliendo en el control SIN OC (OA frente a tareas):
#      es el criterio que de verdad separa la carga cognitiva del efecto
#      Berger, porque con OC dentro del reposo el alfa posterior bajaría en las
#      tareas aunque no hubiera ninguna firma de carga.
BANDAS_CRITERIO_A = ("theta", "alfa")

# Paleta: ranuras 1 (azul) y 2 (naranja) de la paleta de referencia. Validada
# con el validador de la guía de visualización en MODO CLARO (ΔE CVD 24.7,
# visión normal 33.6, contraste >= 3:1). En modo oscuro el mismo validador da
# FALLO en la banda de luminosidad del naranja (0.671): estas figuras se
# generan solo sobre fondo blanco y no están pensadas para fondos oscuros.
# Identidad fija en todas las figuras: azul = reposo, naranja = tarea (la clase
# positiva, como el naranja era OC en el análisis OA/OC).
COLOR_REPOSO = "#2a78d6"
COLOR_TAREA = "#eb6834"
COLOR_GRIS = "#c3c2b7"
COLOR_TEXTO = "#0b0b0b"
COLOR_TEXTO_2 = "#52514e"
COLOR_REJILLA = "#e0e0dc"
# Rampa secuencial de un solo tono para la matriz de confusión.
RAMPA_AZUL = LinearSegmentedColormap.from_list(
    "azul_secuencial", ["#eef4fc", "#2a78d6", "#0f3d80"])

MODELOS = ("Regresión logística", "Random Forest")


def _titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def logistica():
    """Frontera lineal; sus coeficientes tienen signo y permiten ver la dirección."""
    return modelo_logistico()


def bosque():
    """Misma configuración que el Random Forest de las Fases 14, 14b y 17."""
    return modelo_bosque(n_estimators=N_ARBOLES, semilla=SEMILLA)


FABRICAS = {"Regresión logística": logistica, "Random Forest": bosque}


def _ba(y_true, y_pred):
    return metricas_binarias(y_true, y_pred)["exactitud_balanceada"]


def _d_de_cohen(y_true, valores):
    """d de Cohen (tarea - reposo) de un vector; los NaN se ignoran."""
    return float(tamanos_de_efecto(pd.DataFrame({"x": valores}), y_true).iloc[0])


def _spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        raise ValueError("Una de las variables es constante: Spearman no definido.")
    return float(stats.spearmanr(a, b).statistic)


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
    a los 50 minutos, lo ya calculado no se pierde."""
    RUTA_RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    RUTA_RESULTADOS.write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False, default=_a_python),
        encoding="utf-8")


def evaluar(X, y, g, modelo, nombre, con_probabilidad=True, con_ic=True):
    """Valida por persona, calcula los IC bootstrap por persona y lo devuelve."""
    r = evaluar_reposo(X, y, g, modelo, nombre, n_splits=N_PLIEGUES,
                       con_probabilidad=con_probabilidad)
    if con_probabilidad:
        r["ap_reposo"] = precision_media(y, r["puntuaciones"])
    if con_ic:
        boot = bootstrap_por_persona(y, r["y_pred"], g, _ba,
                                     n_replicas=N_BOOTSTRAP, semilla=SEMILLA)
        r["ic95_ba"] = boot["ic95"]
        if con_probabilidad:
            r["ic95_auc"] = bootstrap_por_persona(
                y, r["puntuaciones"], g, auc_roc, n_replicas=N_BOOTSTRAP,
                semilla=SEMILLA)["ic95"]
            r["ic95_ap_reposo"] = bootstrap_por_persona(
                y, r["puntuaciones"], g, precision_media, n_replicas=N_BOOTSTRAP,
                semilla=SEMILLA)["ic95"]
    return r


def _informe(r, prevalencia_reposo, detalle=True):
    m = r["metricas"]
    print(f"\n--- {m['modelo']} ---")
    print(f"  Exactitud balanceada, media por pliegue : "
          f"{m['exactitud_balanceada_media']:.4f}  (SD {m['exactitud_balanceada_sd']:.4f})")
    print(f"  Exactitud balanceada, amontonada        : {m['exactitud_balanceada']:.4f}")
    if "ic95_ba" in r:
        lo, hi = r["ic95_ba"]
        print(f"    IC95 (bootstrap por persona)          : [{lo:.4f}, {hi:.4f}]")
    print(f"  Exactitud simple (engañosa con este desbalance): {m['exactitud']:.4f}")
    if "auc_media" in m:
        print(f"  AUC, media por pliegue                  : {m['auc_media']:.4f}  "
              f"(amontonada {m['auc_agregado']:.4f})")
        if "ic95_auc" in r:
            lo, hi = r["ic95_auc"]
            print(f"    IC95 (bootstrap por persona)          : [{lo:.4f}, {hi:.4f}]")
    if "ap_reposo" in r:
        print(f"  Precisión media de REPOSO (clase rara)  : {r['ap_reposo']:.4f}  "
              f"(azar = prevalencia {prevalencia_reposo:.4f})")
        if "ic95_ap_reposo" in r:
            lo, hi = r["ic95_ap_reposo"]
            print(f"    IC95 (bootstrap por persona)          : [{lo:.4f}, {hi:.4f}]")
    if not detalle:
        return
    print(f"  Sensibilidad (tareas detectadas)        : {m['sensibilidad']:.4f}")
    print(f"  Especificidad (reposo detectado)        : {m['especificidad']:.4f}")
    print(f"  F1 (tarea) = {m['f1']:.4f}   MCC = {m['mcc']:.4f}")
    c = m["matriz_confusion"]
    print("  Matriz de confusión (filas = real, columnas = predicha):")
    print(f"                    pred reposo  pred tarea")
    print(f"      real reposo   {c['vn']:11d}  {c['fp']:10d}")
    print(f"      real tarea    {c['fn']:11d}  {c['vp']:10d}")
    print("  Por pliegue     : "
          + "  ".join(f"{b:.3f}" for b in m["exactitud_balanceada_pliegues"]))


def _guardar(r):
    """Lo que se archiva de una evaluación (sin los vectores por fila)."""
    salida = {"metricas": dict(r["metricas"])}
    for clave in ("ic95_ba", "ic95_auc", "ic95_ap_reposo"):
        if clave in r:
            salida[clave] = list(r[clave])
    if "ap_reposo" in r:
        salida["ap_reposo"] = r["ap_reposo"]
    return salida


def direccion_del_efecto(X, y, g):
    """
    Dirección del efecto por pliegue, para los criterios (b), (c) y (d).

    Devuelve el coeficiente neto de theta frontal y de alfa posterior en la
    logística, el umbral y signo de la referencia de una variable, y si se
    cumplen (b) y (c) (ver las constantes del principio).
    """
    coef = coeficientes_por_pliegue(X, y, g, logistica(), n_splits=N_PLIEGUES)
    neto_theta = coeficiente_neto_por_pliegue(coef, REGION_THETA, BANDA_THETA)
    neto_alfa = coeficiente_neto_por_pliegue(coef, REGION_ALFA, BANDA_ALFA)
    cortes = signos_umbral_por_pliegue(X, y, g, FEATURE_REFERENCIA,
                                       n_splits=N_PLIEGUES)
    return {
        "coef": coef,
        "neto_theta_frontal": neto_theta.tolist(),
        "neto_alfa_posterior": neto_alfa.tolist(),
        "cortes": cortes,
        "cumple_b": bool((neto_theta > 0).all()
                         and all(c["signo"] == 1 for c in cortes)),
        "cumple_c": bool((neto_alfa < 0).all()),
    }


def _imprimir_direccion(d, sangria="    "):
    print(sangria + "coef. neto theta frontal por pliegue : "
          + "  ".join(f"{v:+.3f}" for v in d["neto_theta_frontal"]))
    print(sangria + "coef. neto alfa posterior por pliegue: "
          + "  ".join(f"{v:+.3f}" for v in d["neto_alfa_posterior"]))
    print(sangria + "signo del umbral en theta frontal    : "
          + "  ".join(f"{c['signo']:+d}" for c in d["cortes"]))
    print(sangria + f"(b) theta frontal hacia tarea : {'SI' if d['cumple_b'] else 'NO'}"
          f"     (c) alfa posterior hacia reposo : {'SI' if d['cumple_c'] else 'NO'}")


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
    Matrices de confusión de los dos modelos, lado a lado y con la misma escala.

    El color es el % de cada clase **real** (filas que suman 100 %), no el
    recuento: así el desbalance de clases (1:2.5) no cambia el color de las
    celdas. Rampa secuencial de un solo tono, celdas separadas por un hueco
    blanco.
    """
    fig, ejes = plt.subplots(1, len(modelos), figsize=(9.8, 5.3), dpi=200)
    fig.patch.set_facecolor("white")
    for ax, (nombre, r) in zip(np.atleast_1d(ejes), modelos):
        m = r["metricas"]
        c = m["matriz_confusion"]
        cuentas = np.array([[c["vn"], c["fp"]], [c["fn"], c["vp"]]])
        tasas = cuentas / cuentas.sum(axis=1, keepdims=True)
        # pcolormesh con bordes blancos: el hueco entre celdas lo dibujan las
        # propias celdas. Con imshow + rejilla menor queda un filo de color en
        # el borde derecho.
        ax.pcolormesh(np.arange(3) - 0.5, np.arange(3) - 0.5, tasas, cmap=RAMPA_AZUL,
                      vmin=0, vmax=1, edgecolors="white", linewidth=6)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(1.5, -0.5)
        ax.set_aspect("equal")
        for i in range(2):
            for j in range(2):
                color = "white" if tasas[i, j] > 0.55 else COLOR_TEXTO
                ax.text(j, i - 0.09, f"{100 * tasas[i, j]:.1f} %", ha="center",
                        va="center", fontsize=15, fontweight="bold", color=color)
                ax.text(j, i + 0.17, f"{cuentas[i, j]} grabaciones", ha="center",
                        va="center", fontsize=9, color=color)
        ax.set_xticks([0, 1], ["Reposo", "Tarea"])
        ax.set_yticks([0, 1], ["Reposo", "Tarea"])
        ax.grid(False)
        ax.tick_params(which="both", length=0, colors=COLOR_TEXTO_2, labelsize=11)
        for lado in ax.spines.values():
            lado.set_visible(False)
        ax.set_xlabel("Clase predicha", fontsize=10.5, color=COLOR_TEXTO_2,
                      labelpad=8)
        ax.set_ylabel("Clase real", fontsize=10.5, color=COLOR_TEXTO_2, labelpad=8)
        lo, hi = r["ic95_ba"]
        # Título y subtítulo por separado: en una sola cadena el IC del panel
        # derecho se sale del lienzo.
        ax.set_title(nombre, fontsize=12, color=COLOR_TEXTO, loc="left", pad=30,
                     fontweight="bold")
        ax.text(0, 1.045, f"exactitud balanceada {m['exactitud_balanceada_media']:.3f}"
                f" · IC95 [{lo:.3f}, {hi:.3f}]", transform=ax.transAxes,
                fontsize=9.5, color=COLOR_TEXTO_2, va="bottom")
    fig.suptitle("Reposo frente a tarea ejecutiva: matriz de confusión",
                 x=0.012, ha="left", fontsize=13.5, color=COLOR_TEXTO,
                 fontweight="bold")
    fig.text(0.012, 0.012,
             f"Validación cruzada de {N_PLIEGUES} pliegues agrupada por persona; "
             f"n = {n} grabaciones de train + val (el conjunto de test permanece sin abrir).\n"
             "El color es el porcentaje de cada clase real; cada fila suma 100 %. "
             "Reposo = OA + OC; tarea = STROOP, PASAT, PVTB, MSIT y PVSAT.",
             fontsize=8, color=COLOR_TEXTO_2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    _guardar_figura(fig, ruta)


def figura_por_condicion(paneles, ruta):
    """
    Tasa de acierto de cada condición original, en el orden del protocolo.

    Un punto por condición (con su IC95 remuestreando personas), coloreado por
    la clase real: azul = reposo, naranja = tarea. El orden de arriba abajo es
    el del protocolo (1 = OC ... 7 = PVSAT), que es lo que permite ver de un
    vistazo si el acierto acompaña a la posición en la sesión.
    """
    fig, ejes = plt.subplots(1, len(paneles), figsize=(11.8, 5.6), dpi=200,
                             sharey=True)
    fig.patch.set_facecolor("white")
    for ax, (nombre, d, ba, ic) in zip(np.atleast_1d(ejes), paneles):
        _estilo(ax)
        ax.grid(axis="y", visible=False)
        n_filas = len(d)
        for i, (_, f) in enumerate(d.iterrows()):
            color = COLOR_TAREA if f["clase_real"] == 1 else COLOR_REPOSO
            ax.plot([f["ic95_inf"], f["ic95_sup"]], [i, i], color=COLOR_TEXTO_2,
                    linewidth=1.5, solid_capstyle="round", zorder=4)
            ax.plot([f["tasa_acierto"]], [i], marker="o", markersize=9.5, color=color,
                    markeredgecolor="white", markeredgewidth=2, linestyle="none",
                    zorder=5)
            ax.text(1.04, i, f"{f['tasa_acierto']:.2f}",
                    transform=ax.get_yaxis_transform(), va="center", ha="left",
                    fontsize=10, color=COLOR_TEXTO, fontweight="bold")
        ax.set_ylim(n_filas - 0.5, -0.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        # Separador entre el reposo (posiciones 1-2) y las tareas (3-7).
        if (d["clase_real"] == 0).any() and (d["clase_real"] == 1).any():
            corte = float((d["clase_real"] == 0).sum()) - 0.5
            ax.axhline(corte, color=COLOR_REJILLA, linewidth=1.0, zorder=1)
        # El n va en la etiqueta del eje, no junto al valor: en el panel derecho
        # (sin etiquetas propias, `sharey`) un texto largo se sale del lienzo.
        ax.set_yticks(range(n_filas), [
            f"{ORDEN_PROTOCOLO.index(c) + 1} · {c}  (n = {int(d.loc[c, 'n'])})"
            for c in d.index])
        ax.tick_params(axis="y", labelsize=10.5, colors=COLOR_TEXTO)
        ax.set_xlabel("Tasa de acierto de esa condición", fontsize=10.5,
                      color=COLOR_TEXTO_2, labelpad=8)
        ax.set_title(nombre, fontsize=12, color=COLOR_TEXTO, loc="left", pad=30,
                     fontweight="bold")
        ax.text(0, 1.045, f"exactitud balanceada {ba:.3f} · IC95 [{ic[0]:.3f}, {ic[1]:.3f}]",
                transform=ax.transAxes, fontsize=9.5, color=COLOR_TEXTO_2,
                va="bottom")
    manijas = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=9.5,
               markerfacecolor=COLOR_REPOSO, markeredgecolor="white",
               markeredgewidth=2, label="Reposo (OC, OA)"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=9.5,
               markerfacecolor=COLOR_TAREA, markeredgecolor="white",
               markeredgewidth=2, label="Tarea ejecutiva"),
    ]
    fig.legend(handles=manijas, loc="upper right", bbox_to_anchor=(0.985, 0.985),
               frameon=True, framealpha=0.95, fontsize=9.5, edgecolor=COLOR_REJILLA,
               facecolor="white", borderpad=0.7, ncol=2)
    fig.suptitle("Reposo frente a tarea: acierto por condición original",
                 x=0.012, ha="left", fontsize=13.5, color=COLOR_TEXTO,
                 fontweight="bold", y=0.985)
    fig.text(0.012, 0.012,
             "Train + val (el conjunto de test permanece sin abrir). Punto: fracción de las grabaciones "
             "de esa condición clasificadas bien; línea: IC95 remuestreando personas.\n"
             "El número antepuesto es la posición de la condición en el protocolo, que es fijo "
             "(OC → OA → STROOP → PASAT → PVTB → MSIT → PVSAT).",
             fontsize=8, color=COLOR_TEXTO_2, linespacing=1.5)
    fig.subplots_adjust(left=0.165, right=0.955, wspace=0.24, top=0.78, bottom=0.17)
    _guardar_figura(fig, ruta)


def figura_canales(tabla, ruta):
    """
    Efecto de la tarea sobre theta y alfa, canal a canal, en potencia relativa
    (fila de arriba) y en potencia absoluta (fila de abajo).

    Las dos filas son necesarias, no un adorno: la potencia relativa suma 1
    entre bandas, así que cuando el alfa cae las demás suben aunque no cambien
    en absoluto. Un theta relativo alto en los canales posteriores puede ser
    solo el reflejo del descenso de alfa; el theta absoluto lo separa.

    Los cuatro paneles comparten ejes (mismos canales, misma escala de d), de
    modo que las magnitudes se pueden comparar de un vistazo. Los canales de la
    región donde la teoría espera el efecto, en el color de la tarea; el resto,
    en gris. Orden anatómico de frontal a posterior. Solo se rotula el canal más
    extremo de cada panel.
    """
    orden = [c for r in ("frontal", "frontocentral", "temporal", "posterior")
             for c in REGIONES[r]]
    tabla = tabla.loc[orden]
    fig, ejes = plt.subplots(2, 2, figsize=(10.6, 10.6), dpi=200, sharey=True,
                             sharex=True)
    fig.patch.set_facecolor("white")
    todos = np.array([ic for banda in ("theta", "alfa") for tipo in ("rel", "logabs")
                      for ic in tabla[f"ic95_d_{banda}_{tipo}"].tolist()])
    margen = 0.12 * (todos.max() - todos.min())
    limites = (min(0.0, todos.min()) - margen, max(0.0, todos.max()) + margen)
    paneles = (
        (ejes[0, 0], "theta", "rel", REGION_THETA, "Theta · potencia relativa"),
        (ejes[0, 1], "alfa", "rel", REGION_ALFA, "Alfa · potencia relativa"),
        (ejes[1, 0], "theta", "logabs", REGION_THETA, "Theta · potencia absoluta (log₁₀)"),
        (ejes[1, 1], "alfa", "logabs", REGION_ALFA, "Alfa · potencia absoluta (log₁₀)"),
    )
    for ax, banda, tipo, region, titulo in paneles:
        _estilo(ax)
        ax.grid(axis="y", visible=False)
        ic = np.array(tabla[f"ic95_d_{banda}_{tipo}"].tolist())
        d = tabla[f"d_{banda}_{tipo}"].to_numpy()
        for i, canal in enumerate(orden):
            color = COLOR_TAREA if canal in REGIONES[region] else COLOR_GRIS
            ax.plot(ic[i], [i, i], color=COLOR_TEXTO_2, linewidth=1.5,
                    solid_capstyle="round", zorder=4)
            ax.plot([d[i]], [i], marker="o", markersize=9, color=color,
                    markeredgecolor="white", markeredgewidth=2, linestyle="none",
                    zorder=5)
        ax.axvline(0, color=COLOR_TEXTO_2, linewidth=1.0, zorder=3)
        # Separadores entre regiones.
        acumulado = 0
        for r in ("frontal", "frontocentral", "temporal"):
            acumulado += len(REGIONES[r])
            ax.axhline(acumulado - 0.5, color=COLOR_REJILLA, linewidth=1.0, zorder=1)
        extremo = int(np.argmax(np.abs(d)))
        lado = 1 if d[extremo] >= 0 else -1
        # Signo menos tipográfico, como el de los ejes.
        ax.text(ic[extremo][1 if lado > 0 else 0] + lado * 0.03 * (limites[1] - limites[0]),
                extremo, f"{d[extremo]:+.2f}".replace("-", "−"), va="center",
                ha="left" if lado > 0 else "right", fontsize=9.5, color=COLOR_TEXTO)
        ax.set_xlim(*limites)
        ax.set_ylim(len(orden) - 0.5, -0.5)
        ax.set_yticks(range(len(orden)), orden)
        ax.tick_params(axis="y", labelsize=10.5, colors=COLOR_TEXTO)
        ax.set_title(titulo, fontsize=12, color=COLOR_TEXTO, loc="left", pad=10,
                     fontweight="bold")
    for ax in ejes[1]:
        ax.set_xlabel("d de Cohen (tarea − reposo)", fontsize=10.5,
                      color=COLOR_TEXTO_2, labelpad=8)
    # Leyenda corta arriba a la derecha: el texto largo pisaba el título.
    manijas = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=9,
               markerfacecolor=COLOR_TAREA, markeredgecolor="white",
               markeredgewidth=2, label="Región donde se espera el efecto"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=9,
               markerfacecolor=COLOR_GRIS, markeredgecolor="white",
               markeredgewidth=2, label="Resto de canales"),
    ]
    fig.legend(handles=manijas, loc="upper right", bbox_to_anchor=(0.985, 0.992),
               frameon=True, framealpha=0.95, fontsize=9, edgecolor=COLOR_REJILLA,
               facecolor="white", borderpad=0.7)
    fig.suptitle("Efecto de la tarea ejecutiva sobre theta y alfa, por canal",
                 x=0.012, ha="left", fontsize=13.5, color=COLOR_TEXTO,
                 fontweight="bold", y=0.988)
    fig.text(0.012, 0.010,
             "Train + val (el conjunto de test permanece sin abrir). Punto: d de Cohen; línea: "
             "IC95 remuestreando personas. Canales de frontal a posterior.\n"
             "Hipótesis previa: theta sube (sobre todo en frontal) y alfa baja (sobre todo en "
             "posterior). La potencia relativa suma 1 entre bandas: un\n"
             "descenso de alfa eleva las demás aunque no cambien, por eso se muestra también la "
             "absoluta. Solo grabaciones donde el canal superó el control\n"
             "de calidad de la Fase 11.5. Diferencias de ~0.1-0.2 entre canales vecinos no son "
             "distinguibles.",
             fontsize=8, color=COLOR_TEXTO_2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.945))
    _guardar_figura(fig, ruta)


# --------------------------------------------------------------------------
# Suplemento por canal
# --------------------------------------------------------------------------

def analisis_por_canal(datos, y, g):
    """
    Suplemento: qué canales llevan el efecto.

    Reagrega `matriz_modelado.parquet` con `construir_matriz_rf` y una región por
    canal (la variante que documenta `agregar_por_region`), restringida a las
    mismas grabaciones del análisis principal. Devuelve la tabla por canal.
    """
    # El filtro por archivo se aplica AL LEER: las épocas de las grabaciones
    # apartadas (test) no llegan a cargarse, no solo se descartan después.
    epocas = pd.read_parquet(
        RUTA_MATRIZ_EPOCAS,
        filters=[("archivo_origen", "in", datos["archivo"].tolist())])
    grabaciones = pd.read_csv(RUTA_GRABACIONES)
    with warnings.catch_warnings():
        # `construir_matriz_rf` (código congelado de la Fase 13) inserta
        # columnas de una en una y pandas avisa de la fragmentación; no afecta
        # al resultado y llenaría el log.
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
    region_de = {c: r for r, canales in REGIONES.items() for c in canales}

    filas = []
    for canal in CANALES_EPOC:
        fila = {"canal": canal, "region": region_de[canal]}
        for banda in BANDAS_DEFECTO:
            d = tamanos_de_efecto(por_canal[[f"{canal}_{banda}_rel_media"]], y)
            fila[f"d_{banda}_rel"] = float(d.iloc[0])
        for banda in ("theta", "alfa"):
            fila[f"d_{banda}_logabs"] = float(tamanos_de_efecto(
                por_canal[[f"{canal}_{banda}_logabs_media"]], y).iloc[0])
            # IC95 de la relativa y de la absoluta: la relativa suma 1 entre
            # bandas, así que solo con las dos se separa el efecto real del
            # composicional (el descenso de alfa eleva las demás).
            for tipo in ("rel", "logabs"):
                fila[f"ic95_d_{banda}_{tipo}"] = bootstrap_por_persona(
                    y, por_canal[f"{canal}_{banda}_{tipo}_media"].to_numpy(dtype=float),
                    g, _d_de_cohen, n_replicas=N_BOOTSTRAP_CANALES,
                    semilla=SEMILLA)["ic95"]

        cols = [f"{canal}_{b}_{t}_media" for b in ("theta", "alfa")
                for t in ("rel", "logabs")]
        validas = por_canal[cols].notna().all(axis=1).to_numpy()
        fila["n_validas"] = int(validas.sum())
        fila["n_personas"] = int(len(set(g[validas])))
        if validas.sum() >= 100 and fila["n_personas"] >= N_PLIEGUES:
            r = evaluar_reposo(por_canal.loc[validas, cols], y[validas],
                               g[validas], logistica(), n_splits=N_PLIEGUES)
            fila["ba_un_canal"] = r["metricas"]["exactitud_balanceada_media"]
            fila["auc_un_canal"] = r["metricas"]["auc_media"]
        filas.append(fila)
    return pd.DataFrame(filas).set_index("canal")


def main():
    resultados = {"modo_rapido": RAPIDO}
    if RAPIDO:
        print("#" * 78)
        print("# MODO RÁPIDO: parámetros mínimos. Estas cifras NO son resultados.")
        print(f"# Salidas en {RUTA_RESULTADOS.parent}")
        print("#" * 78)

    # ----------------------------------------------------------------------
    _titulo("1. Datos - grabaciones de reposo y de tarea de train + val")
    matriz = pd.read_parquet(RUTA_MATRIZ)
    n_fase13 = len(columnas_de_features(matriz))
    print(f"Matriz completa      : {matriz.shape[0]} grabaciones, {n_fase13} features (Fase 13)")

    datos, meta = preparar_reposo_tarea(matriz)
    assert set(datos["split"]) <= {"train", "val"}, "el conjunto de test se ha colado"
    print(f"Apartado (test)      : {meta['n_apartadas']} grabaciones, "
          f"{meta['n_personas_apartadas']} personas - no se vuelve a leer")
    print(f"Sin persona_uid      : {meta['n_sin_persona_uid']} grabaciones descartadas "
          f"(sin él no se puede agrupar)")
    print(f"En análisis          : {meta['n_grabaciones']} grabaciones, "
          f"{meta['n_personas']} personas, {meta['n_sesiones']} sesiones")
    print(f"  reposo / tarea     : {meta['n_reposo']} / {meta['n_tarea']}   "
          f"(desbalance {meta['ratio_desbalance']:.2f}:1, mayoritaria = "
          f"{meta['clase_mayoritaria']})")
    print(f"  por condición      : "
          f"{ {c: meta['por_condicion'][c] for c in ORDEN_PROTOCOLO} }")
    print(f"  por split          : {meta['por_split']}")
    por_persona = datos.groupby("persona_uid").size()
    print(f"  grabaciones/persona: mediana {por_persona.median():.0f}, "
          f"rango [{por_persona.min()}, {por_persona.max()}]")
    filas_nan = int(datos[columnas_de_features(datos)].isna().any(axis=1).sum())
    print(f"  NaN en features    : {filas_nan} grabaciones con alguna región sin canales "
          f"válidos (imputación por mediana dentro de cada pliegue)")

    y = etiqueta_tarea(datos)
    g = datos["persona_uid"].to_numpy()
    columnas = columnas_de_features_tarea(datos)
    X = datos[columnas]
    prevalencia_reposo = float((y == 0).mean())
    print(f"\nPrevalencia de reposo: {prevalencia_reposo:.4f}   "
          f"(la exactitud de predecir siempre «tarea» es {1 - prevalencia_reposo:.4f})")

    excluidas = [c for c in columnas_de_features(datos) if c not in columnas]
    print(f"\nFeatures de la Fase 13 : {n_fase13}")
    print(f"Features en X          : {len(columnas)}   (excluidas {len(excluidas)})")
    print("\nColumnas excluidas de X y por qué:")
    print(f"  {COLUMNA_FUGA:<26} ES LA ETIQUETA: posición de la condición en el protocolo")
    for condicion in ORDEN_PROTOCOLO:
        valores = sorted(datos.loc[datos["condicion"] == condicion,
                                   COLUMNA_FUGA].unique().tolist())
        print(f"    {condicion:<7} -> {valores}")
    maximo_reposo = datos.loc[y == 0, COLUMNA_FUGA].max()
    minimo_tarea = datos.loc[y == 1, COLUMNA_FUGA].min()
    print(f"    máx. en reposo = {maximo_reposo:.0f} < mín. en tarea = {minimo_tarea:.0f}: "
          f"un solo corte la separa sin error")
    print(f"\n  Contadores de procesamiento (no son señal cerebral, pero pueden discriminar):")
    filas_proc = []
    for c in COLUMNAS_PROCESAMIENTO:
        d = float(tamanos_de_efecto(datos[[c]], y).iloc[0])
        p = float(stats.mannwhitneyu(datos.loc[y == 1, c], datos.loc[y == 0, c]).pvalue)
        filas_proc.append({"columna": c, "d_cohen": d, "p_mann_whitney": p})
        print(f"    {c:<26} d = {d:+.3f}   p = {p:.4g}")
    verificar_sin_fuga(columnas)
    assert COLUMNA_FUGA not in columnas
    print(f"\nComprobado: `{COLUMNA_FUGA}` no está en las {len(columnas)} features de X.")

    duplicadas = columnas_duplicadas(X)
    print(f"\nColumnas de X que son COPIAS EXACTAS de otra (mismo valor en todas las filas):")
    for conservada, copia in duplicadas:
        print(f"    {copia:<32} = {conservada}")
    print(f"  {len(duplicadas)} duplicados: {len(columnas) - len(duplicadas)} columnas distintas "
          f"de las {len(columnas)}. No se quitan de X (se reutiliza la matriz tal cual);")
    print("  la fila «sin duplicados» de la sensibilidad (8.4) comprueba que no cambian nada.")

    resultados["datos"] = {
        **meta, "n_features_fase13": n_fase13, "n_features_x": len(columnas),
        "prevalencia_reposo": prevalencia_reposo,
        "columnas_excluidas": excluidas, "contadores_procesamiento": filas_proc,
        "columnas_x": columnas,
        "columnas_duplicadas": [list(par) for par in duplicadas],
    }
    _volcar(resultados)

    # ----------------------------------------------------------------------
    _titulo("2. Baselines triviales")
    print("Un clasificador que no usa X. Con la exactitud balanceada el azar es 0.5")
    print("exacto aunque las clases estén desbalanceadas: si aquí saliera otra cosa, el")
    print("montaje estaría roto. La exactitud simple, en cambio, premia predecir siempre la")
    print("clase mayoritaria: por eso NO es la métrica principal.")
    modelos = {}
    for nombre, modelo in (
            ("Baseline: clase mayoritaria",
             DummyClassifier(strategy="most_frequent")),
            ("Baseline: azar estratificado",
             DummyClassifier(strategy="stratified", random_state=SEMILLA))):
        modelos[nombre] = evaluar(X, y, g, modelo, nombre)
        _informe(modelos[nombre], prevalencia_reposo, detalle=False)
    sanidad = all(abs(r["metricas"]["exactitud_balanceada_media"] - 0.5) < 0.05
                  for r in modelos.values())
    print(f"\n  [{'SI' if sanidad else 'NO'}] Los dos baselines dan ~0.5 de exactitud balanceada")
    resultados["baselines"] = {n: _guardar(r) for n, r in modelos.items()}
    resultados["baselines_sanos"] = bool(sanidad)

    # ----------------------------------------------------------------------
    _titulo("3. Modelo A - regresión logística")
    print("Imputación por mediana + tipificación + LogisticRegression(class_weight=")
    print("'balanced'), todo ajustado solo con el pliegue de entrenamiento. Con el peso")
    print("balanceado, P(tarea) > 0.5 da el mismo coste a cada tipo de error, que es lo que")
    print("mide la exactitud balanceada (no se ajusta ningún umbral de decisión).")
    modelos["Regresión logística"] = evaluar(X, y, g, logistica(), "Regresión logística")
    _informe(modelos["Regresión logística"], prevalencia_reposo)

    # ----------------------------------------------------------------------
    _titulo("4. Modelo B - Random Forest")
    print(f"RandomForestClassifier({N_ARBOLES} árboles, hoja>=5, class_weight='balanced'): la misma")
    print("configuración que las Fases 14, 14b y 17, para que la cifra sea comparable.")
    modelos["Random Forest"] = evaluar(X, y, g, bosque(), "Random Forest")
    _informe(modelos["Random Forest"], prevalencia_reposo)
    _volcar({**resultados, "modelos": {n: _guardar(r) for n, r in modelos.items()}})

    # ----------------------------------------------------------------------
    _titulo("5. Referencia de una sola variable")
    print(f"Un único corte sobre `{FEATURE_REFERENCIA}`, con su umbral y su signo")
    print("reestimados dentro de cada pliegue. Responde a: hacía falta un modelo?")
    nombre_ref = f"Umbral en {FEATURE_REFERENCIA}"
    modelos[nombre_ref] = evaluar(X, y, g, UmbralUnaVariable(FEATURE_REFERENCIA),
                                  nombre_ref, con_probabilidad=False)
    _informe(modelos[nombre_ref], prevalencia_reposo)

    cortes = signos_umbral_por_pliegue(X, y, g, FEATURE_REFERENCIA, n_splits=N_PLIEGUES)
    print("\n  Corte aprendido por pliegue (signo +1 = «tarea si theta frontal > corte»):")
    for c in cortes:
        print(f"    pliegue {c['pliegue']}: umbral {c['umbral']:.4f}  signo {c['signo']:+d}")

    umbral_global, signo_global = umbral_una_variable(X[FEATURE_REFERENCIA], y)
    x_ref = X[FEATURE_REFERENCIA].to_numpy(dtype=float)
    validas = ~np.isnan(x_ref)
    auc_univariado = auc_roc(y[validas], x_ref[validas])
    d_ref = float(tamanos_de_efecto(X[[FEATURE_REFERENCIA]], y).iloc[0])
    print(f"\n  Sobre todos los datos (descriptivo): umbral {umbral_global:.4f}, "
          f"signo {signo_global:+d}")
    print(f"  d de Cohen = {d_ref:+.3f}   AUC univariado = {auc_univariado:.4f}")
    resultados["referencia_una_variable"] = {
        "feature": FEATURE_REFERENCIA, "cortes_por_pliegue": cortes,
        "umbral_global": umbral_global, "signo_global": signo_global,
        "d_cohen": d_ref, "auc_univariado": auc_univariado,
    }

    print("\n5.1 Comparaciones PAREADAS entre clasificadores (exactitud balanceada)")
    print("    Mismas personas remuestreadas en las dos ramas de cada réplica: comparar")
    print("    los IC de cada uno por separado no vale, porque comparten la dificultad")
    print("    de las personas y esa parte común se cancela en la diferencia.")
    comparaciones = []
    for a, b in (("Regresión logística", "Random Forest"),
                 ("Regresión logística", nombre_ref),
                 ("Random Forest", nombre_ref),
                 ("Regresión logística", "Baseline: azar estratificado"),
                 ("Random Forest", "Baseline: azar estratificado"),
                 (nombre_ref, "Baseline: azar estratificado")):
        c = bootstrap_diferencia_por_persona(
            y, modelos[a]["y_pred"], modelos[b]["y_pred"], g, _ba,
            n_replicas=N_BOOTSTRAP, semilla=SEMILLA)
        comparaciones.append({
            "a": a, "b": b, "diferencia": c["diferencia"],
            "ic95": list(c["ic95"]), "prop_a_mayor": c["prop_positivas"]})
        print(f"    {a} - {b}: {c['diferencia']:+.4f}  IC95 [{c['ic95'][0]:+.4f}, "
              f"{c['ic95'][1]:+.4f}]  (A > B en el {c['prop_positivas']:.1%} de las réplicas)")
    resultados["comparaciones_pareadas"] = comparaciones
    resultados["modelos"] = {n: _guardar(r) for n, r in modelos.items()}
    _volcar(resultados)

    # ----------------------------------------------------------------------
    _titulo("6. Coherencia neurofisiológica - qué pesa en el modelo")
    print("Criterios declarados antes de ejecutar (ver las constantes del guion):")
    print("  (a) la banda cuya permutación conjunta cuesta más exactitud es theta o alfa")
    print("  (b) theta frontal empuja hacia TAREA: coef. neto > 0 en todos los pliegues")
    print("      y umbral de una variable con signo +1 en todos")
    print("  (c) alfa posterior empuja hacia REPOSO: coef. neto < 0 en todos los pliegues")
    print("  (d) (b) y (c) se cumplen también en el control sin OC (sección 8.2)")

    print("\n6.1 Importancia por permutación del Random Forest, OUT-OF-FOLD")
    print(f"    (caída de exactitud balanceada de validación al barajar la columna;")
    print(f"    {N_REPETICIONES_PERMUTACION} barajados x {N_PLIEGUES} pliegues)")
    imp = importancia_permutacion_cv(X, y, g, bosque(), n_splits=N_PLIEGUES,
                                     n_repeticiones=N_REPETICIONES_PERMUTACION,
                                     semilla=SEMILLA)
    d_todas = tamanos_de_efecto(X, y)
    direccion = direccion_del_efecto(X, y, g)
    coef = direccion["coef"]
    tabla = imp.assign(d_cohen=d_todas, coef_logistica=coef.mean(axis=1),
                       pliegues_coef_positivo=(coef > 0).sum(axis=1))
    ranking = tabla.sort_values("caida_media", ascending=False)
    print(f"\n    {'feature':<34}{'caída BA':>10}{'d Cohen':>9}{'coef LR':>9}{'pliegues+':>10}")
    for nombre, f in ranking.head(15).iterrows():
        print(f"    {nombre:<34}{f['caida_media']:>+10.4f}{f['d_cohen']:>+9.2f}"
              f"{f['coef_logistica']:>+9.3f}{int(f['pliegues_coef_positivo']):>8d}/{N_PLIEGUES}")

    print("\n6.2 Importancia agregada por bloque (fracción del total)")
    por_bloque = importancia_por_bloque(imp["caida_media"])
    print("    por banda   : " + "  ".join(
        f"{k} {v:.1%}" for k, v in sorted(por_bloque["por_banda"].items(),
                                          key=lambda kv: -kv[1])))
    print("    por región  : " + "  ".join(
        f"{k} {v:.1%}" for k, v in sorted(por_bloque["por_region"].items(),
                                          key=lambda kv: -kv[1])))
    mejores = sorted(por_bloque["por_bloque"].items(), key=lambda kv: -kv[1])[:5]
    print("    top región/banda: " + "  ".join(f"{k} {v:.1%}" for k, v in mejores))

    print("\n6.3 Permutación CONJUNTA de bloques enteros (Random Forest)")
    print("    Barajar de una vez todas las columnas de una banda o región, con la")
    print("    misma permutación: no se diluye entre features casi duplicadas.")
    bloques = {}
    for banda in BANDAS_DEFECTO:
        bloques[f"banda/{banda}"] = columnas_del_bloque(columnas, banda=banda)
    for region in REGIONES:
        bloques[f"region/{region}"] = columnas_del_bloque(columnas, region=region)
    bloques["bloque/frontal+theta"] = columnas_del_bloque(
        columnas, region=REGION_THETA, banda=BANDA_THETA)
    bloques["bloque/posterior+alfa"] = columnas_del_bloque(
        columnas, region=REGION_ALFA, banda=BANDA_ALFA)
    conjunta = importancia_permutacion_cv(X, y, g, bosque(), bloques=bloques,
                                          n_splits=N_PLIEGUES,
                                          n_repeticiones=N_REPETICIONES_PERMUTACION,
                                          semilla=SEMILLA)
    conjunta["n_features"] = [len(bloques[k]) for k in conjunta.index]
    print(f"\n    {'bloque':<24}{'features':>9}{'caída BA':>11}{'SD pliegues':>13}")
    for nombre, f in conjunta.sort_values("caida_media", ascending=False).iterrows():
        print(f"    {nombre:<24}{int(f['n_features']):>9d}{f['caida_media']:>+11.4f}"
              f"{f['sd_entre_pliegues']:>13.4f}")

    print("\n6.4 Dirección del efecto (logística y umbral de una variable)")
    print(f"    Features `_media` de theta frontal : "
          f"{columnas_media_del_bloque(columnas, REGION_THETA, BANDA_THETA)}")
    print(f"    Features `_media` de alfa posterior: "
          f"{columnas_media_del_bloque(columnas, REGION_ALFA, BANDA_ALFA)}")
    _imprimir_direccion(direccion)
    theta_media = columnas_media_del_bloque(columnas, REGION_THETA, BANDA_THETA)
    alfa_media = columnas_media_del_bloque(columnas, REGION_ALFA, BANDA_ALFA)
    print("    d de Cohen (tarea - reposo): "
          + "  ".join(f"{c} {d_todas[c]:+.2f}" for c in theta_media + alfa_media))

    print("\n6.5 Mayores tamaños de efecto univariados (d de Cohen, tarea - reposo)")
    por_d = d_todas.reindex(d_todas.abs().sort_values(ascending=False).index)
    for nombre, v in por_d.head(12).items():
        print(f"    {nombre:<34}{v:>+8.3f}")

    # ---- Veredicto parcial: (a), (b) y (c) ----
    bandas_caida = conjunta.loc[[f"banda/{b}" for b in BANDAS_DEFECTO], "caida_media"]
    banda_max = bandas_caida.idxmax().split("/")[1]
    criterio_a = banda_max in BANDAS_CRITERIO_A
    criterio_b, criterio_c = direccion["cumple_b"], direccion["cumple_c"]
    valor_b = ("coef " + ("+" if all(v > 0 for v in direccion["neto_theta_frontal"]) else "x")
               + " / umbral " + ("+" if all(c["signo"] == 1 for c in direccion["cortes"]) else "x"))
    valor_c = "coef " + ("-" if criterio_c else "x")
    print("\n6.6 Veredicto parcial frente a los criterios declarados (el (d) va en 8.5)")
    for nombre, cumple, valor in (
            ("(a) la banda que más cuesta al barajarla entera es theta o alfa",
             criterio_a, f"banda {banda_max}"),
            ("(b) theta frontal hacia tarea (coef. neto > 0 y umbral +1 en todos)",
             criterio_b, valor_b),
            ("(c) alfa posterior hacia reposo (coef. neto < 0 en todos)",
             criterio_c, valor_c)):
        print(f"  [{'SI' if cumple else 'NO'}] {nombre:<76} {valor}")

    resultados["coherencia"] = {
        "criterios_declarados": {"bandas_criterio_a": list(BANDAS_CRITERIO_A),
                                 "feature_referencia": FEATURE_REFERENCIA},
        "importancia_por_permutacion": tabla.reset_index().rename(
            columns={"index": "feature"}).to_dict(orient="records"),
        "importancia_por_bloque": por_bloque,
        "importancia_conjunta": conjunta.reset_index().rename(
            columns={"index": "bloque"}).to_dict(orient="records"),
        "direccion": {k: v for k, v in direccion.items() if k != "coef"},
        "banda_max_permutacion_conjunta": banda_max,
        "d_cohen_features_bloques": {c: float(d_todas[c])
                                     for c in theta_media + alfa_media},
    }
    _volcar(resultados)

    # ----------------------------------------------------------------------
    _titulo("7. Errores por condición original")
    print("Un acierto global esconde qué falla. Aquí cada condición tiene su propia tasa,")
    print("con IC95 remuestreando personas. Para una TAREA el acierto es la fracción")
    print("leída como tarea; para OA/OC es la fracción leída como reposo.")
    desgloses = {}
    for nombre in MODELOS:
        r = modelos[nombre]
        d = desglose_por_condicion(datos, y, r["y_pred"], grupos=g,
                                   n_replicas=N_BOOTSTRAP, semilla=SEMILLA)
        d["p_tarea_media"] = pd.Series(r["puntuaciones"]).groupby(
            datos["condicion"].to_numpy()).mean().reindex(d.index)
        desgloses[nombre] = d
        print(f"\n--- {nombre} ---")
        print(f"    {'posición':<9}{'condición':<10}{'clase':<8}{'n':>5}{'acierto':>9}"
              f"{'IC95':>19}{'leída como tarea':>18}{'P(tarea) media':>16}")
        for cond, f in d.iterrows():
            clase = "tarea" if f["clase_real"] == 1 else "reposo"
            print(f"    {ORDEN_PROTOCOLO.index(cond) + 1:<9}{cond:<10}{clase:<8}{int(f['n']):>5}"
                  f"{f['tasa_acierto']:>9.3f}   [{f['ic95_inf']:.3f}, {f['ic95_sup']:.3f}]"
                  f"{f['frac_predicha_tarea']:>15.3f}{f['p_tarea_media']:>16.3f}")
        tareas = d[d["clase_real"] == 1].sort_values("tasa_acierto")
        print(f"    Tareas de la que más se confunde con reposo a la que menos: "
              + " > ".join(f"{c} ({1 - f['tasa_acierto']:.1%})" for c, f in tareas.iterrows()))

    print("\n7.1 ¿Fallan por igual OA y OC? (diferencia de acierto OA - OC, pareada por persona)")
    oa_oc = {}
    for nombre in MODELOS:
        c = comparar_condiciones(datos, y, modelos[nombre]["y_pred"], g, "OA", "OC",
                                 n_replicas=N_BOOTSTRAP, semilla=SEMILLA)
        oa_oc[nombre] = c
        print(f"    {nombre:<22} OA {c['tasa_a']:.3f} (n = {c['n_a']})  OC {c['tasa_b']:.3f} "
              f"(n = {c['n_b']})  OA - OC = {c['diferencia']:+.3f}  "
              f"IC95 [{c['ic95'][0]:+.3f}, {c['ic95'][1]:+.3f}]  "
              f"(OA > OC en el {c['prop_a_mayor']:.1%} de las réplicas)")
    resultados["desglose_por_condicion"] = {
        n: d.reset_index().rename(columns={"index": "condicion"}).to_dict(orient="records")
        for n, d in desgloses.items()}
    resultados["oa_frente_a_oc"] = oa_oc
    _volcar(resultados)

    # ----------------------------------------------------------------------
    _titulo("8. Controles y sensibilidad")
    print("8.1 Control negativo: etiquetas barajadas DENTRO de cada persona")
    print(f"    {N_PERMUTACIONES_ETIQUETAS} barajados por modelo. Destruye la relación con las features")
    print("    pero conserva las personas y cuántas grabaciones de cada clase tiene cada una.")
    print("    Tiene que dar ~0.5.")
    nulos = {}
    for nombre in MODELOS:
        valores = []
        for s in range(N_PERMUTACIONES_ETIQUETAS):
            y_perm = permutar_etiquetas_dentro_de_persona(y, g, semilla=SEMILLA + s)
            r = evaluar_reposo(X, y_perm, g, FABRICAS[nombre](), n_splits=N_PLIEGUES)
            valores.append(r["metricas"]["exactitud_balanceada_media"])
        valores = np.array(valores)
        real = modelos[nombre]["metricas"]["exactitud_balanceada_media"]
        n_supera = int((valores >= real).sum())
        p_emp = (1 + n_supera) / (1 + len(valores))
        sd = float(valores.std(ddof=1)) if len(valores) > 1 else float("nan")
        nulos[nombre] = {
            "media": float(valores.mean()), "sd": sd,
            "min": float(valores.min()), "max": float(valores.max()),
            "p95": float(np.percentile(valores, 95)), "real": float(real),
            "n_nulos_iguales_o_mayores": n_supera,
            "p_empirico": float(p_emp), "valores": valores.tolist()}
        print(f"    {nombre:<22} nulo {valores.mean():.4f} (SD {sd:.4f}, "
              f"rango [{valores.min():.3f}, {valores.max():.3f}])  real {real:.4f}  "
              f"nulos >= real: {n_supera} de {len(valores)}")
    print(f"    (con {N_PERMUTACIONES_ETIQUETAS} barajados el p empirico no puede bajar de "
          f"{1 / (1 + N_PERMUTACIONES_ETIQUETAS):.3f}: es la resolucion del control, no un "
          f"valor a leer)")
    control_ok = all(abs(v["media"] - 0.5) < 0.03 for v in nulos.values())
    print(f"\n  [{'SI' if control_ok else 'NO'}] El nulo está centrado en 0.5 "
          f"(el arnés no produce acierto de la nada)")
    resultados["control_etiquetas_barajadas"] = {**nulos, "centrado_en_0_5": bool(control_ok)}
    _volcar(resultados)

    print("\n8.2 Sin OC: ¿es solo el efecto Berger?")
    print("    El reposo incluye OC, con el alfa disparado por tener los ojos cerrados. Aquí")
    print("    se repite el análisis con UNA sola de las dos condiciones de reposo. Si con OA")
    print("    solo el acierto se desploma hacia 0.5, el modelo sabía de ojos cerrados y no de")
    print("    carga cognitiva; si aguanta, hay una firma de tarea propia.")
    controles_reposo = {}
    for etiqueta, cond in (("OA solo frente a tareas", ("OA",)),
                           ("OC solo frente a tareas", ("OC",))):
        d_c, meta_c = preparar_reposo_tarea(matriz, condiciones_reposo=cond)
        assert set(d_c["split"]) <= {"train", "val"}
        y_c = etiqueta_tarea(d_c, condiciones_reposo=cond)
        g_c = d_c["persona_uid"].to_numpy()
        X_c = d_c[columnas]
        prev_c = float((y_c == 0).mean())
        print(f"\n  == {etiqueta}: {meta_c['n_grabaciones']} grabaciones "
              f"({meta_c['n_reposo']} reposo, {meta_c['n_tarea']} tarea; "
              f"desbalance {meta_c['ratio_desbalance']:.2f}:1), {meta_c['n_personas']} personas ==")
        salida = {"meta": meta_c, "modelos": {}, "desglose_por_condicion": {}}
        for nombre in MODELOS:
            r = evaluar(X_c, y_c, g_c, FABRICAS[nombre](), nombre)
            _informe(r, prev_c, detalle=False)
            salida["modelos"][nombre] = _guardar(r)
            # Qué tareas se confunden con este reposo (sin la otra condición).
            d_cond = desglose_por_condicion(d_c, y_c, r["y_pred"], grupos=g_c,
                                            n_replicas=N_BOOTSTRAP, semilla=SEMILLA)
            print(f"    Acierto por condición ({nombre}): " + "  ".join(
                f"{c} {f['tasa_acierto']:.3f} [{f['ic95_inf']:.3f}, {f['ic95_sup']:.3f}]"
                for c, f in d_cond.iterrows()))
            salida["desglose_por_condicion"][nombre] = d_cond.reset_index().rename(
                columns={"index": "condicion"}).to_dict(orient="records")
        dir_c = direccion_del_efecto(X_c, y_c, g_c)
        print("\n    Dirección del efecto en este control:")
        _imprimir_direccion(dir_c, sangria="      ")
        salida["direccion"] = {k: v for k, v in dir_c.items() if k != "coef"}
        controles_reposo[etiqueta] = salida
    resultados["control_sin_oc"] = controles_reposo
    _volcar(resultados)

    print("\n8.3 Posición en el protocolo")
    print("    El orden es fijo: el reposo ocupa las posiciones 1-2 y las tareas las 3-7, así")
    print("    que la deriva de sesión (gel, somnolencia, acomodación) está confundida con la")
    print("    condición y este diseño NO puede separarlas. Dos comprobaciones parciales.")
    print("\n    (i) Entre las tareas, ¿crece el acierto con la posición (3..7)? Spearman a nivel")
    print("        de grabación, IC95 remuestreando personas. (Con solo 5 condiciones agregadas")
    print("        el Spearman no tendría potencia.)")
    posicion = np.array([ORDEN_PROTOCOLO.index(c) + 1 for c in datos["condicion"]])
    es_tarea = y == 1
    tendencia = {}
    for nombre in MODELOS:
        r = modelos[nombre]
        fila = {}
        for clave, valores in (("acierto", (r["y_pred"] == 1).astype(float)),
                               ("p_tarea", np.asarray(r["puntuaciones"], dtype=float))):
            rho = _spearman(posicion[es_tarea], valores[es_tarea])
            boot = bootstrap_por_persona(posicion[es_tarea], valores[es_tarea],
                                         g[es_tarea], _spearman,
                                         n_replicas=N_BOOTSTRAP, semilla=SEMILLA)
            fila[clave] = {"rho": rho, "ic95": list(boot["ic95"])}
            print(f"        {nombre:<22} posición ~ {clave:<8}: rho = {rho:+.3f}  "
                  f"IC95 [{boot['ic95'][0]:+.3f}, {boot['ic95'][1]:+.3f}]")
        tendencia[nombre] = fila
    resultados["posicion_protocolo"] = {"spearman_entre_tareas": tendencia}

    print("\n    (ii) El par adyacente OA (posición 2) frente a STROOP (posición 3): el contraste con")
    print("         menos separación temporal posible dentro de la sesión, con clases equilibradas.")
    d_p, meta_p = preparar_reposo_tarea(matriz, condiciones_reposo=("OA",),
                                        condiciones_tarea=("STROOP",))
    assert set(d_p["split"]) <= {"train", "val"}
    y_p = etiqueta_tarea(d_p, condiciones_reposo=("OA",), condiciones_tarea=("STROOP",))
    g_p = d_p["persona_uid"].to_numpy()
    print(f"    {meta_p['n_grabaciones']} grabaciones ({meta_p['n_reposo']} OA, "
          f"{meta_p['n_tarea']} STROOP), {meta_p['n_personas']} personas")
    par = {"meta": meta_p, "modelos": {}}
    for nombre in MODELOS:
        r = evaluar(d_p[columnas], y_p, g_p, FABRICAS[nombre](), nombre)
        _informe(r, float((y_p == 0).mean()), detalle=False)
        par["modelos"][nombre] = _guardar(r)
    resultados["posicion_protocolo"]["oa_frente_a_stroop"] = par
    _volcar(resultados)

    print("\n8.4 Sensibilidad: qué features entran a X")
    sin_alfa = [c for c in columnas
                if bloque_de_feature(c)[1] != "alfa" and not c.startswith("indice_fatiga")]
    sin_theta = [c for c in columnas
                 if bloque_de_feature(c)[1] != "theta" and not c.startswith("indice_fatiga")]
    copias = {copia for _, copia in duplicadas}
    subconjuntos = {
        "Principal": columnas,
        "Principal sin las columnas duplicadas": [c for c in columnas if c not in copias],
        "Principal + 6 contadores de procesamiento":
            columnas_de_features_tarea(datos, incluir_procesamiento=True),
        "Solo los 6 contadores de procesamiento": list(COLUMNAS_PROCESAMIENTO),
        "Solo región frontal": columnas_del_bloque(columnas, region="frontal"),
        "Solo región posterior": columnas_del_bloque(columnas, region="posterior"),
        "Sin banda theta ni índice de fatiga": sin_theta,
        "Sin banda alfa ni índice de fatiga": sin_alfa,
        "Sin theta, índice ni potencia relativa": [c for c in sin_theta if "_rel_" not in c],
        "Sin alfa, índice ni potencia relativa": [c for c in sin_alfa if "_rel_" not in c],
    }
    print("    (el índice de fatiga es theta/alfa: también se quita en las ablaciones de banda;")
    print("     la potencia relativa suma ~1 entre bandas, así que la banda quitada se deduce de")
    print("     las demás: las dos últimas filas quitan también la potencia relativa)")
    print(f"\n    {'subconjunto':<44}{'n':>4}   {'modelo':<20}{'BA/pliegue':>11}"
          f"{'IC95 BA':>19}{'AUC':>8}   {'Δ vs principal (pareado, IC95)':<34}")
    sensibilidad = []
    for etiqueta, cols in subconjuntos.items():
        verificar_sin_fuga(cols)
        for nombre in MODELOS:
            r = evaluar(datos[cols], y, g, FABRICAS[nombre](), nombre, con_ic=False)
            boot = bootstrap_por_persona(y, r["y_pred"], g, _ba,
                                         n_replicas=N_BOOTSTRAP, semilla=SEMILLA)
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
                  f"   [{boot['ic95'][0]:.3f}, {boot['ic95'][1]:.3f}]{m['auc_media']:>8.4f}"
                  f"   {texto_delta}")
            sensibilidad.append({
                "subconjunto": etiqueta, "modelo": nombre, "n_features": len(cols),
                "ba_media_pliegues": m["exactitud_balanceada_media"],
                "ba_sd_pliegues": m["exactitud_balanceada_sd"],
                "ic95_ba": list(boot["ic95"]), "auc_media": m["auc_media"],
                "delta_vs_principal": (None if delta is None else {
                    "diferencia": delta["diferencia"], "ic95": list(delta["ic95"])}),
                "columnas": cols})
    resultados["sensibilidad"] = sensibilidad
    _volcar(resultados)

    print("\n8.5 Veredicto completo frente a los criterios declarados")
    dir_oa = controles_reposo["OA solo frente a tareas"]["direccion"]
    criterio_d = bool(dir_oa["cumple_b"] and dir_oa["cumple_c"])
    criterios = [
        ("(a) la banda que más cuesta al barajarla entera es theta o alfa",
         criterio_a, f"banda {banda_max}"),
        ("(b) theta frontal hacia tarea en todos los pliegues (coef. y umbral)",
         criterio_b, valor_b),
        ("(c) alfa posterior hacia reposo en todos los pliegues (coef.)",
         criterio_c, valor_c),
        ("(d) (b) y (c) se mantienen en el control sin OC (OA frente a tareas)",
         criterio_d, f"(b) {'SI' if dir_oa['cumple_b'] else 'NO'} / (c) "
                     f"{'SI' if dir_oa['cumple_c'] else 'NO'}"),
    ]
    for nombre, cumple, valor in criterios:
        print(f"  [{'SI' if cumple else 'NO'}] {nombre:<76} {valor}")
    n_ok = sum(c for _, c, _ in criterios)
    print(f"\n  Criterios cumplidos: {n_ok} de {len(criterios)}")
    resultados["coherencia"]["veredicto"] = {
        "criterios": [{"criterio": n, "cumple": bool(c), "valor": v}
                      for n, c, v in criterios],
        "n_cumplidos": int(n_ok), "n_total": len(criterios)}
    _volcar(resultados)

    # ----------------------------------------------------------------------
    _titulo("9. Resolución por canal (suplemento)")
    print("matriz_rf.parquet agrega por región. Para ver qué canales llevan el efecto se")
    print("reagrega matriz_modelado.parquet con `construir_matriz_rf` y una región por canal")
    print("(la variante que documenta `agregar_por_region`), sobre las mismas grabaciones.")
    canales = analisis_por_canal(datos, y, g)
    print(f"\n    {'canal':<6}{'valid.':>7}{'d theta rel':>12}{'IC95 (personas)':>19}"
          f"{'d alfa rel':>12}{'IC95 (personas)':>19}{'d theta abs':>13}{'d alfa abs':>12}"
          f"{'d delta':>9}{'d beta':>8}{'BA 1 canal':>12}{'AUC':>8}")
    for canal, f in canales.sort_values("d_theta_rel", ascending=False).iterrows():
        marca = "*" if canal in REGIONES[REGION_THETA] else ("+" if canal in REGIONES[REGION_ALFA] else " ")
        lo_t, hi_t = f["ic95_d_theta_rel"]
        lo_a, hi_a = f["ic95_d_alfa_rel"]
        print(f"    {canal + marca:<6}{int(f['n_validas']):>7d}{f['d_theta_rel']:>+12.2f}"
              f"   [{lo_t:+.2f}, {hi_t:+.2f}]{f['d_alfa_rel']:>+12.2f}"
              f"   [{lo_a:+.2f}, {hi_a:+.2f}]{f['d_theta_logabs']:>+13.2f}"
              f"{f['d_alfa_logabs']:>+12.2f}{f['d_delta_rel']:>+9.2f}{f['d_beta_rel']:>+8.2f}"
              f"{f.get('ba_un_canal', float('nan')):>12.3f}{f.get('auc_un_canal', float('nan')):>8.3f}")
    print("    (* = región frontal, donde se espera theta; + = región posterior, donde se espera")
    print("     alfa; `rel` = potencia relativa, `abs` = log10 de la potencia absoluta. La relativa")
    print("     suma 1 entre bandas: si alfa baja, las demás suben aunque no cambien en absoluto,")
    print("     así que la absoluta es la comprobación de que el efecto en theta no es composicional.")
    print("     `BA 1 canal` = logística con solo theta y alfa (relativo y absoluto) de ese canal,")
    print("     validación agrupada por persona)")

    print(f"\n    Potencia ABSOLUTA (log10), con IC95 remuestreando personas:")
    print(f"    {'canal':<6}{'d theta abs':>12}{'IC95 (personas)':>19}"
          f"{'d alfa abs':>12}{'IC95 (personas)':>19}")
    for canal, f in canales.sort_values("d_theta_logabs", ascending=False).iterrows():
        marca = "*" if canal in REGIONES[REGION_THETA] else ("+" if canal in REGIONES[REGION_ALFA] else " ")
        lo_t, hi_t = f["ic95_d_theta_logabs"]
        lo_a, hi_a = f["ic95_d_alfa_logabs"]
        print(f"    {canal + marca:<6}{f['d_theta_logabs']:>+12.2f}   [{lo_t:+.2f}, {hi_t:+.2f}]"
              f"{f['d_alfa_logabs']:>+12.2f}   [{lo_a:+.2f}, {hi_a:+.2f}]")
    resultados["canales"] = canales
    _volcar(resultados)

    # ----------------------------------------------------------------------
    _titulo("10. Figuras")
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    figura_confusion([(n, modelos[n]) for n in MODELOS], len(datos),
                     RUTA_FIGURAS / "reposo_tarea_confusion.png")
    figura_por_condicion(
        [(n, desgloses[n], modelos[n]["metricas"]["exactitud_balanceada_media"],
          modelos[n]["ic95_ba"]) for n in MODELOS],
        RUTA_FIGURAS / "reposo_tarea_por_condicion.png")
    figura_canales(canales, RUTA_FIGURAS / "reposo_tarea_canales.png")

    # ----------------------------------------------------------------------
    _titulo("11. Resumen")
    filas = []
    for nombre, r in modelos.items():
        m = r["metricas"]
        filas.append({
            "clasificador": nombre,
            "BA/pliegue": f"{m['exactitud_balanceada_media']:.4f}",
            "IC95": f"[{r['ic95_ba'][0]:.3f}, {r['ic95_ba'][1]:.3f}]",
            "AUC": f"{m['auc_media']:.4f}" if "auc_media" in m else "-",
            "AP reposo": f"{r['ap_reposo']:.4f}" if "ap_reposo" in r else "-",
            "sens. (tarea)": f"{m['sensibilidad']:.3f}",
            "espec. (reposo)": f"{m['especificidad']:.3f}"})
    print(pd.DataFrame(filas).to_string(index=False))
    print(f"\n  (azar de la precisión media de reposo = prevalencia = {prevalencia_reposo:.4f})")
    print(f"  Coherencia neurofisiológica: {n_ok} de {len(criterios)} criterios cumplidos")
    print(f"  Control de etiquetas barajadas centrado en 0.5: {'sí' if control_ok else 'NO'}")
    for etiqueta, salida in controles_reposo.items():
        cifras = "  ".join(
            f"{n} {salida['modelos'][n]['metricas']['exactitud_balanceada_media']:.3f}"
            for n in MODELOS)
        print(f"  {etiqueta:<26}: {cifras}")

    resultados["modelos"] = {n: _guardar(r) for n, r in modelos.items()}
    _volcar(resultados)
    print(f"\nResultados -> {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
