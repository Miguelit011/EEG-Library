"""
Clasificación binaria de reposo: ojos abiertos (OA) frente a ojos cerrados (OC).

Control positivo del pipeline. El bloqueo del alfa posterior al abrir los ojos
(efecto Berger) es el fenómeno mejor establecido del EEG de superficie: si las
features de la Fase 13 lo conservan, los resultados negativos sobre el VAS
(Fases 14, 14b y 16) se atribuyen a la fatiga percibida y no a que el
procesamiento destruya la señal. No pretende construir un clasificador útil.

Solo intervienen las grabaciones de reposo (OA y OC); el resto de condiciones
quedan fuera. La unidad de análisis es la **grabación**: a diferencia del VAS,
la condición es una propiedad de cada registro. Las grabaciones de una misma
persona no son independientes, así que la validación cruzada y los intervalos
agrupan por `persona_uid` (nunca por `sujeto_id`).

Se comparan dos modelos (regresión logística y Random Forest) contra dos
baselines triviales y contra una referencia de una sola variable, y se
comprueba si lo que pesa en ellos es neurofisiológicamente coherente.

**Solo usa `train` + `val`.** El conjunto de test se aparta en el paso 1 y no
vuelve a tocarse. `preparar_reposo` se niega a incluirlo sin permiso explícito.

No modifica el pipeline congelado: todo sale de `matriz_rf.parquet` (Fase 13),
salvo el suplemento por canal, que reagrega `matriz_modelado.parquet` (Fase 11.5)
con la propia `construir_matriz_rf` y regiones de un solo canal.

Ejecutar (desde la raíz del repositorio):
    python analisis/analisis_oa_oc.py

La ubicación del dataset se toma de la variable de entorno `EEG_DATA`, con
`C:\\Users\\migue\\Desktop\\TFG\\EEG-Data` por defecto (ver `analisis/README.md`).
"""

import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_library.modelado.reposo import (
    COLUMNA_FUGA,
    COLUMNAS_PROCESAMIENTO,
    UmbralUnaVariable,
    auc_roc,
    bloque_de_feature,
    bootstrap_diferencia_por_persona,
    bootstrap_por_persona,
    coeficientes_por_pliegue,
    columnas_de_features_reposo,
    columnas_del_bloque,
    columnas_duplicadas,
    etiqueta_binaria,
    evaluar_reposo,
    importancia_permutacion_cv,
    importancia_por_bloque,
    metricas_binarias,
    permutar_etiquetas_dentro_de_persona,
    pliegue_por_fila,
    preparar_reposo,
    tamanos_de_efecto,
    umbral_una_variable,
    verificar_sin_fuga,
)
from eeg_library.preprocesado.quality import CANALES_EPOC
from eeg_library.procesado.bands import BANDAS_DEFECTO
from eeg_library.procesado.matrix import (
    REGIONES,
    columnas_de_features,
    construir_matriz_rf,
)

# La salida lleva acentos, ² y signos. En Windows, redirigir a un fichero usa
# cp1252 por defecto y el script moriría con UnicodeEncodeError justo al
# archivar el resultado, que es el caso de uso que importa.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEMILLA = 42
N_PLIEGUES = 5
N_BOOTSTRAP = 2000
N_BOOTSTRAP_CANALES = 2000
N_REPETICIONES_PERMUTACION = 5
N_PERMUTACIONES_ETIQUETAS = 20

RUTA_DATOS = Path(os.environ.get("EEG_DATA", r"C:\Users\migue\Desktop\TFG\EEG-Data"))
RUTA_MATRIZ = RUTA_DATOS / "processed" / "matriz_rf.parquet"
RUTA_MATRIZ_EPOCAS = RUTA_DATOS / "processed" / "matriz_modelado.parquet"
RUTA_GRABACIONES = RUTA_DATOS / "metadata" / "labels_grabaciones.csv"
RUTA_RESULTADOS = RUTA_DATOS / "processed" / "oa_oc.json"
RUTA_FIGURAS = Path(__file__).resolve().parents[1] / "figures"

# La feature de la referencia de una sola variable, elegida por la teoría (el
# ritmo alfa de la región occipito-parietal) y no por los datos.
FEATURE_REFERENCIA = "posterior_alfa_rel_media"
CANALES_POSTERIORES = REGIONES["posterior"]

# Criterios de coherencia neurofisiológica, declarados ANTES de ejecutar. El
# resultado es coherente si se cumplen los cuatro:
#  (a) la banda alfa concentra al menos esta fracción de la importancia por
#      permutación agregada por bloque. Ojo con el listón: alfa tiene 18 de las
#      70 features (25.7 %), así que un reparto sin criterio le daría ~0.26. El
#      0.30 del plan inicial apenas lo superaba y no discriminaba nada; se sube
#      a 0.50 (el doble de lo que le corresponde por número de features) antes
#      de ver ningún resultado.
FRACCION_ALFA_MINIMA = 0.50
#  (b) alguna feature de alfa posterior está entre las N más importantes.
RANGO_ALFA_POSTERIOR = 5
#  (c) barajar de una vez todas las features de una banda cuesta más exactitud
#      con alfa que con cualquier otra banda (importancia conjunta, que no se
#      diluye entre features casi duplicadas).
#  (d) el efecto apunta a OC: el coeficiente neto de las features `_media` de
#      alfa posterior en la logística es positivo en TODOS los pliegues, y el
#      umbral de una sola variable aprende signo +1 en TODOS los pliegues.

# Paleta: ranuras 1 (azul) y 2 (naranja) de la paleta de referencia, validadas
# con el validador de la guía de visualización en claro y oscuro (ΔE CVD 24.7,
# visión normal 33.6, contraste >= 3:1). La identidad es fija en todas las
# figuras: azul = OA, naranja = OC.
COLOR_OA = "#2a78d6"
COLOR_OC = "#eb6834"
COLOR_GRIS = "#c3c2b7"
COLOR_TEXTO = "#0b0b0b"
COLOR_TEXTO_2 = "#52514e"
COLOR_REJILLA = "#e0e0dc"
# Rampa secuencial de un solo tono para la matriz de confusión.
RAMPA_AZUL = LinearSegmentedColormap.from_list(
    "azul_secuencial", ["#eef4fc", "#2a78d6", "#0f3d80"])


def _titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def logistica():
    """Frontera lineal; sus coeficientes tienen signo y permiten ver la dirección."""
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"))


def bosque():
    """Misma configuración que el Random Forest de las Fases 14 y 14b."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                               class_weight="balanced", random_state=SEMILLA,
                               n_jobs=-1))


def _ba(y_true, y_pred):
    return metricas_binarias(y_true, y_pred)["exactitud_balanceada"]


def _d_de_cohen(y_true, valores):
    """d de Cohen (OC - OA) de un vector; los NaN se ignoran."""
    return float(tamanos_de_efecto(pd.DataFrame({"x": valores}), y_true).iloc[0])


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


def evaluar(X, y, g, modelo, nombre, con_probabilidad=True, con_ic=True):
    """Valida por persona, calcula los IC bootstrap por persona y lo devuelve."""
    r = evaluar_reposo(X, y, g, modelo, nombre, n_splits=N_PLIEGUES,
                       con_probabilidad=con_probabilidad)
    if con_ic:
        boot = bootstrap_por_persona(y, r["y_pred"], g, _ba,
                                     n_replicas=N_BOOTSTRAP, semilla=SEMILLA)
        r["ic95_ba"] = boot["ic95"]
        if con_probabilidad:
            boot_auc = bootstrap_por_persona(y, r["puntuaciones"], g, auc_roc,
                                             n_replicas=N_BOOTSTRAP,
                                             semilla=SEMILLA)
            r["ic95_auc"] = boot_auc["ic95"]
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
    if "auc_media" in m:
        print(f"  AUC, media por pliegue                  : {m['auc_media']:.4f}  "
              f"(amontonada {m['auc_agregado']:.4f})")
        if "ic95_auc" in r:
            lo, hi = r["ic95_auc"]
            print(f"    IC95 (bootstrap por persona)          : [{lo:.4f}, {hi:.4f}]")
    if not detalle:
        return
    print(f"  Sensibilidad (OC detectadas)            : {m['sensibilidad']:.4f}")
    print(f"  Especificidad (OA detectadas)           : {m['especificidad']:.4f}")
    print(f"  F1 = {m['f1']:.4f}   MCC = {m['mcc']:.4f}   "
          f"exactitud simple = {m['exactitud']:.4f}")
    c = m["matriz_confusion"]
    print("  Matriz de confusión (filas = real, columnas = predicha):")
    print(f"                  pred OA   pred OC")
    print(f"      real OA     {c['vn']:7d}   {c['fp']:7d}")
    print(f"      real OC     {c['fn']:7d}   {c['vp']:7d}")
    print("  Por pliegue     : "
          + "  ".join(f"{b:.3f}" for b in m["exactitud_balanceada_pliegues"]))


def _guardar(r):
    """Lo que se archiva de una evaluación (sin los vectores por fila)."""
    m = dict(r["metricas"])
    salida = {"metricas": m}
    for clave in ("ic95_ba", "ic95_auc"):
        if clave in r:
            salida[clave] = list(r[clave])
    return salida


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


def figura_confusion(modelos, n, ruta):
    """
    Matrices de confusión de los dos modelos, lado a lado y con la misma escala.

    El color es el % de cada clase **real** (filas que suman 100 %), no el
    recuento: así el desbalance de clases no cambia el color de las celdas.
    Rampa secuencial de un solo tono, celdas separadas por un hueco blanco.
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
        ax.set_xticks([0, 1], ["OA", "OC"])
        ax.set_yticks([0, 1], ["OA", "OC"])
        ax.grid(False)
        ax.tick_params(which="both", length=0, colors=COLOR_TEXTO_2, labelsize=11)
        for lado in ax.spines.values():
            lado.set_visible(False)
        ax.set_xlabel("Clase predicha", fontsize=10.5, color=COLOR_TEXTO_2,
                      labelpad=8)
        ax.set_ylabel("Clase real", fontsize=10.5, color=COLOR_TEXTO_2, labelpad=8)
        lo, hi = r["ic95_ba"]
        # Titulo y subtitulo por separado: en una sola cadena el IC del panel
        # derecho se sale del lienzo.
        ax.set_title(nombre, fontsize=12, color=COLOR_TEXTO, loc="left", pad=30,
                     fontweight="bold")
        ax.text(0, 1.045, f"exactitud balanceada {m['exactitud_balanceada_media']:.3f}"
                f" · IC95 [{lo:.3f}, {hi:.3f}]", transform=ax.transAxes,
                fontsize=9.5, color=COLOR_TEXTO_2, va="bottom")
    fig.suptitle("Ojos abiertos (OA) frente a ojos cerrados (OC): matriz de confusión",
                 x=0.012, ha="left", fontsize=13.5, color=COLOR_TEXTO,
                 fontweight="bold")
    fig.text(0.012, 0.012,
             f"Validación cruzada de {N_PLIEGUES} pliegues agrupada por persona; "
             f"n = {n} grabaciones de train + val (el conjunto de test permanece sin abrir).\n"
             "El color es el porcentaje de cada clase real; cada fila suma 100 %.",
             fontsize=8, color=COLOR_TEXTO_2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.savefig(ruta, facecolor="white")
    plt.close(fig)
    print(f"  figura -> {ruta}")


def figura_alfa_posterior(x_oa, x_oc, umbral, d, auc, ruta):
    """
    Distribución del alfa relativo posterior por condición, con el corte de la
    referencia de una sola variable. Rellenos al ~20 % con contorno de 2 px:
    donde se solapan siguen leyéndose las dos.
    """
    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=200)
    fig.patch.set_facecolor("white")
    _estilo(ax)
    todos = np.concatenate([x_oa, x_oc])
    bins = np.linspace(np.nanmin(todos), np.nanmax(todos), 38)

    picos = {}
    for x, color, nombre in ((x_oa, COLOR_OA, "Ojos abiertos (OA)"),
                             (x_oc, COLOR_OC, "Ojos cerrados (OC)")):
        cuentas, _ = np.histogram(x, bins=bins)
        ax.hist(x, bins=bins, color=color, alpha=0.20, zorder=3)
        ax.hist(x, bins=bins, histtype="step", color=color, linewidth=2.0,
                zorder=4, label=f"{nombre}  (n = {len(x)})")
        i = int(np.argmax(cuentas))
        picos[nombre] = ((bins[i] + bins[i + 1]) / 2, cuentas[i], color)

    ax.axvline(umbral, color=COLOR_TEXTO_2, linewidth=1.4, linestyle=(0, (5, 4)),
               zorder=5)
    tope = ax.get_ylim()[1]
    ax.set_ylim(0, tope * 1.12)
    ax.text(umbral, tope * 1.085, f" corte = {umbral:.3f}", fontsize=9,
            color=COLOR_TEXTO_2, va="top", ha="left")

    # Etiquetas directas: marca de color junto al texto en tinta (el texto no
    # lleva el color de la serie).
    for nombre, (px, py, color) in picos.items():
        ax.plot([px], [py + tope * 0.045], marker="o", markersize=8, color=color,
                markeredgecolor="white", markeredgewidth=2, zorder=6)
        ax.text(px, py + tope * 0.045, "   " + nombre.split(" (")[1][:-1], fontsize=10,
                color=COLOR_TEXTO, va="center", ha="left", zorder=6,
                fontweight="bold")

    ax.set_xlabel("Potencia relativa de alfa, región posterior (P7, P8, O1, O2)\n"
                  "media sobre las ventanas de la grabación", fontsize=10.5,
                  color=COLOR_TEXTO_2, labelpad=9)
    ax.set_ylabel("Grabaciones", fontsize=10.5, color=COLOR_TEXTO_2, labelpad=9)
    ax.set_title("Alfa relativo posterior: ojos abiertos frente a ojos cerrados",
                 fontsize=13.5, color=COLOR_TEXTO, pad=14, loc="left",
                 fontweight="bold")
    ax.text(0.985, 0.975, f"d de Cohen = {d:+.2f}\nAUC univariado = {auc:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
            color=COLOR_TEXTO_2, linespacing=1.6, zorder=10,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="white",
                      edgecolor=COLOR_REJILLA, linewidth=0.9))
    ax.legend(loc="center right", bbox_to_anchor=(0.985, 0.62), frameon=True,
              framealpha=0.95, fontsize=9, edgecolor=COLOR_REJILLA,
              facecolor="white", borderpad=0.7)

    fig.text(0.008, 0.012,
             "Train + val (el conjunto de test permanece sin abrir). El corte es el que "
             "maximiza la exactitud balanceada sobre todas\nlas grabaciones de análisis; "
             "es descriptivo, y la validación cruzada lo reestima dentro de cada pliegue.",
             fontsize=8, color=COLOR_TEXTO_2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(ruta, facecolor="white")
    plt.close(fig)
    print(f"  figura -> {ruta}")


def figura_canales(tabla, ruta):
    """
    Efecto de cerrar los ojos sobre el alfa relativo, canal a canal.

    Forma de énfasis: los canales posteriores en el color de OC (naranja), el
    resto en gris. El valor va en la punta de cada barra.
    """
    orden = tabla.sort_values("d_alfa_rel")
    fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=200)
    fig.patch.set_facecolor("white")
    _estilo(ax)
    ax.grid(axis="y", visible=False)
    colores = [COLOR_OC if c in CANALES_POSTERIORES else COLOR_GRIS
               for c in orden.index]
    ax.barh(orden.index, orden["d_alfa_rel"], color=colores, height=0.62, zorder=3)
    ic = np.array(orden["ic95_d_alfa_rel"].tolist())
    ax.errorbar(orden["d_alfa_rel"], np.arange(len(orden)),
                xerr=[orden["d_alfa_rel"] - ic[:, 0], ic[:, 1] - orden["d_alfa_rel"]],
                fmt="none", ecolor=COLOR_TEXTO_2, elinewidth=1.3, capsize=3,
                zorder=5)
    margen = 0.02 * ic[:, 1].max()
    for i, (canal, v) in enumerate(orden["d_alfa_rel"].items()):
        ax.text(ic[i, 1] + margen, i, f"{v:+.2f}", va="center", ha="left",
                fontsize=9, color=COLOR_TEXTO)
    ax.axvline(0, color=COLOR_TEXTO_2, linewidth=1.0, zorder=4)
    ax.set_xlim(0, ic[:, 1].max() * 1.2)
    ax.set_xlabel("d de Cohen (OC − OA) del alfa relativo", fontsize=10.5,
                  color=COLOR_TEXTO_2, labelpad=9)
    ax.set_title("Efecto de cerrar los ojos sobre el alfa, por canal", fontsize=13.5,
                 color=COLOR_TEXTO, pad=14, loc="left", fontweight="bold")
    manijas = [plt.Rectangle((0, 0), 1, 1, color=COLOR_OC),
               plt.Rectangle((0, 0), 1, 1, color=COLOR_GRIS)]
    ax.legend(manijas, ["Región posterior (P7, P8, O1, O2)", "Resto de canales"],
              loc="lower right", bbox_to_anchor=(0.99, 0.02), frameon=True,
              framealpha=0.95, fontsize=9, edgecolor=COLOR_REJILLA,
              facecolor="white", borderpad=0.7)
    fig.text(0.008, 0.012,
             "Train + val (el conjunto de test permanece sin abrir). Barras de error: IC95 "
             "remuestreando personas. Solo grabaciones donde\nel canal superó el control de "
             "calidad de la Fase 11.5. Diferencias de ~0.1-0.2 entre canales vecinos "
             "no son distinguibles.",
             fontsize=8, color=COLOR_TEXTO_2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(ruta, facecolor="white")
    plt.close(fig)
    print(f"  figura -> {ruta}")


# --------------------------------------------------------------------------
# Análisis
# --------------------------------------------------------------------------

def analisis_por_canal(datos, y, g):
    """
    Suplemento: qué canales llevan el efecto.

    Reagrega `matriz_modelado.parquet` con `construir_matriz_rf` y una región por
    canal (la variante que documenta `agregar_por_region`), restringida a las
    mismas grabaciones del análisis principal. Devuelve la tabla por canal.
    """
    # El filtro por archivo se aplica AL LEER: las epocas de las grabaciones
    # apartadas (test) no llegan a cargarse, no solo se descartan despues.
    epocas = pd.read_parquet(
        RUTA_MATRIZ_EPOCAS,
        filters=[("condicion", "in", ["OA", "OC"]),
                 ("archivo_origen", "in", datos["archivo"].tolist())])
    grabaciones = pd.read_csv(RUTA_GRABACIONES)
    with warnings.catch_warnings():
        # `construir_matriz_rf` (codigo congelado de la Fase 13) inserta
        # columnas de una en una y pandas avisa de la fragmentacion; no afecta
        # al resultado y llenaria el log.
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

    filas = []
    for canal in CANALES_EPOC:
        fila = {"canal": canal, "posterior": canal in CANALES_POSTERIORES}
        for banda in BANDAS_DEFECTO:
            d = tamanos_de_efecto(por_canal[[f"{canal}_{banda}_rel_media"]], y)
            fila[f"d_{banda}_rel"] = float(d.iloc[0])
        fila["d_alfa_logabs"] = float(tamanos_de_efecto(
            por_canal[[f"{canal}_alfa_logabs_media"]], y).iloc[0])
        boot = bootstrap_por_persona(
            y, por_canal[f"{canal}_alfa_rel_media"].to_numpy(dtype=float), g,
            _d_de_cohen, n_replicas=N_BOOTSTRAP_CANALES, semilla=SEMILLA)
        fila["ic95_d_alfa_rel"] = boot["ic95"]

        cols = [f"{canal}_alfa_rel_media", f"{canal}_alfa_logabs_media"]
        validas = por_canal[cols].notna().all(axis=1).to_numpy()
        fila["n_validas"] = int(validas.sum())
        fila["n_personas"] = int(len(set(g[validas])))
        if validas.sum() >= 100 and fila["n_personas"] >= N_PLIEGUES:
            r = evaluar_reposo(por_canal.loc[validas, cols], y[validas],
                               g[validas], logistica(), n_splits=N_PLIEGUES)
            fila["ba_alfa_un_canal"] = r["metricas"]["exactitud_balanceada_media"]
            fila["auc_alfa_un_canal"] = r["metricas"]["auc_media"]
        filas.append(fila)
    return pd.DataFrame(filas).set_index("canal")


def main():
    resultados = {}

    # ----------------------------------------------------------------------
    _titulo("1. Datos - grabaciones de reposo de train + val")
    matriz = pd.read_parquet(RUTA_MATRIZ)
    print(f"Matriz completa      : {matriz.shape[0]} grabaciones, "
          f"{len(columnas_de_features(matriz))} features (Fase 13)")

    datos, meta = preparar_reposo(matriz)
    print(f"Apartado (test)      : {meta['n_apartadas']} grabaciones de reposo, "
          f"{meta['n_personas_apartadas']} personas - no se vuelve a leer")
    print(f"Sin persona_uid      : {meta['n_sin_persona_uid']} grabaciones descartadas "
          f"(sin él no se puede agrupar)")
    print(f"En análisis          : {meta['n_grabaciones']} grabaciones, "
          f"{meta['n_personas']} personas, {meta['n_sesiones']} sesiones")
    print(f"  por condición      : {meta['por_condicion']}")
    print(f"  por split          : {meta['por_split']}")
    por_persona = datos.groupby("persona_uid").size()
    print(f"  grabaciones/persona: mediana {por_persona.median():.0f}, "
          f"rango [{por_persona.min()}, {por_persona.max()}]")
    print(f"  NaN en features    : {int(datos[columnas_de_features(datos)].isna().any(axis=1).sum())} "
          f"grabaciones con alguna región sin canales válidos (imputación por mediana "
          f"dentro de cada pliegue)")

    y = etiqueta_binaria(datos)
    g = datos["persona_uid"].to_numpy()
    columnas = columnas_de_features_reposo(datos)
    X = datos[columnas]

    excluidas = [c for c in columnas_de_features(datos) if c not in columnas]
    print(f"\nFeatures de la Fase 13 : {len(columnas_de_features(datos))}")
    print(f"Features en X          : {len(columnas)}   (excluidas {len(excluidas)})")
    print("\nColumnas excluidas de X y por qué:")
    print(f"  {COLUMNA_FUGA:<26} ES LA ETIQUETA: posición de la condición en el protocolo")
    for condicion, valores in datos.groupby("condicion")[COLUMNA_FUGA].unique().items():
        print(f"    {condicion} -> {sorted(valores.tolist())}")
    print(f"\n  Contadores de procesamiento (no son señal cerebral, pero discriminan):")
    filas_proc = []
    for c in COLUMNAS_PROCESAMIENTO:
        d = float(tamanos_de_efecto(datos[[c]], y).iloc[0])
        p = float(stats.mannwhitneyu(datos.loc[y == 1, c], datos.loc[y == 0, c]).pvalue)
        filas_proc.append({"columna": c, "d_cohen": d, "p_mann_whitney": p})
        print(f"    {c:<26} d = {d:+.3f}   p = {p:.4f}")
    verificar_sin_fuga(columnas)
    assert COLUMNA_FUGA not in columnas
    print(f"\nComprobado: `{COLUMNA_FUGA}` no está en las {len(columnas)} features de X.")

    duplicadas = columnas_duplicadas(X)
    print(f"\nColumnas de X que son COPIAS EXACTAS de otra (mismo valor en todas las filas):")
    for conservada, copia in duplicadas:
        print(f"    {copia:<32} = {conservada}")
    print(f"  {len(duplicadas)} duplicados: {len(columnas) - len(duplicadas)} columnas distintas "
          f"de las {len(columnas)}. No se quitan de X (se reutiliza la matriz tal cual);")
    print("  la fila «sin duplicados» de la sensibilidad (7.2) comprueba que no cambian nada.")

    resultados["datos"] = {
        **meta, "n_features_fase13": len(columnas_de_features(datos)),
        "n_features_x": len(columnas), "columnas_excluidas": excluidas,
        "contadores_procesamiento": filas_proc, "columnas_x": columnas,
        "columnas_duplicadas": [list(par) for par in duplicadas],
    }

    # ----------------------------------------------------------------------
    _titulo("2. Baselines triviales")
    print("Un clasificador que no usa X. Con la exactitud balanceada, el azar es 0.5")
    print("exacto: si aquí saliera otra cosa, el montaje estaría roto.")
    modelos = {}
    for nombre, modelo in (
            ("Baseline: clase mayoritaria",
             DummyClassifier(strategy="most_frequent")),
            ("Baseline: azar estratificado",
             DummyClassifier(strategy="stratified", random_state=SEMILLA))):
        modelos[nombre] = evaluar(X, y, g, modelo, nombre)
        _informe(modelos[nombre], detalle=False)
    sanidad = all(abs(r["metricas"]["exactitud_balanceada_media"] - 0.5) < 0.05
                  for r in modelos.values())
    print(f"\n  [{'SI' if sanidad else 'NO'}] Los dos baselines dan ~0.5 de exactitud balanceada")
    resultados["baselines"] = {n: _guardar(r) for n, r in modelos.items()}
    resultados["baselines_sanos"] = bool(sanidad)

    # ----------------------------------------------------------------------
    _titulo("3. Modelo A - regresión logística")
    print("Imputación por mediana + tipificación + LogisticRegression(class_weight=")
    print("'balanced'), todo ajustado solo con el pliegue de entrenamiento.")
    modelos["Regresión logística"] = evaluar(X, y, g, logistica(), "Regresión logística")
    _informe(modelos["Regresión logística"])

    # ----------------------------------------------------------------------
    _titulo("4. Modelo B - Random Forest")
    print("RandomForestClassifier(500 árboles, hoja>=5, class_weight='balanced'): la misma")
    print("configuración que las Fases 14 y 14b, para que el contraste con el VAS sea directo.")
    modelos["Random Forest"] = evaluar(X, y, g, bosque(), "Random Forest")
    _informe(modelos["Random Forest"])

    # ----------------------------------------------------------------------
    _titulo("5. Referencia de una sola variable")
    print(f"Un único corte sobre `{FEATURE_REFERENCIA}`, con su umbral y su signo")
    print("reestimados dentro de cada pliegue. Responde a: hacía falta un modelo?")
    nombre_ref = f"Umbral en {FEATURE_REFERENCIA}"
    modelos[nombre_ref] = evaluar(X, y, g, UmbralUnaVariable(FEATURE_REFERENCIA),
                                  nombre_ref, con_probabilidad=False)
    _informe(modelos[nombre_ref])

    pliegues = pliegue_por_fila(g, N_PLIEGUES)
    cortes = []
    for p in range(1, N_PLIEGUES + 1):
        ajustado = UmbralUnaVariable(FEATURE_REFERENCIA).fit(
            X[pliegues != p], y[pliegues != p])
        cortes.append({"pliegue": p, "umbral": ajustado.umbral_,
                       "signo": ajustado.signo_})
    print("\n  Corte aprendido por pliegue (signo +1 = «OC si alfa > corte»):")
    for c in cortes:
        print(f"    pliegue {c['pliegue']}: umbral {c['umbral']:.4f}  signo {c['signo']:+d}")
    signo_estable_positivo = all(c["signo"] == 1 for c in cortes)

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
                 ("Random Forest", nombre_ref)):
        c = bootstrap_diferencia_por_persona(
            y, modelos[a]["y_pred"], modelos[b]["y_pred"], g, _ba,
            n_replicas=N_BOOTSTRAP, semilla=SEMILLA)
        comparaciones.append({
            "a": a, "b": b, "diferencia": c["diferencia"],
            "ic95": list(c["ic95"]), "prop_a_mayor": c["prop_positivas"]})
        print(f"    {a} - {b}: {c['diferencia']:+.4f}  IC95 [{c['ic95'][0]:+.4f}, "
              f"{c['ic95'][1]:+.4f}]  (A > B en el {c['prop_positivas']:.1%} de las réplicas)")
    resultados["comparaciones_pareadas"] = comparaciones

    # ----------------------------------------------------------------------
    _titulo("6. Coherencia neurofisiológica - qué pesa en el modelo")
    print("Criterios declarados antes de ejecutar (ver las constantes del guion):")
    print(f"  (a) alfa concentra >= {FRACCION_ALFA_MINIMA:.0%} de la importancia agregada")
    print(f"  (b) alguna feature de alfa posterior entre las {RANGO_ALFA_POSTERIOR} primeras")
    print("  (c) alfa es la banda cuya permutación conjunta cuesta más exactitud")
    print("  (d) el efecto apunta a OC: coeficiente neto de alfa posterior > 0 en todos")
    print("      los pliegues y umbral de una variable con signo +1 en todos")

    print("\n6.1 Importancia por permutación del Random Forest, OUT-OF-FOLD")
    print(f"    (caída de exactitud balanceada de validación al barajar la columna;")
    print(f"    {N_REPETICIONES_PERMUTACION} barajados x {N_PLIEGUES} pliegues)")
    imp = importancia_permutacion_cv(X, y, g, bosque(), n_splits=N_PLIEGUES,
                                     n_repeticiones=N_REPETICIONES_PERMUTACION,
                                     semilla=SEMILLA)
    d_todas = tamanos_de_efecto(X, y)
    coef = coeficientes_por_pliegue(X, y, g, logistica(), n_splits=N_PLIEGUES)
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
    bloques["bloque/posterior+alfa"] = columnas_del_bloque(
        columnas, region="posterior", banda="alfa")
    conjunta = importancia_permutacion_cv(X, y, g, bosque(), bloques=bloques,
                                          n_splits=N_PLIEGUES,
                                          n_repeticiones=N_REPETICIONES_PERMUTACION,
                                          semilla=SEMILLA)
    conjunta["n_features"] = [len(bloques[k]) for k in conjunta.index]
    print(f"\n    {'bloque':<24}{'features':>9}{'caída BA':>11}{'SD pliegues':>13}")
    for nombre, f in conjunta.sort_values("caida_media", ascending=False).iterrows():
        print(f"    {nombre:<24}{int(f['n_features']):>9d}{f['caida_media']:>+11.4f}"
              f"{f['sd_entre_pliegues']:>13.4f}")

    print("\n6.4 Dirección del efecto")
    alfa_posterior = columnas_del_bloque(columnas, region="posterior", banda="alfa")
    alfa_media = [c for c in alfa_posterior if c.endswith("_media")]
    neto = coef.loc[alfa_media].sum()
    print(f"    Features `_media` de alfa posterior: {alfa_media}")
    print("    Coeficiente neto (suma, en unidades de SD) por pliegue de la logística: "
          + "  ".join(f"{v:+.3f}" for v in neto))
    print(f"    Umbral de una variable: signos por pliegue "
          f"{[c['signo'] for c in cortes]}")
    print(f"    d de Cohen de las features de alfa posterior: "
          + "  ".join(f"{c} {d_todas[c]:+.2f}" for c in alfa_media))

    print("\n6.5 Mayores tamaños de efecto univariados (d de Cohen, OC - OA)")
    por_d = d_todas.reindex(d_todas.abs().sort_values(ascending=False).index)
    for nombre, v in por_d.head(12).items():
        print(f"    {nombre:<34}{v:>+8.3f}")

    # ---- Veredicto ----
    fraccion_alfa = por_bloque["por_banda"].get("alfa", 0.0)
    rangos = {c: int(np.flatnonzero(ranking.index == c)[0]) + 1 for c in alfa_posterior}
    mejor_rango = min(rangos.values())
    banda_max = conjunta.loc[[f"banda/{b}" for b in BANDAS_DEFECTO],
                             "caida_media"].idxmax()
    criterios = [
        (f"(a) alfa concentra >= {FRACCION_ALFA_MINIMA:.0%} de la importancia",
         fraccion_alfa >= FRACCION_ALFA_MINIMA, f"{fraccion_alfa:.1%}"),
        (f"(b) alfa posterior entre las {RANGO_ALFA_POSTERIOR} primeras features",
         mejor_rango <= RANGO_ALFA_POSTERIOR, f"mejor puesto {mejor_rango}"),
        ("(c) alfa es la banda que más cuesta al barajarla entera",
         banda_max == "banda/alfa", f"{banda_max}"),
        ("(d) dirección hacia OC (coef. neto > 0 y umbral +1 en todos los pliegues)",
         bool((neto > 0).all() and signo_estable_positivo),
         f"coef {'+' if (neto > 0).all() else 'x'} / umbral "
         f"{'+' if signo_estable_positivo else 'x'}"),
    ]
    print("\n6.6 Veredicto frente a los criterios declarados")
    for nombre, cumple, valor in criterios:
        print(f"  [{'SI' if cumple else 'NO'}] {nombre:<76} {valor}")
    n_ok = sum(c for _, c, _ in criterios)
    print(f"\n  Criterios cumplidos: {n_ok} de {len(criterios)}")

    resultados["coherencia"] = {
        "criterios_declarados": {
            "fraccion_alfa_minima": FRACCION_ALFA_MINIMA,
            "rango_alfa_posterior": RANGO_ALFA_POSTERIOR},
        "importancia_por_permutacion": tabla.reset_index().rename(
            columns={"index": "feature"}).to_dict(orient="records"),
        "importancia_por_bloque": por_bloque,
        "importancia_conjunta": conjunta.reset_index().rename(
            columns={"index": "bloque"}).to_dict(orient="records"),
        "coeficiente_neto_alfa_posterior_por_pliegue": neto.tolist(),
        "features_alfa_posterior": alfa_posterior,
        "rangos_alfa_posterior": rangos,
        "veredicto": {
            "criterios": [{"criterio": n, "cumple": bool(c), "valor": v}
                          for n, c, v in criterios],
            "n_cumplidos": int(n_ok), "n_total": len(criterios)},
    }

    # ----------------------------------------------------------------------
    _titulo("7. Controles y sensibilidad")
    print("7.1 Control negativo: etiquetas barajadas DENTRO de cada persona")
    print(f"    {N_PERMUTACIONES_ETIQUETAS} barajados por modelo. Destruye la relación con las features")
    print("    pero conserva las personas y cuántas OA/OC tiene cada una. Tiene que dar ~0.5.")
    nulos = {}
    for nombre, fabrica in (("Regresión logística", logistica),
                            ("Random Forest", bosque)):
        valores = []
        for s in range(N_PERMUTACIONES_ETIQUETAS):
            y_perm = permutar_etiquetas_dentro_de_persona(y, g, semilla=SEMILLA + s)
            r = evaluar_reposo(X, y_perm, g, fabrica(), n_splits=N_PLIEGUES)
            valores.append(r["metricas"]["exactitud_balanceada_media"])
        valores = np.array(valores)
        real = modelos[nombre]["metricas"]["exactitud_balanceada_media"]
        n_supera = int((valores >= real).sum())
        p_emp = (1 + n_supera) / (1 + len(valores))
        nulos[nombre] = {
            "media": float(valores.mean()), "sd": float(valores.std(ddof=1)),
            "min": float(valores.min()), "max": float(valores.max()),
            "p95": float(np.percentile(valores, 95)), "real": float(real),
            "n_nulos_iguales_o_mayores": n_supera,
            "p_empirico": float(p_emp), "valores": valores.tolist()}
        print(f"    {nombre:<22} nulo {valores.mean():.4f} (SD {valores.std(ddof=1):.4f}, "
              f"rango [{valores.min():.3f}, {valores.max():.3f}])  real {real:.4f}  "
              f"({(real - valores.mean()) / valores.std(ddof=1):.0f} SD por encima)  "
              f"nulos >= real: {n_supera} de {len(valores)}")
    print(f"    (con {N_PERMUTACIONES_ETIQUETAS} barajados el p empirico no puede bajar de "
          f"{1 / (1 + N_PERMUTACIONES_ETIQUETAS):.3f}: es la resolucion del control, no un "
          f"valor a leer)")
    control_ok = all(abs(v["media"] - 0.5) < 0.03 for v in nulos.values())
    print(f"\n  [{'SI' if control_ok else 'NO'}] El nulo está centrado en 0.5 "
          f"(el arnés no produce acierto de la nada)")
    resultados["control_etiquetas_barajadas"] = {**nulos, "centrado_en_0_5": bool(control_ok)}

    print("\n7.2 Sensibilidad: qué features entran a X")
    sin_alfa = [c for c in columnas
                if bloque_de_feature(c)[1] != "alfa" and not c.startswith("indice_fatiga")]
    copias = {copia for _, copia in duplicadas}
    subconjuntos = {
        "Principal": columnas,
        "Principal sin las columnas duplicadas": [c for c in columnas if c not in copias],
        "Principal + 6 contadores de procesamiento":
            columnas_de_features_reposo(datos, incluir_procesamiento=True),
        "Solo los 6 contadores de procesamiento": list(COLUMNAS_PROCESAMIENTO),
        "Solo región posterior": columnas_del_bloque(columnas, region="posterior"),
        "Sin banda alfa ni índice de fatiga": sin_alfa,
        "Sin alfa, índice ni potencia relativa":
            [c for c in sin_alfa if "_rel_" not in c],
    }
    print("    (el índice de fatiga es theta/alfa: también se quita en las ablaciones de alfa;")
    print("     la potencia relativa suma ~1 entre bandas, así que alfa se deduce de las demás)")
    print(f"\n    {'subconjunto':<44}{'n':>4}   {'modelo':<20}{'BA/pliegue':>11}"
          f"{'IC95 BA':>19}{'AUC':>8}   {'Δ vs principal (pareado, IC95)':<34}")
    sensibilidad = []
    for etiqueta, cols in subconjuntos.items():
        verificar_sin_fuga(cols)
        for nombre, fabrica in (("Regresión logística", logistica),
                                ("Random Forest", bosque)):
            r = evaluar(datos[cols], y, g, fabrica(), nombre, con_ic=False)
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

    # ----------------------------------------------------------------------
    _titulo("8. Resolución por canal (suplemento)")
    print("matriz_rf.parquet agrega por región. Para ver qué canales llevan el efecto se")
    print("reagrega matriz_modelado.parquet con `construir_matriz_rf` y una región por canal")
    print("(la variante que documenta `agregar_por_region`), sobre las mismas grabaciones.")
    canales = analisis_por_canal(datos, y, g)
    print(f"\n    {'canal':<6}{'valid.':>7}{'d alfa rel':>12}{'IC95 (personas)':>19}"
          f"{'d alfa log':>12}{'d delta':>9}{'d theta':>9}{'d beta':>9}"
          f"{'BA 1 canal':>12}{'AUC':>8}")
    for canal, f in canales.sort_values("d_alfa_rel", ascending=False).iterrows():
        marca = "*" if f["posterior"] else " "
        lo, hi = f["ic95_d_alfa_rel"]
        print(f"    {canal + marca:<6}{int(f['n_validas']):>7d}{f['d_alfa_rel']:>+12.2f}"
              f"   [{lo:+.2f}, {hi:+.2f}]{f['d_alfa_logabs']:>+12.2f}"
              f"{f['d_delta_rel']:>+9.2f}{f['d_theta_rel']:>+9.2f}"
              f"{f['d_beta_rel']:>+9.2f}{f['ba_alfa_un_canal']:>12.3f}"
              f"{f['auc_alfa_un_canal']:>8.3f}")
    print("    (* = región posterior; `BA 1 canal` = logística con solo el alfa relativo y")
    print("     absoluto de ese canal, validación agrupada por persona)")
    resultados["canales"] = canales

    # ----------------------------------------------------------------------
    _titulo("9. Figuras")
    RUTA_FIGURAS.mkdir(exist_ok=True)
    figura_confusion([(n, modelos[n]) for n in ("Regresión logística", "Random Forest")],
                     len(datos), RUTA_FIGURAS / "oa_oc_confusion.png")
    figura_alfa_posterior(x_ref[y == 0][~np.isnan(x_ref[y == 0])],
                          x_ref[y == 1][~np.isnan(x_ref[y == 1])],
                          umbral_global, d_ref, auc_univariado,
                          RUTA_FIGURAS / "oa_oc_alfa_posterior.png")
    figura_canales(canales, RUTA_FIGURAS / "oa_oc_canales.png")

    # ----------------------------------------------------------------------
    _titulo("10. Resumen")
    filas = []
    for nombre, r in modelos.items():
        m = r["metricas"]
        filas.append({
            "clasificador": nombre,
            "BA/pliegue": f"{m['exactitud_balanceada_media']:.4f}",
            "IC95": f"[{r['ic95_ba'][0]:.3f}, {r['ic95_ba'][1]:.3f}]",
            "AUC": f"{m['auc_media']:.4f}" if "auc_media" in m else "-",
            "sens.": f"{m['sensibilidad']:.3f}", "espec.": f"{m['especificidad']:.3f}"})
    print(pd.DataFrame(filas).to_string(index=False))
    print(f"\n  Coherencia neurofisiológica: {n_ok} de {len(criterios)} criterios cumplidos")
    print(f"  Control de etiquetas barajadas centrado en 0.5: {'sí' if control_ok else 'NO'}")

    resultados["modelos"] = {n: _guardar(r) for n, r in modelos.items()}
    RUTA_RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    RUTA_RESULTADOS.write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False, default=_a_python),
        encoding="utf-8")
    print(f"\nResultados -> {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
