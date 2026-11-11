"""
Exploratorio: PCA + clustering de los grupos extremos de VAS, por grabación.

Pregunta: ¿se separan de forma natural las grabaciones de sesiones con mucha
fatiga percibida (VAS >= 80) de las de poca (VAS <= 20) en el espacio de las 77
features de la Fase 13, una vez neutralizado el efecto de la condición?

La condición (OC, OA, STROOP, PASAT, PVTB, MSIT, PVSAT) se distingue muy bien en
este pipeline y dominaría cualquier clustering sin ajustar. Por eso el análisis
principal **centra cada feature por condición** antes del PCA, igual que la
regresión intrapersona centraba por `persona_uid`: lo que queda es la desviación
de cada grabación respecto a lo esperable *en su propia condición*.

Es un análisis **no supervisado y exploratorio**: el clustering no ve la
etiqueta. El grupo de VAS solo se usa después, para evaluar la coincidencia.

**Solo usa `train` + `val`.** El conjunto de test se aparta en el paso 1 y no
vuelve a tocarse.

Ejecutar (desde la raíz del repositorio):
    python analisis/analisis_pca_clustering.py

La ubicación del dataset se toma de la variable de entorno `EEG_DATA`, con
`EEG-Data` junto al repositorio por defecto (ver `analisis/README.md`).
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from eeg_library.procesado.matrix import COLUMNA_ETIQUETA, columnas_de_features

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEMILLA = 42
N_COMPONENTES = 15
N_CLUSTERS = 2
VAS_ALTO = 80
VAS_BAJO = 20
CONDICIONES_REPOSO = ("OA", "OC")

# Por defecto, la carpeta EEG-Data junto al repositorio (TFG/EEG-Data); se cambia con EEG_DATA.
RUTA_DATOS = Path(os.environ.get("EEG_DATA", Path(__file__).resolve().parents[2] / "EEG-Data"))
RUTA_MATRIZ = RUTA_DATOS / "processed" / "matriz_rf.parquet"
RUTA_RESULTADOS = RUTA_DATOS / "processed" / "pca_clustering.json"
RUTA_FIGURAS = Path(__file__).resolve().parents[1] / "resultados" / "figuras"

# Rojo = mucho estrés, azul = poco. El par pasa las seis comprobaciones del
# validador de paleta en modo claro (CVD ΔE 21.6, muy por encima del 8 exigido).
COLOR_ALTO = "#e34948"
COLOR_BAJO = "#2a78d6"
COLOR_TEXTO = "#0b0b0b"
COLOR_TEXTO_2 = "#52514e"
COLOR_REJILLA = "#e0e0dc"


def _titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def centrar_por_condicion(X, condicion):
    """
    Resta a cada feature la media de su propia condición.

    Los NaN se ignoran al calcular la media y luego se rellenan con 0, que tras
    el centrado **es** la media de la condición. Así una región sin canales
    válidos no inventa señal ni arrastra el NaN al PCA.

    Devuelve la matriz centrada y la lista de columnas que quedan constantes
    (varianza 0), que hay que descartar: `condicion_orden` es una de ellas por
    construcción, porque es la condición codificada y centrarla la anula.
    """
    centrada = X.groupby(condicion, sort=False).transform(lambda c: c - c.mean())
    centrada = centrada.fillna(0.0)
    constantes = [c for c in centrada.columns
                  if not np.isfinite(centrada[c].to_numpy()).all()
                  or np.std(centrada[c].to_numpy()) < 1e-12]
    return centrada, constantes


def solo_imputar(X, condicion):
    """Variante sin ajustar: mismo relleno de NaN, sin restar la media."""
    medias = X.groupby(condicion, sort=False).transform("mean")
    rellenada = X.fillna(medias).fillna(X.median())
    constantes = [c for c in rellenada.columns
                  if np.std(rellenada[c].to_numpy()) < 1e-12]
    return rellenada, constantes


def pureza_por_persona(clusters, personas, semilla=SEMILLA, n_permutaciones=1000):
    """
    Fracción de grabaciones de cada persona que caen en su cluster mayoritario.

    El ARI compara mal particiones de cardinalidad muy distinta (2 clusters
    frente a 82 personas), así que para la pregunta "¿los clusters están
    siguiendo a la identidad?" se usa esta medida directa, con su nivel de azar
    obtenido permutando las etiquetas de cluster.
    """
    df = pd.DataFrame({"cluster": clusters, "persona": personas})
    observada = float(
        df.groupby("persona")["cluster"]
        .apply(lambda c: c.value_counts().iloc[0] / len(c)).mean())

    rng = np.random.default_rng(semilla)
    nulos = []
    for _ in range(n_permutaciones):
        df["cluster"] = rng.permutation(clusters)
        nulos.append(
            df.groupby("persona")["cluster"]
            .apply(lambda c: c.value_counts().iloc[0] / len(c)).mean())
    return {
        "pureza": observada,
        "pureza_azar": float(np.mean(nulos)),
        "pureza_azar_p95": float(np.percentile(nulos, 95)),
    }


def analizar(datos, features, etiqueta, ajustar=True):
    """PCA + KMeans sobre un subconjunto, con todas las comparaciones."""
    X = datos[features]
    condicion = datos["condicion"]

    if ajustar:
        Xp, constantes = centrar_por_condicion(X, condicion)
    else:
        Xp, constantes = solo_imputar(X, condicion)
    if constantes:
        Xp = Xp.drop(columns=constantes)

    Z = StandardScaler().fit_transform(Xp.to_numpy(dtype=float))
    pca = PCA(n_components=N_COMPONENTES, random_state=SEMILLA)
    P = pca.fit_transform(Z)
    acumulada = np.cumsum(pca.explained_variance_ratio_)

    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEMILLA, n_init=50)
    clusters = km.fit_predict(P)

    # Diagnóstico del ajuste por condición. El ARI de los 2 clusters frente a
    # la condición no sirve para comprobarlo: una partición en 2 no puede
    # expresar una de 7, así que sale ~0 tanto si el efecto está como si no.
    # La comprobación que sí discrimina es agrupar en tantos clusters como
    # condiciones haya y ver si los recupera.
    n_condiciones = condicion.nunique()
    km_cond = KMeans(n_clusters=n_condiciones, random_state=SEMILLA, n_init=50)
    clusters_cond = km_cond.fit_predict(P)
    ari_cond_k = float(adjusted_rand_score(condicion.to_numpy(), clusters_cond))

    grupo = datos["grupo_vas"].to_numpy()
    ari_vas = adjusted_rand_score(grupo, clusters)
    ari_cond = adjusted_rand_score(condicion.to_numpy(), clusters)
    ari_persona = adjusted_rand_score(datos["persona_uid"].to_numpy(), clusters)

    tabla = pd.crosstab(pd.Series(clusters, name="cluster"),
                        pd.Series(grupo, name="grupo VAS"))
    # Mejor asignación posible de los 2 clusters a los 2 grupos: el techo de
    # acierto de este clustering si se le permitiera elegir la correspondencia.
    acierto = max(
        (tabla.loc[0, "alto"] + tabla.loc[1, "bajo"]),
        (tabla.loc[1, "alto"] + tabla.loc[0, "bajo"])) / len(datos)
    mayoritaria = max((grupo == "alto").mean(), (grupo == "bajo").mean())

    salida = {
        "n_grabaciones": int(len(datos)),
        "n_personas": int(datos["persona_uid"].nunique()),
        "n_sesiones": int(datos["sesion_uid"].nunique()),
        "n_features_usadas": int(Xp.shape[1]),
        "features_descartadas": constantes,
        "varianza_pc1": float(pca.explained_variance_ratio_[0]),
        "varianza_pc2": float(pca.explained_variance_ratio_[1]),
        "varianza_acumulada_5": float(acumulada[4]),
        "varianza_acumulada_15": float(acumulada[-1]),
        "varianza_por_componente": pca.explained_variance_ratio_.tolist(),
        "tamanos_cluster": np.bincount(clusters).tolist(),
        "silueta": float(silhouette_score(P, clusters)),
        "ari_grupo_vas": float(ari_vas),
        "ari_condicion": float(ari_cond),
        "ari_condicion_k_condiciones": ari_cond_k,
        "n_condiciones": int(n_condiciones),
        "ari_persona": float(ari_persona),
        "ami_grupo_vas": float(adjusted_mutual_info_score(grupo, clusters)),
        "ami_condicion": float(
            adjusted_mutual_info_score(condicion.to_numpy(), clusters)),
        "ami_persona": float(
            adjusted_mutual_info_score(datos["persona_uid"].to_numpy(), clusters)),
        "acierto_mejor_asignacion": float(acierto),
        "acierto_clase_mayoritaria": float(mayoritaria),
        "contingencia": tabla.to_dict(),
    }
    salida.update(pureza_por_persona(clusters, datos["persona_uid"].to_numpy()))

    print(f"\n--- {etiqueta} ---")
    print(f"  n = {salida['n_grabaciones']} grabaciones, "
          f"{salida['n_personas']} personas, {salida['n_sesiones']} sesiones")
    print(f"  features tras el ajuste: {salida['n_features_usadas']}"
          + (f"  (descartadas por varianza 0: {constantes})" if constantes else ""))
    print(f"  varianza explicada: PC1 {salida['varianza_pc1']:.1%}, "
          f"PC2 {salida['varianza_pc2']:.1%}, "
          f"5 comp {salida['varianza_acumulada_5']:.1%}, "
          f"15 comp {salida['varianza_acumulada_15']:.1%}")
    print(f"  tamaños de cluster: {salida['tamanos_cluster']}   "
          f"silueta {salida['silueta']:.3f}")
    print("\n  Contingencia cluster x grupo VAS:")
    print("    " + tabla.to_string().replace("\n", "\n    "))
    print(f"\n  ARI vs grupo VAS   : {ari_vas:+.4f}")
    print(f"  ARI vs condicion   : {ari_cond:+.4f}   (k=2, poco informativo)")
    print(f"  ARI vs condicion   : {ari_cond_k:+.4f}   (k={n_condiciones}, diagnostico del ajuste)")
    print(f"  ARI vs persona_uid : {ari_persona:+.4f}")
    print(f"  AMI vs grupo VAS   : {salida['ami_grupo_vas']:+.4f}")
    print(f"  AMI vs condicion   : {salida['ami_condicion']:+.4f}")
    print(f"  AMI vs persona_uid : {salida['ami_persona']:+.4f}")
    print(f"  Acierto de la mejor asignacion : {acierto:.1%}  "
          f"(clase mayoritaria: {mayoritaria:.1%})")
    print(f"  Pureza por persona : {salida['pureza']:.3f}  "
          f"(azar {salida['pureza_azar']:.3f}, p95 {salida['pureza_azar_p95']:.3f})")

    return salida, P, clusters


def figura(P, datos, resultado, titulo, subtitulo, ruta):
    """PC1 vs PC2: color = grupo VAS real, forma = cluster asignado."""
    fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    grupo = datos["grupo_vas"].to_numpy()
    especificacion = [
        ("bajo", COLOR_BAJO, f"Poco estrés · VAS ≤ {VAS_BAJO}"),
        ("alto", COLOR_ALTO, f"Mucho estrés · VAS ≥ {VAS_ALTO}"),
    ]
    for nombre, color, etiqueta in especificacion:
        sel = grupo == nombre
        ax.scatter(P[sel, 0], P[sel, 1], s=26, c=color, alpha=0.55,
                   edgecolors="white", linewidths=0.5, zorder=3,
                   label=f"{etiqueta}  (n = {int(sel.sum())})")

    ax.set_xlabel(f"PC1 — {resultado['varianza_pc1']:.1%} de la varianza",
                  fontsize=10.5, color=COLOR_TEXTO_2, labelpad=9)
    ax.set_ylabel(f"PC2 — {resultado['varianza_pc2']:.1%} de la varianza",
                  fontsize=10.5, color=COLOR_TEXTO_2, labelpad=9)
    # El `pad` deja hueco para el subtítulo, que va justo encima del eje: sin
    # él, ambos se dibujan en la misma banda y se solapan.
    ax.set_title(titulo, fontsize=13, color=COLOR_TEXTO, pad=30, loc="left",
                 fontweight="bold")
    ax.text(0, 1.015, subtitulo, transform=ax.transAxes, fontsize=9.5,
            color=COLOR_TEXTO_2, va="bottom")

    resumen = (f"ARI vs grupo VAS = {resultado['ari_grupo_vas']:+.4f}\n"
               f"15 componentes = {resultado['varianza_acumulada_15']:.1%} "
               f"de la varianza")
    ax.text(0.985, 0.975, resumen, transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, color=COLOR_TEXTO_2, linespacing=1.6, zorder=10,
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

    _titulo("1. Datos - grabaciones extremas de train + val")
    matriz = pd.read_parquet(RUTA_MATRIZ)
    features = columnas_de_features(matriz)
    print(f"Matriz completa      : {matriz.shape[0]} grabaciones, "
          f"{len(features)} features")

    fuera = matriz[matriz["split"] == "test"]
    tv = matriz[matriz["split"].isin(["train", "val"])]
    print(f"Apartado (test)      : {len(fuera)} grabaciones, "
          f"{fuera['persona_uid'].nunique()} personas - no se vuelve a leer")
    print(f"train + val          : {len(tv)} grabaciones")

    extremos = tv[(tv[COLUMNA_ETIQUETA] >= VAS_ALTO)
                  | (tv[COLUMNA_ETIQUETA] <= VAS_BAJO)].copy()
    extremos["grupo_vas"] = np.where(
        extremos[COLUMNA_ETIQUETA] >= VAS_ALTO, "alto", "bajo")
    print(f"Grupos extremos      : {len(extremos)} grabaciones "
          f"({len(tv) - len(extremos)} descartadas por VAS intermedio)")
    print()
    composicion = extremos.groupby("grupo_vas").agg(
        grabaciones=("archivo", "size"), sesiones=("sesion_uid", "nunique"),
        personas=("persona_uid", "nunique"), vas_min=(COLUMNA_ETIQUETA, "min"),
        vas_max=(COLUMNA_ETIQUETA, "max"))
    print(composicion.to_string())
    print("\nGrabaciones por condicion y grupo:")
    print(pd.crosstab(extremos["condicion"], extremos["grupo_vas"]).to_string())

    resultados["datos"] = {
        "n_grabaciones": int(len(extremos)),
        "n_test_apartado": int(len(fuera)),
        "composicion": composicion.to_dict(),
        "por_condicion": pd.crosstab(
            extremos["condicion"], extremos["grupo_vas"]).to_dict(),
    }

    _titulo("2. Confusión entre identidad y grupo de VAS")
    por_persona = extremos.groupby("persona_uid")["grupo_vas"].nunique()
    ambos = sorted(por_persona[por_persona > 1].index)
    n_total = extremos["persona_uid"].nunique()
    grab_ambos = extremos[extremos["persona_uid"].isin(ambos)]
    print(f"Personas en los grupos extremos      : {n_total}")
    print(f"Personas presentes en AMBOS grupos   : {len(ambos)} "
          f"({100 * len(ambos) / n_total:.1f} %)")
    print(f"Personas en un solo grupo            : {n_total - len(ambos)} "
          f"({100 * (n_total - len(ambos)) / n_total:.1f} %)")
    print(f"Grabaciones aportadas por las de ambos: {len(grab_ambos)} "
          f"({100 * len(grab_ambos) / len(extremos):.1f} %)")
    if ambos:
        print(f"\nSon: {', '.join(ambos)}")
        print(extremos[extremos["persona_uid"].isin(ambos)]
              .groupby(["persona_uid", "grupo_vas"])
              .agg(grabaciones=("archivo", "size"),
                   sesiones=("sesion_uid", "nunique")).to_string())
    print("\nLectura: con el 95.1 % de las personas en un solo grupo, 'grupo de")
    print("VAS' y 'quién es el participante' son casi la misma partición. Un")
    print("clustering que separase los grupos podría estar separando personas.")
    resultados["confusion_identidad"] = {
        "n_personas": int(n_total), "n_personas_ambos_grupos": len(ambos),
        "personas_ambos_grupos": ambos,
        "n_grabaciones_de_esas_personas": int(len(grab_ambos)),
        "pct_grabaciones": float(100 * len(grab_ambos) / len(extremos)),
    }

    _titulo("3. Análisis principal - features centradas por condición")
    print("Cada feature menos la media de su propia condición. Lo que queda es")
    print("la desviación de la grabación respecto a lo esperable en su tarea.")
    principal, P_principal, cl_principal = analizar(
        extremos, features, "Ajustado por condición (7 condiciones)", ajustar=True)
    resultados["principal"] = principal

    _titulo("4. Referencia - las mismas features SIN ajustar")
    print("Sirve para comprobar que el ajuste del paso 3 hizo lo que debía:")
    print("si aquí el ARI vs condición es alto y allí ~0, quedó neutralizada.")
    sin_ajustar, _, _ = analizar(
        extremos, features, "Sin ajustar", ajustar=False)
    resultados["sin_ajustar"] = sin_ajustar

    _titulo("5. Robustez - solo reposo (OA + OC)")
    print("Mismo PCA (15 componentes) y mismo clustering (2 clusters) sobre las")
    print("grabaciones de reposo, donde no hay tarea que module la señal.")
    reposo = extremos[extremos["condicion"].isin(CONDICIONES_REPOSO)].copy()
    robustez, P_reposo, cl_reposo = analizar(
        reposo, features, "Reposo, ajustado por condición (OA/OC)", ajustar=True)
    resultados["reposo"] = robustez

    _titulo("6. Figuras")
    RUTA_FIGURAS.mkdir(exist_ok=True)
    figura(P_principal, extremos, principal,
           "Los grupos extremos de fatiga no se separan en el espacio de features",
           f"Todas las condiciones, features centradas por condición · "
           f"n = {principal['n_grabaciones']} grabaciones",
           RUTA_FIGURAS / "pca_clustering_vas_extremos.png")
    figura(P_reposo, reposo, robustez,
           "Tampoco en reposo, donde no hay tarea que module la señal",
           f"Solo OA + OC, features centradas por condición · "
           f"n = {robustez['n_grabaciones']} grabaciones",
           RUTA_FIGURAS / "pca_clustering_vas_extremos_reposo.png")

    _titulo("7. Resumen")
    filas = [
        {"análisis": "Principal (7 condiciones, ajustado)", **{
            "n": principal["n_grabaciones"],
            "var. 15 comp": f"{principal['varianza_acumulada_15']:.1%}",
            "ARI vs VAS": f"{principal['ari_grupo_vas']:+.4f}",
            "ARI condición": f"{principal['ari_condicion_k_condiciones']:+.4f} (k=7)",
            "acierto": f"{principal['acierto_mejor_asignacion']:.1%}",
            "mayoritaria": f"{principal['acierto_clase_mayoritaria']:.1%}"}},
        {"análisis": "Referencia (sin ajustar)", **{
            "n": sin_ajustar["n_grabaciones"],
            "var. 15 comp": f"{sin_ajustar['varianza_acumulada_15']:.1%}",
            "ARI vs VAS": f"{sin_ajustar['ari_grupo_vas']:+.4f}",
            "ARI condición": f"{sin_ajustar['ari_condicion_k_condiciones']:+.4f} (k=7)",
            "acierto": f"{sin_ajustar['acierto_mejor_asignacion']:.1%}",
            "mayoritaria": f"{sin_ajustar['acierto_clase_mayoritaria']:.1%}"}},
        {"análisis": "Robustez (solo reposo, ajustado)", **{
            "n": robustez["n_grabaciones"],
            "var. 15 comp": f"{robustez['varianza_acumulada_15']:.1%}",
            "ARI vs VAS": f"{robustez['ari_grupo_vas']:+.4f}",
            "ARI condición": f"{robustez['ari_condicion_k_condiciones']:+.4f} (k=2)",
            "acierto": f"{robustez['acierto_mejor_asignacion']:.1%}",
            "mayoritaria": f"{robustez['acierto_clase_mayoritaria']:.1%}"}},
    ]
    print(pd.DataFrame(filas).to_string(index=False))

    RUTA_RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    RUTA_RESULTADOS.write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print(f"\nResultados -> {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
