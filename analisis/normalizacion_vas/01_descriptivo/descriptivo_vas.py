"""
Normalización intra-sujeto del VAS — paso 1: caracterización descriptiva del
VAS bruto (propuesta del tutor).

Antes de normalizar el VAS dentro de cada persona hay que saber qué se va a
normalizar: cuántas sesiones tiene cada persona, cuánta varianza es de nivel
(entre personas) y cuánta de sesión (dentro de persona), y si los alumnos
difieren solo en su nivel medio o también en cuánto recorrido de la escala
usan. Si solo difieren en nivel, centrar basta; si difieren también en rango,
la tipificación (z intra-persona) cambia algo más que el centrado.

**Solo VAS: no se lee EEG ni se entrena ningún modelo.** La fuente es el CSV
completo del VAS (`VAS_100mm_preFE_filtrado.csv`), que cubre M1–M6 — M1 no
tiene EEG, pero sí VAS. Se normaliza con `normalizar_vas` (Fase 12), la misma
función que construye `persona_uid` y aplica la corrección F111/F115.

**Solo `train` + `val`.** El split de cada persona se toma de `labels.csv`
(que lo hereda de `splits.csv` con las correcciones de la Fase 12). Las
personas del test se apartan en el paso 1 y no vuelven a tocarse; las que solo
aparecen en el VAS (sin EEG, sin split) quedan fuera porque no se sabe a qué
lado de la partición pertenecerían.

Se agrupa siempre por `persona_uid`. El `CÓDIGO` del participante no sale en
ninguna tabla ni figura.

Ejecutar (desde la raíz del repositorio):
    python analisis/normalizacion_vas/01_descriptivo/descriptivo_vas.py > analisis/normalizacion_vas/01_descriptivo/ejecucion.log

La ubicación del dataset se toma de la variable de entorno `EEG_DATA`, con
`EEG-Data` junto al repositorio por defecto (ver `analisis/README.md`).
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from eeg_library.procesado.labels import normalizar_vas

SEMILLA = 42
N_BOOTSTRAP = 5000
N_PERMUTACIONES = 5000
MIN_SESIONES = 3  # mínimo para que la desviación por persona signifique algo

# Por defecto, la carpeta EEG-Data junto al repositorio (TFG/EEG-Data); se cambia con EEG_DATA.
EEG_DATA = Path(os.environ.get("EEG_DATA", Path(__file__).resolve().parents[4] / "EEG-Data"))
RUTA_VAS = EEG_DATA / "metadata" / "VAS_100mm_preFE_filtrado.csv"
RUTA_LABELS = EEG_DATA / "metadata" / "labels.csv"

CARPETA = Path(__file__).resolve().parent
CARPETA_TABLAS = CARPETA / "tablas"
CARPETA_FIGURAS = CARPETA / "figuras"

MOMENTOS = ["M1", "M2", "M3", "M4", "M5", "M6"]
# Calendario facilitado por el equipo del estudio (2026-10-21).
MOMENTO_TIPO = {"M1": "inicio", "M2": "final", "M3": "inicio",
                "M4": "final", "M5": "inicio", "M6": "final"}
MOMENTO_TRIMESTRE = {"M1": 1, "M2": 1, "M3": 2, "M4": 2, "M5": 3, "M6": 3}

# Paleta de referencia (validada para daltonismo) y tinta neutra.
COLOR_INICIO = "#2a78d6"
COLOR_FINAL = "#eb6834"
COLOR_BARRA = "#2a78d6"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
REJILLA = "#e4e3df"


def _titulo(texto):
    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


# --- Estadística ---------------------------------------------------------------

def icc1(valores, grupos):
    """
    ICC(1) de efectos aleatorios de una vía, para grupos de tamaño desigual.

    Proporción de la varianza total que es varianza entre grupos (personas).
    Con grupos desiguales el tamaño efectivo es n0 = (N − Σnᵢ²/N) / (k − 1).
    Devuelve también las componentes de varianza y el IC95 % analítico basado
    en la F (exacto para grupos iguales y normalidad; aproximado aquí).
    """
    df = pd.DataFrame({"y": np.asarray(valores, float), "g": np.asarray(grupos)})
    tam = df.groupby("g")["y"].size()
    medias = df.groupby("g")["y"].mean()
    n_total, k = len(df), len(tam)
    media_global = df["y"].mean()
    ss_entre = float((tam * (medias - media_global) ** 2).sum())
    ss_dentro = float(((df["y"] - df["g"].map(medias)) ** 2).sum())
    gl_entre, gl_dentro = k - 1, n_total - k
    ms_entre, ms_dentro = ss_entre / gl_entre, ss_dentro / gl_dentro
    n0 = (n_total - (tam ** 2).sum() / n_total) / gl_entre
    var_entre = max((ms_entre - ms_dentro) / n0, 0.0)
    icc = (ms_entre - ms_dentro) / (ms_entre + (n0 - 1) * ms_dentro)

    f = ms_entre / ms_dentro
    f_inf = f / stats.f.ppf(0.975, gl_entre, gl_dentro)
    f_sup = f * stats.f.ppf(0.975, gl_dentro, gl_entre)
    ic = ((f_inf - 1) / (f_inf + n0 - 1), (f_sup - 1) / (f_sup + n0 - 1))
    return {"icc": float(icc), "ic95_f": [float(ic[0]), float(ic[1])],
            "var_entre": float(var_entre), "var_dentro": float(ms_dentro),
            "n0": float(n0), "n_sesiones": int(n_total), "n_personas": int(k)}


def icc1_bootstrap(valores, grupos, n_rep=N_BOOTSTRAP, semilla=SEMILLA):
    """IC95 % percentil del ICC(1) remuestreando personas (no sesiones)."""
    rng = np.random.default_rng(semilla)
    df = pd.DataFrame({"y": np.asarray(valores, float), "g": np.asarray(grupos)})
    bloques = [b["y"].to_numpy() for _, b in df.groupby("g")]
    k = len(bloques)
    reps = np.empty(n_rep)
    for r in range(n_rep):
        elegidos = rng.integers(0, k, k)
        y = np.concatenate([bloques[i] for i in elegidos])
        # Cada réplica de una persona cuenta como un grupo distinto.
        g = np.concatenate([np.full(len(bloques[i]), j) for j, i in enumerate(elegidos)])
        reps[r] = icc1(y, g)["icc"]
    return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


def heterogeneidad_de_sd(sesiones, n_perm=N_PERMUTACIONES, semilla=SEMILLA):
    """
    ¿Difieren las personas en su desviación más de lo que daría el azar?

    Hipótesis nula: todas las personas tienen la misma dispersión intra-persona
    y lo único que cambia es su nivel. Se construye barajando entre personas
    los residuos respecto a la media propia (re-escalados por √(nᵢ/(nᵢ−1)) para
    que residuos de personas con distinto nº de sesiones sean comparables),
    conservando el nº de sesiones de cada una. El estadístico es la desviación
    típica de las SD por persona: si los alumnos difieren solo en nivel, la
    observada no debería salirse de la distribución nula.

    La nula ignora el techo y el suelo de la escala (residuos barajados pueden
    sacar a una persona de [0, 100]); eso la hace conservadora en el sentido de
    *añadir* dispersión a la nula, no de quitársela.
    """
    rng = np.random.default_rng(semilla)
    s = sesiones.sort_values("persona_uid", kind="stable")
    por = s.groupby("persona_uid")["vas"]
    n_i = por.transform("size")
    r_obs = ((s["vas"] - por.transform("mean")) * np.sqrt(n_i / (n_i - 1))).to_numpy()
    cortes = np.cumsum(por.size().to_numpy())[:-1]

    def sd_de_sd(r):
        return float(np.std([np.std(b, ddof=1) for b in np.split(r, cortes)], ddof=1))

    # Lo observado son las SD reales; los residuos re-escalados tienen la
    # varianza intra-persona como esperanza y solo se usan para la nula.
    observado = float(por.std(ddof=1).std(ddof=1))
    nula = np.array([sd_de_sd(rng.permutation(r_obs)) for _ in range(n_perm)])
    p = (1 + int((nula >= observado).sum())) / (n_perm + 1)
    return {"sd_de_sd_observada": observado,
            "sd_de_sd_nula_media": float(nula.mean()),
            "sd_de_sd_nula_p95": float(np.percentile(nula, 95)),
            "razon_observada_nula": observado / float(nula.mean()),
            "p_unilateral": float(p), "n_permutaciones": n_perm}


# --- Datos ---------------------------------------------------------------------

def cargar():
    vas = normalizar_vas(pd.read_csv(RUTA_VAS, sep=";", encoding="utf-8-sig"))
    labels = pd.read_csv(RUTA_LABELS)

    con_persona = labels.dropna(subset=["persona_uid"])
    splits_por_persona = con_persona.groupby("persona_uid")["split"].unique()
    if (splits_por_persona.map(len) > 1).any():
        raise ValueError("Hay personas en más de un split: revisar labels.csv.")
    split = splits_por_persona.map(lambda s: s[0])

    vas = vas.dropna(subset=["vas"]).copy()
    vas["split"] = vas["persona_uid"].map(split)
    if vas.duplicated(["persona_uid", "medicion"]).any():
        raise ValueError("Más de una respuesta por (persona_uid, medicion).")
    # Una misma persona real no puede tener dos persona_uid dentro del análisis.
    dentro = vas[vas["split"].isin(["train", "val"])]
    if (dentro.groupby("CÓDIGO")["persona_uid"].nunique() > 1).any():
        raise ValueError("Un mismo participante aparece con dos persona_uid en train+val.")

    resumen_exclusion = {
        "respuestas_vas": int(len(vas)),
        "personas_vas": int(vas["persona_uid"].nunique()),
        "personas_test_apartadas": int(vas.loc[vas["split"] == "test", "persona_uid"].nunique()),
        "respuestas_test_apartadas": int((vas["split"] == "test").sum()),
        "personas_sin_split": int(vas.loc[vas["split"].isna(), "persona_uid"].nunique()),
        "respuestas_sin_split": int(vas["split"].isna().sum()),
    }
    sesiones = dentro[["persona_uid", "centro", "medicion", "vas", "split"]].reset_index(drop=True)

    # Subconjunto de referencia: las sesiones etiquetadas con EEG (las de la Fase 12).
    con_eeg = labels[labels["etiquetada"] & labels["split"].isin(["train", "val"])
                     & (labels["n_ventanas"].fillna(0) > 0)]
    con_eeg = con_eeg[["persona_uid", "medicion", "vas"]].reset_index(drop=True)
    return sesiones, con_eeg, resumen_exclusion


# --- Figuras -------------------------------------------------------------------

def _estilo(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(TINTA_2)
    ax.tick_params(colors=TINTA_2, labelsize=9)
    ax.yaxis.grid(True, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)


def figura_histogramas(por_persona, ruta):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, titulo, bins in (
            (ax1, "media", "Media del VAS por persona", np.arange(0, 105, 5)),
            (ax2, "sd", "Desviación típica del VAS por persona", np.arange(0, 47.5, 2.5))):
        ax.hist(por_persona[col], bins=bins, color=COLOR_BARRA,
                edgecolor="white", linewidth=1.5)
        mediana = por_persona[col].median()
        ax.axvline(mediana, color=TINTA, linewidth=1.2, linestyle="--")
        ax.text(mediana, ax.get_ylim()[1] * 0.97, f"  mediana {mediana:.1f}",
                color=TINTA, fontsize=9, va="top")
        ax.set_title(titulo, fontsize=11, color=TINTA, loc="left")
        ax.set_xlabel("mm (escala 0–100)", fontsize=9, color=TINTA_2)
        ax.set_ylabel("personas", fontsize=9, color=TINTA_2)
        _estilo(ax)
    fig.suptitle(f"{len(por_persona)} personas de train+val con ≥{MIN_SESIONES} sesiones",
                 fontsize=9, color=TINTA_2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_media_sd(por_persona, ruta):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    # Máxima SD posible (poblacional) para una media m en [0, 100]: √(m(100−m)).
    m = np.linspace(0, 100, 201)
    ax.plot(m, np.sqrt(m * (100 - m)), color=TINTA_2, linewidth=1, linestyle=":")
    ax.text(50, 51, "SD máxima posible\npara esa media", ha="center",
            fontsize=8, color=TINTA_2, va="bottom")
    ax.scatter(por_persona["media"], por_persona["sd"], s=28, color=COLOR_BARRA,
               edgecolor="white", linewidth=0.8, alpha=0.9)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.set_xlabel("media del VAS de la persona (mm)", fontsize=9, color=TINTA_2)
    ax.set_ylabel("desviación típica de la persona (mm)", fontsize=9, color=TINTA_2)
    ax.set_title("Nivel frente a rango de uso de la escala", fontsize=11,
                 color=TINTA, loc="left")
    _estilo(ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_por_momento(tabla, ruta):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(MOMENTOS))
    for ax, col, ic_inf, ic_sup, titulo in (
            (ax1, "media", "ic95_inf", "ic95_sup", "VAS medio bruto por momento"),
            (ax2, "media_centrada", "ic95_centrada_inf", "ic95_centrada_sup",
             "VAS centrado en la media de cada persona")):
        for t in (1, 3):  # sombreado alterno de trimestres
            ax.axvspan(2 * (t - 1) - 0.5, 2 * (t - 1) + 1.5, color="#f0efec", zorder=0)
        ax.plot(x, tabla[col], color=TINTA_2, linewidth=1, zorder=1)
        for tipo, color, marcador in (("inicio", COLOR_INICIO, "o"),
                                      ("final", COLOR_FINAL, "s")):
            sel = (tabla["tipo"] == tipo).to_numpy()
            ax.errorbar(x[sel], tabla[col][sel],
                        yerr=[tabla[col][sel] - tabla[ic_inf][sel],
                              tabla[ic_sup][sel] - tabla[col][sel]],
                        fmt=marcador, color=color, markersize=8, capsize=3,
                        linewidth=1.5, markeredgecolor="white", label=f"{tipo} de trimestre",
                        zorder=3)
        ax.set_xticks(x, [f"{mo}\n{MOMENTO_TIPO[mo]} T{MOMENTO_TRIMESTRE[mo]}"
                          for mo in MOMENTOS], fontsize=8)
        ax.set_title(titulo, fontsize=11, color=TINTA, loc="left")
        ax.set_ylabel("mm (media ± IC95 %)", fontsize=9, color=TINTA_2)
        _estilo(ax)
    ax2.axhline(0, color=TINTA_2, linewidth=0.8)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# --- Principal -----------------------------------------------------------------

def main():
    CARPETA_TABLAS.mkdir(exist_ok=True)
    CARPETA_FIGURAS.mkdir(exist_ok=True)
    resultados = {}

    _titulo("1. DATOS")
    sesiones, con_eeg, exclusion = cargar()
    resultados["exclusion"] = exclusion
    print(f"Respuestas VAS válidas en el CSV: {exclusion['respuestas_vas']} "
          f"de {exclusion['personas_vas']} persona_uid")
    print(f"  Test apartado:   {exclusion['respuestas_test_apartadas']} respuestas, "
          f"{exclusion['personas_test_apartadas']} personas")
    print(f"  Sin split (sin EEG): {exclusion['respuestas_sin_split']} respuestas, "
          f"{exclusion['personas_sin_split']} personas — excluidas")
    print(f"Analizadas (train+val): {len(sesiones)} sesiones, "
          f"{sesiones['persona_uid'].nunique()} personas "
          f"(train {sesiones.loc[sesiones.split == 'train', 'persona_uid'].nunique()}, "
          f"val {sesiones.loc[sesiones.split == 'val', 'persona_uid'].nunique()})")

    _titulo("2. SESIONES POR PERSONA Y POR MOMENTO")
    n_por_persona = sesiones.groupby("persona_uid").size()
    dist = n_por_persona.value_counts().reindex(range(1, 7), fill_value=0)
    print("Sesiones con VAS por persona (M1–M6):")
    for n, c in dist.items():
        print(f"  {n} sesiones: {c:3d} personas")
    print(f"  media {n_por_persona.mean():.2f}, mediana {n_por_persona.median():.0f}")
    n_eeg = sesiones[sesiones.medicion != "M1"].groupby("persona_uid").size()
    dist_eeg = n_eeg.value_counts().reindex(range(1, 6), fill_value=0)
    print("Solo M2–M6 (el rango con EEG):")
    for n, c in dist_eeg.items():
        print(f"  {n} sesiones: {c:3d} personas")
    por_momento_n = sesiones.groupby("medicion").size().reindex(MOMENTOS)
    print("Sesiones por momento:", {k: int(v) for k, v in por_momento_n.items()})
    pd.DataFrame({"n_sesiones": dist.index, "personas_M1_M6": dist.values,
                  "personas_M2_M6": dist_eeg.reindex(dist.index, fill_value=0).values}
                 ).to_csv(CARPETA_TABLAS / "sesiones_por_persona.csv", index=False)
    resultados["sesiones_por_persona"] = {"M1_M6": {int(k): int(v) for k, v in dist.items()},
                                          "M2_M6": {int(k): int(v) for k, v in dist_eeg.items()}}

    _titulo(f"3. PERSONAS CON ≥{MIN_SESIONES} SESIONES Y CON DESVIACIÓN NULA")
    sd = sesiones.groupby("persona_uid")["vas"].std(ddof=1)
    ge3 = n_por_persona[n_por_persona >= MIN_SESIONES].index
    ge2 = n_por_persona[n_por_persona >= 2].index
    sd0_ge2 = int((sd.loc[ge2] == 0).sum())
    sd0_ge3 = int((sd.loc[ge3] == 0).sum())
    print(f"Personas con ≥{MIN_SESIONES} sesiones: {len(ge3)} de {len(n_por_persona)}")
    print(f"SD = 0 (todas sus respuestas iguales): {sd0_ge3} de las {len(ge3)} con ≥3; "
          f"{sd0_ge2} de las {len(ge2)} con ≥2")
    if sd0_ge3:
        constantes = sesiones[sesiones.persona_uid.isin(sd.loc[ge3][sd.loc[ge3] == 0].index)]
        print("  Valor constante y nº de sesiones de esas personas:",
              constantes.groupby("persona_uid")["vas"].agg(["first", "size"])
              .apply(tuple, axis=1).value_counts().to_dict())
    rango = sesiones.groupby("persona_uid")["vas"].agg(lambda s: s.max() - s.min())
    sd_pequena = int((sd.loc[ge3] < 5).sum())
    print(f"SD < 5 mm (casi constante): {sd_pequena} de {len(ge3)}")
    resultados["personas"] = {"total": int(len(n_por_persona)), "con_ge3": int(len(ge3)),
                              "con_ge2": int(len(ge2)), "sd0_entre_ge3": sd0_ge3,
                              "sd0_entre_ge2": sd0_ge2, "sd_menor_5_entre_ge3": sd_pequena}

    _titulo("4. ICC(1) DEL VAS BRUTO")
    filas_icc = []
    for nombre, datos in (("VAS M1–M6 (todas las respuestas)", sesiones),
                          ("VAS M2–M6", sesiones[sesiones.medicion != "M1"]),
                          ("Sesiones con EEG etiquetadas (Fase 12)", con_eeg)):
        r = icc1(datos["vas"], datos["persona_uid"])
        r["ic95_bootstrap"] = icc1_bootstrap(datos["vas"], datos["persona_uid"])
        filas_icc.append({"conjunto": nombre, **{k: r[k] for k in
                          ("n_sesiones", "n_personas", "icc", "var_entre", "var_dentro", "n0")},
                          "ic95_f_inf": r["ic95_f"][0], "ic95_f_sup": r["ic95_f"][1],
                          "ic95_boot_inf": r["ic95_bootstrap"][0],
                          "ic95_boot_sup": r["ic95_bootstrap"][1]})
        print(f"{nombre}: {r['n_sesiones']} sesiones, {r['n_personas']} personas")
        print(f"  ICC(1) = {r['icc']:.3f}  IC95 F [{r['ic95_f'][0]:.3f}, {r['ic95_f'][1]:.3f}]  "
              f"IC95 bootstrap por persona [{r['ic95_bootstrap'][0]:.3f}, "
              f"{r['ic95_bootstrap'][1]:.3f}]")
        print(f"  σ² entre = {r['var_entre']:.1f}  σ² dentro = {r['var_dentro']:.1f}  "
              f"n0 = {r['n0']:.2f}")
    tabla_icc = pd.DataFrame(filas_icc)
    tabla_icc.to_csv(CARPETA_TABLAS / "icc.csv", index=False, float_format="%.4f")
    resultados["icc"] = filas_icc

    _titulo(f"5. MEDIA Y DESVIACIÓN POR PERSONA (≥{MIN_SESIONES} sesiones)")
    por_persona = (sesiones[sesiones.persona_uid.isin(ge3)].groupby("persona_uid")["vas"]
                   .agg(n="size", media="mean", sd=lambda s: s.std(ddof=1),
                        rango=lambda s: s.max() - s.min()))
    descr = por_persona[["media", "sd", "rango"]].describe(percentiles=[.1, .25, .5, .75, .9]).T
    print(descr.round(1).to_string())
    p10, p90 = por_persona["sd"].quantile([.1, .9])
    print(f"SD por persona: P90/P10 = {p90 / p10:.1f}; CV de las SD = "
          f"{por_persona['sd'].std() / por_persona['sd'].mean():.2f}")
    rho_ms, p_ms = stats.spearmanr(por_persona["media"], por_persona["sd"])
    distancia_borde = np.minimum(por_persona["media"], 100 - por_persona["media"])
    rho_b, p_b = stats.spearmanr(distancia_borde, por_persona["sd"])
    print(f"Spearman media–SD: ρ = {rho_ms:+.3f} (p = {p_ms:.3g})")
    print(f"Spearman distancia al borde de la escala–SD: ρ = {rho_b:+.3f} (p = {p_b:.3g})")
    het = heterogeneidad_de_sd(sesiones[sesiones.persona_uid.isin(ge3)])
    print("¿Difieren las SD más que si todos tuvieran la misma dispersión?")
    print(f"  SD de las SD observada = {het['sd_de_sd_observada']:.2f}; nula: media "
          f"{het['sd_de_sd_nula_media']:.2f}, P95 {het['sd_de_sd_nula_p95']:.2f}; "
          f"razón {het['razon_observada_nula']:.2f}; p = {het['p_unilateral']:.4f}")
    descr.to_csv(CARPETA_TABLAS / "media_sd_por_persona_resumen.csv", float_format="%.2f")
    resultados["por_persona"] = {
        "n_personas": int(len(por_persona)),
        "resumen": descr.round(3).to_dict(orient="index"),
        "sd_p90_p10": float(p90 / p10),
        "spearman_media_sd": [float(rho_ms), float(p_ms)],
        "spearman_borde_sd": [float(rho_b), float(p_b)],
        "heterogeneidad_sd": het}

    _titulo("6. VAS MEDIO POR MOMENTO")
    sesiones["centrado"] = sesiones["vas"] - sesiones.groupby("persona_uid")["vas"].transform("mean")
    # El centrado solo tiene sentido con al menos dos sesiones de la persona.
    centrables = sesiones[sesiones.persona_uid.isin(ge2)]
    filas = []
    for mo in MOMENTOS:
        y = sesiones.loc[sesiones.medicion == mo, "vas"]
        c = centrables.loc[centrables.medicion == mo, "centrado"]
        t = stats.t.ppf(0.975, len(y) - 1)
        tc = stats.t.ppf(0.975, len(c) - 1)
        filas.append({"medicion": mo, "tipo": MOMENTO_TIPO[mo],
                      "trimestre": MOMENTO_TRIMESTRE[mo], "n": len(y),
                      "media": y.mean(), "sd": y.std(ddof=1), "mediana": y.median(),
                      "ic95_inf": y.mean() - t * y.sem(), "ic95_sup": y.mean() + t * y.sem(),
                      "n_centrada": len(c), "media_centrada": c.mean(),
                      "ic95_centrada_inf": c.mean() - tc * c.sem(),
                      "ic95_centrada_sup": c.mean() + tc * c.sem()})
    tabla_m = pd.DataFrame(filas)
    print(tabla_m.round(2).to_string(index=False))
    # Inicio frente a final: diferencia de medias centradas (dentro de persona).
    centrables = centrables.assign(tipo=centrables.medicion.map(MOMENTO_TIPO))
    dif = centrables.groupby(["persona_uid", "tipo"])["centrado"].mean().unstack().dropna()
    d = dif["final"] - dif["inicio"]
    t_res = stats.ttest_1samp(d, 0)
    w_res = stats.wilcoxon(d)
    print(f"Final − inicio de trimestre, dentro de persona ({len(d)} personas con ambos): "
          f"{d.mean():+.2f} mm (IC95 ±{stats.t.ppf(0.975, len(d) - 1) * d.sem():.2f}); "
          f"t p = {t_res.pvalue:.3g}; Wilcoxon p = {w_res.pvalue:.3g}")
    # Dos lecturas alternativas del mismo patrón: que lo arrastre M1 (el valor
    # más bajo, y sin EEG) o que sea una deriva lineal a lo largo del curso y no
    # un efecto de inicio/final. Se separan con un modelo intra-persona:
    # VAS centrado ~ final + nº de visita centrado en la persona.
    def _final_vs_inicio(datos):
        datos = datos.copy()
        datos["centrado"] = datos["vas"] - datos.groupby("persona_uid")["vas"].transform("mean")
        datos["final"] = (datos.medicion.map(MOMENTO_TIPO) == "final").astype(float)
        datos["visita"] = datos.medicion.str[1].astype(float)
        for col in ("final", "visita"):
            datos[col] -= datos.groupby("persona_uid")[col].transform("mean")
        X = datos[["final", "visita"]].to_numpy()
        y = datos["centrado"].to_numpy()
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        # IC por bootstrap de personas: las sesiones de una persona no son independientes.
        rng = np.random.default_rng(SEMILLA)
        bloques = [b.index.to_numpy() for _, b in datos.groupby("persona_uid")]
        pos = {i: p for p, i in enumerate(datos.index)}
        reps = []
        for _ in range(N_BOOTSTRAP):
            idx = np.concatenate([[pos[i] for i in bloques[j]]
                                  for j in rng.integers(0, len(bloques), len(bloques))])
            reps.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0])
        reps = np.array(reps)
        return {"final": float(coef[0]), "visita": float(coef[1]),
                "ic95_final": np.percentile(reps[:, 0], [2.5, 97.5]).tolist(),
                "ic95_visita": np.percentile(reps[:, 1], [2.5, 97.5]).tolist()}

    controles = {}
    for nombre, datos in (("M1–M6", centrables), ("M2–M6", centrables[centrables.medicion != "M1"])):
        datos = datos[datos.groupby("persona_uid")["vas"].transform("size") >= 2]
        controles[nombre] = r = _final_vs_inicio(datos)
        print(f"Intra-persona {nombre}, VAS ~ final + visita: final {r['final']:+.2f} mm "
              f"IC95 [{r['ic95_final'][0]:+.2f}, {r['ic95_final'][1]:+.2f}]; visita "
              f"{r['visita']:+.2f} mm/visita IC95 [{r['ic95_visita'][0]:+.2f}, "
              f"{r['ic95_visita'][1]:+.2f}]")
    resultados["final_vs_inicio_con_deriva"] = controles

    # Varianza explicada por el momento (R² de la media por momento).
    r2_momento = 1 - ((sesiones["vas"] - sesiones.groupby("medicion")["vas"].transform("mean")) ** 2
                      ).sum() / ((sesiones["vas"] - sesiones["vas"].mean()) ** 2).sum()
    print(f"R² de la media por momento sobre el VAS bruto: {r2_momento:.3f}")
    tabla_m.to_csv(CARPETA_TABLAS / "vas_por_momento.csv", index=False, float_format="%.3f")
    resultados["por_momento"] = tabla_m.round(4).to_dict(orient="records")
    resultados["final_menos_inicio"] = {"n_personas": int(len(d)), "media": float(d.mean()),
                                        "p_t": float(t_res.pvalue), "p_wilcoxon": float(w_res.pvalue)}
    resultados["r2_momento"] = float(r2_momento)

    _titulo("7. FIGURAS")
    figura_histogramas(por_persona, CARPETA_FIGURAS / "hist_media_sd_por_persona.png")
    figura_media_sd(por_persona, CARPETA_FIGURAS / "media_vs_sd_por_persona.png")
    figura_por_momento(tabla_m, CARPETA_FIGURAS / "vas_por_momento.png")
    print("Escritas en", CARPETA_FIGURAS)

    with open(CARPETA / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2, default=float)
    print("Resultados en", CARPETA / "resultados.json")


if __name__ == "__main__":
    main()
