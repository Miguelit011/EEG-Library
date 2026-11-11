"""
Normalización intra-sujeto del VAS — paso 2: ¿cambian las etiquetas
normalizadas respecto a la que ya se probó con el EEG?

La Tarea B de las Fases 14/14b ya entrena con una etiqueta normalizada por
persona: el VAS centrado en la media leave-one-out de sus otras sesiones con
EEG (variante (a)). Esta línea solo puede cambiar la respuesta sobre el EEG si
alguna normalización produce una etiqueta distinta. Aquí se construyen las
variantes y se mide cuánto difieren de (a), **sin EEG ni modelos**.

El plan y el criterio de decisión están en `INFORME.md`, escritos antes de
ejecutar este script:

1. Elegir con qué respuestas se estima la referencia de cada persona (R1–R4)
   por el error al predecir la sesión que se deja fuera.
2. Construir (a′) centrado con esa referencia, (b) z con la SD encogida
   (empírico-bayesiano de Smyth) y (c) (a′) sin el efecto de cada momento.
3. Si las tres coinciden con (a) en ≥90 % de los tercios, la línea se cierra.

**Solo `train` + `val`.** El split de cada persona sale de `labels.csv`; el
test se aparta al cargar. Se agrupa siempre por `persona_uid` y el `CÓDIGO`
del participante no sale en ninguna salida.

Ejecutar (desde la raíz del repositorio):
    python analisis/normalizacion_vas/02_etiquetas/etiquetas_normalizadas.py > analisis/normalizacion_vas/02_etiquetas/ejecucion.log

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
from scipy import special, stats
from sklearn.metrics import cohen_kappa_score

from eeg_library.modelado.modeling import (
    a_clases,
    centrar_por_persona_loo,
    cortes_terciles,
    residualizar_por_visita,
)
from eeg_library.procesado.labels import normalizar_vas

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEMILLA = 42
N_BOOTSTRAP = 5000
UMBRAL_ACUERDO = 0.90  # fijado en INFORME.md antes de ejecutar

# Por defecto, la carpeta EEG-Data junto al repositorio (TFG/EEG-Data); se cambia con EEG_DATA.
EEG_DATA = Path(os.environ.get("EEG_DATA", Path(__file__).resolve().parents[4] / "EEG-Data"))
RUTA_VAS = EEG_DATA / "metadata" / "VAS_100mm_preFE_filtrado.csv"
RUTA_LABELS = EEG_DATA / "metadata" / "labels.csv"

CARPETA = Path(__file__).resolve().parent
CARPETA_TABLAS = CARPETA / "tablas"
CARPETA_FIGURAS = CARPETA / "figuras"

MOMENTOS_EEG = ["M2", "M3", "M4", "M5", "M6"]
REFERENCIAS = {
    "R1": "otras sesiones con EEG (Fase 14)",
    "R2": "todas las otras respuestas M2–M6",
    "R3": "R2 + M1 tal cual",
    "R4": "R2 + M1 corregido por su desplazamiento",
}

COLOR_A = "#2a78d6"
COLOR_B = "#eb6834"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
REJILLA = "#e4e3df"


def _titulo(texto):
    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


# --- Datos ---------------------------------------------------------------------

def cargar():
    """
    Devuelve las respuestas del VAS de train+val (M1–M6) y las sesiones
    objetivo (las etiquetadas con EEG), con la coherencia entre ambas fuentes
    comprobada.
    """
    vas = normalizar_vas(pd.read_csv(RUTA_VAS, sep=";", encoding="utf-8-sig"))
    labels = pd.read_csv(RUTA_LABELS)

    splits = labels.dropna(subset=["persona_uid"]).groupby("persona_uid")["split"].unique()
    if (splits.map(len) > 1).any():
        raise ValueError("Hay personas en más de un split: revisar labels.csv.")
    split = splits.map(lambda s: s[0])

    vas = vas.dropna(subset=["vas"]).copy()
    vas["split"] = vas["persona_uid"].map(split)
    respuestas = vas[vas["split"].isin(["train", "val"])][
        ["persona_uid", "medicion", "vas"]].reset_index(drop=True)
    if respuestas.duplicated(["persona_uid", "medicion"]).any():
        raise ValueError("Más de una respuesta por (persona_uid, medicion).")

    objetivo = labels[labels["etiquetada"] & labels["split"].isin(["train", "val"])
                      & (labels["n_ventanas"].fillna(0) > 0)]
    objetivo = objetivo[["sesion_uid", "persona_uid", "medicion", "vas"]].reset_index(drop=True)

    cruce = objetivo.merge(respuestas, on=["persona_uid", "medicion"], how="left",
                           suffixes=("", "_csv"))
    if cruce["vas_csv"].isna().any() or not np.allclose(cruce["vas"], cruce["vas_csv"]):
        raise ValueError("El VAS de labels.csv no coincide con el del CSV del VAS.")

    claves_eeg = set(zip(objetivo["persona_uid"], objetivo["medicion"]))
    respuestas["con_eeg"] = [k in claves_eeg for k in
                             zip(respuestas["persona_uid"], respuestas["medicion"])]
    return respuestas, objetivo


def desplazamiento_m1(respuestas):
    """Media, por persona, de M1 menos la media de sus respuestas M2–M6."""
    m1 = respuestas[respuestas.medicion == "M1"].set_index("persona_uid")["vas"]
    resto = respuestas[respuestas.medicion != "M1"].groupby("persona_uid")["vas"].mean()
    comun = m1.index.intersection(resto.index)
    return float((m1.loc[comun] - resto.loc[comun]).mean()), int(len(comun))


def universos(respuestas, offset_m1):
    """Respuestas que forman cada conjunto de referencia, con el valor a usar."""
    sin_m1 = respuestas[respuestas.medicion != "M1"]
    con_m1_ajustado = respuestas.astype({"vas": float})
    es_m1 = con_m1_ajustado.medicion == "M1"
    con_m1_ajustado.loc[es_m1, "vas"] -= offset_m1
    return {
        "R1": sin_m1[sin_m1.con_eeg],
        "R2": sin_m1,
        "R3": respuestas,
        "R4": con_m1_ajustado,
    }


def referencias_loo(objetivo, universo):
    """
    Para cada sesión objetivo, nº de respuestas, media y varianza de las
    **otras** respuestas de su persona dentro del universo dado.
    """
    por_persona = {p: g for p, g in universo.groupby("persona_uid")}
    filas = []
    for persona, medicion in zip(objetivo["persona_uid"], objetivo["medicion"]):
        g = por_persona.get(persona)
        otros = (g.loc[g.medicion != medicion, "vas"].to_numpy(float)
                 if g is not None else np.array([]))
        n = len(otros)
        filas.append((n, otros.mean() if n else np.nan,
                      otros.var(ddof=1) if n >= 2 else np.nan))
    return pd.DataFrame(filas, index=objetivo.index, columns=["n_ref", "media_ref", "var_ref"])


# --- Estadística ---------------------------------------------------------------

def trigamma_inversa(x, tol=1e-8, max_iter=50):
    """Inversa de la trigamma por Newton (algoritmo de limma::trigammaInverse)."""
    if x > 1e7:
        return 1 / np.sqrt(x)
    if x < 1e-6:
        return 1 / x
    y = 0.5 + 1 / x
    for _ in range(max_iter):
        tri = special.polygamma(1, y)
        dif = tri * (1 - tri / x) / special.polygamma(2, y)
        y += dif
        if -dif / y < tol:
            break
    return float(y)


def ajustar_previa_varianzas(varianzas, gl):
    """
    Previa empírico-bayesiana de las varianzas por persona (Smyth 2004,
    `limma::fitFDist` sin covariables).

    Supone s²ᵢ ~ s0² · F(glᵢ, d0): las varianzas reales de las personas se
    reparten alrededor de una común s0², y d0 mide cuánto se parecen entre sí
    (d0 grande = todas casi iguales). Se estima por momentos sobre log s²,
    descontando la dispersión que el muestreo con glᵢ grados de libertad ya
    produce por sí solo. Si no queda dispersión real, d0 = ∞.
    """
    s2 = np.asarray(varianzas, float)
    gl = np.asarray(gl, float)
    ok = (s2 > 0) & (gl > 0)
    s2, gl = s2[ok], gl[ok]
    e = np.log(s2) - special.digamma(gl / 2) + np.log(gl / 2)
    e_media = e.mean()
    e_var = e.var(ddof=1) - special.polygamma(1, gl / 2).mean()
    if e_var <= 0:
        return {"d0": np.inf, "s0_2": float(np.exp(e_media)), "n": int(ok.sum())}
    d0 = 2 * trigamma_inversa(e_var)
    s0_2 = float(np.exp(e_media + special.digamma(d0 / 2) - np.log(d0 / 2)))
    return {"d0": float(d0), "s0_2": s0_2, "n": int(ok.sum())}


def encoger_varianza(var_ref, n_ref, previa):
    """Varianza posterior: (gl·s² + d0·s0²)/(gl + d0); sin datos, s0²."""
    gl = np.clip(np.asarray(n_ref, float) - 1, 0, None)
    s2 = np.nan_to_num(np.asarray(var_ref, float), nan=0.0)
    if np.isinf(previa["d0"]):
        return np.full(len(gl), previa["s0_2"])
    return (gl * s2 + previa["d0"] * previa["s0_2"]) / (gl + previa["d0"])


def bootstrap_diferencia_mse(err_a, err_b, grupos, n_rep=N_BOOTSTRAP, semilla=SEMILLA):
    """IC95 % de MSE(b) − MSE(a) remuestreando personas."""
    rng = np.random.default_rng(semilla)
    df = pd.DataFrame({"d": err_b ** 2 - err_a ** 2, "g": np.asarray(grupos)})
    por = df.groupby("g")["d"].agg(["sum", "size"])
    suma, tam = por["sum"].to_numpy(), por["size"].to_numpy()
    k = len(por)
    reps = np.empty(n_rep)
    for r in range(n_rep):
        i = rng.integers(0, k, k)
        reps[r] = suma[i].sum() / tam[i].sum()
    return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


def a_tercios(y):
    return a_clases(y, cortes_terciles(y))


def comparar(y_ref, y_var):
    """Acuerdo de tercios, kappa ponderado lineal y ρ entre dos etiquetas."""
    ok = np.isfinite(y_ref) & np.isfinite(y_var)
    c_ref, c_var = a_tercios(y_ref[ok]), a_tercios(y_var[ok])
    return {"n": int(ok.sum()),
            "acuerdo": float((c_ref == c_var).mean()),
            "kappa_lineal": float(cohen_kappa_score(c_ref, c_var, weights="linear")),
            "spearman": float(stats.spearmanr(y_ref[ok], y_var[ok])[0]),
            "cambios_de_dos_niveles": int((np.abs(c_ref - c_var) == 2).sum())}


# --- Figuras -------------------------------------------------------------------

def _estilo(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(TINTA_2)
    ax.tick_params(colors=TINTA_2, labelsize=9)
    ax.yaxis.grid(True, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)


def figura_encogimiento(sd_bruta, sd_encogida, s0, ruta):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    tope = max(np.nanmax(sd_bruta), np.nanmax(sd_encogida)) * 1.05
    ax.plot([0, tope], [0, tope], color=TINTA_2, linewidth=0.8, linestyle=":")
    ax.axhline(s0, color=COLOR_B, linewidth=1.2, linestyle="--")
    ax.text(tope * 0.98, s0 + 0.6, f"SD común s0 = {s0:.1f}", color=TINTA,
            fontsize=8, ha="right", va="bottom")
    ax.scatter(sd_bruta, sd_encogida, s=26, color=COLOR_A, edgecolor="white",
               linewidth=0.8)
    ax.set_xlim(0, tope)
    ax.set_ylim(0, tope)
    ax.set_xlabel("SD bruta de la persona (mm)", fontsize=9, color=TINTA_2)
    ax.set_ylabel("SD encogida (mm)", fontsize=9, color=TINTA_2)
    ax.set_title("Encogimiento de la SD hacia la común", fontsize=11,
                 color=TINTA, loc="left")
    _estilo(ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_extremos(tabla, ruta):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(tabla))
    ancho = 0.36
    for desplaz, col, color, nombre in ((-ancho / 2 - 0.01, "extremas_a", COLOR_A, "(a) centrado"),
                                        (ancho / 2 + 0.01, "extremas_b", COLOR_B, "(b) z encogida")):
        barras = ax.bar(x + desplaz, tabla[col] * 100, width=ancho, color=color, label=nombre)
        for b in barras:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                    f"{b.get_height():.0f} %", ha="center", fontsize=8, color=TINTA)
    ax.axhline(200 / 3, color=TINTA_2, linewidth=0.8, linestyle="--")
    ax.text(-0.45, 200 / 3 + 1, "2/3 si el reparto fuese uniforme",
            ha="left", fontsize=8, color=TINTA_2)
    ax.set_xticks(x, tabla["grupo_sd"], fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% de sesiones en «poca» o «mucha»", fontsize=9, color=TINTA_2)
    ax.set_xlabel("personas agrupadas por su SD del VAS", fontsize=9, color=TINTA_2)
    ax.set_title("Quién ocupa las clases extremas", fontsize=11, color=TINTA, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _estilo(ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_dispersion(y_a, y_b, ruta):
    ok = np.isfinite(y_a) & np.isfinite(y_b)
    y_a, y_b = y_a[ok], y_b[ok]
    cortes_a, cortes_b = cortes_terciles(y_a), cortes_terciles(y_b)
    igual = a_clases(y_a, cortes_a) == a_clases(y_b, cortes_b)
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in cortes_a:
        ax.axvline(c, color=TINTA_2, linewidth=0.8, linestyle=":")
    for c in cortes_b:
        ax.axhline(c, color=TINTA_2, linewidth=0.8, linestyle=":")
    ax.scatter(y_a[igual], y_b[igual], s=16, color="#b9b8b2", edgecolor="none",
               label=f"misma clase ({igual.mean() * 100:.0f} %)")
    ax.scatter(y_a[~igual], y_b[~igual], s=22, color=COLOR_B, edgecolor="white",
               linewidth=0.6, label=f"cambia de clase ({(~igual).mean() * 100:.0f} %)")
    ax.set_xlabel("(a) VAS centrado, Fase 14b (mm)", fontsize=9, color=TINTA_2)
    ax.set_ylabel("(b) z con SD encogida", fontsize=9, color=TINTA_2)
    ax.set_title("Etiqueta probada frente a z encogida", fontsize=11, color=TINTA, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _estilo(ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# --- Principal -----------------------------------------------------------------

def main():
    CARPETA_TABLAS.mkdir(exist_ok=True)
    CARPETA_FIGURAS.mkdir(exist_ok=True)
    resultados = {"umbral_acuerdo": UMBRAL_ACUERDO}

    _titulo("1. DATOS")
    respuestas, objetivo = cargar()
    print(f"Respuestas VAS train+val: {len(respuestas)} de "
          f"{respuestas.persona_uid.nunique()} personas "
          f"({int(respuestas.con_eeg.sum())} con EEG etiquetado)")
    print(f"Sesiones objetivo (con EEG etiquetado): {len(objetivo)} de "
          f"{objetivo.persona_uid.nunique()} personas")
    print("VAS de labels.csv y del CSV del VAS: coinciden en todas.")
    offset, n_offset = desplazamiento_m1(respuestas)
    print(f"Desplazamiento de M1 frente a la media M2–M6 de la persona: {offset:+.2f} mm "
          f"({n_offset} personas)")
    resultados["datos"] = {"respuestas": int(len(respuestas)),
                           "sesiones_objetivo": int(len(objetivo)),
                           "personas": int(objetivo.persona_uid.nunique()),
                           "desplazamiento_m1": offset}

    _titulo("2. DECISIÓN 1 — CONJUNTO DE REFERENCIA (error al predecir la sesión dejada fuera)")
    univ = universos(respuestas, offset)
    refs = {k: referencias_loo(objetivo, u) for k, u in univ.items()}
    comun = np.logical_and.reduce([r["n_ref"].to_numpy() >= 1 for r in refs.values()])
    y = objetivo["vas"].to_numpy(float)
    g = objetivo["persona_uid"].to_numpy()
    err_r1 = y[comun] - refs["R1"]["media_ref"].to_numpy()[comun]
    filas_ref = []
    for k, r in refs.items():
        err = y[comun] - r["media_ref"].to_numpy()[comun]
        ic = (bootstrap_diferencia_mse(err_r1, err, g[comun]) if k != "R1"
              else [0.0, 0.0])
        filas_ref.append({"referencia": k, "descripcion": REFERENCIAS[k],
                          "sesiones_con_referencia": int((r["n_ref"] >= 1).sum()),
                          "n_ref_mediana": float(r.loc[comun, "n_ref"].median()),
                          "mse": float((err ** 2).mean()), "mae": float(np.abs(err).mean()),
                          "dif_mse_vs_R1": float((err ** 2).mean() - (err_r1 ** 2).mean()),
                          "ic95_dif_inf": ic[0], "ic95_dif_sup": ic[1]})
    tabla_ref = pd.DataFrame(filas_ref)
    print(f"Sesiones comunes (con referencia en R1–R4): {int(comun.sum())}")
    print(tabla_ref.drop(columns="descripcion").round(3).to_string(index=False))
    mejoran = tabla_ref[(tabla_ref.referencia != "R1") & (tabla_ref.ic95_dif_sup < 0)]
    elegida = (mejoran.sort_values("mse").iloc[0]["referencia"] if len(mejoran) else "R1")
    print(f"Referencia elegida: {elegida} — {REFERENCIAS[elegida]}")
    tabla_ref.to_csv(CARPETA_TABLAS / "referencias.csv", index=False, float_format="%.4f")
    resultados["referencias"] = filas_ref
    resultados["referencia_elegida"] = elegida

    _titulo("3. ENCOGIMIENTO DE LA SD (empírico-bayesiano de Smyth)")
    u = univ[elegida]
    por_persona = u.groupby("persona_uid")["vas"].agg(["size", "var"])
    previa = ajustar_previa_varianzas(por_persona["var"], por_persona["size"] - 1)
    print(f"Previa ajustada sobre {previa['n']} personas: d0 = {previa['d0']:.2f} "
          f"grados de libertad, SD común s0 = {np.sqrt(previa['s0_2']):.2f} mm")
    print("(d0 es cuánto pesa la SD común frente a la propia: una persona con")
    print(f" gl = n−1 sesiones recibe un peso gl/(gl+d0) para su SD propia.)")
    sd_bruta = np.sqrt(por_persona["var"])
    sd_enc = np.sqrt(encoger_varianza(por_persona["var"], por_persona["size"], previa))
    print(f"SD por persona, P10–P90: bruta {sd_bruta.quantile(.1):.1f}–{sd_bruta.quantile(.9):.1f}"
          f" → encogida {np.quantile(sd_enc, .1):.1f}–{np.quantile(sd_enc, .9):.1f}")
    resultados["previa"] = {"d0": previa["d0"], "s0": float(np.sqrt(previa["s0_2"])),
                            "n_personas": previa["n"]}

    _titulo("4. VARIANTES DE LA ETIQUETA")
    r_el = refs[elegida]
    et = objetivo.copy()
    et["a"] = y - refs["R1"]["media_ref"].to_numpy()
    et["a_prima"] = y - r_el["media_ref"].to_numpy()
    sd_loo_enc = np.sqrt(encoger_varianza(r_el["var_ref"], r_el["n_ref"], previa))
    et["b"] = et["a_prima"] / sd_loo_enc
    et.loc[r_el["n_ref"].to_numpy() < 1, "b"] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        et["z_bruta"] = et["a_prima"] / np.sqrt(r_el["var_ref"].to_numpy())
    et.loc[~np.isfinite(et["z_bruta"]), "z_bruta"] = np.nan
    efecto_momento = et.groupby("medicion")["a_prima"].mean()
    et["c"] = et["a_prima"] - et["medicion"].map(efecto_momento)
    # Etiqueta confirmatoria del pre-registro: (a) residualizada por visita.
    cen = centrar_por_persona_loo(objetivo, ["vas"])
    if not np.allclose(cen["vas"], et["a"], equal_nan=True):
        raise ValueError("(a) no coincide con `centrar_por_persona_loo` de la Fase 14.")
    et["a_visita"] = residualizar_por_visita(cen, ["vas"], cen["medicion"],
                                             cen["persona_uid"])["vas"].to_numpy()
    print(f"(a) Fase 14b: {int(et['a'].notna().sum())} sesiones, SD {et['a'].std():.1f} mm, "
          f"rango [{et['a'].min():.0f}, {et['a'].max():.0f}]  (Fase 14b: 537, SD 18.2, ±50)")
    print(f"(a′) centrado con {elegida}: {int(et['a_prima'].notna().sum())} sesiones")
    print(f"(b) z encogida: {int(et['b'].notna().sum())} sesiones")
    print(f"z bruta: {int(et['z_bruta'].notna().sum())} sesiones "
          f"(sin SD propia o con SD = 0 en el resto)")
    print("Efecto de cada momento sobre (a′), mm:",
          {k: round(float(v), 2) for k, v in efecto_momento.items()})
    resultados["efecto_momento"] = {k: float(v) for k, v in efecto_momento.items()}

    _titulo("5. CRITERIO — ACUERDO DE TERCIOS CON (a)")
    variantes = {"a_prima": "(a′) centrado con la referencia elegida",
                 "b": "(b) z con SD encogida",
                 "c": "(c) (a′) sin efecto de momento",
                 "z_bruta": "z con SD bruta (informativa)",
                 "a_visita": "(a) residualizada por visita (informativa)"}
    filas = []
    for col, nombre in variantes.items():
        r = comparar(et["a"].to_numpy(float), et[col].to_numpy(float))
        filas.append({"variante": nombre, **r})
    # Descomposición (informativa): (a′) ya cambia la referencia, así que (b) y
    # (c) frente a (a′) aíslan lo que aportan la tipificación y el momento.
    for col, nombre in (("b", "(b) frente a (a′): solo la tipificación"),
                        ("c", "(c) frente a (a′): solo el momento"),
                        ("c", "(c) frente a (a) residualizada por visita")):
        base = "a_visita" if "visita" in nombre else "a_prima"
        r = comparar(et[base].to_numpy(float), et[col].to_numpy(float))
        filas.append({"variante": nombre, **r})
    tabla = pd.DataFrame(filas)
    print(tabla.round(3).to_string(index=False))
    tabla.to_csv(CARPETA_TABLAS / "acuerdo_con_a.csv", index=False, float_format="%.4f")
    resultados["acuerdo"] = filas

    en_criterio = {"a_prima": filas[0], "b": filas[1], "c": filas[2]}
    if elegida == "R1":
        en_criterio.pop("a_prima")  # (a′) es (a) exactamente
    no_llegan = [k for k, f in en_criterio.items() if f["acuerdo"] < UMBRAL_ACUERDO]
    if not no_llegan:
        decision = "cerrar"
        print(f"\nDECISIÓN: todas coinciden con (a) en ≥{UMBRAL_ACUERDO:.0%}. "
              f"La línea se cierra sin volver al EEG.")
    else:
        principal = next(k for k in ("b", "c", "a_prima") if k in no_llegan)
        decision = f"paso 3 con variante principal {principal}"
        print(f"\nDECISIÓN: no llegan al {UMBRAL_ACUERDO:.0%}: {no_llegan}. "
              f"Paso 3 con la variante principal {principal}.")
    resultados["decision"] = decision

    _titulo("6. MECANISMO — QUIÉN OCUPA LAS CLASES EXTREMAS")
    ok = et["a"].notna() & et["b"].notna()
    sub = et[ok].copy()
    sub["clase_a"] = a_tercios(sub["a"].to_numpy())
    sub["clase_b"] = a_tercios(sub["b"].to_numpy())
    sd_persona = sd_bruta.reindex(sub["persona_uid"].unique())
    grupo = pd.qcut(sd_persona.rank(method="first"), 3,
                    labels=["SD baja", "SD media", "SD alta"])
    sub["grupo_sd"] = sub["persona_uid"].map(grupo)
    mec = (sub.groupby("grupo_sd", observed=True)
           .agg(personas=("persona_uid", "nunique"), sesiones=("a", "size"),
                sd_mediana=("persona_uid", lambda p: float(sd_bruta.loc[p.unique()].median())),
                extremas_a=("clase_a", lambda c: float((c != 1).mean())),
                extremas_b=("clase_b", lambda c: float((c != 1).mean())))
           .reset_index())
    print(mec.round(3).to_string(index=False))
    mec.to_csv(CARPETA_TABLAS / "clases_extremas_por_sd.csv", index=False, float_format="%.4f")
    resultados["mecanismo"] = mec.to_dict(orient="records")

    _titulo("7. FIGURAS")
    figura_encogimiento(sd_bruta.to_numpy(), sd_enc, np.sqrt(previa["s0_2"]),
                        CARPETA_FIGURAS / "sd_bruta_vs_encogida.png")
    figura_extremos(mec, CARPETA_FIGURAS / "clases_extremas_por_sd.png")
    figura_dispersion(et["a"].to_numpy(float), et["b"].to_numpy(float),
                      CARPETA_FIGURAS / "centrada_vs_z_encogida.png")
    print("Escritas en", CARPETA_FIGURAS)

    with open(CARPETA / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2, default=str)
    print("Resultados en", CARPETA / "resultados.json")


if __name__ == "__main__":
    main()
