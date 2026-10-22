"""
Fatiga general (FG) y fatiga cognitiva (FC) frente al EEG — paso 1:
descriptivo de los cuestionarios (petición de los tutores).

Antes de relacionar FG y FC con el EEG hay que saber qué se mide y en qué
escala se compara: si las puntuaciones son consistentes, qué pasa entre M4 y
M5 (FG y FC caen a la mitad en toda la cohorte), si los alumnos sin EEG sirven
de referencia para tipificar dentro de cada medición, cuánta varianza es de
persona y cuánta de sesión, y si la variación de una sesión a otra es señal o
error de medida.

**Solo cuestionario: no se lee EEG ni se entrena ningún modelo.** Plan y
criterios de decisión (ponderación de la referencia, puerta del paso 3):
`INFORME.md`, escritos antes de ejecutar este script.

**Solo `train` + `val`** para los participantes: las personas de test se
apartan al cargar. La referencia (alumnos sin EEG) no pertenece a ningún split.
El `CÓDIGO` del alumno no sale en ninguna tabla, figura ni log.

Ejecutar (desde la raíz del repositorio):
    python analisis/fatiga_fg_fc/01_descriptivo/descriptivo_fg_fc.py > analisis/fatiga_fg_fc/01_descriptivo/ejecucion.log

La ubicación del dataset se toma de la variable de entorno `EEG_DATA`, con
`C:\\Users\\migue\\Desktop\\TFG\\EEG-Data` por defecto (ver `analisis/README.md`).
"""

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

CARPETA = Path(__file__).resolve().parent
LINEA = CARPETA.parent
sys.path.insert(0, str(LINEA))
import datos_fg_fc as dfg  # noqa: E402

# ICC(1) y su bootstrap por personas: los mismos que en normalizacion_vas, paso 1.
_spec = importlib.util.spec_from_file_location(
    "descriptivo_vas", LINEA.parent / "normalizacion_vas" / "01_descriptivo" / "descriptivo_vas.py")
_descriptivo_vas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_descriptivo_vas)
icc1 = _descriptivo_vas.icc1
icc1_bootstrap = _descriptivo_vas.icc1_bootstrap

SEMILLA = 42
N_BOOTSTRAP = 2000
UMBRAL_V = 0.10        # C1: V de Cramér a partir de la cual se pondera la referencia
MIN_KISH = 100         # C1: tamaño efectivo mínimo tras el raking
UMBRAL_FIABILIDAD = 0.50  # C2: puerta del paso 3
MEDICIONES = dfg.MEDICIONES
MEDIDAS = ["FG", "FC"]

CARPETA_TABLAS = CARPETA / "tablas"
CARPETA_FIGURAS = CARPETA / "figuras"

# Paleta de la línea (validada para daltonismo) y tinta neutra.
COLOR_A = "#2a78d6"
COLOR_B = "#eb6834"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
REJILLA = "#e4e3df"
SOMBRA = "#f0efec"


def _titulo(texto):
    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


# --- Estadística ---------------------------------------------------------------

def alfa_cronbach(items):
    """α de Cronbach sobre las filas con todos los ítems."""
    x = items.dropna()
    k = x.shape[1]
    return float(k / (k - 1) * (1 - x.var(ddof=1).sum() / x.sum(axis=1).var(ddof=1)))


def cramer_v(grupo, variable):
    """V de Cramér entre dos variables categóricas (sin corrección de Yates)."""
    tabla = pd.crosstab(grupo, variable)
    if tabla.shape[0] < 2 or tabla.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(tabla, correction=False)[0]
    return float(np.sqrt(chi2 / (tabla.to_numpy().sum() * (min(tabla.shape) - 1))))


def residuo_efectos_fijos(y, personas, mediciones):
    """Residuo de MCO de `y` sobre efectos fijos de persona y de medición."""
    personas = pd.Categorical(personas)
    mediciones = pd.Categorical(mediciones)
    X = np.column_stack([np.eye(len(personas.categories))[personas.codes],
                         np.eye(len(mediciones.categories))[mediciones.codes][:, 1:]])
    y = np.asarray(y, float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coef


def bootstrap_personas(datos, estadistico, n_rep=N_BOOTSTRAP, semilla=SEMILLA):
    """
    Punto e IC95 percentil de `estadistico(datos)` remuestreando personas.

    En cada réplica, cada copia de una persona cuenta como un grupo distinto.
    `estadistico` devuelve un dict de cifras; se devuelve un dict con el punto y
    el IC de cada una.
    """
    rng = np.random.default_rng(semilla)
    datos = datos.reset_index(drop=True)
    bloques = list(datos.groupby("persona_uid").indices.values())
    punto = estadistico(datos)
    reps = {k: [] for k in punto}
    for _ in range(n_rep):
        elegidos = rng.integers(0, len(bloques), len(bloques))
        filas = np.concatenate([bloques[i] for i in elegidos])
        grupos = np.concatenate([np.full(len(bloques[i]), j) for j, i in enumerate(elegidos)])
        replica = datos.iloc[filas].reset_index(drop=True).assign(persona_uid=grupos)
        r = estadistico(replica)
        for k in punto:
            reps[k].append(r[k])
    return {k: {"punto": float(punto[k]),
                "ic95": [float(np.nanpercentile(reps[k], 2.5)),
                         float(np.nanpercentile(reps[k], 97.5))]} for k in punto}


def fiabilidad_intra(datos, items):
    """
    Fiabilidad de la variación intra-persona por mitades (impares / pares).

    A cada mitad se le quitan los efectos fijos de persona y medición; la
    correlación entre residuos se corrige con Spearman-Brown.
    """
    impares = datos[items[0::2]].sum(axis=1)
    pares = datos[items[1::2]].sum(axis=1)
    r1 = residuo_efectos_fijos(impares, datos["persona_uid"], datos["medicion"])
    r2 = residuo_efectos_fijos(pares, datos["persona_uid"], datos["medicion"])
    r = float(np.corrcoef(r1, r2)[0, 1])
    return {"r_mitades": r, "spearman_brown": 2 * r / (1 + r)}


def media_ic(x):
    x = pd.Series(x).dropna()
    t = stats.t.ppf(0.975, len(x) - 1) if len(x) > 1 else np.nan
    return float(x.mean()), float(x.mean() - t * x.sem()), float(x.mean() + t * x.sem()), int(len(x))


# --- Datos ---------------------------------------------------------------------

def cargar():
    bd = dfg.leer_bd()
    labels = pd.read_csv(dfg.RUTA_LABELS)
    respuestas, resumen_enlace = dfg.respuestas_participantes(bd, labels)
    personas = dfg.caracteristicas_persona(respuestas, labels)
    sesiones = dfg.sesiones_etiquetadas(respuestas, labels)
    ref = dfg.referencia(bd)

    tv = ["train", "val"]
    resumen = {
        "enlace": resumen_enlace,
        "personas_test_apartadas": int(respuestas.loc[respuestas.split == "test", "persona_uid"].nunique()),
        "respuestas_test_apartadas": int((respuestas.split == "test").sum()),
        "personas_sin_eeg_apartadas": int(respuestas.loc[respuestas.split.isna(), "persona_uid"].nunique()),
        "sesiones_etiquetadas_tv": int(sesiones.split.isin(tv).sum()),
        "sesiones_excluidas_tv": sesiones.loc[sesiones.split.isin(tv) & sesiones.excluida,
                                              "sesion_uid"].tolist(),
        "sesiones_sin_fgfc_tv": int((sesiones.split.isin(tv) & ~sesiones.con_fgfc).sum()),
    }
    respuestas = respuestas[respuestas.split.isin(tv)].reset_index(drop=True)
    sesiones = sesiones[sesiones.split.isin(tv) & sesiones.con_fgfc].reset_index(drop=True)
    return respuestas, personas, sesiones, ref, resumen


# --- Figuras -------------------------------------------------------------------

def _estilo(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(TINTA_2)
    ax.tick_params(colors=TINTA_2, labelsize=8)
    ax.yaxis.grid(True, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)


def figura_salto(tabla, ruta):
    paneles = [("FG", "Fatiga general (0–24)"), ("FC", "Fatiga cognitiva (0–24)"),
               ("FSD", "Sueño/descanso PedsQL (media ítem, 0–4)"),
               ("FATIGA_MENTAL", "Fatiga mental ahora (1–10)"),
               ("VAS", "VAS fatiga (0–100)"), ("AE", "Agotamiento MBI-SS (media ítem, 0–6)")]
    fig, ejes = plt.subplots(2, 3, figsize=(12, 6.4))
    x = np.arange(len(MEDICIONES))
    for ax, (var, titulo) in zip(ejes.ravel(), paneles):
        ax.axvspan(3.5, 5.5, color=SOMBRA, zorder=0)
        for grupo, color, marcador, desplazamiento in (("participantes", COLOR_A, "o", -0.06),
                                                       ("referencia", COLOR_B, "s", 0.06)):
            t = tabla[(tabla.variable == var) & (tabla.grupo == grupo)].set_index("medicion")
            if t.empty:
                continue
            t = t.reindex(MEDICIONES)
            ax.errorbar(x + desplazamiento, t["media"],
                        yerr=[t["media"] - t["ic95_inf"], t["ic95_sup"] - t["media"]],
                        fmt=marcador + "-", color=color, markersize=7, capsize=2.5, linewidth=2,
                        markeredgecolor="white", label=grupo, zorder=3)
        ax.set_xticks(x, MEDICIONES)
        ax.set_title(titulo, fontsize=10, color=TINTA, loc="left")
        _estilo(ax)
    ejes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Media por medición (IC95). Sombreado: M5–M6. Participantes: train+val; "
                 "referencia: alumnos sin EEG (no tienen VAS)", fontsize=9, color=TINTA_2,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_razon(tabla, ruta):
    fig, ax = plt.subplots(figsize=(7, 6.5))
    filas = tabla["variable"].unique().tolist()[::-1]
    y = np.arange(len(filas))
    ax.axvline(1, color=TINTA_2, linewidth=1)
    for grupo, color, marcador in (("participantes", COLOR_A, "o"), ("referencia", COLOR_B, "s")):
        t = tabla[tabla.grupo == grupo].set_index("variable").reindex(filas)
        ax.scatter(t["razon"], y, color=color, marker=marcador, s=46, edgecolor="white",
                   linewidth=0.8, label=grupo, zorder=3)
    ax.set_yticks(y, filas, fontsize=8)
    ax.set_xlabel("media M5–M6 / media M1–M4", fontsize=9, color=TINTA_2)
    ax.set_title("Cuánto queda en M5–M6 de lo que había en M1–M4", fontsize=10,
                 color=TINTA, loc="left")
    _estilo(ax)
    ax.xaxis.grid(True, color=REJILLA, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_representatividad(tabla, ruta):
    fig, ejes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    x = np.arange(len(MEDICIONES))
    for ax, var in zip(ejes, ["centro", "sexo", "curso"]):
        t = tabla[tabla.variable == var].set_index("medicion").reindex(MEDICIONES)
        ax.bar(x, t["v_cramer"], color=COLOR_A, width=0.6, edgecolor="white", linewidth=2)
        ax.axhline(UMBRAL_V, color=TINTA, linewidth=1, linestyle="--")
        ax.set_xticks(x, MEDICIONES)
        ax.set_title(f"{var}", fontsize=10, color=TINTA, loc="left")
        _estilo(ax)
    ejes[0].set_ylabel("V de Cramér (participantes vs referencia)", fontsize=8, color=TINTA_2)
    ejes[1].text(-0.4, UMBRAL_V, f"criterio {UMBRAL_V}", fontsize=8, color=TINTA, va="bottom", ha="left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_tipificacion(tabla, ruta):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(MEDICIONES))
    for ax, sufijo, titulo, etiqueta in (
            (ax1, "", "Bruto (suma 0–24)", "puntos"),
            (ax2, "_z", "Tipificado dentro de cada medición", "z (percentil normalizado)")):
        ax.axvspan(3.5, 5.5, color=SOMBRA, zorder=0)
        for medida, color, marcador in (("FG", COLOR_A, "o"), ("FC", COLOR_B, "s")):
            t = tabla[tabla.variable == medida + sufijo].set_index("medicion").reindex(MEDICIONES)
            ax.errorbar(x, t["media"], yerr=[t["media"] - t["ic95_inf"], t["ic95_sup"] - t["media"]],
                        fmt=marcador + "-", color=color, markersize=7, capsize=2.5, linewidth=2,
                        markeredgecolor="white", label=medida, zorder=3)
        if sufijo:
            ax.axhline(0, color=TINTA_2, linewidth=0.8)
        ax.set_xticks(x, MEDICIONES)
        ax.set_ylabel(etiqueta, fontsize=8, color=TINTA_2)
        ax.set_title(titulo, fontsize=10, color=TINTA, loc="left")
        _estilo(ax)
    ax1.legend(frameon=False, fontsize=8)
    fig.suptitle("Participantes de train+val, media por medición (IC95)", fontsize=9,
                 color=TINTA_2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# --- Principal -----------------------------------------------------------------

def main():
    CARPETA_TABLAS.mkdir(exist_ok=True)
    CARPETA_FIGURAS.mkdir(exist_ok=True)
    resultados = {"criterios": {"umbral_v": UMBRAL_V, "min_kish": MIN_KISH,
                                "umbral_fiabilidad": UMBRAL_FIABILIDAD,
                                "n_bootstrap": N_BOOTSTRAP, "semilla": SEMILLA}}

    _titulo("1. DATOS Y ENLACE")
    respuestas, personas, sesiones, ref, resumen = cargar()
    resultados["datos"] = resumen
    enl = resumen["enlace"]
    print(f"Filas con pulsera: {enl['filas_con_pulsera']} -> {enl['respuestas']} respuestas "
          f"de {enl['personas']} persona_uid")
    for c in enl["correcciones"]:
        print(f"  Corrección aplicada: {c['nombre']} ({c['filas']} filas)")
    print(f"  Personas con dos códigos (excepción documentada): {enl['personas_con_dos_codigos']}")
    print(f"Test apartado: {resumen['respuestas_test_apartadas']} respuestas de "
          f"{resumen['personas_test_apartadas']} personas")
    print(f"Sin EEG etiquetado (sin split), apartadas: {resumen['personas_sin_eeg_apartadas']} personas")
    print(f"Participantes train+val: {len(respuestas)} respuestas de "
          f"{respuestas.persona_uid.nunique()} personas "
          f"(con FG y FC: {int(respuestas[['FG', 'FC']].notna().all(axis=1).sum())})")
    print(f"Sesiones etiquetadas train+val: {resumen['sesiones_etiquetadas_tv']}; excluidas "
          f"{resumen['sesiones_excluidas_tv']}; sin FG/FC {resumen['sesiones_sin_fgfc_tv']}")
    print(f"Sesiones con EEG y FG/FC: {len(sesiones)} de {sesiones.persona_uid.nunique()} personas")
    print(f"Referencia (alumnos sin EEG): {len(ref)} respuestas; por medición "
          f"{ref.medicion.value_counts().reindex(MEDICIONES).to_dict()}")
    resultados["datos"].update({
        "respuestas_tv": int(len(respuestas)), "personas_tv": int(respuestas.persona_uid.nunique()),
        "sesiones_con_fgfc": int(len(sesiones)), "personas_con_eeg": int(sesiones.persona_uid.nunique()),
        "referencia": int(len(ref)),
        "referencia_por_medicion": ref.medicion.value_counts().reindex(MEDICIONES).astype(int).to_dict()})

    con_fgfc = respuestas.dropna(subset=["FG", "FC"]).copy()
    # Otros bloques del formulario: media por ítem (solo para describir el salto).
    for df in (con_fgfc, ref):
        df["FSD"] = df[dfg.ITEMS_FSD].mean(axis=1)
        df["WS"] = df[dfg.ITEMS_WS].mean(axis=1)
        df["AE"] = df[dfg.ITEMS_AE].mean(axis=1)
    con_fgfc["VAS"] = con_fgfc["vas_bd"]

    _titulo("2. PUNTUACIÓN Y CONSISTENCIA INTERNA")
    filas = []
    for grupo, datos in (("participantes", con_fgfc), ("referencia", ref)):
        for m in MEDICIONES + ["todas"]:
            d = datos if m == "todas" else datos[datos.medicion == m]
            fila = {"grupo": grupo, "medicion": m, "n": int(len(d))}
            for medida in MEDIDAS:
                fila[f"alfa_{medida}"] = alfa_cronbach(d[dfg.ITEMS[medida]])
                fila[f"media_{medida}"] = float(d[medida].mean())
                fila[f"sd_{medida}"] = float(d[medida].std(ddof=1))
                fila[f"pct_cero_{medida}"] = float((d[medida] == 0).mean() * 100)
            filas.append(fila)
    tabla_alfa = pd.DataFrame(filas)
    print(tabla_alfa.round(2).to_string(index=False))
    tabla_alfa.to_csv(CARPETA_TABLAS / "consistencia_por_medicion.csv", index=False, float_format="%.4f")
    resultados["consistencia"] = tabla_alfa.round(4).to_dict(orient="records")

    _titulo("3. EL SALTO M4 -> M5")
    variables = ["FG", "FC", "FSD", "WS", "AE", "FATIGA_MENTAL", "VAS"]
    filas = []
    for grupo, datos in (("participantes", con_fgfc), ("referencia", ref)):
        for var in variables:
            if var not in datos:
                continue
            for m in MEDICIONES:
                media, inf, sup, n = media_ic(datos.loc[datos.medicion == m, var])
                filas.append({"grupo": grupo, "variable": var, "medicion": m, "n": n,
                              "media": media, "ic95_inf": inf, "ic95_sup": sup,
                              "sd": float(datos.loc[datos.medicion == m, var].std(ddof=1))})
    tabla_m = pd.DataFrame(filas)
    print("Media por medición:")
    print(tabla_m.pivot_table(index=["grupo", "variable"], columns="medicion", values="media")
          .round(2).to_string())
    tabla_m.to_csv(CARPETA_TABLAS / "media_por_medicion.csv", index=False, float_format="%.4f")
    resultados["media_por_medicion"] = tabla_m.round(4).to_dict(orient="records")

    print("\nCambio dentro de persona entre mediciones consecutivas (participantes):")
    ancho = con_fgfc.pivot_table(index="persona_uid", columns="medicion",
                                 values=["FG", "FC", "FATIGA_MENTAL", "VAS"])
    filas = []
    for a, b in zip(MEDICIONES[:-1], MEDICIONES[1:]):
        for var in ["FG", "FC", "FATIGA_MENTAL", "VAS"]:
            if (var, a) not in ancho or (var, b) not in ancho:
                continue
            d = (ancho[(var, b)] - ancho[(var, a)]).dropna()
            media, inf, sup, n = media_ic(d)
            filas.append({"cambio": f"{b}-{a}", "variable": var, "n_personas": n, "media": media,
                          "ic95_inf": inf, "ic95_sup": sup, "pct_baja": float((d < 0).mean() * 100),
                          "pct_sube": float((d > 0).mean() * 100)})
    tabla_cambio = pd.DataFrame(filas)
    print(tabla_cambio.round(2).to_string(index=False))
    tabla_cambio.to_csv(CARPETA_TABLAS / "cambio_entre_mediciones.csv", index=False, float_format="%.4f")
    resultados["cambio_entre_mediciones"] = tabla_cambio.round(4).to_dict(orient="records")

    print("\nM5 − M4 por centro (media de la medición; participantes y referencia por separado):")
    filas = []
    for grupo, datos in (("participantes", con_fgfc), ("referencia", ref)):
        for c, d in datos.groupby("centro"):
            fila = {"grupo": grupo, "centro": c}
            for var in ["FG", "FC", "FATIGA_MENTAL"]:
                fila[f"{var}_M4"] = float(d.loc[d.medicion == "M4", var].mean())
                fila[f"{var}_M5"] = float(d.loc[d.medicion == "M5", var].mean())
                fila[f"{var}_dif"] = fila[f"{var}_M5"] - fila[f"{var}_M4"]
            filas.append(fila)
    tabla_centro = pd.DataFrame(filas)
    print(tabla_centro[["grupo", "centro", "FG_dif", "FC_dif", "FATIGA_MENTAL_dif"]]
          .round(2).to_string(index=False))
    for var in ["FG", "FC", "FATIGA_MENTAL"]:
        print(f"  {var}: baja de M4 a M5 en {int((tabla_centro[f'{var}_dif'] < 0).sum())} "
              f"de {len(tabla_centro)} combinaciones centro × grupo")
    tabla_centro.to_csv(CARPETA_TABLAS / "m5_menos_m4_por_centro.csv", index=False, float_format="%.4f")
    resultados["m5_menos_m4_por_centro"] = tabla_centro.round(4).to_dict(orient="records")

    print("\nRazón media M5–M6 / media M1–M4, por ítem y bloque:")
    filas = []
    columnas = ([(i, i) for i in dfg.ITEMS_FG + dfg.ITEMS_FC]
                + [("FS/D (media)", "FSD"), ("Webex (media)", "WS"),
                   ("MBI-SS agotamiento (media)", "AE"), ("FATIGA_MENTAL", "FATIGA_MENTAL")])
    for grupo, datos in (("participantes", con_fgfc), ("referencia", ref)):
        antes = datos.medicion.isin(["M1", "M2", "M3", "M4"])
        for nombre, col in columnas:
            filas.append({"grupo": grupo, "variable": nombre,
                          "media_M1_M4": float(datos.loc[antes, col].mean()),
                          "media_M5_M6": float(datos.loc[~antes, col].mean())})
    tabla_razon = pd.DataFrame(filas)
    tabla_razon["razon"] = tabla_razon["media_M5_M6"] / tabla_razon["media_M1_M4"]
    print(tabla_razon.pivot(index="variable", columns="grupo", values="razon")
          .reindex([c[0] for c in columnas]).round(2).to_string())
    tabla_razon.to_csv(CARPETA_TABLAS / "razon_por_item.csv", index=False, float_format="%.4f")
    resultados["razon_por_item"] = tabla_razon.round(4).to_dict(orient="records")

    print("\n¿Cambia el constructo o solo la escala? Spearman entre personas dentro de cada medición:")
    filas = []
    for m in MEDICIONES:
        p = con_fgfc[con_fgfc.medicion == m]
        r = ref[ref.medicion == m]
        filas.append({
            "medicion": m,
            "FG_VAS_part": stats.spearmanr(p.FG, p.VAS, nan_policy="omit")[0],
            "FC_VAS_part": stats.spearmanr(p.FC, p.VAS, nan_policy="omit")[0],
            "FG_FM_part": stats.spearmanr(p.FG, p.FATIGA_MENTAL, nan_policy="omit")[0],
            "FC_FM_part": stats.spearmanr(p.FC, p.FATIGA_MENTAL, nan_policy="omit")[0],
            "FG_FM_ref": stats.spearmanr(r.FG, r.FATIGA_MENTAL, nan_policy="omit")[0],
            "FC_FM_ref": stats.spearmanr(r.FC, r.FATIGA_MENTAL, nan_policy="omit")[0],
            "FG_FC_ref": stats.spearmanr(r.FG, r.FC)[0]})
    tabla_constructo = pd.DataFrame(filas)
    print(tabla_constructo.round(3).to_string(index=False))
    tabla_constructo.to_csv(CARPETA_TABLAS / "constructo_por_medicion.csv", index=False,
                            float_format="%.4f")
    resultados["constructo_por_medicion"] = tabla_constructo.round(4).to_dict(orient="records")

    _titulo("4. REPRESENTATIVIDAD DE LA REFERENCIA (criterio C1)")
    eeg = set(sesiones.persona_uid)
    objetivo = (con_fgfc[con_fgfc.persona_uid.isin(eeg)][["persona_uid", "medicion", "FG", "FC"]]
                .join(personas, on="persona_uid"))
    print(f"Participantes con EEG: {len(objetivo)} respuestas de {objetivo.persona_uid.nunique()} personas")
    filas = []
    for m in MEDICIONES:
        o = objetivo[objetivo.medicion == m]
        r = ref[ref.medicion == m]
        for var in ["centro", "sexo", "curso"]:
            g = pd.concat([pd.Series("participantes", index=o.index),
                           pd.Series("referencia", index=r.index)], ignore_index=True)
            v = pd.concat([o[var], r[var]], ignore_index=True)
            ok = v.notna()
            filas.append({"medicion": m, "variable": var, "v_cramer": cramer_v(g[ok], v[ok]),
                          "n_part": int(o[var].notna().sum()), "n_ref": int(r[var].notna().sum())})
    tabla_v = pd.DataFrame(filas)
    print(tabla_v.pivot(index="variable", columns="medicion", values="v_cramer").round(3).to_string())
    print("\nComposición (% dentro de cada grupo, todas las mediciones):")
    composicion = {}
    for var in ["centro", "sexo", "curso"]:
        c = pd.concat([objetivo[var].value_counts(normalize=True).rename("participantes"),
                       ref[var].value_counts(normalize=True).rename("referencia")], axis=1) * 100
        print(c.round(1).to_string())
        composicion[var] = c.round(2).to_dict()
    v_max = float(tabla_v.v_cramer.max())
    ponderar = v_max >= UMBRAL_V
    variables_raking = ["centro", "sexo", "curso"]
    pesos, kish = None, None
    if ponderar:
        pesos, kish = dfg.pesos_referencia(ref, objetivo, variables_raking)
        if min(kish.values()) < MIN_KISH:
            variables_raking = ["centro"]
            pesos, kish = dfg.pesos_referencia(ref, objetivo, variables_raking)
    print(f"\nV máxima = {v_max:.3f} -> {'SE PONDERA' if ponderar else 'no se pondera'} "
          f"la referencia (criterio V >= {UMBRAL_V})")
    if ponderar:
        print(f"  Raking sobre {variables_raking}; tamaño efectivo de Kish por medición: "
              f"{ {k: round(v, 1) for k, v in kish.items()} }")
        print(f"  Peso máximo / mínimo (filas con peso > 0): {pesos[pesos > 0].max():.2f} / "
              f"{pesos[pesos > 0].min():.2f}; filas con peso 0: {int((pesos == 0).sum())}")
    print("\nFG y FC: participantes frente a referencia, diferencia tipificada (d) por medición:")
    filas = []
    for m in MEDICIONES:
        o = objetivo[objetivo.medicion == m]
        en_m = (ref.medicion == m).to_numpy()
        r = ref[en_m]
        fila = {"medicion": m}
        for medida in MEDIDAS:
            sd = r[medida].std(ddof=1)
            fila[f"d_{medida}_sin_pesos"] = float((o[medida].mean() - r[medida].mean()) / sd)
            if ponderar:
                media_w = np.average(r[medida], weights=pesos[en_m])
                fila[f"d_{medida}_con_pesos"] = float((o[medida].mean() - media_w) / sd)
        filas.append(fila)
    tabla_d = pd.DataFrame(filas)
    print(tabla_d.round(3).to_string(index=False))
    tabla_v.to_csv(CARPETA_TABLAS / "representatividad_v_cramer.csv", index=False, float_format="%.4f")
    tabla_d.to_csv(CARPETA_TABLAS / "representatividad_diferencias.csv", index=False, float_format="%.4f")
    resultados["representatividad"] = {
        "v_cramer": tabla_v.round(4).to_dict(orient="records"), "v_max": v_max,
        "composicion_pct": composicion, "ponderar": bool(ponderar),
        "variables_raking": variables_raking if ponderar else None,
        "kish_por_medicion": kish, "diferencias": tabla_d.round(4).to_dict(orient="records")}

    _titulo("5. TIPIFICACIÓN DENTRO DE CADA MEDICIÓN")
    for medida in MEDIDAS:
        con_fgfc[f"{medida}_z"] = dfg.tipificar_por_medicion(con_fgfc, ref, medida, "percentil", pesos)
        con_fgfc[f"{medida}_zlin"] = dfg.tipificar_por_medicion(con_fgfc, ref, medida, "lineal", pesos)
        con_fgfc[f"{medida}_zsp"] = dfg.tipificar_por_medicion(con_fgfc, ref, medida, "percentil", None)
    principal = "percentil normalizado" + (" con referencia ponderada" if ponderar else "")
    print(f"Principal: {principal}")
    filas = []
    acuerdo = {}
    for medida in MEDIDAS:
        z, zl, zs = con_fgfc[f"{medida}_z"], con_fgfc[f"{medida}_zlin"], con_fgfc[f"{medida}_zsp"]
        acuerdo[medida] = {
            "pearson_percentil_lineal": float(np.corrcoef(z, zl)[0, 1]),
            "dif_abs_media_percentil_lineal": float((z - zl).abs().mean()),
            "pearson_con_sin_pesos": float(np.corrcoef(z, zs)[0, 1]),
            "dif_abs_media_con_sin_pesos": float((z - zs).abs().mean()),
            "dif_abs_max_con_sin_pesos": float((z - zs).abs().max())}
        print(f"{medida}: percentil vs lineal r = {acuerdo[medida]['pearson_percentil_lineal']:.3f} "
              f"(|dif| media {acuerdo[medida]['dif_abs_media_percentil_lineal']:.3f}); "
              f"con vs sin pesos r = {acuerdo[medida]['pearson_con_sin_pesos']:.4f} "
              f"(|dif| media {acuerdo[medida]['dif_abs_media_con_sin_pesos']:.3f}, "
              f"máx {acuerdo[medida]['dif_abs_max_con_sin_pesos']:.3f})")
        for sufijo in ("", "_z"):
            for m in MEDICIONES:
                media, inf, sup, n = media_ic(con_fgfc.loc[con_fgfc.medicion == m, medida + sufijo])
                filas.append({"variable": medida + sufijo, "medicion": m, "n": n, "media": media,
                              "ic95_inf": inf, "ic95_sup": sup})
    tabla_z = pd.DataFrame(filas)
    print("\nMedia de los participantes por medición, bruta y tipificada:")
    print(tabla_z.pivot(index="variable", columns="medicion", values="media").round(3).to_string())
    ancho_z = con_fgfc.pivot_table(index="persona_uid", columns="medicion", values=["FG_z", "FC_z"])
    for medida in MEDIDAS:
        d = (ancho_z[(f"{medida}_z", "M5")] - ancho_z[(f"{medida}_z", "M4")]).dropna()
        media, inf, sup, n = media_ic(d)
        print(f"  {medida}_z, M5 − M4 dentro de persona: {media:+.3f} [{inf:+.3f}, {sup:+.3f}] (n = {n})")
        acuerdo[medida]["m5_menos_m4_z"] = [media, inf, sup, n]
    tabla_z.to_csv(CARPETA_TABLAS / "tipificada_por_medicion.csv", index=False, float_format="%.4f")
    resultados["tipificacion"] = {"principal": principal, "acuerdo": acuerdo,
                                  "por_medicion": tabla_z.round(4).to_dict(orient="records")}

    _titulo("6. ICC(1)")
    sesiones_z = sesiones[["persona_uid", "medicion"]].merge(
        con_fgfc[["persona_uid", "medicion", "FG", "FC", "FG_z", "FC_z", "VAS", "FATIGA_MENTAL"]],
        on=["persona_uid", "medicion"], how="left")
    filas = []
    for conjunto, datos in (("respuestas M1–M6", con_fgfc), ("sesiones con EEG", sesiones_z)):
        for var in ["FG", "FC", "FG_z", "FC_z", "VAS", "FATIGA_MENTAL"]:
            d = datos.dropna(subset=[var])
            r = icc1(d[var], d["persona_uid"])
            ic = icc1_bootstrap(d[var], d["persona_uid"], n_rep=N_BOOTSTRAP, semilla=SEMILLA)
            filas.append({"conjunto": conjunto, "variable": var, "n": r["n_sesiones"],
                          "personas": r["n_personas"], "icc": r["icc"],
                          "ic95_boot_inf": ic[0], "ic95_boot_sup": ic[1]})
            print(f"{conjunto:18s} {var:14s} n={r['n_sesiones']:4d} personas={r['n_personas']:3d}  "
                  f"ICC = {r['icc']:.3f} [{ic[0]:.3f}, {ic[1]:.3f}]")
    tabla_icc = pd.DataFrame(filas)
    tabla_icc.to_csv(CARPETA_TABLAS / "icc.csv", index=False, float_format="%.4f")
    resultados["icc"] = tabla_icc.round(4).to_dict(orient="records")

    _titulo("7. FIABILIDAD INTRA-PERSONA (criterio C2)")
    multiples = con_fgfc[con_fgfc.groupby("persona_uid")["FG"].transform("size") >= 2]
    fiabilidad = {}
    for medida in MEDIDAS:
        fiabilidad[medida] = {}
        for nombre, datos in (("M1-M6", multiples),
                              ("M1-M4", multiples[multiples.medicion.isin(["M1", "M2", "M3", "M4"])])):
            datos = datos[datos.groupby("persona_uid")["FG"].transform("size") >= 2]
            b = bootstrap_personas(datos, lambda d, it=dfg.ITEMS[medida]: fiabilidad_intra(d, it))
            alfa = alfa_cronbach(datos[dfg.ITEMS[medida]])
            icc_z = icc1(datos[f"{medida}_z"], datos["persona_uid"])["icc"]
            fiabilidad[medida][nombre] = {
                "n": int(len(datos)), "personas": int(datos.persona_uid.nunique()),
                "r_mitades": b["r_mitades"], "spearman_brown": b["spearman_brown"],
                "formula_alfa_icc": float((alfa - icc_z) / (1 - icc_z)), "alfa": alfa, "icc_z": icc_z}
            sb = b["spearman_brown"]
            print(f"{medida} {nombre}: {len(datos)} respuestas de {datos.persona_uid.nunique()} personas; "
                  f"r mitades = {b['r_mitades']['punto']:.3f}; Spearman-Brown = {sb['punto']:.3f} "
                  f"[{sb['ic95'][0]:.3f}, {sb['ic95'][1]:.3f}]; (α − ICC)/(1 − ICC) = "
                  f"{fiabilidad[medida][nombre]['formula_alfa_icc']:.3f}")
    puerta = {m: fiabilidad[m]["M1-M6"]["spearman_brown"]["punto"] >= UMBRAL_FIABILIDAD for m in MEDIDAS}
    for m in MEDIDAS:
        print(f"  Puerta del paso 3 para {m} (Spearman-Brown M1–M6 >= {UMBRAL_FIABILIDAD}): "
              f"{'SÍ' if puerta[m] else 'NO'}")
    resultados["fiabilidad_intra"] = fiabilidad
    resultados["puerta_paso3"] = puerta

    _titulo("8. RELACIÓN CON VAS Y FATIGA_MENTAL")
    variables = ["FG_z", "FC_z", "VAS", "FATIGA_MENTAL"]
    pares = [("FG_z", "VAS"), ("FC_z", "VAS"), ("FG_z", "FATIGA_MENTAL"), ("FC_z", "FATIGA_MENTAL"),
             ("FG_z", "FC_z"), ("VAS", "FATIGA_MENTAL")]
    completas = con_fgfc.dropna(subset=variables)

    def entre(d):
        medias = d.groupby("persona_uid")[variables].mean()
        return {f"{a}~{b}": stats.spearmanr(medias[a], medias[b])[0] for a, b in pares}

    def intra(d):
        d = d[d.groupby("persona_uid")["VAS"].transform("size") >= 2]
        res = {v: residuo_efectos_fijos(d[v], d["persona_uid"], d["medicion"]) for v in variables}
        return {f"{a}~{b}": stats.spearmanr(res[a], res[b])[0] for a, b in pares}

    relacion = {"entre_personas": bootstrap_personas(completas, entre),
                "intra_persona": bootstrap_personas(completas, intra)}
    filas = []
    for nivel, r in relacion.items():
        for par, v in r.items():
            filas.append({"nivel": nivel, "par": par, "rho": v["punto"],
                          "ic95_inf": v["ic95"][0], "ic95_sup": v["ic95"][1]})
    tabla_rel = pd.DataFrame(filas)
    print(f"{len(completas)} respuestas de {completas.persona_uid.nunique()} personas con las cuatro medidas")
    print(tabla_rel.round(3).to_string(index=False))
    tabla_rel.to_csv(CARPETA_TABLAS / "relacion_vas_fatiga_mental.csv", index=False, float_format="%.4f")
    resultados["relacion"] = tabla_rel.round(4).to_dict(orient="records")

    _titulo("9. SEXO, CENTRO Y CURSO (media por persona de FG y FC tipificadas)")
    por_persona = (con_fgfc[con_fgfc.persona_uid.isin(eeg)].groupby("persona_uid")[["FG_z", "FC_z"]]
                   .mean().join(personas))
    filas = []
    for medida in ["FG_z", "FC_z"]:
        xx = por_persona.loc[por_persona.sexo == "XX", medida]
        xy = por_persona.loc[por_persona.sexo == "XY", medida]
        # Biserial de rangos 2U/(n1·n2) − 1: positiva si el primer grupo puntúa más.
        u = stats.mannwhitneyu(xx, xy)
        c3 = por_persona.loc[por_persona.curso == 3, medida]
        c4 = por_persona.loc[por_persona.curso == 4, medida]
        uc = stats.mannwhitneyu(c4, c3)
        grupos = [g[medida].to_numpy() for _, g in por_persona.groupby("centro")]
        h = stats.kruskal(*grupos)
        n, k = len(por_persona), len(grupos)
        filas += [
            {"medida": medida, "factor": "sexo", "contraste": "XX − XY",
             "diferencia_medias": float(xx.mean() - xy.mean()), "n": f"{len(xx)}/{len(xy)}",
             "p": float(u.pvalue), "efecto": float(2 * u.statistic / (len(xx) * len(xy)) - 1),
             "efecto_nombre": "correlación biserial de rangos"},
            {"medida": medida, "factor": "curso", "contraste": "4.º − 3.º",
             "diferencia_medias": float(c4.mean() - c3.mean()), "n": f"{len(c4)}/{len(c3)}",
             "p": float(uc.pvalue), "efecto": float(2 * uc.statistic / (len(c3) * len(c4)) - 1),
             "efecto_nombre": "correlación biserial de rangos"},
            {"medida": medida, "factor": "centro", "contraste": "Kruskal-Wallis, 8 centros",
             "diferencia_medias": float(por_persona.groupby("centro")[medida].mean().max()
                                        - por_persona.groupby("centro")[medida].mean().min()),
             "n": str(n), "p": float(h.pvalue), "efecto": float((h.statistic - k + 1) / (n - k)),
             "efecto_nombre": "eta² H"}]
    tabla_cov = pd.DataFrame(filas)
    print(f"{len(por_persona)} personas con EEG")
    print(tabla_cov.round(3).to_string(index=False))
    print("\nMedia por centro:")
    print(por_persona.groupby("centro")[["FG_z", "FC_z"]].agg(["mean", "size"]).round(2).to_string())
    tabla_cov.to_csv(CARPETA_TABLAS / "sexo_centro_curso.csv", index=False, float_format="%.4f")
    resultados["sexo_centro_curso"] = tabla_cov.round(4).to_dict(orient="records")

    _titulo("10. COBERTURA CON EEG")
    por_med = sesiones.medicion.value_counts().reindex(MEDICIONES, fill_value=0)
    n_pp = sesiones.groupby("persona_uid").size()
    print("Sesiones con EEG y FG/FC por medición:", por_med.to_dict())
    print("Sesiones por persona:", n_pp.value_counts().sort_index().to_dict(),
          f"(personas con ≥2: {int((n_pp >= 2).sum())}, sesiones en ellas: {int(n_pp[n_pp >= 2].sum())})")
    s24 = sesiones[sesiones.medicion.isin(["M2", "M3", "M4"])]
    n24 = s24.groupby("persona_uid").size()
    print(f"Solo M2–M4: {len(s24)} sesiones de {len(n24)} personas; con ≥2: {int((n24 >= 2).sum())} "
          f"personas, {int(n24[n24 >= 2].sum())} sesiones")
    r14 = con_fgfc[con_fgfc.persona_uid.isin(eeg) & con_fgfc.medicion.isin(["M1", "M2", "M3", "M4"])]
    print(f"Respuestas M1–M4 por persona con EEG (para el rasgo sin M5–M6): "
          f"{r14.groupby('persona_uid').size().value_counts().sort_index().to_dict()}")
    resultados["cobertura"] = {
        "por_medicion": por_med.astype(int).to_dict(),
        "sesiones_por_persona": {int(k): int(v) for k, v in n_pp.value_counts().sort_index().items()},
        "personas_ge2": int((n_pp >= 2).sum()), "sesiones_en_personas_ge2": int(n_pp[n_pp >= 2].sum()),
        "m2_m4": {"sesiones": int(len(s24)), "personas": int(len(n24)),
                  "personas_ge2": int((n24 >= 2).sum()), "sesiones_en_personas_ge2": int(n24[n24 >= 2].sum())}}

    _titulo("11. DECISIONES")
    print(f"C1 — referencia: {'ponderada por raking sobre ' + str(variables_raking) if ponderar else 'sin ponderar'} "
          f"(V máxima {v_max:.3f})")
    for m in MEDIDAS:
        sb = fiabilidad[m]["M1-M6"]["spearman_brown"]["punto"]
        print(f"C2 — paso 3 con {m}: {'sí' if puerta[m] else 'no'} (Spearman-Brown {sb:.3f})")

    _titulo("12. FIGURAS")
    figura_salto(tabla_m, CARPETA_FIGURAS / "salto_por_medicion.png")
    figura_razon(tabla_razon, CARPETA_FIGURAS / "razon_por_item.png")
    figura_representatividad(tabla_v, CARPETA_FIGURAS / "representatividad.png")
    figura_tipificacion(tabla_z, CARPETA_FIGURAS / "tipificacion.png")
    print("Escritas en", CARPETA_FIGURAS)

    with open(CARPETA / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2, default=float)
    print("Resultados en", CARPETA / "resultados.json")


if __name__ == "__main__":
    main()
