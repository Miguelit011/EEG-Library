"""
Fatiga general (FG) y fatiga cognitiva (FC) frente al EEG — paso 1:
descriptivo de los cuestionarios (petición de los tutores).

Antes de relacionar FG y FC con el EEG hay que saber qué se mide: si la base
está bien codificada, si las puntuaciones son consistentes, cómo se mueven a lo
largo del curso, cuánta varianza es de la persona y cuánta de la sesión, y si
la variación de una sesión a otra es señal o error de medida.

**Solo cuestionario: no se lee EEG ni se entrena ningún modelo.** Plan y
criterio de decisión (puerta del paso 3): `INFORME.md`, escritos antes de
ejecutar este script.

**Solo `train` + `val`** para los participantes: las personas de test se
apartan al cargar. El `CÓDIGO` del alumno solo se usa en memoria para
emparejar las dos versiones de la base; no sale en ninguna tabla, figura ni log.

Ejecutar (desde la raíz del repositorio):
    python analisis/fatiga_fg_fc/01_descriptivo/descriptivo_fg_fc.py > analisis/fatiga_fg_fc/01_descriptivo/ejecucion.log

La ubicación del dataset se toma de la variable de entorno `EEG_DATA`, con
`EEG-Data` junto al repositorio por defecto (ver `analisis/README.md`).
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

# Versión de la base con el error de codificación de M5–M6, solo para verificar
# la corrección (punto 0).
RUTA_BD_SIN_CORREGIR = dfg.EEG_DATA / "BD_COMPLETA_sincalificaciones.xlsx"

SEMILLA = 42
N_BOOTSTRAP = 2000
UMBRAL_FIABILIDAD = 0.50  # puerta del paso 3
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
        r = estadistico(datos.iloc[filas].reset_index(drop=True).assign(persona_uid=grupos))
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


# --- Punto 0: la corrección de la base -----------------------------------------

def verificar_correccion(corregida, anterior):
    """
    Empareja las filas de las dos versiones por (CÓDIGO, medida, centro,
    pulsera) y comprueba la regla de la corrección en los 18 ítems PedsQL.

    Devuelve un resumen por medición y la tabla de anomalías de rango de la
    base corregida. No devuelve CÓDIGO.
    """
    clave = ["CÓDIGO", "MEDIDA", "N_CENTRO", "FITBIT"]

    def _con_clave(bd):
        b = bd.copy()
        b["FITBIT"] = b["FITBIT"].fillna(-1)
        b["CÓDIGO"] = b["CÓDIGO"].fillna("").astype(str)
        b["orden"] = b.groupby(clave).cumcount()
        return b

    c, a = _con_clave(corregida), _con_clave(anterior)
    m = c.merge(a, on=clave + ["orden"], how="outer", suffixes=("_c", "_a"), indicator=True)
    emparejamiento = {"emparejadas": int((m._merge == "both").sum()),
                      "solo_corregida": int((m._merge == "left_only").sum()),
                      "solo_anterior": int((m._merge == "right_only").sum())}
    b = m[m._merge == "both"]
    filas = []
    for med in range(1, 7):
        s = b[b["MEDIDA"] == med]
        vc = s[[i + "_c" for i in dfg.ITEMS_PEDSQL]].to_numpy(float)
        va = s[[i + "_a" for i in dfg.ITEMS_PEDSQL]].to_numpy(float)
        ambos = ~np.isnan(vc) & ~np.isnan(va)
        filas.append({
            "medicion": f"M{med}", "filas": int(len(s)), "celdas": int(ambos.sum()),
            "iguales": int(((vc == va) & ambos).sum()),
            "cumplen_regla": int(((va == np.maximum(vc - 1, 0)) & ambos).sum()),
            "nan_distinto": int((np.isnan(vc) != np.isnan(va)).sum()),
            "cuatros_restaurados": int(((vc == 4) & (va == 3) & ambos).sum())})
    anomalias = []
    for item in dfg.ITEMS_PEDSQL:
        fuera = corregida[item].notna() & ~corregida[item].isin([0, 1, 2, 3, 4])
        for (med, valor), n in corregida[fuera].groupby(["medicion", item]).size().items():
            anomalias.append({"item": item, "medicion": med, "valor": float(valor), "filas": int(n)})
    return emparejamiento, pd.DataFrame(filas), pd.DataFrame(anomalias)


# --- Figuras -------------------------------------------------------------------

def _estilo(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(TINTA_2)
    ax.tick_params(colors=TINTA_2, labelsize=8)
    ax.yaxis.grid(True, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)


def figura_correccion(tabla, ruta):
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    x = np.arange(len(MEDICIONES))
    for ax, medida in zip(ejes, MEDIDAS):
        for version, color, marcador, etiqueta in (
                ("anterior", COLOR_B, "s", "versión anterior (M5–M6 mal codificadas)"),
                ("corregida", COLOR_A, "o", "base corregida")):
            t = tabla[(tabla.version == version) & (tabla.variable == medida)].set_index("medicion")
            t = t.reindex(MEDICIONES)
            ax.errorbar(x, t["media"], yerr=[t["media"] - t["ic95_inf"], t["ic95_sup"] - t["media"]],
                        fmt=marcador + "-", color=color, markersize=7, capsize=2.5, linewidth=2,
                        markeredgecolor="white", label=etiqueta, zorder=3)
        ax.set_xticks(x, MEDICIONES)
        ax.set_title({"FG": "Fatiga general", "FC": "Fatiga cognitiva"}[medida] + " (suma 0–24)",
                     fontsize=10, color=TINTA, loc="left")
        _estilo(ax)
    ejes[0].set_ylabel("media (IC95)", fontsize=8, color=TINTA_2)
    ejes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Todas las respuestas de la base (participantes y no participantes)",
                 fontsize=9, color=TINTA_2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_calendario(tabla, ruta):
    paneles = [("FG", "Fatiga general (0–24)"), ("FC", "Fatiga cognitiva (0–24)"),
               ("VAS", "VAS fatiga (0–100)"), ("FATIGA_MENTAL", "Fatiga mental ahora (1–10)")]
    fig, ejes = plt.subplots(1, 4, figsize=(13, 3.4))
    x = np.arange(len(MEDICIONES))
    for ax, (var, titulo) in zip(ejes, paneles):
        t = tabla[tabla.variable == var].set_index("medicion").reindex(MEDICIONES)
        ax.errorbar(x, t["media"], yerr=[t["media"] - t["ic95_inf"], t["ic95_sup"] - t["media"]],
                    fmt="o-", color=COLOR_A, markersize=7, capsize=2.5, linewidth=2,
                    markeredgecolor="white", zorder=3)
        ax.set_xticks(x, MEDICIONES)
        ax.set_title(titulo, fontsize=10, color=TINTA, loc="left")
        _estilo(ax)
    fig.suptitle("Participantes de train+val: media por medición (IC95)", fontsize=9,
                 color=TINTA_2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_distribucion(datos, ruta):
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, medida in zip(ejes, MEDIDAS):
        ax.hist(datos[medida], bins=np.arange(-0.5, 25.5, 1), color=COLOR_A,
                edgecolor="white", linewidth=1.5)
        ax.set_title({"FG": "Fatiga general", "FC": "Fatiga cognitiva"}[medida],
                     fontsize=10, color=TINTA, loc="left")
        ax.set_xlabel("suma de los 6 ítems (0–24)", fontsize=8, color=TINTA_2)
        _estilo(ax)
    ejes[0].set_ylabel("respuestas", fontsize=8, color=TINTA_2)
    fig.suptitle(f"Participantes de train+val, M1–M6 ({len(datos)} respuestas)", fontsize=9,
                 color=TINTA_2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# --- Principal -----------------------------------------------------------------

def main():
    CARPETA_TABLAS.mkdir(exist_ok=True)
    CARPETA_FIGURAS.mkdir(exist_ok=True)
    resultados = {"criterio": {"umbral_fiabilidad": UMBRAL_FIABILIDAD,
                               "n_bootstrap": N_BOOTSTRAP, "semilla": SEMILLA}}

    _titulo("0. LA CORRECCIÓN DE LA BASE")
    bd, respuestas, personas, sesiones, enlace = dfg.cargar_enlace()
    anterior = dfg.leer_bd(RUTA_BD_SIN_CORREGIR)
    print(f"Base corregida: {enlace['base']} (SHA-256 {enlace['sha256_base']})")
    print(f"Versión anterior: {RUTA_BD_SIN_CORREGIR.name} (SHA-256 {dfg.sha256(RUTA_BD_SIN_CORREGIR)})")
    emp, tabla_regla, anomalias = verificar_correccion(bd, anterior)
    print(f"Filas emparejadas por (CÓDIGO, medida, centro, pulsera): {emp}")
    print(tabla_regla.to_string(index=False))
    m14 = tabla_regla[tabla_regla.medicion.isin(MEDICIONES[:4])]
    m56 = tabla_regla[tabla_regla.medicion.isin(MEDICIONES[4:])]
    regla_ok = bool((m14.iguales == m14.celdas).all() and (m56.cumplen_regla == m56.celdas).all()
                    and (tabla_regla.nan_distinto == 0).all())
    print(f"M1–M4 idénticas y M5–M6 = max(corregida − 1, 0) en todas las celdas: "
          f"{'SÍ' if regla_ok else 'NO'}")
    print("Valores de la PedsQL fuera de 0–4 en la base corregida:")
    print(anomalias.to_string(index=False) if len(anomalias) else "  ninguno")
    filas = []
    for version, datos in (("anterior", anterior), ("corregida", bd)):
        for medida in MEDIDAS:
            for m in MEDICIONES:
                media, inf, sup, n = media_ic(datos.loc[datos.medicion == m, medida])
                filas.append({"version": version, "variable": medida, "medicion": m, "n": n,
                              "media": media, "ic95_inf": inf, "ic95_sup": sup})
    tabla_version = pd.DataFrame(filas)
    print("\nMedia por medición de toda la base, antes y después de corregir:")
    print(tabla_version.pivot_table(index=["variable", "version"], columns="medicion",
                                    values="media").round(2).to_string())
    tabla_regla.to_csv(CARPETA_TABLAS / "correccion_regla.csv", index=False)
    tabla_version.to_csv(CARPETA_TABLAS / "correccion_medias.csv", index=False, float_format="%.4f")
    resultados["correccion"] = {
        "sha256_anterior": dfg.sha256(RUTA_BD_SIN_CORREGIR), "emparejamiento": emp,
        "regla_cumplida": regla_ok, "por_medicion": tabla_regla.to_dict(orient="records"),
        "anomalias_rango": anomalias.to_dict(orient="records"),
        "medias": tabla_version.round(4).to_dict(orient="records")}

    _titulo("1. DATOS Y ENLACE")
    tv = ["train", "val"]
    print(f"Filas con pulsera: {enlace['filas_con_pulsera']} -> {enlace['respuestas']} respuestas "
          f"de {enlace['personas']} persona_uid")
    for c in enlace["correcciones"]:
        print(f"  Corrección aplicada: {c['nombre']} ({c['filas']} filas)")
    print(f"  Personas con dos códigos (excepción documentada): {enlace['personas_con_dos_codigos']}")
    en_test = respuestas.split == "test"
    print(f"Test apartado: {int(en_test.sum())} respuestas de "
          f"{respuestas.loc[en_test, 'persona_uid'].nunique()} personas")
    print(f"Sin EEG etiquetado (sin split), apartadas: "
          f"{respuestas.loc[respuestas.split.isna(), 'persona_uid'].nunique()} personas")
    sesiones_tv = sesiones[sesiones.split.isin(tv)]
    respuestas = respuestas[respuestas.split.isin(tv)].reset_index(drop=True)
    con_fgfc = respuestas.dropna(subset=["FG", "FC"]).copy()
    sesiones = sesiones_tv[sesiones_tv.con_fgfc].reset_index(drop=True)
    print(f"Participantes train+val: {len(respuestas)} respuestas de "
          f"{respuestas.persona_uid.nunique()} personas (con FG y FC: {len(con_fgfc)})")
    sin_ventanas = sesiones_tv.loc[~sesiones_tv.con_ventanas, "sesion_uid"].tolist()
    sin_fgfc = int((sesiones_tv.con_ventanas & ~sesiones_tv.con_fgfc).sum())
    print(f"Sesiones etiquetadas train+val: {len(sesiones_tv)}; sin ventanas EEG: "
          f"{len(sin_ventanas)} {sin_ventanas}; con ventanas pero sin FG/FC: {sin_fgfc}; "
          f"con EEG y FG/FC: {len(sesiones)} de {sesiones.persona_uid.nunique()} personas")
    resultados["datos"] = {
        **enlace, "respuestas_tv": int(len(respuestas)), "con_fgfc_tv": int(len(con_fgfc)),
        "personas_tv": int(respuestas.persona_uid.nunique()),
        "sesiones_etiquetadas_tv": int(len(sesiones_tv)), "sesiones_sin_ventanas": sin_ventanas,
        "sesiones_con_ventanas_sin_fgfc": sin_fgfc, "sesiones_con_fgfc": int(len(sesiones)),
        "personas_con_eeg": int(sesiones.persona_uid.nunique())}

    con_fgfc["VAS"] = con_fgfc["vas_bd"]
    for var in ["FG", "FC", "VAS", "FATIGA_MENTAL"]:
        con_fgfc[var + "_c"] = dfg.centrar_por_medicion(con_fgfc.dropna(subset=[var]), var) \
            .reindex(con_fgfc.index)

    _titulo("2. PUNTUACIÓN Y CONSISTENCIA")
    filas = []
    for m in MEDICIONES + ["todas"]:
        d = con_fgfc if m == "todas" else con_fgfc[con_fgfc.medicion == m]
        fila = {"medicion": m, "n": int(len(d))}
        for medida in MEDIDAS:
            fila[f"alfa_{medida}"] = alfa_cronbach(d[dfg.ITEMS[medida]])
            fila[f"media_{medida}"] = float(d[medida].mean())
            fila[f"sd_{medida}"] = float(d[medida].std(ddof=1))
            fila[f"asimetria_{medida}"] = float(d[medida].skew())
            fila[f"pct_cero_{medida}"] = float((d[medida] == 0).mean() * 100)
            fila[f"pct_24_{medida}"] = float((d[medida] == 24).mean() * 100)
        filas.append(fila)
    tabla_cons = pd.DataFrame(filas)
    print(tabla_cons.round(2).to_string(index=False))
    tabla_cons.to_csv(CARPETA_TABLAS / "consistencia_por_medicion.csv", index=False, float_format="%.4f")
    resultados["consistencia"] = tabla_cons.round(4).to_dict(orient="records")

    _titulo("3. EFECTO DE CALENDARIO")
    variables = ["FG", "FC", "VAS", "FATIGA_MENTAL"]
    filas = []
    for var in variables:
        for m in MEDICIONES:
            media, inf, sup, n = media_ic(con_fgfc.loc[con_fgfc.medicion == m, var])
            filas.append({"variable": var, "medicion": m, "n": n, "media": media,
                          "ic95_inf": inf, "ic95_sup": sup})
    tabla_cal = pd.DataFrame(filas)
    print("Media por medición:")
    print(tabla_cal.pivot(index="variable", columns="medicion", values="media")
          .reindex(variables).round(2).to_string())
    ancho = con_fgfc.pivot_table(index="persona_uid", columns="medicion", values=variables)
    filas = []
    for a, b in zip(MEDICIONES[:-1], MEDICIONES[1:]):
        for var in variables:
            d = (ancho[(var, b)] - ancho[(var, a)]).dropna()
            media, inf, sup, n = media_ic(d)
            filas.append({"cambio": f"{b}-{a}", "variable": var, "n_personas": n, "media": media,
                          "ic95_inf": inf, "ic95_sup": sup})
    tabla_cambio = pd.DataFrame(filas)
    print("\nCambio dentro de persona entre mediciones consecutivas:")
    print(tabla_cambio.pivot(index="variable", columns="cambio", values="media")
          .reindex(variables).round(2).to_string())
    filas = []
    for var in variables:
        d = con_fgfc.dropna(subset=[var])
        d = d[d.groupby("persona_uid")[var].transform("size") >= 2]
        c = d[var] - d.groupby("persona_uid")[var].transform("mean")
        r2 = 1 - ((c - c.groupby(d.medicion).transform("mean")) ** 2).sum() / (c ** 2).sum()
        filas.append({"variable": var, "r2_medicion_intra_persona": float(r2), "n": int(len(d))})
    tabla_r2 = pd.DataFrame(filas)
    print("\nParte de la varianza intra-persona que explica la medición:")
    print(tabla_r2.round(3).to_string(index=False))
    tabla_cal.to_csv(CARPETA_TABLAS / "media_por_medicion.csv", index=False, float_format="%.4f")
    tabla_cambio.to_csv(CARPETA_TABLAS / "cambio_entre_mediciones.csv", index=False, float_format="%.4f")
    tabla_r2.to_csv(CARPETA_TABLAS / "r2_medicion.csv", index=False, float_format="%.4f")
    resultados["calendario"] = {"medias": tabla_cal.round(4).to_dict(orient="records"),
                                "cambios": tabla_cambio.round(4).to_dict(orient="records"),
                                "r2_medicion": tabla_r2.round(4).to_dict(orient="records")}

    _titulo("4. ICC(1)")
    sesiones_v = sesiones[["persona_uid", "medicion"]].merge(con_fgfc, on=["persona_uid", "medicion"],
                                                             how="left")
    filas = []
    for conjunto, datos in (("respuestas M1–M6", con_fgfc), ("sesiones con EEG", sesiones_v)):
        for var in ["FG", "FC", "FG_c", "FC_c", "VAS", "FATIGA_MENTAL"]:
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

    _titulo("5. FIABILIDAD INTRA-PERSONA (puerta del paso 3)")
    multiples = con_fgfc[con_fgfc.groupby("persona_uid")["FG"].transform("size") >= 2]
    fiabilidad = {}
    for medida in MEDIDAS:
        b = bootstrap_personas(multiples, lambda d, it=dfg.ITEMS[medida]: fiabilidad_intra(d, it))
        fiabilidad[medida] = {"n": int(len(multiples)), "personas": int(multiples.persona_uid.nunique()),
                              "r_mitades": b["r_mitades"], "spearman_brown": b["spearman_brown"]}
        sb = b["spearman_brown"]
        print(f"{medida}: {len(multiples)} respuestas de {multiples.persona_uid.nunique()} personas; "
              f"r mitades = {b['r_mitades']['punto']:.3f}; Spearman-Brown = {sb['punto']:.3f} "
              f"[{sb['ic95'][0]:.3f}, {sb['ic95'][1]:.3f}]")
    puerta = {m: fiabilidad[m]["spearman_brown"]["punto"] >= UMBRAL_FIABILIDAD for m in MEDIDAS}
    for m in MEDIDAS:
        print(f"  Puerta del paso 3 para {m} (Spearman-Brown >= {UMBRAL_FIABILIDAD}): "
              f"{'SÍ' if puerta[m] else 'NO'}")
    resultados["fiabilidad_intra"] = fiabilidad
    resultados["puerta_paso3"] = puerta

    _titulo("6. RELACIÓN CON VAS Y FATIGA_MENTAL")
    pares = [("FG", "VAS"), ("FC", "VAS"), ("FG", "FATIGA_MENTAL"), ("FC", "FATIGA_MENTAL"),
             ("FG", "FC"), ("VAS", "FATIGA_MENTAL")]
    completas = con_fgfc.dropna(subset=variables)

    def entre(d):
        medias = d.groupby("persona_uid")[[v + "_c" for v in variables]].mean()
        return {f"{a}~{b}": stats.spearmanr(medias[a + "_c"], medias[b + "_c"])[0] for a, b in pares}

    def intra(d):
        d = d[d.groupby("persona_uid")["VAS"].transform("size") >= 2]
        res = {v: residuo_efectos_fijos(d[v], d["persona_uid"], d["medicion"]) for v in variables}
        return {f"{a}~{b}": stats.spearmanr(res[a], res[b])[0] for a, b in pares}

    relacion = {"entre_personas": bootstrap_personas(completas, entre),
                "intra_persona": bootstrap_personas(completas, intra)}
    filas = [{"nivel": nivel, "par": par, "rho": v["punto"], "ic95_inf": v["ic95"][0],
              "ic95_sup": v["ic95"][1]} for nivel, r in relacion.items() for par, v in r.items()]
    tabla_rel = pd.DataFrame(filas)
    print(f"{len(completas)} respuestas de {completas.persona_uid.nunique()} personas con las cuatro medidas")
    print(tabla_rel.round(3).to_string(index=False))
    tabla_rel.to_csv(CARPETA_TABLAS / "relacion_vas_fatiga_mental.csv", index=False, float_format="%.4f")
    resultados["relacion"] = tabla_rel.round(4).to_dict(orient="records")

    _titulo("7. SEXO, CURSO Y CENTRO (media por persona de FG y FC centradas)")
    eeg = set(sesiones.persona_uid)
    por_persona = (con_fgfc[con_fgfc.persona_uid.isin(eeg)].groupby("persona_uid")[["FG_c", "FC_c"]]
                   .mean().join(personas))
    filas = []
    for medida in ["FG_c", "FC_c"]:
        # Biserial de rangos 2U/(n1·n2) − 1: positiva si el primer grupo puntúa más.
        xx = por_persona.loc[por_persona.sexo == "XX", medida]
        xy = por_persona.loc[por_persona.sexo == "XY", medida]
        u = stats.mannwhitneyu(xx, xy)
        c4 = por_persona.loc[por_persona.curso == 4, medida]
        c3 = por_persona.loc[por_persona.curso == 3, medida]
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
             "p": float(uc.pvalue), "efecto": float(2 * uc.statistic / (len(c4) * len(c3)) - 1),
             "efecto_nombre": "correlación biserial de rangos"},
            {"medida": medida, "factor": "centro", "contraste": "Kruskal-Wallis, 8 centros",
             "diferencia_medias": float(por_persona.groupby("centro")[medida].mean().max()
                                        - por_persona.groupby("centro")[medida].mean().min()),
             "n": str(n), "p": float(h.pvalue), "efecto": float((h.statistic - k + 1) / (n - k)),
             "efecto_nombre": "eta² H"}]
    tabla_cov = pd.DataFrame(filas)
    print(f"{len(por_persona)} personas con EEG (puntos de la escala 0–24)")
    print(tabla_cov.round(3).to_string(index=False))
    tabla_cov.to_csv(CARPETA_TABLAS / "sexo_curso_centro.csv", index=False, float_format="%.4f")
    resultados["sexo_curso_centro"] = tabla_cov.round(4).to_dict(orient="records")

    _titulo("8. COBERTURA CON EEG")
    por_med = sesiones.medicion.value_counts().reindex(MEDICIONES, fill_value=0)
    n_pp = sesiones.groupby("persona_uid").size()
    print("Sesiones con EEG y FG/FC por medición:", por_med.to_dict())
    print("Sesiones por persona:", n_pp.value_counts().sort_index().to_dict(),
          f"(personas con ≥2: {int((n_pp >= 2).sum())}, sesiones en ellas: {int(n_pp[n_pp >= 2].sum())})")
    r_pp = con_fgfc[con_fgfc.persona_uid.isin(eeg)].groupby("persona_uid").size()
    print("Respuestas M1–M6 por persona con EEG (para el rasgo del paso 2):",
          r_pp.value_counts().sort_index().to_dict())
    resultados["cobertura"] = {
        "por_medicion": por_med.astype(int).to_dict(),
        "sesiones_por_persona": {int(k): int(v) for k, v in n_pp.value_counts().sort_index().items()},
        "personas_ge2": int((n_pp >= 2).sum()), "sesiones_en_personas_ge2": int(n_pp[n_pp >= 2].sum()),
        "respuestas_por_persona": {int(k): int(v) for k, v in r_pp.value_counts().sort_index().items()}}

    _titulo("9. DECISIÓN")
    for m in MEDIDAS:
        sb = fiabilidad[m]["spearman_brown"]["punto"]
        print(f"Paso 3 con {m}: {'sí' if puerta[m] else 'no'} (Spearman-Brown {sb:.3f})")

    _titulo("10. FIGURAS")
    figura_correccion(tabla_version, CARPETA_FIGURAS / "correccion_base.png")
    figura_calendario(tabla_cal, CARPETA_FIGURAS / "calendario.png")
    figura_distribucion(con_fgfc, CARPETA_FIGURAS / "distribucion.png")
    print("Escritas en", CARPETA_FIGURAS)

    with open(CARPETA / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2, default=float)
    print("Resultados en", CARPETA / "resultados.json")


if __name__ == "__main__":
    main()
