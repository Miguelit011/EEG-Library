"""
Fatiga general (FG) y fatiga cognitiva (FC) frente al EEG — paso 2: entre
personas. ¿Tienen otro EEG habitual los alumnos que suelen estar más
fatigados, una vez descontados su sexo, su curso y su centro?

Una fila por persona: y = media de sus respuestas FG (o FC) de M1–M6 centradas
por medición; X = media de las 70 features espectrales de sus sesiones EEG,
centradas por medición. Dentro de cada pliegue se le quita a y lo que explican
sexo, curso y centro, y el modelo predice ese residuo.

- Principal: Random Forest (500 árboles, hoja ≥ 5), ρ de Spearman fuera de
  pliegue, IC por bootstrap de personas y p unilateral por permutación de
  personas dentro de centro (todo el proceso se repite en cada permutación).
- Secundario: ridge con α por validación interna.
- A priori: índice θ/α y compuesto `_rel_sd`, correlación parcial de Spearman.
- Referencia: el VAS por el mismo camino, diferencia emparejada.
- Sensibilidades: sin ajuste por covariables y semillas 1–4.

Plan, estadístico principal y reglas de lectura: `INFORME.md`, escritos antes
de ejecutar este script. **Solo `train` + `val`**: el test se aparta al leer la
matriz, antes de agregar. Exploratorio: el test está gastado.

Ejecutar (desde la raíz del repositorio):
    python analisis/fatiga_fg_fc/02_entre_personas/entre_personas.py > analisis/fatiga_fg_fc/02_entre_personas/ejecucion.log
    python analisis/fatiga_fg_fc/02_entre_personas/entre_personas.py --rapido   # depuración: pocas réplicas, salida temporal

La ubicación del dataset se toma de la variable de entorno `EEG_DATA`, con
`EEG-Data` junto al repositorio por defecto (ver `analisis/README.md`).
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_library.modelado.modeling import (
    agregar_a_sesion,
    columnas_de_features_sesion,
    compuesto_rel_sd,
)
from eeg_library.modelado.reposo import COLUMNAS_PROCESAMIENTO

CARPETA = Path(__file__).resolve().parent
LINEA = CARPETA.parent
sys.path.insert(0, str(LINEA))
import datos_fg_fc as dfg  # noqa: E402

RAPIDO = "--rapido" in sys.argv
SEMILLA = 42
SEMILLAS_ESTABILIDAD = [1, 2, 3, 4]
N_PLIEGUES = 5
N_BOOTSTRAP = 100 if RAPIDO else 2000
N_PERMUTACIONES = 20 if RAPIDO else 1000
N_PERMUTACIONES_APRIORI = 200 if RAPIDO else 10000

RUTA_MATRIZ = dfg.EEG_DATA / "processed" / "matriz_rf.parquet"
SALIDA = Path(tempfile.gettempdir()) / "fatiga_fg_fc_paso2_rapido" if RAPIDO else CARPETA
CARPETA_TABLAS = SALIDA / "tablas"
CARPETA_FIGURAS = SALIDA / "figuras"

MEDIDAS = ["FG", "FC"]
OBJETIVOS = ["FG", "FC", "VAS"]
COVARIABLES = ["sexo", "curso", "centro"]
APRIORI = {"indice_theta_alfa": "indice_fatiga_media", "compuesto_rel_sd": "compuesto_rel_sd"}

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


def bosque(semilla=SEMILLA, n_jobs=-1):
    # Con random_state fijo, el bosque es idéntico con cualquier n_jobs.
    return RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                                 random_state=semilla, n_jobs=n_jobs)


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         RidgeCV(alphas=np.logspace(-2, 4, 25)))


# --- Datos ---------------------------------------------------------------------

def cargar():
    """
    Respuestas (FG, FC y VAS por persona y medición), X por sesión y personas.

    El test se aparta antes de agregar la matriz a sesión. Solo entran las
    sesiones de train+val con FG y FC (paso 1) y sus personas.
    """
    _, respuestas, personas, sesiones, enlace = dfg.cargar_enlace()
    tv = ["train", "val"]
    sesiones = sesiones[sesiones.split.isin(tv) & sesiones.con_fgfc]
    ids = sorted(sesiones.persona_uid.unique())

    resp = respuestas[respuestas.split.isin(tv) & respuestas.persona_uid.isin(ids)].copy()
    resp["VAS"] = resp["vas_bd"]

    matriz = pd.read_parquet(RUTA_MATRIZ)
    n_test = int((matriz.split == "test").sum())
    matriz = matriz[matriz.split.isin(tv)]
    por_sesion = agregar_a_sesion(matriz)
    faltan = sorted(set(sesiones.sesion_uid) - set(por_sesion.sesion_uid))
    if faltan:
        raise ValueError(f"Sesiones con FG/FC sin fila en la matriz: {faltan[:5]}…")
    por_sesion = por_sesion[por_sesion.sesion_uid.isin(sesiones.sesion_uid)].reset_index(drop=True)
    excluidas = set(COLUMNAS_PROCESAMIENTO) | {"n_grabaciones"}
    columnas = [c for c in columnas_de_features_sesion(por_sesion) if c not in excluidas]
    if len(columnas) != 70:
        raise ValueError(f"Se esperaban 70 features espectrales y hay {len(columnas)}.")
    por_sesion["compuesto_rel_sd"] = compuesto_rel_sd(por_sesion, por_sesion)

    pers = personas.loc[ids, COVARIABLES]
    if pers.isna().any().any():
        raise ValueError(f"Personas sin sexo, curso o centro: {list(pers[pers.isna().any(axis=1)].index)}.")
    resumen = {**{k: enlace[k] for k in ("base", "sha256_base")},
               "matriz": RUTA_MATRIZ.name, "sha256_matriz": dfg.sha256(RUTA_MATRIZ),
               "grabaciones_test_apartadas": n_test, "personas": len(ids),
               "sesiones": int(len(por_sesion)), "respuestas": int(len(resp)),
               "respuestas_por_objetivo": {o: int(resp[o].notna().sum()) for o in OBJETIVOS},
               "features": len(columnas)}
    return resp, por_sesion, columnas, pers, resumen


def matriz_covariables(pers):
    """Diseño de las covariables (one-hot con la primera categoría de referencia)."""
    return pd.get_dummies(pers[COVARIABLES].astype(str), drop_first=True).astype(float)


# --- Construcción de y y X dentro de un pliegue --------------------------------

def y_por_persona(resp, objetivo, ids, ids_ajuste):
    """Media por persona de `objetivo` centrado por medición (medias de `ids_ajuste`)."""
    r = resp.dropna(subset=[objetivo])
    centrado = dfg.centrar_por_medicion(r, objetivo, r.persona_uid.isin(ids_ajuste))
    y = centrado.groupby(r.persona_uid).mean().reindex(ids)
    if y.isna().any():
        raise ValueError(f"Personas sin ninguna respuesta de {objetivo}: {list(y[y.isna()].index)}.")
    return y


def x_por_persona(por_sesion, columnas, ids, ids_ajuste):
    """Media por persona de las features, centradas por medición (medias de `ids_ajuste`)."""
    ajuste = por_sesion.persona_uid.isin(ids_ajuste)
    medias = por_sesion[ajuste].groupby("medicion")[columnas].mean()
    centrado = por_sesion[columnas] - medias.reindex(por_sesion.medicion).to_numpy()
    return centrado.groupby(por_sesion.persona_uid).mean().reindex(ids)


def residualizar(y, C, entrenamiento):
    """y menos su predicción lineal por covariables ajustada en `entrenamiento`."""
    D = np.column_stack([np.ones(len(y)), C])
    beta, *_ = np.linalg.lstsq(D[entrenamiento], y[entrenamiento], rcond=None)
    return y - D @ beta


def pliegues(pers, semilla):
    cv = StratifiedKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=semilla)
    return list(cv.split(np.zeros(len(pers)), pers["centro"].to_numpy()))


def evaluar(resp, objetivo, X_pliegues, C, ids, cortes, modelo, ajustar=True):
    """
    Predicciones fuera de pliegue del y residual de cada persona.

    `X_pliegues[i]` es X construida con las personas de entrenamiento del
    pliegue i (no depende de y, así que se calcula una vez y se reutiliza en
    las permutaciones).
    """
    pred = np.full(len(ids), np.nan)
    obs = np.full(len(ids), np.nan)
    for i, (tr, te) in enumerate(cortes):
        ids_tr = [ids[j] for j in tr]
        y = y_por_persona(resp, objetivo, ids, ids_tr).to_numpy(float)
        entrenamiento = np.zeros(len(ids), bool)
        entrenamiento[tr] = True
        if ajustar:
            y = residualizar(y, C, entrenamiento)
        else:
            y = y - y[tr].mean()
        ajustado = clone(modelo).fit(X_pliegues[i].iloc[tr], y[tr])
        pred[te] = ajustado.predict(X_pliegues[i].iloc[te])
        obs[te] = y[te]
    return pred, obs


def preparar_x(por_sesion, columnas, ids, cortes):
    return [x_por_persona(por_sesion, columnas, ids, [ids[j] for j in tr]) for tr, _ in cortes]


# --- Estadística ---------------------------------------------------------------

def rho(a, b):
    return float(stats.spearmanr(a, b)[0])


def r2_frente_a_cero(obs, pred):
    """R² frente a predecir la media del entrenamiento (0 para el y residual)."""
    return float(1 - ((obs - pred) ** 2).sum() / (obs ** 2).sum())


def bootstrap_rho(pares, n_rep=N_BOOTSTRAP, semilla=SEMILLA):
    """
    IC95 por bootstrap de personas de ρ (o de la diferencia de dos ρ con las
    mismas personas). `pares` es una lista de (obs, pred); con dos, se da la
    diferencia primero − segundo.
    """
    rng = np.random.default_rng(semilla)
    n = len(pares[0][0])
    reps = []
    for _ in range(n_rep):
        idx = rng.integers(0, n, n)
        valores = [rho(o[idx], p[idx]) for o, p in pares]
        reps.append(valores[0] if len(valores) == 1 else valores[0] - valores[1])
    return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


def rho_permutado(resp, mapa, X_pliegues, C, ids, cortes):
    """ρ del Random Forest con las respuestas reasignadas según `mapa`, FG y FC."""
    resp_perm = resp.assign(persona_uid=resp.persona_uid.map(mapa))
    salida = {}
    for medida in MEDIDAS:
        pred, obs = evaluar(resp_perm, medida, X_pliegues, C, ids, cortes, bosque(n_jobs=1))
        salida[medida] = rho(obs, pred)
    return salida


def permutar_dentro_de_centro(ids, centros, rng):
    """Reasignación aleatoria de personas dentro de cada centro: {original: nueva}."""
    mapa = {}
    for c in np.unique(centros):
        grupo = [ids[i] for i in np.flatnonzero(centros == c)]
        for a, b in zip(grupo, rng.permutation(grupo)):
            mapa[a] = b
    return mapa


def rho_parcial(x, y, C):
    """ρ de Spearman parcial: Pearson entre los residuos de los rangos sobre C."""
    D = np.column_stack([np.ones(len(x)), C])
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    ex = rx - D @ np.linalg.lstsq(D, rx, rcond=None)[0]
    ey = ry - D @ np.linalg.lstsq(D, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


# --- Figuras -------------------------------------------------------------------

def _estilo(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(TINTA_2)
    ax.tick_params(colors=TINTA_2, labelsize=8)
    ax.xaxis.grid(True, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)


def figura_rho(tabla, ruta):
    variantes = [("RF", "Random Forest (principal)", COLOR_A, "o"),
                 ("ridge", "ridge", COLOR_B, "s"),
                 ("RF sin ajuste", "Random Forest sin ajuste", TINTA_2, "D")]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y0 = np.arange(len(OBJETIVOS))[::-1]
    ax.axvline(0, color=TINTA, linewidth=1)
    for k, (clave, etiqueta, color, marcador) in enumerate(variantes):
        t = tabla[tabla.variante == clave].set_index("objetivo").reindex(OBJETIVOS)
        y = y0 + (1 - k) * 0.22
        ax.errorbar(t["rho"], y, xerr=[t["rho"] - t["ic95_inf"], t["ic95_sup"] - t["rho"]],
                    fmt=marcador, color=color, markersize=7, capsize=2.5, linewidth=1.8,
                    markeredgecolor="white", label=etiqueta)
    ax.set_yticks(y0, ["Fatiga general", "Fatiga cognitiva", "VAS (referencia)"], fontsize=9)
    ax.set_xlabel("ρ de Spearman, predicción fuera de pliegue frente a y (IC95)",
                  fontsize=8, color=TINTA_2)
    ax.set_title("¿Ordena el EEG a las personas según su fatiga habitual?", fontsize=10,
                 color=TINTA, loc="left")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_dispersion(oof, ruta):
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, medida in zip(ejes, MEDIDAS):
        obs, pred = oof[medida]
        ax.scatter(pred, obs, s=26, color=COLOR_A, edgecolor="white", linewidth=0.8, alpha=0.9)
        ax.axhline(0, color=TINTA_2, linewidth=0.8)
        ax.set_xlabel("predicción fuera de pliegue (Random Forest)", fontsize=8, color=TINTA_2)
        ax.set_ylabel("y residual (puntos, 0–24)", fontsize=8, color=TINTA_2)
        ax.set_title(f"{'Fatiga general' if medida == 'FG' else 'Fatiga cognitiva'}: "
                     f"ρ = {rho(obs, pred):+.3f}", fontsize=10, color=TINTA, loc="left")
        _estilo(ax)
        ax.yaxis.grid(True, color=REJILLA, linewidth=0.8)
    fig.suptitle(f"Una persona por punto ({len(oof['FG'][0])} personas de train+val)",
                 fontsize=9, color=TINTA_2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_nula(nulas, observados, ruta):
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, medida in zip(ejes, MEDIDAS):
        ax.hist(nulas[medida], bins=30, color=COLOR_A, edgecolor="white", linewidth=1)
        ax.axvline(observados[medida], color=TINTA, linewidth=1.5, linestyle="--")
        ax.text(observados[medida], ax.get_ylim()[1] * 0.95, f"  observado {observados[medida]:+.3f}",
                fontsize=8, color=TINTA, va="top")
        ax.set_title({"FG": "Fatiga general", "FC": "Fatiga cognitiva"}[medida], fontsize=10,
                     color=TINTA, loc="left")
        ax.set_xlabel("ρ con las personas permutadas dentro de centro", fontsize=8, color=TINTA_2)
        _estilo(ax)
    ejes[0].set_ylabel("permutaciones", fontsize=8, color=TINTA_2)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# --- Principal -----------------------------------------------------------------

def main():
    inicio = time.time()
    CARPETA_TABLAS.mkdir(parents=True, exist_ok=True)
    CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)
    if RAPIDO:
        print(f"*** MODO RÁPIDO: réplicas mínimas, salida en {SALIDA}. Sus cifras NO son resultados. ***")
    resultados = {"parametros": {"semilla": SEMILLA, "pliegues": N_PLIEGUES, "n_bootstrap": N_BOOTSTRAP,
                                 "n_permutaciones": N_PERMUTACIONES,
                                 "n_permutaciones_apriori": N_PERMUTACIONES_APRIORI,
                                 "semillas_estabilidad": SEMILLAS_ESTABILIDAD, "rapido": RAPIDO}}

    _titulo("1. DATOS")
    resp, por_sesion, columnas, pers, resumen = cargar()
    resultados["datos"] = resumen
    ids = list(pers.index)
    C = matriz_covariables(pers).to_numpy()
    centros = pers["centro"].to_numpy()
    print(f"Base: {resumen['base']} (SHA-256 {resumen['sha256_base']})")
    print(f"Matriz: {resumen['matriz']} (SHA-256 {resumen['sha256_matriz']}); "
          f"{resumen['grabaciones_test_apartadas']} grabaciones de test apartadas antes de agregar")
    print(f"Personas: {resumen['personas']}; sesiones EEG: {resumen['sesiones']}; "
          f"features: {resumen['features']}")
    print(f"Respuestas M1–M6 por objetivo: {resumen['respuestas_por_objetivo']}")
    print("Covariables:", {c: pers[c].value_counts().to_dict() for c in COVARIABLES})

    cortes = pliegues(pers, SEMILLA)
    X_pliegues = preparar_x(por_sesion, columnas, ids, cortes)
    print("Personas por pliegue de evaluación:", [len(te) for _, te in cortes])

    _titulo("2. PRINCIPAL: RANDOM FOREST, ρ FUERA DE PLIEGUE")
    oof, filas = {}, []
    for objetivo in OBJETIVOS:
        for variante, modelo, ajustar in (("RF", bosque(), True), ("ridge", ridge(), True),
                                          ("RF sin ajuste", bosque(), False)):
            pred, obs = evaluar(resp, objetivo, X_pliegues, C, ids, cortes, modelo, ajustar)
            if variante == "RF":
                oof[objetivo] = (obs, pred)
            r = rho(obs, pred)
            ic = bootstrap_rho([(obs, pred)])
            filas.append({"objetivo": objetivo, "variante": variante, "rho": r,
                          "ic95_inf": ic[0], "ic95_sup": ic[1], "r2": r2_frente_a_cero(obs, pred),
                          "n": int(np.isfinite(pred).sum())})
            print(f"{objetivo:3s} {variante:14s} ρ = {r:+.3f} [{ic[0]:+.3f}, {ic[1]:+.3f}]  "
                  f"R² = {filas[-1]['r2']:+.3f}")
    tabla = pd.DataFrame(filas)

    _titulo("3. PERMUTACIONES (Random Forest, personas dentro de centro)")
    # Las reasignaciones se generan antes y en orden con la misma semilla, y las
    # permutaciones se reparten entre núcleos: el resultado no depende del reparto.
    rng = np.random.default_rng(SEMILLA)
    mapas = [permutar_dentro_de_centro(ids, centros, rng) for _ in range(N_PERMUTACIONES)]
    t0 = time.time()
    salidas = Parallel(n_jobs=-1)(delayed(rho_permutado)(resp, mapa, X_pliegues, C, ids, cortes)
                                  for mapa in mapas)
    nulas = {m: [s[m] for s in salidas] for m in MEDIDAS}
    print(f"  {N_PERMUTACIONES} permutaciones en {time.time() - t0:.0f} s")
    principal = {}
    for medida in MEDIDAS:
        observado = rho(*oof[medida])
        nula = np.array(nulas[medida])
        p = (1 + int((nula >= observado).sum())) / (N_PERMUTACIONES + 1)
        principal[medida] = {"rho": observado, "p_unilateral": p, "nula_media": float(nula.mean()),
                             "nula_p95": float(np.percentile(nula, 95))}
        tabla.loc[(tabla.objetivo == medida) & (tabla.variante == "RF"), "p_unilateral"] = p
        print(f"{medida}: ρ = {observado:+.3f}; nula media {nula.mean():+.3f}, P95 "
              f"{np.percentile(nula, 95):+.3f}; p unilateral = {p:.4f} (sin corregir)")
    resultados["principal"] = principal
    tabla.to_csv(CARPETA_TABLAS / "rho_por_variante.csv", index=False, float_format="%.4f")
    resultados["por_variante"] = tabla.round(4).to_dict(orient="records")

    _titulo("4. REFERENCIA VAS: DIFERENCIA EMPAREJADA (Random Forest)")
    filas = []
    for medida in MEDIDAS:
        d = rho(*oof[medida]) - rho(*oof["VAS"])
        ic = bootstrap_rho([oof[medida], oof["VAS"]])
        filas.append({"comparacion": f"{medida} − VAS", "diferencia": d, "ic95_inf": ic[0], "ic95_sup": ic[1]})
        print(f"ρ({medida}) − ρ(VAS) = {d:+.3f} [{ic[0]:+.3f}, {ic[1]:+.3f}]")
    tabla_vas = pd.DataFrame(filas)
    tabla_vas.to_csv(CARPETA_TABLAS / "diferencia_con_vas.csv", index=False, float_format="%.4f")
    resultados["diferencia_con_vas"] = tabla_vas.round(4).to_dict(orient="records")

    _titulo("5. HIPÓTESIS A PRIORI: CORRELACIÓN PARCIAL (sexo, curso, centro)")
    ids_todos = ids
    x_apriori = x_por_persona(por_sesion, list(APRIORI.values()), ids, ids_todos)
    rng = np.random.default_rng(SEMILLA)
    filas = []
    for medida in MEDIDAS:
        y = y_por_persona(resp, medida, ids, ids_todos).to_numpy(float)
        for nombre, columna in APRIORI.items():
            x = x_apriori[columna].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            r = rho_parcial(x[ok], y[ok], C[ok])
            nula = []
            for _ in range(N_PERMUTACIONES_APRIORI):
                perm = np.arange(len(y))
                for c in np.unique(centros):
                    idx = np.flatnonzero(centros == c)
                    perm[idx] = rng.permutation(idx)
                yp = y[perm]
                okp = np.isfinite(x) & np.isfinite(yp)
                nula.append(rho_parcial(x[okp], yp[okp], C[okp]))
            p = (1 + int((np.array(nula) >= r).sum())) / (N_PERMUTACIONES_APRIORI + 1)
            brng = np.random.default_rng(SEMILLA)
            reps = []
            sel = np.flatnonzero(ok)
            for _ in range(N_BOOTSTRAP):
                b = brng.choice(sel, len(sel))
                reps.append(rho_parcial(x[b], y[b], C[b]))
            filas.append({"medida": medida, "variable": nombre, "rho_parcial": r,
                          "ic95_inf": float(np.percentile(reps, 2.5)),
                          "ic95_sup": float(np.percentile(reps, 97.5)),
                          "p_unilateral": p, "n": int(ok.sum())})
            print(f"{medida} ~ {nombre:18s} ρ parcial = {r:+.3f} [{filas[-1]['ic95_inf']:+.3f}, "
                  f"{filas[-1]['ic95_sup']:+.3f}]; p unilateral = {p:.4f} (sin corregir); n = {ok.sum()}")
    tabla_apriori = pd.DataFrame(filas)
    tabla_apriori.to_csv(CARPETA_TABLAS / "apriori.csv", index=False, float_format="%.4f")
    resultados["apriori"] = tabla_apriori.round(4).to_dict(orient="records")

    _titulo("6. ESTABILIDAD: SEMILLAS 1–4 (reparto en pliegues y bosque)")
    filas = []
    for s in [SEMILLA] + SEMILLAS_ESTABILIDAD:
        cortes_s = pliegues(pers, s)
        X_s = X_pliegues if s == SEMILLA else preparar_x(por_sesion, columnas, ids, cortes_s)
        for medida in MEDIDAS:
            pred, obs = evaluar(resp, medida, X_s, C, ids, cortes_s, bosque(s))
            filas.append({"semilla": s, "medida": medida, "rho": rho(obs, pred)})
    tabla_sem = pd.DataFrame(filas)
    print(tabla_sem.pivot(index="semilla", columns="medida", values="rho").round(3).to_string())
    print("Media (SD):", {m: f"{g.rho.mean():+.3f} ({g.rho.std():.3f})"
                          for m, g in tabla_sem.groupby("medida")})
    tabla_sem.to_csv(CARPETA_TABLAS / "estabilidad_semillas.csv", index=False, float_format="%.4f")
    resultados["estabilidad"] = tabla_sem.round(4).to_dict(orient="records")

    _titulo("7. FIGURAS")
    figura_rho(tabla, CARPETA_FIGURAS / "rho_por_variante.png")
    figura_dispersion(oof, CARPETA_FIGURAS / "prediccion_frente_a_observado.png")
    figura_nula(nulas, {m: principal[m]["rho"] for m in MEDIDAS},
                CARPETA_FIGURAS / "distribucion_nula.png")
    print("Escritas en", CARPETA_FIGURAS)

    resultados["duracion_s"] = round(time.time() - inicio, 1)
    with open(SALIDA / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2, default=float)
    print(f"Resultados en {SALIDA / 'resultados.json'} ({resultados['duracion_s']:.0f} s)")


if __name__ == "__main__":
    main()
