"""
Fatiga general (FG) y fatiga cognitiva (FC) frente al EEG — paso 3: intra
persona. Cuando un alumno está más fatigado de lo habitual en él, ¿se aparta
también su EEG de su EEG habitual? Cierra las familias de pruebas con Holm.

Una fila por sesión (personas con ≥ 2 sesiones). y = FG (o FC) de la sesión
centrado por medición y menos la media leave-one-out de las otras respuestas
M1–M6 de la persona; X = las 70 features espectrales centradas por medición y
menos la media leave-one-out de las otras sesiones EEG de la persona
(`centrar_por_persona_loo`, Fase 14). Las medias por medición salen de las
personas del pliegue de entrenamiento.

- Principal: Random Forest (500 árboles, hoja ≥ 5), GroupKFold(5) por persona,
  ρ de Spearman fuera de pliegue, IC por bootstrap de personas y p unilateral
  por permutación dentro de persona (todo el proceso se repite).
- Secundario: ridge. A priori: θ/α y `_rel_sd`. Referencia: el VAS.
- Sensibilidades: control de visita y semillas 1–4 del bosque.
- Holm sobre las familias completas (con las p del paso 2).

Plan, estadístico principal y reglas de lectura: `INFORME.md`, escritos antes
de ejecutar este script. **Solo `train` + `val`**: el test se aparta al leer la
matriz, antes de agregar. Exploratorio: el test está gastado.

Ejecutar (desde la raíz del repositorio):
    python analisis/fatiga_fg_fc/03_intra_persona/intra_persona.py > analisis/fatiga_fg_fc/03_intra_persona/ejecucion.log
    python analisis/fatiga_fg_fc/03_intra_persona/intra_persona.py --rapido   # depuración: pocas réplicas, salida temporal

La ubicación del dataset se toma de la variable de entorno `EEG_DATA`, con
`EEG-Data` junto al repositorio por defecto (ver `analisis/README.md`).
"""

import importlib.util
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
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from eeg_library.modelado.modeling import (
    centrar_por_persona_loo,
    medias_loo_por_grupo,
    residualizar_por_visita,
)

CARPETA = Path(__file__).resolve().parent
LINEA = CARPETA.parent
sys.path.insert(0, str(LINEA))
import datos_fg_fc as dfg  # noqa: E402

# Carga, modelos y utilidades del paso 2 (mismo X, mismas personas de partida).
_spec = importlib.util.spec_from_file_location(
    "entre_personas", LINEA / "02_entre_personas" / "entre_personas.py")
paso2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paso2)

RAPIDO = "--rapido" in sys.argv
SEMILLA = 42
SEMILLAS_ESTABILIDAD = [1, 2, 3, 4]
N_PLIEGUES = 5
N_BOOTSTRAP = 100 if RAPIDO else 2000
N_PERMUTACIONES = 20 if RAPIDO else 1000
N_PERMUTACIONES_APRIORI = 200 if RAPIDO else 10000
RUTA_RESULTADOS_PASO2 = LINEA / "02_entre_personas" / "resultados.json"

SALIDA = Path(tempfile.gettempdir()) / "fatiga_fg_fc_paso3_rapido" if RAPIDO else CARPETA
CARPETA_TABLAS = SALIDA / "tablas"
CARPETA_FIGURAS = SALIDA / "figuras"

MEDIDAS = ["FG", "FC"]
OBJETIVOS = ["FG", "FC", "VAS"]
APRIORI = paso2.APRIORI
ALFA = 0.05

COLOR_A, COLOR_B = paso2.COLOR_A, paso2.COLOR_B
TINTA, TINTA_2, REJILLA = paso2.TINTA, paso2.TINTA_2, paso2.REJILLA
rho = paso2.rho
r2_frente_a_cero = paso2.r2_frente_a_cero
bosque = paso2.bosque
ridge = paso2.ridge


def _titulo(texto):
    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


# --- Datos ---------------------------------------------------------------------

def cargar():
    """Sesiones de personas con ≥ 2 sesiones EEG y sus respuestas M1–M6."""
    resp, por_sesion, columnas, _, resumen = paso2.cargar()
    n = por_sesion.groupby("persona_uid").size()
    ids = sorted(n[n >= 2].index)
    ses = por_sesion[por_sesion.persona_uid.isin(ids)].sort_values(
        ["persona_uid", "medicion"]).reset_index(drop=True)
    resp = resp[resp.persona_uid.isin(ids)].reset_index(drop=True)
    for o in OBJETIVOS:
        clave = ses[["persona_uid", "medicion"]].merge(
            resp[["persona_uid", "medicion", o]], how="left")[o]
        if clave.isna().any():
            raise ValueError(f"Sesiones sin respuesta de {o}.")
    resumen = {**resumen, "personas": len(ids), "sesiones": int(len(ses)),
               "personas_excluidas_una_sesion": int((n < 2).sum()),
               "respuestas": int(len(resp))}
    return resp, ses, columnas, resumen


# --- y y X de cada sesión dentro de un pliegue ---------------------------------

def y_por_sesion(resp, objetivo, ses, ids_ajuste):
    """
    Desviación de cada sesión respecto al nivel habitual de la persona.

    Centrado por medición (medias de `ids_ajuste`) y menos la media
    leave-one-out de las otras respuestas M1–M6 de la persona.
    """
    r = resp.dropna(subset=[objetivo])
    c = dfg.centrar_por_medicion(r, objetivo, r.persona_uid.isin(ids_ajuste))
    desviacion = c - medias_loo_por_grupo(c, r.persona_uid)
    tabla = pd.DataFrame({"persona_uid": r.persona_uid, "medicion": r.medicion, "y": desviacion})
    y = ses[["persona_uid", "medicion"]].merge(tabla, how="left")["y"].to_numpy(float)
    if np.isnan(y).any():
        raise ValueError(f"Sesiones sin desviación de {objetivo}.")
    return y


def x_por_sesion(ses, columnas, ids_ajuste):
    """Features centradas por medición (medias de `ids_ajuste`) y por persona (LOO)."""
    ajuste = ses.persona_uid.isin(ids_ajuste)
    medias = ses[ajuste].groupby("medicion")[columnas].mean()
    centrado = ses[["persona_uid"]].copy()
    centrado[columnas] = ses[columnas] - medias.reindex(ses.medicion).to_numpy()
    return centrado_persona(centrado, columnas)


def centrado_persona(datos, columnas):
    return centrar_por_persona_loo(datos, columnas)[columnas].reset_index(drop=True)


def cortes_por_persona(ses):
    return list(GroupKFold(n_splits=N_PLIEGUES).split(ses, groups=ses.persona_uid))


def con_visita(X, y, ses):
    """Quita la recta por persona sobre el número de visita, en X y en y."""
    datos = X.copy()
    datos["__y"] = y
    r = residualizar_por_visita(datos, list(datos.columns), ses.medicion, ses.persona_uid)
    return r.drop(columns="__y"), r["__y"].to_numpy(float)


def preparar_x(ses, columnas, cortes):
    salida = []
    for tr, _ in cortes:
        ids_tr = set(ses.persona_uid.iloc[tr])
        salida.append(x_por_sesion(ses, columnas, ids_tr))
    return salida


def evaluar(resp, objetivo, ses, X_pliegues, cortes, modelo, visita=False):
    """Predicciones fuera de pliegue de la desviación de cada sesión."""
    pred = np.full(len(ses), np.nan)
    obs = np.full(len(ses), np.nan)
    for i, (tr, te) in enumerate(cortes):
        ids_tr = set(ses.persona_uid.iloc[tr])
        y = y_por_sesion(resp, objetivo, ses, ids_tr)
        X = X_pliegues[i]
        if visita:
            X, y = con_visita(X, y, ses)
        ajustado = clone(modelo).fit(X.iloc[tr], y[tr])
        pred[te] = ajustado.predict(X.iloc[te])
        obs[te] = y[te]
    return pred, obs


# --- Permutaciones dentro de persona -------------------------------------------

def permutacion_dentro_de_persona(resp, ses, rng):
    """
    Índices de `resp` con las respuestas de las sesiones EEG de cada persona
    barajadas entre esas sesiones. Las respuestas sin EEG no se mueven.
    """
    con_eeg = resp.merge(ses[["persona_uid", "medicion"]], how="left", indicator=True)["_merge"] == "both"
    orden = np.arange(len(resp))
    for _, idx in resp[con_eeg.to_numpy()].groupby("persona_uid").indices.items():
        filas = np.flatnonzero(con_eeg.to_numpy())[idx]
        orden[filas] = rng.permutation(filas)
    return orden


def rho_permutado(resp, orden, ses, X_pliegues, cortes):
    """ρ del Random Forest con FG y FC reasignados según `orden` (mismo para las dos)."""
    resp_perm = resp.copy()
    for medida in MEDIDAS:
        resp_perm[medida] = resp[medida].to_numpy()[orden]
    return {m: rho(*evaluar(resp_perm, m, ses, X_pliegues, cortes, bosque(n_jobs=1)))
            for m in MEDIDAS}


def permutar_y_dentro(y, grupos, rng):
    salida = y.copy()
    for idx in pd.Series(np.arange(len(y))).groupby(np.asarray(grupos)).indices.values():
        salida[idx] = y[rng.permutation(idx)]
    return salida


# --- Bootstrap por personas ----------------------------------------------------

def bootstrap_personas(grupos, estadistico, n_rep=N_BOOTSTRAP, semilla=SEMILLA):
    """IC95 percentil de `estadistico(idx)` remuestreando personas con sus sesiones."""
    rng = np.random.default_rng(semilla)
    bloques = list(pd.Series(np.arange(len(grupos))).groupby(np.asarray(grupos)).indices.values())
    reps = []
    for _ in range(n_rep):
        idx = np.concatenate([bloques[i] for i in rng.integers(0, len(bloques), len(bloques))])
        reps.append(estadistico(idx))
    return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


def holm(pvalores):
    """p ajustadas por Holm, en el orden de entrada."""
    p = np.asarray(pvalores, float)
    orden = np.argsort(p)
    m = len(p)
    ajustadas = np.empty(m)
    maximo = 0.0
    for rango, i in enumerate(orden):
        maximo = max(maximo, (m - rango) * p[i])
        ajustadas[i] = min(1.0, maximo)
    return ajustadas


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
                 ("RF con control de visita", "Random Forest con control de visita", TINTA_2, "D")]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y0 = np.arange(len(OBJETIVOS))[::-1]
    ax.axvline(0, color=TINTA, linewidth=1)
    for k, (clave, etiqueta, color, marcador) in enumerate(variantes):
        t = tabla[tabla.variante == clave].set_index("objetivo").reindex(OBJETIVOS)
        ax.errorbar(t["rho"], y0 + (1 - k) * 0.22,
                    xerr=[t["rho"] - t["ic95_inf"], t["ic95_sup"] - t["rho"]],
                    fmt=marcador, color=color, markersize=7, capsize=2.5, linewidth=1.8,
                    markeredgecolor="white", label=etiqueta)
    ax.set_yticks(y0, ["Fatiga general", "Fatiga cognitiva", "VAS (referencia)"], fontsize=9)
    ax.set_xlabel("ρ de Spearman, predicción fuera de pliegue frente a la desviación (IC95)",
                  fontsize=8, color=TINTA_2)
    ax.set_title("¿Sigue el EEG los cambios de fatiga de la misma persona?", fontsize=10,
                 color=TINTA, loc="left")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
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
        ax.set_xlabel("ρ con las respuestas permutadas dentro de persona", fontsize=8, color=TINTA_2)
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
    resp, ses, columnas, resumen = cargar()
    resultados["datos"] = resumen
    grupos = ses.persona_uid.to_numpy()
    print(f"Base: {resumen['base']} (SHA-256 {resumen['sha256_base']})")
    print(f"Matriz: {resumen['matriz']} (SHA-256 {resumen['sha256_matriz']}); "
          f"{resumen['grabaciones_test_apartadas']} grabaciones de test apartadas antes de agregar")
    print(f"Personas con ≥ 2 sesiones: {resumen['personas']} (fuera {resumen['personas_excluidas_una_sesion']} "
          f"con una sola); sesiones: {resumen['sesiones']}; respuestas M1–M6: {resumen['respuestas']}; "
          f"features: {len(columnas)}")
    print("Sesiones por medición:", ses.medicion.value_counts().sort_index().to_dict())

    cortes = cortes_por_persona(ses)
    X_pliegues = preparar_x(ses, columnas, cortes)
    print("Personas / sesiones por pliegue de evaluación:",
          [(len(set(grupos[te])), len(te)) for _, te in cortes])

    _titulo("2. PRINCIPAL: RANDOM FOREST, ρ FUERA DE PLIEGUE")
    oof, filas = {}, []
    for objetivo in OBJETIVOS:
        for variante, modelo, visita in (("RF", bosque(), False), ("ridge", ridge(), False),
                                         ("RF con control de visita", bosque(), True)):
            pred, obs = evaluar(resp, objetivo, ses, X_pliegues, cortes, modelo, visita)
            if variante == "RF":
                oof[objetivo] = (obs, pred)
            r = rho(obs, pred)
            ic = bootstrap_personas(grupos, lambda i, o=obs, p=pred: rho(o[i], p[i]))
            filas.append({"objetivo": objetivo, "variante": variante, "rho": r, "ic95_inf": ic[0],
                          "ic95_sup": ic[1], "r2": r2_frente_a_cero(obs, pred), "n": int(len(obs))})
            print(f"{objetivo:3s} {variante:25s} ρ = {r:+.3f} [{ic[0]:+.3f}, {ic[1]:+.3f}]  "
                  f"R² = {filas[-1]['r2']:+.3f}")
    tabla = pd.DataFrame(filas)

    _titulo("3. PERMUTACIONES (Random Forest, respuestas dentro de persona)")
    rng = np.random.default_rng(SEMILLA)
    ordenes = [permutacion_dentro_de_persona(resp, ses, rng) for _ in range(N_PERMUTACIONES)]
    t0 = time.time()
    salidas = Parallel(n_jobs=-1)(delayed(rho_permutado)(resp, o, ses, X_pliegues, cortes)
                                  for o in ordenes)
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
        (o1, p1), (o2, p2) = oof[medida], oof["VAS"]
        d = rho(o1, p1) - rho(o2, p2)
        ic = bootstrap_personas(grupos, lambda i: rho(o1[i], p1[i]) - rho(o2[i], p2[i]))
        filas.append({"comparacion": f"{medida} − VAS", "diferencia": d, "ic95_inf": ic[0], "ic95_sup": ic[1]})
        print(f"ρ({medida}) − ρ(VAS) = {d:+.3f} [{ic[0]:+.3f}, {ic[1]:+.3f}]")
    tabla_vas = pd.DataFrame(filas)
    tabla_vas.to_csv(CARPETA_TABLAS / "diferencia_con_vas.csv", index=False, float_format="%.4f")
    resultados["diferencia_con_vas"] = tabla_vas.round(4).to_dict(orient="records")

    _titulo("5. HIPÓTESIS A PRIORI: ρ INTRA-PERSONA")
    todas = set(grupos)
    x_apriori = x_por_sesion(ses, list(APRIORI.values()), todas)
    rng = np.random.default_rng(SEMILLA)
    filas = []
    for medida in MEDIDAS:
        y = y_por_sesion(resp, medida, ses, todas)
        for nombre, columna in APRIORI.items():
            x = x_apriori[columna].to_numpy(float)
            ok = np.isfinite(x)
            r = rho(x[ok], y[ok])
            nula = np.array([rho(x[ok], permutar_y_dentro(y[ok], grupos[ok], rng))
                             for _ in range(N_PERMUTACIONES_APRIORI)])
            p = (1 + int((nula >= r).sum())) / (N_PERMUTACIONES_APRIORI + 1)
            xo, yo, go = x[ok], y[ok], grupos[ok]
            ic = bootstrap_personas(go, lambda i: rho(xo[i], yo[i]))
            filas.append({"medida": medida, "variable": nombre, "rho": r, "ic95_inf": ic[0],
                          "ic95_sup": ic[1], "p_unilateral": p, "n": int(ok.sum())})
            print(f"{medida} ~ {nombre:18s} ρ = {r:+.3f} [{ic[0]:+.3f}, {ic[1]:+.3f}]; "
                  f"p unilateral = {p:.4f} (sin corregir); n = {ok.sum()}")
    tabla_apriori = pd.DataFrame(filas)
    tabla_apriori.to_csv(CARPETA_TABLAS / "apriori.csv", index=False, float_format="%.4f")
    resultados["apriori"] = tabla_apriori.round(4).to_dict(orient="records")

    _titulo("6. ESTABILIDAD: SEMILLAS 1–4 DEL BOSQUE")
    filas = []
    for s in [SEMILLA] + SEMILLAS_ESTABILIDAD:
        for medida in MEDIDAS:
            pred, obs = evaluar(resp, medida, ses, X_pliegues, cortes, bosque(s))
            filas.append({"semilla": s, "medida": medida, "rho": rho(obs, pred)})
    tabla_sem = pd.DataFrame(filas)
    print(tabla_sem.pivot(index="semilla", columns="medida", values="rho").round(3).to_string())
    print("Media (SD):", {m: f"{g.rho.mean():+.3f} ({g.rho.std():.3f})" for m, g in tabla_sem.groupby("medida")})
    tabla_sem.to_csv(CARPETA_TABLAS / "estabilidad_semillas.csv", index=False, float_format="%.4f")
    resultados["estabilidad"] = tabla_sem.round(4).to_dict(orient="records")

    _titulo("7. CIERRE DE LAS FAMILIAS (Holm)")
    with open(RUTA_RESULTADOS_PASO2, encoding="utf-8") as f:
        paso2_res = json.load(f)
    if paso2_res["parametros"]["rapido"]:
        raise ValueError("Los resultados del paso 2 son de una ejecución --rapido.")
    familias = []
    for medida in MEDIDAS:
        familias.append({"familia": "principal", "paso": 2, "prueba": f"{medida}, entre personas, RF",
                         "p": paso2_res["principal"][medida]["p_unilateral"],
                         "rho": paso2_res["principal"][medida]["rho"]})
    for medida in MEDIDAS:
        familias.append({"familia": "principal", "paso": 3, "prueba": f"{medida}, intra persona, RF",
                         "p": principal[medida]["p_unilateral"], "rho": principal[medida]["rho"]})
    for fila in paso2_res["apriori"]:
        familias.append({"familia": "secundaria", "paso": 2,
                         "prueba": f"{fila['medida']} ~ {fila['variable']}, entre personas",
                         "p": fila["p_unilateral"], "rho": fila["rho_parcial"]})
    for fila in tabla_apriori.to_dict(orient="records"):
        familias.append({"familia": "secundaria", "paso": 3,
                         "prueba": f"{fila['medida']} ~ {fila['variable']}, intra persona",
                         "p": fila["p_unilateral"], "rho": fila["rho"]})
    tabla_holm = pd.DataFrame(familias)
    tabla_holm["p_holm"] = np.nan
    for fam, g in tabla_holm.groupby("familia"):
        tabla_holm.loc[g.index, "p_holm"] = holm(g["p"].to_numpy())
    tabla_holm["significativa"] = tabla_holm["p_holm"] < ALFA
    print(tabla_holm.round(4).to_string(index=False))
    tabla_holm.to_csv(CARPETA_TABLAS / "holm_familias.csv", index=False, float_format="%.4f")
    resultados["holm"] = tabla_holm.round(4).to_dict(orient="records")

    _titulo("8. FIGURAS")
    figura_rho(tabla, CARPETA_FIGURAS / "rho_por_variante.png")
    figura_nula(nulas, {m: principal[m]["rho"] for m in MEDIDAS}, CARPETA_FIGURAS / "distribucion_nula.png")
    print("Escritas en", CARPETA_FIGURAS)

    resultados["duracion_s"] = round(time.time() - inicio, 1)
    with open(SALIDA / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2, default=float)
    print(f"Resultados en {SALIDA / 'resultados.json'} ({resultados['duracion_s']:.0f} s)")


if __name__ == "__main__":
    main()
