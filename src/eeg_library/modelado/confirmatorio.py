"""
Piezas del análisis confirmatorio sobre el conjunto de test.

El arnés de validación (`modeling`, `reposo`, `reposo_tarea`,
`tareas_ejecutivas`) evalúa con validación cruzada agrupada. El test es otra
cosa: un modelo final ajustado con train+val que predice **una sola vez** a 28
personas que no ha visto. Este módulo contiene lo que ese paso necesita y el
arnés no tenía, tal como lo fija `PREREGISTRO_TEST.md`:

- inferencia sobre predicciones fijas, sin reentrenar: p por permutación de la
  etiqueta real **dentro de cada persona** (definición convencional, que
  garantiza un error de tipo I <= alfa), réplicas bootstrap remuestreando
  personas, p unilateral a partir de esas réplicas y la corrección de Holm;
- la preparación de la Tarea B (persona conocida) con el control de la deriva
  por visita, idéntica en el conjunto de entrenamiento y en el evaluado;
- la importancia por permutación sobre datos no vistos (el modelo ya está
  ajustado: se barajan columnas del conjunto evaluado);
- la trazabilidad de la ejecución: una carpeta de salida que no se sobrescribe
  nunca, hashes de archivos y el estado de git.

Nada de aquí abre `test` por sí mismo: recibe los datos que le pasa el guion
(`analisis/analisis_test.py`), que es quien decide qué split evalúa.
"""

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_library.modelado.modeling import (
    COLUMNA_ETIQUETA,
    COLUMNA_GRUPO,
    centrar_por_persona_loo,
    residualizar_por_visita,
)


# --------------------------------------------------------------------------
# Métrica e inferencia sobre predicciones fijas
# --------------------------------------------------------------------------

def exactitud_balanceada_lote(y_true, y_pred, clases):
    """
    Exactitud balanceada de una o varias versiones de `y_true` frente a un
    mismo `y_pred`.

    `y_true` puede ser (n,) o (P, n): la forma (P, n) es la de P permutaciones,
    y se puntúan todas de una vez. Las clases de `clases` ausentes en una fila
    de `y_true` no cuentan en la media (como `balanced_accuracy_score`).

    Devuelve un array (P,), o (1,) si `y_true` es un vector.
    """
    y_true = np.atleast_2d(np.asarray(y_true))
    y_pred = np.asarray(y_pred)
    if y_true.shape[1] != len(y_pred):
        raise ValueError(
            f"y_true e y_pred deben tener el mismo número de filas: "
            f"{y_true.shape[1]} y {len(y_pred)}."
        )
    exhaustividades = []
    for c in clases:
        reales = y_true == c
        n_c = reales.sum(axis=1)
        aciertos = (reales & (y_pred == c)).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            exhaustividades.append(np.where(n_c > 0, aciertos / n_c, np.nan))
    return np.nanmean(np.vstack(exhaustividades), axis=0)


def permutaciones_dentro_de_grupo(grupos, n, rng):
    """
    Índices (n, filas): cada fila es una permutación que solo mueve valores
    entre filas del mismo grupo.
    """
    _, codigo = np.unique(np.asarray(grupos), return_inverse=True)
    clave = rng.random((n, len(codigo)))
    return np.lexsort((clave, np.broadcast_to(codigo, clave.shape)), axis=1)


def p_permutacion(y_true, y_pred, grupos, clases, n_permutaciones=10000,
                  semilla=42, lote=500):
    """
    p unilateral de la exactitud balanceada por permutación **dentro de cada
    persona**, con las predicciones fijas.

    Bajo la hipótesis nula las predicciones no dicen nada de la etiqueta real
    dentro de la persona, así que barajar `y_true` entre las filas de una misma
    persona no cambia la distribución de la métrica. Se conservan la persona y
    su reparto de clases; no se reentrena nada.

    Definición convencional: el observado cuenta como un valor más de la
    referencia y los empates cuentan enteros en contra,
    `p = (1 + #{nulo >= observado}) / (N + 1)`. Garantiza un error de tipo I
    <= alfa por construcción (PREREGISTRO_TEST.md, sección 3).

    Devuelve un dict con `observado`, `p`, `n_permutaciones` y `nulo` (el
    vector de las N exactitudes balanceadas permutadas).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    grupos = np.asarray(grupos)
    if not (len(y_true) == len(y_pred) == len(grupos)):
        raise ValueError(
            f"y_true, y_pred y grupos deben tener la misma longitud: "
            f"{len(y_true)}, {len(y_pred)}, {len(grupos)}."
        )
    rng = np.random.default_rng(semilla)
    observado = float(exactitud_balanceada_lote(y_true, y_pred, clases)[0])
    nulo = []
    hechas = 0
    while hechas < n_permutaciones:
        k = min(lote, n_permutaciones - hechas)
        idx = permutaciones_dentro_de_grupo(grupos, k, rng)
        nulo.append(exactitud_balanceada_lote(y_true[idx], y_pred, clases))
        hechas += k
    nulo = np.concatenate(nulo)
    p = (1 + np.sum(nulo >= observado)) / (n_permutaciones + 1)
    return {"observado": observado, "p": float(p),
            "n_permutaciones": int(n_permutaciones), "nulo": nulo}


def replicas_bootstrap(grupos, estadistico, n_replicas=10000, semilla=42):
    """
    Réplicas de un estadístico remuestreando **personas** con reposición.

    `estadistico(idx)` recibe los índices de fila de la remuestra (las filas de
    una persona elegida dos veces aparecen dos veces) y devuelve un número o un
    vector. Evaluar varios estadísticos en la misma llamada es lo que hace que
    sus diferencias sean **pareadas**: misma remuestra en todas las ramas.
    Una réplica en la que el estadístico no está definido (lanza `ValueError`)
    queda a NaN y no se descarta en silencio: se ve en el recuento.

    Devuelve un array (n_replicas,) o (n_replicas, k).
    """
    grupos = np.asarray(grupos)
    unicos = pd.unique(grupos)
    filas = {u: np.flatnonzero(grupos == u) for u in unicos}
    rng = np.random.default_rng(semilla)
    salida = []
    for _ in range(n_replicas):
        elegidas = rng.choice(unicos, size=len(unicos), replace=True)
        idx = np.concatenate([filas[u] for u in elegidas])
        try:
            salida.append(np.asarray(estadistico(idx), dtype=float))
        except ValueError:
            salida.append(None)
    forma = next((np.shape(v) for v in salida if v is not None), ())
    return np.array([np.full(forma, np.nan) if v is None else v for v in salida])


def ic_percentil(replicas, nivel=0.95):
    """IC de percentiles de unas réplicas (ignora NaN). Devuelve [inf, sup]."""
    r = np.asarray(replicas, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        raise ValueError("No hay réplicas válidas para el intervalo.")
    cola = 100 * (1 - nivel) / 2
    return [float(np.percentile(r, cola)), float(np.percentile(r, 100 - cola))]


def p_bootstrap(replicas, umbral, h0):
    """
    p unilateral a partir de réplicas bootstrap.

    `h0` dice de qué lado del `umbral` está la hipótesis nula:
    `"menor_o_igual"` (H1: el estadístico supera el umbral, p. ej. ρ > 0) o
    `"mayor_o_igual"` (H1: el estadístico queda por debajo, p. ej. un ΔR²
    por debajo del margen de equivalencia).
    `p = (#{réplicas en el lado de H0} + 1) / (B + 1)`, con B las réplicas
    válidas.
    """
    r = np.asarray(replicas, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        raise ValueError("No hay réplicas válidas para el p.")
    if h0 == "menor_o_igual":
        en_h0 = np.sum(r <= umbral)
    elif h0 == "mayor_o_igual":
        en_h0 = np.sum(r >= umbral)
    else:
        raise ValueError(f"`h0` debe ser 'menor_o_igual' o 'mayor_o_igual', no {h0!r}.")
    return float((en_h0 + 1) / (len(r) + 1))


def holm(p_valores):
    """
    p ajustados por Holm-Bonferroni, en el mismo orden de entrada.

    Se rechaza la hipótesis i a nivel alfa si su p ajustado es < alfa.
    """
    p = np.asarray(p_valores, dtype=float)
    if np.isnan(p).any():
        raise ValueError("Holm no admite p ausentes.")
    m = len(p)
    orden = np.argsort(p, kind="stable")
    ajustados = np.empty(m)
    acumulado = 0.0
    for rango, i in enumerate(orden):
        acumulado = max(acumulado, min(1.0, (m - rango) * p[i]))
        ajustados[i] = acumulado
    return ajustados


def veredicto_principal(p, ic_superior, referencia, alfa=0.05):
    """
    Veredicto de la prueba principal según PREREGISTRO_TEST.md, sección 4.1.

    - `confirma`: p < alfa;
    - `no confirma`: p >= alfa y el extremo superior del IC95 queda por debajo
      de la referencia de validación;
    - `no concluye`: cualquier otro caso.
    """
    if p < alfa:
        return "confirma"
    if ic_superior < referencia:
        return "no confirma"
    return "no concluye"


# --------------------------------------------------------------------------
# Preparación de los análisis intra-persona del VAS
# --------------------------------------------------------------------------

def preparar_tarea_b(sesiones, features, controlar_visita=True):
    """
    Tarea B (persona conocida): X e y centrados por la media LOO de la persona
    y, con `controlar_visita`, residualizados por la recta de visita dentro de
    cada persona.

    Es exactamente el paso de `analisis_control_visita.py` y de la Fase 14b, y
    se aplica igual a train+val (para entrenar) que al conjunto evaluado (con
    las medias y las rectas calculadas dentro de cada persona evaluada). Se
    descartan las sesiones sin referencia LOO (persona con una sola sesión).

    Devuelve un DataFrame con la identidad de la sesión, `features` y
    `COLUMNA_ETIQUETA` transformados, `vas_original` (el VAS sin tocar),
    `vas_loo` (la media LOO de la persona) y `desviacion_sin_control`.
    """
    centradas = centrar_por_persona_loo(sesiones, list(features) + [COLUMNA_ETIQUETA])
    validas = centradas[COLUMNA_ETIQUETA].notna()
    cen = centradas[validas].reset_index(drop=True)
    cen["vas_original"] = sesiones.loc[validas.values, COLUMNA_ETIQUETA].to_numpy()
    cen["desviacion_sin_control"] = cen[COLUMNA_ETIQUETA].to_numpy()
    cen = cen.rename(columns={f"{COLUMNA_ETIQUETA}_loo": "vas_loo"})
    if controlar_visita:
        g = cen[COLUMNA_GRUPO].to_numpy()
        x_res = residualizar_por_visita(cen, list(features), cen["medicion"], g)
        y_res = residualizar_por_visita(cen[[COLUMNA_ETIQUETA]], [COLUMNA_ETIQUETA],
                                        cen["medicion"], g)
        cen[list(features)] = x_res[list(features)]
        cen[COLUMNA_ETIQUETA] = y_res[COLUMNA_ETIQUETA]
        cen.attrs["personas_residualizadas"] = y_res.attrs["personas_residualizadas"]
    return cen


def centrar_y_controlar(valores, medicion, grupos):
    """
    Centra por la media de la propia persona y, donde hay recta (>= 3
    sesiones), quita la tendencia de visita. Es el control de θ/α y del
    compuesto `_rel_sd` (MODELADO.md, «Control de la deriva por visita»).

    Devuelve `(valores_controlados, personas_con_recta)`.
    """
    s = pd.Series(np.asarray(valores, dtype=float))
    grupos = np.asarray(grupos)
    centrado = pd.DataFrame({"v": (s - s.groupby(grupos).transform("mean")).to_numpy()})
    res = residualizar_por_visita(centrado, ["v"], medicion, grupos)
    return res["v"].to_numpy(), int(res.attrs["personas_residualizadas"])


def centrar_por_persona(valores, grupos):
    """Resta la media de la propia persona (en muestra)."""
    s = pd.Series(np.asarray(valores, dtype=float))
    return (s - s.groupby(np.asarray(grupos)).transform("mean")).to_numpy()


# --------------------------------------------------------------------------
# Importancia por permutación sobre datos no vistos
# --------------------------------------------------------------------------

def importancia_permutacion_evaluada(modelo, X, y, puntuar, bloques=None,
                                     n_repeticiones=10, semilla=42):
    """
    Caída de la puntuación al barajar columnas del conjunto **evaluado**, con
    el modelo ya ajustado.

    Con `bloques=None` se baraja cada columna por separado; con
    `bloques={nombre: [columnas]}` se barajan juntas todas las columnas del
    bloque con la **misma** permutación (no se diluye entre features casi
    duplicadas). `puntuar(y_true, y_pred)` devuelve un número mayor = mejor.

    Devuelve un DataFrame largo con `nombre`, `repeticion`, `puntuacion_base`
    y `caida` (base − barajada).
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    y = np.asarray(y)
    if bloques is None:
        bloques = {c: [c] for c in X.columns}
    faltan = sorted({c for cols in bloques.values() for c in cols} - set(X.columns))
    if faltan:
        raise ValueError(f"Columnas de bloque que no están en X: {faltan}.")
    base = float(puntuar(y, modelo.predict(X)))
    rng = np.random.default_rng(semilla)
    filas = []
    for nombre, columnas in bloques.items():
        for r in range(n_repeticiones):
            barajada = X.copy()
            orden = rng.permutation(len(X))
            barajada[list(columnas)] = X[list(columnas)].to_numpy()[orden]
            caida = base - float(puntuar(y, modelo.predict(barajada)))
            filas.append({"nombre": nombre, "repeticion": r,
                          "puntuacion_base": base, "caida": caida})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# Trazabilidad
# --------------------------------------------------------------------------

def carpeta_salida_nueva(ruta):
    """
    Crea la carpeta de salida y falla si ya existe: una ejecución nunca
    sobrescribe a otra.
    """
    ruta = Path(ruta)
    if ruta.exists():
        raise FileExistsError(
            f"La carpeta de salida ya existe: {ruta}. Una ejecución no "
            f"sobrescribe a otra."
        )
    ruta.mkdir(parents=True)
    return ruta


def sha256_archivo(ruta, bloque=1 << 20):
    """SHA-256 de un archivo, leído por bloques."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(bloque), b""):
            h.update(trozo)
    return h.hexdigest()


def estado_git(repositorio):
    """
    Commit actual y cambios sin commitear de un repositorio.

    Devuelve un dict con `commit`, `limpio` (sin cambios ni archivos sin
    seguimiento) y `cambios` (las líneas de `git status --porcelain`).
    """
    def git(*args):
        return subprocess.run(["git", *args], cwd=repositorio, capture_output=True,
                              text=True, check=True).stdout
    commit = git("rev-parse", "HEAD").strip()
    cambios = [l for l in git("status", "--porcelain").splitlines() if l.strip()]
    return {"commit": commit, "limpio": not cambios, "cambios": cambios}
