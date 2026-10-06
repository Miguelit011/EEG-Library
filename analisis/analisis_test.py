"""
Análisis confirmatorio sobre el conjunto de test, tal como lo fija
`PREREGISTRO_TEST.md`.

Ajusta los modelos finales con un conjunto de entrenamiento, predice **una sola
vez** el conjunto evaluado y calcula todo lo que el pre-registro declara: la
familia VAS (prueba principal + 5 secundarias con Holm + lo descriptivo) y la
familia de condición (OA/OC, reposo/tarea y las cinco tareas, con Holm entre
las tres pruebas principales y los criterios de coherencia). Guarda los datos
en bruto (predicciones, matrices evaluadas, modelos, parámetros, réplicas
bootstrap y de permutación, importancias) y los metadatos de la ejecución.

Modos (hay que elegir uno):

  --ensayo         entrena con `train` y evalúa `val`. Es el ensayo general del
                   pre-registro: comprueba que el guion corre de principio a
                   fin y escribe todos los artefactos. **Sus cifras no son
                   resultados**: `val` ya se usó en la validación cruzada.
  --confirmatorio  entrena con `train` + `val` y evalúa `test`. Exige el árbol
                   de git limpio, se niega si ya existe una ejecución anterior
                   (`processed/test_*`) y solo se lanza una vez.
  --rapido         (con --ensayo) mismo código con réplicas mínimas y salida a
                   un directorio temporal, para depurar. Sus cifras no valen.

Ejecutar (desde la raíz del repositorio):
    python analisis/analisis_test.py --ensayo
    python analisis/analisis_test.py --confirmatorio

El registro se escribe a la vez en pantalla y en `ejecucion.log`, dentro de la
carpeta de salida. La ubicación del dataset se toma de `EEG_DATA`, con
`C:\\Users\\migue\\Desktop\\TFG\\EEG-Data` por defecto (ver `analisis/README.md`).
"""

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score

from eeg_library.modelado.confirmatorio import (
    carpeta_salida_nueva,
    centrar_por_persona,
    centrar_y_controlar,
    estado_git,
    exactitud_balanceada_lote,
    holm,
    ic_percentil,
    importancia_permutacion_evaluada,
    p_bootstrap,
    p_permutacion,
    preparar_tarea_b,
    replicas_bootstrap,
    sha256_archivo,
    veredicto_principal,
)
from eeg_library.modelado.modeling import (
    COLUMNA_ETIQUETA,
    COLUMNA_GRUPO,
    a_clases,
    agregar_a_sesion,
    columnas_de_features_sesion,
    compuesto_rel_sd,
    cortes_terciles,
    metricas_clasificacion,
)
from eeg_library.modelado.reposo import (
    COLUMNAS_PROCESAMIENTO,
    UmbralUnaVariable,
    auc_roc,
    bloque_de_feature,
    columnas_de_features_reposo,
    columnas_del_bloque,
    etiqueta_binaria,
    importancia_por_bloque,
    metricas_binarias,
    preparar_reposo,
    tamanos_de_efecto,
)
from eeg_library.modelado.reposo_tarea import (
    columnas_media_del_bloque,
    comparar_condiciones,
    desglose_por_condicion,
    etiqueta_tarea,
    modelo_bosque,
    modelo_logistico,
    precision_media,
    preparar_reposo_tarea,
)
from eeg_library.modelado.tareas_ejecutivas import (
    CONDICIONES_EJECUTIVAS,
    confusion_por_pares,
    distancias_de_duracion,
    estructura_de_la_confusion,
    etiqueta_multiclase,
    matriz_confusion_normalizada,
    metricas_multiclase,
    modelo_bosque_multiclase,
    modelo_logistico_multinomial,
    preparar_tareas_ejecutivas,
    tamanos_de_efecto_una_contra_resto,
)
from eeg_library.procesado.bands import BANDAS_DEFECTO
from eeg_library.procesado.matrix import (
    COLUMNAS_IDENTIDAD,
    ORDEN_PROTOCOLO,
    REGIONES,
)

# Se reutiliza tal cual la regresión de REGRESION_LINEAL.md para θ/α.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisis_regresion_lineal import ols  # noqa: E402

RAIZ_REPO = Path(__file__).resolve().parents[1]
RUTA_DATOS = Path(os.environ.get("EEG_DATA", r"C:\Users\migue\Desktop\TFG\EEG-Data"))
RUTA_MATRIZ = RUTA_DATOS / "processed" / "matriz_rf.parquet"
RUTA_PREREGISTRO = RAIZ_REPO / "PREREGISTRO_TEST.md"
ARCHIVOS_DATOS = {
    "matriz_rf.parquet": RUTA_MATRIZ,
    "splits.csv": RUTA_DATOS / "metadata" / "splits.csv",
    "labels.csv": RUTA_DATOS / "metadata" / "labels.csv",
    "labels_grabaciones.csv": RUTA_DATOS / "metadata" / "labels_grabaciones.csv",
    "VAS_100mm_preFE_filtrado.csv": RUTA_DATOS / "metadata" / "VAS_100mm_preFE_filtrado.csv",
}

SEMILLA = 42
ALFA = 0.05

# Referencias de validación (PREREGISTRO_TEST.md, secciones 4 y 6).
REFERENCIA_PRINCIPAL = 0.3795
MARGEN_R2 = 0.05
MARGEN_MAE = 2.0
PENDIENTE_UTIL = 10.0
GEMELOS = ({"PASAT", "PVSAT"}, {"STROOP", "MSIT"})
FRACCION_ALFA_MINIMA = 0.50
RANGO_ALFA_POSTERIOR = 5

BANDAS = tuple(BANDAS_DEFECTO)
REGIONES_N = tuple(REGIONES)


class Config:
    """Número de réplicas: el del pre-registro, o mínimo con --rapido."""

    def __init__(self, rapido):
        self.n_bootstrap = 200 if rapido else 10000
        self.n_permutaciones = 200 if rapido else 10000
        self.n_repeticiones_importancia = 2 if rapido else 10


# --------------------------------------------------------------------------
# Registro y artefactos
# --------------------------------------------------------------------------

class Duplicador:
    """Escribe en pantalla y en el log de la carpeta de salida a la vez."""

    def __init__(self, *flujos):
        self.flujos = flujos

    def write(self, texto):
        for f in self.flujos:
            f.write(texto)
            f.flush()

    def flush(self):
        for f in self.flujos:
            f.flush()


def _a_python(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index().to_dict(orient="records")
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"No serializable: {type(obj)}")


class Artefactos:
    """Acumula los datos en bruto y los escribe al final (sección 7)."""

    def __init__(self, carpeta):
        self.carpeta = Path(carpeta)
        self.predicciones = []
        self.inventario = []
        self.importancias = []
        self.parametros = {}
        for sub in ("modelos", "bootstrap", "permutaciones"):
            (self.carpeta / sub).mkdir()

    def prediccion(self, analisis, modelo, identidad, y_true, y_pred,
                   probabilidades=None, clases=None, extra=None):
        df = identidad.reset_index(drop=True).copy()
        df.insert(0, "modelo", modelo)
        df.insert(0, "analisis", analisis)
        df["y_true"] = np.asarray(y_true)
        df["y_pred"] = np.asarray(y_pred)
        if probabilidades is not None:
            for k, c in enumerate(clases):
                df[f"prob_{c}"] = probabilidades[:, k]
        for nombre, valores in (extra or {}).items():
            df[nombre] = np.asarray(valores)
        self.predicciones.append(df)

    def matriz(self, nombre, df):
        df.reset_index(drop=True).to_parquet(self.carpeta / f"X_test_{nombre}.parquet")

    def modelo(self, nombre, modelo):
        joblib.dump(modelo, self.carpeta / "modelos" / f"{nombre}.joblib")
        self.parametros.setdefault("modelos", {})[nombre] = _parametros_de(modelo)

    def replicas(self, nombre, valores, tipo="bootstrap"):
        np.save(self.carpeta / tipo / f"{nombre}.npy", np.asarray(valores))

    def escribir(self, resultados):
        pd.concat(self.predicciones, ignore_index=True).to_parquet(
            self.carpeta / "predicciones_test.parquet")
        pd.DataFrame(self.inventario).to_csv(self.carpeta / "inventario.csv", index=False)
        if self.importancias:
            pd.concat(self.importancias, ignore_index=True).to_parquet(
                self.carpeta / "importancias_test.parquet")
        with open(self.carpeta / "parametros.json", "w", encoding="utf-8") as f:
            json.dump(self.parametros, f, indent=2, ensure_ascii=False, default=_a_python)
        with open(self.carpeta / "resultados.json", "w", encoding="utf-8") as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False, default=_a_python)


def _parametros_de(modelo):
    """Lo que el modelo aprendió, en un formato legible sin joblib."""
    salida = {"clase": type(modelo).__name__,
              "get_params": {k: repr(v) for k, v in modelo.get_params(deep=True).items()}}
    pasos = getattr(modelo, "named_steps", {"modelo": modelo})
    for nombre, paso in pasos.items():
        p = {}
        for atributo in ("statistics_", "mean_", "scale_", "coef_", "intercept_",
                         "classes_", "feature_importances_", "umbral_", "signo_",
                         "mediana_"):
            if hasattr(paso, atributo):
                p[atributo] = getattr(paso, atributo)
        if hasattr(paso, "feature_names_in_"):
            p["feature_names_in_"] = list(paso.feature_names_in_)
        salida[nombre] = p
    return salida


def _titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def _identidad(df):
    return df[[c for c in COLUMNAS_IDENTIDAD if c in df.columns]]


def _ic_texto(ic):
    return f"[{ic[0]:.4f}, {ic[1]:.4f}]"


# --------------------------------------------------------------------------
# Utilidades de puntuación
# --------------------------------------------------------------------------

def _ba(y_true, y_pred, clases):
    return float(exactitud_balanceada_lote(y_true, y_pred, clases)[0])


def _r2(y, pred):
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot <= 0:
        raise ValueError("Sin varianza: R² no definido.")
    return 1 - np.sum((y - pred) ** 2) / ss_tot


def _spearman(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        raise ValueError("Variable constante: Spearman no definido.")
    return float(stats.spearmanr(a, b).statistic)


def _d_cohen(x, y):
    """d de Cohen (clase 1 − clase 0), como `reposo.tamanos_de_efecto`."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    uno, cero = x[(y == 1) & ~np.isnan(x)], x[(y == 0) & ~np.isnan(x)]
    if len(uno) < 2 or len(cero) < 2:
        raise ValueError("Hacen falta 2 observaciones por clase.")
    agrupada = np.sqrt(((len(uno) - 1) * uno.var(ddof=1)
                        + (len(cero) - 1) * cero.var(ddof=1))
                       / (len(uno) + len(cero) - 2))
    if agrupada <= 0:
        raise ValueError("Sin varianza.")
    return float((uno.mean() - cero.mean()) / agrupada)


def _d_con_ic(x, y, g, cfg, nombre, sal):
    d = _d_cohen(x, y)
    rep = replicas_bootstrap(g, lambda idx: _d_cohen(x[idx], y[idx]),
                             cfg.n_bootstrap, SEMILLA)
    sal.replicas(nombre, rep)
    ic = ic_percentil(rep)
    return {"d": d, "ic95": ic, "excluye_0": bool(ic[0] > 0 or ic[1] < 0)}


# --------------------------------------------------------------------------
# Familia VAS
# --------------------------------------------------------------------------

def _rf_clasificador(**extra):
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                  random_state=SEMILLA, n_jobs=-1, **extra)


def _rf_regresor():
    return RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                                 random_state=SEMILLA, n_jobs=-1)


def _tres_niveles(ses_tr, ses_ev, features, controlar, modelo):
    """Ajusta la prueba de la Tarea B en tres niveles y predice el evaluado."""
    cen_tr = preparar_tarea_b(ses_tr, features, controlar_visita=controlar)
    cen_ev = preparar_tarea_b(ses_ev, features, controlar_visita=controlar)
    cortes = cortes_terciles(cen_tr[COLUMNA_ETIQUETA])
    y_tr = a_clases(cen_tr[COLUMNA_ETIQUETA], cortes).astype(int)
    y_ev = a_clases(cen_ev[COLUMNA_ETIQUETA], cortes).astype(int)
    modelo.fit(cen_tr[features], y_tr)
    pred = modelo.predict(cen_ev[features]).astype(int)
    proba = modelo.predict_proba(cen_ev[features]) if hasattr(modelo, "predict_proba") else None
    return {"cen_tr": cen_tr, "cen_ev": cen_ev, "cortes": cortes, "y_tr": y_tr,
            "y_ev": y_ev, "pred": pred, "proba": proba, "modelo": modelo,
            "g": cen_ev[COLUMNA_GRUPO].to_numpy()}


def familia_vas(datos, entrena, evalua, cfg, sal):
    res = {}
    _titulo("FAMILIA VAS — datos")
    ses_tr = agregar_a_sesion(datos[datos["split"].isin(entrena)])
    ses_ev = agregar_a_sesion(datos[datos["split"].isin(evalua)])
    features = columnas_de_features_sesion(ses_tr)
    if features != columnas_de_features_sesion(ses_ev):
        raise ValueError("Las features de sesión no coinciden entre los dos conjuntos.")
    por_persona_ev = ses_ev.groupby(COLUMNA_GRUPO).size()
    print(f"Entrenamiento: {len(ses_tr)} sesiones, {ses_tr[COLUMNA_GRUPO].nunique()} personas")
    print(f"Evaluado     : {len(ses_ev)} sesiones, {ses_ev[COLUMNA_GRUPO].nunique()} personas")
    print("Sesiones por persona evaluada: " + ", ".join(
        f"{k}: {v}" for k, v in por_persona_ev.value_counts().sort_index().items()))
    print(f"Features por sesión: {len(features)}")
    for _, fila in ses_ev.iterrows():
        n = por_persona_ev[fila[COLUMNA_GRUPO]]
        sal.inventario.append({
            "familia": "VAS", "unidad": "sesion", "id": fila["sesion_uid"],
            "persona_uid": fila[COLUMNA_GRUPO],
            "tarea_a": "usada", "tarea_b_e_intrapersona":
                "usada" if n >= 2 else "descartada: persona con una sola sesión",
            "recta_de_visita": "sí" if n >= 3 else "no (< 3 sesiones)"})
    res["datos"] = {
        "sesiones_entrenamiento": len(ses_tr),
        "personas_entrenamiento": int(ses_tr[COLUMNA_GRUPO].nunique()),
        "sesiones_evaluadas": len(ses_ev),
        "personas_evaluadas": int(ses_ev[COLUMNA_GRUPO].nunique()),
        "sesiones_por_persona_evaluada": {int(k): int(v) for k, v in
                                          por_persona_ev.value_counts().items()},
        "n_features": len(features)}
    sal.matriz("vas_sesiones_evaluadas", ses_ev)

    # -- Prueba principal --------------------------------------------------
    _titulo("PRUEBA PRINCIPAL — Tarea B, tres niveles, RF hoja>=5, con control de visita")
    pr = _tres_niveles(ses_tr, ses_ev, features, True, _rf_clasificador())
    g, y, pred = pr["g"], pr["y_ev"], pr["pred"]
    print(f"Sesiones evaluadas: {len(y)} de {pr['cen_ev'][COLUMNA_GRUPO].nunique()} personas "
          f"(entrenamiento: {len(pr['y_tr'])}); cortes de train: "
          f"[{pr['cortes'][0]:+.2f}, {pr['cortes'][1]:+.2f}]")
    perm = p_permutacion(y, pred, g, [0, 1, 2], cfg.n_permutaciones, SEMILLA)
    sal.replicas("principal_nulo", perm["nulo"], "permutaciones")

    # Regresión discretizada con los mismos cortes (descriptivo, pareado)
    reg = _rf_regresor().fit(pr["cen_tr"][features], pr["cen_tr"][COLUMNA_ETIQUETA])
    cont = reg.predict(pr["cen_ev"][features])
    pred_reg = a_clases(cont, pr["cortes"]).astype(int)
    rep = replicas_bootstrap(
        g, lambda idx: (_ba(y[idx], pred[idx], [0, 1, 2]),
                        _ba(y[idx], pred_reg[idx], [0, 1, 2])),
        cfg.n_bootstrap, SEMILLA)
    sal.replicas("principal_ba_y_regresion_discretizada", rep)
    ic = ic_percentil(rep[:, 0])
    veredicto = veredicto_principal(perm["p"], ic[1], REFERENCIA_PRINCIPAL, ALFA)
    m = metricas_clasificacion(y, pred)
    print(f"Exactitud balanceada: {perm['observado']:.4f}  IC95 {_ic_texto(ic)}  "
          f"p = {perm['p']:.4f}  (azar 0.3333; referencia {REFERENCIA_PRINCIPAL})")
    print(f"VEREDICTO: {veredicto.upper()}")
    exhaustividad = {c: float(np.mean(pred[y == c] == c)) for c in range(3)}
    confusion = pd.crosstab(pd.Series(y, name="real"), pd.Series(pred, name="predicha")
                            ).reindex(index=range(3), columns=range(3), fill_value=0)
    print("Matriz de confusión (filas = real):\n" + confusion.to_string())
    ba_reg = _ba(y, pred_reg, [0, 1, 2])
    dif = rep[:, 0] - rep[:, 1]
    print(f"Regresión discretizada: {ba_reg:.4f}; clasificador − regresión = "
          f"{perm['observado'] - ba_reg:+.4f} IC95 {_ic_texto(ic_percentil(dif))}")
    res["principal"] = {
        "exactitud_balanceada": perm["observado"], "ic95": ic, "p": perm["p"],
        "veredicto": veredicto, "referencia": REFERENCIA_PRINCIPAL,
        "n_sesiones": int(len(y)), "n_personas": int(len(pd.unique(g))),
        "personas_residualizadas": pr["cen_ev"].attrs.get("personas_residualizadas"),
        "cortes": pr["cortes"], "metricas": m, "exhaustividad_por_clase": exhaustividad,
        "matriz_confusion": confusion.to_numpy(),
        "reparto_real": np.bincount(y, minlength=3), "reparto_predicho": np.bincount(pred, minlength=3),
        "regresion_discretizada": {
            "exactitud_balanceada": ba_reg, "ic95": ic_percentil(rep[:, 1]),
            "reparto_predicho": np.bincount(pred_reg, minlength=3),
            "diferencia_clasificador_menos_regresion": perm["observado"] - ba_reg,
            "ic95_diferencia": ic_percentil(dif)}}
    sal.parametros["principal_cortes"] = pr["cortes"]
    sal.modelo("vas_principal_rf", pr["modelo"])
    sal.modelo("vas_principal_regresion_discretizada", reg)
    sal.matriz("vas_principal", pd.concat(
        [_identidad(pr["cen_ev"]), pr["cen_ev"][features + [COLUMNA_ETIQUETA]]], axis=1))
    sal.prediccion("vas_principal", "RF hoja>=5", _identidad(pr["cen_ev"]), y, pred,
                   pr["proba"], [0, 1, 2], extra={
                       "vas_original": pr["cen_ev"]["vas_original"],
                       "vas_loo": pr["cen_ev"]["vas_loo"],
                       "desviacion_sin_control": pr["cen_ev"]["desviacion_sin_control"],
                       "desviacion_controlada": pr["cen_ev"][COLUMNA_ETIQUETA]})
    sal.prediccion("vas_principal", "regresion discretizada", _identidad(pr["cen_ev"]),
                   y, pred_reg, extra={"prediccion_continua": cont})

    # -- Descriptivo: sin control y otras configuraciones -----------------
    _titulo("PRINCIPAL — descriptivo: sin control y otras configuraciones")
    sin = _tres_niveles(ses_tr, ses_ev, features, False, _rf_clasificador())
    perm_sin = p_permutacion(sin["y_ev"], sin["pred"], sin["g"], [0, 1, 2],
                             cfg.n_permutaciones, SEMILLA)
    rep_sin = replicas_bootstrap(
        sin["g"], lambda idx: _ba(sin["y_ev"][idx], sin["pred"][idx], [0, 1, 2]),
        cfg.n_bootstrap, SEMILLA)
    sal.replicas("principal_sin_control_ba", rep_sin)
    sal.replicas("principal_sin_control_nulo", perm_sin["nulo"], "permutaciones")
    print(f"Sin control de visita: {perm_sin['observado']:.4f} IC95 "
          f"{_ic_texto(ic_percentil(rep_sin))} p = {perm_sin['p']:.4f} (descriptivo)")
    sal.prediccion("vas_principal_sin_control", "RF hoja>=5", _identidad(sin["cen_ev"]),
                   sin["y_ev"], sin["pred"], sin["proba"], [0, 1, 2])
    sal.modelo("vas_principal_sin_control_rf", sin["modelo"])
    res["principal_sin_control"] = {
        "exactitud_balanceada": perm_sin["observado"], "ic95": ic_percentil(rep_sin),
        "p": perm_sin["p"], "cortes": sin["cortes"],
        "metricas": metricas_clasificacion(sin["y_ev"], sin["pred"])}
    res["otras_configuraciones"] = {}
    for nombre, modelo in (("RF hoja>=1", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=1, random_state=SEMILLA, n_jobs=-1)),
                           ("RF hoja>=15", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=15, random_state=SEMILLA, n_jobs=-1)),
                           ("RF hoja>=5 balanceado", _rf_clasificador(class_weight="balanced"))):
        r = _tres_niveles(ses_tr, ses_ev, features, True, modelo)
        rep_c = replicas_bootstrap(
            r["g"], lambda idx: _ba(r["y_ev"][idx], r["pred"][idx], [0, 1, 2]),
            cfg.n_bootstrap, SEMILLA)
        ba_c = _ba(r["y_ev"], r["pred"], [0, 1, 2])
        print(f"{nombre:<24s} {ba_c:.4f} IC95 {_ic_texto(ic_percentil(rep_c))}")
        res["otras_configuraciones"][nombre] = {"exactitud_balanceada": ba_c,
                                                "ic95": ic_percentil(rep_c)}
        sal.prediccion("vas_principal_otras", nombre, _identidad(r["cen_ev"]),
                       r["y_ev"], r["pred"], r["proba"], [0, 1, 2])

    # -- Secundarias ---------------------------------------------------------
    sec = {}
    g_ev = ses_ev[COLUMNA_GRUPO].to_numpy()
    varias = ses_ev[COLUMNA_GRUPO].map(por_persona_ev).to_numpy() >= 2
    vas_ev = ses_ev[COLUMNA_ETIQUETA].to_numpy(dtype=float)

    _titulo("S1 — compuesto _rel_sd, ρ intra-persona con control de visita")
    rel = sorted(c for c in features if c.endswith("_rel_sd"))
    sal.parametros["compuesto_rel_sd"] = {"columnas": rel,
                                          "media": ses_tr[rel].mean(), "sd": ses_tr[rel].std()}
    comp = compuesto_rel_sd(ses_ev, ses_tr).to_numpy()
    cx, _ = centrar_y_controlar(comp, ses_ev["medicion"], g_ev)
    cy, _ = centrar_y_controlar(vas_ev, ses_ev["medicion"], g_ev)
    cx, cy, gv = cx[varias], cy[varias], g_ev[varias]
    rho = _spearman(cx, cy)
    rep = replicas_bootstrap(gv, lambda idx: _spearman(cx[idx], cy[idx]),
                             cfg.n_bootstrap, SEMILLA)
    sal.replicas("s1_rho", rep)
    sec["S1"] = {"afirmacion": "rho intra-persona del compuesto _rel_sd > 0",
                 "rho": rho, "ic95": ic_percentil(rep), "n": int(len(cx)),
                 "p": p_bootstrap(rep, 0.0, "menor_o_igual"), "referencia": 0.130}
    ux = centrar_por_persona(comp, g_ev)[varias]
    uy = centrar_por_persona(vas_ev, g_ev)[varias]
    rep_u = replicas_bootstrap(gv, lambda idx: _spearman(ux[idx], uy[idx]),
                               cfg.n_bootstrap, SEMILLA)
    sec["S1"]["sin_control"] = {"rho": _spearman(ux, uy), "ic95": ic_percentil(rep_u)}
    print(f"ρ = {rho:+.4f} IC95 {_ic_texto(sec['S1']['ic95'])} p = {sec['S1']['p']:.4f} "
          f"(sin control: {sec['S1']['sin_control']['rho']:+.4f})")

    _titulo("S2 — θ/α intrapersona con control de visita: R² < 0.05")
    x_ta = ses_ev["indice_fatiga_media"].to_numpy(dtype=float)
    dx, rectas = centrar_y_controlar(x_ta, ses_ev["medicion"], g_ev)
    dy, _ = centrar_y_controlar(vas_ev, ses_ev["medicion"], g_ev)
    n_personas_ev = int(ses_ev[COLUMNA_GRUPO].nunique())
    ajuste = ols(dx, dy, gl_extra=n_personas_ev + rectas)

    def _r2_y_pendiente(idx):
        a, b = dx[idx], dy[idx]
        sa = np.sum((a - a.mean()) ** 2)
        if sa <= 0 or np.sum((b - b.mean()) ** 2) <= 0:
            raise ValueError("sin varianza")
        pendiente = np.sum((a - a.mean()) * (b - b.mean())) / sa
        return np.corrcoef(a, b)[0, 1] ** 2, pendiente
    rep = replicas_bootstrap(g_ev, _r2_y_pendiente, cfg.n_bootstrap, SEMILLA)
    sal.replicas("s2_r2_y_pendiente", rep)
    sec["S2"] = {"afirmacion": "R² intrapersona θ/α < 0.05", "r2": ajuste["r2"],
                 "ic95_r2": ic_percentil(rep[:, 0]),
                 "p": p_bootstrap(rep[:, 0], MARGEN_R2, "mayor_o_igual"),
                 "pendiente": ajuste["pendiente"], "ic95_pendiente_t": ajuste["ic95_pendiente"],
                 "ic95_pendiente_bootstrap": ic_percentil(rep[:, 1]),
                 "excluye_pendiente_util": bool(ic_percentil(rep[:, 1])[1] < PENDIENTE_UTIL),
                 "ajuste": ajuste}
    ux, uy = centrar_por_persona(x_ta, g_ev), centrar_por_persona(vas_ev, g_ev)
    sec["S2"]["sin_control"] = ols(ux, uy, gl_extra=n_personas_ev)
    sec["S2"]["regresion_global"] = ols(x_ta, vas_ev)
    print(f"R² = {ajuste['r2']:.5f} IC95 {_ic_texto(sec['S2']['ic95_r2'])} p = {sec['S2']['p']:.4f}; "
          f"β = {ajuste['pendiente']:+.3f} IC95 boot {_ic_texto(sec['S2']['ic95_pendiente_bootstrap'])}")

    _titulo("S3 y S5 — Tarea A (persona nueva): regresión RF sobre el VAS absoluto")
    reg_a = _rf_regresor().fit(ses_tr[features], ses_tr[COLUMNA_ETIQUETA])
    pred_a = reg_a.predict(ses_ev[features])
    media_tr = float(ses_tr[COLUMNA_ETIQUETA].mean())
    sal.parametros["tarea_a_media_entrenamiento"] = media_tr
    sal.modelo("vas_tarea_a_regresor", reg_a)
    sal.prediccion("vas_tarea_a", "RF regresor hoja>=5", _identidad(ses_ev), vas_ev, pred_a)

    def _tarea_a(idx):
        yv, pv = vas_ev[idx], pred_a[idx]
        const = np.full(len(yv), media_tr)
        return (_r2(yv, pv) - _r2(yv, const),
                np.mean(np.abs(yv - const)) - np.mean(np.abs(yv - pv)),
                _spearman(pv, yv), _r2(yv, pv))
    obs = _tarea_a(np.arange(len(vas_ev)))
    rep = replicas_bootstrap(g_ev, _tarea_a, cfg.n_bootstrap, SEMILLA)
    sal.replicas("s3_s5_tarea_a", rep)
    sec["S3"] = {"afirmacion": "Tarea A: ΔR² < 0.05 y mejora del MAE < 2 frente a la media de entrenamiento",
                 "delta_r2": obs[0], "ic95_delta_r2": ic_percentil(rep[:, 0]),
                 "mejora_mae": obs[1], "ic95_mejora_mae": ic_percentil(rep[:, 1]),
                 "r2_modelo": obs[3], "ic95_r2_modelo": ic_percentil(rep[:, 3]),
                 "p_r2": p_bootstrap(rep[:, 0], MARGEN_R2, "mayor_o_igual"),
                 "p_mae": p_bootstrap(rep[:, 1], MARGEN_MAE, "mayor_o_igual")}
    sec["S3"]["p"] = max(sec["S3"]["p_r2"], sec["S3"]["p_mae"])
    sec["S5"] = {"afirmacion": "Tarea A: rho(predicción, VAS) > 0", "rho": obs[2],
                 "ic95": ic_percentil(rep[:, 2]),
                 "p": p_bootstrap(rep[:, 2], 0.0, "menor_o_igual"), "referencia": 0.117}
    print(f"S3: ΔR² = {obs[0]:+.4f} IC95 {_ic_texto(sec['S3']['ic95_delta_r2'])}; mejora MAE = "
          f"{obs[1]:+.3f} IC95 {_ic_texto(sec['S3']['ic95_mejora_mae'])}; p = {sec['S3']['p']:.4f}")
    print(f"S5: ρ = {obs[2]:+.4f} IC95 {_ic_texto(sec['S5']['ic95'])} p = {sec['S5']['p']:.4f}")

    _titulo("S4 — Tarea B, regresión: el EEG no aporta sobre la media de la persona")
    b_tr = preparar_tarea_b(ses_tr, features, controlar_visita=False)
    b_ev = preparar_tarea_b(ses_ev, features, controlar_visita=False)
    reg_b = _rf_regresor().fit(b_tr[features], b_tr[COLUMNA_ETIQUETA])
    dev = reg_b.predict(b_ev[features])
    y_b = b_ev["vas_original"].to_numpy(dtype=float)
    ref_b = b_ev["vas_loo"].to_numpy(dtype=float)
    con_b = ref_b + dev
    g_b = b_ev[COLUMNA_GRUPO].to_numpy()
    sal.modelo("vas_tarea_b_regresor", reg_b)
    sal.prediccion("vas_tarea_b_regresion", "RF regresor hoja>=5", _identidad(b_ev), y_b,
                   con_b, extra={"vas_loo": ref_b, "desviacion_predicha": dev})

    def _tarea_b(idx):
        yv = y_b[idx]
        return (_r2(yv, con_b[idx]) - _r2(yv, ref_b[idx]),
                np.mean(np.abs(yv - ref_b[idx])) - np.mean(np.abs(yv - con_b[idx])),
                _r2(yv, con_b[idx]), _r2(yv, ref_b[idx]))
    obs = _tarea_b(np.arange(len(y_b)))
    rep = replicas_bootstrap(g_b, _tarea_b, cfg.n_bootstrap, SEMILLA)
    sal.replicas("s4_tarea_b", rep)
    sec["S4"] = {"afirmacion": "Tarea B: con EEG − sin EEG, ΔR² < 0.05 y mejora del MAE < 2",
                 "delta_r2": obs[0], "ic95_delta_r2": ic_percentil(rep[:, 0]),
                 "mejora_mae": obs[1], "ic95_mejora_mae": ic_percentil(rep[:, 1]),
                 "r2_con_eeg": obs[2], "r2_sin_eeg": obs[3],
                 "p_r2": p_bootstrap(rep[:, 0], MARGEN_R2, "mayor_o_igual"),
                 "p_mae": p_bootstrap(rep[:, 1], MARGEN_MAE, "mayor_o_igual"),
                 "n_sesiones": int(len(y_b))}
    sec["S4"]["p"] = max(sec["S4"]["p_r2"], sec["S4"]["p_mae"])
    print(f"ΔR² = {obs[0]:+.4f} IC95 {_ic_texto(sec['S4']['ic95_delta_r2'])}; mejora MAE = "
          f"{obs[1]:+.3f} IC95 {_ic_texto(sec['S4']['ic95_mejora_mae'])}; p = {sec['S4']['p']:.4f}")

    _titulo("SECUNDARIAS — Holm")
    claves = ["S1", "S2", "S3", "S4", "S5"]
    ajustados = holm([sec[k]["p"] for k in claves])
    for k, pa in zip(claves, ajustados):
        sec[k]["p_holm"] = float(pa)
        sec[k]["significativa"] = bool(pa < ALFA)
        print(f"  {k}: p = {sec[k]['p']:.4f}  p Holm = {pa:.4f}  "
              f"{'SIGNIFICATIVA' if pa < ALFA else 'no significativa'} — {sec[k]['afirmacion']}")
    res["secundarias"] = sec

    _titulo("DESCRIPTIVO — Tarea A en tres niveles")
    cortes_a = cortes_terciles(ses_tr[COLUMNA_ETIQUETA])
    clf_a = _rf_clasificador().fit(ses_tr[features],
                                   a_clases(ses_tr[COLUMNA_ETIQUETA], cortes_a).astype(int))
    y_a = a_clases(vas_ev, cortes_a).astype(int)
    p_a = clf_a.predict(ses_ev[features]).astype(int)
    rep = replicas_bootstrap(g_ev, lambda idx: _ba(y_a[idx], p_a[idx], [0, 1, 2]),
                             cfg.n_bootstrap, SEMILLA)
    res["tarea_a_tres_niveles"] = {"exactitud_balanceada": _ba(y_a, p_a, [0, 1, 2]),
                                   "ic95": ic_percentil(rep), "cortes": cortes_a,
                                   "metricas": metricas_clasificacion(y_a, p_a)}
    sal.prediccion("vas_tarea_a_tres_niveles", "RF hoja>=5", _identidad(ses_ev), y_a, p_a,
                   clf_a.predict_proba(ses_ev[features]), [0, 1, 2])
    print(f"{res['tarea_a_tres_niveles']['exactitud_balanceada']:.4f} IC95 "
          f"{_ic_texto(ic_percentil(rep))} (cortes {cortes_a[0]:.1f}, {cortes_a[1]:.1f})")
    return res


# --------------------------------------------------------------------------
# Familia de condición
# --------------------------------------------------------------------------

def _preparar(preparar, datos, splits, **kw):
    return preparar(datos, splits=splits, permitir_test="test" in splits, **kw)


def _bloques(columnas):
    bloques = {f"banda {b}": columnas_del_bloque(columnas, banda=b) for b in BANDAS}
    bloques.update({f"región {r}": columnas_del_bloque(columnas, region=r) for r in REGIONES_N})
    bloques["posterior ∩ alfa"] = columnas_del_bloque(columnas, region="posterior", banda="alfa")
    bloques["frontal ∩ theta"] = columnas_del_bloque(columnas, region="frontal", banda="theta")
    return {k: v for k, v in bloques.items() if v}


def _coeficientes(lr, columnas, clases_nombres=None):
    coef = lr[-1].coef_
    if coef.shape[0] == 1:
        return pd.Series(coef[0], index=columnas)
    return pd.DataFrame(coef.T, index=columnas, columns=clases_nombres)


def _evaluar_condicion(clave, datos_tr, datos_ev, y_tr, y_ev, clases, fabrica_lr,
                       fabrica_rf, cfg, sal, umbral=None, binaria=True,
                       clase_ap=None):
    """Ajusta LR, RF, baselines, umbral y contadores; puntúa el evaluado."""
    columnas = columnas_de_features_reposo(datos_tr)
    if columnas != columnas_de_features_reposo(datos_ev):
        raise ValueError("Las columnas de X no coinciden entre los dos conjuntos.")
    X_tr, X_ev = datos_tr[columnas], datos_ev[columnas]
    g = datos_ev[COLUMNA_GRUPO].to_numpy()
    print(f"Entrenamiento: {len(datos_tr)} grabaciones, {datos_tr[COLUMNA_GRUPO].nunique()} personas")
    print(f"Evaluado     : {len(datos_ev)} grabaciones, {len(pd.unique(g))} personas")
    print("Reparto evaluado: " + ", ".join(
        f"{int(k)}: {int(v)}" for k, v in zip(*np.unique(y_ev, return_counts=True))))
    sal.matriz(clave, pd.concat([_identidad(datos_ev), X_ev], axis=1))

    modelos = {"logistica": fabrica_lr().fit(X_tr, y_tr), "random_forest": fabrica_rf().fit(X_tr, y_tr)}
    pred, proba = {}, {}
    for n, m in modelos.items():
        pred[n] = m.predict(X_ev).astype(int)
        proba[n] = m.predict_proba(X_ev)
        sal.modelo(f"{clave}_{n}", m)
        sal.prediccion(clave, n, _identidad(datos_ev), y_ev, pred[n], proba[n], clases)
    if umbral is not None:
        modelos["umbral"] = UmbralUnaVariable(umbral).fit(X_tr, y_tr)
        pred["umbral"] = modelos["umbral"].predict(X_ev).astype(int)
        sal.modelo(f"{clave}_umbral", modelos["umbral"])
        sal.prediccion(clave, f"umbral {umbral}", _identidad(datos_ev), y_ev, pred["umbral"],
                       extra={"x": X_ev[umbral]})
    nombres = list(pred)

    def _todas(idx):
        valores = [_ba(y_ev[idx], pred[n][idx], clases) for n in nombres]
        if binaria:
            valores += [auc_roc(y_ev[idx], proba[n][idx, 1]) for n in ("logistica", "random_forest")]
        if clase_ap is not None:
            valores += [precision_media(y_ev[idx], proba[n][idx, 1], clase_positiva=clase_ap)
                        for n in ("logistica", "random_forest")]
        return valores
    obs = _todas(np.arange(len(y_ev)))
    rep = replicas_bootstrap(g, _todas, cfg.n_bootstrap, SEMILLA)
    sal.replicas(f"{clave}_modelos", rep)
    res = {"n_grabaciones": int(len(y_ev)), "n_personas": int(len(pd.unique(g))),
           "reparto_evaluado": {int(k): int(v) for k, v in zip(*np.unique(y_ev, return_counts=True))},
           "modelos": {}}
    for i, n in enumerate(nombres):
        res["modelos"][n] = {"exactitud_balanceada": obs[i], "ic95": ic_percentil(rep[:, i])}
        if n != "umbral":
            res["modelos"][n]["metricas"] = (metricas_binarias(y_ev, pred[n]) if binaria
                                             else metricas_multiclase(y_ev, pred[n]))
    k = len(nombres)
    if binaria:
        for j, n in enumerate(("logistica", "random_forest")):
            res["modelos"][n]["auc"] = obs[k + j]
            res["modelos"][n]["ic95_auc"] = ic_percentil(rep[:, k + j])
        k += 2
    if clase_ap is not None:
        prevalencia = float(np.mean(y_ev == clase_ap))
        for j, n in enumerate(("logistica", "random_forest")):
            res["modelos"][n]["precision_media_reposo"] = obs[k + j]
            res["modelos"][n]["ic95_precision_media"] = ic_percentil(rep[:, k + j])
        res["prevalencia_reposo"] = prevalencia
    res["diferencias_pareadas"] = {}
    for a, b in [("logistica", "random_forest"), ("logistica", "umbral"),
                 ("random_forest", "umbral")]:
        if a in nombres and b in nombres:
            ia, ib = nombres.index(a), nombres.index(b)
            res["diferencias_pareadas"][f"{a} − {b}"] = {
                "diferencia": obs[ia] - obs[ib], "ic95": ic_percentil(rep[:, ia] - rep[:, ib])}
    if umbral is not None:
        res["umbral"] = {"feature": umbral, "umbral": modelos["umbral"].umbral_,
                         "signo": int(modelos["umbral"].signo_)}

    # p por permutación dentro de persona (la logística es la prueba principal)
    for n in ("logistica", "random_forest"):
        pp = p_permutacion(y_ev, pred[n], g, clases, cfg.n_permutaciones, SEMILLA)
        sal.replicas(f"{clave}_{n}_nulo", pp["nulo"], "permutaciones")
        res["modelos"][n]["p_permutacion"] = pp["p"]

    # baselines triviales
    for n, estrategia in (("mayoritaria", "most_frequent"), ("azar_estratificado", "stratified")):
        d = DummyClassifier(strategy=estrategia, random_state=SEMILLA).fit(X_tr, y_tr)
        res.setdefault("baselines", {})[n] = _ba(y_ev, d.predict(X_ev), clases)

    # solo los 6 contadores de procesamiento
    contadores = list(COLUMNAS_PROCESAMIENTO)
    res["solo_contadores"] = {}
    for n, fab in (("logistica", fabrica_lr), ("random_forest", fabrica_rf)):
        m = fab().fit(datos_tr[contadores], y_tr)
        pc = m.predict(datos_ev[contadores]).astype(int)
        r = replicas_bootstrap(g, lambda idx: _ba(y_ev[idx], pc[idx], clases),
                               cfg.n_bootstrap, SEMILLA)
        res["solo_contadores"][n] = {"exactitud_balanceada": _ba(y_ev, pc, clases),
                                     "ic95": ic_percentil(r)}
        sal.prediccion(clave, f"solo contadores, {n}", _identidad(datos_ev), y_ev, pc)

    # importancia por permutación sobre el evaluado (RF)
    def puntuar(a, b):
        return balanced_accuracy_score(a, b)
    rf = modelos["random_forest"]
    imp = importancia_permutacion_evaluada(rf, X_ev, y_ev, puntuar, None,
                                           cfg.n_repeticiones_importancia, SEMILLA)
    imp_b = importancia_permutacion_evaluada(rf, X_ev, y_ev, puntuar, _bloques(columnas),
                                             cfg.n_repeticiones_importancia, SEMILLA)
    imp.insert(0, "tipo", "feature")
    imp_b.insert(0, "tipo", "bloque")
    for t in (imp, imp_b):
        t.insert(0, "analisis", clave)
        sal.importancias.append(t)
    por_feature = imp.groupby("nombre")["caida"].mean().sort_values(ascending=False)
    por_bloque = imp_b.groupby("nombre")["caida"].mean().sort_values(ascending=False)
    res["importancia"] = {"top10": por_feature.head(10), "bloques": por_bloque,
                          "reparto": importancia_por_bloque(por_feature)}
    print("Importancia por bloque (caída de exactitud balanceada, RF):")
    for n, v in por_bloque.items():
        print(f"    {n:<22s} {v:+.4f}")

    for n in ("logistica", "random_forest") + (("umbral",) if umbral else ()):
        r = res["modelos"][n]
        extra = (f" AUC {r['auc']:.4f}" if "auc" in r else "")
        p_txt = f" p = {r['p_permutacion']:.4f}" if "p_permutacion" in r else ""
        print(f"  {n:<14s} BA {r['exactitud_balanceada']:.4f} IC95 {_ic_texto(r['ic95'])}{extra}{p_txt}")
    return {"res": res, "modelos": modelos, "pred": pred, "proba": proba, "columnas": columnas,
            "X_ev": X_ev, "g": g, "por_feature": por_feature, "por_bloque": por_bloque}


def _banda_que_mas_cuesta(por_bloque):
    bandas = {k.replace("banda ", ""): v for k, v in por_bloque.items() if k.startswith("banda ")}
    return max(bandas, key=bandas.get), bandas


def analisis_oa_oc(datos, entrena, evalua, cfg, sal):
    _titulo("CONDICIÓN C1 — OA frente a OC")
    tr, _ = _preparar(preparar_reposo, datos, entrena)
    ev, meta = _preparar(preparar_reposo, datos, evalua)
    _inventario_condicion(sal, "C1 OA/OC", ev, meta)
    y_tr, y_ev = etiqueta_binaria(tr), etiqueta_binaria(ev)
    ctx = _evaluar_condicion("c1_oa_oc", tr, ev, y_tr, y_ev, [0, 1], modelo_logistico,
                             modelo_bosque, cfg, sal, umbral="posterior_alfa_rel_media")
    res = ctx["res"]
    banda, bandas = _banda_que_mas_cuesta(ctx["por_bloque"])
    top = list(ctx["por_feature"].index[:RANGO_ALFA_POSTERIOR])
    coef = _coeficientes(ctx["modelos"]["logistica"], ctx["columnas"])
    neto = float(coef[columnas_media_del_bloque(ctx["columnas"], "posterior", "alfa")].sum())
    d = _d_con_ic(ctx["X_ev"]["posterior_alfa_rel_media"].to_numpy(), y_ev, ctx["g"], cfg,
                  "c1_d_alfa_posterior", sal)
    fraccion_alfa = res["importancia"]["reparto"]["por_banda"].get("alfa", 0.0)
    res["coherencia"] = {
        "a_fraccion_alfa": {"valor": fraccion_alfa, "cumple": fraccion_alfa >= FRACCION_ALFA_MINIMA},
        "b_alfa_posterior_entre_las_5": {
            "top5": top, "cumple": any(bloque_de_feature(c) == ("posterior", "alfa") for c in top)},
        "c_alfa_es_la_banda_que_mas_cuesta": {"bandas": bandas, "cumple": banda == "alfa"},
        "d_direccion_hacia_oc": {
            "d_test": d, "cumple_test": d["d"] > 0 and d["excluye_0"],
            "modelo_coef_neto_alfa_posterior": neto,
            "modelo_signo_umbral": res["umbral"]["signo"],
            "modelo_cumple": neto > 0 and res["umbral"]["signo"] == 1}}
    _imprimir_coherencia(res["coherencia"])
    return res


def analisis_reposo_tarea(datos, entrena, evalua, cfg, sal):
    _titulo("CONDICIÓN C2 — reposo frente a tarea")
    tr, _ = _preparar(preparar_reposo_tarea, datos, entrena)
    ev, meta = _preparar(preparar_reposo_tarea, datos, evalua)
    _inventario_condicion(sal, "C2 reposo/tarea", ev, meta)
    y_tr, y_ev = etiqueta_tarea(tr), etiqueta_tarea(ev)
    ctx = _evaluar_condicion("c2_reposo_tarea", tr, ev, y_tr, y_ev, [0, 1], modelo_logistico,
                             modelo_bosque, cfg, sal, umbral="frontal_theta_rel_media",
                             clase_ap=0)
    res = ctx["res"]
    g = ctx["g"]

    # errores por condición, OA − OC, PVTB y posición en el protocolo
    pred_lr = ctx["pred"]["logistica"]
    res["por_condicion"] = {}
    for n in ("logistica", "random_forest"):
        res["por_condicion"][n] = {
            "desglose": desglose_por_condicion(ev, y_ev, ctx["pred"][n], g,
                                               n_replicas=cfg.n_bootstrap, semilla=SEMILLA),
            "oa_menos_oc": comparar_condiciones(ev, y_ev, ctx["pred"][n], g, "OA", "OC",
                                                n_replicas=cfg.n_bootstrap, semilla=SEMILLA)}
    tareas = ev["condicion"].isin(CONDICIONES_EJECUTIVAS).to_numpy()
    posicion = ev["condicion"].map(lambda c: ORDEN_PROTOCOLO.index(c) + 1).to_numpy()[tareas]
    acierto = (pred_lr == y_ev).astype(float)[tareas]
    rep = replicas_bootstrap(g[tareas], lambda idx: _spearman(posicion[idx], acierto[idx]),
                             cfg.n_bootstrap, SEMILLA)
    res["posicion_vs_acierto_logistica"] = {"rho": _spearman(posicion, acierto),
                                            "ic95": ic_percentil(rep)}

    # control sin OC
    tr_oa, _ = _preparar(preparar_reposo_tarea, datos, entrena, condiciones_reposo=("OA",))
    ev_oa, _ = _preparar(preparar_reposo_tarea, datos, evalua, condiciones_reposo=("OA",))
    yt_oa = etiqueta_tarea(tr_oa, condiciones_reposo=("OA",))
    ye_oa = etiqueta_tarea(ev_oa, condiciones_reposo=("OA",))
    cols = ctx["columnas"]
    lr_oa = modelo_logistico().fit(tr_oa[cols], yt_oa)
    p_oa = lr_oa.predict(ev_oa[cols]).astype(int)
    g_oa = ev_oa[COLUMNA_GRUPO].to_numpy()
    rep = replicas_bootstrap(g_oa, lambda idx: _ba(ye_oa[idx], p_oa[idx], [0, 1]),
                             cfg.n_bootstrap, SEMILLA)
    sal.modelo("c2_sin_oc_logistica", lr_oa)
    sal.prediccion("c2_sin_oc", "logistica", _identidad(ev_oa), ye_oa, p_oa,
                   lr_oa.predict_proba(ev_oa[cols]), [0, 1])
    res["sin_oc"] = {"exactitud_balanceada": _ba(ye_oa, p_oa, [0, 1]), "ic95": ic_percentil(rep),
                     "referencia": 0.816}
    print(f"Sin OC (logística): {res['sin_oc']['exactitud_balanceada']:.4f} "
          f"IC95 {_ic_texto(res['sin_oc']['ic95'])}")

    # coherencia
    banda, bandas = _banda_que_mas_cuesta(ctx["por_bloque"])
    coef = _coeficientes(ctx["modelos"]["logistica"], cols)
    coef_oa = _coeficientes(lr_oa, cols)
    neto = {(r, b): float(c[columnas_media_del_bloque(cols, r, b)].sum())
            for c in (coef,) for r, b in (("frontal", "theta"), ("posterior", "alfa"))}
    neto_oa = {(r, b): float(coef_oa[columnas_media_del_bloque(cols, r, b)].sum())
               for r, b in (("frontal", "theta"), ("posterior", "alfa"))}
    d_theta = _d_con_ic(ctx["X_ev"]["frontal_theta_rel_media"].to_numpy(), y_ev, g, cfg,
                        "c2_d_theta_frontal", sal)
    d_alfa = _d_con_ic(ctx["X_ev"]["posterior_alfa_rel_media"].to_numpy(), y_ev, g, cfg,
                       "c2_d_alfa_posterior", sal)
    d_theta_oa = _d_con_ic(ev_oa["frontal_theta_rel_media"].to_numpy(), ye_oa, g_oa, cfg,
                           "c2_sin_oc_d_theta_frontal", sal)
    d_alfa_oa = _d_con_ic(ev_oa["posterior_alfa_rel_media"].to_numpy(), ye_oa, g_oa, cfg,
                          "c2_sin_oc_d_alfa_posterior", sal)
    signo_umbral = res["umbral"]["signo"]
    sube = lambda d: d["d"] > 0 and d["excluye_0"]  # noqa: E731
    baja = lambda d: d["d"] < 0 and d["excluye_0"]  # noqa: E731
    res["coherencia"] = {
        "a_banda_que_mas_cuesta_theta_o_alfa": {"banda": banda, "bandas": bandas,
                                                "cumple": banda in ("theta", "alfa")},
        "b_theta_frontal_hacia_tarea": {
            "d_test": d_theta, "cumple_test": sube(d_theta),
            "modelo_coef_neto": neto[("frontal", "theta")], "modelo_signo_umbral": signo_umbral,
            "modelo_cumple": neto[("frontal", "theta")] > 0 and signo_umbral == 1},
        "c_alfa_posterior_hacia_reposo": {
            "d_test": d_alfa, "cumple_test": baja(d_alfa),
            "modelo_coef_neto": neto[("posterior", "alfa")],
            "modelo_cumple": neto[("posterior", "alfa")] < 0},
        "d_sin_oc": {
            "d_test_theta": d_theta_oa, "d_test_alfa": d_alfa_oa,
            "cumple_test": sube(d_theta_oa) and baja(d_alfa_oa),
            "modelo_coef_neto_theta": neto_oa[("frontal", "theta")],
            "modelo_coef_neto_alfa": neto_oa[("posterior", "alfa")],
            "modelo_cumple": neto_oa[("frontal", "theta")] > 0 and neto_oa[("posterior", "alfa")] < 0}}
    _imprimir_coherencia(res["coherencia"])
    return res


def analisis_tareas(datos, entrena, evalua, cfg, sal):
    _titulo("CONDICIÓN C3 — las cinco tareas ejecutivas")
    tr, _ = _preparar(preparar_tareas_ejecutivas, datos, entrena)
    ev, meta = _preparar(preparar_tareas_ejecutivas, datos, evalua)
    _inventario_condicion(sal, "C3 tareas", ev, meta)
    y_tr, y_ev = etiqueta_multiclase(tr), etiqueta_multiclase(ev)
    clases = list(range(len(CONDICIONES_EJECUTIVAS)))
    ctx = _evaluar_condicion("c3_tareas", tr, ev, y_tr, y_ev, clases,
                             modelo_logistico_multinomial, modelo_bosque_multiclase, cfg, sal,
                             binaria=False)
    res = ctx["res"]
    pred_lr = ctx["pred"]["logistica"]
    res["matriz_confusion_normalizada"] = {
        n: matriz_confusion_normalizada(y_ev, ctx["pred"][n]) for n in ("logistica", "random_forest")}
    pares = confusion_por_pares(y_ev, pred_lr).sort_values("confusion_simetrica", ascending=False)
    estructura = estructura_de_la_confusion(pares, distancias_de_duracion(tr))
    exhaustividad = {c: float(np.mean(pred_lr[y_ev == i] == i))
                     for i, c in enumerate(CONDICIONES_EJECUTIVAS)}
    top3 = [set(par.split("-")) for par in pares.index[:3]]
    efectos = tamanos_de_efecto_una_contra_resto(ctx["X_ev"], y_ev)
    theta = columnas_media_del_bloque(ctx["columnas"], "frontal", "theta")
    d_theta = {c: float(efectos[c][theta].mean()) for c in CONDICIONES_EJECUTIVAS}
    coef = _coeficientes(ctx["modelos"]["logistica"], ctx["columnas"], list(CONDICIONES_EJECUTIVAS))
    neto = coef.loc[theta].sum()
    ic_lr, ic_rf = res["modelos"]["logistica"]["ic95"], res["modelos"]["random_forest"]["ic95"]
    res["pares"] = pares
    res["estructura_confusion"] = estructura
    res["exhaustividad_logistica"] = exhaustividad
    res["coherencia"] = {
        "a_los_dos_superan_el_azar": {"ic95_logistica": ic_lr, "ic95_random_forest": ic_rf,
                                      "cumple": ic_lr[0] > 0.2 and ic_rf[0] > 0.2},
        "b_gemelos_entre_los_3_mas_confundidos": {
            "top3": list(pares.index[:3]), "cumple": all(gm in top3 for gm in GEMELOS)},
        "c_pvtb_mayor_exhaustividad": {"exhaustividad": exhaustividad,
                                       "cumple": max(exhaustividad, key=exhaustividad.get) == "PVTB"},
        "d_theta_frontal_pasat_pvsat_sobre_pvtb": {
            "d_test": d_theta,
            "cumple_test": d_theta["PASAT"] > d_theta["PVTB"] and d_theta["PVSAT"] > d_theta["PVTB"],
            "modelo_coef_neto": neto,
            "modelo_cumple": bool(neto["PASAT"] > neto["PVTB"] and neto["PVSAT"] > neto["PVTB"])},
        "e_confusion_sigue_al_parecido_cognitivo": {
            "rho_similitud": estructura["confusion_vs_similitud_cognitiva"],
            "rho_protocolo": estructura["confusion_vs_distancia_protocolo"],
            "rho_duracion": estructura.get("confusion_vs_distancia_duracion"),
            "cumple": (estructura["confusion_vs_similitud_cognitiva"] > 0
                       and abs(estructura["confusion_vs_similitud_cognitiva"])
                       > abs(estructura["confusion_vs_distancia_protocolo"]))}}
    print("Pares más confundidos (logística): " + ", ".join(
        f"{p} {v:.3f}" for p, v in pares["confusion_simetrica"].head(4).items()))
    _imprimir_coherencia(res["coherencia"])
    return res


def _inventario_condicion(sal, analisis, ev, meta):
    for _, fila in ev.iterrows():
        sal.inventario.append({"familia": "condición", "unidad": "grabacion",
                               "analisis": analisis, "id": fila.get("archivo"),
                               "persona_uid": fila[COLUMNA_GRUPO], "estado": "usada"})
    if meta.get("n_sin_persona_uid"):
        sal.inventario.append({"familia": "condición", "unidad": "grabacion",
                               "analisis": analisis, "id": None, "persona_uid": None,
                               "estado": f"descartadas sin persona_uid: {meta['n_sin_persona_uid']}"})


def _imprimir_coherencia(coherencia):
    print("Criterios de coherencia:")
    for nombre, c in coherencia.items():
        partes = []
        if "cumple" in c:
            partes.append(f"test {'SÍ' if c['cumple'] else 'NO'}")
        if "cumple_test" in c:
            partes.append(f"test {'SÍ' if c['cumple_test'] else 'NO'}")
        if "modelo_cumple" in c:
            partes.append(f"modelo final {'sí' if c['modelo_cumple'] else 'no'} (se reporta)")
        print(f"  {nombre:<44s} " + " | ".join(partes))


def familia_condicion(datos, entrena, evalua, cfg, sal):
    res = {"C1_oa_oc": analisis_oa_oc(datos, entrena, evalua, cfg, sal),
           "C2_reposo_tarea": analisis_reposo_tarea(datos, entrena, evalua, cfg, sal),
           "C3_tareas": analisis_tareas(datos, entrena, evalua, cfg, sal)}
    _titulo("FAMILIA DE CONDICIÓN — Holm entre las 3 pruebas (logística)")
    claves = list(res)
    p = [res[k]["modelos"]["logistica"]["p_permutacion"] for k in claves]
    for k, pa in zip(claves, holm(p)):
        r = res[k]["modelos"]["logistica"]
        r["p_holm"] = float(pa)
        r["veredicto"] = "confirma" if pa < ALFA else "no confirma"
        print(f"  {k:<16s} BA {r['exactitud_balanceada']:.4f} p = {r['p_permutacion']:.4f} "
              f"p Holm = {pa:.4f}  {r['veredicto'].upper()}")
    return res


# --------------------------------------------------------------------------
# Metadatos y ejecución
# --------------------------------------------------------------------------

def metadatos(modo, entrena, evalua, cfg, git, inicio):
    def ejecutar(*args):
        try:
            return subprocess.run(args, capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError) as e:
            return f"no disponible: {e}"
    cpu = ejecutar("powershell", "-NoProfile", "-Command",
                   "(Get-CimInstance Win32_Processor).Name") if os.name == "nt" else platform.processor()
    return {
        "modo": modo, "entrena": entrena, "evalua": evalua,
        "inicio": inicio.isoformat(timespec="seconds"),
        "comando": " ".join([sys.executable] + sys.argv), "EEG_DATA": str(RUTA_DATOS),
        "git": git, "sha256_guion": sha256_archivo(Path(__file__)),
        "preregistro": {"ruta": str(RUTA_PREREGISTRO),
                        "sha256": sha256_archivo(RUTA_PREREGISTRO) if RUTA_PREREGISTRO.exists() else None},
        "datos_sha256": {n: sha256_archivo(r) if r.exists() else None for n, r in ARCHIVOS_DATOS.items()},
        "entorno": {"python": sys.version, "plataforma": platform.platform(),
                    "cpu": cpu.strip(), "nucleos": os.cpu_count(), "n_jobs": -1,
                    "pip_freeze": ejecutar(sys.executable, "-m", "pip", "freeze").splitlines()},
        "parametros": {"semilla": SEMILLA, "alfa": ALFA, "n_bootstrap": cfg.n_bootstrap,
                       "n_permutaciones": cfg.n_permutaciones,
                       "n_repeticiones_importancia": cfg.n_repeticiones_importancia,
                       "margen_r2": MARGEN_R2, "margen_mae": MARGEN_MAE,
                       "referencia_principal": REFERENCIA_PRINCIPAL},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    modo = parser.add_mutually_exclusive_group(required=True)
    modo.add_argument("--ensayo", action="store_true")
    modo.add_argument("--confirmatorio", action="store_true")
    parser.add_argument("--rapido", action="store_true")
    args = parser.parse_args()
    if args.rapido and args.confirmatorio:
        parser.error("--rapido solo vale con --ensayo.")

    inicio = dt.datetime.now()
    git = estado_git(RAIZ_REPO)
    if args.confirmatorio:
        entrena, evalua, nombre_modo = ("train", "val"), ("test",), "confirmatorio"
        if not git["limpio"]:
            sys.exit("El árbol de git no está limpio: el confirmatorio exige que el guion y "
                     "el pre-registro estén commiteados. Cambios:\n  " + "\n  ".join(git["cambios"]))
        previas = sorted((RUTA_DATOS / "processed").glob("test_*"))
        if previas:
            sys.exit(f"Ya existe una ejecución confirmatoria: {previas}. El test se gasta una vez.")
        carpeta = RUTA_DATOS / "processed" / f"test_{inicio:%Y%m%d-%H%M%S}"
    else:
        entrena, evalua = ("train",), ("val",)
        nombre_modo = "ensayo rápido" if args.rapido else "ensayo"
        base = Path(tempfile.gettempdir()) if args.rapido else RUTA_DATOS / "processed"
        carpeta = base / f"ensayo_val_{inicio:%Y%m%d-%H%M%S}"
    carpeta_salida_nueva(carpeta)
    if hasattr(sys.__stdout__, "reconfigure"):
        sys.__stdout__.reconfigure(encoding="utf-8")
    log = open(carpeta / "ejecucion.log", "w", encoding="utf-8")
    sys.stdout = Duplicador(sys.__stdout__, log)
    cfg = Config(args.rapido)
    sal = Artefactos(carpeta)
    t0 = time.time()

    _titulo(f"ANÁLISIS CONFIRMATORIO — modo {nombre_modo.upper()}")
    print(f"Entrena con {entrena}; evalúa {evalua}")
    print(f"Salida: {carpeta}")
    print(f"Git: {git['commit'][:12]}  limpio = {git['limpio']}")
    if args.ensayo:
        print("ENSAYO: las cifras de esta ejecución NO son resultados (val ya se usó en la CV).")
    meta = metadatos(nombre_modo, entrena, evalua, cfg, git, inicio)

    matriz = pd.read_parquet(RUTA_MATRIZ)
    datos = matriz[matriz["split"].isin(entrena + evalua)].reset_index(drop=True)
    print(f"Grabaciones leídas: {len(datos)} de los splits {entrena + evalua} "
          f"(se descartan {len(matriz) - len(datos)} de otros splits sin usarlas)")
    del matriz

    resultados = {"metadatos": meta}
    resultados["familia_vas"] = familia_vas(datos, entrena, evalua, cfg, sal)
    resultados["familia_condicion"] = familia_condicion(datos, entrena, evalua, cfg, sal)

    meta["fin"] = dt.datetime.now().isoformat(timespec="seconds")
    meta["duracion_s"] = round(time.time() - t0, 1)
    sal.escribir(resultados)
    _titulo("RESUMEN")
    vas = resultados["familia_vas"]
    print(f"Prueba principal: {vas['principal']['exactitud_balanceada']:.4f} IC95 "
          f"{_ic_texto(vas['principal']['ic95'])} p = {vas['principal']['p']:.4f} -> "
          f"{vas['principal']['veredicto'].upper()}")
    print(f"Duración: {meta['duracion_s']:.0f} s. Artefactos en {carpeta}")
    log.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
