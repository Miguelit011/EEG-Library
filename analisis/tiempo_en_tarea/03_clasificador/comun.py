"""
Tiempo en tarea — paso 3: piezas comunes a `seleccion.py` y `comprobacion_val.py`.

Carga por partición (nunca `test`), recorte a igual tiempo, features por
tercio de las cuatro familias candidatas, modelos candidatos, pliegues
repetidos por persona y acierto pareado. Reutiliza las funciones de los pasos
1 y 2, que no se modifican.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent / "01_descriptivo"))
import anexo_igual_tiempo as anexo  # noqa: E402
import descriptivo_tiempo_en_tarea as d1  # noqa: E402

from eeg_library.modelado.reposo_tarea import modelo_bosque  # noqa: E402
from eeg_library.procesado.features import anotar_validez_de_canales  # noqa: E402
from eeg_library.procesado.matrix import REGIONES  # noqa: E402

FAMILIA = ["PASAT", "PVTB"]
CONDICIONES = ["PASAT", "PVTB", "PVSAT", "OA", "OC"]
N_PLIEGUES = 5
SEMILLAS = list(range(20))
TOLERANCIA_EMPATE = 0.01

REL, ABS = d1.REL, d1.ABS
REG = [f"{r}_{b}_{t}" for r in REGIONES for b in d1.BANDAS for t in ("abs", "rel")]
FEATURES = {"regiones": REG, "rel": REL, "abs": ABS, "rel+abs": REL + ABS}
MODELOS = ["lr_0.01", "lr_0.1", "lr_1", "rf"]
ESQUEMAS = ["separado", "conjunto"]
# Orden de simplicidad para el desempate (menor = más simple).
SIMPLICIDAD_FEATURES = {"regiones": 0, "rel": 1, "abs": 1, "rel+abs": 2}
SIMPLICIDAD_MODELO = {"lr_0.01": 0, "lr_0.1": 1, "lr_1": 2, "rf": 3}
SIMPLICIDAD_ESQUEMA = {"separado": 0, "conjunto": 1}


def modelo(nombre):
    if nombre == "rf":
        return modelo_bosque()
    c = float(nombre.split("_")[1])
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))


def modelo_regresion(nombre):
    if nombre == "rf":
        return RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=d1.SEMILLA, n_jobs=-1)
    c = float(nombre.split("_")[1])
    return make_pipeline(StandardScaler(), Ridge(alpha=1 / c))


# --- Carga -------------------------------------------------------------------


def cargar(particiones):
    """Como `d1.cargar`, pero para las particiones pedidas. Nunca carga `test`."""
    particiones = set(particiones)
    assert "test" not in particiones, "Ningún script de los pasos 1–5 carga test"
    matriz = pd.read_parquet(d1.RUTA_MATRIZ, columns=["archivo", "persona_uid", "sesion_uid", "split"])
    etiquetadas = matriz[matriz.persona_uid.notna() & matriz.split.isin(particiones)]
    archivos = sorted(etiquetadas.archivo)
    columnas = ["condicion", "archivo_origen", "indice_ventana", "tmin_s"] + REL + ABS
    w = pd.read_parquet(d1.RUTA_FEATURES, columns=columnas, filters=[("archivo_origen", "in", archivos)])
    assert set(w.archivo_origen) <= set(archivos)
    w = w.merge(etiquetadas[["archivo", "persona_uid", "sesion_uid", "split"]],
                left_on="archivo_origen", right_on="archivo").drop(columns="archivo")
    assert set(w.split) <= particiones
    log = pd.read_csv(d1.RUTA_LOG)
    mascara = anotar_validez_de_canales(pd.DataFrame({"archivo_origen": w.archivo_origen.unique()}), log)
    w = w.merge(mascara[["archivo_origen"] + d1.VALIDEZ], on="archivo_origen")
    w["t_rec"] = w.tmin_s / 60
    w["n_vent"] = w.groupby("archivo_origen").tmin_s.transform("size")
    w["dur"] = w.groupby("archivo_origen").t_rec.transform("max")
    return w[w.condicion.isin(CONDICIONES) & (w.n_vent >= d1.MIN_VENTANAS)].reset_index(drop=True)


def versiones(w):
    """Grabación completa e igual tiempo (tareas recortadas al OA de su sesión; OA y OC intactos)."""
    recortadas, _ = anexo.recortar(w)
    igual = pd.concat([recortadas, w[w.condicion == "OC"]], ignore_index=True)
    igual = igual[igual.n_vent >= d1.MIN_VENTANAS].reset_index(drop=True)
    return {"igual_tiempo": igual, "completa": w}


# --- Features ----------------------------------------------------------------


def _por_segmento(g, claves):
    """Media por segmento de las potencias por canal, features en log10 y máscaras."""
    S = g.groupby(claves)[REL + ABS].mean().reset_index()
    validez = g.drop_duplicates("archivo_origen").set_index("archivo_origen")[d1.VALIDEZ]
    canal_ok = validez.loc[S.archivo_origen].to_numpy(dtype=bool)  # (n, 14)
    salida = S[claves].copy()
    for pref, cols in (("pot_rel", REL), ("pot_abs", ABS)):
        salida[cols] = np.log10(S[cols].clip(lower=1e-30 if pref == "pot_abs" else 1e-6)).to_numpy()
    for region, canales in REGIONES.items():
        idx = [d1.CANALES_EPOC.index(c) for c in canales]
        ok = canal_ok[:, idx]
        n_ok = ok.sum(axis=1)
        for b in d1.BANDAS:
            for t, pref in (("abs", "pot_abs"), ("rel", "pot_rel")):
                v = S[[f"{pref}_{c}_{b}" for c in canales]].to_numpy(float)
                media = np.where(n_ok > 0, (v * ok).sum(axis=1) / np.maximum(n_ok, 1), np.nan)
                salida[f"{region}_{b}_{t}"] = np.log10(np.clip(media, 1e-30, None))
    mascaras = {
        "rel": np.repeat(canal_ok, len(d1.BANDAS), axis=1),
        "abs": np.repeat(canal_ok, len(d1.BANDAS), axis=1),
    }
    mascaras["rel+abs"] = np.hstack([mascaras["rel"], mascaras["abs"]])
    mascaras["regiones"] = salida[REG].notna().to_numpy()
    return salida, mascaras


def tercios(w, cond):
    g = w[w.condicion == cond]
    seg = np.where(g.t_rec <= g.dur / 3, 0, np.where(g.t_rec >= 2 * g.dur / 3, 1, -1))
    g = g[seg >= 0].assign(seg=seg[seg >= 0])
    S, mascaras = _por_segmento(g, ["archivo_origen", "persona_uid", "seg"])
    return S, mascaras


def bloques(w, cond, segundos=30):
    """Bloques de `segundos` para la regresión del minuto; `t_bloque` en minutos (centro del bloque)."""
    g = w[w.condicion == cond]
    g = g.assign(bloque=(g.tmin_s // segundos).astype(int))
    # Solo bloques completos: los de cola tienen pocas ventanas.
    n = g.groupby(["archivo_origen", "bloque"]).tmin_s.transform("size")
    g = g[n >= segundos // 2 - 1]
    S, mascaras = _por_segmento(g, ["archivo_origen", "persona_uid", "bloque"])
    S["t_bloque"] = (S.bloque + 0.5) * segundos / 60
    return S, mascaras


def matriz(S, mascaras, features):
    """Features centradas por grabación, con canales (o regiones) inválidos a 0."""
    cols = FEATURES[features]
    F = S[cols] - S[cols].groupby(S.archivo_origen).transform("mean")
    F = F.mask(~mascaras[features], 0.0).fillna(0.0)
    return F.to_numpy(float)


# --- Pliegues y métricas -----------------------------------------------------


def pliegues_por_persona(personas, semilla):
    """persona_uid -> pliegue, equilibrado y reproducible con la semilla."""
    orden = np.random.default_rng(semilla).permutation(np.array(sorted(personas)))
    return {p: i % N_PLIEGUES for i, p in enumerate(orden)}


def aciertos_por_grabacion(S, p):
    """Acierto pareado por grabación (empate = 0.5) a partir de la probabilidad de cada tercio."""
    par = S.assign(p=p).pivot_table(index=["archivo_origen", "persona_uid"], columns="seg",
                                    values="p").dropna().reset_index()
    par["acierto"] = np.where(par[1] > par[0], 1.0, np.where(par[1] == par[0], 0.5, 0.0))
    return par[["archivo_origen", "persona_uid", "acierto"]]
