"""
Fatiga aguda — deep learning, paso 4: piezas comunes a `preparar.py` y
`evaluar.py` (entorno `.venv`).

Carga con el cargador del paso 6 (`06_test/evaluacion_test.py`), que solo lee
`test` con permiso explícito; tensor de ventanas del piloto
(`02_piloto_cnn/preparar.py`); y las cifras del preregistro (P1, S1,
descriptivos y sensibilidades), que se calculan igual en `val` (fase A) y en
`test` (fase D). Ninguna de las funciones reutilizadas se modifica.
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CARPETA = Path(__file__).resolve().parent
DL = CARPETA.parent
LINEA = DL.parent
RAIZ_REPO = LINEA.parent.parent
sys.path.insert(0, str(LINEA / "06_test"))
import evaluacion_test as et  # noqa: E402
from evaluacion_test import FAMILIA, comun, d1, md  # noqa: E402


def cargar_modulo(nombre, ruta):
    """Importa un script por su ruta. Los del piloto se llaman igual que los de este paso."""
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = modulo
    spec.loader.exec_module(modulo)
    return modulo


piloto = cargar_modulo("piloto_preparar", DL / "02_piloto_cnn" / "preparar.py")

from eeg_library.modelado.confirmatorio import (  # noqa: E402
    holm, ic_percentil, p_bootstrap, replicas_bootstrap, sha256_archivo,
)

CACHE = d1.EEG_DATA.parent / "EEG-DL-cache" / "test_eegnet"
CONFIGURACION = CARPETA / "configuracion_dl.json"
PREREGISTRO = CARPETA / "PREREGISTRO.md"
SEMILLAS = [0, 1, 2]
CONDICIONES_CNN = FAMILIA + ["OA"]
N_BOOT = et.N_BOOT
SEMILLA_BOOT = et.SEMILLA
ALFA = et.ALFA
MAX_DIFERENCIA_EMPAREJAMIENTO = 2


class Tee(et.Tee):
    """El `Tee` del paso 6, que además crea la carpeta del registro."""

    def __init__(self, ruta):
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        super().__init__(ruta)


def git_limpio():
    """Sale del programa si el repositorio tiene cambios sin commitear."""
    salida = subprocess.run(["git", "status", "--porcelain"], cwd=RAIZ_REPO, capture_output=True, text=True,
                            check=True).stdout
    cambios = [l for l in salida.splitlines() if l.strip()]
    if cambios:
        sys.exit("El árbol de git no está limpio: el preregistro, los scripts y configuracion_dl.json deben estar "
                 "commiteados.\n  " + "\n  ".join(cambios))
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ_REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def json_seguro(ruta, datos):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))


# --- Ventanas ------------------------------------------------------------------


def ventanas_cnn(w, particiones_oa):
    """PASAT y PVTB de todas las particiones de `w`, y OA solo de las pedidas, ordenadas por grabación."""
    sel = w.condicion.isin(FAMILIA) | ((w.condicion == "OA") & w.split.isin(particiones_oa))
    v = w[sel].sort_values(["archivo_origen", "indice_ventana"]).reset_index(drop=True)
    v["seg"] = piloto.segmento(v)
    return v


def marcar_igual_tiempo(v, w_evalua):
    """Columna `igual`: la ventana de tarea cae dentro del recorte a igual tiempo (`comun.versiones`)."""
    igual = comun.versiones(w_evalua)["igual_tiempo"]
    claves = set(zip(igual.archivo_origen[igual.condicion.isin(FAMILIA)],
                     igual.indice_ventana[igual.condicion.isin(FAMILIA)]))
    return np.array([(a, i) in claves for a, i in zip(v.archivo_origen, v.indice_ventana)])


def integridad_npy(v):
    """Comprueba que cada `.npy` tiene tantas ventanas como filas en `features_all`; hash combinado."""
    hashes, desalineados = [], []
    n = v.groupby("archivo_origen").n_vent.first()
    for a in sorted(n.index):
        ruta = piloto.p1.RUTA_VENTANAS / a.replace(".edf", ".npy")
        x = np.load(ruta, mmap_mode="r")
        if x.shape[0] != n[a] or x.shape[1:] != (14, 1024):
            desalineados.append(a)
        hashes.append(f"{ruta.name}:{d1.sha256(ruta)}")
    return desalineados, hashlib.sha256("\n".join(hashes).encode()).hexdigest(), len(n)


def guardar_tensor(v, carpeta):
    """Tensor a 128 Hz y máscara (función del piloto) y tabla de ventanas, en la caché."""
    X, mascara = piloto.tensor(v)
    assert not np.isnan(X).any()
    carpeta.mkdir(parents=True, exist_ok=True)
    np.save(carpeta / "X.npy", X)
    np.save(carpeta / "mascara.npy", mascara)
    cols = ["archivo_origen", "persona_uid", "condicion", "indice_ventana", "t_rec", "seg", "papel", "igual"]
    v[cols].to_parquet(carpeta / "ventanas.parquet", index=False)
    return X.shape, float(X.nbytes / 1e9), float(1 - mascara.mean())


# --- Medidas de EEGNet -----------------------------------------------------------


def rho_cnn(v, punt):
    """ρ por grabación de una semilla: versión completa (PASAT, PVTB, OA) e igual tiempo (PASAT, PVTB)."""
    filas = v[["archivo_origen", "persona_uid", "t_rec"]]
    completa = md.rho_por_grabacion(filas, punt).assign(version="completa")
    sel = v.igual.to_numpy()
    igual = md.rho_por_grabacion(filas[sel], punt[sel]).assign(version="igual_tiempo")
    oa = completa[completa.archivo_origen.isin(v.archivo_origen[v.condicion == "OA"])]
    r = pd.concat([completa, igual, oa.assign(version="igual_tiempo")], ignore_index=True)
    return r.merge(v.drop_duplicates("archivo_origen")[["archivo_origen", "condicion"]])


def tercios_cnn(v, punt):
    """Acierto pareado de tercios por grabación (media de la puntuación de cada tercio; empate = 0.5)."""
    d = v.assign(s=punt)
    d = d[d.seg >= 0].groupby(["archivo_origen", "persona_uid", "seg"]).s.mean().unstack().dropna()
    return pd.Series(np.where(d[1] > d[0], 1.0, np.where(d[1] == d[0], 0.5, 0.0)), index=d.index.get_level_values(0))


def tabla_por_grabacion(v, puntuaciones, rho_lr):
    """Una fila por grabación y versión: ρ de cada semilla, su media (`cnn`), el conjunto (`ens`) y la logística."""
    por_semilla = {s: rho_cnn(v, p) for s, p in puntuaciones.items()}
    base = por_semilla[SEMILLAS[0]][["archivo_origen", "persona_uid", "condicion", "version"]].copy()
    clave = ["archivo_origen", "version"]
    for s, r in por_semilla.items():
        base = base.merge(r[clave + ["acierto"]].rename(columns={"acierto": f"cnn_s{s}"}), on=clave, how="left")
    base["cnn"] = base[[f"cnn_s{s}" for s in puntuaciones]].mean(axis=1)
    z = np.mean([(p - p.mean()) / p.std() for p in puntuaciones.values()], axis=0)
    base = base.merge(rho_cnn(v, z)[clave + ["acierto"]].rename(columns={"acierto": "ens"}), on=clave, how="left")
    tareas = v.condicion.isin(FAMILIA).to_numpy()
    terc = pd.concat([tercios_cnn(v[tareas], p[tareas]) for p in puntuaciones.values()], axis=1).mean(axis=1)
    base["tercios"] = np.where(base.version == "completa", base.archivo_origen.map(terc), np.nan)

    lr = rho_lr[["archivo_origen", "version", "acierto"]].rename(columns={"acierto": "lr"})
    tabla = base.merge(lr, on=clave, how="left")
    # Emparejamiento de la P1: grabaciones de tarea (completa) con ρ de las dos fuentes.
    cnn_t = set(base.archivo_origen[(base.version == "completa") & base.condicion.isin(FAMILIA)])
    lr_t = set(rho_lr.archivo_origen[(rho_lr.version == "completa") & rho_lr.condicion.isin(FAMILIA)])
    solo = {"solo_cnn": sorted(cnn_t - lr_t), "solo_logistica": sorted(lr_t - cnn_t)}
    n_dif = len(solo["solo_cnn"]) + len(solo["solo_logistica"])
    if n_dif > MAX_DIFERENCIA_EMPAREJAMIENTO:
        raise RuntimeError(f"El emparejamiento difiere en {n_dif} grabaciones (máximo "
                           f"{MAX_DIFERENCIA_EMPAREJAMIENTO}): se para antes de calcular nada. {solo}")
    return tabla, por_semilla, solo


def auc_cnn(v, puntuaciones):
    """AUC por ventana (primer frente a último tercio de PASAT + PVTB), media de las semillas."""
    sel = (v.seg >= 0).to_numpy() & v.condicion.isin(FAMILIA).to_numpy()
    return float(np.mean([roc_auc_score(v.seg[sel], p[sel]) for p in puntuaciones.values()]))


# --- Estadísticos y bootstrap ----------------------------------------------------


def _media(t, idx, version, conds, col):
    sel = idx[(t.version.to_numpy()[idx] == version) & np.isin(t.condicion.to_numpy()[idx], conds)]
    x = t[col].to_numpy()[sel]
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("remuestra sin grabaciones")
    return x.mean()


def _media_dif(t, idx, version, conds, a, b):
    sel = idx[(t.version.to_numpy()[idx] == version) & np.isin(t.condicion.to_numpy()[idx], conds)]
    x = t[a].to_numpy()[sel] - t[b].to_numpy()[sel]
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("remuestra sin grabaciones")
    return x.mean()


# Nombre → función del estadístico sobre los índices de fila de la remuestra.
ESTADISTICOS = {
    "P1": lambda t, i: _media_dif(t, i, "completa", FAMILIA, "cnn", "lr"),
    "S1_PASAT": lambda t, i: _media(t, i, "completa", ["PASAT"], "cnn"),
    "S1_PVTB": lambda t, i: _media(t, i, "completa", ["PVTB"], "cnn"),
    "dif_PASAT": lambda t, i: _media_dif(t, i, "completa", ["PASAT"], "cnn", "lr"),
    "dif_PVTB": lambda t, i: _media_dif(t, i, "completa", ["PVTB"], "cnn", "lr"),
    "cnn_familia": lambda t, i: _media(t, i, "completa", FAMILIA, "cnn"),
    "lr_familia": lambda t, i: _media(t, i, "completa", FAMILIA, "lr"),
    "lr_PASAT": lambda t, i: _media(t, i, "completa", ["PASAT"], "lr"),
    "lr_PVTB": lambda t, i: _media(t, i, "completa", ["PVTB"], "lr"),
    "igual_cnn_PASAT": lambda t, i: _media(t, i, "igual_tiempo", ["PASAT"], "cnn"),
    "igual_cnn_PVTB": lambda t, i: _media(t, i, "igual_tiempo", ["PVTB"], "cnn"),
    "igual_cnn_familia": lambda t, i: _media(t, i, "igual_tiempo", FAMILIA, "cnn"),
    "igual_lr_familia": lambda t, i: _media(t, i, "igual_tiempo", FAMILIA, "lr"),
    "igual_dif_familia": lambda t, i: _media_dif(t, i, "igual_tiempo", FAMILIA, "cnn", "lr"),
    "cnn_OA": lambda t, i: _media(t, i, "completa", ["OA"], "cnn"),
    "cnn_tareas_menos_OA": lambda t, i: (_media(t, i, "completa", FAMILIA, "cnn")
                                         - _media(t, i, "completa", ["OA"], "cnn")),
    "igual_cnn_tareas_menos_OA": lambda t, i: (_media(t, i, "igual_tiempo", FAMILIA, "cnn")
                                               - _media(t, i, "igual_tiempo", ["OA"], "cnn")),
    "tercios_cnn_familia": lambda t, i: _media(t, i, "completa", FAMILIA, "tercios"),
    "ens_familia": lambda t, i: _media(t, i, "completa", FAMILIA, "ens"),
    "ens_menos_lr": lambda t, i: _media_dif(t, i, "completa", FAMILIA, "ens", "lr"),
}
PRINCIPALES = ["P1", "S1_PASAT", "S1_PVTB"]


def bootstrap(tabla, nombres):
    """Réplicas pareadas (mismas personas en todos los estadísticos) y valores observados."""
    t = tabla.reset_index(drop=True)

    def estadistico(idx):
        return np.array([ESTADISTICOS[n](t, idx) for n in nombres])

    observado = estadistico(np.arange(len(t)))
    reps = replicas_bootstrap(t.persona_uid.to_numpy(), estadistico, n_replicas=N_BOOT, semilla=SEMILLA_BOOT)
    return observado, reps


def p_bilateral(reps):
    r = reps[np.isfinite(reps)]
    return float(min(1.0, 2 * min(np.sum(r <= 0) + 1, np.sum(r >= 0) + 1) / (len(r) + 1)))


def veredicto_p1(ic):
    if ic[0] > 0:
        return "EEGNET SUPERA"
    if ic[1] < 0:
        return "EEGNET QUEDA POR DEBAJO"
    return "NO SE DISTINGUE"


def analizar(tabla, expuestas, a_128):
    """Todas las cifras del preregistro (secciones 5 y 6) sobre una tabla por grabación."""
    nombres = list(ESTADISTICOS)
    obs, reps = bootstrap(tabla, nombres)
    res = {}
    for i, n in enumerate(nombres):
        res[n] = {"valor": float(obs[i]), "ic95": ic_percentil(reps[:, i]),
                  "replicas_validas": int(np.isfinite(reps[:, i]).sum())}
    p1 = res["P1"]
    p1["p_bilateral"] = p_bilateral(reps[:, 0])
    p1["veredicto"] = veredicto_p1(p1["ic95"])
    p1["dentro_de_0.02"] = bool(p1["ic95"][0] > -0.02 and p1["ic95"][1] < 0.02)
    p_s1 = [p_bootstrap(reps[:, nombres.index(n)], 0, "menor_o_igual") for n in ("S1_PASAT", "S1_PVTB")]
    for n, p, p_aj in zip(("S1_PASAT", "S1_PVTB"), p_s1, holm(p_s1)):
        res[n].update({"p_unilateral": float(p), "p_holm": float(p_aj),
                       "veredicto": "CONFIRMA" if p_aj < ALFA else "NO CONCLUYE"})

    sensibilidades = {}
    for etiqueta, sel in (("sin_personas_expuestas", ~tabla.persona_uid.isin(expuestas)),
                          ("sin_grabaciones_128_hz", ~tabla.archivo_origen.isin(a_128))):
        o, r = bootstrap(tabla[sel.to_numpy()], PRINCIPALES)
        sensibilidades[etiqueta] = {n: {"valor": float(o[i]), "ic95": ic_percentil(r[:, i])}
                                    for i, n in enumerate(PRINCIPALES)}
        sensibilidades[etiqueta]["P1"]["veredicto_que_tendria"] = veredicto_p1(sensibilidades[etiqueta]["P1"]["ic95"])
    return res, reps, sensibilidades


def imprimir(res, sensibilidades, extra):
    p1 = res["P1"]
    d1._titulo("PRINCIPAL")
    print(f"  P1 Δρ (EEGNet − logística), PASAT + PVTB, completa: {p1['valor']:+.4f}  "
          f"IC95 [{p1['ic95'][0]:+.4f}, {p1['ic95'][1]:+.4f}]  p bilateral = {p1['p_bilateral']:.4f}  "
          f"→ {p1['veredicto']}")
    d1._titulo("SECUNDARIA")
    for n in ("S1_PASAT", "S1_PVTB"):
        x = res[n]
        print(f"  {n:9s} ρ = {x['valor']:+.4f}  IC95 [{x['ic95'][0]:+.4f}, {x['ic95'][1]:+.4f}]  "
              f"p = {x['p_unilateral']:.4f}, Holm {x['p_holm']:.4f}  → {x['veredicto']}")
    d1._titulo("DESCRIPTIVOS (sin veredicto)")
    for n, x in res.items():
        if n not in PRINCIPALES:
            print(f"  {n:28s} {x['valor']:+.4f} [{x['ic95'][0]:+.4f}, {x['ic95'][1]:+.4f}]")
    for k, x in extra.items():
        print(f"  {k:28s} {x}")
    d1._titulo("SENSIBILIDADES (sin veredicto)")
    for etiqueta, s in sensibilidades.items():
        print(f"  {etiqueta}: " + "  ".join(f"{n} {x['valor']:+.4f} [{x['ic95'][0]:+.4f}, {x['ic95'][1]:+.4f}]"
                                           for n, x in s.items())
              + f"  (P1 tendría: {s['P1']['veredicto_que_tendria']})")


def personas_y_128(v):
    """Personas expuestas en los sondeos y grabaciones grabadas a 128 Hz, entre las evaluadas."""
    expuestas = et.personas_expuestas(v)
    log = pd.read_csv(d1.RUTA_LOG)
    sfreq = log.drop_duplicates("archivo").set_index("archivo").sfreq_original
    a_128 = set(v.archivo_origen[v.archivo_origen.map(sfreq).eq(128).to_numpy()])
    return expuestas, a_128


def ahora():
    return time.strftime("%Y-%m-%d %H:%M:%S")
