"""
Fatiga aguda — deep learning, paso 4, parte 3 de 3: evaluar.

Dos modos, con las mismas cifras del preregistro (P1, S1, descriptivos y
sensibilidades, en `comun_dl.py`):

- `--val` (fase A): EEGNet entrenada con `train` frente a la logística del
  paso 6 entrenada con `train`, en `val`. Comprueba los requisitos técnicos,
  fija las épocas del modelo final y escribe `configuracion_dl.json` (sin
  pesos: los añade `entrenar.py --fase-b`). Es de desarrollo.
- `--test --permitir-test` (fase D): EEGNet final frente a las ρ por grabación
  ya guardadas de la logística del paso 6, en `test`. Se niega a correr si el
  árbol de git no está limpio. Reporta todo.

Ejecutar (desde la raíz del repositorio; el script escribe su propio registro):
    .venv/Scripts/python.exe analisis/fatiga_aguda/deep_learning/04_test_eegnet/evaluar.py --val
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comun_dl as cd  # noqa: E402
from comun_dl import FAMILIA, d1, et  # noqa: E402

PREDICCIONES_LOGISTICA = (d1.EEG_DATA / "processed" / "fatiga_aguda_test_20261027-190547"
                          / "predicciones_por_grabacion.parquet")
MAX_EPOCAS_A = 80
# Hiperparámetros del piloto (`02_piloto_cnn/entrenar.py`), que `entrenar.py` reutiliza.
HIPERPARAMETROS = {"arquitectura": "EEGNet-8,2 de 02_piloto_cnn/entrenar.py (BN tras la conv temporal, "
                                   "2 mapas de entrada: señal y máscara)",
                   "frecuencia_hz": 128, "lr": 3e-4, "weight_decay": 0.01, "lote": 256, "dropout": 0.25,
                   "optimizador": "AdamW", "perdida": "BCEWithLogitsLoss", "semillas": cd.SEMILLAS}
SCRIPTS = ["comun_dl.py", "preparar.py", "entrenar.py", "evaluar.py", "ejecutar_test.ps1"]


def cifras(v, puntuaciones, rho_lr):
    """Tabla por grabación y todas las cifras del preregistro."""
    tabla, por_semilla, solo = cd.tabla_por_grabacion(v, puntuaciones, rho_lr)
    expuestas, a_128 = cd.personas_y_128(v)
    res, reps, sens = cd.analizar(tabla, expuestas, a_128)
    fam = {s: float(r[(r.version == "completa") & r.condicion.isin(FAMILIA)].acierto.mean())
           for s, r in por_semilla.items()}
    extra = {"auc_ventanas_media_semillas": cd.auc_cnn(v, puntuaciones),
             "rho_familia_por_semilla": fam,
             "desviacion_entre_semillas": float(np.std(list(fam.values()), ddof=1))}
    completa = tabla[tabla.version == "completa"]
    inventario = (completa.groupby("condicion").agg(grabaciones=("archivo_origen", "size"),
                                                    personas=("persona_uid", "nunique"),
                                                    con_logistica=("lr", lambda x: int(x.notna().sum())))
                  .reset_index().to_dict(orient="records"))
    meta = {"inventario": inventario, "personas_expuestas": len(expuestas), "grabaciones_128_hz": len(a_128),
            "emparejamiento": {k: len(x) for k, x in solo.items()}}
    cd.imprimir(res, sens, extra)
    print(f"\nInventario: {inventario}\nEmparejamiento: {meta['emparejamiento']}; "
          f"personas expuestas {len(expuestas)}; grabaciones a 128 Hz {len(a_128)}")
    return tabla, res, reps, sens, extra, meta, solo


def figura_rho(res, ruta, titulo):
    filas = [("EEGNet PASAT", res["S1_PASAT"]), ("Logística PASAT", res["lr_PASAT"]),
             ("EEGNet PVTB", res["S1_PVTB"]), ("Logística PVTB", res["lr_PVTB"]),
             ("EEGNet OA", res["cnn_OA"])]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for i, (nombre, x) in enumerate(filas[::-1]):
        color = "#2a78d6" if nombre.startswith("EEGNet") else "#8a8984"
        ax.errorbar(x["valor"], i, xerr=[[x["valor"] - x["ic95"][0]], [x["ic95"][1] - x["valor"]]],
                    fmt="o", color=color, capsize=3)
    ax.set_yticks(range(len(filas)))
    ax.set_yticklabels([n for n, _ in filas[::-1]], fontsize=8)
    ax.axvline(0, color="#cccccc", lw=0.8)
    ax.set_xlabel(f"ρ media por grabación (grabación completa), IC95 — {titulo}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    ruta.parent.mkdir(exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def modo_val():
    sys.stdout = cd.Tee(cd.CARPETA / "evaluar_val.log")
    d1._titulo(f"FASE A: EEGNet (train) frente a la logística del paso 6 (train), en val. {cd.ahora()}")
    v = pd.read_parquet(cd.CACHE / "dev" / "ventanas.parquet")
    v = v[v.papel == "val"].reset_index(drop=True)
    with open(cd.CACHE / "fase_a" / "fase_a.json", encoding="utf-8") as f:
        fase_a = json.load(f)
    puntuaciones = {s: np.load(cd.CACHE / "fase_a" / f"s{s}" / "puntuacion_val.npy") for s in cd.SEMILLAS}
    assert all(len(p) == len(v) for p in puntuaciones.values())
    rho_lr = pd.read_parquet(cd.CACHE / "dev" / "lr_val.parquet")
    tabla, res, reps, sens, extra, meta, _ = cifras(v, puntuaciones, rho_lr)

    d1._titulo("ÉPOCAS Y REQUISITOS TÉCNICOS")
    eje = pd.DataFrame([{k: r[k] for k in ("nombre", "semilla", "epocas", "mejor_epoca", "mejor_perdida_interna",
                                          "auc_interna_mejor", "segundos_por_epoca", "nan")}
                        for r in fase_a["ejecuciones"]])
    print(eje.to_string(index=False))
    curvas = pd.DataFrame([dict(c, semilla=r["semilla"]) for r in fase_a["ejecuciones"] for c in r["curva"]])
    epocas = int(np.median(eje.mejor_epoca))
    with open(cd.CARPETA / "resultados_preparar_val.json", encoding="utf-8") as f:
        preparar = json.load(f)
    requisitos = {
        "perdida_interna_baja": bool(all(g.perdida_interna.min() < g.perdida_interna.iloc[0]
                                         for _, g in curvas.groupby("semilla"))),
        "mejor_epoca_ni_1_ni_max": bool(((eje.mejor_epoca > 1) & (eje.mejor_epoca < MAX_EPOCAS_A)).all()),
        "sin_nan": bool(not eje.nan.any()),
        "logistica_reproduce_el_ensayo": bool(preparar["logistica"]["reproduce_ensayo"]),
    }
    for k, ok in requisitos.items():
        print(f"  {k}: {'sí' if ok else 'NO'}")
    print(f"\n  Épocas del modelo final (mediana de las mejores épocas {list(eje.mejor_epoca)}): {epocas}")

    if all(requisitos.values()):
        if cd.CONFIGURACION.exists():
            with open(cd.CONFIGURACION, encoding="utf-8") as f:
                if json.load(f).get("pesos") is not None:
                    sys.exit("configuracion_dl.json ya tiene pesos (fase B hecha): no se sobrescribe.")
        cd.json_seguro(cd.CONFIGURACION, {
            "fecha": cd.ahora(), **HIPERPARAMETROS, "epocas": epocas,
            "regla_epocas": "mediana de las mejores épocas de las 3 semillas de la fase A",
            "mejores_epocas_fase_a": [int(x) for x in eje.mejor_epoca], "pesos": None})
        print(f"  Escrito {cd.CONFIGURACION.name} (sin pesos; los añade entrenar.py --fase-b).")
    else:
        print("  NO se escribe configuracion_dl.json: falla algún requisito. Se corrige y se repite la fase A.")

    (cd.CARPETA / "tablas").mkdir(exist_ok=True)
    curvas.to_csv(cd.CARPETA / "tablas" / "curvas_fase_a.csv", index=False)
    pd.DataFrame([{"cifra": n, "valor": x["valor"], "ic95_inf": x["ic95"][0], "ic95_sup": x["ic95"][1]}
                  for n, x in res.items()]).to_csv(cd.CARPETA / "tablas" / "cifras_val.csv", index=False)
    figura_curvas(curvas)
    figura_rho(res, cd.CARPETA / "figuras" / "rho_val.png", "val")
    cd.json_seguro(cd.CARPETA / "resultados_val.json", {
        "fecha": cd.ahora(), "particion": "val (EEGNet y logística entrenadas con train)",
        "cifras": res, "sensibilidades": sens, "extra": extra, **meta,
        "fase_a": eje.to_dict(orient="records"), "epocas_modelo_final": epocas, "requisitos": requisitos,
        "entorno_entrenamiento": fase_a["entorno"], "segundos_fase_a": fase_a["segundos"]})


def figura_curvas(curvas):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for s, g in curvas.groupby("semilla"):
        ax.plot(g.epoca, g.perdida_interna, lw=1.2, label=f"semilla {s}")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida en el conjunto interno")
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    (cd.CARPETA / "figuras").mkdir(exist_ok=True)
    fig.savefig(cd.CARPETA / "figuras" / "curvas_fase_a.png", dpi=150)
    plt.close(fig)


def modo_test():
    commit = cd.git_limpio()
    salida = Path((cd.CACHE / "test" / "salida.txt").read_text(encoding="utf-8"))
    sys.stdout = cd.Tee(salida / "evaluar.log")
    inicio = cd.ahora()
    d1._titulo(f"FASE D: EVALUACIÓN ÚNICA EN TEST. {inicio}. Commit {commit[:12]}")
    v = pd.read_parquet(cd.CACHE / "test" / "ventanas.parquet")
    assert set(v.papel) == {"test"}
    puntuaciones = {s: np.load(cd.CACHE / "test" / f"puntuacion_s{s}.npy") for s in cd.SEMILLAS}
    assert all(len(p) == len(v) for p in puntuaciones.values())

    pred = pd.read_parquet(PREDICCIONES_LOGISTICA)
    # Las filas de rho_ventanas son las que no llevan `medida`; las de tercios llevan «tercios_pareado».
    assert set(pred.medida.dropna()) == {"tercios_pareado"}
    rho_lr = pred[pred.medida.isna() & pred.condicion.isin(cd.CONDICIONES_CNN) & pred.version.isin(et.VERSIONES)]
    rho_lr = rho_lr[["archivo_origen", "persona_uid", "acierto", "condicion", "version"]].reset_index(drop=True)
    tabla, res, reps, sens, extra, meta, solo = cifras(v, puntuaciones, rho_lr)

    (salida / "bootstrap").mkdir(exist_ok=True)
    np.save(salida / "bootstrap" / "replicas.npy", reps)
    (salida / "bootstrap" / "nombres.json").write_text(json.dumps(list(cd.ESTADISTICOS)), encoding="utf-8")
    tabla.to_parquet(salida / "rho_por_grabacion.parquet", index=False)
    with open(cd.CONFIGURACION, encoding="utf-8") as f:
        config = json.load(f)
    resultados = {
        "modo": "test", "inicio": inicio, "fin": cd.ahora(), "commit": commit,
        "cifras": res, "sensibilidades": sens, "extra": extra, **meta,
        "configuracion": config,
        "sha256": {**{s: cd.sha256_archivo(cd.CARPETA / s) for s in SCRIPTS},
                   "preregistro": cd.sha256_archivo(cd.PREREGISTRO),
                   "configuracion_dl.json": cd.sha256_archivo(cd.CONFIGURACION),
                   **{p.name: cd.sha256_archivo(p) for p in (d1.RUTA_FEATURES, d1.RUTA_MATRIZ, d1.RUTA_LOG,
                                                             PREDICCIONES_LOGISTICA)}},
        "entorno_evaluacion": {"python": platform.python_version(), "numpy": np.__version__,
                               "pandas": pd.__version__, "scipy": scipy.__version__, "so": platform.platform(),
                               "cpu": platform.processor()},
        "parametros": {"n_boot": cd.N_BOOT, "semilla": cd.SEMILLA_BOOT, "alfa": cd.ALFA},
    }
    cd.json_seguro(salida / "resultados.json", resultados | {"emparejamiento_archivos": solo})
    # Copia agregada en el repositorio (sin identificadores de persona ni nombres de archivo).
    cd.json_seguro(cd.CARPETA / "resultados_test.json", resultados)
    figura_rho(res, cd.CARPETA / "figuras" / "rho_test.png", "test")
    print(f"\nFin: {cd.ahora()}. Artefactos en {salida}")
    sys.stdout.flush()
    partes = [(salida / n).read_text(encoding="utf-8") for n in ("preparar.log", "puntuar.log", "evaluar.log")]
    (cd.CARPETA / "ejecucion_test.log").write_text("\n".join(partes), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    modo = ap.add_mutually_exclusive_group(required=True)
    modo.add_argument("--val", action="store_true")
    modo.add_argument("--test", action="store_true")
    ap.add_argument("--permitir-test", action="store_true")
    args = ap.parse_args()
    if args.test and not args.permitir_test:
        sys.exit("El modo --test exige --permitir-test.")
    t0 = time.time()
    modo_test() if args.test else modo_val()
    print(f"Tiempo: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
