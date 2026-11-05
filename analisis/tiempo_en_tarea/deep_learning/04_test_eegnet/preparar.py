"""
Fatiga aguda — deep learning, paso 4, parte 1 de 3: preparar.

Dos modos:

- `--val` (fases A y B): tensor de PASAT y PVTB de `train` + `val` y de OA de
  `val`, a 128 Hz, con la función del piloto; tabla de ventanas con su papel
  (entreno, interno o `val`) y la marca de igual tiempo; y la logística
  preregistrada del paso 6 entrenada con `train` y aplicada a `val`, que debe
  reproducir las cifras del ensayo general del paso 6. **No carga `test`.**
- `--test --permitir-test` (fase D): tensor de PASAT, PVTB y OA de `test`.
  Se niega a correr si el árbol de git no está limpio. Crea la carpeta de
  salida en `EEG-Data/processed/` (falla si ya existe).

Plan: `INFORME.md` y `PREREGISTRO.md`, escritos antes de ejecutar.

Ejecutar (desde la raíz del repositorio; el script escribe su propio registro):
    .venv/Scripts/python.exe analisis/fatiga_aguda/deep_learning/04_test_eegnet/preparar.py --val
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comun_dl as cd  # noqa: E402
from comun_dl import comun, d1, et  # noqa: E402

from eeg_library.modelado.confirmatorio import carpeta_salida_nueva  # noqa: E402

FRACCION_INTERNA = 0.15
SEMILLA_INTERNA = 0
TOLERANCIA_ENSAYO = 1e-6


def configuracion_logistica():
    with open(et.CONFIG, encoding="utf-8") as f:
        config = json.load(f)
    with open(et.MEDIDA, encoding="utf-8") as f:
        medida = json.load(f)
    assert (config["features"], config["modelo"], config["esquema"], medida["medida"]) == (
        "rel", "lr_0.01", "separado", "rho_ventanas")
    return config["features"], config["modelo"]


def resumen_papeles(v):
    return (v.groupby(["papel", "condicion"])
            .agg(personas=("persona_uid", "nunique"), grabaciones=("archivo_origen", "nunique"),
                 ventanas=("seg", "size"), tercios=("seg", lambda s: int((s >= 0).sum())))
            .reset_index())


def tensor_e_integridad(v, carpeta):
    desalineados, hash_npy, n_archivos = cd.integridad_npy(v)
    assert not desalineados, f"Archivos desalineados: {desalineados}"
    print(f"Archivos .npy: {n_archivos}. Desalineados: 0. Hash combinado: {hash_npy[:16]}…")
    forma, gb, excluidos = cd.guardar_tensor(v, carpeta)
    print(f"Tensor {forma}, {gb:.2f} GB, {100 * excluidos:.1f} % de canales excluidos. Caché: {carpeta}")
    return {"forma_tensor": list(forma), "gb": gb, "fraccion_canales_excluidos": excluidos,
            "ventanas_npy_combinado": hash_npy, "archivos_npy": n_archivos}


def modo_val():
    sys.stdout = cd.Tee(cd.CARPETA / "preparar_val.log")
    t0 = time.time()
    particiones = {"train", "val"}
    assert "test" not in particiones
    d1._titulo(f"0. CARGA (train + val; test no se carga). {cd.ahora()}")
    w, _ = et.cargar(particiones, permitir_test=False)
    assert set(w.split) <= particiones

    v = cd.ventanas_cnn(w, particiones_oa={"val"})
    entreno = sorted(v[v.split == "train"].persona_uid.unique())
    rng = np.random.default_rng(SEMILLA_INTERNA)
    internas = set(rng.choice(entreno, size=round(FRACCION_INTERNA * len(entreno)), replace=False))
    v["papel"] = np.where(v.split == "val", "val", np.where(v.persona_uid.isin(internas), "interno", "entreno"))
    v["igual"] = False
    es_val = (v.split == "val").to_numpy()
    v.loc[es_val, "igual"] = cd.marcar_igual_tiempo(v[es_val], w[w.split == "val"])
    resumen = resumen_papeles(v)
    print(resumen.to_string(index=False))

    d1._titulo("1. INTEGRIDAD Y TENSOR A 128 Hz")
    tensor = tensor_e_integridad(v, cd.CACHE / "dev")

    d1._titulo("2. LOGÍSTICA PREREGISTRADA DEL PASO 6: train → val")
    feats, mod = configuracion_logistica()
    v_ent, v_ev = comun.versiones(w[w.split == "train"]), comun.versiones(w[w.split == "val"])
    filas = []
    for version in et.VERSIONES:
        _, rho, _ = et.rho_y_tercios(v_ent[version], v_ev[version], feats, mod)
        filas += [rho[c].assign(version=version) for c in cd.CONDICIONES_CNN]
    lr = pd.concat(filas, ignore_index=True)
    lr.to_parquet(cd.CACHE / "dev" / "lr_val.parquet", index=False)

    with open(et.CARPETA / "resultados_ensayo.json", encoding="utf-8") as f:
        ensayo = json.load(f)["descriptivos"]
    reproduccion = {}
    for version in et.VERSIONES:
        for c in cd.CONDICIONES_CNN:
            x = lr[(lr.version == version) & (lr.condicion == c)]
            ref = ensayo[f"{version} | {c} | rho"]
            reproduccion[f"{version} | {c}"] = {"ahora": float(x.acierto.mean()), "ensayo": ref["valor"],
                                                "grabaciones": len(x), "grabaciones_ensayo": ref["grabaciones"]}
            print(f"  {version:12s} {c:5s} ρ = {x.acierto.mean():+.6f}  (ensayo {ref['valor']:+.6f}; "
                  f"{len(x)} grab., ensayo {ref['grabaciones']})")
    reproduce = all(abs(r["ahora"] - r["ensayo"]) < TOLERANCIA_ENSAYO and r["grabaciones"] == r["grabaciones_ensayo"]
                    for r in reproduccion.values())
    print(f"  Reproduce el ensayo general del paso 6: {'sí' if reproduce else 'NO'}")

    cd.json_seguro(cd.CARPETA / "resultados_preparar_val.json", {
        "fecha": cd.ahora(), "particiones": sorted(particiones),
        "interno": {"fraccion": FRACCION_INTERNA, "semilla": SEMILLA_INTERNA, "personas": len(internas)},
        "resumen": resumen.to_dict(orient="records"), **tensor,
        "logistica": {"features": feats, "modelo": mod, "reproduccion_ensayo": reproduccion,
                      "reproduce_ensayo": reproduce},
        "sha256": {q.name: d1.sha256(q) for q in (d1.RUTA_FEATURES, d1.RUTA_MATRIZ, d1.RUTA_LOG, et.CONFIG, et.MEDIDA)},
        "segundos": round(time.time() - t0),
    })
    print(f"\nTiempo total: {time.time() - t0:.0f} s")


def modo_test():
    commit = cd.git_limpio()
    inicio = datetime.now()
    salida = carpeta_salida_nueva(d1.EEG_DATA / "processed" / f"fatiga_aguda_dl_test_{inicio:%Y%m%d-%H%M%S}")
    sys.stdout = cd.Tee(salida / "preparar.log")
    t0 = time.time()
    d1._titulo(f"0. CARGA DE TEST (evaluación única). {cd.ahora()}. Commit {commit[:12]}")
    print(f"Salida: {salida}")
    carpeta = cd.CACHE / "test"
    if carpeta.exists():
        shutil.rmtree(carpeta)
    w, _ = et.cargar({"test"}, permitir_test=True)
    assert set(w.split) == {"test"}
    v = cd.ventanas_cnn(w, particiones_oa={"test"})
    v["papel"] = "test"
    v["igual"] = cd.marcar_igual_tiempo(v, w)
    resumen = resumen_papeles(v)
    print(resumen.to_string(index=False))

    d1._titulo("1. INTEGRIDAD Y TENSOR A 128 Hz")
    tensor = tensor_e_integridad(v, carpeta)
    (carpeta / "salida.txt").write_text(str(salida), encoding="utf-8")
    cd.json_seguro(salida / "preparar.json", {
        "inicio": f"{inicio:%Y-%m-%d %H:%M:%S}", "commit": commit, "resumen": resumen.to_dict(orient="records"),
        **tensor, "sha256": {q.name: d1.sha256(q) for q in (d1.RUTA_FEATURES, d1.RUTA_MATRIZ, d1.RUTA_LOG)},
        "segundos": round(time.time() - t0),
    })
    print(f"\nTiempo total: {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    modo = ap.add_mutually_exclusive_group(required=True)
    modo.add_argument("--val", action="store_true")
    modo.add_argument("--test", action="store_true")
    ap.add_argument("--permitir-test", action="store_true")
    args = ap.parse_args()
    if args.test and not args.permitir_test:
        sys.exit("El modo --test exige --permitir-test.")
    modo_test() if args.test else modo_val()


if __name__ == "__main__":
    main()
