"""
Tiempo en tarea — deep learning, paso 2 (piloto), parte 1 de 3: preparar.

Construye en la caché, fuera de los repos y de `EEG-Data`:
- el tensor de ventanas de PASAT y PVTB de `train` a 128 Hz, normalizado por
  grabación y canal, con los canales excluidos a 0, y su máscara;
- la tabla de ventanas (grabación, persona, condición, instante, tercio y papel
  en el pliegue 0 de la semilla 0: entreno, interno o evaluación);
- las puntuaciones de las logísticas de referencia (F con C = 0.003 y B con
  C = 0.03, del paso 1) en las ventanas de evaluación.

**Solo `train`.** Plan: `INFORME.md`, escrito antes de ejecutar.

Ejecutar (desde la raíz del repositorio):
    .venv/Scripts/python.exe analisis/tiempo_en_tarea/deep_learning/02_piloto_cnn/preparar.py > analisis/tiempo_en_tarea/deep_learning/02_piloto_cnn/preparar.log
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent / "01_psd_fina"))
import psd_fina as p1  # noqa: E402
from psd_fina import FAMILIA, comun, d1, md  # noqa: E402

CACHE = d1.EEG_DATA.parent / "EEG-DL-cache" / "piloto"
SEMILLA_PLIEGUES = 0
PLIEGUE = 0
FRACCION_INTERNA = 0.15
SEMILLA_INTERNA = 0
C_REFERENCIA = {"F": 0.003, "B": 0.03}  # mejores C del paso 1


def segmento(g):
    """0 = primer tercio, 1 = último, −1 = tercio medio (como `comun.tercios`)."""
    return np.where(g.t_rec <= g.dur / 3, 0, np.where(g.t_rec >= 2 * g.dur / 3, 1, -1))


def tensor(w):
    """Ventanas a 128 Hz, z-score por grabación y canal, canales excluidos a 0; máscara (n, 14)."""
    X = np.zeros((len(w), 14, 512), dtype=np.float32)
    validez = w[d1.VALIDEZ].to_numpy(dtype=bool)
    for a, g in w.groupby("archivo_origen", sort=True):
        x = np.load(p1.RUTA_VENTANAS / a.replace(".edf", ".npy"))[g.indice_ventana.to_numpy()]
        x = resample_poly(x.astype(np.float64), 1, 2, axis=-1)
        media = x.mean(axis=(0, 2), keepdims=True)
        desv = x.std(axis=(0, 2), keepdims=True)
        x = (x - media) / np.where(desv > 0, desv, 1.0)
        pos = w.index.get_indexer(g.index)
        x[:, ~validez[pos[0]], :] = 0.0
        X[pos] = x.astype(np.float32)
    return X, validez.astype(np.uint8)


def referencias(w, papel):
    """Puntuación de las logísticas F y B (por condición) en las ventanas de evaluación."""
    salida = {}
    entreno = papel != "evaluacion"
    for r in ["F", "B"]:
        punt = pd.Series(np.nan, index=w.index)
        for c in FAMILIA:
            S, X, filas, Xv = p1.datos(w, c, r)
            en_S = S.persona_uid.map(dict(zip(w.persona_uid, entreno))).to_numpy()
            idx = w.index[w.condicion == c]
            en_v = entreno[w.index.get_indexer(idx)]
            m = comun.modelo(f"lr_{C_REFERENCIA[r]}").fit(X[en_S], S.seg.to_numpy()[en_S])
            punt.loc[idx[~en_v]] = m.decision_function(Xv[~en_v])
        salida[r] = punt
    return salida


def main():
    t0 = time.time()
    d1._titulo("0. CARGA (solo train) Y PLIEGUE")
    w = comun.cargar({"train"})
    assert set(w.split) == {"train"}
    # Mismas personas que en el paso 1 (PASAT, PVTB, OA y OC) para tener la misma asignación a pliegues.
    personas = sorted(w[w.condicion.isin(p1.CONDICIONES)].persona_uid.unique())
    pliegue = comun.pliegues_por_persona(personas, SEMILLA_PLIEGUES)
    w = w[w.condicion.isin(FAMILIA)].sort_values(["archivo_origen", "indice_ventana"]).reset_index(drop=True)

    entreno = sorted(p for p in personas if pliegue[p] != PLIEGUE)
    rng = np.random.default_rng(SEMILLA_INTERNA)
    internas = set(rng.choice(entreno, size=round(FRACCION_INTERNA * len(entreno)), replace=False))
    papel = np.where(w.persona_uid.map(pliegue) == PLIEGUE, "evaluacion",
                     np.where(w.persona_uid.isin(internas), "interno", "entreno"))
    w["seg"] = segmento(w)
    w["papel"] = papel
    resumen = w.groupby("papel").agg(personas=("persona_uid", "nunique"), grabaciones=("archivo_origen", "nunique"),
                                     ventanas=("seg", "size"), tercios=("seg", lambda s: int((s >= 0).sum())))
    print(resumen.to_string())

    d1._titulo("1. ESPECTRO FINO (para la referencia F) E INTEGRIDAD")
    w, desalineados, _, hash_npy, n_archivos = p1.anadir_espectro(w)
    assert not desalineados, f"Archivos desalineados: {len(desalineados)}"
    w = w.reset_index(drop=True)
    print(f"Archivos: {n_archivos}. Desalineados: 0. ({time.time() - t0:.0f} s)")

    d1._titulo("2. REFERENCIAS LOGÍSTICAS EN EL PLIEGUE 0")
    ref = referencias(w, w.papel.to_numpy())
    ev = w.papel == "evaluacion"
    for r, punt in ref.items():
        filas = w.loc[ev, ["archivo_origen", "persona_uid", "t_rec", "condicion"]]
        rho = md.rho_por_grabacion(filas.drop(columns="condicion"), punt[ev].to_numpy())
        rho = rho.merge(filas.drop_duplicates("archivo_origen")[["archivo_origen", "condicion"]])
        print(f"  {r} (C = {C_REFERENCIA[r]}): " + "  ".join(
            f"{c} ρ = {rho[rho.condicion == c].acierto.mean():.3f}" for c in FAMILIA)
            + f"  PASAT + PVTB ρ = {rho.acierto.mean():.3f}")

    d1._titulo("3. TENSOR A 128 Hz")
    X, mascara = tensor(w)
    print(f"Forma: {X.shape}, {X.nbytes / 1e9:.2f} GB. NaN: {int(np.isnan(X).sum())}. ({time.time() - t0:.0f} s)")
    assert not np.isnan(X).any()

    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "X.npy", X)
    np.save(CACHE / "mascara.npy", mascara)
    tabla = w[["archivo_origen", "persona_uid", "condicion", "t_rec", "seg", "papel"]].assign(
        punt_F=ref["F"].to_numpy(), punt_B=ref["B"].to_numpy())
    tabla.to_parquet(CACHE / "ventanas.parquet", index=False)

    with open(CARPETA / "resultados_preparar.json", "w", encoding="utf-8") as f:
        json.dump({
            "fecha": time.strftime("%Y-%m-%d"),
            "particion": "train",
            "pliegue": {"semilla": SEMILLA_PLIEGUES, "pliegue": PLIEGUE, "fraccion_interna": FRACCION_INTERNA,
                        "semilla_interna": SEMILLA_INTERNA},
            "resumen": resumen.reset_index().to_dict(orient="records"),
            "C_referencia": C_REFERENCIA,
            "forma_tensor": list(X.shape),
            "sha256": {q.name: d1.sha256(q) for q in (d1.RUTA_FEATURES, d1.RUTA_MATRIZ, d1.RUTA_LOG)}
                      | {"ventanas_npy_combinado": hash_npy},
            "segundos": round(time.time() - t0),
        }, f, ensure_ascii=False, indent=1, default=int)
    print(f"\nCaché: {CACHE}. Tiempo total: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
