"""
Fatiga aguda — deep learning, paso 4, parte 2 de 3: entrenar y puntuar EEGNet.

Tres modos, con la EEGNet-8,2 y las funciones del piloto
(`02_piloto_cnn/entrenar.py`, sin modificarlas):

- `--fase-a`: 3 semillas entrenadas con las personas de `train`, parada
  temprana sobre el conjunto interno (paciencia 8, máximo 80 épocas);
  puntúa las ventanas de `val`.
- `--fase-b`: 3 semillas con todo `train` + `val`, sin conjunto interno,
  durante las épocas fijadas en `configuracion_dl.json` (las escribe
  `evaluar.py --val`). Añade a ese archivo los SHA-256 de los pesos.
- `--puntuar-test --permitir-test`: puntúa `test` con los pesos congelados,
  después de comprobar sus SHA-256. Se niega a correr si el árbol de git no
  está limpio.

Guarda un checkpoint por época; si se corta, al relanzarlo continúa donde lo
dejó.

Ejecutar con el entorno de PyTorch (desde la raíz del repositorio; el script
escribe su propio registro):
    C:/venvs/eeg-dl/Scripts/python.exe analisis/fatiga_aguda/deep_learning/04_test_eegnet/entrenar.py --fase-a
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

CARPETA = Path(__file__).resolve().parent
RAIZ_REPO = CARPETA.parents[3]
EEG_DATA = Path(os.environ.get("EEG_DATA", r"C:\Users\migue\Desktop\TFG\EEG-Data"))
CACHE = EEG_DATA.parent / "EEG-DL-cache" / "test_eegnet"
CONFIGURACION = CARPETA / "configuracion_dl.json"
FAMILIA = ["PASAT", "PVTB"]
LR = 3e-4
SEMILLAS = [0, 1, 2]
MAX_EPOCAS_A = 80


def cargar_modulo(nombre, ruta):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = modulo
    spec.loader.exec_module(modulo)
    return modulo


piloto = cargar_modulo("piloto_entrenar", CARPETA.parent / "02_piloto_cnn" / "entrenar.py")
EEGNet, lote, puntuar, auc = piloto.EEGNet, piloto.lote, piloto.puntuar, piloto.auc
LOTE, PACIENCIA, WEIGHT_DECAY, HILOS = piloto.LOTE, piloto.PACIENCIA, piloto.WEIGHT_DECAY, piloto.HILOS


class Tee:
    def __init__(self, ruta):
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        self.archivo = open(ruta, "a", encoding="utf-8")
        self.consola = sys.stdout

    def write(self, texto):
        self.consola.write(texto)
        self.archivo.write(texto)

    def flush(self):
        self.consola.flush()
        self.archivo.flush()


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


def git_limpio():
    def git(*args):
        return subprocess.run(["git", *args], cwd=RAIZ_REPO, capture_output=True, text=True, check=True).stdout
    cambios = [l for l in git("status", "--porcelain").splitlines() if l.strip()]
    if cambios:
        sys.exit("El árbol de git no está limpio.\n  " + "\n  ".join(cambios))
    return git("rev-parse", "HEAD").strip()


def cargar_cache(carpeta):
    X = np.ascontiguousarray(np.load(carpeta / "X.npy", mmap_mode="r"))
    M = np.load(carpeta / "mascara.npy")
    v = pd.read_parquet(carpeta / "ventanas.parquet")
    assert len(X) == len(M) == len(v)
    return X, M, v


def entrenar(carpeta, semilla, X, M, y, ent, inter=None, epocas=None):
    """
    Una ejecución reanudable desde su último checkpoint.

    Con `inter`: parada temprana sobre la pérdida interna (paciencia 8, máximo
    80 épocas) y se guarda la mejor época en `mejor.pt`. Sin `inter`: exactamente
    `epocas` épocas y se guarda la última en `final.pt`.
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    if (carpeta / "resumen.json").exists():
        print(f"[{carpeta.name}] ya terminada; se reutiliza.")
        with open(carpeta / "resumen.json", encoding="utf-8") as f:
            return json.load(f)
    torch.manual_seed(semilla)
    modelo = EEGNet()
    opt = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    perdida = nn.BCEWithLogitsLoss()
    curva, epoca0, mejor, sin_mejora = [], 0, np.inf, 0
    if (carpeta / "ultimo.pt").exists():
        cp = torch.load(carpeta / "ultimo.pt", weights_only=False)
        modelo.load_state_dict(cp["modelo"])
        opt.load_state_dict(cp["opt"])
        curva, epoca0, mejor, sin_mejora = cp["curva"], cp["epoca"], cp["mejor"], cp["sin_mejora"]
        torch.set_rng_state(cp["rng_torch"])
        print(f"[{carpeta.name}] reanudada en la época {epoca0 + 1}")

    tope = MAX_EPOCAS_A if inter is not None else epocas
    for epoca in range(epoca0, tope):
        t = time.time()
        orden = np.random.default_rng([semilla, epoca]).permutation(ent)
        modelo.train()
        suma = 0.0
        for i in range(0, len(orden), LOTE):
            idx = np.sort(orden[i:i + LOTE])
            opt.zero_grad()
            l = perdida(modelo(lote(X, M, idx)), torch.from_numpy(y[idx]))
            l.backward()
            opt.step()
            suma += l.item() * len(idx)
        fila = {"epoca": epoca + 1, "perdida_entreno": suma / len(ent)}
        parar = not np.isfinite(fila["perdida_entreno"])
        if inter is not None:
            s_int = puntuar(modelo, X, M, inter)
            l_int = float(perdida(torch.from_numpy(s_int), torch.from_numpy(y[inter])))
            fila.update({"perdida_interna": l_int, "auc_interna": auc(y[inter], s_int)})
            if l_int < mejor:
                mejor, sin_mejora = l_int, 0
                torch.save(modelo.state_dict(), carpeta / "mejor.pt")
            else:
                sin_mejora += 1
            parar = parar or not np.isfinite(l_int) or sin_mejora >= PACIENCIA
        fila["segundos"] = time.time() - t
        curva.append(fila)
        print(f"[{carpeta.name}] época {fila['epoca']:2d}  entreno {fila['perdida_entreno']:.4f}"
              + (f"  interna {fila['perdida_interna']:.4f}  AUC interna {fila['auc_interna']:.3f}"
                 if inter is not None else "") + f"  {fila['segundos']:.0f} s", flush=True)
        torch.save({"modelo": modelo.state_dict(), "opt": opt.state_dict(), "curva": curva, "epoca": epoca + 1,
                    "mejor": mejor, "sin_mejora": sin_mejora, "rng_torch": torch.get_rng_state()},
                   carpeta / "ultimo.pt")
        if parar:
            break

    c = pd.DataFrame(curva)
    resumen = {"nombre": carpeta.name, "lr": LR, "semilla": semilla, "epocas": len(c),
               "segundos_por_epoca": float(c.segundos.median()), "curva": curva}
    if inter is not None:
        modelo.load_state_dict(torch.load(carpeta / "mejor.pt"))
        resumen.update({"mejor_epoca": int(c.perdida_interna.idxmin() + 1),
                        "mejor_perdida_interna": float(c.perdida_interna.min()),
                        "auc_interna_mejor": float(c.auc_interna[c.perdida_interna.idxmin()]),
                        "nan": bool(not np.isfinite(c.perdida_interna).all())})
    else:
        torch.save(modelo.state_dict(), carpeta / "final.pt")
        resumen["nan"] = bool(not np.isfinite(c.perdida_entreno).all())
    with open(carpeta / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=1)
    return resumen


def entorno():
    return {"torch": torch.__version__, "python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "hilos": HILOS}


def fase_a():
    sys.stdout = Tee(CARPETA / "entrenar_fase_a.log")
    t0 = time.time()
    X, M, v = cargar_cache(CACHE / "dev")
    assert "test" not in set(v.papel)
    tercio = (v.seg >= 0).to_numpy()
    ent = np.flatnonzero((v.papel == "entreno").to_numpy() & tercio)
    inter = np.flatnonzero((v.papel == "interno").to_numpy() & tercio)
    eva = np.flatnonzero((v.papel == "val").to_numpy())
    y = v.seg.to_numpy().astype(np.float32)
    print(f"Fase A: tensor {X.shape}; entreno {len(ent)}, internas {len(inter)}, val {len(eva)} ventanas. "
          f"PyTorch {torch.__version__}, {HILOS} hilos. Parámetros: {sum(p.numel() for p in EEGNet().parameters())}")
    resumenes = []
    for s in SEMILLAS:
        r = entrenar(CACHE / "fase_a" / f"s{s}", s, X, M, y, ent, inter=inter)
        modelo = EEGNet()
        modelo.load_state_dict(torch.load(CACHE / "fase_a" / f"s{s}" / "mejor.pt"))
        np.save(CACHE / "fase_a" / f"s{s}" / "puntuacion_val.npy", puntuar(modelo, X, M, eva))
        resumenes.append(r)
    with open(CACHE / "fase_a" / "fase_a.json", "w", encoding="utf-8") as f:
        json.dump({"ejecuciones": resumenes, "entorno": entorno(), "segundos": round(time.time() - t0)}, f,
                  ensure_ascii=False, indent=1)
    print(f"\nTiempo total: {(time.time() - t0) / 60:.0f} min")


def fase_b():
    sys.stdout = Tee(CARPETA / "entrenar_fase_b.log")
    t0 = time.time()
    with open(CONFIGURACION, encoding="utf-8") as f:
        config = json.load(f)
    assert config["lr"] == LR and config["semillas"] == SEMILLAS and config["pesos"] is None
    epocas = int(config["epocas"])
    X, M, v = cargar_cache(CACHE / "dev")
    ent = np.flatnonzero(v.condicion.isin(FAMILIA).to_numpy() & (v.seg >= 0).to_numpy())
    y = v.seg.to_numpy().astype(np.float32)
    print(f"Fase B: tensor {X.shape}; entreno (train + val, tercios de PASAT y PVTB) {len(ent)} ventanas; "
          f"{epocas} épocas fijas. PyTorch {torch.__version__}, {HILOS} hilos")
    resumenes, pesos = [], {}
    for s in SEMILLAS:
        resumenes.append(entrenar(CACHE / "fase_b" / f"s{s}", s, X, M, y, ent, epocas=epocas))
        pesos[f"s{s}"] = sha256(CACHE / "fase_b" / f"s{s}" / "final.pt")
        print(f"  SHA-256 de s{s}/final.pt: {pesos[f's{s}']}")
    (CARPETA / "tablas").mkdir(exist_ok=True)
    pd.DataFrame([dict(c, semilla=r["semilla"]) for r in resumenes for c in r["curva"]]).to_csv(
        CARPETA / "tablas" / "curvas_fase_b.csv", index=False)
    config.update({"pesos": pesos, "tensor_entrenamiento_sha256": sha256(CACHE / "dev" / "X.npy"),
                   "ventanas_entrenamiento": int(len(ent)), "entorno_entrenamiento": entorno(),
                   "fase_b": {"segundos": round(time.time() - t0),
                              "nan": any(r["nan"] for r in resumenes)}})
    with open(CONFIGURACION, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=1)
    print(f"\nPesos añadidos a {CONFIGURACION.name}. Tiempo total: {(time.time() - t0) / 60:.0f} min")


def puntuar_test():
    commit = git_limpio()
    carpeta = CACHE / "test"
    salida = Path((carpeta / "salida.txt").read_text(encoding="utf-8"))
    sys.stdout = Tee(salida / "puntuar.log")
    print(f"Puntuación de test con los pesos congelados. Commit {commit[:12]}. Salida: {salida}")
    with open(CONFIGURACION, encoding="utf-8") as f:
        config = json.load(f)
    X, M, v = cargar_cache(carpeta)
    assert set(v.papel) == {"test"}
    (salida / "modelos").mkdir(exist_ok=True)
    todas = np.arange(len(v))
    for s in SEMILLAS:
        ruta = CACHE / "fase_b" / f"s{s}" / "final.pt"
        h = sha256(ruta)
        if h != config["pesos"][f"s{s}"]:
            sys.exit(f"El SHA-256 de {ruta} no coincide con configuracion_dl.json: {h}")
        modelo = EEGNet()
        modelo.load_state_dict(torch.load(ruta))
        np.save(carpeta / f"puntuacion_s{s}.npy", puntuar(modelo, X, M, todas))
        shutil.copy2(ruta, salida / "modelos" / f"s{s}.pt")
        print(f"  s{s}: SHA-256 comprobado; {len(todas)} ventanas puntuadas")
    with open(salida / "puntuar.json", "w", encoding="utf-8") as f:
        json.dump({"commit": commit, "entorno": entorno(), "pesos": config["pesos"]}, f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    modo = ap.add_mutually_exclusive_group(required=True)
    modo.add_argument("--fase-a", action="store_true")
    modo.add_argument("--fase-b", action="store_true")
    modo.add_argument("--puntuar-test", action="store_true")
    ap.add_argument("--permitir-test", action="store_true")
    args = ap.parse_args()
    if args.puntuar_test and not args.permitir_test:
        sys.exit("El modo --puntuar-test exige --permitir-test.")
    torch.set_num_threads(HILOS)
    if args.fase_a:
        fase_a()
    elif args.fase_b:
        fase_b()
    else:
        puntuar_test()


if __name__ == "__main__":
    main()
