"""
Fatiga aguda — deep learning, paso 3 (piloto de ShallowConvNet), parte 1 de 2: entrenar.

Lee la caché del paso 2 (sin modificarla), entrena ShallowConvNet con filtro
temporal de 1 s en el pliegue 0 con las ejecuciones cerradas del `INFORME.md`
(dos lr con la semilla 0 y dos semillas más de la mejor lr) y guarda, por
ejecución, la curva de pérdida, el mejor modelo, sus filtros temporales y la
puntuación de las ventanas de evaluación. Guarda un checkpoint por época; si se
corta, al relanzarlo continúa donde lo dejó.

La mejor lr se elige por la pérdida en el conjunto interno, nunca con las
personas de evaluación. `lote`, `puntuar` y `auc` son los del paso 2.

Ejecutar con el entorno de PyTorch (desde la raíz del repositorio):
    C:/venvs/eeg-dl/Scripts/python.exe analisis/fatiga_aguda/deep_learning/03_piloto_shallow/entrenar.py > analisis/fatiga_aguda/deep_learning/03_piloto_shallow/entrenar.log
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

CARPETA = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("piloto2_entrenar", CARPETA.parent / "02_piloto_cnn" / "entrenar.py")
p2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2)

CACHE = p2.CACHE
SALIDA = CACHE / "ejecuciones_shallow"
LRS = [3e-4, 1e-3]
SEMILLAS_EXTRA = [1, 2]
LOTE = p2.LOTE
MAX_EPOCAS = p2.MAX_EPOCAS
PACIENCIA = p2.PACIENCIA
WEIGHT_DECAY = p2.WEIGHT_DECAY
HILOS = p2.HILOS


class ShallowConvNet(nn.Module):
    """ShallowConvNet (Schirrmeister et al., 2017) a 128 Hz con filtro temporal de 1 s y 2 mapas de entrada."""

    def __init__(self, canales=14, muestras=512, n=40, k=128, pool=35, paso=7, p=0.5, mapas=2):
        super().__init__()
        self.temporal = nn.Conv2d(mapas, n, (1, k))
        self.espacial = nn.Conv2d(n, n, (canales, 1), bias=False)
        self.bn = nn.BatchNorm2d(n)
        self.pool = nn.AvgPool2d((1, pool), (1, paso))
        self.dropout = nn.Dropout(p)
        with torch.no_grad():
            m = self.pool(self.espacial(self.temporal(torch.zeros(1, mapas, canales, muestras)))).numel()
        self.salida = nn.Linear(m, 1)

    def forward(self, x):
        x = self.bn(self.espacial(self.temporal(x)))
        x = torch.log(torch.clamp(self.pool(x * x), min=1e-6))
        return self.salida(self.dropout(x).flatten(1)).squeeze(1)


def ejecutar(nombre, lr, semilla, X, M, v):
    """Una ejecución completa, reanudable desde su último checkpoint (como en el paso 2)."""
    carpeta = SALIDA / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    if (carpeta / "resumen.json").exists():
        print(f"[{nombre}] ya terminada; se reutiliza.")
        with open(carpeta / "resumen.json", encoding="utf-8") as f:
            return json.load(f)

    ent = np.flatnonzero((v.papel == "entreno").to_numpy() & (v.seg >= 0).to_numpy())
    inter = np.flatnonzero((v.papel == "interno").to_numpy() & (v.seg >= 0).to_numpy())
    eva = np.flatnonzero((v.papel == "evaluacion").to_numpy())
    y = v.seg.to_numpy().astype(np.float32)

    torch.manual_seed(semilla)
    modelo = ShallowConvNet()
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    perdida = nn.BCEWithLogitsLoss()
    curva, epoca0, mejor, sin_mejora = [], 0, np.inf, 0
    if (carpeta / "ultimo.pt").exists():
        cp = torch.load(carpeta / "ultimo.pt", weights_only=False)
        modelo.load_state_dict(cp["modelo"])
        opt.load_state_dict(cp["opt"])
        curva, epoca0, mejor, sin_mejora = cp["curva"], cp["epoca"], cp["mejor"], cp["sin_mejora"]
        torch.set_rng_state(cp["rng_torch"])
        print(f"[{nombre}] reanudada en la época {epoca0 + 1}")

    for epoca in range(epoca0, MAX_EPOCAS):
        t = time.time()
        orden = np.random.default_rng([semilla, epoca]).permutation(ent)
        modelo.train()
        suma = 0.0
        for i in range(0, len(orden), LOTE):
            idx = np.sort(orden[i:i + LOTE])
            opt.zero_grad()
            l = perdida(modelo(p2.lote(X, M, idx)), torch.from_numpy(y[idx]))
            l.backward()
            opt.step()
            suma += l.item() * len(idx)
        s_int = p2.puntuar(modelo, X, M, inter)
        l_int = float(perdida(torch.from_numpy(s_int), torch.from_numpy(y[inter])))
        fila = {"epoca": epoca + 1, "perdida_entreno": suma / len(ent), "perdida_interna": l_int,
                "auc_interna": p2.auc(y[inter], s_int), "segundos": time.time() - t}
        curva.append(fila)
        if l_int < mejor:
            mejor, sin_mejora = l_int, 0
            torch.save(modelo.state_dict(), carpeta / "mejor.pt")
        else:
            sin_mejora += 1
        print(f"[{nombre}] época {fila['epoca']:2d}  entreno {fila['perdida_entreno']:.4f}  "
              f"interna {l_int:.4f}  AUC interna {fila['auc_interna']:.3f}  {fila['segundos']:.0f} s", flush=True)
        torch.save({"modelo": modelo.state_dict(), "opt": opt.state_dict(), "curva": curva, "epoca": epoca + 1,
                    "mejor": mejor, "sin_mejora": sin_mejora, "rng_torch": torch.get_rng_state()},
                   carpeta / "ultimo.pt")
        if not np.isfinite(l_int) or sin_mejora >= PACIENCIA:
            break

    modelo.load_state_dict(torch.load(carpeta / "mejor.pt"))
    np.save(carpeta / "puntuacion_evaluacion.npy", p2.puntuar(modelo, X, M, eva))
    np.save(carpeta / "filtros_temporales.npy", modelo.temporal.weight.detach().numpy())
    c = pd.DataFrame(curva)
    resumen = {"nombre": nombre, "lr": lr, "semilla": semilla, "epocas": len(c),
               "mejor_epoca": int(c.perdida_interna.idxmin() + 1), "mejor_perdida_interna": float(c.perdida_interna.min()),
               "auc_interna_mejor": float(c.auc_interna[c.perdida_interna.idxmin()]),
               "segundos_por_epoca": float(c.segundos.median()), "nan": bool(not np.isfinite(c.perdida_interna).all()),
               "curva": curva}
    with open(carpeta / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=1)
    return resumen


def main():
    torch.set_num_threads(HILOS)
    t0 = time.time()
    X = np.ascontiguousarray(np.load(CACHE / "X.npy", mmap_mode="r"))
    M = np.load(CACHE / "mascara.npy")
    v = pd.read_parquet(CACHE / "ventanas.parquet")
    print(f"Tensor {X.shape}; ventanas de entreno {int(((v.papel == 'entreno') & (v.seg >= 0)).sum())}, "
          f"internas {int(((v.papel == 'interno') & (v.seg >= 0)).sum())}, "
          f"evaluación {int((v.papel == 'evaluacion').sum())}. PyTorch {torch.__version__}, {HILOS} hilos")
    print(f"Parámetros de ShallowConvNet: {sum(p.numel() for p in ShallowConvNet().parameters())}")

    resumenes = []
    for lr in LRS:
        resumenes.append(ejecutar(f"lr{lr:g}_s0", lr, 0, X, M, v))
    mejor = min(resumenes, key=lambda r: r["mejor_perdida_interna"])
    print(f"\nMejor lr (menor pérdida interna): {mejor['lr']:g} ({mejor['mejor_perdida_interna']:.4f})\n")
    for s in SEMILLAS_EXTRA:
        resumenes.append(ejecutar(f"lr{mejor['lr']:g}_s{s}", mejor["lr"], s, X, M, v))

    with open(SALIDA / "entrenamiento.json", "w", encoding="utf-8") as f:
        json.dump({"mejor_lr": mejor["lr"], "ejecuciones": [r["nombre"] for r in resumenes],
                   "torch": torch.__version__, "python": sys.version.split()[0], "hilos": HILOS,
                   "segundos": round(time.time() - t0)}, f, ensure_ascii=False, indent=1)
    print(f"\nTiempo total: {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
