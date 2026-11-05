"""
Tiempo en tarea — deep learning, paso 2 (piloto), parte 2 de 3: entrenar EEGNet.

Lee la caché de `preparar.py`, entrena EEGNet-8,2 en el pliegue 0 con las
ejecuciones cerradas del `INFORME.md` (rejilla de lr con la semilla 0 y dos
semillas más de la mejor lr) y guarda, por ejecución, la curva de pérdida, el
mejor modelo y la puntuación de las ventanas de evaluación. Guarda un
checkpoint por época; si se corta, al relanzarlo continúa donde lo dejó.

La mejor lr se elige por la pérdida en el conjunto interno, nunca con las
personas de evaluación.

Ejecutar con el entorno de PyTorch (desde la raíz del repositorio):
    C:/venvs/eeg-dl/Scripts/python.exe analisis/tiempo_en_tarea/deep_learning/02_piloto_cnn/entrenar.py > analisis/tiempo_en_tarea/deep_learning/02_piloto_cnn/entrenar.log
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

EEG_DATA = Path(os.environ.get("EEG_DATA", r"C:\Users\migue\Desktop\TFG\EEG-Data"))
CACHE = EEG_DATA.parent / "EEG-DL-cache" / "piloto"
LRS = [3e-4, 1e-3, 3e-3]
SEMILLAS_EXTRA = [1, 2]
LOTE = 256
MAX_EPOCAS = 60
PACIENCIA = 8
WEIGHT_DECAY = 0.01
HILOS = 6


class EEGNet(nn.Module):
    """EEGNet-8,2 (Lawhern et al., 2018) a 128 Hz, con 2 mapas de entrada: señal y máscara de canales."""

    def __init__(self, canales=14, muestras=512, F1=8, D=2, F2=16, k=64, p=0.25, mapas=2):
        super().__init__()
        self.bloque1 = nn.Sequential(
            nn.Conv2d(mapas, F1, (1, k), padding=(0, k // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (canales, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p),
        )
        self.bloque2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(p),
        )
        with torch.no_grad():
            n = self.bloque2(self.bloque1(torch.zeros(1, mapas, canales, muestras))).numel()
        self.salida = nn.Linear(n, 1)

    def forward(self, x):
        return self.salida(self.bloque2(self.bloque1(x)).flatten(1)).squeeze(1)


def lote(X, M, idx):
    """Entrada (b, 2, 14, 512): la señal y la máscara repetida a lo largo del tiempo."""
    x = torch.from_numpy(X[idx])
    m = torch.from_numpy(M[idx].astype(np.float32))[:, :, None].expand_as(x)
    return torch.stack([x, m], dim=1)


def puntuar(modelo, X, M, idx):
    modelo.eval()
    salida = []
    with torch.no_grad():
        for i in range(0, len(idx), 1024):
            salida.append(modelo(lote(X, M, idx[i:i + 1024])).numpy())
    return np.concatenate(salida) if salida else np.zeros(0)


def auc(y, s):
    """AUC de Mann–Whitney (los empates entre puntuaciones continuas son despreciables)."""
    r = np.empty(len(s))
    r[np.argsort(s)] = np.arange(1, len(s) + 1)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def ejecutar(nombre, lr, semilla, X, M, v):
    """Una ejecución completa, reanudable desde su último checkpoint."""
    carpeta = CACHE / "ejecuciones" / nombre
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
    modelo = EEGNet()
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
            l = perdida(modelo(lote(X, M, idx)), torch.from_numpy(y[idx]))
            l.backward()
            opt.step()
            suma += l.item() * len(idx)
        s_int = puntuar(modelo, X, M, inter)
        l_int = float(perdida(torch.from_numpy(s_int), torch.from_numpy(y[inter])))
        fila = {"epoca": epoca + 1, "perdida_entreno": suma / len(ent), "perdida_interna": l_int,
                "auc_interna": auc(y[inter], s_int), "segundos": time.time() - t}
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
    np.save(carpeta / "puntuacion_evaluacion.npy", puntuar(modelo, X, M, eva))
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
    X = np.load(CACHE / "X.npy", mmap_mode="r")
    X = np.ascontiguousarray(X)
    M = np.load(CACHE / "mascara.npy")
    v = pd.read_parquet(CACHE / "ventanas.parquet")
    print(f"Tensor {X.shape}; ventanas de entreno {int(((v.papel == 'entreno') & (v.seg >= 0)).sum())}, "
          f"internas {int(((v.papel == 'interno') & (v.seg >= 0)).sum())}, "
          f"evaluación {int((v.papel == 'evaluacion').sum())}. PyTorch {torch.__version__}, {HILOS} hilos")
    print(f"Parámetros de EEGNet: {sum(p.numel() for p in EEGNet().parameters())}")

    resumenes = []
    for lr in LRS:
        resumenes.append(ejecutar(f"lr{lr:g}_s0", lr, 0, X, M, v))
    mejor = min(resumenes, key=lambda r: r["mejor_perdida_interna"])
    print(f"\nMejor lr (menor pérdida interna): {mejor['lr']:g} ({mejor['mejor_perdida_interna']:.4f})\n")
    for s in SEMILLAS_EXTRA:
        resumenes.append(ejecutar(f"lr{mejor['lr']:g}_s{s}", mejor["lr"], s, X, M, v))

    with open(CACHE / "entrenamiento.json", "w", encoding="utf-8") as f:
        json.dump({"mejor_lr": mejor["lr"], "ejecuciones": [r["nombre"] for r in resumenes],
                   "torch": torch.__version__, "python": sys.version.split()[0], "hilos": HILOS,
                   "segundos": round(time.time() - t0)}, f, ensure_ascii=False, indent=1)
    print(f"\nTiempo total: {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
