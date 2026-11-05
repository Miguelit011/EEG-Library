"""
Tiempo en tarea — deep learning, paso 5 (2 de 2): exporta EEGNet a ONNX y comprueba que da lo mismo.

1. Comprueba que el SHA-256 de `s0.pt`, `s1.pt` y `s2.pt` es el de
   `04_test_eegnet/configuracion_dl.json`.
2. Exporta cada semilla a ONNX con el lote dinámico.
3. Puntúa el tensor de referencia de `referencia.py` (20 grabaciones de train) con PyTorch y
   con onnxruntime. Criterio: diferencia máxima < 1e-5 en las tres semillas.
4. Solo si se cumple, escribe los `.onnx` y la ficha `tt_eegnet.json` en
   `src/eeg_library/modelos/tiempo_en_tarea/`, y `resultados.json` en esta carpeta.

Ejecutar desde la raíz del repositorio con el entorno de PyTorch:
    C:/venvs/eeg-dl/Scripts/python.exe analisis/tiempo_en_tarea/deep_learning/05_onnx/exportar.py
"""

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

CARPETA = Path(__file__).resolve().parent
DL = CARPETA.parent
RAIZ_REPO = DL.parents[2]
EEG_DATA = Path(os.environ.get("EEG_DATA", r"C:\Users\migue\Desktop\TFG\EEG-Data"))
DIR_PESOS = EEG_DATA / "processed" / "fatiga_aguda_dl_test_20261031-224217" / "modelos"
CACHE = EEG_DATA.parent / "EEG-DL-cache" / "onnx"
CONFIGURACION = DL / "04_test_eegnet" / "configuracion_dl.json"
RESULTADOS_TEST = DL / "04_test_eegnet" / "resultados_test.json"
DESTINO = RAIZ_REPO / "src" / "eeg_library" / "modelos" / "tiempo_en_tarea"
SEMILLAS = [0, 1, 2]
TOLERANCIA = 1e-5
OPSET = 17
CANALES = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]


def sha256(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def cargar_modulo(nombre, ruta):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = modulo
    spec.loader.exec_module(modulo)
    return modulo


piloto = cargar_modulo("piloto_entrenar", DL / "02_piloto_cnn" / "entrenar.py")


def exportar(modelo, ruta):
    """ONNX con el lote dinámico. Primero el exportador nuevo (dynamo); si falla, el clásico."""
    ejemplo = torch.zeros(2, 2, 14, 512)
    try:
        programa = torch.onnx.export(modelo, (ejemplo,), dynamo=True, opset_version=OPSET,
                                     input_names=["entrada"], output_names=["logit"],
                                     dynamic_shapes=({0: torch.export.Dim("n", min=1, max=100000)},),
                                     external_data=False)
        programa.save(str(ruta), external_data=False)
        metodo = "dynamo"
    except Exception as error:  # noqa: BLE001
        print(f"  exportador dynamo no disponible ({type(error).__name__}: {error}); se usa el clásico")
        torch.onnx.export(modelo, (ejemplo,), str(ruta), dynamo=False, opset_version=OPSET,
                          input_names=["entrada"], output_names=["logit"],
                          dynamic_axes={"entrada": {0: "n"}, "logit": {0: "n"}})
        metodo = "clasico"
    onnx.checker.check_model(onnx.load(str(ruta)))
    return metodo


def puntuar_ort(ruta, X, M):
    sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
    salida = []
    for i in range(0, len(X), 512):
        idx = np.arange(i, min(i + 512, len(X)))
        entrada = piloto.lote(X, M, idx).numpy()
        salida.append(sesion.run(None, {sesion.get_inputs()[0].name: entrada})[0].reshape(-1))
    return np.concatenate(salida)


def main():
    torch.set_num_threads(6)
    config = json.loads(CONFIGURACION.read_text(encoding="utf-8"))
    X = np.load(CACHE / "X.npy")
    M = np.load(CACHE / "mascara.npy")
    referencia = json.loads((CACHE / "referencia.json").read_text(encoding="utf-8"))
    print(f"Tensor de referencia: {X.shape} ({referencia['grabaciones']} grabaciones de train)")

    temporal = Path(tempfile.mkdtemp(prefix="onnx_"))
    resultado = {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__,
                 "opset": OPSET, "referencia": referencia, "semillas": {}}
    for s in SEMILLAS:
        pt = DIR_PESOS / f"s{s}.pt"
        if sha256(pt) != config["pesos"][f"s{s}"]:
            sys.exit(f"El SHA-256 de {pt.name} no es el de configuracion_dl.json: se para.")
        modelo = piloto.EEGNet()
        modelo.load_state_dict(torch.load(pt))
        modelo.eval()
        ruta = temporal / f"tt_eegnet_s{s}.onnx"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metodo = exportar(modelo, ruta)
        torch_p = piloto.puntuar(modelo, X, M, np.arange(len(X)))
        ort_p = puntuar_ort(ruta, X, M)
        dif = float(np.max(np.abs(torch_p - ort_p)))
        resultado["semillas"][f"s{s}"] = {"exportador": metodo, "dif_max": dif, "sha256_pt": sha256(pt),
                                          "bytes": ruta.stat().st_size}
        print(f"  s{s}: exportador {metodo}, {ruta.stat().st_size} bytes, diferencia máxima {dif:.2e}")

    peor = max(v["dif_max"] for v in resultado["semillas"].values())
    resultado["cumple"] = peor < TOLERANCIA
    if not resultado["cumple"]:
        (CARPETA / "resultados.json").write_text(json.dumps(resultado, indent=1), encoding="utf-8")
        sys.exit(f"ONNX ≠ PyTorch (diferencia {peor:.2e} ≥ {TOLERANCIA}): no se escribe nada en la librería.")

    DESTINO.mkdir(parents=True, exist_ok=True)
    semillas = []
    for s in SEMILLAS:
        origen = temporal / f"tt_eegnet_s{s}.onnx"
        destino = DESTINO / origen.name
        destino.write_bytes(origen.read_bytes())
        resultado["semillas"][f"s{s}"]["sha256_onnx"] = sha256(destino)
        semillas.append({"semilla": s, "archivo": destino.name, "sha256": sha256(destino),
                         "sha256_pt": resultado["semillas"][f"s{s}"]["sha256_pt"]})

    test = json.loads(RESULTADOS_TEST.read_text(encoding="utf-8"))
    c = test["cifras"]
    personas = {f["condicion"]: f["personas"] for f in test["inventario"]}
    grabaciones = sum(f["grabaciones"] for f in test["inventario"] if f["condicion"] in ("PASAT", "PVTB"))
    ficha = {
        "id": "tt_eegnet",
        "titulo": "Tiempo en tarea · Red neuronal (EEGNet)",
        "que_predice": "A qué parte de la tarea se parece cada ventana de la grabación, a partir de la señal "
                       "(no de la potencia por banda): al primer tercio o al último.",
        "condiciones_aplicables": ["PASAT", "PVTB"],
        "parametros_validos": {"referencia": "car_condicional", "banda_hz": [1.0, 30.0], "notch_hz": 50.0},
        "canales": CANALES,
        "entrada": {"forma": [2, 14, 512], "frecuencia_hz": config["frecuencia_hz"],
                    "preparacion": "ventana de 4 s remuestreada a 128 Hz (resample_poly 1/2), z-score por "
                                   "grabación y canal sobre todas sus ventanas, canales excluidos a 0; "
                                   "segundo mapa: máscara de canales válidos (02_piloto_cnn/preparar.py, tensor)"},
        "semillas": semillas,
        "combinacion": "media de los logits de las tres semillas",
        "resultado_test": {
            "metrica": "acierto_pareado_tercios", "valor": c["tercios_cnn_familia"]["valor"],
            "ic95": c["tercios_cnn_familia"]["ic95"], "azar": 0.5, "n_grabaciones": grabaciones,
            "n_personas": personas["PASAT"],
            "fuente": "analisis/tiempo_en_tarea/deep_learning/04_test_eegnet/resultados_test.json"},
        "resultado_test_rho": {"valor": c["cnn_familia"]["valor"], "ic95": c["cnn_familia"]["ic95"], "azar": 0.0},
        "comparacion_logistica": {"diferencia_rho": c["P1"]["valor"], "ic95": c["P1"]["ic95"],
                                  "p_bilateral": c["P1"]["p_bilateral"], "veredicto": c["P1"]["veredicto"]},
        "entrenamiento": {"fecha": config["fecha"][:10], "datos": "train + val; evaluado una sola vez en test",
                          "arquitectura": config["arquitectura"], "epocas": config["epocas"],
                          "commit_preregistro": test["commit"], "torch": config["entorno_entrenamiento"]["torch"]},
        "exportacion": {"fecha_paso": "2026-11-01", "torch": torch.__version__, "onnx": onnx.__version__,
                        "opset": OPSET, "dif_max_onnx_torch": peor},
        "avisos": [
            "Mide el cambio del EEG con el tiempo en tarea, no el nivel de fatiga de la persona.",
            "En el test no se distingue de la regresión logística (diferencia de ρ −0,011, IC95 −0,047 a +0,028).",
            "Las cifras del test se calcularon con cada una de las tres semillas por separado y se promediaron; "
            "la plataforma combina las tres en una sola puntuación por ventana (media de los logits).",
            "Solo es válido con los parámetros de procesado de su entrenamiento.",
        ],
    }
    (DESTINO / "tt_eegnet.json").write_text(json.dumps(ficha, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    (CARPETA / "resultados.json").write_text(json.dumps(resultado, indent=1), encoding="utf-8")
    print(f"Cumple (diferencia máxima {peor:.2e} < {TOLERANCIA}). Escritos {len(semillas)} .onnx y tt_eegnet.json.")


if __name__ == "__main__":
    main()
