"""
Fatiga aguda con EEGNet: la red neuronal de la línea, ejecutada con onnxruntime.

Son los tres modelos finales del paso 4 de `analisis/tiempo_en_tarea/deep_learning/` (EEGNet-8,2,
semillas 0, 1 y 2, entrenadas con `train` + `val` y evaluadas una vez en test), exportados a
ONNX en el paso 5 (`05_onnx/exportar.py`, ONNX = PyTorch con diferencia < 1e-5). La librería no
necesita PyTorch.

A diferencia de la logística, EEGNet trabaja con la **señal** de cada ventana, no con la potencia
por banda. Las ventanas crudas no se guardan (ocuparían mucho), así que la puntuación se calcula
**al procesar** (`analizar_grabacion` → `por_ventana["eegnet"]`) y aquí solo se resume.

Decisiones de diseño:

- **La entrada se prepara exactamente como en el entrenamiento** (`02_piloto_cnn/preparar.py`,
  `tensor`): remuestreo a 128 Hz con `resample_poly(x, 1, 2)`, z-score por canal con la media y
  la desviación de todas las ventanas de la grabación, canales excluidos a 0 y un segundo mapa
  con la máscara de canales válidos.
- **Una puntuación por ventana: la media de los logits de las tres semillas.** En el test, cada
  semilla se evaluó por separado y se promediaron las cifras; aquí se combinan para dar un único
  resultado. Se declara en la ficha.
- **Veredicto de tercios como en el análisis** (`04_test_eegnet/comun_dl.py`, `tercios_cnn`): la
  media de la puntuación del último tercio frente a la del primero.
- **Un hilo por sesión de onnxruntime**: el procesador de la plataforma ya ejecuta varias
  grabaciones en paralelo.
"""

import hashlib
import json
from importlib import resources
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.signal import resample_poly

CONDICIONES = ("PASAT", "PVTB")
MIN_VENTANAS = 10
MUESTRAS_ENTRADA = 512

_sesiones = []


def _dir_modelos():
    return Path(str(resources.files("eeg_library") / "modelos" / "fatiga_aguda"))


def leer_ficha_eegnet():
    """Ficha de EEGNet (cifras del test, semillas y sus SHA-256, avisos)."""
    return json.loads((_dir_modelos() / "fa_eegnet.json").read_text(encoding="utf-8"))


def _sesiones_onnx():
    """Las tres sesiones de onnxruntime, comprobando antes el SHA-256 de cada `.onnx`."""
    if not _sesiones:
        import onnxruntime as ort
        opciones = ort.SessionOptions()
        opciones.intra_op_num_threads = 1
        opciones.inter_op_num_threads = 1
        for s in leer_ficha_eegnet()["semillas"]:
            ruta = _dir_modelos() / s["archivo"]
            if hashlib.sha256(ruta.read_bytes()).hexdigest() != s["sha256"]:
                raise ValueError(f"El SHA-256 de '{ruta.name}' no es el de su ficha.")
            _sesiones.append(ort.InferenceSession(str(ruta), opciones,
                                                  providers=["CPUExecutionProvider"]))
    return _sesiones


def preparar_entrada(ventanas, canal_valido):
    """
    Entrada de EEGNet a partir de las ventanas limpias de una grabación.

    Parámetros
    ----------
    ventanas : array (n, 14, 1024)
        Ventanas de 4 s a 256 Hz, en el orden de `CANALES_EPOC` (las de `ejecutar_fases` con
        `devolver_ventanas=True`).
    canal_valido : sequence of bool (14,)
        Canales válidos de la grabación.

    Devuelve
    --------
    (X, M): `X` float32 (n, 14, 512) y `M` uint8 (n, 14), como `tensor` del piloto.
    """
    x = resample_poly(np.asarray(ventanas).astype(np.float64), 1, 2, axis=-1)
    media = x.mean(axis=(0, 2), keepdims=True)
    desv = x.std(axis=(0, 2), keepdims=True)
    x = (x - media) / np.where(desv > 0, desv, 1.0)
    valido = np.asarray(canal_valido, dtype=bool)
    x[:, ~valido, :] = 0.0
    mascara = np.repeat(valido[None, :], len(x), axis=0).astype(np.uint8)
    return x.astype(np.float32), mascara


def puntuar_entrada(X, M, lote=512):
    """Logit de cada ventana con las tres semillas: array (3, n)."""
    salida = []
    for sesion in _sesiones_onnx():
        nombre = sesion.get_inputs()[0].name
        partes = []
        for i in range(0, len(X), lote):
            x = X[i:i + lote]
            m = np.broadcast_to(M[i:i + lote].astype(np.float32)[:, :, None], x.shape)
            partes.append(sesion.run(None, {nombre: np.stack([x, m], axis=1)})[0].reshape(-1))
        salida.append(np.concatenate(partes) if partes else np.zeros(0, dtype=np.float32))
    return np.vstack(salida)


def puntuar_ventanas(ventanas, canal_valido):
    """Media de los logits de las tres semillas, una por ventana (lo que guarda `por_ventana`)."""
    X, M = preparar_entrada(ventanas, canal_valido)
    return puntuar_entrada(X, M).mean(axis=0)


def _no_disponible(condicion, motivo):
    return {"disponible": False, "condicion": condicion, "modelo": "fa_eegnet", "motivo": motivo}


def puntuar_eegnet(resultado, condicion):
    """
    Resume la puntuación de EEGNet de una grabación PASAT o PVTB.

    Devuelve, si se puede, `disponible=True`, `modelo`, `condicion`, `n_ventanas`, `instante_s`,
    `puntuacion` (una por ventana; positiva = se parece al último tercio), `rho` (Spearman entre
    puntuación e instante) y `tercios` (`media_primero`, `media_ultimo` y `final_mayor`). Si no,
    `disponible=False` con el `motivo`.
    """
    from eeg_library.modelado.prediccion import comparar_parametros

    condicion = (condicion or "").strip().upper()
    if condicion not in CONDICIONES:
        return _no_disponible(condicion, "Solo hay modelo de fatiga aguda para PASAT y PVTB.")
    ficha = leer_ficha_eegnet()
    distintos, _, _ = comparar_parametros(ficha, (resultado or {}).get("parametros_aplicados"))
    if distintos:
        return _no_disponible(condicion, "El modelo solo es válido con los parámetros con que se "
                              f"entrenó; difieren: {', '.join(distintos)}.")
    por_ventana = (resultado or {}).get("por_ventana") or {}
    if por_ventana.get("eegnet") is None:
        return _no_disponible(condicion, "La grabación se procesó con una versión anterior de la "
                              "librería, que no calculaba la puntuación de EEGNet. Hay que volver "
                              "a procesarla.")
    instantes = np.asarray(por_ventana["instante_s"], dtype=float)
    puntuacion = np.asarray(por_ventana["eegnet"], dtype=float)
    if len(instantes) < MIN_VENTANAS:
        return _no_disponible(condicion, f"La grabación tiene {len(instantes)} ventanas y el "
                              f"modelo necesita al menos {MIN_VENTANAS}.")
    t = instantes / 60
    rho = float(stats.spearmanr(puntuacion, t)[0])
    primero, ultimo = t <= t.max() / 3, t >= 2 * t.max() / 3
    a, b = float(puntuacion[primero].mean()), float(puntuacion[ultimo].mean())
    return {
        "disponible": True, "modelo": ficha["id"], "condicion": condicion,
        "n_ventanas": int(len(instantes)), "instante_s": instantes.tolist(),
        "puntuacion": puntuacion.tolist(), "rho": None if np.isnan(rho) else rho,
        "tercios": {"media_primero": a, "media_ultimo": b,
                    "final_mayor": None if a == b else bool(b > a)},
    }
