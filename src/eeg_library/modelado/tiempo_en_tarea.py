"""
Tiempo en tarea: cómo cambia el EEG a lo largo de una tarea PASAT o PVTB.

Aplica a una grabación suelta los modelos de la línea
`analisis/tiempo_en_tarea/` (paso 6, evaluados una vez en test el 2026-10-27):
una regresión logística por tarea, entrenada con `train` + `val` para
distinguir el primer tercio de la grabación del último. Mide el **cambio con
el tiempo en tarea**, no el nivel de fatiga de la persona.

Los modelos viven en `eeg_library/modelos/tiempo_en_tarea/` (los empaqueta
`analisis/empaquetar_modelos_tiempo_en_tarea.py`) y trabajan con el bloque
`por_ventana` de `analizar_grabacion`.

Decisiones de diseño:

- **Exactamente el cálculo del análisis**, para que la cifra del test que
  acompaña al resultado sea la del mismo cálculo:
  - puntuación por ventana como `medidas.ventanas` + `decision_function`
    (lo que mide la ρ del test, `rho_ventanas`);
  - veredicto principio/final como `comun.tercios` + `comun.matriz` +
    `predict_proba` (lo que mide el acierto pareado del test).
- **Separados de c1–c3.** Sus 56 entradas son por ventana y no tienen
  nombres de columna en el Pipeline, así que no pasan por `cargar_modelo`;
  su carga hace las mismas comprobaciones de integridad.
- **«No disponible» con motivo**, nunca una puntuación dudosa: otra
  condición, otros parámetros de procesado, menos de 10 ventanas o una
  grabación procesada antes de que existiera `por_ventana`.
"""

import hashlib
from importlib import resources
from pathlib import Path

import joblib
import numpy as np
from scipy import stats

from eeg_library.modelado.prediccion import Modelo, comparar_parametros, leer_ficha

CONDICIONES = ("PASAT", "PVTB")
BANDAS = ("delta", "theta", "alfa", "beta")
MIN_VENTANAS = 10          # el mínimo de la línea (`d1.MIN_VENTANAS`)
SUELO_POT_REL = 1e-6       # recorte antes del log10, como en el análisis

IDS = {"PASAT": "tt_pasat_logistica", "PVTB": "tt_pvtb_logistica"}

_cache = {}


def dir_modelos():
    """Carpeta de los modelos de tiempo en tarea empaquetados."""
    return Path(str(resources.files("eeg_library") / "modelos" / "tiempo_en_tarea"))


def leer_ficha_tiempo_en_tarea(condicion):
    """Ficha del modelo de una condición (PASAT o PVTB), sin cargar el `.joblib`."""
    return leer_ficha(dir_modelos() / f"{IDS[condicion.upper()]}.json")[0]


def cargar_modelo_tiempo_en_tarea(condicion):
    """
    Carga el modelo de una condición y comprueba que es el que describe su ficha.

    Raises
    ------
    KeyError
        Si la condición no es PASAT ni PVTB.
    ValueError
        Si el SHA-256, el número de entradas o las clases del Pipeline no son
        los de la ficha.
    """
    condicion = condicion.upper()
    if condicion in _cache:
        return _cache[condicion]
    ficha, ruta_ficha = leer_ficha(dir_modelos() / f"{IDS[condicion]}.json")
    ruta_modelo = ruta_ficha.parent / ficha["archivo"]
    sha = hashlib.sha256(ruta_modelo.read_bytes()).hexdigest()
    if sha != ficha["sha256_modelo"]:
        raise ValueError(f"El SHA-256 de '{ruta_modelo.name}' no es el de su ficha.")
    pipeline = joblib.load(ruta_modelo)
    if pipeline.n_features_in_ != len(ficha["columnas"]):
        raise ValueError(f"El modelo '{ficha['id']}' no tiene las entradas de su ficha.")
    if [int(c) for c in pipeline.classes_] != [c["codigo"] for c in ficha["clases"]]:
        raise ValueError(f"Las clases del modelo '{ficha['id']}' no son las de su ficha.")
    modelo = Modelo(id=ficha["id"], ficha=ficha, pipeline=pipeline)
    _cache[condicion] = modelo
    return modelo


def _columnas(canales):
    """Orden de las 56 entradas: canal por canal, y dentro de cada uno, las 4 bandas."""
    return [f"pot_rel_{c}_{b}" for c in canales for b in BANDAS]


def _matriz(por_ventana, canales):
    """`pot_rel` lineal (n_ventanas × 56) y máscara de canales válidos (56,)."""
    pot = np.column_stack([np.asarray(por_ventana["pot_rel"][c][b], dtype=float)
                           for c in canales for b in BANDAS])
    valido = np.repeat([bool(por_ventana["canal_valido"][c]) for c in canales], len(BANDAS))
    return pot, valido


def _centrar(X, valido):
    """Resta la media de las filas (por grabación) y pone a 0 los canales excluidos."""
    X = X - X.mean(axis=0, keepdims=True)
    return np.where(valido, X, 0.0)


def _no_disponible(condicion, motivo, ficha=None):
    return {"disponible": False, "condicion": condicion, "motivo": motivo,
            "modelo": ficha["id"] if ficha else None}


def puntuar(resultado, condicion):
    """
    Puntúa una grabación con el modelo de tiempo en tarea de su condición.

    Parámetros
    ----------
    resultado : dict
        Salida de `analizar_grabacion` (o el JSON guardado por la web).
    condicion : str
        Condición de la grabación. Solo PASAT y PVTB tienen modelo.

    Devuelve
    --------
    dict
        Si se puede: `disponible=True`, `modelo`, `condicion`, `n_ventanas`,
        `instante_s` y `puntuacion` (una por ventana; positiva = se parece al
        último tercio de la tarea, negativa = al primero), `rho` (Spearman
        entre puntuación e instante), `tercios` (`p_primero`, `p_ultimo`:
        probabilidad de «último tercio» de cada uno, y `final_mayor`: True,
        False o None si empatan) y `pesos` {canal: {banda: peso}} (coeficiente
        dividido por la escala del `StandardScaler`; descriptivo).
        Si no: `disponible=False`, `condicion`, `modelo` y `motivo`.
    """
    condicion = (condicion or "").strip().upper()
    if condicion not in CONDICIONES:
        return _no_disponible(condicion, "Solo hay modelo de tiempo en tarea para PASAT y PVTB.")
    modelo = cargar_modelo_tiempo_en_tarea(condicion)
    ficha = modelo.ficha

    distintos, _, _ = comparar_parametros(ficha, (resultado or {}).get("parametros_aplicados"))
    if distintos:
        return _no_disponible(condicion, "El modelo solo es válido con los parámetros con que "
                              f"se entrenó; difieren: {', '.join(distintos)}.", ficha)
    por_ventana = (resultado or {}).get("por_ventana")
    if not por_ventana:
        return _no_disponible(condicion, "La grabación se procesó con una versión anterior de "
                              "la librería, que no guardaba la potencia por ventana. Hay que "
                              "volver a procesarla.", ficha)
    instantes = np.asarray(por_ventana["instante_s"], dtype=float)
    if len(instantes) < MIN_VENTANAS:
        return _no_disponible(condicion, f"La grabación tiene {len(instantes)} ventanas y el "
                              f"modelo necesita al menos {MIN_VENTANAS}.", ficha)

    canales = list(ficha["canales"])
    pot, valido = _matriz(por_ventana, canales)
    pipeline = modelo.pipeline

    # Por ventana (medidas.ventanas): log10 de cada ventana, centrado en la grabación.
    X = _centrar(np.log10(np.clip(pot, SUELO_POT_REL, None)), valido)
    puntuacion = pipeline.decision_function(X)
    t = instantes / 60
    rho = float(stats.spearmanr(puntuacion, t)[0])

    # Tercios (comun.tercios + comun.matriz): media lineal por tercio, log10, centrado.
    dur = t.max()
    primero, ultimo = t <= dur / 3, t >= 2 * dur / 3
    T = np.log10(np.clip(np.vstack([pot[primero].mean(axis=0), pot[ultimo].mean(axis=0)]),
                         SUELO_POT_REL, None))
    p_primero, p_ultimo = pipeline.predict_proba(_centrar(T, valido))[:, 1]
    final_mayor = None if p_ultimo == p_primero else bool(p_ultimo > p_primero)

    escalado, logistica = pipeline.steps[0][1], pipeline.steps[-1][1]
    pesos = (logistica.coef_[0] / escalado.scale_).reshape(len(canales), len(BANDAS))
    return {
        "disponible": True,
        "modelo": ficha["id"],
        "condicion": condicion,
        "n_ventanas": int(len(instantes)),
        "instante_s": instantes.tolist(),
        "puntuacion": [float(v) for v in puntuacion],
        "rho": None if np.isnan(rho) else rho,
        "tercios": {"p_primero": float(p_primero), "p_ultimo": float(p_ultimo),
                    "final_mayor": final_mayor},
        "pesos": {c: {b: float(pesos[i, j]) for j, b in enumerate(BANDAS)}
                  for i, c in enumerate(canales)},
    }
