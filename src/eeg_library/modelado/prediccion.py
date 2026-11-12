"""
Modelos enchufables: carga con su ficha y predicción sobre una grabación.

Un modelo es un `.joblib` (un Pipeline de scikit-learn) más una ficha `.json`
con el mismo nombre al lado. La ficha dice qué predice, cómo se llaman sus
clases, con qué columnas y con qué parámetros de procesado es válido, y cuál
fue su resultado en el test independiente. Los modelos del estudio (c1, c2 y
c3) viven en `eeg_library/modelos/`; un modelo externo se carga dando la
ruta de su ficha.

Decisiones de diseño:

- **Sin reglas por modelo en el código.** `predecir` solo compara los
  parámetros que declara la ficha con los aplicados a la grabación. Si no
  coinciden, devuelve «no disponible» con lo que haría falta; nunca predice
  sobre una señal procesada de otra forma que la de entrenamiento.
- **Integridad.** La ficha guarda el SHA-256 del `.joblib` y la carga falla
  si no coincide: una ficha no puede describir un modelo distinto del que
  se ejecuta.
- **Clases con nombre.** El Pipeline predice códigos enteros; la ficha los
  traduce a nombres confirmados en el código de entrenamiento.
"""

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from eeg_library.procesado.grabacion import ALIAS_CARACTERISTICAS
from eeg_library.procesado.parametros import OPCIONES, normalizar_parametros

CLAVES_FICHA = (
    "id", "archivo", "sha256_modelo", "que_predice", "clases",
    "condiciones_aplicables", "parametros_validos", "columnas",
)


@dataclass
class Modelo:
    """Un Pipeline de scikit-learn junto con su ficha."""
    id: str
    ficha: dict
    pipeline: object


def _dir_modelos(*subcarpetas):
    """
    Carpeta de los modelos empaquetados, o una subcarpeta suya.

    Es la única definición de dónde viven los modelos: la usan también los de
    tiempo en tarea (`_dir_modelos("tiempo_en_tarea")`).
    """
    return Path(str(resources.files("eeg_library") / "modelos")).joinpath(*subcarpetas)


def _sha256(ruta):
    """SHA-256 de un archivo de modelo, para comprobarlo contra su ficha."""
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def listar_modelos():
    """Identificadores de los modelos empaquetados con la librería."""
    return sorted(p.stem for p in _dir_modelos().glob("*.json"))


def leer_ficha(id_o_ruta):
    """
    Lee la ficha de un modelo sin cargar el `.joblib`.

    `id_o_ruta` es el identificador de un modelo empaquetado (ver
    `listar_modelos`) o la ruta a la ficha `.json` de un modelo externo.
    Devuelve `(ficha, ruta_de_la_ficha)`.
    """
    ruta = Path(id_o_ruta)
    if ruta.suffix.lower() != ".json":
        ruta = _dir_modelos() / f"{id_o_ruta}.json"
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No existe la ficha '{ruta}'. Modelos empaquetados: {listar_modelos()}."
        )
    ficha = json.loads(ruta.read_text(encoding="utf-8"))
    faltan = [c for c in CLAVES_FICHA if c not in ficha]
    if faltan:
        raise ValueError(f"A la ficha '{ruta.name}' le faltan claves: {faltan}.")
    return ficha, ruta


def cargar_modelo(id_o_ruta):
    """
    Carga un modelo y su ficha, comprobando que se corresponden.

    Raises
    ------
    ValueError
        Si el SHA-256 del `.joblib` no es el de la ficha, o si las columnas o
        las clases del Pipeline no son las que declara la ficha.
    """
    ficha, ruta_ficha = leer_ficha(id_o_ruta)
    ruta_modelo = ruta_ficha.parent / ficha["archivo"]
    sha = _sha256(ruta_modelo)
    if sha != ficha["sha256_modelo"]:
        raise ValueError(
            f"El SHA-256 de '{ruta_modelo.name}' ({sha[:12]}…) no es el de su "
            f"ficha ({ficha['sha256_modelo'][:12]}…)."
        )
    pipeline = joblib.load(ruta_modelo)

    columnas = list(getattr(pipeline, "feature_names_in_", []))
    if columnas != list(ficha["columnas"]):
        raise ValueError(
            f"Las columnas del modelo '{ficha['id']}' no son las de su ficha."
        )
    codigos = [int(c) for c in pipeline.classes_]
    if codigos != [c["codigo"] for c in ficha["clases"]]:
        raise ValueError(
            f"Las clases del modelo '{ficha['id']}' ({codigos}) no son las de su "
            f"ficha ({[c['codigo'] for c in ficha['clases']]})."
        )
    return Modelo(id=ficha["id"], ficha=ficha, pipeline=pipeline)


def _fila(ficha, caracteristicas):
    """Una fila con las columnas del modelo, deshaciendo los alias neutros."""
    fila, faltan = {}, []
    for columna in ficha["columnas"]:
        alias = ALIAS_CARACTERISTICAS.get(columna)
        if columna in caracteristicas:
            valor = caracteristicas[columna]
        elif alias in caracteristicas:
            valor = caracteristicas[alias]
        else:
            faltan.append(columna)
            continue
        fila[columna] = np.nan if valor is None else float(valor)
    if faltan:
        raise ValueError(
            f"Faltan {len(faltan)} característica(s) que necesita el modelo "
            f"'{ficha['id']}': {faltan[:5]}" + (" ..." if len(faltan) > 5 else "")
        )
    return pd.DataFrame([fila], columns=list(ficha["columnas"]))


def comparar_parametros(ficha, parametros):
    """
    Compara los parámetros de procesado de una grabación con los de la ficha.

    Solo cuentan las claves elegibles (`OPCIONES`); las demás se ignoran, así
    que `parametros_aplicados` sirve tal cual. Devuelve `(distintos,
    requeridos, aplicados)`: la lista de claves que no coinciden y los dos
    juegos de parámetros normalizados.
    """
    aplicados = normalizar_parametros({k: v for k, v in (parametros or {}).items()
                                       if k in OPCIONES})
    requeridos = normalizar_parametros(ficha["parametros_validos"])
    distintos = [k for k in requeridos if requeridos[k] != aplicados[k]]
    return distintos, requeridos, aplicados


def predecir(modelo, caracteristicas, parametros):
    """
    Predice la clase de una grabación, si sus parámetros lo permiten.

    Parámetros
    ----------
    modelo : Modelo o str
        Modelo cargado, o identificador / ruta de ficha para `cargar_modelo`.
    caracteristicas : dict
        `analizar_grabacion(...)["caracteristicas"]` (admite también los
        nombres originales de la matriz). `None` cuenta como valor ausente y
        lo imputa el propio Pipeline.
    parametros : dict
        Parámetros de procesado aplicados a la grabación
        (`analizar_grabacion(...)["parametros_aplicados"]` sirve tal cual:
        las claves que no son de `OPCIONES` se ignoran).

    Devuelve
    --------
    dict
        Con parámetros válidos: `disponible=True`, `modelo`, `clase` (nombre),
        `codigo` y `probabilidades` {nombre: p} en el orden de la ficha.
        Si no: `disponible=False`, `modelo`, `motivo`,
        `parametros_requeridos` y `parametros_aplicados`.
    """
    if not isinstance(modelo, Modelo):
        modelo = cargar_modelo(modelo)
    ficha = modelo.ficha

    distintos, requeridos, aplicados = comparar_parametros(ficha, parametros)
    if distintos:
        return {
            "disponible": False,
            "modelo": modelo.id,
            "motivo": ("El modelo solo es válido con los parámetros con que se "
                       f"entrenó; difieren: {', '.join(distintos)}."),
            "parametros_requeridos": requeridos,
            "parametros_aplicados": aplicados,
        }

    X = _fila(ficha, caracteristicas)
    proba = modelo.pipeline.predict_proba(X)[0]
    nombres = {int(c["codigo"]): c["nombre"] for c in ficha["clases"]}
    codigos = [int(c) for c in modelo.pipeline.classes_]
    ganadora = codigos[int(np.argmax(proba))]
    return {
        "disponible": True,
        "modelo": modelo.id,
        "clase": nombres[ganadora],
        "codigo": ganadora,
        "probabilidades": {nombres[c]: float(p) for c, p in zip(codigos, proba)},
    }
