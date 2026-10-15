"""
Empaqueta los modelos finales c1, c2 y c3 (regresión logística) en la librería.

Copia sin modificar los `.joblib` del análisis confirmatorio del 2026-10-07
(`<EEG_DATA>/processed/test_20261007-133358/modelos/`) a
`src/eeg_library/modelos/` y escribe al lado la ficha `.json` de cada uno.

Nada de la ficha se teclea a mano:

- las cifras del test (exactitud balanceada, IC95, n) salen de
  `resultados.json`;
- los nombres de las clases salen de las constantes del código de
  entrenamiento (`reposo`, `reposo_tarea`, `tareas_ejecutivas`), las mismas
  con las que `analisis_test.py` construyó las etiquetas;
- las columnas, del propio Pipeline (`feature_names_in_`);
- el SHA-256 de `matriz_rf.parquet` se recalcula y se exige que sea el que
  registró el análisis confirmatorio.

No escribe nada en `EEG_DATA`. Volver a ejecutarlo con los mismos datos da
archivos idénticos.

Uso:
    .venv/Scripts/python.exe analisis/empaquetar_modelos.py
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import joblib
import pandas as pd

from eeg_library.modelado.reposo import (CLASE_NEGATIVA as OA, CLASE_POSITIVA as OC,
                                         CONDICIONES_REPOSO, columnas_de_features_reposo)
from eeg_library.modelado.reposo_tarea import (CLASE_NEGATIVA as REPOSO,
                                               CLASE_POSITIVA as TAREA)
from eeg_library.modelado.tareas_ejecutivas import CODIGO_DE_CLASE, CONDICIONES_EJECUTIVAS
from eeg_library.procesado.matrix import ORDEN_PROTOCOLO
from eeg_library.procesado.parametros import PARAMETROS_ESTUDIO

RAIZ_REPO = Path(__file__).resolve().parents[1]
RUTA_DATOS = Path(os.environ.get("EEG_DATA", r"C:\Users\migue\Desktop\TFG\EEG-Data"))
DIR_TEST = RUTA_DATOS / "processed" / "test_20261007-133358"
RUTA_MATRIZ = RUTA_DATOS / "processed" / "matriz_rf.parquet"
DESTINO = RAIZ_REPO / "src" / "eeg_library" / "modelos"

AVISOS_COMUNES = [
    "No mide fatiga ni ningún estado del participante: clasifica la condición "
    "de la grabación.",
    "Parte del acierto puede deberse a la duración de la grabación o al orden "
    "de las tareas en el protocolo, no solo a la actividad cerebral.",
    "Solo es válido con los parámetros de procesado de su entrenamiento.",
]

MODELOS = {
    "c1_oa_oc_logistica": {
        "clave_resultados": "C1_oa_oc",
        "titulo": "Ojos abiertos / ojos cerrados",
        "que_predice": "Si una grabación de reposo es con ojos abiertos (OA) o "
                       "con ojos cerrados (OC).",
        "clases": {0: OA, 1: OC},
        "condiciones_aplicables": list(CONDICIONES_REPOSO),
    },
    "c2_reposo_tarea_logistica": {
        "clave_resultados": "C2_reposo_tarea",
        "titulo": "Reposo / tarea",
        "que_predice": "Si una grabación es de reposo (OA u OC) o de una tarea "
                       "cognitiva.",
        "clases": {0: REPOSO, 1: TAREA},
        "condiciones_aplicables": list(ORDEN_PROTOCOLO),
    },
    "c3_tareas_logistica": {
        "clave_resultados": "C3_tareas",
        "titulo": "Cuál de las 5 tareas",
        "que_predice": "Cuál de las cinco tareas cognitivas (STROOP, PASAT, "
                       "PVTB, MSIT, PVSAT) se estaba haciendo.",
        "clases": {codigo: nombre for nombre, codigo in CODIGO_DE_CLASE.items()},
        "condiciones_aplicables": list(CONDICIONES_EJECUTIVAS),
    },
}


def sha256(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def version_en_freeze(freeze, paquete):
    for linea in freeze:
        if linea.lower().startswith(f"{paquete}=="):
            return linea.split("==", 1)[1]
    raise ValueError(f"'{paquete}' no aparece en el pip freeze del análisis.")


def main():
    resultados = json.loads((DIR_TEST / "resultados.json").read_text(encoding="utf-8"))
    meta = resultados["metadatos"]
    if meta["modo"] != "confirmatorio" or meta["entrena"] != ["train", "val"]:
        sys.exit("resultados.json no es el del análisis confirmatorio con train+val.")

    sha_matriz = sha256(RUTA_MATRIZ)
    if sha_matriz != meta["datos_sha256"]["matriz_rf.parquet"]:
        sys.exit("matriz_rf.parquet no es la que usó el análisis confirmatorio.")
    matriz = pd.read_parquet(RUTA_MATRIZ)
    columnas_esperadas = columnas_de_features_reposo(matriz)

    entorno = meta["entorno"]
    entrenamiento = {
        "fecha": meta["inicio"][:10],
        "datos": "train + val; evaluado una sola vez en test",
        "variante": "regresión logística, la principal del pre-registro "
                    "(PREREGISTRO_TEST.md §6)",
        "sha256_matriz_rf": sha_matriz,
        "commit_eeg_library": meta["git"]["commit"],
        "python": entorno["python"].split()[0],
        "scikit_learn": version_en_freeze(entorno["pip_freeze"], "scikit-learn"),
    }

    DESTINO.mkdir(parents=True, exist_ok=True)
    for id_modelo, spec in MODELOS.items():
        origen = DIR_TEST / "modelos" / f"{id_modelo}.joblib"
        destino = DESTINO / origen.name
        shutil.copyfile(origen, destino)
        sha_modelo = sha256(destino)
        if sha_modelo != sha256(origen):
            sys.exit(f"La copia de {origen.name} no es idéntica al original.")

        pipeline = joblib.load(destino)
        columnas = [str(c) for c in pipeline.feature_names_in_]
        if columnas != columnas_esperadas:
            sys.exit(f"{id_modelo}: sus columnas no son columnas_de_features_reposo.")
        codigos = [int(c) for c in pipeline.classes_]
        if codigos != sorted(spec["clases"]):
            sys.exit(f"{id_modelo}: clases {codigos} distintas de las esperadas.")

        res = resultados["familia_condicion"][spec["clave_resultados"]]
        logistica = res["modelos"]["logistica"]
        ficha = {
            "id": id_modelo,
            "titulo": spec["titulo"],
            "archivo": origen.name,
            "sha256_modelo": sha_modelo,
            "que_predice": spec["que_predice"],
            "clases": [{"codigo": c, "nombre": spec["clases"][c]} for c in codigos],
            "condiciones_aplicables": spec["condiciones_aplicables"],
            "parametros_validos": dict(PARAMETROS_ESTUDIO),
            "columnas": columnas,
            "resultado_test": {
                "metrica": "exactitud_balanceada",
                "valor": logistica["exactitud_balanceada"],
                "ic95": logistica["ic95"],
                "azar": 1 / len(codigos),
                "n_grabaciones": res["n_grabaciones"],
                "n_personas": res["n_personas"],
                "fuente": "resultados.json del análisis confirmatorio "
                          f"({DIR_TEST.name})",
            },
            "entrenamiento": entrenamiento,
            "avisos": AVISOS_COMUNES,
        }
        ruta_ficha = DESTINO / f"{id_modelo}.json"
        ruta_ficha.write_text(json.dumps(ficha, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
        r = ficha["resultado_test"]
        print(f"{id_modelo}: BA {r['valor']:.3f} [{r['ic95'][0]:.3f}–{r['ic95'][1]:.3f}], "
              f"azar {r['azar']:.2f}, clases {[c['nombre'] for c in ficha['clases']]}")


if __name__ == "__main__":
    main()
