"""
Empaqueta los modelos de tiempo en tarea (logística de PASAT y PVTB) en la librería.

Copia sin modificar `completa_PASAT.joblib` y `completa_PVTB.joblib` de la
evaluación única en test de la línea (2026-10-27,
`<EEG_DATA>/processed/fatiga_aguda_test_20261027-190547/modelos/`) a
`src/eeg_library/modelos/tiempo_en_tarea/` y escribe al lado la ficha `.json`
de cada uno. Los usa `eeg_library.modelado.tiempo_en_tarea`.

Como en `empaquetar_modelos.py`, nada de la ficha se teclea a mano:

- las cifras del test (acierto pareado de tercios y ρ, con IC95 y número de
  grabaciones; personas del inventario) salen de
  `analisis/tiempo_en_tarea/06_test/resultados_test.json`;
- la configuración (`rel` + `lr_0.01`, una por condición) se comprueba con
  `03_clasificador/configuracion_elegida.json`;
- el orden de las 56 entradas es el de `REL` del análisis (canal × banda);
- los SHA-256 de los datos de entrenamiento y el commit, del mismo
  `resultados_test.json`, y se exige que el de `features_all.parquet` siga
  siendo el mismo.

No escribe nada en `EEG_DATA`. Volver a ejecutarlo da archivos idénticos.

Uso:
    .venv/Scripts/python.exe analisis/empaquetar_modelos_tiempo_en_tarea.py
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import joblib

from eeg_library.modelado.tiempo_en_tarea import BANDAS, IDS
from eeg_library.preprocesado.quality import CANALES_EPOC
from eeg_library.procesado.parametros import PARAMETROS_ESTUDIO

RAIZ_REPO = Path(__file__).resolve().parents[1]
RUTA_DATOS = Path(os.environ.get("EEG_DATA", r"C:\Users\migue\Desktop\TFG\EEG-Data"))
DIR_TEST = RUTA_DATOS / "processed" / "fatiga_aguda_test_20261027-190547"
LINEA = RAIZ_REPO / "analisis" / "tiempo_en_tarea"
RESULTADOS = LINEA / "06_test" / "resultados_test.json"
CONFIG = LINEA / "03_clasificador" / "configuracion_elegida.json"
DESTINO = RAIZ_REPO / "src" / "eeg_library" / "modelos" / "tiempo_en_tarea"

AVISOS = [
    "Mide el cambio del EEG con el tiempo en tarea, no el nivel de fatiga de la persona.",
    "Una ventana suelta de 4 s apenas dice nada (AUC por ventana ≈ 0,58); lo fiable es la "
    "tendencia de la grabación entera.",
    "Parte del cambio puede deberse al asentamiento del casco: en reposo con ojos abiertos "
    "también aparece (acierto pareado 0,670 en test).",
    "El orden de las tareas en el protocolo es fijo.",
    "Solo es válido con los parámetros de procesado de su entrenamiento.",
]


def sha256(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def main():
    resultados = json.loads(RESULTADOS.read_text(encoding="utf-8"))
    if resultados["modo"] != "test":
        sys.exit("resultados_test.json no es el de la evaluación en test.")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (config["features"], config["modelo"], config["esquema"]) != ("rel", "lr_0.01", "separado"):
        sys.exit("La configuración elegida no es rel + lr_0.01 + separado.")
    ruta_features = RUTA_DATOS / "processed" / "features_all.parquet"
    if sha256(ruta_features) != resultados["sha256"]["features_all.parquet"]:
        sys.exit("features_all.parquet no es el que usó la evaluación en test.")

    personas = {f["condicion"]: f["personas"] for f in resultados["inventario"]}
    columnas = [f"pot_rel_{c}_{b}" for c in CANALES_EPOC for b in BANDAS]
    entrenamiento = {
        "fecha": resultados["inicio"][:10],
        "datos": "train + val; evaluado una sola vez en test",
        "variante": "potencia relativa por canal y banda (log10, centrada por grabación) + "
                    "regresión logística C = 0,01, un modelo por tarea "
                    "(analisis/tiempo_en_tarea/06_test/PREREGISTRO.md)",
        "objetivo": "distinguir el primer tercio de la grabación (0) del último (1)",
        "sha256_datos": {k: resultados["sha256"][k] for k in
                         ("features_all.parquet", "matriz_rf.parquet", "procesamiento_log.csv")},
        "commit_eeg_library": resultados["commit"],
        "python": resultados["entorno"]["python"],
        "scikit_learn": resultados["entorno"]["sklearn"],
    }

    DESTINO.mkdir(parents=True, exist_ok=True)
    for condicion, id_modelo in IDS.items():
        origen = DIR_TEST / "modelos" / f"completa_{condicion}.joblib"
        destino = DESTINO / f"{id_modelo}.joblib"
        shutil.copyfile(origen, destino)
        sha_modelo = sha256(destino)
        if sha_modelo != sha256(origen):
            sys.exit(f"La copia de {origen.name} no es idéntica al original.")
        pipeline = joblib.load(destino)
        if pipeline.n_features_in_ != len(columnas) or list(pipeline.classes_) != [0, 1]:
            sys.exit(f"{id_modelo}: no tiene 56 entradas y las clases 0 y 1.")

        tercios = resultados["descriptivos"][f"completa | {condicion} | tercios_pareado"]
        rho = resultados["descriptivos"][f"completa | {condicion} | rho"]
        ficha = {
            "id": id_modelo,
            "titulo": f"Tiempo en tarea · {condicion}",
            "archivo": destino.name,
            "sha256_modelo": sha_modelo,
            "que_predice": "A qué parte de la tarea se parece cada ventana de la grabación: al "
                           "primer tercio o al último.",
            "clases": [{"codigo": 0, "nombre": "primer tercio"},
                       {"codigo": 1, "nombre": "último tercio"}],
            "condiciones_aplicables": [condicion],
            "parametros_validos": dict(PARAMETROS_ESTUDIO),
            "canales": list(CANALES_EPOC),
            "columnas": columnas,
            "resultado_test": {
                "metrica": "acierto_pareado_tercios",
                "valor": tercios["valor"],
                "ic95": tercios["ic95"],
                "azar": 0.5,
                "n_grabaciones": tercios["grabaciones"],
                "n_personas": personas[condicion],
                "fuente": f"analisis/tiempo_en_tarea/06_test/resultados_test.json ({DIR_TEST.name})",
            },
            "resultado_test_rho": {
                "valor": rho["valor"], "ic95": rho["ic95"], "azar": 0.0,
                "n_grabaciones": rho["grabaciones"],
            },
            "entrenamiento": entrenamiento,
            "avisos": AVISOS,
        }
        (DESTINO / f"{id_modelo}.json").write_text(
            json.dumps(ficha, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        t = ficha["resultado_test"]
        print(f"{id_modelo}: acierto pareado {t['valor']:.3f} "
              f"[{t['ic95'][0]:.3f}–{t['ic95'][1]:.3f}], {t['n_grabaciones']} grabaciones, "
              f"{t['n_personas']} personas; ρ {rho['valor']:+.3f}")


if __name__ == "__main__":
    main()
