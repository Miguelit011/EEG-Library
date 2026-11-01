"""
Paridad de los modelos de fatiga aguda con el análisis, sobre EDF reales.

Necesita `EEG_DATA` (EDF, `features_all.parquet`, `matriz_rf.parquet`,
`procesamiento_log.csv` y los modelos de la evaluación en test de la línea),
así que va con el marcador `datos`:

    .venv/Scripts/python.exe -m pytest -m datos tests/datos/test_fatiga_aguda_datos.py -s

Usa 20 grabaciones PASAT y PVTB de `train` y `val`, elegidas con semilla fija.
**Nunca de `test`**: la lista se filtra por la `split` de `matriz_rf.parquet`
antes de abrir ningún EDF, y el análisis se carga con `comun.cargar`, que se
niega a leer `test`.

Para cada grabación, la librería (`analizar_grabacion` + `puntuar`) y el
análisis (`medidas.ventanas` y `comun.tercios` + `comun.matriz` sobre
`features_all.parquet`, con los `.joblib` originales) tienen que dar:

1. la misma `pot_rel` por ventana (mismo código de procesado);
2. la misma puntuación por ventana (< 1e-6) y la misma ρ;
3. el mismo veredicto principio/final y las mismas probabilidades por tercio.
"""

import contextlib
import io
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from eeg_library.modelado.fatiga_aguda import BANDAS, CONDICIONES, puntuar
from eeg_library.preprocesado.quality import CANALES_EPOC
from eeg_library.procesado.grabacion import analizar_grabacion

RUTA_DATOS = Path(os.environ.get("EEG_DATA", r"C:\Users\migue\Desktop\TFG\EEG-Data"))
RUTA_MATRIZ = RUTA_DATOS / "processed" / "matriz_rf.parquet"
DIR_MODELOS = RUTA_DATOS / "processed" / "fatiga_aguda_test_20261027-190547" / "modelos"
CLASIFICADOR = Path(__file__).resolve().parents[2] / "analisis" / "fatiga_aguda" / "03_clasificador"

pytestmark = [
    pytest.mark.datos,
    pytest.mark.skipif(not RUTA_MATRIZ.is_file(), reason="no hay dataset en EEG_DATA"),
]

SPLITS = ("train", "val")
N_POR_CONDICION = 10
SEMILLA = 0
TOL_PUNTUACION = 1e-6


@pytest.fixture(scope="module")
def analisis():
    """Los módulos del análisis, sin modificarlos (se importan desde su carpeta)."""
    sys.path.insert(0, str(CLASIFICADOR))
    with contextlib.redirect_stdout(io.StringIO()):
        import comun
        import medidas
    return comun, medidas


def _muestra():
    matriz = pd.read_parquet(RUTA_MATRIZ, columns=["archivo", "persona_uid", "split", "condicion",
                                                   "n_canales_buenos"])
    candidatas = matriz[matriz.persona_uid.notna() & matriz.split.isin(SPLITS)
                        & matriz.condicion.isin(CONDICIONES)]
    elegidas = []
    for c in CONDICIONES:
        de_c = candidatas[candidatas.condicion == c]
        # Que entren grabaciones con canales excluidos, que es donde puede fallar la máscara.
        con_excluidos = de_c[de_c.n_canales_buenos < 14].sample(3, random_state=SEMILLA)
        resto = de_c.drop(con_excluidos.index).sample(N_POR_CONDICION - 3, random_state=SEMILLA)
        elegidas += list(zip(pd.concat([con_excluidos, resto]).archivo, [c] * N_POR_CONDICION))
    assert set(matriz.set_index("archivo").loc[[a for a, _ in elegidas], "split"]) <= set(SPLITS)
    return elegidas


def test_paridad_con_el_analisis(analisis):
    import mne
    mne.set_log_level("ERROR")
    comun, md = analisis
    elegidas = _muestra()
    archivos = [a for a, _ in elegidas]

    with contextlib.redirect_stdout(io.StringIO()):
        w = comun.cargar(set(SPLITS))
    assert set(w.split) <= set(SPLITS)
    w = w[w.archivo_origen.isin(archivos)].reset_index(drop=True)
    modelos = {c: joblib.load(DIR_MODELOS / f"completa_{c}.joblib") for c in CONDICIONES}

    filas = []
    for archivo, condicion in elegidas:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            r = analizar_grabacion(RUTA_DATOS / "dataset_final" / archivo)
        lib = puntuar(r, condicion)
        assert lib["disponible"], lib.get("motivo")

        g = w[w.archivo_origen == archivo].sort_values("indice_ventana")
        # 1. Potencia por ventana.
        pv = r["por_ventana"]
        assert pv["instante_s"] == g.tmin_s.tolist()
        mia = np.column_stack([pv["pot_rel"][c][b] for c in CANALES_EPOC for b in BANDAS])
        suya = g[comun.REL].to_numpy(float)
        dif_pot = float(np.max(np.abs(mia - suya) / np.maximum(np.abs(suya), 1e-300)))
        validez = [bool(pv["canal_valido"][c]) for c in CANALES_EPOC]
        assert validez == g[[f"canal_valido_{c}" for c in CANALES_EPOC]].iloc[0].tolist()

        # 2. Puntuación por ventana y ρ (rho_ventanas del test).
        sub = w[w.archivo_origen == archivo]
        fv, Xv = md.ventanas(sub, condicion)
        orden = np.argsort(fv.t_rec.to_numpy(), kind="stable")
        puntuacion = modelos[condicion].decision_function(Xv)[orden]
        rho = md.rho_por_grabacion(fv, modelos[condicion].decision_function(Xv)).acierto.iloc[0]
        dif_punt = float(np.max(np.abs(np.asarray(lib["puntuacion"]) - puntuacion)))

        # 3. Tercios (tercios_pareado del test).
        S, m = comun.tercios(sub, condicion)
        p = modelos[condicion].predict_proba(comun.matriz(S, m, "rel"))[:, 1]
        p_seg = dict(zip(S.seg, p))
        acierto = comun.aciertos_por_grabacion(S, p).acierto.iloc[0]
        final = lib["tercios"]["final_mayor"]

        filas.append({
            "archivo": archivo, "condicion": condicion, "n_ventanas": lib["n_ventanas"],
            "canales_validos": sum(validez), "dif_pot_rel": dif_pot, "dif_puntuacion": dif_punt,
            "rho_libreria": lib["rho"], "rho_analisis": rho,
            "dif_p_tercios": max(abs(lib["tercios"]["p_primero"] - p_seg[0]),
                                 abs(lib["tercios"]["p_ultimo"] - p_seg[1])),
            "veredicto_igual": (1.0 if final else 0.5 if final is None else 0.0) == acierto,
        })

    tabla = pd.DataFrame(filas)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n" + tabla.drop(columns="archivo").to_string(index=False, float_format="%.2e"))
    print(f"\nMáximos: pot_rel {tabla.dif_pot_rel.max():.1e} · puntuación "
          f"{tabla.dif_puntuacion.max():.1e} · ρ {np.max(np.abs(tabla.rho_libreria - tabla.rho_analisis)):.1e}"
          f" · p de tercios {tabla.dif_p_tercios.max():.1e} · veredictos iguales "
          f"{int(tabla.veredicto_igual.sum())}/{len(tabla)}")

    assert tabla.dif_pot_rel.max() <= 1e-9
    assert tabla.dif_puntuacion.max() < TOL_PUNTUACION
    np.testing.assert_allclose(tabla.rho_libreria, tabla.rho_analisis, rtol=0, atol=1e-9)
    assert tabla.dif_p_tercios.max() < TOL_PUNTUACION
    assert tabla.veredicto_igual.all()
