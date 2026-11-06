"""
Tiempo en tarea — paso 7, anexo de rendimiento (descriptivo, 2026-11-06).

Describe las métricas de rendimiento por tarea de la base del estudio
(`Reaction_`, `Accuracy_`, `Speed_`, `Variation_`, `RCS_`, `Lapses_`) **por sí
solas**, sin cruzarlas con el EEG ni con el VAS: cobertura, valores extremos,
efecto techo, proporción de varianza entre personas (ICC(1)) y efecto de
práctica entre mediciones. Además estima la potencia que tendría replicar en
`test` el contraste A2 del paso 5 (cambio del EEG en el PVTB ↔ lapsos).

Sirve para justificar en `rendimiento.md` por qué no se amplía la línea con
más métricas ni con otro test. No es un análisis de asociación.

**Carga `train` y `val`; nunca `test`.** El tamaño de `test` para la potencia
se toma del recuento ya publicado en `06_test/INFORME.md` (103 grabaciones de
PVTB de 28 personas). Solo agregados en la salida.

Ejecutar (desde la raíz del repositorio):
    .venv/Scripts/python.exe analisis/tiempo_en_tarea/07_redaccion/rendimiento_descriptivo.py > analisis/tiempo_en_tarea/07_redaccion/rendimiento_descriptivo.log
"""

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent.parent / "fatiga_fg_fc"))
import datos_fg_fc as dfg  # noqa: E402

from eeg_library.procesado.labels import normalizar_vas  # noqa: E402

PATRON = re.compile(r"(Reaction|Accuracy|Speed|Variation|RCS|Lapses)_(STROOP|PASAT|PVTB|MSIT|PVSAT)$")
SPLITS_DESARROLLO = ["train", "val"]
# Tamaño de test publicado en 06_test/INFORME.md (no se lee test aquí).
N_TEST_SESIONES, N_TEST_PERSONAS = 103, 28
N_COVARIABLES = 4  # efectos fijos de medición (M2–M6) en el modelo del paso 5
R_ESCENARIOS = [0.07, 0.10, 0.14]  # 0.14 = r parcial observado en A2; los otros, prudentes
SESIONES_POR_PERSONA = N_TEST_SESIONES / N_TEST_PERSONAS
SALIDA = CARPETA / "rendimiento_descriptivo.json"


def icc1(valores, grupos):
    """ICC(1) de un factor con tamaños de grupo desiguales (ANOVA)."""
    d = pd.DataFrame({"x": valores, "g": grupos}).dropna()
    k = d.groupby("g").size()
    if (k > 1).sum() < 20 or d["x"].std() == 0:
        return None
    n, total = k.size, len(d)
    k0 = (total - (k ** 2).sum() / total) / (n - 1)
    media = d["x"].mean()
    medias = d.groupby("g")["x"].mean()
    msb = (k * (medias - media) ** 2).sum() / (n - 1)
    msw = ((d["x"] - d.groupby("g")["x"].transform("mean")) ** 2).sum() / (total - n)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def practica(valores, grupos, medicion):
    """Correlación dentro de persona entre la medición (2–6) y el valor."""
    d = pd.DataFrame({"x": valores, "g": grupos, "m": medicion}).dropna()
    xc = d["x"] - d.groupby("g")["x"].transform("mean")
    mc = d["m"] - d.groupby("g")["m"].transform("mean")
    if xc.std() == 0:
        return None
    return float(np.corrcoef(xc, mc)[0, 1])


def potencia(n_sesiones, n_personas, r, alfa=0.05):
    """Potencia unilateral de un coeficiente dentro de persona (t no central)."""
    gl = n_sesiones - n_personas - N_COVARIABLES - 1
    no_central = r * np.sqrt(gl) / np.sqrt(1 - r ** 2)
    return float(1 - stats.nct.cdf(stats.t.ppf(1 - alfa, gl), gl, no_central))


def personas_para(r, objetivo=0.8):
    for p in range(20, 1000, 5):
        if potencia(int(p * SESIONES_POR_PERSONA), p, r) >= objetivo:
            return p
    return None


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bd = dfg.leer_bd()
        labels = pd.read_csv(dfg.RUTA_LABELS)
        corregidas, _ = dfg.aplicar_correcciones(bd[bd["FITBIT"].notna()])
        base = normalizar_vas(corregidas)
    columnas = [c for c in base.columns if PATRON.match(str(c))]
    sesiones = labels[labels["etiquetada"] & labels["split"].isin(SPLITS_DESARROLLO)]
    sesiones = sesiones[["persona_uid", "medicion"]].drop_duplicates()
    assert not base.duplicated(["persona_uid", "medicion"]).any()
    d = sesiones.merge(base[["persona_uid", "medicion"] + columnas],
                       on=["persona_uid", "medicion"], how="left")
    medicion = d["medicion"].str[1].astype(int)

    filas = []
    for c in columnas:
        x = pd.to_numeric(d[c], errors="coerce")
        con_dato = x.notna()
        filas.append({
            "variable": c,
            "sesiones_con_dato": int(con_dato.sum()),
            "cobertura_pct": round(100 * con_dato.mean(), 1),
            "mediana": float(x.median()),
            "minimo": float(x.min()),
            "maximo": float(x.max()),
            "pct_en_maximo": round(100 * float((x[con_dato] == x.max()).mean()), 1),
            "valores_distintos": int(x.nunique()),
            "icc1": icc1(x, d["persona_uid"]),
            "r_practica_intra": practica(x, d["persona_uid"], medicion),
        })
    tabla = pd.DataFrame(filas)

    pot = {f"r={r}": potencia(N_TEST_SESIONES, N_TEST_PERSONAS, r) for r in R_ESCENARIOS}
    necesarias = {f"r={r}": personas_para(r) for r in (0.10, 0.14)}

    pd.set_option("display.width", 200)
    print(f"Sesiones etiquetadas de train + val: {len(d)} ({d['persona_uid'].nunique()} personas)")
    print(tabla.round(2).to_string(index=False))
    print(f"\nPotencia de replicar A2 en test ({N_TEST_SESIONES} sesiones, {N_TEST_PERSONAS} personas, "
          f"unilateral, alfa 0.05): " + ", ".join(f"{k}: {v:.2f}" for k, v in pot.items()))
    print("Personas de test necesarias para potencia 0.80: "
          + ", ".join(f"{k}: {v}" for k, v in necesarias.items()))

    salida = {
        "fecha": "2026-11-06",
        "particiones": SPLITS_DESARROLLO,
        "sesiones": int(len(d)),
        "personas": int(d["persona_uid"].nunique()),
        "sha256_base": dfg.sha256(dfg.RUTA_BD),
        "sha256_labels": dfg.sha256(dfg.RUTA_LABELS),
        "variables": filas,
        "potencia_replica_A2": {"n_sesiones_test": N_TEST_SESIONES, "n_personas_test": N_TEST_PERSONAS,
                                "fuente_n": "06_test/INFORME.md", "potencia": pot,
                                "personas_para_0.80": necesarias},
    }
    SALIDA.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
