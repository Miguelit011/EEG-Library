"""
Tiempo en tarea — paso 5, desviación 1 (exploratorio): el
contraste A2 (`rho_ventanas` del PVTB ↔ lapsos del PVTB, dentro de persona)
con el movimiento de cada grabación como covariable.

Plan y regla de lectura en `INFORME.md` («Desviación 1»), escritos antes de
ejecutar. **Carga `train` y `val`; nunca `test`.** Solo agregados en la salida.

Ejecutar (desde la raíz del repositorio):
    .venv/Scripts/python.exe analisis/tiempo_en_tarea/05_entre_sesiones/movimiento.py > analisis/tiempo_en_tarea/05_entre_sesiones/ejecucion_movimiento.log
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent / "02_artefactos"))
import artefactos as art  # noqa: E402
import entre_sesiones as es  # noqa: E402
from entre_sesiones import comun, controles, d1, dfg, potencia  # noqa: E402

TAREA, CONDUCTA = "PVTB", "Lapses_PVTB"
MOVIMIENTO = ["mov_acc", "mov_ang"]
COVARIABLES = ["nivel_mov_acc", "nivel_mov_ang", "tendencia_mov_acc", "tendencia_mov_ang"]


def movimiento_por_grabacion(w):
    """Nivel (media) y tendencia (ρ con el tiempo) de `mov_acc` y `mov_ang` en cada grabación de la tarea."""
    g = w[w.condicion == TAREA].reset_index(drop=True)
    por_grab = g.groupby("archivo_origen").agg(n=("indice_ventana", "size"), ultimo=("indice_ventana", "max"))
    assert (por_grab.n == por_grab.ultimo + 1).all(), "Índices de ventana no consecutivos"
    validez = g.drop_duplicates("archivo_origen").set_index("archivo_origen")[d1.VALIDEZ]
    t0 = time.time()
    partes = []
    for archivo, n in por_grab.n.items():
        validos = [c for c, v in zip(d1.CANALES_EPOC, validez.loc[archivo]) if v]
        partes.append(art.medidas_grabacion(archivo, n, validos)[["archivo_origen", "indice_ventana"] + MOVIMIENTO])
    mov = pd.concat(partes, ignore_index=True)
    print(f"EDF leídos: {len(partes)} en {time.time() - t0:.0f} s")
    g = g[["archivo_origen", "indice_ventana", "t_rec"]].merge(mov, on=["archivo_origen", "indice_ventana"],
                                                               validate="one_to_one")
    sin_mov = g.groupby("archivo_origen").mov_acc.apply(lambda s: s.isna().all())
    g = g[~g.archivo_origen.isin(sin_mov[sin_mov].index)].reset_index(drop=True)
    salida = g.groupby("archivo_origen")[MOVIMIENTO].mean().add_prefix("nivel_")
    for m in MOVIMIENTO:
        salida[f"tendencia_{m}"] = potencia.rho_vectorizada(g, g[m].to_numpy())
    return salida.reset_index(), int(sin_mov.sum())


def dentro_con_covariables(d, x, y, covs):
    """Como `es.dentro_de_persona`, con covariables en z de rangos (mismas sesiones con y sin ellas)."""
    d = d[["persona_uid", "medicion", x, y] + covs].dropna()
    d = d[d.groupby("persona_uid").persona_uid.transform("size") >= 2].reset_index(drop=True)
    zy = pd.Series(es.z_rango(d[y].to_numpy()))
    regs = pd.DataFrame({c: es.z_rango(d[c].to_numpy()) for c in [x] + covs})
    dummies = pd.get_dummies(d.medicion, drop_first=True).astype(float)
    R = pd.concat([regs, dummies], axis=1)
    Rw = R - R.groupby(d.persona_uid).transform("mean")
    yw = zy - zy.groupby(d.persona_uid).transform("mean")
    beta, se = controles.mco_agrupado(Rw.to_numpy(float), yw.to_numpy(float), d.persona_uid.to_numpy())
    G = d.persona_uid.nunique()
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), G - 1)
    return pd.DataFrame({"coef": beta, "ee": se, "t": t, "p": p}, index=R.columns).loc[[x] + covs], len(d), G


def main():
    w = comun.cargar(["train", "val"])
    assert set(w.split) <= {"train", "val"}
    print(f"Personas: {w.persona_uid.nunique()}")

    d1._titulo(f"rho_ventanas del {TAREA} (igual que en entre_sesiones.py)")
    rho = es.rho_out_of_fold(w)
    rho = rho[rho.condicion == TAREA].drop(columns="condicion")

    d1._titulo("MOVIMIENTO POR GRABACIÓN")
    mov, n_sin_mov = movimiento_por_grabacion(w)
    print(f"Grabaciones {TAREA} con movimiento: {len(mov)}; sin movimiento (excluidas): {n_sin_mov}")

    g = rho.merge(mov, on="archivo_origen", how="left").merge(
        w.drop_duplicates("archivo_origen")[["archivo_origen", "persona_uid", "sesion_uid"]])
    g["medicion"] = g.sesion_uid.str.split("_").str[-1]
    cond, _ = es.conducta()
    g = g.merge(cond[["persona_uid", "medicion", CONDUCTA]], on=["persona_uid", "medicion"], how="left")
    assert not g.duplicated("sesion_uid").any(), "Más de una grabación de la tarea por sesión"

    d1._titulo("A2 SIN Y CON COVARIABLES DE MOVIMIENTO (mismas sesiones)")
    original = es.dentro_de_persona(g, "rho", CONDUCTA)
    print(f"A2 en el paso 5 (todas las sesiones con dato): coef {original['coef']:+.4f}, p {original['p']:.4f}, "
          f"{original['sesiones']} sesiones")
    muestra = g.dropna(subset=COVARIABLES)
    sin, n, G = dentro_con_covariables(muestra, "rho", CONDUCTA, [])
    con, n2, G2 = dentro_con_covariables(muestra, "rho", CONDUCTA, COVARIABLES)
    assert (n, G) == (n2, G2)
    print(f"\nMisma muestra con movimiento: {n} sesiones, {G} personas")
    print("Sin covariables:\n" + sin.round(4).to_string())
    print("Con las 4 covariables:\n" + con.round(4).to_string())

    d1._titulo("CAMINOS DE LA EXPLICACIÓN ALTERNATIVA (dentro de persona, cada covariable por separado)")
    caminos = []
    for c in COVARIABLES:
        a = es.dentro_de_persona(muestra, c, CONDUCTA)
        b = es.dentro_de_persona(muestra, c, "rho")
        caminos.append({"covariable": c, "con_lapsos_coef": a["coef"], "con_lapsos_p": a["p"],
                        "con_rho_coef": b["coef"], "con_rho_p": b["p"]})
    caminos = pd.DataFrame(caminos)
    print(caminos.round(4).to_string(index=False))

    b_sin, b_con, p_con = sin.loc["rho", "coef"], con.loc["rho", "coef"], con.loc["rho", "p"]
    aguanta = bool(p_con < 0.05 and b_con >= 0.5 * b_sin)
    d1._titulo("LECTURA (regla fijada en el INFORME)")
    print(f"coef sin {b_sin:+.4f}; con {b_con:+.4f} ({b_con / b_sin:.0%} del original); p con {p_con:.4f}")
    print("AGUANTA" if aguanta else "NO SE DISTINGUE DEL MOVIMIENTO")

    tabla = pd.concat([sin.assign(modelo="sin_covariables"), con.assign(modelo="con_movimiento")])
    tabla.rename_axis("termino").reset_index().to_csv(CARPETA / "tablas" / "movimiento.csv", index=False)
    resultados = {
        "fecha": pd.Timestamp.now().isoformat(timespec="seconds"),
        "particiones": ["train", "val"], "tarea": TAREA, "conducta": CONDUCTA,
        "grabaciones_sin_movimiento_excluidas": n_sin_mov,
        "A2_paso5": original,
        "muestra": {"sesiones": int(n), "personas": int(G)},
        "sin_covariables": sin.to_dict(orient="index"),
        "con_movimiento": con.to_dict(orient="index"),
        "caminos": caminos.to_dict(orient="records"),
        "proporcion_conservada": float(b_con / b_sin),
        "aguanta": aguanta,
        "sha256": {p.name: d1.sha256(p) for p in (d1.RUTA_FEATURES, d1.RUTA_MATRIZ, d1.RUTA_LOG, dfg.RUTA_BD,
                                                 dfg.RUTA_LABELS)},
    }
    with open(CARPETA / "resultados_movimiento.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
