"""
Fatiga aguda (tiempo en tarea) — paso 1, anexo A: comparación a igual tiempo
transcurrido.

Cada grabación de PASAT, PVTB y PVSAT se recorta a la duración útil de la
grabación de OA de la misma sesión, y se repite el clasificador de tercios del
paso 1. Pregunta: ¿siguen superando PASAT y PVTB a OA cuando el tiempo
transcurrido es el mismo? Añade las pendientes por minuto a igual tiempo,
pareadas por persona frente a OA (de interés para la H3).

**Solo `train`.** Reutiliza la carga, los segmentos, el clasificador y el
bootstrap de `descriptivo_fatiga_aguda.py`, que no se modifica. Plan y lectura
fijados antes de ejecutar: `INFORME.md`, anexo A.

Ejecutar (desde la raíz del repositorio):
    .venv/Scripts/python.exe analisis/fatiga_aguda/01_descriptivo/anexo_igual_tiempo.py > analisis/fatiga_aguda/01_descriptivo/ejecucion_anexo.log
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA))
import descriptivo_fatiga_aguda as d1  # noqa: E402

TAREAS = ["PASAT", "PVTB", "PVSAT"]
# Magnitudes de la tabla 3.2 que entran en la H3 o la rozan.
MAGNITUDES_H3 = [m for m in d1.MAGNITUDES_SONDEO
                 if m[0] in ("alfa_post_abs", "theta_front_abs", "beta_post_rel")]


def recortar(w):
    """Recorta cada grabación de tarea al `tmin_s` máximo del OA de su sesión."""
    oa = (w[w.condicion == "OA"].groupby("sesion_uid").tmin_s.max().rename("tmax_oa"))
    tareas = w[w.condicion.isin(TAREAS)].merge(oa, left_on="sesion_uid", right_index=True, how="left")
    sin_oa = tareas[tareas.tmax_oa.isna()].groupby("condicion").archivo_origen.nunique()
    recortadas = tareas[tareas.tmin_s <= tareas.tmax_oa].drop(columns="tmax_oa")
    # Duración y número de ventanas, recalculados sobre la grabación recortada.
    recortadas["n_vent"] = recortadas.groupby("archivo_origen").tmin_s.transform("size")
    recortadas["dur"] = recortadas.groupby("archivo_origen").t_rec.transform("max")
    return pd.concat([recortadas, w[w.condicion == "OA"]], ignore_index=True), sin_oa


def sumas_por_persona(aciertos, personas):
    s = aciertos.groupby("persona_uid").acierto.agg(["sum", "size"]).reindex(personas, fill_value=0)
    return s["sum"].to_numpy(), s["size"].to_numpy()


def diferencia_bootstrap(a, b, rng):
    """Diferencia de acierto pareado a − b (por grabación) e IC95 remuestreando personas."""
    personas = np.array(sorted(set(a.persona_uid) | set(b.persona_uid)))
    sa, na = sumas_por_persona(a, personas)
    sb, nb = sumas_por_persona(b, personas)
    idx = rng.integers(len(personas), size=(d1.N_BOOTSTRAP, len(personas)))
    reps = sa[idx].sum(1) / na[idx].sum(1) - sb[idx].sum(1) / nb[idx].sum(1)
    return a.acierto.mean() - b.acierto.mean(), np.percentile(reps, [2.5, 97.5])


def lectura(dif, ic):
    if dif <= 0:
        return "no supera a OA"
    return "sigue superando a OA" if ic[0] > 0 else "ventaja no concluyente"


def clasificador(w_completo, w_recortado):
    d1._titulo("2-4. CLASIFICADOR DE TERCIOS A IGUAL TIEMPO")
    rng = np.random.default_rng(d1.SEMILLA)
    filas, aciertos = [], {}
    for validos in (True, False):
        variante = f"centrado+{'validos' if validos else '14'}"
        for version, w in (("completa", w_completo), ("recortada", w_recortado)):
            for cond in TAREAS + ["OA"]:
                if version == "recortada" and cond == "OA":
                    continue  # OA no se recorta: es la de la versión completa
                X, mascara = d1.segmentos(w, cond)
                m, par = d1.clasificar(X, mascara, centrar=True, validos=validos)
                porp = par.groupby("persona_uid").acierto.agg(["sum", "size"])
                ic = d1.bootstrap_personas(porp["sum"].to_numpy(), porp["size"].to_numpy(), rng)
                aciertos[(variante, version, cond)] = par
                filas.append({"variante": variante, "version": version, "condicion": cond, **m,
                              "pareado_ic95": ic.tolist()})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(d1.CARPETA_TABLAS / "anexo_clasificador_igual_tiempo.csv", index=False)
    for variante in tabla.variante.unique():
        print(f"\nVariante {variante}:")
        sub = tabla[tabla.variante == variante].copy()
        sub["ic95"] = sub.pareado_ic95.apply(lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]")
        print(sub[["version", "condicion", "grabaciones", "personas", "auc", "pareado", "ic95", "empates"]]
              .round(3).to_string(index=False))

    filas_d = []
    for variante in tabla.variante.unique():
        oa = aciertos[(variante, "completa", "OA")]
        print(f"\nDiferencias ({variante}), IC95 bootstrap de personas:")
        for cond in TAREAS:
            rec = aciertos[(variante, "recortada", cond)]
            com = aciertos[(variante, "completa", cond)]
            dif, ic = diferencia_bootstrap(rec, oa, rng)
            # Pérdida al recortar: mismas grabaciones en las dos versiones.
            comunes = rec.merge(com, on=["archivo_origen", "persona_uid"], suffixes=("_rec", "_com"))
            perdida, ic_p = diferencia_bootstrap(
                comunes.rename(columns={"acierto_com": "acierto"})[["persona_uid", "acierto"]],
                comunes.rename(columns={"acierto_rec": "acierto"})[["persona_uid", "acierto"]], rng)
            texto = lectura(dif, ic)
            print(f"  {cond:6s} recortada − OA: {dif:+.3f} [{ic[0]:+.3f}, {ic[1]:+.3f}]  → {texto};"
                  f"  completa − recortada: {perdida:+.3f} [{ic_p[0]:+.3f}, {ic_p[1]:+.3f}]")
            filas_d.append({"variante": variante, "condicion": cond, "recortada_menos_oa": dif,
                            "ic95": ic.tolist(), "lectura": texto,
                            "completa_menos_recortada": perdida, "ic95_perdida": ic_p.tolist()})
    pd.DataFrame(filas_d).to_csv(d1.CARPETA_TABLAS / "anexo_diferencias_igual_tiempo.csv", index=False)
    return tabla, filas_d


def pendientes_pareadas(w):
    d1._titulo("5. PENDIENTES POR MINUTO A IGUAL TIEMPO, PAREADAS POR PERSONA FRENTE A OA")
    w = w[w.n_vent >= d1.MIN_VENTANAS]
    filas = []
    for nombre, pref, reg, banda in MAGNITUDES_H3:
        y = d1.magnitud_regional(w, pref, d1.REGIONES_SONDEO[reg], banda, validos=True)
        d = pd.DataFrame({"archivo": w.archivo_origen.to_numpy(), "persona": w.persona_uid.to_numpy(),
                          "cond": w.condicion.to_numpy(), "t": w.t_rec.to_numpy(), "y": y})
        d = d[np.isfinite(d.y)]
        d["tc"] = d.t - d.groupby("archivo").t.transform("mean")
        d["yc"] = d.y - d.groupby("archivo").y.transform("mean")
        d["txy"], d["txx"] = d.tc * d.yc, d.tc ** 2
        g = d.groupby(["archivo", "persona", "cond"])[["txy", "txx"]].sum().reset_index()
        g["pend"] = g.txy / g.txx
        pp = g.groupby(["persona", "cond"]).pend.mean().unstack()
        for cond in TAREAS + ["OA"]:
            v = pp[cond].dropna()
            t, p = stats.ttest_1samp(v, 0)
            fila = {"magnitud": nombre, "condicion": cond, "pendiente": v.mean(), "t": t, "p": p,
                    "personas": len(v)}
            if cond != "OA":
                dif = (pp[cond] - pp["OA"]).dropna()
                td, pd_ = stats.ttest_1samp(dif, 0)
                fila.update({"dif_con_oa": dif.mean(), "t_dif": td, "p_dif": pd_, "personas_dif": len(dif)})
            filas.append(fila)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(d1.CARPETA_TABLAS / "anexo_pendientes_igual_tiempo.csv", index=False)
    print(tabla.round(4).to_string(index=False))
    return tabla


def main():
    d1._titulo("0. CARGA (solo train)")
    w, _, _ = d1.cargar()

    d1._titulo("1. RECORTE A LA DURACIÓN DEL OA DE LA MISMA SESIÓN")
    w_rec, sin_oa = recortar(w)
    print("Grabaciones de tarea sin OA en su sesión (fuera del recorte):",
          sin_oa.to_dict() if len(sin_oa) else "ninguna")
    por_grab = w_rec.groupby("archivo_origen").agg(cond=("condicion", "first"), n=("n_vent", "first"),
                                                   dur_s=("tmin_s", "max"))
    resumen = por_grab.groupby("cond").agg(grabaciones=("n", "size"), dur_mediana_s=("dur_s", "median"),
                                           dur_p10=("dur_s", lambda s: s.quantile(.1)),
                                           dur_p90=("dur_s", lambda s: s.quantile(.9)),
                                           ventanas_mediana=("n", "median"),
                                           menos_10=("n", lambda s: int((s < d1.MIN_VENTANAS).sum())))
    print(resumen.to_string())

    tabla, diferencias = clasificador(w, w_rec)
    pend = pendientes_pareadas(w_rec)

    resultados = {
        "fecha": "2026-10-27",
        "particion": "solo train (split de matriz_rf), etiquetadas",
        "sha256": {p.name: d1.sha256(p) for p in (d1.RUTA_FEATURES, d1.RUTA_MATRIZ, d1.RUTA_LOG)},
        "recorte": {"sin_oa_en_sesion": {k: int(v) for k, v in sin_oa.items()},
                    "resumen": resumen.reset_index().to_dict(orient="records")},
        "clasificador": tabla.to_dict(orient="records"),
        "diferencias": diferencias,
        "pendientes": pend.round(5).to_dict(orient="records"),
        "miradas": {"val": 0, "test": 0},
    }
    with open(CARPETA / "resultados_anexo.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
