"""
Tiempo en tarea — paso 5 (exploratorio): el cambio dentro de la
grabación frente al rendimiento total en la tarea y frente a covariables.

Contrastes cerrados del `INFORME.md` (escrito antes de ejecutar):

- A. Dentro de persona: `rho_ventanas` de la tarea frente a su conducta, con
  efectos fijos de medición y errores agrupados por persona; Holm sobre A1–A3.
- B. Entre personas: medias por persona, ρ de Spearman.
- C. Control: lo mismo con la `rho_ventanas` de OA de la misma sesión.
- D. Lo mismo con la pendiente de la beta posterior relativa.
- E. Covariables descriptivas: sexo, centro, medición, hora y casco.

**Carga `train` y `val`; nunca `test`.** Solo agregados en la salida.

Ejecutar (desde la raíz del repositorio):
    .venv/Scripts/python.exe analisis/tiempo_en_tarea/05_entre_sesiones/entre_sesiones.py > analisis/tiempo_en_tarea/05_entre_sesiones/ejecucion.log
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent / "03_clasificador"))
sys.path.insert(0, str(CARPETA.parent / "04_potencia_y_controles"))
sys.path.insert(0, str(CARPETA.parent.parent / "fatiga_fg_fc"))
import comun  # noqa: E402
import controles  # noqa: E402
import datos_fg_fc as dfg  # noqa: E402
import medidas as md  # noqa: E402
import potencia  # noqa: E402
from comun import N_PLIEGUES, SEMILLAS, d1  # noqa: E402

from eeg_library.procesado.labels import normalizar_vas  # noqa: E402

CONDICIONES = ["PASAT", "PVTB", "OA"]
PARES = [("A1", "PASAT", "Accuracy_PASAT"), ("A2", "PVTB", "Lapses_PVTB"), ("A3", "PVTB", "Reaction_PVTB")]
RANGOS = {"Accuracy_PASAT": (1.0, 100.0, "(1, 100]"), "Lapses_PVTB": (0.0, np.inf, ">= 0"),
          "Reaction_PVTB": (100.0, 1000.0, "[100, 1000]")}
CASCOS = (172358, 236373)
RUTA_MANIFEST = d1.EEG_DATA / "metadata" / "manifest_canonical.csv"
CARPETA_TABLAS = CARPETA / "tablas"
N_BOOT = d1.N_BOOTSTRAP


# --- Cambio por grabación -------------------------------------------------------


def rho_out_of_fold(w):
    """`rho_ventanas` por grabación: media de las 20 particiones, puntuación out-of-fold."""
    personas = sorted(w.persona_uid.unique())
    salida = []
    for c in CONDICIONES:
        S, masc = comun.tercios(w, c)
        X, y, p_seg = comun.matriz(S, masc, "rel"), S.seg.to_numpy(), S.persona_uid.to_numpy()
        filas, Xv = md.ventanas(w, c)
        p_ven = filas.persona_uid.to_numpy()
        por_semilla = []
        for s in SEMILLAS:
            pliegue = comun.pliegues_por_persona(personas, s)
            f_seg, f_ven = pd.Series(p_seg).map(pliegue).to_numpy(), pd.Series(p_ven).map(pliegue).to_numpy()
            r = []
            for k in range(N_PLIEGUES):
                m = comun.modelo("lr_0.01").fit(X[f_seg != k], y[f_seg != k])
                sel = f_ven == k
                r.append(potencia.rho_vectorizada(filas[sel], m.decision_function(Xv[sel])))
            por_semilla.append(pd.concat(r))
        rho = pd.concat(por_semilla, axis=1).mean(axis=1)
        salida.append(pd.DataFrame({"archivo_origen": rho.index, "condicion": c, "rho": rho.to_numpy()}))
        print(f"{c}: {len(rho)} grabaciones, ρ media {rho.mean():+.3f}")
    return pd.concat(salida, ignore_index=True)


def pendiente_beta(w):
    """Pendiente por minuto (MCO) de la beta posterior relativa en cada grabación."""
    nombre, pref, reg, banda = next(m for m in d1.MAGNITUDES_SONDEO if m[0] == "beta_post_rel")
    y = d1.magnitud_regional(w, pref, d1.REGIONES_SONDEO[reg], banda, validos=True)
    d = pd.DataFrame({"a": w.archivo_origen.to_numpy(), "c": w.condicion.to_numpy(),
                      "t": w.t_rec.to_numpy(), "y": y})
    d = d[np.isfinite(d.y) & d.c.isin(CONDICIONES)]
    d["tc"] = d.t - d.groupby("a").t.transform("mean")
    d["yc"] = d.y - d.groupby("a").y.transform("mean")
    d["xy"], d["xx"] = d.tc * d.yc, d.tc ** 2
    g = d.groupby(["a", "c"])[["xy", "xx"]].sum().reset_index()
    return pd.DataFrame({"archivo_origen": g.a, "condicion": g.c, "beta": g.xy / g.xx})


# --- Conducta y covariables por sesión -----------------------------------------


def conducta():
    """Conducta por (persona_uid, medicion), enlazada como en `datos_fg_fc`, con los fuera de rango a NaN."""
    bd = dfg.leer_bd()
    labels = pd.read_csv(dfg.RUTA_LABELS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Valida el enlace (duplicados, códigos, splits) con la función de la línea FG/FC.
        respuestas, _ = dfg.respuestas_participantes(bd, labels)
        corregidas, _ = dfg.aplicar_correcciones(bd[bd["FITBIT"].notna()])
        v = normalizar_vas(corregidas)
    columnas = [x for _, _, x in PARES]
    v = v[["persona_uid", "medicion"] + columnas].copy()
    assert not v.duplicated(["persona_uid", "medicion"]).any()
    assert set(map(tuple, v[["persona_uid", "medicion"]].to_numpy())) == \
        set(map(tuple, respuestas[["persona_uid", "medicion"]].to_numpy()))
    excluidos = {}
    for col, (lo, hi, texto) in RANGOS.items():
        x = pd.to_numeric(v[col], errors="coerce")
        fuera = x.notna() & ~((x > lo if col == "Accuracy_PASAT" else x >= lo) & (x <= hi))
        excluidos[col] = {"rango_valido": texto, "excluidos": int(fuera.sum())}
        v[col] = x.mask(fuera)
    return v, excluidos


def covariables(w):
    """Hora local de inicio de la sesión, casco (moda), sexo y centro por sesión."""
    man = pd.read_csv(RUTA_MANIFEST, usecols=["nombre_nuevo", "timestamp_iso", "serie_headset"])
    man = man[man.nombre_nuevo.isin(set(w.archivo_origen))]
    hora_local = man.timestamp_iso.str[11:13].astype(float) + man.timestamp_iso.str[14:16].astype(float) / 60
    man = man.assign(hora=hora_local).merge(
        w.drop_duplicates("archivo_origen")[["archivo_origen", "sesion_uid"]],
        left_on="nombre_nuevo", right_on="archivo_origen")
    ses = man.groupby("sesion_uid").agg(hora=("hora", "min"),
                                        casco=("serie_headset", lambda s: s.mode().iloc[0]))
    bd = dfg.leer_bd()
    labels = pd.read_csv(dfg.RUTA_LABELS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        respuestas, _ = dfg.respuestas_participantes(bd, labels)
        personas = dfg.caracteristicas_persona(respuestas, labels)
    return ses.reset_index(), personas


# --- Estadística -------------------------------------------------------------------


def z_rango(x):
    r = stats.rankdata(x)
    return (r - r.mean()) / r.std()


def dentro_de_persona(d, x, y):
    """Coeficiente de z(rango de y) sobre z(rango de x) con EF de persona y de medición; EE agrupados."""
    d = d[["persona_uid", "medicion", x, y]].dropna()
    d = d[d.groupby("persona_uid").persona_uid.transform("size") >= 2].reset_index(drop=True)
    zx, zy = z_rango(d[x].to_numpy()), z_rango(d[y].to_numpy())
    dummies = pd.get_dummies(d.medicion, drop_first=True).astype(float)
    R = pd.concat([pd.Series(zx, name="x"), dummies], axis=1)
    Rw = R - R.groupby(d.persona_uid).transform("mean")
    yw = pd.Series(zy) - pd.Series(zy).groupby(d.persona_uid).transform("mean")
    beta, se = controles.mco_agrupado(Rw.to_numpy(float), yw.to_numpy(float), d.persona_uid.to_numpy())
    G = d.persona_uid.nunique()
    t = beta[0] / se[0]
    # Correlación parcial dentro de persona (residuos de x e y sobre las medidas de medición).
    D = Rw.drop(columns="x").to_numpy(float)
    rx = Rw.x.to_numpy() - D @ np.linalg.lstsq(D, Rw.x.to_numpy(), rcond=None)[0]
    ry = yw.to_numpy() - D @ np.linalg.lstsq(D, yw.to_numpy(), rcond=None)[0]
    return {"coef": float(beta[0]), "ee": float(se[0]), "t": float(t),
            "p": float(2 * stats.t.sf(abs(t), G - 1)), "r_parcial": float(np.corrcoef(rx, ry)[0, 1]),
            "sesiones": int(len(d)), "personas": int(G)}


def entre_personas(d, x, y, rng):
    """ρ de Spearman entre las medias por persona, con IC95 por bootstrap de personas."""
    m = d[["persona_uid", x, y]].dropna().groupby("persona_uid")[[x, y]].mean()
    rho = stats.spearmanr(m[x], m[y])[0]
    idx = rng.integers(len(m), size=(N_BOOT, len(m)))
    a, b = m[x].to_numpy(), m[y].to_numpy()
    reps = [stats.spearmanr(a[i], b[i])[0] for i in idx]
    return {"rho": float(rho), "ic95": np.nanpercentile(reps, [2.5, 97.5]).tolist(), "personas": int(len(m))}


def holm(p):
    orden = np.argsort(p)
    ajustada = np.empty(len(p))
    acumulado = 0.0
    for i, j in enumerate(orden):
        acumulado = max(acumulado, min(1.0, (len(p) - i) * p[j]))
        ajustada[j] = acumulado
    return ajustada


def media_por_grupo(d, grupo, valor, rng):
    """Media de `valor` por grupo, con IC95 remuestreando personas dentro del grupo."""
    filas = []
    for g, x in d[["persona_uid", grupo, valor]].dropna().groupby(grupo):
        s = x.groupby("persona_uid")[valor].agg(["sum", "size"])
        ic = d1.bootstrap_personas(s["sum"].to_numpy(), s["size"].to_numpy(), rng)
        filas.append({grupo: g, "media": float(x[valor].mean()), "ic95": ic.tolist(),
                      "sesiones": int(len(x)), "personas": int(len(s))})
    return filas


# --- Principal -----------------------------------------------------------------------


def main():
    rng = np.random.default_rng(d1.SEMILLA)
    CARPETA_TABLAS.mkdir(exist_ok=True)
    w = comun.cargar(["train", "val"])
    assert set(w.split) <= {"train", "val"}
    print(f"Ventanas: {len(w)}; grabaciones: {w.archivo_origen.nunique()}; personas: {w.persona_uid.nunique()}")

    d1._titulo("CAMBIO POR GRABACIÓN (grabación completa)")
    rho = rho_out_of_fold(w)
    beta = pendiente_beta(w)
    cambio = rho.merge(beta, on=["archivo_origen", "condicion"], how="left")
    cambio = cambio.merge(w.drop_duplicates("archivo_origen")[["archivo_origen", "persona_uid", "sesion_uid"]])
    ses = cambio.pivot_table(index=["persona_uid", "sesion_uid"], columns="condicion",
                             values=["rho", "beta"], aggfunc="mean")
    ses.columns = [f"{v}_{c}" for v, c in ses.columns]
    ses = ses.reset_index()
    ses["medicion"] = ses.sesion_uid.str.split("_").str[-1]

    d1._titulo("CONDUCTA")
    cond, excluidos = conducta()
    ses = ses.merge(cond, on=["persona_uid", "medicion"], how="left")
    print("Valores fuera de rango excluidos:", excluidos)
    for _, _, col in PARES:
        print(f"{col}: {ses[col].notna().sum()} sesiones con dato; mediana {ses[col].median():.1f}")

    d1._titulo("A/C/D. DENTRO DE PERSONA (EF de persona y medición; EE agrupados por persona)")
    filas = []
    for clave, tarea, col in PARES:
        for bloque, x in (("A", f"rho_{tarea}"), ("C_control_OA", "rho_OA"), ("D_beta_post_rel", f"beta_{tarea}")):
            r = dentro_de_persona(ses, x, col)
            filas.append({"contraste": clave, "bloque": bloque, "cambio": x, "conducta": col, **r})
    dentro = pd.DataFrame(filas)
    principal = dentro.bloque == "A"
    dentro.loc[principal, "p_holm"] = holm(dentro.loc[principal, "p"].to_numpy())
    print(dentro.round(4).to_string(index=False))
    dentro.to_csv(CARPETA_TABLAS / "dentro_de_persona.csv", index=False)

    d1._titulo("B. ENTRE PERSONAS (medias por persona, ρ de Spearman)")
    filas = []
    for clave, tarea, col in PARES:
        for bloque, x in (("B", f"rho_{tarea}"), ("C_control_OA", "rho_OA"), ("D_beta_post_rel", f"beta_{tarea}")):
            r = entre_personas(ses, x, col, rng)
            filas.append({"contraste": clave, "bloque": bloque, "cambio": x, "conducta": col, **r})
    entre = pd.DataFrame(filas)
    print(entre.round(4).to_string(index=False))
    entre.to_csv(CARPETA_TABLAS / "entre_personas.csv", index=False)

    d1._titulo("E. COVARIABLES (descriptivo)")
    cov_ses, personas = covariables(w)
    ses = ses.merge(cov_ses, on="sesion_uid", how="left")
    ses["sexo"] = ses.persona_uid.map(personas["sexo"])
    ses["centro"] = ses.persona_uid.str.split("_").str[0]
    covs = {}
    for tarea in ("PASAT", "PVTB"):
        v = f"rho_{tarea}"
        c = {g: media_por_grupo(ses, g, v, rng) for g in ("sexo", "centro", "medicion")}
        x = ses[["persona_uid", "hora", v]].dropna()
        pers = np.array(sorted(x.persona_uid.unique()))
        grupos = {p: x.index[x.persona_uid == p].to_numpy() for p in pers}
        reps = []
        for i in rng.integers(len(pers), size=(N_BOOT, len(pers))):
            filas_b = np.concatenate([grupos[p] for p in pers[i]])
            reps.append(stats.spearmanr(x.loc[filas_b, "hora"], x.loc[filas_b, v])[0])
        c["hora"] = {"rho": float(stats.spearmanr(x.hora, x[v])[0]),
                     "ic95": np.percentile(reps, [2.5, 97.5]).tolist(), "sesiones": int(len(x)),
                     "hora_mediana": float(x.hora.median()), "hora_p10_p90": np.percentile(x.hora, [10, 90]).tolist()}
        k = ses[ses.medicion.isin(["M4", "M5", "M6"]) & ses.casco.isin(CASCOS)][["persona_uid", "casco", v]].dropna()
        m = k.groupby(["persona_uid", "casco"])[v].mean().unstack().dropna()
        dif = (m[CASCOS[1]] - m[CASCOS[0]]).to_numpy()
        reps = dif[rng.integers(len(dif), size=(N_BOOT, len(dif)))].mean(1)
        c["casco_M4_M6"] = {"diferencia": f"{CASCOS[1]} − {CASCOS[0]}", "media": float(dif.mean()),
                            "ic95": np.percentile(reps, [2.5, 97.5]).tolist(), "personas_con_los_dos": int(len(dif))}
        covs[tarea] = c
        print(f"\n{tarea}:")
        for g in ("sexo", "centro", "medicion"):
            print(f"  {g}: " + "; ".join(f"{f[g]} {f['media']:+.3f} [{f['ic95'][0]:+.3f}, {f['ic95'][1]:+.3f}] "
                                         f"(n={f['personas']})" for f in c[g]))
        h = c["hora"]
        print(f"  hora: ρ {h['rho']:+.3f} [{h['ic95'][0]:+.3f}, {h['ic95'][1]:+.3f}], "
              f"mediana {h['hora_mediana']:.1f} h, p10–p90 {h['hora_p10_p90']}")
        k = c["casco_M4_M6"]
        print(f"  casco ({k['diferencia']}): {k['media']:+.3f} [{k['ic95'][0]:+.3f}, {k['ic95'][1]:+.3f}], "
              f"{k['personas_con_los_dos']} personas")

    a = dentro[principal]
    reportable = bool(((a.p_holm <= 0.05).to_numpy()
                       & (np.sign(dentro[dentro.bloque == "D_beta_post_rel"].coef.to_numpy())
                          == -np.sign(a.coef.to_numpy()))).any())
    d1._titulo("LECTURA")
    print("Algún A sobrevive a Holm:", bool((a.p_holm <= 0.05).any()))
    print("(Signo esperado de D: opuesto al de A, porque más ρ = más cambio hacia el final y la beta baja.)")

    resultados = {
        "fecha": pd.Timestamp.now().isoformat(timespec="seconds"),
        "particiones": ["train", "val"], "personas": int(w.persona_uid.nunique()),
        "sesiones": int(ses.sesion_uid.nunique()),
        "conducta_excluida_por_rango": excluidos,
        "dentro_de_persona": dentro.to_dict(orient="records"),
        "entre_personas": entre.to_dict(orient="records"),
        "covariables": covs,
        "algun_A_sobrevive_holm": bool((a.p_holm <= 0.05).any()),
        "A_holm_y_D_en_sentido_coherente": reportable,
        "sha256": {p.name: d1.sha256(p) for p in (d1.RUTA_FEATURES, d1.RUTA_MATRIZ, d1.RUTA_LOG, dfg.RUTA_BD,
                                                 dfg.RUTA_LABELS, RUTA_MANIFEST)},
    }
    with open(CARPETA / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
