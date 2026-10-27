"""
Fatiga aguda — paso 3, reajuste 1: las tres medidas candidatas a igual tiempo.

- `tercios_pareado`: primer frente a último tercio, acierto pareado.
- `mitades_pareado`: primera frente a segunda mitad, acierto pareado.
- `rho_ventanas`: modelo entrenado con tercios que puntúa cada ventana; ρ de
  Spearman por grabación entre la puntuación y el instante de la ventana.

Cada medida devuelve un valor por grabación en una columna `acierto` (así lo
esperan los bootstrap de los pasos anteriores), con `archivo_origen` y
`persona_uid`.
"""

import numpy as np
import pandas as pd
from scipy import stats

import comun
from comun import d1

MEDIDAS = ["tercios_pareado", "mitades_pareado", "rho_ventanas"]
AZAR = {"tercios_pareado": 0.5, "mitades_pareado": 0.5, "rho_ventanas": 0.0}


def mitades(w, cond):
    g = w[w.condicion == cond]
    g = g.assign(seg=np.where(g.t_rec <= g.dur / 2, 0, 1))
    return comun._por_segmento(g, ["archivo_origen", "persona_uid", "seg"])


def segmentos_de_entreno(w, cond, medida):
    """Segmentos con los que se entrena: mitades para `mitades_pareado`; tercios en los demás casos."""
    return mitades(w, cond) if medida == "mitades_pareado" else comun.tercios(w, cond)


def ventanas(w, cond):
    """Una fila por ventana: log10(pot_rel) centrado por grabación, con los canales excluidos a 0."""
    g = w[w.condicion == cond].reset_index(drop=True)
    X = np.log10(np.clip(g[comun.REL].to_numpy(float), 1e-6, None))
    X = X - pd.DataFrame(X).groupby(g.archivo_origen.to_numpy()).transform("mean").to_numpy()
    mascara = np.repeat(g[d1.VALIDEZ].to_numpy(dtype=bool), len(d1.BANDAS), axis=1)
    X = np.where(mascara, X, 0.0)
    return g[["archivo_origen", "persona_uid", "t_rec"]], X


def rho_por_grabacion(filas, puntuacion):
    d = filas.assign(s=puntuacion)
    r = d.groupby(["archivo_origen", "persona_uid"]).apply(
        lambda x: stats.spearmanr(x.s, x.t_rec)[0], include_groups=False)
    return r.rename("acierto").reset_index().dropna()


# --- Bootstrap que devuelve las réplicas (para el error típico) ---------------


def _sumas(df, personas):
    s = df.groupby("persona_uid").acierto.agg(["sum", "size"]).reindex(personas, fill_value=0)
    return s["sum"].to_numpy(), s["size"].to_numpy()


def bootstrap(a, b=None, rng=None, n=d1.N_BOOTSTRAP):
    """Réplicas de la media de `a` (o de media(a) − media(b)) remuestreando personas."""
    personas = np.array(sorted(set(a.persona_uid) | (set(b.persona_uid) if b is not None else set())))
    idx = rng.integers(len(personas), size=(n, len(personas)))
    sa, na = _sumas(a, personas)
    reps = sa[idx].sum(1) / na[idx].sum(1)
    if b is not None:
        sb, nb = _sumas(b, personas)
        reps = reps - sb[idx].sum(1) / nb[idx].sum(1)
    return reps


def resumen(aciertos, medida, rng):
    """Valor, IC95 y z frente al azar por condición; H1 y su z."""
    salida = {}
    for c, x in aciertos.items():
        reps = bootstrap(x, rng=rng)
        salida[c] = {"valor": float(x.acierto.mean()), "ic95": np.percentile(reps, [2.5, 97.5]).tolist(),
                     "z_azar": float((x.acierto.mean() - AZAR[medida]) / reps.std()),
                     "grabaciones": len(x), "personas": int(x.persona_uid.nunique())}
    fam = pd.concat([aciertos[c] for c in comun.FAMILIA], ignore_index=True)
    reps = bootstrap(fam, rng=rng)
    salida["familia"] = {"valor": float(fam.acierto.mean()), "ic95": np.percentile(reps, [2.5, 97.5]).tolist()}
    reps = bootstrap(fam, aciertos["OA"], rng=rng)
    h1 = fam.acierto.mean() - aciertos["OA"].acierto.mean()
    salida["H1"] = {"valor": float(h1), "ic95": np.percentile(reps, [2.5, 97.5]).tolist(),
                    "error_tipico": float(reps.std()), "z": float(h1 / reps.std())}
    return salida
