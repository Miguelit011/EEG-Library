"""
Normalización intra-sujeto del VAS — paso 3: ¿cambia lo que el EEG consigue
en la Tarea B (persona conocida) cuando la etiqueta se normaliza mejor?

Repite la prueba principal de la Fase 14b (tres niveles, RF 500 hoja>=5,
GroupKFold(5) por persona, X centrada por la media LOO de la persona) sobre
las mismas 537 sesiones, cambiando **solo la etiqueta** por las variantes del
paso 2: (a) la de la Fase 14b, (a′) centrado con la referencia R4, (b) z con
SD encogida (principal) y (c) (a′) sin efecto de momento. Cada una con y sin
control de visita.

Todo parámetro poblacional de la etiqueta (desplazamiento de M1, previa del
encogimiento, efecto de momento, cortes de los tercios) se estima **dentro de
cada pliegue de entrenamiento**. La referencia de cada persona usa solo sus
propias respuestas, así que no cruza pliegues.

Plan, estadístico principal y reglas de lectura: `INFORME.md`, escritos antes
de ejecutar este script.

**Solo `train` + `val`.** El test se aparta antes de agregar a sesión. El test
está gastado (MODELADO.md): este análisis es exploratorio.

Ejecutar (desde la raíz del repositorio):
    python analisis/normalizacion_vas/03_eeg/eeg_etiqueta_normalizada.py > analisis/normalizacion_vas/03_eeg/ejecucion.log

La ubicación del dataset se toma de la variable de entorno `EEG_DATA`, con
`C:\\Users\\migue\\Desktop\\TFG\\EEG-Data` por defecto (ver `analisis/README.md`).
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from eeg_library.modelado.modeling import (
    COLUMNA_ETIQUETA,
    COLUMNA_GRUPO,
    a_clases,
    agregar_a_sesion,
    centrar_por_persona_loo,
    columnas_de_features_sesion,
    cortes_terciles,
    metricas_clasificacion_por_pliegue,
    residualizar_por_visita,
)

CARPETA = Path(__file__).resolve().parent
RAIZ_ANALISIS = CARPETA.parents[1]

# Mismas piezas que los guiones cuyas cifras se reproducen (como hace
# `analisis_control_visita.py`), y las del paso 2 para construir las etiquetas.
sys.path.insert(0, str(RAIZ_ANALISIS))
from analisis_fase14b_clases import bootstrap_metrica, matriz_confusion  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "paso2", CARPETA.parent / "02_etiquetas" / "etiquetas_normalizadas.py")
paso2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paso2)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEMILLA = 42
SEMILLAS_ESTABILIDAD = (1, 2, 3, 4)
N_BOOTSTRAP = 2000  # el de la Fase 14b
REFERENCIA_SIN_CONTROL = 0.3947  # Fase 14b
REFERENCIA_CON_CONTROL = 0.3795  # control de visita (2026-10-06)

EEG_DATA = Path(os.environ.get("EEG_DATA", r"C:\Users\migue\Desktop\TFG\EEG-Data"))
RUTA_MATRIZ = EEG_DATA / "processed" / "matriz_rf.parquet"
CARPETA_FIGURAS = CARPETA / "figuras"
CARPETA_TABLAS = CARPETA / "tablas"

VARIANTES = {
    "a": "(a) centrado R1 — Fase 14b",
    "a_prima": "(a′) centrado R4",
    "b": "(b) z SD encogida, R4 — principal",
    "c": "(c) (a′) sin efecto de momento",
}

COLOR_SIN = "#2a78d6"
COLOR_CON = "#eb6834"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
REJILLA = "#e4e3df"


def _titulo(texto):
    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


# --- Etiquetas por pliegue -----------------------------------------------------

class ConstructorEtiquetas:
    """
    Construye X e y de cada variante para un pliegue, estimando los parámetros
    poblacionales solo con las personas de entrenamiento de ese pliegue.
    """

    def __init__(self, sesiones, vas_bruto, Xc, features, respuestas):
        self.sesiones = sesiones[[COLUMNA_GRUPO, "medicion"]].reset_index(drop=True)
        self.vas = np.asarray(vas_bruto, float)
        self.Xc = Xc.reset_index(drop=True)
        self.features = features
        self.respuestas = respuestas
        self.g = self.sesiones[COLUMNA_GRUPO].to_numpy()
        self.medicion = self.sesiones["medicion"].to_numpy()
        self.y_a = None  # se fija desde fuera: la etiqueta de la Fase 14b
        self.registro = []
        self._cache = {}

    def _r4(self, idx_tr):
        clave = tuple(idx_tr)
        if clave in self._cache:
            return self._cache[clave]
        personas_tr = set(self.g[idx_tr])
        resp_tr = self.respuestas[self.respuestas[COLUMNA_GRUPO].isin(personas_tr)]
        offset, _ = paso2.desplazamiento_m1(resp_tr)
        u4 = paso2.universos(self.respuestas, offset)["R4"]
        refs = paso2.referencias_loo(self.sesiones, u4)
        a_prima = self.vas - refs["media_ref"].to_numpy()
        por = (u4[u4[COLUMNA_GRUPO].isin(personas_tr)]
               .groupby(COLUMNA_GRUPO)["vas"].agg(["size", "var"]))
        previa = paso2.ajustar_previa_varianzas(por["var"], por["size"] - 1)
        sd = np.sqrt(paso2.encoger_varianza(refs["var_ref"], refs["n_ref"], previa))
        efecto_y = pd.Series(a_prima[idx_tr]).groupby(self.medicion[idx_tr]).mean()
        efecto_x = self.Xc.iloc[idx_tr].groupby(self.medicion[idx_tr]).mean()
        salida = {"a_prima": a_prima, "b": a_prima / sd,
                  "c": a_prima - pd.Series(self.medicion).map(efecto_y).to_numpy(),
                  "Xc_c": self.Xc - efecto_x.reindex(self.medicion).to_numpy()}
        self.registro.append({"offset_m1": offset, "d0": previa["d0"],
                              "s0": float(np.sqrt(previa["s0_2"])),
                              "efecto_momento": {k: float(v) for k, v in efecto_y.items()}})
        self._cache[clave] = salida
        return salida

    def _visita(self, X, y):
        Xr = residualizar_por_visita(X, self.features, self.medicion, self.g)
        yr = residualizar_por_visita(pd.DataFrame({"y": y}), ["y"], self.medicion,
                                     self.g)["y"].to_numpy()
        return Xr, yr

    def construir(self, variante, control, idx_tr):
        if variante == "a":
            X, y = self.Xc, self.y_a
        else:
            r = self._r4(idx_tr)
            X = r["Xc_c"] if variante == "c" else self.Xc
            y = r[variante]
        if control:
            X, y = self._visita(X, y)
        return X, y


def cv_clases(constructor, variante, control, semilla=SEMILLA, n_splits=5):
    """
    Réplica de `predecir_clases_cv_agrupada` con la etiqueta construida dentro
    de cada pliegue. Con la variante (a), que no tiene parámetros poblacionales,
    es exactamente el mismo cálculo (se comprueba contra 0.3947 / 0.3795).
    """
    n = len(constructor.g)
    t = np.full(n, np.nan)
    p = np.full(n, np.nan)
    pliegue = np.full(n, -1, dtype=int)
    cortes_por_pliegue = []
    for i, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(
            constructor.Xc, groups=constructor.g), start=1):
        if set(constructor.g[tr]) & set(constructor.g[va]):
            raise ValueError(f"El pliegue {i} comparte personas entre train y validación.")
        X, y = constructor.construir(variante, control, tr)
        cortes = cortes_terciles(y[tr])
        ytr = a_clases(y[tr], cortes)
        t[va] = a_clases(y[va], cortes)
        pliegue[va] = i
        usables = ~np.isnan(ytr)
        modelo = RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                        random_state=semilla, n_jobs=-1)
        modelo.fit(X.iloc[tr][usables], ytr[usables])
        p[va] = modelo.predict(X.iloc[va])
        cortes_por_pliegue.append([float(c) for c in cortes])
    return t, p, pliegue, cortes_por_pliegue


def bootstrap_diferencia(res_a, res_b, grupos, n=N_BOOTSTRAP, semilla=SEMILLA):
    """
    IC95 de la diferencia de exactitud balanceada (b − a) remuestreando las
    **mismas** personas para las dos: la comparación es emparejada.
    """
    rng = np.random.default_rng(semilla)
    unicos = pd.unique(grupos)
    indices = {u: np.flatnonzero(grupos == u) for u in unicos}
    difs = []
    for _ in range(n):
        filas = np.concatenate([indices[u] for u in
                                rng.choice(unicos, size=len(unicos), replace=True)])
        try:
            ma = metricas_clasificacion_por_pliegue(
                res_a[0][filas], res_a[1][filas], res_a[2][filas])
            mb = metricas_clasificacion_por_pliegue(
                res_b[0][filas], res_b[1][filas], res_b[2][filas])
        except ValueError:
            continue
        difs.append(mb["exactitud_balanceada"] - ma["exactitud_balanceada"])
    difs = np.array(difs)
    return ([float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5))],
            float((difs <= 0).mean()))


# --- Figuras -------------------------------------------------------------------

def _estilo(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(TINTA_2)
    ax.tick_params(colors=TINTA_2, labelsize=9)
    ax.xaxis.grid(True, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)


def figura_exactitud(tabla, ruta):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    claves = list(VARIANTES)
    for j, (control, color, marcador, nombre) in enumerate(
            ((False, COLOR_SIN, "o", "sin control de visita"),
             (True, COLOR_CON, "s", "con control de visita"))):
        sub = tabla[tabla.control == control].set_index("variante").loc[claves]
        yy = np.arange(len(claves))[::-1] + (0.15 if not control else -0.15)
        ax.errorbar(sub["exactitud_balanceada"], yy,
                    xerr=[sub["exactitud_balanceada"] - sub["ic95_inf"],
                          sub["ic95_sup"] - sub["exactitud_balanceada"]],
                    fmt=marcador, color=color, markersize=8, capsize=3, linewidth=1.5,
                    markeredgecolor="white", label=nombre)
    ax.axvline(1 / 3, color=TINTA_2, linewidth=1, linestyle="--")
    ax.text(1 / 3, -0.75, " azar 1/3", color=TINTA_2, fontsize=8, va="bottom")
    ax.set_ylim(-0.8, len(claves) - 0.5)
    ax.set_yticks(np.arange(len(claves))[::-1], [VARIANTES[k] for k in claves], fontsize=9)
    ax.set_xlabel("exactitud balanceada (media de pliegues, IC95 por personas)",
                  fontsize=9, color=TINTA_2)
    ax.set_title("Tarea B con cada etiqueta", fontsize=11, color=TINTA, loc="left", pad=22)
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=2,
              bbox_to_anchor=(1, 1.0), borderaxespad=0.2)
    _estilo(ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_diferencias(difs, ruta):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    claves = [k for k in VARIANTES if k != "a"]
    for control, color, marcador, nombre in (
            (False, COLOR_SIN, "o", "sin control de visita"),
            (True, COLOR_CON, "s", "con control de visita")):
        sub = difs[difs.control == control].set_index("variante").loc[claves]
        yy = np.arange(len(claves))[::-1] + (0.15 if not control else -0.15)
        ax.errorbar(sub["diferencia"], yy,
                    xerr=[sub["diferencia"] - sub["ic95_inf"],
                          sub["ic95_sup"] - sub["diferencia"]],
                    fmt=marcador, color=color, markersize=8, capsize=3, linewidth=1.5,
                    markeredgecolor="white", label=nombre)
    ax.axvline(0, color=TINTA_2, linewidth=1)
    ax.set_yticks(np.arange(len(claves))[::-1], [VARIANTES[k] for k in claves], fontsize=9)
    ax.set_xlabel("diferencia emparejada con (a), exactitud balanceada (IC95)",
                  fontsize=9, color=TINTA_2)
    ax.set_title("¿Cuánto cambia frente a la etiqueta de la Fase 14b?", fontsize=11,
                 color=TINTA, loc="left", pad=22)
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=2,
              bbox_to_anchor=(1, 1.0), borderaxespad=0.2)
    _estilo(ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# --- Principal -----------------------------------------------------------------

def solo_figuras():
    """Regenera las figuras desde las tablas ya guardadas, sin reentrenar."""
    figura_exactitud(pd.read_csv(CARPETA_TABLAS / "exactitud_por_variante.csv"),
                     CARPETA_FIGURAS / "exactitud_por_variante.png")
    figura_diferencias(pd.read_csv(CARPETA_TABLAS / "diferencias_con_a.csv"),
                       CARPETA_FIGURAS / "diferencia_con_a.png")


def main():
    CARPETA_FIGURAS.mkdir(exist_ok=True)
    CARPETA_TABLAS.mkdir(exist_ok=True)
    resultados = {}

    _titulo("1. DATOS (train + val; test apartado antes de agregar)")
    matriz = pd.read_parquet(RUTA_MATRIZ)
    n_test = int((matriz["split"] == "test").sum())
    matriz = matriz[matriz["split"].isin(["train", "val"])]
    ses = agregar_a_sesion(matriz)
    assert set(ses["split"]) <= {"train", "val"}
    features = columnas_de_features_sesion(ses)
    respuestas, objetivo = paso2.cargar()
    if set(ses["sesion_uid"]) != set(objetivo["sesion_uid"]):
        raise ValueError("Las sesiones de la matriz no son las etiquetadas del paso 2.")
    centradas = centrar_por_persona_loo(ses, features + [COLUMNA_ETIQUETA])
    validas = centradas[COLUMNA_ETIQUETA].notna()
    cen = centradas[validas].reset_index(drop=True)
    vas_bruto = ses.loc[validas, COLUMNA_ETIQUETA].to_numpy(float)
    print(f"Grabaciones de test apartadas: {n_test} (no se agregan ni se leen)")
    print(f"Sesiones train+val: {len(ses)}; con X centrable (Tarea B): {len(cen)} "
          f"de {cen[COLUMNA_GRUPO].nunique()} personas; {len(features)} features")
    resultados["datos"] = {"sesiones": int(len(cen)),
                           "personas": int(cen[COLUMNA_GRUPO].nunique()),
                           "features": len(features)}

    constructor = ConstructorEtiquetas(cen, vas_bruto, cen[features], features, respuestas)
    constructor.y_a = cen[COLUMNA_ETIQUETA].to_numpy(float)
    g = constructor.g

    _titulo("2. EXACTITUD BALANCEADA DE CADA VARIANTE")
    res, filas = {}, []
    for control in (False, True):
        for v, nombre in VARIANTES.items():
            t, p, pl, cortes = cv_clases(constructor, v, control)
            m = metricas_clasificacion_por_pliegue(t, p, pl)
            lo, hi = bootstrap_metrica(t, p, g, pl, "exactitud_balanceada", n=N_BOOTSTRAP)
            res[(v, control)] = (t, p, pl)
            filas.append({"variante": v, "nombre": nombre, "control": control,
                          "exactitud_balanceada": m["exactitud_balanceada"],
                          "sd_pliegues": m["exactitud_balanceada_sd"],
                          "ic95_inf": lo, "ic95_sup": hi,
                          "f1_macro": m["f1_macro"], "kendall_tau": m["kendall_tau"],
                          "por_pliegue": [round(f["exactitud_balanceada"], 4)
                                          for f in m["por_pliegue"]],
                          "cortes": cortes})
            etiqueta = "con control" if control else "sin control"
            print(f"{nombre:<38s} {etiqueta:<12s} bal = {m['exactitud_balanceada']:.4f} "
                  f"IC95 [{lo:.4f}, {hi:.4f}]  tau {m['kendall_tau']:+.3f}  pliegues "
                  + " ".join(f"{f['exactitud_balanceada']:.3f}" for f in m["por_pliegue"]))
            if v == "a":
                ref = REFERENCIA_CON_CONTROL if control else REFERENCIA_SIN_CONTROL
                if round(m["exactitud_balanceada"], 4) != ref:
                    raise ValueError(
                        f"(a) {etiqueta} da {m['exactitud_balanceada']:.4f} y debía "
                        f"reproducir {ref}. La comparación no sería válida.")
                print(f"{'':<38s} → reproduce el {ref} documentado")
    tabla = pd.DataFrame(filas)
    tabla.drop(columns=["por_pliegue", "cortes"]).to_csv(
        CARPETA_TABLAS / "exactitud_por_variante.csv", index=False, float_format="%.4f")
    resultados["variantes"] = filas
    resultados["parametros_por_pliegue"] = constructor.registro[:5]

    print("\nMatriz de confusión de (b) sin control (amontonada; filas = real):")
    t, p, _ = res[("b", False)]
    print("  " + matriz_confusion(t, p).to_string().replace("\n", "\n  "))

    _titulo("3. ESTADÍSTICO PRINCIPAL — DIFERENCIA EMPAREJADA CON (a)")
    difs = []
    for control in (False, True):
        base = res[("a", control)]
        m_a = metricas_clasificacion_por_pliegue(*base)["exactitud_balanceada"]
        for v in ("a_prima", "b", "c"):
            m_v = metricas_clasificacion_por_pliegue(*res[(v, control)])["exactitud_balanceada"]
            ic, p_menor = bootstrap_diferencia(base, res[(v, control)], g)
            t_a, t_v = base[0], res[(v, control)][0]
            difs.append({"variante": v, "control": control, "diferencia": m_v - m_a,
                         "ic95_inf": ic[0], "ic95_sup": ic[1],
                         "fraccion_boot_menor_igual_0": p_menor,
                         "clases_reales_que_cambian": float((t_a != t_v).mean())})
            etiqueta = "con control" if control else "sin control"
            print(f"{VARIANTES[v]:<38s} {etiqueta:<12s} Δ = {m_v - m_a:+.4f} "
                  f"IC95 [{ic[0]:+.4f}, {ic[1]:+.4f}]  "
                  f"(clase real distinta de (a) en {(t_a != t_v).mean():.1%} de sesiones)")
    tabla_difs = pd.DataFrame(difs)
    tabla_difs.to_csv(CARPETA_TABLAS / "diferencias_con_a.csv", index=False,
                      float_format="%.4f")
    resultados["diferencias"] = difs

    _titulo("4. ESTABILIDAD FRENTE A LA SEMILLA DEL RF (sin control)")
    estab = {}
    for v in ("a", "b"):
        valores = [metricas_clasificacion_por_pliegue(
            *cv_clases(constructor, v, False, semilla=s)[:3])["exactitud_balanceada"]
            for s in (SEMILLA,) + SEMILLAS_ESTABILIDAD]
        estab[v] = valores
        print(f"{VARIANTES[v]:<38s} semillas 42,1–4: " + " ".join(f"{x:.4f}" for x in valores)
              + f"   media {np.mean(valores):.4f}, SD {np.std(valores, ddof=1):.4f}")
    dif_semillas = np.array(estab["b"]) - np.array(estab["a"])
    print(f"Diferencia (b) − (a) por semilla: " + " ".join(f"{d:+.4f}" for d in dif_semillas)
          + f"   media {dif_semillas.mean():+.4f}")
    resultados["estabilidad_semillas"] = {**estab, "diferencia": dif_semillas.tolist()}

    _titulo("5. LECTURA SEGÚN LAS REGLAS FIJADAS")
    principal = tabla_difs[(tabla_difs.variante == "b") & (~tabla_difs.control)].iloc[0]
    fila_b = tabla[(tabla.variante == "b") & (~tabla.control)].iloc[0]
    if principal.ic95_inf <= 0 <= principal.ic95_sup:
        regla = 1
        texto = ("IC95 de (b) − (a) contiene 0: normalizar la etiqueta no cambia la "
                 "respuesta del EEG; la conclusión de las Fases 14/14b se mantiene.")
    elif principal.ic95_inf > 0:
        regla = 2
        texto = ("IC95 de (b) − (a) por encima de 0: la etiqueta mejor estimada deja "
                 "ver más señal (exploratorio).")
    else:
        regla = 3
        texto = ("IC95 de (b) − (a) por debajo de 0: la cifra de la Fase 14b dependía "
                 "de la construcción de la etiqueta.")
    azar = "excluye" if fila_b.ic95_inf > 1 / 3 else "no excluye"
    print(f"Regla {regla}: {texto}")
    print(f"El IC95 de (b) sin control [{fila_b.ic95_inf:.4f}, {fila_b.ic95_sup:.4f}] "
          f"{azar} el azar (1/3).")
    resultados["lectura"] = {"regla": regla, "texto": texto, "ic_b_frente_azar": azar}

    _titulo("6. FIGURAS")
    figura_exactitud(tabla, CARPETA_FIGURAS / "exactitud_por_variante.png")
    figura_diferencias(tabla_difs, CARPETA_FIGURAS / "diferencia_con_a.png")
    print("Escritas en", CARPETA_FIGURAS)

    with open(CARPETA / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2, default=float)
    print("Resultados en", CARPETA / "resultados.json")


if __name__ == "__main__":
    if "--solo-figuras" in sys.argv:
        solo_figuras()
    else:
        main()
