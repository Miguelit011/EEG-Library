"""
Tests de las piezas del análisis confirmatorio (`confirmatorio.py`).

Lo que se fija aquí son las reglas que hacen válido el test, no sus cifras:

1. El p por permutación dentro de persona es el convencional, `(1 + #≥)/(N+1)`,
   solo mueve etiquetas dentro de cada persona y, sin información, rechaza a
   su nivel nominal.
2. Las réplicas bootstrap remuestrean personas enteras y, con varios
   estadísticos a la vez, son pareadas.
3. Holm y el veredicto de la principal hacen exactamente lo que dice el
   pre-registro.
4. La Tarea B se prepara igual con y sin control de visita salvo el control.
5. Una ejecución nunca sobrescribe a otra.
"""

import subprocess

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score

from eeg_library.modelado.confirmatorio import (
    carpeta_salida_nueva,
    centrar_y_controlar,
    estado_git,
    exactitud_balanceada_lote,
    holm,
    ic_percentil,
    importancia_permutacion_evaluada,
    p_bootstrap,
    p_permutacion,
    permutaciones_dentro_de_grupo,
    preparar_tarea_b,
    replicas_bootstrap,
    sha256_archivo,
    veredicto_principal,
)


# -- exactitud balanceada ---------------------------------------------------

def test_exactitud_balanceada_lote_coincide_con_sklearn():
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 3, 200)
    yp = rng.integers(0, 3, 200)
    assert exactitud_balanceada_lote(yt, yp, [0, 1, 2])[0] == pytest.approx(
        balanced_accuracy_score(yt, yp))


def test_exactitud_balanceada_lote_puntua_varias_versiones_a_la_vez():
    rng = np.random.default_rng(1)
    yp = rng.integers(0, 2, 50)
    varias = rng.integers(0, 2, (4, 50))
    lote = exactitud_balanceada_lote(varias, yp, [0, 1])
    for fila, valor in zip(varias, lote):
        assert valor == pytest.approx(balanced_accuracy_score(fila, yp))


def test_exactitud_balanceada_lote_exige_longitudes_coherentes():
    with pytest.raises(ValueError, match="mismo número de filas"):
        exactitud_balanceada_lote([0, 1, 0], [0, 1], [0, 1])


# -- permutación dentro de persona -------------------------------------------

def test_permutaciones_no_sacan_valores_de_su_persona():
    grupos = np.array(["a"] * 4 + ["b"] * 3 + ["c"] * 5)
    idx = permutaciones_dentro_de_grupo(grupos, 50, np.random.default_rng(2))
    assert (grupos[idx] == grupos).all()
    # y de verdad barajan
    assert not (idx == np.arange(len(grupos))).all()


def test_p_permutacion_usa_la_definicion_convencional():
    """Un acierto perfecto y repetible: ninguna permutación lo supera, p = 1/(N+1)."""
    grupos = np.repeat(np.arange(20), 6)
    yt = np.tile([0, 1, 2, 0, 1, 2], 20)
    r = p_permutacion(yt, yt, grupos, [0, 1, 2], n_permutaciones=999, semilla=3)
    assert r["observado"] == 1.0
    assert r["p"] == pytest.approx((1 + np.sum(r["nulo"] >= 1.0)) / 1000)
    assert r["p"] < 0.01
    assert len(r["nulo"]) == 999


def test_p_permutacion_con_una_clase_por_persona_no_puede_rechazar():
    """Si cada persona tiene una sola clase, barajar dentro no cambia nada: p = 1."""
    grupos = np.repeat(np.arange(10), 4)
    yt = np.repeat(np.arange(10) % 3, 4)
    r = p_permutacion(yt, yt, grupos, [0, 1, 2], n_permutaciones=200)
    assert r["p"] == 1.0


def test_p_permutacion_sin_informacion_rechaza_a_su_nivel():
    """Predicciones independientes de la etiqueta: tasa de rechazo ~0.05."""
    rng = np.random.default_rng(4)
    rechazos = []
    for s in range(300):
        grupos = np.repeat(np.arange(25), 4)
        yt = rng.integers(0, 3, 100)
        yp = rng.integers(0, 3, 100)
        rechazos.append(p_permutacion(yt, yp, grupos, [0, 1, 2],
                                      n_permutaciones=199, semilla=s)["p"] < 0.05)
    assert 0.01 <= np.mean(rechazos) <= 0.10


def test_p_permutacion_exige_longitudes_coherentes():
    with pytest.raises(ValueError, match="misma longitud"):
        p_permutacion([0, 1], [0, 1], [0], [0, 1])


# -- bootstrap ---------------------------------------------------------------

def test_replicas_bootstrap_remuestrean_personas_enteras():
    grupos = np.array(["a"] * 3 + ["b"] * 1 + ["c"] * 2)
    tamanos = replicas_bootstrap(grupos, lambda idx: len(idx), n_replicas=200)
    # cada réplica es una suma de 3 tamaños de persona elegidos con reposición
    posibles = {a + b + c for a in (3, 1, 2) for b in (3, 1, 2) for c in (3, 1, 2)}
    assert set(tamanos.astype(int)) <= posibles


def test_replicas_bootstrap_vectoriales_son_pareadas():
    """Dos estadísticos en la misma llamada ven la misma remuestra."""
    grupos = np.repeat(np.arange(30), 3)
    x = np.random.default_rng(5).normal(size=90)
    rep = replicas_bootstrap(grupos, lambda idx: (x[idx].mean(), x[idx].mean() + 1),
                             n_replicas=100)
    np.testing.assert_allclose(rep[:, 1] - rep[:, 0], 1.0)


def test_replicas_bootstrap_dejan_nan_donde_no_hay_estadistico():
    grupos = np.arange(5)

    def a_veces_falla(idx):
        if 0 in idx:
            raise ValueError("no definido")
        return 1.0
    rep = replicas_bootstrap(grupos, a_veces_falla, n_replicas=100)
    assert np.isnan(rep).any() and (rep[~np.isnan(rep)] == 1.0).all()


def test_ic_percentil_y_p_bootstrap():
    rep = np.arange(1, 101, dtype=float)
    assert ic_percentil(rep, 0.9) == pytest.approx([5.95, 95.05])
    assert p_bootstrap(rep, 10, "menor_o_igual") == pytest.approx(11 / 101)
    assert p_bootstrap(rep, 91, "mayor_o_igual") == pytest.approx(11 / 101)
    with pytest.raises(ValueError):
        p_bootstrap(rep, 0, "otro")


# -- Holm y veredicto --------------------------------------------------------

def test_holm_ajusta_en_orden_y_es_monotono():
    p = [0.01, 0.04, 0.03, 0.005]
    # ordenados: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> 0.06
    np.testing.assert_allclose(holm(p), [0.03, 0.06, 0.06, 0.02])
    assert holm([0.9, 0.8]).max() <= 1.0


def test_veredicto_principal_sigue_el_pre_registro():
    assert veredicto_principal(0.01, 0.50, 0.3795) == "confirma"
    assert veredicto_principal(0.20, 0.35, 0.3795) == "no confirma"
    assert veredicto_principal(0.20, 0.45, 0.3795) == "no concluye"
    # decide el p aunque el IC llegue por debajo de la referencia
    assert veredicto_principal(0.04, 0.37, 0.3795) == "confirma"


# -- preparación de la Tarea B ----------------------------------------------

def _sesiones():
    filas = []
    for p, base in (("a", 30.0), ("b", 60.0), ("c", 45.0)):
        for k, m in enumerate(("M2", "M3", "M4", "M5")):
            filas.append({"sesion_uid": f"{p}{m}", "persona_uid": p, "medicion": m,
                          "vas": base + 5 * k + (k % 2), "f1": base / 10 + k})
    filas.append({"sesion_uid": "dM2", "persona_uid": "d", "medicion": "M2",
                  "vas": 10.0, "f1": 1.0})
    return pd.DataFrame(filas)


def test_preparar_tarea_b_descarta_personas_sin_referencia_y_guarda_el_original():
    ses = _sesiones()
    cen = preparar_tarea_b(ses, ["f1"], controlar_visita=False)
    assert "d" not in set(cen["persona_uid"])
    np.testing.assert_allclose(cen["vas_original"], ses["vas"].iloc[:12])
    np.testing.assert_allclose(cen["vas"] + cen["vas_loo"], cen["vas_original"])


def test_preparar_tarea_b_el_control_quita_la_tendencia_de_visita():
    ses = _sesiones()
    cen = preparar_tarea_b(ses, ["f1"], controlar_visita=True)
    # f1 es nivel + visita exacta: con el control no queda nada
    np.testing.assert_allclose(cen["f1"], 0.0, atol=1e-12)
    assert cen.attrs["personas_residualizadas"] == 3
    # la desviación sin control se conserva para lo descriptivo
    sin = preparar_tarea_b(ses, ["f1"], controlar_visita=False)
    np.testing.assert_allclose(cen["desviacion_sin_control"], sin["vas"])


def test_centrar_y_controlar_centra_tambien_a_quien_no_tiene_recta():
    valores = [1.0, 3.0, 10.0, 20.0, 30.0]
    medicion = ["M2", "M3", "M2", "M3", "M4"]
    grupos = ["a", "a", "b", "b", "b"]
    v, rectas = centrar_y_controlar(valores, medicion, grupos)
    np.testing.assert_allclose(v[:2], [-1.0, 1.0])     # centrado, sin recta
    np.testing.assert_allclose(v[2:], 0.0, atol=1e-12)  # tendencia exacta
    assert rectas == 1


# -- importancia por permutación sobre datos evaluados ----------------------

class _Umbral:
    """Modelo mínimo ya ajustado: clase 1 si x > 0."""

    def predict(self, X):
        return (X["x"].to_numpy() > 0).astype(int)


def test_importancia_evaluada_solo_penaliza_lo_que_usa_el_modelo():
    rng = np.random.default_rng(6)
    X = pd.DataFrame({"x": rng.normal(size=400), "ruido": rng.normal(size=400)})
    y = (X["x"] > 0).astype(int).to_numpy()
    imp = importancia_permutacion_evaluada(
        _Umbral(), X, y, lambda a, b: balanced_accuracy_score(a, b), n_repeticiones=5)
    media = imp.groupby("nombre")["caida"].mean()
    assert media["x"] > 0.3
    assert media["ruido"] == 0.0
    bloques = importancia_permutacion_evaluada(
        _Umbral(), X, y, lambda a, b: balanced_accuracy_score(a, b),
        bloques={"todo": ["x", "ruido"]}, n_repeticiones=3)
    assert set(bloques["nombre"]) == {"todo"}
    with pytest.raises(ValueError, match="no están en X"):
        importancia_permutacion_evaluada(_Umbral(), X, y, balanced_accuracy_score,
                                         bloques={"b": ["falta"]})


# -- trazabilidad ------------------------------------------------------------

def test_carpeta_salida_nueva_no_sobrescribe(tmp_path):
    ruta = carpeta_salida_nueva(tmp_path / "salida")
    assert ruta.is_dir()
    with pytest.raises(FileExistsError):
        carpeta_salida_nueva(tmp_path / "salida")


def test_sha256_archivo(tmp_path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"abc")
    assert sha256_archivo(f) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


def test_estado_git_detecta_cambios_sin_commitear(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    git("init")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (tmp_path / "a.txt").write_text("1")
    git("add", "a.txt")
    git("commit", "-m", "x")
    limpio = estado_git(tmp_path)
    assert limpio["limpio"] and len(limpio["commit"]) == 40
    (tmp_path / "a.txt").write_text("2")
    assert not estado_git(tmp_path)["limpio"]
