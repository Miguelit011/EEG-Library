"""
Tests para el módulo de evaluación de calidad (Fase 4).
"""

import numpy as np
import pytest
import mne

from eeg_library.preprocesado.quality import evaluar_calidad, CANALES_EPOC

# Frecuencia de muestreo y duración usada en todos los tests
SFREQ = 256
N_MUESTRAS = SFREQ * 10  # 10 segundos


def _raw_sintetico(señales_eeg: dict, señales_cq: dict | None = None) -> mne.io.RawArray:
    """
    Construye un RawArray de MNE a partir de dicts {nombre_canal: array}.

    señales_eeg : {nombre: array de shape (N_MUESTRAS,)}
    señales_cq  : {nombre_cq: array}  — opcional
    """
    canales = dict(señales_eeg)
    if señales_cq:
        canales.update(señales_cq)

    nombres = list(canales.keys())
    data = np.array([canales[n] for n in nombres])  # (n_ch, n_muestras)

    # Tipos de canal: "eeg" para los EEG, "misc" para los CQ
    tipos = []
    for n in nombres:
        tipos.append("eeg" if not n.startswith("CQ_") else "misc")

    info = mne.create_info(ch_names=nombres, sfreq=SFREQ, ch_types=tipos)
    return mne.io.RawArray(data, info, verbose=False)


def _señal_normal():
    """Señal EEG sintética con varianza típica (~1e-11 V²)."""
    rng = np.random.default_rng(0)
    return rng.normal(0, 5e-6, N_MUESTRAS)  # std ≈ 5 µV


def _cq_buena():
    """Señal CQ con valor constante 4 (buen contacto)."""
    return np.full(N_MUESTRAS, 4.0)


# ---------------------------------------------------------------------------
# Criterio 1 — CQ medio < 3.0
# ---------------------------------------------------------------------------

def test_excluye_por_cq_medio_bajo():
    cq_baja = np.full(N_MUESTRAS, 2.0)  # media = 2.0 < 3.0
    raw = _raw_sintetico(
        {"AF3": _señal_normal()},
        {"CQ_AF3": cq_baja},
    )
    good, excluidos, _ = evaluar_calidad(raw)
    assert "AF3" not in good
    assert "AF3" in excluidos
    assert "CQ medio bajo" in excluidos["AF3"]


def test_acepta_canal_con_cq_medio_exactamente_3():
    cq_limite = np.full(N_MUESTRAS, 3.0)
    raw = _raw_sintetico(
        {"AF3": _señal_normal()},
        {"CQ_AF3": cq_limite},
    )
    good, excluidos, _ = evaluar_calidad(raw)
    assert "AF3" in good
    assert "AF3" not in excluidos


# ---------------------------------------------------------------------------
# Criterio 2 — más del 20 % de muestras con CQ <= 1
# ---------------------------------------------------------------------------

def test_excluye_por_exceso_muestras_malas():
    # 30 % de muestras con CQ = 1, el resto con CQ = 4
    cq = np.full(N_MUESTRAS, 4.0)
    n_malas = int(N_MUESTRAS * 0.30)
    cq[:n_malas] = 1.0
    raw = _raw_sintetico(
        {"F7": _señal_normal()},
        {"CQ_F7": cq},
    )
    good, excluidos, _ = evaluar_calidad(raw)
    assert "F7" not in good
    assert "Exceso de muestras malas" in excluidos["F7"]


def test_acepta_canal_con_20_pct_malas_exacto():
    cq = np.full(N_MUESTRAS, 4.0)
    n_malas = int(N_MUESTRAS * 0.20)
    cq[:n_malas] = 1.0
    raw = _raw_sintetico(
        {"F7": _señal_normal()},
        {"CQ_F7": cq},
    )
    good, excluidos, _ = evaluar_calidad(raw)
    # pct_malo = exactamente 0.20 → no excluido (condición estricta > 0.20)
    assert "F7" in good


# ---------------------------------------------------------------------------
# Criterio 3 — señal plana (varianza < umbral_flat)
# ---------------------------------------------------------------------------

def test_excluye_señal_plana():
    señal_plana = np.zeros(N_MUESTRAS)
    raw = _raw_sintetico({"F3": señal_plana})
    good, excluidos, _ = evaluar_calidad(raw)
    assert "F3" not in good
    assert "plana" in excluidos["F3"].lower()


def test_umbral_flat_configurable():
    # Señal con varianza pequeña pero > umbral_flat personalizado
    rng = np.random.default_rng(1)
    señal = rng.normal(0, 1e-6, N_MUESTRAS)  # var ≈ 1e-12
    raw = _raw_sintetico({"FC5": señal})
    # Con umbral_flat muy pequeño no debería excluirse
    good, excluidos, _ = evaluar_calidad(raw, umbral_flat=1e-15)
    assert "FC5" in good


# ---------------------------------------------------------------------------
# Criterio 4 — amplitud excesiva (z-score robusto inter-canal, FASTER)
# ---------------------------------------------------------------------------

def _señales_normales_distintas(canales, semilla=42):
    """
    Construye señales EEG sintéticas con varianzas ligeramente distintas
    entre canales (para que el MAD inter-canal sea > 0).
    """
    rng = np.random.default_rng(semilla)
    señales = {}
    for i, ch in enumerate(canales):
        std = 5e-6 * (1 + 0.05 * i)  # std entre 5 y ~8 µV
        señales[ch] = rng.normal(0, std, N_MUESTRAS)
    return señales


def test_excluye_amplitud_excesiva_por_z_score():
    # 13 canales con amplitud normal y uno con varianza ~10000x mayor
    canales = CANALES_EPOC[:13]
    señales = _señales_normales_distintas(canales)
    rng = np.random.default_rng(99)
    canal_outlier = CANALES_EPOC[13]
    señales[canal_outlier] = rng.normal(0, 5e-4, N_MUESTRAS)  # var ≈ 2.5e-7

    raw = _raw_sintetico(señales)
    good, excluidos, metricas = evaluar_calidad(raw)

    assert canal_outlier not in good
    assert "Amplitud excesiva" in excluidos[canal_outlier]
    assert metricas[canal_outlier]["z_robusto"] > 3.0


def test_no_excluye_amplitud_normal():
    # Varianzas ligeramente distintas pero todas en rango fisiológico
    canales = CANALES_EPOC[:14]
    señales = _señales_normales_distintas(canales)
    raw = _raw_sintetico(señales)
    good, excluidos, _ = evaluar_calidad(raw)
    # Ningún canal debe ser excluido por amplitud excesiva
    assert all("Amplitud excesiva" not in motivo for motivo in excluidos.values())
    assert set(good) == set(canales)


def test_umbral_z_configurable():
    # Un canal con varianza moderadamente más alta que el resto
    canales = CANALES_EPOC[:13]
    señales = _señales_normales_distintas(canales)
    rng = np.random.default_rng(11)
    canal_alto = CANALES_EPOC[13]
    señales[canal_alto] = rng.normal(0, 2e-5, N_MUESTRAS)  # ~4x la mediana

    raw = _raw_sintetico(señales)

    # Con umbral z muy alto, no se excluye
    good_relajado, _, _ = evaluar_calidad(raw, umbral_z=100.0)
    assert canal_alto in good_relajado

    # Con umbral z muy bajo, sí se excluye
    good_estricto, excluidos_estricto, _ = evaluar_calidad(raw, umbral_z=1.0)
    assert canal_alto not in good_estricto
    assert "Amplitud excesiva" in excluidos_estricto[canal_alto]


def test_caso_degenerado_mad_cero(capsys):
    # Todos los canales con exactamente la misma señal → MAD = 0
    señal_comun = _señal_normal()
    señales = {ch: señal_comun.copy() for ch in CANALES_EPOC}
    raw = _raw_sintetico(señales)

    good, excluidos, metricas = evaluar_calidad(raw)

    # Ningún canal excluido por amplitud excesiva
    assert all("Amplitud excesiva" not in motivo for motivo in excluidos.values())
    # z_robusto debe ser None para todos los canales (criterio no aplicable)
    for ch in CANALES_EPOC:
        assert metricas[ch]["z_robusto"] is None
    # Se debe avisar de la dispersión nula
    salida = capsys.readouterr().out
    assert "MAD = 0" in salida


def test_distingue_motivo_plana_de_amplitud_excesiva():
    # Un canal plano y otro saturado en el mismo registro
    canales = CANALES_EPOC[:12]
    señales = _señales_normales_distintas(canales)
    señales[CANALES_EPOC[12]] = np.zeros(N_MUESTRAS)  # plano
    rng = np.random.default_rng(3)
    señales[CANALES_EPOC[13]] = rng.normal(0, 5e-4, N_MUESTRAS)  # saturado

    raw = _raw_sintetico(señales)
    _, excluidos, _ = evaluar_calidad(raw)

    assert "plana" in excluidos[CANALES_EPOC[12]].lower()
    assert "Amplitud excesiva" in excluidos[CANALES_EPOC[13]]


# ---------------------------------------------------------------------------
# Canal sin señal CQ — solo criterios de varianza
# ---------------------------------------------------------------------------

def test_canal_sin_cq_usa_solo_varianza():
    raw = _raw_sintetico({"P7": _señal_normal()})  # sin CQ_P7
    good, excluidos, metricas = evaluar_calidad(raw)
    assert "P7" in good
    assert metricas["P7"]["cq_medio"] is None
    assert metricas["P7"]["pct_muestras_malas"] is None


# ---------------------------------------------------------------------------
# Canal OK — pasa todos los criterios
# ---------------------------------------------------------------------------

def test_canal_ok_incluido():
    raw = _raw_sintetico(
        {"O1": _señal_normal()},
        {"CQ_O1": _cq_buena()},
    )
    good, excluidos, metricas = evaluar_calidad(raw)
    assert "O1" in good
    assert "O1" not in excluidos
    assert metricas["O1"]["cq_medio"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Métricas devueltas
# ---------------------------------------------------------------------------

def test_metricas_contienen_campos_esperados():
    raw = _raw_sintetico(
        {"AF3": _señal_normal()},
        {"CQ_AF3": _cq_buena()},
    )
    _, _, metricas = evaluar_calidad(raw)
    assert "cq_medio" in metricas["AF3"]
    assert "pct_muestras_malas" in metricas["AF3"]
    assert "varianza" in metricas["AF3"]
    assert "z_robusto" in metricas["AF3"]


# ---------------------------------------------------------------------------
# Advertencias por número de canales buenos
# ---------------------------------------------------------------------------

def test_advertencia_menos_de_8_canales(capsys):
    # 7 canales buenos (uno excluido por señal plana)
    señales = {ch: _señal_normal() for ch in CANALES_EPOC[:7]}
    señales[CANALES_EPOC[7]] = np.zeros(N_MUESTRAS)  # excluido
    # Rellenar el resto con señal plana también para tener exactamente 7 buenos
    for ch in CANALES_EPOC[8:]:
        señales[ch] = np.zeros(N_MUESTRAS)
    raw = _raw_sintetico(señales)
    good, _, _ = evaluar_calidad(raw)
    salida = capsys.readouterr().out
    if len(good) < 6:
        assert "CRITICA" in salida
    elif len(good) < 8:
        assert "ADVERTENCIA" in salida


def test_advertencia_critica_menos_de_6_canales(capsys):
    # Solo 5 canales con señal normal, el resto planas
    señales = {ch: _señal_normal() for ch in CANALES_EPOC[:5]}
    for ch in CANALES_EPOC[5:]:
        señales[ch] = np.zeros(N_MUESTRAS)
    raw = _raw_sintetico(señales)
    good, _, _ = evaluar_calidad(raw)
    assert len(good) == 5
    assert "CRITICA" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Solo procesa canales EPOC presentes
# ---------------------------------------------------------------------------

def test_ignora_canales_no_epoc():
    señales = {"AF3": _señal_normal(), "EXG1": _señal_normal()}
    raw = _raw_sintetico(señales)
    _, _, metricas = evaluar_calidad(raw)
    assert "EXG1" not in metricas
    assert "AF3" in metricas
