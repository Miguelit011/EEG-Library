"""
Tests para el módulo de construcción del dataset supervisado (Fase 12).

Los datos son sintéticos pero reproducen las tres situaciones que hacen que
este enlace no sea un `merge` trivial, y que son las que de verdad hay que
proteger con tests:

1. Una pulsera con **dos ocupantes** repartidos entre momentos, que es la razón
   por la que la clave es `(sujeto_id, medicion)` y no el sujeto.
2. Grabaciones **archivadas bajo el centro equivocado**, que cambian de
   `sujeto_id` y por tanto de split.
3. La matriz de features, que lleva el `sujeto` y el `split` de **antes** de
   corregir, y que por eso solo se puede unir por `archivo_origen`.
"""

import numpy as np
import pandas as pd
import pytest

from eeg_library.procesado.labels import (
    CORRECCIONES_MANIFEST,
    CORRECCIONES_VAS,
    construir_etiquetas,
    corregir_manifest,
    expandir_a_grabaciones,
    normalizar_vas,
    unir_etiquetas,
)

CONDICIONES = ["OC", "OA", "STROOP", "PASAT", "PVTB", "MSIT", "PVSAT"]


def _fila_vas(centro, medida, codigo, fitbit, genero, vas, fecha="2025-01-09 09:00:00"):
    return {
        "Marca temporal": fecha, "N_CENTRO": centro, "MEDIDA": medida,
        "CÓDIGO": codigo, "FITBIT": fitbit, "GÉNERO": genero,
        "VAS_100mm_preFE": vas,
    }


def _vas_minimo():
    """VAS con una pulsera de un solo ocupante y otra con dos."""
    return pd.DataFrame([
        _fila_vas(1, 2, "AAA010101", 50, 1, 40, "2025-01-09 09:00:00"),
        _fila_vas(1, 3, "AAA010101", 50, 1, 60, "2025-02-09 09:00:00"),
        # Pulsera 60 de C1: ocupante 1 en M2, ocupante 2 (2060) en M3.
        _fila_vas(1, 2, "BBB020202", 60, 2, 10, "2025-01-09 09:00:00"),
        _fila_vas(1, 3, "CCC030303", 2060, 1, 90, "2025-02-09 09:00:00"),
    ])


def _grabaciones_manifest(centro, sujeto_local, medicion, fecha, condiciones=None,
                          sexo="XX"):
    filas = []
    for i, condicion in enumerate(condiciones or CONDICIONES):
        filas.append({
            "nombre_nuevo": f"sub-{centro}{sujeto_local}_ses-{medicion}_task-{condicion}_eeg.edf",
            "sujeto_id": f"{centro}_{sujeto_local}",
            "sujeto_local": sujeto_local,
            "medicion": medicion,
            "centro_efectivo": centro,
            "condicion_norm": condicion,
            "sexo_filename": sexo,
            "timestamp_iso": f"{fecha}T{9 + i:02d}:00:00+01:00",
        })
    return filas


def _manifest_minimo():
    filas = []
    filas += _grabaciones_manifest("C1", "F050", "M2", "2025-01-09")
    filas += _grabaciones_manifest("C1", "F050", "M3", "2025-02-09")
    filas += _grabaciones_manifest("C1", "F060", "M2", "2025-01-09")
    filas += _grabaciones_manifest("C1", "F060", "M3", "2025-02-09")
    return pd.DataFrame(filas)


def _splits_minimo(manifest):
    ids = sorted(set(manifest["centro_efectivo"] + "_" + manifest["sujeto_local"]))
    return pd.DataFrame({"sujeto_id": ids, "split": ["train"] * len(ids)})


# --------------------------------------------------------------------------
# normalizar_vas
# --------------------------------------------------------------------------

def test_normalizar_vas_deriva_clave_y_ocupante():
    v = normalizar_vas(_vas_minimo())
    fila = v[v["FITBIT"] == 2060].iloc[0]
    assert fila["ocupante"] == 2
    assert fila["fitbit_base"] == 60
    assert fila["persona_uid"] == "C1_F2060"
    fila1 = v[v["FITBIT"] == 60].iloc[0]
    assert fila1["ocupante"] == 1
    assert fila1["fitbit_base"] == 60
    # Los dos ocupantes de la misma pulsera no pueden compartir persona_uid:
    # es justo lo que hace que agrupar por persona sea correcto.
    assert fila["persona_uid"] != fila1["persona_uid"]


def test_normalizar_vas_rellena_a_tres_digitos():
    v = normalizar_vas(pd.DataFrame([_fila_vas(5, 2, "DDD040404", 7, 1, 20)]))
    assert v.iloc[0]["persona_uid"] == "C5_F007"


def test_normalizar_vas_traduce_genero_y_deja_el_cero_como_desconocido():
    v = normalizar_vas(pd.DataFrame([
        _fila_vas(1, 2, "AAA010101", 50, 1, 40),
        _fila_vas(1, 2, "BBB020202", 51, 2, 40),
        _fila_vas(1, 2, "CCC030303", 52, 0, 40),
    ]))
    assert list(v["sexo_vas"].iloc[:2]) == ["XX", "XY"]
    assert pd.isna(v["sexo_vas"].iloc[2])


def test_normalizar_vas_elimina_la_fila_duplicada_exacta():
    fila = _fila_vas(6, 3, "EEE050505", 26, 2, 60)
    v = normalizar_vas(pd.DataFrame([fila, dict(fila)]))
    assert len(v) == 1


def test_normalizar_vas_aplica_el_intercambio_f111_f115():
    base = [
        _fila_vas(2, m, "FFF060606", 111, 1, 40 + m) for m in range(1, 7)
    ] + [
        _fila_vas(2, m, "GGG070707", 115, 2, 10 + m) for m in range(1, 7)
    ]
    v = normalizar_vas(pd.DataFrame(base))
    # El valor que el CSV daba a la pulsera 111 tiene que acabar en la 115.
    m3_115 = v[(v["fitbit_base"] == 115) & (v["medicion"] == "M3")].iloc[0]
    assert m3_115["vas"] == 43
    m3_111 = v[(v["fitbit_base"] == 111) & (v["medicion"] == "M3")].iloc[0]
    assert m3_111["vas"] == 13


def test_normalizar_vas_falla_si_el_intercambio_no_cuadra():
    # Solo 2 filas por lado en vez de las 6 declaradas.
    base = [_fila_vas(2, m, "FFF060606", 111, 1, 40) for m in range(1, 3)]
    base += [_fila_vas(2, m, "GGG070707", 115, 2, 10) for m in range(1, 3)]
    with pytest.raises(ValueError, match="swap_f111_f115"):
        normalizar_vas(pd.DataFrame(base))


def test_normalizar_vas_avisa_de_ocupantes_que_chocan_pero_no_falla(capsys):
    v = normalizar_vas(pd.DataFrame([
        _fila_vas(8, 1, "HHH080808", 44, 2, 40),
        _fila_vas(8, 1, "JJJ090909", 2044, 2, 30),
    ]))
    assert len(v) == 2
    assert "dos ocupantes" in capsys.readouterr().out


# --------------------------------------------------------------------------
# corregir_manifest
# --------------------------------------------------------------------------

def _manifest_con_correccion():
    """Manifest con las 41 grabaciones mal archivadas y el PASAT de C3F103."""
    filas = []
    for sujeto in ("F125", "F128", "F130", "F135", "F136"):
        filas += _grabaciones_manifest("C3", sujeto, "M5", "2025-04-28")
    filas += _grabaciones_manifest("C3", "F131", "M5", "2025-04-28",
                                   condiciones=CONDICIONES[:6])
    filas += _grabaciones_manifest("C3", "F103", "M3", "2025-01-09",
                                   condiciones=["PASAT"])
    filas += _grabaciones_manifest("C2", "F103", "M3", "2025-01-09",
                                   condiciones=[c for c in CONDICIONES if c != "PASAT"])
    return pd.DataFrame(filas)


def test_corregir_manifest_reasigna_centro_y_sujeto_id():
    m = corregir_manifest(_manifest_con_correccion())
    movidas = m[m["corregida"]]
    assert len(movidas) == 42
    assert set(movidas["centro_efectivo"]) == {"C4", "C2"}
    f125 = m[m["sujeto_local"] == "F125"]
    assert set(f125["sujeto_id"]) == {"C4_F125"}
    assert set(f125["sesion_uid"]) == {"C4_F125_M5"}


def test_corregir_manifest_reune_las_siete_tareas_de_la_sesion():
    """El PASAT de 'C3_F103' completa la M3 de C2_F103, que tenía seis."""
    m = corregir_manifest(_manifest_con_correccion())
    sesion = m[m["sesion_uid"] == "C2_F103_M3"]
    assert len(sesion) == 7
    assert sorted(sesion["condicion_norm"]) == sorted(CONDICIONES)


def test_corregir_manifest_falla_si_el_recuento_no_cuadra():
    m = _manifest_con_correccion()
    m = m[m["sujeto_local"] != "F136"]  # ahora la primera corrección ve 34, no 41
    with pytest.raises(ValueError, match="c3_a_c4_m5"):
        corregir_manifest(m)


def test_corregir_manifest_es_idempotente():
    una = corregir_manifest(_manifest_con_correccion())
    dos = corregir_manifest(una)
    pd.testing.assert_frame_equal(una, dos)


def test_corregir_manifest_rechaza_sujeto_local_del_rango_de_ocupante_2():
    m = _manifest_minimo()
    m.loc[0, "sujeto_local"] = "F2060"
    with pytest.raises(ValueError, match="1000"):
        corregir_manifest(m)


# --------------------------------------------------------------------------
# construir_etiquetas
# --------------------------------------------------------------------------

def test_construir_etiquetas_da_a_cada_ocupante_su_momento():
    """
    El test central del módulo: `C1_F060` es un solo `sujeto_id` con dos
    personas. Unir por sujeto le daría a las dos el mismo VAS.
    """
    manifest = _manifest_minimo()
    etiquetas, meta = construir_etiquetas(_vas_minimo(), manifest,
                                          _splits_minimo(manifest))
    m2 = etiquetas[etiquetas["sesion_uid"] == "C1_F060_M2"].iloc[0]
    m3 = etiquetas[etiquetas["sesion_uid"] == "C1_F060_M3"].iloc[0]
    assert (m2["vas"], m2["persona_uid"], m2["ocupante"]) == (10, "C1_F060", 1)
    assert (m3["vas"], m3["persona_uid"], m3["ocupante"]) == (90, "C1_F2060", 2)
    assert meta["n_sujeto_id_con_dos_personas"] == 1
    assert meta["n_personas"] == 3


def test_construir_etiquetas_conserva_las_sesiones_sin_vas():
    manifest = _manifest_minimo()
    manifest = pd.concat([manifest, pd.DataFrame(
        _grabaciones_manifest("C1", "F050", "M4", "2025-03-09"))], ignore_index=True)
    etiquetas, meta = construir_etiquetas(_vas_minimo(), manifest,
                                          _splits_minimo(manifest))
    sin = etiquetas[etiquetas["sesion_uid"] == "C1_F050_M4"].iloc[0]
    assert not sin["etiquetada"]
    assert pd.isna(sin["vas"])
    # Sin VAS no se sabe quién llevaba la pulsera, así que persona_uid queda a
    # NaN en vez de heredar el sujeto_id: en una pulsera con dos ocupantes eso
    # atribuiría la sesión a la persona equivocada.
    assert pd.isna(sin["persona_uid"])
    assert meta["sesiones_sin_vas"] == ["C1_F050_M4"]


def test_construir_etiquetas_marca_la_concordancia_de_fecha():
    manifest = _manifest_minimo()
    vas = _vas_minimo()
    vas.loc[0, "Marca temporal"] = "2025-01-11 09:00:00"   # dos días después
    vas.loc[1, "Marca temporal"] = None                     # sin marca
    etiquetas, meta = construir_etiquetas(vas, manifest, _splits_minimo(manifest))
    por_sesion = etiquetas.set_index("sesion_uid")["concordancia_fecha"]
    assert por_sesion["C1_F050_M2"] == "discrepa"
    assert por_sesion["C1_F050_M3"] == "desconocida"
    assert por_sesion["C1_F060_M2"] == "coincide"
    assert meta["sesiones_fecha_discrepante"] == ["C1_F050_M2"]


def test_construir_etiquetas_detecta_fuga_entre_splits():
    """
    Reproduce el fallo real que motivó `CORRECCIONES_MANIFEST`: una persona con
    grabaciones archivadas bajo dos centros acaba con dos `sujeto_id`, y si
    esos dos caen en splits distintos hay fuga entre train y test. El módulo
    tiene que parar en vez de escribir un dataset silenciosamente inservible.
    """
    # Se usa una pulsera fuera de CORRECCIONES_MANIFEST para que lo que falle
    # sea la comprobación de fuga y no el recuento de una corrección conocida.
    manifest = pd.DataFrame(
        _grabaciones_manifest("C5", "F129", "M3", "2025-01-15")
        + _grabaciones_manifest("C6", "F129", "M5", "2025-04-28"))
    vas = pd.DataFrame([
        _fila_vas(5, 3, "KKK101010", 129, 2, 30, "2025-01-15 08:00:00"),
        _fila_vas(6, 5, "KKK101010", 129, 2, 40, "2025-04-28 08:00:00"),
    ])
    splits = pd.DataFrame([{"sujeto_id": "C5_F129", "split": "val"},
                           {"sujeto_id": "C6_F129", "split": "train"}])
    with pytest.raises(ValueError, match="más de un split"):
        construir_etiquetas(vas, manifest, splits)


def test_construir_etiquetas_no_devuelve_el_codigo_del_participante():
    """
    `CÓDIGO` son iniciales y fecha de nacimiento de un menor. Se usa dentro de
    `construir_etiquetas` para detectar fugas, pero no puede salir en ninguna
    de las tablas ni acabar escrito en `labels.csv`.
    """
    manifest = _manifest_minimo()
    etiquetas, meta = construir_etiquetas(_vas_minimo(), manifest,
                                          _splits_minimo(manifest))
    grabaciones = expandir_a_grabaciones(etiquetas, manifest)
    unidas = unir_etiquetas(_features_de(manifest), grabaciones)
    for tabla in (etiquetas, grabaciones, unidas):
        assert not [c for c in tabla.columns if "DIGO" in c.upper()]
    codigos = set(_vas_minimo()["CÓDIGO"])
    for tabla in (etiquetas, grabaciones):
        planos = set(tabla.astype(str).values.ravel())
        assert not (codigos & planos)
    assert "CÓDIGO" not in str(meta)


def test_construir_etiquetas_falla_si_dos_personas_comparten_clave_con_eeg():
    manifest = pd.DataFrame(_grabaciones_manifest("C8", "F044", "M2", "2025-01-09"))
    vas = pd.DataFrame([
        _fila_vas(8, 2, "HHH080808", 44, 2, 40, "2025-01-09 09:00:00"),
        _fila_vas(8, 2, "JJJ090909", 2044, 2, 30, "2025-01-09 09:00:00"),
    ])
    with pytest.raises(ValueError, match="ambiguo"):
        construir_etiquetas(vas, manifest, _splits_minimo(manifest))


def test_construir_etiquetas_cuenta_ventanas_del_log():
    manifest = _manifest_minimo()
    log = pd.DataFrame({
        "archivo": manifest["nombre_nuevo"],
        "estado": ["ok"] * (len(manifest) - 1) + ["error"],
        "n_epocas": [10] * len(manifest),
    })
    etiquetas, meta = construir_etiquetas(_vas_minimo(), manifest,
                                          _splits_minimo(manifest), log=log)
    # 28 grabaciones, una falla: 27 x 10 ventanas.
    assert int(etiquetas["n_ventanas"].sum()) == 270
    assert meta["n_ventanas_etiquetadas"] == 270


# --------------------------------------------------------------------------
# expandir_a_grabaciones
# --------------------------------------------------------------------------

def test_expandir_replica_la_etiqueta_sin_cambiar_el_numero_de_filas():
    manifest = _manifest_minimo()
    etiquetas, _ = construir_etiquetas(_vas_minimo(), manifest,
                                       _splits_minimo(manifest))
    grabaciones = expandir_a_grabaciones(etiquetas, manifest)
    assert len(grabaciones) == len(manifest)
    sesion = grabaciones[grabaciones["sesion_uid"] == "C1_F060_M3"]
    assert len(sesion) == 7
    assert set(sesion["vas"]) == {90}
    assert set(sesion["persona_uid"]) == {"C1_F2060"}


# --------------------------------------------------------------------------
# unir_etiquetas
# --------------------------------------------------------------------------

def _features_de(manifest, n_epocas=3):
    """Matriz de features como la que deja `procesar_lote`: con el `sujeto` y
    el `split` de ANTES de corregir el manifest."""
    filas = []
    for fila in manifest.to_dict("records"):
        for i in range(n_epocas):
            filas.append({
                "sujeto": fila["sujeto_id"], "condicion": fila["condicion_norm"],
                "sesion": fila["medicion"], "archivo_origen": fila["nombre_nuevo"],
                "indice_ventana": i, "centro": fila["centro_efectivo"],
                "split": "train", "pot_abs_AF3_theta": 1.0 * i,
            })
    return pd.DataFrame(filas)


def test_unir_etiquetas_llega_a_las_grabaciones_mal_archivadas():
    """
    Regresión del error más fácil de cometer aquí: la matriz de features lleva
    `sujeto='C3_F125'`, que ya no existe tras la corrección. Unir por sujeto
    dejaría estas épocas sin etiqueta; unir por archivo funciona.
    """
    manifest = _manifest_con_correccion()
    corregido = corregir_manifest(manifest)
    vas = pd.DataFrame([
        _fila_vas(4, 5, "LLL111111", 125, 2, 40, "2025-04-28 08:00:00"),
    ])
    splits = pd.DataFrame([{"sujeto_id": "C4_F125", "split": "val"}])
    etiquetas, _ = construir_etiquetas(vas, corregido, splits)
    grabaciones = expandir_a_grabaciones(etiquetas, corregido)

    features = _features_de(manifest)   # con el sujeto_id SIN corregir
    unidas = unir_etiquetas(features, grabaciones)

    de_f125 = unidas[unidas["archivo_origen"].str.contains("C3F125")]
    assert len(de_f125) == 21
    assert set(de_f125["vas"]) == {40}
    assert set(de_f125["sujeto_id"]) == {"C4_F125"}
    assert set(de_f125["split"]) == {"val"}          # ya no el 'train' original
    assert set(de_f125["persona_uid"]) == {"C4_F125"}


def test_unir_etiquetas_no_cambia_el_numero_de_epocas():
    manifest = _manifest_minimo()
    etiquetas, _ = construir_etiquetas(_vas_minimo(), manifest,
                                       _splits_minimo(manifest))
    grabaciones = expandir_a_grabaciones(etiquetas, manifest)
    features = _features_de(manifest)
    unidas = unir_etiquetas(features, grabaciones)
    assert len(unidas) == len(features)
    assert unidas["vas"].notna().all()


def test_unir_etiquetas_falla_si_falta_un_archivo():
    manifest = _manifest_minimo()
    etiquetas, _ = construir_etiquetas(_vas_minimo(), manifest,
                                       _splits_minimo(manifest))
    grabaciones = expandir_a_grabaciones(etiquetas, manifest)
    features = _features_de(manifest)
    features.loc[0, "archivo_origen"] = "sub-C9F999_ses-M9_task-OC_eeg.edf"
    with pytest.raises(ValueError, match="no aparecen"):
        unir_etiquetas(features, grabaciones)


def test_unir_etiquetas_exige_archivo_origen():
    manifest = _manifest_minimo()
    etiquetas, _ = construir_etiquetas(_vas_minimo(), manifest,
                                       _splits_minimo(manifest))
    grabaciones = expandir_a_grabaciones(etiquetas, manifest)
    with pytest.raises(ValueError, match="archivo_origen"):
        unir_etiquetas(pd.DataFrame({"sujeto": ["C1_F050"]}), grabaciones)


# --------------------------------------------------------------------------
# Trazabilidad de las correcciones
# --------------------------------------------------------------------------

def test_cada_correccion_documenta_su_evidencia():
    """
    No es un test de comportamiento: fija que ninguna corrección pueda entrar
    sin dejar escrito por qué, que es lo que la hace defendible en la memoria.
    """
    for correccion in CORRECCIONES_MANIFEST + CORRECCIONES_VAS:
        assert correccion["evidencia"].strip()
        assert correccion["n_esperado"] > 0
        assert correccion["nombre"]
