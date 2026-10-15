"""
Módulo de eliminación de artefactos oculares mediante ICA (Fase 6).

Se aplica sobre el Raw ya recortado (Fase 2), filtrado (Fase 3), evaluado
en calidad (Fase 4) y re-referenciado con CAR (Fase 5). Expone tres
funciones que se usan encadenadas:

  1. entrenar_ica                    — ajusta ICA (Infomax Extended).
  2. identificar_componentes_oculares — clasifica componentes con ICLabel
                                         y selecciona solo las oculares.
  3. eliminar_artefactos_oculares      — reconstruye la señal sin esas
                                         componentes.

Alcance reducido a componentes oculares (parpadeos y movimientos oculares):
el Emotiv Epoc X no tiene canales EOG/ECG dedicados y el presupuesto del
TFG es de 300h. Las demás categorías de ICLabel (muscle artifact, heart
beat, line noise, channel noise, other) se detectan y documentan a modo
de diagnóstico, pero nunca se excluyen — ver identificar_componentes_oculares.
"""

import datetime
import time

import mne
from mne_icalabel.iclabel import iclabel_label_components

# Orden de columnas de la matriz de probabilidades devuelta por ICLabel
# (documentado en mne_icalabel.iclabel.iclabel_label_components).
CATEGORIAS_ICLABEL = (
    "brain",
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
    "other",
)

_INDICE_EYE_BLINK = CATEGORIAS_ICLABEL.index("eye blink")


def entrenar_ica(raw, good_channels, random_state=42, n_components=None):
    """
    Ajusta ICA (Infomax Extended) sobre los canales buenos del registro.

    Se ajusta únicamente sobre `good_channels` —no sobre todos los canales
    EEG del Raw— porque aplicar_car (Fase 5) no elimina del objeto Raw los
    canales excluidos por evaluar_calidad, solo los deja fuera del promedio.
    Incluirlos aquí contaminaría la descomposición y rompería la relación
    entre `n_components` y la deficiencia de rango real introducida por CAR.

    n_components = len(good_channels) - 1: CAR resta el promedio de esos
    mismos canales a sí mismos, introduciendo una dependencia lineal entre
    ellos (rango = n_canales - 1). Pedir más componentes de las que el
    rango permite genera componentes espurias ("ghost ICs") sin
    interpretación física.

    Nota: esta resta asume que CAR se aplicó realmente (aplicar_car con
    >= 6 canales buenos). Si CAR se omitió (< 6 canales buenos), no hay
    deficiencia de rango inducida por CAR y `-1` es conservador pero no
    estrictamente necesario; se documenta como limitación conocida. Se
    mantiene así porque es como se procesó el dataset del estudio.

    Cuando no se re-referencia en absoluto (referencia `original`, ver
    `procesado/parametros.py`), quien llama pasa `n_components` igual al
    número de canales buenos: sin CAR no hay pérdida de rango.

    Parámetros
    ----------
    raw : mne.io.BaseRaw
        Raw recortado, filtrado, evaluado en calidad y con CAR aplicado
        (Fases 2-5), con preload=True y montaje asignado (ver cargar_eeg).
    good_channels : list[str]
        Canales buenos devueltos por evaluar_calidad / usados en aplicar_car.
    random_state : int
        Semilla para reproducibilidad (Infomax es estocástico). Por
        defecto 42.
    n_components : int o None
        Número de componentes. None (por defecto) usa
        `len(good_channels) - 1`, el valor del estudio.

    Devuelve
    --------
    ica : mne.preprocessing.ICA
        Objeto ICA ajustado.
    metadata : dict
        Diccionario con las claves:
          - metodo         : "infomax_extended"
          - n_components   : nº de componentes ajustadas
          - n_good_channels: nº de canales buenos usados
          - canales_usados : list[str] con los nombres de esos canales
          - tiempo_ajuste_s: duración del ajuste en segundos
          - random_state   : semilla usada
          - timestamp      : ISO 8601 del momento de aplicación
    """
    if len(good_channels) < 2:
        raise ValueError(
            f"Se requieren al menos 2 canales buenos para ajustar ICA "
            f"(recibidos: {len(good_channels)})."
        )

    faltantes = [ch for ch in good_channels if ch not in raw.ch_names]
    if faltantes:
        raise ValueError(
            f"Los siguientes canales de good_channels no están en el Raw: "
            f"{faltantes}."
        )

    if n_components is None:
        n_components = len(good_channels) - 1
    if not 1 <= n_components <= len(good_channels):
        raise ValueError(
            f"n_components={n_components} fuera de rango para "
            f"{len(good_channels)} canales buenos."
        )

    raw_picked = raw.copy().pick(good_channels)

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="infomax",
        fit_params={"extended": True},
        random_state=random_state,
    )

    inicio = time.perf_counter()
    ica.fit(raw_picked, verbose=False)
    tiempo_ajuste = time.perf_counter() - inicio

    metadata = {
        "metodo": "infomax_extended",
        "n_components": n_components,
        "n_good_channels": len(good_channels),
        "canales_usados": list(good_channels),
        "tiempo_ajuste_s": tiempo_ajuste,
        "random_state": random_state,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    print(
        f"ICA ajustada: {n_components} componentes sobre "
        f"{len(good_channels)} canales buenos ({tiempo_ajuste:.1f} s)."
    )

    return ica, metadata


def identificar_componentes_oculares(raw, ica, umbral_prob=0.8):
    """
    Clasifica las componentes de ICA con ICLabel y selecciona las oculares.

    Usa la red de ICLabel (mne_icalabel.iclabel.iclabel_label_components)
    para clasificar cada componente en una de 7 categorías estándar: brain,
    muscle artifact, eye blink, heart beat, line noise, channel noise, other.

    Solo se seleccionan como `componentes_a_excluir` las etiquetadas
    'eye blink' con probabilidad >= umbral_prob. Las demás categorías nunca
    se excluyen (fuera de alcance del TFG: el Epoc X no tiene canales
    EOG/ECG dedicados para identificarlas con fiabilidad), pero como las
    probabilidades ya se calculan igualmente, se registran en
    `diagnostico_otras_categorias` como dato gratuito para justificar en la
    memoria por qué se detectaron y no se trataron.

    Parámetros
    ----------
    raw : mne.io.BaseRaw
        Mismo Raw (o compatible) usado para entrenar_ica, con montaje
        asignado (necesario para las topografías de ICLabel).
    ica : mne.preprocessing.ICA
        Objeto ICA ya ajustado por entrenar_ica.
    umbral_prob : float
        Probabilidad mínima para considerar una componente perteneciente
        a una categoría. Por defecto 0.8.

    Devuelve
    --------
    componentes_a_excluir : list[int]
        Índices de componentes clasificadas como 'eye blink' con
        probabilidad >= umbral_prob.
    diagnostico_otras_categorias : dict[str, list[dict]]
        Para cada categoría distinta de 'eye blink' con al menos una
        componente >= umbral_prob: lista de {"componente": int,
        "probabilidad": float}.
    proba_matrix : numpy.ndarray de forma (n_components, 7)
        Matriz completa de probabilidades por componente y categoría,
        columnas en el orden de CATEGORIAS_ICLABEL.
    """
    proba_matrix = iclabel_label_components(raw, ica)

    componentes_a_excluir = [
        idx
        for idx in range(proba_matrix.shape[0])
        if proba_matrix[idx, _INDICE_EYE_BLINK] >= umbral_prob
    ]

    diagnostico_otras_categorias = {}
    for idx in range(proba_matrix.shape[0]):
        for j, categoria in enumerate(CATEGORIAS_ICLABEL):
            if j == _INDICE_EYE_BLINK:
                continue
            probabilidad = float(proba_matrix[idx, j])
            if probabilidad >= umbral_prob:
                diagnostico_otras_categorias.setdefault(categoria, []).append(
                    {"componente": idx, "probabilidad": probabilidad}
                )

    print(
        f"ICLabel: {len(componentes_a_excluir)} componente(s) ocular(es) "
        f"detectada(s) (umbral={umbral_prob}). "
        f"Otras categorías con probabilidad >= umbral: "
        f"{ {k: len(v) for k, v in diagnostico_otras_categorias.items()} }."
    )

    return componentes_a_excluir, diagnostico_otras_categorias, proba_matrix


def eliminar_artefactos_oculares(raw, ica, componentes_a_excluir):
    """
    Reconstruye la señal excluyendo las componentes oculares indicadas.

    Trabaja sobre una copia de `ica` (ica.exclude no se modifica en el
    objeto recibido) y sobre una copia de `raw`, para no mutar ninguno
    de los dos objetos de entrada.

    Parámetros
    ----------
    raw : mne.io.BaseRaw
        Raw sobre el que reconstruir la señal limpia.
    ica : mne.preprocessing.ICA
        Objeto ICA ya ajustado.
    componentes_a_excluir : list[int]
        Índices de componentes a eliminar (típicamente el resultado de
        identificar_componentes_oculares).

    Devuelve
    --------
    raw_limpio : mne.io.BaseRaw
        Copia del Raw con las componentes indicadas eliminadas.
    metadata : dict
        Diccionario con las claves:
          - n_componentes_eliminadas   : int
          - componentes_eliminadas    : list[int]
          - varianza_explicada_eliminada: float | None (None si no se
            elimina ninguna componente)
          - timestamp                 : ISO 8601 del momento de aplicación
    """
    ica_copia = ica.copy()
    ica_copia.exclude = list(componentes_a_excluir)

    varianza_eliminada = None
    if componentes_a_excluir:
        ratio = ica_copia.get_explained_variance_ratio(
            raw, components=componentes_a_excluir, ch_type="eeg"
        )
        varianza_eliminada = ratio.get("eeg")

    raw_limpio = raw.copy()
    ica_copia.apply(raw_limpio, verbose=False)

    metadata = {
        "n_componentes_eliminadas": len(componentes_a_excluir),
        "componentes_eliminadas": list(componentes_a_excluir),
        "varianza_explicada_eliminada": varianza_eliminada,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    print(
        f"Artefactos oculares eliminados: {len(componentes_a_excluir)} "
        f"componente(s)."
    )

    return raw_limpio, metadata
