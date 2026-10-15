"""
Parámetros de procesado que puede elegir quien usa la librería.

Son **opciones cerradas**, no valores libres: cada combinación que se admite
es una que se puede declarar en la ficha de un modelo y comparar después con
la que se aplicó a una grabación. Un valor libre (p. ej. un paso banda de
1.3-27 Hz) no coincidiría nunca con el de ningún modelo y solo serviría para
producir resultados que nadie ha validado.

- `referencia`:
  - `car_condicional` (la del estudio): CAR sobre los canales buenos si hay
    al menos 6; con menos se omite (ver `aplicar_car`).
  - `original`: se conserva la referencia de hardware del Emotiv (CMS/DRL),
    sin re-referenciar.
- `banda_hz`: paso banda FIR, `[1, 30]` (la del estudio) o `[3, 30]`.
- `notch_hz`: `50` (la del estudio) o `None` (sin notch).

`PARAMETROS_ESTUDIO` son los valores con los que se construyó
`matriz_rf.parquet` y se entrenaron los modelos.
"""

PARAMETROS_ESTUDIO = {
    "referencia": "car_condicional",
    "banda_hz": [1.0, 30.0],
    "notch_hz": 50.0,
}

OPCIONES = {
    "referencia": ("car_condicional", "original"),
    "banda_hz": ([1.0, 30.0], [3.0, 30.0]),
    "notch_hz": (50.0, None),
}


def _normalizar_valor(clave, valor):
    """Lleva un valor a su forma canónica (floats y listas) para compararlo."""
    if clave == "banda_hz":
        try:
            return [float(v) for v in valor]
        except TypeError:
            return valor
    if clave == "notch_hz":
        return None if valor is None else float(valor)
    return valor


def normalizar_parametros(parametros=None):
    """
    Completa y valida un diccionario de parámetros de procesado.

    Las claves que falten toman el valor de `PARAMETROS_ESTUDIO`. Los valores
    se llevan a su forma canónica (`banda_hz` como lista de dos floats,
    `notch_hz` como float o None), de modo que dos diccionarios equivalentes
    salgan iguales al compararlos.

    Parámetros
    ----------
    parametros : dict o None
        Parámetros pedidos. None equivale a `PARAMETROS_ESTUDIO`.

    Devuelve
    --------
    dict
        Copia completa y canónica, con las tres claves de `OPCIONES`.

    Raises
    ------
    ValueError
        Si llega una clave desconocida o un valor que no está entre las
        opciones admitidas.
    """
    parametros = dict(parametros or {})
    desconocidas = sorted(set(parametros) - set(OPCIONES))
    if desconocidas:
        raise ValueError(
            f"Parámetros desconocidos: {desconocidas}. Se admiten: {sorted(OPCIONES)}."
        )

    normalizados = {}
    for clave, opciones in OPCIONES.items():
        valor = _normalizar_valor(clave, parametros.get(clave, PARAMETROS_ESTUDIO[clave]))
        if valor not in opciones:
            raise ValueError(
                f"Valor no admitido para '{clave}': {valor!r}. "
                f"Opciones: {list(opciones)}."
            )
        normalizados[clave] = list(valor) if isinstance(valor, list) else valor
    return normalizados
