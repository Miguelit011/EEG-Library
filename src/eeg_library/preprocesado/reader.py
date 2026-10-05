"""
Módulo de lectura de archivos EEG.

Proporciona funciones para cargar señales EEG desde archivos EDF
registrados con el casco Emotiv Epoc X.
"""

import datetime
import os

import mne
import numpy as np

from eeg_library.preprocesado.quality import CANALES_EPOC, buscar_canal_cq

TARGET_SFREQ = 256
"""Frecuencia de muestreo objetivo (Hz) a la que se armoniza todo el
dataset. La gran mayoría de registros ya se graban a esta frecuencia, pero
una fracción minoritaria del dataset final (2.2 %, 100 de 4574 grabaciones
válidas — ver ESTADO.md) se grabó a 128 Hz: son todas de la medición M2 y
de solo dos centros (C1 y C8), lo que apunta a un lote grabado con otra
configuración de EmotivPro y no a un fallo puntual. El Nyquist de 128 Hz
(64 Hz) sigue muy por encima de la banda 1-30 Hz que usa el filtro de la
Fase 3, así que esas grabaciones no tienen menos información útil real para
este análisis: el resampleo es puramente para que todas las señales
compartan el mismo número de muestras por segundo antes del epoching
(Fase 7) y del Welch (Fase 8), no para "recuperar" contenido de alta
frecuencia que nunca existió."""

MIN_SFREQ = 128
"""Frecuencia de muestreo mínima (Hz) aceptada para un registro de origen.
Por debajo de este umbral se rechaza el archivo en vez de resamplearlo:
"subir" de, p. ej., 100 Hz a TARGET_SFREQ no añade información real y
generaría una falsa sensación de calidad uniforme en todo el dataset — el
resampleo ascendente solo está justificado para el caso conocido y
documentado de 128 Hz (Nyquist a 64 Hz, muy por encima de la banda 1-30 Hz
de interés), no como norma general para cualquier frecuencia inferior."""

_CQ_VALOR_MINIMO_SIN_ESCALAR = 1.0
"""Valor por debajo del cual se considera que los canales CQ llegaron
escalados por MNE y hay que revertir la escala (ver `cargar_eeg`). La escala
CQ del Emotiv es discreta con paso 1 en el rango [0, 4], así que cualquier
valor no nulo que no haya sido escalado es >= 1: un máximo por debajo de
este umbral solo puede venir de un archivo escalado por el factor de voltaje
(los valores llegarían como ~4e-6) o de un CQ idénticamente nulo, caso en el
que revertir la escala es inocuo."""


def cargar_eeg(ruta: str) -> tuple[mne.io.BaseRaw, dict]:
    """
    Carga un archivo EDF y devuelve el objeto Raw de MNE.

    Realiza una validación básica de la frecuencia de muestreo, armoniza
    esa frecuencia a TARGET_SFREQ si hace falta, e imprime un resumen del
    archivo sin aplicar ningún otro procesamiento a los datos.

    Parámetros
    ----------
    ruta : str
        Ruta al archivo EDF que se desea cargar.

    Devuelve
    --------
    raw : mne.io.BaseRaw
        Objeto Raw de MNE con los datos EEG cargados en memoria, ya a
        TARGET_SFREQ Hz (resampleado si el archivo original no lo estaba).
        Solo los canales de CANALES_EPOC quedan tipados como 'eeg'; el resto
        (CQ, EQ, telemetría, marcadores, movimiento) se tipa como 'misc',
        para que las fases que seleccionan picks="eeg" operen únicamente
        sobre los 14 electrodos reales.
    metadata : dict
        Diccionario con las claves:
          - metodo         : "carga_edf"
          - archivo        : nombre del archivo cargado
          - sfreq          : frecuencia de muestreo final (Hz, == TARGET_SFREQ)
          - original_sfreq : frecuencia de muestreo del archivo original (Hz)
          - resampled      : True si se aplicó resampleo a TARGET_SFREQ
          - duracion_s     : duración del registro en segundos
          - n_canales      : número total de canales del archivo
          - n_canales_eeg  : número de canales tipados como 'eeg' (los de
                             CANALES_EPOC presentes en el archivo)
          - timestamp      : ISO 8601 del momento de carga

    Raises
    ------
    FileNotFoundError
        Si el archivo indicado en ``ruta`` no existe.
    ValueError
        Si la frecuencia de muestreo original del archivo es inferior a
        MIN_SFREQ (128 Hz): resamplear al alza desde una calidad tan baja
        no recupera información real y se rechaza en vez de armonizarse.
    """

    if not os.path.isfile(ruta):
        raise FileNotFoundError(
            f"No se encontró el archivo EEG: '{ruta}'. "
            "Comprueba que la ruta sea correcta."
        )

    # preload = True -> Carga los datos en la memoria RAM para permitir el filtrado y procesamiento posterior.
    # verbose = False -> Desactiva los mensajes automáticos en la terminal para una ejecución más limpia.
    raw = mne.io.read_raw_edf(ruta, preload=True, verbose=False)

    # Armonización de frecuencia de muestreo — debe ser el primer paso tras
    # la carga, antes de cualquier otra transformación. La Fase 3 diseña el
    # filtro en función de sfreq, la Fase 7 convierte "ventana de N s" a
    # muestras usando sfreq, y la Fase 8 calcula la resolución espectral a
    # partir de sfreq: si esta armonización no queda garantizada aquí, en
    # el punto de entrada, cualquier archivo grabado a una frecuencia
    # distinta de TARGET_SFREQ puede romper esas fases silenciosamente.
    #
    # Por debajo de MIN_SFREQ se rechaza el archivo en vez de resamplearlo
    # al alza: a diferencia del caso de 128 Hz (documentado en MIN_SFREQ),
    # una frecuencia inferior no garantiza el margen de Nyquist necesario
    # para la banda 1-30 Hz de interés, y "estirar" esos datos a
    # TARGET_SFREQ generaría una falsa sensación de calidad uniforme sin
    # aportar información real.
    original_sfreq = float(raw.info["sfreq"])
    if original_sfreq < MIN_SFREQ:
        raise ValueError(
            f"Frecuencia de muestreo original insuficiente: "
            f"{original_sfreq:g} Hz. Se requieren al menos {MIN_SFREQ:g} Hz "
            "para garantizar una calidad mínima; no se resamplea al alza "
            "por debajo de ese umbral."
        )

    # Solo se resamplea si hace falta, para no reprocesar innecesariamente
    # el ~98 % de archivos que ya están a TARGET_SFREQ. raw.resample()
    # aplica anti-aliasing por FFT internamente (no es una interpolación
    # lineal ingenua); npad="auto" elige el relleno que minimiza artefactos
    # de borde. La transformación es solo en memoria: no se escribe ni
    # modifica ningún .edf en disco.
    resampled = original_sfreq != TARGET_SFREQ
    if resampled:
        raw.resample(TARGET_SFREQ, npad="auto", verbose=False)

    # MNE tipa todos los canales del EDF como 'eeg' por defecto, incluidos los
    # que no son señal cerebral (calidad de contacto, calidad de señal,
    # telemetría, marcadores, acelerómetros y magnetómetros). Sin corregir
    # esto, todas las fases que seleccionan picks="eeg" operarían también
    # sobre ellos: aplicar_filtros (Fase 3) filtraría los CQ_* —una señal
    # casi constante que un bandpass 1-30 Hz colapsa a ~0, invalidando los
    # criterios de calidad— y calcular_psd (Fase 8) calcularía el espectro de
    # los 71 canales, con lo que extraer_bandas (Fase 9) devolvía 568
    # columnas de potencia (con features sin sentido como
    # pot_abs_MOT.AccX_theta) en vez de las 112 de los 14 canales reales.
    #
    # Se decide por *allowlist* y no enumerando los nombres no-EEG a excluir:
    # EmotivPro exporta el mismo casco con convenciones distintas según su
    # versión —"CQ_AF3"/"RAW_CQ"/"TIME_STAMP_s" en el export de 60 canales
    # frente a "CQ.AF3"/"RawCq"/"TimestampS" en los de 67 y 71— y una lista
    # de exclusiones por nombre falla en silencio en cuanto aparece una
    # variante nueva: deja canales de telemetría tipados como EEG sin que
    # nada avise. Los 14 nombres de CANALES_EPOC, en cambio, son idénticos en
    # todos los exports porque provienen del sistema 10-20.
    # on_unit_change="ignore": pasar de 'eeg' a 'misc' cambia la unidad del
    # canal de voltios a adimensional y MNE avisa de ello por cada canal. Es
    # exactamente el cambio que se busca —telemetría y CQ no son voltajes— y
    # el aviso se repetiría en cada uno de los miles de archivos del lote
    # (Fase 11), así que se silencia aquí a propósito.
    canales_no_eeg = [ch for ch in raw.ch_names if ch not in CANALES_EPOC]
    if canales_no_eeg:
        raw.set_channel_types(
            {ch: "misc" for ch in canales_no_eeg},
            on_unit_change="ignore",
            verbose=False,
        )

    # Los canales CQ llegan en una escala u otra según cómo declare el EDF su
    # unidad física: si la declara como voltaje ("µV", export de 60 canales),
    # MNE les aplica el factor de escala estándar y los valores originales del
    # rango [0, 4] llegan como ~4e-6; si la declara adimensional ("n/a",
    # exports de 67 y 71 canales), MNE no los toca y llegan ya correctos.
    #
    # Por eso el rescalado se decide por el rango real de los valores y no por
    # el nombre del canal ni por la versión del export: aplicarlo siempre
    # convertiría un CQ correcto de 4.0 en 4e6, con lo que el criterio "CQ
    # medio < 3.0" de evaluar_calidad no volvería a excluir un canal nunca.
    canales_cq = []
    for canal in CANALES_EPOC:
        nombre_cq = buscar_canal_cq(canal, raw.ch_names)
        if nombre_cq is not None:
            canales_cq.append(nombre_cq)

    if canales_cq:
        indices_cq = [raw.ch_names.index(ch) for ch in canales_cq]
        cq_ya_escalados = (
            float(np.abs(raw._data[indices_cq]).max()) < _CQ_VALOR_MINIMO_SIN_ESCALAR
        )
        if cq_ya_escalados:
            raw._data[indices_cq] *= 1e6

    # Asigna las posiciones de electrodo del sistema 10-20 estándar. Los 14
    # canales EEG del Epoc X usan nombres 10-20 y quedan georreferenciados;
    # los canales no-EEG no tienen posición estándar y se omiten con
    # on_missing="warn" (aviso, no error). Necesario para la Fase 6 (ICLabel
    # calcula topografías de scalp a partir de estas posiciones).
    # "colin27_1020" es el nombre que MNE 1.13 da al antiguo "standard_1020"
    # (que deja de existir en MNE 1.14); las posiciones son idénticas, así que
    # los datos procesados con el nombre anterior no cambian.
    raw.set_montage("colin27_1020", on_missing="warn", verbose=False)

    # Red de seguridad: tras la armonización anterior, frecuencia_muestreo
    # debería ser siempre exactamente TARGET_SFREQ. Si no lo es, algo falló
    # en raw.resample() (o en la comprobación de MIN_SFREQ de más arriba) y
    # es preferible abortar aquí a propagar un sfreq inconsistente a las
    # fases posteriores.
    frecuencia_muestreo = raw.info["sfreq"]
    if frecuencia_muestreo < TARGET_SFREQ:
        raise ValueError(
            f"Frecuencia de muestreo final insuficiente tras la armonización: "
            f"{frecuencia_muestreo} Hz. Se esperaban {TARGET_SFREQ} Hz."
        )

    nombre_archivo = os.path.basename(ruta)
    duracion_segundos = raw.times[-1]
    canales = raw.ch_names
    canales_eeg = [ch for ch in canales if ch in CANALES_EPOC]

    print(f"Archivo       : {nombre_archivo}")
    print(f"Frec. muestreo: {frecuencia_muestreo} Hz")
    if resampled:
        print(f"Resampleado   : {original_sfreq:g} Hz -> {TARGET_SFREQ:g} Hz")
    print(f"Duración      : {duracion_segundos:.2f} s")
    print(
        f"Canales EEG ({len(canales_eeg)}): {canales_eeg} "
        f"[+{len(canales) - len(canales_eeg)} no-EEG como 'misc']"
    )

    metadata = {
        "metodo": "carga_edf",
        "archivo": nombre_archivo,
        "sfreq": float(frecuencia_muestreo),
        "original_sfreq": original_sfreq,
        "resampled": resampled,
        "duracion_s": float(duracion_segundos),
        "n_canales": len(canales),
        "n_canales_eeg": len(canales_eeg),
        "timestamp": datetime.datetime.now().isoformat(),
    }

    return raw, metadata
