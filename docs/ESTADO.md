# Estado de implementación — EEG-Library

> Documento de uso interno. No se sube al repositorio.
> Última actualización: 2026-08-23 | Versión librería: 0.9.0

> **Este documento se congela en la Fase 13** (matriz de features del Random
> Forest). Cubre el pipeline de señal y la construcción de la matriz — lo que
> ya está cerrado e implementado como librería. El entrenamiento del modelo,
> su evaluación y todo lo posterior (Fase 14 en adelante, incluida la ruta
> CNN) vive en [`MODELADO.md`](MODELADO.md), para no volver a mezclar
> documentación de librería (estable) con cuaderno de análisis (que cambia
> cada sesión). Se actualiza solo si cambia el propio pipeline de las
> Fases 1-13.

---

## El dataset final (medido el 2026-08-10)

Ubicación: `C:\Users\migue\Desktop\TFG\EEG-Data\` — fuera del repo y **fuera de OneDrive**. Estuvo en
`OneDrive\Escritorio\EEG-Data` hasta el 2026-08-10 y en `C:\EEG-Data\` hasta
el 2026-10-06, cuando se trasladó aquí tras formatear el equipo; los guiones
la localizan con la variable de entorno `EEG_DATA`. Que no
sincronice es un requisito, no una casualidad: las ventanas crudas para la CNN
que genera la Fase 11 ocupan 16.2 GB.

```
dataset_final/                4574 archivos EDF
metadata/manifest_canonical.csv   4574 filas x 37 columnas (ya filtrado a
                                  válidos y sin copias duplicadas)
metadata/splits.csv               178 filas (partición por sujeto)
processed/                        salida de la Fase 11 (features, log, resumen)
ventanas/                         salida de la Fase 11 (ventanas crudas CNN)
```

| Dimensión              | Valor real |
|------------------------|------------|
| Sujetos                | 178        |
| Grabaciones válidas    | 4574       |
| Mediciones             | 5 (M2–M6)  |
| Condiciones            | 7 (MSIT, OA, OC, PASAT, PVSAT, PVTB, STROOP) |
| Centros                | 8 (C1–C8)  |
| Señal bruta total      | 172.8 h    |
| Duración por grabación | 25–379 s (mediana 127 s) |
| Grabadas a 128 Hz      | 100 (2.2 %) — todas de M2, centros C1 y C8, 16 sujetos |

> Estas cifras **corrigen** las del "Contexto científico" de CLAUDE.md
> (160 sujetos, 960 archivos, 6 momentos, 5 condiciones, 30–37 h), que
> describían una estimación previa a la organización del dataset.

**Significado de las mediciones (M1–M6).** Información facilitada por el equipo
del estudio el 2026-10-21 (testimonio del equipo, no deducido de los archivos).
Las medidas se tomaron al inicio y al final de cada trimestre de un curso
escolar:

| Medición | Momento del curso            | En el dataset EEG |
|----------|------------------------------|-------------------|
| M1       | inicio del 1.er trimestre    | no (descartada, ver abajo) |
| M2       | final del 1.er trimestre     | sí |
| M3       | inicio del 2.º trimestre     | sí |
| M4       | final del 2.º trimestre      | sí |
| M5       | inicio del 3.er trimestre    | sí |
| M6       | final del 3.er trimestre     | sí |

Consecuencias: las mediciones son **ordinales y no equiespaciadas** (M2→M3 cubre
las vacaciones entre trimestres; M3→M4 un trimestre lectivo), de modo que no
conviene tratar `medicion` como variable numérica con paso constante. Los pares
(M2, M3), (M4, M5) son fin de un trimestre e inicio del siguiente, y por tanto
casi contiguos en el tiempo. M1 solo aparece en el VAS, no en las grabaciones
(ver "Regla de enlace", Fase 12).

**Por qué se descartó M1.** Decisión del autor tras la auditoría inicial de los
datos crudos de M1 (1014 grabaciones EDF, 15 076 archivos, ~30 GB). Los motivos
que recogía esa auditoría:

- **Protocolo distinto:** M1 solo tiene 5 condiciones (OC, OA, STROOP, PASAT,
  PVTB); MSIT y PVSAT no aparecen. M2–M6 tienen las 7.
- **Identidad sin resolver:** los `.edf` crudos no llevan sujeto/centro/sexo en
  el nombre (esquema crudo de EmotivPro: iniciales + serie + timestamp). De las
  1014 grabaciones, ~291 (29 %) no tenían entrada canónica F/C/sexo; en C8 el
  82 %. En C3 y C4 ocurría lo inverso: más entradas canónicas que EDF reales.
- **Colisión de identidad:** F144 aparecía en C2 y C7, y la pulsera 44 de C8 tenía
  dos ocupantes en M1 (la única colisión de clave `(centro, pulsera, medición)`
  del VAS). Con identidad de persona dudosa no se puede hacer un split por sujeto
  sin riesgo de fuga.
- **Contaminación:** grabaciones de apps demo de Emotiv (Cerebral Circles, Star
  Search, EmotivBCI-*) mezcladas con las experimentales en C5, y mucho ruido de
  naming (typos, tomas inválidas, `prueba`).

Estos son los hallazgos de la auditoría tal como se recogieron entonces; no se
han vuelto a verificar contra `EEG-Data`. Las preguntas que dejó abiertas (por
qué MSIT/PVSAT no están en M1, origen de los ~291 sin mapear) no consta que se
hayan respondido.

**Partición (`splits.csv`):** por sujeto y estratificada por centro, sin
solape — train 125 sujetos / val 25 / test 28. Su columna `n_ventanas`
(305 272 en total) **no descuenta el recorte de 5+5 s de la Fase 2**: el
epocado real de la Fase 7 produce **282 402** ventanas, exactamente 5 menos
por grabación. Usar la cifra calculada, no la de la columna.

**Orden del protocolo dentro de cada sesión (fijo):**
`OC → OA → STROOP → PASAT → PVTB → MSIT → PVSAT`, ~15.5 min por sesión;
622 de 671 sesiones tienen las 7 grabaciones. La posición en ese orden es un
proxy de tiempo acumulado en tarea **independiente de la señal EEG**, y es lo
que codifica `condicion_orden` en la matriz de la Fase 13.

> Como el orden es fijo, la condición y el tiempo acumulado en tarea están
> **perfectamente confundidos**: no hay forma de separar "esto es un PVSAT" de
> "esto es lo último de la sesión". No es un defecto del análisis sino del
> diseño del protocolo, y hay que decirlo así en la memoria.

> **Cifras del subconjunto etiquetado (Fase 12, 2026-08-23):** no todo el
> dataset tiene etiqueta VAS. Quedan **4453 grabaciones / 653 sesiones /
> 174 personas / 274 718 ventanas**. Para cualquier cuenta relacionada con
> modelado, usar esas cifras y no las de esta tabla.

---

## Pendientes

- **Confirmar en test la pista de las `_rel_sd`.** Es lo único con señal
  intra-persona que ha aparecido (ρ = +0.157 para el compuesto), y se eligió
  mirando train+val, así que reportarlo sobre esos mismos datos sería
  circular. Hay que confirmarlo con selección dentro de la validación cruzada
  o gastando test una sola vez. Detalle en la Fase 13.
- **Ruta CNN.** Las ventanas crudas (16.2 GB) están generadas desde la Fase 11
  pero no se ha construido el tensor: falta la normalización z por canal
  ajustada solo sobre train y decidir el tratamiento de los canales
  enmascarados (un canal extra de máscara, no ceros).

---

## Fase 10 — retirada (2026-08-23)

La Fase 10 era el **auto-etiquetado de fatiga**: derivar la etiqueta
(baja/media/alta y un continuo) aplicando umbrales sobre el índice theta/alfa
de la Fase 9. Se retira del pipeline. No había código que borrar —nunca llegó
a implementarse—, así que el cambio es de diseño y de documentación.

**Por qué se retira:** existía solo porque no había etiqueta. La Fase 12 aportó
una **medida externa y real** de la variable de interés —el VAS de fatiga
percibida que responde el propio participante— sobre 653 sesiones y 174
personas. Con eso disponible, derivar la etiqueta de la propia señal pasa de
ser un apaño razonable a ser un error metodológico: se estaría entrenando un
modelo para reproducir un umbral que nos hemos inventado, y llamando "fatiga"
a lo que en realidad es "potencia theta alta".

Tres consecuencias:

- **Desaparece el riesgo de circularidad** que este documento venía arrastrando
  como pendiente. Frente a una etiqueta externa, `pot_abs_AF3_theta` y el
  propio índice pueden entrar en la matriz sin reservas: el modelo no puede
  aprender el umbral del etiquetado porque no hay umbral.
- **La numeración no se renumera.** Las Fases 11, 11.5, 12 y 13 conservan su
  número. Renumerar invalidaría todas las referencias cruzadas del código, de
  los tests y del historial de git, a cambio de nada. El hueco entre la 9 y la
  11 queda documentado aquí, que es más útil que un salto sin explicación.
- **La comparación índice ↔ VAS sigue siendo interesante**, pero como
  *análisis*, no como fase: ¿coinciden los tercios del índice con los del VAS?
  La respuesta, medida en la Fase 13, es que no.

---

## ⚠️ Nota de corrección de orden (2026-07-28)

El orden de fases y el flujo de uso típico de versiones anteriores de este
documento tenían un error: `evaluar_calidad` y `aplicar_car` se ejecutaban
**antes** de `recortar_registro` y `aplicar_filtros`. Esto significa que:

- La varianza y el z-score robusto (criterios 3 y 4 de `evaluar_calidad`) se
  calculaban incluyendo los 5 s de artefactos de colocación/retirada del
  casco al inicio y final del registro, pudiendo generar falsos positivos de
  "amplitud excesiva".
- Esas mismas métricas se calculaban sobre señal sin filtrar, dominada por
  deriva de baja frecuencia y ruido de red (50 Hz), no por el contenido real
  en la banda 1–30 Hz de interés.
- `aplicar_car` promediaba canales potencialmente mal evaluados por lo anterior.

**Las funciones en sí (`reader.py`, `quality.py`, `referencing.py`,
`filtering.py`) no necesitan cambios internos** — el problema es solo el
orden de las llamadas. Ver el "Flujo de uso típico" corregido más abajo.

---

## Visión general del pipeline

| Fase | Descripción                                             | Estado      | Módulo              |
|------|---------------------------------------------------------|-------------|---------------------|
| 1    | Lectura de archivos EDF                                 | Completada  | `reader.py`         |
| 2    | Recorte de márgenes (5 s inicio/final)                  | Completada  | `filtering.py`      |
| 3    | Filtro notch (50 Hz) + bandpass FIR (1–30 Hz)           | Completada  | `filtering.py`      |
| 4    | Evaluación de calidad de canales                        | Completada  | `quality.py`        |
| 5    | Re-referenciación (CAR)                                 | Completada  | `referencing.py`    |
| 6    | Eliminación de artefactos (ICA, solo ocular)            | Completada  | `artifacts.py`      |
| 7    | Epoching (segmentación en ventanas)                     | Completada  | `epoching.py`       |
| 8    | Análisis espectral (PSD Welch, por canal y por época)   | Completada  | `spectral.py`       |
| 9    | Extracción de potencia por bandas + índice theta/alfa   | Completada  | `bands.py`          |
| ~~10~~ | ~~Auto-etiquetado de fatiga~~                          | **Retirada** | — (ver arriba)     |
| 11   | Procesamiento por lotes                                 | Completada  | `batch.py`          |
| 11.5 | Máscara de validez de canal                             | Completada  | `features.py`       |
| 12   | Dataset supervisado (etiqueta VAS)                      | Completada  | `labels.py`         |
| 13   | Matriz de features del Random Forest                    | Completada  | `matrix.py`         |

> **El hueco del 10 es deliberado.** La Fase 10 (auto-etiquetado de fatiga por
> umbrales sobre el índice theta/alfa) se retiró el 2026-08-23 al disponer de
> una etiqueta externa real; nunca llegó a implementarse. La numeración de las
> demás no se toca, para no invalidar las referencias del código, los tests y
> el historial de git. Justificación completa en "Fase 10 — retirada".

> Nota: las Fases 2–5 ya estaban implementadas antes de detectar el error de
> orden; el código de cada función es válido, solo cambia el orden en que se
> invocan (ver flujo de uso típico corregido).

> Nota (Fase 6): requiere que `cargar_eeg` (Fase 1) asigne montaje
> (`raw.set_montage("standard_1020", ...)`), añadido junto con esta fase —
> ICLabel necesita posiciones de electrodo para calcular topografías de scalp.

---

## Fase 1 — Lectura de archivos EDF

**Archivo:** `src/eeg_library/preprocesado/reader.py`
**Función pública:** `cargar_eeg(ruta: str) -> tuple[mne.io.BaseRaw, dict]`
**Constantes exportadas:** `TARGET_SFREQ` (256), `MIN_SFREQ` (128)

### Qué hace
1. Comprueba que el archivo existe; lanza `FileNotFoundError` si no.
2. Carga el EDF con `mne.io.read_raw_edf(..., preload=True)`.
3. **Rechaza el archivo si su sfreq original es inferior a `MIN_SFREQ`
   (128 Hz)** — lanza `ValueError` antes de intentar ningún resampleo (ver
   "Armonización de sfreq" más abajo).
4. **Armoniza la frecuencia de muestreo a `TARGET_SFREQ` (256 Hz)** —
   inmediatamente después de cargar, antes de cualquier otra transformación.
5. **Tipa como `'misc'` todo canal que no esté en `CANALES_EPOC`** (ver
   "Tipado de canales" más abajo).
6. **Rescala los canales CQ solo si llegaron escalados** (ver "Escala de los
   canales CQ" más abajo).
7. Asigna el montaje `standard_1020` a los canales EEG.
8. Valida que la frecuencia de muestreo final sea >= `TARGET_SFREQ`; lanza
   `ValueError` si no (defensivo: tras el paso 4 siempre se cumple).
9. Imprime un resumen: nombre de archivo, frecuencia (y resampleo si lo hubo),
   duración y canales EEG (más el recuento de no-EEG).

### Tipado de canales (corregido el 2026-08-10)

**El error:** la versión anterior identificaba los canales no-EEG con una
lista de nombres tomada de los dos archivos de `data/samples/` con los que se
desarrolló la librería (`CQ_`, `EQ_`, `TIME_STAMP`, `COUNTER`, `BATTERY`…).
EmotivPro exporta el mismo casco con convenciones distintas según su versión.
En el dataset final conviven **cuatro esquemas de canal**, y ninguno usa la
convención de los archivos de desarrollo:

| Origen | Canales (MNE) | Archivos | Convención |
|--------|---------------|----------|------------|
| `data/samples/` (desarrollo) | 60 | 2 | `CQ_AF3`, `RAW_CQ`, `TIME_STAMP_s` |
| Dataset final | 59 | 78   | `CQ.AF3`, `RawCq`, `TimestampS` |
| Dataset final | 63 | 22   | ídem |
| Dataset final | 67 | 562  | ídem |
| Dataset final | 71 | 3912 | ídem |

Lo único constante en los cinco esquemas son los 14 electrodos: están
presentes y con el mismo nombre en todos. Sobre el dataset final **no
coincidía ni un nombre** de la lista de exclusión, así que no se retipaba
ningún canal. Consecuencias reales, medidas sobre
`sub-C1F081_ses-M4_task-PASAT_eeg.edf` (71 canales):

- `calcular_psd` (Fase 8, `picks="eeg"`) calculaba el espectro de los 71
  canales y `extraer_bandas` (Fase 9) devolvía **568 columnas** de potencia
  en vez de 112, con features como `pot_abs_MOT.AccX_theta` o
  `pot_abs_Battery_alfa` entrando en la matriz del Random Forest.
- Los criterios 1 y 2 de `evaluar_calidad` (Fase 4) **no se habían ejecutado
  nunca** sobre el dataset final: buscaban `CQ_<canal>` y los archivos traen
  `CQ.<canal>`. El reporte imprimía `N/A` en esas columnas y solo actuaban
  los criterios de varianza — en silencio, sin error.
- `aplicar_filtros` (Fase 3) filtraba también los CQ/EQ/telemetría.

**La corrección:** se invierte a *allowlist* — solo los canales de
`CANALES_EPOC` quedan como `'eeg'`, todo lo demás pasa a `'misc'`. Los 14
nombres de electrodo provienen del sistema 10-20 y son idénticos en los cinco
esquemas, mientras que una lista de nombres a excluir habría exigido
enumerarlos todos y volvería a fallar **en silencio** con el siguiente que
apareciera. Efecto medido sobre el archivo real: 112 columnas de potencia,
criterios CQ operativos y un 24 % menos de tiempo por archivo (3.91 s frente
a 5.16 s), al no filtrar ni calcular PSD de 57 canales irrelevantes.

`set_channel_types` se llama con `on_unit_change="ignore"`: el cambio de
unidad de voltios a adimensional es justo el que se busca, y su aviso se
repetiría en cada archivo del lote de la Fase 11.

### Escala de los canales CQ (corregido el 2026-08-10)

Los canales CQ llegan en una escala u otra según cómo declare el EDF su
unidad física, comprobado en `raw._orig_units`:

| Export | Unidad declarada | Qué hace MNE | Valor que llega |
|--------|------------------|--------------|-----------------|
| 60 canales | `µV`  | aplica el factor de voltaje | ~4e-6 |
| 67 y 71    | `n/a` | no la toca                  | 4.0 (correcto) |

La versión anterior multiplicaba por `1e6` incondicionalmente todo canal
`CQ_*`. Corregir solo los nombres habría sido peor que no tocar nada: habría
convertido un CQ correcto de 4.0 en 4e6, dejando el criterio "CQ medio < 3.0"
permanentemente en verde. Por eso el rescalado se decide **por el rango real
de los valores** y no por el nombre ni por la versión del export: la escala CQ
es discreta con paso 1 en [0, 4], así que cualquier valor sin escalar no nulo
es >= 1, y un máximo por debajo de `_CQ_VALOR_MINIMO_SIN_ESCALAR` (1.0) solo
puede venir de un archivo escalado — o de un CQ idénticamente nulo, caso en el
que revertir la escala es inocuo.

`buscar_canal_cq` (en `quality.py`, junto a `CANALES_EPOC`) localiza el canal
CQ de un electrodo probando los prefijos de `PREFIJOS_CQ` (`"CQ_"`, `"CQ."`),
y lo usan tanto `cargar_eeg` como `evaluar_calidad`.

### Armonización de frecuencia de muestreo (sfreq)

**Hallazgo que motiva esto:** de las 4574 grabaciones válidas del dataset
final, 100 (2.2 %) están grabadas a 128 Hz en vez de los 256 Hz
nominales/esperados — confirmado en `manifest_canonical.csv`, que registra la
`sampling_rate_hz` leída de la cabecera EDF real de cada archivo. Afecta solo
a la medición M2, en 2 centros (C1 y C8) y 16 sujetos distintos —
probablemente un lote grabado con otra configuración de EmotivPro, no un
fallo puntual de un sujeto.

**Por qué en la Fase 1 y no más adelante:** la Fase 3 diseña el filtro
bandpass en función de `sfreq`, la Fase 7 convierte "ventana de N s" a
muestras usando `sfreq`, y la Fase 8 calcula la resolución espectral a
partir de `sfreq`. Si la armonización no queda garantizada en el punto de
entrada del pipeline, cualquier fase downstream puede romperse
silenciosamente para ese 4.2 % de archivos. Por eso `cargar_eeg` resamplea
inmediatamente después de leer el EDF, antes de devolver el `Raw`.

**Por qué no se pierde información real:** el Nyquist de 128 Hz es 64 Hz,
muy por encima de la banda 1-30 Hz que usa el filtro bandpass de la Fase 3.
Es decir, esas 252 grabaciones no tienen menos información útil real para
este análisis — el resampleo es puramente para que todas las señales tengan
el mismo número de muestras por segundo antes del epoching (Fase 7) y del
Welch (Fase 8), no para "recuperar" contenido de alta frecuencia que nunca
existió.

**Cómo se hace:**
- Si `raw.info["sfreq"]` original es inferior a `MIN_SFREQ` (128 Hz), se
  rechaza el archivo con `ValueError` y no se resamplea al alza: por debajo
  de ese umbral no hay garantía del margen de Nyquist necesario para la
  banda 1-30 Hz, y "estirar" esos datos a 256 Hz generaría una falsa
  sensación de calidad uniforme sin aportar información real. El único caso
  de resampleo ascendente permitido y validado es el de 128 Hz exactos,
  documentado arriba.
- Se compara `raw.info["sfreq"]` contra `TARGET_SFREQ` tras cargar. Si
  coinciden, no se toca nada (evita reprocesar innecesariamente el ~98 % de
  archivos que ya están bien).
- Si no coinciden (y la sfreq original ya pasó la validación de `MIN_SFREQ`),
  `raw.resample(TARGET_SFREQ, npad="auto")` de MNE — resampleo por FFT con
  anti-aliasing interno, no interpolación lineal. Esto cubre tanto el
  resampleo ascendente desde 128 Hz como uno descendente, si algún archivo
  se hubiera grabado por encima de 256 Hz.
- Es solo en memoria: ningún `.edf` se escribe ni se modifica en disco.
- Se registra en la metadata devuelta (`resampled`, `original_sfreq`) para
  poder auditar después qué grabaciones fueron tocadas — útil para un
  chequeo de sensibilidad (comparar métricas del modelo con y sin ese
  subconjunto).

### Decisiones de diseño
- `preload=True` es obligatorio: las fases posteriores acceden a `raw._data`
  directamente como un array NumPy.
- Aparte de la armonización de sfreq, no se aplica ningún otro
  procesamiento; la función solo carga, resamplea si hace falta y valida.
- El rescalado CQ se hace aquí para que `evaluar_calidad` reciba valores en
  el rango correcto sin necesidad de conocer detalles del formato EDF.
- `cargar_eeg` ahora devuelve `(raw, metadata)`, igual que el resto de fases
  del pipeline (Fases 2-9), en vez de solo `raw`: rompe la firma anterior a
  propósito, para que la Fase 1 sea trazable de la misma forma que las
  demás en vez de ser la única excepción.

### Firma y salida
```python
raw, metadata = cargar_eeg("ruta/al/archivo.edf")

# raw      : mne.io.BaseRaw — preload=True, ya a TARGET_SFREQ Hz. Solo los
#            canales de CANALES_EPOC quedan tipados 'eeg'; el resto, 'misc'.
# metadata : dict con claves:
#   - metodo         : "carga_edf"
#   - archivo        : nombre del archivo cargado
#   - sfreq          : frecuencia final (Hz, == TARGET_SFREQ)
#   - original_sfreq : frecuencia del archivo original (Hz)
#   - resampled      : bool — True si se resampleó a TARGET_SFREQ
#   - duracion_s     : duración del registro en segundos
#   - n_canales      : número total de canales del archivo (60, 67 o 71)
#   - n_canales_eeg  : canales tipados 'eeg' (los de CANALES_EPOC presentes)
#   - timestamp      : str ISO 8601
```

---

## Fase 2 y 3 — Preprocesamiento de señal (recorte + filtro)

**Archivo:** `src/eeg_library/preprocesado/filtering.py`
**Funciones públicas:**
- `recortar_registro(raw, inicio=5.0, fin=5.0) -> tuple`
- `aplicar_filtros(raw, notch_freq=50.0, banda=(1.0, 30.0)) -> tuple`

> Ver nota de corrección de orden al inicio del documento.

El orden correcto de aplicación es: **recortar primero, filtrar después**.
Recortar los márgenes antes de filtrar reduce los artefactos de borde de
los filtros FIR, que son sensibles a transitorios al inicio y final de la
señal.

### `recortar_registro` — Eliminación de márgenes (Fase 2)

#### Qué hace
Elimina los primeros `inicio` segundos y los últimos `fin` segundos del
registro usando `raw.copy().crop(tmin=inicio, tmax=raw.times[-1] - fin)`.
Los márgenes contienen artefactos de colocación y retirada del casco
(transitorio de amplificador, movimiento de electrodos).

#### Validaciones
- `inicio >= 0` y `fin >= 0`.
- `inicio + fin < duración total` del registro.

#### Firma y salida
```python
raw_rec, metadata = recortar_registro(raw, inicio=5.0, fin=5.0)

# raw_rec  : mne.io.BaseRaw — copia del Raw recortada
# metadata : dict con claves:
#   - metodo              : "crop"
#   - inicio_s            : float
#   - fin_s               : float
#   - duracion_original_s : float
#   - duracion_final_s    : float
#   - timestamp           : str ISO 8601
```

---

### `aplicar_filtros` — Notch + bandpass FIR (Fase 3)

#### Qué hace
Aplica sobre los datos continuos dos filtros en cadena, afectando únicamente
a los canales de tipo `'eeg'`:

1. **Notch FIR zero-phase** a `notch_freq` (50 Hz por defecto) — elimina la
   interferencia de la red eléctrica europea.
2. **Bandpass FIR zero-phase** con diseño `firwin` en la banda `banda`
   (1–30 Hz por defecto) — descarta componentes DC, deriva de base y
   frecuencias altas irrelevantes para el análisis de fatiga.

Ambos filtros usan `phase='zero'` (fase lineal), lo que preserva las
latencias de los eventos sin introducir desfase.

#### Validaciones
- `l_freq > 0` y `l_freq < h_freq`.
- `h_freq < Nyquist` y `notch_freq < Nyquist`.

#### Firma y salida
```python
raw_filt, metadata = aplicar_filtros(raw)

# raw_filt : mne.io.BaseRaw — copia del Raw filtrado
# metadata : dict con claves:
#   - metodo      : "notch + bandpass FIR zero-phase"
#   - notch_hz    : float
#   - banda_hz    : [l_freq, h_freq]
#   - fir_design  : "firwin"
#   - phase       : "zero"
#   - timestamp   : str ISO 8601
```

#### Decisiones de diseño
- El notch se aplica **antes** del bandpass: elimina la componente de red
  antes de que el filtro paso banda pueda amplificar transitorios cerca de
  sus bordes de transición.
- `picks='eeg'` garantiza que los canales `CQ_*`, markers o cualquier
  canal no-EEG queden intactos.
- Trabaja sobre `raw.copy()` para no mutar el objeto original.
- Delegamos en MNE la elección de la longitud del FIR (`filter_length='auto'`),
  que adapta la longitud al cociente `sfreq / l_freq`.

---

## Fase 4 — Evaluación de calidad de canales (CQ)

**Archivo:** `src/eeg_library/preprocesado/quality.py`
**Función pública:** `evaluar_calidad(raw, umbral_flat=1e-12, umbral_z=3.0)`
**Constante exportada:** `CANALES_EPOC` (lista de 14 nombres de canal)

> Se ejecuta sobre el Raw ya recortado y filtrado (Fases 2 y 3), no sobre el
> Raw crudo. Ver nota de corrección de orden al inicio del documento.

### Qué hace
Recorre los 14 canales del Emotiv Epoc X que estén presentes en el Raw y aplica
cuatro criterios de exclusión independientes (basta cumplir uno para excluir):

| # | Criterio                                       | Umbral por defecto |
|---|------------------------------------------------|--------------------|
| 1 | CQ medio < 3.0                                 | 3.0 (fijo)         |
| 2 | > 20 % de muestras con CQ <= 1                 | 20 % (fijo)        |
| 3 | Varianza < `umbral_flat` (señal plana)         | 1e-12 V² (absoluto)|
| 4 | z-score robusto de varianza > `umbral_z`       | 3.0 (FASTER)       |

Los criterios 1 y 2 solo se evalúan si existe el canal `CQ_<nombre>` en el Raw.
Si no existe, se omiten y solo se aplican los criterios de varianza.

Los criterios 3 y 4 atacan fallos distintos y usan umbrales de naturaleza
distinta a propósito:

- **Criterio 3 (umbral absoluto):** una señal casi plana revela un fallo
  físico del electrodo (sin contacto eléctrico). Su detección no debe
  depender del resto del registro, por eso se usa un umbral fijo.
- **Criterio 4 (umbral relativo robusto):** lo que cuenta como "amplitud
  excesiva" depende del participante y del montaje, así que se compara
  con el resto de canales de la sesión usando el enfoque de FASTER
  (Nolan et al., 2010): se calcula la mediana y el MAD de las varianzas
  entre canales y se excluye el canal cuyo z-score robusto supera 3
  (umbral estándar de FASTER). Si la dispersión inter-canal es nula
  (MAD = 0), el criterio se omite para evitar exclusiones espurias.

Tras evaluar todos los canales imprime una tabla resumen por consola
(`_imprimir_reporte`) y lanza una advertencia si hay menos de 8 canales buenos
o una advertencia crítica si hay menos de 6.

### Firma y salida
```python
good_channels, excluidos, metricas = evaluar_calidad(raw)

# good_channels : list[str]  — canales que pasan todos los criterios
# excluidos     : dict[str, str]  — {canal: "motivo de exclusión"}
# metricas      : dict[str, dict] — {canal: {cq_medio, pct_muestras_malas, varianza, z_robusto}}
```

### Decisiones de diseño
- Solo procesa canales presentes en `CANALES_EPOC`; ignora cualquier otro canal
  (p. ej. `EXG1`, acelerómetros).
- Los canales CQ se buscan por nombre con prefijo `"CQ_"` dentro del propio Raw,
  sin necesitar un argumento adicional.
- Los umbrales (`umbral_flat`, `umbral_z`) son configurables para facilitar
  la experimentación.
- `CANALES_EPOC` se define aquí y se importa desde `referencing.py` para evitar
  duplicar la lista.

---

## Fase 5 — Re-referenciación CAR

**Archivo:** `src/eeg_library/preprocesado/referencing.py`
**Función pública:** `aplicar_car(raw, good_channels) -> tuple`

> Se ejecuta con los `good_channels` obtenidos en la Fase 4, ya sobre señal
> recortada y filtrada.

### Qué hace
Implementa **Common Average Reference (CAR)**: calcula el promedio instantáneo
de las señales de `good_channels` y lo sustrae a cada uno de esos canales:

```
señal_car[canal] = señal[canal] - mean(señal[good_channels], axis=canales)
```

La lógica varía según el número de canales buenos:

| Canales buenos | Comportamiento                              |
|----------------|---------------------------------------------|
| >= 8           | CAR estándar, sin advertencias              |
| 6 – 7          | CAR aplicada + imprime ADVERTENCIA en log   |
| < 6            | CAR omitida; devuelve el Raw sin modificar  |

### Firma y salida
```python
raw_ref, metadata = aplicar_car(raw, good_channels)

# raw_ref  : mne.io.BaseRaw — copia del raw (con o sin CAR)
# metadata : dict con claves:
#   - metodo            : "CAR" o "ninguno"
#   - canales_incluidos : list[str]
#   - canales_excluidos : list[str]  (canales EPOC presentes pero no usados)
#   - timestamp         : str ISO 8601
```

### Decisiones de diseño
- Trabaja sobre `raw.copy()` para no mutar el objeto original; el test
  `test_car_modifica_copia_no_el_original` verifica este comportamiento.
- La operación se hace directamente sobre `raw._data` (array NumPy) para
  evitar la sobrecarga de la API de MNE.
- Los canales excluidos por la Fase 4 no participan en el promedio ni son
  modificados (verificado en `test_canales_excluidos_no_se_modifican`).

---

## Fase 6 — Eliminación de artefactos oculares mediante ICA

**Archivo:** `src/eeg_library/preprocesado/artifacts.py`
**Funciones públicas:**
- `entrenar_ica(raw, good_channels, random_state=42) -> tuple`
- `identificar_componentes_oculares(raw, ica, umbral_prob=0.8) -> tuple`
- `eliminar_artefactos_oculares(raw, ica, componentes_a_excluir) -> tuple`

Aplicada sobre la señal ya recortada, filtrada, evaluada en calidad y
re-referenciada (CAR) — Fases 2-5. Las tres funciones se usan encadenadas.

### Alcance (reducido, decisión de diseño)
- **Solo componentes oculares** (parpadeos y movimientos oculares) se
  excluyen de la señal.
- Cardíaco, muscular, line noise residual y otros artefactos quedan **fuera
  de alcance de exclusión** para este TFG: el Emotiv Epoc X no tiene canales
  EOG/ECG dedicados que permitan identificarlos con fiabilidad sin
  heurísticas complejas, y el presupuesto del TFG es de 300h. Sí se
  **detectan y documentan** como diagnóstico (ver
  `identificar_componentes_oculares` más abajo), pero nunca se eliminan.
- El line noise a 50 Hz ya queda cubierto por el filtro notch (Fase 3).

### Algoritmo de ICA
- **Infomax Extended** (`method="infomax"`, `fit_params={"extended": True}`)
  — variante de Infomax que maneja tanto distribuciones super-gaussianas
  como sub-gaussianas, adecuada para señales EEG con artefactos mixtos.

### Clasificación de componentes: ICLabel
Se usa **ICLabel** (vía `mne-icalabel`) en vez de `find_bads_eog` con un
canal frontal como proxy de EOG:

- `find_bads_eog` necesita elegir a mano un canal "proxy" (p. ej. AF3)
  y correlacionar cada componente con su señal cruda. Ese canal frontal
  también contiene actividad cerebral genuina relevante para fatiga
  mental (theta/alpha frontal), así que usarlo como referencia de "esto
  es ojo" es circular: puede acabar marcando como artefacto ocular
  actividad frontal que es justo la señal de interés del estudio.
- ICLabel es un clasificador entrenado sobre miles de descomposiciones
  ICA de EEG real etiquetadas manualmente (Pion-Tonachini et al., 2019);
  no depende de la señal de ningún canal concreto, sino de topografía +
  espectro + autocorrelación de cada componente, y da una probabilidad
  por 7 categorías estándar (brain, muscle artifact, eye blink, heart
  beat, line noise, channel noise, other) en vez de una decisión binaria.
  Esto habilita además, sin coste adicional, el diagnóstico de qué otras
  categorías de artefacto aparecen con alta probabilidad aunque no se
  traten (ver `identificar_componentes_oculares` más abajo).

**Caveat conocido:** ICLabel se validó sobre EEG filtrado 1–100 Hz; este
pipeline filtra 1–30 Hz (Fase 3), por debajo de ese rango. MNE-ICALabel
emite un aviso en tiempo de ejecución pero sigue funcionando (ver
"Pendientes" al inicio del documento).

---

### `entrenar_ica` — Ajuste de ICA (Infomax Extended)

#### Qué hace
Ajusta ICA únicamente sobre `good_channels` (no sobre todos los canales EEG
del Raw): `aplicar_car` no elimina del objeto Raw los canales excluidos por
`evaluar_calidad`, solo los deja fuera del promedio CAR. Ajustar ICA sobre
todos los canales EEG contaminaría la descomposición con canales de mala
calidad y rompería la relación entre `n_components` y la deficiencia de
rango real introducida por CAR.

`n_components = len(good_channels) - 1`: CAR resta el promedio de esos
mismos canales a sí mismos, introduciendo una dependencia lineal entre
ellos (rango = n_canales - 1). Pedir más componentes de las que el rango
permite genera componentes espurias ("ghost ICs") sin interpretación
física — enfoque estándar del pipeline FASTER/EEGLAB.

**Limitación conocida:** la resta `-1` asume que CAR se aplicó realmente
(`aplicar_car` con >= 6 canales buenos). Si CAR se omitió (< 6 canales
buenos), no hay deficiencia de rango inducida por CAR y `-1` es
conservador pero no estrictamente necesario.

#### Validaciones
- `len(good_channels) >= 2`.
- Todos los nombres en `good_channels` deben existir en `raw.ch_names`.

#### Firma y salida
```python
ica, metadata = entrenar_ica(raw, good_channels, random_state=42)

# ica      : mne.preprocessing.ICA — ajustada sobre good_channels
# metadata : dict con claves:
#   - metodo         : "infomax_extended"
#   - n_components   : int (len(good_channels) - 1)
#   - n_good_channels: int
#   - canales_usados : list[str]
#   - tiempo_ajuste_s: float
#   - random_state   : int
#   - timestamp      : str ISO 8601
```

---

### `identificar_componentes_oculares` — Clasificación con ICLabel

#### Qué hace
Llama a `mne_icalabel.iclabel.iclabel_label_components(raw, ica)`, que
devuelve la matriz completa de probabilidades por componente y categoría
(7 columnas: brain, muscle artifact, eye blink, heart beat, line noise,
channel noise, other — orden fijo documentado en `CATEGORIAS_ICLABEL`).

- `componentes_a_excluir`: índices con categoría `"eye blink"` y
  probabilidad >= `umbral_prob`.
- `diagnostico_otras_categorias`: para las 6 categorías restantes, lista de
  `{"componente": idx, "probabilidad": p}` de las que superan el mismo
  umbral — dato diagnóstico gratuito (las probabilidades ya se calculan)
  para justificar en la memoria por qué se detectaron pero no se trataron.

#### Firma y salida
```python
componentes_a_excluir, diagnostico_otras_categorias, proba_matrix = (
    identificar_componentes_oculares(raw, ica, umbral_prob=0.8)
)

# componentes_a_excluir        : list[int]
# diagnostico_otras_categorias : dict[str, list[dict]]  {"componente": int, "probabilidad": float}
# proba_matrix                 : np.ndarray (n_components, 7)
```

---

### `eliminar_artefactos_oculares` — Reconstrucción sin componentes oculares

#### Qué hace
Copia `ica` internamente (`ica.copy()`) y fija `exclude` en la copia — el
objeto `ica` recibido **no se muta**, igual que `raw` (convención del
proyecto). Aplica `ica_copia.apply()` sobre `raw.copy()` para reconstruir
la señal sin las componentes indicadas, y calcula la varianza explicada
eliminada con `ica_copia.get_explained_variance_ratio(...)`.

#### Firma y salida
```python
raw_limpio, metadata = eliminar_artefactos_oculares(raw, ica, componentes_a_excluir)

# raw_limpio : mne.io.BaseRaw — copia sin las componentes indicadas
# metadata   : dict con claves:
#   - n_componentes_eliminadas    : int
#   - componentes_eliminadas      : list[int]
#   - varianza_explicada_eliminada: float | None (None si no se elimina ninguna)
#   - timestamp                   : str ISO 8601
```

---

## Fase 7 — Epocado (segmentación en ventanas)

**Archivo:** `src/eeg_library/procesado/epoching.py`
**Función pública:** `crear_epocas(raw, sujeto, condicion, sesion, archivo_origen, duracion=4.0, solape=2.0) -> tuple`

Aplicada sobre la señal ya recortada, filtrada, evaluada en calidad,
re-referenciada y limpiada de artefactos oculares (Fases 2-6). Este
epocado es **compartido**: sus épocas alimentan tanto la extracción de
features para Random Forest (Fases 9-10) como las ventanas crudas
normalizadas para la CNN, así que la función no aplica ninguna
transformación específica de ninguna de las dos rutas.

### Qué hace
Usa `mne.make_fixed_length_epochs(raw, duration=duracion, overlap=solape,
preload=True)` para segmentar en ventanas de longitud fija con
solapamiento. El tramo final de cada archivo que no completa una ventana
entera se descarta automáticamente (MNE no genera una época a partir de
`n_epocas * paso_s + duracion > duración_total`).

Por defecto: ventanas de **4 s** (1024 muestras a 256 Hz) con
solapamiento del **50 %** (paso de 2 s) entre ventanas consecutivas.

### Justificación científica

- **Duración de 4 s:** Wosiak, Sumiński y Żykwińska (2025) usan una
  ventana fija de 4 s como punto de partida de su pipeline de detección
  de fatiga cognitiva por EEG, con muestras suficientes para estadísticos
  de ventana fiables y que abarcan varios ciclos de los ritmos theta/alfa
  relevantes para carga mental. Es compatible con Lee y An (2023), que
  comparan explícitamente longitudes de ventana en modelos LSTM, CNN y
  LSTM-CNN para clasificación de somnolencia por EEG: el modelo
  combinado LSTM-CNN obtiene su mejor índice kappa (0.77) justo con
  ventanas de 4 s, y el rendimiento de los modelos basados en redes
  neuronales cae de forma notable por encima de 8 s, mientras que por
  debajo de 8 s los cambios son pequeños.
  - Wosiak, A., Sumiński, M., & Żykwińska, K. (2025). *Impact of
    Temporal Window Shift on EEG-Based Machine Learning Models for
    Cognitive Fatigue Detection.* Algorithms, 18(10), 629.
    https://doi.org/10.3390/a18100629
  - Lee, C., & An, J. (2023). *LSTM-CNN model of drowsiness detection
    from multiple consciousness states acquired by EEG.* Expert
    Systems with Applications, 213, 119032.
  - Para el lado de Random Forest (no CNN): Tzimourta et al. (2019)
    evalúan explícitamente distintas longitudes de época — no solapadas,
    de 5 a 12 s — extrayendo features estadísticos y espectrales por
    banda (delta, theta, alfa, beta, gamma) para un clasificador Random
    Forest, y encuentran que las épocas de 12 s dan el mejor rendimiento
    de clasificación. Es un estudio de detección de Alzheimer por EEG, no
    de fatiga, pero confirma que el rango de duración de época elegido
    (4 s) es coherente con el usado en la práctica para alimentar un
    Random Forest con features espectrales por banda, aunque quede algo
    por debajo del óptimo que encuentran ellos (12 s) — la duración de 4 s
    de este proyecto prioriza la duración validada para fatiga/CNN
    (Wosiak et al. y Lee & An, arriba) sobre el óptimo específico de ese
    estudio de Alzheimer.
    - Tzimourta, K. D., Giannakeas, N., Tzallas, A. T., Astrakas, L. G.,
      Afrantou, T., Ioannidis, P., Grigoriadis, N., Angelidis, P.,
      Tsalikakis, D. G., & Tsipouras, M. G. (2019). *EEG Window Length
      Evaluation for the Detection of Alzheimer's Disease over Different
      Brain Regions.* Brain Sciences, 9(4), 81.
      https://doi.org/10.3390/brainsci9040081
- **Solapamiento del 50 % (paso de 2 s):** compensa la corta duración de
  cada archivo (~3 min) sin generar redundancia excesiva entre ventanas
  consecutivas. Wosiak et al. (2025) — mismo estudio citado arriba —
  investigan directamente el efecto del *shift* (paso) entre ventanas en
  detección de fatiga cognitiva, probando pasos de 0.25, 0.5 y 1 s sobre
  una ventana fija (equivalentes a solapamientos del 93.75 %, 87.5 % y
  75 %): los pasos más pequeños (mayor solapamiento) aumentan la
  precisión y el F1, pero a costa de reducir la independencia entre
  muestras — con datasets cortos como el de este proyecto, un
  solapamiento del 50 % es el punto intermedio más citado en la práctica
  general de EEG entre ese riesgo y la escasez de épocas por archivo.
  - Referencia: la misma cita de Wosiak et al. (2025) de arriba.

### Metadata de origen por época
Cada época debe poder trazarse hasta su sujeto, condición, sesión/momento
y archivo fuente — imprescindible para la partición train/test **por
sujeto** (evitar fuga de datos) y para poder unir después la etiqueta VAS
a cada época (Fase 12). En vez de inventar una estructura de
datos propia, se usa el mecanismo nativo de MNE: `epochs.metadata` es un
`pandas.DataFrame` con una fila por época y las columnas `sujeto`,
`condicion`, `sesion`, `archivo_origen`, `indice_ventana` y `tmin_s`
(inicio de la ventana en segundos relativo al inicio de `raw`). Ventaja
práctica: `mne.concatenate_epochs` (necesario en la Fase 11 para juntar
épocas de varios archivos) concatena `metadata` automáticamente, sin
trabajo adicional.

`sujeto`, `condicion` y `sesion` se reciben como parámetros de la función
en vez de parsearse del nombre de archivo dentro de `epoching.py`: el
parseo de nombres es responsabilidad de la Fase 11 (procesamiento por
lotes), no de esta fase, para mantener el epocado desacoplado de la
convención de nombres de archivo.

### Validaciones
- `duracion > 0`.
- `0 <= solape < duracion`.
- La duración total del Raw debe ser >= `duracion` (si no, `ValueError`:
  no hay señal suficiente para una sola ventana).

### Firma y salida
```python
epochs, metadata = crear_epocas(
    raw, sujeto="S07", condicion="C3", sesion=4,
    archivo_origen="M4_XX_F007_C3.edf",
    duracion=4.0, solape=2.0,
)

# epochs   : mne.Epochs — preload=True, con epochs.metadata (DataFrame):
#   sujeto, condicion, sesion, archivo_origen, indice_ventana, tmin_s
# metadata : dict con claves:
#   - metodo               : "ventanas_fijas_solapadas"
#   - duracion_s           : float
#   - solape_s             : float
#   - paso_s               : float (duracion - solape)
#   - sfreq                : float
#   - n_epocas             : int
#   - duracion_descartada_s: float (segundos finales descartados)
#   - timestamp            : str ISO 8601
```

### Decisiones de diseño
- No mutación: `mne.make_fixed_length_epochs` opera sobre los datos de
  `raw` sin modificar el objeto original (convención del proyecto,
  verificado en `test_no_muta_el_raw_original`).
- `tmin_s` se calcula analíticamente (`indice_ventana * paso_s`) en vez de
  derivarse de `epochs.events`, para no depender de detalles internos de
  numeración de muestras de MNE (`first_samp` tras un `crop` previo).
- No se valida `sujeto`/`condicion`/`sesion` contra listas fijas de
  valores permitidos: la función es agnóstica a la convención de
  codificación de esos identificadores.

---

## Fase 8 — Análisis espectral (PSD, método de Welch)

**Archivo:** `src/eeg_library/procesado/spectral.py`
**Función pública:** `calcular_psd(epochs, banda=(1.0, 30.0), duracion_ventana=1.0, solape_ventana=0.5) -> tuple`

Aplicada sobre las épocas ya generadas en la Fase 7. Calcula la densidad
espectral de potencia por canal individual y por época, sin promediar por
banda ni por región de canales — esa reducción es responsabilidad de la
Fase 9 (extracción de potencia por bandas delta/theta/alfa/beta y
clústeres frontal/parietal para el índice de fatiga), que queda
explícitamente fuera de alcance de esta fase.

### Qué hace
Usa `epochs.compute_psd(method="welch", ...)` de MNE, que sub-segmenta
internamente cada época en ventanas más cortas (parámetro estándar del
método de Welch: promediar varios periodogramas reduce la varianza de la
estimación de la PSD frente a una FFT única sobre toda la época) y
promedia los periodogramas resultantes por canal.

Por defecto: sub-ventanas de **1 s** (256 muestras a 256 Hz) con **50 %**
de solapamiento (128 muestras) y función de ventana **Hann**, calculadas
solo en el rango **1–30 Hz**.

### Justificación científica

- **Sub-ventana de 1 s (256 muestras) con 50 % de solapamiento (128
  muestras) y ventana de Hann:** convención estándar en el análisis
  espectral de EEG epocado. El paquete `eegUtils` (R), especializado en
  análisis EEG, usa por defecto una longitud de FFT de 256 muestras con
  un 50 % de solapamiento al calcular la PSD de Welch sobre datos ya
  divididos en épocas (`compute_psd`/`plot_psd`), con ventana de Hann
  para reducir el *leakage* espectral. Da una resolución en frecuencia
  de `sfreq / n_per_seg` = 256 / 256 = **1 Hz**, suficiente para separar
  theta (4–8 Hz) de alfa (8–13 Hz) sin perder información relevante. Se
  cumple además la regla general de que la sub-ventana debe ser al menos
  el doble del periodo de la frecuencia más baja de interés — para theta
  (mínimo 4 Hz) eso exige solo ≥ 0.5 s, así que 1 s da margen de sobra.
  - Craddock, M. *eegUtils: Import and process EEG data* (paquete R).
    Documentación de `compute_psd`/`plot_psd`:
    https://craddm.github.io/eegUtils/reference/compute_psd.html

> **Nota:** en la investigación previa a esta entrada se citó también un
> estudio de EEG de imaginación motora con PSD de Welch sobre ventanas de
> 1 s y 0.5 s de solapamiento como segunda fuente convergente. No ha
> sido posible identificar con certeza el artículo exacto entre los
> candidatos encontrados (posibles candidatos: Francis, Keran, Chetan &
> Niranjana Krupa (2021), *EEG-Controlled Robot Navigation using Hjorth
> Parameters and Welch-PSD*; u otros estudios de clasificación de
> imaginación motora con Welch-PSD), así que se deja fuera de esta lista
> de referencias hasta verificarlo. La justificación de 1 s/50 % ya queda
> cubierta por la cita de `eegUtils` de arriba.
- **Rango 1–30 Hz:** el filtro bandpass de la Fase 3 ya eliminó el
  contenido fuera de esa banda: calcular PSD por encima de 30 Hz no
  aportaría señal real, solo ruido de cuantización/numérico. `fmax=30`
  se pasa explícitamente a `compute_psd` para no devolver esas
  frecuencias.
- `n_fft = n_per_seg` (sin zero-padding): la resolución en frecuencia
  queda determinada únicamente por la duración de la sub-ventana, sin
  interpolación espectral que sugiera una resolución mayor de la
  realmente disponible.

### Firma y salida
```python
spectrum, metadata = calcular_psd(epochs)

# spectrum : mne.time_frequency.EpochsSpectrum
#   - forma (n_epocas, n_canales, n_frecuencias)
#   - spectrum.freqs     : array de frecuencias (Hz), resolución 1 Hz
#   - spectrum.ch_names  : nombres de canal, mismo orden que el eje 1
#   - spectrum.metadata  : heredada de epochs.metadata (sujeto, condicion,
#                          sesion, archivo_origen, ...)
#   - spectrum.get_data(fmin=..., fmax=...) / .pick(...) : la Fase 9 puede
#     extraer potencia de banda o de un subconjunto de canales
#     (clústeres frontal/parietal) directamente sobre este objeto.
# metadata : dict con claves:
#   - metodo             : "welch"
#   - banda_hz            : [fmin, fmax]
#   - duracion_ventana_s  : float
#   - solape_ventana_s    : float
#   - paso_ventana_s      : float (duracion_ventana - solape_ventana)
#   - n_per_seg           : int (muestras por sub-ventana)
#   - n_overlap           : int (muestras de solapamiento)
#   - n_segmentos         : int (sub-ventanas promediadas por época)
#   - ventana             : "hann"
#   - resolucion_hz       : float (sfreq / n_per_seg)
#   - n_epocas            : int
#   - n_canales           : int
#   - sfreq               : float
#   - timestamp           : str ISO 8601
```

### Decisiones de diseño
- Se reutiliza `mne.Epochs.compute_psd` en vez de reimplementar Welch con
  `scipy.signal.welch` a mano: ya implementa la sub-segmentación, el
  promediado y el recorte a `[fmin, fmax]`, y devuelve un objeto
  (`EpochsSpectrum`) que hereda automáticamente `epochs.metadata` sin
  trabajo adicional — mismo mecanismo de propagación de metadata que
  `mne.concatenate_epochs` en la Fase 7.
- `picks="eeg"` (no configurable): excluye los canales no-EEG (p. ej.
  `CQ_*`) que `crear_epocas` puede conservar en el objeto `Epochs`, igual
  que `aplicar_filtros` (Fase 3) excluye esos mismos canales del filtrado.
- No se devuelve la estructura de salida en un formato propio (p. ej. un
  DataFrame por banda): se prioriza que la Fase 9 pueda operar
  directamente sobre `EpochsSpectrum.get_data()`/`.pick()`, tanto a nivel
  de canal individual (features del Random Forest) como agrupando
  canales en clústeres frontal/parietal, sin necesitar convertir entre
  estructuras.
- `n_segmentos` se expone en la metadata como dato informativo/verificable
  (usado en los tests), aunque `compute_psd` no lo necesita como
  parámetro: se calcula con la misma fórmula que usa `scipy.signal.welch`
  internamente (`1 + (n_muestras_epoca - n_per_seg) // paso_muestras`).

---

## Fase 9 — Extracción de potencia por bandas + índice theta/alfa

**Archivo:** `src/eeg_library/procesado/bands.py`
**Función pública:** `extraer_bandas(spectrum, bandas=None, cluster_frontal=None, cluster_posterior=None, canales_validos=None) -> tuple`
**Constantes exportadas:** `BANDAS_DEFECTO`, `CLUSTER_FRONTAL`, `CLUSTER_POSTERIOR`

Aplicada sobre el espectro por canal y por época de la Fase 8. Reduce ese
espectro a potencia por banda de frecuencia (por canal individual, sin
promediar por región) y calcula el índice de fatiga theta/alfa a partir
de dos clústeres de canales fijos. No agrega por región ni construye la
matriz del Random Forest — eso es la Fase 13.

> **Por qué se sigue calculando el índice theta/alfa.** Nació como base del
> auto-etiquetado (Fase 10, retirada). Hoy no es la etiqueta —lo es el VAS— y
> **no predice el VAS**: ρ = +0.004 (p = 0.93) dentro de una misma persona,
> y quitarlo de la matriz junto con sus dos componentes deja el R² del Random
> Forest en −0.0145 frente a −0.0150 con ellos. Se mantiene por dos razones,
> ninguna de rendimiento: es el biomarcador de referencia de la literatura de
> fatiga mental y un TFG sobre el tema tiene que reportarlo; y decidir no
> medirlo *después* de ver que sale nulo sería reporte selectivo. El resultado
> negativo es un hallazgo, y solo existe si el índice se calcula.

### Qué hace
1. Para cada banda (delta, theta, alfa, beta) y cada canal, calcula:
   - **Potencia absoluta:** suma de la PSD en los bins de frecuencia de la
     banda, ponderada por la resolución en frecuencia del espectro de
     entrada (integración por rectángulos — válida porque los bins de
     Welch de la Fase 8 están espaciados uniformemente).
   - **Potencia relativa:** potencia absoluta de la banda entre la
     potencia total del espectro calculado (1–30 Hz).
   Con las 4 bandas por defecto y 14 canales: 14 × 4 × 2 = **112 columnas**
   de potencia por época.
2. Calcula el **índice de fatiga theta/alfa** por época: potencia theta
   promediada sobre `CLUSTER_FRONTAL` dividida entre potencia alfa
   promediada sobre `CLUSTER_POSTERIOR`.
3. Conserva la metadata de origen (`spectrum.metadata`, heredada de la
   Fase 7) en el mismo `DataFrame` de salida.

### No se calcula banda gamma
El filtro bandpass de la Fase 3 (1–30 Hz) ya eliminó ese contenido; no
habría señal real que extraer, solo ruido de cuantización/numérico —
mismo razonamiento que el límite superior de 30 Hz en la Fase 8.

### Potencia absoluta y relativa, ambas
La evidencia sobre cuál representación (absoluta o relativa) es más
fiable por sujeto es contradictoria según la banda, así que se calculan
ambas para las 4 bandas y se deja que el análisis de
`feature_importances_` del Random Forest (fases posteriores) decida cuál
aporta más señal, en vez de descartar una a priori.

### Límites de banda: semiabiertos, sin solape ni huecos
Los límites de cada banda son semiabiertos por la derecha
(`fmin <= f < fmax`), salvo la banda cuyo límite superior alcanza el
máximo del espectro calculado (`fmin <= f <= fmax`, ahora mismo la banda
beta con `fmax=30`). Así cada bin de frecuencia del espectro pertenece a
exactamente una banda: sin solapamiento en 4, 8 y 13 Hz (que de otro modo
se contarían dos veces) ni huecos entre bandas. Consecuencia verificada
por tests: para las 4 bandas por defecto, la suma de las 4 potencias
relativas de un canal es exactamente 1 (salvo error de punto flotante), y
la suma de las 4 potencias absolutas reproduce exactamente la potencia
total del espectro.

### Los clústeres se restringen a los canales buenos (`canales_validos`)

**Añadido el 2026-08-10.** `extraer_bandas` acepta `canales_validos`, la lista
de canales que devuelve `evaluar_calidad` (Fase 4). Si se pasa, los promedios
de clúster del índice de fatiga se calculan solo sobre esos canales. **Debe
pasarse siempre que se disponga de él.**

**Por qué:** un canal excluido por `evaluar_calidad` lo fue casi siempre por
amplitud excesiva, y además no pasó por CAR (Fase 5) ni por ICA (Fase 6),
porque ambas operan únicamente sobre los canales buenos — pero seguía
entrando en la media del clúster, donde su potencia de banda la domina.

Medido sobre una muestra aleatoria de 147 archivos reales del dataset:

| Canales frontales excluidos | Índice actual | Solo canales buenos |
|-----------------------------|---------------|---------------------|
| 0                           | 1.24          | 1.44                |
| 1                           | 2.17          | 1.52                |
| 2                           | 3.38          | 1.74                |
| 3                           | 8.38          | 2.39                |
| 4                           | 5.63          | 1.81                |

El índice sin corregir **crece de forma monótona con el número de electrodos
que falló el casco**, mientras que restringido a canales buenos se mantiene
plano — es decir, buena parte de lo que medía era la calidad del contacto y
no la fatiga. Afecta al **64 %** de los archivos (factor de inflación mediano
1.75, p90 3.54, máximo 47.6) y la corrección comprime la cola: el p99 del
índice baja de 22.0 a 9.3. Sin esto, cualquier lectura del índice —incluida
la comparación con el VAS de la Fase 13— heredaría esa contaminación.

**Qué no cambia:** las columnas `pot_abs_*`/`pot_rel_*` se siguen calculando
para los 14 canales pase lo que pase, así que la matriz de features conserva
las mismas columnas y el mismo orden en todos los archivos y el
procesamiento por lotes (Fase 11) puede concatenarlas. Qué hacer con las 8
columnas de un canal excluido dentro de la matriz del Random Forest (dejarlas,
ponerlas a NaN, o añadir una máscara de validez) queda **pendiente para la
Fase 11.5**, ya con el lote procesado delante.

**Caso límite:** si todos los canales de un clúster presentes en el espectro
quedan excluidos por calidad, se lanza `ValueError` en vez de calcular el
índice sobre canales malos — esa grabación debe descartarse del lote. El
mensaje distingue ese caso del de un clúster ausente del espectro, porque
piden acciones distintas (grabación mala frente a montaje o `picks` mal
puestos).

### Clústeres del índice de fatiga theta/alfa
- **Theta → clúster frontal:** AF3, AF4, F3, F4, F7, F8.
- **Alfa → clúster posterior:** P7, P8, O1, O2.

Esta separación espacial (theta frontal, alfa posterior/parietal — **no**
ambos en frontal, que era la definición inicialmente asumida) está
respaldada por dos fuentes, la primera sobre el mismo hardware que este
proyecto:

- Raufi, B., & Longo, L. (2022). *An Evaluation of the EEG
  Alpha-to-Theta and Theta-to-Alpha Band Ratios as Indexes of Mental
  Workload.* Frontiers in Neuroinformatics, 16, 861967.
  https://doi.org/10.3389/fninf.2022.861967 — construye, sobre el mismo
  casco Emotiv EPOC de 14 canales que este proyecto, un clúster de theta
  promediando AF3, AF4, F3, F4, F7 y F8, y un clúster de alfa promediando
  P7 y P8, para calcular ratios alfa/theta como índice de carga mental.
- Un segundo estudio independiente de carga de memoria con el mismo
  hardware localiza las diferencias de potencia delta/theta en los
  canales frontales AF3, AF4, F7 y F8, y las diferencias de alfa en
  sitios occipitales/parietales O1, O2, P7 y P8 — de ahí que el clúster
  posterior de este proyecto amplíe el de Raufi & Longo (P7, P8) con O1 y
  O2. (Cita exacta pendiente de verificar — ver "Pendientes".)

### Validaciones
- `bandas` debe incluir las claves `'theta'` y `'alfa'` (el índice de
  fatiga depende de ambas); si no, `ValueError`.
- Cada banda debe tener al menos un bin de frecuencia disponible en el
  espectro de entrada; si no, `ValueError`.
- `cluster_frontal` y `cluster_posterior` deben tener al menos un canal
  presente en `spectrum.ch_names`; si no, `ValueError` (evita promediar
  sobre una lista vacía y devolver `NaN` silenciosamente).

### Firma y salida
```python
features, metadata = extraer_bandas(spectrum, canales_validos=good_channels)

# features : pandas.DataFrame — una fila por época. Columnas:
#   - metadata heredada de spectrum.metadata (sujeto, condicion, sesion,
#     archivo_origen, ...), sin modificar.
#   - pot_abs_<canal>_<banda> y pot_rel_<canal>_<banda> por cada canal x
#     banda (14 x 4 x 2 = 112 columnas con los valores por defecto).
#   - theta_frontal, alfa_posterior: potencia de banda promediada sobre
#     cada clúster.
#   - indice_fatiga_theta_alfa: theta_frontal / alfa_posterior.
# metadata : dict con claves:
#   - metodo             : "potencia_por_banda_welch"
#   - bandas_hz           : dict {nombre_banda: [fmin, fmax]}
#   - cluster_frontal     : canales realmente promediados (frontal)
#   - cluster_posterior   : canales realmente promediados (posterior)
#   - cluster_frontal_descartados   : canales del clúster frontal presentes
#                           en el espectro pero excluidos por calidad
#   - cluster_posterior_descartados : ídem, clúster posterior
#   - canales             : lista de canales del espectro
#   - resolucion_hz       : resolución en frecuencia del espectro de entrada
#   - n_epocas            : int
#   - n_canales           : int
#   - n_features_banda    : int (canales x bandas x 2)
#   - timestamp           : str ISO 8601
```

### Decisiones de diseño
- Potencia por **canal individual**, no promediada por región: un
  estudio con el mismo casco de 14 canales encontró que la mejor
  separación entre condiciones aparecía en canales concretos (F3, O2,
  F4), no en promedios regionales — promediar de antemano perdería
  precisamente esa información, y además haría el análisis de
  `feature_importances_` del Random Forest menos informativo ("el canal
  F3 en banda alfa" en vez de "la región frontal"). (Cita exacta
  pendiente de verificar — ver "Pendientes".)
- Se devuelve un único `pandas.DataFrame` con metadata y features juntos
  (en vez de dos estructuras separadas a mantener sincronizadas): permite
  filtrar/agrupar por sujeto o condición directamente y construir la matriz
  `X` del Random Forest (Fase 13) quitando las columnas de metadata
  conocidas, sin más conversión.
- Potencia absoluta calculada por **suma ponderada por la resolución en
  frecuencia** (regla de rectángulos), no por la regla del trapecio: con
  bandas contiguas que comparten límites exactos (4, 8, 13 Hz), el
  trapecio asignaría medio peso a cada bin de límite en las dos bandas
  adyacentes, rompiendo la partición exacta; la suma por rectángulos con
  máscaras semiabiertas asigna cada bin a una única banda y garantiza que
  las 4 potencias absolutas sumen exactamente la potencia total.
- No se aplica ninguna transformación logarítmica ni normalización
  adicional a la potencia (p. ej. `log(potencia)`, típica en algunos
  pipelines de EEG): se deja la decisión a la fase de modelado, que
  conoce el algoritmo final (Random Forest es invariante a
  transformaciones monótonas por canal, a diferencia de modelos lineales
  o basados en distancia).

---

## Fase 11 — Procesamiento por lotes

**Archivo:** `src/eeg_library/procesado/batch.py`
**Funciones públicas:**
- `cargar_tareas(ruta_manifest, dir_edf, ruta_splits=None) -> DataFrame`
- `procesar_archivo(ruta, sujeto, condicion, sesion, extras=None, devolver_ventanas=False, random_state=42) -> tuple`
- `procesar_lote(tareas, dir_salida, dir_ventanas=None, n_procesos=None, reanudar=True, random_state=42, limite=None) -> tuple`
- `consolidar_features(dir_salida, ruta_destino=None) -> DataFrame`
**Excepción exportada:** `ErrorDeFase`

Encadena las Fases 1-9 sobre muchos archivos y persiste dos salidas nacidas
del mismo epocado de la Fase 7: la tabla de features por época para el Random
Forest y las ventanas crudas para la CNN. La fila `i` del parquet y la ventana
`i` del `.npy` de un archivo son la misma época (verificado en tests).

### Estructura de salida
```
dir_salida/
    features/<archivo>.parquet     una tabla por EDF
    features_all.parquet           concatenación (consolidar_features)
    procesamiento_log.csv          una fila por archivo
    procesamiento_resumen.json     parámetros y agregados del lote
dir_ventanas/
    <archivo>.npy                  (n_epocas, 14, 1024) float32
```

### Decisiones de diseño

- **La metadata viene del manifest, no del nombre de archivo.** ESTADO.md
  asignaba a esta fase el parseo de nombres, pero `manifest_canonical.csv` ya
  trae sujeto, condición y medición parseados y validados. `cargar_tareas`
  añade además dos columnas que no están en él:
  - `split`, unido desde `splits.csv` **por sujeto** (la partición es por
    sujeto, no por grabación).
  - `orden_en_sesion`, derivado de `timestamp_iso`: la posición de la
    grabación dentro de su sesión. Como el protocolo tiene orden fijo, es un
    proxy de tiempo acumulado en tarea **independiente del EEG**. Es un entero
    *nullable*: hay una sesión del dataset sin marca de tiempo en ninguna de
    sus 7 grabaciones, y ahí el orden queda como `<NA>` en vez de inventarse.
    La Fase 13 usa su equivalente `condicion_orden`, derivado del nombre de la
    condición, que es estable aunque falten grabaciones.
- **Un proceso por archivo.** El coste lo domina ICA (~60 %), que es CPU pura.
  Los workers reciben rutas y devuelven registros de log; los objetos de MNE
  nunca cruzan la frontera entre procesos. Cada worker escribe su propia
  salida, así que reanudar es simplemente saltarse lo que ya existe.
- **`n_procesos=1` ejecuta en serie, sin pool.** Más lento, pero deja ver la
  traza completa y permite depurar un archivo concreto — que es lo que hace
  falta cuando falla uno entre miles.
- **Un fallo no detiene el lote.** Cada archivo va dentro de un try/except que
  registra la fase en la que falló (`ErrorDeFase`). Con 4574 archivos, saber
  que algo lanzó `ValueError` no es accionable; saber si falló al leer el EDF,
  al epocar una grabación demasiado corta o al calcular el índice sobre un
  clúster entero excluido, sí.
- **La consolidación es un paso aparte** y verifica que todos los parquet
  tengan exactamente las mismas columnas y en el mismo orden. Concatenar a
  ciegas rellenaría de NaN en silencio si un archivo se desviara.
- **El log es también la tabla de control de calidad de la memoria:** guarda
  por archivo los canales buenos y excluidos, si se aplicó CAR, las
  componentes ICA totales y oculares, la varianza explicada eliminada, las
  categorías de artefacto detectadas pero no tratadas (`diagnostico_otras`,
  ver Fase 6) y los canales de clúster descartados (ver Fase 9).

### Rendimiento medido
Sobre 98 grabaciones reales (3 sujetos completos, uno de cada split) con 10
procesos en una máquina de 12 CPUs: **0.93 s/archivo**, 0 errores, 6031
épocas. Proyección al dataset completo:

| | Valor |
|---|---|
| Tiempo | ~71 min |
| Features (parquet) | ~648 MB |
| Ventanas CNN (`.npy` float32) | **16.2 GB** |
| Épocas totales | 282 402 |

> Los 16.2 GB de ventanas son la razón por la que el dataset se sacó de
> OneDrive: ahí dentro nada sincroniza, así que features y ventanas pueden
> convivir con el dataset sin subir 16 GB a la nube.

---

## Fase 11.5 — Máscara de validez de canal

**Archivo:** `src/eeg_library/procesado/features.py`
**Funciones públicas:** `anotar_validez_de_canales(features, log, canales=None)`,
`columnas_de_validez(canales=None)`

Resuelve la decisión que quedaba abierta desde la Fase 9: qué hacer con las 8
columnas de potencia (4 bandas × absoluta/relativa) de un canal que
`evaluar_calidad` descartó. Esas columnas existen igual que las de un canal
bueno —la Fase 9 las calcula para los 14 electrodos siempre, para que la
matriz sea concatenable— pero salen de señal que no pasó por CAR ni por ICA.

### La evidencia (dataset completo: 4543 grabaciones, 280 339 épocas)

| | |
|---|---|
| Archivos con ≥1 canal excluido | 3863 (**85.0 %**), media de 1.87 |
| Épocas afectadas | 240 409 (**85.8 %**) |
| Celdas de potencia afectadas | **13.7 %** |

Concentrado en los electrodos de la frente, donde las almohadillas del Epoc X
se secan antes — AF4 33 %, AF3 26 %, F8 24 %, F7 18 % de los archivos; ningún
otro pasa del 13 %. Los valores de un canal excluido están fuera de escala por
factores de **1.3× a 7.9×** respecto a la mediana del mismo canal cuando está
bueno.

### Por qué máscara y no imputación ni exclusión

- **Esa desviación de escala ya es una señal legible para un árbol.** El
  modelo puede usar la calidad de contacto como atajo tanto si se le da la
  máscara como si no; dejarla implícita no elimina el atajo, solo lo vuelve
  inmedible. Con la máscara se puede cuantificar su peso en
  `feature_importances_` y entrenar con y sin ella.
- **Imputar** habría fabricado ese 13.7 % de celdas justo en los canales
  frontales donde vive el biomarcador theta.
- **Descartar** las grabaciones afectadas no era viable: 21 de los 178
  sujetos no tienen ninguna grabación sin exclusiones.

### Dos hallazgos colaterales

- **El contacto no se degrada a lo largo de la sesión.** Spearman entre
  canales excluidos y posición en el protocolo: **0.048**, plano de la
  posición 1 a la 7 (1.81 → 2.17). Es lo que permite usar `condicion_orden`
  (Fase 13) sin que arrastre un efecto de calidad de contacto.
- **Sí queda un confundido con el índice de fatiga.** Spearman **0.167**; el
  índice mediano por archivo sube monótonamente de 1.14 con 0 canales
  excluidos a 1.88 con 5. Esto **sobrevive** a la restricción de clústeres a
  canales buenos de la Fase 9, así que no es el mismo problema: apunta a que
  en una grabación con mal contacto general también los canales que sobreviven
  son más ruidosos. **Hay que tenerlo presente al interpretar cualquier
  resultado sobre el índice** — la máscara no lo resuelve, solo lo hace
  visible. En particular, matiza su correlación bruta con el VAS (−0.091),
  aunque no la intra-persona, que ya es cero.

### Firma y salida
```python
anotadas = anotar_validez_de_canales(features, log)

# anotadas : copia de features con 15 columnas añadidas al final:
#   - canal_valido_<canal> (bool) x 14, en el orden de CANALES_EPOC
#   - n_canales_buenos (int16)
# Las columnas originales no se tocan ni se reordenan.
```

La máscara es **por archivo, no por época**: `evaluar_calidad` decide sobre el
registro completo, así que todas las épocas de una grabación comparten los
mismos canales válidos.

Valida que el número de válidos de la máscara coincida con el
`n_canales_buenos` que registró `evaluar_calidad`, y lanza `ValueError` si no
—delataría una grabación a la que le faltaba algún electrodo, que la máscara
estaría dando por válido— o si algún `archivo_origen` no aparece como
procesado con éxito en el log.

### Salida generada
`C:\Users\migue\Desktop\TFG\EEG-Data\processed\matriz_modelado.parquet` — 280 339 × **139** columnas.
Nombre distinto de `features_all.parquet` a propósito: volver a ejecutar
`consolidar_features` regenera este último y perdería la anotación, así que la
matriz de modelado vive aparte.

Reparto de canales válidos por época: 14 en 39 930, 13 en 72 695, 12 en
81 125, 11 en 53 228, 10 en 24 560, y menos de 10 en 8801.

---

## Fase 12 — Dataset supervisado (etiqueta VAS)

**Archivo:** `src/eeg_library/procesado/labels.py`
**Funciones públicas:** `normalizar_vas`, `corregir_manifest`,
`construir_etiquetas`, `expandir_a_grabaciones`, `unir_etiquetas`
**Salida:** `C:\Users\migue\Desktop\TFG\EEG-Data\metadata\labels.csv` (670 filas, una por sesión) y
`labels_grabaciones.csv` (4574 filas, una por grabación)
**Origen:** estudio de viabilidad del cruce `VAS_100mm_preFE_filtrado.csv`
(945 filas, VAS 0–100 mm por sujeto × momento) × `manifest_canonical.csv`
(4574 grabaciones). Informe completo con todas las tablas de evidencia:
`https://claude.ai/code/artifact/e49540e6-e032-4767-843d-d890504e34ff`.

Esta fase sustituye al auto-etiquetado como fuente de verdad del TFG: la
variable a predecir pasa a ser la **fatiga mental percibida** medida con una
escala visual analógica, no un umbral derivado de la propia señal.

### El protocolo, resuelto (información del equipo del estudio, 2026-08-23)

La pregunta que el informe dejó abierta —si el EEG canónico se grabó antes o
después de los ejercicios de fatiga (FE1/FE2) del tracker— **no era
determinable con los archivos** y la respondió el equipo del estudio:

> El EEG (todas las condiciones, reposo y tareas ejecutivas) se registra en la
> misma sesión, **inmediatamente después de completar el VAS**. No hay ninguna
> tarea ni intervención entre el VAS y el EEG que busque inducir fatiga. El VAS
> refleja un nivel de fatiga relativamente estable, ligado al momento del curso
> escolar y no a las tareas de esa sesión; el EEG se registra a continuación
> para comprobar si ese nivel ya existente se refleja en la señal.

Se registra aquí como **testimonio del equipo, no como hallazgo de los datos**:
los nombres canónicos no llevan marca PRE/POST (0 de 4574), los
`_intervalMarker.csv` de EmotivPro están vacíos (400 de 400 muestreados) y las
marcas temporales del formulario son transcripción en bloque en C2, C4 y C7
(desfases de +60 a +160 min), así que ningún archivo del dataset lo confirma
ni lo desmiente. Si en la memoria hace falta respaldarlo, la fuente es el
equipo, no el dataset.

Tres consecuencias de diseño:

1. **VAS y EEG son concurrentes.** La etiqueta describe el estado del sujeto
   durante el registro, no un estado futuro. Es un diseño de validez
   concurrente, limpio, y así debe presentarse.
2. **El VAS no lo inducen las tareas de la sesión**, luego la etiqueta es
   constante dentro de la sesión por construcción y no por simplificación
   nuestra. Replicarla a las 7 grabaciones del momento es correcto.
3. **La etiqueta está ligada al momento del curso escolar.** Esto convierte
   `medicion` (M2–M6) en un predictor fuerte del VAS **sin usar EEG**, y
   obliga a las líneas base de más abajo.

### Regla de enlace

Clave de unión, validada con tres fuentes independientes:

```
(centro_efectivo, fitbit_base, medicion)   →   fila del VAS
    donde fitbit_base = int(sujeto_local[1:])
```

Evidencia de que la clave es correcta:

| Comprobación | Resultado |
|---|---|
| Fecha de la sesión EEG == fecha del VAS | **635 coinciden, 1 discrepa**, 20 sin fecha en el VAS |
| `sexo_filename` == `GÉNERO` (por sujeto) | 172 de 174; las 2 excepciones son F111 y F115 |
| VAS del tracker `RESUMEN DATOS(TOTAL).csv` == VAS del CSV, por F-code | 97.8 % |
| `persona_uid` construido vs. personas reales (`CÓDIGO`) | 174 = 174 |

La única fecha que discrepa de verdad es `C3_F068_M5` (EEG el 2025-04-30, VAS
el 2025-05-02, dos días después). Las otras 20 "discrepancias" son filas del
VAS sin `Marca temporal`, y por eso `concordancia_fecha` tiene tres valores
—`coincide`, `discrepa`, `desconocida`— en vez de ser un booleano: mezclarlas
daría una tasa de error inflada por 20.

El mapeo `GÉNERO` → sexo no está documentado en el archivo; se dedujo cruzando
con `sexo_filename`: **1 = XX, 2 = XY** (85 de 86 y 87 de 88 sujetos
respectivamente). El valor 0 aparece en 7 grabaciones y se trata como
desconocido.

> **La hipótesis de partida era que la pulsera se reutiliza entre centros.**
> No es así: solo la 116 aparece en dos centros y resulta ser la misma persona
> registrada dos veces. La reutilización real es **dentro de un centro y a lo
> largo del tiempo** — y eso es lo que produce la serie 2000+.

**La serie 2000+ es el segundo ocupante de la pulsera**, no un sujeto sin
emparejar. Confirmado por tres vías: el tracker los lista como filas propias
(`F2094`, `F2162`, `F2166`), las fechas del VAS del ocupante 2 coinciden
exactamente con las de sus sesiones EEG, y el sexo del nombre de archivo
cambia en el momento exacto en que cambia el ocupante (`C1_F162` es XX en M2
y XY de M3 en adelante).

Hay **10 pulseras con dos ocupantes** (C1/94, C1/162, C2/166, C5/12, C6/33,
C8/41, C8/42, C8/43, C8/44, C8/47), pero en 4 de ellas uno de los dos solo
aparece en M1, que el manifest no cubre. Quedan **6 `sujeto_id` que contienen
dos personas distintas con EEG**: `C1_F162`, `C2_F166`, `C6_F033`, `C8_F041`,
`C8_F042`, `C8_F043`. En `C6_F033` los dos ocupantes **alternan** momentos
(M3 y M5 uno, M4 el otro), así que ni siquiera vale partir por un corte
temporal.

El `join` debe hacerse **siempre por `(sujeto_id, medicion)`**. Unir por
sujeto asignaría el VAS de una persona a las grabaciones de otra.

> Dentro de M2–M6 la clave `(centro, pulsera, medición)` es única, así que
> resuelve el ocupante sin ambigüedad. La **única** colisión de dos personas
> en la misma clave está en M1 (C8, pulsera 44), fuera del rango que cubre el
> manifest. `normalizar_vas` avisa de ella y `construir_etiquetas` solo falla
> si una clave ambigua llega a tener EEG.

### Claves canónicas propuestas

| Clave | Definición | Cardinalidad | Para qué |
|---|---|---|---|
| `grabacion_uid` | `nombre_nuevo` (equivalente: `md5`) | 4453 | fila de la matriz RF; grupo de ventanas de la CNN |
| `sesion_uid` | `<sujeto_id>_<medicion>` | **653** | **unidad de etiqueta** — n real del estudio |
| `sujeto_id` | `<centro>_F<nnn>` (el del manifest) | 168 | clave del EEG; **no es una persona** en 6 casos |
| `persona_uid` | `<centro>_<fcode>`, con el `fcode` del tracker que conserva el prefijo `2` del segundo ocupante (`C1_F094` vs `C1_F2094`) | **174** | **unidad de agrupación** para splits y bootstrap |

Regla: agrupar siempre por `persona_uid`, etiquetar siempre por `sesion_uid`.
`sujeto_id` solo sirve para localizar archivos.

### Las correcciones de metadata

Ninguna toca la señal; son ediciones de metadata y **no exigen reprocesar EEG**.
Cada una vive en `CORRECCIONES_MANIFEST` / `CORRECCIONES_VAS` con su evidencia
escrita y su número esperado de filas, y se aplica comprobando ese número: si
el manifest o el VAS cambian y una corrección deja de cuadrar, el módulo para
en vez de aplicarla a medias. Un test fija que ninguna corrección pueda
añadirse sin dejar escrito por qué.

1. **`F111` ↔ `F115` intercambiados en el CSV del VAS.** Dos fuentes
   independientes contradicen al CSV en los 6 momentos de ambos sujetos:
   (a) el VAS da `GÉNERO=1` (XX) a la pulsera 111, pero sus EDF son XY en todas
   las mediciones, y al revés con la 115 — son los **dos únicos sujetos
   discordantes de 174**; (b) el tracker, indexado por F-code y no por pulsera,
   da a F111 la serie `[60,30,20,15,20,20]` y a F115 la `[40,60,40,40,40,65]`,
   exactamente al revés que el CSV. Afecta a 8 sesiones etiquetadas.
2. **41 grabaciones archivadas en C3 son la M5 de seis sujetos de C4**
   (`F125`, `F128`, `F130`, `F131`, `F135`, `F136`). Evidencia convergente:
   están fechadas el 2025-04-28, que es la jornada M5 de C4 (la de C3 es el
   04-30); esos seis existen en C4 con M3/M4/M6 **pero sin M5**; en C3 no
   tienen ninguna otra medición, o sea que `C3_F125` y compañía son sujetos
   fantasma que solo existen por este archivado; el sexo coincide en los seis;
   y las 41 filas ya venían marcadas con `a_revisar=True` y `centro_filename`
   vacío.
3. **`sub-C3F103_ses-M3_task-PASAT_eeg.edf` es de C2_F103.** Una sola
   grabación. `C2_F103` tiene la M3 del 2025-01-09 con seis tareas y le falta
   **exactamente PASAT**; esta es un PASAT del mismo día y el mismo sexo.
   Juntas completan el protocolo de 7. La fecha por sí sola no habría bastado
   (C3 también midió M3 ese día): lo decisivo es la tarea que falta.

Las correcciones 2 y 3 **no son cosméticas: eliminan fuga entre splits.**
`C3_F128` estaba en train y `C4_F128` en val; `C3_F131` en train y `C4_F131`
en test; `C3_F103` en test y `C2_F103` en val. Eran la misma persona a los dos
lados de la partición. No llegó a manifestarse porque esas grabaciones no
tenían etiqueta —el enlace fallaba justo por estar mal archivadas—, pero en
cuanto se corrige el centro sin corregir el split, la fuga es real.
`construir_etiquetas` lo comprueba y falla si vuelve a ocurrir.

> **Corrección de una afirmación anterior.** Un borrador de este documento
> listaba `C1_F157`, `C8_F155` y `C5_F163` como "recuperaciones menores"
> pendientes. **Es incorrecto y no debe recuperarse.** En los tres casos hay
> una fila del VAS con ese número de pulsera en otro centro, pero el sexo y la
> fecha la contradicen a la vez: `C1_F157` es XX en el EEG y XY en el VAS de
> C7, con fechas 2024-12-03 vs 2024-11-28; `C8_F155` es XX vs XY, 2024-12-04
> vs 2024-11-28; `C5_F163` es XY vs XX, 2025-03-07 vs 2025-03-20. Son
> personas distintas que llevaron la misma pulsera en centros distintos, y se
> quedan sin etiqueta a propósito.

### Recuento final

Cifras medidas el 2026-08-23 con las correcciones aplicadas.

| | Valor |
|---|---|
| Sesiones EEG totales | 670 |
| Sesiones con VAS | 656 (14 sin etiqueta) |
| **Sesiones con VAS y ventanas (`sesion_uid`)** | **653** |
| Personas (`persona_uid`, y verificado contra `CÓDIGO`) | **174** de 178 |
| `sujeto_id` con etiqueta | 168 (6 de ellos contienen 2 personas) |
| Grabaciones etiquetadas y procesadas | **4453** |
| Ventanas de 4 s etiquetadas | **274 718** |

Cadena de pérdidas: 4574 en el manifest → 4543 procesadas con éxito (31
errores: 30 en ICA, 1 en bandas) → 4484 con etiqueta VAS → 4453 que además
sobrevivieron al procesado.

**Reparto sobre el `splits.csv` actual — viable tal cual:**

| Split | Ventanas | Grabaciones | Sesiones | Personas | `sujeto_id` |
|---|---|---|---|---|---|
| train | 192 315 | 3117 | 458 | 121 | 118 |
| val   |  37 683 |  611 |  89 |  25 |  24 |
| test  |  44 720 |  725 | 106 |  28 |  26 |

Con las correcciones aplicadas, **ninguna persona cruza de split** — y ahora
está comprobado sobre el `CÓDIGO` real del participante, no sobre una clave
derivada del centro (ver "Las correcciones de metadata"). No hace falta
rehacer la partición.

> **274 718 ventanas, pero 653 etiquetas independientes.** Todas las ventanas
> de una sesión comparten el mismo valor de VAS al milímetro. El n efectivo
> para cualquier intervalo de confianza es 653 (**106 en test**), no 274 718.
> Reportar métricas por ventana produce intervalos falsamente estrechos.

### Unidad de análisis y tratamiento de la condición

**Decisión (2026-08-23): opción B — todas las condiciones, con la tarea como
variable explícita** (feature categórica en el RF, embedding en la CNN) como
modelo principal. Las alternativas A (restringir a reposo OC+OA) y C (un
modelo por condición) quedan como análisis complementarios si hay presupuesto
de horas.

Por qué B no filtra la etiqueta: cada sesión aporta las 7 condiciones **con el
mismo VAS**, así que la condición está balanceada dentro de cada sesión y es
ortogonal a la etiqueta por construcción — no puede predecirla. Lo que hace es
darle al modelo el contexto para interpretar la potencia de banda **relativa a
la tarea**, en vez de obligarle a promediar sobre siete estados espectrales
muy distintos. 622 de 671 sesiones tienen las 7 grabaciones, así que el
desbalance residual es despreciable.

Niveles: `MSIT`, `OA`, `OC`, `PASAT`, `PVSAT`, `PVTB`, `STROOP`. Codificación
ordinal por posición en el protocolo (`OC→1 … PVSAT→7`) para el RF, que además
la hace intercambiable con `orden_en_sesion` de la Fase 11.

### Líneas base obligatorias

Medidas sobre las 653 sesiones etiquetadas:

| Línea base | R² | Qué recupera |
|---|---|---|
| Media global | 0.000 | nada, por definición |
| Media por `medicion` | **0.015** | el calendario escolar |
| Media por `centro` | 0.016 | el efecto de sitio |
| Media por `persona_uid`, en muestra | 0.600 | "quién eres", optimista |
| **Media por `persona_uid`, leave-one-out** | **0.260** | **la que hay que batir** |

> **Esto corrige una previsión que había escrito en este documento.** Yo
> esperaba que `medicion` fuese un predictor fuerte, porque el equipo describe
> el VAS como un nivel ligado al momento del curso escolar. **Medido, no lo
> es:** explica el 1.5 % de la varianza, y las medias por momento van de 39.8
> (M3, M5) a 46.4 (M6) sobre una escala de 0 a 100. La descripción del equipo
> es correcta como justificación del diseño, pero en estos datos el momento no
> mueve el VAS de forma apreciable. Lo que sí lo mueve es la persona.

La decisión de **no meter `medicion` como feature del modelo principal se
mantiene**, pero ahora se sabe que es barata: no se renuncia a nada medible.
El argumento pasa de "es un atajo peligroso" a "es una variable de diseño, no
un predictor, y el modelo debe explicarse sin ella". Sigue usándose como línea
base y como eje de estratificación en el reporte.

Dos precisiones sobre la tabla:

- **La línea base honesta es la de leave-one-out (R² = 0.260)**, no la de
  0.600. Esta última ajusta la media de cada persona sobre las mismas sesiones
  con las que luego se evalúa, así que está inflada. Con una media por persona
  estimada sobre sus *otras* sesiones —que es lo que un modelo tendría
  disponible— el listón real es 0.26.
- **"Media por persona + momento" no es estimable.** Hay exactamente una
  sesión por celda `(persona, momento)`, así que esa media *es* el valor y
  daría R² = 1 trivialmente. No es una línea base, es la etiqueta.

### Varianza de la etiqueta

ICC(1) sobre las 653 sesiones etiquetadas, agrupando por `persona_uid`:

| | |
|---|---|
| ICC(1) | **0.457** |
| σ² entre personas | 228.9 |
| σ² dentro de persona | 272.2 |
| Personas con ≥3 momentos | **157** de 174 |

**El 54 % de la varianza es intra-persona**, es decir, hay señal real que
capturar más allá de la identidad del participante. Es el mejor de los
escenarios razonables. Lo que no se sabe es **qué** la mueve: no es el momento
del curso (R² = 0.015) ni el centro (0.016), así que es variación de sesión a
sesión que estas variables de diseño no explican — exactamente el hueco que le
toca llenar al EEG, si es que puede.

Distribución: media 42.7, mediana 40, desviación 22.4, rango completo 0–100.
Reparto de sesiones por persona: 12 personas con 1, 5 con 2, 25 con 3, 104 con
4 y 28 con 5.

El VAS es **ordinal de facto**: el 76.7 % de las respuestas son múltiplos de 10
y el 99.8 % múltiplos de 5. Consecuencia práctica: además del MAE conviene
reportar correlación de Spearman, y un análisis secundario en 3 clases
(tercios calibrados **solo sobre train**) es defendible y probablemente más
interpretable para la Facultad de Ciencias del Deporte que un continuo.

### Protocolo de evaluación

- **Predicción a nivel de sesión:** promediar las predicciones de las ventanas
  de una `sesion_uid` y evaluar contra su único VAS. Reportar también el nivel
  grabación como diagnóstico, nunca como métrica principal.
- **Validación cruzada agrupada por `persona_uid`** (`GroupKFold`), nunca
  aleatoria por ventana.
- **Intervalos por bootstrap agrupado por persona**, n = 28 personas en test
  (26 `sujeto_id`).
- Cualquier normalización, umbral o imputación se ajusta **solo con train**.

### Comprobación de sensibilidad a 128 Hz

Las 100 grabaciones a 128 Hz son el 2.2 % del dataset pero el **27 % de la M2
superviviente** (M2 solo sobrevive en C1, C6 y C8). La comprobación de que el
resampleo de la Fase 1 no introduce un efecto sistemático hay que hacerla
**dentro de M2**; sobre el dataset entero el efecto queda diluido y no se ve.

### Duraciones: no hay ninguna grabación inservible

La duración mediana de 127 s por grabación no es un defecto ni contradice los
~15 min de sesión: son cosas distintas. La sesión completa de 7 tareas dura
15.5 min de mediana; cada grabación individual dura lo que dura su tarea.

| Condición | Duración mediana |
|---|---|
| MSIT | 69 s |
| STROOP | 68 s |
| OA / OC | ~122 s |
| PASAT / PVSAT / PVTB | ~187 s |

La más corta del dataset son 25 s → 15 s útiles tras el recorte de 5+5 s de la
Fase 2 → **6 ventanas**. **Cero grabaciones quedan por debajo del mínimo** para
el epocado de 4 s.

### Protección de datos

La columna `CÓDIGO` del CSV del VAS son **iniciales + fecha de nacimiento de
menores**: dato personal pseudonimizado, no anonimizado. No debe aparecer en
`labels.csv`, ni en la matriz de modelado, ni en el repositorio. Si hace falta
trazabilidad hacia atrás, guardar un hash con sal, con la sal fuera del repo.
El enlace ya está hecho por `(centro, fitbit, medicion)`, así que el `CÓDIGO`
no es necesario en ningún artefacto posterior.

### Firma y salida

```python
from eeg_library import (corregir_manifest, construir_etiquetas,
                         expandir_a_grabaciones, unir_etiquetas)

manifest = corregir_manifest(pd.read_csv(RUTA_MANIFEST, encoding="utf-8-sig"))
etiquetas, meta = construir_etiquetas(vas, manifest, splits, log=log)
grabaciones = expandir_a_grabaciones(etiquetas, manifest, log=log)
matriz = unir_etiquetas(features, grabaciones)

# etiquetas   : una fila por sesión EEG (670), INCLUIDAS las 14 sin VAS, con
#               vas=NaN y etiquetada=False. Columnas: sesion_uid, sujeto_id,
#               centro, fitbit_base, medicion, n_grabaciones, n_corregidas,
#               fecha_eeg, persona_uid, ocupante, vas, fecha_vas, sexo_vas,
#               etiquetada, concordancia_fecha, split, n_grabaciones_ok,
#               n_ventanas.
# grabaciones : una fila por grabación (4574), con la etiqueta de su sesión
#               replicada, más condicion, corregida, procesada y n_ventanas.
# meta        : recuentos y trazabilidad (correcciones aplicadas, sesiones sin
#               VAS, concordancia de fechas).
```

Tres decisiones que no se ven en la firma:

- **`corregir_manifest` es idempotente**, para que el flujo normal —corregir
  una vez y pasar el resultado a las dos funciones siguientes— no intente
  aplicar dos veces lo mismo.
- **`persona_uid` queda a `NaN` en las sesiones sin VAS**, en vez de heredar el
  `sujeto_id`. En una pulsera con dos ocupantes el número por sí solo no dice
  quién la llevaba, y el caso es real: `C1_F162_M5` tiene EEG pero no VAS, y su
  `sujeto_id` es el mismo que el del ocupante 1, que es otra persona. Con NaN,
  cualquier `groupby` por persona las descarta en vez de mezclarlas.
- **`unir_etiquetas` une por `archivo_origen`, no por sujeto ni por sesión.**
  Es la única clave que sobrevive a las correcciones de centro: la matriz de
  features lleva el `sujeto` y el `split` que tenía el manifest **antes** de
  corregirlo (`C3_F125`, que ya no existe). Si además la matriz trae `centro` o
  `split`, se sustituyen por los corregidos informando de cuántos valores
  cambian. Hay un test de regresión para exactamente esto.

### Lo que queda por hacer

1. **Matriz RF** — agregación por grabación (media y desviación de cada banda a
   lo largo de las ventanas), agregando **por región** en lugar de por canal
   para que la máscara de la Fase 11.5 degrade en vez de romper.
2. **Tensor CNN** — normalización z por canal ajustada **solo sobre train**, y
   un canal extra de máscara en vez de poner a cero los canales inválidos
   (poner a cero introduce un sesgo que la red aprende).
3. **Líneas base primero, modelo después.** Ya están medidas (tabla de arriba):
   el listón es R² = 0.260 sobre la media por persona leave-one-out.

---

## Fase 13 — Matriz de features del Random Forest

**Archivo:** `src/eeg_library/procesado/matrix.py`
**Funciones públicas:** `agregar_por_region`, `construir_matriz_rf`,
`columnas_de_features`
**Salida:** `C:\Users\migue\Desktop\TFG\EEG-Data\processed\matriz_rf.parquet` — **4543 × 91**
(13 columnas de identidad + `vas` + **77 features**)

Convierte la matriz por época de la Fase 11.5 (280 339 × 139) en una matriz por
grabación lista para `RandomForestRegressor`, con la etiqueta ya unida.

### Composición de las 77 features

| Bloque | Cuántas | Qué es |
|---|---|---|
| `<region>_<banda>_logabs_{media,sd}` | 32 | log₁₀ de la potencia absoluta, promediada sobre los canales **válidos** de la región |
| `<region>_<banda>_rel_{media,sd}` | 32 | potencia relativa, ídem |
| `theta_frontal_log_{media,sd}`, `alfa_posterior_log_{media,sd}` | 4 | componentes del índice de fatiga |
| `indice_fatiga_{media,sd}` | 2 | el biomarcador theta/alfa de la Fase 9 |
| `n_validos_<region>` | 4 | canales buenos por región |
| `n_canales_buenos`, `n_ventanas` | 2 | tamaño y calidad de la grabación |
| `condicion_orden` | 1 | posición en el protocolo (OC=1 … PVSAT=7) |

Regiones: `frontal` (AF3, AF4, F3, F4, F7, F8), `frontocentral` (FC5, FC6),
`temporal` (T7, T8), `posterior` (P7, P8, O1, O2).

### Cuatro decisiones

- **Se agrega por región y sobre los canales válidos.** Un canal descartado
  trae valores fuera de escala por factores de hasta 7.9× (Fase 11.5); al
  promediar solo los válidos, un electrodo seco degrada la precisión de su
  región en vez de meter basura en la matriz. Bajar de 112 columnas de potencia
  a 64 importa cuando hay 653 etiquetas independientes.
- **La potencia absoluta se agrega en log₁₀.** Los valores del dataset van de
  4e-38 a 3e-7: una media aritmética sobre las ventanas la decidiría la ventana
  más ruidosa. La media del logaritmo es la media geométrica, que es la forma
  estándar de promediar potencia espectral. La relativa ya está en [0, 1] y se
  agrega tal cual.
- **Media y desviación típica a lo largo de las ventanas.** La media es el
  nivel; la desviación es cuánto fluctúa la grabación. Resultó ser la parte
  interesante (ver más abajo).
- **Una región sin canales válidos da NaN, no un valor inventado.** Pasa en el
  2.9 % de las grabaciones, siempre en las regiones de dos electrodos (temporal
  2.3 %, frontocentral 0.6 %); frontal y posterior nunca se quedan vacías. Son
  2096 celdas, el **0.6 %** de la matriz. `RandomForestRegressor` admite NaN
  desde scikit-learn 1.4 (aquí 1.7.2), así que imputar sería inventar datos.

### `columnas_de_features` es la definición de X

Todo lo que no devuelve esa función no entra al `fit`. Existe como función y no
como convención mental porque la exclusión de `medicion` (Fase 12) es una
decisión de diseño que hay que poder comprobar con un test, no recordar. Hay un
test que falla si `medicion` o `vas` aparecen en X.

### Prueba de humo y diagnóstico (2026-08-23)

Sobre **train+val**, `GroupKFold(5)` por `persona_uid`, evaluando a nivel de
sesión (547 sesiones, 146 personas). **Test no se ha tocado.**

| | R² | MAE |
|---|---|---|
| Media global | 0.000 | 18.01 |
| Media por persona (leave-one-out) | **0.299** | 14.25 |
| Random Forest (300 árboles, `min_samples_leaf=5`) | **−0.015** | 18.20 |

**El Random Forest no bate a la media global.** Spearman entre predicción y
VAS: 0.043. Las `feature_importances_` salen planas (~0.02 cada una de las 77),
que es la firma de un bosque partiendo ruido.

**La matriz no está rota** — hay dos controles positivos que lo descartan:

| Control | Resultado | Azar |
|---|---|---|
| Predecir la **condición** desde las mismas features (7 clases, sin `condicion_orden`) | **0.608** | 0.143 |
| Identificar a la **persona** (129 clases) | **0.383** | 0.008 |

> **Corrección (2026-08-26).** La fila de la condición estaba inflada por
> circularidad: `condicion_orden` es la condición codificada uno a uno y
> `columnas_de_features()` la devuelve, así que el clasificador leía la
> respuesta de una columna que era la respuesta. Con ella, 0.9795; sin
> ella, 0.6084 sobre las 7 condiciones (azar 0.143). El control sigue
> siendo positivo —4.3x el azar— pero la cifra correcta es 0.608.

Las features llevan muchísima información EEG real y estructurada. Lo que no
llevan, o llevan muy poco, es información sobre el VAS.

### El tamaño real del efecto

Correlación de Spearman de cada feature con el VAS a nivel de sesión, separando
la variación **entre** personas de la **intra**-persona (centrando cada
variable en la media de su participante):

- **|ρ| intra máximo de las 77 features: 0.141.**
- 15 de 77 features con p < 0.05 intra-persona, frente a 3.9 esperadas por
  azar. Hay algo por encima del azar, pero los tamaños de efecto son ρ ≈ 0.10–0.14,
  o sea R² de un 1–2 % por feature. Un RF con 77 columnas mayormente ruidosas y
  n = 547 se ahoga: el R² ≈ 0 es coherente con esto, no lo contradice.

**El biomarcador clásico no funciona en esta cohorte.** Es un resultado
negativo limpio y publicable en la memoria:

| Feature | ρ bruto | ρ intra-persona |
|---|---|---|
| `indice_fatiga_media` (theta/alfa) | −0.091 (p=0.033) | **+0.004 (p=0.93)** |
| `theta_frontal_log_media` | −0.145 (p=0.001) | −0.030 (p=0.49) |
| `alfa_posterior_log_media` | −0.033 (p=0.44) | −0.036 (p=0.40) |

El índice theta/alfa correlaciona **cero** con la fatiga percibida dentro de
persona. Su correlación bruta de −0.091 es un artefacto entre-sujetos, y
además va en **sentido contrario** al que predice la literatura (más theta
frontal → *menos* fatiga reportada).

### El único patrón que asoma: la variabilidad, no el nivel

De las 15 features con señal intra-persona, **la mayoría son `_rel_sd`** — la
desviación de la potencia relativa a lo largo de las ventanas, no su media:

| Feature | ρ intra | tras residualizar |
|---|---|---|
| `frontal_delta_rel_sd` | +0.141 | +0.140 (p=0.001) |
| `posterior_delta_rel_sd` | +0.136 | +0.136 (p=0.002) |
| `frontal_alfa_rel_sd` | +0.116 | +0.115 (p=0.008) |
| `frontal_beta_rel_sd` | +0.113 | +0.111 (p=0.010) |
| `posterior_alfa_rel_sd` | +0.110 | +0.110 (p=0.011) |

Todas con **el mismo signo**: más fluctuación del espectro a lo largo de la
grabación → más fatiga reportada. Es coherente en frontal, frontocentral y
posterior; la temporal (2 electrodos, la región más ruidosa) es la excepción.
**No es un artefacto de longitud ni de calidad**: la correlación sobrevive
intacta a residualizar contra `n_ventanas` y `n_canales_buenos`. Un índice
compuesto (media z de las 16 `_rel_sd`) da **ρ intra = +0.157, p = 0.0003,
n = 537**.

> **Es una pista, no un resultado.** Ese compuesto se eligió **después** de
> mirar train+val, así que reportarlo sobre los mismos datos sería circular.
> Hay que confirmarlo en test una sola vez, o con selección dentro de la
> validación cruzada. Y aunque se confirme, ρ = 0.157 son unos 2.5 puntos de
> R²: es un efecto pequeño, y hay que presentarlo como tal.

### Lo que esto implica para el siguiente paso

Afinar hiperparámetros del RF sobre 77 columnas no va a arreglar un ρ máximo de
0.14. Lo que tiene sentido:

1. **Agregar a nivel de sesión** (547 filas, no 3728): es donde vive la
   etiqueta, y promediar las 7 condiciones reduce ruido.
2. **Reducir drásticamente la dimensión**, con la selección **dentro** de la
   validación cruzada para no repetir la circularidad de arriba.
3. **Modelar la desviación respecto a la media de la persona**, que es donde
   está la señal — con la advertencia de que para una persona nueva esa media
   no se conoce, así que la tarea "en frío" es genuinamente más difícil.
4. **Reportar el resultado negativo del índice theta/alfa como hallazgo.** Un
   TFG que construye un pipeline riguroso y mide que el biomarcador clásico no
   sigue a la fatiga autopercibida en esta cohorte es un TFG defendible.

---

## API pública (`__init__.py`)

```python
from eeg_library import (
    cargar_eeg, recortar_registro, aplicar_filtros,
    evaluar_calidad, aplicar_car,
    entrenar_ica, identificar_componentes_oculares, eliminar_artefactos_oculares,
    crear_epocas, calcular_psd, extraer_bandas,
    # Fase 11 — procesamiento por lotes
    cargar_tareas, procesar_archivo, procesar_lote, consolidar_features,
    # Fase 11.5 — máscara de validez de canal
    anotar_validez_de_canales, columnas_de_validez,
    # Fase 12 — dataset supervisado (etiqueta VAS)
    normalizar_vas, corregir_manifest, construir_etiquetas,
    expandir_a_grabaciones, unir_etiquetas,
    # Fase 13 — matriz de features del Random Forest
    agregar_por_region, construir_matriz_rf, columnas_de_features, REGIONES,
)
```

Versión actual: `0.11.0`

### Procesamiento del dataset completo (Fase 11)
```python
from eeg_library import cargar_tareas, procesar_lote, consolidar_features

tareas = cargar_tareas(
    r"C:\Users\migue\Desktop\TFG\EEG-Data\metadata\manifest_canonical.csv",
    r"C:\Users\migue\Desktop\TFG\EEG-Data\dataset_final",
    r"C:\Users\migue\Desktop\TFG\EEG-Data\metadata\splits.csv",
)
log, resumen = procesar_lote(
    tareas,
    dir_salida=r"C:\Users\migue\Desktop\TFG\EEG-Data\processed",
    dir_ventanas=r"C:\Users\migue\Desktop\TFG\EEG-Data\ventanas",
    n_procesos=10,
)
features = consolidar_features(r"C:\Users\migue\Desktop\TFG\EEG-Data\processed")

# Fase 11.5 — máscara de validez de canal
anotadas = anotar_validez_de_canales(features, log)
anotadas.to_parquet(r"C:\Users\migue\Desktop\TFG\EEG-Data\processed\matriz_modelado.parquet", index=False)
```

> Debe ejecutarse desde un script con guarda `if __name__ == "__main__":` —
> los workers arrancan con *spawn* en Windows y reimportan el módulo.

### Flujo de uso típico (Fases 1-9)
```python
from eeg_library import (
    cargar_eeg, recortar_registro, aplicar_filtros,
    evaluar_calidad, aplicar_car,
    entrenar_ica, identificar_componentes_oculares, eliminar_artefactos_oculares,
    crear_epocas, calcular_psd, extraer_bandas,
)

raw,      meta_carga         = cargar_eeg("archivo.edf")
raw_rec,  meta_rec           = recortar_registro(raw)
raw_filt, meta_filt          = aplicar_filtros(raw_rec)
good_channels, excluidos, _  = evaluar_calidad(raw_filt)
raw_ref,  meta_ref           = aplicar_car(raw_filt, good_channels)

ica, meta_ica                = entrenar_ica(raw_ref, good_channels)
componentes, diagnostico, _  = identificar_componentes_oculares(raw_ref, ica)
raw_limpio, meta_ica_aplicada = eliminar_artefactos_oculares(raw_ref, ica, componentes)

epochs, meta_epocado         = crear_epocas(
    raw_limpio, sujeto="S07", condicion="C3", sesion=4,
    archivo_origen="archivo.edf",
)

spectrum, meta_psd           = calcular_psd(epochs)
features, meta_bandas        = extraer_bandas(
    spectrum, canales_validos=good_channels,
)
```

> `canales_validos=good_channels` no es opcional en la práctica: sin él, el
> índice de fatiga se calcula también sobre los canales que descartó la Fase 4
> y acaba midiendo la calidad del contacto del casco (ver Fase 9).

---

## Tests

| Archivo                     | Tests | Cubre                                      |
|-----------------------------|-------|--------------------------------------------|
| `tests/test_reader.py`      | 25    | Armonización de sfreq a `TARGET_SFREQ`: archivo ya a 256 Hz (no se toca), archivo a 128 Hz (se resamplea, duración preservada), rechazo `ValueError` por debajo de `MIN_SFREQ` (127 Hz y 100 Hz), no escritura/modificación del `.edf` fuente, formato de metadata general, `FileNotFoundError`. Tipado de canales y escala CQ, parametrizados sobre los **tres esquemas de export reales** (60 con guion bajo, 67 y 71 con punto): solo los 14 de `CANALES_EPOC` quedan como `'eeg'` y el resto como `'misc'`, `n_canales` frente a `n_canales_eeg`, reversión de la escala CQ solo cuando llegó escalada, y regresión end-to-end de que un CQ bajo excluye el canal en `evaluar_calidad` bajo las dos convenciones de escala |
| `tests/test_quality.py`     | 17    | Los 4 criterios (z-score robusto + caso MAD=0), métricas, advertencias, canales no EPOC |
| `tests/test_referencing.py` | 10    | CAR correcta, copia, canales excluidos, metadatos |
| `tests/test_filtering.py`   | 22    | Recorte de márgenes, notch a 50 Hz, bandpass 1–30 Hz, DC, Nyquist, canales no-EEG, metadatos |
| `tests/test_artifacts.py`   | 6     | Detección/eliminación de componente ocular, ausencia de exclusión sin ocular claro, diagnóstico de otras categorías sin excluirlas, verificación de rango (n_components = n_good_channels - 1), validaciones de entrenar_ica |
| `tests/test_epoching.py`    | 12    | Nº de épocas y solape, tamaño de ventana en muestras (1024 a 256 Hz), descarte del resto final, validaciones de duracion/solape, metadata de origen por época, no-mutación del Raw, formato de metadata general |
| `tests/test_pipeline_integracion.py` | 4 | Flujo completo Fases 2-6 encadenado en el orden corregido (recortar → filtrar → calidad → CAR → ICA → ICLabel mockeado → eliminación); regresión explícita: un canal con deriva de baja frecuencia + transitorio de borde se excluye si se evalúa la calidad sobre el Raw crudo, y se acepta tras recortar/filtrar. Segunda regresión de integración (Fases 1→9, saltando ICA): partiendo de un Raw con CQ/telemetría/movimiento tipados como `'eeg'` —lo que devuelve MNE al leer el EDF— la matriz de features contiene exactamente 112 columnas de potencia y ninguna procedente de un canal no-EEG |
| `tests/test_batch.py`       | 18    | `cargar_tareas`: derivación de `orden_en_sesion` desde las marcas de tiempo (no desde el nombre de la condición), unión del split por sujeto, orden `<NA>` cuando la sesión no tiene timestamps, validación de columnas del manifest. `procesar_archivo`: features con 112 columnas de potencia y sin canales no-EEG, ventanas alineadas fila a fila con las features (forma, dtype float32, 4 s), columnas extra, e identificación de la fase en la que falla. `procesar_lote`: escritura de parquet/log/resumen, aislamiento de fallos (un archivo ilegible no tumba el lote), reanudación saltando lo ya hecho, generación de ventanas, y equivalencia entre ejecución en serie y en paralelo. `consolidar_features`: concatenación y detección de parquet con columnas incompatibles. **Único archivo de test que escribe EDF reales en disco** (`mne.export.export_raw`) en vez de mockear la lectura: los workers corren en otros procesos, donde un `monkeypatch` del proceso padre no existiría |
| `tests/test_spectral.py`    | 16    | Resolución en frecuencia (1 Hz), tamaño/solape de sub-ventana de Welch (256/128 muestras), nº de segmentos promediados, localización de un pico espectral conocido, rango 1–30 Hz (sin contenido por encima), banda configurable, forma de salida por canal y por época, preservación de metadata de origen, no-mutación de las épocas, validaciones de banda/ventana/solape, formato de metadata general |
| `tests/test_matrix.py`      | 17    | `agregar_por_region`: el promedio de región **ignora los canales inválidos** (con un AF3 contaminado a 1e-3 y marcado como malo, para verificar que no entra), agregación de la potencia absoluta en log₁₀ frente a la relativa sin transformar, NaN —y no un valor inventado— cuando una región se queda sin canales válidos, sin contagiar a las demás, mapa por canal como caso particular del mapa por región, y validación de `archivo_origen`. `construir_matriz_rf`: una fila por grabación con media y desviación a lo largo de sus ventanas y equivalencia con el agregado por época, `sd = 0` (no NaN) en una grabación de una sola ventana, codificación de la condición por su posición en el protocolo y error ante una condición desconocida, llegada de etiqueta y claves desde `labels_grabaciones.csv`, conservación de las grabaciones sin etiqueta con `vas` a NaN, y error si `labels` repite un archivo. `columnas_de_features`: **`medicion` no es una feature** aunque viaje en la matriz, ni la etiqueta ni ninguna clave de identidad lo son, todas las features son numéricas, y una prueba de extremo a extremo de que la matriz con NaN entra en un `RandomForestRegressor` sin imputar |
| `tests/test_labels.py`      | 25    | `normalizar_vas`: derivación de ocupante/`fitbit_base`/`persona_uid` a partir del número de pulsera (94 vs 2094), relleno a tres dígitos, traducción de `GÉNERO` con el 0 como desconocido, eliminación de la fila duplicada exacta, intercambio F111/F115 y fallo si el intercambio no cuadra por ambos lados, y aviso —no error— ante dos ocupantes que chocan en M1. `corregir_manifest`: reasignación de centro y recálculo de `sujeto_id`/`sesion_uid`, reunión de las 7 tareas de una sesión partida entre dos centros, fallo si el recuento no cuadra, idempotencia, y rechazo de `sujeto_local` en el rango del ocupante 2. `construir_etiquetas`: **cada ocupante recibe el VAS de su propio momento** (el test central del módulo), conservación de las sesiones sin VAS con `persona_uid` a NaN, los tres estados de `concordancia_fecha`, detección de la fuga entre splits de una persona con dos `sujeto_id`, error si una clave ambigua llega a tener EEG, recuento de ventanas desde el log, y que el `CÓDIGO` del participante no salga en ninguna tabla ni en la metadata. `unir_etiquetas`: regresión de que la etiqueta llega a las grabaciones mal archivadas pese a que la matriz de features lleva el `sujeto` sin corregir, invariancia del número de épocas, y errores por archivo desconocido o por falta de `archivo_origen`. Un test fija además que ninguna corrección pueda añadirse sin evidencia escrita |
| `tests/test_bands.py`       | 22    | `canales_validos`: equivalencia con el comportamiento anterior cuando es None, exclusión del clúster de un canal contaminado (con verificación de que inflaba theta_frontal y el índice), invariancia de las columnas de potencia por canal frente al conjunto de canales válidos, `ValueError` si un clúster queda entero excluido por calidad y distinción de ese error frente al de clúster ausente del espectro. Ausencia de banda gamma, columnas pot_abs/pot_rel por canal x banda (112), potencia localizada en la banda correcta ante una senoidal conocida, potencias relativas sumando 1 por canal, partición exacta de la potencia total entre bandas (sin solape en 4/8/13 Hz), composición de los clústeres frontal/posterior, cálculo de theta_frontal/alfa_posterior como media del clúster, índice de fatiga como cociente theta/alfa, errores por clúster o bandas theta/alfa ausentes, preservación de metadata de origen, no-mutación del espectro, formato de metadata general |

Todos los tests usan datos sintéticos generados con `mne.io.RawArray` y
`numpy.random` (sin necesidad de archivos EDF reales).

**Nota sobre `test_artifacts.py`:** ICLabel es una red entrenada sobre EEG
real; se comprobó manualmente que ni una inyección deliberada de pulsos
frontales lentos en datos sintéticos logra una clasificación fiable de
`"eye blink"`. Por eso los tests de `identificar_componentes_oculares` y
`eliminar_artefactos_oculares` mockean
`mne_icalabel.iclabel.iclabel_label_components` con matrices de
probabilidad controladas (vía `monkeypatch`), probando así la lógica de
selección/diagnóstico y no la precisión del clasificador. El test de
verificación de rango sí ejecuta ICA real, ya que no depende de ICLabel.

Ejecutar:
```bash
pytest tests/ -v
```

---

## Estructura de archivos relevante

```
src/eeg_library/
    __init__.py       # API pública (re-exporta los tres subpaquetes)
    preprocesado/     # Fases 1-6 — EDF crudo -> señal limpia
        reader.py         # Fase 1 — cargar_eeg (resampleo a TARGET_SFREQ, tipado de
                           #          canales por allowlist, escala CQ, set_montage)
        quality.py        # Fase 4 — evaluar_calidad, CANALES_EPOC, PREFIJOS_CQ,
                           #          buscar_canal_cq
        referencing.py    # Fase 5 — aplicar_car
        filtering.py      # Fases 2 y 3 — recortar_registro, aplicar_filtros
        artifacts.py      # Fase 6 — entrenar_ica, identificar_componentes_oculares,
                           #          eliminar_artefactos_oculares
    procesado/        # Fases 7-13 — señal limpia -> matriz de features
        epoching.py       # Fase 7 — crear_epocas
        spectral.py       # Fase 8 — calcular_psd
        bands.py          # Fase 9 — extraer_bandas, BANDAS_DEFECTO,
                           #          CLUSTER_FRONTAL, CLUSTER_POSTERIOR
        batch.py          # Fase 11 — cargar_tareas, procesar_archivo,
                           #          procesar_lote, consolidar_features
                           #          (orquesta preprocesado + procesado)
        features.py       # Fase 11.5 — anotar_validez_de_canales,
                           #          columnas_de_validez
        labels.py         # Fase 12 — normalizar_vas, corregir_manifest,
                           #          construir_etiquetas, expandir_a_grabaciones,
                           #          unir_etiquetas, CORRECCIONES_MANIFEST,
                           #          CORRECCIONES_VAS
        matrix.py         # Fase 13 — agregar_por_region, construir_matriz_rf,
                           #          columnas_de_features, REGIONES,
                           #          ORDEN_PROTOCOLO, COLUMNAS_IDENTIDAD
    modelado/         # Fase 14 en adelante — ver MODELADO.md
        modeling.py       # arnés de evaluación del Random Forest
tests/
    test_reader.py
    test_quality.py
    test_referencing.py
    test_filtering.py
    test_artifacts.py
    test_epoching.py
    test_pipeline_integracion.py
    test_spectral.py
    test_bands.py
    test_batch.py
    test_labels.py
    test_matrix.py
data/samples/         # Archivos EDF reales (no en repo, en .gitignore)
probar.py             # Script manual de prueba (no es parte de la librería)
```

---

## Notas técnicas

- **Python:** 3.10.11 (el proyecto especifica 3.11+ pero funciona en 3.10).
- **Dependencias clave:** MNE-Python, NumPy, SciPy, pytest, mne-icalabel, onnxruntime.
- **Entorno virtual:** `.venv` (creado en esta sesión con
  `pip install -e ".[dev]"` + `mne-icalabel` + `onnxruntime`; no en repo).
- **Rescalado CQ:** depende del export — el de 60 canales declara la unidad CQ
  como `µV` y MNE le aplica el factor de voltaje (hay que revertirlo); los de
  67 y 71 la declaran `n/a` y llegan ya en el rango [0, 4] correcto.
  `cargar_eeg` lo decide por el rango de los valores, no por el nombre del
  canal (ver Fase 1, "Escala de los canales CQ").
- **Montaje:** `cargar_eeg` asigna `standard_1020` a los canales EEG (Fase 6
  lo necesita para las topografías de ICLabel); los canales no-EEG ya están
  tipados como `'misc'` en ese punto, así que no reclaman posición.
- **Armonización de sfreq (Fase 1):** 100 de 4574 grabaciones válidas del
  dataset final (2.2 %) se grabaron a 128 Hz en vez de los 256 Hz
  nominales/esperados (medición M2, centros C1 y C8, 16 sujetos —
  probablemente un lote grabado con otra configuración de EmotivPro).
  `cargar_eeg` las
  resamplea a `TARGET_SFREQ` (256 Hz) con `raw.resample(..., npad="auto")`
  de MNE (FFT + anti-aliasing) inmediatamente tras la carga, para que las
  Fases 3, 7 y 8 (que derivan parámetros de `sfreq`) operen siempre sobre
  una frecuencia uniforme. El Nyquist de 128 Hz (64 Hz) ya está muy por
  encima de la banda 1-30 Hz de interés, así que no se pierde información
  real relevante para el análisis de fatiga.
- `probar.py` en la raíz es un script exploratorio temporal; actualmente
  prueba el filtro notch visual con `compute_psd()` de forma manual.
- **Epocado (Fase 7):** `crear_epocas` delega en `mne.make_fixed_length_epochs`
  en vez de reimplementar el troceado a mano; el descarte del resto final
  y el cálculo de eventos con solapamiento ya vienen resueltos ahí.
  `epochs.metadata` (pandas.DataFrame) es el mecanismo elegido para llevar
  sujeto/condición/sesión/archivo por época porque sobrevive a
  `mne.concatenate_epochs`, necesario en la Fase 11.
- **Análisis espectral (Fase 8):** `calcular_psd` usa `Epochs.compute_psd`
  de MNE en vez de `scipy.signal.welch` directamente; `probar.py` ya
  usaba `compute_psd()` de forma manual/exploratoria antes de esta fase,
  confirmando que es la vía idiomática del proyecto para PSD con MNE.