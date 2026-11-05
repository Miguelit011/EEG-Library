# Deep learning — paso 1: espectro fino frente a bandas (filtro barato)

> Plan escrito el 2026-10-29, **después** de la evaluación única en `test` de
> la línea (paso 6) y **antes** de ejecutar nada de este paso. Usa **solo
> `train`** (121 personas con etiqueta en las condiciones de la línea) y no
> carga ni `val` ni `test`. Es de desarrollo y no cambia ningún veredicto.

## Por qué se hace

Una CNN sobre señal cruda puede aprender, en principio, tres tipos de
información que las 56 `log10(pot_rel)` (14 canales × 4 bandas) no tienen:

1. **Detalle espectral dentro de las bandas.** Por ejemplo, que el pico de
   alfa se desplace a frecuencias más bajas con el tiempo en tarea (un
   marcador conocido de fatiga que una banda fija de 8–13 Hz no ve), o un
   cambio en la beta alta y no en la baja.
2. **Información espacial**: acoplamiento entre canales y filtros espaciales.
3. **Información temporal fina** dentro de los 4 s: forma de onda, ráfagas y
   no estacionariedad.

El punto 1 se puede medir casi gratis con la logística ya fijada, usando la PSD
de 1 Hz en lugar de las bandas. ShallowConvNet, el comparador natural de la
logística (PLAN 7.2), aprende sobre todo eso: potencia en bandas que elige
ella misma. Así que este paso sirve de **filtro**. Si el espectro fino no
mejora a las bandas, la vía por la que una CNN pequeña tiene más opciones de
ganar está vacía. Si mejora, se sabe en qué frecuencias, y eso orienta (y
permite interpretar) la CNN.

**Lo que este paso no demuestra.** Un resultado nulo no descarta los puntos 2
y 3. Solo baja la expectativa para la CNN. Se reporta así.

## Datos

- **Ventanas crudas:** `EEG-Data/ventanas/<archivo>.npy`, forma
  `(n_ventanas, 14, 1024)`, float32, 256 Hz. La ventana `i` del `.npy` es la
  fila `i` de `features_all.parquet` para ese `archivo_origen`
  (`procesado/batch.py`, que lo comprueba al generarlas).
- **Partición e identidad:** `matriz_rf.parquet`, como en la línea. Se toma la
  lista de archivos con `split == "train"` **antes** de abrir ningún `.npy`
  (con `assert`).
- **Condiciones:** PASAT, PVTB y OA (las de la H1). OC, como descriptiva. No
  entran ni PVSAT ni STROOP ni MSIT.
- **Filas, recorte y exclusiones:** las de `comun.cargar` y `comun.versiones`
  del paso 3 (mínimo de 10 ventanas, recorte a igual tiempo con la duración
  del OA de la sesión). Los canales excluidos por calidad se anulan tras el
  centrado, igual que en el paso 3.
- **Volumen:** ≈ 1 500 archivos, ≈ 7–8 GB de lectura del NVMe. La PSD de 1 Hz
  por ventana (≈ 150 000 ventanas × 420 valores, ≈ 250 MB en float32) se
  calcula **en memoria y no se guarda**.

## Las dos representaciones

| Id | Features por ventana | Nº |
|---|---|---|
| **B** (referencia, la del paso 3) | `log10(pot_rel)` por canal en delta, theta, alfa y beta, de `features_all` | 56 |
| **F** (espectro fino) | `log10(PSD(f) / Σ PSD(1–30 Hz))` por canal, en f = 1, 2, …, 30 Hz | 420 |

- **Cálculo de F:** la PSD se obtiene con `calcular_psd` de la librería (Fase
  8: Welch, sub-ventanas de 1 s con Hann y 50 % de solape, 1–30 Hz). Se
  construye un `mne.EpochsArray` desde el `.npy`. No se reimplementa Welch.
- **Relativa, como B:** se divide entre la potencia total de 1–30 Hz del canal
  y la ventana. Así B y F solo se diferencian en la resolución en frecuencia,
  no en la normalización.
- **Mismo tratamiento que B:** para entrenar, media lineal de la potencia
  relativa en cada tercio y después `log10`. Para puntuar ventanas, `log10`
  por ventana. En los dos casos se centra por grabación y se ponen a 0 los
  canales excluidos.

## Comprobación de integridad (antes de cualquier modelo)

1. **Alineación:** para cada archivo, `n_ventanas` del `.npy` es igual al
   número de filas de `features_all`. Los que no cumplan se excluyen y se
   cuentan. Si son más del 1 %, se para.
2. **Reproducción de las bandas:** sumando F por bandas (con las máscaras de
   `BANDAS_DEFECTO`) se recalcula `pot_rel` y se compara con la de
   `features_all`. Criterio: correlación ≥ 0.999 en cada una de las 56
   columnas. **Si no se cumple, se para y se investiga** antes de seguir: la
   comparación B frente a F solo tiene sentido si las dos salen de la misma
   señal.

## Modelo y validación (fijados antes de ejecutar)

- Logística con `StandardScaler`, un modelo por condición (`separado`),
  entrenada con el primer y el último tercio. Es la configuración del paso 3.
- **Regularización:** con 420 features, C = 0.01 puede no ser el óptimo de F.
  Para no favorecer a ninguna de las dos, **las dos** se evalúan con la misma
  rejilla C ∈ {0.001, 0.003, 0.01, 0.03, 0.1} y se compara **la mejor de cada
  una**. La mejor C es la de mayor ρ media de PASAT + PVTB con la grabación
  completa. Se reporta la rejilla entera.
- **CV repetida:** 20 particiones por persona (semillas 0–19, 5 pliegues,
  `comun.pliegues_por_persona`). Las dos representaciones y todas las C usan
  **la misma asignación** de personas a pliegues.
- **Medida:** `rho_ventanas`, igual que `reajuste_medida.py`. Se calcula la
  puntuación *out-of-fold* de cada ventana, se promedia sobre las 20
  particiones y se halla la ρ de Spearman con el instante por grabación.
  Como cifra interpretable, también el acierto pareado de tercios.
- **IC95:** bootstrap de personas (`medidas.bootstrap`) sobre la diferencia
  **pareada por grabación** F − B.

## Contrastes

| # | Qué | Versión | Papel |
|---|---|---|---|
| **D1** | Δρ (F − B) en PASAT + PVTB | Grabación completa | **Principal**: ¿hay información espectral extra sobre el tiempo en tarea? |
| D2 | Δρ (F − B) en PASAT y en PVTB por separado | Grabación completa | Descriptivo |
| D3 | Δρ (F − B) en OA y en OC | — | ¿La mejora es común con el reposo (deriva) o específica de la tarea? |
| D4 | Δ de la H1 (tareas − OA), F frente a B | Igual tiempo | Descriptivo. ¿Ayudaría F a la pregunta que quedó abierta? |

**Se elige la grabación completa para D1** porque la pregunta es si hay
información extra, no si es específica de la tarea (eso es D3 y D4), y porque
usa todas las ventanas.

## Criterio de decisión (fijado antes de ejecutar)

| Resultado de D1 | Lectura | Qué se propone |
|---|---|---|
| Δρ ≥ +0.02 **y** límite inferior del IC95 > 0 | **Hay margen:** el espectro fino tiene información que las bandas pierden | Piloto de CNN (paso 2) y mirar en qué frecuencias está la mejora |
| Límite inferior del IC95 > 0, pero Δρ < +0.02 | Mejora real pero pequeña | Se valora a la vista del resultado. Una CNN difícilmente aporta algo apreciable por la vía espectral |
| IC95 incluye 0, o Δρ < 0 | **Sin margen:** las bandas recogen lo que da el espectro | La CNN solo tiene sentido como convergencia (CNN ≈ logística) o por los puntos 2 y 3. El piloto se valora a la vista del resultado |

- **Por qué +0.02.** En `train`, la ρ de PASAT + PVTB a igual tiempo es
  ≈ 0.14 (paso 3, reajuste 1). Un +0.02 es una mejora relativa de ≈ 15 %, más
  o menos lo mínimo que justificaría 30–45 h de CPU. Es una convención, no una
  prueba, y se declara así.
- **Matiz de D3.** Si F mejora OA tanto como las tareas, la información extra
  es probablemente deriva común (asentamiento del casco), no tiempo en tarea.
  Se dice explícitamente, sea cual sea D1.

## Interpretación si hay margen

- **Perfil de frecuencias:** coeficientes medios de la logística F por Hz
  (promediados sobre canales y pliegues) en PASAT, PVTB y OA, en una figura.
- **Señales de alarma de artefacto:** si la mejora se concentra en 20–30 Hz
  (posible EMG) o en 1–3 Hz (parpadeo residual o deriva de contacto), se
  anota como sospechosa. En ese caso, en el paso 2 se añade a la CNN la
  referencia de movimiento y contacto del paso 2 de la línea.

## Qué se produce

- `psd_fina.py` (se ejecuta desde la raíz del repo con `.venv/Scripts/python.exe`;
  no necesita PyTorch). Reutiliza `03_clasificador/comun.py` y `medidas.py`
  sin modificarlos. Lo que esos módulos asumen de las 56 columnas se escribe en
  paralelo dentro del script nuevo.
- `ejecucion.log`, `resultados.json` (con el SHA-256 de `features_all.parquet`,
  `matriz_rf.parquet` y `procesamiento_log.csv`, y un hash combinado de los
  `.npy` leídos).
- `tablas/`, solo agregados: `integridad.csv`, `rejilla.csv` (representación ×
  C × condición × versión) y `contrastes.csv` (D1–D4).
- `figuras/`: Δρ por condición y perfil de frecuencias.

## Coste

- CPU: lectura y PSD, unos minutos. La logística con 20 particiones × 5 C × 2
  representaciones × 4 condiciones × 2 versiones, del orden de 10–30 min.
- Trabajo: ≈ 3–4 h (script, comprobaciones e informe).

## Recuento de miradas

- `train`: se añaden 2 representaciones × 5 C (las 32 configuraciones del paso
  3 ya se miraron en `train`).
- `val`: sigue en 2 (no se carga).
- `test`: no se carga (gastado el 2026-10-27).

## Resultados

Ejecutado el 2026-10-29 con `psd_fina.py` → `ejecucion.log`,
`resultados.json`, `tablas/` y `figuras/`. Duración: 350 s.

### Integridad

- 1 785 grabaciones de `train` (121 personas, 128 492 ventanas). Ningún `.npy`
  desalineado.
- La `pot_rel` recalculada desde la PSD fina reproduce la de `features_all`:
  r mínima = 1.000000 y diferencia absoluta máxima de 4.9·10⁻⁸.
- **Comprobación adicional:** B con C = 0.01 reproduce exactamente las cifras
  del paso 3 en `train` a igual tiempo (PASAT 0.143, PVTB 0.136, OA 0.063).

### Rejilla (ρ media por grabación, grabación completa)

| C | B PASAT + PVTB | F PASAT + PVTB |
|---|---|---|
| 0.001 | 0.135 | 0.170 |
| 0.003 | 0.148 | **0.174** |
| 0.01 | 0.154 | 0.168 |
| 0.03 | **0.154** | 0.159 |
| 0.1 | 0.151 | 0.149 |

Mejor C: **0.03 para B** y **0.003 para F**. Como era de esperar con 420
features, F pide más regularización. La rejilla completa por condición está en
`tablas/rejilla.csv`.

### Contrastes (F − B con la mejor C de cada una; IC95 por bootstrap de personas)

| # | Qué | Versión | Δρ | IC95 |
|---|---|---|---|---|
| **D1** | PASAT + PVTB | Completa | **+0.019** | [+0.013, +0.026] |
| D2 | PASAT | Completa | +0.023 | [+0.015, +0.032] |
| D2 | PVTB | Completa | +0.015 | [+0.005, +0.025] |
| D1′ | PASAT + PVTB | Igual tiempo | +0.028 | [+0.018, +0.038] |
| D2 | PASAT | Igual tiempo | +0.035 | [+0.021, +0.050] |
| D2 | PVTB | Igual tiempo | +0.022 | [+0.011, +0.033] |
| D3 | OA | — | +0.003 | [−0.015, +0.021] |
| D3 | OC | — | **+0.052** | [+0.032, +0.074] |
| D4 | H1 (tareas − OA) | Igual tiempo | **+0.025** | [+0.005, +0.046] |

ρ y acierto pareado con la mejor C de cada representación:

| | ρ B | ρ F | Pareado B | Pareado F |
|---|---|---|---|---|
| PASAT (completa) | 0.157 | 0.180 | 0.793 | 0.791 |
| PVTB (completa) | 0.152 | 0.167 | 0.767 | 0.760 |
| PASAT (igual tiempo) | 0.142 | 0.177 | 0.743 | 0.766 |
| PVTB (igual tiempo) | 0.138 | 0.160 | 0.705 | 0.749 |
| OA | 0.069 | 0.073 | 0.620 | 0.616 |
| OC | 0.104 | 0.156 | 0.662 | 0.733 |

### Veredicto según el criterio fijado

**D1 = +0.019 [+0.013, +0.026]: mejora real pero pequeña.** El IC excluye el
0, pero el valor se queda justo por debajo del umbral de +0.02. La
continuación se valora a la vista del resultado.

### Lectura

1. **El espectro fino tiene algo de información que las bandas pierden**, en
   las dos tareas y con la grabación completa. Es una mejora de un 12 % sobre
   la mejor B. Con el acierto pareado de la grabación completa no se nota; solo
   aparece en la medida continua.
2. **La mejora no está en OA** (D3: +0.003), y por eso **mejora el contraste
   tarea − OA a igual tiempo** (D4: +0.025 [+0.005, +0.046]). Es la pregunta
   que quedó abierta en el test (H1 no concluyó). Aquí es una cifra de
   desarrollo en `train`, elegida con C ajustada sobre los mismos datos, y no
   sustituye a ningún veredicto.
3. **OC es la condición que más gana** (+0.052). Encaja con la deriva de
   asentamiento del casco al principio de la sesión (paso 4 de la línea): el
   espectro fino describe mejor ese cambio. Es un aviso para la CNN. Una red
   más flexible aprenderá también esa deriva, y OA es el único control que no
   la tiene.
4. **Perfil de frecuencias** (`figuras/perfil_frecuencias.png`): es ruidoso y
   los coeficientes de features tan correlacionadas no se pueden leer uno a
   uno. Descriptivamente, en PASAT el último tercio pesa más en 6–9 Hz y menos
   en 11 Hz y en 20–30 Hz. Eso es compatible con un desplazamiento del pico de
   alfa hacia frecuencias bajas y con la bajada de beta ya conocida (H3). En
   PVTB el patrón es distinto (más 10–11 Hz y menos 5–6 Hz y 13–14 Hz). No hay
   una concentración en 1–3 Hz ni en 20–30 Hz que haga sospechar de un
   artefacto.
5. **Para la CNN:** por la vía espectral, el margen sobre la logística ronda
   +0.02 de ρ. Una CNN puede sumar información espacial o temporal (puntos 2 y
   3 de «Por qué se hace»), pero este paso no dice nada de eso.

### Desviaciones del plan

- **Primera ejecución interrumpida** por la codificación de la consola de
  Windows al imprimir «ρ», después de la comprobación de integridad y antes de
  entrenar ningún modelo. No se vio ninguna cifra de modelo. Se relanzó sin
  cambios en el código, con `PYTHONIOENCODING=utf-8`.
- Se añadió `tablas/descriptivos.csv` (ρ, IC y acierto pareado por condición
  con la mejor C). No estaba en la lista de productos.
- El perfil de frecuencias se calculó aunque D1 no llegara al umbral, porque
  no cuesta nada. Se usan los coeficientes de los 5 pliegues de la semilla 0.

### Recuento de miradas

- `train`: 2 representaciones × 5 C × 4 condiciones × 2 versiones.
- `val`: 2 (no se cargó).
- `test`: no se cargó.
