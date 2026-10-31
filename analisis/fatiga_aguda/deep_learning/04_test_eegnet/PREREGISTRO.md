# Pre-registro — EEGNet frente a la logística en el conjunto de test (fatiga aguda)

> **Estado: FIJADO en el commit `e20279d` (2026-10-31) y EJECUTADO ese mismo
> día.** Resultados en `INFORME.md`; desviaciones en la sección 11.
> Quedó **FIJADO** en el commit que lo
> incluye junto con los scripts de `04_test_eegnet/` y
> `configuracion_dl.json`, que contiene los SHA-256 de los pesos finales. Ese
> commit lo hace el autor después de las fases A y B (`INFORME.md`) y antes de
> cualquier lectura de `test` para el deep learning. Desde entonces no se
> modifica: cualquier cambio se anota en la sección 11 («Desviaciones») con
> fecha, motivo y si ocurrió antes o después de ver algún resultado de `test`.
>
> **Es la segunda evaluación de la línea en `test`.** La primera, con la
> logística, se hizo el 2026-10-27 (`06_test/`), y sus resultados se conocen.
> Esta evaluación **no cambia ninguno de aquellos veredictos**. En la memoria
> se presenta como «segunda evaluación preregistrada, posterior a la
> principal».

---

## 1. Qué se evalúa y qué no

**Pregunta.** En personas no vistas, ¿extrae una CNN pequeña (EEGNet) sobre la
señal cruda **más, igual o menos** información sobre el tiempo en tarea que la
logística sobre la potencia relativa por banda?

**Lo que no se afirma.** Que el EEG «mide la fatiga» (`PLAN.md`, 1.3). Que la
CNN confirme o refute la H1 del paso 6. Que ninguna CNN pueda superar a la
logística: solo se prueba esta, con este tamaño de muestra.

`test` se usa **una sola vez** para esta pregunta: todo lo que se ejecuta está
declarado aquí, se ejecuta de una pasada y se reporta entero, salga lo que
salga.

| Familia | Contenido | Control del error |
|---|---|---|
| **Principal** | **P1**: Δρ (EEGNet − logística) | IC95 bilateral; una prueba |
| Secundaria | **S1**: ρ de EEGNet > 0, en PASAT y en PVTB | Holm sobre 2, α = 0.05 |

Todo lo que no figura como prueba es **descriptivo**: se reporta con su IC,
sin p y sin veredicto.

## 2. Qué se sabe ya de `test`

El lector debe saber que, al escribir esto, se conocen los resultados de la
logística en `test` (`06_test/INFORME.md`). Con la grabación completa: ρ PASAT
+0.162, PVTB +0.180 y OA +0.104 (más alta que en el desarrollo).

Las decisiones del deep learning **se tomaron después** de conocerlos
(pilotos del 2026-10-29 y 30), pero **solo con criterios de `train`**:

| Decisión | Criterio |
|---|---|
| EEGNet y no ShallowConvNet | ρ en el pliegue 0 de `train` (paso 3) |
| lr = 3·10⁻⁴ | Pérdida interna en `train` (paso 2) |
| 128 Hz, máscara, dropout de 0.25, normalización por grabación | Fijados en `PLAN.md`, sección 7, antes del test de la logística |
| Grabación completa como versión principal | La del desarrollo (pasos 1–3 del deep learning) |
| Hacer este paso | Decisión del autor (2026-10-31), tras ver pilotos negativos. No depende de cifras de `test` |

## 3. Datos

| | |
|---|---|
| Partición | `split` de `matriz_rf.parquet` (Fase 12). Nunca la de `features_all.parquet` |
| Test | **28 personas**. PASAT 102, PVTB 103 y OA 103 grabaciones según la matriz |
| Entrenamiento del modelo final | `train` + `val`: **145 personas** con grabaciones válidas de PASAT o PVTB (de las 146 de `train` + `val`), 62 661 ventanas de los tercios primero y último |
| Ventanas | `EEG-Data/ventanas/*.npy` (14 × 1024 a 256 Hz), alineadas con `features_all.parquet`; canales válidos de `procesamiento_log.csv` (Fase 11.5) |
| Referencia logística | `EEG-Data/processed/fatiga_aguda_test_20261027-190547/predicciones_por_grabacion.parquet`, filas de `rho_ventanas` (las que no llevan `medida`; las de tercios llevan `tercios_pareado`), versiones completa e igual tiempo (solo lectura) |

**Exclusiones** (las de la línea): grabaciones sin `persona_uid` y con < 10
ventanas. En la versión a igual tiempo, además, las tareas cuya sesión no
tiene OA.

**Emparejamiento.** P1 se calcula sobre las grabaciones que tienen ρ de las
dos fuentes (unión por `archivo_origen`). Como las dos aplican los mismos
filtros, deberían coincidir todas. **Si difieren en más de 2 grabaciones, se
para antes de calcular nada** y se investiga (error de código, sección 10.4).
Se cuentan en el inventario.

**Persona expuesta.** La de `06_test/PREREGISTRO.md`, sección 2. Se mantiene
en `test`, con una sensibilidad sin ella (sección 6).

## 4. Lo que queda fijado

### 4.1. EEGNet

- **Entrada:** ventanas de 4 s a 128 Hz (`resample_poly(x, 1, 2)`), z-score
  por grabación y canal con todas las ventanas de la grabación, canales
  excluidos a 0 y mapa de máscara: `(2, 14, 512)`.
- **Arquitectura:** EEGNet-8,2 de `02_piloto_cnn/entrenar.py` (F1 = 8, D = 2,
  F2 = 16, BN tras la conv temporal, dropout de 0.25, 2 097 parámetros).
- **Entrenamiento:** un modelo conjunto para PASAT + PVTB con el primer (0) y
  el último tercio (1) de todas las grabaciones de `train` + `val`. AdamW (lr
  3·10⁻⁴, `weight_decay` 0.01), lote de 256, `BCEWithLogitsLoss`, **sin
  parada temprana**, durante **E épocas**.
- **E** = mediana de las mejores épocas de las 3 semillas de la fase A
  (38, 24 y 27): **E = 27**.
- **Semillas:** 0, 1 y 2. Los SHA-256 de sus pesos (`fase_b/s{k}/final.pt`)
  están en `configuracion_dl.json`. El script se niega a puntuar si no
  coinciden:

  | Semilla | SHA-256 de los pesos |
  |---|---|
  | 0 | `533ddbda6bdde5899d1dfd2240b9031d1c7eea562ea58323aebc4b0e57f73e81` |
  | 1 | `90b221d84175089ab76b46424da6f005b554840523a6ccc9d413a0d65cc06e52` |
  | 2 | `f0bd68d50289399587feff69ab48cebc5294f500c8813715e984eb28f1090313` |

  SHA-256 del tensor de entrenamiento:
  `f6cc66f2f3435b35c79566a9500683963d91d109892af434dbc0b654545d112b`.
  Fase B: 65 min y sin NaN.
- **Puntuación:** el logit de cada ventana de `test` (los tres tercios y las
  ventanas de OA, puntuadas con el mismo modelo de tareas).

### 4.2. Medida

- **`rho_ventanas`:** por grabación, ρ de Spearman entre la puntuación de cada
  ventana y su instante (`tmin_s`). Es la medida de la línea.
- **ρ de EEGNet por grabación:** la **media de las ρ de las 3 semillas**, como
  en los pilotos.

### 4.3. Referencia

La **logística preregistrada del paso 6** (`rel` + `StandardScaler` +
`LogisticRegression(C=0.01)`, una por condición, entrenada con `train` +
`val`). No se reentrena ni se aplica: se usan sus ρ por grabación ya
guardadas. No es la logística F ni la B de los pilotos.

**No se evalúa en `test` ningún otro modelo clásico**, tampoco la logística
sobre el espectro fino (F), que fue la referencia de los pilotos. Sería un
modelo más evaluado en `test`, y el preregistrado de la línea es el de bandas
(decisión del autor, 2026-10-31).

### 4.4. Inferencia

- **Bootstrap:** remuestreando **personas** de `test`, 10 000 réplicas, semilla
  42, IC95 de percentiles. Todas las pruebas y descriptivos se calculan en las
  mismas réplicas (pareadas).
- **p bilateral bootstrap:** `min(1, 2 · min(#réplicas ≤ 0 + 1, #réplicas ≥ 0
  + 1) / (B + 1))`.
- **p unilateral bootstrap** (H0: ≤ 0): `(#réplicas ≤ 0 + 1) / (B + 1)`.

## 5. Pruebas

### 5.1. Principal

**P1.** `Δρ = media sobre las grabaciones de PASAT + PVTB de (ρ_EEGNet −
ρ_logística)`, con la **grabación completa**, pareada por grabación.

| Resultado del IC95 | Veredicto | Lectura |
|---|---|---|
| Límite inferior > 0 | **EEGNET SUPERA** | La señal cruda tiene información sobre el tiempo en tarea que la potencia por banda no recoge. Habría que averiguar cuál (`PLAN.md`, 7.1) |
| Límite superior < 0 | **EEGNET QUEDA POR DEBAJO** | Con este tamaño de muestra, la potencia por banda recoge el efecto mejor que una CNN pequeña |
| Incluye el 0 | **NO SE DISTINGUE** | No hay evidencia de diferencia. **No equivale a «son iguales»** (sección 7) |

Se reportan también Δρ, el IC95 y la p bilateral.

### 5.2. Secundaria

**S1.** ¿Ordena EEGNet en el tiempo las ventanas de las tareas en personas no
vistas? ρ media de EEGNet por tarea (PASAT y PVTB por separado), grabación
completa. H0: ρ ≤ 0 / H1: ρ > 0, p unilateral bootstrap, **Holm sobre 2**.
CONFIRMA si la p de Holm < 0.05; si no, NO CONCLUYE.

## 6. Descriptivos y sensibilidades (sin veredicto)

- Δρ (EEGNet − logística) en PASAT y en PVTB por separado.
- ρ media de EEGNet y de la logística por tarea, con IC95.
- **Igual tiempo:** ρ de EEGNet en las grabaciones de tarea recortadas a la
  duración del OA de la sesión (solo se puntúan las ventanas dentro del
  recorte), frente a la logística de la versión a igual tiempo. **Aviso:**
  EEGNet se entrenó con grabaciones completas y la logística de esa versión,
  con recortadas. No es una comparación limpia.
- **Control OA:** ρ de EEGNet en OA (ventanas puntuadas con el modelo de
  tareas) y la diferencia tareas − OA de EEGNet, en las dos versiones. Si
  EEGNet ordena OA casi tanto como las tareas, ha aprendido sobre todo deriva
  común. No se compara con la ρ de OA de la logística, que tiene su propio
  modelo de OA.
- Acierto pareado de tercios y AUC por ventana de EEGNet.
- **Conjunto de semillas:** ρ con la media de los logits estandarizados de
  las 3 semillas, y la desviación típica de la ρ entre semillas.
- **Sensibilidad 1:** P1 y S1 sin la persona expuesta.
- **Sensibilidad 2:** P1 y S1 sin las grabaciones grabadas a 128 Hz
  (`sfreq_original` = 128; M2 de C1 y C8).
- Si una sensibilidad cambia el veredicto, se dice explícitamente, pero **el
  veredicto oficial es el de la sección 5**.

## 7. Qué cabe esperar (no es una predicción)

*(Escrita antes de la fase A y no se ha retocado después. La fase A en `val`
dio un resultado distinto del esperado: ver la sección 12.)*

Cifras de desarrollo del paso 2 (pliegue 0 de `train`, 25 personas): EEGNet −
logística B = −0.021 [−0.046, +0.003]. De ahí, el error típico de Δρ con 28
personas es ≈ 0.012, y la semiamplitud del IC95, ≈ 0.023.

| Si la diferencia real es | Probabilidad aproximada de «QUEDA POR DEBAJO» |
|---|---|
| −0.020 (la del piloto frente a B) | ≈ 0.40 |
| −0.035 (la del piloto frente a F) | ≈ 0.83 |
| 0 | ≈ 0.025 |

- **Lo más probable** es «NO SE DISTINGUE» o «QUEDA POR DEBAJO». «SUPERA» es
  muy improbable a la vista del desarrollo.
- **Este test no puede demostrar equivalencia.** Con un IC de ±0.023, más
  ancho que el margen de ±0.02 del paso 1, no hay potencia para ello. Por eso
  no se plantea como prueba.
- **S1:** con ρ ≈ 0.15 en el desarrollo y un error típico ≈ 0.02, la potencia
  es cercana a 1 en las dos tareas.

## 8. Datos en bruto que se guardan

En `<EEG_DATA>/processed/fatiga_aguda_dl_test_<fecha-hora>/`. **Excepción
declarada a la regla de solo lectura de `EEG-Data`**, como en el paso 6
(autorizada por el autor el 2026-10-31). El script se niega a escribir si la
carpeta ya existe.

| Artefacto | Contenido |
|---|---|
| `rho_por_grabacion.parquet` | ρ de EEGNet por grabación y semilla, y de la logística, por versión y condición, con `archivo_origen` y `persona_uid` |
| `bootstrap/*.npy` | Las 10 000 réplicas de P1, S1 y los descriptivos |
| `modelos/*.pt` | Los pesos de las 3 semillas (los mismos de `configuracion_dl.json`) |
| `resultados.json` | Todas las cifras, p, p de Holm, veredictos, inventario y metadatos |
| `ejecucion.log` | Salida completa |

En el repositorio solo quedan `resultados_test.json` (sin identificadores de
persona) y `ejecucion_test.log`. El tensor de `test` y las puntuaciones por
ventana se borran de la caché al terminar.

## 9. Metadatos de la ejecución

Commit y comprobación de que `git status` está limpio (los scripts se niegan a
correr si no). SHA-256 de los scripts, de este documento, de
`configuracion_dl.json`, de los pesos, de `matriz_rf.parquet`,
`features_all.parquet`, `procesamiento_log.csv` y
`predicciones_por_grabacion.parquet`, y hash combinado de los `.npy` de
`test`. Versiones de Python, PyTorch, NumPy, pandas y SciPy; sistema operativo
y CPU; fecha y hora de inicio y fin; recuentos de personas, grabaciones y
ventanas.

## 10. Ejecución

1. **Fases A y B** (`INFORME.md`): comprobación en `val` y entrenamiento del
   modelo final. No tocan `test`.
2. Se rellenan las cifras pendientes de la sección 4.1 y de la sección 12.
3. **Commit** de este documento, los scripts y `configuracion_dl.json` (lo
   hace el autor).
4. **Una sola ejecución** de `ejecutar_test.ps1`. Si falla antes de terminar,
   se corrige solo el error de código, se anota en la sección 11 (incluido lo
   que se hubiera visto) y se vuelve a ejecutar completa. Los pesos no se
   tocan.
5. Se reporta **todo**: pruebas con sus veredictos, descriptivos y
   sensibilidades.

## 11. Desviaciones

*(Cada entrada: fecha, qué cambió respecto a este documento, por qué y si
ocurrió antes o después de ver algún resultado de `test`.)*

**Ejecución: 2026-10-31, 22:42–22:43, commit `e20279d`.** Una sola ejecución
de `ejecutar_test.ps1`, sin fallos. Se comprobaron los SHA-256 de los pesos.
Ninguna desviación en datos, modelo, pruebas ni análisis.

1. **2026-10-31, después de ver los resultados. Solo cambia el nombre del
   registro.** La sección 8 dice `ejecucion.log` en la carpeta de datos en
   bruto. Lo que se guardó allí son los tres registros por separado:
   `preparar.log`, `puntuar.log` y `evaluar.log`. Su unión está en el
   repositorio como `ejecucion_test.log`. El contenido es el mismo.

## 12. Fase A en `val` (rellenada antes del commit, 2026-10-31)

Cifras de **desarrollo**: entrenamiento con `train` y puntuación en `val`
(25 personas). Detalle completo en `resultados_val.json`, `evaluar_val.log`,
`tablas/cifras_val.csv` y `figuras/`. No cambian nada de lo fijado arriba: la
fase A no era una puerta.

**Épocas y requisitos técnicos**

| Semilla | Épocas | Mejor época | Pérdida interna | AUC interna |
|---|---|---|---|---|
| 0 | 46 | 38 | 0.6795 | 0.592 |
| 1 | 32 | 24 | 0.6791 | 0.589 |
| 2 | 35 | 27 | 0.6813 | 0.586 |

E = 27. Se cumplen los cuatro requisitos: la pérdida interna baja, la mejor
época no es ni la 1 ni la 80, no hay NaN y la logística reproduce el ensayo
general del paso 6 (diferencia 0 en las ρ medias y en los recuentos). Fase A:
70 min.

**Cifras de la sección 5 en `val`**

| Cifra | Valor | IC95 |
|---|---|---|
| P1: Δρ (EEGNet − logística), completa | **+0.033** | [+0.004, +0.065] (p bilateral 0.028; aquí sería «EEGNET SUPERA») |
| S1: ρ EEGNet PASAT | +0.163 | [+0.119, +0.206] (Holm 0.0002) |
| S1: ρ EEGNet PVTB | +0.169 | [+0.113, +0.225] (Holm 0.0002) |
| ρ logística PASAT / PVTB | +0.119 / +0.148 | [+0.077, +0.158] / [+0.102, +0.195] |
| Δρ PASAT / PVTB | +0.044 / +0.021 | [+0.002, +0.090] / [−0.022, +0.064] |
| Igual tiempo: Δρ (EEGNet − logística) | −0.001 | [−0.046, +0.047] |
| Control OA: ρ EEGNet en OA | +0.026 | [−0.032, +0.084] |
| Tareas − OA (completa / igual tiempo) | +0.140 / +0.070 | [+0.076, +0.200] / [+0.010, +0.133] |
| Acierto pareado de tercios | 0.766 | [0.711, 0.816] |
| Conjunto EEGNet + logística: ρ / ganancia sobre la logística | +0.181 / +0.047 | [+0.139, +0.219] / [+0.015, +0.082] |
| AUC por ventana (media de semillas) | 0.583 | — |
| ρ por semilla (0, 1, 2) | 0.172, 0.160, 0.166 | desviación típica 0.006 |
| Sensibilidades: P1 sin la persona expuesta / sin 128 Hz | +0.036 / +0.033 | [+0.006, +0.068] / [+0.004, +0.066] |

Emparejamiento: todas las grabaciones tienen ρ de las dos fuentes (0 sin
pareja). Inventario de `val`: OA 86, PASAT 88 y PVTB 87 grabaciones de 25
personas.

**Lectura (de desarrollo)**

- El signo es el **contrario** al que esperaba la sección 7: en `val`, EEGNet
  supera a la logística del paso 6. No contradice los pilotos. Allí la
  referencia era la logística F (espectro fino) o la B en un pliegue de
  `train`. Aquí es la logística preregistrada, que en `val` saca ρ 0.133.
- **La ventaja depende de la versión.** A igual tiempo desaparece (−0.001).
  Si en `test` pasa lo mismo, parte de la diferencia de P1 vendría de los
  tramos finales de las grabaciones largas. Esos tramos no tienen control:
  el OA de la sesión es más corto.
- La semiamplitud del IC95 de P1 en `val` es ≈ 0.031, más ancha que la ≈ 0.023
  prevista en la sección 7 para `test`. La potencia real puede ser menor que
  la calculada allí.
