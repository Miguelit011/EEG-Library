# Fatiga aguda: el paso del tiempo dentro de una tarea — plan

> Plan escrito el 2026-10-27, **antes de ejecutar nada en el repositorio**. Las
> cifras de la sección 3 vienen de sondeos exploratorios hechos el 2026-10-26
> fuera del repo (scratchpad temporal de una sesión de Claude Code, que puede no
> conservarse). Por eso este documento recoge todo lo necesario para retomar la
> línea sin ellos: datos, definiciones, resultados, el código esencial
> (apéndice A) y las decisiones pendientes. Nada de lo de la sección 3 es
> citable en la memoria hasta que el paso 1 lo reproduzca desde el repo.

## 0. Resumen

- **Pregunta.** ¿Distingue el EEG el principio del final de **una misma
  grabación** de tarea (primer tercio frente a último tercio), en personas que
  el modelo no ha visto? Es la operacionalización estándar de la fatiga mental
  aguda o decremento de vigilancia (*time-on-task*) en la literatura EEG.
- **Por qué.** Todas las líneas sobre la fatiga percibida (VAS, FG, FC) dan
  resultados nulos o frágiles. La etiqueta de aquí es objetiva y gratuita (el
  instante de cada ventana) y la comparación es dentro de la grabación, donde
  no influyen el casco, la colocación ni el día.
- **Lo que se sabe (exploratorio, solo `train`).** Acierto pareado de 0.785 en
  PASAT, 0.751 en PVTB y 0.682 en PVSAT, frente a 0.620 en OA (control de
  reposo).
- **Validación.** Exploración en `train`, pre-registro, confirmación **una
  sola vez en `val`** y, si `val` cumple el criterio fijado de antemano,
  evaluación **una sola vez en `test`** (sección 5).
- **Deep learning.** Medido en este ordenador: un programa cerrado con EEGNet
  y ShallowConvNet a 128 Hz cuesta 30–45 h de CPU (unas 4–6 noches). Es
  viable, pero no deja margen para búsquedas de hiperparámetros. Con GPU
  serían unas pocas horas, pero mover los datos tiene restricciones (sección
  7.6).

---

## 1. Pregunta y motivación

### 1.1. De dónde viene

| Línea | Etiqueta | Resultado |
|---|---|---|
| Fases 14 y 14b (`docs/MODELADO.md`) | VAS por sesión | Tarea A R² = −0.010; Tarea B igual que sin EEG |
| Test confirmatorio (2026-10-07) | VAS | Prueba principal NO CONCLUYE (p = 0.77); S1 no se replica |
| `normalizacion_vas/` | VAS normalizado | No mejora |
| `fatiga_fg_fc/` | PedsQL FG y FC | FG intra-persona ρ = +0.17, pero desaparece con el control de visita |
| Controles positivos (Fases 17–19, test) | Condición | OA/OC 0.855, reposo/tarea 0.851, cinco tareas 0.488: **confirman** |

Lo que el EEG de consumo detecta en esta cohorte depende de la **escala
temporal** del fenómeno:

| Escala | Ejemplo | ¿Se detecta? |
|---|---|---|
| Segundos–minutos | ojos abiertos/cerrados, reposo/tarea, tipo de tarea | Sí (confirmado en test) |
| **Minutos dentro de una tarea** | **primer frente a último tercio** | **Sí en el sondeo (esta línea)** |
| Semanas | VAS, FG, inicio/fin de trimestre, rendimiento intra-persona | No |
| Rasgo estable | identidad, sexo, memoria de trabajo | Sí (sondeos del 2026-10-26) |

Esta línea aporta el resultado positivo que le falta a la memoria. Además
convierte el nulo del VAS en la otra mitad de un argumento: la variabilidad
entre sesiones (casco, colocación, gel salino, día) supera a los efectos lentos
y pequeños, y desaparece cuando la comparación se hace dentro de la grabación.

### 1.2. Literatura de referencia (comprobar las referencias exactas antes de citarlas)

- Boksem, Meijman y Lorist (2005), *Cognitive Brain Research*: con el tiempo en
  tarea aumentan alfa y theta (y la potencia lenta en general).
- Wascher et al. (2014), *Biological Psychology*: fatiga por tiempo en tarea y
  potencia frontal theta/alfa.
- Craig et al. (2012), *Psychophysiology*: cambios EEG regionales con la
  fatiga mental.
- Tran et al. (2020), revisión sobre EEG y fatiga mental: patrón general de
  aumento de la actividad lenta.
- Lawhern et al. (2018), *J. Neural Eng.*: EEGNet. Schirrmeister et al.
  (2017), *Hum. Brain Mapp.*: ShallowConvNet y Deep4Net.

### 1.3. Cómo se llama y cómo no

- En la memoria: **«fatiga aguda» / «decremento de vigilancia» / «efecto del
  tiempo en tarea»**. Nunca «la fatiga del alumno» ni «la fatiga del VAS».
- No hay medida subjetiva ni conductual **dentro** de la grabación (sección
  4.6). Lo que se afirma es «el EEG cambia de forma sistemática con el tiempo en
  tarea, compatible con fatiga aguda», no «el EEG mide la fatiga».
- **Plataforma:** la regla «No detecta ni mide fatiga» sigue vigente. Una
  vista de evolución temporal de la potencia dentro de la grabación sí encaja,
  sin la palabra «fatiga». Un modelo de esta línea en la plataforma requiere
  hablarlo antes con el tutor.

---

## 2. Qué se va a producir

Carpeta `analisis/fatiga_aguda/`, con la misma convención que `fatiga_fg_fc/`:
un `README.md` con la tabla de pasos y, por paso, una subcarpeta con su script,
sus tablas, sus figuras, su `resultados.json` y un `INFORME.md` cuyo plan se
escribe **antes** de ejecutar. Este `PLAN.md` es el documento de partida.

Reglas comunes:

- **Solo `train` hasta el pre-registro.** `val` se abre una vez (paso 6) y
  `test` también una sola vez, solo si `val` cumple el criterio pre-registrado
  (paso 6b, sección 5).
- La partición válida es la columna `split` de `matriz_rf.parquet` /
  `labels.csv` (nivel `persona_uid`). **No** se usa la `split` de
  `features_all.parquet`, que va por `sujeto` (ver la sección 3.5).
- Se agrupa siempre por `persona_uid` (GroupKFold y bootstrap).
- Ejecutar desde la raíz del repo con `.venv/Scripts/python.exe`, salvo el
  deep learning (sección 7.7).
- Registrar en `resultados.json` el SHA-256 de cada archivo de datos leído.
- Si alguna función nueva va a `src/eeg_library/`, necesita su test en
  `tests/`. Los scripts de `analisis/` no.

---

## 3. Lo que ya se sabe (sondeos del 2026-10-26, exploratorios)

### 3.1. Datos que se usaron

- `EEG-Data/processed/features_all.parquet`: una fila por ventana de 4 s
  (paso de 2 s) con `pot_abs_<canal>_<banda>` y `pot_rel_<canal>_<banda>`.
  Son 14 canales × 4 bandas (`delta`, `theta`, `alfa`, `beta`) y **todos los
  canales**, también los excluidos por calidad (sección 4.4).
- Filtro `split == "train"` **de `features_all`** (122 personas, ≈ 447
  grabaciones por tarea).
- `matriz_rf.parquet` para `persona_uid` y `sesion_uid`, y
  `manifest_canonical.csv` (`timestamp_iso`) para el inicio de cada archivo.
- Tiempo dentro de la grabación: `t_rec = tmin_s / 60` (minutos).

### 3.2. Pendiente intra-grabación de la potencia (por minuto)

Se ajusta la pendiente en cada grabación (con un mínimo de 10 ventanas), se
promedia por persona y se hace un t de una muestra frente a 0 sobre las
personas. Las magnitudes son log₁₀ de la potencia media de la región:
posterior = P7, O1, O2, P8; frontal = AF3, F3, F4, AF4. Formato: pendiente
(t).

| Condición | alfa post abs | theta front abs | beta post abs | beta front abs | delta front abs | alfa post rel | theta front rel | beta post rel |
|---|---|---|---|---|---|---|---|---|
| OC | −0.023 (−2.8) | −0.020 (−2.1) | −0.023 (−3.2) | −0.021 (−3.3) | −0.043 (−3.4) | +0.020 (+4.0) | +0.012 (+3.8) | +0.019 (+3.7) |
| OA | +0.033 (+4.4) | +0.017 (+1.8) | +0.021 (+3.1) | +0.031 (+6.5) | +0.016 (+1.2) | +0.005 (+0.9) | −0.005 (−1.3) | −0.005 (−1.0) |
| STROOP | +0.055 (+3.4) | +0.041 (+2.4) | +0.003 (+0.2) | +0.017 (+1.3) | +0.042 (+1.8) | +0.032 (+3.4) | +0.009 (+1.4) | −0.026 (−2.7) |
| **PASAT** | +0.027 (**+8.9**) | +0.022 (+5.8) | +0.013 (+4.5) | +0.005 (+1.6) | +0.023 (+4.5) | +0.002 (+1.0) | +0.002 (+1.3) | −0.012 (**−4.6**) |
| **PVTB** | +0.024 (**+6.8**) | +0.018 (+3.7) | +0.005 (+1.7) | +0.017 (+4.8) | +0.032 (+4.6) | +0.003 (+1.5) | −0.008 (−5.2) | −0.016 (**−7.8**) |
| MSIT | +0.021 (+1.4) | −0.023 (−1.1) | −0.023 (−2.3) | −0.011 (−1.0) | −0.085 (−3.2) | +0.060 (+6.4) | +0.040 (+5.4) | +0.014 (+1.4) |
| PVSAT | +0.016 (+4.6) | +0.005 (+0.9) | +0.008 (+2.3) | +0.007 (+2.3) | −0.002 (−0.3) | +0.013 (+6.9) | +0.004 (+2.5) | +0.006 (+2.2) |

Lectura honesta (más matizada que la explicación oral del 2026-10-26):

- **En PASAT y PVTB** sube la potencia absoluta lenta (alfa, theta y delta) y
  baja la beta **relativa**. Es un desplazamiento hacia frecuencias lentas,
  compatible con la literatura de *time-on-task*.
- **El argumento «no es secado del electrodo»** (el secado añadiría ruido de
  banda ancha y subiría la beta relativa) **solo vale para PASAT y PVTB**. En
  PVSAT y MSIT la beta relativa sube ligeramente, y en OA, sin tarea, la
  potencia absoluta también sube (alfa +4.4, beta frontal +6.5). Hay una deriva
  común a todas las grabaciones con ojos abiertos. El efecto de la tarea es lo
  que excede a esa deriva, y por eso OA es el control principal.
- OC hace lo contrario en absoluta (baja). No se ha interpretado.
- El patrón difiere entre tareas a igual momento de la sesión (MSIT y PVSAT,
  seguidas, se comportan distinto). Eso apunta a algo específico de la tarea y
  no a una deriva del hardware, que sería igual para todas. Es un indicio, no
  una prueba.

### 3.3. Clasificador primer tercio frente a último tercio

Diseño (código en el apéndice A):

- Tareas: PASAT, PVTB, PVSAT, más OA como control.
- Segmentos: `dur` es el `tmin_s` máximo de la grabación; tercio inicial si
  `t ≤ dur/3` y final si `t ≥ 2·dur/3`. El tercio medio se descarta.
- Features: las 56 `pot_rel_<canal>_<banda>`, promediadas por segmento y en
  log₁₀.
- **Centrado por grabación:** a cada segmento se le resta la media de los dos
  segmentos de su grabación. Así el modelo solo ve el cambio, no el nivel de la
  persona.
- Modelo: `StandardScaler` + `LogisticRegression(C=0.1, max_iter=3000)`, con
  `cross_val_predict` y `GroupKFold(5)` por `persona_uid`.
- **Acierto pareado:** proporción de grabaciones en que el tercio final recibe
  más probabilidad que el inicial. El azar es 0.5. Con el centrado coincide con
  la BA.

| Tarea | Grabaciones / personas | AUC sin centrar | Pareado sin centrar | **AUC centrado** | **Pareado centrado** |
|---|---|---|---|---|---|
| PASAT | 447 / 122 | 0.604 | 0.732 | **0.860** | **0.785** |
| PVTB | 445 / 122 | 0.601 | 0.708 | **0.831** | **0.751** |
| PVSAT | 443 / 121 | 0.543 | 0.616 | **0.720** | **0.682** |
| OA (control) | 447 / 122 | 0.537 | 0.570 | **0.653** | **0.620** |

Lo que hay que defender es la diferencia **tarea − OA** (0.785 − 0.620 en
PASAT), no la distancia al 0.5.

### 3.4. Otros resultados de los sondeos que afectan a esta línea

- **Momento de inicio de cada tarea en la sesión** (mediana en minutos,
  calculada desde `timestamp_iso`): OC 0, OA 2.3, STROOP 4.9, PASAT 6.7, PVTB
  10.3, MSIT 13.9, PVSAT 15.5.
- **«Fatigabilidad» por sesión** (compuesto fijado antes de mirar: la media de
  las z de las pendientes de `alfa_post_abs` y `theta_front_abs` en PASAT, PVTB
  y PVSAT):
  - ICC ≈ 0.37;
  - efecto de fin de trimestre nulo (t = +0.06, con efectos fijos de persona y
    visita lineal);
  - sin relación con FG ni con los lapsos del PVT;
  - frente al VAS, intra-persona ρ = −0.128 (p = 0.006), **en dirección
    contraria** a la esperada.

  **Línea cerrada:** la fatigabilidad aguda no se conecta con la crónica.
  Se reporta como resultado y no se vuelve a explorar.
- **Casco:** en M4–M6 dominan dos números de serie (172358 y 236373). Dentro de
  persona, el cambio de casco mueve 27 de las 70 features de reposo. Para esta
  línea no confunde, porque el casco es constante dentro de la grabación, pero
  conviene usarlo como covariable en el análisis entre sesiones (paso 5).
- **Rendimiento:** el EEG por sesión no sigue a los lapsos ni a los tiempos de
  reacción dentro de persona. No hay marcadores de ensayo en los EDF
  (`MarkerHardware` = 0).

### 3.5. Incidencias de los sondeos (hay que declararlas)

- **Contaminación mínima de la partición.** Los sondeos filtraron por la
  `split` de `features_all.parquet`, que va por `sujeto`. Tres `persona_uid`
  cambian de partición al usar la `split` correcta (la de la matriz):
  - una de `test` aparecía como `train` (6 grabaciones);
  - una de `val` aparecía como `train` (7 grabaciones);
  - una de `val` aparecía como `test` (1 grabación; no se usó).

  Se identifican con `matriz.split != features_all.split`.
  - **Recomendación:** mantener a esas dos personas (la de `val` y la de
    `test`) en sus conjuntos de evaluación, declarar la exposición en el
    pre-registro y añadir en cada conjunto un análisis de sensibilidad sin
    ellas. Excluirlas a posteriori sería también un grado de libertad.
- **Los sondeos usaron los 14 canales**, también los excluidos por calidad. El
  paso 1 lo corrige (sección 4.4).
- **El guion del sondeo se editó después de ejecutarlo:** las listas de
  condiciones se dejaron vacías (`for cond in []`) para no repetir secciones.
  El registro fiel es la salida que se transcribe en 3.2 y 3.3. Las listas
  originales están en el apéndice A.
- **Recuento de exploración:** estas miradas (pendientes en 7 condiciones, 4
  clasificadores en 2 variantes y la fatigabilidad frente a 4 variables) se
  anotan en el pre-registro como exploración previa.

---

## 4. Datos y definiciones

### 4.1. Fuentes

| Qué | Dónde | Notas |
|---|---|---|
| Features por ventana | `EEG-Data/processed/features_all.parquet` | 280 339 ventanas, 4543 grabaciones. Columnas útiles: `sujeto`, `condicion`, `sesion`, `archivo_origen`, `indice_ventana`, `tmin_s`, `centro`, `split` (no usar), `orden_en_sesion`, `pot_abs_*`, `pot_rel_*` |
| Ventanas crudas | `EEG-Data/ventanas/<archivo>.npy` | (n, 14, 1024) float32 en voltios, 256 Hz tras el procesado. La ventana `i` del `.npy` es la fila `indice_ventana == i` del archivo en `features_all`. Orden de canales: AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4 |
| Identidad, sesión y partición | `EEG-Data/processed/matriz_rf.parquet` (`archivo`, `persona_uid`, `sesion_uid`, `split`, `condicion`) y `metadata/labels.csv` | Partición válida |
| Canales excluidos y procesado | `EEG-Data/processed/procesamiento_log.csv` | `canales_excluidos` (separados por `|`), `n_canales_buenos`, `car`, `sfreq_original`, `resampleado`, `duracion_s`, `n_epocas` |
| Hora de inicio y casco | `EEG-Data/metadata/manifest_canonical.csv` | `nombre_nuevo`, `timestamp_iso`, `serie_headset`, `sampling_rate_hz`, `n_signals`, `sexo_filename` |
| EDF originales (MOT, CQ) | `EEG-Data/dataset_final/**.edf` | Solo lectura |
| Conducta por sesión | `EEG-Data/BD_COMPLETA_corregida.xlsx`, hoja `CUESTIONARIOS` | `Reaction_`, `Accuracy_`, `Lapses_`, `RCS_`… por tarea. Enlace con `analisis/fatiga_fg_fc/datos_fg_fc.py` |

### 4.2. Tamaños

Medidos sobre `features_all`: mediana por grabación de `tmin_s` máximo.

| Condición | Grabaciones | Ventanas | Ventanas/grabación | Duración útil (s) |
|---|---|---|---|---|
| PASAT | 648 | 57 501 | 89 | 176 (p10–p90: 168–182) |
| PVTB | 647 | 55 986 | 87 | 172 (166–178) |
| PVSAT | 646 | 56 058 | 87 | 172 (166–178) |
| OA | 645 | 36 138 | 55 | 108 (106–118) |
| OC | 659 | 37 420 | 55 | 108 (106–120) |
| STROOP | 650 | 18 413 | 28 | 54 (48–62) |
| MSIT | 648 | 18 823 | 28 | 54 (48–64) |

Cada tercio de PASAT, PVTB o PVSAT dura ≈ 1 min (≈ 29 ventanas). STROOP y MSIT
son demasiado cortas para que los tercios tengan sentido: quedan fuera del
análisis principal.

Grabaciones etiquetadas (`split` de la matriz):

| Partición | OA | PASAT | PVSAT | PVTB | Personas |
|---|---|---|---|---|---|
| train | 445 | 445 | 442 | 443 | 120 |
| val | 86 | 88 | 86 | 88 | 25 |
| test | 103 | 102 | 105 | 103 | 28 |

### 4.3. Tiempo

- `tmin_s` es el inicio de la ventana **relativo al registro ya recortado**
  (`recortar_registro` quita 5 s al principio y 5 s al final). En tiempo del
  EDF original, la ventana `i` cubre `[5 + tmin_s, 5 + tmin_s + 4)` s. El
  remuestreo de 128 a 256 Hz conserva el tiempo. **Verificar** en el paso 2 que
  ninguna otra fase recorta (la ICA no lo hace).
- Tiempo dentro de la sesión: `t_ses = (timestamp_iso − min(timestamp_iso de la
  sesión)) + tmin_s`. `timestamp_iso` es el inicio del archivo.

### 4.4. Canales excluidos

`features_all` trae los 14 canales, también los marcados como malos. La matriz
de la Fase 13 los restringe con la Fase 11.5. Aquí:

- **Clasificador clásico:** tras centrar por grabación, se pone a 0 el cambio de
  los canales que figuran en `canales_excluidos` de esa grabación. Así no
  aportan nada.
- **Deep learning:** se ponen a 0 tras la normalización por grabación, con la
  opción de un canal de máscara (como prevé `docs/MODELADO.md`, «Ruta CNN»).
- Los agregados por región solo promedian canales válidos.
- Comparar el resultado con los 14 canales (como en el sondeo) para medir
  cuánto cambia.

### 4.5. Otras particularidades

- **128 Hz:** 100 grabaciones de M2 (C1 y C8) se grabaron a 128 Hz y se
  remuestrearon a 256 Hz. Análisis de sensibilidad: repetir sin ellas.
- **Versiones de export:** `n_signals` es 60, 64, 68 o 72 (los EDF muestran
  59–71 canales). Las de 60 y 64 solo aparecen en M2. **Comprobar** que todas
  traen `MOT.*` y `CQ.*`. Si no, el control de movimiento será parcial en M2.
- Ejemplo de canales de un EDF de 71: `MOT.AccX/Y/Z`, `MOT.MagX/Y/Z`,
  `MOT.Q0–Q3` (cuaterniones), `MOT.Interpolated`, `CQ.<canal>` ×14,
  `CQ.Overall`, `RawCq`, `Battery`, `MarkerHardware`.

### 4.6. Limitación estructural

No hay marcadores de ensayo, así que **no se puede relacionar el EEG con el
rendimiento minuto a minuto** (p. ej., si los lapsos del PVT aumentan en el
último tercio). Solo hay totales por sesión. Se declara en la memoria. Como
máximo, en el paso 5 se mira entre personas si la pendiente intra-grabación se
asocia al rendimiento total en esa tarea.

---

## 5. Diseño de validación

**Decisión del usuario (2026-10-27): si los resultados en `val` son buenos,
la línea se evalúa también en `test`.** El test ya se abrió el 2026-10-07 para
el VAS y la condición, pero **nunca para esta pregunta**. La única excepción es
la persona declarada en 3.5.

> `docs/MODELADO.md` (sección «Pendientes») recoge desde el 2026-10-27 la
> regla matizada: el test está gastado para el VAS y la condición, y una
> pregunta nueva puede evaluarse en él con pre-registro y criterio de paso en
> `val`. En la memoria hay que explicar por qué el test sigue siendo válido
> aquí: el modelo, las features y la etiqueta se fijan sin haber mirado el
> test para el tiempo en tarea.

| Conjunto | Uso en esta línea |
|---|---|
| `train` (120 personas) | Exploración libre, selección del modelo, CV por persona. Toda la sección 3 y los pasos 1–5 |
| `val` (25 personas) | **Confirmación única** tras el pre-registro (paso 6). Nunca se ha mirado para esta pregunta, salvo la persona declarada en 3.5 |
| `test` (28 personas) | **Evaluación final única** (paso 6b), solo si `val` cumple el criterio de paso pre-registrado. No se mira hasta entonces |

Reglas de la secuencia `val` → `test`, que se fijan **en el mismo
pre-registro**, antes de abrir `val`:

1. **Qué significa «buenos resultados en `val`».** Hay que escribir un criterio
   de paso concreto; si no, se decide a la vista de los datos. Propuesta: «H1
   CONFIRMA en `val`», es decir, el límite inferior del IC unilateral de la
   diferencia tarea − OA es > 0. Si no hay potencia para H1 (paso 4), usar H2
   en PASAT y PVTB y declararlo.
2. **Qué modelo llega a `test`.** O el mismo modelo entrenado solo con `train`
   (el que se evaluó en `val`), o uno reentrenado con `train` + `val` con
   **exactamente** la misma configuración. Recomendación: reentrenar con
   `train` + `val`, que tiene más datos y no implica ninguna decisión nueva.
   Lo que no vale es cambiar nada (features, C, arquitectura) después de ver
   `val`.
3. **Qué se evalúa en `test`.** Las mismas hipótesis, contrastes y reglas de
   decisión que en `val`, en una sola ejecución. Se reportan las cifras de los
   dos conjuntos por separado, sin combinarlos ni escoger el mejor.
4. **Si `val` no pasa el criterio**, el test no se abre y se reporta el
   resultado de `val` como está. Abrir el test «para ver» tras un fallo en
   `val` convertiría el test en una segunda oportunidad.
5. **Si se cambia algo después de abrir el test**, todo lo que venga después
   es exploratorio y se cita así.

- **Potencia:** con 25 personas en `val` (≈ 88 grabaciones por tarea) y 28 en
  `test` (≈ 103), un acierto pareado de 0.78 frente a 0.5 sale holgado. El
  contraste **tarea − OA** (≈ 0.16 en train) es el que puede quedarse corto.
  **El paso 4 estima la potencia** submuestreando 25 y 28 personas de `train`
  (≥ 2000 réplicas) antes de fijar la hipótesis principal y el criterio de
  paso.

---

## 6. Pasos

Las horas son de trabajo humano. El cómputo del deep learning va aparte
(sección 7).

### Paso 1 — `01_descriptivo/`: reproducir y describir (≈ 5 h)

- Reproducir 3.2 y 3.3 desde el repo con la partición correcta y los canales
  excluidos (4.4). Reportar también la versión de 14 canales.
- Curvas por minuto (media por persona e IC por bootstrap de personas) de
  alfa y theta absolutas y de beta relativa, por tarea y región, con OA como
  referencia.
- Descriptivo de duraciones, número de ventanas por tercio y grabaciones
  excluidas (< 10 ventanas).
- **Criterio de continuidad:** el acierto pareado centrado de PASAT y PVTB
  sigue siendo ≥ 0.70 y supera a OA.

### Paso 2 — `02_artefactos/`: movimiento y calidad de contacto (≈ 8 h)

Es el paso que más protege la línea frente a la crítica de «es el casco».

- Leer de cada EDF de PASAT, PVTB, PVSAT y OA los canales `MOT.*` y `CQ.*`
  (solo lectura), alineados a las ventanas según 4.3.
- Por ventana:
  - movimiento = desviación típica del módulo de la aceleración
    (`AccX/Y/Z`) y cambio angular a partir de los cuaterniones;
  - contacto = media de `CQ.<canal>` en los canales válidos, y `CQ.Overall`.
- Preguntas:
  - ¿Aumentan el movimiento o empeora el contacto con el tiempo en tarea?
  - ¿El clasificador sigue funcionando si se añaden como features «de
    confusión» y se comparan los resultados?
  - ¿Y si se residualiza cada feature EEG por movimiento y contacto dentro de
    la grabación?
  - ¿Qué acierto pareado alcanza un clasificador que **solo** usa movimiento y
    contacto? Es la referencia de artefacto que hay que superar.
- Comprobar la existencia de `MOT`/`CQ` por versión de export (4.5).

### Paso 3 — `03_clasificador/`: modelo clásico (≈ 8 h)

- Variantes de features (fijar la lista antes de mirar):
  1. 56 `pot_rel` (como el sondeo);
  2. 56 `pot_abs` en log;
  3. ambas;
  4. agregados por región.
- Modelos: logística (C en {0.01, 0.1, 1}) y Random Forest (el ya validado en
  la Fase 14, sin búsqueda de hiperparámetros).
- Tareas: PASAT, PVTB y PVSAT por separado y conjuntas. OA y OC como
  controles.
- **Transferencia entre tareas:** entrenar en PASAT y evaluar en PVTB (y al
  revés). Si funciona, el cambio es común a las tareas y no específico de una.
- Variante de regresión: predecir el minuto (ρ de Spearman dentro de la
  grabación, promediado).
- Importancias e interpretación: qué canales y bandas pesan.
- **Salidas:** el modelo y las features que se fijarán en el pre-registro. Se
  elige por el criterio declarado aquí (el mayor acierto pareado medio en
  PASAT + PVTB con CV por persona), no a ojo.

### Paso 4 — `04_potencia_y_controles/`: potencia y tiempo desde la colocación (≈ 6 h)

- **Potencia para `val` y `test`:** submuestrear 25 y 28 personas de `train`,
  repetir la CV y estimar la probabilidad de que los contrastes de la sección 8
  sean significativos. De aquí sale la hipótesis principal y el criterio de
  paso a `test` (sección 5).
- **Tiempo en tarea frente a tiempo desde que se colocó el casco.** El orden
  fijo de las tareas impide separarlos del todo. Comprobaciones parciales:
  1. **Especificidad:** si fuera deriva del hardware, el patrón espectral sería
     igual en todas las tareas a igual tiempo de sesión. Comparar MSIT y PVSAT
     (seguidas) y PASAT y PVTB.
  2. **«Dientes de sierra»:** ¿vuelve a bajar la alfa al empezar la tarea
     siguiente? Comparar el último minuto de la tarea k con el primero de la
     k+1. Hay un confusor: el cambio de tarea cambia la demanda. Se reporta
     como indicio.
  3. **Desaceleración:** una deriva de asentamiento debería ser más fuerte al
     principio de la sesión. Comparar las pendientes de tareas tempranas y
     tardías, teniendo en cuenta que también son tareas distintas.
  4. **Modelo de ventanas** con efectos fijos de sesión y condición, más
     `t_rec` y `t_ses`. Solo se identifica por la variación de los inicios
     entre sesiones, así que hay que reportar si el ajuste es frágil.
- Declarar la limitación en la memoria: solo un diseño con orden
  contrabalanceado la resolvería.

### Paso 5 — `05_entre_sesiones/` (opcional, exploratorio): relación con otras variables (≈ 4 h)

- Pendiente por grabación frente al rendimiento total en esa tarea (entre
  personas): PASAT con `Accuracy_PASAT` y PVTB con `Lapses_PVTB` y
  `Reaction_PVTB`.
- Pendiente frente al casco, la hora del día, el sexo y el centro, para
  descripción y como covariables.
- La conexión con la fatiga crónica ya está cerrada (3.4): no se repite.

### Paso 6 — `06_confirmatorio/`: pre-registro y confirmación en `val` (≈ 6 h)

- Escribir `PREREGISTRO.md` (con el formato de `PREREGISTRO_TEST.md`): las
  hipótesis de la sección 8 ya concretadas, el modelo y las features fijados,
  las métricas, los contrastes, el α, la corrección por multiplicidad, las
  reglas de decisión, el recuento de exploración (3.5 y los pasos 1–5) y el
  tratamiento de las personas expuestas. También la secuencia `val` → `test`
  completa (sección 5): el criterio de paso, qué modelo llega a `test` y que el
  test no se abre si `val` no pasa.
- Commit del pre-registro **antes** de ejecutar (lo hace el usuario).
- Entrenar con todo `train` y evaluar una sola vez en `val`. Si se ejecuta el
  deep learning, su configuración también se fija en el pre-registro (sección
  7.5).
- Escribir el resultado de `val` y la decisión de paso (que aplica la regla, no
  la decide) en el `INFORME.md` del paso. Commit antes de abrir el test.

### Paso 6b — `06_confirmatorio/`: evaluación en `test` (≈ 3 h, solo si `val` pasa)

- Una sola ejecución, con el modelo que dice el pre-registro (reentrenado con
  `train` + `val` o el mismo de `val`) y las mismas hipótesis y contrastes.
- Reportar `val` y `test` por separado, con la sensibilidad sin las personas
  expuestas (3.5).
- Anotar las desviaciones, si las hay, en el pre-registro.

### Paso 7 — Redacción (≈ 8 h) y plataforma (opcional, ≈ 6 h)

- Memoria: resultados (capítulo 6) y discusión con el mapa de escalas
  temporales (1.1). Figuras: curvas por minuto, acierto pareado por tarea
  frente a OA y referencia de artefactos.
- Plataforma: una vista de la evolución temporal de la potencia por banda
  dentro de la grabación, sin mencionar la fatiga.

---

## 7. Deep learning

### 7.1. Qué pregunta responde

¿Contiene la señal cruda información sobre el tiempo en tarea **más allá** de
la potencia por banda? `docs/MODELADO.md` deja la ruta CNN como «trabajo
futuro» porque con el VAS había 653 etiquetas. Aquí, en cambio, hay ≈ 110 000
ventanas de primer o último tercio en las tres tareas largas, y la etiqueta es
objetiva.

Las dos respuestas son publicables:

- **CNN ≈ logística:** la potencia por banda recoge el efecto, y la CNN lo
  confirma por otra vía.
- **CNN > logística:** hay información adicional. Hay que averiguar cuál
  (oclusión por canal y por banda) y descartar que sean artefactos
  (parpadeo, EMG residual).

**Es opcional y depende del paso 3.** Si el modelo clásico no supera a OA,
no se hace.

### 7.2. Diseño

- **Entrada:** ventanas de 4 s, 14 canales, **remuestreadas a 128 Hz**
  (512 muestras). La señal está filtrada a 1–30 Hz, así que 128 Hz (Nyquist en
  64 Hz) no pierde información. Además, las 100 grabaciones de M2 eran
  originalmente de 128 Hz. Remuestrear con anti-aliasing (`mne.filter.resample`
  o `scipy.signal.resample_poly`).
- **Normalización por grabación:** z-score por canal con la media y la
  desviación de **toda la grabación**. Es el equivalente del centrado de 3.3:
  quita el nivel de la persona y la sesión y conserva el cambio interno. Es
  simétrica respecto a la etiqueta, así que no hay fuga. Los canales excluidos
  se ponen a 0 (4.4).
- **Etiqueta:** 0 en el primer tercio y 1 en el último. El tercio medio no se
  entrena.
- **Evaluación comparable con el modelo clásico:** media de las probabilidades
  de las ventanas de cada tercio, y acierto pareado por grabación. Además, AUC
  por ventana.
- **Mismos pliegues:** el mismo `GroupKFold(5)` por persona que el modelo
  clásico (guardar la asignación de pliegues en el paso 3). Así la comparación
  es pareada por grabación.
- **Variante B, por pares:** dos ventanas de la misma grabación y se predice
  cuál es posterior. Optimiza directamente el acierto pareado. Cuesta el doble
  por ejemplo; se prueba solo si la variante A funciona.
- **Arquitecturas:**
  - **EEGNet-8,2** (principal: pequeña, 1800 parámetros);
  - **ShallowConvNet** (aprende algo parecido a la log-potencia por banda, así
    que es el comparador natural de la logística);
  - Deep4Net solo con GPU.
- **Entrenamiento:** AdamW (lr 1e-3), lote de 256 y 30–40 épocas, con parada
  temprana sobre un subconjunto interno de personas de cada pliegue.
- **Referencia por ventana:** la logística sobre la potencia de **las mismas
  ventanas individuales**, no solo de los tercios agregados.

### 7.3. Coste medido en este ordenador

Medido el 2026-10-27 en un entorno aparte (PyTorch 2.14.1 CPU), con datos
sintéticos de la misma forma (sin datos reales), lote de 256 y 6 hilos.
Equipo: AMD Ryzen 5 5600G (6 núcleos/12 hilos), 15.3 GB de RAM, SSD NVMe y
**sin GPU dedicada** (la Radeon integrada no sirve para PyTorch en Windows).
Rendimiento bruto de la CPU: 350–460 GFLOPS en matmul float32.

| Arquitectura | Frecuencia | Parámetros | Entrenamiento (ventanas/s) | Inferencia (ventanas/s) |
|---|---|---|---|---|
| EEGNet-8,2 | 128 Hz | 1 842 | **1 272** | 3 779 |
| ShallowConvNet | 128 Hz | 28 402 | 505 | 1 696 |
| Deep4Net | 128 Hz | 152 102 | 482 | 1 631 |
| EEGNet-8,2 | 256 Hz | 2 866 | 450 | 1 222 |
| ShallowConvNet | 256 Hz | 34 562 | 293 | 871 |
| Deep4Net | 256 Hz | 164 902 | 246 | 700 |

Por pliegue de entrenamiento: ≈ 119 000 ventanas de las tres tareas en
`train` × 2/3 (primer y último tercio) × 4/5 (pliegue) ≈ **64 000 ventanas por
época**. Con los datos reales hay que sumar un 10–30 % por la carga y la
validación.

| Arquitectura (128 Hz) | Por época | Por entrenamiento (≈ 40 épocas) | CV de 5 pliegues |
|---|---|---|---|
| EEGNet-8,2 | ≈ 1 min | ≈ 35–45 min | **≈ 3–4 h** |
| ShallowConvNet | ≈ 2 min | ≈ 1.5 h | **≈ 7–8 h** |
| Deep4Net | ≈ 2.2 min | ≈ 1.5–2 h | ≈ 7.5–9 h |
| (a 256 Hz, todas) | ×2.5–3 | | 8–15 h por CV |

**Memoria:** las tres tareas a 128 Hz en float32, solo los tercios
utilizables de `train` + `val`, ocupan ≈ 2.7 GB y caben en RAM. Leer los
`.npy` necesarios (≈ 10 GB a 256 Hz) del NVMe cuesta segundos: se hace una
vez y se guarda el tensor reducido.

### 7.4. Programa mínimo y viabilidad en CPU

| Bloque | Cómputo |
|---|---|
| Piloto (épocas y lr; un pliegue, EEGNet) | ≈ 3 h |
| EEGNet + ShallowConvNet, CV de 5 pliegues | ≈ 11–12 h |
| 2 semillas más de la mejor | ≈ 6–16 h |
| Variante B por pares (mejor arquitectura) | ≈ 6–8 h |
| Control OA (≈ 1/4 de los datos) | ≈ 1–2 h |
| Entrenamiento final con todo `train` y evaluación en `val` | ≈ 1–2 h |
| Reentrenamiento con `train` + `val` y evaluación en `test` (si `val` pasa) | ≈ 1–2 h |
| **Total** | **≈ 30–45 h de CPU** (a 256 Hz, ×2.5) |

**Veredicto:** **es viable en tu ordenador** si se cumplen tres condiciones:
128 Hz, EEGNet y ShallowConvNet, y un programa cerrado sin búsqueda de
hiperparámetros. Son unas 4–6 noches de 8 h, con el ordenador ocupado mientras
tanto. **No es viable** si se quieren Deep4Net, 256 Hz, búsqueda de
hiperparámetros o muchas semillas: se pasa de 100 h, y un error en mitad del
programa obliga a repetir noches. Además hay que sumar ≈ 15–20 h de trabajo
humano (código, depuración y análisis).

### 7.5. Pre-registro del deep learning

Antes del paso 6 hay que fijar:

- la arquitectura;
- la normalización;
- las épocas o la regla de parada;
- la semilla o semillas;
- la métrica;
- la comparación con la logística (diferencia del acierto pareado, bootstrap
  por persona).

La CNN sigue la misma secuencia que la logística: una vez en `val` y, si pasa
el criterio, una vez en `test`. Si para `test` se reentrena con `train` +
`val`, hay que fijar de antemano el número de épocas, porque ya no queda un
conjunto de parada temprana. Lo razonable es usar la mediana de las épocas
elegidas en la CV.

### 7.6. Opciones con GPU

**Estimación, no medida:** con una GPU de gama media (NVIDIA T4, RTX 3060 o
equivalente), estas redes pequeñas entrenan entre 10 y 40 veces más rápido, a
menudo limitadas por la carga de datos. El programa de 7.4 cabría en ≈ 2–5 h.
Eso permitiría Deep4Net, 256 Hz y más semillas.

**Restricción previa:** son **EEG de menores**, pseudonimizados pero datos
personales de categoría especial. El `CLAUDE.md` del espacio de trabajo prohíbe
enviar los datos de la plataforma a servicios externos, y el mismo criterio
aplica a `EEG-Data`. Quitar los nombres no los convierte en anónimos.

| Opción | Datos fuera del ordenador | Valoración |
|---|---|---|
| **1. Recursos de cálculo de la universidad** (clúster o servidor del departamento, laboratorio con GPU) | Sí, pero dentro de la institución responsable del estudio | **Recomendada.** Preguntar al tutor si existe y cómo se pide acceso. Mover solo el tensor reducido (≈ 2.7 GB) cifrado y borrarlo al terminar |
| **2. Ordenador con GPU NVIDIA del grupo de investigación o del tutor** | Igual que la 1 | Buena si existe. Las mismas precauciones |
| 3. Google Colab, Kaggle o GPU de alquiler (RunPod, vast.ai…) | Sí, a un tercero fuera de la UE o sin contrato | **Descartada salvo autorización escrita** del investigador responsable del estudio (y, en su caso, del DPD) que cubra ese tratamiento |
| 4. GPU integrada AMD con DirectML (`torch-directml`) | No | No medida. En una iGPU Vega que comparte la RAM la ganancia es incierta y el soporte de operaciones es irregular. No merece las horas |
| 5. Ordenador personal de un tercero | Sí, fuera de cualquier control | Descartada |

**Recomendación:** empezar en CPU con el piloto (3 h), que de paso valida el
código, y preguntar en paralelo al tutor por la opción 1. Si llega el acceso a
GPU, ampliar el programa. Si no, el programa de 7.4 basta para la memoria.

### 7.7. Entorno

- PyTorch **no** está en `.venv` y no se instala ahí (regla: no actualizar
  paquetes del entorno de validación). Usar un entorno aparte, por ejemplo
  `.venv-dl` (añadirlo a `.gitignore`) o un grupo opcional `[dl]` en
  `pyproject.toml`. Decide el usuario.
- La rueda CPU que funcionó con Python 3.10:
  ```bash
  python -m pip install --upgrade pip
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
  ```
- **Windows y rutas largas:** la instalación de PyTorch falló en una ruta
  profunda («Windows Long Path support»). Soluciones: crear el entorno en una
  ruta corta, o activar `LongPathsEnabled` (requiere administrador).
- El script de medida está en el apéndice B. Si se repite en otra máquina, da
  la cifra comparable.

---

## 8. Borrador de hipótesis para el pre-registro

Se concretan en el paso 6 con lo que digan los pasos 3 y 4.

| Id | Hipótesis | Contraste propuesto |
|---|---|---|
| **H1 (principal)** | El acierto pareado en PASAT + PVTB supera al de OA de las mismas sesiones | Diferencia de acierto pareado; bootstrap por persona (10 000); unilateral, α = 0.05 |
| H2 | El acierto pareado supera 0.5 en PASAT, PVTB y PVSAT | Bootstrap por persona; Holm sobre 3 |
| H3 | Firma espectral en PASAT y PVTB: pendiente de alfa posterior absoluta > 0 y de beta posterior relativa < 0 | t sobre las pendientes por persona; Holm sobre 4 |
| H4 | El efecto supera a la referencia de artefactos: el acierto pareado del EEG es mayor que el de solo movimiento y contacto | Diferencia pareada; bootstrap por persona |
| H5 (si hay DL) | La CNN fijada iguala o supera a la logística | Diferencia del acierto pareado; IC al 95 %. Se reporta salga lo que salga |

Reglas de decisión: escribirlas en el pre-registro, como en `PREREGISTRO_TEST.md`.
Por ejemplo, «H1 CONFIRMA si el límite inferior del IC unilateral > 0». Las
mismas reglas se aplican en `val` y en `test`. El criterio de paso a `test` es
el de la sección 5.

---

## 9. Riesgos y confusores

| Riesgo | Qué haría | Cómo se aborda |
|---|---|---|
| Tiempo desde la colocación del casco (asentamiento, secado del salino) | Producir el cambio sin fatiga | OA como control, especificidad por tarea y las pruebas del paso 4. **No se resuelve del todo**: se declara |
| Movimiento creciente (aburrimiento) | Meter artefacto creciente | Paso 2 (MOT) |
| Contacto que empeora | Lo mismo | Paso 2 (CQ) |
| Parpadeo y EMG residuales (la ICA solo quita componentes oculares) | Que la CNN aprenda la tasa de parpadeo | Oclusión e importancias; comparar con la logística. La tasa de parpadeo es también un marcador de fatiga, pero no es «EEG»: si aparece, se reporta así |
| Grabaciones de 128 Hz | Diferencias de procesado en M2 | Sensibilidad sin ellas |
| Canales excluidos usados en el sondeo | Cifras infladas o desinfladas | Paso 1 (4.4) |
| Personas de `val` y `test` vistas en el sondeo | Fuga mínima | Declarar y hacer sensibilidad (3.5) |
| Potencia baja en `val` (25) o `test` (28) para tarea − OA | Un «no concluye» con efecto real | Paso 4; elegir la hipótesis principal y el criterio de paso según la potencia |
| Decidir a la vista de `val` si se abre el test | Garden of forking paths | Criterio de paso escrito en el pre-registro (sección 5) |
| Reutilizar un test ya abierto para otra pregunta | Crítica del tribunal | Explicar en la memoria que es una pregunta y una etiqueta nuevas, fijadas sin mirar el test, y matizar `docs/MODELADO.md` |
| Sobreinterpretación («mide la fatiga») | Choca con la evidencia y con la regla de la plataforma | 1.3 |

---

## 10. Lo que no hay que hacer

- **Mirar `test` antes de tiempo**: ni descriptivos ni comprobaciones antes
  del paso 6b. **Abrirlo si `val` no pasa el criterio**, o tocar el modelo
  entre `val` y `test`.
- **Usar la `split` de `features_all.parquet`.**
- **Clasificar ventanas sin centrar o normalizar por grabación** y presentarlo
  como resultado. Sin centrar, el modelo aprende quién es la persona y no el
  momento: el AUC baja a ≈ 0.60.
- **Comparar contra 0.5 en vez de contra OA** en el titular.
- **Mezclar STROOP o MSIT** (≈ 1 min) en el análisis de tercios.
- **Volver a conectar la pendiente con el VAS, FG o el trimestre.** Ya está
  hecho y es nulo (3.4); repetirlo es pescar.
- **Lanzar el deep learning antes de cerrar el paso 3**, o buscar
  hiperparámetros con `val`.
- **Subir datos a Colab, Kaggle o la nube** sin autorización escrita.

---

## 11. Preguntas abiertas

**Para el tutor**

1. ¿Le parece bien la secuencia `val` → `test` con criterio de paso
   (sección 5) y la justificación de reutilizar el test para una pregunta
   nueva?
2. ¿Entra el deep learning en el alcance (presupuesto total del TFG de 300 h)?
3. ¿Hay recursos de cálculo con GPU en la universidad o en el grupo
   (opción 1 o 2 de 7.6)?
4. ¿Puede un modelo de cambio de estado dentro de la grabación entrar en la
   plataforma, o se queda en la vista descriptiva?

**Para el equipo del estudio**

5. Diseño exacto de PASAT, PVTB y PVSAT: duración, ritmo, si la dificultad
   cambia dentro de la tarea y si hay pausas. **Si la dificultad aumenta con el
   tiempo**, el tercio final no solo es «más tarde», también es «más difícil».
   Es un confusor importante que hay que descartar.
6. ¿Por qué el orden es fijo? ¿Hay alguna sesión con otro orden?
7. Cascos usados por medición (`serie_headset` en M2/M3).
8. ¿El consentimiento o el protocolo permite procesar los datos en
   infraestructura de la universidad fuera del ordenador del alumno, o en la
   nube?

---

## 12. Presupuesto de horas

| Bloque | Horas humanas |
|---|---|
| Paso 1 — Descriptivo | 5 |
| Paso 2 — Artefactos (MOT, CQ) | 8 |
| Paso 3 — Clasificador clásico | 8 |
| Paso 4 — Potencia y controles | 6 |
| Paso 5 — Entre sesiones (opcional) | 4 |
| Paso 6 — Pre-registro y confirmación en `val` | 6 |
| Paso 6b — Evaluación en `test` (si `val` pasa) | 3 |
| Paso 7 — Redacción | 8 |
| **Subtotal sin DL** | **≈ 44–48** |
| Deep learning (opcional) | +15–20 humanas, más 30–45 h de CPU nocturna |
| Plataforma (opcional) | +6 |

---

## Apéndice A — Código esencial del sondeo (2026-10-26)

Extracto de `sondeo3.py` (scratchpad). Hay que corregir la partición y los
canales excluidos antes de reutilizarlo (secciones 3.5 y 4.4). Listas
originales de condiciones: pendientes = `["OC", "OA", "STROOP", "PASAT",
"PVTB", "MSIT", "PVSAT"]`; clasificador = `["PASAT", "PVTB", "PVSAT", "OA"]`.

```python
D = Path(r"C:\Users\migue\Desktop\TFG\EEG-Data")
names = pq.ParquetFile(D / "processed" / "features_all.parquet").schema.names
rel = [n for n in names if n.startswith("pot_rel_")]
ab = [n for n in names if n.startswith("pot_abs_")]
w = pd.read_parquet(D / "processed" / "features_all.parquet",
                    columns=["sujeto", "condicion", "sesion", "archivo_origen", "tmin_s", "split"] + rel + ab)
w = w[w.split == "train"].copy()          # ← CORREGIR: usar la split de matriz_rf
man = pd.read_csv(D / "metadata" / "manifest_canonical.csv")
man["t0"] = pd.to_datetime(man.timestamp_iso, utc=True)
w = w.merge(man[["nombre_nuevo", "t0"]], left_on="archivo_origen", right_on="nombre_nuevo")
mat = pd.read_parquet(D / "processed" / "matriz_rf.parquet", columns=["archivo", "persona_uid", "sesion_uid"])
w = w.merge(mat, left_on="archivo_origen", right_on="archivo")
w = w[w.persona_uid.notna()]
w["t_ses"] = (w.t0 - w.groupby("sesion_uid").t0.transform("min")).dt.total_seconds() / 60 + w.tmin_s / 60
w["t_rec"] = w.tmin_s / 60

def agg(pref, chs, band):   # log10 de la potencia media de una región
    return np.log10(w[[f"{pref}_{c}_{band}" for c in chs]].mean(axis=1).clip(lower=1e-30))
POST, FRONT = ["P7", "O1", "O2", "P8"], ["AF3", "F3", "F4", "AF4"]
# p. ej. w["alfa_post_abs"] = agg("pot_abs", POST, "alfa"); w["beta_post_rel"] = agg("pot_rel", POST, "beta")

def pend_por_grabacion(df, col):   # pendiente por minuto en cada grabación
    return df.groupby("archivo_origen").apply(
        lambda g: np.polyfit(g.t_rec, g[col], 1)[0] if len(g) >= 10 else np.nan)
# tabla 3.2: pendientes -> media por persona -> stats.ttest_1samp(pp, 0)

# clasificador 3.3
for cond in ["PASAT", "PVTB", "PVSAT", "OA"]:
    g = w[w.condicion == cond].copy()
    dur = g.groupby("archivo_origen").t_rec.transform("max")
    g["seg"] = np.where(g.t_rec <= dur / 3, 0, np.where(g.t_rec >= 2 * dur / 3, 1, -1))
    g = g[g.seg >= 0]
    X = g.groupby(["archivo_origen", "persona_uid", "seg"])[rel].mean().reset_index()
    X[rel] = np.log10(X[rel].clip(lower=1e-6))
    Xn = X[rel] - X.groupby("archivo_origen")[rel].transform("mean")   # centrado por grabación
    pr = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)),
                           Xn.values, X.seg.values, groups=X.persona_uid, cv=GroupKFold(5),
                           method="predict_proba")[:, 1]
    X["p"] = pr
    par = X.pivot_table(index="archivo_origen", columns="seg", values="p")
    pareado = (par[1] > par[0]).mean()
    # AUC = roc_auc_score(X.seg, pr); BA = balanced_accuracy_score(X.seg, pr > .5)
```

Fatigabilidad (3.4): para PASAT, PVTB y PVSAT y para `alfa_post_abs` y
`theta_front_abs`, la pendiente por grabación se tipifica sobre todas las
grabaciones de esa tarea y magnitud. Después se promedian las seis z por
`sesion_uid` y se analiza con efectos fijos de persona (centrado por persona),
visita lineal e indicador de fin de trimestre, con errores robustos agrupados
por persona.

## Apéndice B — Script de medida del coste de las CNN

Datos sintéticos `torch.randn(256, 14, T)` y etiquetas aleatorias. Se cuentan
3 pasos de calentamiento y 12 pasos medidos de `AdamW` +
`CrossEntropyLoss` en `train()`, y la inferencia en `eval()` sin gradiente.

Arquitecturas reimplementadas a mano, sin braindecode:

- **EEGNet-8,2:** conv temporal (1, k) con k = 64 a 128 Hz y 128 a 256 Hz →
  conv espacial depthwise (14, 1) con D = 2 → ELU → AvgPool 4 → separable
  (1, 16) → AvgPool 8 → lineal.
- **ShallowConvNet:** conv temporal (1, 13) a 128 Hz o (1, 25) a 256 Hz con 40
  filtros → conv espacial (14, 1) → BN → cuadrado → AvgPool (1, 35) con paso 7
  → log → lineal.
- **Deep4Net:** 4 bloques conv (1, 5) con 25, 50, 100 y 200 filtros y MaxPool 2.

Con estas especificaciones se puede rehacer en ≈ 15 minutos.
