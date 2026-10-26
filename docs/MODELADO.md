# Modelado y resultados — EEG-Library

> Documento de uso interno. No se sube al repositorio.
> Última actualización: 2026-10-07 | Continúa a [`docs/ESTADO.md`](ESTADO.md) en la Fase 13.
> **El conjunto de test se ejecutó el 2026-10-07** (sección final de este
> documento). Todo lo anterior a ella se calculó sobre `train` + `val`.

> Este documento recoge el entrenamiento del modelo, su evaluación y todo lo
> posterior a la Fase 13 (Random Forest y, más adelante, la ruta CNN): es
> cuaderno de análisis, no documentación de librería — cambia cada sesión de
> trabajo, a diferencia de `docs/ESTADO.md`, que se congela salvo que cambie el
> propio pipeline. Ver la nota al principio de `docs/ESTADO.md` para el porqué del
> split.

---

## Punto de partida (heredado de `docs/ESTADO.md`, Fase 13)

- **Matriz:** `matriz_rf.parquet`, 4543 grabaciones × 77 features (agregación
  por región y por grabación, restringida a canales válidos — detalle en
  `docs/ESTADO.md`, Fase 13).
- **Etiqueta:** VAS 0-100 mm por sesión (no por grabación ni por ventana) —
  653 sesiones / 174 personas etiquetadas. `columnas_de_features()` en
  `matrix.py` sigue siendo la única definición válida de X.
- **Agrupación obligatoria:** `persona_uid`, nunca `sujeto_id` — ver
  `docs/ESTADO.md`, Fase 12.
- **Líneas base a batir:** R² = 0.260 (media por persona, leave-one-out);
  ICC(1) = 0.457. Detalle y derivación en `docs/ESTADO.md`, Fase 12,
  "Líneas base obligatorias". **Ver la corrección de la Fase 14:** esa cifra no
  es una única línea base, son dos tareas distintas con dos rivales distintos.
- **Ruta CNN (pendiente):** construir el tensor desde las ventanas crudas de la
  Fase 11 (normalización z por canal ajustada solo sobre train, canal de
  máscara en vez de ceros para los canales enmascarados).

---

## Pendientes

- ~~**Cerrar el paquete de test y ejecutarlo una sola vez.**~~ **Hecho el
  2026-10-07**: paquete fijado en `PREREGISTRO_TEST.md` (commit `bb55aee`) y
  ejecutado de una pasada. Resultados en la sección «Resultados sobre el
  conjunto de test». **El test está gastado para las preguntas de ese
  paquete** (fatiga percibida —VAS— y condición): no se vuelve a ejecutar ni
  se usan sus cifras para decidir nada nuevo sobre ellas.
  > **Para una misma pregunta, el test no se gasta "una vez por modelo".** Se
  > gasta una vez, punto: cualquier análisis posterior del VAS o de la
  > condición sobre esos datos es exploratorio y así debe citarse.
  >
  > **Una pregunta nueva, con una etiqueta nueva, sí puede evaluarse en el
  > test** (decisión del 2026-10-27), siempre que se cumpla todo esto:
  > - que el test no se haya mirado para esa pregunta;
  > - que el modelo, las features y las reglas de decisión se fijen en un
  >   pre-registro antes de abrir `val`;
  > - que el test se abra una sola vez, y solo si `val` cumple un criterio de
  >   paso escrito de antemano.
  >
  > Primer caso: la línea de fatiga aguda por tiempo en tarea
  > (`analisis/fatiga_aguda/PLAN.md`, sección 5).
- **Ruta CNN.** No se hizo y **queda fuera del paquete de test por escrito**
  (`PREREGISTRO_TEST.md`, sección 1): es trabajo futuro. Como el test ya está
  gastado para el VAS, una CNN futura sobre el VAS no podrá evaluarse sobre él
  como prueba confirmatoria. Una CNN para una pregunta nueva, como el tiempo en
  tarea, sigue la regla anterior.
  La Fase 14 no la condiciona — mide que las features agregadas no llevan la
  señal, no que la señal cruda no la tenga.
- **Completar el registro del test:** anotar en la sección 11 de
  `PREREGISTRO_TEST.md` («Desviaciones») lo que corresponda, y archivar o
  comitear la carpeta de resultados.

---

## Fase 14 — Entrenamiento y evaluación del Random Forest

**Archivos:** `src/eeg_library/modelado/modeling.py`, `tests/modelado/test_modeling.py` (23 tests),
`analisis/analisis_fase14.py` (script reproducible)
**Salida:** `C:\Users\migue\Desktop\TFG\EEG-Data\processed\fase14_resultados.json`
**Alcance:** solo `train` + `val`. **El conjunto de test no se ha tocado.**

### Qué son las 77 features de esta fase (y de la 14b)

No son las mismas 77 de la Fase 13. Cada sesión tiene hasta 7 grabaciones (una
por condición) y **una sola** etiqueta VAS, así que `agregar_a_sesion` colapsa la
matriz a una fila por sesión: cada feature es la **media de esa feature sobre las
grabaciones de la sesión** (ignorando NaN). Las `_media` pasan a ser la media de
las medias; las `_sd`, la media de las desviaciones *dentro* de cada grabación
(variabilidad ventana a ventana promedio), no la dispersión entre grabaciones.
Una sesión incompleta promedia otra mezcla de condiciones que una completa.

La composición es 76 + 1: las 77 de la Fase 13 **menos `condicion_orden`**
(promediada vale ~4 en toda sesión completa) **más `n_grabaciones`** (1-7; 503 de
las 547 sesiones de train+val tienen las 7). La X la define
`columnas_de_features_sesion`, derivada de `columnas_de_features`.
`n_grabaciones` no está asociada al VAS (ρ = −0.052, p = 0.22, n = 547), pero se
conserva para que el código siga produciendo las cifras citadas. Una reejecución
futura podría excluirla (76 features) y obligaría a actualizar estas tablas; el
test `test_columnas_de_features_sesion_conserva_n_grabaciones` avisa si alguien
la quita sin hacerlo. La Fase 16 no se ve afectada: trabaja por grabación, donde
no existe `n_grabaciones`.

### Por qué este paso y no "entrenar y ver qué sale"

La Fase 13 dejó una prueba de humo con R² = −0.015 y un diagnóstico: el tamaño
de efecto máximo de cualquier feature es ρ ≈ 0.14, así que afinar
hiperparámetros no va a arreglar nada. Con eso sobre la mesa, lanzar más
Random Forests a ver si alguno sube no es un paso, es una lotería — y una
lotería peligrosa, porque probando suficientes configuraciones siempre sale
alguna con R² positivo por azar, y quedarse con esa es el fraude estadístico
más común en TFGs de machine learning.

Lo que faltaba no era un modelo mejor, era **un banco de pruebas que hiciera
imposible engañarse**. Por eso este paso construye primero las reglas de
evaluación (como módulo de librería, con tests) y solo después entrena. Las
tres reglas:

1. **Evaluar a nivel de sesión, no de grabación.** La prueba de humo de la
   Fase 13 midió sobre 3728 grabaciones, pero solo hay 547 etiquetas
   independientes: las 7 grabaciones de una sesión comparten el mismo VAS al
   milímetro. Medir por grabación no añade información y estrecha falsamente
   todas las barras de error.
2. **Partir por persona.** Ver el control del apartado siguiente.
3. **Separar dos tareas que la Fase 12 tenía mezcladas.** Es la corrección
   principal de esta fase.

### Corrección: la línea base 0.260 no aplica a la tarea que se estaba evaluando

`docs/ESTADO.md` (Fase 12) fija dos cosas que, juntas, no se sostienen:

- el protocolo de evaluación es `GroupKFold` por `persona_uid` — el modelo
  nunca ve a la persona que evalúa;
- la línea base a batir es R² = 0.260, la media de la persona calculada sobre
  sus **otras** sesiones.

Pero esa media solo existe si se conocen otras sesiones etiquetadas de esa
persona, que es justo lo que `GroupKFold` prohíbe. **Son dos escenarios de uso
distintos, y compararlos entre sí no significa nada:**

| | Tarea A — persona nueva | Tarea B — persona conocida |
|---|---|---|
| Situación | El casco se le pone a alguien por primera vez | Se hace seguimiento a lo largo del curso |
| Qué se conoce | Solo su EEG | Su EEG **y** su nivel habitual de VAS |
| Qué predice el modelo | El VAS | La **desviación** respecto a su nivel habitual |
| Rival | Media global, R² = 0.000 | Media por persona LOO, R² = 0.299 |
| Partición | GroupKFold por persona | GroupKFold por persona (+ centrado LOO) |

La Tarea B es la científicamente interesante: el 54 % de la varianza del VAS es
intra-persona (ICC = 0.457), así que ahí es donde hay algo que explicar. Pero
tiene una trampa que hay que cerrar explícitamente, y es la razón de que el
módulo centre **X e y a la vez**: si solo se centrase `y`, el bosque podría
reconocer a la persona por su EEG y devolver su nivel habitual, marcándose un
R² = 0.30 que no es fatiga sino reidentificación. Centrando ambos lados, el eje
entre-personas desaparece de la entrada y de la salida, y reconocer a la
persona deja de servir de nada.

### El control que justifica todo lo anterior

Antes de creerse ninguna cifra, la pregunta es si el riesgo de fuga es real:

| | Resultado |
|---|---|
| Identificar a la persona desde las features de sesión (132 clases) | **0.248** |
| Azar | 0.0076 |
| | **33× el azar** |

Sí lo es. Con 547 sesiones y una partición aleatoria por sesión, el modelo
reconocería a la persona en una de cada cuatro y devolvería su nivel habitual.
Todo el R² que saliera de ahí sería falso. Hay un test de regresión
(`test_agrupar_por_persona_impide_predecir_lo_que_es_pura_identidad`) que monta
un dataset donde la etiqueta es pura identidad y comprueba que agrupar la
vuelve impredecible (R² 0.96 → ≈0) — para que nadie vuelva a quitar el
`GroupKFold` sin que salte algo.

### Líneas base (547 sesiones, 146 personas, train+val)

| Regla sin EEG | R² | MAE |
|---|---|---|
| Media global | 0.0000 | 18.01 |
| Media por `medicion` | 0.0146 | 17.72 |
| Media por `centro` | 0.0154 | 17.69 |
| Media por persona, **en muestra** (circular) | 0.6099 | 10.47 |
| **Media por persona, leave-one-out** | **0.2961** | 14.23 |

> **El banco queda validado contra la Fase 13.** Restringido a las mismas 537
> sesiones que usa la prueba de humo (las de personas con ≥2 sesiones), da
> R² = 0.2988 y MAE = 14.2517, frente al 0.299 / 14.25 que reporta `docs/ESTADO.md`.
> La diferencia con el 0.2961 de la tabla es solo que aquí las 10 sesiones de
> personas con una única sesión reciben la media global en vez de descartarse.

### Tarea A — persona nueva

`GroupKFold(5)` por `persona_uid`: ~117 personas entrenan, ~29 se evalúan, y
cada sesión recibe una predicción de un modelo que nunca vio a su participante.

| Modelo | R² | MAE | ρ Spearman | p |
|---|---|---|---|---|
| *media global* | 0.0000 | 18.01 | — | — |
| RF 500, hoja≥5 | −0.0102 | 18.17 | +0.117 | 0.006 |
| RF 500, hoja≥1 | −0.0079 | 18.12 | +0.127 | 0.003 |
| RF 500, hoja≥15 | **+0.0021** | 18.09 | +0.126 | 0.003 |
| RF 500, hoja≥5, `max_features`=0.3 | −0.0091 | 18.13 | +0.109 | 0.011 |
| Ridge (imputado + escalado) | −0.0678 | 18.61 | +0.008 | 0.849 |

**R² = −0.010, IC 95 % [−0.074, +0.038]** (bootstrap remuestreando personas).
El intervalo contiene el cero holgadamente: no hay capacidad predictiva.

Se probaron cinco configuraciones y **se reportan las cinco**, no la mejor. La
mejor (hoja≥15, R² = +0.0021) está dentro del ruido de las demás; quedarse solo
con ella sería exactamente el sesgo de selección que este banco existe para
evitar.

> **Pero ρ = +0.12 con p = 0.006 no es cero.** El modelo no acierta el número
> —predecir la media global da menos error— pero **ordena** las sesiones algo
> mejor que el azar, de forma consistente en las cuatro configuraciones de RF.
> Es un efecto real y diminuto: ordena en la dirección correcta y es incapaz de
> decir cuánto.

### Tarea B — persona conocida

537 sesiones / 136 personas (se caen las 10 de personas con una sola sesión,
que no tienen nivel propio estimable).

En la escala de la **desviación**, que es lo que el modelo predice:

| Modelo | R² | MAE | ρ | p |
|---|---|---|---|---|
| *sin EEG (desviación 0)* | 0.0000 | 14.25 | — | — |
| RF 500, hoja≥5 | −0.0029 | 14.52 | +0.101 | 0.020 |
| RF 500, hoja≥15 | +0.0025 | 14.45 | +0.103 | 0.017 |
| Ridge | −0.0315 | 14.43 | +0.109 | 0.011 |

Reconstruido en la escala original del VAS (`nivel propio + desviación
predicha`), que es lo comparable con las líneas base:

| | R² | IC 95 % |
|---|---|---|
| Sin EEG (media persona LOO) | **0.2988** | [+0.166, +0.405] |
| Con EEG (RF 500, hoja≥5) | 0.2967 | [+0.173, +0.394] |

**El EEG no aporta nada por encima de conocer a la persona.** Los intervalos se
solapan casi por completo y la diferencia (−0.002) va en el sentido contrario.
Todo el 0.30 sale de saber quién es el participante; ninguna parte sale de su
señal cerebral.

### Reducción de dimensión con selección dentro de la validación cruzada

Era la recomendación 2 de la Fase 13, y ahora está motivada por medición: con
77 features y 537 sesiones, un efecto de ρ = 0.14 se ahoga en el ruido de las
otras 76 columnas. La selección se repite **dentro de cada pliegue** —la
mediana de imputación, la tipificación, el ranking de correlaciones y la recta
salen solo del pliegue de entrenamiento— para que el número final ya descuente
el optimismo de haber elegido mirando los datos.

| k | R² (desviación) | ρ | p | R² (escala original) |
|---|---|---|---|---|
| 3 | −0.0023 | +0.113 | 0.009 | 0.2972 |
| 5 | +0.0055 | +0.117 | 0.007 | **0.3026** |
| 10 | −0.0132 | +0.120 | 0.005 | 0.2895 |
| 16 | +0.0034 | **+0.144** | **0.0008** | 0.3011 |
| 25 | +0.0001 | +0.130 | 0.003 | 0.2988 |

Dos lecturas, y las dos importan:

- **El R² sigue siendo cero.** El mejor caso sube de 0.2988 a 0.3026: cuatro
  milésimas, dentro del ruido. Reducir la dimensión no rescata la predicción.
- **El ρ es estable y significativo en las cinco configuraciones** (p entre
  0.0008 y 0.009), con selección honesta. Eso ya no se explica por azar de
  selección: hay una relación monótona débil, reproducible, entre estas
  features y la fatiga percibida.

Las features que sobreviven a la selección **en los cinco pliegues** son las
mismas que señaló la Fase 13, lo que refuerza que no es un artefacto de una
partición concreta:

`frontal_delta_rel_sd` (en los 5 pliegues y en las 5 configuraciones),
`frontal_alfa_rel_sd`, `posterior_delta_rel_sd`, `frontocentral_alfa_rel_sd`,
`posterior_alfa_rel_sd`, `posterior_theta_rel_media`.

Cinco de las seis son `_rel_sd`: **variabilidad** del espectro a lo largo de la
grabación, no su nivel. El índice compuesto de la Fase 13 (media z de las 16
`_rel_sd`) da a nivel de sesión ρ = +0.1561, p = 0.00028, n = 537 — reproduce
el +0.157 de la Fase 13, así que la señal sobrevive a la agregación por sesión.

### En qué se fija el bosque

| Feature | Importancia |
|---|---|
| `frontocentral_delta_rel_sd` | 0.0367 |
| `frontocentral_beta_rel_sd` | 0.0314 |
| `posterior_theta_rel_media` | 0.0288 |
| `frontal_beta_rel_sd` | 0.0281 |
| `posterior_delta_rel_sd` | 0.0227 |

Una feature neutra valdría 1/77 = 0.0130, así que el máximo es solo **2.83×**
lo uniforme. Sigue siendo un reparto casi plano —la firma de un bosque que no
encuentra nada dominante— pero ya no es *totalmente* plano como en la Fase 13,
y las que asoman son otra vez las `_rel_sd`. Tres métodos independientes
(correlación intra-persona, importancia del bosque, selección anidada) apuntan
al mismo sitio.

### Conclusión

**El Random Forest sobre features espectrales agregadas no predice la fatiga
percibida en esta cohorte**, ni para una persona nueva (R² = −0.010, IC
[−0.074, +0.038]) ni como añadido a conocer su nivel habitual (0.2967 frente a
0.2988 sin EEG). No es un fallo de implementación ni de ajuste: los controles
positivos muestran que la matriz lleva muchísima información EEG real —
identifica a la persona 33× por encima del azar, y en la Fase 13 acertaba la
condición con 0.608 sobre 7 clases (azar 0.143). **Corregido el 2026-08-26:**
la cifra 0.978 que figuraba aquí incluía `condicion_orden`, que es la propia
condición codificada, así que era circular. Lo que no lleva es información
sobre el VAS.

Lo único que sobrevive a una evaluación honesta es una relación **de orden,
débil y consistente**: ρ ≈ +0.12–0.14 (p < 0.01), concentrada en la
variabilidad del espectro (`_rel_sd`) y no en su nivel, y en la dirección
"más fluctuación → más fatiga reportada". Es demasiado pequeña para predecir un
número, pero es reproducible bajo tres métodos distintos y bajo selección
anidada sin fuga.

Para la memoria del TFG esto es un resultado, no un fracaso: el pipeline está
validado extremo a extremo, el biomarcador clásico theta/alfa se midió y **no
sigue a la fatiga percibida** dentro de persona (ρ = +0.004, p = 0.93), y lo
que sí asoma es un marcador distinto del que predice la literatura.

### Qué queda por decidir

> **Superado por la Fase 14b (2026-08-25).** El punto 2 se ejecutó y cambió el
> punto 1: el compuesto `_rel_sd` restringido rinde **peor** que el conjunto
> completo en la tarea de tres niveles (0.348 [0.309, 0.387] frente a 0.395
> [0.348, 0.438]), así que deja de ser la candidata principal del test. Lo que
> sigue se conserva como registro de lo que se pensaba antes de medirlo.

1. **Gastar o no el test con la hipótesis `_rel_sd`.** Es la decisión más
   importante y solo se puede tomar una vez. A favor: la hipótesis está ahora
   pre-registrada (features fijadas en la Fase 13, confirmadas por selección
   anidada en la 14) y 106 sesiones / 28 personas darían un ρ con su intervalo,
   reportable en la memoria. En contra: con n = 28 personas el intervalo será
   ancho, y un ρ = 0.14 podría salir no significativo sin que eso signifique
   que no existe. **Recomendación: hacerlo, una sola vez, con el compuesto de
   las 16 `_rel_sd` fijado de antemano**, y reportar el intervalo salga lo que
   salga.
2. **Clasificación en tres clases** como análisis secundario. El VAS es ordinal
   de facto (76.7 % múltiplos de 10) y una señal que ordena pero no cuantifica
   —que es exactamente lo que hay— puede rendir mejor en tercios que en
   regresión. Barato de probar sobre train+val.
3. **La ruta CNN sigue viva.** Esta fase mide que las features *agregadas* no
   llevan la señal: 4 s de ventana resumidos a media y desviación por región y
   banda. No mide que no esté en la señal cruda. Es el argumento más fuerte
   para justificar la CNN en la memoria.

### Lo que **no** hay que hacer

- Afinar más hiperparámetros del RF. Cinco configuraciones cubren desde
  `hoja≥1` hasta `hoja≥15` y todas dan lo mismo dentro del ruido. El problema
  es el tamaño de efecto, no el ajuste.
- Volver a mirar train+val buscando features nuevas. Cada mirada adicional
  consume validez: el compuesto `_rel_sd` ya se eligió ahí, y añadirle
  hallazgos posteriores lo contaminaría.
- Reportar el 0.30 de la Tarea B como logro del modelo. Es la persona, no el
  EEG.

---

## Fase 14b — ¿Aporta tratar la fatiga como tres niveles?

**Archivos:** `src/eeg_library/modelado/modeling.py` (funciones de clasificación ordinal),
`tests/modelado/test_modeling.py` (40 tests), `analisis/analisis_fase14b_clases.py`

**Solo train+val.** El test sigue sin tocarse.

### Por qué

La Fase 14 deja un diagnóstico preciso: la señal disponible **ordena**
(ρ ≈ +0.12, p < 0.01) pero **no cuantifica** (R² ≈ 0). Una regresión exige el
número exacto y por tanto no puede aprovechar una señal así. Tres niveles
(poca / media / mucha) solo exigen el escalón. La pregunta es si esa métrica
más indulgente recupera algo real o simplemente disimula el mismo vacío.

Se hace **antes** de decidir el paquete de test y sobre train+val, porque cada
análisis extra sobre el test sube el listón de todos los demás: solo debe entrar
lo que aquí demuestre que merece la pena.

### Un artefacto que hubo que corregir primero

La primera ejecución dio una línea base absurda: un `DummyClassifier`, que
ignora X por completo, sacaba **0.407 de exactitud balanceada** (el azar es
1/3 exacto) y un Kendall tau de **+0.105 con p = 0.0098**. Información salida
de la nada.

La causa: los cortes de los tercios se estiman dentro de cada pliegue, así que
**la clase "media" de un pliegue no está cortada donde la de otro**. Amontonar
las predicciones de los cinco pliegues y puntuarlas de golpe mezcla escalas, y
la variación de los cortes entre pliegues se convierte en señal espuria. Es un
problema exclusivo de la discretización: en la Fase 14 la etiqueta era la misma
en todos los pliegues y agrupar era correcto (y preferible, porque el R² por
pliegue es inestable con ~30 personas).

Solución: `metricas_clasificacion_por_pliegue` puntúa **dentro** de cada pliegue
y promedia. Con ella el `DummyClassifier` vuelve a su 1/3 exacto. Hay un test
que lo fija (`test_puntuar_por_pliegue_devuelve_el_azar_a_un_clasificador_sin_informacion`).
Todas las cifras de abajo están calculadas así.

### Tarea A — persona nueva, tercios absolutos de VAS

Cortes por pliegue: p33 = 30.0 en los cinco; p67 entre 50.0 y 55.0. La
concentración de respuestas en múltiplos de 10 hace que los tercios no salgan
equilibrados (128 / 215 / 204 sesiones).

| Modelo | Exactitud balanceada | F1 macro | Adyacente | tau |
|---|---|---|---|---|
| *línea base (mayoritaria)* | **0.3333** | 0.2125 | 0.9032 | n/d |
| RF 500, hoja≥5 | 0.3103 (±0.033) | 0.2902 | 0.8940 | +0.036 |
| RF 500, hoja≥1 | 0.3246 (±0.069) | 0.3096 | 0.8940 | +0.061 |
| RF 500, hoja≥15 | 0.3289 (±0.024) | 0.2933 | 0.8886 | +0.030 |
| RF 500, hoja≥5, balanceado | 0.3335 (±0.052) | 0.3287 | 0.8702 | +0.070 |
| regresión discretizada | 0.3385 (±0.011) | 0.2387 | 0.9872 | +0.105 |

RF principal: **0.3103, IC 95 % [0.2770, 0.3475]** — el intervalo contiene el
azar y la media está por debajo. Por pliegue: 0.360 / 0.281 / 0.320 / 0.309 /
0.282, ninguno destacable.

**Nada.** Coherente con la Fase 14 (R² = −0.010): para una persona nueva, estas
features no dicen cuánta fatiga tiene.

### Tarea B — persona conocida, tercios de la desviación propia

537 sesiones, 146 personas (10 descartadas: persona con una sola sesión, sin
referencia propia). X e y centrados a la vez por la media leave-one-out de la
persona, para que reidentificarla no sirva de nada.

| Modelo | Exactitud balanceada | F1 macro | Adyacente | tau |
|---|---|---|---|---|
| *línea base (mayoritaria)* | 0.3333 | 0.1713 | 0.6723 | n/d |
| **RF 500, hoja≥5** | **0.3947** (±0.043) | 0.3905 | 0.7950 | +0.105 |
| RF 500, hoja≥1 | 0.3784 (±0.027) | 0.3722 | 0.7988 | +0.102 |
| RF 500, hoja≥15 | 0.3853 (±0.052) | 0.3788 | 0.7894 | +0.087 |
| RF 500, hoja≥5, balanceado | 0.3774 (±0.046) | 0.3748 | 0.8044 | +0.099 |
| regresión discretizada | 0.3331 (±0.038) | 0.2211 | 0.9740 | +0.045 |

RF principal: **0.3947, IC 95 % [0.3484, 0.4376]**. El intervalo **excluye el
azar**. Por pliegue: 0.412 / 0.453 / 0.393 / 0.378 / 0.337 — los cinco en el
mismo lado, cuatro claramente por encima. Las cuatro configuraciones de RF caen
entre 0.377 y 0.395: no es una elegida entre muchas, es el rango entero.

Matriz de confusión (filas = real, columnas = predicha):

|  | poca | media | mucha |
|---|---|---|---|
| **poca** | 73 | 45 | 58 |
| **media** | 50 | 57 | 68 |
| **mucha** | 52 | 53 | 81 |

La diagonal domina en los extremos (poca→poca 73, mucha→mucha 81) y se pierde
en el centro (media→media 57 de 175). Es exactamente el patrón de una señal que
distingue los extremos y no resuelve el medio.

**Es el primer resultado del proyecto que bate su línea base con un intervalo
que excluye el azar.** Y es modesto: 0.3947 frente a 0.3333 son ~6 puntos
porcentuales, un 18 % relativo.

### La respuesta a la pregunta: sí, clasificar aporta — y se sabe por qué

La comparación decisiva es la última fila de cada tabla. En la Tarea B:

- **clasificador entrenado sobre las clases: 0.3947**
- **regresor entrenado sobre el continuo y luego discretizado: 0.3331** (azar)

El mismo bosque, los mismos datos, la misma partición: la diferencia es solo si
se le pide un número o un escalón. El mecanismo se ve en el reparto de lo que
predice cada uno sobre las 537 sesiones:

| | poca | media | mucha |
|---|---|---|---|
| clases reales | 176 | 175 | 186 |
| predichas por el clasificador | 175 | 155 | 207 |
| predichas por la regresión | 20 | **481** | 36 |

La desviación real del VAS tiene sd = 18.2 y rango [−50, +50], pero las
predicciones del regresor apenas se separan del cero — el comportamiento normal
de un modelo con poca señal, que se refugia en la media para minimizar el error
cuadrático. Al discretizarlas, el 90 % cae en el escalón central y el orden se
pierde. Su exactitud *adyacente* de 0.974 es el síntoma: casi nunca se equivoca
por dos escalones porque casi nunca sale del medio.

El clasificador no tiene esa salida: la entropía cruzada le obliga a repartirse
entre las tres clases, y al comprometerse conserva el poco orden que hay. **No
hay más señal en la tarea de tres niveles; hay una métrica y una función de
pérdida que sí pueden usar la que había.**

### El compuesto `_rel_sd` en tres niveles: peor que usar todo

| Modelo (solo las 16 `_rel_sd`) | Exactitud balanceada | IC 95 % | tau |
|---|---|---|---|
| RF 500, hoja≥5 | 0.3485 (±0.041) | [0.3093, 0.3865] | +0.068 |
| RF 500, hoja≥5, balanceado | 0.3443 (±0.033) | [0.3056, 0.3828] | +0.074 |

Ambos intervalos **contienen el azar**, y ambos quedan por debajo del RF sobre
las 77 features (0.3947). Restringirse a las `_rel_sd` **empeora** la
clasificación.

Esto no contradice la Fase 14 —allí las `_rel_sd` eran las features con más
señal *individual*— pero sí matiza la conclusión: la señal individual más fuerte
no es lo mismo que el mejor conjunto predictor. El bosque sobre las 77 encuentra
combinaciones que las 16 solas no dan.

**Consecuencia para el test:** el candidato principal ya no es el compuesto
`_rel_sd`, sino el clasificador de tres niveles sobre el conjunto completo.

### Conclusión

1. Para **persona nueva** no hay nada, ni en regresión ni en tres niveles. Es un
   resultado consistente medido de dos formas independientes.
2. Para **persona conocida** hay un efecto pequeño, reproducible en los cinco
   pliegues y en las cuatro configuraciones, cuyo intervalo excluye el azar:
   exactitud balanceada 0.3947 [0.3484, 0.4376] frente a 0.3333.
3. La clasificación **sí aporta** frente a la regresión, y por una razón
   entendible y reportable en la memoria: la señal ordena pero no cuantifica, y
   la regresión no tiene forma de usar eso.
4. El `_rel_sd` restringido queda **descartado como primaria** del test.

### Qué cambia respecto a la Fase 14

La recomendación de la Fase 14 ("gastar el test con el compuesto `_rel_sd`") se
sustituye por el paquete de test de la Fase 14c, con el clasificador de tres
niveles como prueba principal.

### Lo que **no** hay que hacer

- Puntuar clasificación amontonando pliegues cuyos cortes difieren. Da 0.407 a
  un modelo sin información. Usar `metricas_clasificacion_por_pliegue`.
- Fijar los cortes de los tercios sobre todo el conjunto "porque es más
  cómodo": filtra la distribución de la etiqueta de las personas evaluadas.
- Leer la exactitud sin balancear. Con tercios desequilibrados (Tarea A:
  128/215/204) premia predecir siempre la clase mayoritaria.
- Presentar el 0.3947 como "el modelo acierta el 39 %". Es acierto medio por
  clase sobre tres clases, con azar en 33 %. La mejora es de ~6 puntos.
- Seguir probando variantes sobre train+val. Con esto la exploración se cierra:
  cada mirada adicional consume validez del test.

---

## Control de la deriva por número de visita (2026-10-06)

**Archivos:** `src/eeg_library/modelado/modeling.py` (`numero_de_visita`,
`residualizar_por_visita`, `compuesto_rel_sd`), `tests/modelado/test_modeling.py`,
`analisis/analisis_control_visita.py`
**Salida:** `<EEG_DATA>/processed/control_visita.json`,
`resultados/logs/control_visita.log`

**Solo train+val.** El test se aparta al leer la matriz, antes de agregar.

### Por qué

Las sesiones de una persona están ordenadas en el tiempo (M2 … M6). Si el VAS y
el EEG derivan a la vez a lo largo del curso (impedancia, gel, habituación al
protocolo), aparece una relación intra-persona entre ambos sin que el EEG lea
la fatiga. La deriva del VAS existe: ρ intra-persona (VAS, visita) = +0.119,
IC95 por personas [+0.009, +0.229], n = 537.

El control ya se midió en la Fase 15 (retirada el 2026-09-15; su código no se
llegó a commitear). Se **reincorpora** porque el pre-registro del test fija la
cifra controlada como confirmatoria en los tres análisis intra-persona del VAS.
El método es el de la Fase 15, recuperado del historial
(`fase15e_reposo.py`, `fase15f_confusor_visita.py`) y ahora con tests:

- **Visita** = número de `medicion` (M2 → 2 … M6 → 6), no el orden de la visita
  dentro de la persona.
- **Residuo** de una recta de mínimos cuadrados sobre la visita ajustada **dentro
  de cada persona**, en X y en y. Las personas con menos de 3 valores válidos o
  una sola visita distinta se dejan sin residualizar (con dos puntos el residuo
  sería 0 por construcción): 132 de 136 personas en la Tarea B.
- **Prueba principal:** se aplica sobre X e y ya centrados por la media LOO de
  la persona; los cortes de los tercios siguen saliendo de cada pliegue.
- **θ/α y `_rel_sd`:** se centra por la media de la persona (como en
  `docs/analisis/ANALISIS_REGRESION_LINEAL.md`) y después se residualiza. Cada recta gasta un grado de
  libertad más en la regresión intrapersona.

Es un control **conservador**: si la fatiga real evolucionase de forma monótona
durante el curso, también se llevaría señal legítima. Se lleva el 38 % de la
varianza intra-persona de la desviación del VAS (453.9 → 282.2).

### Resultado

| Análisis | Sin control | Con control de visita |
|---|---|---|
| **Principal** (14b, Tarea B, RF hoja≥5): exactitud balanceada | 0.3947, IC95 [0.3484, 0.4376] | **0.3795, IC95 [0.3302, 0.4290]** |
| ↳ por pliegue | 0.412 / 0.453 / 0.393 / 0.378 / 0.337 | 0.428 / 0.344 / 0.412 / 0.330 / 0.384 |
| ↳ tau de Kendall | +0.105 | +0.092 |
| θ/α intrapersona: pendiente β (mm) | −0.073, IC95 t [−1.61, +1.46], p = 0.925 | −0.455, IC95 t [−2.15, +1.23], p = 0.596 |
| ↳ ρ | +0.004 | −0.059 |
| Compuesto `_rel_sd`: ρ intra-persona | +0.156, IC95 [+0.063, +0.245] | **+0.130, IC95 [+0.007, +0.247]** |

IC de la principal: bootstrap de 2000 réplicas por personas (el de la Fase 14b);
los de θ/α y `_rel_sd`, 10 000 (el de `docs/analisis/ANALISIS_REGRESION_LINEAL.md`).

**Reproducción de lo documentado.** Las tres cifras sin control reproducen las
publicadas: 0.3947 con el mismo IC (Fase 14b), β = −0.073 con p = 0.925
(`docs/analisis/ANALISIS_REGRESION_LINEAL.md`) y ρ = +0.1559 para el compuesto frente al +0.1561 de la
Fase 14 (p = 0.00029 en ambos). El código de ese compuesto no se conservó; la
definición reconstruida (z de cada `_rel_sd` sobre las sesiones de train+val,
media de las 16, centrado por persona, Spearman, n = 537) es la más cercana de
las cinco variantes probadas, y es la que queda fijada para el test. Con
control, la principal reproduce el 0.3795 de la Fase 15 (su IC, de 1000
réplicas, era [0.3302, 0.4255]).

### Lectura

- **La prueba principal aguanta en el punto pero no en el intervalo.** Con
  control baja 1.5 puntos y su IC95 **contiene el azar**. La referencia de
  validación del pre-registro es 0.3795 (+4.6 puntos sobre 1/3), no 0.3947.
- **θ/α:** sigue sin relación dentro de persona, con o sin control.
- **`_rel_sd`:** baja de +0.156 a +0.130 y el IC queda justo por encima de 0.
  Parte de la asociación era deriva común.

Añade **3 análisis** al recuento de lo explorado en train+val.

---

## Potencia de la prueba principal en test (2026-10-06)

**Archivo:** `analisis/analisis_potencia_test.py` · **Salida:**
`<EEG_DATA>/processed/potencia_test.json`, `resultados/logs/potencia_test.log`

**Solo train+val.** Se toman las predicciones out-of-fold de la prueba
principal (Tarea B, tres niveles, RF hoja≥5) y se simulan 5000 tests
remuestreando 28 personas con reposición entre las 136 elegibles. En cada uno
se aplica la regla pre-registrada: exactitud balanceada amontonada con un solo
juego de cortes y p unilateral por permutación de las clases reales dentro de
cada persona, con la definición convencional `(1 + #{nulo ≥ obs}) / (N + 1)` y
1000 permutaciones por simulación.

Puntuar amontonado es legítimo aquí porque los cortes apenas cambian entre
pliegues (con control: [−4.6, +5.0] a [−4.7, +5.3]). La exactitud balanceada
amontonada es 0.3817, frente a 0.3795 por pliegue, y un `DummyClassifier`
amontonado da 0.3281.

| | Con control (confirmatoria) | Sin control |
|---|---|---|
| Referencia de validación | 0.3795 | 0.3947 |
| Sesiones por test simulado | mediana 111 [105, 116] | ídem |
| Error típico de la exactitud balanceada | 0.054 | 0.049 |
| **Potencia, P(p < 0.05 unilateral)** | **0.33** (0.329) | 0.49 |
| ↳ con mid-p, solo como comparación | 0.33 (0.331) | 0.50 |
| P(IC95 excluye el azar) | 0.16 | 0.25 |
| P(IC95 contiene el azar y la referencia) | **0.81** | 0.71 |
| P(IC95 por debajo de la referencia) | 0.03 | 0.04 |
| Efecto detectable con potencia del 80 % (aprox.) | ≈ 0.467 | ≈ 0.455 |

**Control de error de tipo I** (predicciones barajadas de nuevo en cada
simulación, 20 000 simulaciones):

| | α = 0.05 | α = 0.01 |
|---|---|---|
| **p convencional** (el del pre-registro) | 0.0524 ± 0.0016 | 0.0109 ± 0.0007 |
| mid-p | 0.0536 ± 0.0016 | 0.0112 ± 0.0007 |

Una segunda tanda independiente de 20 000 (otra semilla, fuera del guion) dio
0.0487 ± 0.0015 y 0.0493 ± 0.0015 a α = 0.05; juntas, el p convencional queda en
≈ 0.0506 ± 0.0011, en el nominal. Los dos p deciden distinto en el 0.1 % de las
simulaciones.

**Por qué el p convencional y no el mid-p.** El mid-p se eligió primero porque
una versión defectuosa de este control —un único barajado reutilizado en todas
las simulaciones— daba 0.022 y hacía parecer conservador al p convencional.
Corregido el control, los dos tienen el mismo error de tipo I y la misma
potencia, y solo el convencional garantiza un error ≤ α por construcción. Se
vuelve a él (decisión del 2026-10-06).

**Lectura.** Si el efecto real es el de validación, el test lo detecta una de
cada tres veces. El resultado más probable (81 %) es un intervalo que no
distingue el azar del efecto de validación: «no concluye». Un resultado no
significativo en test **no** es evidencia contra el efecto. Para tener un 80 %
de potencia, la exactitud balanceada real tendría que rondar 0.47. Supuestos:
el efecto real es el estimado (si es menor, la potencia también) y las personas
de test aportan las mismas sesiones que las de train+val. Que los modelos de
validación salgan de ~109 personas y el final de 146 solo podría hacer la cifra
algo conservadora.

---

## Fase 16 — PCA + clustering: ¿hay estructura de fatiga sin supervisión?

**Archivos:** `analisis/analisis_pca_clustering.py` (script reproducible)
**Salida:** `C:\Users\migue\Desktop\TFG\EEG-Data\processed\pca_clustering.json`,
`resultados/logs/pca_clustering.log`
**Figuras:** `resultados/figuras/pca_clustering_vas_extremos.png`,
`resultados/figuras/pca_clustering_vas_extremos_reposo.png`
**Fecha:** 2026-08-27

**Solo `train` + `val`.** El test se aparta en el paso 1 (732 grabaciones,
28 personas) y no se vuelve a leer.

### Advertencia de estatus

**Nada de esto es confirmatorio.** Aporta 3 configuraciones más de exploración
sobre train+val, que hay que sumar al recuento total que se declare al fijar el
alfa del paquete de test.

### La pregunta

La Fase 14 preguntó si un modelo *supervisado* puede predecir el VAS.
Esta pregunta es anterior y más débil: **¿existe siquiera estructura de fatiga en
el espacio de features, la vea o no un clasificador?** Si las grabaciones de
sesiones con mucha fatiga (VAS ≥ 80) y poca (VAS ≤ 20) formaran dos nubes
separadas, un algoritmo que no ve la etiqueta debería encontrarlas solo.

El clustering **no ve el VAS**. El grupo se usa únicamente después, para evaluar.

### Diseño

| | |
|---|---|
| Unidad | La **grabación** (no la sesión) — 1 fila por grabación |
| Muestra | 1076 grabaciones: 257 con VAS ≥ 80, 819 con VAS ≤ 20 |
| Descartadas | 2735 grabaciones de VAS intermedio |
| Cobertura | 157 sesiones, 82 personas |
| Features | Las 77 de `columnas_de_features()`, menos `condicion_orden` |
| Ajuste | Cada feature menos la media de **su propia condición** |
| Reducción | PCA a 15 componentes |
| Clustering | K-means, k = 2, `n_init=50`, semilla 42 |

**`condicion_orden` se cae sola.** Es la condición codificada, así que centrar
por condición la deja constante a 0. Se descarta por varianza nula: quedan 76
features. Es la comprobación de que el ajuste hace lo que dice.

**Los NaN se rellenan con la media de su condición** (equivalente: centrar
ignorando NaN y rellenar con 0). Una región sin canales válidos no inventa señal
ni arrastra el NaN al PCA.

### El problema que hay que declarar antes de leer ningún resultado

**El grupo de VAS y la identidad del participante son casi la misma partición.**

| | |
|---|---|
| Personas en los grupos extremos | 82 |
| Presentes en **ambos** grupos | **4 (4.9 %)** — `C1_F100`, `C2_F103`, `C6_F035`, `C6_F038` |
| Presentes en un solo grupo | **78 (95.1 %)** |
| Grabaciones aportadas por las de ambos | 76 (7.1 %) |

Solo 4 personas de 82 aportan contraste *dentro* del individuo. Para las otras
78, "pertenece al grupo alto" y "es esta persona" son la misma etiqueta.

Esto no es un detalle: en este pipeline las features **identifican al
participante 33 veces por encima del azar** (Fase 14, control de identidad). Un
clustering que hubiera separado limpiamente los dos grupos **no habría sido
evidencia de fatiga** — habría sido, casi con seguridad, identidad. El resultado
negativo esquiva el problema, pero la limitación queda registrada porque **acota
lo que este diseño podía haber demostrado en el mejor de los casos**.

Añádase que las grabaciones no son independientes: las 7 condiciones de una
sesión comparten literalmente el mismo VAS. El *n* efectivo para cualquier
afirmación sobre la fatiga son **157 sesiones / 82 personas**, no 1076.

### Resultado — análisis principal (7 condiciones, ajustado)

| | |
|---|---|
| Varianza explicada | PC1 25.6 %, PC2 12.4 %, 5 comp 64.1 %, **15 comp 85.5 %** |
| Tamaños de cluster | 723 / 353 |
| Silueta | 0.194 (estructura débil) |
| **ARI vs grupo VAS** | **−0.0264** |
| AMI vs grupo VAS | +0.0114 |
| Acierto de la mejor asignación | **54.3 %** |
| Clase mayoritaria (referencia) | **76.1 %** |

Contingencia:

| cluster | VAS alto | VAS bajo |
|---|---|---|
| 0 | 198 | 525 |
| 1 | 59 | 294 |

**El ARI es negativo.** No es "bajo": es *peor que el azar*. Y el acierto de la
mejor asignación posible de los 2 clusters a los 2 grupos (54.3 %) queda **22
puntos por debajo** de lo que se consigue diciendo "bajo" a todo (76.1 %). Los
clusters no están captando una versión débil del eje de fatiga: están cortando
por un eje que no tiene nada que ver con él.

### ¿Funcionó el ajuste por condición?

Sí, y hubo que medirlo bien para saberlo. **El ARI de los 2 clusters frente a la
condición no sirve como comprobación**: una partición en 2 no puede expresar una
de 7, así que sale ≈ 0 esté el efecto o no (sin ajustar: +0.0046; ajustado:
−0.0000 — indistinguibles, y ambos engañosos).

La comprobación que sí discrimina es agrupar en **k = 7** y ver si el algoritmo
recupera las condiciones:

| | ARI vs condición (k = 7) |
|---|---|
| Sin ajustar | **+0.0733** |
| Ajustado por condición | **+0.0043** |

El ajuste reduce la recuperabilidad de la condición **17 veces**. Hizo su
trabajo. (Que +0.0733 tampoco sea alto es otro asunto: k-means sobre 15
componentes es un detector flojo de una estructura de 7 grupos.)

### ¿Con qué se alinean entonces los clusters?

Con la identidad, débil pero por encima del azar:

| | Principal | Sin ajustar | Reposo |
|---|---|---|---|
| Pureza por persona | **0.791** | 0.787 | **0.924** |
| Azar (permutación, media) | 0.694 | 0.699 | 0.881 |
| Azar (p95) | 0.706 | 0.711 | 0.900 |
| AMI vs `persona_uid` | +0.0667 | +0.0575 | +0.0426 |

La pureza observada supera el p95 del nulo en los tres casos. Es coherente con
todo lo demás del proyecto: **lo que estas features llevan es quién eres, no cómo
de cansado estás.**

> El ARI frente a `persona_uid` (+0.0057) parece contradecirlo, pero no dice
> nada: comparar una partición de 2 con una de 82 lo deprime mecánicamente. Por
> eso la pregunta de identidad se responde con pureza y AMI, no con ARI.

### Robustez — solo reposo (OA + OC)

| | |
|---|---|
| Muestra | 310 grabaciones (74 alto / 236 bajo) |
| Varianza explicada | PC1 27.4 %, PC2 14.2 %, **15 comp 87.5 %** |
| Tamaños de cluster | 270 / 40 |
| Silueta | 0.360 |
| **ARI vs grupo VAS** | **−0.0508** |
| Acierto de la mejor asignación | 66.5 % (mayoritaria: 76.1 %) |

**Consistente con el principal, y algo peor.** El ARI sigue siendo negativo y de
hecho se aleja más del cero (−0.051 frente a −0.026); el acierto sigue por debajo
de la clase mayoritaria. La silueta sube a 0.360, pero eso no es una buena
noticia: el reparto 270/40 delata un cluster pequeño de grabaciones atípicas
—perfil de calidad de señal, no de fatiga—, que es justo lo que k-means encuentra
cuando no hay dos grupos que encontrar.

### Las figuras

`resultados/figuras/pca_clustering_vas_extremos.png` y `..._reposo.png`: PC1 vs PC2, rojo =
VAS ≥ 80, azul = VAS ≤ 20. En ambas, una sola nube con los dos colores
completamente entremezclados. No hay dos lóbulos, ni un gradiente, ni siquiera un
desplazamiento de centroides visible. Es la forma gráfica de un ARI negativo.

### Conclusión

**No hay estructura natural de fatiga percibida en el espacio de features, ni
con todas las condiciones ni en reposo.**

Los tres análisis coinciden: ARI negativo frente al grupo de VAS, acierto por
debajo de la clase mayoritaria, y clusters que se alinean —débilmente— con la
identidad del participante en vez de con su fatiga. El ajuste por condición
funcionó (17× menos recuperabilidad de la condición), así que el resultado no se
explica porque la tarea estuviera dominando el clustering.

Es coherente con todo lo anterior y **refuerza** el diagnóstico de la Fase 14:
el problema no es el modelo supervisado ni la elección de hiperparámetros. Es
que las 77 features agregadas por región y grabación **no contienen un eje de
fatiga percibida separable**. Un método no supervisado, que no puede
sobreajustar a la etiqueta porque no la ve, tampoco lo encuentra.

Lo que **no** demuestra: que la señal cruda no lo tenga. La ruta CNN sigue sin
estar condicionada por esto, por la misma razón que en la Fase 14.

### Lo que **no** hay que hacer

- **Presentar este resultado sin la confusión identidad/grupo al lado.** Con el
  95.1 % de las personas en un solo grupo, un resultado *positivo* aquí no
  habría sido interpretable. El negativo sí lo es, pero la asimetría hay que
  contarla.
- **Usar el ARI de k = 2 frente a la condición como prueba de que el ajuste
  funcionó.** No discrimina. La comprobación es a k = 7.
- **Usar el ARI para preguntar por la identidad.** Cardinalidades de 2 y 82: usar
  pureza por persona con su nulo por permutación.
- **Leer la silueta de 0.360 del reposo como estructura de fatiga.** Es un
  cluster de 40 grabaciones atípicas.
- **Contar las grabaciones como *n*.** Son 1076 filas, pero 157 sesiones y 82
  personas.
- **Probar más valores de k o más métodos de clustering.** El ARI es *negativo*
  en los tres análisis; no es un problema de sintonía. Cada intento adicional
  engorda el recuento de configuraciones que hay que declarar.

---

## Fase 17 — Ojos abiertos frente a ojos cerrados: control positivo del pipeline

**Archivos:** `src/eeg_library/modelado/reposo.py`, `tests/modelado/test_reposo.py`
(78 tests), `analisis/analisis_oa_oc.py` (script reproducible)
**Salida:** `C:\Users\migue\Desktop\TFG\EEG-Data\processed\oa_oc.json`, `resultados/logs/oa_oc.log`
**Figuras:** `resultados/figuras/oa_oc_confusion.png`, `resultados/figuras/oa_oc_alfa_posterior.png`,
`resultados/figuras/oa_oc_canales.png`
**Documento completo:** [`docs/analisis/ANALISIS_OA_OC.md`](analisis/ANALISIS_OA_OC.md)
**Fecha:** 2026-09-19

**Solo `train` + `val`.** El test no se abre (`preparar_reposo` se niega sin
`permitir_test=True`).

Control positivo del pipeline, no un clasificador útil: ¿conservan las features
de la Fase 13 el efecto Berger (más alfa posterior con los ojos cerrados)? Sobre
1072 grabaciones de reposo de 146 personas, validando por `persona_uid` en
personas no vistas, la regresión logística distingue OA de OC con **0.849** de
exactitud balanceada (IC95 [0.826, 0.873]; azar 0.500) y el Random Forest de las
Fases 14/14b con 0.804. El alfa posterior es la feature con más efecto
(d = +1.30), apunta hacia OC en los cinco pliegues, y O1 y O2 son los canales con
más efecto. Cumple 3 de los 4 criterios de coherencia declarados de antemano; el
(a) **no** se cumple (alfa concentra el 47.1 % de la importancia frente al 50 %
exigido).

Dos avisos para el resto del proyecto: `condicion_orden` es la propia etiqueta de
OA/OC (el mismo problema que ya apareció en la Fase 14), y cuatro features de la
Fase 13 son copias exactas de otra. Por ser la etiqueta la condición y no el VAS,
este análisis no consume el alfa del paquete de test pendiente.

**Detalle, diseño y lo que no hay que hacer: `docs/analisis/ANALISIS_OA_OC.md`.**

---

## Fase 18 — Reposo frente a tarea ejecutiva: ¿distingue el pipeline la carga cognitiva?

**Archivos:** `src/eeg_library/modelado/reposo_tarea.py`, `tests/modelado/test_reposo_tarea.py`
(73 tests), `analisis/analisis_reposo_tarea.py` (script reproducible)
**Salida:** `C:\Users\migue\Desktop\TFG\EEG-Data\processed\reposo_tarea.json`, `resultados/logs/reposo_tarea.log`
**Figuras:** `resultados/figuras/reposo_tarea_confusion.png`,
`resultados/figuras/reposo_tarea_por_condicion.png`, `resultados/figuras/reposo_tarea_canales.png`
**Documento completo:** [`docs/analisis/ANALISIS_REPOSO_TAREA.md`](analisis/ANALISIS_REPOSO_TAREA.md)
**Fecha:** 2026-09-21

**Solo `train` + `val`.** El test no se abre (`preparar_reposo_tarea` se niega sin
`permitir_test=True`).

Escalón siguiente al control OA/OC: ¿conservan las features de la Fase 13 una señal de
carga cognitiva? Sobre 3728 grabaciones (1072 de reposo y 2656 de tarea, desbalance
2.48 : 1) de 146 personas, validando por `persona_uid` en personas no vistas, la
regresión logística separa reposo de tarea con **0.857** de exactitud balanceada (IC95
[0.839, 0.874]; azar 0.500) y el Random Forest de las Fases 14/14b/17 con 0.831. Lo que
pesa es sobre todo el **descenso del alfa** (d hasta −1.6, en todos los canales); el
aumento de theta es más modesto y, en potencia absoluta, frontal. Sin OC dentro del
reposo baja a 0.816 (logística) pero sigue muy por encima del azar. Cumple los 4
criterios de coherencia declarados de antemano (el (a) es débil por construcción). OA
se confunde con tarea mucho más que OC (28 % frente a 5 % con la logística) y PVTB es
la única tarea mal reconocida (32 % leída como reposo).

Tres avisos para el resto del proyecto: **el orden fijo del protocolo confunde
condición y posición en la sesión y este diseño no puede separarlos**;
`condicion_orden` es la etiqueta (vale 1-2 en el reposo y 3-7 en las tareas); y **los
seis contadores de procesamiento, con solo ellos, dan 0.990 con el Random Forest**
(la duración de la grabación identifica casi a la condición), por lo que su exclusión
de X es imprescindible. Por ser la etiqueta la condición y no el VAS, este análisis
no consume el alfa del paquete de test pendiente.

**Detalle, diseño y lo que no hay que hacer: `docs/analisis/ANALISIS_REPOSO_TAREA.md`.**

---

## Fase 19 — Clasificación multiclase de las cinco tareas ejecutivas

**Archivos:** `src/eeg_library/modelado/tareas_ejecutivas.py`,
`tests/modelado/test_tareas_ejecutivas.py` (77 tests), `analisis/analisis_tareas_ejecutivas.py`
(script reproducible)
**Salida:** `C:\Users\migue\Desktop\TFG\EEG-Data\processed\tareas_ejecutivas.json`,
`resultados/logs/tareas_ejecutivas.log`
**Figuras:** `resultados/figuras/tareas_ejecutivas_confusion.png`,
`resultados/figuras/tareas_ejecutivas_pares.png`, `resultados/figuras/tareas_ejecutivas_canales.png`
**Documento completo:** [`docs/analisis/ANALISIS_TAREAS_EJECUTIVAS.md`](analisis/ANALISIS_TAREAS_EJECUTIVAS.md)
**Fecha:** 2026-09-22

**Solo `train` + `val`.** El test no se abre (`preparar_tareas_ejecutivas` se niega
sin `permitir_test=True`).

Escalón siguiente a la Fase 18: ¿distinguen las features de la Fase 13 **entre**
las cinco tareas ejecutivas (STROOP, PASAT, PVTB, MSIT, PVSAT), no solo si hay
carga cognitiva? Sobre 2656 grabaciones de 145 personas, casi perfectamente
equilibradas entre las cinco clases (azar = 0.200), validando por `persona_uid`
en personas no vistas, la regresión logística multinomial separa las cinco tareas
con **0.470** de exactitud balanceada (IC95 [0.452, 0.488]) y el Random Forest de
las Fases 14/14b/17/18 con 0.434. Se cumplen los 5 criterios de coherencia
declarados de antemano: los dos pares más confundidos (STROOP-MSIT, PASAT-PVSAT)
son justo los dos declarados como cognitivamente más parecidos, PVTB (vigilancia,
sin conflicto ni memoria de trabajo) es la mejor reconocida y la única con alfa
positivo en los 14 canales sin excepción, y theta frontal es mayor en las tareas
de máxima carga de memoria de trabajo (PASAT, PVSAT) que en PVTB, en los cinco
pliegues.

Dos avisos serios para el resto del proyecto. Primero, **los 6 contadores de
procesamiento, solos y sin ninguna feature espectral, igualan o superan el
modelo completo** (0.465–0.471 frente a 0.434–0.470): la duración de la
grabación predice el patrón de confusión mejor que la propia similitud
cognitiva (ρ = 0.76 frente a 0.47), así que una parte no acotada del resultado
podría ser duración y no neurofisiología. Segundo, `condicion_orden` es la
etiqueta (constante y distinta por tarea: 3–7) y **el orden fijo del protocolo
confunde tarea con momento de la grabación** sin que este diseño pueda
separarlos. Por ser la etiqueta la condición y no el VAS, este análisis no
consume el alfa del paquete de test pendiente.

**Detalle, diseño y lo que no hay que hacer: `docs/analisis/ANALISIS_TAREAS_EJECUTIVAS.md`.**

---

## Resultados sobre el conjunto de test (2026-10-07)

**Guion:** `analisis/analisis_test.py --confirmatorio` · **Pre-registro:**
[`PREREGISTRO_TEST.md`](../PREREGISTRO_TEST.md) (fijado el 2026-10-06)
**Salida:** `C:\Users\migue\Desktop\TFG\EEG-Data\processed\test_20261007-133358\`
(`resultados.json`, `ejecucion.log`, `predicciones_test.parquet`,
`inventario.csv`, `parametros.json`, modelos, bootstraps y permutaciones)

**Este es el único uso del test.** Todo lo que sigue estaba declarado en el
pre-registro y se reporta entero, salga lo que salga. Las cifras de validación
que se citan como referencia son las de `train` + `val` de las secciones
anteriores.

### Registro de la ejecución

| | |
|---|---|
| Inicio / fin | 2026-10-07 13:33:58 → 13:38:08 (250 s), una sola pasada |
| Commit | `bb55aee8d783` (el que fija el pre-registro), `git status` limpio |
| SHA-256 del guion | `20e25ce8aa3b…f3f0a4` |
| SHA-256 del pre-registro | `bdca1a7bca68…325121` |
| Semilla / bootstrap / permutaciones | 42 / 10 000 / 10 000 |
| Entrenamiento | `train` + `val` completos: 547 sesiones y 146 personas (VAS); 1072, 3728 y 2656 grabaciones (condición) |
| Entorno | Python 3.10.11, Windows 11, Ryzen 5 5600G (12 núcleos) |

Los SHA-256 completos de la matriz, `splits.csv`, las etiquetas y el VAS, el
`pip freeze` y el comando exacto están en `resultados.json`, bloque
`metadatos`.

### Datos de test

| | VAS | Condición |
|---|---|---|
| Personas | 28 (26 en la prueba principal) | 28 |
| Unidad | 106 sesiones | C1: 207, C2: 725, C3: 518 grabaciones |
| Descartes | 2 sesiones de personas con una sola sesión (Tarea B); 4 sesiones de personas con menos de 3 sesiones, sin recta de visita | 3 grabaciones sin `persona_uid` |

Sesiones por persona evaluada: 4 en 19 personas, 5 en 4, 3 en 2, 1 en 2 y 2 en
1. Con tan pocas sesiones por persona y 26 personas, **los intervalos son
anchos por construcción**; es la limitación que el cálculo de potencia ya
anticipaba.

### Familia VAS: ninguna prueba de fatiga se confirma

#### Prueba principal — Tarea B, tres niveles, RF hoja ≥ 5, con control de visita

| | Test | Validación (referencia) |
|---|---|---|
| Exactitud balanceada | **0.3513**, IC95 [0.2440, 0.4545] | 0.3795 [0.3302, 0.4290] |
| p unilateral (permutación) | **0.7707** | — |
| Azar | 0.3333 | 0.3333 |
| Sesiones / personas | 104 / 26 | 537 / 136 |
| **Veredicto** | **NO CONCLUYE** | — |

Con las reglas declaradas: «confirma» exigía p < 0.05 (no); «no confirma»
exigía p ≥ 0.05 **y** límite superior del IC < 0.3795, y el límite es 0.4545
(no). Queda el tercer caso. Es el desenlace que el cálculo de potencia daba por
más probable (81 %) y **no es evidencia contra el efecto**; tampoco es evidencia
a favor.

Lo que sí hay que decir con la misma claridad: el punto estimado queda **por
debajo de la referencia y a solo 1.8 puntos del azar**, y el IC contiene a las
dos. No hay indicio de réplica.

Matriz de confusión (filas = real; poca / media / mucha):

|  | poca | media | mucha |
|---|---|---|---|
| **poca** | 13 | 12 | 12 |
| **media** | 13 | 16 | 7 |
| **mucha** | 14 | 9 | 8 |

Descriptivo: F1 macro 0.351, exactitud adyacente 0.750, tau de Kendall −0.079
(p = 0.37; en validación +0.092). Exhaustividad por clase 0.351 / 0.444 / 0.258:
**no se reproduce el patrón de validación** (extremos mejor que el centro), y
«mucha fatiga» es la peor clase.

| Descriptivo de la principal | Exactitud balanceada [IC95] |
|---|---|
| Sin control de visita (p = 0.78) | 0.2808 [0.1853, 0.3759] |
| RF hoja ≥ 1 | 0.3758 [0.2750, 0.4713] |
| RF hoja ≥ 15 | 0.3618 [0.2844, 0.4399] |
| RF hoja ≥ 5, balanceado | 0.3591 [0.2568, 0.4604] |
| Regresión RF discretizada con los mismos cortes | 0.3326 [0.2894, 0.3773] |

- **Sin el control de visita el resultado cae por debajo del azar** (0.2808).
  En validación era 0.3947. Con y sin control, el test no apoya el efecto.
- **La regresión discretizada reproduce su comportamiento** (0.3326 frente a
  0.3331 en validación): se refugia en la clase central (8 / 93 / 3 sesiones
  predichas por clase). Pero **la ventaja de clasificar sobre regresar no se
  confirma**: +0.019, IC95 [−0.077, +0.114]. La afirmación de la Fase 14b («clasificar
  aporta») queda sin apoyo en test.
- Las cuatro configuraciones de RF caen en 0.359–0.376, todas con IC que
  contiene el azar.

#### Secundarias (Holm, 5 pruebas)

| # | Prueba | Resultado en test | p | p Holm | Veredicto |
|---|---|---|---|---|---|
| S1 | `_rel_sd` intra-persona con control de visita, ρ > 0 | ρ = −0.098 [−0.355, +0.167] (sin control: +0.124) | 0.766 | 1.000 | no significativa |
| S2 | θ/α intra-persona con control, R² < 0.05 | R² = 0.118 [0.009, 0.295]; β = +5.20 mm, IC boot [+1.81, +8.74] | 0.840 | 1.000 | no significativa |
| S3 | Persona nueva: el EEG no mejora a la media (ΔR² < 0.05 y mejora MAE < 2) | ΔR² = −0.093 [−0.191, −0.011]; mejora MAE = −0.52 [−1.84, +0.72] | 0.0005 | **0.0025** | **significativa** |
| S4 | Persona conocida: el EEG no aporta sobre la media de la persona | ΔR² = −0.053 [−0.159, +0.022]; mejora MAE = −0.52 [−1.42, +0.38] | 0.0027 | **0.0108** | **significativa** |
| S5 | Persona nueva: el RF ordena, ρ > 0 | ρ = −0.010 [−0.204, +0.178] | 0.551 | 1.000 | no significativa |

Descriptivo de la familia: persona nueva en tres niveles, 0.3140 [0.2346, 0.4005]
(validación 0.3103).

**Cómo leer S3 y S4.** Son afirmaciones **nulas**: «significativa» significa que
se confirma que el EEG **no** mejora a la media por encima del margen de
equivalencia (ΔR² < 0.05 y mejora del MAE < 2). No es un hallazgo positivo. Y
cumplen el margen «con holgura» en el sentido incómodo: el RF **empeora** a la
media de entrenamiento (ΔR² negativo) en vez de igualarla. En S4 el IC de ΔR²
incluye valores ligeramente positivos (hasta +0.022), por debajo del margen.

**S1 no se replica.** El compuesto `_rel_sd`, la única pista positiva de todo el
proyecto, pasa de +0.130 (validación, con control) a −0.098 en test. El IC
([−0.355, +0.167]) es ancho, así que contiene la referencia de validación y no
se puede **descartar** un efecto de ese tamaño con 26 personas; pero la
hipótesis no obtiene ningún apoyo independiente. Sin control de visita el test
da +0.124, parecido a lo visto en validación (+0.156); en test el control
elimina por completo la asociación.

**S2 es la única cifra que apunta hacia el lado «positivo» y requiere cautela.**
Dentro de persona, θ/α sigue al VAS con β = +5.2 mm y R² = 0.118, con IC
bootstrap de β que excluye el 0 (y que excluye también la pendiente útil de
+10 mm). Contradice la validación (β = −0.455 mm, R² = 0.001). Pero: (i) es una
de cinco secundarias y **p Holm = 1.0**; (ii) la hipótesis pre-registrada (R²
por debajo del margen) **no se rechaza ni se confirma** — el IC del R² abarca de
0.009 a 0.295; (iii) son 26 personas y 104 sesiones; (iv) no hay réplica
independiente. Debe citarse como **resultado no concluyente discordante con la
validación**, no como evidencia a favor del índice ni como un fallo de la
validación. No debe usarse para volver a interpretar θ/α.

### Familia de condición: las tres pruebas se confirman

Regresión logística, `train` + `val` → test (p por permutación dentro de
persona, 10 000 permutaciones; el valor 0.0001 es el mínimo alcanzable,
`1/10 001`):

| # | Prueba | Test | Validación | p | p Holm | Veredicto |
|---|---|---|---|---|---|---|
| C1 | OA frente a OC | **0.8553** [0.7937, 0.9118] | 0.849 | 0.0001 | 0.0003 | **CONFIRMA** |
| C2 | Reposo frente a tarea | **0.8508** [0.8126, 0.8851] | 0.857 | 0.0001 | 0.0003 | **CONFIRMA** |
| C3 | Cinco tareas ejecutivas | **0.4884** [0.4408, 0.5314] | 0.470 | 0.0001 | 0.0003 | **CONFIRMA** |

Las tres cifras reproducen las de validación dentro de su intervalo. El control
sin información (clase mayoritaria o azar estratificado) da 0.50 / 0.50 / 0.20
en las tres.

Random Forest (secundario, descriptivo): C1 **0.8410** [0.7714, 0.9029] (AUC
0.947), C2 **0.8469** [0.8082, 0.8834] (AUC 0.923), C3 **0.4560** [0.4160, 0.4937].
Diferencia pareada logística − RF: C1 +0.014 [−0.028, +0.061], C2 +0.004 [−0.027,
+0.035], C3 +0.032 [−0.014, +0.073]; **en test ninguna es distinguible de 0**
(en validación la logística superaba al RF en las tres).

Referencia de un solo umbral: C1 0.7738 (`posterior_alfa_rel_media`), C2 0.5944
(`frontal_theta_rel_media`). El modelo multivariante la supera en C1 por
+0.081 [+0.037, +0.131] (logística) y en C2 por +0.256 [+0.189, +0.324].

#### Criterios de coherencia (SÍ/NO, sin p)

| Prueba | Criterio | Test | Modelo final |
|---|---|---|---|
| **C1** | (a) alfa concentra ≥ 50 % de la importancia | **NO** (12.7 %) | — |
| | (b) alfa posterior entre las 5 más importantes | SÍ | — |
| | (c) alfa es la banda que más cuesta al permutarla | SÍ (+0.178 frente a +0.006 de theta) | — |
| | (d) dirección hacia OC | SÍ (d = +1.63 [+1.32, +2.05]) | sí |
| **C2** | (a) la banda que más cuesta es theta o alfa | SÍ (alfa, +0.248) | — |
| | (b) theta frontal hacia tarea | SÍ (d = +0.40 [+0.09, +0.72]) | sí |
| | (c) alfa posterior hacia reposo | SÍ (d = −1.67 [−1.88, −1.48]) | sí |
| | (d) (b) y (c) se mantienen sin OC | **NO** | sí |
| **C3** | (a) los dos modelos superan el azar | SÍ | — |
| | (b) los pares gemelos entre los 3 más confundidos | SÍ | — |
| | (c) PVTB, mayor exhaustividad (0.641) | SÍ | — |
| | (d) theta frontal PASAT y PVSAT > PVTB | SÍ | sí |
| | (e) confusión sigue al parecido cognitivo | SÍ (ρ = +0.26 frente a −0.08) | — |

- **C1 repite lo de validación:** 3 de 4, con el mismo criterio (a) en NO. En
  test el alfa concentra además una fracción mucho menor (12.7 % frente a
  47.1 %), pero la permutación conjunta (c) sigue siendo clara.
- **C2 cumple 3 de 4 en test (en validación 4 de 4).** El (d) falla por un
  margen estrecho: con OA como único reposo, el alfa posterior baja con claridad
  (d = −1.09 [−1.43, −0.81]) pero el **IC del theta frontal incluye el 0**
  (d = +0.21 [−0.01, +0.40]). El modelo final sí cumple el criterio. Se cita
  como NO.
- **C3 cumple 5 de 5**, pero (b) es muy ajustado: PASAT-PVSAT 0.164 frente a
  PVTB-PVSAT 0.163 en el tercer puesto, y STROOP-MSIT 0.260 destaca como el par
  más confundido.

#### Los dos confusores se replican

**Contadores de procesamiento, solos** (exactitud balanceada, logística / RF):

| | Test | Validación |
|---|---|---|
| C1 | 0.508 / 0.542 | 0.576 / 0.530 |
| C2 | 0.593 / **0.993** | 0.624 / **0.990** |
| C3 | 0.465 / 0.481 | 0.465 / 0.471 |

El atajo de la duración está **tan presente en test como en validación**: en C2
seis números sin señal cerebral dan 0.993 con el RF, y en C3 igualan al modelo
espectral (0.465–0.481 frente a 0.456–0.488). La exclusión de los contadores
de X sigue siendo imprescindible, y estos resultados se citan siempre con esa
salvedad.

**Estructura de la confusión entre tareas (C3, 10 pares):**

| Confusión frente a… | ρ test | ρ validación |
|---|---|---|
| similitud cognitiva | +0.26 | +0.47 |
| −distancia en el protocolo | −0.08 | −0.27 |
| −distancia de duración | **+0.86** | +0.76 |

En test la **duración explica la confusión aún mejor** (+0.86) y el parecido
cognitivo peor (+0.26). El criterio (e) se cumple formalmente, pero la evidencia
de que el patrón sea cognitivo y no de duración es **más débil que en
validación**. Con n = 10 pares, las tres cifras son descriptivas.

**Orden fijo del protocolo.** Persiste sin solución. Entre las tareas, la
posición apenas se relaciona con el acierto (ρ = +0.037 [−0.008, +0.083]).

#### Otras réplicas (descriptivo)

- **OA se reconoce peor que OC como reposo** (0.748 frente a 0.933; OA − OC =
  −0.185 [−0.315, −0.066], OA > OC en el 0.06 % de las réplicas): igual que en
  validación.
- **PVTB es la tarea peor reconocida frente a reposo** (acierto 0.621, es decir,
  37.9 % de PVTB leído como reposo, frente a ≥ 88 % en las otras cuatro).
- **Sin OC dentro del reposo**: 0.7967 [0.7509, 0.8384] (validación 0.816),
  muy por encima del azar.

### Conclusión

1. **El pipeline queda validado en un conjunto independiente.** Las tres
   pruebas de condición se confirman con p Holm = 0.0003 y reproducen las cifras
   de validación. La señal espectral de OA/OC, de carga cognitiva y de tipo de
   tarea sobrevive a 28 personas que el modelo nunca vio.
2. **No hay evidencia de que el EEG, con estas features, prediga la fatiga
   percibida.** La prueba principal no concluye (0.3513, p = 0.77) y su punto
   estimado no replica la validación; para persona nueva y para persona conocida
   se **confirma**, con el margen declarado, que el EEG no mejora a la media
   (S3, S4). Ni `_rel_sd` (S1) ni el orden de las predicciones (S5) se replican.
3. **El único resultado discordante es S2** (θ/α intra-persona con β positiva).
   No es significativo tras Holm y contradice la validación; se reporta tal cual.
4. **Los confusores de duración y de orden del protocolo se replican.** Los
   resultados de condición no deben citarse sin esa salvedad.

### Alcance y limitaciones

- **Potencia baja y declarada.** La probabilidad de confirmar la prueba
  principal si el efecto real era el de validación era 0.33. Un «no concluye»
  **no** equivale a «no existe». Los IC de 26 personas son anchos y contienen
  tanto el azar como la referencia.
- **El efecto de validación no se reproduce, y este análisis no dice por qué.**
  La prueba principal era la configuración declarada de la Fase 14b (no la mejor
  de varias; las otras tres caían en el mismo rango), pero la exploración
  previa sobre `train` + `val` (recuento en `PREREGISTRO_TEST.md`, sección 9)
  pudo dejar un efecto sobreestimado. Es una hipótesis, no algo demostrado: con
  26 personas tampoco se puede descartar que el efecto sea real y pequeño.
- **El test no permite separar** entre «no hay señal de fatiga en las features
  agregadas» y «no hay señal en el EEG». La ruta CNN, que mide otra cosa, no se
  ha ejecutado.
- **Las pruebas S3 y S4 son equivalencias unilaterales.** Que el ΔR² sea
  negativo no es mérito de ningún modelo: significa que el RF empeora a la
  media.
- **Test gastado para estas preguntas.** Cualquier análisis adicional del VAS
  o de la condición sobre estas 28 personas, o cualquier ajuste posterior
  motivado por estos resultados, es exploratorio y debe citarse así. Para una
  pregunta nueva, ver la regla de la sección «Pendientes».

### Lo que **no** hay que hacer

- **Presentar la prueba principal como confirmada, o como refutada.** Es «no
  concluye», con ese texto.
- **Citar el 0.3947 de la Fase 14b como resultado del proyecto sobre el VAS.**
  La cifra confirmatoria es 0.3513 (controlada) y 0.2808 (sin control); la de
  validación es una exploración.
- **Citar S3 y S4 como que «el modelo funciona».** Son resultados nulos.
- **Reinterpretar θ/α a partir de S2.** Una secundaria sin corrección
  significativa, con 26 personas, no justifica un giro.
- **Volver a ejecutar el test, o a elegir modelo, umbral o features mirando sus
  resultados.**
- **Citar los resultados de condición sin los confusores** (contadores de
  procesamiento y orden fijo del protocolo).
- **Leer el p = 0.0001 de la familia de condición como un valor exacto.** Es el
  mínimo alcanzable con 10 000 permutaciones.

---

## Fase 20 — Normalización intra-sujeto del VAS (cerrada)

**Carpeta:** [`analisis/normalizacion_vas/`](../analisis/normalizacion_vas/README.md)
(una subcarpeta por paso, cada una con script, tablas, figuras e `INFORME.md`)
**Inicio:** 2026-10-21, a propuesta del tutor. **Solo `train` + `val`.**

¿Da una etiqueta más útil expresar el VAS relativo a cada persona (centrado o
tipificado dentro de `persona_uid`)?

- **Paso 1 — descriptivo del VAS bruto** (solo VAS, sin EEG ni modelos;
  [`01_descriptivo/INFORME.md`](../analisis/normalizacion_vas/01_descriptivo/INFORME.md)):
  146 personas y 786 respuestas M1–M6; 139 con ≥3 sesiones y ninguna con
  SD = 0. ICC(1) = 0.40 [0.32, 0.48] (0.47 sobre las sesiones con EEG). Los
  alumnos difieren sobre todo en nivel y solo modestamente en rango de uso de
  la escala. Dentro de persona, el final de trimestre queda ≈ +4–6 mm por
  encima del inicio.
- **Paso 2 — variantes de la etiqueta** (solo VAS, criterio fijado antes de
  calcular; [`02_etiquetas/INFORME.md`](../analisis/normalizacion_vas/02_etiquetas/INFORME.md)):
  ninguna variante coincide con la etiqueta de la Fase 14b en ≥ 90 % de los
  tercios (80.8 % / 80.3 % / 78.0 %), así que hay paso 3 con la z de SD
  encogida como principal. Pero la tipificación apenas cambia nada (96.7 %
  frente al centrado sobre la misma referencia). Lo que cambia la etiqueta es
  estimar el nivel habitual con todas las respuestas del VAS (incluida M1,
  corregida en −7.8 mm), que reduce un 6 % el error de la referencia.
- **Paso 3 — Tarea B con cada etiqueta** (exploratorio, reglas fijadas antes;
  [`03_eeg/INFORME.md`](../analisis/normalizacion_vas/03_eeg/INFORME.md)):
  mismo banco que la Fase 14b, que se reproduce exacto (0.3947 / 0.3795). La
  variante principal da **0.351 [0.313, 0.390]**, con Δ frente a la Fase 14b
  de −0.044 [−0.091, +0.004]: **normalizar no mejora la respuesta del EEG**.
  Sin control de visita, cada etiqueta "más limpia" baja el resultado, hasta
  0.321 al quitar el efecto de momento. Con control de visita todas quedan en
  0.377–0.398. El 0.3947 está además en lo alto del rango de semillas del RF
  (0.359–0.395, media 0.383).

**Conclusión de la línea (cerrada el 2026-10-21):** la señal de la Tarea B es
frágil (0.32–0.40 según cómo se construya la misma etiqueta) y la
normalización no la hace más robusta. Refuerza la conclusión de las Fases
14/14b y del test.

---

## Fase 21 — Fatiga general y cognitiva (PedsQL) frente al EEG (cerrada)

**Carpeta:** [`analisis/fatiga_fg_fc/`](../analisis/fatiga_fg_fc/README.md)
(una subcarpeta por paso, cada una con script, tablas, figuras e `INFORME.md`
con el plan escrito antes de ejecutar). **Inicio:** 2026-10-22, a petición de
los tutores. **Solo `train` + `val`.**

¿Guardan relación con el EEG la fatiga general (FG) y la fatiga cognitiva (FC)
de la PedsQL Multidimensional Fatigue Scale (6 ítems cada una, suma 0–24),
respondidas el mismo día que el EEG?

- **Paso 1 — descriptivo**, solo cuestionario
  ([`01_descriptivo/INFORME.md`](../analisis/fatiga_fg_fc/01_descriptivo/INFORME.md)).
  - **La base:** la primera versión de este paso detectó que en las bases
    anteriores los ítems PedsQL de M5–M6 estaban codificados como
    `max(respuesta − 1, 0)`. El equipo lo corrigió en
    `BD_COMPLETA_corregida.xlsx` y la corrección se verificó celda a celda.
  - **Las medidas:** α 0.79–0.90, un efecto de calendario pequeño (M1 más
    baja) e ICC 0.60–0.72 (frente a 0.40–0.49 del VAS).
  - **Puerta del paso 3:** la fiabilidad intra-persona es 0.67 (FG) y 0.76
    (FC), así que hay paso 3 para las dos.
  - **Relación con el VAS dentro de persona:** FG lo acompaña (ρ = 0.40); FC
    apenas (0.14).
  - **Universo:** 531 sesiones de 146 personas.
- **Paso 2 — entre personas**
  ([`02_entre_personas/INFORME.md`](../analisis/fatiga_fg_fc/02_entre_personas/INFORME.md)).
  - **Diseño:** una fila por persona, con y centrado por medición y ajustado
    por sexo, curso y centro dentro de cada pliegue, y Random Forest.
  - **FG:** ρ = +0.146 [−0.029, +0.311], p = 0.081.
  - **FC:** ρ = +0.008.
  - Sin el ajuste, FG sube a +0.23, que en parte no es fatiga.
- **Paso 3 — intra persona**, la Tarea B con FG y FC
  ([`03_intra_persona/INFORME.md`](../analisis/fatiga_fg_fc/03_intra_persona/INFORME.md)).
  - **FG:** ρ = **+0.170 [+0.085, +0.251], p = 0.004, p de Holm = 0.016**. Lo
    reproducen ridge (+0.131) y cinco semillas, y supera al VAS por el mismo
    camino (diferencia +0.178 [+0.066, +0.287]).
  - **Con control de visita**, FG baja a **+0.067 [−0.037, +0.175]**.
  - **FC:** +0.059, p de Holm = 0.36.
  - **Hipótesis a priori (θ/α y `_rel_sd`):** ninguna con Holm.

**Conclusión de la línea (cerrada el 2026-10-22):** entre personas no hay
relación.

Dentro de persona, FG es la única medida de fatiga del proyecto que acompaña al
EEG de forma estadísticamente robusta (p de Holm = 0.016). Pero la relación
desaparece al controlar la tendencia de cada alumno a lo largo del curso.
Según la regla fijada antes de ejecutar, se reporta como dependiente de la
deriva a lo largo del curso, no como fatiga de la sesión. Con 2–5 sesiones por
persona no se puede separar una fatiga que se acumula de una deriva del
registro. Es pequeña (R² = +0.013) y exploratoria.

FC no tiene relación con el EEG.
