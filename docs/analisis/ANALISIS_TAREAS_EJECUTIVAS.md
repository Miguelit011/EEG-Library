# Clasificación multiclase de las cinco tareas ejecutivas

> **Qué es este documento.** Clasificación multiclase (5 clases) de las
> grabaciones de tarea ejecutiva —STROOP, PASAT, PVTB, MSIT, PVSAT— a partir de
> las features de la Fase 13. OA y OC quedan fuera; ya están cubiertas en
> `docs/analisis/ANALISIS_OA_OC.md` y `docs/analisis/ANALISIS_REPOSO_TAREA.md`. Es el escalón siguiente a
> ambas: la Fase 17 preguntaba si el pipeline conserva el efecto Berger
> (sí/no) y la Fase 18 si conserva una señal de carga cognitiva en general
> (sí/no); esta pregunta es más fina — si el pipeline distingue **entre**
> tareas ejecutivas, y **qué tareas se confunden entre sí**. No busca un
> clasificador útil: busca si el patrón de confusiones sigue el parecido
> cognitivo de las tareas (señal neurofisiológica) o la posición fija de cada
> tarea en el protocolo (artefacto de diseño).
>
> **Resultado en una línea:** sí lo distingue, muy por encima del azar
> (regresión logística: **0.470** de exactitud balanceada, IC95 [0.452, 0.488];
> azar 0.200), y el patrón de confusión sigue el parecido cognitivo declarado
> antes de ejecutar (ρ = +0.47 frente a −0.27 con la distancia en el
> protocolo): los pares más confundidos son STROOP-MSIT (ambas de
> interferencia) y PASAT-PVSAT (misma tarea, distinta modalidad), y PVTB, la
> más distinta cognitivamente, es la mejor reconocida. Se cumplen los **5 de 5**
> criterios de coherencia declarados de antemano. **Pero los 6 contadores de
> procesamiento, solos y sin una sola feature espectral, igualan o superan
> ese rendimiento** (logística 0.465, Random Forest 0.471): una parte que no
> se puede acotar de lo anterior podría ser duración de la grabación, no
> neurofisiología (sección 8.2). Y **el orden del protocolo es fijo, así que
> la tarea coincide con el momento de la grabación**; este diseño no puede
> separar del todo parecido cognitivo de proximidad temporal (sección 12).
>
> **Guion reproducible:** `analisis/analisis_tareas_ejecutivas.py`
> **Código de apoyo:** `src/eeg_library/modelado/tareas_ejecutivas.py`
> (`tests/test_tareas_ejecutivas.py`, 77 pruebas)
> **Salida numérica:** `<EEG_DATA>/processed/tareas_ejecutivas.json`
> **Registro de ejecución:** `resultados/logs/tareas_ejecutivas.log`
> **Figuras:** `resultados/figuras/tareas_ejecutivas_confusion.png`,
> `resultados/figuras/tareas_ejecutivas_pares.png`, `resultados/figuras/tareas_ejecutivas_canales.png`
> **Resumen en el cuaderno de modelado:** `docs/MODELADO.md`, Fase 19
>
> Fecha de ejecución: 2026-09-22. **Solo `train` + `val`; el conjunto de test
> no se ha abierto.**

---

## 1. La pregunta, y por qué es distinta de las Fases 17 y 18

La Fase 17 (`docs/analisis/ANALISIS_OA_OC.md`) mostró que el pipeline conserva el efecto
Berger (0.849 de exactitud balanceada entre OA y OC). La Fase 18
(`docs/analisis/ANALISIS_REPOSO_TAREA.md`) mostró que conserva una señal de carga cognitiva
en general: reposo frente a cualquiera de las cinco tareas ejecutivas se
separa con 0.857. Ninguna de las dos dice si esa señal distingue **entre**
tareas, o si toda la carga cognitiva se ve igual sea cual sea la tarea.

Este análisis responde a esa pregunta con las cinco tareas ejecutivas del
protocolo como clases: STROOP, PASAT, PVTB, MSIT y PVSAT. No es un
clasificador útil —nadie necesita adivinar qué tarea hacía alguien—: lo
informativo es **qué pares de tareas se confunden entre sí** y **por qué**.
Hay tres estructuras candidatas que predicen confusiones distintas:

1. **Parecido cognitivo.** Si el modelo confunde tareas que exigen procesos
   similares (dos tareas de interferencia, o la misma tarea en dos
   modalidades), la confusión es señal neurofisiológica genuina.
2. **Distancia en el protocolo.** El orden es fijo dentro de cada sesión
   (`docs/ESTADO.md`, «Orden del protocolo»): STROOP → PASAT → PVTB → MSIT → PVSAT.
   Si el modelo confunde tareas que están cerca en el tiempo, puede ser una
   deriva de sesión (fatiga, acomodación, calidad de contacto) y no una
   propiedad de la tarea.
3. **Duración de la grabación.** STROOP y MSIT duran ~60 s; PASAT, PVTB y
   PVSAT ~150 s (`docs/ESTADO.md`, «Duraciones»). Si el modelo confunde tareas de
   duración parecida, puede estar usando un artefacto de la grabación
   (`n_ventanas`, fuera de X, pero potencialmente correlacionado con features
   que sí entran).

El diseño tiene una disociación aprovechable: los dos pares cognitivamente más
parecidos —PASAT-PVSAT (la misma tarea de suma serial, auditiva frente a
visual) y STROOP-MSIT (las dos de interferencia)— están ambos a distancia 3 en
el protocolo, mientras que los pares contiguos en el tiempo (PASAT-PVTB,
PVTB-MSIT) son cognitivamente distintos. Similitud cognitiva y proximidad
temporal apuntan en direcciones opuestas, así que el patrón de confusión puede
en principio discriminar entre ellas (sección 7). La duración, en cambio, **no
se disocia bien de nada**: STROOP y MSIT comparten duración y son también el
par cognitivamente más parecido; PASAT y PVSAT comparten duración y son el
otro. Por eso el resultado de la sección 8.2 —los contadores de duración solos
igualan al modelo espectral— es la limitación más seria de todo el análisis
(sección 12), no un detalle técnico.

## 2. Datos

| | |
|---|---|
| Grabaciones en la matriz de la Fase 13 (las 7 condiciones) | 4543 |
| Apartadas (test), de las 5 tareas ejecutivas | 523 (28 personas) — **no se leen** |
| Descartadas por no tener `persona_uid` (train + val) | 60 |
| **En el análisis** | **2656 grabaciones** |
| Por condición | STROOP 534, PASAT 533, PVTB 531, MSIT 530, PVSAT 528 |
| Personas / sesiones | 145 / 537 |
| Por split | 2219 train + 437 val |
| Grabaciones por persona | mediana 20, rango [5, 25] |
| **Desbalance de clases** | **1.011 : 1** (mayoritaria STROOP, minoritaria PVSAT) |

**La unidad de análisis es la grabación**, igual que en las Fases 17 y 18: la
condición es una propiedad de cada registro, con su propio EEG. Lo que no es
independiente es la persona (cada una aporta una mediana de 20 grabaciones), y
eso se controla agrupando por `persona_uid` — nunca `sujeto_id`, porque seis de
ellos contienen dos personas distintas — tanto al validar como al calcular
intervalos.

A diferencia de reposo frente a tarea (2.48 : 1), las cinco tareas ejecutivas
están **casi perfectamente equilibradas entre sí** (528 a 534 grabaciones, un
rango del 1.1 %): el reparto de condiciones en el protocolo es fijo por
sesión, así que el pequeño desbalance observado viene solo de sesiones
incompletas. La exactitud balanceada de azar es, por tanto, **1/5 = 0.200
exacto**. El reparto de clases por pliegue de la CV agrupada por persona
también queda casi equilibrado (104–108 grabaciones de cada clase en cada uno
de los 5 pliegues), así que `GroupKFold` sin estratificar no introduce sesgo
apreciable aquí.

## 3. Qué entra a X, y qué no

`columnas_de_features()` sigue siendo la única definición de X del proyecto y
no se ha tocado. `columnas_de_features_ejecutivas()` es exactamente la de
OA/OC y reposo/tarea (`columnas_de_features_reposo`, y un test comprueba que
coinciden): parte de las 77 features de la Fase 13 y resta 7, de modo que
quedan **70**.

### `condicion_orden` es la etiqueta

Codifica la posición de la condición en el protocolo. Dentro de las cinco
tareas ejecutivas toma un valor **constante y distinto en cada una**: STROOP=3,
PASAT=4, PVTB=5, MSIT=6, PVSAT=7. No hace falta ni un corte: es literalmente
la etiqueta con otro nombre, uno a uno con la clase — una fuga todavía más
directa que en reposo/tarea (donde hacía falta un solo corte entre 2 y 3) y
una exclusión obligatoria, no una decisión discutible. Hay un test que prueba
que, metida a mano, da exactitud balanceada de 1.0.

### Los seis contadores de procesamiento

`n_ventanas`, `n_canales_buenos` y los cuatro `n_validos_*` no son señal
cerebral. En la Fase 18, solos, daban 0.990 de exactitud balanceada en el
contraste binario reposo/tarea. Aquí dan **0.465–0.471**, prácticamente lo
mismo que las 70 features espectrales juntas (0.470/0.434): la sección 8.2
desarrolla por qué esto es la limitación más seria del análisis.

### Observación sobre la matriz: cuatro features son copias exactas

Igual que en OA/OC y reposo/tarea: `theta_frontal_log_media` =
`frontal_theta_logabs_media`, `theta_frontal_log_sd` =
`frontal_theta_logabs_sd`, `alfa_posterior_log_media` =
`posterior_alfa_logabs_media`, `alfa_posterior_log_sd` =
`posterior_alfa_logabs_sd`. Quedan 66 columnas distintas de las 70. No se
quitan de X (se reutiliza la matriz tal cual); la sensibilidad (sección 8.3)
comprueba que no cambian nada.

## 4. Modelos, y por qué estos dos

El caso sigue siendo tabular: 70 features numéricas continuas y muy
correlacionadas, y ~2656 filas que en realidad son 145 personas. Se pedía un
máximo de dos modelos y la elección es la misma que en las Fases 17 y 18, por
las mismas razones, extendida a multiclase:

- **Regresión logística multinomial** (`LogisticRegression`, softmax
  automático con más de dos clases desde scikit-learn ≥ 1.5, sin pasar
  `multi_class`, que está deprecado). Es el único modelo cuyos coeficientes
  —uno por clase, en unidades de desviación típica tras `StandardScaler`—
  permiten comprobar la **dirección** del efecto de cada tarea, que es lo que
  ponen a prueba los criterios (c) y (d).
- **Random Forest** (500 árboles, hoja ≥ 5, `class_weight="balanced"`,
  `random_state=42`): la configuración de las Fases 14, 14b, 17 y 18, sin
  ajustar hiperparámetros, para que la cifra sea comparable con todo lo
  anterior del proyecto y cubra las interacciones y no linealidades que la
  logística no puede.

Se descartaron, por las mismas razones que en `docs/analisis/ANALISIS_REPOSO_TAREA.md`: el
gradient boosting (misma familia que el bosque, más hiperparámetros y sin la
ventaja de la comparabilidad), una SVM con núcleo (sensible a la escala y sin
coeficientes interpretables), el análisis discriminante lineal (redundante con
la logística) y una red neuronal (con tan pocas personas independientes,
tendría mucha más capacidad que información y sus pesos no darían un signo
interpretable). Una sola configuración por modelo: elegir la mejor de varias
sobre estos datos es el sesgo que el arnés de `modeling.py` existe para
evitar.

**El desbalance se trata igual que en las Fases 17/18 aunque aquí sea
pequeño** (1.011 : 1, no 2.48 : 1): `class_weight="balanced"` en el
entrenamiento, por consistencia y porque el reparto dentro de cada pliegue de
la CV puede no ser exactamente igual. La métrica principal es la **exactitud
balanceada** (azar = 1/5 = 0.200 exacto). De apoyo: **F1 macro** y el
**coeficiente kappa de Cohen** (que sí depende del reparto de clases
observado, a diferencia de la exactitud balanceada, y sirve de comprobación
cruzada).

Referencias que no usan X: `DummyClassifier` de clase mayoritaria, azar
estratificado y uniforme — con 5 clases casi equilibradas, los tres deberían
dar ~0.200 de exactitud balanceada (comprobado en la sección 2). No se incluye
una referencia de una sola variable (como el umbral univariante de OA/OC y
reposo/tarea) porque no hay un análogo natural a un solo corte binario con 5
clases sin imponer una jerarquía arbitraria entre ellas; el papel de
referencia mínima lo cumplen aquí los tres baselines.

**Validación.** `predecir_cv_agrupada` de la Fase 14, sin modificarla: 5
pliegues de `GroupKFold` por `persona_uid`, con la comprobación de que ninguna
persona cae a ambos lados. Imputación y tipificación se ajustan **solo con el
pliegue de entrenamiento** (van dentro del `Pipeline`). **Intervalos:**
bootstrap remuestreando **personas** (2000 réplicas, semilla 42); las
comparaciones entre modelos son **pareadas**.

### Qué se decidió antes de ver datos y qué después

Los dos modelos, los criterios de coherencia (a)-(e), las métricas, la
comparación de estructuras de la sección 7.3, los controles de la sección 8 y
las filas de la sensibilidad estaban en la primera versión del guion, antes de
ninguna ejecución. Antes de la ejecución completa se hizo **una pasada en modo
rápido** (`--rapido`: parámetros mínimos, salidas fuera del repositorio) para
comprobar que el guion corría de principio a fin; sus cifras no se usan como
resultado, aunque —al no reducir el modo rápido el propio conjunto de datos ni
la logística— sus puntos estimados de la regresión logística coincidieron
exactamente con los de la ejecución completa. No se cambió nada del guion
entre esa pasada y la ejecución que produjo estos resultados.

## 5. Resultado — rendimiento

| Clasificador | BA media/pliegue | IC95 (bootstrap por persona) | BA amontonada | Exactitud simple | F1 macro | Kappa |
|---|---|---|---|---|---|---|
| Baseline: clase mayoritaria | 0.2000 | [0.198, 0.200] | 0.1993 | 0.2003 | 0.1037 | −0.0008 |
| Baseline: azar estratificado | 0.1927 | [0.180, 0.206] | 0.1927 | 0.1928 | 0.1917 | −0.0091 |
| Baseline: uniforme | 0.2100 | [0.196, 0.224] | 0.2101 | 0.2101 | 0.2092 | 0.0126 |
| **Regresión logística** | **0.4701** | **[0.452, 0.488]** | 0.4699 | 0.4699 | 0.4660 | 0.3374 |
| Random Forest | 0.4345 | [0.415, 0.454] | 0.4342 | 0.4341 | 0.4288 | 0.2927 |

Los tres baselines dan ~0.200, como debe ser con la exactitud balanceada
(sección 2 del guion). Los dos modelos superan claramente el azar: **la
regresión logística separa las cinco tareas ejecutivas con 0.470 de exactitud
balanceada, 2.35 veces el azar**, y el Random Forest con 0.434.

Por clase (regresión logística):

| Tarea | Precisión | Exhaustividad | F1 | n |
|---|---|---|---|---|
| STROOP | 0.457 | 0.472 | 0.465 | 534 |
| PASAT | 0.405 | 0.336 | 0.367 | 533 |
| **PVTB** | 0.549 | **0.638** | 0.590 | 531 |
| MSIT | 0.484 | 0.492 | 0.488 | 530 |
| PVSAT | 0.429 | 0.411 | 0.420 | 528 |

Por clase (Random Forest): STROOP 0.439/0.393/0.415, PASAT 0.347/0.313/0.329,
**PVTB 0.499/0.659/0.568**, MSIT 0.477/0.408/0.439, PVSAT 0.387/0.398/0.392
(precisión/exhaustividad/F1). **PVTB es, con los dos modelos, la tarea mejor
reconocida** (sección 6 y 7 desarrollan por qué).

### Comparaciones pareadas (exactitud balanceada)

| Comparación | Diferencia | IC95 | A > B en |
|---|---|---|---|
| Regresión logística − Random Forest | **+0.0357** | [+0.018, +0.054] | 100.0 % |
| Regresión logística − Baseline azar | +0.2772 | [+0.256, +0.299] | 100.0 % |
| Random Forest − Baseline azar | +0.2415 | [+0.218, +0.265] | 100.0 % |

### Lectura

La regresión logística supera al Random Forest en el 100 % de las réplicas
del bootstrap pareado: **la señal es, otra vez, esencialmente lineal en
features tipificadas** (mismo patrón que en OA/OC y reposo/tarea). Con 5
clases y solo 145 personas independientes, el Random Forest tiene más
parámetros que ajustar por unidad de información y generaliza algo peor a
personas nuevas.

## 6. Coherencia neurofisiológica — qué pesa en el modelo

Cinco criterios, **declarados como constantes del guion antes de ejecutar**:

| | Criterio |
|---|---|
| (a) | los dos modelos superan el azar: el límite inferior del IC95 de la exactitud balanceada está por encima de 1/5 = 0.200 |
| (b) | los dos pares cognitivamente gemelos —PASAT-PVSAT y STROOP-MSIT— están entre los 3 pares más confundidos de los 10 |
| (c) | PVTB es la clase con mayor exhaustividad (regresión logística) |
| (d) | el coeficiente neto de las features `_media` de theta frontal en la logística es mayor para PASAT y PVSAT que para PVTB, en todos los pliegues |
| (e) | la confusión sigue el parecido cognitivo más que la posición en el protocolo: ρ(confusión, similitud) > 0 y \|ρ(confusión, similitud)\| > \|ρ(confusión, −distancia protocolo)\| |

Con la salvedad, declarada junto a ellos: **(b) y (e) están parcialmente
confundidos con la duración de la grabación** (STROOP y MSIT duran lo mismo, y
PASAT y PVSAT también), que es justo lo que mide la sección 8.2. Se citan
siempre juntos.

### 6.1 Importancia por permutación (Random Forest, out-of-fold)

Caída de exactitud balanceada de validación al barajar cada columna (5
barajados × 5 pliegues):

| Feature | Caída BA | SD pliegues |
|---|---|---|
| `frontal_beta_rel_sd` | +0.0103 | 0.0026 |
| `frontal_alfa_rel_sd` | +0.0055 | 0.0106 |
| `posterior_alfa_rel_sd` | +0.0051 | 0.0128 |
| `frontal_theta_rel_sd` | +0.0040 | 0.0104 |
| `frontocentral_delta_rel_sd` | +0.0032 | 0.0100 |
| `posterior_alfa_rel_media` | +0.0031 | 0.0047 |
| `frontocentral_delta_logabs_media` | +0.0031 | 0.0020 |
| `indice_fatiga_media` | +0.0030 | 0.0047 |

Ninguna columna individual cuesta más de 1 punto de exactitud balanceada: con
70 features correlacionadas, cada una por separado aporta poco (la misma
observación que en OA/OC y reposo/tarea, y la razón de que exista la
permutación conjunta de 6.3).

### 6.2 Agregación por bloque (fracción de la importancia total)

| Por banda | | Por región | |
|---|---|---|---|
| **beta** | **30.8 %** | **frontal** | **41.2 %** |
| alfa | 30.0 % | posterior | 23.5 % |
| delta | 18.3 % | frontocentral | 19.9 % |
| theta | 16.1 % | temporal | 10.6 % |
| sin asignar | 4.8 % | sin asignar | 4.8 % |

A diferencia de OA/OC (alfa dominante) y reposo/tarea (alfa muy por delante de
theta), aquí **ninguna banda domina claramente**: beta y alfa reparten el 61 %
y delta y theta el resto. Es consistente con el problema: distinguir 5 tareas
entre sí no tiene un único biomarcador de referencia como el efecto Berger o
la carga ejecutiva general.

### 6.3 Permutación conjunta de bloques enteros

Barajar de una vez todas las columnas de un bloque, con la misma permutación
(no se diluye entre features casi duplicadas):

| Bloque | Caída de exactitud balanceada | SD entre pliegues |
|---|---|---|
| **región frontal** | **+0.0958** | 0.0081 |
| banda theta | +0.0459 | 0.0174 |
| banda alfa | +0.0455 | 0.0188 |
| banda delta | +0.0449 | 0.0162 |
| región posterior | +0.0379 | 0.0142 |
| banda beta | +0.0271 | 0.0118 |
| región frontocentral | +0.0254 | 0.0139 |
| región temporal | +0.0085 | 0.0034 |

**La región frontal es, con diferencia, el bloque que más cuesta al barajarlo
entero** (9.6 puntos, más del doble que la segunda). Entre bandas, theta,
alfa y delta cuestan casi lo mismo (4.5–4.6 puntos) y beta bastante menos (2.7).
Esto no coincide con 6.2 (donde beta lideraba por importancia individual
sumada): la permutación conjunta, que no se diluye entre columnas
correlacionadas, es la lectura más fiable, y apunta a la **región**, no a una
banda concreta, como lo más informativo para separar estas cinco tareas.

### 6.4 Dirección del efecto

Coeficiente neto (suma, en unidades de desviación típica) de las tres
features `_media` de **theta frontal**, por clase (media entre los 5
pliegues, con el rango):

| Tarea | Coef. neto theta frontal | Por pliegue |
|---|---|---|
| STROOP | **−3.101** | −3.085, −3.010, −2.948, −3.120, −3.340 |
| PASAT | **+1.227** | +1.204, +1.215, +0.965, +1.273, +1.478 |
| PVTB | +0.852 | +1.095, +0.752, +0.790, +0.734, +0.891 |
| MSIT | −1.033 | −1.105, −1.021, −0.897, −1.043, −1.098 |
| PVSAT | **+2.054** | +1.891, +2.064, +2.090, +2.156, +2.069 |

Con softmax el coeficiente de cada clase se lee frente a la combinación de las
demás, no frente a un cero absoluto. **PASAT y PVSAT tienen el coeficiente más
alto, PVTB queda claramente por debajo de las dos, y STROOP y MSIT son
negativos** (theta frontal los distingue empujando hacia las otras clases, no
hacia sí mismas). El criterio (d) —PASAT y PVSAT por encima de PVTB, en todos
los pliegues— **se cumple con margen** en los 5 pliegues.

### 6.5 d de Cohen una-contra-resto

| Tarea | d theta frontal | d alfa posterior |
|---|---|---|
| STROOP | −0.149 | −0.093 |
| PASAT | +0.069 | −0.160 |
| PVTB | **−0.370** | **+0.276** |
| MSIT | +0.172 | −0.012 |
| PVSAT | +0.277 | −0.006 |

Tamaños de efecto pequeños a moderados (máximo |d| = 0.37): nada aquí se
acerca a los d ≈ 1.3–1.6 de OA/OC o reposo/tarea, porque el contraste ya no es
«hay carga o no la hay» sino «qué tipo de carga», un efecto más fino por
construcción. **PVTB tiene el theta frontal más bajo y el alfa posterior más
alto de las cinco tareas**: es la que menos se parece, en estas dos bandas de
referencia, a un patrón de carga ejecutiva, coherente con ser la de vigilancia
sostenida sin memoria de trabajo ni conflicto de respuesta.

### 6.6 Veredicto

| | Criterio | Resultado |
|---|---|---|
| SÍ | (a) los dos modelos superan el azar con el IC95 | logística [0.452,0.488], RF [0.415,0.454] |
| SÍ | (b) los pares gemelos entre los 3 más confundidos | STROOP-MSIT y PASAT-PVSAT son los 2 más confundidos de los 10 |
| SÍ | (c) PVTB es la clase con mayor exhaustividad | 0.638 (logística), 0.659 (Random Forest) |
| SÍ | (d) theta frontal mayor en PASAT/PVSAT que en PVTB, todos los pliegues | +1.23/+2.05 frente a +0.85, en los 5 |
| SÍ | (e) la confusión sigue más el parecido cognitivo que la distancia en el protocolo | ρ = +0.47 frente a −0.27 |

**Se cumplen los 5 de 5.** Con la salvedad declarada de antemano: (b) y (e)
están parcialmente confundidos con la duración de la grabación (sección 8.2),
así que no se pueden citar como prueba limpia de que el patrón de confusión
sea puramente cognitivo.

## 7. Qué tareas se confunden

### Confusión por pares

De más a menos confundido (regresión logística, matriz de confusión completa
en la sección siguiente):

| Par | Confusión | Asimetría | Similitud cognitiva | Distancia protocolo |
|---|---|---|---|---|
| **STROOP-MSIT** | **0.257** | +0.001 | **3** | 3 |
| **PASAT-PVSAT** | **0.230** | +0.028 | **3** | 3 |
| PASAT-PVTB | 0.163 | +0.018 | 1 | 1 |
| PVTB-PVSAT | 0.128 | −0.034 | 1 | 2 |
| MSIT-PVSAT | 0.113 | −0.015 | 2 | 1 |
| STROOP-PASAT | 0.108 | −0.029 | 2 | 1 |
| STROOP-PVSAT | 0.094 | +0.000 | 2 | 4 |
| STROOP-PVTB | 0.087 | +0.012 | 1 | 2 |
| PASAT-MSIT | 0.080 | +0.010 | 2 | 2 |
| PVTB-MSIT | 0.065 | −0.018 | 1 | 1 |

**Los dos pares más confundidos son exactamente los dos declarados de antemano
como cognitivamente más parecidos** (similitud 3), y la asimetría de ambos es
prácticamente cero (el error va igual en las dos direcciones, no es que una
tarea "se lea como" la otra sistemáticamente).

Matriz de confusión completa (% de cada fila; regresión logística / Random
Forest):

| Real \ Predicha | STROOP | PASAT | PVTB | MSIT | PVSAT |
|---|---|---|---|---|---|
| STROOP | 47.2 / 39.3 | 7.9 / 11.6 | 9.9 / 11.8 | **25.7 / 28.3** | 9.4 / 9.0 |
| PASAT | 13.7 / 12.2 | 33.6 / 31.3 | 18.0 / 24.4 | 9.0 / 5.3 | **25.7 / 26.8** |
| PVTB | 7.5 / 4.3 | 14.5 / 14.5 | **63.8 / 65.9** | 4.7 / 2.3 | 9.4 / 13.0 |
| MSIT | **25.7 / 26.6** | 7.0 / 9.6 | 8.3 / 9.2 | 49.2 / 40.8 | 9.8 / 13.8 |
| PVSAT | 9.5 / 7.4 | **20.3 / 23.5** | 16.3 / 20.6 | 12.9 / 8.7 | 41.1 / 39.8 |

### Separabilidad uno-contra-uno (10 logísticas binarias)

De más difícil a más fácil de separar (agrupado por persona, IC95 con 2000
réplicas de bootstrap):

| Par | n | Personas | BA | IC95 |
|---|---|---|---|---|
| **PASAT-PVSAT** | 1061 | 145 | **0.592** | [0.567, 0.617] |
| **STROOP-MSIT** | 1064 | 145 | **0.615** | [0.589, 0.639] |
| PASAT-PVTB | 1064 | 145 | 0.742 | [0.719, 0.767] |
| MSIT-PVSAT | 1058 | 145 | 0.751 | [0.725, 0.776] |
| STROOP-PASAT | 1067 | 145 | 0.759 | [0.731, 0.786] |
| STROOP-PVSAT | 1062 | 145 | 0.763 | [0.736, 0.789] |
| PASAT-MSIT | 1063 | 145 | 0.764 | [0.740, 0.788] |
| PVTB-PVSAT | 1059 | 145 | 0.781 | [0.756, 0.807] |
| STROOP-PVTB | 1065 | 145 | 0.816 | [0.793, 0.839] |
| **PVTB-MSIT** | 1061 | 145 | **0.844** | [0.821, 0.867] |

Coincide con la confusión: **los dos pares más difíciles de separar de forma
aislada (PASAT-PVSAT, STROOP-MSIT) son los mismos que más se confunden dentro
del modelo de 5 clases**, y el más fácil de separar (PVTB-MSIT, 0.844) tiene la
menor confusión de los 10 (0.065). La separabilidad uno-contra-uno y la
confusión observada cuentan la misma historia por dos caminos independientes.

### Estructura de la confusión: qué la predice mejor

Correlación de Spearman de la confusión simétrica de los 10 pares contra las
tres estructuras candidatas (n = 10; descriptivo, sin potencia para un
contraste formal):

| Comparación | ρ |
|---|---|
| confusión ~ similitud cognitiva | **+0.467** |
| confusión ~ (−) distancia en el protocolo | −0.267 |
| similitud cognitiva ~ (−) distancia en el protocolo | −0.517 |
| confusión ~ (−) distancia de duración | **+0.758** |
| similitud cognitiva ~ (−) distancia de duración | +0.027 |
| distancia en el protocolo ~ distancia de duración | −0.331 |

**La confusión sigue el parecido cognitivo (+0.47) más que la distancia en el
protocolo (−0.27), cumpliendo el criterio (e)**, pero la sigue todavía **más
la distancia de duración (+0.76)** — el predictor individual más fuerte de
los tres. Y la similitud cognitiva y la distancia de duración son casi
independientes entre sí (ρ = +0.03): eso significa que **la duración no
puede explicar por sí sola por qué la similitud cognitiva también predice la
confusión** (si la duración fuera la única causa, similitud y duración
estarían correlacionadas y no lo están), pero tampoco se puede afirmar que el
parecido cognitivo por sí solo explique la confusión sin controlar la
duración, porque ambas apuntan en la misma dirección para los pares más
confundidos (STROOP-MSIT y PASAT-PVSAT comparten duración *y* similitud
cognitiva). Este análisis no puede separar limpiamente las dos.

## 8. Controles y sensibilidad

### 8.1 Control negativo: etiquetas barajadas dentro de cada persona

Destruye la relación entre las features y la tarea pero conserva las personas
y cuántas grabaciones de cada clase tiene cada una (20 barajados por modelo).

| Modelo | Exactitud balanceada con etiquetas barajadas | Real |
|---|---|---|
| Logística | 0.1981 (SD 0.0092; rango 0.177–0.215) | 0.4701 |
| Random Forest | 0.1990 (SD 0.0077; rango 0.183–0.217) | 0.4345 |

Ninguno de los 40 nulos se acerca al valor real. **El arnés no produce acierto
de la nada**; el nulo está centrado casi exactamente en 0.200.

### 8.2 Control de duración: los 6 contadores de procesamiento, solos

**Este es el resultado que más condiciona la lectura de todo el análisis.**
STROOP y MSIT duran ~60 s; PASAT, PVTB y PVSAT ~150 s (`docs/ESTADO.md`,
«Duraciones»). Si los 6 contadores —`n_ventanas`, `n_canales_buenos` y los
cuatro `n_validos_*`, ninguno de ellos señal cerebral— bastan por sí solos
para distinguir las tareas, la duración es un confusor real, no hipotético.

| Modelo | Solo 6 contadores: BA/pliegue | IC95 | Principal (70 features espectrales) |
|---|---|---|---|
| Logística | **0.4652** | [0.449, 0.484] | 0.4701 [0.452, 0.488] |
| Random Forest | **0.4711** | [0.454, 0.488] | 0.4345 [0.415, 0.454] |

**Los 6 contadores, sin una sola feature espectral, igualan a la logística
principal y superan al Random Forest principal.** No es un efecto pequeño ni
marginal como en OA/OC (donde los contadores discriminaban con d pequeños): es
un modelo completo, comparable en rendimiento al que usa las 70 features de
EEG. Añadir los contadores a las 70 features (sección 8.3) sube el resultado
en +0.122 (logística) y +0.135 (Random Forest) sobre el principal, y con solo
los contadores el Random Forest (0.471) ya es más alto que el Random Forest
espectral completo (0.434).

**Consecuencia para la lectura de las secciones 6 y 7.** No se puede afirmar
que el 0.470 de exactitud balanceada del modelo principal sea puramente
neurofisiológico: una parte no acotada de ese acierto podría provenir de que
las features espectrales absolutas y de variabilidad (`_sd`) arrastran
indirectamente cuánto dura la grabación (más ventanas cambia la media y la
desviación típica de la potencia estimada), sin que eso implique nada sobre la
demanda cognitiva de la tarea. Es la misma preocupación de la Fase 18 (allí
0.990 con solo los contadores, en un problema binario más fácil) pero aquí
proporcionalmente más grave: **el confusor iguala al efecto de interés, no
solo lo acompaña**. Los 6 contadores permanecen fuera de X (no hay
alternativa: son metadatos de calidad y de tamaño, no señal cerebral), pero
este resultado se cita siempre junto con el 0.470, no en su lugar.

### 8.3 Sensibilidad: qué features entran a X

Δ es la diferencia **pareada** frente al principal (misma remuestra de
personas en las dos ramas), con su IC95.

| Subconjunto | n | Modelo | BA/pliegue | IC95 | Δ vs principal |
|---|---|---|---|---|---|
| **Principal** | 70 | Logística | 0.4701 | [0.452, 0.488] | — |
| | | Random Forest | 0.4345 | [0.415, 0.454] | — |
| Sin las 4 copias exactas | 66 | Logística | 0.4671 | [0.449, 0.484] | −0.0030 [−0.006, −0.000] |
| | | Random Forest | 0.4379 | [0.417, 0.458] | +0.0034 [−0.006, +0.013] |
| **+ 6 contadores de procesamiento** | 76 | Logística | **0.5921** | [0.573, 0.610] | **+0.1220** [+0.106, +0.137] |
| | | Random Forest | **0.5700** | [0.551, 0.588] | **+0.1354** [+0.119, +0.152] |
| Solo las features `_media` | 35 | Logística | 0.3916 | [0.371, 0.411] | −0.0787 [−0.098, −0.058] |
| | | Random Forest | 0.3508 | [0.333, 0.370] | −0.0835 [−0.104, −0.062] |
| Solo las features `_sd` | 35 | Logística | 0.3905 | [0.373, 0.409] | −0.0794 [−0.097, −0.061] |
| | | Random Forest | 0.4037 | [0.386, 0.421] | −0.0306 [−0.044, −0.017] |
| Solo banda delta | 16 | Logística | 0.3769 | [0.361, 0.394] | −0.0931 [−0.114, −0.073] |
| | | Random Forest | 0.3759 | [0.358, 0.393] | −0.0584 [−0.077, −0.041] |
| Solo banda theta | 18 | Logística | 0.3830 | [0.365, 0.401] | −0.0869 [−0.106, −0.068] |
| | | Random Forest | 0.3760 | [0.358, 0.393] | −0.0583 [−0.078, −0.039] |
| Solo banda alfa | 18 | Logística | 0.3772 | [0.361, 0.394] | −0.0928 [−0.111, −0.074] |
| | | Random Forest | 0.3797 | [0.362, 0.397] | −0.0547 [−0.075, −0.035] |
| Solo banda beta | 16 | Logística | 0.3714 | [0.354, 0.388] | −0.0987 [−0.121, −0.077] |
| | | Random Forest | 0.3871 | [0.370, 0.404] | −0.0475 [−0.068, −0.027] |
| Solo región frontal | 18 | Logística | 0.4180 | [0.402, 0.434] | −0.0521 [−0.071, −0.035] |
| | | Random Forest | 0.4069 | [0.385, 0.428] | −0.0275 [−0.048, −0.008] |
| Solo región frontocentral | 16 | Logística | 0.3680 | [0.351, 0.385] | −0.1020 [−0.121, −0.083] |
| | | Random Forest | 0.3640 | [0.345, 0.383] | −0.0704 [−0.089, −0.051] |
| Solo región temporal | 16 | Logística | 0.3333 | [0.316, 0.351] | −0.1368 [−0.157, −0.117] |
| | | Random Forest | 0.3528 | [0.338, 0.369] | −0.0814 [−0.104, −0.059] |
| Solo región posterior | 18 | Logística | 0.3612 | [0.343, 0.379] | −0.1089 [−0.131, −0.087] |
| | | Random Forest | 0.3718 | [0.354, 0.390] | −0.0626 [−0.083, −0.042] |

Lo que se lee:

- **Las cuatro copias exactas no cambian nada** (Δ pequeño, IC bordeando o
  incluyendo 0).
- **Los contadores son otra vez un atajo enorme** (sección 8.2): añadirlos
  suma más de 12 puntos con los dos modelos, muy por encima de lo que aporta
  cualquier subconjunto espectral individual.
- **Ninguna banda ni región sola basta**: todas caen entre 5 y 14 puntos
  respecto al principal, y ninguna se acerca a lo que aportan los contadores.
  La región frontal es la que menos cuesta perder (−5.2/−2.8 puntos),
  coherente con 6.3 (es el bloque más importante en la permutación conjunta).
  La región temporal es la que más cuesta perder sola —es decir, aislada
  aporta menos que las demás—.
- **`_media` y `_sd` aportan de forma parecida y ninguna basta por
  separado** (ambas caen 7.9–8.4 puntos con la logística); no hay una de las
  dos que sea claramente prescindible.

### 8.4 Veredicto completo frente a los 5 criterios declarados

| | Criterio | Resultado |
|---|---|---|
| SÍ | (a) los dos modelos superan el azar con el IC95 | logística y RF, IC95 > 0.200 |
| SÍ | (b) los pares gemelos entre los 3 más confundidos | STROOP-MSIT y PASAT-PVSAT, los 2 más confundidos |
| SÍ | (c) PVTB es la clase con mayor exhaustividad | 0.638 / 0.659 |
| SÍ | (d) theta frontal mayor en PASAT/PVSAT que en PVTB, todos los pliegues | +1.23/+2.05 frente a +0.85 |
| SÍ | (e) la confusión sigue más el parecido cognitivo que la distancia en el protocolo | ρ +0.47 frente a −0.27 |

**Se cumplen los 5 de 5.** Con la salvedad citada desde el principio: (b) y
(e) están parcialmente confundidos con la duración de la grabación, que
predice la confusión (ρ = +0.76) todavía mejor que la propia similitud
cognitiva (ρ = +0.47), y que **por sí sola, sin ninguna feature espectral,
iguala el rendimiento del modelo principal** (sección 8.2). Ningún criterio de
los cinco controla por duración de forma limpia.

## 9. Resolución por canal (suplemento)

`matriz_rf.parquet` agrega por región. Para ver qué canales llevan el efecto
se reagrega `matriz_modelado.parquet` con `construir_matriz_rf` y una región
por canal, restringida a las mismas 2656 grabaciones y leída ya filtrada (las
épocas del test no llegan a cargarse).

**Banda theta** (d de Cohen una-contra-resto, potencia relativa):

| Canal | STROOP | PASAT | PVTB | MSIT | PVSAT |
|---|---|---|---|---|---|
| AF3 | −0.13 | +0.09 | −0.24 | +0.04 | +0.28 |
| AF4 | −0.13 | +0.09 | −0.22 | −0.03 | **+0.36** |
| F3 | −0.08 | 0.00 | −0.34 | +0.21 | +0.21 |
| F4 | −0.10 | −0.01 | −0.32 | +0.22 | +0.22 |
| F7 | −0.13 | +0.09 | −0.12 | −0.08 | +0.25 |
| F8 | −0.13 | +0.12 | −0.21 | −0.03 | +0.26 |
| FC5 | −0.09 | +0.06 | −0.20 | −0.03 | +0.26 |
| FC6 | −0.05 | +0.01 | −0.17 | +0.01 | +0.21 |
| O1 | +0.09 | −0.16 | −0.41 | **+0.45** | +0.03 |
| O2 | +0.15 | −0.18 | **−0.44** | **+0.49** | −0.01 |
| P7 | +0.04 | −0.10 | −0.21 | +0.24 | +0.03 |
| P8 | −0.04 | −0.08 | −0.19 | +0.26 | +0.05 |
| T7 | +0.03 | −0.01 | −0.26 | +0.12 | +0.11 |
| T8 | −0.03 | −0.01 | −0.23 | +0.15 | +0.12 |

**Banda alfa:**

| Canal | STROOP | PASAT | PVTB | MSIT | PVSAT |
|---|---|---|---|---|---|
| AF3 | −0.07 | −0.21 | +0.42 | −0.07 | −0.08 |
| AF4 | −0.01 | −0.24 | +0.36 | −0.04 | −0.11 |
| F3 | −0.03 | −0.22 | +0.45 | −0.08 | −0.10 |
| F4 | +0.06 | −0.26 | +0.37 | −0.05 | −0.11 |
| F7 | −0.10 | −0.12 | +0.42 | −0.16 | −0.02 |
| F8 | −0.05 | −0.17 | +0.35 | −0.08 | −0.06 |
| FC5 | −0.11 | −0.11 | +0.32 | −0.17 | +0.08 |
| FC6 | −0.05 | −0.15 | +0.28 | −0.13 | +0.05 |
| O1 | −0.12 | −0.24 | **+0.53** | −0.02 | −0.15 |
| O2 | −0.09 | −0.25 | **+0.55** | −0.06 | −0.14 |
| P7 | −0.12 | −0.21 | +0.49 | +0.01 | −0.16 |
| P8 | −0.07 | −0.22 | +0.46 | −0.07 | −0.10 |
| T7 | +0.02 | **−0.30** | +0.32 | −0.01 | −0.03 |
| T8 | −0.05 | −0.25 | +0.35 | −0.07 | +0.03 |

Dos lecturas:

- **El alfa de PVTB es positivo en los 14 canales, sin una sola excepción**
  (+0.28 a +0.55), el patrón más consistente por canal de todo el análisis, y
  máximo en los occipitales (O1, O2). Es coherente con ser la tarea de
  vigilancia sostenida: sin conflicto de respuesta ni carga de memoria de
  trabajo, el alfa se comporta más como en reposo.
- **El theta no tiene un patrón único por canal.** STROOP y MSIT (las dos de
  interferencia) tienen theta ligeramente **negativo** en los canales
  frontales pero **positivo** en O1 y O2; PVSAT es justo lo contrario, positivo
  en frontal y casi nulo en posterior. Esto significa que el coeficiente neto
  de «theta frontal» de la sección 6.4 —una única cifra por región— **no
  describe un patrón homogéneo dentro de la propia región** para todas las
  clases; conviene leerlo como una dirección promedio, no como una firma
  espacial limpia.

## 10. Las figuras

### `resultados/figuras/tareas_ejecutivas_confusion.png`

Las dos matrices de confusión 5×5, lado a lado y con la misma escala. El color
es el porcentaje de cada clase real (cada fila suma 100 %). La diagonal más
oscura es PVTB (64–66 %); los bloques claros más visibles de fuera de la
diagonal son STROOP↔MSIT y PASAT↔PVSAT, los dos pares gemelos.

### `resultados/figuras/tareas_ejecutivas_pares.png`

Dos paneles ordenados de mayor a menor similitud cognitiva declarada:
separabilidad uno-contra-uno (izquierda, con IC95) y confusión real dentro del
modelo de 5 clases (derecha). Los dos pares de similitud 3 (rojo) son a la vez
los menos separables y los más confundidos; los de similitud 1 (azul) están
repartidos, porque incluyen tanto el par más fácil (PVTB-MSIT) como uno
intermedio (PASAT-PVTB).

### `resultados/figuras/tareas_ejecutivas_canales.png`

Dos mapas de calor (theta, alfa), canales en filas y las 5 tareas en columnas,
con el d de Cohen una-contra-resto de cada celda. El patrón más visible es la
franja roja casi uniforme de PVTB en el panel de alfa: sube en los 14 canales.

Paleta cualitativa de 5 colores resistente a daltonismo (Okabe & Ito, 2008),
una por tarea; a diferencia de las Fases 17/18 (que usaban solo 2 colores
validados con el validador de la guía de visualización del proyecto), esta
paleta de 5 no se ha pasado por ese validador y se declara como elección
razonable, no como validación formal.

## 11. Conclusión

**El pipeline distingue entre las cinco tareas ejecutivas, muy por encima del
azar y con un patrón de confusión que sigue el parecido cognitivo declarado
antes de ejecutar.** Sobre personas no vistas, la regresión logística separa
las cinco clases con **0.470** de exactitud balanceada (IC95 [0.452, 0.488];
azar 0.200) y el Random Forest con 0.434 [0.415, 0.454]; el control con
etiquetas barajadas da ~0.199 en los dos. Se cumplen los 5 criterios de
coherencia declarados de antemano: los dos pares más confundidos son
exactamente los dos declarados como cognitivamente más parecidos
(STROOP-MSIT, PASAT-PVSAT), PVTB —la tarea sin conflicto de respuesta ni
carga de memoria de trabajo— es la mejor reconocida y la que muestra el alfa
más alto en los 14 canales sin excepción, y el coeficiente de theta frontal es
mayor en las dos tareas de máxima demanda de memoria de trabajo (PASAT,
PVSAT) que en PVTB, en los cinco pliegues.

**Pero el resultado no se puede leer como puramente neurofisiológico.** Los
seis contadores de procesamiento —duración de la grabación y calidad de
contacto, ninguno señal cerebral— igualan o superan por sí solos el
rendimiento del modelo con las 70 features espectrales (sección 8.2), y la
duración de la grabación predice el patrón de confusión mejor que la propia
similitud cognitiva (ρ = 0.76 frente a 0.47; sección 7). Esto no invalida los
5 criterios —se miden y se cumplen sobre el modelo espectral, sin los
contadores—, pero sí significa que **no se puede afirmar que el patrón
observado sea evidencia limpia de una firma cognitiva por tarea**: podría ser,
en una parte no acotada, un reflejo de cuánto dura cada grabación.

### Alcance de la afirmación

- **Sí establece** que las features de la Fase 13 contienen información que
  separa las cinco tareas ejecutivas entre sí, muy por encima del azar, y que
  el patrón de qué se confunde con qué no es aleatorio: covaría con el
  parecido cognitivo declarado antes de ver los datos.
- **No establece** que esa información sea una firma cognitiva "limpia": el
  orden fijo del protocolo confunde tarea con posición en la sesión, y la
  duración de la grabación —confundida a su vez con ambas— iguala por sí sola
  el rendimiento del modelo espectral.
- **No establece** qué tarea concreta explica mejor la fatiga percibida. Este
  análisis no toca el VAS ni la etiqueta de fatiga; solo la condición.
- **No establece** nada sobre la señal cruda de la ruta CNN.
- **No es confirmatorio.** Se ejecutó sobre `train` + `val`. Al ser la
  etiqueta la condición y no el VAS, **no** consume el alfa del paquete de
  test pendiente, que es sobre la fatiga; las configuraciones exploradas aquí
  no se suman a ese recuento. El test sigue sin abrirse.

## 12. Limitaciones

- **La duración de la grabación es un confusor que iguala el efecto
  principal, no solo lo acompaña.** Los 6 contadores de procesamiento, solos
  y sin ninguna feature espectral, dan 0.465–0.471 de exactitud balanceada,
  frente a 0.434–0.470 del modelo completo (sección 8.2). Es la limitación
  más seria de este análisis y no tiene solución con este diseño: la duración
  de cada condición es fija (STROOP y MSIT ~60 s; PASAT, PVTB, PVSAT ~150 s),
  así que no hay forma de comparar tareas de duración distinta sin que la
  duración compita como explicación.
- **El orden del protocolo es fijo: STROOP → PASAT → PVTB → MSIT → PVSAT**
  (`docs/ESTADO.md`). La condición y el tiempo acumulado en la sesión están
  perfectamente confundidos, y este diseño **no puede separarlos**. A
  diferencia de la Fase 18, aquí no hay un par adyacente con clases
  equilibradas y suficiente n para un control limpio de posición (cada tarea
  aparece una sola vez por sesión, en un único punto del protocolo), así que
  esta limitación **no se ha podido acotar con un control adicional**, más
  allá de la comparación de estructuras de la sección 7.3.
- **Duración y distancia en el protocolo están parcialmente confundidas entre
  sí y con la similitud cognitiva** (ver la matriz de correlaciones de la
  sección 7.3): los pares de similitud cognitiva máxima comparten también
  duración. El diseño no permite atribuir limpiamente la confusión observada
  a una sola de las tres causas.
- **PVTB se comporta de forma distinta a las otras cuatro** (mejor
  reconocida, alfa positivo en los 14 canales) y no se ha investigado más allá
  de la hipótesis de que exige menos control ejecutivo. Es la misma
  observación que en la Fase 18 (allí PVTB era la única tarea mal reconocida
  como reposo); aquí se lee de forma consistente pero desde el ángulo
  contrario (mejor reconocida entre tareas, no peor reconocida frente a
  reposo).
- **El theta no tiene un patrón espacial homogéneo por clase** (sección 9):
  el coeficiente neto de «theta frontal» de la sección 6.4 es una cifra
  agregada por región que no describe una firma limpia y uniforme dentro de
  esa región para todas las tareas.
- **Tamaños de efecto pequeños a moderados** (d ≤ 0.37 en los descriptores de
  referencia, sección 6.5): un problema de 5 clases con diferencias más finas
  que "hay carga o no la hay" tiene, por construcción, efectos univariados más
  pequeños que en OA/OC o reposo/tarea.
- **Artefactos oculares, musculares y de movimiento** propios de hacer una
  tarea (mirar una pantalla, responder) no se pueden separar de una firma
  cortical de la tarea con estas features, igual que en la Fase 18.
- **Sin Fz y con 14 electrodos.** El theta frontal medial se aproxima con la
  región de seis canales frontales; las regiones de la Fase 13 son gruesas.
- **Una configuración por modelo, sin ajustar hiperparámetros.** El Random
  Forest por debajo de la logística no es una afirmación sobre lo que un
  Random Forest ajustado podría hacer.
- **Importancia por permutación con features correlacionadas y duplicadas**
  infravalora cada una por separado; por eso se añade la permutación
  conjunta de bloques (sección 6.3). El control con etiquetas barajadas usa
  20 barajados por modelo: su p empírico no puede bajar de 1/21 = 0.048.
- **Correlaciones de la sección 7.3 con n = 10** (los 10 pares de tareas): son
  descriptivas, sin potencia estadística para un contraste formal.
- **Cohorte y montaje:** 145 personas de un estudio de la Facultad de
  Ciencias del Deporte (edad y demografía no confirmadas), Emotiv Epoc X,
  ventanas de 4 s agregadas por grabación. Los resultados pueden no
  trasladarse a otro montaje o población.

## 13. Lo que **no** hay que hacer

- **Usar `columnas_de_features()` tal cual.** Devuelve `condicion_orden`, que
  dentro de las 5 tareas es la etiqueta con otro nombre. Usar
  `columnas_de_features_ejecutivas()`.
- **Reincorporar los contadores de procesamiento a X.** Solos ya igualan al
  modelo espectral completo (sección 8.2); mezclarlos con las 70 features
  espectrales sube el resultado más de 12 puntos sin que sea señal cerebral.
- **Citar 0.470 como «el pipeline reconoce la tarea por su neurofisiología»**
  sin la salvedad de la sección 8.2: el mismo rendimiento se obtiene con 6
  números que no son señal cerebral.
- **Citar el criterio (b) o (e) como evidencia de que la confusión es
  puramente cognitiva.** Los dos están declarados desde el principio como
  parcialmente confundidos con la duración, y la duración predice la
  confusión mejor que la similitud cognitiva (ρ 0.76 frente a 0.47).
- **Leer la diferencia logística − Random Forest como diferencia de
  capacidad de un Random Forest bien ajustado.** Solo se probó una
  configuración, la de las Fases 14/14b/17/18.
- **Leer las importancias o los coeficientes individuales como pesos
  separados.** Hay copias exactas y features muy correlacionadas; la lectura
  fiable es la permutación conjunta por bloque (6.3) y el coeficiente neto por
  bloque (6.4), no una columna aislada.
- **Presentar la ausencia de una banda dominante (sección 6.2) como que
  «ninguna banda importa».** La permutación conjunta (6.3) sí distingue: la
  región frontal pesa más del doble que cualquier banda sola.
- **Extrapolar el patrón por canal de theta (sección 9) como una firma
  espacial limpia.** Su signo cambia entre canales frontales y occipitales
  según la tarea; la cifra agregada de la sección 6.4 es un promedio, no una
  descripción espacial.
- **Contar las 2656 grabaciones como n independiente.** Son 145 personas.
- **Comparar directamente con los IC95 de OA/OC o reposo/tarea** para decir
  que el efecto aquí es "más débil": son problemas con distinto número de
  clases y distinto azar (0.5, 0.5 y 0.2 respectivamente), no comparables
  cifra a cifra.
- **Ajustar el modelo o el umbral a este problema para subir la cifra.** No
  es el objetivo, y cada configuración adicional engorda el recuento de
  configuraciones exploradas.
- **Abrir el test.** `preparar_tareas_ejecutivas` se niega salvo
  `permitir_test=True`.

## 14. Reproducir

```bash
# desde la raíz del repositorio; una sola vez y nunca en paralelo (la RAM es limitada)
python analisis/analisis_tareas_ejecutivas.py > resultados/logs/tareas_ejecutivas.log 2>&1
```

Duración: **~26 min** en la ejecución que produjo estos resultados. Semilla
fija (`SEMILLA = 42`) en los modelos, las permutaciones y todos los
bootstraps.

**Reproducción comprobada el 2026-10-06.** Se repitió la ejecución completa
en otra máquina, con las mismas versiones y regenerando desde cero las
etiquetas y `matriz_rf.parquet` (24 min 25 s). El log coincide línea a línea
con el original salvo las rutas de los archivos y un aviso de matplotlib
(`tight_layout`) que esta vez no se emitió, y las tres figuras son idénticas
byte a byte. El JSON no se pudo comparar: el original se perdió y este es el
regenerado.

`--rapido` ejecuta el mismo código con parámetros mínimos (40 réplicas de
bootstrap, 30 árboles, 2 barajados de etiquetas) y escribe en un directorio
temporal, en menos de un minuto. Sirve para comprobar que el guion corre y
revisar las figuras; **sus cifras de bootstrap/permutación no son
resultados** (aunque, como no reduce el propio dataset, la logística da
exactamente los mismos puntos estimados que la ejecución completa).

Además de `matriz_rf.parquet`, el suplemento por canal necesita
`<EEG_DATA>/processed/matriz_modelado.parquet` y
`<EEG_DATA>/metadata/labels_grabaciones.csv`. El guion aparta el `test` en su
primer paso, `preparar_tareas_ejecutivas` se niega a incluirlo sin permiso
explícito, y el suplemento por canal lee el parquet de épocas ya filtrado a
las grabaciones de análisis.

`*.log` y `resultados/figuras/` están en el `.gitignore` del repositorio (igual que en
las Fases 17 y 18): el log y las figuras existen en disco pero no se
versionan salvo que se fuerce.

Versiones con las que se generaron estos resultados: Python 3.10.11,
scikit-learn 1.7.2, numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, matplotlib
3.10.9.

## 15. Documentos relacionados

| Documento | Qué cubre |
|---|---|
| `docs/MODELADO.md`, Fase 19 | Este análisis, resumido en el cuaderno de modelado |
| `docs/analisis/ANALISIS_REPOSO_TAREA.md` (Fase 18) | El contraste binario anterior (reposo frente a cualquier tarea); de aquí se reutilizan el arnés, las exclusiones de X y la estructura del guion |
| `docs/analisis/ANALISIS_OA_OC.md` (Fase 17) | El control positivo del efecto Berger, origen del módulo `reposo.py` que ambos análisis posteriores reutilizan |
| `docs/ESTADO.md`, Fase 13 | Cómo se construyen las 77 features y qué significa cada una |
| `docs/ESTADO.md`, «Orden del protocolo» y «Duraciones» | El orden fijo que confunde condición y posición, y las duraciones por condición que explican el confusor de la sección 8.2 |
| `docs/ESTADO.md`, Fase 12 | Por qué la agrupación es `persona_uid` y nunca `sujeto_id` |
