# Reposo frente a tarea ejecutiva: ¿distingue el pipeline la carga cognitiva?

> **Qué es este documento.** La clasificación binaria de las grabaciones de
> **reposo** (ojos abiertos OA u ojos cerrados OC) frente a las de **tarea
> ejecutiva** (STROOP, PASAT, PVTB, MSIT, PVSAT), a partir de las features de la
> Fase 13. Es el escalón siguiente al control OA/OC: allí el efecto era el más
> grande y localizado del EEG de superficie (efecto Berger); aquí es un efecto de
> **carga cognitiva**, más pequeño y más parecido en naturaleza a la fatiga
> mental, que es la variable de interés del proyecto. No busca un clasificador
> útil —nadie necesita adivinar si alguien está haciendo una tarea—: busca saber
> si el pipeline conserva esa señal y qué es lo que la lleva.
>
> **Resultado en una línea:** sí, y lo que la lleva es sobre todo el **descenso
> del alfa**. Validando sobre personas no vistas, la regresión logística acierta
> con **0.857** de exactitud balanceada (IC95 [0.839, 0.874]; azar 0.500) y el
> Random Forest con 0.831 [0.812, 0.851]. Sin OC dentro del reposo baja a 0.816
> (logística) pero sigue muy por encima del azar. Cumple los 4 criterios de
> coherencia declarados antes de ejecutar, uno de los cuales es débil por
> construcción. **El orden fijo del protocolo confunde condición y posición en la
> sesión, y este diseño no puede separarlas** (sección 12).
>
> **Guion reproducible:** `analisis/analisis_reposo_tarea.py`
> **Código de apoyo:** `src/eeg_library/modelado/reposo_tarea.py`
> (`tests/test_reposo_tarea.py`, 73 pruebas)
> **Salida numérica:** `<EEG_DATA>/processed/reposo_tarea.json`
> **Registro de ejecución:** `analisis/salidas/reposo_tarea.log`
> **Figuras:** `figures/reposo_tarea_confusion.png`,
> `figures/reposo_tarea_por_condicion.png`, `figures/reposo_tarea_canales.png`
> **Resumen en el cuaderno de modelado:** `MODELADO.md`, Fase 18
>
> Fecha de ejecución: 2026-09-21. **Solo `train` + `val`; el conjunto de test no
> se ha abierto.**

---

## 1. La pregunta, y por qué es distinta de la Fase 17

La Fase 17 (`ANALISIS_OA_OC.md`) mostró que el pipeline conserva el efecto Berger
(0.849 de exactitud balanceada entre OA y OC, con un d de Cohen ≈ 1.3 en el alfa
posterior). Su propia sección de alcance dice lo que **no** establece: que el
pipeline conserve efectos más sutiles. Un control positivo con el efecto más
grande del EEG de superficie solo descarta que el pipeline esté roto para efectos
grandes.

Reposo frente a tarea es el escalón intermedio. Lo que la literatura predice para
una tarea que exige control ejecutivo, con clase 1 = tarea:

- **theta frontal sube** (el theta frontal medial es el marcador canónico de la
  carga ejecutiva);
- **alfa baja**, sobre todo en las regiones posteriores (desincronización del
  alfa al implicarse en una tarea).

Como en la Fase 17, eso hace del control uno *falsable*: predice **qué** debe
pesar, **dónde** y **con qué signo**. Y añade una pregunta que la Fase 17 no
tenía, porque aquí hay dos amenazas a la interpretación que el diseño no puede
eliminar y que por eso el análisis **mide** en vez de esquivar:

1. **El reposo incluye OC.** La mitad de la clase «reposo» tiene el alfa disparado
   por tener los ojos cerrados. Un acierto alto podría ser el efecto Berger otra
   vez, no carga cognitiva. → Sección 8.2.
2. **El orden del protocolo es fijo** (OC → OA → STROOP → PASAT → PVTB → MSIT →
   PVSAT): el reposo ocupa siempre las posiciones 1–2 y las tareas las 3–7, así que
   la condición y el tiempo acumulado en la sesión están confundidos
   (`ESTADO.md`, «Orden del protocolo»). → Sección 8.3, y no se resuelve.

## 2. Datos

| | |
|---|---|
| Grabaciones en la matriz de la Fase 13 | 4543 |
| Apartadas (test) | 732 (28 personas) — **no se leen** |
| Descartadas por no tener `persona_uid` (train + val) | 83 (sin él no se puede agrupar) |
| **En el análisis** | **3728 grabaciones: 1072 de reposo + 2656 de tarea** |
| Reposo | 531 OA + 541 OC |
| Tarea | 534 STROOP + 533 PASAT + 531 PVTB + 530 MSIT + 528 PVSAT |
| Personas / sesiones | 146 / 547 |
| Por split | 3117 train + 611 val |
| Grabaciones por persona | mediana 28, rango 1–35 |
| **Desbalance de clases** | **2.48 : 1** (prevalencia de reposo 0.2876; predecir siempre «tarea» acierta 0.7124) |
| Grabaciones con alguna región sin canales válidos (NaN) | 96 — mediana del entrenamiento dentro de cada pliegue |

**La unidad de análisis es la grabación**, no la sesión: la condición es una
propiedad de cada registro, con su propio EEG, y cada fila es una etiqueta
genuinamente independiente. Lo que **no** es independiente es la persona (cada una
aporta ~28 grabaciones), y eso se controla agrupando por `persona_uid` —nunca
`sujeto_id`, porque seis de ellos contienen dos personas— tanto al validar como al
calcular intervalos.

## 3. Qué entra a X, y qué no

`columnas_de_features()` sigue siendo la única definición de X del proyecto y no se
ha tocado. `columnas_de_features_tarea()` es exactamente la de OA/OC
(`columnas_de_features_reposo`, y un test comprueba que coinciden): parte de las 77
features de la Fase 13 y resta 7, de modo que quedan **70**.

### `condicion_orden` es la etiqueta

Codifica la posición de la condición en el protocolo: **1 (OC), 2 (OA), 3 (STROOP),
4 (PASAT), 5 (PVTB), 6 (MSIT), 7 (PVSAT)**. El máximo en el reposo es 2 y el mínimo
en las tareas es 3: **un solo corte la separa sin error**. Es una fuga más grosera
que en OA/OC y una exclusión obligatoria, no una decisión discutible. Hay un test
que prueba que, metida a mano, da 1.0 de exactitud balanceada.

### Los seis contadores de procesamiento

`n_ventanas`, `n_canales_buenos` y los cuatro `n_validos_*` no son señal cerebral.
Aquí **todos discriminan** reposo de tarea:

| Columna | d de Cohen (tarea − reposo) | p (Mann–Whitney) |
|---|---|---|
| `n_validos_frontal` | −0.349 | 1.4 × 10⁻²⁰ |
| `n_ventanas` | +0.299 | 3.6 × 10⁻²¹ |
| `n_validos_posterior` | +0.227 | 9.7 × 10⁻¹⁰ |
| `n_validos_temporal` | +0.131 | 0.0002 |
| `n_validos_frontocentral` | +0.125 | 0.0006 |
| `n_canales_buenos` | −0.101 | 0.0059 |

(Los p están calculados sobre grabaciones sin agrupar por persona: son
orientativos.) Que discriminen con un d pequeño podría parecer poca cosa. **No lo
es**, como muestra la sensibilidad de la sección 8.4:

| Con los contadores | Exactitud balanceada | Δ vs principal (pareado, IC95) |
|---|---|---|
| **Solo** los 6 contadores, Random Forest | **0.9902** [0.986, 0.994] (AUC 0.994) | +0.1590 [+0.139, +0.178] |
| **Solo** los 6 contadores, logística | 0.6238 [0.605, 0.642] | −0.2329 [−0.255, −0.211] |
| Principal + 6 contadores (76), Random Forest | 0.9440 [0.932, 0.956] | +0.1128 [+0.099, +0.127] |
| Principal + 6 contadores (76), logística | 0.8667 [0.851, 0.882] | +0.0100 [+0.003, +0.017] |

**Con solo seis números de procesamiento, sin una sola feature espectral, el Random
Forest distingue reposo de tarea con 0.990.** La explicación más probable está en
`ESTADO.md` («Duraciones»): la duración mediana de la grabación depende de la
condición —MSIT 69 s, STROOP 68 s, OA y OC ~122 s, PASAT, PVSAT y PVTB ~187 s—, de
modo que `n_ventanas` (que es la duración) identifica casi por sí sola la
condición. Y como el reposo queda en **medio** de las tareas cortas y las largas, la
relación no es monotónica: un modelo lineal no la puede usar (0.624) y un bosque sí
(0.990). Es una explicación, **no una medición**: no se ha desglosado qué contador
lo hace.

Esto convierte la exclusión de los contadores de una precaución en algo que hay que
defender con esta cifra: un análisis que los hubiera dejado en X habría publicado un
0.94–0.99 con el Random Forest que no dice nada del EEG.

### Observación sobre la matriz: cuatro features son copias exactas

Como en la Fase 17, cuatro de las 70 columnas de X son **idénticas valor a valor** a
otra (`theta_frontal_log_media` = `frontal_theta_logabs_media`,
`theta_frontal_log_sd` = `frontal_theta_logabs_sd`, `alfa_posterior_log_media` =
`posterior_alfa_logabs_media`, `alfa_posterior_log_sd` =
`posterior_alfa_logabs_sd`). No se han quitado (el encargo es reutilizar la matriz
tal cual), pero importa para leer las importancias. La sensibilidad comprueba que
no cambian el resultado: Δ = −0.0012 [−0.003, +0.000] con la logística y +0.0035
[−0.002, +0.010] con el Random Forest. Efectivamente hay 66 columnas distintas.

## 4. Modelos, y por qué estos dos

| | Configuración | Qué aporta |
|---|---|---|
| **Regresión logística** | imputación por mediana → tipificación → `LogisticRegression(class_weight="balanced")` | Frontera lineal. Sus coeficientes **tienen signo**, que es lo único que permite comprobar la dirección del efecto (theta hacia tarea, alfa hacia reposo). Ganó en la Fase 17, así que el contraste es directo. |
| **Random Forest** | imputación por mediana → 500 árboles, hoja ≥ 5, `class_weight="balanced"` | Interacciones y no linealidades que la logística no puede capturar. Es **la misma configuración de las Fases 14, 14b y 17**: la cifra es comparable con todo lo anterior del proyecto. |

**Por qué estos dos y no otros.** El caso es tabular: 70 features numéricas continuas
y muy correlacionadas, y ~3700 filas que en realidad son 146 personas. Con tan pocos
datos independientes, una red neuronal tendría mucha más capacidad que información y
sus pesos no dan un signo interpretable. Se pedía un máximo de dos modelos, y la
elección es un par complementario:

- la **logística** es el modelo lineal, regularizado (L2 por defecto, que tolera bien
  la colinealidad) y de probabilidades razonables, y es el único cuyos coeficientes
  permiten comprobar la **dirección** del efecto, que es lo que pone a prueba los
  criterios (b), (c) y (d);
- el **Random Forest** cubre lo que la logística no puede, interacciones y
  no linealidades, apenas exige tipificar ni ajustar, y da la importancia por
  permutación. Que sea la configuración de las Fases 14, 14b y 17 hace la cifra
  comparable con todo lo anterior.

Su comparación es informativa por sí misma: si el bosque mejorara claramente a la
logística, habría estructura no lineal que un modelo lineal pierde. No es lo que pasa
(sección 5): **la señal es esencialmente lineal en features tipificadas**. Se
descartaron, por este orden, el gradient boosting (de la misma familia que el bosque,
con más hiperparámetros que ajustar y sin la ventaja de la comparabilidad), una SVM
con núcleo (sensible a la escala y sin coeficientes interpretables) y el análisis
discriminante lineal (redundante con la logística). Solo una configuración por
modelo, **sin ajustar hiperparámetros**: elegir la mejor de varias sobre estos datos
es el sesgo que el arnés de `modeling.py` existe para evitar.

**El desbalance se trata en dos sitios.** *En el entrenamiento*, con
`class_weight="balanced"`: cada clase pesa en la función de coste como si tuviera el
mismo número de grabaciones, así que la frontera no se desplaza hacia la mayoritaria
y `P(tarea) > 0.5` da el mismo coste a cada tipo de error. No se ajusta ningún umbral
de decisión. *En la evaluación*, con la **exactitud balanceada** como métrica
principal (su valor de azar es 0.5 exacto pase lo que pase con el reparto) y la
**precisión media de la clase reposo** como métrica de apoyo, porque es la clase
rara y la que la exactitud balanceada y el AUC no dejan ver; su valor de azar no es
0.5 sino la prevalencia, 0.2876.

Además, referencias que no usan X o solo una variable:

- **Baseline de la clase mayoritaria** y **baseline estratificado al azar**. Con la
  exactitud balanceada el azar es 0.5 aunque haya 2.5 tareas por reposo.
- **Umbral sobre una sola variable** (`frontal_theta_rel_media`, fijada por la
  teoría y no por los datos): un único corte con umbral **y signo** reestimados
  dentro de cada pliegue. El Epoc X no tiene Fz; la región frontal (AF3, AF4, F3,
  F4, F7, F8) es la más cercana al theta frontal medial.

**Validación.** `predecir_cv_agrupada` de la Fase 14, sin modificarla: 5 pliegues de
`GroupKFold` por `persona_uid`, con la comprobación de que ninguna persona cae a
ambos lados. Imputación y tipificación se ajustan **solo con el pliegue de
entrenamiento** (van dentro del `Pipeline`). Es la situación de un investigador que
pone el casco a alguien por primera vez, sin calibración. **Intervalos:** bootstrap
remuestreando **personas** (2000 réplicas, semilla 42), no grabaciones; las
comparaciones entre modelos son **pareadas**.

**Criterios de coherencia, declarados como constantes del guion antes de ejecutar.**
Más modestos que los de Berger, porque el efecto esperado lo es:

| | Criterio |
|---|---|
| (a) | la banda cuya permutación conjunta cuesta más exactitud es theta o alfa |
| (b) | el coeficiente neto de las features `_media` de **theta frontal** es positivo (hacia tarea) en los 5 pliegues, y el umbral de una variable aprende signo +1 en los 5 |
| (c) | el coeficiente neto de las features `_media` de **alfa posterior** es negativo (hacia reposo) en los 5 pliegues |
| (d) | (b) y (c) se siguen cumpliendo en el control sin OC |

El criterio (a) es **débil por construcción**: por azar lo cumpliría la mitad de las
veces (dos de cuatro bandas). Los informativos son (b), (c) y sobre todo (d).

### Qué se decidió antes de ver datos y qué después

Los criterios, los modelos, las métricas, el desglose por condición, los controles
de la sección 8 y las 10 filas de la sensibilidad estaban en la primera versión del
guion, antes de ninguna ejecución. Antes de la
ejecución completa se hicieron **tres pasadas en modo rápido** (`--rapido`:
parámetros mínimos, salidas fuera del repositorio) para comprobar que el guion corría
de principio a fin; sus cifras no se usan. **De esas pasadas se cambiaron solo** la
maquetación de las figuras y tres cosas de contenido, **las tres decididas después
de ver resultados provisionales**: (1) la segunda fila de la figura por canal
(potencia absoluta), tras ver que el theta relativo era mayor en posterior que en
frontal; (2) las columnas de potencia absoluta en la tabla por canal; (3) el
desglose por condición del control sin OC. Ninguna toca un criterio ni una cifra
principal.

## 5. Resultado — rendimiento

| Clasificador | Exactitud balanceada (media de pliegues) | IC95 | AUC | Precisión media de reposo | Sensibilidad (tarea) | Especificidad (reposo) |
|---|---|---|---|---|---|---|
| Baseline: clase mayoritaria | 0.5000 | [0.500, 0.500] | 0.5000 | 0.2876 | 1.000 | 0.000 |
| Baseline: azar estratificado | 0.4991 | [0.483, 0.515] | 0.4991 | 0.2872 | 0.716 | 0.282 |
| Umbral en `frontal_theta_rel_media` | 0.6344 | [0.617, 0.653] | — | — | 0.652 | 0.618 |
| Random Forest | 0.8313 | [0.812, 0.851] | 0.9319 | 0.8689 | 0.922 | 0.741 |
| **Regresión logística** | **0.8568** | **[0.839, 0.874]** | **0.9335** | **0.8813** | 0.880 | 0.834 |

- **IC95 del AUC:** logística [0.918, 0.944], Random Forest [0.916, 0.943]. **IC95 de
  la precisión media de reposo:** [0.859, 0.904] y [0.842, 0.896].
- **Amontonada frente a por pliegue:** 0.8567 / 0.8568 (logística), 0.8312 / 0.8313
  (RF), 0.6346 / 0.6344 (umbral). Coinciden, como deben: la etiqueta no se deriva de
  ningún corte estimado con los datos.
- **Por pliegue** (SD 0.015 / 0.023 / 0.028): logística 0.858, 0.836, 0.852, 0.860,
  0.878; RF 0.837, 0.803, 0.814, 0.844, 0.859; umbral 0.613, 0.602, 0.635, 0.649,
  0.673.
- **Matrices de confusión** (filas = real, columnas = predicha):

| | Logística | | Random Forest | | Umbral | |
|---|---|---|---|---|---|---|
| | pred reposo | pred tarea | pred reposo | pred tarea | pred reposo | pred tarea |
| **real reposo** | 894 | 178 | 794 | 278 | 662 | 410 |
| **real tarea** | 320 | 2336 | 208 | 2448 | 925 | 1731 |

### Comparaciones pareadas (exactitud balanceada)

| A − B | Diferencia | IC95 | A > B en |
|---|---|---|---|
| Logística − Random Forest | +0.0256 | [+0.0111, +0.0400] | 100 % de réplicas |
| Logística − umbral de una variable | +0.2221 | [+0.2001, +0.2444] | 100 % |
| Random Forest − umbral de una variable | +0.1965 | [+0.1724, +0.2223] | 100 % |
| Logística − azar estratificado | +0.3576 | [+0.3356, +0.3808] | 100 % |
| Random Forest − azar estratificado | +0.3321 | [+0.3081, +0.3559] | 100 % |
| Umbral − azar estratificado | +0.1355 | [+0.1149, +0.1556] | 100 % |

### Lectura

**Ambos modelos distinguen reposo de tarea muy por encima del azar en personas no
vistas.** Ningún IC se acerca a 0.5 y el arnés no produce acierto de la nada
(sección 8.1).

**El desbalance es real y está tratado.** El baseline mayoritario acierta el 71 %
sin saber nada, y su exactitud balanceada es 0.500. La logística tiene una precisión
media de reposo de 0.881 frente a un azar de 0.288: la clase rara no se sacrifica
para ganar acierto global.

**La logística supera al Random Forest en exactitud balanceada (+2.6 puntos, IC que
excluye 0), pero no en AUC** (0.9335 frente a 0.9319, con IC casi idénticos; no se
ha contrastado de forma pareada). El Random Forest tiene un punto de operación
sesgado hacia «tarea» (sensibilidad 0.922, especificidad 0.741) pese a los pesos, así
que parte de esa diferencia es de umbral y no de capacidad para separar. No se ha
ajustado el umbral porque no es el objetivo. Que un bosque, capaz de capturar
no linealidades, no mejore a la logística indica que la señal es esencialmente lineal
en features tipificadas.

**El modelo multivariante aporta mucho sobre theta frontal solo** (+22 puntos), pero
esa referencia se fijó en theta y **no** en alfa. Como el d univariante del alfa
posterior (−1.54) es mucho mayor que el del theta frontal (+0.67), un umbral sobre
alfa lo haría bastante mejor que 0.634. No se ha medido: «aporta 22 puntos» no
significa «aporta 22 sobre la mejor variable única».

**El tope es ~0.86, no ~0.99, y el error no es uniforme:** casi todo el fallo está
en dos sitios, el reposo con ojos abiertos y la tarea PVTB (sección 7).

## 6. Coherencia neurofisiológica — qué pesa en el modelo

### 6.1 Importancia por permutación (Random Forest, out-of-fold)

Caída de la exactitud balanceada **de validación** al barajar la columna (5 barajados
× 5 pliegues). Es la importancia de lo que el modelo usa para generalizar a gente
nueva, no la de Gini de un `fit` sobre todos los datos.

| Feature | Caída | d de Cohen (tarea − reposo) | Coef. logística | Pliegues con coef > 0 |
|---|---|---|---|---|
| `posterior_alfa_rel_sd` | +0.0177 | −1.62 | −0.856 | 0/5 |
| `theta_frontal_log_media` | +0.0060 | +0.63 | +0.567 | 5/5 |
| `frontal_theta_logabs_media` | +0.0055 | +0.63 | +0.567 | 5/5 |
| `indice_fatiga_media` | +0.0050 | +0.68 | −0.540 | 0/5 |
| `posterior_beta_rel_media` | +0.0048 | +0.65 | +0.068 | 4/5 |
| `posterior_beta_logabs_media` | +0.0047 | +0.40 | +2.040 | 5/5 |
| `frontal_theta_rel_media` | +0.0028 | +0.67 | +0.308 | 5/5 |

Las cifras individuales son pequeñas porque el modelo es redundante: barajar una
columna deja las demás cubriéndola. **No hay que leer estos coeficientes como pesos
independientes**: `theta_frontal_log_media` y `frontal_theta_logabs_media` son la
misma columna, y hay signos que se invierten respecto al efecto univariante
(`indice_fatiga_media`: d = +0.68 y coeficiente −0.540; `frontal_alfa_logabs_media`:
d = −0.38 y coeficiente +1.072, 5/5 pliegues), que es lo que hace una regresión
regularizada con features correlacionadas.

Los **mayores tamaños de efecto univariantes** son todos de alfa: `posterior_alfa_rel_sd`
(−1.62), `posterior_alfa_rel_media` (−1.54), `frontal_alfa_rel_sd` (−1.53),
`frontocentral_alfa_rel_sd` (−1.36), `frontal_alfa_rel_media` (−1.33), y así hasta
diez. El mayor efecto positivo es `posterior_theta_rel_media` (+0.80), y la sección 9
explica por qué esa cifra engaña.

### 6.2 Agregación por bloque (fracción de la importancia total)

| Por banda | | Por región | |
|---|---|---|---|
| **alfa** | **32.8 %** | **posterior** | **40.9 %** |
| theta | 28.4 % | frontal | 31.4 % |
| beta | 21.5 % | frontocentral | 12.3 % |
| delta | 10.4 % | temporal | 8.5 % |
| sin asignar | 6.9 % | sin asignar | 6.9 % |

Por bloque: posterior/alfa 22.3 %, frontal/theta 18.0 %, posterior/beta 13.3 %,
frontal/alfa 6.0 %. «Sin asignar» (6.9 %) es `indice_fatiga_*`, el cociente
theta/alfa.

Esta agregación **subestima la ventaja del alfa** (32.8 % frente a 28.4 %) porque
suma importancias individuales que se diluyen entre features casi duplicadas. La
permutación conjunta, que no se diluye, da una imagen mucho más clara:

### 6.3 Permutación conjunta de bloques enteros

Barajar de una vez todas las columnas de un bloque, con la misma permutación.

| Bloque | Features | Caída de exactitud balanceada | SD entre pliegues |
|---|---|---|---|
| **banda alfa** | 18 | **+0.2344** | 0.013 |
| región posterior | 18 | +0.1765 | 0.017 |
| posterior ∩ alfa | 6 | +0.0968 | 0.009 |
| banda theta | 18 | +0.0369 | 0.014 |
| región frontal | 18 | +0.0340 | 0.014 |
| banda beta | 16 | +0.0220 | 0.010 |
| frontal ∩ theta | 6 | +0.0166 | 0.014 |
| región temporal | 16 | +0.0074 | 0.010 |
| banda delta | 16 | +0.0050 | 0.010 |
| región frontocentral | 16 | +0.0024 | 0.006 |

**Alfa cuesta 23 puntos, seis veces más que theta (3.7).** El modelo depende
sobre todo del alfa y de la región posterior; theta es la segunda banda, con un peso
modesto.

### 6.4 Dirección del efecto

- Coeficiente neto (suma, en unidades de desviación típica) de las tres features
  `_media` de **theta frontal** en la logística, por pliegue: **+1.330, +1.519,
  +1.418, +1.395, +1.552**. Positivo en los cinco.
- Coeficiente neto de las tres de **alfa posterior**: **−2.702, −3.502, −2.897,
  −2.799, −2.701**. Negativo en los cinco.
- Signo del umbral de una variable sobre theta frontal: **+1, +1, +1, +1, +1** (con
  corte en 0.1405–0.1549; sobre todos los datos 0.1501).

Theta frontal empuja hacia tarea y alfa posterior hacia reposo, en todos los
pliegues y por dos vías independientes. Coincide con la teoría.

### 6.5 Veredicto

| | Criterio | Resultado |
|---|---|---|
| SÍ | (a) la banda que más cuesta al barajarla entera es theta o alfa | alfa (+0.234) |
| SÍ | (b) theta frontal hacia tarea en los 5 pliegues (coef. y umbral) | coef. + / umbral + |
| SÍ | (c) alfa posterior hacia reposo en los 5 pliegues (coef.) | coef. − |
| SÍ | (d) (b) y (c) se mantienen sin OC (sección 8.2) | (b) SÍ / (c) SÍ |

**Se cumplen 4 de 4.** Con dos salvedades que hay que citar con ellos: el (a) es
débil por construcción (lo cumpliría la mitad de los resultados posibles), y el
resultado no es «theta y alfa por igual»: es alfa muy por delante de theta.

## 7. Errores por condición original

Un acierto global esconde qué falla. Aquí cada condición tiene su propia tasa, con
IC95 remuestreando personas. Para una **tarea** el acierto es la fracción leída
como tarea; para **OA y OC** es la fracción leída como reposo.

| Posición | Condición | Clase | n | Logística: acierto [IC95] | Random Forest: acierto [IC95] |
|---|---|---|---|---|---|
| 1 | OC | reposo | 541 | **0.945** [0.923, 0.965] | 0.887 [0.846, 0.924] |
| 2 | OA | reposo | 531 | **0.721** [0.668, 0.777] | 0.591 [0.531, 0.653] |
| 3 | STROOP | tarea | 534 | 0.919 [0.889, 0.947] | 0.968 [0.947, 0.987] |
| 4 | PASAT | tarea | 533 | 0.916 [0.879, 0.948] | 0.940 [0.913, 0.963] |
| 5 | PVTB | tarea | 531 | **0.676** [0.617, 0.733] | 0.763 [0.709, 0.817] |
| 6 | MSIT | tarea | 530 | 0.947 [0.924, 0.968] | 0.975 [0.960, 0.989] |
| 7 | PVSAT | tarea | 528 | 0.939 [0.912, 0.964] | 0.962 [0.937, 0.982] |

Probabilidad media de «tarea» asignada por el modelo: logística OC 0.090, OA 0.311,
STROOP 0.828, PASAT 0.824, PVTB 0.611, MSIT 0.882, PVSAT 0.854; Random Forest OC
0.168, OA 0.439, STROOP 0.848, PASAT 0.824, PVTB 0.669, MSIT 0.880, PVSAT 0.850.

### ¿Fallan por igual OA y OC? No

Diferencia de acierto OA − OC, pareada por persona (mismas personas en las dos
ramas de cada réplica):

| Modelo | OA | OC | OA − OC | IC95 | OA > OC en |
|---|---|---|---|---|---|
| Regresión logística | 0.721 | 0.945 | **−0.223** | [−0.274, −0.175] | 0.0 % de réplicas |
| Random Forest | 0.591 | 0.887 | **−0.296** | [−0.351, −0.245] | 0.0 % |

**OC se reconoce como reposo casi siempre; OA se confunde con una tarea el 28 % de
las veces con la logística y el 41 % con el Random Forest.** Es lo que predice el
efecto Berger: con los ojos cerrados el alfa es tan alto que ninguna tarea se le
parece, y con los ojos abiertos el reposo se parece más a una tarea. Consecuencia
para el titular: el 0.857 **mezcla un reposo fácil (OC) con uno difícil (OA)**, y
por eso la sección 8.2 mide qué pasa sin OC.

### ¿Qué tareas se confunden con reposo?

Fracción de cada tarea leída como reposo, de más a menos: logística **PVTB 32.4 %**
> PASAT 8.4 % > STROOP 8.1 % > PVSAT 6.1 % > MSIT 5.3 %; Random Forest **PVTB 23.7 %**
> PASAT 6.0 % > PVSAT 3.8 % > STROOP 3.2 % > MSIT 2.5 %.

**PVTB es la única tarea que el modelo no reconoce bien**; las otras cuatro se
reconocen entre el 91.6 % y el 97.5 % según el modelo. También es la de menor probabilidad media de
«tarea», así que no es un efecto del umbral. Qué hay detrás **no se ha investigado**.
El proyecto solo la lista como una de las cinco condiciones de carga cognitiva. Una
hipótesis razonable es que exija menos control ejecutivo que las demás (no se ha
comprobado la demanda de cada tarea); no es un asunto de duración, porque PVTB dura
lo mismo que PASAT y PVSAT (~187 s) y los contadores de duración no están en X.

## 8. Controles y sensibilidad

### 8.1 Control negativo: etiquetas barajadas dentro de cada persona

Destruye la relación entre las features y la condición pero conserva las personas y
cuántas grabaciones de cada clase tiene cada una (20 barajados por modelo).

| Modelo | Exactitud balanceada con etiquetas barajadas | Real |
|---|---|---|
| Logística | 0.4949 (SD 0.0106; rango 0.471–0.513) | 0.8568 |
| Random Forest | 0.5005 (SD 0.0021; rango 0.498–0.505) | 0.8313 |

Ninguno de los 20 nulos alcanza el valor real. **El arnés no produce acierto de la
nada.** La media de la logística queda algo por debajo de 0.5, dentro del criterio
declarado (±0.03). Con 20 barajados el p empírico no puede bajar de 1/21 = 0.048: es
la resolución del control, no un valor que leer.

### 8.2 Sin OC: ¿es solo el efecto Berger?

Se repite el análisis con **una sola** de las dos condiciones de reposo. Si con OA
solo el acierto se desploma hacia 0.5, el modelo sabía de ojos cerrados y no de
carga cognitiva.

| Contraste | Grabaciones | Desbalance | Logística: BA [IC95] | AUC (logística) | Random Forest: BA [IC95] |
|---|---|---|---|---|---|
| Principal: OA + OC frente a tareas | 3728 | 2.48 : 1 | 0.8568 [0.839, 0.874] | 0.9335 | 0.8313 [0.812, 0.851] |
| **OA solo frente a tareas** | 3187 (531 + 2656) | 5.00 : 1 | **0.8163** [0.796, 0.837] | 0.8968 [0.880, 0.914] | **0.7378** [0.712, 0.764] |
| OC solo frente a tareas | 3197 (541 + 2656) | 4.91 : 1 | 0.9269 [0.910, 0.943] | 0.9734 [0.962, 0.982] | 0.8936 [0.867, 0.920] |

Precisión media de reposo (logística): OA solo 0.683 frente a un azar de 0.167; OC
solo 0.936 frente a 0.169.

**Dirección del efecto en los controles:** con OA solo, coeficiente neto de theta
frontal +0.788, +0.240, +0.391, +0.939, +0.661 y de alfa posterior −0.959, −0.682,
−0.999, −0.891, −0.852, umbral con signo +1 en los cinco: **(b) y (c) se mantienen**,
y por tanto se cumple el criterio (d). Con OC solo también (theta +3.310 a +4.256,
alfa −4.426 a −5.788).

Acierto por condición en el control con OA solo (logística / Random Forest): OA
0.802 / 0.539, STROOP 0.899 / 0.978, PASAT 0.876 / 0.957, **PVTB 0.552 / 0.783**,
MSIT 0.930 / 0.989, PVSAT 0.896 / 0.979.

**Lectura.** Sin OC el acierto baja (−4.1 puntos con la logística, −9.3 con el
Random Forest), pero **sigue muy por encima del azar**: existe una firma de tarea
que no es solo cerrar los ojos. Con OC solo, en cambio, se llega a 0.927: OC es lo
que empuja hacia arriba el resultado agregado. Con OA solo, el Random Forest acierta
la propia OA el 54 % de las veces, cerca de una moneda, y la logística el 80 %; PVTB
cae al 55 % con la logística, es decir, **con OA como reposo casi la mitad de PVTB se
confunde con reposo**. Estas comparaciones **no son pareadas** (son filas distintas y
un desbalance de 5 : 1), así que las diferencias con el principal son indicativas.

### 8.3 Posición en el protocolo

El orden es fijo: el reposo ocupa las posiciones 1–2 y las tareas las 3–7. Cualquier
deriva sistemática dentro de la sesión (el gel secándose, el participante
acomodándose, somnolencia creciente) queda confundida con la condición y **este
diseño no puede separarlas**. Dos comprobaciones parciales, ninguna concluyente.

**(i) Entre las tareas, ¿crece el acierto con la posición (3 a 7)?** Correlación de
Spearman a nivel de grabación, con IC95 remuestreando personas (con solo cinco
condiciones agregadas no habría potencia):

| Modelo | Posición ~ acierto | Posición ~ P(tarea) |
|---|---|---|
| Logística | ρ = +0.030 [+0.005, +0.054] | ρ = +0.104 [+0.076, +0.131] |
| Random Forest | ρ = +0.011 [−0.010, +0.034] | ρ = +0.072 [+0.045, +0.099] |

No hay una tendencia clara del acierto con la posición (el IC del Random Forest
incluye 0 y el de la logística apenas lo excluye). La probabilidad media de «tarea»
sí sube un poco (ρ ≈ 0.07–0.10, IC que excluye 0), pero es pequeña y no es una
deriva limpia: PVTB (posición 5) es la peor reconocida y MSIT (posición 6) la mejor,
así que domina *qué tarea es*, no *cuándo ocurre*.

**(ii) El par adyacente OA (posición 2) frente a STROOP (posición 3)**: el contraste
con menos separación temporal posible dentro de la sesión, con clases equilibradas
(531 OA y 534 STROOP, 145 personas).

| Modelo | Exactitud balanceada [IC95] | AUC [IC95] |
|---|---|---|
| Logística | 0.8706 [0.847, 0.895] | 0.9483 [0.932, 0.960] |
| Random Forest | 0.8493 [0.826, 0.872] | 0.9328 [0.913, 0.947] |

Dos grabaciones consecutivas se separan con 0.87. Una deriva lenta de la sesión
explicaría mal una separación tan grande entre dos registros contiguos, pero un
efecto de *cambio de estado al empezar la tarea* no se distingue de un efecto de
*tarea*, y el diseño no permite separarlos. Además, STROOP es de las tareas mejor
reconocidas (sección 7): este par no es representativo de las cinco.

### 8.4 Sensibilidad: qué features entran a X

Δ es la diferencia **pareada** frente al principal (misma remuestra de personas en
las dos ramas), con su IC95.

| Subconjunto | n | Modelo | Exactitud bal. | IC95 | AUC | Δ vs principal |
|---|---|---|---|---|---|---|
| **Principal** | 70 | Logística | 0.8568 | [0.839, 0.874] | 0.934 | — |
| | | Random Forest | 0.8313 | [0.812, 0.851] | 0.932 | — |
| Sin las 4 copias exactas | 66 | Logística | 0.8556 | [0.838, 0.873] | 0.934 | −0.0012 [−0.003, +0.000] |
| | | Random Forest | 0.8348 | [0.815, 0.854] | 0.930 | +0.0035 [−0.002, +0.010] |
| + 6 contadores de procesamiento | 76 | Logística | 0.8667 | [0.851, 0.882] | 0.943 | +0.0100 [+0.003, +0.017] |
| | | Random Forest | 0.9440 | [0.932, 0.956] | 0.994 | +0.1128 [+0.099, +0.127] |
| **Solo** los 6 contadores | 6 | Logística | 0.6238 | [0.605, 0.642] | 0.644 | −0.2329 [−0.255, −0.211] |
| | | Random Forest | **0.9902** | [0.986, 0.994] | 0.994 | +0.1590 [+0.139, +0.178] |
| Solo región frontal | 18 | Logística | 0.8187 | [0.800, 0.838] | 0.896 | −0.0381 [−0.054, −0.022] |
| | | Random Forest | 0.8007 | [0.781, 0.822] | 0.895 | −0.0305 [−0.042, −0.019] |
| Solo región posterior | 18 | Logística | 0.8349 | [0.815, 0.855] | 0.915 | −0.0219 [−0.034, −0.009] |
| | | Random Forest | 0.8274 | [0.808, 0.847] | 0.920 | −0.0038 [−0.015, +0.007] |
| Sin theta ni índice | 50 | Logística | 0.8582 | [0.841, 0.875] | 0.933 | +0.0014 [−0.005, +0.008] |
| | | Random Forest | 0.8224 | [0.801, 0.842] | 0.925 | −0.0089 [−0.018, +0.000] |
| Sin alfa ni índice | 50 | Logística | 0.8562 | [0.839, 0.873] | 0.931 | −0.0006 [−0.007, +0.006] |
| | | Random Forest | 0.8226 | [0.802, 0.842] | 0.919 | −0.0086 [−0.022, +0.004] |
| Sin theta, índice ni potencia relativa | 26 | Logística | 0.8112 | [0.792, 0.829] | 0.895 | −0.0456 [−0.062, −0.029] |
| | | Random Forest | 0.7969 | [0.776, 0.817] | 0.901 | −0.0344 [−0.049, −0.020] |
| Sin alfa, índice ni potencia relativa | 26 | Logística | 0.7402 | [0.722, 0.757] | 0.812 | −0.1166 [−0.134, −0.100] |
| | | Random Forest | 0.7100 | [0.690, 0.728] | 0.811 | −0.1213 [−0.144, −0.098] |

Lo que se lee:

- **Las cuatro copias exactas no cambian nada** (los IC de Δ incluyen 0).
- **Los contadores son un atajo enorme**: sección 3. Excluirlos no cuesta acierto de
  EEG; incluirlos lo inflaría con algo que no es señal cerebral.
- **Cada región sola casi basta.** Solo posterior: −2.2 puntos (logística) y una
  diferencia indistinguible de 0 con el Random Forest; solo frontal: −3.8 y −3.1.
  La información está repartida por el cuero cabelludo (el alfa cae en todos los
  canales, sección 9); ninguna región es imprescindible.
- **Quitar una banda dejando las potencias relativas no la quita**: «sin theta» y
  «sin alfa» apenas cambian nada (Δ entre −0.009 y +0.001). No es un colapso ni una
  prueba de que la banda no importe: las cuatro potencias relativas de cada región
  suman 1, así que la banda quitada se reconstruye de las demás (la misma lección de
  la Fase 17).
- **Las ablaciones limpias**, que quitan también la potencia relativa, sí cuestan, y
  cuestan más quitando alfa: **−11.7 puntos** (logística) y −12.1 (Random Forest)
  sin alfa, frente a **−4.6** y −3.4 sin theta. La información de alfa pesa unas
  2.5–3.5 veces la de theta.
- **Sin alfa, índice ni relativas el modelo aún acierta 0.74 (logística) y 0.71
  (Random Forest)**, por encima del azar: la potencia absoluta de delta, theta y beta
  sigue separando. Con estas features no se puede distinguir un efecto de banda ancha
  (actividad muscular, ocular residual, movimiento) de una firma cortical de tarea.

## 9. Resolución por canal (suplemento)

`matriz_rf.parquet` agrega por región. Para responder qué **canales** llevan el
efecto, el guion reagrega `matriz_modelado.parquet` con `construir_matriz_rf` y una
región por canal (la variante que documenta `agregar_por_region`), restringida a las
mismas 3728 grabaciones y leída ya filtrada, de modo que las épocas de las
grabaciones de test no llegan a cargarse.

**Potencia relativa** (d de Cohen, tarea − reposo; IC95 remuestreando personas):

| Canal | Grab. válidas | d theta rel. | IC95 | d alfa rel. | IC95 | d delta rel. | d beta rel. | Exactitud bal. (1 canal) | AUC |
|---|---|---|---|---|---|---|---|---|---|
| **O1** | 3431 | +0.98 | [+0.89, +1.09] | **−1.59** | [−1.73, −1.47] | +0.67 | +0.52 | 0.793 | 0.866 |
| **O2** | 3461 | +0.91 | [+0.81, +1.02] | −1.49 | [−1.62, −1.38] | +0.50 | +0.67 | 0.790 | 0.868 |
| AF3 | 2786 | +0.65 | [+0.56, +0.75] | −1.24 | [−1.36, −1.14] | +0.53 | +0.02 | 0.782 | 0.850 |
| F3 | 3527 | +0.57 | [+0.49, +0.65] | −1.26 | [−1.37, −1.15] | +0.52 | +0.03 | 0.766 | 0.836 |
| AF4 | 2536 | +0.56 | [+0.47, +0.66] | −1.17 | [−1.27, −1.07] | +0.55 | −0.05 | 0.783 | 0.848 |
| P7 | 3346 | +0.50 | [+0.42, +0.59] | −1.34 | [−1.47, −1.23] | +0.42 | +0.46 | 0.762 | 0.840 |
| F4 | 3469 | +0.50 | [+0.43, +0.59] | −1.17 | [−1.29, −1.06] | +0.50 | −0.06 | 0.758 | 0.830 |
| P8 | 3421 | +0.46 | [+0.37, +0.55] | −1.25 | [−1.37, −1.15] | +0.33 | +0.58 | 0.741 | 0.811 |
| F8 | 2835 | +0.41 | [+0.33, +0.49] | −1.14 | [−1.25, −1.03] | +0.50 | +0.03 | 0.742 | 0.828 |
| F7 | 3040 | +0.40 | [+0.32, +0.50] | −1.19 | [−1.31, −1.08] | +0.51 | +0.08 | 0.739 | 0.826 |
| FC5 | 3446 | +0.36 | [+0.29, +0.45] | −1.19 | [−1.30, −1.08] | +0.53 | +0.03 | 0.735 | 0.817 |
| T7 | 3252 | +0.33 | [+0.25, +0.42] | −1.04 | [−1.16, −0.92] | +0.25 | +0.17 | 0.726 | 0.791 |
| T8 | 3247 | +0.27 | [+0.18, +0.36] | −1.09 | [−1.21, −0.99] | +0.20 | +0.26 | 0.719 | 0.798 |
| FC6 | 3433 | +0.24 | [+0.17, +0.32] | −1.15 | [−1.25, −1.04] | +0.56 | −0.06 | 0.730 | 0.807 |

**Potencia absoluta** (log₁₀), con la que se separa el efecto real del composicional:

| Canal | d theta abs. | IC95 | d alfa abs. | IC95 |
|---|---|---|---|---|
| **F4** | **+0.59** | [+0.51, +0.67] | −0.35 | [−0.44, −0.26] |
| F7 | +0.52 | [+0.45, +0.59] | −0.17 | [−0.26, −0.09] |
| F8 | +0.51 | [+0.44, +0.59] | −0.17 | [−0.26, −0.08] |
| FC5 | +0.51 | [+0.43, +0.59] | −0.33 | [−0.43, −0.23] |
| FC6 | +0.43 | [+0.34, +0.52] | −0.30 | [−0.39, −0.21] |
| F3 | +0.43 | [+0.33, +0.53] | −0.58 | [−0.69, −0.48] |
| AF3 | +0.37 | [+0.29, +0.45] | −0.46 | [−0.56, −0.36] |
| T7 | +0.36 | [+0.29, +0.43] | −0.41 | [−0.51, −0.33] |
| T8 | +0.34 | [+0.27, +0.41] | −0.44 | [−0.55, −0.35] |
| AF4 | +0.34 | [+0.25, +0.42] | −0.41 | [−0.53, −0.32] |
| O2 | +0.25 | [+0.17, +0.34] | −0.84 | [−0.96, −0.73] |
| O1 | +0.20 | [+0.12, +0.28] | **−1.04** | [−1.18, −0.91] |
| P8 | +0.17 | [+0.10, +0.24] | −0.71 | [−0.82, −0.61] |
| P7 | +0.16 | [+0.07, +0.25] | −0.76 | [−0.88, −0.65] |

(«Exactitud bal. (1 canal)»: logística con solo theta y alfa, relativos y absolutos,
de ese canal, con validación agrupada por persona, sobre las grabaciones donde el
canal superó el control de calidad.)

- **El alfa baja en todos los canales** (relativo: entre −1.04 y −1.59). La mayor
  caída está en O1 y O2 y, **en potencia absoluta, el gradiente posterior es
  nítido**: O1 −1.04, O2 −0.84, P7 −0.76, P8 −0.71 frente a −0.17 (F7, F8) a −0.58
  (F3) en los frontales, con IC disjuntos (O1 [−1.18, −0.91] frente a F3 [−0.69,
  −0.48]).
- **El theta relativo engaña.** Es mayor en O1 y O2 (+0.98, +0.91) que en cualquier
  canal frontal (máximo +0.65). Pero la potencia relativa suma 1 entre bandas: cuando
  el alfa cae mucho en posterior, las demás suben aunque no cambien. **En potencia
  absoluta el orden se invierte**: el aumento de theta es mayor en los canales
  frontales y frontocentrales (F4 +0.59, F7 +0.52, F8 +0.51, FC5 +0.51, con IC
  disjuntos de los de los cuatro posteriores, cuyo máximo es O2 [+0.17, +0.34]) y
  menor en los posteriores (+0.16 a +0.25). Es el patrón que la teoría predice para
  el theta de carga ejecutiva, **con las reservas de la sección 12** (sin Fz; los
  canales frontales son los más expuestos a la actividad ocular).
- **Un solo canal ya da 0.72–0.79** con solo su theta y su alfa, frente a 0.857 del
  modelo completo. Los de mayor acierto son O1 (0.793), O2 (0.790), AF4 (0.783) y AF3
  (0.782): los frontales anteriores compiten con los occipitales.
- **El número de grabaciones válidas varía por canal** (2536 en AF4 a 3527 en F3 de
  3728), porque se usan solo las que superaron la Fase 11.5.
- Diferencias de 0.1–0.2 entre canales vecinos **no** son distinguibles.

## 10. Las figuras

### `figures/reposo_tarea_confusion.png`

Las dos matrices de confusión, lado a lado y en la misma escala. El color es el
**porcentaje de cada clase real** (cada fila suma 100 %), no el recuento: así el
desbalance de 2.5 : 1 no cambia el color de las celdas. La diagonal es oscura y los
errores claros. El error más visible es «real reposo, predicha tarea» en el Random
Forest (25.1 %).

### `figures/reposo_tarea_por_condicion.png`

La tasa de acierto de cada condición original, en el orden del protocolo (de arriba
abajo, 1 = OC … 7 = PVSAT), con su IC95 remuestreando personas. Azul = reposo,
naranja = tarea. Es la figura que muestra de un vistazo los dos huecos de la sección
7: OA por debajo de OC y PVTB por debajo de las demás tareas.

### `figures/reposo_tarea_canales.png`

El d de Cohen de theta y alfa por canal, en potencia relativa (fila de arriba) y
absoluta (fila de abajo), con IC95 remuestreando personas. En naranja, los canales
de la región donde la teoría espera el efecto (theta: frontal; alfa: posterior). La
segunda fila es necesaria: la relativa suma 1 entre bandas y, sola, haría parecer
que el theta es posterior.

Paleta: azul = reposo, naranja = tarea (la clase positiva, como el naranja era OC en
la Fase 17), validada con el validador de la guía de visualización en **modo claro**
(ΔE CVD 24.7, visión normal 33.6, contraste ≥ 3 : 1). En modo oscuro el mismo
validador da fallo en la banda de luminosidad del naranja, así que estas figuras se
generan solo sobre fondo blanco.

## 11. Conclusión

**El pipeline conserva una señal que separa reposo de tarea ejecutiva.** Sobre
personas no vistas, la regresión logística la separa con una exactitud balanceada de
**0.857** (IC95 [0.839, 0.874]; AUC 0.934) y el Random Forest con la configuración de
las Fases 14, 14b y 17 con **0.831** [0.812, 0.851]; el azar es 0.500 y el control con
etiquetas barajadas da 0.495 y 0.501. Lo que pesa es lo que la neurofisiología
predice, con una jerarquía clara: el **descenso del alfa** (d hasta −1.6, en todos
los canales y más marcado en posterior) explica la mayor parte, y el **aumento de
theta**, más modesto, es frontal y frontocentral en potencia absoluta. La dirección
es estable en los cinco pliegues y en los dos controles.

**No es solo el efecto Berger.** Sin OC dentro del reposo, la logística sigue en 0.816
[0.796, 0.837] y los criterios de dirección se mantienen. Con OC como único reposo
sube a 0.927: cerrar los ojos infla el resultado agregado, pero no lo explica.

Para la memoria del TFG, esto **refuerza el diagnóstico de la Fase 17 con un efecto
más cercano al de interés**: los resultados negativos sobre la fatiga percibida (Fases
14, 14b y 16) no se explican porque el procesamiento destruya la señal espectral de
carga cognitiva, un efecto de tamaño moderado a grande.

### Alcance de la afirmación

- **Sí establece** que un efecto de tamaño grande a moderado a nivel de grabación
  (alfa d ≈ −1.5; theta absoluto frontal d ≈ +0.6) sobrevive a las Fases 1–13 y se
  recupera en personas nuevas, y que no depende solo de tener los ojos cerrados.
- **No establece** que esa señal sea una firma de *carga cognitiva*: el orden fijo del
  protocolo la confunde con la posición en la sesión, y la actividad ocular, muscular
  y motora propias de hacer una tarea (mirar una pantalla, responder) no se pueden
  separar con estas features.
- **No establece** que el pipeline conserve la señal de **fatiga**. Aquí los efectos
  son grandes (d ≈ 1.5 en alfa); la fatiga percibida, si está en la señal, es mucho
  más sutil, y este resultado no dice si un efecto de ese tamaño sobreviviría.
- **No establece** nada sobre la señal cruda de la ruta CNN.
- **No es confirmatorio.** Se ejecutó sobre `train` + `val`. Al ser la etiqueta la
  condición y no el VAS, **no** consume el alfa del paquete de test pendiente, que es
  sobre la fatiga; las configuraciones exploradas aquí no se suman a ese recuento. El
  test sigue sin abrirse.

## 12. Limitaciones

- **El orden del protocolo es fijo: el reposo siempre precede a las tareas** (OC →
  OA → STROOP → PASAT → PVTB → MSIT → PVSAT). La condición y el tiempo acumulado en
  la sesión están perfectamente confundidos (`ESTADO.md`), y este diseño **no puede
  separarlos**. Es la limitación más seria y no tiene solución con estos datos. La
  sección 8.3 la acota: no hay una tendencia limpia del acierto con la posición, y el
  par adyacente OA–STROOP se separa con 0.87. Ninguna de las dos la descarta.
- **El reposo incluye OC.** Medido en la sección 8.2, pero el resultado agregado
  mezcla un reposo fácil con uno difícil.
- **Las cinco tareas no son homogéneas y «reposo» tampoco.** PVTB se comporta distinto
  de las demás (sección 7) y no se ha investigado por qué. Etiquetar las cinco como
  una sola clase da una cifra media que no describe a ninguna.
- **Artefactos oculares, musculares y de movimiento.** Hacer una tarea implica mirar
  una pantalla y responder. La ICA con ICLabel de la Fase 6 trata los componentes
  oculares, pero la eliminación no es perfecta y no hay tratamiento para el resto. Los
  canales frontales, donde se ve el theta, son los más expuestos. Que sin alfa ni
  potencia relativa el modelo aún acierte 0.71–0.74 es compatible con un efecto de
  banda ancha de ese tipo.
- **Los contadores de procesamiento son un atajo enorme** (0.990 con solo ellos y el
  Random Forest) y están fuera de X. Tampoco se ha comprobado que las features `_sd`
  no arrastren indirectamente la duración de la grabación.
- **Sin Fz y con 14 electrodos.** El theta frontal medial se aproxima con la región
  de seis canales frontales; las regiones de la Fase 13 son gruesas.
- **Umbral de decisión sin ajustar.** El punto de operación del Random Forest está
  sesgado hacia «tarea», y su exactitud balanceada infravalora su capacidad de
  separar (AUC prácticamente igual al de la logística).
- **Una configuración por modelo, sin ajustar.** El Random Forest por debajo de la
  logística no es una afirmación sobre lo que un Random Forest ajustado puede hacer.
- **Importancia por permutación con features correlacionadas y duplicadas** infravalora
  cada una por separado; por eso se añade la permutación conjunta. El control con
  etiquetas barajadas tiene una resolución de p ≥ 0.048.
- **Los controles de la sección 8.2 no son pareados** con el principal (filas
  distintas, desbalance de 5 : 1).
- **Cohorte y montaje:** 146 personas de un estudio de la Facultad de Ciencias del
  Deporte (edad y demografía no confirmadas), Emotiv Epoc X, ventanas de 4 s agregadas
  por grabación. Los resultados pueden no trasladarse a otro montaje o población.

## 13. Lo que **no** hay que hacer

- **Usar `columnas_de_features()` tal cual.** Devuelve `condicion_orden`, que es la
  etiqueta: da 100 % sin significar nada. Usar `columnas_de_features_tarea()`.
- **Reincorporar los contadores de procesamiento a X.** Con el Random Forest, el
  acierto sube a 0.944 (y a 0.990 con ellos solos) sin que haya una sola feature de
  EEG detrás.
- **Citar 0.857 como «el pipeline detecta carga cognitiva»** sin la salvedad del orden
  fijo del protocolo y de la actividad ocular y motora.
- **Leer 0.857 como la sensibilidad del pipeline a la fatiga.** Es la de un efecto con
  d ≈ 1.5.
- **Citar el criterio (a) como evidencia fuerte.** Lo cumple la mitad de los resultados
  posibles; los informativos son (b), (c), (d) y la permutación conjunta.
- **Leer la diferencia logística − Random Forest como diferencia de capacidad.** Los
  AUC son casi iguales; la diferencia es de punto de operación.
- **Leer las importancias o los coeficientes individuales como pesos separados.** Hay
  copias exactas y signos que se invierten por la colinealidad.
- **Presentar «sin theta» o «sin alfa» (dejando las relativas) como «el modelo no
  depende de esa banda».** Las relativas suman 1 y la reconstruyen.
- **Leer el theta relativo por canal sin la potencia absoluta.** Es composicional.
- **Comparar la sección 8.2 con el principal como si fuera una diferencia pareada.**
- **Contar las 3728 grabaciones como n independiente.** Son 146 personas.
- **Leer el p = 0.048 del control como un hallazgo.** Es el mínimo alcanzable con 20
  barajados.
- **Ajustar el modelo o el umbral a este problema para subir la cifra.** No es el
  objetivo, y cada configuración adicional engorda el recuento.
- **Abrir el test.** `preparar_reposo_tarea` se niega salvo `permitir_test=True`.

## 14. Reproducir

```bash
# desde la raíz del repositorio; una sola vez y nunca en paralelo (la RAM es limitada)
python analisis/analisis_reposo_tarea.py > analisis/salidas/reposo_tarea.log 2>&1
```

Duración: **15 min 11 s** en la ejecución que produjo estos resultados. Semilla fija
(`SEMILLA = 42`) en los modelos, las permutaciones y todos los bootstraps.

**Reproducción comprobada el 2026-10-06.** Se repitió la ejecución completa en otra
máquina, con las mismas versiones y regenerando desde cero las etiquetas y
`matriz_rf.parquet` (15 min 43 s). El log coincide línea a línea con el original salvo
las rutas de los archivos, y las tres figuras son idénticas byte a byte. El JSON no se
pudo comparar: el original se perdió y este es el regenerado.

`--rapido` ejecuta el mismo código con parámetros mínimos (40 réplicas de bootstrap,
30 árboles, 2 barajados) y escribe en un directorio temporal, en unos 40 s. Sirve para
comprobar que el guion corre y revisar las figuras; **sus cifras no son resultados**.

Además de `matriz_rf.parquet`, el suplemento por canal necesita
`<EEG_DATA>/processed/matriz_modelado.parquet` y
`<EEG_DATA>/metadata/labels_grabaciones.csv`. El guion aparta el `test` en su primer
paso, `preparar_reposo_tarea` se niega a incluirlo sin permiso explícito, y el
suplemento por canal lee el parquet de épocas ya filtrado a las grabaciones de
análisis.

`*.log` y `figures/` están en el `.gitignore` del repositorio (igual que los de la
Fase 17): el log y las figuras existen en disco pero no se versionan salvo que se
fuerce.

Versiones con las que se generaron estos resultados: Python 3.10.11, scikit-learn
1.7.2, numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, matplotlib 3.10.9.

## 15. Documentos relacionados

| Documento | Qué cubre |
|---|---|
| `MODELADO.md`, Fase 18 | Este análisis, resumido en el cuaderno de modelado |
| `ANALISIS_OA_OC.md` (Fase 17) | El control positivo anterior, con el efecto Berger; de aquí se reutilizan el módulo, las exclusiones de X y la estructura del guion |
| `MODELADO.md`, Fases 14 y 14b | El arnés de validación reutilizado y el resultado sobre la fatiga que este control contextualiza |
| `REGRESION_LINEAL.md` | El índice theta/alfa contra el VAS: el mismo biomarcador, aquí sin etiqueta de fatiga |
| `PCA_CLUSTER.md` | La vía no supervisada sobre los grupos extremos de VAS |
| `ESTADO.md`, Fase 13 | Cómo se construyen las 77 features y qué significa cada una |
| `ESTADO.md`, «Orden del protocolo» y «Duraciones» | El orden fijo que confunde condición y posición, y las duraciones por condición que explican el atajo de los contadores |
| `ESTADO.md`, Fase 12 | Por qué la agrupación es `persona_uid` y nunca `sujeto_id` |
| `ESTADO.md`, Fase 11.5 | La máscara de canales válidos |
