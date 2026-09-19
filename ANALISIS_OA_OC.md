# Ojos abiertos frente a ojos cerrados: ¿conserva el pipeline el efecto Berger?

> **Qué es este documento.** La clasificación binaria de las grabaciones de
> reposo con los ojos abiertos (OA) y con los ojos cerrados (OC) a partir de las
> features de la Fase 13, como **control positivo del pipeline**. No busca un
> clasificador útil —nadie necesita adivinar si alguien tenía los ojos
> cerrados—: busca saber si el efecto neurofisiológico más establecido del EEG de
> superficie sobrevive a todo el procesamiento, y si lo que pesa en el modelo es
> lo que la neurofisiología predice.
>
> **Resultado en una línea:** sí. Validando sobre personas no vistas, la
> regresión logística acierta con **0.849** de exactitud balanceada (IC95
> [0.826, 0.873]; azar 0.500), la feature más discriminativa es el alfa
> posterior (d = +1.30) y los canales con más efecto son O1 y O2. Cumple 3 de los
> 4 criterios de coherencia declarados antes de ejecutar; el cuarto falla, y se
> cuenta en la sección 6.
>
> **Guion reproducible:** `analisis/analisis_oa_oc.py`
> **Código de apoyo:** `src/eeg_library/modelado/reposo.py`
> (`tests/test_reposo.py`, 78 pruebas)
> **Salida numérica:** `<EEG_DATA>/processed/oa_oc.json`
> **Registro de ejecución:** `analisis/salidas/oa_oc.log`
> **Figuras:** `figures/oa_oc_confusion.png`, `figures/oa_oc_alfa_posterior.png`,
> `figures/oa_oc_canales.png`
> **Resumen en el cuaderno de modelado:** `MODELADO.md`, Fase 17
>
> Fecha de ejecución: 2026-09-19. **Solo `train` + `val`; el conjunto de test no
> se ha abierto.**

---

## 1. La pregunta, y por qué es distinta de las anteriores

Las Fases 14, 14b y 16 y `REGRESION_LINEAL.md` preguntan si las features de la
Fase 13 llevan **fatiga percibida**. Lo que dice `MODELADO.md`: para una persona
nueva no hay nada, y para una persona conocida hay un efecto pequeño (tres
niveles: 0.3947 [0.3484, 0.4376] frente a 0.3333 de azar). Ese resultado admite
dos lecturas: que la fatiga no esté en la señal, o que el procesamiento destruya
parte de ella.

La Fase 14 ya aporta controles positivos (las features identifican a la persona
33× por encima del azar y acertaban la condición con 0.608 sobre 7 clases, azar
0.143), pero son **poco específicos**: clasificar 7 condiciones puede apoyarse en
cualquier diferencia entre tareas y no dice *qué* está usando el modelo.

OA frente a OC es el contraste con la firma neurofisiológica más establecida:
el ritmo alfa (8–13 Hz) sobre las regiones posteriores aumenta al cerrar los ojos
(efecto Berger, 1929). Predice **qué** debe pesar (alfa), **dónde** (occipito-
parietal) y **con qué signo** (más alfa en OC). Eso hace del control uno
*falsable*: si el pipeline conserva el efecto, el modelo tiene que apoyarse en
él; si acierta por otra vía, o el signo sale al revés, el resultado es
sospechoso aunque el acierto sea alto.

## 2. Datos

| | |
|---|---|
| Grabaciones de reposo en el dataset | 1304 (645 OA + 659 OC) |
| Apartadas (test) | 209 (28 personas) — **no se leen** |
| Descartadas por no tener `persona_uid` | 23 (sin él no se puede agrupar) |
| **En el análisis** | **1072 grabaciones: 531 OA + 541 OC** |
| Personas / sesiones | 146 / 547 |
| Por split | 898 train + 174 val |
| Grabaciones por persona | mediana 8, rango 1–10 |
| Grabaciones con alguna región sin canales válidos (NaN) | 33 — mediana del entrenamiento dentro de cada pliegue |

**La unidad de análisis es la grabación**, no la sesión. Con el VAS había que
colapsar a sesión porque se responde una vez por sesión y las 7 grabaciones
compartían etiqueta (Fase 14). La condición ojos abiertos/cerrados, en cambio, es
una propiedad de cada grabación, con su propio registro EEG: cada fila es una
etiqueta genuinamente independiente. Lo que **no** es independiente es la
persona (cada una aporta ~7 grabaciones), y eso se controla agrupando por
`persona_uid` —nunca `sujeto_id`, porque seis de ellos contienen dos personas—
tanto al validar como al calcular intervalos.

## 3. Qué entra a X, y qué no

`columnas_de_features()` sigue siendo la única definición de X del proyecto y no
se ha tocado. `reposo.columnas_de_features_reposo()` parte de ella y le resta 7
columnas, de forma explícita y con tests: de las 77 quedan **70**.

### `condicion_orden` es la etiqueta

Codifica la posición de la condición en el protocolo: vale **1 en las 541 OC y 2
en las 531 OA, sin una sola excepción**. Con ella el acierto sería 100 % por
construcción. Es la misma trampa que la Fase 14 ya encontró en el control de
condición sobre 7 clases («la cifra 0.978 … incluía `condicion_orden`, que es la
propia condición codificada, así que era circular», corregido el 2026-08-26). Es
una exclusión obligatoria, no una decisión discutible: ni
`incluir_procesamiento=True` la devuelve, y hay un test que prueba que metida a
mano da 1.0.

### Los seis contadores de procesamiento

`n_ventanas`, `n_canales_buenos` y los cuatro `n_validos_*` no son señal
cerebral. Algunos discriminan OA/OC:

| Columna | d de Cohen (OC − OA) | p (Mann–Whitney) |
|---|---|---|
| `n_validos_frontal` | +0.277 | < 0.0001 |
| `n_validos_posterior` | −0.229 | 0.0004 |
| `n_ventanas` | +0.183 | 0.057 |
| `n_validos_temporal` | −0.103 | 0.045 |
| `n_validos_frontocentral` | −0.073 | 0.280 |
| `n_canales_buenos` | +0.020 | 0.506 |

Solo `n_validos_frontal` y `n_validos_posterior` discriminan con p < 0.001 (las
otras cuatro no sobreviven a una corrección por seis contrastes). Una hipótesis
razonable es que con los ojos cerrados hay menos parpadeos y más canales
frontales superan el control de calidad de la Fase 4, pero **no se ha comprobado**.
Sea cual sea el mecanismo, es una vía que no pasa por el alfa: un acierto apoyado
ahí no diría nada sobre si el pipeline lo conserva. Por eso quedan fuera del
modelo principal, y la sección 7 mide cuánto aportarían.

### Observación sobre la matriz: cuatro features son copias exactas

Cuatro de las 70 columnas de X son **idénticas valor a valor** a otra:

| Copia | Original |
|---|---|
| `alfa_posterior_log_media` | `posterior_alfa_logabs_media` |
| `alfa_posterior_log_sd` | `posterior_alfa_logabs_sd` |
| `theta_frontal_log_media` | `frontal_theta_logabs_media` |
| `theta_frontal_log_sd` | `frontal_theta_logabs_sd` |

Lo que se ha comprobado es que son idénticas; la causa probable es que las
regiones frontal y posterior de la Fase 13 son los mismos canales que los clústeres
del índice de fatiga de la Fase 9 (`CLUSTER_FRONTAL`, `CLUSTER_POSTERIOR`).
**No se han quitado** (el encargo es reutilizar la matriz tal cual y no modificar
el pipeline congelado), pero importa para leer los resultados: reparten la
importancia entre las copias, reparten el peso en una regresión regularizada y
hacen que un Random Forest muestree esa señal más veces. La sección 7 comprueba
que no cambian el resultado (Δ = −0.002 [−0.007, +0.003] con la logística).
Efectivamente hay 66 columnas distintas, no 70.

## 4. Modelos, y por qué estos dos

| | Configuración | Qué aporta |
|---|---|---|
| **Regresión logística** | imputación por mediana → tipificación → `LogisticRegression(class_weight="balanced")` | Frontera lineal. Sus coeficientes **tienen signo**, que es lo que permite comprobar la dirección del efecto: más alfa posterior tiene que empujar hacia OC. |
| **Random Forest** | imputación por mediana → 500 árboles, hoja ≥ 5, `class_weight="balanced"` | Interacciones y no linealidades. Es **la misma configuración de las Fases 14 y 14b**: el contraste con lo que da sobre el VAS es directo. |

Solo una configuración por modelo, **sin ajustar hiperparámetros**: no hay
selección del mejor de varios, que es el sesgo que el arnés de `modeling.py`
existe para evitar. Que el Random Forest salga por debajo de la logística
(sección 5) no dice que un Random Forest ajustado no pueda igualarla.

Además, tres referencias:

- **Baseline de la clase mayoritaria** y **baseline estratificado al azar**: no
  usan X. Con la exactitud balanceada el azar es 0.5 exacto, pase lo que pase con
  el reparto; si aquí saliera otra cosa, el montaje estaría roto.
- **Umbral sobre una sola variable** (`posterior_alfa_rel_media`, elegida por
  la teoría y no por los datos): un único corte, con umbral **y signo**
  reestimados dentro de cada pliegue. Responde a si hacía falta un modelo.

**Validación.** `predecir_cv_agrupada` de la Fase 14, sin modificarla: 5 pliegues
de `GroupKFold` por `persona_uid`, con la comprobación de que ninguna persona cae
a ambos lados. Imputación y tipificación se ajustan **solo con el pliegue de
entrenamiento** (van dentro del `Pipeline`). Es la situación de un investigador
que pone el casco a alguien por primera vez, sin calibración.

**Métrica principal:** exactitud balanceada, calculada dentro de cada pliegue
(media ± SD) y sobre las predicciones out-of-fold amontonadas. A diferencia de la
Fase 14b, aquí la etiqueta no se deriva de ningún corte estimado con los datos,
así que amontonar es legítimo y las dos cifras deben coincidir; se reportan
ambas para comprobarlo (difieren en la cuarta decimal). **Intervalos:** bootstrap
remuestreando **personas** (2000 réplicas, semilla 42), no grabaciones.
Las comparaciones entre modelos son **pareadas** (mismas personas en las dos
ramas de cada réplica).

## 5. Resultado — rendimiento

| Clasificador | Exactitud balanceada (media de pliegues) | IC95 | AUC | Sensibilidad (OC) | Especificidad (OA) |
|---|---|---|---|---|---|
| Baseline: clase mayoritaria | 0.5000 | [0.500, 0.500] | 0.5000 | 1.000 | 0.000 |
| Baseline: azar estratificado | 0.4999 | [0.471, 0.528] | 0.4999 | 0.512 | 0.488 |
| Umbral en `posterior_alfa_rel_media` | 0.7451 | [0.713, 0.776] | — | 0.658 | 0.832 |
| Random Forest | 0.8036 | [0.775, 0.832] | 0.8870 | 0.734 | 0.874 |
| **Regresión logística** | **0.8490** | **[0.826, 0.873]** | **0.9089** | 0.813 | 0.885 |

- **Amontonada frente a por pliegue:** 0.8492 / 0.8490 (logística), 0.8038 /
  0.8036 (RF), 0.7452 / 0.7451 (umbral). Coinciden.
- **Por pliegue** (SD 0.017 / 0.032 / 0.035): logística 0.828, 0.837, 0.855,
  0.869, 0.856; RF 0.768, 0.776, 0.846, 0.809, 0.819. Ningún pliegue de los dos modelos baja de 0.76.
- **Matrices de confusión** (filas = real, columnas = predicha):

| | Logística | | Random Forest | | Umbral | |
|---|---|---|---|---|---|---|
| | pred OA | pred OC | pred OA | pred OC | pred OA | pred OC |
| **real OA** | 470 | 61 | 464 | 67 | 442 | 89 |
| **real OC** | 101 | 440 | 144 | 397 | 185 | 356 |

### Comparaciones pareadas (exactitud balanceada)

| A − B | Diferencia | IC95 | A > B en |
|---|---|---|---|
| Logística − Random Forest | +0.0454 | [+0.0252, +0.0678] | 100 % de réplicas |
| Logística − umbral de una variable | +0.1040 | [+0.0782, +0.1311] | 100 % |
| Random Forest − umbral de una variable | +0.0586 | [+0.0374, +0.0788] | 100 % |

### Lectura

**Ambos modelos distinguen OA de OC muy por encima del azar en personas no
vistas.** Ningún IC se acerca a 0.5, y el arnés no produce acierto de la nada
(sección 7).

**El modelo multivariante aporta sobre una sola feature**: +10.4 puntos la
logística y +5.9 el Random Forest sobre el umbral en el alfa posterior. El
alfa posterior solo ya da 0.745, así que la mayor parte del efecto está ahí, pero
no toda.

**La logística supera al Random Forest** (+4.5 puntos, IC que excluye 0). La
señal es esencialmente lineal en features tipificadas. El Random Forest se
configuró para ser comparable con las Fases 14 y 14b, no para este problema;
no se ha probado a ajustarlo.

**El tope es ~0.85, no ~0.99.** Los errores no son simétricos: la logística
falla más en OC leídas como OA (101, 18.7 %) que al revés (61, 11.5 %). El umbral
de una variable deja pasar como OA **185 de las 541 OC (34 %)**. La figura
`oa_oc_alfa_posterior.png` lo muestra: la distribución de OC tiene una cola larga
hacia alfa bajo que se solapa con la de OA. Qué hay detrás no se ha investigado:
podrían ser personas con poco alfa, somnolencia, ojos no cerrados del todo o
calidad de señal. Con estas features y este diseño no se puede distinguir.

## 6. Coherencia neurofisiológica — qué pesa en el modelo

Cuatro criterios, **declarados como constantes del guion antes de ejecutar**:

| | Criterio |
|---|---|
| (a) | la banda alfa concentra ≥ 50 % de la importancia por permutación agregada por bloque |
| (b) | alguna feature de alfa posterior está entre las 5 más importantes |
| (c) | barajar de una vez **todas** las features de una banda cuesta más exactitud con alfa que con cualquier otra banda |
| (d) | el efecto apunta a OC: el coeficiente neto de las features `_media` de alfa posterior en la logística es positivo en **todos** los pliegues, y el umbral de una variable aprende signo +1 en **todos** |

Dos aclaraciones sobre cómo se fijaron. El listón de (a) era 0.30 en el plan
inicial y **se subió a 0.50 antes de ejecutar**: alfa tiene 18 de las 70 features
(25.7 %), así que 0.30 apenas lo superaba por número de features y no
discriminaba. Y el sondeo univariante previo (d = +1.30 para el alfa posterior)
ya anticipaba (b) y (d); los criterios realmente informativos son (a) y (c), que
dependen del modelo multivariante.

### 6.1 Importancia por permutación (Random Forest, out-of-fold)

Caída de la exactitud balanceada **de validación** al barajar la columna (5
barajados × 5 pliegues). Es la importancia de lo que el modelo usa para
generalizar a gente nueva, no la de Gini de un `fit` sobre todos los datos.

| Feature | Caída | d de Cohen | Coef. logística | Pliegues con coef > 0 |
|---|---|---|---|---|
| `posterior_alfa_rel_media` | +0.0109 | +1.30 | +0.586 | 5/5 |
| `posterior_alfa_logabs_media` | +0.0098 | +1.32 | +1.463 | 5/5 |
| `indice_fatiga_media` | +0.0087 | −0.61 | +0.026 | 4/5 |
| `alfa_posterior_log_media` | +0.0086 | +1.32 | +1.463 | 5/5 |
| `posterior_alfa_logabs_sd` | +0.0085 | −0.27 | −0.372 | 0/5 |
| `frontal_theta_rel_media` | +0.0064 | −0.69 | −1.190 | 0/5 |

Las cifras individuales son pequeñas (~1 punto) porque el modelo es redundante:
barajar una columna deja las demás cubriéndola. **No hay que leer estos
coeficientes como pesos independientes**: `posterior_alfa_logabs_media` y
`alfa_posterior_log_media` tienen exactamente el mismo (son la misma columna), y
`posterior_theta_logabs_sd` sale con +1.526 teniendo d = +0.04, que es lo que
hace una regresión regularizada con features correlacionadas.

### 6.2 Agregación por bloque (fracción de la importancia total)

| Por banda | | Por región | |
|---|---|---|---|
| **alfa** | **47.1 %** | **posterior** | **58.7 %** |
| theta | 18.8 % | frontal | 12.2 % |
| sin asignar | 12.9 % | temporal | 10.3 % |
| delta | 12.8 % | frontocentral | 5.8 % |
| beta | 8.5 % | sin asignar | 12.9 % |

El bloque posterior/alfa solo concentra el 44.7 %. «Sin asignar» (12.9 %) son las
features sin banda ni región propias, sobre todo `indice_fatiga_*`, que es el
cociente theta/alfa. No se ha reasignado: el criterio se declaró antes y el
índice mezcla dos bandas.

### 6.3 Permutación conjunta de bloques enteros

Barajar de una vez todas las columnas de un bloque, con la misma permutación. No
se diluye entre features casi duplicadas, que es el problema de 6.1.

| Bloque | Features | Caída de exactitud balanceada | SD entre pliegues |
|---|---|---|---|
| región posterior | 18 | +0.1720 | 0.024 |
| **banda alfa** | 18 | **+0.1519** | 0.027 |
| posterior ∩ alfa | 6 | +0.1189 | 0.019 |
| banda theta | 18 | +0.0216 | 0.023 |
| banda delta | 16 | +0.0099 | 0.014 |
| región frontal | 18 | +0.0083 | 0.013 |
| banda beta | 16 | +0.0020 | 0.005 |
| región temporal | 16 | +0.0011 | 0.006 |
| región frontocentral | 16 | +0.0000 | 0.008 |

Alfa cuesta 15 puntos, siete veces más que la siguiente banda (theta, 2).
**La región frontal casi no cuesta nada (0.8 puntos)**, en contra de que el
modelo dependa de artefacto ocular frontal.

### 6.4 Dirección del efecto

- Coeficiente neto (suma, en unidades de desviación típica) de las tres features
  `_media` de alfa posterior en la logística, por pliegue: **+3.44, +3.24, +3.62,
  +3.73, +3.53**. Positivo en los cinco.
- Signo del umbral de una variable, por pliegue: **+1, +1, +1, +1, +1** (con corte
  en 0.256–0.287; sobre todos los datos 0.286).

Más alfa posterior empuja hacia OC, en todos los pliegues y por dos vías
independientes.

### 6.5 Veredicto

| | Criterio | Resultado |
|---|---|---|
| **NO** | (a) alfa concentra ≥ 50 % de la importancia | **47.1 %** |
| SÍ | (b) alfa posterior entre las 5 primeras features | puesto 1 |
| SÍ | (c) alfa es la banda que más cuesta al barajarla entera | alfa (+0.152) |
| SÍ | (d) dirección hacia OC en todos los pliegues | coef. + / umbral + |

**Se cumplen 3 de 4.** El criterio (a) **no se cumple** con el listón declarado:
47.1 % queda por debajo del 50 %. Habría pasado con el 0.30 del plan inicial, pero
ese listón no discriminaba. Lo que sí se puede decir es por qué mide mal aquí:
suma importancias individuales que se diluyen entre features correlacionadas y
duplicadas, y deja fuera el 12.9 % del índice theta/alfa. El criterio (c), diseñado
justamente para no sufrir eso, da una diferencia clara. Pero (a) sigue siendo un
«no», y así hay que citarlo.

## 7. Controles y sensibilidad

### 7.1 Control negativo: etiquetas barajadas dentro de cada persona

Destruye la relación entre las features y OA/OC pero conserva las personas y
cuántas OA/OC tiene cada una (20 barajados por modelo).

| Modelo | Exactitud balanceada con etiquetas barajadas | Real | Diferencia |
|---|---|---|---|
| Logística | 0.5013 (SD 0.018; rango 0.466–0.540) | 0.8490 | 19 SD |
| Random Forest | 0.5006 (SD 0.024; rango 0.449–0.537) | 0.8036 | 13 SD |

Ninguno de los 20 nulos alcanza el valor real. **El arnés no produce acierto de
la nada.** Con 20 barajados el p empírico no puede bajar de 1/21 = 0.048: es la
resolución del control, no un valor que leer.

### 7.2 Sensibilidad: qué features entran a X

Δ es la diferencia **pareada** frente al principal (misma remuestra de personas
en las dos ramas), con su IC95.

| Subconjunto | n | Modelo | Exactitud bal. | IC95 | AUC | Δ vs principal |
|---|---|---|---|---|---|---|
| **Principal** | 70 | Logística | 0.8490 | [0.826, 0.873] | 0.909 | — |
| | | Random Forest | 0.8036 | [0.775, 0.832] | 0.887 | — |
| Sin las 4 copias exactas | 66 | Logística | 0.8472 | [0.823, 0.871] | 0.908 | −0.002 [−0.007, +0.003] |
| | | Random Forest | 0.8008 | [0.771, 0.829] | 0.884 | −0.003 [−0.011, +0.005] |
| + 6 contadores de procesamiento | 76 | Logística | 0.8491 | [0.826, 0.872] | 0.909 | +0.000 [−0.012, +0.011] |
| | | Random Forest | 0.8035 | [0.775, 0.831] | 0.886 | −0.000 [−0.007, +0.007] |
| **Solo** los 6 contadores | 6 | Logística | 0.5763 | [0.551, 0.602] | 0.603 | −0.273 [−0.307, −0.240] |
| | | Random Forest | 0.5304 | [0.505, 0.556] | 0.555 | −0.274 [−0.309, −0.237] |
| Solo región posterior | 18 | Logística | 0.8240 | [0.794, 0.852] | 0.893 | −0.025 [−0.045, −0.007] |
| | | Random Forest | 0.7963 | [0.766, 0.825] | 0.879 | −0.007 [−0.022, +0.008] |
| Sin banda alfa ni índice | 50 | Logística | 0.8089 | [0.784, 0.834] | 0.892 | −0.040 [−0.058, −0.022] |
| | | Random Forest | 0.7766 | [0.748, 0.803] | 0.861 | −0.027 [−0.044, −0.011] |
| Sin alfa, índice ni potencia relativa | 26 | Logística | 0.7315 | [0.708, 0.756] | 0.792 | −0.118 [−0.144, −0.091] |
| | | Random Forest | 0.6722 | [0.647, 0.699] | 0.739 | −0.132 [−0.166, −0.095] |

Lo que se lee:

- **Las cuatro copias exactas no cambian nada** (los IC de Δ incluyen 0).
- **Los seis contadores tampoco aportan cuando se añaden** (Δ = 0.000), así que
  excluirlos no cuesta acierto: es una cuestión de interpretabilidad. **Solos**
  dan 0.576 con la logística (IC excluye 0.5) y 0.530 con el Random Forest: el
  atajo artefactual existe, pero es pequeño.
- **La región posterior sola casi basta**: −2.5 puntos con la logística y una
  diferencia no distinguible de 0 con el Random Forest.
- **Quitar el alfa cuesta, pero no destruye el acierto**: −4.0 puntos (logística) y
  −2.7 (RF), significativos, pero el modelo conserva 0.81 y 0.78. **Esto no es
  un colapso** y hay que decirlo, porque la ablación **no elimina el alfa**: las
  cuatro potencias relativas de cada región suman *exactamente* 1 (comprobado:
  mínimo = máximo = 1.0000 en las cuatro regiones), así que
  `posterior_alfa_rel_media` = 1 − (delta + theta + beta relativas) y una
  regresión lineal la reconstruye sin error a partir de las que quedan. Lo que
  se pierde es la información propia de la potencia absoluta y el índice. Además
  delta y theta posteriores cambian por sí mismas con el cierre de ojos
  (d = −0.86 y −0.79 en relativas).
- **Sin alfa ni potencia relativa** (solo log-potencia absoluta de delta, theta y
  beta) baja a 0.73 / 0.67, todavía por encima del azar. Esta sí quita la
  reconstrucción exacta, y aun así queda información. Con estas features no se
  puede separar un efecto de banda ancha (o de artefacto ocular residual, o de
  orden de registro) de una posible fuga espectral del pico alfa hacia las bandas
  vecinas, que con una ventana de 1 s (resolución de 1 Hz, Fase 8) es plausible
  pero no se ha medido.

## 8. Resolución por canal (suplemento)

`matriz_rf.parquet` agrega por región (P7, P8, O1, O2 forman la «posterior»).
Para responder qué **canales** llevan el efecto, el guion reagrega
`matriz_modelado.parquet` con `construir_matriz_rf` y una región por canal (la
variante que documenta `agregar_por_region`), restringida a las mismas 1072
grabaciones y leída ya filtrada, de modo que las épocas de las grabaciones de test
no llegan a cargarse.

| Canal | Grab. válidas | d alfa relativo | IC95 (personas) | d alfa log-abs | d delta | d theta | d beta | Exactitud bal. (1 canal) | AUC |
|---|---|---|---|---|---|---|---|---|---|
| **O2** | 985 | **+1.34** | [+1.19, +1.51] | +1.22 | −0.98 | −0.93 | −0.38 | 0.755 | 0.823 |
| **O1** | 944 | **+1.33** | [+1.18, +1.50] | +1.31 | −0.97 | −0.90 | −0.43 | 0.761 | 0.826 |
| **P7** | 938 | +1.06 | [+0.93, +1.21] | +1.06 | −0.57 | −0.62 | −0.40 | 0.716 | 0.784 |
| **P8** | 958 | +1.02 | [+0.90, +1.15] | +1.08 | −0.58 | −0.48 | −0.41 | 0.715 | 0.784 |
| AF3 | 883 | +0.92 | [+0.80, +1.06] | +1.02 | −0.19 | −0.57 | −0.57 | 0.733 | 0.800 |
| F3 | 1007 | +0.92 | [+0.80, +1.04] | +0.83 | −0.37 | −0.51 | −0.39 | 0.689 | 0.758 |
| FC5 | 957 | +0.90 | [+0.79, +1.04] | +0.84 | −0.27 | −0.40 | −0.45 | 0.687 | 0.761 |
| F4 | 986 | +0.88 | [+0.76, +1.01] | +0.67 | −0.36 | −0.42 | −0.26 | 0.675 | 0.738 |
| AF4 | 843 | +0.87 | [+0.74, +1.01] | +1.00 | −0.10 | −0.62 | −0.58 | 0.715 | 0.795 |
| F7 | 914 | +0.87 | [+0.75, +1.00] | +0.79 | −0.30 | −0.65 | −0.39 | 0.692 | 0.756 |
| F8 | 885 | +0.80 | [+0.67, +0.92] | +0.84 | −0.26 | −0.57 | −0.37 | 0.696 | 0.760 |
| T7 | 916 | +0.77 | [+0.67, +0.89] | +0.77 | −0.24 | −0.27 | −0.23 | 0.679 | 0.739 |
| FC6 | 985 | +0.77 | [+0.66, +0.88] | +0.67 | −0.19 | −0.32 | −0.36 | 0.658 | 0.732 |
| T8 | 905 | +0.76 | [+0.65, +0.88] | +0.92 | −0.22 | −0.25 | −0.23 | 0.695 | 0.763 |

(«Exactitud bal. (1 canal)»: logística con solo el alfa relativo y absoluto de ese
canal, validación agrupada por persona, sobre las grabaciones donde el canal
superó el control de calidad.)

- **O1 y O2 son los canales con más efecto** (d ≈ +1.33) y sus IC **no solapan**
  con los de ningún canal frontal, central ni temporal. Es la topografía clásica
  del efecto Berger.
- **P7 y P8** (≈ +1.04) quedan por encima de la mayoría, pero sus IC solapan con
  los de los frontales más altos (AF3, F3, FC5): no se distinguen. Diferencias de
  0.1–0.2 entre canales vecinos **no** son distinguibles.
- **El efecto es amplio**: en todos los canales el alfa relativo sube con los ojos
  cerrados (d entre +0.76 y +1.34) y las relativas de delta y theta bajan. Alfa
  frontal alto con ojos cerrados es un fenómeno conocido, pero también es
  compatible con actividad ocular residual en OA que reduzca el alfa relativo por
  aumento de delta/theta frontal; la sección 6.3 (la región frontal casi no cuesta
  al barajarla) va contra que el modelo dependa de eso, sin descartarlo.
- **Un solo canal occipital ya da 0.76** (O1) con solo su alfa, casi igual que el
  umbral sobre toda la región posterior (0.745).
- **El número de grabaciones válidas varía por canal** (843–1007 de 1072), porque
  se usan solo las que superaron la Fase 11.5.

## 9. Las figuras

### `figures/oa_oc_confusion.png`

Las dos matrices de confusión, lado a lado y en la misma escala. El color es el
**porcentaje de cada clase real** (cada fila suma 100 %), no el recuento: así el
desbalance de clases no cambia el color de las celdas. La diagonal es oscura y los
errores claros; el error dominante en ambos modelos es la celda «real OC, predicha
OA».

### `figures/oa_oc_alfa_posterior.png`

Distribución del alfa relativo posterior por condición, con el corte de la
referencia de una variable (0.286). Azul = OA, naranja = OC. Es la figura que
hace visible el efecto en estos datos: las dos nubes están desplazadas
(d = +1.30, AUC univariado 0.811), pero se solapan mucho, y la de OC tiene esa
cola larga hacia alfa bajo. El corte es descriptivo (ajustado sobre todos los
datos); la validación cruzada lo reestima dentro de cada pliegue.

### `figures/oa_oc_canales.png`

El d de Cohen del alfa relativo por canal, con su IC95 remuestreando personas.
Forma de énfasis: los cuatro canales posteriores en naranja (el color de OC), el
resto en gris.

## 10. Conclusión

**El pipeline conserva el efecto Berger.** Sobre personas no vistas, la regresión
logística distingue OA de OC con una exactitud balanceada de **0.849** (IC95
[0.826, 0.873]; AUC 0.909), y el Random Forest con la configuración de las Fases
14 y 14b con **0.804** [0.775, 0.832]; el azar es 0.500 y el control con etiquetas
barajadas da 0.501. Lo que pesa es lo que la neurofisiología predice: el alfa
posterior es la feature con más efecto (d = +1.30), es la banda que más cuesta al
barajarla, apunta hacia OC en los cinco pliegues, y los canales con más efecto son
los occipitales O1 y O2.

Para la memoria del TFG, esto **refuerza** el diagnóstico de la Fase 14: los
resultados sobre la fatiga percibida no se explican porque el procesamiento
destruya el efecto espectral más robusto que hay. Los controles positivos
anteriores eran poco específicos; este dice *qué* usa el modelo y comprueba que es
lo esperable.

### Alcance de la afirmación

- **Sí establece** que un efecto grande, localizado y bien conocido a nivel de
  grabación (d ≈ 1.3, alfa posterior) sobrevive a las Fases 1–13 y se recupera en
  personas nuevas.
- **No establece** que el pipeline conserve la señal de **fatiga**. El efecto
  Berger es enorme; la fatiga, si está, es mucho más sutil, y este resultado no
  dice si un efecto de ese tamaño sobreviviría. Un control positivo con un efecto
  grande solo descarta que el pipeline esté roto para efectos grandes.
- **No establece** nada sobre la señal cruda de la ruta CNN.
- **No demuestra que el modelo dependa únicamente del alfa.** Quitar todas las
  features de alfa cuesta entre 2.7 y 4.0 puntos (sección 7.2), y esa ablación
  no elimina el alfa relativo.
- **No es confirmatorio.** Se ejecutó sobre `train` + `val`. Al ser la etiqueta la
  condición y no el VAS, **no** consume el alfa del paquete de test pendiente,
  que es sobre la fatiga; las configuraciones exploradas aquí no se suman a ese
  recuento. El test sigue sin abrirse.

## 11. Limitaciones

- **El orden del protocolo es fijo: OC siempre precede a OA** (OC → OA → STROOP
  → …). Cualquier deriva sistemática dentro de la sesión —el gel secándose, el
  participante acomodándose, somnolencia creciente— queda confundida con la
  condición, y este diseño **no puede separarlas**. Es la limitación más seria y
  no tiene solución con estos datos. Hay dos indicios parciales a favor de la
  lectura fisiológica, ninguno concluyente: el efecto se concentra en O1 y O2 con
  el gradiente occipital → frontal que predice el efecto Berger; y el número de
  canales buenos no difiere entre OC y OA (d = +0.02, p = 0.51), y la Fase 11.5
  midió que el contacto no se degrada a lo largo de la sesión (Spearman entre
  canales excluidos y posición en el protocolo: 0.048). Pero
  calidad de contacto no es deriva espectral ni somnolencia.
- **La etiqueta es la condición asignada, no el estado verificado.** No se sabe
  si el participante tenía de verdad los ojos cerrados o abiertos durante toda la
  grabación.
- **Residuo ocular.** La ICA con ICLabel de la Fase 6 se aplica en el
  procesamiento por lotes, así que los parpadeos —la diferencia obvia entre OA y
  OC— están tratados, pero la eliminación no es perfecta. Lo que se ve
  (posterior ≫ frontal al barajar bloques) va en contra de que el acierto sea
  artefacto ocular, sin descartarlo.
- **Personas nuevas sin calibración.** Los errores (~15 %) probablemente incluyen
  las diferencias de amplitud basal de alfa entre personas, que un modelo con un
  registro de referencia de la propia persona no tendría. No se ha medido: la
  unidad es la grabación, no el par OA/OC de una sesión.
- **La ablación del alfa no es limpia**: las relativas suman exactamente 1 y
  permiten reconstruir el alfa relativo; sección 7.2.
- **Cuatro features de la Fase 13 son copias exactas de otra.** No cambian el
  resultado, pero distorsionan las importancias individuales y los coeficientes.
- **Una configuración por modelo, sin ajustar.** El Random Forest por debajo de la
  logística no es una afirmación sobre lo que un Random Forest ajustado puede
  hacer.
- **Importancia por permutación con features correlacionadas** infravalora cada
  una por separado; por eso se añade la permutación conjunta. El control con
  etiquetas barajadas tiene una resolución de p ≥ 0.048.
- **Cohorte y montaje:** 146 personas, Emotiv Epoc X (14 electrodos), ventanas de
  4 s agregadas por grabación. Los resultados pueden no trasladarse a otro
  montaje.

## 12. Lo que **no** hay que hacer

- **Usar `columnas_de_features()` tal cual para OA/OC.** Devuelve
  `condicion_orden`, que es la etiqueta: da 100 % sin significar nada. Usar
  `columnas_de_features_reposo()`.
- **Citar el criterio (a) como cumplido.** No lo está (47.1 % frente a 50 %).
  Se cumplen 3 de 4, y el documento lo dice así.
- **Leer 0.85 como la sensibilidad del pipeline a la fatiga.** Es la de un efecto
  con d ≈ 1.3.
- **Leer las importancias o los coeficientes individuales de las features de alfa
  posterior como pesos separados.** Hay copias exactas y correlaciones altas.
- **Presentar la ablación sin alfa como «el modelo no depende del alfa».** Ni como
  «depende solo del alfa»: el alfa relativo se reconstruye exactamente con las
  otras tres bandas.
- **Contar las 1072 grabaciones como n independientes.** Son 146 personas.
- **Leer el p = 0.048 del control como un hallazgo.** Es el mínimo alcanzable con
  20 barajados.
- **Ajustar el Random Forest o la logística a este problema para subir la cifra.**
  No es el objetivo, y cada configuración adicional engorda el recuento.
- **Abrir el test.** `preparar_reposo` se niega salvo `permitir_test=True`.

## 13. Reproducir

```bash
# desde la raíz del repositorio
python analisis/analisis_oa_oc.py > analisis/salidas/oa_oc.log 2>&1
```

Duración: ~7.5 min (453 s y 460 s en las dos últimas ejecuciones). Semilla fija
(`SEMILLA = 42`) en los modelos, las permutaciones y todos los bootstraps. Dos
ejecuciones completas consecutivas del guion tal como está dieron un **log, un
JSON y unas figuras idénticos byte a byte** (comprobado el 2026-09-19 con SHA-256
y comparando el JSON campo a campo). Una ejecución anterior, con una versión previa
del guion que solo tenía menos salidas y figuras menos pulidas, ya había dado las
mismas cifras en todo lo que comparte.

Además de `matriz_rf.parquet`, el suplemento por canal necesita
`<EEG_DATA>/processed/matriz_modelado.parquet` y
`<EEG_DATA>/metadata/labels_grabaciones.csv`.

El guion aparta el `test` en su primer paso; `preparar_reposo` se niega a
incluirlo sin permiso explícito, y el suplemento por canal lee el parquet de
épocas ya filtrado a las grabaciones de análisis.

Versiones con las que se generaron estos resultados: Python 3.10.11,
scikit-learn 1.7.2, numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, matplotlib 3.10.9.

## 14. Documentos relacionados

| Documento | Qué cubre |
|---|---|
| `MODELADO.md`, Fase 17 | Este análisis, resumido en el cuaderno de modelado |
| `MODELADO.md`, Fase 14 | El arnés de validación reutilizado y los controles positivos anteriores |
| `MODELADO.md`, Fase 14b | El resultado sobre la fatiga en tres niveles que este control contextualiza |
| `REGRESION_LINEAL.md` | El índice theta/alfa contra el VAS — el mismo biomarcador, aquí sin etiqueta de fatiga |
| `PCA_CLUSTER.md` | La vía no supervisada sobre los grupos extremos de VAS |
| `ESTADO.md`, Fase 13 | Cómo se construyen las 77 features y qué significa cada una |
| `ESTADO.md`, Fase 12 | Por qué la agrupación es `persona_uid` y nunca `sujeto_id` |
| `ESTADO.md`, Fase 11.5 | La máscara de canales válidos y el contacto a lo largo de la sesión |
