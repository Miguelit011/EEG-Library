# Pre-registro del análisis confirmatorio sobre el conjunto de test

> **Estado: BORRADOR, 2026-10-06.** Queda fijado en el commit que lo incluya,
> que debe ser **anterior** a cualquier lectura del split `test`. Desde ese
> momento no se modifica: cualquier cambio posterior se anota en la sección 11
> («Desviaciones») con fecha y motivo, y se reporta junto a los resultados.
>
> **El conjunto de test no se ha abierto.** Todas las cifras de referencia de
> este documento salen de `train` + `val`.

---

## 1. Qué se confirma y qué no

El test se gasta **una sola vez**: todo lo que se ejecuta sobre él está
declarado aquí, se ejecuta de una pasada y se reporta entero, salga lo que
salga. Se renuncia por escrito a la **ruta CNN**: no existe y no entra en el
paquete; queda como trabajo futuro.

Hay dos familias de pruebas, con su multiplicidad controlada por separado:

| Familia | Qué contiene | Control de multiplicidad |
|---|---|---|
| **VAS** (fatiga percibida) | 1 prueba principal + 5 secundarias | Principal a α = 0.05 unilateral; secundarias con Holm a α = 0.05 |
| **Condición** (controles positivos del pipeline) | 3 pruebas principales: OA/OC, reposo/tarea, 5 tareas | Holm a α = 0.05 entre las 3 |

Todo lo que no figura como prueba en las secciones 4 y 6 es **descriptivo**: se
reporta con su IC, sin p y sin veredicto de confirmación.

## 2. Datos

| | |
|---|---|
| Partición | `<EEG_DATA>/metadata/splits.csv`, por sujeto y estratificada por centro |
| Test | **28 personas** (26 `sujeto_id`), **106 sesiones** etiquetadas, 725 grabaciones etiquetadas (732 grabaciones de test en `matriz_rf.parquet`) |
| Entrenamiento de los modelos finales | `train` + `val` completos: 547 sesiones / 146 personas (VAS); 1072, 3728 y 2656 grabaciones (condición) |
| Matriz | `matriz_rf.parquet` (Fase 13), sin regenerar; se registra su SHA-256 |

Los modelos finales se reajustan **con train+val completo** (146 personas). Las
cifras de validación de referencia salían de validación cruzada agrupada en 5
pliegues, con **~117 personas de entrenamiento por pliegue** (~109 en la Tarea
B); se declara esa diferencia.

Unidad de análisis: la **sesión** en la familia VAS y la **grabación** en la
familia de condición. La agrupación (bootstrap, permutaciones) es siempre por
`persona_uid`. Las grabaciones sin `persona_uid` se descartan, como en
validación.

## 3. Métodos comunes

- **Semilla:** 42 en todos los modelos, bootstraps y permutaciones. Es la cifra
  oficial; no hay análisis de robustez por semillas.
- **IC:** bootstrap de percentiles remuestreando **personas** de test, **10 000
  réplicas**, IC95 bilateral.
- **p de las pruebas principales:** unilateral por permutación, **10 000
  permutaciones** de la etiqueta real **dentro de cada persona** frente a las
  predicciones fijas (no se reentrena), con la definición convencional: el
  observado cuenta como un valor más de la referencia y los empates cuentan
  enteros, `p = (1 + #{nulo ≥ obs}) / (N + 1)`. Garantiza un error de tipo I
  ≤ α. Se descartó el mid-p: con 20 000 simulaciones nulas los dos tienen el
  error de tipo I nominal (p convencional 0.0524 ± 0.0016 y 0.0487 ± 0.0015 en
  dos tandas) y la misma potencia (0.329 frente a 0.331), y el mid-p no tiene
  esa garantía (`MODELADO.md`, «Potencia de la prueba principal en test»).
- **p de las secundarias del VAS:** unilateral por bootstrap de personas
  (10 000 réplicas): `p = (#{réplicas en el lado de H0} + 1) / (B + 1)`, con
  el lado de H0 definido en cada prueba (sección 4.2). Se usa el bootstrap y no
  la permutación porque es la única forma común a las pruebas de superioridad
  (S1, S5) y a las de no superioridad (S2–S4).
- **Holm:** sobre los p de cada familia, a α = 0.05: las 5 secundarias del VAS
  y, por separado, las 3 pruebas de condición.
- **Control de la deriva por número de visita** (`residualizar_por_visita`,
  `MODELADO.md`, «Control de la deriva por número de visita»): recta de
  mínimos cuadrados sobre el número de visita (M2 → 2 … M6 → 6) ajustada dentro
  de cada persona, en X y en y; las personas con menos de 3 sesiones válidas o
  una sola visita distinta se dejan sin residualizar. Se aplica a **todos los
  análisis intra-persona del VAS** (prueba principal, θ/α intrapersona y
  compuesto `_rel_sd`), en train+val para entrenar y en test para evaluar. **La
  cifra confirmatoria es la controlada**; la no controlada se reporta como
  descriptiva.
- **Margen de equivalencia** para las afirmaciones nulas: ΔR² < 0.05 y mejora
  del MAE < 2 puntos (escala 0–100). Cuando una prueba tiene las dos métricas
  (S3, S4) se exigen **las dos**: es lo más estricto para afirmar un nulo.

## 4. Familia VAS

### 4.1 Prueba principal

**Hipótesis.** En la tarea *persona conocida* (Tarea B), un clasificador
entrenado sobre el EEG asigna el nivel de fatiga de la sesión —por debajo, en
o por encima de lo habitual en esa persona— mejor que el azar, una vez
eliminada la deriva por visita.

**Modelo (fijado):** `RandomForestClassifier(n_estimators=500,
min_samples_leaf=5, random_state=42, n_jobs=-1)`, sin `class_weight` ni
imputación. X = las 77 columnas de `columnas_de_features_sesion` (76 features
espectrales agregadas por sesión + `n_grabaciones`).

**Entrenamiento (train+val):**
1. `agregar_a_sesion` sobre las grabaciones de train+val.
2. `centrar_por_persona_loo` sobre X y VAS; se descartan las sesiones de
   personas con una sola sesión (quedan 537 / 136).
3. `residualizar_por_visita` sobre X y sobre la desviación del VAS.
4. Cortes de los tercios = percentiles 33.3 y 66.7 de esa desviación
   controlada en las 537 sesiones (`cortes_terciles`). Se calculan **una vez** y
   se guardan.
5. Ajuste del bosque sobre las clases resultantes.

**Evaluación (test):** los mismos pasos 1–3 sobre las sesiones de test, con las
medias LOO y las rectas de visita calculadas **dentro de cada persona de
test**, con sus otras sesiones de test. Las personas de test con una sola
sesión quedan fuera. La clase real sale de los cortes de train+val (paso 4).

**Métrica principal:** exactitud balanceada amontonada sobre las sesiones de
test (un solo juego de cortes, así que amontonar es legítimo). Azar = 1/3.

**Referencia de validación:** **0.3795**, IC95 [0.3302, 0.4290]. Sin control
era 0.3947 [0.3484, 0.4376].

**Reglas de lectura**, declaradas antes:

| Veredicto | Condición |
|---|---|
| **Confirma** | p unilateral < 0.05 |
| **No confirma** | p ≥ 0.05 **y** límite superior del IC95 < 0.3795 |
| **No concluye** | cualquier otro caso |

Se reporta siempre la cifra con su IC, aunque «confirma». Si el p y el IC95
bilateral discrepan (p < 0.05 con un IC que roza 1/3), decide el p y se dice.

**Potencia** (`analisis/analisis_potencia_test.py`, `MODELADO.md`): si el efecto
real es el de validación, P(confirma) = **0.33** (0.329), P(IC95 contiene el azar y la
referencia) = 0.81 y P(IC95 por debajo de la referencia) = 0.03. Para un 80 %
de potencia, la exactitud balanceada real tendría que rondar 0.47. **Un
resultado «no concluye» es lo más probable y no es evidencia contra el
efecto.**

**Descriptivo junto a la principal:** F1 macro, exactitud adyacente, MAE
ordinal, tau de Kendall, matriz de confusión 3×3, exhaustividad por clase,
reparto de clases reales y predichas; la misma prueba **sin** control de
visita; las otras tres configuraciones de la Fase 14b (hoja≥1, hoja≥15, hoja≥5
balanceado); y el regresor RF hoja≥5 discretizado con los mismos cortes, con la
diferencia **pareada** clasificador − regresión y su IC (la afirmación de la
Fase 14b de que clasificar aporta porque la regresión se refugia en el centro).

### 4.2 Secundarias (Holm, 5 pruebas)

| # | Afirmación de validación | Prueba en test | H0 (lado del p) | Referencia de validación |
|---|---|---|---|---|
| S1 | El compuesto `_rel_sd` sigue al VAS dentro de persona | ρ de Spearman intra-persona **con control de visita** > 0 | réplicas de ρ ≤ 0 | +0.130 [+0.007, +0.247] |
| S2 | El índice θ/α no sigue al VAS dentro de persona | R² intrapersona **con control de visita** por debajo del margen | réplicas de R² ≥ 0.05 | R² = 0.001 [0.000, 0.015]; β = −0.455 mm |
| S3 | Persona nueva: el EEG no predice el VAS (Tarea A, regresión) | RF frente a la media de VAS de train+val: ΔR² y mejora del MAE por debajo del margen | réplicas de ΔR² ≥ 0.05 o de mejora de MAE ≥ 2 (p = el mayor de los dos) | R² = −0.010 [−0.074, +0.038] |
| S4 | Persona conocida: el EEG no aporta sobre la media de la persona (Tarea B, regresión) | con EEG − sin EEG, en escala original: ΔR² y mejora del MAE por debajo del margen | ídem S3 | 0.2967 frente a 0.2988 |
| S5 | Persona nueva: el RF **ordena** las sesiones (ρ ≈ +0.12) | ρ de Spearman entre la predicción del RF y el VAS > 0 | réplicas de ρ ≤ 0 | +0.117, p = 0.006 |

Detalle de cada una:

- **S1 — compuesto `_rel_sd`.** `compuesto_rel_sd(sesiones_test,
  referencia=sesiones_train_val)`: z de las 16 columnas `_rel_sd` con la media
  y la SD de **train+val**, media de las 16. En test: centrado por la media de
  la persona, `residualizar_por_visita` sobre el compuesto y sobre el VAS, y
  Spearman sobre las sesiones de personas con ≥ 2 sesiones. Sin modelo. La
  versión sin control (+0.156 en validación) es descriptiva.
- **S2 — θ/α intrapersona.** `indice_fatiga_media` y VAS centrados por la
  media de la persona y residualizados por visita; MCO con un grado de libertad
  menos por persona y otro por recta (como `analisis_control_visita.py`). Se
  reporta además β con IC por *t* y por bootstrap, y si el IC excluye la
  pendiente útil de +10 mm declarada en `REGRESION_LINEAL.md`. La regresión
  global, sin centrar, y la versión sin control son descriptivas.
- **S3 — Tarea A.** `RandomForestRegressor(n_estimators=500,
  min_samples_leaf=5, random_state=42)` sobre las 77 features, entrenado con las
  547 sesiones de train+val sobre el VAS absoluto; evaluado en las 106 sesiones
  de test. ΔR² = R²(RF) − R²(media de train+val), ambos con `r2_score` sobre
  test; mejora del MAE = MAE(media de train+val) − MAE(RF). Se reporta también
  el R² del RF tal cual.
- **S4 — Tarea B, regresión.** El mismo regresor entrenado con la desviación
  centrada LOO de train+val (537 sesiones, **sin** control de visita, como en
  la Fase 14). En test: predicción reconstruida = media LOO de la persona en
  test + desviación predicha; rival = media LOO de la persona en test.
  ΔR² = R²(con EEG) − R²(sin EEG); mejora del MAE = MAE(sin EEG) − MAE(con EEG).
- **S5 — orden en Tarea A.** Spearman entre las predicciones de S3 y el VAS de
  test.

**Descriptivo de la familia:** la Tarea A en tres niveles (`RandomForestClassifier`
de la principal, cortes de los tercios del VAS absoluto de train+val), con su
exactitud balanceada e IC; MAE, RMSE y ρ de todos los modelos; las
importancias Gini de los modelos finales (como parámetro, no como resultado de
test).

## 5. Lo que no entra

| Excluido | Motivo |
|---|---|
| Ruta CNN | No existe; trabajo futuro |
| PCA + clustering (Fase 16) | Exploratorio y no supervisado, sin afirmación que confirmar; el diseño confunde grupo de VAS con identidad |
| Control de identidad | Propiedad de las features, ya medida |
| Selección anidada top-k (Fase 14) | 5 configuraciones exploratorias |
| Ridge y `max_features=0.3` (Fase 14) | Configuraciones de exploración, no candidatas |
| Tablas de ablación por banda o región (Fases 17–19) | Eran mecanísticas; las sustituye la permutación por bloques en test |
| Logísticas de un canal, 10 binarias uno-contra-uno | Redundantes con la matriz de confusión y la permutación |
| Reentrenar con etiquetas barajadas | Lo sustituye la permutación sobre test |
| Robustez por semillas | Decisión explícita: la semilla 42 es la cifra oficial |
| Métricas del VAS por grabación o por ventana | El n es la sesión |

## 6. Familia de condición

Las tres pruebas principales usan la **regresión logística** (imputación por
mediana → tipificación → `LogisticRegression(max_iter=2000,
class_weight="balanced")`, multinomial en las 5 tareas) sobre las 70 columnas
de `columnas_de_features_reposo` (idénticas a las de tarea y a las ejecutivas):
sin `condicion_orden` ni los 6 contadores de procesamiento. El **Random Forest**
de las Fases 17–19 (imputación + 500 árboles, hoja≥5, `class_weight="balanced"`,
semilla 42) se evalúa como modelo secundario, descriptivo. Ambos se entrenan con
train+val completo y predicen test una vez.

### 6.1 Pruebas principales (Holm, 3 pruebas)

| # | Prueba | H1 | Referencia de validación (logística) |
|---|---|---|---|
| C1 | OA frente a OC | exactitud balanceada > 0.5 | 0.849 [0.826, 0.873] |
| C2 | Reposo (OA + OC) frente a tarea | exactitud balanceada > 0.5 | 0.857 [0.839, 0.874] |
| C3 | Cinco tareas ejecutivas | exactitud balanceada > 0.2 | 0.470 [0.452, 0.488] |

p: unilateral por permutación de la etiqueta dentro de persona (sección 3).
Veredicto: **confirma** si el p ajustado por Holm < 0.05.

### 6.2 Métricas descriptivas en test (los dos modelos)

- **Todas:** exactitud balanceada, IC95 por personas, matriz de confusión en
  recuentos y normalizada, diferencia **pareada** logística − RF con su IC, y
  baselines triviales (clase mayoritaria y azar estratificado).
- **OA/OC:** AUC, sensibilidad (OC), especificidad (OA), MCC y F1. Referencia
  univariante: umbral y signo sobre `posterior_alfa_rel_media` aprendidos en
  train+val (`umbral_una_variable`), con las diferencias pareadas frente a los
  dos modelos.
- **Reposo/tarea:** AUC, precisión media de la clase reposo junto a su
  prevalencia en test, sensibilidad y especificidad. Referencia univariante:
  umbral sobre `frontal_theta_rel_media`. Acierto por cada una de las 7
  condiciones, diferencia pareada OA − OC, tasa de PVTB leída como reposo, y
  ρ entre la posición en el protocolo y el acierto entre las tareas.
  **Control sin OC:** modelo «OA frente a tareas» entrenado en train+val y
  evaluado en test (referencia: 0.816 [0.796, 0.837]).
- **Cinco tareas:** F1 macro, kappa, precisión, exhaustividad y F1 por clase,
  confusión por pares y su asimetría.
- **Control de los contadores** (los 6 de procesamiento, solos, logística y RF)
  en los tres análisis. Referencias de validación: OA/OC 0.576 / 0.530;
  reposo/tarea 0.624 / **0.990**; tareas **0.465 / 0.471**.
- **Opcional, si el tiempo de ejecución lo permite:** d de Cohen por canal con
  IC por personas (OA/OC y reposo/tarea), a partir de las épocas de test de
  `matriz_modelado.parquet`.

### 6.3 Criterios de coherencia

Solo se **contrasta** en test lo que depende de predicciones o datos de test.
Lo que es propiedad del modelo final (coeficientes, signo del umbral) se
**reporta**, no se confirma. Las importancias por permutación se calculan con el
RF sobre test, con **10 repeticiones**, por feature y por bloque (permutación
conjunta), con la exactitud balanceada como puntuación.

| Análisis | Criterio | Se contrasta en test | Se reporta (modelo final) |
|---|---|---|---|
| OA/OC | (a) alfa concentra ≥ 50 % de la importancia agregada | Sí | — |
| | (b) alguna feature de alfa posterior entre las 5 más importantes | Sí | — |
| | (c) la permutación conjunta de alfa cuesta más que la de cualquier otra banda | Sí | — |
| | (d) dirección hacia OC | d de Cohen OC − OA de `posterior_alfa_rel_media` > 0 con IC que excluye 0 | coeficiente neto de alfa posterior `_media` > 0; signo del umbral +1 |
| Reposo/tarea | (a) la banda que más cuesta es theta o alfa | Sí | — |
| | (b) theta frontal hacia tarea | d (tarea − reposo) de `frontal_theta_rel_media` > 0, IC sin 0 | coeficiente neto > 0; signo del umbral +1 |
| | (c) alfa posterior hacia reposo | d (tarea − reposo) de `posterior_alfa_rel_media` < 0, IC sin 0 | coeficiente neto < 0 |
| | (d) (b) y (c) sin OC | las mismas d con OA como único reposo | coeficientes del modelo sin OC |
| Tareas | (a) los dos modelos superan el azar | límite inferior del IC95 > 0.2 | — |
| | (b) PASAT-PVSAT y STROOP-MSIT entre los 3 pares más confundidos | Sí (logística) | — |
| | (c) PVTB, la clase con mayor exhaustividad | Sí (logística) | — |
| | (d) theta frontal mayor en PASAT y PVSAT que en PVTB | d una-contra-resto (media de las `_media` de theta frontal): PASAT y PVSAT > PVTB | coeficiente neto por clase |
| | (e) la confusión sigue más el parecido cognitivo que la distancia en el protocolo | ρ(confusión, similitud) > 0 y mayor en valor absoluto que ρ(confusión, −distancia) | — |

En (e) se reporta además ρ(confusión, −distancia de duración), con las
distancias de duración fijadas con train+val. Es el confusor declarado en
validación (+0.76 frente a +0.47).

Los listones son los de validación. Cada criterio se reporta como SÍ o NO, sin
p.

## 7. Datos en bruto que se guardan

Todo en `<EEG_DATA>/processed/test_<fecha>/`. El guion se niega a escribir si
la carpeta ya existe.

| Artefacto | Contenido |
|---|---|
| `predicciones_test.parquet` | Una fila por sesión (VAS) o grabación (condición) × análisis × modelo: `sesion_uid`/`grabacion_uid`, `persona_uid`, `sujeto_id`, `centro`, `medicion`, `condicion`, posición en el protocolo, `y_true`, `y_pred` y la probabilidad de cada clase. En el VAS añade `vas`, `vas_loo`, la desviación sin y con control, la clase real, la clase predicha y la predicción continua de los regresores |
| `X_test_*.parquet` | La matriz exacta que entró a cada `predict` y la del compuesto y θ/α |
| `modelos/*.joblib` | Los modelos finales |
| `parametros.json` | Cortes de los tercios, medianas del imputador, media y SD del escalado, coeficientes e interceptos, umbral y signo de las referencias univariantes, media y SD del compuesto, `feature_importances_`, `get_params()` |
| `bootstrap/*.npy` | Las 10 000 réplicas de cada métrica y de cada diferencia pareada |
| `permutaciones/*.npy` | Las 10 000 réplicas nulas de cada prueba |
| `importancias_test.parquet` | Importancia por permutación por feature, por bloque y por repetición |
| `inventario.csv` | Sesiones y grabaciones de test usadas y descartadas, con el motivo |
| `resultados.json` | Todas las cifras de este documento, con p brutos, p de Holm y veredictos |
| `ejecucion.log` | Salida completa |

## 8. Metadatos de la ejecución

- **Código:** hash del commit, comprobación de que `git status` está limpio (el
  guion se niega a correr si no) y SHA-256 del guion.
- **Pre-registro:** ruta y SHA-256 de este documento, y el hash del commit que
  lo fija.
- **Datos:** SHA-256 de `matriz_rf.parquet`, `splits.csv`, `labels.csv`,
  `labels_grabaciones.csv`, `VAS_100mm_preFE_filtrado.csv` y, si se usa,
  `matriz_modelado.parquet`.
- **Entorno:** versión de Python, `pip freeze` del `.venv`, sistema operativo,
  modelo de CPU, número de núcleos y `n_jobs`.
- **Ejecución:** fecha y hora de inicio y fin, comando exacto y `EEG_DATA`.
- **Parámetros:** semillas, número de réplicas y de permutaciones, número de
  repeticiones de las importancias.
- **Recuentos:** personas, sesiones y grabaciones de test por análisis, con lo
  descartado.

## 9. Recuento de lo explorado en train+val

Se declara para que el lector pueda ponderar los resultados. Configuraciones
evaluadas contra el VAS:

| Fase | Configuraciones |
|---|---|
| 13 (prueba de humo y compuesto `_rel_sd`) | 2 |
| 14 | 5 modelos × 2 tareas + selección anidada con 5 valores de k = 15 |
| 14b | 5 modelos × 2 tareas + 2 sobre `_rel_sd` = 12 |
| 15 (retirada el 2026-09-15) | Varias decenas sobre 374 features, bloques por condición y controles; no cuantificadas con precisión y su código no se conserva |
| 16 (PCA + clustering) | 3 |
| `REGRESION_LINEAL.md` | 4 (principal, log₁₀, recorte y la intrapersona) |
| Control de visita | 3 |

Las Fases 17–19 (condición) añaden su propio recuento a su familia: una
configuración por modelo y sus controles y sensibilidades, documentados en cada
`ANALISIS_*.md`.

La prueba principal **no** se eligió como la mejor de estas: era la
configuración principal declarada de la Fase 14b, y las otras tres caen en el
mismo rango (0.377–0.395). El control de visita se adoptó **después** de ver
que la Fase 15 lo necesitaba, y baja la referencia.

## 10. Ejecución

1. **Guion nuevo** `analisis/analisis_test.py`, con tests de las piezas nuevas
   (ajuste final y predicción, puntuación y permutación dentro de persona,
   p por permutación, Holm, escritura de artefactos y negativa a sobrescribir).
2. **Ensayo general completo** del mismo guion con `train` como entrenamiento y
   `val` como test: comprueba que corre de principio a fin y que escribe todos
   los artefactos. Sus cifras **no son resultados** (`val` ya se usó en la
   validación cruzada) y no se usan para cambiar nada que no sea un error de
   código.
3. **Commit** del guion y de este documento. Después, **una sola ejecución**
   sobre test, con `permitir_test=True`.
4. Si la ejecución falla antes de terminar, se corrige solo el error de código,
   se anota en la sección 11 y se vuelve a ejecutar completa. Si algún
   resultado ya se ha mostrado, se anota también qué se vio.
5. Se reporta **todo**: las pruebas, con sus veredictos, y lo descriptivo.

## 11. Desviaciones

*(Vacía hasta la ejecución. Cada entrada: fecha, qué cambió respecto a este
documento, por qué y si ocurrió antes o después de ver algún resultado de
test.)*
