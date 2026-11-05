# Paso 3 — Clasificador clásico

> Línea: tiempo en tarea. Fecha: 2026-10-27. Desarrollo en
> `train` con CV repetida y **primera mirada a `val`** con la configuración
> elegida. `test` no se carga.

## Plan y criterio de decisión (escrito antes de ejecutar)

Esta sección se redactó **antes** de escribir y ejecutar el script y no se
modifica después. Los resultados van en las secciones siguientes.

### Qué tiene que salir de aquí

La configuración (features, modelo y esquema de entrenamiento) que se fijará
en el pre-registro, elegida por una regla escrita aquí antes de mirar, y su
primera comprobación en `val`.

### Datos

- `train` para la selección y `val` para la comprobación (partición de
  `matriz_rf`), etiquetadas, ≥ 10 ventanas por grabación. Funciones de carga
  del paso 1; la lectura de `val` se añade con el mismo filtro de archivos.
- Condiciones: **PASAT y PVTB** (familia), PVSAT (descriptiva), OA (control
  del contraste) y OC (control descriptivo).
- Dos versiones, como en los pasos anteriores: **igual tiempo** (cada tarea
  recortada a la duración del OA de su sesión; **principal**) y grabación
  completa. OA y OC no se recortan.

### Candidatos (lista cerrada)

**Features**, por tercio (primer y último), centradas por grabación y con los
canales excluidos a 0 tras el centrado (como en el paso 1):

1. `rel`: las 56 `log10(pot_rel)` (como el paso 1);
2. `abs`: las 56 `log10(pot_abs)`;
3. `rel+abs`: las 112;
4. `regiones`: las 4 regiones del proyecto (`REGIONES`) × 4 bandas ×
   {absoluta, relativa}, en `log10` de la media de los canales válidos
   (32 features). Una región sin canales válidos queda en 0 tras centrar.

En todos los casos, primero se hace la media del tercio y después el `log10`.

**Modelos:**

- logística: `StandardScaler` + `LogisticRegression(C, max_iter=3000)` con
  C ∈ {0.01, 0.1, 1};
- Random Forest: `modelo_bosque()` de la librería (500 árboles, hoja ≥ 5,
  `class_weight="balanced"`, semilla 42), la configuración de las Fases 14 y
  17, sin búsqueda de hiperparámetros.

**Esquema de entrenamiento:**

- `separado`: un modelo por tarea;
- `conjunto`: un solo modelo entrenado con PASAT + PVTB y evaluado en cada una.

OA (y OC) se entrena siempre con su propio modelo, con las mismas features y
el mismo modelo. Son 4 × 4 × 2 = **32 configuraciones**.

### Validación cruzada repetida (`PLAN.md`, sección 2)

- 20 repeticiones (semillas 0–19). En cada una, las personas de `train` se
  reparten al azar en 5 pliegues equilibrados y **esa misma asignación** se
  usa en todas las condiciones y configuraciones (`PredefinedSplit` sobre el
  pliegue de cada persona).
- Por configuración y repetición: acierto pareado en PASAT, en PVTB, en PASAT +
  PVTB juntas (media de las grabaciones de las dos) y en OA, y la diferencia
  **H1 = PASAT + PVTB − OA**. Se promedia sobre las 20 repeticiones y se da el
  rango.
- IC95 de la configuración elegida: bootstrap de personas (2000 réplicas,
  semilla 42) sobre la probabilidad media de cada tercio en las 20
  repeticiones.

### Regla de selección (decisión del autor, 2026-10-27)

Se elige la configuración con **mayor H1 = PASAT + PVTB − OA**, media de las 20
repeticiones, **a igual tiempo**. El acierto en las tareas se reporta al lado.
Motivo: la H1 es ahora la ventaja sobre OA, y elegir solo por el acierto en
las tareas premiaría features que captan la deriva común, que también suben
en OA. **Desempate por simplicidad:** entre las configuraciones a menos de
0.01 de la mejor se elige la más simple, en este orden: menos features
(`regiones` < `rel` = `abs` < `rel+abs`), logística antes que RF, C menor (más
regularización) y `separado` antes que `conjunto`.

### Con la configuración elegida (en `train`, descriptivo)

1. Acierto pareado de PVSAT y OC.
2. **Transferencia entre tareas:** entrenar con PASAT y evaluar en PVTB, y al
   revés, con la misma CV repetida (se entrena con las personas de los pliegues
   de entrenamiento y se evalúa con las del pliegue de prueba). Si funciona, el
   cambio es común a las dos tareas.
3. **Regresión del minuto** (grabación completa): bloques de 30 s, features
   del bloque centradas por grabación, `Ridge(alpha = 1/C)` si se elige la
   logística o `RandomForestRegressor` (500 árboles, hoja ≥ 5) si se elige el
   RF. Métrica: ρ de Spearman entre el tiempo predicho y el real dentro de
   cada grabación, media por persona y después entre personas. PASAT, PVTB y
   OA.
4. **Importancias:** modelo ajustado con todo `train`. Coeficientes
   tipificados (logística) o importancias por impureza (RF), en un mapa de
   canal (o región) × banda.

### Comprobación en `val` (mirada 1)

La configuración elegida se entrena con **todo `train`** y se evalúa **una
vez** en las 25 personas de `val`, con las dos versiones:

- acierto pareado de PASAT, PVTB, PASAT + PVTB, PVSAT, OA y OC;
- **H1**: PASAT + PVTB − OA a igual tiempo, con IC95 por bootstrap de personas
  (2000 réplicas);
- H1b: lo mismo con la grabación completa;
- H2: acierto pareado de PASAT y de PVTB frente a 0.5, con IC95;
- **H3**: diferencia de pendientes de la beta posterior relativa, tarea − OA a
  igual tiempo, pareada por persona (t), como en el anexo A del paso 1.

### Qué se considera «bueno» en `val` (decisión del autor, 2026-10-27)

Es la condición del `PLAN.md` (sección 5, punto 3) para preparar el
pre-registro y abrir `test`. `val` es **bueno** si, con la configuración
elegida:

- la **H1** a igual tiempo tiene estimación puntual **> 0**; y
- **PASAT y PVTB** tienen cada una un acierto pareado con el **límite
  inferior del IC95 > 0.5** (H2).

La H3 se reporta sin decidir: con 25 personas, su signo puede fallar por azar.

**Si `val` no es bueno**, el script no reajusta nada: se reporta, se para y se
discute antes de volver a `train`.

### Qué no se hace aquí

- No se mira `test`.
- No se estima la potencia (paso 4).
- No se prueba deep learning (sección 7 del `PLAN.md`; depende de este paso).

### Detalles de ejecución (fijados antes de ejecutar)

- **Dos scripts:** `seleccion.py` solo carga `train`, aplica la regla y
  escribe `configuracion_elegida.json`. `comprobacion_val.py` lee ese archivo
  y es el único que carga `val`. Así la configuración queda congelada antes de
  la mirada.
- La regresión del minuto usa 5 repeticiones (semillas 0–4) y no 20, por
  coste con el RF. Es descriptiva y no decide nada.
- La H2 del criterio de «bueno» se evalúa en la versión a igual tiempo, como
  la H1. La de la grabación completa se reporta al lado.
- La asignación de personas a pliegues no se guarda (contiene identificadores
  de persona). Se reproduce con la semilla: permutación con
  `numpy.random.default_rng(semilla)` de los `persona_uid` de `train`
  ordenados, y pliegue = posición módulo 5.

### Decisiones tomadas con el autor antes de ejecutar

1. La regla de selección: la mayor H1 a igual tiempo.
2. «Bueno» en `val`: H1 > 0 y H2 con IC95 > 0.5 en PASAT y en PVTB.

### Recuento de miradas

- `val`: se prevé **1** mirada en este paso. `test`: 0.

---

## Resultados

Ejecutado el 2026-10-27. `seleccion.py` → `ejecucion_seleccion.log`,
`resultados_seleccion.json`, `configuracion_elegida.json`; `comprobacion_val.py`
→ `ejecucion_val.log`, `resultados_val.json`. Tablas en `tablas/` y figuras en
`figuras/`. **Comprobación superada:** con la partición del paso 1, `rel` +
logística C = 0.1 da exactamente 0.7730 (PASAT) y 0.6472 (OA).

### Veredicto

- **Selección:** `rel` + logística **C = 0.01** + modelo **separado** por
  tarea. Su H1 a igual tiempo en `train` es +0.120 de media sobre 20
  particiones, a 0.005 de la mejor (RF con `rel`, +0.125), y es la más simple
  de las tres empatadas.
- **`val` NO es bueno** según la definición fijada antes de mirar. La H1 es
  > 0 (+0.056), pero el acierto pareado de **PASAT a igual tiempo tiene el
  límite inferior del IC95 en 0.494**, justo por debajo de 0.5. PVTB sí
  cumple (0.575).
- **Con la grabación completa, `val` replica con claridad** (H1b = +0.173
  [+0.062, +0.285]) y la **H3 va en la dirección esperada** en las dos tareas.
- Según el plan, no se reajusta nada: se para y se discute.

### 1. Las 32 configuraciones en `train` (igual tiempo, media de 20 particiones)

`tablas/candidatas_resumen.csv`, `tablas/candidatas_por_semilla.csv` y
`figuras/configuraciones.png`. Extracto:

| Features | Modelo | Esquema | H1 (rango en 20) | PASAT + PVTB | OA |
|---|---|---|---|---|---|
| `rel` | RF | separado | **+0.125** (+0.085, +0.159) | 0.696 | 0.571 |
| **`rel`** | **lr C = 0.01** | **separado** | **+0.120** (+0.097, +0.155) | 0.712 | 0.593 |
| `rel` | lr C = 0.01 | conjunto | +0.116 (+0.079, +0.142) | 0.709 | 0.593 |
| `regiones` | lr C = 0.1 | separado | +0.101 (+0.072, +0.132) | 0.691 | 0.590 |
| `rel` | lr C = 0.1 | separado (paso 1) | +0.096 (+0.070, +0.135) | 0.715 | 0.619 |
| `rel+abs` | RF | separado | +0.043 (+0.006, +0.069) | **0.725** | 0.682 |
| `abs` | lr C = 0.01 | separado | +0.001 (−0.034, +0.038) | 0.663 | 0.662 |

- **La potencia absoluta acierta en las tareas, pero igual en OA.** Con `abs`,
  la H1 es casi 0. La configuración con más acierto en las tareas
  (`rel+abs` + RF, 0.725) es de las peores en H1, porque también acierta en OA
  (0.682). Es la deriva común del paso 1, ahora en el clasificador. Si se
  hubiera elegido por el acierto en las tareas, como decía el plan original,
  se habría escogido justo la configuración que mide la deriva común.
- **La potencia relativa es la que separa la tarea del reposo.** Las cuatro
  mejores usan `rel`.
- El esquema `conjunto` no mejora al `separado` en ningún caso.

### 2. La elegida en `train` (probabilidad media de 20 particiones, IC95 por bootstrap de personas)

| | Igual tiempo | Grabación completa |
|---|---|---|
| PASAT | 0.730 [0.678, 0.776] | 0.789 [0.747, 0.829] |
| PVTB | 0.699 [0.657, 0.740] | 0.758 [0.718, 0.798] |
| PASAT + PVTB | 0.714 [0.681, 0.746] | 0.773 [0.747, 0.800] |
| OA | 0.598 [0.549, 0.646] | 0.598 [0.548, 0.644] |
| **H1** | **+0.116** [+0.051, +0.182] | +0.176 [+0.119, +0.235] |
| PVSAT (descriptiva) | 0.691 [0.641, 0.736] | 0.652 [0.604, 0.699] |
| OC (descriptiva) | 0.667 [0.622, 0.711] | 0.667 [0.624, 0.709] |

- Con C = 0.01, OC (0.667) y PVSAT a igual tiempo (0.691) quedan por encima
  de OA. El reposo con ojos cerrados también cambia de forma detectable dentro
  de la grabación, y OA es el control más bajo, no el único posible. No se
  interpreta aquí.

**Transferencia entre tareas.** A igual tiempo, PASAT → PVTB **0.610**
[0.567, 0.656] y PVTB → PASAT **0.614** [0.572, 0.658]; con la grabación
completa, 0.656 y 0.654. **Una parte del cambio es común a las dos tareas**,
pero la mayor parte es específica de cada una (0.61 frente a 0.70–0.73
dentro de la tarea).

**Regresión del minuto** (grabación completa, bloques de 30 s, Ridge):
ρ medio dentro de la grabación de **+0.350** en PASAT (t = 13.5) y **+0.338** en
PVTB (t = 12.9), frente a **+0.132** en OA (t = 3.3). El orden temporal de la
grabación se recupera mucho mejor en las tareas.

**Importancias** (`figuras/importancias.png`, cada celda es la media de los
coeficientes de PASAT y PVTB; `tablas/importancias.csv`): el patrón dominante
es la **bajada de la beta relativa** en los canales posteriores y temporales
(P8, P7, T7, T8, O1), con subida de la alfa relativa posterior (O1, P7) y de la
theta relativa temporal y frontal izquierda (T7, F7). Es la firma de la H3 y
del paso 1. Los coeficientes de PASAT y PVTB coinciden en la beta posterior
pero no en la theta frontal (AF3, FC6), lo que encaja con la transferencia
parcial.

### 3. Comprobación en `val` (mirada 1; 25 personas)

| | Igual tiempo | Grabación completa |
|---|---|---|
| PASAT | **0.593** [**0.494**, 0.685] | 0.727 [0.613, 0.824] |
| PVTB | 0.659 [0.575, 0.742] | 0.759 [0.682, 0.833] |
| PASAT + PVTB | 0.626 [0.557, 0.690] | 0.743 [0.659, 0.814] |
| OA | 0.570 [0.477, 0.658] | 0.570 [0.479, 0.659] |
| **H1** | **+0.056** [−0.058, +0.165] | **+0.173** [+0.062, +0.285] |
| PVSAT (descriptiva) | 0.690 | 0.616 |
| OC (descriptiva) | 0.705 | 0.705 |

**H3 en `val`** (pendiente de la beta posterior relativa, tarea − OA a igual
tiempo, por persona): PASAT **−0.064** (t = −2.29, p = 0.031) y PVTB **−0.034**
(t = −1.97, p = 0.060). Va en la dirección esperada en las dos, y con más
magnitud que en `train` (−0.014 y −0.022).

**Criterio de «bueno»:** H1 > 0 → sí (+0.056); H2 PASAT → **no** (límite
0.494); H2 PVTB → sí (0.575). **`val` no es bueno.**

Lectura:

- **A igual tiempo, el clasificador pierde mucho en `val`, sobre todo en
  PASAT** (de 0.730 en `train` a 0.593). A igual tiempo cada tercio dura
  ≈ 36 s (≈ 19 ventanas) y el acierto por grabación es mucho más ruidoso que
  con la grabación completa. Con 25 personas, el IC de PASAT tiene una
  anchura de ±0.10.
- **Con la grabación completa todo replica**, con cifras parecidas a `train`
  (PASAT + PVTB 0.743 frente a 0.773; H1b +0.173 frente a +0.176). Pero es la
  versión que confunde tarea con tiempo transcurrido.
- **La H3, que no depende del clasificador, replica a igual tiempo.**
- La caída de PASAT a igual tiempo (0.73 → 0.59) supera lo que se esperaría
  solo por el tamaño de `val` (≈ 2.8 errores típicos del acierto en `val`). No se investiga sin
  hablarlo: cualquier análisis de `val` buscando la causa sería otra mirada.

### Desviaciones del plan

Ninguna en lo planificado. Avisos de rendimiento de pandas (fragmentación) en
la salida de errores; no afectan a las cifras.

### Recuento de miradas

- `val`: **1** (la comprobación planificada, con la configuración congelada).
- `test`: 0.
- `train`: 32 configuraciones × 20 particiones × 2 versiones, más la elegida
  en PVSAT y OC, la transferencia (2 × 2), la regresión del minuto (3
  condiciones) y las importancias.

---

## Reajuste 1 — Medida más eficiente a igual tiempo

> Decidido el 2026-10-27, **después de la mirada 1 a `val`** (no buena), por
> encargo del autor: «la opción más inteligente si lo que queremos es obtener
> resultados concluyentes». Por tanto, **está informado por `val`** y se
> declara así en el pre-registro. Las cifras de `val` de la mirada 2 son de
> desarrollo, no de confirmación.

### Plan (escrito antes de ejecutar)

**Por qué la medida y no otra cosa.** Que `test` sea concluyente depende de la
potencia, y lo que más la limita es la medida. El acierto pareado reduce cada
grabación a 0 o 1, y a igual tiempo los tercios solo tienen ≈ 19 ventanas y
descartan el tercio medio. Esto se veía ya en `train`, sin `val`: a igual
tiempo la H1 tiene un IC más ancho que con la grabación completa, y OA es la
condición más inestable entre particiones (paso 2). Una medida continua por
grabación, que use todas las ventanas, debería tener más señal por unidad de
ruido. **Esto no garantiza un resultado positivo:** solo aumenta la
probabilidad de que `test` sea concluyente en cualquiera de las dos
direcciones.

**Qué queda fijo.** La configuración elegida (`rel` + logística C = 0.01 +
`separado`), la versión a igual tiempo, la familia (PASAT + PVTB), OA como
control, la CV repetida de 20 particiones y el bootstrap de personas. **No se
vuelven a elegir features ni modelo.**

**Medidas candidatas (lista cerrada):**

1. `tercios_pareado`: la actual (primer frente a último tercio; acierto
   pareado).
2. `mitades_pareado`: primera frente a segunda mitad (`t ≤ dur/2` y `t >
   dur/2`). Se entrena y se evalúa con mitades. Usa todas las ventanas, pero
   las mitades están más cerca en el tiempo.
3. `rho_ventanas`: el modelo se entrena con los tercios, como ahora, y
   puntúa **cada ventana** de las grabaciones del pliegue de prueba
   (`decision_function` sobre `log10(pot_rel)` de la ventana, con los canales
   excluidos a 0). Por grabación, la ρ de Spearman entre la puntuación y el
   instante de la ventana. Es continua, va de −1 a 1 y usa todas las ventanas.
   La puntuación es la media de las 20 repeticiones.

Para cada medida, por condición: valor medio por grabación (acierto pareado o
ρ), **H1 = PASAT + PVTB − OA** e IC95 por bootstrap de personas (2000
réplicas, semilla 42), con la predicción media de las 20 particiones.

**Regla de elección.** Se elige la medida con mayor **eficiencia de la H1**:
`z = H1 / error típico bootstrap de H1`, a igual tiempo en `train`. Las
escalas de las medidas son distintas y la z es lo que predice la potencia. Si
la actual (`tercios_pareado`) queda a menos de 0.25 de la mejor z, se mantiene
la actual, para no cambiar sin necesidad.

**«Bueno» en `val`** (mirada 2, misma estructura que la definición anterior):
con la medida elegida, a igual tiempo,

- la H1 tiene estimación puntual > 0; y
- PASAT y PVTB tienen cada una el límite inferior del IC95 por encima del
  azar: > 0.5 si la medida es un acierto pareado y > 0 si es la ρ.

**Disciplina en `val`:** `comprobacion_val_2.py` **solo calcula la medida
elegida** (no las tres), con la misma configuración congelada entrenada con
todo `train`. La grabación completa y la H3 se reportan al lado, como en la
mirada 1.

**Dos scripts**, como antes: `reajuste_medida.py` (solo `train`; escribe
`medida_elegida.json`) y `comprobacion_val_2.py` (lee ese archivo; único que
carga `val`).

**Recuento previsto:** `val`, 2 miradas en total (esta es la segunda). `test`:
0.

### Resultados del reajuste 1

`reajuste_medida.py` → `ejecucion_reajuste.log`, `resultados_reajuste.json`,
`medida_elegida.json`, `tablas/reajuste_medidas.csv`;
`comprobacion_val_2.py` → `ejecucion_val_2.log`, `resultados_val_2.json`.
**Comprobación superada:** `tercios_pareado` reproduce exactamente las cifras
de la selección (0.730 / 0.699 / 0.598; H1 +0.116).

**Elección en `train`** (igual tiempo, predicción media de 20 particiones):

| Medida | PASAT | PVTB | OA | H1 [IC95] | Error típico | **z(H1)** |
|---|---|---|---|---|---|---|
| `tercios_pareado` (actual) | 0.730 | 0.699 | 0.598 | +0.116 [+0.049, +0.181] | 0.034 | 3.44 |
| `mitades_pareado` | 0.698 | 0.664 | 0.602 | +0.079 [+0.018, +0.137] | 0.030 | 2.62 |
| **`rho_ventanas`** | ρ = 0.143 | 0.136 | 0.063 | **+0.077** [+0.046, +0.108] | **0.016** | **4.89** |

**Elegida: `rho_ventanas`.** Su z supera en 1.45 a la de la actual: más del
doble de información por persona (z² 23.9 frente a 11.8). Las mitades son
peores que los tercios, porque acercar los segmentos en el tiempo cuesta más
efecto del que se gana en ventanas. Con la grabación completa, el orden es el
mismo (z = 6.05 frente a 5.83).

**`val`, mirada 2** (solo la medida elegida; ρ media por grabación, IC95 por
bootstrap de personas):

| | Igual tiempo | Grabación completa |
|---|---|---|
| PASAT | **+0.097** [+0.051, +0.146] | +0.119 [+0.074, +0.159] |
| PVTB | **+0.098** [+0.056, +0.139] | +0.148 [+0.102, +0.194] |
| PASAT + PVTB | +0.097 [+0.061, +0.131] | +0.133 [+0.098, +0.167] |
| OA | +0.039 [−0.000, +0.079] | +0.039 [−0.000, +0.079] |
| **H1** | **+0.059** [+0.003, +0.116], z = 2.00 | +0.095 [+0.046, +0.138], z = 3.98 |
| PVSAT (descriptiva) | +0.105 | +0.060 |
| OC (descriptiva) | +0.092 | +0.092 |

H3 en `val`: la misma que en la mirada 1 (PASAT −0.064, p = 0.031; PVTB
−0.034, p = 0.060).

**Criterio de «bueno»:** H1 > 0 → sí (+0.059); H2 PASAT, límite +0.051 > 0 →
sí; H2 PVTB, límite +0.056 > 0 → sí. **`val` es bueno.**

**Lectura y cautelas:**

- **Con la medida eficiente, `val` replica a igual tiempo.** Las dos tareas
  ordenan sus ventanas en el tiempo por encima del azar, y más que OA.
- **La mirada 2 está informada por la 1.** La medida se eligió en `train` por
  un criterio de eficiencia que no mira `val`, pero la decisión de buscar una
  medida mejor se tomó porque la mirada 1 falló. Las cifras de `val` son de
  desarrollo y se reportan con el recuento de miradas (2).
- **La H1 en `val` va justa** (límite inferior +0.003). Una estimación gruesa
  de potencia para `test` (28 personas), escalando la z de `train`
  (4.89 × √(28/121) ≈ 2.35), da del orden del 75 % con α = 0.05 unilateral. El
  paso 4 la estima bien.
- **OC se comporta como las tareas** (ρ = 0.092 en `val`; acierto pareado 0.667
  en `train`). El reposo con ojos cerrados también cambia de forma ordenada
  dentro de la grabación, y OA es el control más bajo, no el único. OC es la
  primera grabación de la sesión, justo después de colocar el casco: encaja
  con una deriva de asentamiento. Es un dato importante para el paso 4
  (tiempo en tarea frente a tiempo desde la colocación) y para la discusión.
  No cambia la H1, que se fijó frente a OA antes de ver esto.

### Recuento de miradas (final del paso 3)

- `val`: **2** (mirada 1 con `tercios_pareado`, no buena; mirada 2 con
  `rho_ventanas`, buena).
- `test`: 0.
