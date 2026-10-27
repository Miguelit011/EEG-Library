# Paso 4 — Potencia y tiempo desde la colocación

> Línea: fatiga aguda (tiempo en tarea). Fecha: 2026-10-27. Dos partes:
> **potencia** (con `train` + `val`, como prevé el `PLAN.md`) y **controles del
> tiempo desde la colocación** (solo `train`). `test` no se carga.

## Plan (escrito antes de ejecutar)

Esta sección se redactó **antes** de escribir y ejecutar los scripts y no se
modifica después. Los resultados van en las secciones siguientes.

### Qué tiene que salir de aquí

1. La **potencia** de cada hipótesis en `test` (28 personas), para decidir con
   el autor, antes del pre-registro, cuál es la principal (`PLAN.md`, sección
   5, punto 5).
2. Cuánto del cambio dentro de la grabación puede atribuirse al **tiempo desde
   la colocación del casco** y no al tiempo en la tarea. No se resuelve del
   todo (el orden del protocolo es fijo), pero se acota. El paso 3 añadió un
   indicio a favor de la deriva de asentamiento: OC, la primera grabación,
   cambia casi como las tareas.

Todo lo que se fija aquí es lo del paso 3: 56 `log10(pot_rel)` + logística
C = 0.01, un modelo por condición, entrenado con los tercios; medida
`rho_ventanas`.

---

### Parte A — Potencia (`potencia.py`, `train` + `val`)

**Simulación del `test`.** 2000 réplicas (semilla 42). En cada una:

1. Se sacan 28 personas al azar de las 146 de `train` + `val` como pseudo-test.
2. Con las 118 restantes se entrenan los modelos de PASAT, PVTB y OA (y de
   PVSAT y OC, descriptivos). Es algo pesimista: en el `test` real se
   entrenará con 146.
3. Se calcula la H1 en las 28 personas exactamente como se hará en `test` y se
   aplica su contraste.

**Contrastes simulados** (los de la sección 8 del `PLAN.md`):

| Id | Qué se mide en las 28 personas | Se rechaza H0 si |
|---|---|---|
| **H1** | ρ media de PASAT + PVTB − ρ media de OA, a igual tiempo | percentil 5 del bootstrap de personas (2000 réplicas internas) > 0 |
| H1b | Lo mismo con la grabación completa | Ídem |
| H2 | ρ media de PASAT y de PVTB (igual tiempo) | p unilateral bootstrap (fracción de réplicas ≤ 0), Holm sobre 2, α = 0.05 |
| H3 | Pendiente de la beta posterior relativa, tarea − OA por persona (igual tiempo) | t unilateral (diferencia < 0), Holm sobre 2, α = 0.05 |

En el pre-registro la H1 usará 10 000 réplicas bootstrap; aquí 2000 bastan para
estimar la potencia.

**Salidas:** potencia de cada contraste (fracción de réplicas que rechazan),
con su error de Monte Carlo, y la distribución de la H1 en muestras de 28.

**Cautelas que se declaran:** las magnitudes del efecto salen de los mismos
datos con los que se eligió la configuración y la medida, así que la potencia
estimada es **optimista** (sesgo del ganador). Se reporta también la potencia
que resultaría con un efecto un 25 % menor, para dar un margen.

**Uso de `val`:** esta parte usa las personas de `val`, como prevé el
`PLAN.md`. No evalúa ninguna configuración nueva ni decide nada sobre ella; se
anota igualmente en el recuento como uso de `val` para la potencia.

**Decisión de la hipótesis principal.** No hay regla automática: con las
potencias delante, se decide con el autor antes del pre-registro. La
referencia habitual es ≥ 80 %.

---

### Parte B — Tiempo en tarea frente a tiempo desde la colocación (`controles.py`, solo `train`)

Las siete condiciones (OC, OA, STROOP, PASAT, PVTB, MSIT y PVSAT), con la
grabación completa y ≥ 10 ventanas. Tiempo desde el inicio de la sesión:
`t_ses = (timestamp_iso − mínimo de la sesión) + 5 s + tmin_s` (en minutos;
los 5 s son el recorte inicial). Las grabaciones sin marca de tiempo válida
quedan fuera de B2–B4 y se cuentan.

Magnitudes (regiones del sondeo, canales válidos): alfa posterior absoluta,
theta frontal absoluta y beta posterior relativa (la de la H3).

**B1. Especificidad: matriz de transferencia.** Con la CV repetida de 20
particiones (los pliegues del paso 3), el modelo entrenado con los tercios de
la condición A puntúa las ventanas de la condición B de las personas del
pliegue de prueba. Se calcula `rho_ventanas` para las 7 × 7 combinaciones.

- Si el cambio fuera una deriva común del hardware, los modelos de PASAT y
  PVTB ordenarían igual de bien las ventanas de OC, OA, STROOP o MSIT que las
  suyas.
- Si es específico, la diagonal domina.
- Se reportan la matriz, el IC95 de las celdas clave y la diferencia entre
  «propia» y «ajena».

**B2. Dientes de sierra.** Para cada par consecutivo del protocolo (OC → OA →
STROOP → PASAT → PVTB → MSIT → PVSAT) dentro de la sesión:

- cambio **dentro** de la tarea k: últimos 30 s menos primeros 30 s;
- **salto** al pasar a k+1: primeros 30 s de k+1 menos últimos 30 s de k.

Media por persona, t frente a 0, y el hueco mediano entre archivos. Una
deriva monótona del hardware predice saltos ≥ 0 (la deriva sigue); una
«recuperación» al cambiar de tarea predice saltos de signo contrario al cambio
dentro. Confusor declarado: el cambio de tarea cambia la demanda y el nivel.
Se reporta como indicio.

**B3. Desaceleración.** Pendiente por minuto de cada magnitud en cada
condición (media por persona, IC95) frente al momento mediano de inicio de la
condición en la sesión. Una deriva de asentamiento debería ser más fuerte al
principio de la sesión (OC, OA) y frenarse después. Confusor: las
condiciones también son tareas distintas.

**B4. Modelo de ventanas con efectos fijos.** Para cada magnitud:

`y = efecto fijo de sesión + efecto fijo de condición + β_c · t_rec (una pendiente por condición) + γ · t_ses`

Se ajusta por MCO con la transformación *within* de sesión (los efectos de
condición, como variables indicadoras) y con errores típicos robustos
agrupados por persona.

- `γ` mide el efecto del tiempo de sesión.
- `β_c` mide el cambio dentro de la grabación que **no** explica el tiempo de
  sesión.
- `γ` solo se identifica por la variación entre sesiones del momento de inicio
  de cada condición (dentro de la grabación, `t_ses` y `t_rec` avanzan juntos),
  así que el ajuste puede ser frágil. Se reportan la correlación entre los
  regresores tras la transformación, la variación disponible de los inicios y
  una variante con una sola pendiente común de `t_rec`.

### Lectura fijada antes de mirar

No hay criterio de continuidad: el siguiente paso es el pre-registro, y antes
hay que hablar de la potencia. Para la parte B, la lectura se anticipa así:

- **A favor del tiempo en tarea:** diagonal dominante en B1 (las tareas no
  ordenan el reposo) y `β` de PASAT y PVTB > 0 (o < 0 para la beta relativa)
  con `γ` pequeño en B4.
- **A favor de la colocación:** modelos que transfieren a OC y OA tan bien
  como a sí mismos, `γ` grande con `β_c` cercanos a 0 y pendientes que se
  frenan a lo largo de la sesión.
- Lo más probable es una mezcla. Se reporta tal cual y la limitación se
  declara en la memoria: solo un orden contrabalanceado la resolvería.

### Qué no se hace aquí

- No se mira `test` ni se cambia la configuración, la medida ni las hipótesis.
- La parte B no usa `val`.

### Recuento de miradas

- `val`: la parte A usa sus personas para la potencia (no es una evaluación
  de la configuración). Se anota como uso 3 de `val`.
- `test`: 0.

---

## Resultados

Ejecutado el 2026-10-27. `potencia.py` → `ejecucion_potencia.log`,
`resultados_potencia.json`, `tablas/potencia_*.csv`; `controles.py` →
`ejecucion_controles.log`, `resultados_controles.json`, `tablas/` y
`figuras/`. Todas las grabaciones de `train` tienen marca de tiempo válida
(los 7 archivos sin ella no son de `train`).

### Parte A — Potencia en `test` (28 personas; 2000 réplicas; error de Monte Carlo ≈ ±0.02)

| Contraste | Efecto estimado | Efecto un 25 % menor | H1 en muestras de 28: mediana [p5, p95] |
|---|---|---|---|
| **H1** (igual tiempo) | **0.75** | 0.58 | +0.071 [+0.025, +0.122] |
| **H1b** (grabación completa) | **0.92** | 0.83 | +0.090 [+0.044, +0.137] |
| H2 PASAT (ρ > 0, Holm) | 1.00 | 1.00 | — |
| H2 PVTB (ρ > 0, Holm) | 1.00 | 1.00 | — |
| H3 PASAT (Holm) | 0.26 | — | — |
| H3 PVTB (Holm) | 0.38 | — | — |

- **La H1 a igual tiempo tiene una potencia del 75 %**, que coincide con la
  estimación gruesa del paso 3, y del 58 % si el efecto real es un 25 % menor.
  No llega al 80 % habitual.
- **La H1b (grabación completa) llega al 92 % (83 % en el escenario
  prudente)**, pero es la versión que confunde tarea con tiempo transcurrido.
- **La H2 está asegurada:** las dos tareas ordenan sus ventanas en el tiempo
  en prácticamente todas las réplicas.
- **La H3 tiene poca potencia (26–38 %).** Con 28 personas, un resultado
  nulo en la H3 no se podrá interpretar como ausencia de efecto.
- La potencia es optimista: el efecto sale de los datos con los que se
  eligieron la configuración y la medida. El escenario del 25 % menor da el
  margen.

### Parte B — Tiempo en tarea frente a tiempo desde la colocación (solo `train`)

**B1. Especificidad (`figuras/transferencia.png`, `tablas/transferencia*.csv`).**
ρ media cuando el modelo de la fila puntúa las ventanas de la columna:

| Modelo ↓ / puntuada → | OC | OA | STROOP | PASAT | PVTB | MSIT | PVSAT |
|---|---|---|---|---|---|---|---|
| PASAT | 0.063 | 0.019 | 0.073 | **0.159** | 0.085 | 0.067 | 0.046 |
| PVTB | −0.013 | 0.027 | 0.079 | 0.094 | **0.148** | 0.015 | 0.009 |
| OC | **0.108** | 0.009 | 0.053 | 0.082 | 0.006 | 0.096 | 0.053 |
| OA | 0.006 | **0.063** | 0.045 | −0.007 | 0.030 | 0.034 | 0.013 |

- **La diagonal domina en PASAT y PVTB.** Su modelo ordena sus propias
  ventanas mucho mejor que las de cualquier otra condición: la diferencia
  propia − ajena va de +0.054 a +0.161, con todos los IC95 por encima de 0.
  **El modelo de PVTB no ordena nada en OC (−0.013) ni en OA (0.027).** Una
  deriva común del hardware haría que todos los modelos transfirieran igual.
- PASAT y PVTB comparten una parte (0.085 y 0.094 en la transferencia
  cruzada). STROOP y MSIT, dos tareas de interferencia cortas, transfieren
  mucho entre sí (0.161 y 0.139).
- **OC cambia, pero de otra manera:** su modelo no transfiere a OA ni a PVTB.
  En OC, la alfa y la theta absolutas **bajan** y la beta relativa **sube**
  (B3), justo al revés que en las tareas. **El cambio de OC que inquietaba en el
  paso 3 no es la firma de las tareas.**

**B2. Dientes de sierra (`tablas/dientes_de_sierra.csv`).** Cambio dentro de
la tarea k (últimos 30 s − primeros 30 s) y salto al empezar k+1:

| Transición | alfa post abs: dentro / salto | beta post rel: dentro / salto |
|---|---|---|
| OA → STROOP | +0.038 (t = +4.9) / **−0.086** (−4.3) | −0.006 (−0.9) / +0.108 (+9.9) |
| STROOP → PASAT | +0.032 (+5.9) / **−0.065** (−7.3) | −0.014 (−3.0) / **+0.024** (+3.1) |
| PASAT → PVTB | +0.064 (+8.9) / −0.012 (−1.2) | −0.037 (−4.8) / **+0.040** (+4.5) |
| PVTB → MSIT | +0.064 (+9.2) / **−0.049** (−4.3) | −0.036 (−6.2) / **+0.036** (+4.4) |
| MSIT → PVSAT | +0.017 (+3.4) / −0.027 (−3.2) | +0.000 (+0.1) / −0.014 (−1.8) |

Huecos medianos entre archivos: 14–37 s. **Hay dientes de sierra:** dentro de
cada tarea la alfa sube y la beta relativa baja, y al empezar la siguiente
tarea las dos **retroceden** en parte. Una deriva monótona del hardware no se
recuperaría en 20–30 s. Hay dos confusores declarados: el cambio de demanda
entre tareas mueve el nivel (OC → OA es el caso extremo, −0.56 en alfa al
abrir los ojos), y entre archivos hay instrucciones e interacción. El
«reinicio» puede ser de la tarea o de la pausa.

**B3. Desaceleración (`figuras/desaceleracion.png`).** No hay un patrón de
asentamiento que se frene a lo largo de la sesión. Las pendientes de la alfa
absoluta son +0.027 en OA (minuto 2.4 de la sesión), +0.065 en STROOP (5.0),
+0.024 en PASAT (6.8) y en PVTB (10.4) y +0.016 en PVSAT (15.6). La beta
relativa solo baja en las tareas (STROOP, PASAT y PVTB); en OA, MSIT y PVSAT,
no. OC va al revés en todo.

**B4. Efectos fijos (`tablas/efectos_fijos.csv`).** Los inicios de cada
condición varían entre sesiones con una DE de ≈ 4.8 min (OC 1.2). Tras la
transformación *within*, la correlación entre `t_ses` y `t_rec` es 0.29: el
modelo está identificado, aunque no con holgura.

| Magnitud | `t_ses` (por minuto de sesión) | `t_rec` PASAT | `t_rec` PVTB | `t_rec` OA |
|---|---|---|---|---|
| alfa post abs | −0.0046 (t = −1.7) | **+0.027 (t = 6.6)** | **+0.027 (7.6)** | +0.032 (5.1) |
| theta front abs | −0.0035 (−1.7) | +0.030 (7.7) | +0.022 (6.6) | +0.023 (3.2) |
| beta post rel | +0.0017 (+0.8) | **−0.013 (−3.8)** | **−0.016 (−5.5)** | −0.004 (−0.7) |

- **El tiempo de sesión no explica el cambio dentro de la grabación.** Su
  coeficiente es pequeño, no significativo y, en alfa y theta, **de signo
  contrario** (con la sesión baja un poco). Las pendientes dentro de la
  grabación de PASAT y PVTB se mantienen intactas al controlarlo.

### Lectura conjunta

- **Lo que dice la parte B, con las tres pruebas en la misma dirección, es
  que el cambio de PASAT y PVTB es del tiempo en la tarea y no del tiempo
  desde la colocación:**
  - es específico de la tarea (B1);
  - se reinicia al cambiar de tarea (B2);
  - no lo explica el tiempo de sesión (B4).
- Sigue sin ser concluyente por diseño: el orden fijo y las pausas entre
  archivos no permiten descartarlo del todo. La limitación se declara.
- La subida de la **alfa absoluta** es la parte común: también aparece en OA
  y, en B4, OA tiene la misma pendiente. Lo específico de las tareas es la
  **beta relativa**, como ya decía el paso 1.
- **Decisión pendiente para el pre-registro:** la H1 a igual tiempo tiene una
  potencia del 75 % (58 % en el escenario prudente) y la H1b del 92 % (83 %).
  Opciones en el resumen al autor.
