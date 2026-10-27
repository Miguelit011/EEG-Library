# Pre-registro — fatiga aguda (tiempo en tarea) en el conjunto de test

> **Estado: BORRADOR (2026-10-27).** Queda **FIJADO** en el commit que lo
> incluye junto con el guion `evaluacion_test.py`. Ese commit lo hace el
> autor, antes de cualquier lectura de `test` para esta pregunta. Desde
> entonces no se modifica: cualquier cambio posterior se anota en la sección 11
> («Desviaciones») con fecha, motivo y si ocurrió antes o después de ver algún
> resultado de `test`.
>
> **El conjunto de test no se ha abierto para esta pregunta.** Se abrió una vez
> el 2026-10-07 para el VAS y la condición (`PREREGISTRO_TEST.md`), con otra
> etiqueta y otra pregunta. La única exposición de esta línea es la persona
> declarada en la sección 2. Todas las cifras de referencia de este documento
> salen de `train` + `val`.

---

## 1. Qué se confirma y qué no

**Pregunta.** ¿Cambia el EEG de forma sistemática con el tiempo en la tarea
(efecto compatible con fatiga aguda o decremento de vigilancia), en personas
que el modelo no ha visto, más de lo que cambia en reposo con ojos abiertos?

**Lo que no se afirma.** Que el EEG «mide la fatiga». No hay medida subjetiva
ni conductual dentro de la grabación (`PLAN.md`, 1.3 y 4.6).

El test se usa **una sola vez**: todo lo que se ejecuta está declarado aquí, se
ejecuta de una pasada y se reporta entero, salga lo que salga.

| Familia | Contenido | Control del error |
|---|---|---|
| **Principal** | **H1b → H1**, en secuencia fija | H1b a α = 0.05 unilateral; **solo si confirma**, H1 a α = 0.05 unilateral. La secuencia controla el error de familia |
| Secundaria 1 | H2 (PASAT, PVTB) | Holm sobre 2, α = 0.05 |
| Secundaria 2 | H3 (PASAT, PVTB) | Holm sobre 2, α = 0.05 |
| Secundaria 3 | H4 | Una prueba, α = 0.05 unilateral |

Todo lo que no figura como prueba es **descriptivo**: se reporta con su IC, sin
p y sin veredicto.

**Se renuncia por escrito al deep learning en este test** (`PLAN.md`, sección
7). Si se desarrolla después, su evaluación en `test` será exploratoria.

## 2. Datos

| | |
|---|---|
| Partición | `split` de `matriz_rf.parquet` (corregida en la Fase 12). Nunca la de `features_all.parquet` |
| Test | **28 personas**; ≈ 102–105 grabaciones etiquetadas por tarea (PASAT 102, PVTB 103, OA 103 según la matriz) |
| Entrenamiento del modelo final | `train` + `val` completos: **146 personas** |
| Ventanas y potencias | `features_all.parquet` (Fases 9–11); canales válidos de `procesamiento_log.csv` (Fase 11.5) |
| Movimiento y contacto (H4) | `MOT.*` y `CQ.*` de los EDF de `dataset_final/` (solo lectura), como en el paso 2 |

Se registran los SHA-256 de todos los archivos de datos, de este documento, de
`configuracion_elegida.json`, de `medida_elegida.json` y del guion.

**Exclusiones** (las mismas de todo el desarrollo): grabaciones sin
`persona_uid`, grabaciones con < 10 ventanas y, en la versión a igual tiempo,
grabaciones de tarea cuya sesión no tiene OA. En la H4, además, las grabaciones
sin acelerómetro ni cuaterniones (exports de 60 o 64 señales de M2). Todo se
cuenta en el inventario.

**Persona expuesta.** Los sondeos del 2026-10-26 filtraron por la `split` de
`features_all` y vieron como `train` a una persona que es de `test` (6
grabaciones; `PLAN.md`, 3.5). **Se mantiene en `test`.** Excluirla a
posteriori sería también un grado de libertad. Se añade una sensibilidad sin
ella (sección 6).

## 3. Lo que queda fijado

Todo sale de `train` + `val` (pasos 1–4) y está en
`03_clasificador/configuracion_elegida.json` y `medida_elegida.json`.

- **Features:** las 56 `log10(pot_rel)` (14 canales × 4 bandas), medias por
  tercio de grabación, **centradas por grabación**. Los canales excluidos por
  calidad se ponen a 0 tras el centrado.
- **Modelo:** `StandardScaler` + `LogisticRegression(C=0.01, max_iter=3000)`,
  **uno por condición** (PASAT, PVTB, OA; y PVSAT y OC, descriptivos).
  Etiqueta: primer tercio (0) frente a último tercio (1); el tercio medio no se
  entrena.
- **Entrenamiento final:** con todo `train` + `val`, una vez por versión: la
  versión a igual tiempo se entrena con las grabaciones recortadas y la
  completa con las completas.
- **Versión a igual tiempo:** cada grabación de tarea se recorta a la duración
  útil (`tmin_s` máximo) del OA de su misma sesión. OA y OC no se recortan.
- **Medida `rho_ventanas`:** el modelo puntúa con `decision_function` **cada
  ventana** de la grabación (las 56 `log10(pot_rel)` de la ventana, centradas
  por grabación y con los canales excluidos a 0). Por grabación, la **ρ de
  Spearman** entre la puntuación y el instante de la ventana (`tmin_s`).
- **Semilla:** 42. **Bootstrap:** remuestreando **personas** de test, 10 000
  réplicas, IC95 de percentiles. p unilateral bootstrap: `(#réplicas ≤ 0 + 1) /
  (B + 1)`.

## 4. Pruebas principales (secuencia fija)

**Estadístico:** `Δρ = ρ media de las grabaciones de PASAT + PVTB − ρ media de
las grabaciones de OA`, con la ρ de `rho_ventanas`. Las réplicas bootstrap de
H1b y H1 se generan en la misma remuestra (pareadas).

| Paso | Id | Versión | H0 / H1 | Regla |
|---|---|---|---|---|
| 1 | **H1b** | Grabación completa | Δρ ≤ 0 / Δρ > 0 | **CONFIRMA** si p < 0.05; si no, **NO CONCLUYE** |
| 2 | **H1** | **Igual tiempo** | Δρ ≤ 0 / Δρ > 0 | Solo si H1b confirma: **CONFIRMA** si p < 0.05; si no, **NO CONCLUYE**. Si H1b no confirma: **NO SE CONTRASTA** y se reporta como descriptiva |

**Lectura de los resultados posibles:**

- H1b y H1 confirman: el EEG ordena en el tiempo las grabaciones de tarea mejor
  que las de reposo, **también a igual tiempo transcurrido**.
- Solo H1b confirma: las tareas superan al reposo, pero no se puede separar de
  que duran más.
- Ninguna confirma: no hay evidencia confirmatoria; se reporta el desarrollo
  como tal.

**Referencias de desarrollo** (no son predicciones). En `val` (25 personas,
mirada 2): H1 = +0.059 [+0.003, +0.116]; H1b = +0.095 [+0.046, +0.138]. Potencia
estimada en el paso 4: **H1b 0.92** (0.83 con un efecto un 25 % menor) y **H1
0.75** (0.58).

## 5. Pruebas secundarias

| Id | Qué | Estadístico y prueba | Regla |
|---|---|---|---|
| **H2** | PASAT y PVTB ordenan sus ventanas en el tiempo (igual tiempo) | ρ media de `rho_ventanas` por tarea; p unilateral bootstrap (H0: ρ ≤ 0) | Holm sobre 2; CONFIRMA si p de Holm < 0.05 |
| **H3** | Firma específica: la beta posterior relativa baja más en la tarea que en OA (igual tiempo) | Pendiente por minuto de `log10` de la potencia relativa beta media de P7, O1, O2 y P8 (canales válidos), por grabación; media por persona; diferencia tarea − OA por persona; t de una muestra unilateral (H1: < 0) | Holm sobre 2. **Potencia estimada 0.26–0.38:** un NO CONCLUYE no indica ausencia de efecto |
| **H4** | El efecto supera a la referencia de artefactos (igual tiempo) | Acierto pareado de tercios del EEG (configuración fijada) menos el de un clasificador de solo movimiento y contacto (`mov_acc`, `mov_ang`, `cq_validos`, `cq_overall` por tercio, centrados por grabación, `StandardScaler` + `LogisticRegression(C=0.1)`, como en el paso 2), en las mismas grabaciones de PASAT + PVTB; p unilateral bootstrap (H0: ≤ 0) | CONFIRMA si p < 0.05 |

## 6. Descriptivos y sensibilidades (sin veredicto)

- ρ media (`rho_ventanas`) y acierto pareado de tercios, con IC95, de PASAT,
  PVTB, OA, PVSAT y OC, en las dos versiones.
- **Sensibilidad 1:** H1b, H1 y H2 sin la persona expuesta (sección 2).
- **Sensibilidad 2:** H1b, H1 y H2 sin las grabaciones grabadas a 128 Hz
  (`sfreq_original` = 128 en el log; M2 de C1 y C8).
- Si alguna sensibilidad cambia el veredicto que tendría la prueba, se dice
  explícitamente en el informe, pero **el veredicto oficial es el de las
  secciones 4 y 5**.

## 7. Datos en bruto que se guardan

En `<EEG_DATA>/processed/fatiga_aguda_test_<fecha-hora>/`, **excepción
declarada a la regla de solo lectura de `EEG-Data`** (decisión del autor,
2026-10-27, como en el test del 2026-10-07). El guion se niega a escribir si la
carpeta ya existe.

| Artefacto | Contenido |
|---|---|
| `predicciones_por_grabacion.parquet` | ρ por grabación (`rho_ventanas`) y acierto pareado de tercios, por versión y condición, con `archivo_origen` y `persona_uid` |
| `h4_por_grabacion.parquet` | Acierto del EEG y de solo artefactos por grabación |
| `bootstrap/*.npy` | Las 10 000 réplicas de H1b, H1, H2 (pareadas) y de H4 |
| `modelos/*.joblib` | Los modelos finales por versión y condición |
| `resultados.json` | Todas las cifras, p, p de Holm, veredictos, inventario, metadatos |
| `ejecucion.log` | Salida completa |

En el repositorio solo quedan `resultados_test.json` (sin identificadores de
persona) y `ejecucion_test.log`.

## 8. Metadatos de la ejecución

Commit y comprobación de que `git status` está limpio (el guion se niega a
correr si no), SHA-256 del guion, de este documento y de los datos, versiones
de Python, scikit-learn, NumPy y pandas, sistema operativo y CPU, fecha y hora
de inicio y fin, y recuentos de personas y grabaciones.

## 9. Recuento de lo explorado en `train` + `val`

| Fase | Qué se miró |
|---|---|
| Sondeos (2026-10-26, solo `train` con la partición antigua) | Pendientes en 7 condiciones; 4 clasificadores × 2 variantes; fatigabilidad frente a 4 variables (`PLAN.md`, 3.5) |
| Paso 1 + anexo A (`train`) | Pendientes 7 condiciones × 8 magnitudes × 2 versiones de canales; clasificador 4 condiciones × 4 variantes; curvas; comparación a igual tiempo |
| Paso 2 (`train`) | Tendencias de 5 medidas de artefacto; 8 modelos × 4 condiciones × 2 versiones; estabilidad de 20 particiones |
| Paso 3 (`train`) | **32 configuraciones** (4 features × 4 modelos × 2 esquemas) × 20 particiones × 2 versiones; transferencia, regresión del minuto, importancias; **3 medidas** en el reajuste 1 |
| `val` | **Mirada 1** (tercios: no buena) → **reajuste de la medida en `train`** → **mirada 2** (`rho_ventanas`: buena). Uso 3: simulación de potencia (paso 4). Uso 4: ensayo general (sección 10), que además mostró la H4 y las sensibilidades en `val` |
| Paso 4 (`train`) | Matriz de transferencia 7 × 7, dientes de sierra, desaceleración y modelo de efectos fijos |

Decisiones tomadas **después de ver resultados**, que el lector debe ponderar:

- Familia PASAT + PVTB (PVSAT fuera), por los resultados del paso 1 en `train`.
- Contraste a igual tiempo, propuesto tras ver en `train` que OA dura menos.
- Medida `rho_ventanas`: elegida en `train` por eficiencia estadística, pero
  buscada **porque la mirada 1 a `val` falló**. Las cifras de `val` son de
  desarrollo y optimistas.
- Secuencia H1b → H1, elegida a la vista de las potencias del paso 4 (sin datos
  de `test`).

## 10. Ejecución

1. **Ensayo general** del mismo guion (`--ensayo`): entrena con `train` y
   evalúa en `val`. Comprueba que corre de principio a fin y que reproduce las
   cifras de la mirada 2 a `val`. No es un resultado ni una mirada nueva (las
   cifras de `val` ya se vieron). Solo se corrigen errores de código.
2. **Commit** del guion y de este documento (lo hace el autor).
3. **Una sola ejecución** en `test`: `evaluacion_test.py --test --permitir-test`.
4. Si la ejecución falla antes de terminar, se corrige solo el error de código,
   se anota en la sección 11 (incluido lo que se hubiera visto) y se vuelve a
   ejecutar completa.
5. Se reporta **todo**: pruebas con sus veredictos, descriptivos y
   sensibilidades.

**Ensayo general, hecho el 2026-10-27 antes del commit** (`ejecucion_ensayo.log`,
`resultados_ensayo.json`). Corre de principio a fin y reproduce **exactamente**
las cifras puntuales ya vistas en `val`: las de la mirada 2 (H1 +0.0587, H1b
+0.0947, ρ PASAT +0.097, PVTB +0.098, OA +0.039), la H3 (−0.0635 y −0.0343) y
el acierto de tercios de la mirada 1 (0.593, 0.659 y 0.570). **Dos cifras de
`val` no se habían visto antes:** la H4 (+0.089 [−0.005, +0.185]) y las dos
sensibilidades. Se declaran aquí. No se cambia nada por ellas. La potencia de
la H4 se había estimado a ojo con `train` («cercana al 100 %»), pero en `val`
sale mucho más justa, así que su potencia real en `test` es incierta. Los
artefactos del ensayo (carpeta temporal) se borraron tras comprobarlo.

## 11. Desviaciones

*(Vacía hasta la ejecución. Cada entrada: fecha, qué cambió respecto a este
documento, por qué y si ocurrió antes o después de ver algún resultado de
test.)*
