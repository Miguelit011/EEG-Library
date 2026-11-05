# Resultados — el efecto del tiempo en tarea dentro de la grabación

> Borrador para el capítulo 6. Cada bloque dice de qué datos sale y con qué
> estatus:
> - **desarrollo** (`train`/`val`, con decisiones tomadas a la vista de los
>   datos);
> - **confirmatorio** (`test`, una sola vez, pre-registrado);
> - **exploratorio** (posterior al test).
>
> Las cifras están copiadas de los `INFORME.md` y JSON indicados.

## 1. Desarrollo: qué cambia y frente a qué (`train`, 120 personas)

*Fuente: `01_descriptivo/INFORME.md` y su anexo A. Figuras F1 y F2.*

**El EEG distingue el principio del final de la misma grabación.** Con el
centrado por grabación y solo canales válidos, el acierto pareado de tercios
es:

| Condición | Grabación completa | Igual tiempo que OA |
|---|---|---|
| PASAT | 0.773 [0.735, 0.810] | 0.739 [0.691, 0.785] |
| PVTB | 0.758 [0.717, 0.797] | 0.715 [0.673, 0.755] |
| PVSAT | 0.645 [0.597, 0.693] | 0.632 [0.584, 0.677] |
| OA (control) | 0.647 [0.601, 0.691] | (sin recortar) |

- **El reposo también cambia** (OA 0.647, muy por encima de 0.5), así que el
  dato relevante es la diferencia con OA.
- **Con la grabación completa:** PASAT − OA = +0.126 [+0.069, +0.183] y
  PVTB − OA = +0.111 [+0.047, +0.170].
- **A igual tiempo transcurrido** se mantienen unos dos tercios: +0.091
  [+0.021, +0.160] y +0.067 [+0.006, +0.127].
- **PVSAT no se distingue de OA.** Por eso la familia de tareas se fijó en
  PASAT + PVTB, una decisión tomada en `train` y declarada.

**Qué cambia (figura F1).** Pendientes por minuto, a igual tiempo y pareadas
con el OA de la misma persona:

- **Alfa y theta absolutas:** suben en las tareas, pero **también en el
  reposo**, con la misma inclinación (alfa posterior: +0.037 por minuto en las
  tareas frente a +0.027 en OA; la diferencia no es significativa).
- **La beta posterior relativa:** baja en PASAT (−0.018 por minuto) y en PVTB
  (−0.026), y apenas en OA (−0.003). Es **el único componente que separa la
  tarea del reposo** (diferencia frente a OA: PASAT −0.014, p = 0.037; PVTB
  −0.022, p = 0.0009).

## 2. Desarrollo: no es movimiento ni contacto (`train`)

*Fuente: `02_artefactos/INFORME.md`. Figura F3.*

- **El movimiento de la cabeza no crece con el tiempo en tarea más que en
  reposo.** En PASAT, la velocidad angular sube lo mismo que en OA, y en PVTB
  baja respecto a OA.
- **Un clasificador que solo ve el movimiento y la calidad de contacto apenas
  supera el azar:** 0.542 en PASAT y 0.568 en PVTB, frente a 0.721 y 0.727 del
  EEG, a igual tiempo.
- **El EEG residualizado** por esas medidas conserva la ventaja sobre OA:
  +0.120 [+0.061, +0.180].
- **Límites:**
  - la calidad de contacto del casco está saturada (el 99.4 % de las ventanas
    tiene el valor máximo), así que su control es débil;
  - no se controla la actividad muscular.

## 3. Desarrollo: tiempo en la tarea, no tiempo desde la colocación (`train`)

*Fuente: `04_potencia_y_controles/INFORME.md`. Figura F4.*

El protocolo tiene un orden fijo (OC → OA → STROOP → PASAT → PVTB → MSIT →
PVSAT), así que el tiempo en la tarea y el tiempo desde que se colocó el casco
no se pueden separar del todo. Tres comprobaciones parciales apuntan en la
misma dirección:

- **Especificidad (F4):** el modelo de cada tarea ordena sus propias ventanas
  mucho mejor que las de otras condiciones.
  - PASAT: ρ 0.159 propia frente a 0.019 en OA.
  - PVTB: 0.148 frente a −0.013 en OC y 0.027 en OA.
  - Una deriva común del equipo haría que todos los modelos transfirieran por
    igual.
- **Dientes de sierra:** dentro de cada tarea, la alfa sube y la beta relativa
  baja. Al empezar la tarea siguiente, ambas retroceden en parte. Por ejemplo,
  de PVTB a MSIT la beta relativa baja −0.036 dentro de PVTB y salta +0.036 al
  empezar MSIT.
- **Efectos fijos:** al controlar el tiempo desde el inicio de la sesión, las
  pendientes de la beta relativa dentro de la grabación se mantienen:
  - PASAT −0.013 (t = −3.8);
  - PVTB −0.016 (t = −5.5);
  - el tiempo de sesión es pequeño y no significativo (+0.0017; t = +0.8).

**El reposo con ojos cerrados (OC) también cambia, pero al revés:** baja la
potencia absoluta lenta y sube la beta relativa. Su modelo no transfiere a las
tareas.

## 4. Desarrollo: modelo fijado y comprobación en `val`

*Fuente: `03_clasificador/INFORME.md`.*

- **Configuración elegida** con validación cruzada repetida (20 particiones)
  en `train`: las 56 potencias relativas + logística C = 0.01, con un modelo
  por condición (Δρ a igual tiempo en `train` = +0.120).
- **Las potencias absolutas** captaban sobre todo la deriva común con el
  reposo.
- **En `val` (25 personas)**, con la medida `rho_ventanas`, elegida en `train`
  **después** de una primera mirada a `val` que no fue buena:
  - Δρ a igual tiempo = +0.059 [+0.003, +0.116];
  - con la grabación completa = +0.095 [+0.046, +0.138].

  **Son cifras de desarrollo y optimistas.**

## 5. Confirmatorio: evaluación única en `test` (28 personas)

*Fuente: `06_test/INFORME.md` y `06_test/resultados_test.json`. Ejecutado una
vez el 2026-10-27, según el pre-registro y sin desviaciones. Figura F5.*

| Prueba | Estadístico | IC95 | p | Veredicto |
|---|---|---|---|---|
| **H1b:** tareas > OA, grabación completa | Δρ = +0.068 | [+0.002, +0.128] | 0.022 | **CONFIRMA** |
| **H1:** tareas > OA, igual tiempo | Δρ = +0.027 | [−0.038, +0.087] | 0.208 | **NO CONCLUYE** |
| H2 PASAT: ordena sus ventanas en el tiempo | ρ = +0.092 | [+0.048, +0.138] | Holm 0.0002 | **CONFIRMA** |
| H2 PVTB | ρ = +0.169 | [+0.120, +0.213] | Holm 0.0002 | **CONFIRMA** |
| H3 PASAT: la beta relativa baja más que en OA | Δpendiente = −0.029 | — | Holm 0.055 | NO CONCLUYE (potencia 26–38 %) |
| H3 PVTB | Δpendiente = −0.049 | — | Holm 0.018 | **CONFIRMA** |
| H4: EEG > solo movimiento y contacto | 0.743 − 0.579 = +0.163 | [+0.070, +0.257] | 0.0004 | **CONFIRMA** |

**ρ media de `rho_ventanas` por condición (descriptivo):**

| | Grabación completa | Igual tiempo |
|---|---|---|
| PASAT | +0.162 [+0.111, +0.217] | +0.092 [+0.052, +0.137] |
| PVTB | +0.180 [+0.141, +0.220] | +0.169 [+0.122, +0.215] |
| OA | +0.104 [+0.052, +0.160] | (sin recortar) |
| PVSAT | +0.034 [−0.003, +0.069] | +0.054 [+0.016, +0.094] |
| OC | +0.104 [+0.060, +0.152] | (sin recortar) |

**Acierto pareado de tercios:**
- grabación completa: PASAT 0.735, PVTB 0.835 y OA 0.670;
- igual tiempo: PASAT 0.686 y PVTB 0.775.

**Sensibilidad que el pre-registro obliga a declarar:** sin la persona de
`test` expuesta en los sondeos, la H1b queda en +0.061 [−0.006, +0.124]. **Su
IC incluye el 0.** El veredicto oficial no cambia, pero **la H1b confirma por
poco y no es robusta**. Sin las grabaciones a 128 Hz: +0.071 [+0.004, +0.134].

**Por qué la H1 no concluye.** En `test`, el reposo con ojos abiertos ordenó
sus ventanas en el tiempo bastante más que durante el desarrollo:

| | OA | PVTB | PASAT (igual tiempo) |
|---|---|---|---|
| `train` | 0.063 | | |
| `val` | 0.039 | | |
| `test` | 0.104 | 0.169 | 0.092 |

PVTB sigue por encima de OA, pero PASAT a igual tiempo queda a su altura.
Cualquier explicación de por qué OA cambia más en `test` sería un análisis
posterior y, por tanto, exploratorio.

## 6. Exploratorio: relación con el rendimiento (`train` + `val`, posterior al test)

*Fuente: `05_entre_sesiones/INFORME.md`.*

**Dentro de una misma persona** (efectos fijos de persona y medición, ≈ 500
sesiones de 135 personas):

| Cambio del EEG en la tarea ↔ conducta en esa tarea | Coeficiente | p (Holm) | Control: cambio en OA |
|---|---|---|---|
| PASAT ↔ precisión | +0.007 | 0.82 | −0.042 |
| **PVTB ↔ lapsos** | **+0.090 (r ≈ 0.14)** | **0.022** | +0.020 (p 0.53) |
| PVTB ↔ tiempo de reacción | +0.039 | 0.28 | −0.013 |

- **Hallazgo:** en las sesiones en que el EEG de una persona cambia más a lo
  largo del PVTB, esa persona comete más lapsos en el PVTB.
- **El cambio del reposo** de la misma sesión no se asocia con los lapsos.
- **Con cuatro covariables de movimiento** (nivel y tendencia del
  acelerómetro y del giro), el coeficiente conserva el 97 % (+0.093,
  p = 0.006).
- **Lo que no aparece:**
  - con la pendiente de la beta relativa (solo coincide el signo);
  - en PASAT;
  - entre personas.
- **Covariables:** ni el casco, ni la hora, ni el sexo, ni la medición mueven
  la medida.

**Es un resultado exploratorio, sin replicación independiente.**

## Figuras

| Figura | Archivo (`figuras/`) | Estatus | Pie propuesto |
|---|---|---|---|
| F1 | `f1_curvas_por_minuto.pdf` | Desarrollo (`train`) | Cambio por minuto de la alfa absoluta y la beta relativa posteriores, centrado por grabación (media e IC95 por personas). La alfa sube igual en las tareas y en el reposo; la beta relativa solo baja en las tareas. |
| F2 | `f2_acierto_pareado.pdf` | Desarrollo (`train`) | Acierto pareado de tercios con la grabación completa y recortada a la duración del reposo de la misma sesión. |
| F3 | `f3_referencia_artefactos.pdf` | Desarrollo (`train`) | El EEG frente a un clasificador de solo movimiento y calidad de contacto, a igual tiempo. |
| F4 | `f4_transferencia.pdf` | Desarrollo (`train`) | ρ media cuando el modelo entrenado en una condición (fila) ordena las ventanas de otra (columna). La diagonal domina en PASAT y PVTB. |
| F5 | `f5_resultado_test.pdf` | **Confirmatorio** (`test`) | Izquierda: ρ media por condición (descriptivo). Derecha: las dos pruebas principales con su veredicto pre-registrado. |
