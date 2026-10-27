# Paso 6 — Resultado de la evaluación única en `test`

> Ejecutado **una sola vez** el 2026-10-27 a las 19:05, sobre el commit
> `063a41f` (árbol limpio), con `evaluacion_test.py --test --permitir-test`,
> según `PREREGISTRO.md`. **Sin desviaciones.** Registro:
> `ejecucion_test.log`; cifras agregadas: `resultados_test.json`; datos en
> bruto: `<EEG_DATA>/processed/fatiga_aguda_test_20261027-190547/`.

## Datos

28 personas de `test`. Modelo final entrenado con 146 personas (`train` +
`val`). Grabaciones evaluadas: PASAT 102, PVTB 103 (102 a igual tiempo), OA
103, PVSAT 105 y OC 104. Persona expuesta en los sondeos: 1. Grabaciones a
128 Hz: 5. Grabaciones sin movimiento (H4): 2.

## Veredictos (pre-registrados)

| Prueba | Estadístico | IC95 | p | Veredicto |
|---|---|---|---|---|
| **H1b** (grabación completa) | Δρ = **+0.068** | [+0.002, +0.128] | 0.022 | **CONFIRMA** |
| **H1** (igual tiempo) | Δρ = **+0.027** | [−0.038, +0.087] | 0.208 | **NO CONCLUYE** |
| H2 PASAT | ρ = +0.092 | [+0.048, +0.138] | 0.0001 (Holm 0.0002) | **CONFIRMA** |
| H2 PVTB | ρ = +0.169 | [+0.120, +0.213] | 0.0001 (Holm 0.0002) | **CONFIRMA** |
| H3 PASAT | Δpendiente = −0.029 (t = −1.66) | — | 0.055 (Holm 0.055) | NO CONCLUYE (potencia 26–38 %: no indica ausencia) |
| H3 PVTB | Δpendiente = −0.049 (t = −2.53) | — | 0.009 (Holm 0.018) | **CONFIRMA** |
| H4 | EEG 0.743 − artefactos 0.579 = **+0.163** | [+0.070, +0.257] | 0.0004 | **CONFIRMA** |

## Descriptivos

ρ media de `rho_ventanas` (IC95):

| | Grabación completa | Igual tiempo |
|---|---|---|
| PASAT | +0.162 [+0.111, +0.217] | +0.092 [+0.052, +0.137] |
| PVTB | +0.180 [+0.141, +0.220] | +0.169 [+0.122, +0.215] |
| **OA** | **+0.104** [+0.052, +0.160] | +0.104 (sin recortar) |
| PVSAT | +0.034 [−0.003, +0.069] | +0.054 [+0.016, +0.094] |
| OC | +0.104 [+0.060, +0.152] | +0.104 |

Acierto pareado de tercios (la cifra más interpretable): PASAT 0.735, PVTB
0.835 y OA 0.670 con la grabación completa; a igual tiempo, PASAT 0.686 y
PVTB 0.775.

## Sensibilidades (sin veredicto)

| | H1b | H1 | ρ PASAT | ρ PVTB |
|---|---|---|---|---|
| Oficial | +0.068 [+0.002, +0.128] | +0.027 [−0.038, +0.087] | +0.092 | +0.169 |
| Sin la persona expuesta | +0.061 [**−0.006**, +0.124] | +0.020 [−0.046, +0.083] | +0.091 | +0.165 |
| Sin las grabaciones a 128 Hz | +0.071 [+0.004, +0.134] | +0.032 [−0.035, +0.094] | +0.093 | +0.168 |

**Como exige el pre-registro (sección 6), se dice explícitamente: sin la
persona expuesta, el IC95 de la H1b incluye el 0**, así que con esa
sensibilidad la H1b no habría confirmado. El veredicto oficial sigue siendo el
de la tabla de veredictos, pero **la H1b confirma por poco y no es robusta**.

## Lectura

1. **Lo que se confirma con holgura: el EEG cambia de forma ordenada con el
   tiempo dentro de PASAT y PVTB, en personas no vistas (H2), y ese cambio no
   lo explican el movimiento ni el contacto (H4: +0.163).** La firma de la
   beta relativa se confirma en PVTB (H3); en PASAT se queda en p = 0.055, con
   una potencia baja de antemano.
2. **Lo que no se confirma: que ese cambio supere al del reposo con ojos
   abiertos a igual tiempo transcurrido (H1).** En `test`, OA ordena sus
   ventanas en el tiempo bastante más que en el desarrollo (ρ = 0.104, frente a
   0.063 en `train` y 0.039 en `val`). PVTB sigue por encima (0.169), pero
   PASAT a igual tiempo (0.092) queda igual que OA.
3. **La H1b (grabación completa) confirma, pero por poco** y depende de una
   persona. Además, es la versión que mezcla tarea con tiempo transcurrido.
   No permite afirmar que el efecto sea específico de la tarea.
4. **Formulación honesta para la memoria:** «En personas no vistas, el EEG de
   consumo registra un cambio sistemático a lo largo de las tareas cognitivas
   largas (PASAT, PVTB), compatible con el efecto del tiempo en tarea
   descrito en la literatura, que no se explica por el movimiento ni por el
   contacto de los electrodos. No se ha podido confirmar que ese cambio sea
   mayor que el que se produce en reposo con los ojos abiertos durante el
   mismo tiempo; con la grabación completa, la diferencia es positiva pero
   frágil.»
5. Se mantiene lo de los pasos 1 y 4: lo específico de las tareas en el
   desarrollo era el desplazamiento relativo (beta relativa), y la subida de la
   potencia absoluta es común con el reposo. El orden fijo del protocolo sigue
   siendo una limitación.

Cualquier análisis posterior sobre `test` (por ejemplo, entender por qué OA
cambia más en `test`) es **exploratorio** y se cita así.
