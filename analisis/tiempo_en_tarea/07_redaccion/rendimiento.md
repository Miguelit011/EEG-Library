# Rendimiento en las tareas: qué hay, qué se buscó y qué se escribe

> Borrador del 2026-11-06 para pasar a la memoria. No se ha escrito nada en
> `EEG-Memoria`. Las cifras salen de `05_entre_sesiones/INFORME.md` (asociación
> con el EEG) y de `rendimiento_descriptivo.py` → `rendimiento_descriptivo.json`
> y `.log` (descriptivos del rendimiento solo, `train` + `val`, y potencia).

## 1. Qué datos de rendimiento hay

La base del estudio (`BD_COMPLETA_*.xlsx`) guarda, para cada alumno y cada
medición, **un resumen de cómo hizo cada tarea**: tiempo de reacción
(`Reaction_`), porcentaje de aciertos (`Accuracy_`), velocidad (`Speed_`),
variabilidad (`Variation_`), una medida combinada de rapidez y acierto (`RCS_`)
y, en tres tareas, lapsos (`Lapses_`, respuestas que no llegan a tiempo).

- Es **un valor por tarea entera**, no por ensayo ni por minuto.
- Los EDF no tienen marcadores de ensayo (`MarkerHardware` = 0), así que no se
  sabe en qué momento de la grabación ocurrió cada respuesta.

Calidad de las métricas en las 550 sesiones etiquetadas de `train` + `val`
(146 personas):

- **Cobertura alta:** entre el 96 % y el 99.6 % de las sesiones.
- **Inutilizables o casi:**
  - `Lapses_PASAT` y `Lapses_PVSAT` valen 0 en todas las sesiones;
  - efecto techo en la precisión del STROOP (74 % de las sesiones en 100),
    del PVTB (39 %) y del MSIT (34 %);
  - valores imposibles en `Accuracy_PASAT` (máximo 2536), `RCS_PASAT` (22) y
    `Reaction_PVSAT` (11 763 ms), que hunden su ICC(1) a 0.06–0.10.
- **Mucho componente de persona:** ICC(1) entre 0.5 y 0.7 en la mayoría.
- **Efecto de práctica:** dentro de persona, el tiempo de reacción baja con la
  medición en STROOP, PASAT y MSIT (r −0.24 a −0.32), y la precisión del PVSAT
  y los RCS suben (+0.25 a +0.33). Cualquier análisis con rendimiento tiene que
  controlar la medición.

## 2. Qué se buscó con ellos (explicado sin tecnicismos)

La línea de tiempo en tarea encontró que **el EEG cambia de forma ordenada a lo
largo de PASAT y PVTB**. La duda que deja es **por qué**: puede ser cansancio,
pero también habituación a la tarea o pérdida de atención. El rendimiento
sirve para mirar esa duda de forma indirecta:

> Si el cambio del EEG tiene que ver con que el alumno «se apaga», los días en
> que ese cambio es mayor deberían ser también los días en que rinde peor en
> esa tarea.

Se hizo dos veces, siempre con datos de desarrollo:

1. **Sondeos del 2026-10-26** (`PLAN.md`, 3.4; solo `train`): un índice de
   «fatigabilidad» por sesión no se relacionó con los lapsos del PVT, y el EEG
   por sesión tampoco siguió a los lapsos ni a los tiempos de reacción. Se
   midió con pendientes absolutas, que luego resultaron ser en buena parte una
   deriva común con el reposo.
2. **Paso 5 del 2026-10-27** (`train` + `val`, después del test): con la medida
   propia de la línea (`rho_ventanas`), tres comparaciones fijadas antes de
   mirar, siempre **dentro de la misma persona** (cada alumno comparado consigo
   mismo en distintas mediciones):

| Cambio del EEG en la tarea ↔ rendimiento en esa tarea | Coeficiente | p (Holm) | Control: cambio en reposo (OA) |
|---|---|---|---|
| PASAT ↔ precisión | +0.007 | 0.82 | −0.042 |
| **PVTB ↔ lapsos** | **+0.090 (r ≈ 0.14)** | **0.022** | +0.020 (p 0.53) |
| PVTB ↔ tiempo de reacción | +0.039 | 0.28 | −0.013 |

- **Lo que sale:** en las sesiones en que el EEG de un alumno cambia más a lo
  largo del PVTB, ese alumno comete más lapsos en el PVTB.
- **Lo que lo apoya:** el cambio del reposo de la misma sesión no se relaciona
  con los lapsos, y con cuatro covariables de movimiento se conserva el 97 %
  del efecto (+0.093, p 0.006).
- **Lo que lo debilita:** es una comparación de tres; no aparece con la otra
  medida del cambio (pendiente de la beta), ni en PASAT, ni entre personas; y
  se obtuvo después del test, sin replicación.

## 3. Texto propuesto para la memoria

### 3.1. Cap. 1, `01_introduccion.tex`, línea 224

Actual:

> …y no se dispone de medidas subjetivas ni de rendimiento dentro de la
> grabación.

Propuesto:

> …y no se dispone de medidas subjetivas a lo largo de la grabación; el
> rendimiento solo se registró como total de cada tarea.

### 3.2. Cap. 3, `03_estado_del_arte.tex`, líneas 31–32

Actual:

> …aunque en este estudio no se dispone de medidas de rendimiento dentro de
> cada grabación.

Propuesto:

> …aunque en este estudio el rendimiento solo se registró como un total por
> tarea y sesión, sin marcadores de ensayo que permitan seguir su evolución a
> lo largo de cada grabación.

### 3.3. Resultados o discusión de la línea (párrafo exploratorio)

> Como análisis exploratorio, posterior a la evaluación en test y realizado
> solo con los datos de desarrollo, se relacionó el cambio del EEG a lo largo
> de cada tarea con el rendimiento total en esa tarea. Dentro de una misma
> persona, las sesiones en las que el EEG cambió más a lo largo del PVTB fueron
> las de más lapsos en esa tarea (r ≈ 0.14; p = 0.022 tras la corrección de
> Holm). El cambio en reposo de la misma sesión no mostró esa relación, y el
> movimiento de la cabeza no la explica. Va en la dirección esperada si el
> cambio reflejara una pérdida de vigilancia, pero es un efecto pequeño, en
> una sola de las tres comparaciones, que no aparece en PASAT ni con una
> medida espectral más simple, y no se ha replicado en datos independientes.

### 3.4. Trabajo futuro (añadir a la lista)

> Registrar marcadores de cada ensayo junto al EEG, para relacionar la señal
> con el rendimiento minuto a minuto, y contar con más participantes: para
> replicar con garantías una relación del tamaño observada (r ≈ 0.14) harían
> falta unas 120 personas en el conjunto de evaluación, frente a las 28
> disponibles.

## 4. Por qué no se amplía la línea (decisión del 2026-11-06)

Se valoró hacer otro test o mirar más métricas. Se descarta:

- **Potencia de un test nuevo muy baja.** En `test` hay 103 sesiones de PVTB
  de 28 personas (`06_test/INFORME.md`). La probabilidad de confirmar la
  relación con los lapsos sería del 32 % si el efecto real fuera el observado
  (r 0.14), y del 21 % con r 0.10, más realista porque los efectos hallados
  explorando suelen estar inflados. Para llegar al 80 % harían falta 120 y 235
  personas. No hay más datos: M1 se descartó en la auditoría.
- **Resultado poco informativo en cualquier caso.** Lo más probable es un «no
  concluye», que no distingue entre ausencia de efecto y falta de datos.
- **Más métricas en los mismos datos no confirman nada.** Mirar
  `Variation_PVTB` o PVSAT en `train` + `val` sería buscar más en el mismo
  sitio, y el resto de métricas tiene techo, ceros o valores imposibles.
- **Coste:** un pre-registro y su ejecución llevan días que rinden más en la
  redacción.

## 5. Notas para quien pase el texto

- Estatus: **exploratorio** (regla de tres estatus del `INFORME.md` del paso 7).
- Las cifras de asociación son las de `05_entre_sesiones/INFORME.md`; las de
  calidad y potencia, las de `rendimiento_descriptivo.log`.
- `discusion.md` ya recoge esta evidencia y la limitación de los marcadores;
  este archivo añade las dos correcciones de los capítulos, el párrafo
  redactado y el argumento de la potencia.
