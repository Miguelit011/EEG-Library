# Paso 5 — Relación con el rendimiento y covariables (exploratorio)

> Plan escrito el 2026-10-27, **después** de la evaluación única en `test`
> (paso 6) y **antes** de ejecutar este paso. Todo lo de aquí es exploratorio:
> usa `train` + `val` (146 personas), que ya sirvieron para desarrollar, y
> **no carga `test`**. Ninguna cifra de este paso cambia los veredictos del
> paso 6.

## Por qué se hace (versión reducida)

El test confirmó que el EEG cambia de forma ordenada a lo largo de PASAT y PVTB
(H2) y que no lo explican los artefactos (H4). En cambio, no confirmó que ese
cambio supere al del reposo con ojos abiertos a igual tiempo (H1). La pregunta
que queda abierta es si ese cambio tiene relación con la tarea. Este paso añade
un argumento indirecto e independiente: **si el cambio en la tarea se asocia
con el rendimiento en esa tarea y el cambio en OA no**, eso apoya que es
específico de la tarea.

Expectativa honesta: baja. En los sondeos del 2026-10-26 (PLAN 3.4), la
«fatigabilidad» por sesión no se relacionó con los lapsos del PVT, aunque se
midió con pendientes **absolutas**, que el paso 1 mostró que son deriva común
con el reposo. Aquí se usan las medidas específicas de la línea. Además, la
conducta solo existe como total por sesión (PLAN 4.6): no se puede ver si el
rendimiento empeora dentro de la tarea.

Se recorta el plan original (≈ 4 h) a un conjunto cerrado de contrastes, fijado
aquí antes de mirar.

## Datos

- Grabaciones PASAT, PVTB y OA de `train` + `val` (`comun.cargar`), grabación
  completa (el rendimiento es el total de la tarea entera).
- Conducta por sesión: `BD_COMPLETA_corregida.xlsx`, hoja `CUESTIONARIOS`,
  enlazada con `analisis/fatiga_fg_fc/datos_fg_fc.py` por
  `(persona_uid, medicion)`.
- Variables fijadas en el PLAN: `Accuracy_PASAT`, `Lapses_PVTB` y
  `Reaction_PVTB`.
- **Valores fuera de rango, fijados antes de mirar asociaciones.** La
  descripción de la base muestra `Accuracy_PASAT` con máximo 2536 y mínimo
  0.67. Regla: `Accuracy_PASAT` válida en (1, 100]; los valores ≤ 1 son
  ambiguos (¿proporción?) y no se corrigen, se excluyen. `Reaction_PVTB` es
  válida en [100, 1000] ms y `Lapses_PVTB` en ≥ 0. Se cuenta cuántos valores
  se excluyen.
- Hora, casco y sexo: `manifest_canonical.csv` (`timestamp_iso`, hora local
  de la primera grabación de la sesión; `serie_headset`) y
  `datos_fg_fc.caracteristicas_persona` (sexo). Centro, desde `persona_uid`.

## Medidas de cambio por grabación

1. **Principal: `rho_ventanas`**, la medida del paso 6: modelo `rel` +
   logística C = 0.01, uno por condición, entrenado con tercios. Puntuación
   *out-of-fold*: con los 20 particionados por persona (semillas 0–19, 5
   pliegues) y la **media de las 20 ρ** de cada grabación, para quitar ruido
   de partición. El modelo de OA se usa para OA.
2. **Interpretable:** pendiente por minuto de la beta posterior relativa
   (`beta_post_rel`, la magnitud de la H3), MCO por grabación.

## Contrastes (conjunto cerrado)

**A. Dentro de persona (principal).** Para cada par (medida de la tarea,
variable de conducta de la misma tarea), se convierte cada variable a rangos
sobre las sesiones y se pasa a z. Luego se aplica la transformación *within*
por persona, con efectos fijos de medición (que quitan el calendario), y se
estima por MCO con errores típicos agrupados por persona. Así, el coeficiente
equivale aproximadamente a una correlación de rangos dentro de persona. Solo
entran personas con ≥ 2 sesiones válidas. Contraste bilateral.

| # | Cambio EEG | Conducta |
|---|---|---|
| A1 | `rho_ventanas` PASAT | `Accuracy_PASAT` |
| A2 | `rho_ventanas` PVTB | `Lapses_PVTB` |
| A3 | `rho_ventanas` PVTB | `Reaction_PVTB` |

**Holm sobre A1–A3.** Es la única familia con p corregida.

**B. Entre personas (descriptivo).** La media por persona del cambio frente a
la media por persona de la conducta, con ρ de Spearman e IC95 por bootstrap de
personas. Mismos tres pares.

**C. Control con OA (descriptivo).** Los mismos A1–A3 y B, con la
`rho_ventanas` de **OA** de la misma sesión en lugar de la de la tarea. La
lectura de especificidad exige que la tarea se asocie y OA no.

**D. Medida interpretable (descriptivo).** A1–A3 con la pendiente de
`beta_post_rel`. Signo esperado, si hay relación: a más bajada de beta
(pendiente más negativa), peor conducta.

**E. Covariables (descriptivo, sin p).** La `rho_ventanas` de PASAT y PVTB
frente a:
- sexo y centro: media por grupo, con IC95 por bootstrap de personas;
- medición (M2–M6);
- hora del día: ρ de Spearman entre sesiones, con IC95 por bootstrap de
  personas;
- casco: en M4–M6, 172358 frente a 236373, dentro de persona. Se calcula la
  diferencia media en las personas que usaron los dos.

## Cómo se lee

- **Hallazgo reportable** (exploratorio): algún A sobrevive a Holm, la medida
  interpretable (D) va en el mismo sentido y el control con OA (C) no muestra
  lo mismo. Se citaría como asociación exploratoria, sin replicación
  independiente.
- **Si no:** se reporta que, con totales por sesión, el cambio dentro de la
  tarea no se asocia con el rendimiento. Es coherente con PLAN 3.4 y con la
  limitación estructural de 4.6, y no indica ausencia de relación ensayo a
  ensayo.
- Pase lo que pase, la línea sigue al paso 7. Este paso no tiene criterio de
  continuidad.

## Archivos

- `entre_sesiones.py`: script; ejecución en `ejecucion.log`.
- `resultados.json`: solo agregados, sin `persona_uid`, con los SHA-256 de los
  archivos leídos.
- `tablas/`: solo agregados.

## Resultados (ejecutado el 2026-10-27, sin desviaciones del plan)

146 personas de `train` + `val`. `rho_ventanas` media (grabación completa,
out-of-fold, media de 20 particiones): PASAT +0.153 (533 grabaciones), PVTB
+0.150 (529) y OA +0.061 (531). Valores de conducta excluidos por rango: 2 de
`Accuracy_PASAT` y ninguno en las otras dos. Sesiones con dato: 534, 516 y 529.

### A, C y D: dentro de persona (efectos fijos de persona y medición)

El coeficiente es entre z de rangos y equivale aproximadamente a una
correlación dentro de persona.

| # | Conducta | **A: ρ de la tarea** | p (Holm) | C: ρ de OA (control) | D: pendiente de la beta |
|---|---|---|---|---|---|
| A1 | `Accuracy_PASAT` | +0.007 (t 0.23) | 0.82 (0.82) | −0.042 (p 0.24) | −0.015 (p 0.66) |
| **A2** | **`Lapses_PVTB`** | **+0.090 (t 2.73; r parcial 0.14)** | **0.007 (0.022)** | +0.020 (p 0.53) | −0.020 (p 0.55) |
| A3 | `Reaction_PVTB` | +0.039 (t 1.48) | 0.14 (0.28) | −0.013 (p 0.61) | −0.015 (p 0.62) |

Cada contraste usa unas 500 sesiones de 135–136 personas.

### B: entre personas (medias por persona, ρ de Spearman, 145 personas)

Ninguna asociación de la `rho_ventanas` de la tarea con la conducta: A1 −0.08,
A2 +0.06 y A3 +0.08, con todos los IC95 incluyendo el 0. Con OA tampoco. La
única cifra con IC95 que excluye el 0 es la pendiente de la beta de PASAT
frente a `Accuracy_PASAT`: +0.19 [+0.03, +0.36]. Es una de 9 cifras sin
corregir y no se interpreta.

### E: covariables (descriptivo)

En PASAT y PVTB no se ve ningún patrón:
- **Sexo:** sin diferencias (PASAT 0.149 frente a 0.157; PVTB 0.160 frente a
  0.139).
- **Hora del día:** ρ −0.04 y −0.08, con IC95 que incluyen el 0; las sesiones
  van de 8.9 h a 12.8 h (p10–p90).
- **Casco:** el 236373 frente al 172358, dentro de persona, en M4–M6: +0.02 y
  +0.03, con IC95 que incluyen el 0 (≈ 72 personas).
- **Medición:** M2–M6 oscilan entre 0.12 y 0.20, sin tendencia.
- **Centro:** de 0.08 a 0.22, con IC solapados; C3 está en el extremo bajo en
  PASAT y en el alto en PVTB, sin un patrón común.

### Lectura

1. **Se cumple, por los pelos, la regla de «hallazgo reportable»:**
   - A2 sobrevive a Holm (p = 0.022);
   - D va en el sentido coherente, aunque solo por el signo (−0.020, p = 0.55):
     la medida interpretable **no replica en magnitud**;
   - el control con OA no muestra lo mismo (+0.020, p = 0.53).
2. **Qué dice:** dentro de una misma persona, **en las sesiones en que el EEG
   cambia más a lo largo del PVTB, esa persona comete más lapsos en el PVTB**.
   El cambio del EEG en OA de esa misma sesión no se asocia con los lapsos. Es
   el sentido que espera la interpretación de decremento de vigilancia, y es
   específico de la tarea. La H1 no podía dar este tipo de argumento.
3. **Cuánto pesa:** poco, y se cita como exploratorio.
   - El efecto es pequeño (r ≈ 0.14 dentro de persona).
   - Es uno de tres contrastes.
   - No replica con la pendiente de la beta.
   - Entre personas no aparece.
   - En PASAT, la precisión no se asocia.

   A favor: el modelo y la medida se fijaron sin mirar la conducta, que no se
   había usado en ningún paso de esta línea. No es un ajuste a esta variable.
4. **No controlado (lo que habría que mirar antes de darle más peso):** una
   sesión con más lapsos puede ser una sesión con más movimiento, y el
   movimiento podría inflar la ρ. La H4 descarta que el movimiento explique el
   cambio medio, pero no esta covariación entre sesiones. No estaba en el plan.
5. **Las covariables no confunden:** ni el casco, ni la hora, ni el sexo, ni la
   medición mueven la ρ de las tareas de forma apreciable.

**Formulación para la memoria (exploratoria):** «En un análisis exploratorio
sobre las personas de desarrollo, las sesiones en que el EEG cambió más a lo
largo del PVTB fueron también aquellas con más lapsos en esa tarea (dentro de
persona, r ≈ 0.14; p corregida = 0.022). El cambio en el reposo de la misma
sesión no mostró esta asociación. No se observó relación con la precisión en
el PASAT ni entre personas, y no se ha replicado en datos independientes.»
