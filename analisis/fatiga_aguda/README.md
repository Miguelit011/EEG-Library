# Fatiga aguda: el paso del tiempo dentro de una tarea

¿Distingue el EEG el principio del final de **una misma grabación** de tarea
(primer tercio frente a último tercio), en personas que el modelo no ha visto?
Es la operacionalización estándar de la fatiga mental aguda o decremento de
vigilancia (*time-on-task*). En la memoria se nombra «fatiga aguda»,
«decremento de vigilancia» o «efecto del tiempo en tarea», nunca «la fatiga
del alumno».

El documento de partida, con la motivación, los sondeos previos, las
definiciones y el diseño de validación, es [`PLAN.md`](PLAN.md).

Cada paso vive en su propia subcarpeta con su script, sus tablas, sus figuras,
su `resultados.json` y un `INFORME.md` cuyo plan se escribe **antes** de
ejecutar. Reglas comunes a todos los pasos:

- **`train` y `val` para desarrollar; `test` una sola vez** (paso 6). Ningún
  script de los pasos 1–5 carga personas de `test`: se apartan al leer.
- **Las features por ventana salen de `features_all.parquet`; la partición y la
  identidad, de `matriz_rf.parquet`** (unión por `archivo_origen` ↔
  `archivo`). La `split` de `features_all` es la anterior a las correcciones de
  la Fase 12 y no se usa.
- Agrupar siempre por `persona_uid` (GroupKFold y bootstrap).
- Los canales excluidos por calidad (Fase 11.5) no aportan: se anulan tras el
  centrado por grabación y los agregados por región solo promedian canales
  válidos.
- Registrar en `resultados.json` el SHA-256 de cada archivo de datos leído.
- Ejecutar desde la raíz del repositorio con `.venv/Scripts/python.exe`.

## Pasos

| Paso | Carpeta | Qué hace | Estado |
|---|---|---|---|
| 1 | [`01_descriptivo/`](01_descriptivo/INFORME.md) | Reproducir los sondeos (pendientes y clasificador de tercios) con la partición correcta y los canales excluidos; curvas por minuto; descriptivo de duraciones | Hecho (2026-10-27): se reproduce. Acierto pareado PASAT 0.773 y PVTB 0.758 frente a OA 0.647; PVSAT ≈ OA. Criterio de continuidad cumplido. Anexo A (igual tiempo): PASAT +0.091 y PVTB +0.067 sobre OA; solo la beta relativa se separa del reposo |
| 2 | [`02_artefactos/`](02_artefactos/INFORME.md) | Movimiento (`MOT.*`) y calidad de contacto (`CQ.*`) frente al tiempo en tarea | Hecho (2026-10-27): el movimiento no crece más que en reposo; solo artefactos 0.54–0.57; EEG residualizado − OA +0.120 a igual tiempo. Criterio cumplido. Diagnóstico: la cifra depende de la partición en pliegues (±0.02–0.05) |
| 3 | [`03_clasificador/`](03_clasificador/INFORME.md) | Modelo clásico: variantes de features, transferencia entre tareas, regresión del minuto | Hecho (2026-10-27): elegida `rel` + logística C = 0.01, separado (H1 +0.120 en train). **`val` no es bueno**: H1 +0.056, pero PASAT a igual tiempo tiene IC [0.494, 0.685]. Con la grabación completa replica (H1b +0.173). **Reajuste 1**: se elige en train una medida más eficiente (`rho_ventanas`, z 4.89 frente a 3.44); en la mirada 2, **`val` es bueno** (H1 +0.059 [+0.003, +0.116]). OC se comporta como las tareas |
| 4 | [`04_potencia_y_controles/`](04_potencia_y_controles/INFORME.md) | Potencia para `test`; tiempo en tarea frente a tiempo desde la colocación | Hecho (2026-10-27): potencia H1 (igual tiempo) 0.75, H1b 0.92, H2 1.00, H3 0.26–0.38. El cambio de PASAT y PVTB es específico de la tarea, se reinicia al cambiar de tarea y no lo explica el tiempo de sesión. Pendiente: elegir la hipótesis principal |
| 5 | `05_entre_sesiones/` | (Opcional) Pendiente frente a rendimiento y covariables | Pendiente |
| 6 | [`06_test/`](06_test/PREREGISTRO.md) | Pre-registro y evaluación única en `test` | Pre-registro y guion escritos (2026-10-27); ensayo `train` → `val` correcto. **Pendiente: commit del autor y ejecución única** |
