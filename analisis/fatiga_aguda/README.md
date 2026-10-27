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
| 2 | `02_artefactos/` | Movimiento (`MOT.*`) y calidad de contacto (`CQ.*`) frente al tiempo en tarea | Pendiente |
| 3 | `03_clasificador/` | Modelo clásico: variantes de features, transferencia entre tareas, regresión del minuto | Pendiente |
| 4 | `04_potencia_y_controles/` | Potencia para `test`; tiempo en tarea frente a tiempo desde la colocación | Pendiente |
| 5 | `05_entre_sesiones/` | (Opcional) Pendiente frente a rendimiento y covariables | Pendiente |
| 6 | `06_test/` | Pre-registro y evaluación única en `test` | Pendiente |
