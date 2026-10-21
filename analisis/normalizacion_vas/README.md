# Normalización intra-sujeto del VAS

Línea de exploración propuesta por el tutor: estudiar si expresar el VAS
relativo a cada persona (centrado o tipificado dentro de `persona_uid`) da una
etiqueta más útil que el VAS bruto.

Cada paso vive en su propia subcarpeta con su script, sus tablas, sus figuras
y un `INFORME.md` con resultados y conclusión. Reglas comunes a todos los
pasos:

- **Solo `train` + `val`.** El test está gastado (ver `docs/MODELADO.md`) y no
  se toca.
- Agrupar siempre por `persona_uid`.
- Cualquier estadístico de normalización se estima solo con la partición de
  entrenamiento.
- Ejecutar desde la raíz del repositorio con `.venv/Scripts/python.exe`.

## Pasos

| Paso | Carpeta | Qué hace | Estado |
|---|---|---|---|
| 1 | [`01_descriptivo/`](01_descriptivo/INFORME.md) | Caracterización descriptiva del VAS bruto: sesiones por persona, ICC, nivel frente a rango, media por momento | Hecho (2026-10-21) |
