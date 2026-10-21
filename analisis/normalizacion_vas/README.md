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
| 2 | [`02_etiquetas/`](02_etiquetas/INFORME.md) | Variantes normalizadas de la etiqueta (sin EEG) y criterio previo de si merece la pena volver al EEG | Hecho (2026-10-21): hay paso 3, principal (b) |
| 3 | [`03_eeg/`](03_eeg/INFORME.md) | Tarea B de la Fase 14b con cada etiqueta normalizada, diferencia emparejada con la original | Hecho (2026-10-21): no mejora; **línea cerrada** |

## Conclusión de la línea

Normalizar el VAS dentro de cada persona **no mejora** lo que el EEG consigue
decir sobre la fatiga percibida. La variante principal da 0.351
[0.313, 0.390] de exactitud balanceada (azar 0.333) y −0.044
[−0.091, +0.004] frente a la etiqueta de la Fase 14b. Con distintas formas
razonables de construir la etiqueta, la Tarea B va de 0.32 a 0.40: la señal es
frágil. Según la regla fijada antes de ejecutar, la línea se cierra.
