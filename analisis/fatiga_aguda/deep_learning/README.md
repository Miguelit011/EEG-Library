# Fatiga aguda — deep learning (sublínea exploratoria)

¿Contiene la señal cruda información sobre el tiempo en tarea **más allá** de
la potencia por banda? Es la pregunta de `../PLAN.md`, sección 7, que se dejó
fuera del test de la línea: `../06_test/PREREGISTRO.md` renuncia por escrito al
deep learning, y la línea principal (pasos 1–7) está cerrada.

Esta subcarpeta empieza el 2026-10-29, **después** de la evaluación única en
`test` (2026-10-27). Por eso todo lo de aquí es **de desarrollo o
exploratorio** y no cambia ningún veredicto del paso 6. Se cita así en la
memoria.

## Qué cambia respecto a `PLAN.md`, sección 7

- **Se añade un paso previo barato** (`01_psd_fina/`), que no estaba en el
  PLAN. Antes de gastar 30–45 h de CPU, se mira con la logística ya fijada si
  un espectro más fino (1 Hz) aporta algo sobre las cuatro bandas. Sirve de
  filtro: si el espectro fino no mejora, es poco probable que la CNN mejore
  por la vía espectral.
- **Ya no hay H5 confirmatoria.** El test está gastado para la línea. La
  comparación CNN frente a logística se hace con validación cruzada por
  persona y se reporta como desarrollo.
- Lo demás de la sección 7 sigue como punto de partida: entrada a 128 Hz,
  normalización por grabación con canal de máscara, EEGNet-8,2 y
  ShallowConvNet, `rho_ventanas` como medida, mismos pliegues que la logística
  y el entorno aparte para PyTorch (7.7). Se revisa en el plan de cada paso.

## Reglas (las de `../README.md`, más estas)

- **No se carga `test`.** Los scripts filtran la lista de archivos por la
  `split` de `matriz_rf.parquet` **antes** de abrir ningún `.npy`.
- **`val`:** propuesta pendiente de visto bueno. Se reserva para **una sola**
  comprobación de la configuración final (sería la mirada 3 de `val` en la
  línea). Los pasos de cribado usan solo `train`.
- **Nada derivado de los datos se guarda en el repositorio** salvo tablas
  agregadas y figuras. Tampoco se escribe en `EEG-Data` (solo lectura). Lo que
  haga falta cachear, en memoria o en una carpeta fuera del repo que se
  decida antes.
- Igual que en la línea: plan en el `INFORME.md` **antes** de ejecutar, el
  SHA-256 de los datos leídos en `resultados.json`, agrupar por `persona_uid`
  y parar entre pasos hasta tener el visto bueno.

## Pasos

| Paso | Carpeta | Qué hace | Estado |
|---|---|---|---|
| 1 | [`01_psd_fina/`](01_psd_fina/INFORME.md) | Filtro barato: logística con la PSD de 1 Hz (14 × 30) frente a las 4 bandas (14 × 4), solo en `train` | Plan escrito (2026-10-29), sin ejecutar |
| 2 | `02_piloto_cnn/` | Piloto de EEGNet en un pliegue (≈ 3 h de CPU): validar el código, épocas y lr | Depende del paso 1 y de la decisión del autor |
| 3 | `03_cv_cnn/` | EEGNet y ShallowConvNet en CV de 5 pliegues, con los mismos pliegues que la logística y el control OA | Depende del paso 2 |
