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
- **`val`** se reserva para **una sola** comprobación de la configuración
  final (sería la mirada 3 de `val` en la línea). Los pasos de cribado usan
  solo `train`.
- **Nada derivado de los datos se guarda en el repositorio** salvo tablas
  agregadas y figuras. Tampoco se escribe en `EEG-Data` (solo lectura). Lo que
  haga falta cachear, en memoria o en una carpeta fuera del repo que se
  decida antes.
- Igual que en la línea: plan en el `INFORME.md` **antes** de ejecutar, el
  SHA-256 de los datos leídos en `resultados.json`, agrupar por `persona_uid`
  y no empezar un paso sin haber revisado el anterior.

## Pasos

| Paso | Carpeta | Qué hace | Estado |
|---|---|---|---|
| 1 | [`01_psd_fina/`](01_psd_fina/INFORME.md) | Filtro barato: logística con la PSD de 1 Hz (14 × 30) frente a las 4 bandas (14 × 4), solo en `train` | Hecho (2026-10-29): mejora real pero pequeña. D1 = +0.019 [+0.013, +0.026], justo bajo el umbral de +0.02. Sin mejora en OA; el contraste tarea − OA a igual tiempo mejora en +0.025 [+0.005, +0.046]. OC gana +0.052 (deriva de asentamiento) |
| 2 | [`02_piloto_cnn/`](02_piloto_cnn/INFORME.md) | Piloto de EEGNet en un pliegue (≈ 2–3 h de CPU): validar el código, fijar lr y épocas, y comprobar si llega al nivel de la logística sobre el espectro fino | Hecho (2026-10-29): no iguala a las bandas. ρ EEGNet = 0.156 (3 semillas, lr 3·10⁻⁴), frente a 0.190 de la logística F y 0.176 de la B. EEGNet − F = −0.035 [−0.060, −0.010]. Requisitos técnicos cumplidos. Regla de decisión: parar |
| 3 | [`03_piloto_shallow/`](03_piloto_shallow/INFORME.md) | Piloto de ShallowConvNet (filtro temporal de 1 s) en el mismo pliegue que el paso 2, reutilizando su caché: ¿llega al nivel de la logística F? | Hecho (2026-10-30): no. ρ Shallow = 0.128 (3 semillas, lr 3·10⁻⁴), por debajo de EEGNet (0.156) y de las logísticas F (0.190) y B (0.176). Shallow − F = −0.063 [−0.086, −0.039]. Sobreajusta desde la época 3–4. Requisitos técnicos cumplidos. Regla de decisión: parar |
| 4 | `04_cv_cnn/` | ShallowConvNet en CV de 5 pliegues, con los mismos pliegues que la logística y el control OA | No se ejecuta según la regla del paso 3. **Sublínea cerrada** |
