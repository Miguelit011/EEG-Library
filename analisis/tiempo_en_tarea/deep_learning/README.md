# Tiempo en tarea — deep learning (sublínea exploratoria)

¿Contiene la señal cruda información sobre el tiempo en tarea **más allá** de
la potencia por banda? Es la pregunta de `../PLAN.md`, sección 7, que se dejó
fuera del test de la línea: `../06_test/PREREGISTRO.md` renuncia por escrito al
deep learning, y la línea principal (pasos 1–7) está cerrada.

Esta subcarpeta empieza el 2026-10-29, **después** de la evaluación única en
`test` (2026-10-27). Por eso los pasos 1–3 son **de desarrollo o
exploratorios** y no cambian ningún veredicto del paso 6. El paso 4 es una
**segunda evaluación preregistrada en `test`**, posterior a la principal, que
tampoco cambia aquellos veredictos. Se cita así en la memoria.

## Qué cambia respecto a `PLAN.md`, sección 7

- **Se añade un paso previo barato** (`01_psd_fina/`), que no estaba en el
  PLAN. Antes de gastar 30–45 h de CPU, se mira con la logística ya fijada si
  un espectro más fino (1 Hz) aporta algo sobre las cuatro bandas. Sirve de
  filtro: si el espectro fino no mejora, es poco probable que la CNN mejore
  por la vía espectral.
- **No hay H5 en el test de la línea.** La comparación CNN frente a
  logística se desarrolla en `train` (pasos 2 y 3). Desde el 2026-10-31 (decisión
  del autor) se evalúa además en `test` con su propio preregistro
  ([`04_test_eegnet/PREREGISTRO.md`](04_test_eegnet/PREREGISTRO.md)), como
  segunda evaluación de la línea.
- Lo demás de la sección 7 sigue como punto de partida: entrada a 128 Hz,
  normalización por grabación con canal de máscara, EEGNet-8,2 y
  ShallowConvNet, `rho_ventanas` como medida, mismos pliegues que la logística
  y el entorno aparte para PyTorch (7.7). Se revisa en el plan de cada paso.

## Reglas (las de `../README.md`, más estas)

- **No se carga `test`, salvo en la evaluación única del paso 4**, con
  `--test --permitir-test`, `git status` limpio y el preregistro ya
  commiteado. El resto de los modos filtran la lista de archivos por la
  `split` de `matriz_rf.parquet` **antes** de abrir ningún `.npy`.
- **`val`** se reserva para **una sola** comprobación de la configuración
  final (sería la mirada 3 de `val` en la línea). Los pasos de cribado usan
  solo `train`.
- **Nada derivado de los datos se guarda en el repositorio** salvo tablas
  agregadas y figuras. Tampoco se escribe en `EEG-Data` (solo lectura). Lo que
  haga falta cachear, en memoria o en una carpeta fuera del repo que se
  decida antes. Excepción, solo para el paso 4: los datos en bruto del test
  van a `EEG-Data/processed/fatiga_aguda_dl_test_<fecha-hora>/`, como en el
  paso 6 de la línea (autorizada por el autor el 2026-10-31).
- Igual que en la línea: plan en el `INFORME.md` **antes** de ejecutar, el
  SHA-256 de los datos leídos en `resultados.json`, agrupar por `persona_uid`
  y no empezar un paso sin haber revisado el anterior.

## Pasos

| Paso | Carpeta | Qué hace | Estado |
|---|---|---|---|
| 1 | [`01_psd_fina/`](01_psd_fina/INFORME.md) | Filtro barato: logística con la PSD de 1 Hz (14 × 30) frente a las 4 bandas (14 × 4), solo en `train` | Hecho (2026-10-29): mejora real pero pequeña. D1 = +0.019 [+0.013, +0.026], justo bajo el umbral de +0.02. Sin mejora en OA; el contraste tarea − OA a igual tiempo mejora en +0.025 [+0.005, +0.046]. OC gana +0.052 (deriva de asentamiento) |
| 2 | [`02_piloto_cnn/`](02_piloto_cnn/INFORME.md) | Piloto de EEGNet en un pliegue (≈ 2–3 h de CPU): validar el código, fijar lr y épocas, y comprobar si llega al nivel de la logística sobre el espectro fino | Hecho (2026-10-29): no iguala a las bandas. ρ EEGNet = 0.156 (3 semillas, lr 3·10⁻⁴), frente a 0.190 de la logística F y 0.176 de la B. EEGNet − F = −0.035 [−0.060, −0.010]. Requisitos técnicos cumplidos. Regla de decisión: parar |
| 3 | [`03_piloto_shallow/`](03_piloto_shallow/INFORME.md) | Piloto de ShallowConvNet (filtro temporal de 1 s) en el mismo pliegue que el paso 2, reutilizando su caché: ¿llega al nivel de la logística F? | Hecho (2026-10-30): no. ρ Shallow = 0.128 (3 semillas, lr 3·10⁻⁴), por debajo de EEGNet (0.156) y de las logísticas F (0.190) y B (0.176). Shallow − F = −0.063 [−0.086, −0.039]. Sobreajusta desde la época 3–4. Requisitos técnicos cumplidos. Regla de decisión: parar |
| 4 | [`04_test_eegnet/`](04_test_eegnet/INFORME.md) | EEGNet entrenada con `train` + `val` y evaluada **una vez** en `test` frente a la logística preregistrada del paso 6 ([preregistro](04_test_eegnet/PREREGISTRO.md)). Sustituye a la CV de 5 pliegues de ShallowConvNet (`04_cv_cnn/`), que no se hace según la regla del paso 3 | **Hecho (2026-10-31).** Preregistro fijado en `e20279d` y evaluado una vez en `test`. **P1: NO SE DISTINGUE**, Δρ (EEGNet − logística) −0.011 [−0.047, +0.028]. **S1 CONFIRMA** en PASAT (+0.180) y en PVTB (+0.141). La ventaja de `val` (+0.033) no se repite. El conjunto no suma (+0.001) |
| 5 | [`05_onnx/`](05_onnx/INFORME.md) | Exporta las tres EEGNet finales del paso 4 a ONNX para la plataforma (Fase 11 de EEG-Platform). No entrena ni carga `val` ni `test` | **Hecho (2026-11-01).** ONNX = PyTorch (diferencia máxima 2,2·10⁻⁶) y paridad exacta de la librería con el análisis en 20 grabaciones de `train` y `val` (diferencia 0) |

**Caché borrada el 2026-10-30.** Al cerrar los pilotos se borró
`TFG\EEG-DL-cache\` (tensor, máscara, tabla de ventanas, checkpoints y
puntuaciones de los pasos 2 y 3), como fijaba el plan del paso 2. Para volver a
ejecutar `evaluar.py` de los pasos 2 o 3, hay que regenerarla con
`02_piloto_cnn/preparar.py` y los `entrenar.py`. Las cifras y los SHA-256
quedan en los `resultados.json`.
