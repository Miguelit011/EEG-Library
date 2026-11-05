# Deep learning — paso 5: EEGNet en ONNX para la plataforma

> Plan escrito el 2026-11-01, **antes** de ejecutar nada de este paso. Es la
> Fase 11 de EEG-Platform (`EEG-Platform/PLAN.md`). **No entrena nada, no
> carga `test` ni cambia ningún resultado:** convierte los pesos finales del
> paso 4 a un formato que la librería pueda ejecutar sin PyTorch y comprueba
> que dan exactamente lo mismo.

## Para qué sirve

La plataforma enseña, en la pestaña Tiempo en tarea, una segunda cinta con la
puntuación de EEGNet junto a la de la logística. La web y la librería no
tienen PyTorch (`PLAN.md` de la línea, 7.7), pero sí `onnxruntime` (ICLabel
lo usa). Así que los tres modelos finales del paso 4 se exportan a ONNX.

## Qué se usa

- **Pesos:** `s0.pt`, `s1.pt` y `s2.pt` de
  `EEG-Data/processed/fatiga_aguda_dl_test_20261031-224217/modelos/`. Antes de
  nada se comprueba que su SHA-256 es el de `04_test_eegnet/configuracion_dl.json`.
- **Arquitectura:** la clase `EEGNet` de `02_piloto_cnn/entrenar.py`, sin
  modificarla.
- **Entrada:** `(n, 2, 14, 512)`: la ventana a 128 Hz normalizada por
  grabación y canal, con los canales excluidos a 0, y el mapa de máscara. Es
  `piloto.tensor` + `lote` de los pasos 2 y 4.

## Scripts

| Script | Entorno | Qué hace |
|---|---|---|
| `referencia.py` | `.venv` | Construye con `02_piloto_cnn/preparar.py` (`tensor`) el tensor de 20 grabaciones PASAT y PVTB de **`train`** (semilla 0) y lo guarda, con su tabla de ventanas, en `TFG/EEG-DL-cache/onnx/` (fuera de los repos, sin git; se borra al cerrar el paso) |
| `exportar.py` | `C:\venvs\eeg-dl` | Comprueba los SHA-256 de los `.pt`, exporta cada semilla a ONNX (lote dinámico), puntúa el tensor de referencia con PyTorch y con `onnxruntime` y compara. Si pasa, escribe los `.onnx` y la ficha `tt_eegnet.json` en `src/eeg_library/modelos/tiempo_en_tarea/` |

En `eeg-dl` se instalaron `onnx`, `onnxscript` y `onnxruntime` (visto bueno
del autor, 2026-11-01). `onnxruntime` es la 1.23.2 en los tres entornos.

## Criterios (fijados antes de ejecutar)

- **ONNX = PyTorch:** diferencia absoluta máxima de la puntuación (logit) por
  ventana < 10⁻⁵ en las tres semillas, sobre todas las ventanas del tensor de
  referencia.
- Si no se cumple, no se escribe nada en la librería y se para. La
  alternativa (anotada en el plan de la Fase 11) sería reimplementar la red
  en NumPy.

## La ficha

Copia las cifras del test de `04_test_eegnet/resultados_test.json`: ρ de
PASAT + PVTB (`cnn_familia`), acierto pareado de tercios
(`tercios_cnn_familia`) y la comparación con la logística (`P1`).

**Una diferencia con el test, que se declara en la ficha:** en el test, cada
semilla se evaluó por separado y se promediaron las cifras. La plataforma da
una sola puntuación por ventana, la media de los logits de las tres semillas.
Es la forma natural de enseñar un único resultado, pero no es exactamente el
cálculo que dio la cifra del test.

## Recuento de miradas

- `train`: se puntúan 20 grabaciones con modelos ya fijados; no se elige nada.
- `val` y `test`: no se cargan.

## Resultados

Ejecutado el 2026-11-01: `referencia.py` (`.venv`) y `exportar.py` (`eeg-dl`).

### Exportación (`exportar.py` → `resultados.json`)

- SHA-256 de los tres `.pt`: iguales a `configuracion_dl.json`.
- Exportador nuevo de PyTorch (dynamo), opset 17, lote dinámico. Cada `.onnx` ocupa 11 400 bytes.
- **ONNX = PyTorch** en las 1 757 ventanas de 20 grabaciones de `train` (17 % de canales
  excluidos): diferencia máxima de 1,9·10⁻⁶ (s0), 7,2·10⁻⁷ (s1) y 2,2·10⁻⁶ (s2). **Cumple**
  (< 10⁻⁵).
- Escritos `tt_eegnet_s0.onnx`, `tt_eegnet_s1.onnx`, `tt_eegnet_s2.onnx` y `tt_eegnet.json` en
  `src/eeg_library/modelos/tiempo_en_tarea/`. *(Se escribieron como `fa_*` en
  `modelos/fatiga_aguda/`; el 2026-11-05, contrato 0.17.0, se renombraron sin cambiar su
  contenido: los `.onnx` conservan su SHA-256.)*

### Librería (contrato 0.16.0)

- `modelado/tiempo_en_tarea_eegnet.py`: `preparar_entrada` (la preparación del piloto),
  `puntuar_ventanas` (onnxruntime, un hilo, media de las tres semillas) y `puntuar_eegnet`
  (ρ y veredicto de tercios como `tercios_cnn`).
- `analizar_grabacion` pide las ventanas a `ejecutar_fases` y guarda `por_ventana["eegnet"]`
  (o `null` con otros parámetros). Coste medido: 0,13 s en una grabación típica (90 ventanas) y
  0,33 s en la más larga (188).
- **Paridad con el análisis** (`tests/datos/test_tiempo_en_tarea_eegnet_datos.py`, 20 grabaciones
  PASAT y PVTB de `train` y `val`): la entrada que prepara la librería es **idéntica** a la de
  `tensor` (diferencia 0) y la puntuación que guarda `analizar_grabacion` es **idéntica** a la de
  puntuar el tensor del análisis (diferencia 0).
- Los tests de las fases anteriores siguen pasando: matriz idéntica, 70 características con
  paridad exacta y paridad de la logística de tiempo en tarea.

### Caché

`TFG/EEG-DL-cache/onnx/` (el tensor de referencia, 50 MB) se borró al cerrar el paso. Para
repetir la comprobación: `referencia.py` y después `exportar.py`.
