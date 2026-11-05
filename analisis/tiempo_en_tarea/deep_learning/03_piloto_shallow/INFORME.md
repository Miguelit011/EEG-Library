# Deep learning — paso 3: piloto de ShallowConvNet (un pliegue)

> Plan escrito el 2026-10-30, después del paso 2 y **antes** de ejecutar nada
> de este paso. Usa **solo `train`** y no carga ni `val` ni `test`. Es de
> desarrollo.

## Para qué sirve

El paso 2 paró EEGNet: ρ = 0.156, por debajo de la logística de bandas (0.176)
y de la del espectro fino (0.190) en el mismo pliegue. Ese piloto dejó una
arquitectura sin probar, ShallowConvNet, y es la que tiene más opciones aquí.

- **Hace lo mismo que la logística, pero aprendido.** Filtra cada canal en el
  tiempo (como elegir bandas), combina canales (filtro espacial), eleva al
  cuadrado, promedia y toma el logaritmo. Eso es una log-potencia por banda,
  con bandas y combinaciones de canales que elige la red.
- **Puede tener algo que la logística no tiene:** la logística F usa la
  potencia de cada canal por separado. ShallowConvNet puede usar la potencia de
  **combinaciones de canales**, que es información espacial (punto 2 de
  «Por qué se hace» en el paso 1).
- **EEGNet no hace eso igual de bien:** tiene solo 2 filtros espaciales por
  filtro temporal y no calcula potencias de forma explícita.

**La pregunta del piloto:** en un pliegue, ¿llega ShallowConvNet al nivel de
la logística F? Si llega, merece la pena el programa completo (paso 4). Si no,
se cierra la sublínea con dos arquitecturas probadas.

**Expectativa, escrita antes:** el paso 1 acotó la vía espectral en ≈ +0.02 de
ρ sobre las bandas. Lo razonable es que ShallowConvNet quede cerca de la
logística F (≈ 0.19). Superarla con claridad exigiría que la información
espacial aporte algo, y eso no se ha medido nunca.

**Lo que no responde:** si ShallowConvNet supera a la logística. Con un
pliegue (25 personas), el IC es demasiado ancho. Eso sería el paso 4.

## Datos: los del paso 2, sin volver a leer nada

Se reutiliza la caché del paso 2 en `TFG\EEG-DL-cache\piloto\`. **No se abre
`EEG-Data`** y no hace falta un `preparar.py` nuevo.

| Archivo de la caché | Qué es |
|---|---|
| `X.npy` | Tensor 77 864 × 14 × 512 (PASAT + PVTB de `train`, 128 Hz, z-score por grabación, canales excluidos a 0) |
| `mascara.npy` | Máscara de canales válidos por ventana |
| `ventanas.parquet` | Papel de cada ventana (entreno / interno / evaluación), tercio, instante y las puntuaciones de las logísticas F y B |

Ventajas:

- **Mismo pliegue, mismo conjunto interno y mismas personas de evaluación**
  que EEGNet y las logísticas. La comparación es pareada por grabación con las
  tres.
- **Las referencias no se recalculan.** ρ_F = 0.190 y ρ_B = 0.176 salen de las
  puntuaciones ya guardadas.
- **Comprobación de integridad:** `evaluar.py` debe reproducir exactamente esas
  dos cifras y la de EEGNet (0.156). Si no, se para. En `resultados.json` se
  guarda el SHA-256 de los tres archivos de la caché.

Los archivos de la caché solo se leen. Las ejecuciones nuevas van a
`EEG-DL-cache\piloto\ejecuciones_shallow\`, separadas de las de EEGNet.

## Modelo

**ShallowConvNet** (Schirrmeister et al., 2017), reimplementada a mano como en
el apéndice B del PLAN, con un cambio en el filtro temporal (ver abajo):

| Capa | Detalle |
|---|---|
| Conv temporal | (1, **128**), 40 filtros, sin relleno. Entrada de 2 mapas (señal y máscara), como en el paso 2 |
| Conv espacial | (14, 1), 40 → 40, sin sesgo |
| BatchNorm | — |
| Cuadrado | x² |
| AvgPool | (1, 35) con paso 7 |
| Log | log(max(x, 10⁻⁶)) |
| Dropout | 0.5 (el valor de Schirrmeister) |
| Lineal | a 1 logit |

**34 801 parámetros** (medido).

### Cambio respecto al PLAN: filtro temporal de 1 s en lugar de 0.1 s

El PLAN (apéndice B) fijaba un filtro de (1, 13) a 128 Hz, el equivalente del
original a 250 Hz. Se cambia a **(1, 128) = 1 s** por un motivo técnico:

- **Un filtro de 13 muestras no distingue frecuencias cercanas.** Su
  resolución es de ≈ 10 Hz: no separa theta (4–8 Hz) de alfa (8–13 Hz) y aún
  menos el desplazamiento de 1–2 Hz del pico de alfa.
- **La única mejora medida (paso 1) viene precisamente del detalle espectral
  de 1 Hz.** La logística F usa sub-ventanas de Welch de 1 s. Un filtro de 1 s
  da a la red la misma resolución que F. Con 13 muestras, la comparación con F
  estaría perdida de antemano.
- **Apenas cuesta más.** Medido con datos sintéticos el 2026-10-30 (mismo
  equipo, 6 hilos, lote de 256):

| Filtro temporal | Parámetros | Entreno (ventanas/s) | Inferencia (ventanas/s) |
|---|---|---|---|
| (1, 13) — PLAN | 26 241 | 549 | 1 221 |
| (1, 64) | 30 041 | 582–594 | 1 885–1 906 |
| **(1, 128) — elegido** | **34 801** | **506** | **1 643** |

> **Decisión pendiente de tu visto bueno.** Si prefieres seguir el PLAN al pie
> de la letra, se cambia a (1, 13) sin más cambios. La alternativa intermedia
> es (1, 64), igual que el filtro temporal de EEGNet en el paso 2.

## Entrenamiento: igual que el paso 2

- Un modelo conjunto para PASAT + PVTB. Etiqueta 0 en el primer tercio y 1 en
  el último. El tercio medio se puntúa, pero no se entrena.
- AdamW (`weight_decay` = 0.01), lote de 256, `BCEWithLogitsLoss`.
- Parada temprana sobre la pérdida del conjunto interno: paciencia de 8 épocas
  y máximo de 60. Se guarda el modelo de la mejor época.
- Checkpoint en cada época (modelo, optimizador, época, estado aleatorio). Si
  algo se corta, se reanuda.
- `torch.set_num_threads(6)`.

## Ejecuciones (lista cerrada)

| # | lr | Semilla | Para qué |
|---|---|---|---|
| 1 | 3·10⁻⁴ | 0 | Rejilla de lr |
| 2 | 1·10⁻³ | 0 | Rejilla de lr |
| 3 | mejor lr | 1 | Variabilidad entre semillas |
| 4 | mejor lr | 2 | Variabilidad entre semillas |

- **Dos lr en vez de tres.** En el paso 2, la pérdida interna de las tres lr
  difería menos de 0.003. Las dos elegidas rodean el valor por defecto de
  braindecode para ShallowConvNet (6.25·10⁻⁴). Se quita 3·10⁻³, que fue la
  peor en el paso 2.
- **Mejor lr:** la de menor pérdida en el conjunto interno, nunca la ρ de
  evaluación.

## Qué se mide

- Por ejecución: curva de pérdida (entreno e interna), mejor época, segundos
  por época, ρ media de PASAT, de PVTB y de las dos juntas, y AUC por ventana
  en los tercios.
- Diferencias pareadas por grabación con IC95 por bootstrap de personas (con
  `medidas.bootstrap`, como en el paso 2): **Shallow − F** (la principal),
  Shallow − B y Shallow − EEGNet. **Son orientativas**: 25 personas.
- Desviación típica de ρ entre las 3 semillas de la mejor lr.
- **Descriptivo, solo si llega a F:** respuesta en frecuencia de los 40
  filtros temporales (FFT de los pesos) del mejor modelo, para ver qué
  frecuencias usa. Señales de alarma de artefacto, como en el paso 1: si se
  concentran en 1–3 Hz (parpadeo o deriva de contacto) o en 20–30 Hz (EMG),
  se anota como sospechoso.

## Regla de decisión (fijada antes de ejecutar)

Con ρ_S la media de las 3 semillas de la mejor lr (PASAT + PVTB) y las
referencias ya conocidas en este pliegue (ρ_F = 0.190 y ρ_B = 0.176):

| Resultado | Lectura | Qué se propone |
|---|---|---|
| ρ_S ≥ **0.180** (ρ_F − 0.01) | ShallowConvNet llega al nivel de la mejor logística | Paso 4: CV de 5 pliegues con ShallowConvNet y el control OA. Antes se decide, en su plan, si se evalúa en `test` (ver abajo) |
| **0.166** ≤ ρ_S < 0.180 | Iguala a las bandas, pero no al espectro fino | Se valora. Lo más probable es parar |
| ρ_S < **0.166** (ρ_B − 0.01) | Ni siquiera iguala a las bandas | Parar y cerrar la sublínea |

- El margen de 0.01 es el del paso 2: el error típico aproximado de una
  diferencia pareada de ρ con ≈ 25 personas.
- **No hay una casilla para «supera a F».** Aunque ρ_S > 0.190, con un
  pliegue no se puede afirmar. Se anota y se lleva al paso 4.

**Requisitos técnicos** (si alguno falla, se corrige y se repite, sin pasar a
la regla):

- la pérdida interna baja respecto a la época 1;
- la mejor época no es la 1 ni la 60;
- no hay NaN;
- `evaluar.py` reproduce ρ_F = 0.190, ρ_B = 0.176 y ρ_EEGNet = 0.156 con la
  caché.

## Sobre `test`

El README de esta sublínea dice que el test está gastado para la línea. Se
puede volver a usar para evaluar **una técnica nueva** si se cumplen tres
condiciones: se fija antes en un preregistro, se evalúa una sola vez y se
declara en la memoria como segunda evaluación en `test`. **Este piloto no lo
decide.** Solo se plantea si el piloto pasa al paso 4, y entonces se escribe
en el plan del paso 4. Aquí no se carga `test`.

## Scripts

| Script | Entorno | Qué hace |
|---|---|---|
| `entrenar.py` | `C:\venvs\eeg-dl` | Entrena las 4 ejecuciones y guarda curvas, mejor modelo y puntuaciones de evaluación. Importa `lote`, `puntuar` y `auc` de `02_piloto_cnn/entrenar.py` sin modificarlo |
| `evaluar.py` | `.venv` | Calcula ρ, diferencias, regla de decisión, requisitos y figuras. Reutiliza `rho` e `ic` de `02_piloto_cnn/evaluar.py` y `medidas.py` sin modificarlos |

Los dos se ejecutan desde la raíz del repo con `PYTHONIOENCODING=utf-8`, como
en los pasos 1 y 2.

## Antes de lanzar

- **Memoria:** el paso 2 se cortó una vez por falta de RAM. El tensor ocupa
  2.2 GB, y ShallowConvNet necesita más memoria por lote que EEGNet
  (≈ 1 GB). Cerrar el navegador y otros programas pesados mientras dure.
- Pausar Windows Update y vigilar las temperaturas, como en el paso 2.

## Qué se produce

- `entrenar.py`, `evaluar.py`, sus `.log` y `resultados.json` (SHA-256 de la
  caché y versiones de PyTorch y de Python).
- `tablas/`: `ejecuciones.csv` (una fila por ejecución más las tres
  referencias) y `curvas.csv`.
- `figuras/`: `curvas.png` (pérdida interna), `rho_por_modelo.png` (Shallow,
  EEGNet y las dos logísticas) y, solo si llega a F, `filtros.png`.

## Coste

- **CPU: ≈ 4–6 h.** Con 506 ventanas/s sintéticas y el sobrecoste del paso 2
  con datos reales (×1.45), salen ≈ 100–110 s por época. Con 30–55 épocas,
  cada ejecución tarda ≈ 50–100 min, y hay 4.
- Trabajo: ≈ 2–3 h (dos scripts que reutilizan los del paso 2 e informe).

## Recuento de miradas

- `train`: pliegue 0 de la semilla 0, 4 ejecuciones de ShallowConvNet. Es la
  **segunda arquitectura** que se mira con las mismas 25 personas de
  evaluación. Además, la decisión de probarla se tomó después de ver el
  resultado de EEGNet en ese pliegue. Las dos cosas se declaran en la memoria.
- `val`: 2 (no se carga).
- `test`: no se carga.

## Resultados

Ejecutado el 2026-10-30, con el filtro temporal de 1 s (visto bueno del
usuario). `entrenar.py` → `entrenar.log`; `evaluar.py` → `evaluar.log`,
`resultados.json`, `tablas/` y `figuras/`. Entrenamiento: **53 min de CPU**
para las 4 ejecuciones (≈ 67 s por época), muy por debajo de las 4–6 h
estimadas, porque todas pararon pronto (ver Lectura).

### Ejecuciones (personas de evaluación del pliegue 0: 25 personas, 188 grabaciones)

| Modelo | Épocas | Mejor época | Pérdida interna | ρ PASAT | ρ PVTB | ρ PASAT + PVTB | AUC por ventana |
|---|---|---|---|---|---|---|---|
| Shallow, lr 3·10⁻⁴, semilla 0 | 11 | 3 | 0.6904 | 0.135 | 0.140 | 0.137 | 0.570 |
| Shallow, lr 1·10⁻³, semilla 0 | 12 | 4 | 0.6939 | 0.075 | 0.086 | 0.080 | 0.544 |
| Shallow, lr 3·10⁻⁴, semilla 1 | 11 | 3 | **0.6876** | 0.150 | 0.116 | 0.133 | 0.570 |
| Shallow, lr 3·10⁻⁴, semilla 2 | 12 | 4 | 0.6907 | 0.099 | 0.127 | 0.113 | 0.561 |
| EEGNet (paso 2, media de 3 semillas) | — | — | — | — | — | 0.156 | 0.578–0.585 |
| Logística F (C = 0.003) | — | — | — | **0.200** | **0.181** | **0.190** | **0.623** |
| Logística B (C = 0.03) | — | — | — | 0.177 | 0.176 | 0.176 | 0.613 |

**Mejor lr:** 3·10⁻⁴, por la menor pérdida interna (0.6904 frente a 0.6939).

### Regla de decisión

- ρ_S (media de las 3 semillas con lr 3·10⁻⁴) = **0.128**. La desviación entre
  semillas es de 0.013.
- Umbrales: ρ_F − 0.01 = 0.180 y ρ_B − 0.01 = 0.166.
- Shallow − F: **−0.063 [−0.086, −0.039]**.
- Shallow − B: −0.049 [−0.074, −0.025].
- Shallow − EEGNet: −0.028 [−0.042, −0.014].

**Resultado: ρ_S < ρ_B − 0.01. Ni siquiera iguala a las bandas, así que se
para y se cierra la sublínea.** Ninguna ejecución por separado llega al
umbral; la mejor da 0.137. No se calcula la respuesta en frecuencia de los
filtros, porque el plan la reservaba para el caso en que llegara a F.

### Requisitos técnicos

Se cumplen todos:

- la pérdida interna baja respecto a la época 1 en todas las ejecuciones;
- la mejor época es la 3 o la 4, nunca la 1 ni la 60;
- no hay NaN;
- `evaluar.py` reproduce **exactamente** (diferencia < 10⁻⁶) ρ_F = 0.190,
  ρ_B = 0.176 y ρ_EEGNet = 0.156 del paso 2 con la caché. Los SHA-256 de los
  tres archivos de la caché están en `resultados.json`.

El resultado no se debe a un fallo de la cadena.

### Lectura

1. **ShallowConvNet sobreajusta enseguida.** La pérdida de entreno baja de
   0.73 a 0.59 en 11–12 épocas, pero la interna toca su mínimo en la época 3–4
   (≈ 0.69, apenas por debajo del azar, 0.693) y después sube hasta 0.73–0.77
   (`figuras/curvas.png`). La red aprende patrones de las 81 personas de
   entreno que no se trasladan a personas nuevas.
2. **Queda por debajo de todo lo anterior**: de las dos logísticas y de
   EEGNet, con IC que excluyen el 0 en las tres comparaciones. Con lr 1·10⁻³
   es aún peor (0.080).
3. **Por qué la flexibilidad no ayuda aquí.** ShallowConvNet tiene 34 801
   parámetros (EEGNet, 2 097). La mayoría están en el filtro espacial (40 × 40
   × 14 = 22 400). Con una señal tan débil y ≈ 80 personas para entrenar, esa
   capacidad se gasta en diferencias entre personas y grabaciones, no en el
   tiempo en tarea.
4. **El filtro de 1 s no explica el resultado.** Solo añade ≈ 8 600
   parámetros a la versión de 13 muestras. El sobreajuste lo produce sobre
   todo la capa espacial, que es igual en las dos. No se ha probado la versión
   de 13 muestras, así que es una lectura, no una medida.
5. **Lo que no descarta:** una ShallowConvNet mucho más regularizada (menos
   filtros, más dropout o *weight decay*) o con más personas. Eso sería una
   búsqueda de hiperparámetros, que el programa de la sección 7 del PLAN
   excluía por coste. Además, para llegar al umbral tendría que pasar de 0.128
   a 0.180.
6. **Para la memoria:** con dos arquitecturas probadas (EEGNet y
   ShallowConvNet), ninguna CNN pequeña sobre la señal cruda llega a la
   logística sobre la potencia espectral. La más flexible, ShallowConvNet,
   queda peor que la más compacta, EEGNet. Es una cifra de desarrollo en un
   pliegue de `train`.

### Desviaciones del plan

- **Coste:** 53 min en lugar de 4–6 h, porque la parada temprana actuó en la
  época 11–12. No cambia nada del diseño.
- El filtro temporal de 1 s ya estaba en el plan como decisión pendiente; el
  usuario lo aprobó antes de ejecutar.
- Ninguna otra.

### Recuento de miradas

- `train`: pliegue 0 de la semilla 0, 4 ejecuciones de ShallowConvNet (segunda
  arquitectura con las mismas 25 personas de evaluación).
- `val`: 2 (no se cargó).
- `test`: no se cargó.
