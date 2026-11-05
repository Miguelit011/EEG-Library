# Paso 2 — Movimiento y calidad de contacto

> Línea: tiempo en tarea. Fecha: 2026-10-27. **Solo `train`**
> (partición de `matriz_rf`). Los EDF originales se leen en modo solo lectura.

## Plan y criterio de decisión (escrito antes de ejecutar)

Esta sección se redactó **antes** de escribir y ejecutar el script y no se
modifica después. Los resultados van en las secciones siguientes.

### Por qué este paso

El cambio del EEG con el tiempo en tarea podría ser un artefacto: más
movimiento a medida que el alumno se cansa o se aburre, o un contacto de los
electrodos que empeora. Es la crítica más directa a la línea («es el casco»).
Este paso mide movimiento y contacto ventana a ventana y comprueba si explican
el resultado del paso 1.

### Comprobaciones previas (hechas antes de escribir este plan, sin mirar resultados)

- **Alineación ventana ↔ EDF.** El único recorte del pipeline es
  `recortar_registro` (5 s al principio y 5 s al final). El epocado empieza en
  la primera muestra del registro recortado. `make_fixed_length_epochs` podría
  descartar ventanas que caigan en anotaciones `BAD_*` y descolocar
  `tmin_s = 2·i`, pero **en las 1775 grabaciones de `train` de las cuatro
  condiciones `n_epocas` coincide con el esperado por la duración**. Por tanto,
  la ventana `i` cubre `[5 + 2i, 9 + 2i)` s del EDF original.
- **Versiones de export.** Las de 68 y 72 señales (1735 grabaciones de
  `train`) traen `MOT.Q0–Q3`, `MOT.AccX/Y/Z` y `MOT.MagX/Y/Z` a 64 Hz. Las de 60
  y 64 señales (40 grabaciones, todas de M2 a 128 Hz) solo traen
  `MOT.GyroX/Y`: **no tienen acelerómetro ni cuaterniones**. Todas traen
  `CQ.<canal>`, `CQ.Overall` y `EQ.*`. `MOT.Interpolated` marca el 0.1 % de las
  muestras.
- **El `CQ` está casi siempre saturado.** El 98.9 % de las muestras de
  `CQ.<canal>` vale 4 (el máximo) y `CQ.Overall` es constante en el 76 % de las
  grabaciones. Como medida continua del contacto tiene muy poco recorrido.
  `EQ.OVERALL` sí varía (mediana por grabación 75, rango 0–100), pero Emotiv lo
  calcula a partir de la propia señal EEG.

### Datos

- PASAT, PVTB, PVSAT y OA de `train`, etiquetadas, con ≥ 10 ventanas (como en
  el paso 1). Ventanas y features EEG de `features_all`, identidad y partición
  de `matriz_rf` y canales válidos de `procesamiento_log` (funciones del paso
  1).
- EDF de `dataset_final/`, leídos con un lector propio de la cabecera EDF que
  devuelve cada señal a su frecuencia nativa, **sin remuestrear**. MNE
  remuestrearía `MOT` (64 Hz) y `CQ` (escalones) a 256 Hz e introduciría
  oscilaciones.
- **Las 40 grabaciones sin acelerómetro ni cuaterniones** quedan fuera de todo
  análisis que use movimiento, como ya preveía `PLAN.md` (4.5). Para que las
  comparaciones sean justas, **todos los modelos de este paso se ajustan sobre
  el mismo conjunto de grabaciones** (las que tienen movimiento), incluido el
  de solo EEG.
- SHA-256 de los archivos tabulares en `resultados.json`. De los EDF se
  registra el número de archivos leídos (no se calcula el hash de cada uno).

### Medidas por ventana

Sobre las muestras de cada ventana `[5 + 2i, 9 + 2i)` s:

| Medida | Definición | Escala |
|---|---|---|
| `mov_acc` | Desviación típica del módulo de la aceleración `√(AccX² + AccY² + AccZ²)` | `log10(x + 1e-4)` |
| `mov_ang` | Velocidad angular media (°/s) a partir de cuaterniones consecutivos: `θ = 2·arccos(|q_t · q_t+1|)` × 64 Hz | `log10(x + 1e-3)` |
| `cq_validos` | Media de `CQ.<canal>` (0–4) en los canales válidos de la grabación | tal cual |
| `cq_overall` | Media de `CQ.Overall` (0–100) | tal cual |
| `eq_overall` | Media de `EQ.OVERALL` (0–100). Solo en la sensibilidad (ver abajo) | tal cual |

Variables «de artefacto» (principales): `mov_acc`, `mov_ang`, `cq_validos` y
`cq_overall`.

**Sensibilidad con `EQ`** (decisión del autor, 2026-10-27): los análisis 2–4 se
repiten añadiendo `eq_overall` a las variables de artefacto. Como `EQ` se
calcula a partir de la propia señal EEG, si «explica» el efecto puede ser
porque forma parte de él; se lee así y no decide el criterio.

### Qué se calcula

Cada análisis en dos versiones: **grabación completa** e **igual tiempo** (cada
tarea recortada a la duración del OA de su sesión, como en el anexo A del paso
1). La versión principal es la de **igual tiempo**. PVSAT es descriptiva.

1. **¿Aumentan el movimiento o empeora el contacto con el tiempo en tarea?**
   Pendiente por minuto de cada medida en cada grabación, media por persona y
   t frente a 0. Diferencia pareada por persona tarea − OA a igual tiempo.
   Además, la diferencia media último tercio − primer tercio.
2. **Clasificador de solo artefactos.** Las variables de artefacto promediadas
   por tercio y centradas por grabación, con el mismo modelo, los mismos
   tercios y la misma CV que el paso 1. Acierto pareado. Es la **referencia de
   artefactos** de la H4.
3. **EEG + artefactos.** Las 56 `pot_rel` del paso 1 más las variables de
   artefacto, todas centradas por grabación. ¿Cambia el acierto pareado
   respecto a solo EEG?
4. **EEG residualizado.** Dentro de cada grabación, cada una de las 56
   `log10(pot_rel)` por ventana se regresa (MCO) sobre las variables de
   artefacto de la ventana, centradas en la grabación. Las variables
   constantes en esa grabación se omiten. Los residuos se promedian por tercio
   y van al clasificador. Como aquí el logaritmo se toma por ventana (en el
   paso 1 se tomaba sobre la media del tercio), la referencia **«solo EEG»
   se recalcula con el logaritmo por ventana** y es con ella con la que se
   compara.
5. **Contrastes** (bootstrap de 2000 réplicas por persona, semilla 42), en
   PASAT, en PVTB y con PASAT + PVTB juntas (acierto pareado medio de las
   grabaciones de las dos):
   - **EEG − solo artefactos** (pareado por grabación) → anticipa la H4;
   - **EEG residualizado − OA** (OA también residualizado) → ¿sobrevive la
     ventaja sobre OA?
   - EEG + artefactos − EEG y EEG residualizado − EEG, como descripción de
     cuánto se pierde.

### Criterio de continuidad (confirmado por el autor el 2026-10-27)

Se pasa al paso 3 si, **a igual tiempo** y con las variables de artefacto
principales:

- (a) el EEG (las 56 `pot_rel` como en el paso 1) supera al clasificador de
  solo artefactos en PASAT y en PVTB (diferencia > 0 con el límite inferior
  del IC95 > 0); y
- (b) el EEG residualizado mantiene una ventaja sobre OA con PASAT + PVTB
  juntas (estimación puntual > 0; el IC se reporta, pero quien decide es el
  `test`).

Si (a) falla, el cambio del EEG no se distingue de lo que explican el
movimiento y el contacto: la línea se para y se discute. Si solo falla (b), se
discute antes de seguir.

### Qué no se hace aquí

- No se mira `val` ni `test`.
- No se eligen features ni modelo (paso 3).
- No se separa el tiempo en tarea del tiempo desde la colocación (paso 4).

### Decisiones tomadas con el autor antes de ejecutar

1. `EQ.OVERALL`, solo como sensibilidad (el `CQ` está saturado).
2. El criterio de continuidad de arriba (el `PLAN.md` no fijaba ninguno para
   este paso).

### Privacidad

Las tablas del repositorio son agregadas (por condición y persona media). Las
medidas por ventana se calculan en memoria y no se guardan.

### Recuento de miradas

- `val`: 0. `test`: 0.

---

## Resultados

Ejecutado el 2026-10-27 (`ejecucion.log`, `resultados.json`, `tablas/`,
`figuras/`). Se leyeron 1774 EDF en 21 s. Quedan fuera 40 grabaciones sin
movimiento (10 por condición; las de M2 exportadas con 60 o 64 señales).
Quedan 435 grabaciones de PASAT y OA y 432 de PVTB y PVSAT con la grabación
completa, y 427–435 a igual tiempo.

### Veredicto

**El criterio de continuidad se cumple con margen.** A igual tiempo:

- (a) el EEG supera al clasificador de solo artefactos en PASAT (**+0.179**
  [+0.109, +0.254]) y en PVTB (**+0.159** [+0.103, +0.218]);
- (b) el EEG residualizado mantiene la ventaja sobre OA con PASAT + PVTB
  (**+0.120** [+0.061, +0.180]).

El movimiento de la cabeza no aumenta con el tiempo en tarea más que en
reposo. Un clasificador que solo ve movimiento y contacto apenas supera el
azar. Añadir los artefactos o residualizar por ellos casi no mueve el
resultado. **El efecto no es «el casco moviéndose» ni «el contacto
empeorando»**, con dos límites: el `CQ` está saturado, así que el control del
contacto es débil, y no se controla la actividad muscular (EMG).

### 1. ¿Cambian el movimiento y el contacto con el tiempo en tarea?

Pendiente por minuto (t sobre ≈ 120 personas) y, para las tareas, diferencia
con el OA de la misma persona. `tablas/tendencias_artefactos.csv` y
`figuras/artefactos_por_minuto.png`.

| Medida (igual tiempo) | PASAT | PVTB | PVSAT | OA |
|---|---|---|---|---|
| `mov_acc` (log10) | +0.001 (+0.3); − OA −0.002 (−0.4) | −0.002 (−0.5); − OA −0.004 (−1.2) | **−0.017 (−4.5)**; − OA −0.019 (−4.4) | +0.003 (+1.3) |
| `mov_ang` (log10) | +0.013 (+1.7); − OA −0.000 (−0.0) | −0.016 (−2.1); − OA −0.029 (−3.5) | **−0.036 (−5.2)**; − OA −0.049 (−5.8) | +0.013 (+2.8) |
| `cq_validos` (0–4) | +0.002 (+2.0); − OA +0.001 (+0.5) | −0.000 (−0.0) | −0.000 (−0.4) | +0.001 (+0.3) |
| `cq_overall` (0–100) | +0.11 (+1.2) | −0.01 (−0.1) | −0.09 (−1.5) | −0.18 (−0.9) |
| `eq_overall` (0–100) | −0.32 (−0.6) | −1.04 (−2.1); − OA −0.73 (−1.0) | +0.95 (+1.9) | −0.31 (−0.6) |

- **El movimiento no crece con el tiempo en tarea más que en reposo.** En
  PASAT la velocidad angular sube lo mismo que en OA (diferencia 0.000). En
  PVTB **baja** respecto a OA. Con la grabación completa ocurre lo mismo:
  PASAT +0.012 frente a OA +0.013.
- **En PVSAT el movimiento baja claramente** a lo largo de la tarea (t = −4.5
  y −5.2). Es la única tarea con una tendencia de artefacto marcada, y va en
  sentido contrario al de un alumno que se mueve más al cansarse.
- **El contacto (`CQ`) no cambia**, pero tampoco podría: el 99.4 % de las
  ventanas tiene `cq_validos` = 4 y el 78 % de las grabaciones tiene
  `cq_overall` constante. `EQ.OVERALL` baja un poco en PVTB (t = −2.1), sin
  diferencia con OA.

### 2–4. Clasificadores

Acierto pareado, misma CV por persona que el paso 1 y mismo conjunto de
grabaciones para todos los modelos (`tablas/clasificadores.csv`,
`figuras/acierto_por_modelo.png`).

| Modelo (igual tiempo) | PASAT | PVTB | PVSAT | OA |
|---|---|---|---|---|
| EEG (56 `pot_rel`, como el paso 1) | 0.721 | 0.727 | 0.670 | 0.600 |
| **Solo artefactos** | **0.542** | **0.568** | 0.600 | 0.515 |
| EEG + artefactos | 0.721 | 0.715 | 0.658 | 0.602 |
| EEG con log por ventana (referencia) | 0.742 | 0.727 | 0.651 | 0.611 |
| **EEG residualizado** | **0.698** | **0.715** | 0.630 | 0.586 |
| Solo artefactos + EQ | 0.537 | 0.570 | 0.588 | 0.499 |
| EEG + artefactos + EQ | 0.728 | 0.713 | 0.665 | 0.595 |
| EEG residualizado por artefactos + EQ | 0.688 | 0.685 | 0.630 | 0.589 |

Contrastes a igual tiempo (bootstrap de personas; `tablas/contrastes.csv`):

| Contraste | PASAT | PVTB | PASAT + PVTB |
|---|---|---|---|
| **EEG − solo artefactos** | **+0.179** [+0.109, +0.254] | **+0.159** [+0.103, +0.218] | +0.169 [+0.124, +0.213] |
| EEG + artefactos − EEG | +0.000 | −0.012 | −0.006 [−0.020, +0.008] |
| EEG residualizado − EEG (log por ventana) | −0.044 [−0.079, −0.009] | −0.012 | −0.028 [−0.050, −0.006] |
| EEG − OA (en este subconjunto) | +0.121 | +0.127 | +0.124 [+0.061, +0.184] |
| **EEG residualizado − OA residualizado** | +0.111 | +0.129 | **+0.120** [+0.061, +0.180] |
| Ídem, residualizado también por EQ | +0.100 | +0.096 | +0.098 [+0.043, +0.152] |

Con la grabación completa todo es igual o más holgado (EEG − solo artefactos
+0.233 con PASAT + PVTB; residualizado − OA +0.161).

Lectura:

- **El movimiento y el contacto no predicen el tercio.** Solo artefactos da
  0.54–0.57 en PASAT y PVTB, cerca del azar. La excepción es PVSAT (0.600),
  por la bajada del movimiento.
- **Añadirlos al EEG no cambia nada** (−0.006): no aportan información que el
  EEG no tenga.
- **Residualizar quita un poco en PASAT** (−0.044 respecto a su referencia):
  una parte pequeña del cambio de PASAT covaría ventana a ventana con el
  movimiento. En PVTB la pérdida es despreciable. La ventaja sobre OA
  sobrevive entera (+0.120 frente a +0.124).
- **Sensibilidad con EQ:** residualizar también por `EQ` baja la ventaja sobre
  OA a +0.098, que sigue lejos de 0. Como `EQ` sale del propio EEG, esa
  bajada puede ser efecto real arrastrado por la corrección. No cambia la
  conclusión.

### Diagnóstico añadido: la cifra depende de la partición en pliegues

**No estaba en el plan; se añadió después de ver la salida.** En este
subconjunto (40 grabaciones menos que en el paso 1), el mismo clasificador del
paso 1 da OA = 0.600, frente a 0.647. Se comprobó que no es un error: la
función del paso 1 sobre este subconjunto da exactamente 0.800 y 0.600. La
diferencia viene de que `GroupKFold(5)` sin barajar reparte las personas según
el orden de los datos, y al quitar grabaciones cambia el reparto.

`estabilidad_pliegues.py` repite el clasificador del paso 1, con todas las
grabaciones de `train`, sobre 20 particiones por persona barajadas
(`ejecucion_estabilidad.log`, `tablas/estabilidad_pliegues.csv`):

| | Partición del paso 1 | Media de 20 | DE | Mín.–máx. |
|---|---|---|---|---|
| PASAT (completa) | 0.773 | 0.780 | 0.009 | 0.764–0.802 |
| PVTB (completa) | 0.758 | 0.749 | 0.012 | 0.724–0.771 |
| OA | 0.647 | **0.611** | 0.022 | 0.562–**0.647** |
| PASAT + PVTB − OA, completa | 0.118 | 0.153 | 0.020 | 0.108–0.201 |
| **PASAT + PVTB − OA, igual tiempo** | 0.079 | **0.102** | 0.024 | 0.060–0.140 |

- **La partición del paso 1 era, por azar, la más favorable a OA de las 20.**
  Las diferencias frente a OA del paso 1 y de su anexo están
  **infraestimadas**. A igual tiempo, la media sobre particiones es +0.102 y
  no +0.079. Las conclusiones del paso 1 no cambian (eran favorables igual),
  pero sus cifras puntuales tienen una incertidumbre de partición de
  ±0.02–0.05 **que los IC por bootstrap no recogen**.
- OA es la condición más inestable (DE 0.022): tiene menos ventanas por
  tercio y un efecto más débil.
- **No afecta a `test`.** Allí el modelo final se entrena una vez con `train`
  + `val` y se evalúa una vez: no hay pliegues. Sí afecta a las decisiones del
  desarrollo (selección del paso 3 y potencia del paso 4).

### Desviaciones del plan

1. **Diagnóstico de estabilidad de pliegues** (sección anterior), añadido
   después de ver la salida. No cambia ninguna cifra del análisis planificado
   ni el criterio.
2. Se leyeron 1774 EDF y no 1775: la grabación de PVTB con menos de 10
   ventanas se aparta antes de leer, como en el paso 1.

### Límites

- **Contacto:** con el `CQ` saturado, el control del contacto es casi nulo. La
  sensibilidad con `EQ` lo compensa en parte, con la ambigüedad de que `EQ`
  sale del propio EEG.
- **EMG:** el movimiento de la cabeza no mide la tensión muscular. Que la beta
  relativa **baje** con el tiempo va en contra de un EMG creciente (el EMG sube
  la potencia alta), pero no lo descarta del todo.
- **Tiempo desde la colocación:** sigue pendiente (paso 4).

### Para los pasos siguientes (no decidido aquí)

- **Validación cruzada repetida.** Dado el diagnóstico, la selección del paso 3
  y la potencia del paso 4 deberían promediar varias particiones por persona
  (por ejemplo, 20 semillas de `GroupKFold` barajado) en vez de usar una sola.
  Es un cambio al `PLAN.md` que hay que hablar.

### Recuento de miradas

- `val`: 0. `test`: 0.
- `train`: tendencias de 5 medidas × 4 condiciones × 2 versiones; 8 modelos ×
  4 condiciones × 2 versiones; 7 contrastes × 4 grupos × 2 versiones; y el
  diagnóstico de 20 particiones × 4 condiciones × 2 versiones.
