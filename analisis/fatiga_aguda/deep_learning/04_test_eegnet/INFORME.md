# Deep learning — paso 4: EEGNet entrenada con `train` + `val` y evaluada en `test`

> Plan escrito el 2026-10-31, **antes** de ejecutar nada de este paso. Ese
> mismo día se ejecutaron las fases A (`val`) y B (modelo final); sus
> resultados están al final. `test` **no se ha cargado** para el deep
> learning. Las pruebas sobre `test` están en [`PREREGISTRO.md`](PREREGISTRO.md).
> Este informe cubre el cómo: datos, scripts, orden de ejecución y coste.

## Para qué sirve

Los pilotos (pasos 2 y 3) dicen que una CNN pequeña no llega a la logística,
pero es **una cifra de un pliegue de `train`** (25 personas). Este paso la
convierte en una cifra con **las 28 personas de `test`**, comparable con la
logística ya evaluada allí (paso 6 de la línea). Así la memoria puede decir
«CNN frente a logística en personas no vistas», no solo «en desarrollo».

**Lo que no cambia:** ningún veredicto del paso 6 (H1b, H1, H2, H3 y H4).

## Desviaciones respecto a lo escrito antes

| Documento | Qué decía | Qué cambia | Por qué |
|---|---|---|---|
| `03_piloto_shallow/INFORME.md` | Regla del paso 3: parar, no hacer el paso 4 (`04_cv_cnn/`) | No se hace la CV de 5 pliegues, pero sí una evaluación en `test` de EEGNet | Decisión del autor (2026-10-31): obtener una cifra con personas no vistas. **No la motiva ningún resultado nuevo**: los últimos que se vieron son los de los pilotos |
| `deep_learning/README.md` | «No se carga `test`» y «Ya no hay H5 confirmatoria» | `test` se carga **una vez**, con preregistro, como segunda evaluación de la línea | La misma decisión. `06_test/PREREGISTRO.md` ya preveía que, si el deep learning se desarrollaba después, su evaluación en `test` sería exploratoria |
| `deep_learning/README.md` | `val` reservado para **una** comprobación de la configuración final | Se mantiene: es la fase A de este paso | — |

**Decisiones del autor sobre este plan (2026-10-31):**

1. Los datos en bruto del test se guardan en `EEG-Data/processed/`, con la
   misma excepción que en el paso 6.
2. No se evalúa en `test` la logística sobre el espectro fino (F). La única
   referencia es la logística preregistrada del paso 6.
3. Se actualiza `deep_learning/README.md` (reglas y fila 4).

## Qué se reutiliza de los pilotos (sin cambios)

Lo fijado en `02_piloto_cnn/INFORME.md`:

- **Datos:** ventanas de `EEG-Data/ventanas/*.npy`, remuestreadas a 128 Hz con
  `resample_poly(x, 1, 2)`, z-score por grabación y canal, canales excluidos a
  0 y **mapa de máscara** como segunda entrada: `(2, 14, 512)`.
- **Filas:** las de `comun.cargar` (mínimo de 10 ventanas y `persona_uid`
  conocido); partición e identidad de `matriz_rf.parquet`.
- **Etiqueta:** 0 en el primer tercio y 1 en el último; el tercio medio no se
  entrena, pero se puntúa.
- **Modelo:** EEGNet-8,2 con BN tras la conv temporal (2 097 parámetros), un
  modelo conjunto para PASAT + PVTB, dropout de 0.25, AdamW (`weight_decay` =
  0.01), lote de 256, `BCEWithLogitsLoss`, **lr = 3·10⁻⁴** (la elegida en el
  piloto por la pérdida interna) y semillas 0, 1 y 2.
- **Medida:** `rho_ventanas` (ρ de Spearman por grabación entre la puntuación
  de cada ventana y su instante), con las funciones `rho` e `ic` de
  `02_piloto_cnn/evaluar.py` y `medidas.py`.
- **Entorno:** `C:\venvs\eeg-dl` (Python 3.10.11, PyTorch 2.14.1+cpu) para
  entrenar; `.venv` para preparar y evaluar.

**Se descarta ShallowConvNet:** en el paso 3 quedó por debajo de EEGNet
(−0.028 [−0.042, −0.014]) y sobreajusta desde la época 3–4.

## Qué es nuevo

1. **Más personas para entrenar:** las de `train` en la fase A y las de
   `train` + `val` en la fase B, frente a las 81 del piloto.
2. **Épocas fijas en el modelo final.** Al entrenar con `train` + `val` no
   queda conjunto para la parada temprana. Se usa la **mediana de las mejores
   épocas** de las 3 semillas de la fase A (`PLAN.md`, 7.5).
3. **Se puntúan también las ventanas de OA**, con el modelo de tareas, como
   control descriptivo de deriva común (asentamiento del casco).
4. **Referencia logística:** la del test, la que está preregistrada
   (`rel` + `LogisticRegression(C=0.01)`, una por condición). No son las F y B
   de los pilotos. En `test` **no se vuelve a entrenar ni a aplicar**: se leen
   sus `rho_ventanas` por grabación de
   `EEG-Data/processed/fatiga_aguda_test_20261027-190547/predicciones_por_grabacion.parquet`
   (solo lectura, con SHA-256) y se emparejan por `archivo_origen`.

## Fases

### Fase A — comprobación en `val` (mirada única del deep learning a `val`)

- Entrena con las personas de `train`. Un 15 % de ellas (semilla 0) forma el
  conjunto interno para la parada temprana: paciencia de 8 y **máximo de 80
  épocas**. El piloto paró entre las épocas 39 y 48 con un máximo de 60; con
  más datos podría necesitar más.
- 3 semillas (0, 1 y 2) y lr 3·10⁻⁴.
- Puntúa las ventanas de PASAT, PVTB y OA de `val`.
- Calcula en `val` las mismas cifras que el preregistro calculará en `test`
  (sección 4 del preregistro), con la logística del test reentrenada con
  `train` (funciones de `06_test/evaluacion_test.py`, sin modificarlas). Las
  cifras de esa logística en `val` ya se vieron en el ensayo general del paso
  6: no son una mirada nueva.
- **Para qué:** (1) ensayo general de toda la cadena con personas no vistas,
  (2) fijar el número de épocas y (3) dar una cifra de desarrollo.
- **No es una puerta.** La fase B y el test se hacen salga lo que salga en
  `val`, salvo un fallo técnico. Así `val` no se convierte en un grado de
  libertad más.

**Requisitos técnicos** (si alguno falla, se corrige el código y se repite la
fase A; se anota):

- la pérdida interna baja respecto a la época 1;
- la mejor época no es la 1 ni la 80;
- no hay NaN;
- la logística reentrenada reproduce en `val` las ρ medias y los recuentos de
  grabaciones del ensayo general del paso 6 (PASAT, PVTB y OA, las dos
  versiones; diferencia < 10⁻⁶). El ensayo no guardó las ρ por grabación.

### Fase B — modelo final

- Entrena con **todas** las personas de `train` + `val`, sin conjunto interno,
  durante las épocas fijadas en la fase A, con las semillas 0, 1 y 2.
- Guarda los pesos (`.pt`) y su **SHA-256**, que pasan a
  `configuracion_dl.json` junto con las épocas, lr, semillas, versiones y el
  SHA-256 del tensor de entrenamiento.
- No toca `test`.

### Fase C — preregistro fijado y commit (lo hace el autor)

Se rellenan en `PREREGISTRO.md` las cifras de la fase A (épocas y requisitos)
y se commitean juntos `PREREGISTRO.md`, los scripts y
`configuracion_dl.json`. **Desde ese commit, el modelo ya no puede cambiar**:
los pesos están identificados por su hash antes de abrir `test`.

### Fase D — evaluación única en `test`

Una sola ejecución de `ejecutar_test.ps1` (ver «Scripts»), con `git status`
limpio. Construye el tensor de `test`, lo puntúa con los pesos congelados
(comprobando su hash) y calcula todo lo del preregistro. Se reporta entero.

## Scripts

| Script | Entorno | Qué hace |
|---|---|---|
| `preparar.py --val` | `.venv` | Tensor y máscara de PASAT y PVTB de `train` + `val` y de OA de `val` (el OA de `train` no se usa); tabla de ventanas con papel (entreno, interno, `val`); referencias logísticas en `val` |
| `entrenar.py --fase-a` | `eeg-dl` | 3 semillas con parada temprana; puntuaciones de `val`; checkpoints por época |
| `evaluar.py --val` | `.venv` | Cifras de la fase A, requisitos, épocas fijadas y figuras |
| `entrenar.py --fase-b` | `eeg-dl` | 3 semillas con todo `train` + `val` y épocas fijas; pesos y hashes en `configuracion_dl.json` |
| `preparar.py --test --permitir-test` | `.venv` | Tensor de `test`. Se niega a correr sin las dos opciones, con `git status` sucio o si la carpeta de salida ya existe |
| `entrenar.py --puntuar-test --permitir-test` | `eeg-dl` | Puntúa `test` con los pesos congelados; comprueba sus SHA-256 antes |
| `evaluar.py --test --permitir-test` | `.venv` | Pruebas, descriptivos y sensibilidades del preregistro |
| `ejecutar_test.ps1` | — | Lanza los tres pasos de `test` seguidos y para si alguno falla |

Los scripts **importan** las funciones de `02_piloto_cnn/` y de `06_test/`, sin
modificarlas. Lo que no cuadre se adapta en el script nuevo.

**Filtro de `test`:** los modos `--val` y `--fase-*` filtran la lista de
archivos por la `split` de `matriz_rf.parquet` **antes** de abrir ningún
`.npy`, con `assert "test" not in particiones`.

## Dónde se guarda cada cosa

| Qué | Dónde |
|---|---|
| Tensores, tabla de ventanas, checkpoints y puntuaciones | `TFG\EEG-DL-cache\test_eegnet\` (fuera de los repos, sin git). Se borra al cerrar el paso, como en los pilotos |
| Pesos finales (`.pt`, ≈ 10 KB cada uno) | En la caché hasta el test. Después, con los datos en bruto del test (abajo) |
| Datos en bruto del test: ρ por grabación con `persona_uid`, réplicas bootstrap y pesos | `EEG-Data/processed/fatiga_aguda_dl_test_<fecha-hora>/`. Es la misma excepción a la regla de solo lectura de `EEG-Data` que en el paso 6, autorizada por el autor el 2026-10-31 |
| En el repositorio | Solo `resultados_val.json`, `resultados_test.json` (sin identificadores), `configuracion_dl.json`, `tablas/` agregadas, `figuras/` y los `.log` |

## Coste

Medido en el piloto: ≈ 1.1 ms por ventana y época (40 s por época con 35 000
ventanas).

| Fase | Ventanas por época | Cómputo |
|---|---|---|
| Preparar `train` + `val` (lectura de los `.npy`) | — | ≈ 10–15 min |
| A: 3 semillas con parada temprana (≈ 45–60 épocas) | ≈ 45 000 | ≈ 2.5–3 h |
| B: 3 semillas con épocas fijas | ≈ 60 000 | ≈ 2.5–3 h |
| D: preparar y puntuar `test`; bootstrap | — | ≈ 15 min |
| **Total** | | **≈ 5.5–6.5 h de CPU** (una noche) |

Trabajo humano: ≈ 8–12 h (adaptar los tres scripts, el `.ps1`, comprobaciones
e informe).

**Memoria:** el tensor de `train` + `val` (PASAT + PVTB, y OA de `val`)
ocupa ≈ 3 GB en float32. Se abre con `mmap_mode="r"`, como en el piloto, que se
cortó una vez por falta de memoria. Hay checkpoint por época y se reanuda
desde ahí.

## Antes de lanzar

- Comprobar las temperaturas con HWiNFO64 durante la primera hora. Si se
  acercan a 95 °C, poner el «estado máximo del procesador» al 99 %.
- Pausar Windows Update durante las fases A y B.
- Comprobar que `C:\venvs\eeg-dl` sigue teniendo las versiones del piloto.

## Recuento de miradas

- `train`: entrenamiento de las fases A y B (sin evaluar en `train`).
- `val`: **primera** mirada del deep learning (la línea lleva 4 usos de `val`
  con la logística, `06_test/PREREGISTRO.md`, sección 9).
- `test`: **segunda** apertura para la línea (la primera fue la logística, el
  2026-10-27) y primera para el deep learning.

## Resultados

### Preparación (`preparar.py --val`, 73 s)

| Papel | Personas | PASAT (grab. / ventanas) | PVTB (grab. / ventanas) | OA (grab.) |
|---|---|---|---|---|
| Entreno | 102 | 376 / 33 332 | 373 / 32 473 | — |
| Interno | 18 | 69 / 6 137 | 69 / 5 922 | — |
| `val` | 25 | 88 / 7 814 | 87 / 7 450 | 86 |

1 148 `.npy`, ninguno desalineado. Tensor (97 941, 14, 512), 2.81 GB, con un
13.8 % de canales excluidos. La logística del paso 6 reentrenada con `train`
reproduce exactamente el ensayo general en `val`, en las dos versiones y en
las tres condiciones.

### Fase A (`entrenar.py --fase-a`, 70 min; `evaluar.py --val`, 18 s)

Paró por paciencia en las épocas 46, 32 y 35. Las mejores fueron la 38, la 24
y la 27, así que **E = 27**. Se cumplen los cuatro requisitos técnicos. Cada
época tardó ≈ 35 s con 44 272 ventanas (≈ 0.8 ms por ventana y época), algo
más rápido que en el piloto.

Cifras de desarrollo en `val` (tabla completa en `PREREGISTRO.md`, sección 12):

- **P1:** Δρ = **+0.033** [+0.004, +0.065]. En `val`, EEGNet supera a la
  logística del paso 6 (ρ 0.166 frente a 0.133). Es el signo contrario al
  que esperaba la sección 7 del preregistro.
- **A igual tiempo, la diferencia desaparece:** −0.001 [−0.046, +0.047].
- **Control OA:** ρ +0.026 [−0.032, +0.084], sin diferencia clara con 0.
- **Semillas:** muy estables (ρ entre 0.160 y 0.172).
- **Sensibilidades:** sin la persona expuesta o sin las grabaciones a 128 Hz,
  P1 apenas cambia.

Figuras: `figuras/curvas_fase_a.png` (pérdida interna por época) y
`figuras/rho_val.png`.

### Fase B (`entrenar.py --fase-b`, 65 min)

3 semillas × 27 épocas con 62 661 ventanas de 145 personas (`train` +
`val`), sin NaN. La pérdida de entrenamiento final es 0.678–0.679. Los pesos y
sus SHA-256 están en `configuracion_dl.json`
(`tablas/curvas_fase_b.csv`), y también en el preregistro (sección 4.1).

**Coste real hasta aquí:** ≈ 2 h 20 min de CPU, frente a las 5.5–6.5 h
estimadas.

### Fase C

Pendiente: el commit del autor que fija el preregistro. Fase D (test): sin
ejecutar.
