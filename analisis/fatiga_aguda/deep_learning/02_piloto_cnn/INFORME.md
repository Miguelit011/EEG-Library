# Deep learning — paso 2: piloto de EEGNet (un pliegue)

> Plan escrito el 2026-10-29, después del paso 1 y **antes** de instalar nada
> ni ejecutar este paso. Usa **solo `train`** y no carga ni `val` ni `test`.
> Es de desarrollo.

## Para qué sirve

El paso 1 dio una mejora pequeña del espectro fino sobre las bandas (D1 =
+0.019 [+0.013, +0.026]), sin mejora en OA. Antes de gastar las 25–40 h de CPU
del paso 3, este piloto responde a tres preguntas prácticas:

1. **¿Funciona la cadena completa?** Construcción del tensor, normalización,
   máscara, entrenamiento, puntuación *out-of-fold* y cálculo de
   `rho_ventanas` con las mismas funciones que la logística.
2. **¿Qué lr y cuántas épocas?** Hay que fijarlos para el paso 3, que no tiene
   presupuesto para buscarlos.
3. **¿Llega EEGNet al nivel de la logística?** En un solo pliegue, como
   orientación, con una regla de parada fijada aquí.

**Lo que no responde:** si la CNN supera a la logística. Con un pliegue (≈ 24
personas de evaluación), el IC es demasiado ancho. Eso es el paso 3.

## Cambio respecto al PLAN (sección 7)

**La referencia de la CNN es la logística sobre el espectro fino (F, C =
0.003), no la de bandas.** Tras el paso 1, F es el mejor modelo clásico. Si la
CNN solo iguala a las bandas, no aporta nada que no dé ya una logística más
rica. Se reportan las dos.

## Datos

- **Filas:** las de `comun.cargar({"train"})`, PASAT y PVTB, **grabación
  completa** (la versión de D1). Mínimo de 10 ventanas por grabación.
- **Ventanas crudas:** `EEG-Data/ventanas/*.npy` (14 × 1024 a 256 Hz),
  alineadas con `features_all` (comprobado en el paso 1: 0 desalineados).
- **Remuestreo a 128 Hz** (512 muestras) con `scipy.signal.resample_poly(x, 1,
  2)` por ventana. La señal está filtrada a 1–30 Hz, así que no se pierde
  contenido.
- **Normalización por grabación:** z-score por canal con la media y la
  desviación de **todas** las ventanas de la grabación (también el tercio
  medio). Es simétrica respecto a la etiqueta, así que no hay fuga.
- **Canales excluidos** (Fase 11.5): se ponen a 0 tras normalizar, y la red
  recibe un **segundo mapa de entrada con la máscara** (1 = válido, 0 =
  excluido, constante en el tiempo). Entrada: `(2, 14, 512)`.
- **Etiqueta:** 0 en el primer tercio y 1 en el último. El tercio medio no se
  entrena, pero se puntúa (lo necesita `rho_ventanas`).
- **Volumen:** ≈ 78 000 ventanas de PASAT + PVTB en `train`, ≈ 2.2 GB en
  float32 a 128 Hz. Cabe en RAM.

## Pliegue y partición interna

- **Pliegue 0 de la semilla 0** de `comun.pliegues_por_persona` (la misma
  asignación que la logística en los pasos 3 y 1). ≈ 97 personas para
  entrenar y ≈ 24 para evaluar.
- **Parada temprana:** el 15 % de las personas de entrenamiento (≈ 15),
  elegidas con una semilla fija, forman un conjunto interno. Las mismas en
  todas las ejecuciones.
- **Referencias en el mismo pliegue:** logística F (C = 0.003) y logística B
  (C = 0.03), entrenadas con **todas** las personas de entrenamiento del
  pliegue (la logística no necesita conjunto interno), con el esquema y las
  features del paso 1.

## Modelo y entrenamiento

- **EEGNet-8,2** a 128 Hz (apéndice B del PLAN): conv temporal (1, 64) con
  F1 = 8 → conv espacial depthwise (14, 1) con D = 2 → BN + ELU → AvgPool (1, 4)
  → dropout → separable (1, 16) con F2 = 16 → BN + ELU → AvgPool (1, 8) →
  dropout → lineal a 1 logit. La primera conv recibe 2 mapas (señal y
  máscara). Dropout de 0.25 (el valor de Lawhern et al. para evaluación entre
  personas).
- **Un modelo conjunto para PASAT + PVTB.** La logística va por separado, pero
  la CNN necesita más datos. La ρ se calcula por condición igualmente.
- AdamW (`weight_decay` = 0.01), lote de 256, `BCEWithLogitsLoss`.
- **Parada temprana** sobre la pérdida del conjunto interno: paciencia de 8
  épocas y máximo de 60. Se guarda el modelo de la mejor época.
- **Puntuación:** logit de cada ventana de las personas de evaluación (los tres
  tercios) → `rho_ventanas` por grabación.
- **Checkpoint** al final de cada época: modelo, optimizador, época y semilla.
  Si algo se corta, se reanuda desde ahí.
- `torch.set_num_threads(6)`, como en la medida de coste del PLAN (7.3).

## Ejecuciones (lista cerrada)

| # | lr | Semilla de inicialización | Para qué |
|---|---|---|---|
| 1 | 3·10⁻⁴ | 0 | Rejilla de lr |
| 2 | 1·10⁻³ | 0 | Rejilla de lr (el valor del PLAN) |
| 3 | 3·10⁻³ | 0 | Rejilla de lr |
| 4 | mejor lr | 1 | Variabilidad entre semillas |
| 5 | mejor lr | 2 | Variabilidad entre semillas |

- **Mejor lr:** la de menor pérdida en el conjunto interno (no la ρ de
  evaluación, para no elegir con las personas con las que se compara).
- Coste estimado: ≈ 35 000 ventanas por época × 1 272 ventanas/s ≈ 30–40 s
  por época, y ≈ 15–30 min por ejecución con parada temprana. Con la
  construcción del tensor, **≈ 2–3 h de CPU**.

## Qué se mide

- Por ejecución: curva de pérdida (entrenamiento e interna), mejor época,
  segundos por época, ρ media de PASAT, de PVTB y de las dos juntas (personas
  de evaluación), AUC por ventana en los tercios.
- Las mismas cifras para la logística F y B en el mismo pliegue.
- Diferencia pareada por grabación (EEGNet − F) con IC95 por bootstrap de
  personas. **Es orientativa**: 24 personas.
- Desviación típica de la ρ entre las 3 semillas de la mejor lr: dice cuántas
  semillas hacen falta en el paso 3.

## Regla de decisión (fijada antes de ejecutar)

Con ρ_CNN la media de las 3 semillas de la mejor lr (PASAT + PVTB) y ρ_F y ρ_B
las de las logísticas en el mismo pliegue:

| Resultado | Lectura | Qué se propone |
|---|---|---|
| ρ_CNN ≥ ρ_F − 0.01 | EEGNet llega al nivel de la mejor logística | Paso 3 (CV de 5 pliegues, con ShallowConvNet y el control OA) |
| ρ_B − 0.01 ≤ ρ_CNN < ρ_F − 0.01 | Iguala a las bandas, pero no al espectro fino | Se valora. Lo más probable es parar y reportar «CNN ≈ logística de bandas» |
| ρ_CNN < ρ_B − 0.01 | Ni siquiera iguala a las bandas | Parar y documentar |

**Requisitos técnicos** (si alguno falla, se corrige y se repite, sin pasar de
la regla):

- la pérdida interna baja respecto a la época 1;
- la mejor época no es la 1 ni la 60 (si es la 60, faltan épocas);
- no hay NaN;
- la logística F en el pliegue 0 da una ρ del orden de la del paso 1 (≈ 0.17).

**Por qué 0.01:** es el error típico aproximado de una diferencia pareada de ρ
con ≈ 24 personas. Por debajo de eso, un pliegue no distingue.

## Entorno (decisión pendiente)

PyTorch no se instala en `.venv` (regla del PLAN, 7.7). Propuesta:

- **Entorno aparte en ruta corta:** `C:\venvs\eeg-dl` (Python 3.10, PyTorch
  CPU, numpy, pandas, scipy y pyarrow). La ruta corta evita el fallo de rutas
  largas de Windows que dio la instalación de PyTorch (PLAN 7.7). Al estar
  fuera del repo, no hace falta tocar `.gitignore`.
- **Tres scripts, para que las métricas salgan del entorno validado:**
  1. `preparar.py` (`.venv`): construye el tensor, la máscara y las etiquetas,
     calcula las referencias logísticas y guarda la asignación de pliegues.
  2. `entrenar.py` (`C:\venvs\eeg-dl`): entrena EEGNet y guarda las
     puntuaciones de las ventanas de evaluación y los checkpoints.
  3. `evaluar.py` (`.venv`): calcula `rho_ventanas`, los contrastes y las
     figuras con `comun.py` y `medidas.py`.
- **Caché intermedia** (tensor ≈ 2.2 GB, puntuaciones y checkpoints): fuera de
  los repos y de `EEG-Data`. Propuesta: `C:\Users\migue\Desktop\TFG\EEG-DL-cache\`,
  sin git, con la misma regla que los datos (son datos de menores derivados:
  nunca a git ni a servicios externos). Se borra al cerrar la sublínea.

## Antes de lanzar

- Limpiar el polvo del equipo y comprobar las temperaturas con HWiNFO64 durante
  la primera ejecución. Si se acercan a 95 °C, poner el «estado máximo del
  procesador» al 99 % (desactiva el turbo).
- Pausar Windows Update mientras dure el piloto.

## Qué se produce

- `preparar.py`, `entrenar.py`, `evaluar.py`, sus `.log` y `resultados.json`
  (SHA-256 de los datos leídos, hash combinado de los `.npy` y versiones de
  PyTorch y de Python).
- `tablas/` (agregados): `ejecuciones.csv` (una fila por ejecución) y
  `curvas.csv` (pérdida por época).
- `figuras/`: curvas de pérdida y ρ por modelo.

## Coste

- CPU: ≈ 2–3 h.
- Trabajo: ≈ 4–6 h (entorno, tres scripts, comprobaciones e informe).

## Recuento de miradas

- `train`: pliegue 0 de la semilla 0, 5 ejecuciones de EEGNet y 2 logísticas.
- `val`: 2 (no se carga).
- `test`: no se carga.

## Resultados

*(Pendiente de ejecutar.)*
