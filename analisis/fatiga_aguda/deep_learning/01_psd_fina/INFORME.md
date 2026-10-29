# Deep learning — paso 1: espectro fino frente a bandas (filtro barato)

> Plan escrito el 2026-10-29, **después** de la evaluación única en `test` de
> la línea (paso 6) y **antes** de ejecutar nada de este paso. Usa **solo
> `train`** (121 personas con etiqueta en las condiciones de la línea) y no
> carga ni `val` ni `test`. Es de desarrollo y no cambia ningún veredicto.

## Por qué se hace

Una CNN sobre señal cruda puede aprender, en principio, tres tipos de
información que las 56 `log10(pot_rel)` (14 canales × 4 bandas) no tienen:

1. **Detalle espectral dentro de las bandas.** Por ejemplo, que el pico de
   alfa se desplace a frecuencias más bajas con el tiempo en tarea (un
   marcador conocido de fatiga que una banda fija de 8–13 Hz no ve), o un
   cambio en la beta alta y no en la baja.
2. **Información espacial**: acoplamiento entre canales y filtros espaciales.
3. **Información temporal fina** dentro de los 4 s: forma de onda, ráfagas y
   no estacionariedad.

El punto 1 se puede medir casi gratis con la logística ya fijada, usando la PSD
de 1 Hz en lugar de las bandas. ShallowConvNet, el comparador natural de la
logística (PLAN 7.2), aprende sobre todo eso: potencia en bandas que elige
ella misma. Así que este paso sirve de **filtro**. Si el espectro fino no
mejora a las bandas, la vía por la que una CNN pequeña tiene más opciones de
ganar está vacía. Si mejora, se sabe en qué frecuencias, y eso orienta (y
permite interpretar) la CNN.

**Lo que este paso no demuestra.** Un resultado nulo no descarta los puntos 2
y 3. Solo baja la expectativa para la CNN. Se reporta así.

## Datos

- **Ventanas crudas:** `EEG-Data/ventanas/<archivo>.npy`, forma
  `(n_ventanas, 14, 1024)`, float32, 256 Hz. La ventana `i` del `.npy` es la
  fila `i` de `features_all.parquet` para ese `archivo_origen`
  (`procesado/batch.py`, que lo comprueba al generarlas).
- **Partición e identidad:** `matriz_rf.parquet`, como en la línea. Se toma la
  lista de archivos con `split == "train"` **antes** de abrir ningún `.npy`
  (con `assert`).
- **Condiciones:** PASAT, PVTB y OA (las de la H1). OC, como descriptiva. No
  entran ni PVSAT ni STROOP ni MSIT.
- **Filas, recorte y exclusiones:** las de `comun.cargar` y `comun.versiones`
  del paso 3 (mínimo de 10 ventanas, recorte a igual tiempo con la duración
  del OA de la sesión). Los canales excluidos por calidad se anulan tras el
  centrado, igual que en el paso 3.
- **Volumen:** ≈ 1 500 archivos, ≈ 7–8 GB de lectura del NVMe. La PSD de 1 Hz
  por ventana (≈ 150 000 ventanas × 420 valores, ≈ 250 MB en float32) se
  calcula **en memoria y no se guarda**.

## Las dos representaciones

| Id | Features por ventana | Nº |
|---|---|---|
| **B** (referencia, la del paso 3) | `log10(pot_rel)` por canal en delta, theta, alfa y beta, de `features_all` | 56 |
| **F** (espectro fino) | `log10(PSD(f) / Σ PSD(1–30 Hz))` por canal, en f = 1, 2, …, 30 Hz | 420 |

- **Cálculo de F:** la PSD se obtiene con `calcular_psd` de la librería (Fase
  8: Welch, sub-ventanas de 1 s con Hann y 50 % de solape, 1–30 Hz). Se
  construye un `mne.EpochsArray` desde el `.npy`. No se reimplementa Welch.
- **Relativa, como B:** se divide entre la potencia total de 1–30 Hz del canal
  y la ventana. Así B y F solo se diferencian en la resolución en frecuencia,
  no en la normalización.
- **Mismo tratamiento que B:** para entrenar, media lineal de la potencia
  relativa en cada tercio y después `log10`. Para puntuar ventanas, `log10`
  por ventana. En los dos casos se centra por grabación y se ponen a 0 los
  canales excluidos.

## Comprobación de integridad (antes de cualquier modelo)

1. **Alineación:** para cada archivo, `n_ventanas` del `.npy` es igual al
   número de filas de `features_all`. Los que no cumplan se excluyen y se
   cuentan. Si son más del 1 %, se para.
2. **Reproducción de las bandas:** sumando F por bandas (con las máscaras de
   `BANDAS_DEFECTO`) se recalcula `pot_rel` y se compara con la de
   `features_all`. Criterio: correlación ≥ 0.999 en cada una de las 56
   columnas. **Si no se cumple, se para y se investiga** antes de seguir: la
   comparación B frente a F solo tiene sentido si las dos salen de la misma
   señal.

## Modelo y validación (fijados antes de ejecutar)

- Logística con `StandardScaler`, un modelo por condición (`separado`),
  entrenada con el primer y el último tercio. Es la configuración del paso 3.
- **Regularización:** con 420 features, C = 0.01 puede no ser el óptimo de F.
  Para no favorecer a ninguna de las dos, **las dos** se evalúan con la misma
  rejilla C ∈ {0.001, 0.003, 0.01, 0.03, 0.1} y se compara **la mejor de cada
  una**. La mejor C es la de mayor ρ media de PASAT + PVTB con la grabación
  completa. Se reporta la rejilla entera.
- **CV repetida:** 20 particiones por persona (semillas 0–19, 5 pliegues,
  `comun.pliegues_por_persona`). Las dos representaciones y todas las C usan
  **la misma asignación** de personas a pliegues.
- **Medida:** `rho_ventanas`, igual que `reajuste_medida.py`. Se calcula la
  puntuación *out-of-fold* de cada ventana, se promedia sobre las 20
  particiones y se halla la ρ de Spearman con el instante por grabación.
  Como cifra interpretable, también el acierto pareado de tercios.
- **IC95:** bootstrap de personas (`medidas.bootstrap`) sobre la diferencia
  **pareada por grabación** F − B.

## Contrastes

| # | Qué | Versión | Papel |
|---|---|---|---|
| **D1** | Δρ (F − B) en PASAT + PVTB | Grabación completa | **Principal**: ¿hay información espectral extra sobre el tiempo en tarea? |
| D2 | Δρ (F − B) en PASAT y en PVTB por separado | Grabación completa | Descriptivo |
| D3 | Δρ (F − B) en OA y en OC | — | ¿La mejora es común con el reposo (deriva) o específica de la tarea? |
| D4 | Δ de la H1 (tareas − OA), F frente a B | Igual tiempo | Descriptivo. ¿Ayudaría F a la pregunta que quedó abierta? |

**Se elige la grabación completa para D1** porque la pregunta es si hay
información extra, no si es específica de la tarea (eso es D3 y D4), y porque
usa todas las ventanas.

## Criterio de decisión (fijado antes de ejecutar)

| Resultado de D1 | Lectura | Qué se propone |
|---|---|---|
| Δρ ≥ +0.02 **y** límite inferior del IC95 > 0 | **Hay margen:** el espectro fino tiene información que las bandas pierden | Piloto de CNN (paso 2) y mirar en qué frecuencias está la mejora |
| Límite inferior del IC95 > 0, pero Δρ < +0.02 | Mejora real pero pequeña | Decide el autor. Una CNN difícilmente aporta algo apreciable por la vía espectral |
| IC95 incluye 0, o Δρ < 0 | **Sin margen:** las bandas recogen lo que da el espectro | La CNN solo tiene sentido como convergencia (CNN ≈ logística) o por los puntos 2 y 3. Decide el autor si se hace el piloto |

- **Por qué +0.02.** En `train`, la ρ de PASAT + PVTB a igual tiempo es
  ≈ 0.14 (paso 3, reajuste 1). Un +0.02 es una mejora relativa de ≈ 15 %, más
  o menos lo mínimo que justificaría 30–45 h de CPU. Es una convención, no una
  prueba, y se declara así.
- **Matiz de D3.** Si F mejora OA tanto como las tareas, la información extra
  es probablemente deriva común (asentamiento del casco), no tiempo en tarea.
  Se dice explícitamente, sea cual sea D1.

## Interpretación si hay margen

- **Perfil de frecuencias:** coeficientes medios de la logística F por Hz
  (promediados sobre canales y pliegues) en PASAT, PVTB y OA, en una figura.
- **Señales de alarma de artefacto:** si la mejora se concentra en 20–30 Hz
  (posible EMG) o en 1–3 Hz (parpadeo residual o deriva de contacto), se
  anota como sospechosa. En ese caso, en el paso 2 se añade a la CNN la
  referencia de movimiento y contacto del paso 2 de la línea.

## Qué se produce

- `psd_fina.py` (se ejecuta desde la raíz del repo con `.venv/Scripts/python.exe`;
  no necesita PyTorch). Reutiliza `03_clasificador/comun.py` y `medidas.py`
  sin modificarlos. Lo que esos módulos asumen de las 56 columnas se escribe en
  paralelo dentro del script nuevo.
- `ejecucion.log`, `resultados.json` (con el SHA-256 de `features_all.parquet`,
  `matriz_rf.parquet` y `procesamiento_log.csv`, y un hash combinado de los
  `.npy` leídos).
- `tablas/`, solo agregados: `integridad.csv`, `rejilla.csv` (representación ×
  C × condición × versión) y `contrastes.csv` (D1–D4).
- `figuras/`: Δρ por condición y perfil de frecuencias.

## Coste

- CPU: lectura y PSD, unos minutos. La logística con 20 particiones × 5 C × 2
  representaciones × 4 condiciones × 2 versiones, del orden de 10–30 min.
- Trabajo: ≈ 3–4 h (script, comprobaciones e informe).

## Recuento de miradas

- `train`: se añaden 2 representaciones × 5 C (las 32 configuraciones del paso
  3 ya se miraron en `train`).
- `val`: sigue en 2 (no se carga).
- `test`: no se carga (gastado el 2026-10-27).

## Resultados

*(Pendiente de ejecutar.)*
