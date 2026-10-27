# Paso 1 — Reproducir y describir

> Línea: fatiga aguda (tiempo en tarea). Fecha: 2026-10-27. **Solo `train`**
> (partición de `matriz_rf`): ni `val` ni `test` se cargan en este paso.

## Plan y criterio de decisión (escrito antes de ejecutar)

Esta sección se redactó **antes** de escribir y ejecutar el script y no se
modifica después. Los resultados van en las secciones siguientes.

### Por qué este paso

Las cifras de `PLAN.md` (secciones 3.2 y 3.3) salen de sondeos hechos fuera
del repositorio con dos defectos conocidos (`PLAN.md`, 3.5): filtraron por la
`split` de `features_all`, anterior a las correcciones de la Fase 12, y usaron
los 14 canales, también los excluidos por calidad. Este paso las rehace desde
el repositorio con los dos defectos corregidos y mide cuánto cambian. Nada de
la sección 3 del plan es citable hasta que este paso lo reproduzca.

### Datos

- **Ventanas:** `features_all.parquet` (`pot_abs_*`, `pot_rel_*`, `tmin_s`,
  `indice_ventana`, `archivo_origen`, `condicion`). Se leen **solo las filas de
  los archivos de `train`** (filtro en la lectura del parquet), así que las
  ventanas de `val` y `test` no llegan a cargarse.
- **Partición e identidad:** `matriz_rf.parquet` (`archivo`, `persona_uid`,
  `sesion_uid`, `split`). Solo grabaciones etiquetadas (`persona_uid` no nulo)
  con `split == "train"`.
- **Canales excluidos:** `procesamiento_log.csv`, con
  `anotar_validez_de_canales` de la librería (Fase 11.5), que además comprueba
  que la máscara cuadra con `n_canales_buenos`.
- SHA-256 de los tres archivos en `resultados.json`.

**Incidencia de la partición (3.5).** Se cuentan las grabaciones y personas
cuya `split` difiere entre `features_all` y `matriz_rf`, comparando **solo las
dos columnas de partición** (sin leer EEG de `val` ni de `test`).

### Definiciones

- Tiempo en la grabación: `t_rec = tmin_s / 60` (minutos). `dur` = `tmin_s`
  máximo de la grabación.
- **Mínimo de 10 ventanas por grabación** para entrar en cualquier análisis
  (pendientes y clasificador). El sondeo solo lo aplicaba a las pendientes; se
  generaliza para que todas las cifras salgan del mismo conjunto, y se reporta
  cuántas grabaciones caen por esta regla.
- **Regiones.**
  - Para reproducir 3.2 se usan las del sondeo: posterior = P7, O1, O2, P8;
    frontal = AF3, F3, F4, AF4.
  - Para las curvas por minuto se usan las cuatro regiones del proyecto
    (`REGIONES` de `matrix.py`: frontal = AF3, AF4, F3, F4, F7, F8;
    frontocentral; temporal; posterior). La posterior coincide en ambas.
  - Magnitud regional = `log10` de la potencia media de los canales
    **válidos** de la región (como el sondeo, pero sin los excluidos). Una
    región sin canales válidos da NaN y esa grabación no entra en esa
    magnitud.

### Qué se calcula

1. **Datos y descriptivo.** Por condición: grabaciones, personas, ventanas,
   ventanas por grabación y duración útil (mediana, p10–p90); grabaciones
   excluidas por tener < 10 ventanas; ventanas por tercio (inicial y final) en
   PASAT, PVTB, PVSAT y OA; canales excluidos por grabación; grabaciones en
   las que una región se queda sin canales válidos.
2. **Pendientes intra-grabación (reproduce 3.2).** Para las 7 condiciones y las
   8 magnitudes de la tabla 3.2: pendiente por minuto en cada grabación
   (`polyfit` de grado 1 sobre `t_rec`), media por persona y t de una muestra
   frente a 0 sobre las personas. Dos versiones: **canales válidos**
   (principal) y **14 canales** (como el sondeo).
3. **Clasificador primer tercio frente a último tercio (reproduce 3.3).**
   PASAT, PVTB, PVSAT y OA (control). Tercio inicial `t ≤ dur/3`, final
   `t ≥ 2·dur/3`; el medio se descarta. Features: las 56 `pot_rel`,
   promediadas por segmento y en `log10`. `StandardScaler` +
   `LogisticRegression(C=0.1, max_iter=3000)`, `cross_val_predict` con
   `GroupKFold(5)` por `persona_uid`. Métricas: AUC, BA y **acierto pareado**
   (proporción de grabaciones en las que el tercio final recibe más
   probabilidad que el inicial; un empate cuenta 0.5 y se reporta cuántos
   hay, porque una grabación sin canales válidos tiene los dos segmentos
   idénticos tras anular). Cuatro variantes:
   - **centrado + canales válidos** (principal): a cada segmento se le resta
     la media de los dos segmentos de su grabación y después se pone a 0 el
     cambio de los canales excluidos de esa grabación (`PLAN.md`, 4.4);
   - centrado + 14 canales (como el sondeo);
   - sin centrar + canales válidos: los canales excluidos se dejan como NaN y
     se imputan con la media del pliegue de entrenamiento (`SimpleImputer`
     dentro del pipeline); poner 0 no tiene sentido sin centrar;
   - sin centrar + 14 canales (como el sondeo).

   Para la variante principal, IC95 del acierto pareado y de la diferencia
   **tarea − OA** por bootstrap de personas (las predicciones de la CV quedan
   fijas; se remuestrean personas). Es descriptivo: no es un contraste.
4. **Curvas por minuto.** Alfa y theta absolutas y beta relativa, en las cuatro
   regiones, para PASAT, PVTB y PVSAT con OA como referencia. Por ventana se
   calcula la magnitud regional (canales válidos), se **centra por grabación**
   (se resta la media de la grabación, como en el clasificador, para ver el
   cambio y no el nivel de la persona) y se agrupa en minutos
   (`tmin_s` en [0, 60), [60, 120), [120, 180)…). Media por grabación y
   minuto → media por persona → media de personas, con IC95 por bootstrap de
   personas. Se descartan los minutos con menos de 5 ventanas en la
   grabación.

Bootstrap: 2000 réplicas remuestreando personas, semilla 42.

### Criterio de continuidad (fijado en `PLAN.md`)

El paso 2 se hace si, en la **variante principal** (centrado + canales
válidos), el acierto pareado de **PASAT y de PVTB es ≥ 0.70 en ambas** y
**cada una supera al de OA**. Se evalúa sobre las estimaciones puntuales de la
CV; los IC se reportan como información.

### Qué no se hace aquí

- No se mira `val` ni `test` (`val` se usa a partir del paso 3).
- No se eligen features ni modelo: el clasificador es el del sondeo, sin
  cambios, para que la comparación con 3.3 sea directa.
- No se interpreta el tiempo en tarea frente al tiempo desde la colocación
  (paso 4) ni los artefactos (paso 2).

### Recuento de miradas

- `val`: 0 miradas en este paso.
- `test`: 0.

---

## Resultados

Ejecutado el 2026-10-27 (`ejecucion.log`, `resultados.json`, `tablas/`,
`figuras/`). Determinista: dos ejecuciones dan un log idéntico.

### Veredicto

**El criterio de continuidad se cumple.** Acierto pareado centrado con canales
válidos: PASAT **0.773** y PVTB **0.758**, ambos ≥ 0.70 y por encima de OA
(**0.647**). El sondeo se reproduce en lo esencial, pero la ventaja sobre OA
**se reduce** al quitar los canales excluidos: de +0.165 a **+0.126** en PASAT
y de +0.131 a **+0.111** en PVTB. PVSAT no supera a OA.

### Datos

- `train` de `matriz_rf`, etiquetadas: **3117 grabaciones, 121 personas,
  192 315 ventanas** (120 personas por condición).
- **Incidencia de la partición (3.5), confirmada:** coincide exactamente con lo
  declarado en `PLAN.md`. Una persona pasa de `train` a `test` (6
  grabaciones), una de `train` a `val` (7) y una de `test` a `val` (1).
- Excluidas por tener < 10 ventanas: **3** (una de STROOP, una de PVTB y una de
  MSIT). En las tareas largas la regla solo quita una grabación de PVTB.
- Canales excluidos: entre 1.7 y 2.2 por grabación de media; el 81–91 % de las
  grabaciones tiene al menos uno. Ninguna grabación se queda sin canales
  válidos, así que **no hay empates** en el acierto pareado.
- Regiones vacías: la frontal del sondeo (4 canales) en 8 grabaciones; la
  frontal y la posterior del proyecto nunca; la frontocentral en 22 y la
  temporal en 53.

| Condición | Grabaciones | Ventanas/grab. | Duración útil (s), mediana [p10–p90] |
|---|---|---|---|
| OC | 453 | 55 | 108 [106–120] |
| OA | 445 | 55 | 108 [106–119] |
| STROOP | 446 | 28 | 54 [48–62] |
| PASAT | 445 | 88 | 174 [168–182] |
| PVTB | 443 | 87 | 172 [166–178] |
| MSIT | 443 | 28 | 54 [48–64] |
| PVSAT | 442 | 87 | 172 [166–178] |

Ventanas por tercio (mediana): PASAT 30, PVTB 29, PVSAT 29 y OA 19. El mínimo
es de 6–9 en alguna grabación corta de PVTB, PVSAT y OA.

### Clasificador de tercios (reproduce 3.3)

Acierto pareado (con el centrado coincide con la BA); entre paréntesis, el AUC.

| Tarea | Sondeo (14 canales, `split` antigua) | 14 canales, `split` correcta | **Canales válidos (principal)** | IC95 (bootstrap de personas) |
|---|---|---|---|---|
| PASAT | 0.785 (0.860) | 0.793 (0.854) | **0.773** (0.843) | [0.735, 0.810] |
| PVTB | 0.751 (0.831) | 0.753 (0.833) | **0.758** (0.829) | [0.717, 0.797] |
| PVSAT | 0.682 (0.720) | 0.663 (0.716) | **0.645** (0.703) | [0.597, 0.693] |
| OA | 0.620 (0.653) | 0.622 (0.670) | **0.647** (0.677) | [0.601, 0.691] |

Diferencia **tarea − OA** (principal, bootstrap de personas, descriptiva):

| | Sondeo | Principal | IC95 |
|---|---|---|---|
| PASAT − OA | +0.165 | **+0.126** | [+0.069, +0.183] |
| PVTB − OA | +0.131 | **+0.111** | [+0.047, +0.170] |
| PVSAT − OA | +0.062 | **−0.002** | [−0.071, +0.064] |

Sin centrar, el acierto pareado es 0.59–0.73 y el AUC 0.54–0.62 en las cuatro
condiciones, como en el sondeo: sin centrar, el modelo aprende sobre todo el
nivel de la persona.

Lectura:

- **La corrección de la partición apenas mueve las cifras** (columna 2 frente
  a la 1). Lo que cambia es **quitar los canales excluidos**: PASAT baja
  0.020 y **OA sube 0.025**. Parte del contraste del sondeo venía de los
  canales malos, que en las tareas aportaban cambio y en OA ruido.
- **PVSAT no se distingue de OA.** El 0.682 del sondeo era en parte artefacto
  de canales. Afecta a la H2 del borrador (que incluía PVSAT) y a cómo se
  define «las tareas» en el paso 3.

### Pendientes intra-grabación (reproduce 3.2)

Con 14 canales, la tabla 3.2 se reproduce a la segunda cifra decimal (p. ej.,
PASAT alfa post abs +0.027 (+8.7) frente a +0.027 (+8.9); PVTB beta post rel
−0.015 (−7.7) frente a −0.016 (−7.8)). Con canales válidos
(`tablas/pendientes.csv`):

| Condición | alfa post abs | theta front abs | beta post abs | beta front abs | delta front abs | alfa post rel | theta front rel | beta post rel |
|---|---|---|---|---|---|---|---|---|
| OC | −0.015 (−2.1) | −0.018 (−2.0) | −0.017 (−3.1) | −0.018 (−3.2) | −0.034 (−3.0) | +0.018 (+3.5) | +0.011 (+3.6) | +0.018 (+3.5) |
| OA | +0.027 (+4.9) | +0.021 (+2.8) | +0.015 (+3.5) | +0.029 (+6.4) | +0.020 (+2.0) | +0.007 (+1.2) | −0.003 (−1.0) | −0.003 (−0.7) |
| STROOP | +0.065 (+6.1) | +0.042 (+3.6) | +0.009 (+0.9) | +0.018 (+1.6) | +0.028 (+1.6) | +0.026 (+2.9) | +0.012 (+2.0) | −0.031 (−3.4) |
| **PASAT** | +0.023 (**+9.2**) | +0.028 (+9.2) | +0.010 (+3.8) | +0.007 (+2.2) | +0.026 (+6.7) | +0.001 (+0.6) | +0.004 (+2.4) | −0.012 (**−4.7**) |
| **PVTB** | +0.024 (**+10.0**) | +0.019 (+6.3) | +0.005 (+2.5) | +0.017 (+5.6) | +0.032 (+6.6) | +0.004 (+1.8) | −0.008 (−5.0) | −0.014 (**−7.4**) |
| MSIT | +0.031 (+2.7) | +0.008 (+0.6) | −0.015 (−2.1) | −0.007 (−0.9) | −0.068 (−3.8) | +0.060 (+6.3) | +0.044 (+7.0) | +0.013 (+1.3) |
| PVSAT | +0.016 (+5.3) | +0.008 (+2.4) | +0.008 (+2.5) | +0.008 (+3.1) | +0.000 (+0.0) | +0.014 (+6.8) | +0.004 (+2.4) | +0.006 (+2.2) |

Pendiente por minuto de `log10` de la potencia (t de una muestra sobre 120
personas). Los signos de la tabla 3.2 se mantienen en todas las celdas con
|t| > 2. Con canales válidos, los t de PASAT y PVTB en potencia absoluta
**suben** (alfa post +9.2 y +10.0; theta front de +5.8 a +9.2 en PASAT).

**Matiz importante para el paso 4 y la H3:** **por minuto**, la subida de la
alfa posterior absoluta en OA (+0.027) es **igual** que en PASAT y PVTB
(+0.023 y +0.024). Su t es menor solo porque OA dura un tercio menos. La subida
de la potencia absoluta lenta **no es específica de la tarea**. Lo que separa a
PASAT y PVTB de OA es la **bajada de la beta relativa** (−0.012 y −0.014 frente
a −0.003) y, en menor medida, la theta frontal.

### Curvas por minuto

`figuras/curvas_por_minuto.png` y `tablas/curvas_por_minuto.csv`: cambio
centrado por grabación, canales válidos y las cuatro regiones del proyecto.

- **Alfa y theta absolutas** suben de forma casi lineal del minuto 1 al 3 en
  las cuatro regiones, en PASAT y PVTB (≈ +0.03 a +0.07 en `log10` entre el
  minuto 1 y el 3, un 8–16 % de potencia). PVSAT sube menos, y en theta posterior y temporal casi
  nada. **OA sube con la misma inclinación** durante sus dos minutos.
- **Beta relativa:** baja en PASAT y PVTB en las cuatro regiones (más en
  frontocentral y temporal). En OA y PVSAT, plana.
- Concuerda con las pendientes: el rasgo diferencial de las tareas frente al
  reposo es el desplazamiento relativo hacia frecuencias lentas, no la subida
  absoluta.

`figuras/acierto_pareado.png`: acierto pareado con IC95 de la variante
principal.

### Desviaciones del plan

1. **Figura de curvas: minutos de cola sin dibujar.** Decidido **después** de
   ver la primera salida. El minuto 4 de las tareas y el 3 de OA solo
   contienen las grabaciones más largas que la mediana (1–16 personas), con
   medias e IC sin sentido. En la figura no se dibuja un minuto con menos de la
   mitad de las personas de su condición. **La tabla los conserva** con su
   número de personas. Es solo de presentación: no cambia ninguna cifra ni el
   criterio. La reejecución dio un log idéntico.

### Para los pasos siguientes (no decidido aquí)

- **OA es más corta que las tareas** (108 s frente a 172–174 s). Sus tercios
  están separados ≈ 36 s y los de las tareas ≈ 58 s. Si hay una deriva común
  por minuto (y la hay en la potencia absoluta), el contraste tarea − OA mezcla
  «tarea» con «más tiempo». Un control a igual tiempo (p. ej., los primeros
  108 s de cada tarea frente a OA) aclararía cuánto es de la tarea. Es una
  propuesta para el paso 3 o el 4 y hay que decidirla antes de mirar.
- **PVSAT ≈ OA:** conviene revisar la H2 y la definición de la familia de
  tareas antes del paso 3.

### Recuento de miradas

- `val`: 0. `test`: 0.
- `train`: pendientes en 7 condiciones × 8 magnitudes × 2 versiones de canales;
  clasificador en 4 condiciones × 4 variantes; curvas en 4 condiciones × 3
  magnitudes × 4 regiones. Es reproducción y descripción de lo ya sondeado,
  sin elegir nada.
