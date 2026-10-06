# Regresión lineal simple: VAS ~ índice theta/alfa

> **Qué es este documento.** El contraste directo entre el biomarcador de fatiga
> de referencia de la literatura (índice theta/alfa) y la etiqueta real del
> estudio (VAS de fatiga percibida), con su gráfica. Es el análisis descriptivo
> que la memoria del TFG tiene que reportar sobre la Fase 9.
>
> **Guion reproducible:** `analisis/analisis_regresion_lineal.py`
> **Salida numérica:** `<EEG_DATA>/processed/regresion_lineal.json`
> **Registro de ejecución:** `analisis/salidas/regresion_lineal.log`
> **Figuras:** `figures/regresion_vas_theta_alfa.png`,
> `figures/regresion_vas_theta_alfa_intrapersona.png`
>
> Fecha de ejecución: 2026-08-27.

---

## 1. Diseño

| | |
|---|---|
| Variable dependiente | `vas` — VAS de fatiga percibida, 0–100 mm |
| Variable independiente | `indice_fatiga_media` — theta frontal / alfa posterior, promediado sobre las ventanas y las grabaciones de la sesión |
| Unidad de análisis | **La sesión** (sujeto × momento) |
| Muestra | 547 sesiones, 146 personas (`train` + `val`) |
| Apartado | 106 sesiones, 28 personas (`test`) — **no se leyó** |
| Estimación | Mínimos cuadrados ordinarios, IC al 95 % por *t* de Student |
| IC complementario | Bootstrap de 10 000 réplicas remuestreando **personas** |
| Semilla | 42 |

La unidad es la sesión y no la grabación porque el VAS se responde **una vez por
sesión**, inmediatamente antes del EEG. Las 4543 grabaciones de
`matriz_rf.parquet` aportan 653 etiquetas independientes (547 aquí), no 4543:
las siete condiciones de una misma sesión comparten literalmente el mismo número
de VAS. Analizar por grabación multiplicaría por ~7 el *n* aparente sin añadir ni
una sola observación de la variable dependiente, y estrecharía los intervalos de
confianza por un factor de ~2.6 sin justificación.

---

## 2. El resultado óptimo, declarado antes de ejecutar

Escrito y fijado en el código (`R2_LINEA_BASE_DECLARADO`, `PENDIENTE_OPTIMA`)
**antes** de mirar el resultado, para que la interpretación no se ajuste a
posteriori a lo que salga. Si el VAS y el índice theta/alfa midieran la misma
fatiga, la gráfica tendría que cumplir estas siete condiciones **a la vez**:

| # | Criterio | Qué exige | Por qué |
|---|---|---|---|
| 1 | **Pendiente positiva** | β > 0 | El índice sube con la fatiga: la teoría dice que la fatiga mental eleva la theta frontal y altera la alfa posterior. Fatiga ↑ → índice ↑ |
| 2 | **Nube estrecha** | SD residual claramente < 21.7 mm | Lo que importa no es que la recta se incline, sino que los puntos estén cerca de ella |
| 3 | **R² > 0.260** | Superar la línea base sin EEG | Ese 0.260 es lo que ya consigue predecir la media de VAS **de esa misma persona** (leave-one-out). Un índice espectral que no lo bata no aporta información nueva |
| 4 | **Magnitud clínica** | β ≈ +10 mm por unidad | El RIC del índice va de ~1.3 a ~2.6; para mover el VAS unos 15 de sus 100 puntos hace falta esa pendiente |
| 5 | **IC95 lejos del 0** | Estrecho y sin cruzar el cero | Y que sus dos extremos sigan siendo magnitudes que importan |
| 6 | **Sin artefactos de forma** | El resultado no lo dictan la asimetría ni las colas | El índice está muy sesgado a la derecha (asimetría 2.21) |
| 7 | **Se sostiene *dentro* de la persona** | β intrapersona sigue siendo ≠ 0 | **El criterio decisivo.** Con ~4 sesiones por participante, una correlación puede salir solo porque quien tiene más alfa de base marca VAS más altos: un rasgo estable, no una lectura de fatiga |

El criterio 7 es el que separa un biomarcador de un artefacto. Un detector de
fatiga tiene que saber que **hoy** estás más cansado que **ayer**, no solo que
eres una persona que suele estarlo.

**El fallo característico** —la hipótesis que había que poder descartar— es una
nube plana y redonda: recta horizontal, R² ≈ 0, IC cruzando el cero.

---

## 3. Resultado

### 3.1 Regresión principal

```
VAS = 46.72 − 1.71 × índice_theta/alfa
```

| Estadístico | Valor |
|---|---|
| n | 547 sesiones (146 personas) |
| Pendiente β | **−1.710 mm** por unidad de índice (EE 0.720) |
| IC 95 % de β (MCO) | **[−3.124, −0.296]** |
| IC 95 % de β (bootstrap por persona) | **[−3.278, −0.013]** |
| Intercepto | 46.72 mm |
| **R²** | **0.0102** |
| r de Pearson | −0.101 |
| t(545) | −2.375 |
| **p** | **0.018** |
| Spearman ρ | −0.091 (p = 0.033) |
| SD del VAS | 21.67 mm |
| SD residual | 21.56 mm (**reducción del 0.51 %**) |

### 3.2 Comprobaciones de forma (criterio 6)

| Versión | Pendiente | R² | p |
|---|---|---|---|
| Cruda | −1.710 | 0.0102 | 0.018 |
| log₁₀ del índice | −8.026 | 0.0079 | 0.038 |
| Recortando el 1 % de cada cola | −2.082 | 0.0108 | 0.016 |

Las tres coinciden en signo y en orden de magnitud. **El resultado no es un
artefacto de la asimetría ni de los valores extremos.**

### 3.3 La comprobación decisiva (criterio 7)

Restando a cada sesión la media de su propia persona en ambas variables —lo que
queda es la variación día a día del mismo individuo:

| Estadístico | Valor |
|---|---|
| Pendiente β intrapersona | **−0.073 mm** (EE 0.782) |
| IC 95 % | **[−1.610, +1.463]** |
| R² | **0.00002** |
| t(399) | −0.094 |
| **p** | **0.925** |
| Spearman ρ | +0.004 (p = 0.929) |

**La asociación desaparece por completo.** Coincide con la cifra que ya figuraba
en `CLAUDE.md` (ρ = +0.004, p = 0.93).

---

## 4. Comparación con el óptimo declarado

| # | Criterio | Óptimo | Obtenido | |
|---|---|---|---|---|
| 1 | Pendiente positiva | β > 0 | **β = −1.71** | ❌ |
| 2 | Nube estrecha | SD residual ≪ 21.67 | 21.56 mm (−0.51 %) | ❌ |
| 3 | R² > 0.260 | > 0.260 | **0.0102** | ❌ |
| 4 | Magnitud clínica | β ≈ +10 | −1.71 | ❌ |
| 5 | IC95 excluye el 0 | Sí, con margen | [−3.12, −0.30] | ✅ |
| 6 | Sin artefactos de forma | Estable | Estable | ✅ |
| 7 | Se sostiene intrapersona | p < 0.05 | **p = 0.925** | ❌ |

**2 de 7.** Y los dos que se cumplen no sostienen la hipótesis:

- El **criterio 5** se cumple *en la dirección contraria*. El IC excluye el cero,
  pero por el lado negativo, que contradice al criterio 1. Además, con el IC
  honesto —el que remuestrea personas en vez de sesiones— el extremo superior es
  **−0.013**: prácticamente tocando el cero. La significación nominal (p = 0.018)
  descansa en tratar 547 sesiones como independientes cuando en realidad son 146
  personas.
- El **criterio 6** solo dice que el resultado es *estable*, no que sea el
  esperado. Es un no-resultado robusto, no un resultado.

### Qué significa el R² = 0.0102

El índice theta/alfa explica el **1 % de la varianza** del VAS. La línea base sin
EEG —predecir para cada sesión la media de VAS de esa misma persona— explica el
**29.6 %** en esta muestra. Dicho de otro modo: **saber quién es el participante
informa 29 veces más sobre su fatiga percibida que medirle el biomarcador.**

En términos prácticos, recorrer el rango intercuartílico completo del índice
(de 1.33 a 2.64) mueve la predicción del VAS **2.2 mm** sobre una escala de 100,
con una dispersión residual de 21.6 mm alrededor. La reducción del error es del
0.51 %.

### Por qué la pendiente sale negativa

No hay que sobreinterpretarla. La lectura correcta es que **es un efecto entre
personas, no de fatiga**: desaparece por completo (β = −0.07, p = 0.93) al
comparar a cada participante consigo mismo. Lo que la regresión global está
captando es una diferencia estable entre individuos —quien tiene más potencia
alfa posterior de base tiende a marcar VAS algo más altos— y eso es anatomía,
colocación del casco o estilo de respuesta, no fatiga. Es exactamente el
confusor que el criterio 7 estaba diseñado para detectar, y lo detectó.

---

## 5. Las gráficas

### `figures/regresion_vas_theta_alfa.png`

La dispersión pedida, con la recta de MCO, su banda de confianza al 95 % y —en
gris discontinuo— la pendiente +10 que el índice tendría que tener para ser
útil. El contraste entre ambas rectas **es** el resultado.

Dos rasgos de la gráfica que son reales y no defectos de dibujo:

- **Las bandas horizontales.** El 77.0 % de las respuestas al VAS de esta
  muestra son múltiplos de 10 (76.7 % sobre las 653 sesiones del proyecto): los
  participantes marcan la escala en valores redondos. El VAS es ordinal de
  facto, y por eso se reporta también Spearman.
- **El apelotonamiento a la izquierda.** El índice tiene asimetría +2.21. La
  versión en log₁₀ (§ 3.2) confirma que no cambia nada.

### `figures/regresion_vas_theta_alfa_intrapersona.png`

La misma regresión tras centrar ambas variables por persona. Es la nube plana y
redonda del "fallo característico" descrito en la § 2, en su forma de manual:
recta indistinguible de la horizontal, R² = 0.00002, IC simétrico alrededor del
cero.

---

## 6. Conclusión

**El índice theta/alfa no mide la fatiga que el VAS mide, en esta cohorte.**

No es un fallo de potencia estadística: con 547 sesiones y 146 personas, el IC
de la pendiente intrapersona ([−1.61, +1.46]) es lo bastante estrecho como para
**descartar** cualquier efecto de magnitud útil, no solo para no encontrarlo.
Si existiera la pendiente de +10 del óptimo, esta muestra la habría visto sin
dificultad.

Este es un **resultado negativo informativo**, y como tal entra en la memoria.
El índice theta/alfa se sigue calculando y reportando (Fase 9) porque es el
biomarcador de referencia de la literatura y el TFG debe pronunciarse sobre él
— y aquí se pronuncia: medido contra una etiqueta externa real, con la unidad de
análisis correcta y controlando la identidad del participante, no sigue a la
fatiga percibida.

Es también la justificación empírica de dos decisiones ya tomadas en el
proyecto:

1. **La retirada de la Fase 10** (auto-etiquetado de fatiga por umbrales sobre
   el índice theta/alfa). Umbralizar este índice para generar etiquetas habría
   producido una variable con ρ = +0.004 respecto a la fatiga real: ruido con
   nombre científico. Ver `ESTADO.md`, "Fase 10 — retirada".
2. **El uso del VAS como etiqueta** (Fase 12) en lugar de una etiqueta derivada
   de la propia señal.

### Alcance de la afirmación

Lo que este análisis establece y lo que no:

- **Sí establece** que el índice theta/alfa, agregado por sesión y medido con un
  Emotiv Epoc X, no predice el VAS de fatiga percibida respondido al inicio de
  la sesión.
- **No establece** que el índice theta/alfa no mida nada. El VAS es fatiga
  *autopercibida antes de la tarea*; no es rendimiento, ni somnolencia, ni
  fatiga inducida por la tarea. La discrepancia entre marcadores fisiológicos y
  autoinforme está documentada en la literatura.
- **No es confirmatorio.** Se ejecutó sobre `train` + `val`. El conjunto de test
  (106 sesiones, 28 personas) sigue sin abrirse y se reserva para un único
  análisis pre-registrado.

---

## 7. Reproducir

```bash
# desde la raíz del repositorio
python analisis/analisis_regresion_lineal.py
```

Duración: ~20 s (el grueso son las 10 000 réplicas de bootstrap). Semilla fija:
dos ejecuciones sobre la misma matriz dan cifras idénticas.

Versiones con las que se generaron estos resultados: Python 3.11, numpy 2.2.6,
pandas 2.3.3, scipy 1.15.3, matplotlib 3.10.9.
