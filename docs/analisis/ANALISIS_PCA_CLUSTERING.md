# PCA + clustering: ¿hay estructura de fatiga sin supervisión?

> **Qué es este documento.** El análisis exploratorio **no supervisado** de los
> grupos extremos de fatiga percibida en el espacio de features de la Fase 13.
> Pregunta si las grabaciones de sesiones con mucha fatiga (VAS ≥ 80) y poca
> (VAS ≤ 20) forman nubes separadas por sí solas, una vez neutralizado el efecto
> de la condición.
>
> **Guion reproducible:** `analisis/analisis_pca_clustering.py`
> **Salida numérica:** `<EEG_DATA>/processed/pca_clustering.json`
> **Registro de ejecución:** `resultados/logs/pca_clustering.log`
> **Figuras:** `resultados/figuras/pca_clustering_vas_extremos.png`,
> `resultados/figuras/pca_clustering_vas_extremos_reposo.png`
> **Resumen en el cuaderno de modelado:** `docs/MODELADO.md`, Fase 16
>
> Fecha de ejecución: 2026-08-27.

---

## 1. La pregunta, y por qué es distinta de la Fase 14

La Fase 14 preguntó si un modelo **supervisado** puede predecir el VAS, y la
respuesta fue que apenas. Esta pregunta es **anterior y más débil**:

> ¿Existe siquiera estructura de fatiga en el espacio de features, la vea o no un
> clasificador?

Si las grabaciones de mucha y poca fatiga formaran dos nubes distintas, un
algoritmo que **no ve la etiqueta** debería encontrarlas solo. La ventaja de
preguntarlo así es que un método no supervisado **no puede sobreajustar a la
etiqueta, porque no la tiene**: si no encuentra nada, no cabe el argumento de
que faltó regularización, que el modelo era el equivocado o que los
hiperparámetros no estaban afinados.

El clustering no ve el VAS en ningún momento. El grupo se usa **solo después**,
para evaluar la coincidencia.

---

## 2. Diseño

| | |
|---|---|
| Unidad de análisis | La **grabación** — 1 fila por grabación, sin promediar entre condiciones |
| Muestra | 1076 grabaciones: **257** con VAS ≥ 80, **819** con VAS ≤ 20 |
| Descartadas | 2735 grabaciones de VAS intermedio (20 < VAS < 80) |
| Cobertura | 157 sesiones, 82 personas |
| Apartado | 732 grabaciones, 28 personas (`test`) — **no se leyó** |
| Features | Las 77 de `columnas_de_features()`, menos `condicion_orden` → **76** |
| Ajuste | Cada feature menos la media de **su propia condición** |
| Reducción | PCA a **15 componentes** |
| Clustering | K-means, **k = 2**, `n_init=50` |
| Semilla | 42 |

### Reparto por condición

El diseño está equilibrado: las siete condiciones aportan prácticamente el mismo
número de grabaciones dentro de cada grupo, así que ninguna tarea está
sobrerrepresentada en un extremo del VAS.

| Condición | VAS alto | VAS bajo |
|---|---|---|
| MSIT | 37 | 115 |
| OA | 37 | 117 |
| OC | 37 | 119 |
| PASAT | 35 | 118 |
| PVSAT | 37 | 115 |
| PVTB | 37 | 117 |
| STROOP | 37 | 118 |

### El ajuste por condición

La condición (OC, OA, STROOP, PASAT, PVTB, MSIT, PVSAT) se distingue muy bien en
este pipeline y podría dominar el clustering, respondiendo "qué tarea estaba
haciendo" en vez de "cómo de cansado estaba". Por eso, **antes del PCA**, a cada
feature se le resta la media de su propia condición:

```
x_ajustada = x − media(x | condición de esa grabación)
```

Es el mismo recurso que la regresión intrapersona de `docs/analisis/ANALISIS_REGRESION_LINEAL.md`, pero
centrando por condición en vez de por `persona_uid`. Lo que queda es **la
desviación de cada grabación respecto a lo esperable en su propia tarea**.

**`condicion_orden` se cae sola, y eso es la comprobación de que el ajuste
funciona.** Esa columna es la condición codificada como número; centrarla por
condición la deja constante a 0. El guion la descarta por varianza nula y lo
reporta. Si no desapareciera, el centrado no estaría haciendo lo que dice.

**Los NaN se rellenan con la media de su condición.** Operativamente: se centra
ignorando los NaN y luego se rellenan con 0, que tras el centrado *es* la media
de la condición. Así una región sin canales válidos (Fase 11.5) no inventa señal
ni arrastra el NaN al PCA. Afecta a 32 de las 77 features, con un máximo de 23
grabaciones por feature.

---

## 3. Lo que hay que declarar antes de leer ningún resultado

### El grupo de VAS y la identidad del participante son casi la misma partición

| | |
|---|---|
| Personas en los grupos extremos | **82** |
| Presentes en **ambos** grupos | **4 (4.9 %)** — `C1_F100`, `C2_F103`, `C6_F035`, `C6_F038` |
| Presentes en **un solo** grupo | **78 (95.1 %)** |
| Grabaciones aportadas por las de ambos | 76 (**7.1 %**) |

Detalle de las cuatro:

| Persona | Grabaciones alto | Grabaciones bajo | Sesiones alto | Sesiones bajo |
|---|---|---|---|---|
| `C1_F100` | 7 | 7 | 1 | 1 |
| `C2_F103` | 14 | 12 | 2 | 2 |
| `C6_F035` | 8 | 14 | 2 | 2 |
| `C6_F038` | 7 | 7 | 1 | 1 |

Solo 4 personas de 82 aportan contraste **dentro** del individuo. Para las otras
78, "pertenece al grupo alto" y "es esta persona" son literalmente la misma
etiqueta.

**Esto no es un detalle metodológico menor.** En este pipeline las features
identifican al participante **33 veces por encima del azar** (Fase 14, control de
identidad). Por tanto:

> Un clustering que hubiera separado limpiamente los dos grupos **no habría sido
> evidencia de fatiga.** Habría sido, casi con seguridad, identidad.

El resultado negativo esquiva el problema —no hay nada que reinterpretar—, pero
la limitación queda registrada porque **acota lo que este diseño podía haber
demostrado en el mejor de los casos**. La asimetría es importante: aquí un
resultado negativo es interpretable y uno positivo no lo habría sido.

### Las grabaciones no son independientes

Las siete condiciones de una misma sesión comparten **literalmente el mismo
número de VAS**. Las 1076 filas son 157 sesiones y 82 personas. El *n* efectivo
para cualquier afirmación sobre la fatiga son **157 sesiones / 82 personas**, no
1076.

Trabajar por grabación es correcto para *buscar estructura* —cada grabación es
una observación real de la señal, y promediar entre condiciones destruiría
justo la variación que se quiere examinar—, pero no convierte 1076 en el tamaño
muestral de la etiqueta.

### Contra qué se comparan los resultados

Los tres puntos de referencia son **mecánicos**, no elegidos a posteriori:

| Referencia | Valor | Qué significa |
|---|---|---|
| ARI de dos particiones sin relación | **0** | Es el valor esperado por construcción del índice ajustado |
| Clase mayoritaria | **76.1 %** | El acierto de decir "bajo" a todas las grabaciones |
| Nulo de pureza por persona | por permutación | Se estima remuestreando 1000 permutaciones de las etiquetas de cluster |

---

## 4. Resultado — análisis principal (7 condiciones, ajustado)

| | |
|---|---|
| Muestra | 1076 grabaciones, 157 sesiones, 82 personas |
| Features tras el ajuste | 76 (descartada: `condicion_orden`) |
| Varianza explicada | PC1 **25.6 %**, PC2 **12.4 %**, PC3 11.6 % |
| Acumulada | 5 comp **64.1 %** · **15 comp 85.5 %** |
| Tamaños de cluster | 723 / 353 |
| Silueta | **0.194** (estructura débil) |
| **ARI vs grupo VAS** | **−0.0264** |
| AMI vs grupo VAS | +0.0114 |
| **Acierto de la mejor asignación** | **54.3 %** |
| Clase mayoritaria (referencia) | **76.1 %** |

### Tabla de contingencia

| cluster | VAS alto | VAS bajo | total |
|---|---|---|---|
| **0** | 198 | 525 | 723 |
| **1** | 59 | 294 | 353 |
| **total** | 257 | 819 | 1076 |

La proporción de "alto" es 27.4 % en el cluster 0 y 16.7 % en el cluster 1. Hay
una diferencia, pero va en la dirección de un desequilibrio residual, no de una
separación: **ningún cluster es mayoritariamente de fatiga alta.**

### Lectura

**El ARI es negativo.** No es "bajo": es *peor que el azar*. Una partición
aleatoria daría 0 en promedio; esta da −0.0264.

**El acierto queda por debajo del listón trivial.** 54.3 % es el techo de este
clustering si se le permite elegir la correspondencia más favorable entre los 2
clusters y los 2 grupos. Decir "bajo" a todo acierta el **76.1 %**. El clustering
es **22 puntos peor** que no hacer nada.

Los clusters no están captando una versión débil del eje de fatiga: están
cortando el espacio por un eje que no guarda relación con él.

---

## 5. ¿Funcionó el ajuste por condición?

Sí — y hubo que cambiar la comprobación para poder afirmarlo.

### La comprobación obvia no discrimina

Comparar los 2 clusters contra la condición **no sirve**: una partición en 2 no
puede expresar una de 7, así que el ARI sale ≈ 0 esté el efecto presente o no.

| | ARI vs condición (k = 2) |
|---|---|
| Sin ajustar | +0.0046 |
| Ajustado | −0.0000 |

Indistinguibles, y ambos engañosos. Si uno se quedara aquí, concluiría que "la
condición no influía", cuando lo único que ha visto es una limitación del índice.

### La comprobación que sí discrimina

Agrupar en **k = 7** —tantos clusters como condiciones— y ver si el algoritmo las
recupera:

| | ARI vs condición (k = 7) |
|---|---|
| **Sin ajustar** | **+0.0733** |
| **Ajustado por condición** | **+0.0043** |

**El ajuste reduce la recuperabilidad de la condición 17 veces.** Hizo su
trabajo, y ahora hay evidencia de ello en vez de una suposición.

> Que +0.0733 tampoco sea un valor alto es otro asunto: k-means sobre 15
> componentes es un detector flojo de una estructura de 7 grupos. Lo que importa
> aquí es la **razón entre ambos**, no la magnitud absoluta.

### Y no cambió la conclusión

La comparación sin ajustar sirve además como control del análisis entero: el
resultado frente al VAS es prácticamente el mismo con y sin ajuste.

| | Ajustado | Sin ajustar |
|---|---|---|
| ARI vs grupo VAS | −0.0264 | −0.0223 |
| Acierto | 54.3 % | 55.9 % |
| Silueta | 0.194 | 0.183 |
| Varianza en 15 comp | 85.5 % | 86.1 % |

El resultado negativo **no se explica** porque el ajuste hubiera borrado la señal
de fatiga junto con la de condición: sin ajustar tampoco aparece.

---

## 6. ¿Con qué se alinean entonces los clusters?

Con la identidad del participante — débilmente, pero por encima del azar.

| | Principal | Sin ajustar | Reposo |
|---|---|---|---|
| **Pureza por persona** | **0.791** | 0.787 | **0.924** |
| Azar (permutación, media) | 0.694 | 0.699 | 0.881 |
| Azar (p95) | 0.706 | 0.711 | 0.900 |
| AMI vs `persona_uid` | +0.0667 | +0.0575 | +0.0426 |
| AMI vs grupo VAS | +0.0114 | +0.0057 | +0.0093 |

La **pureza por persona** es la fracción de grabaciones de cada individuo que
caen en su cluster mayoritario, promediada sobre los 82. La observada supera el
percentil 95 del nulo por permutación en los tres análisis.

Además, el AMI frente a `persona_uid` es entre **4.6 y 10 veces** el AMI frente
al grupo de VAS (5.8× en el principal, 10.1× sin ajustar, 4.6× en reposo). Es
coherente con todo lo demás del proyecto:

> Lo que estas features llevan es **quién eres**, no **cómo de cansado estás**.

### Por qué aquí no se usa el ARI

El ARI frente a `persona_uid` (+0.0057) parece contradecir lo anterior, pero no
dice nada: comparar una partición de 2 con una de 82 lo deprime mecánicamente,
igual que pasaba en la § 5 con las 7 condiciones. Por eso la pregunta de
identidad se responde con **pureza y AMI**, que sí toleran cardinalidades
distintas, y no con ARI.

---

## 7. Robustez — solo reposo (OA + OC)

Mismo procedimiento restringido a las grabaciones de reposo, donde no hay tarea
que module la señal y el estado basal debería verse más limpio.

| | |
|---|---|
| Muestra | 310 grabaciones (**74** alto / **236** bajo) |
| Varianza explicada | PC1 **27.4 %**, PC2 **14.2 %** |
| Acumulada | 5 comp 66.8 % · **15 comp 87.5 %** |
| Tamaños de cluster | **270 / 40** |
| Silueta | **0.360** |
| **ARI vs grupo VAS** | **−0.0508** |
| Acierto de la mejor asignación | **66.5 %** (mayoritaria: 76.1 %) |

| cluster | VAS alto | VAS bajo | total |
|---|---|---|---|
| **0** | 69 | 201 | 270 |
| **1** | 5 | 35 | 40 |

### Lectura

**Consistente con el principal, y algo peor.** El ARI sigue siendo negativo y de
hecho se aleja más del cero (−0.051 frente a −0.026). El acierto sigue por debajo
de la clase mayoritaria. La respuesta a "¿difiere sustancialmente del análisis
principal?" es **no**: difiere en la dirección de ser aún menos favorable.

**La silueta sube a 0.360, y eso no es una buena noticia.** El reparto 270/40
delata un cluster pequeño de grabaciones atípicas —perfil de calidad de señal—,
que es exactamente lo que k-means produce cuando no hay dos grupos que encontrar:
aísla los valores extremos y llama a eso estructura. La proporción de "alto" es
25.6 % en el cluster grande y 12.5 % en el pequeño; el cluster atípico es, si
acaso, *menos* de fatiga alta.

---

## 8. Las figuras

Ambas: PC1 vs PC2, un punto por grabación, **rojo = VAS ≥ 80**, **azul =
VAS ≤ 20**. El par de colores pasa las seis comprobaciones del validador de
paleta en modo claro (separación CVD ΔE 21.6, muy por encima del 8 exigido).

### `resultados/figuras/pca_clustering_vas_extremos.png`

Todas las condiciones, features centradas por condición, n = 1076. Una sola nube
con los dos colores **completamente entremezclados**. No hay dos lóbulos, ni un
gradiente de rojo hacia un lado, ni siquiera un desplazamiento visible de
centroides. Es la forma gráfica de un ARI negativo.

### `resultados/figuras/pca_clustering_vas_extremos_reposo.png`

Solo OA + OC, n = 310. Lo mismo, con menos puntos. Los valores extremos de PC1
que forman el cluster de 40 se ven a la derecha, y son de ambos colores.

> **Cómo se leería un resultado positivo**, para contraste: los rojos ocuparían
> preferentemente una región del plano y los azules otra, aunque se solaparan en
> la frontera; los centroides estarían separados; y la frontera de k-means
> caería aproximadamente entre ambos. Nada de eso ocurre.

---

## 9. Conclusión

**No hay estructura natural de fatiga percibida en el espacio de features, ni con
todas las condiciones ni en reposo.**

Los tres análisis coinciden:

| Análisis | n | var. 15 comp | ARI vs VAS | ARI condición | Acierto | Mayoritaria |
|---|---|---|---|---|---|---|
| Principal (7 cond., ajustado) | 1076 | 85.5 % | **−0.0264** | +0.0043 (k=7) | 54.3 % | 76.1 % |
| Referencia (sin ajustar) | 1076 | 86.1 % | −0.0223 | +0.0733 (k=7) | 55.9 % | 76.1 % |
| Robustez (solo reposo) | 310 | 87.5 % | **−0.0508** | −0.0014 (k=2) | 66.5 % | 76.1 % |

ARI negativo frente al grupo de VAS en los tres; acierto por debajo de la clase
mayoritaria en los tres; y clusters que se alinean —débilmente— con la identidad
del participante en vez de con su fatiga. El ajuste por condición funcionó (17×
menos recuperabilidad de la condición), así que el resultado **no** se explica
porque la tarea estuviera dominando el clustering.

Esto **refuerza** el diagnóstico de la Fase 14: el problema no es el
modelo supervisado, ni la familia de algoritmo, ni los hiperparámetros. Es que
las 77 features agregadas por región y por grabación **no contienen un eje de
fatiga percibida separable**. Un método que no puede sobreajustar a la etiqueta
porque no la ve tampoco lo encuentra.

Encaja además con el resultado de `docs/analisis/ANALISIS_REGRESION_LINEAL.md`: el biomarcador de
referencia de la literatura tampoco sigue al VAS (R² = 0.010; intrapersona
p = 0.93). Dos rutas independientes —una supervisada y univariante, otra no
supervisada y multivariante— llegan al mismo sitio.

### Alcance de la afirmación

- **Sí establece** que las features de la Fase 13, agregadas por región y
  grabación, no organizan las grabaciones extremas de VAS en grupos separables.
- **No establece** que la señal cruda no lleve la información. La ruta CNN sigue
  **sin estar condicionada** por esto, por la misma razón que en la Fase 14: mide
  lo que las features agregadas destruyen, no lo que el EEG contiene.
- **No establece** que no haya *ninguna* relación. Una tendencia monótona débil
  es perfectamente compatible con la ausencia total de conglomerados.
- **No es confirmatorio.** Se ejecutó sobre `train` + `val`. Aporta **3
  configuraciones** más que hay que sumar al recuento total que se declare al
  fijar el alfa del paquete de test. El test (732 grabaciones, 28 personas)
  sigue sin abrirse.

---

## 10. Lo que **no** hay que hacer

- **Presentar este resultado sin la confusión identidad/grupo al lado.** Con el
  95.1 % de las personas en un solo grupo, un resultado *positivo* aquí no habría
  sido interpretable. El negativo sí lo es, pero la asimetría hay que contarla.
- **Usar el ARI de k = 2 frente a la condición como prueba de que el ajuste
  funcionó.** No discrimina: sale ≈ 0 en ambos casos. La comprobación es a k = 7.
- **Usar el ARI para preguntar por la identidad.** Cardinalidades de 2 y 82: hay
  que usar pureza por persona con su nulo por permutación, y AMI.
- **Leer la silueta de 0.360 del reposo como estructura de fatiga.** Es un
  cluster de 40 grabaciones atípicas por calidad de señal.
- **Contar las grabaciones como *n*.** Son 1076 filas, pero **157 sesiones y 82
  personas**.
- **Probar más valores de k, o más métodos de clustering.** El ARI es *negativo*
  en los tres análisis: no es un problema de sintonía. Cada intento adicional
  engorda el recuento de configuraciones que hay que declarar y no cambiará el
  signo.

---

## 11. Reproducir

```bash
# desde la raíz del repositorio
python analisis/analisis_pca_clustering.py
```

Duración: ~40 s. Semilla fija (`SEMILLA = 42`) en el PCA, en k-means y en las
1000 permutaciones del nulo de pureza: dos ejecuciones sobre la misma matriz dan
cifras idénticas.

El guion aparta el `test` en su primer paso y no vuelve a leerlo; el filtro a
`["train", "val"]` es lo primero que ocurre tras cargar la matriz.

Versiones con las que se generaron estos resultados: Python 3.11,
scikit-learn 1.7.2, numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, matplotlib 3.10.9.

---

## 12. Documentos relacionados

| Documento | Qué cubre |
|---|---|
| `docs/MODELADO.md`, Fase 16 | El mismo análisis, resumido en el cuaderno de modelado |
| `docs/MODELADO.md`, Fases 14–15 | Los intentos **supervisados** que este análisis complementa |
| `docs/analisis/ANALISIS_REGRESION_LINEAL.md` | El índice theta/alfa contra el VAS — la vía univariante |
| `docs/ESTADO.md`, Fase 13 | Cómo se construyen las 77 features y qué significa cada una |
| `docs/ESTADO.md`, Fase 12 | Por qué la agrupación es `persona_uid` y nunca `sujeto_id` |
