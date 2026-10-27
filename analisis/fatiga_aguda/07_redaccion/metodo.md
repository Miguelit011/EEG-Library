# Método — el efecto del tiempo en tarea dentro de la grabación

> Borrador para la memoria (metodología o implementación). Fuentes:
> `PLAN.md`, `03_clasificador/configuracion_elegida.json`,
> `03_clasificador/medida_elegida.json` y `06_test/PREREGISTRO.md`. Los
> términos de este análisis son «fatiga aguda», «decremento de vigilancia» o
> «efecto del tiempo en tarea»; en ningún caso se dice que el EEG «mide la
> fatiga» del alumno.

## Pregunta

Las líneas anteriores buscaron en el EEG la fatiga percibida (VAS, PedsQL)
entre sesiones separadas por semanas, y no la encontraron. Este análisis
cambia de escala temporal: pregunta si el EEG cambia de forma sistemática
**dentro de una misma grabación de tarea**, del principio al final, en
personas que el modelo no ha visto.

Es la operacionalización habitual de la fatiga mental aguda en la literatura
EEG (*time-on-task*). Tiene dos ventajas:

- **La etiqueta es objetiva y no cuesta nada:** es el instante de cada
  ventana.
- **La comparación es dentro de la grabación:** el casco, la colocación de los
  electrodos y el día son los mismos al principio y al final.

## Datos

- Grabaciones de PASAT y PVTB (tareas de ≈ 3 min; duración útil mediana de
  174 y 172 s) y de reposo con ojos abiertos (OA; 108 s), de las cinco
  mediciones (M2–M6). PVSAT y reposo con ojos cerrados (OC) se describen sin
  contraste.
- Las mismas ventanas de 4 s con un 50 % de solapamiento y las mismas
  potencias por banda (delta, theta, alfa y beta, 14 canales) que el resto del
  TFG.
- La partición y la identidad son las corregidas en la Fase 12 (agrupación por
  `persona_uid`).
- Los canales excluidos por calidad (Fase 11.5) no aportan.
- Se excluyen las grabaciones con menos de 10 ventanas.
- **Partición:** `train` (121 personas) y `val` (25) para desarrollar; `test`
  (28 personas) se usa **una sola vez**, con un pre-registro fijado antes.

## Definiciones

- **Tercios:** inicial si `t ≤ dur/3` y final si `t ≥ 2·dur/3`. El tercio medio
  no se usa para entrenar.
- **Centrado por grabación:** a cada segmento o ventana se le resta la media de
  su grabación. El modelo solo ve el **cambio**, no el nivel de la persona ni
  el del casco.
- **Igual tiempo transcurrido:** cada grabación de tarea se recorta a la
  duración útil del OA de su misma sesión. Así la tarea y el reposo se
  comparan con los mismos minutos desde el inicio de la grabación. Sin
  recorte, las tareas duran un 60 % más que OA y tienen más tiempo para
  cambiar.
- **Modelo (fijado en desarrollo):** las 56 potencias relativas
  (`log10`, 14 canales × 4 bandas) con `StandardScaler` + regresión logística
  (C = 0.01). Hay **un modelo por condición**, entrenado a distinguir el
  primer tercio del último.
- **Medida `rho_ventanas`:** el modelo puntúa cada ventana. Por grabación se
  calcula la ρ de Spearman entre esa puntuación y el instante de la ventana:
  0 si el EEG no se ordena en el tiempo y positiva si avanza en la dirección
  aprendida.
- **Acierto pareado de tercios (descriptivo):** la proporción de grabaciones
  en que el último tercio recibe más probabilidad que el primero. El azar es
  0.5.

## El control: el reposo con ojos abiertos

Las tareas no se comparan con el azar, sino con **OA**. Toda grabación cambia
algo con el tiempo, aunque no haya tarea: el gel se asienta, la persona se
relaja y el amplificador deriva. Lo que interesa es si las tareas cambian
**más** que el reposo. El contraste principal es

> Δρ = ρ media (PASAT + PVTB) − ρ media (OA)

con IC por bootstrap de personas.

## Controles de confusores (desarrollo)

1. **Movimiento y contacto (paso 2):**
   - medidas: acelerómetro y giroscopio (canales `MOT.*`) y calidad de
     contacto (`CQ.*`) de los EDF, alineados con cada ventana;
   - comparaciones: un clasificador de solo artefactos frente al EEG, y el
     EEG residualizado por ellos.
2. **Tiempo en tarea frente a tiempo desde la colocación (paso 4):**
   - matriz de transferencia entre condiciones (el modelo de una condición
     puntúa otra);
   - «dientes de sierra»: el cambio dentro de cada tarea frente al salto al
     empezar la siguiente;
   - modelo de ventanas con efectos fijos de sesión y condición, que separa
     el tiempo dentro de la grabación (`t_rec`) del tiempo desde el inicio de
     la sesión (`t_ses`).

## Diseño de validación

- **Desarrollo en `train` + `val`:**
  - selección del modelo entre 32 configuraciones (4 conjuntos de features × 4
    modelos × 2 esquemas), con validación cruzada por persona **repetida 20
    veces**;
  - regla fijada de antemano: la mayor Δρ a igual tiempo.
- **Miradas a `val`:**
  - la primera, con el acierto de tercios, no fue buena;
  - se eligió en `train` una medida más eficiente (`rho_ventanas`);
  - la segunda mirada fue buena (Δρ = +0.059 [+0.003, +0.116]).
- **Esa decisión se tomó después de ver `val`, y se declara.**
- **Potencia para `test`, simulada con 28 personas de desarrollo:**
  - H1 a igual tiempo: 0.75 (0.58 con un efecto un 25 % menor);
  - H1b con la grabación completa: 0.92 (0.83).
- **Pre-registro** (`06_test/PREREGISTRO.md`), fijado en un commit antes de
  abrir `test`, con una **secuencia fija** que controla el error de familia:
  - **H1b**, con la grabación completa: Δρ > 0;
  - **solo si H1b confirma, H1** a igual tiempo: Δρ > 0.
- **Secundarias:**
  - **H2:** PASAT y PVTB ordenan sus ventanas en el tiempo (Holm sobre 2);
  - **H3:** la beta posterior relativa baja más que en OA (Holm sobre 2;
    potencia de solo 0.26–0.38);
  - **H4:** el EEG supera a la referencia de solo movimiento y contacto.
- **Modelo final:** entrenado con `train` + `val` (146 personas) y evaluado
  una vez en `test` (28 personas), con bootstrap de personas (10 000 réplicas).
- **Sensibilidades:**
  - sin la persona de `test` que se vio en los sondeos iniciales (una
    contaminación declarada de la partición antigua);
  - sin las grabaciones a 128 Hz.

## Análisis exploratorio posterior (paso 5)

Se hizo después del test y solo con `train` + `val`. Compara el cambio dentro
de la grabación con el rendimiento total en esa tarea en la misma sesión:

- precisión del PASAT;
- lapsos y tiempo de reacción del PVTB.

Se usan efectos fijos de persona y medición, con errores agrupados por persona
y Holm sobre los tres contrastes. El control es el cambio en OA de la misma
sesión. El movimiento se añadió como covariable en una desviación declarada.

Los EDF no tienen marcadores de ensayo, así que el rendimiento solo existe
como total por sesión. **No se puede relacionar el EEG con el rendimiento
minuto a minuto.**
