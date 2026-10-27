# Paso 7 — Redacción

> Plan escrito el 2026-10-27, antes de redactar. **Decisión del autor:** no se
> escribe nada en `EEG-Memoria`. Todo (textos y figuras) queda en esta carpeta,
> en Markdown, para que el autor lo pase a la memoria. La vista de la
> plataforma se hará en un paso o fase posterior (no forma parte de este paso).

## Qué se produce

| Archivo | Para qué parte de la memoria | Contenido |
|---|---|---|
| `metodo.md` | Metodología / implementación | Pregunta, datos, definiciones (tercios, igual tiempo, centrado, canales válidos), modelo, `rho_ventanas`, diseño de validación (`train`/`val`/`test`, CV repetida, pre-registro, secuencia H1b → H1), controles |
| `resultados.md` | Resultados (cap. 6) | Desarrollo (pasos 1–4), evaluación única en `test` (paso 6) y análisis exploratorio (paso 5), cada bloque etiquetado por su estatus |
| `discusion.md` | Discusión y conclusiones (cap. 7) | Mapa de escalas temporales, qué se puede afirmar y qué no, comparación con la literatura, limitaciones, trabajo futuro |
| `referencias.md` | Bibliografía | Las referencias del PLAN 1.2 **comprobadas** (autores, año, título, revista, DOI). Las que no se puedan comprobar se marcan como tales y no se citan |
| `figuras.py` → `figuras/` | Figuras | Se generan **solo a partir de agregados** ya calculados (tablas y JSON de los pasos 1, 2, 4 y 6), en PDF vectorial (para LaTeX) y PNG. No se recalcula ningún análisis ni se carga `test` |

## Figuras previstas

1. **Curvas por minuto:** alfa absoluta y beta relativa en la región
   posterior, para PASAT, PVTB y OA (`train`; paso 1).
2. **Acierto pareado de tercios:** tarea frente a OA, con la grabación
   completa y a igual tiempo (`train`; paso 1 y anexo A).
3. **Referencia de artefactos:** el EEG frente a solo movimiento y contacto, a
   igual tiempo (`train`; paso 2).
4. **Especificidad:** la matriz de transferencia (`train`; paso 4, B1).
5. **Resultado en `test`:** la ρ media de `rho_ventanas` por condición, en las
   dos versiones, con su IC95, y Δρ de la H1b y la H1 (paso 6).

Cada figura lleva en su pie de qué partición sale y si es de desarrollo,
confirmatoria o exploratoria.

## Reglas de redacción

- Nombres: «fatiga aguda», «decremento de vigilancia» o «efecto del tiempo en
  tarea»; nunca «la fatiga del alumno» (PLAN 1.3). No se afirma que el EEG
  «mide la fatiga».
- **Tres estatus, siempre explícitos:**
  - **desarrollo:** `train` y `val`, con las decisiones post hoc declaradas;
  - **confirmatorio:** los veredictos del test, tal cual;
  - **exploratorio:** el paso 5 y todo lo posterior al test.
- Las cifras se copian de los `INFORME.md` y JSON de cada paso, sin redondeos
  nuevos que cambien el sentido. Las cifras de `val` se citan como de
  desarrollo.
- El resultado negativo de la H1 se cuenta igual de visible que los positivos.
  La H1b se cita con su fragilidad (sensibilidad sin la persona expuesta).
- Sin identificadores de persona.

## Comprobación final

- Cada cifra de `resultados.md` se contrasta con su fuente (la ruta se cita al
  lado de cada tabla).
- Las figuras se abren y se revisan (solapes, ejes, pies).

## Hecho (2026-10-28)

- **Textos:** `metodo.md`, `resultados.md`, `discusion.md` y
  `referencias.md`. Las cifras de `resultados.md` se copiaron de los INFORME y
  JSON de cada paso y se contrastaron con su fuente.
- **Figuras:** `figuras/f1`–`f5`, en PDF y PNG, generadas por `figuras.py`
  solo a partir de agregados. Se revisaron visualmente. Dos correcciones tras
  la primera revisión:
  - en F1 se quitan los minutos de cola con menos del 80 % de las personas
    (OA en el minuto 3: 16 personas; tareas en el minuto 4: 1–14), que
    deformaban los ejes;
  - las leyendas de F1, F2, F3 y F5 pasan fuera del área de datos.
- **Paleta:** la de las tareas del paso 1, validada como paleta categórica
  (separación para daltonismo ≥ 9.2). OA va en gris neutro de referencia, con
  marcador cuadrado y nombre en el eje.
- **Referencias:** comprobadas por búsqueda web. Hay que confirmar a mano tres
  datos antes de pasarlos al `.bib`:
  - el DOI de Craig 2012;
  - el de Tran 2020;
  - el volumen, las páginas y el DOI de Wascher 2014.
- **Hallazgo para la discusión:** la literatura mide la fatiga por tiempo en
  tarea en **horas**, y aquí las tareas duran **≈ 3 minutos**. Se dice
  explícitamente que no es una réplica a la misma escala.
- **Sin cambios en `EEG-Memoria`**, por decisión del autor.
- **Pendiente para después** (fuera de este paso): la vista de evolución
  temporal en la plataforma, como fase propia con su plan en EEG-Library
  (contrato) y en EEG-Platform.
