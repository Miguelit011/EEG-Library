# Fatiga general y fatiga cognitiva (PedsQL) frente al EEG

Línea de exploración pedida por los tutores: ¿guardan alguna relación con la
señal EEG las medidas de **fatiga general (FG)** y **fatiga cognitiva (FC)** de
los cuestionarios?

FG y FC son dos subescalas de la *PedsQL Multidimensional Fatigue Scale*
(Varni & Limbers, 2008): seis ítems cada una, de 0 (totalmente en desacuerdo) a
4 (totalmente de acuerdo), con más puntuación = más fatiga. Se puntúan como
suma de 0 a 24. Se respondieron el mismo día que el EEG, en todas las
mediciones M1–M6.

Cada paso vive en su propia subcarpeta con su script, sus tablas, sus figuras
y un `INFORME.md` cuyo plan se escribe **antes** de ejecutar. Reglas comunes a
todos los pasos:

- **Solo `train` + `val`.** El test está gastado (ver `docs/MODELADO.md`) y no
  se toca. Las personas de test se apartan en el primer paso de cada script.
- Agrupar siempre por `persona_uid`. La identidad del lado EEG es la de la Fase
  12 (`labels.csv`) y solo entran sesiones etiquetadas.
- **FG y FC se centran por medición** (M1…M6) antes de cualquier análisis. Hay
  un efecto común de calendario (M1 es la más baja y hay una subida suave a lo
  largo del curso), del mismo orden que el del VAS. Sin quitarlo, una deriva del
  EEG a lo largo del curso podría parecer relación con FG o FC. Las medias por
  medición se estiman solo con la partición de entrenamiento (o dentro de cada
  pliegue).
- No se tipifica dentro de persona ni se transforma la escala (ver el paso 1).
- El `CÓDIGO` del alumno (iniciales + fecha de nacimiento de menores) solo se
  usa en memoria para comprobar el enlace. No sale en ninguna tabla, figura,
  log ni archivo del repositorio.
- Ejecutar desde la raíz del repositorio con `.venv/Scripts/python.exe`.

## Datos

- **Cuestionarios:** `<EEG_DATA>/BD_COMPLETA_corregida.xlsx`, hoja
  `CUESTIONARIOS`. Necesita `openpyxl` (grupo `dev`). Cada paso registra su
  SHA-256 en `resultados.json`.
- **Sesiones EEG:** `<EEG_DATA>/metadata/labels.csv` (Fase 12).

> **Las versiones anteriores de la base no sirven para esta línea.** En
> `BD_COMPLETA_ConCalificaciones.xlsx` y `BD_COMPLETA_sincalificaciones.xlsx`,
> los 18 ítems PedsQL (FG, FS/D y FC) de M5 y M6 están codificados como
> `max(respuesta − 1, 0)`, lo que hace caer FG y FC a la mitad. El primer
> descriptivo de esta línea lo detectó y el equipo del estudio lo corrigió
> (hoja `AUDIT_CORRECCION_PEDS`). El paso 1 verifica la corrección celda a
> celda.

## Enlace y correcciones

Todo lo hace [`datos_fg_fc.py`](datos_fg_fc.py), que importan todos los pasos.

El enlace es el de la Fase 12:

1. `(centro, pulsera)` → `persona_uid`, con `normalizar_vas`, que aplica la
   corrección F111/F115.
2. Unión con las sesiones etiquetadas por `(persona_uid, medición)`.

**No se usa el `CÓDIGO` para enlazar** (un alumno tiene dos) **ni
`PARTICIPANTES_PROYC`** (`C2_F108` tiene cuatro sesiones EEG y figura como no
participante). El VAS de la base tiene que coincidir con el de `labels.csv` en
cada sesión que entra; si no, el script se detiene.

| Incidencia en la base | Resolución |
|---|---|
| `C2_F105` figura en C4 solo en M4 y M5 | Corrección: esas dos filas van a C2 (mismo alumno; su EEG está archivado en C2) |
| La M3 de `C2_F116` figura en C7 | Corrección: va a C2 (caso ya conocido de la Fase 12) |
| C3/66 aparece con dos códigos según la medición | Misma persona (`C3_F066`): misma pulsera, centro, sexo y curso, sin mediciones repetidas. Excepción documentada a la regla un código ↔ un `persona_uid` |
| Filas sin sexo, edad ni curso | El sexo y el curso de cada persona son la moda de sus filas; el sexo se contrasta con el del nombre de los EDF |

## Pasos

| Paso | Carpeta | Qué hace | Estado |
|---|---|---|---|
| 1 | [`01_descriptivo/`](01_descriptivo/INFORME.md) | Solo cuestionario: corrección de la base, enlace, consistencia, efecto de calendario, ICC, fiabilidad intra-persona, relación con el VAS, sexo/curso/centro | Hecho (2026-10-22): base verificada; hay paso 3 con FG y FC |
| 2 | [`02_entre_personas/`](02_entre_personas/INFORME.md) | Rasgo: FG/FC por persona frente al EEG | Hecho (2026-10-22): sin evidencia de relación (FG ρ = +0.15, p = 0.08; FC +0.01) |
| 3 | `03_intra_persona/` | Estado: FG/FC por sesión, centrado por persona y medición, frente al EEG | Pendiente |
