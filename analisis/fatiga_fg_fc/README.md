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
- Agrupar siempre por `persona_uid`. Identidad del lado EEG: la de la Fase 12
  (`labels.csv`); solo entran sesiones etiquetadas.
- **Las puntuaciones se tipifican dentro de cada medición** (M1…M6). Entre M4
  y M5 FG y FC caen a la mitad en toda la cohorte, sin que se muevan el VAS ni
  el ítem de fatiga mental del mismo formulario: se trata como un cambio de
  administración, no de fatiga (paso 1). La referencia para tipificar son los
  alumnos que no hicieron EEG.
- Cualquier estadístico estimado con datos de los participantes se estima solo
  con la partición de entrenamiento (o dentro de cada pliegue).
- El `CÓDIGO` del alumno (iniciales + fecha de nacimiento de menores) solo se
  usa en memoria para comprobar el enlace. No sale en ninguna tabla, figura,
  log ni archivo del repositorio.
- Ejecutar desde la raíz del repositorio con `.venv/Scripts/python.exe`.

## Datos

- **Cuestionarios:** `<EEG_DATA>/BD_COMPLETA_sincalificaciones.xlsx`, hoja
  `CUESTIONARIOS` (versión pedida por el tutor; sustituye a
  `BD_COMPLETA_ConCalificaciones.xlsx`). Necesita `openpyxl` (grupo `dev`).
- **Sesiones EEG:** `<EEG_DATA>/metadata/labels.csv` (Fase 12).

## Enlace y correcciones

Todo lo hace [`datos_fg_fc.py`](datos_fg_fc.py), que importan todos los pasos.

El enlace es el de la Fase 12: `(centro, pulsera)` → `persona_uid` con
`normalizar_vas` (que aplica la corrección F111/F115), y unión con las sesiones
etiquetadas por `(persona_uid, medición)`. **No se usa el `CÓDIGO` para
enlazar** (un alumno tiene dos) **ni `PARTICIPANTES_PROYC`** (`C2_F108` tiene
cuatro sesiones EEG y figura como no participante). El VAS de la base tiene que
coincidir con el de `labels.csv` en cada sesión que entra; si no, el script se
detiene.

| Incidencia | Resolución |
|---|---|
| El VAS y las pruebas cognitivas de M6 de C3/63 pasan al alumno de la pulsera 2063 | Se excluye la sesión `C3_F063_M6`: los datos no permiten saber de quién es el EEG. Pendiente de confirmar con el equipo; si la versión nueva es la correcta, `labels.csv` atribuye ese VAS (1 sesión de val) a otra persona |
| C3/66 aparece con dos códigos según la medición | Misma persona (`C3_F066`): misma pulsera, centro, sexo y curso, sin mediciones repetidas. Excepción documentada a la regla un código ↔ un `persona_uid` |
| `C2_F105` figura en C4 solo en M4 y M5 | Corrección: esas dos filas van a C2 (mismo alumno; su EEG está archivado en C2) |
| La M3 de `C2_F116` figura en C7 | Corrección: va a C2 (caso ya conocido de la Fase 12) |
| Filas sin sexo, edad ni curso | El sexo y el curso de cada persona son la moda de sus filas; el sexo se contrasta con el del nombre de los EDF |

## Pasos

| Paso | Carpeta | Qué hace | Estado |
|---|---|---|---|
| 1 | [`01_descriptivo/`](01_descriptivo/INFORME.md) | Solo cuestionario: enlace, salto M4→M5, representatividad de la referencia, tipificación, ICC, fiabilidad intra-persona, relación con VAS | Hecho (2026-10-22): referencia ponderada (C1); hay paso 3 con FG y FC (C2) |
| 2 | `02_entre_personas/` | Rasgo: FG/FC por persona frente al EEG | Pendiente |
| 3 | `03_intra_persona/` | Estado: FG/FC por sesión, centrado por persona, frente al EEG | Pendiente |
