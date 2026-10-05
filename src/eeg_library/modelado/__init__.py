"""
Modelado (Fase 14 en adelante): evaluación y modelos supervisados sobre el VAS.

Arnés de evaluación del Random Forest: agregación a nivel de sesión, cortes
de terciles estimados por pliegue, medias leave-one-out por persona y
métricas de regresión y clasificación ordinal. Detalle de diseño en
MODELADO.md.

`reposo` es el control positivo del pipeline: clasificación binaria de ojos
abiertos (OA) frente a ojos cerrados (OC), sin relación con el VAS. Detalle en
ANALISIS_OA_OC.md.

`reposo_tarea` es el contraste siguiente: reposo (OA u OC) frente a tarea
ejecutiva, un efecto de carga cognitiva. Reutiliza la evaluación de `reposo` y
añade el manejo del desbalance de clases y el desglose de errores por
condición. Detalle en ANALISIS_REPOSO_TAREA.md.

`tareas_ejecutivas` es el escalón siguiente a `reposo_tarea`: clasificación
multiclase de las cinco tareas ejecutivas (STROOP, PASAT, PVTB, MSIT, PVSAT),
sin OA ni OC. Reutiliza el arnés de `reposo` (agnóstico al número de clases) y
añade las métricas multiclase, la matriz de confusión y el análisis de qué
tareas se confunden entre sí. Detalle en ANALISIS_TAREAS_EJECUTIVAS.md.
"""
