# Figuras de la Fase 15 (retirada) — 2026-08-26

> **No son resultados vigentes.** Las generó `analisis/fase15i_figuras.py` el
> 2026-08-26 para explicar al tutor por qué el EEG no predice el VAS. La
> Fase 15 entera (guiones `fase15*.py`, sus logs y su sección de `docs/MODELADO.md`)
> se retiró el 2026-09-15 y su código no se llegó a commitear, así que **estas
> figuras ya no se pueden regenerar** con el repositorio actual. Se conservan
> solo como registro.

Algunas cifras no coinciden con la documentación vigente porque la Fase 15 las
calculaba de otra forma (otro conjunto de features, otra muestra o una
ejecución distinta). Para la memoria hay que usar las de `docs/MODELADO.md` y de los
`ANALISIS_*.md`.

| Archivo | Qué muestra | Cifra (`figuras_cifras.json`) | Cifra vigente |
|---|---|---|---|
| `fig1_controles_positivos.png` | Controles positivos: la matriz identifica a la persona y acierta la condición muy por encima del azar | persona 0.248 (azar 0.0076); condición 0.608 (azar 0.143) | Igual (`docs/MODELADO.md`, Fase 14) |
| `fig2_varianza_vas.png` | Dónde está la varianza del VAS: ICC, reparto entre y dentro de persona, y R² de conocer solo a la persona | ICC 0.470; R² LOO 0.296 | ICC(1) 0.457 sobre las 653 sesiones (`docs/ESTADO.md`, Fase 12); R² LOO 0.2961 |
| `fig3_prediccion_vs_real.png` | La nube plana: predicción del RF frente al VAS real, junto a la media por persona | R² EEG −0.010; R² LOO 0.296 | Igual (`docs/MODELADO.md`, Fase 14) |
| `fig4_techo_efecto.png` | Correlaciones individuales de las **374 features de la Fase 15** con el VAS: ninguna supera Bonferroni | máx. \|ρ\| 0.162; 0 significativas | No hay equivalente: las 374 features eran de la Fase 15 |
| `fig5_theta_alfa.png` | El índice theta/alfa frente al VAS, dentro de persona | ρ −0.003, p 0.955 (gl 400) | ρ +0.004, p 0.929 (`docs/analisis/ANALISIS_REGRESION_LINEAL.md`) |
| `fig6_permutacion.png` | Nulos por permutación con el valor observado encima | BA 3 clases 0.394 (p 0.017); ρ reposo 0.167 (p 0.005) | BA 0.3947 (`docs/MODELADO.md`, Fase 14b); la ρ de reposo era un hallazgo de la Fase 15 |
| `figuras_cifras.json` | Las cifras que se dibujan en cada figura | — | — |
