# Rutas originales y ficheros congelados

`ESTADO.md` y `PREREGISTRO_TEST.md` son registros históricos y no se editan, así
que sus menciones a otros ficheros conservan los nombres que tenían antes de la
reorganización del 2026-10-08. Esta tabla las traduce. El estado anterior
completo está en el tag `pre-reorganizacion`.

| Ruta original | Ruta actual |
|---|---|
| `ESTADO.md` | `docs/ESTADO.md` |
| `MODELADO.md` | `docs/MODELADO.md` |
| `ANALISIS_OA_OC.md` | `docs/analisis/ANALISIS_OA_OC.md` |
| `ANALISIS_REPOSO_TAREA.md` | `docs/analisis/ANALISIS_REPOSO_TAREA.md` |
| `ANALISIS_TAREAS_EJECUTIVAS.md` | `docs/analisis/ANALISIS_TAREAS_EJECUTIVAS.md` |
| `PCA_CLUSTER.md` | `docs/analisis/ANALISIS_PCA_CLUSTERING.md` |
| `REGRESION_LINEAL.md` | `docs/analisis/ANALISIS_REGRESION_LINEAL.md` |
| `figures/*.png` | `resultados/figuras/*.png` |
| `figures/fase15_retirada_2026-08-26/` | `archivo/fase15_retirada_2026-08-26/` |
| `analisis/salidas/*.log` | `resultados/logs/*.log` |
| `tests/test_<módulo>.py` | `tests/<subpaquete>/test_<módulo>.py` |

`PREREGISTRO_TEST.md` y `analisis/analisis_test.py` **no se han movido**.
`analisis_test.py` depende de esa ubicación (`parents[1]`, `sys.path` y
`from analisis_regresion_lineal import ols`).

## Ficheros congelados

| Fichero | SHA-256 (registrado en `resultados.json`) |
|---|---|
| `PREREGISTRO_TEST.md` | `bdca1a7bca689b92856fe30769df059f8fabc0f66f7d1023226a844d27325121` |
| `analisis/analisis_test.py` | `20e25ce8aa3b1a90e26b87c46884c0fb43688842e4b5a5a4552b199115f3f0a4` |
| `docs/ESTADO.md` | `363e62df7f0d8126ffba861212a0ea4424805a8d27dcaf82c86d502c26dde132` |

Los dos primeros se calcularon sobre la copia de trabajo de Windows, con saltos
de línea CRLF; git los guarda con LF. `.gitattributes` fija `eol=crlf` en ellos
para que el hash sea el mismo en cualquier plataforma. Desde el tag:

```bash
git show preregistro-test-v1:PREREGISTRO_TEST.md | sed 's/$/\r/' | sha256sum
```

El test se ejecutó una sola vez (`processed/test_20261007-133358`, commit
`bb55aee`) y **no se vuelve a ejecutar**.
