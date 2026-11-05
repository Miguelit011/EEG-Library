# Referencias de la línea (comprobadas el 2026-10-28)

Comprobadas por búsqueda web el 2026-10-28. **«DOI por confirmar»** significa
que se confirmaron los autores, el título, la revista, el volumen y las
páginas, pero no el DOI. Hay que resolverlo en doi.org o en la web de la
revista antes de pasarlo al `.bib`.

| Clave | Referencia | Estado | Para qué se cita |
|---|---|---|---|
| `boksem2005` | Boksem, M. A. S., Meijman, T. F. y Lorist, M. M. (2005). Effects of mental fatigue on attention: An ERP study. *Cognitive Brain Research*, 25(1), 107–116. doi:10.1016/j.cogbrainres.2005.04.011 | Comprobada (PDF del autor) | Paradigma de tiempo en tarea: 3 h de tarea atencional sin descanso; aumento de la potencia lenta y pérdida de la atención dirigida a metas |
| `wascher2014` | Wascher, E., Rasch, B., Sänger, J., Hoffmann, S., Schneider, D., Rinkenauer, G., Heuer, H. y Gutberlet, I. (2014). Frontal theta activity reflects distinct aspects of mental fatigue. *Biological Psychology*, 96, 57–65. | Título y resumen comprobados (repositorio); **volumen, páginas y DOI por confirmar** | La alfa sube pronto (máximo hacia la primera hora) y la theta frontal sube de forma continua con el tiempo en tarea |
| `craig2012` | Craig, A., Tran, Y., Wijesuriya, N. y Nguyen, H. (2012). Regional brain wave activity changes associated with fatigue. *Psychophysiology*, 49(4), 574–582. | Comprobada (repositorio UTS); **DOI por confirmar** | Con la fatiga suben theta y alfa en toda la corteza, y también la beta (sobre todo frontal), en conducción simulada hasta la fatiga |
| `tran2020` | Tran, Y., Craig, A., Craig, R., Chai, R. y Nguyen, H. (2020). The influence of mental fatigue on brain activity: Evidence from a systematic review with meta-analyses. *Psychophysiology*, 57(5), e13554. | Comprobada; **DOI por confirmar** (probablemente 10.1111/psyp.13554) | Metaanálisis de 21 estudios: la theta es el marcador más robusto y la alfa el segundo. g global = 0.68 |
| `lawhern2018` | Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P. y Lance, B. J. (2018). EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*, 15(5), 056013. doi:10.1088/1741-2552/aace8c | Comprobada | Solo si se cita el deep learning como trabajo futuro |
| `schirrmeister2017` | Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., *et al.* (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391–5420. doi:10.1002/hbm.23730 | Comprobada | Ídem (ShallowConvNet, Deep4Net) |

## Avisos para la discusión

- **Escala temporal distinta.** La literatura mide la fatiga por tiempo en
  tarea a lo largo de **1–3 horas** (Boksem 2005: 3 h; Wascher 2014: la alfa
  llega a su máximo hacia la hora). Aquí cada tarea dura **≈ 3 minutos**. Lo
  que se observa es compatible con el inicio de ese proceso (o con la
  habituación y la pérdida de vigilancia en minutos), pero **no es el mismo
  fenómeno a la misma escala**. No hay que presentarlo como réplica de esos
  estudios.
- **Beta.** Craig 2012 encuentra que la beta **absoluta** sube con la fatiga.
  Aquí lo que baja es la beta **relativa**: el peso de la beta respecto al
  total, compatible con que suba la potencia lenta. No es contradictorio, pero
  conviene decir que son magnitudes distintas.
- **Theta.** En el metaanálisis (Tran 2020), la theta es el marcador principal.
  Aquí la theta frontal absoluta sube en las tareas, pero **también en el
  reposo** (paso 1, anexo A), así que no distingue la tarea del reposo a igual
  tiempo.

## Fuentes de la comprobación

- [Boksem et al. 2005 (PDF del autor)](https://www.boksem.nl/pdf/boksem2005cbr.pdf)
- [Wascher et al. 2014 (registro en sonar.ch)](https://sonar.ch/global/documents/136799)
- [Craig et al. 2012 (repositorio UTS)](https://opus.lib.uts.edu.au/handle/10453/18398)
- [Tran et al. 2020 (Macquarie University)](https://researchers.mq.edu.au/en/publications/the-influence-of-mental-fatigue-on-brain-activity-evidence-from-a/)
- [Lawhern et al. 2018 (IOP)](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta)
- [Schirrmeister et al. 2017 (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5655781/)
