# Discusión — el efecto del tiempo en tarea

> Borrador para la discusión y las conclusiones (capítulo 7). Las referencias
> están en `referencias.md`; dos de ellas tienen el DOI por confirmar.

## Qué se puede afirmar

En personas que el modelo no había visto, y con un análisis pre-registrado y
evaluado una sola vez:

1. **El EEG de un casco de consumo cambia de forma sistemática a lo largo de
   las tareas cognitivas largas** (PASAT y PVTB, unos 3 minutos). Es la H2,
   confirmada con holgura en las dos tareas.
2. **Ese cambio no lo explican el movimiento de la cabeza ni la calidad de
   contacto.** Es la H4, confirmada: un clasificador de solo artefactos queda
   cerca del azar.
3. **En PVTB, la beta posterior relativa baja más que en el reposo.** Es la
   H3, confirmada en PVTB. En PASAT no concluye, con una potencia baja de
   antemano.
4. **Con la grabación completa, las tareas cambian más que el reposo con ojos
   abiertos.** Es la H1b, confirmada por poco y frágil: sin la persona de
   `test` expuesta en los sondeos, su IC incluye el 0.

## Qué no se puede afirmar

- **Que el cambio de las tareas supere al del reposo a igual tiempo
  transcurrido** (la H1 no concluye). Es la versión que separa «estar haciendo
  la tarea» de «llevar más tiempo grabando». Sin ella, no se puede atribuir el
  cambio a la tarea con evidencia confirmatoria. La H1b sí concluye, pero
  mezcla las dos cosas, porque las tareas duran un 60 % más que el reposo.
- **Que el EEG mida la fatiga.** No hay ninguna medida subjetiva ni conductual
  **dentro** de la grabación. Lo que se afirma es un cambio compatible con el
  efecto del tiempo en tarea, no una medida del estado de la persona.
- **Que el efecto sea el de la literatura sobre fatiga mental.** Esos
  estudios miden a lo largo de 1–3 horas (Boksem et al., 2005; Wascher et al.,
  2014). Aquí cada tarea dura unos 3 minutos. Lo observado es compatible con
  la primera fase de ese proceso, o con la pérdida de vigilancia y la
  habituación en minutos, pero no es una réplica a la misma escala.

## Lectura conjunta: por qué el resultado es mixto

**A favor de que el cambio es de la tarea** (desarrollo):

- **Especificidad:** el modelo de cada tarea ordena sus propias ventanas y no
  las del reposo (el de PVTB no ordena nada en OA ni en OC).
- **Dientes de sierra:** el cambio se reinicia en parte al empezar la tarea
  siguiente, cosa que una deriva continua del equipo no haría.
- **El tiempo desde la colocación del casco no lo explica:** su coeficiente
  es pequeño y, en la alfa, de signo contrario.
- **La firma diferencial es la que cabe esperar:** se desplaza el peso hacia
  las frecuencias lentas (baja la beta relativa).

**En contra, o al menos sin resolver:**

- **El reposo también cambia, y bastante.** En OA, la alfa y la theta
  absolutas suben con la misma inclinación que en las tareas. Esa parte del
  cambio es común a cualquier grabación con ojos abiertos y no es una firma de
  la tarea. Coincide con los marcadores que la literatura asocia a la fatiga
  (theta y alfa: Tran et al., 2020), lo que obliga a ser prudentes al
  interpretarlos en registros cortos.
- **En `test`, OA cambió más que en el desarrollo** (ρ 0.104 frente a 0.063 y
  0.039), y PASAT a igual tiempo quedó a su altura. Con 28 personas, la
  diferencia entre conjuntos puede ser azar de muestreo. No se ha investigado
  porque el test está gastado.
- **El orden fijo del protocolo** impide separar por completo el tiempo en la
  tarea del tiempo de sesión. Solo un diseño contrabalanceado lo resolvería.

**Evidencia exploratoria complementaria** (posterior al test, sin
replicación):

- dentro de la misma persona, las sesiones en que el EEG cambió más a lo largo
  del PVTB fueron las de más lapsos en el PVTB (r ≈ 0.14);
- el cambio del reposo no mostró esa asociación y el movimiento no la explica;
- va en el sentido del decremento de vigilancia, pero es un efecto pequeño, en
  un solo contraste de tres, y no replica con la medida espectral simple.

## El mapa de escalas temporales

Este resultado completa el argumento del TFG sobre **qué detecta el EEG de
consumo en esta cohorte según la escala temporal del fenómeno**:

| Escala | Ejemplo | ¿Se detecta en personas no vistas? |
|---|---|---|
| Segundos–minutos, cambios grandes de estado | ojos abiertos/cerrados, reposo/tarea, tipo de tarea | **Sí** (test del 2026-10-07) |
| Minutos dentro de una tarea | principio frente a final de PASAT y PVTB | **Sí** el cambio (H2); **no confirmado** que sea mayor que en reposo a igual tiempo (H1) |
| Semanas | fatiga percibida (VAS, PedsQL), inicio/fin de trimestre | **No** |

El patrón es coherente:

- **Dentro de una grabación** se comparan momentos con el mismo casco, la
  misma colocación y el mismo día, y aparece un cambio sistemático.
- **Entre sesiones**, la variabilidad de la colocación, del gel, del casco y
  del día supera a los efectos lentos y pequeños que se buscaban.

El nulo del VAS deja así de ser un resultado aislado: es la otra cara de la
misma limitación de medida.

## Limitaciones

- **Orden fijo del protocolo:** el tiempo en la tarea y el tiempo de sesión
  están parcialmente confundidos.
- **Grabaciones cortas** (≈ 3 min): la escala no es la de la literatura sobre
  fatiga mental.
- **Sin marcadores de ensayo en los EDF:** el rendimiento solo existe como
  total por sesión.
- **Control de contacto débil**, porque la medida de calidad del casco está
  saturada. No se controla la actividad muscular (EMG).
- **Decisiones del desarrollo tomadas a la vista de los datos**, todas
  declaradas en el pre-registro:
  - la familia PASAT + PVTB;
  - el contraste a igual tiempo;
  - la medida `rho_ventanas`, buscada tras una primera mirada a `val` que no
    fue buena;
  - la secuencia H1b → H1.
- **Una persona de `test` estuvo expuesta** en los sondeos iniciales; se
  declaró y se dio la sensibilidad sin ella.
- **Potencia limitada:** con 28 personas en `test`, la potencia estimada para
  la H1 era del 75 % (58 % en el escenario prudente).

## Trabajo futuro

- **Un diseño con orden contrabalanceado** y un reposo de la misma duración
  que las tareas, para separar el tiempo en la tarea del tiempo de sesión.
- **Marcadores de ensayo en el registro**, para relacionar el EEG con los
  lapsos minuto a minuto.
- **Registros más largos** (decenas de minutos), para acercarse a la escala
  de la literatura.
- **Deep learning** (EEGNet o ShallowConvNet; Lawhern et al., 2018;
  Schirrmeister et al., 2017), para ver si la señal cruda contiene información
  más allá de la potencia por banda. Coste medido en CPU: 30–45 h. En el test
  de esta línea se renunció a él por escrito, así que cualquier evaluación en
  `test` sería exploratoria.

## Frase de resumen

> En personas no vistas, el EEG de un casco de consumo registra un cambio
> sistemático a lo largo de tareas cognitivas de unos 3 minutos (PASAT y PVTB).
> Ese cambio no se explica por el movimiento ni por el contacto de los
> electrodos y es compatible con el efecto del tiempo en tarea. No se ha
> podido confirmar que sea mayor que el que se produce en reposo con los ojos
> abiertos durante el mismo tiempo. Con la grabación completa, la diferencia
> es positiva pero frágil.
