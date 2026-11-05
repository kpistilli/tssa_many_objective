# TSSA many-objective: extensión del prototipo Conacyt-Uninter (v2)

> Basado en: Telesca, E.; López-Pires, F. (2025). *A Unified Framework for
> Multi-Objective Assignment of Teachers and Students to Educational
> Institutions*. Preprint 202512.0426.v1. Universidad Internacional Tres
> Fronteras (UNINTER), Paraguay. Proyecto original Conacyt-Uninter:
> Dr. Fabio López (tutor investigador), Ing. Eliana Telesca (investigadora
> en formación).
>
> Extensión many-objective (NSGA-III, U-NSGA-III, C-TAEA, RVEA, SPEA2; 3 a 6 objetivos):
> "Many-Objective Assignment of Teachers and Students to Schools: An Experimental 
> Evaluation of Algorithms" (K. Pistilli, F. López-Pires, R. Benítez — 
> Facultad Politécnica, Universidad Nacional de Asunción)

Esta carpeta extiende el código del preprint hacia una formulación de **muchos objetivos** resuelta con **C-TAEA** y otros algoritmos many-objective. Incorpora un protocolo experimental reproducible al estilo de *Optimizing Student Assignment to Educational Establishments: A Many-Objective Approach* (Casco et al., CACIC 2024).

La **v2** incluye:

- el resultado de una corrida es el **Conjunto Pareto completo**, acumulado con un archivo externo en todas las generaciones;
- ese conjunto se persiste en **CSV/NPZ y en PostgreSQL**;
- la **decisión multicriterio a posteriori** (compromiso, ASF, rodilla, pseudo-pesos);
- la **sección derivada**;
- **Kruskal–Wallis + corrección por comparaciones múltiples + GD+**;
- el **greedy que atiende primero a los alumnos más lejanos**;
- el objetivo opcional **f6 de estabilidad**;
- la variante **f4 = máx(máx_est, máx_doc)** como opción.

Los datos de entrada no se modifican. Las tablas `estudiantes`, `docentes`, `clases`, `establecimientos`, `instituciones` y `asignacion_mec` se leen igual que antes.

## 1. Formulación

| Clave | Sentido | Definición |
|---|---|---|
| `f1_distancia` | min | Distancia promedio estudiante→establecimiento + distancia promedio docente→establecimiento (ec. 22) |
| `f2_balance` | min | Varianza de alumnos por clase (ec. 23). Con `--f2 std` se usa el desvío estándar de la v1.3 |
| `f3_colocalizacion` | max | Proporción de docentes con 2 clases en el mismo establecimiento (ec. 24, versión del código) |
| **`f4_distancia_max`** | **min** | **Nueva.** Distancia máxima (criterio minimax, equidad del peor caso). Ver la variante abajo |
| `f5_afinidad_grado` | max | Opcional. Proporción de clases activas cuyo docente tiene el mismo grado que la clase (usa G_j, que ya forma parte de D_j) |
| `f6_estabilidad` | min | Opcional (propuesta de la v2). Proporción de estudiantes que **cambian de establecimiento** respecto de `asignacion_mec`: el costo político de implementar el plan |

**Formulación de la tesis: f1–f4** (`--n-obj 4`, cuatro objetivos, es decir, muchos objetivos). f6 se usa en un análisis complementario (fase 4). f5 queda implementado pero **fuera del alcance de la tesis**, como trabajo futuro (ver abajo).

Conjuntos predefinidos: `--n-obj 3` (preprint), `--n-obj 4` (f1–f4, formulación de la tesis) y `--n-obj 5` (f1–f5). Cualquier combinación se arma con `--objetivos`, por ejemplo `--objetivos f1_distancia f2_balance f3_colocalizacion f4_distancia_max f6_estabilidad`.

### Dos definiciones de f4 (`--f4-modo`)

```
suma (defecto):  f4 = max_i U_{i,E(i)}  +  max_j V_{j,E(j)}
max  (v2):       f4 = max( max_i U_{i,E(i)},  max_j V_{j,E(j)} )
```

La versión `max` es la del documento `f4_formulacion_many_objective.md`. Se puede linealizar para un modelo MIP como `min z s.a. z ≥ U·xA, z ≥ V·xD`. Tiene un punto ciego: sólo "ve" al grupo con el máximo más grande.

Con la solución greedy sobre 15.763 alumnos, el máximo de alumnos es 18,3 km y el de docentes 5,9 km. En ese caso el peor desplazamiento docente no se optimiza hasta que supere al de los alumnos. La `suma` mantiene a ambos grupos sensibles. Si en la tesis se quiere la versión `max`, conviene declarar esta limitación o separar f4 en dos objetivos, como propone la alternativa "f4b" del mismo documento.

### f6 y la asignación del MEC

Con f6 activo, la asignación actual del MEC tiene f6 = 0 por definición: no cambia a nadie. Por eso **ninguna solución puede dominarla**. Es un resultado esperable, no un error.

Con f6 la pregunta pasa a ser *"cuánto mejora cada objetivo por cada punto porcentual de estudiantes reasignados"*. Esa curva se lee en el frente, por ejemplo en `matriz_dispersion.png`.

**Semillas de estabilidad.** Cuando f6 es un objetivo, la población inicial incluye, además del greedy, la asignación del MEC (f6 = 0) y tres soluciones intermedias en las que el 25 %, 50 % y 75 % de los alumnos conservan su clase del MEC y el resto toma la del greedy (reparadas para ser factibles). Es lo mismo para todos los algoritmos (`semillas_estabilidad` en `operators.py`). Sin ellas, la búsqueda partía sólo de la zona del greedy, que mueve al 94 % de los alumnos, y el frente cubría únicamente f6 entre 0,93 y 0,96: la curva de estabilidad no se podía trazar. Con los datos de Asunción, las semillas tienen f6 = 0; 0,24; 0,47 y 0,71.

Las restricciones son:

- **g1:** R2, capacidad, más un mínimo opcional `--cap-min`;
- **g2:** R5;
- **g3:** R3;
- **g4:** R4, sin superposición de franjas horarias para un mismo docente (ver abajo);
- **g5:** grado del estudiante igual al grado de la clase.

### Diferencias entre paper y código que conviene declarar en la tesis

- **f3.** La ecuación 24 suma también los pares (l, l), de modo que un docente con una sola clase aporta 1 (en el ejemplo didáctico, f3 = 1). El código mide la proporción de docentes con exactamente 2 clases en el mismo establecimiento, que es lo que se quiere premiar. El paper debería corregirse.
- **f2.** El paper define la varianza y la v1.3 usaba el desvío estándar. Por defecto se usa la varianza.
- **R2 mínimo.** El paper exige al menos 25 alumnos por clase, pero la v1.3 no lo controlaba. `--cap-min` elige el mínimo: 0 como la v1.3, 25 como el paper, 10 como CACIC.
- **g4 y el turno Mañana-Tarde.** R4 se evalúa sobre **franjas horarias**, no sobre el texto del turno: Mañana = {mañana}, Tarde = {tarde}, Mañana-Tarde = {mañana, tarde} (`franjas_de_turno` en `integrated_problem.py`). Dos clases de un mismo docente están en conflicto si comparten una franja, así que un docente con una clase Mañana-Tarde (jornada completa) no puede tener una segunda clase. g4 cuenta las clases de más en cada franja ocupada. Con turnos simples coincide con la v1.3; antes, "Mañana-Tarde" se trataba como un tercer turno independiente. La interpretación se verificó en `asignacion_mec`: ningún docente con una clase Mañana-Tarde tiene otra clase. Consecuencia: f3 solo puede provenir de pares Mañana + Tarde.
- **Capacidad por aula.** En los datos de Asunción la capacidad registrada coincidía exactamente con la matrícula de cada grado, lo que dejaba f2 constante. Se adopta una capacidad estándar de **40 alumnos por aula** para 1º a 3º grado, salvo dos aulas que hoy ya funcionan con más alumnos y conservan su tamaño real: clase 1600 (48) y clase 1612 (46). Es un supuesto que debe declararse.
- **Una clase por sección real.** La tabla `clases` tenía 259 filas para 318 secciones del MEC, por lo que varias secciones caían en la misma clase y el MEC violaba la capacidad. Se agregaron las 59 secciones faltantes (318 clases). Con ellas el MEC es completamente factible (g1–g5 = 0).
- **Mapeo MEC → clases.** Cada sección se asigna a una clase según su **orden** dentro de (establecimiento, grado, turno), no según su letra. Así, secciones con letras no consecutivas ("B" sola, o "A" y "C") no se juntan en una misma clase. `referencias.csv` informa `mapeo_seccion_sin_clase_propia` si alguna sección quedara sin clase propia.
- **f2 cuenta las clases vacías.** La varianza se calcula sobre todas las clases, incluidas las que quedan sin alumnos (con 0). Por eso f2 penaliza cerrar aulas y favorece repartir a los alumnos entre todas las clases disponibles.
- **f3 se divide por todos los docentes** (281), no solo por los que tienen dos clases, igual que en la v1.3.
- **Clases sin alumnos.** No cuentan para la carga docente (g3, g4) ni para f3 y f5.
- **Infraestructura.** La f3 de infraestructura de CACIC 2024 no es replicable, porque el esquema no tiene los vectores Vi/Vs/Vm. La observación es del documento v2.

## 2. Archivos

| Archivo | Contenido |
|---|---|
| `integrated_problem.py` | U y V precalculadas y evaluación vectorizada. Objetivos f1–f6 configurables, `f4_modo`, `x_base` para f6. API compatible con la v1.3 |
| `operators.py` | Reparación R1–R5 (Sección 5.2 del paper), greedy que atiende primero a los más lejanos (Sección 6.2.2), muestreo por grado con semilla greedy y mutación categórica |
| `pareto.py` | **Nuevo, desde `pareto_export.py`.** `ParetoArchive`, `extraer_pareto_set`, decisión a posteriori, sección derivada, exportación CSV/NPZ y persistencia en PostgreSQL |
| `integrated_optimization.py` | NSGA-II, NSGA-III, U-NSGA-III, C-TAEA, RVEA, SPEA2 (y AGE-MOEA con numba). `result.pareto_set` con el conjunto completo. `select_best_individual(result, metodo)`. `DatabaseManager.save_pareto_set()` |
| `instancia.py` | Carga desde BD, CSV o el SQL de ejemplo, instancia sintética, escenarios por grado y conversión de `asignacion_mec` en un vector x |
| `metrics.py` | HV, IGD+, **GD+**, spacing, **Kruskal–Wallis**, Mann–Whitney con corrección **Holm/Bonferroni**, A12, dominancia y tabla de mejora contra MEC/greedy |
| `experiments.py` | Protocolo experimental completo (ver secciones 3 a 5) |
| `integrated_app.py` | Una corrida: imprime el Conjunto Pareto, las soluciones elegidas por cada método y la comparación con el MEC. Opciones `--exportar-pareto`, `--metodo-decision`, `--guardar-pareto` y `--max-archivo` |
| `tests/test_tssa.py` | 26 pruebas: ejemplo didáctico, regresión contra la v1.3, reparación, greedy, los 6 algoritmos, archivo Pareto, decisión, f4_modo, f6, Holm/Bonferroni, exportación, franjas Mañana-Tarde (R4), mapeo de secciones no consecutivas, exclusión del greedy infactible, reproducibilidad por semilla, semillas de estabilidad (f6) y decisión `domina_mec` |
| `campania_tesis.sh` | Campaña experimental completa y reanudable (fases 0 a 6, y 7 opcional; ver sección 4) |
| `cotas_exactas.py` | Cotas exactas de la componente de alumnos y brecha de los frentes (fase 7; sección 12) |

`database.py` y `integrated_viewer_optimizado.py` de tu proyecto no se modifican. El viewer sigue funcionando porque se sigue escribiendo `asignacion_mec_opt`, ahora con la solución elegida y la sección derivada.

### Qué se tomó para la v2 y qué no

| Para la v2 | Decisión |
|---|---|
| Archivo externo no dominado (`ParetoArchive`) | ✅ Integrado. En NSGA-III la última población tenía 5–10 no dominadas y el archivo 27–52 |
| Persistencia del Conjunto Pareto (`opt_run`, `pareto_solucion`, `asignacion_pareto`) | ✅ Integrado. F se guarda en signo natural y el conjunto en NPZ binario, más liviano que CSV |
| Decisión a posteriori (ASF, rodilla, pseudo-pesos) | ✅ Integrado, más "compromiso", "lexicográfico", "min_fk" y "domina_mec" |
| Sección derivada | ✅ Integrado |
| Kruskal–Wallis, Bonferroni, GD+ | ✅ Integrado. Holm es la corrección por defecto (más potente que Bonferroni y válida igual) |
| Greedy que empieza por los más lejanos | ✅ Integrado como comportamiento por defecto |
| f6 estabilidad | ✅ Integrado como objetivo opcional |
| f4 = máx de máximos | ✅ Integrado como `--f4-modo max`. El defecto sigue siendo `suma` (ver 1) |
| SBX + redondeo sobre índices | ❌ No integrado. En la misma instancia y con los objetivos de la v2, los operadores categóricos dieron HV 0,95 contra 0,62 (p = 0,008). Se puede reproducir con `--operadores sbx` |
| Codificación restringida por grado | ➖ Innecesaria. Muestreo, mutación y reparación ya respetan el grado mediante listas de candidatas ordenadas por distancia |
| `tasa_factibles` sobre `res.G` | ❌ Corregido. Ahora se mide sobre la población final (`tasa_factibles_pob_final`) |
| Escenarios por grado con todos los docentes | ❌ Corregido. En `por-grado` se filtran los docentes por grado, y además se ofrece el escenario `completo` |

## 3. Estudio comparativo de algoritmos

Por defecto `experiments.py` compara 6 algoritmos evolutivos y la heurística **greedy** como línea de base no evolutiva:

| Algoritmo | Tipo | Notas |
|---|---|---|
| NSGA-II | dominancia + crowding | enfoque del preprint (con `@3` reproduce sus 3 objetivos) |
| NSGA-III | direcciones de referencia | direcciones "energy" = población exacta (100) |
| U-NSGA-III | NSGA-III unificado | mismas direcciones que NSGA-III |
| C-TAEA | dos archivos coevolutivos (convergencia + diversidad) | pensado para problemas fuertemente restringidos |
| RVEA | vectores de referencia con adaptación angular | puede rendir mal en instancias muy chicas; es esperable |
| SPEA2 | fuerza de Pareto + densidad | referencia clásica |
| **greedy** | heurística determinista | se corre una sola vez por escenario |

La sintaxis `algoritmo@k` (por ejemplo `nsga2@3`) hace que ese algoritmo optimice con los primeros k objetivos. Su Conjunto Pareto se re-evalúa después en el espacio del experimento, lo que permite comparar la formulación de 3 objetivos del preprint en igualdad de condiciones.

## 4. Instalación y uso


Crear entorno virtual (opcional):

```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Configurar variables de entorno en `.env`:

```
DB_USER=postgres
DB_PASSWORD=tu_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=Asignacion_MEC
```

Ejecuta el test:

```bash
python -m pytest -q tests                 # 26 pruebas
```

Una corrida:

```bash
python integrated_app.py                                   # BD, NSGA-III, 4 obj.; guarda el Conjunto Pareto
python integrated_app.py --algoritmo nsga2 --n-obj 3       # configuración del preprint
python integrated_app.py --fuente ejemplo --exportar-pareto pareto_out   # sin PostgreSQL
python integrated_app.py --metodo-decision rodilla --guardar-pareto todas
python integrated_app.py --objetivos f1_distancia f2_balance f3_colocalizacion \
                                     f4_distancia_max f6_estabilidad
```

Campaña completa de la tesis (recomendado):

```bash
bash campania_tesis.sh            # fases 0 a 6
bash campania_tesis.sh 5 6        # sólo las fases indicadas
bash campania_tesis.sh 7          # cotas exactas (opcional, segundos)
bash campania_tesis.sh 8          # cruce de dos puntos, revisión R1-1 (opcional, ~1,5-2 h)
```

| Fase | Contenido |
|---|---|
| 0 | Pruebas unitarias (aborta si fallan) |
| 1 | Prueba la conexión a PostgreSQL y exporta un snapshot a `datos_asu/` con huella SHA-256 |
| 2 | Piloto de convergencia: NSGA-II, NSGA-III y C-TAEA, 5 corridas, 800 generaciones |
| 3 | Experimento principal: `nsga2@3`, NSGA-II, NSGA-III, U-NSGA-III, `ctaea@3`, C-TAEA, RVEA, SPEA2 y greedy; escenarios completo y por grado. Incluye el diseño 2×2 algoritmo × formulación (C-TAEA y NSGA-II con 3 y 4 objetivos) |
| 4 | Análisis complementario: 5 objetivos con estabilidad (f1–f4 + f6). Mide cuánto mejoran los demás objetivos según la proporción de alumnos reasignados |
| 5 | 8 ablaciones sobre NSGA-III (SBX, f4 = max, sin reparación, sin semilla greedy, cap-min 25 y 10, cruce 2p, f2 = std) |
| 6 | Corrida con C-TAEA que guarda el Conjunto Pareto en PostgreSQL para el viewer. La solución que va a `asignacion_mec_opt` se elige con `domina_mec`: la de compromiso entre las que mejoran al MEC en todos los objetivos (respaldar antes `asignacion_mec_opt`) |
| 7 | Cotas exactas de la componente de alumnos (`cotas_exactas.py`, sin MOEA): óptimo de la distancia media (LP de transporte) y de la peor distancia (cuello de botella por flujo), con certificado de que existe una asignación de docentes factible. Sólo necesita la fase 1; con la fase 3 calcula además la brecha de cada algoritmo. Opcional: no se ejecuta por defecto (ver sección 12) |
| 8 | Revisión R1-1: escenario completo con cruce de dos puntos para C-TAEA, NSGA-III, NSGA-II y SPEA2 (+ greedy), 30 corridas con las mismas semillas que la fase 3 (120 corridas). Opcional; salida en `resultados/tesis/principal_cruce2p/` |

Parámetros: `RUNS=30`, `POP=100`, `GEN=500`, `MAX_ARCHIVO=2000`, `PROCESOS=4`. Es reanudable: salta los experimentos que ya tienen `informe.txt`. Si un experimento falla, se registra y la campaña continúa. Los logs quedan en `resultados/tesis/logs/`.

Comandos sueltos equivalentes:

```bash
# 1) Congelar los datos (reproducibilidad)
python experiments.py --fuente db --exportar-csv datos_asu --runs 0 --salida resultados/tesis/exportacion

# 2) Experimento principal: preprint (NSGA-II, 3 obj.) vs el estudio comparativo con 4 objetivos
python experiments.py --fuente csv --csv-dir datos_asu \
    --algoritmos nsga2@3 nsga2 nsga3 unsga3 ctaea@3 ctaea rvea spea2 greedy \
    --runs 30 --pop 100 --gen 500 --max-archivo 2000 --escenario ambos --procesos 4 \
    --guardar-pareto conjunto --salida resultados/principal

# 3) Formulación de 5 objetivos con estabilidad
python experiments.py ... --objetivos f1_distancia f2_balance f3_colocalizacion \
    f4_distancia_max f6_estabilidad --algoritmos nsga3 unsga3 ctaea rvea greedy

# 4) Ablaciones (una a la vez)
python experiments.py ... --operadores sbx          # operadores del preprint / v2
python experiments.py ... --f4-modo max             # definición de f4 de la v2
python experiments.py ... --sin-reparacion
python experiments.py ... --sin-semilla-greedy
python experiments.py ... --cap-min 25              # R2 completo del paper
python experiments.py ... --correccion bonferroni
```

## 5. Salidas (por escenario)

| Archivo | Contenido |
|---|---|
| `config.json` | parámetros, versiones y fecha |
| `referencias.csv` | objetivos y violaciones de la asignación MEC y del greedy |
| `corridas.csv` | por corrida: semilla, tiempo, HV, IGD+, GD+, spacing, tamaño del Conjunto Pareto, no dominadas de la última población, tasa de factibles, mejor valor y valor de compromiso por objetivo, cuántas soluciones dominan al MEC |
| `resumen.csv` | media, desvío, mediana, mínimo y máximo por algoritmo |
| `kruskal_wallis.csv` | prueba global por métrica |
| `estadistica.csv` | Mann–Whitney por pares con p crudo y corregido (Holm/Bonferroni) y A12 |
| `mejora_vs_referencias.csv` | Tablas 4 y 5 de CACIC: dominancia y % de mejora, frente vs MEC y vs greedy |
| `frente_referencia.csv` | frente no dominado global |
| `frente_union_<algoritmo>.csv` | frente de todas las corridas del algoritmo, con la solución elegida por compromiso, rodilla y pseudo-pesos |
| `frentes/<alg>_runNN_frente.csv` | PF* de cada corrida (signo natural, con la columna `seleccionada`) |
| `conjuntos_pareto/<alg>_runNN/` | con `--guardar-pareto conjunto`: `pareto_front.csv`, `pareto_set.npz` (P*) y `metadata.json`; con `asignaciones`, además las asignaciones de las soluciones seleccionadas |
| `convergencia.csv` | HV por generación (sobre el archivo acumulado) |
| `informe.txt` | resumen legible |
| `*.png` | boxplots, convergencia, coordenadas paralelas, matriz de dispersión |

En la base de datos (`integrated_app.py` o `DatabaseManager.save_pareto_set`):

- `opt_run`: una fila por corrida;
- `pareto_solucion`: una fila por solución, con F y la etiqueta de decisión;
- `asignacion_pareto`: las asignaciones detalladas de las soluciones seleccionadas, o de todas;
- `asignacion_mec_opt`: la solución elegida, para el viewer.

## 6. Recomendaciones para el capítulo experimental

1. Correr **30 ejecuciones** por algoritmo.
2. Declarar **500 generaciones como presupuesto fijo** de evaluaciones (100 por generación en todos los algoritmos). El HV sigue creciendo lentamente hasta la generación 800, pero el orden de los algoritmos no cambia desde la 300 (sección 8).
3. Reportar la **mediana y el RIC** de HV e IGD+, el Kruskal–Wallis global y los pares con p corregido y A12.
4. Presentar el **Conjunto Pareto** (coordenadas paralelas o matriz de dispersión) y, aparte, la solución elegida por cada método de decisión. Así se separa la optimización de la decisión, tal como pide la formulación (Algoritmo 1: "Salida: Pareto Set").
5. Usar escenarios **por grado** y **completo**. El completo muestra el acoplamiento de docentes entre grados y turnos, que es el aporte del TSSA.
6. En la instancia de ejemplo del SQL `f3` es siempre 0, porque ningún establecimiento tiene clases en ambos turnos. Con datos reales sí informa.
7. La instancia **sintética** sirve sólo para pruebas de escala.
8. Presentar al **greedy como un punto del frente**: se inyecta como semilla en todos los algoritmos y es el extremo de mínima distancia (f1 y f4) de los frentes de las fases 3 y 4. **No es el óptimo en distancia**: en la ablación `cap-min 10` NSGA-III encontró f1 = 1,80 km (contra 1,93 km del greedy), con soluciones que también son factibles en el problema base. La búsqueda simplemente no explora esa zona (cerrar clases chicas). La ablación `--sin-semilla-greedy` muestra que sin la semilla se pierde todo el extremo de distancia (mejor f1 = 3,33 km).
9. Si el greedy viola alguna restricción (por ejemplo con `--cap-min`, que no contempla), se excluye de los indicadores y del frente de referencia, y sólo queda como referencia en `referencias.csv`.
10. Las ablaciones usan **NSGA-III** como algoritmo base; aclararlo en el texto para no confundirlas con el algoritmo recomendado (C-TAEA).

## 7. Tiempos medidos

Asunción (7.540 alumnos, 281 docentes, 318 clases, 69 establecimientos), 4 corridas en paralelo:

- Una corrida de 800 generaciones tarda entre 230 s (NSGA-II) y 250 s (C-TAEA).
- La campaña completa (fases 3 a 6, 500 generaciones) lleva unas 10 horas.

Instancia sintética (15.763 alumnos, 456 clases):

- Con NSGA-III y población 100, una generación tarda ~0,65 s. El archivo Pareto agrega menos de 0,01 s por generación.
- 100 generaciones tardan ~65–90 s por corrida.
- 30 corridas × 7 métodos con `--procesos 4` llevan aproximadamente entre 1 y 1,5 horas por escenario.

## 8. Calibración y piloto (Asunción, octubre de 2026)

Instancia: 7.540 alumnos, 281 docentes, 318 clases (una por sección real), 69 establecimientos. Capacidad 40 (48 y 46 en las clases 1600 y 1612). El MEC es factible: g1–g5 = 0.

**Generaciones (`GEN=500`).** Piloto de 5 corridas × 800 generaciones. HV medio por generación:

| Generación | C-TAEA | NSGA-II | NSGA-III |
|---|---|---|---|
| 300 | 0,568 | 0,536 | 0,541 |
| 500 | 0,605 | 0,564 | 0,564 |
| 800 | 0,651 | 0,593 | 0,590 |

El orden de los algoritmos es el mismo desde la generación 300. En la 500 cada algoritmo alcanzó el 93–96 % de su HV final.

**Archivo de Pareto (`MAX_ARCHIVO=2000`).** Con 500 el archivo se llenaba y se recortaban soluciones por crowding. Con 2.000, el máximo observado con 4 objetivos fue 1.518 soluciones por corrida. El Conjunto Pareto final une el archivo con la última población, así que puede superar el tope hasta en el tamaño de la población.

**Declaración: tope de 2.000 con 5 objetivos.** Con 5 objetivos el frente es mucho más grande. En la primera ejecución de la fase 4, 6 de 30 corridas de NSGA-III y 11 de 30 de U-NSGA-III llegaron al tope (C-TAEA y RVEA no). Se decidió **mantener el tope en 2.000** para todas las fases: cuando se supera, el archivo conserva las soluciones de mayor distancia de crowding, lo que preserva la extensión y la diversidad del frente, pero descarta soluciones en las zonas más pobladas. En esas corridas el Conjunto Pareto es una muestra representativa y no el conjunto completo, lo que puede subestimar levemente su HV. Subir el tope alargaría mucho el posprocesamiento, porque el HV exacto con 5 objetivos ya cuesta alrededor de 1,2 minutos por corrida. `corridas.csv` permite identificar las corridas recortadas (`n_nd` ≥ 2.000).

**Comparación contra C-TAEA (verificada).**

- Presupuesto igual: los tres algoritmos hacen 100 evaluaciones por generación (contado con `evaluator.n_eval`).
- El HV recalculado de forma independiente coincide con el oficial (diferencia < 1e-15).
- C-TAEA supera a NSGA-II y NSGA-III en HV, IGD+ y GD+ en las 5 corridas (Mann–Whitney p = 0,0079, el mínimo posible con 5 contra 5; Holm p = 0,024; A12 = 1). El resultado se mantiene con punto de referencia 1,5 y normalizando con MEC y greedy incluidos.
- Cobertura (fracción del frente B dominada por A): C(C-TAEA, NSGA-II) = 0,705 y C(C-TAEA, NSGA-III) = 0,723, contra C(NSGA-II, C-TAEA) = 0,153 y C(NSGA-III, C-TAEA) = 0,188. C-TAEA aporta 960 de las 2.090 soluciones del frente de referencia.
- El archivo externo sólo lee `algorithm.pop`, que en C-TAEA es el archivo de convergencia (CA); su archivo de diversidad (DA) no se archiva. Si algo, esto perjudica a C-TAEA, así que la ventaja medida es conservadora.

**Frente vs MEC.** Sólo C-TAEA encuentra soluciones que dominan al MEC (84 en la unión de corridas). La mejor tiene f1 = 3,62 km contra 8,05 km del MEC (−55 %) sin empeorar f2, f3 ni f4. NSGA-II alcanza el balance del MEC sólo con peor distancia máxima, y NSGA-III no llega a esa zona del frente. Con 5 corridas, las 30 de la campaña deben confirmarlo.

**Reproducibilidad por semilla (corregido el 29/10/2026).** Desde pymoo 0.6.2, `minimize(seed=...)` usa un generador propio y ya no siembra `np.random`, que es el que usan los operadores del TSSA (muestreo, mutación y reparación). Por eso, en las fases 2, 3 y 3b las corridas no eran exactamente reproducibles a partir de su semilla, aunque sí eran réplicas aleatorias independientes, que es lo que suponen las pruebas estadísticas. Ahora cada corrida fija `np.random.seed(seed)` antes de `minimize` (`experiments.py` e `integrated_optimization.py`), y la prueba `test_reproducibilidad_semilla` lo controla. Las corridas de `ctaea`, `nsga2` y `nsga2@3` de la fase 3b funcionaron como réplica independiente de la fase 3: ninguna medida difirió significativamente (Mann–Whitney, todos los p > 0,05).

## 9. Alcance: formulación de 4 objetivos y f5 como trabajo futuro

La tesis adopta **f1–f4** como formulación principal. La fase de afinidad de grado (f1–f5) se retiró de la campaña el 30/10/2026 y las fases siguientes se renumeraron (ablaciones = fase 5; guardado en la BD = fase 6): en su ejecución, la búsqueda partía del greedy, que asigna el docente más cercano sin mirar el grado (f5 = 0,37), y la mutación sólo elige entre los 5 docentes más cercanos. Los algoritmos no superaron f5 = 0,54, mientras que la asignación del MEC tiene f5 = 1. Esos resultados no son válidos para comparar algoritmos y no se reportan.

Observación para trabajo futuro (no evaluada experimentalmente): un greedy que asigna primero docentes del mismo grado alcanza f5 = 1 con una distancia promedio de 2,09 km, contra 1,93 km del greedy actual, y es factible. Asignar los docentes a su propio grado parecería costar muy poca distancia. Para estudiarlo habría que inyectar semillas de afinidad, como se hizo con f6 (`semillas_estabilidad`).

## 10. Método de decisión `domina_mec` (recomendación que mejora al MEC en todo)

`--metodo-decision domina_mec` elige, entre las soluciones del Conjunto Pareto que **dominan** a la asignación actual del MEC (iguales o mejores en todos los objetivos y estrictamente mejores en alguno), la de compromiso, calculada sólo sobre ese subconjunto. Con `domina_mec:<método>` se usa otro criterio dentro del subconjunto, por ejemplo `domina_mec:min_f1` (la de menor distancia promedio). Si ninguna solución domina al MEC, se aplica el método sobre todo el frente y se registra una advertencia. Lo usan `integrated_app.py` (fase 6) y el viewer; la referencia es la asignación del MEC del problema (`x_base`).

Sobre el frente de C-TAEA de la fase 3 (escenario completo; MEC: f1 = 8,05 km, f2 = 41,2, f3 = 0,128, f4 = 21,87 km):

| Método | f1 (km) | f2 | f3 | f4 (km) | ¿Domina al MEC? |
|---|---|---|---|---|---|
| `compromiso` | 3,12 | 91,1 | 0,146 | 19,57 | No (peor balance) |
| `domina_mec` | 4,24 | 32,3 | 0,153 | 20,60 | Sí |
| `domina_mec:min_f1` | 3,82 | 38,5 | 0,153 | 21,55 | Sí |

## 11. Ablaciones (fase 5) y trabajo futuro sobre el operador de cruce

Resultados completos en el documento del proyecto `resultados_fase5_ablaciones.md`. Comparación contra NSGA-III base de la fase 3 con normalización común y Holm sobre 21 comparaciones:

| Variante | HV (mediana; base 0,798) | Conclusión |
|---|---|---|
| Cruce de 2 puntos | 0,883 | Mejor en HV, IGD+ y GD+, y más rápido (121 s contra 194 s por corrida) |
| f2 = desvío estándar | 0,790 | Sin diferencia |
| f4 = máx (en f1–f3) | — | Sin diferencia |
| Sin semilla greedy | 0,649 | Pierde el extremo de distancia |
| cap-min 10 | 0,479 | Costo de exigir ≥ 10 alumnos por clase |
| Sin reparación | 0,356 | Sólo el 1 % de la población es factible; el frente se reduce al greedy |
| Operadores SBX | 0,278 | Mucho peor: confirma los operadores categóricos |
| cap-min 25 | 0,053 | El frente casi colapsa |

**Decisión (31/10/2026): el cruce de 2 puntos se reporta como hallazgo y trabajo futuro; no se repiten las fases.** Todas las fases usaron cruce uniforme para todos los algoritmos, así que la comparación entre algoritmos es justa, pero la calidad absoluta de los frentes podría mejorar con cruce de 2 puntos. Hipótesis: el cruce de 2 puntos preserva bloques contiguos del vector, mientras que el uniforme rompe la relación entre la asignación de los alumnos y la de los docentes de sus clases. Trabajo futuro: (1) verificar si el orden de los algoritmos (C-TAEA primero) se mantiene con cruce de 2 puntos; (2) estudiar operadores que respeten explícitamente esa relación (por ejemplo, cruce por establecimiento).

## 12. Cotas exactas de la componente de alumnos (fase 7, noviembre de 2026)

Respuesta al revisor R1-3 (¿qué tan lejos del óptimo están los frentes?). No es una corrida evolutiva ni obliga a repetir fases: es un cálculo determinista sobre `datos_asu/` que tarda unos segundos (`bash campania_tesis.sh 7`, salida en `resultados/tesis/cotas_exactas/`). El mismo script está en el artículo como `codigo/cotas_exactas_asu.py`.

- **Media de alumnos (componente de f1).** Problema de transporte por grado: alumno → establecimiento con la capacidad agregada de sus clases del grado. Es totalmente unimodular, así que el LP (HiGHS) da la solución entera óptima.
- **Peor distancia de alumnos (componente de f4).** Cuello de botella: búsqueda binaria sobre las distancias con factibilidad por flujo máximo. También se calcula la media mínima sujeta a ese peor viaje.
- **Certificado.** Para cada óptimo se arma la asignación a clases y un MILP de factibilidad comprueba que existe una asignación de docentes que cumple R3–R5. Como existe, las cotas se **alcanzan**: son los óptimos exactos de esas componentes en el problema completo. Los docentes y R7 no intervienen, así que valen con y sin R7.

| Componente de alumnos | Óptimo exacto | Greedy | Mejor de los MOEAs (fase 3) | MEC |
|---|---|---|---|---|
| Distancia media (km) | 0,607 | 0,711 (+17,1 %) | 0,664, RVEA (+9,4 %) | 3,112 |
| Peor distancia (km) | 2,418 | 3,262 (+34,9 %) | 3,262, todos (+34,9 %) | 11,206 |

- La capacidad sube la media del ideal sin capacidad (0,556 km, clase más cercana) al óptimo de 0,607 km, pero no limita el peor viaje: 2,418 km es la distancia del alumno más alejado a su clase más cercana.
- Lograr el peor viaje mínimo casi no cuesta media: la media mínima con peor viaje ≤ 2,418 km es 0,607 km (+0,05 %). En esta instancia f1 y f4 no compiten en la componente de alumnos.
- Ningún algoritmo mejora la peor distancia de alumnos del greedy en ninguna de las 240 corridas, y en la media sólo se acercan algunas corridas (mediana por algoritmo = greedy). Esto refuerza la observación de la sección 6, punto 8: el greedy no es el óptimo en distancia y la búsqueda no explora esa zona.
- Trabajo futuro barato: sembrar los MOEAs con las soluciones del LP y del cuello de botella además del greedy.

## 13. Cruce de dos puntos en el escenario completo (fase 8, noviembre de 2026)

Respuesta al revisor R1-1. Las 120 corridas (C-TAEA, SPEA2, NSGA-II y NSGA-III, mismas semillas que la fase 3) se compararon con las de la fase 3 usando una normalización común (`articulo_tssa/codigo/cruce2p_asu.py`). Medianas:

| Algoritmo | HV UX | HV 2P | IGD+ UX | IGD+ 2P | GD+ UX | GD+ 2P | Dominan al MEC UX | Dominan al MEC 2P |
|---|---|---|---|---|---|---|---|---|
| C-TAEA | 0,594 | 0,678 | 0,097 | 0,087 | 0,041 | 0,067 | 28/30 | 12/30 |
| SPEA2 | 0,559 | 0,710 | 0,120 | 0,057 | 0,104 | 0,057 | 0/30 | 5/30 |
| NSGA-II | 0,547 | 0,684 | 0,136 | 0,089 | 0,099 | 0,086 | 0/30 | 3/30 |
| NSGA-III | 0,548 | 0,658 | 0,125 | 0,071 | 0,094 | 0,055 | 0/30 | 0/30 |

- El cruce de dos puntos mejora el HV de todos los algoritmos, pero el orden cambia: SPEA2 pasa a ser el mejor en HV e IGD+.
- C-TAEA sigue siendo el que más veces domina al MEC, pero ya no es el único. La diferencia con SPEA2 no es significativa (Fisher + Holm, p = 0,08).
- Para el objetivo práctico de mejorar la asignación vigente, la mejor configuración sigue siendo C-TAEA con cruce uniforme.
- Las conclusiones de la tesis sobre algoritmos quedan condicionadas al cruce uniforme. El 2×2, la estabilidad y las ablaciones no se repitieron con 2P.

