#!/bin/bash
# =====================================================================
# Campaña experimental completa para la tesis (TSSA many-objective)
#
# Uso:   bash campania_tesis.sh            # todas las fases
#        bash campania_tesis.sh 3 4        # sólo las fases indicadas
#
# Fases:
#   0  Pruebas unitarias
#   1  Congelar datos reales de PostgreSQL en CSV (datos_asu/)
#   2  Piloto de convergencia (fija el número de generaciones)
#   3  Experimento principal: 7 algoritmos + ctaea@3 + greedy, escenarios completo
#      y por grado. Incluye el diseño 2x2 algoritmo x formulación (C-TAEA y NSGA-II
#      con 3 y con 4 objetivos), que separa el efecto del algoritmo del de agregar f4
#   4  Análisis complementario: 5 objetivos con estabilidad (f1-f4 + f6)
#      (la formulación principal de la tesis es f1-f4; f5 queda como trabajo futuro)
#   5  Ablaciones (una a la vez, contra NSGA-III)
#   6  Guardar en PostgreSQL el Conjunto Pareto final (para el viewer)
#   7  Cotas exactas de la componente de alumnos (LP de transporte y cuello de
#      botella por flujo; sin MOEA, segundos). Sólo necesita la fase 1; si existe
#      la fase 3, calcula además la brecha de cada algoritmo al óptimo.
#      No forma parte de la ejecución por defecto: bash campania_tesis.sh 7
#   8  Revisión R1-1: escenario completo con cruce de DOS PUNTOS para C-TAEA,
#      NSGA-III, NSGA-II y SPEA2 (+ greedy), 30 corridas con las mismas semillas
#      que la fase 3. Responde si el orden de los algoritmos se conserva con el
#      operador que ganó la ablación. Opcional: bash campania_tesis.sh 8
#
# Es reanudable: si un experimento ya tiene informe.txt, se salta.
# Si un experimento falla, se registra y la campaña continúa.
# =====================================================================

cd "$(dirname "$0")"
source venv/bin/activate

# ---------------- Parámetros del protocolo ----------------
CSV_DIR=datos_asu
RUNS=30            # ejecuciones independientes por algoritmo
POP=100
GEN=500            # ajustar según el piloto (fase 2)
PROCESOS=4         # núcleos del equipo
RUNS_ABL=30        # corridas por ablación
MAX_ARCHIVO=2000   # tope del archivo de Pareto (con 500 el piloto saturaba)
SALIDA=resultados/tesis

DATOS="--fuente csv --csv-dir $CSV_DIR"
BASE="$DATOS --runs $RUNS --pop $POP --gen $GEN --procesos $PROCESOS --max-archivo $MAX_ARCHIVO"
ABL="$DATOS --runs $RUNS_ABL --pop $POP --gen $GEN --procesos $PROCESOS --max-archivo $MAX_ARCHIVO \
     --escenario completo --algoritmos nsga3 greedy --guardar-pareto frente"

FASES="${*:-0 1 2 3 4 5 6}"
mkdir -p "$SALIDA/logs"
LOG="$SALIDA/logs/campania_$(date +%Y%m%d_%H%M%S).log"
FALLOS=()

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# Evita que el Mac se duerma durante la campaña
if command -v caffeinate >/dev/null; then caffeinate -i -w $$ & fi

# Ejecuta un experimento salvo que ya exista su informe
exp() {
  local nombre=$1; shift
  local out="$SALIDA/$nombre"
  if [ -f "$out/completo/informe.txt" ]; then
    log "SALTO  $nombre (ya terminado)"; return
  fi
  log "INICIO $nombre"
  local t0=$SECONDS
  if python experiments.py "$@" --salida "$out" >>"$SALIDA/logs/$nombre.log" 2>&1; then
    log "FIN    $nombre ($(( (SECONDS-t0)/60 )) min)"
  else
    log "FALLO  $nombre (ver $SALIDA/logs/$nombre.log)"; FALLOS+=("$nombre")
  fi
}

# Trazabilidad del entorno
{ echo "fecha: $(date)"; echo "host: $(uname -a)"; python --version
  git rev-parse HEAD 2>/dev/null || echo "sin git"; pip freeze; } > "$SALIDA/entorno.txt"
log "Campaña iniciada. Fases: $FASES  (RUNS=$RUNS POP=$POP GEN=$GEN MAX_ARCHIVO=$MAX_ARCHIVO)"

for f in $FASES; do case $f in
  0) log "Fase 0: pruebas unitarias"
     python -m pytest -q tests 2>&1 | tee -a "$LOG" || { log "Pruebas fallidas: abortando"; exit 1; } ;;

  1) log "Fase 1: verificando conexión a PostgreSQL"
     python -c "import sys; from database import test_conexion; sys.exit(0 if test_conexion() else 1)" \
         2>&1 | tee -a "$LOG"
     [ ${PIPESTATUS[0]} -eq 0 ] || { log "Sin conexión a la BD (revisar .env): abortando"; exit 1; }
     if [ -f "$CSV_DIR/estudiantes.csv" ]; then
       log "Fase 1: $CSV_DIR ya existe (snapshot previo de la BD; borrarlo para reexportar)"
     else
       log "Fase 1: exportando snapshot de la BD a $CSV_DIR/"
       python experiments.py --fuente db --exportar-csv "$CSV_DIR" --runs 0 --salida "$SALIDA/exportacion" 2>&1 | tee -a "$LOG"
       [ ${PIPESTATUS[0]} -eq 0 ] || { log "Fallo al exportar desde la BD: abortando"; exit 1; }
     fi
     (cd "$CSV_DIR" && shasum -a 256 *.csv) > "$SALIDA/datos_sha256.txt" ;;

  2) log "Fase 2: piloto de convergencia (5 corridas, 800 generaciones)"
     exp piloto_convergencia $DATOS --runs 5 --pop $POP --gen 800 --procesos $PROCESOS --max-archivo $MAX_ARCHIVO \
         --escenario completo --algoritmos nsga2 nsga3 ctaea --conv-cada 10
     log "Revisar $SALIDA/piloto_convergencia/completo/convergencia*.png y ajustar GEN" ;;

  3) log "Fase 3: experimento principal"
     exp principal $BASE --escenario ambos \
         --algoritmos nsga2@3 nsga2 nsga3 unsga3 ctaea@3 ctaea rvea spea2 greedy \
         --guardar-pareto conjunto ;;

  4) log "Fase 4 (complementaria): 5 objetivos con estabilidad (f6)"
     exp obj5_estabilidad $BASE --escenario completo \
         --objetivos f1_distancia f2_balance f3_colocalizacion f4_distancia_max f6_estabilidad \
         --algoritmos nsga3 unsga3 ctaea rvea greedy --guardar-pareto conjunto ;;

  5) log "Fase 5: ablaciones"
     exp abl_operadores_sbx   $ABL --operadores sbx
     exp abl_f4_max           $ABL --f4-modo max
     exp abl_sin_reparacion   $ABL --sin-reparacion
     exp abl_sin_semilla      $ABL --sin-semilla-greedy
     exp abl_capmin25         $ABL --cap-min 25
     exp abl_capmin10         $ABL --cap-min 10
     exp abl_cruce_2p         $ABL --cruce 2p
     exp abl_f2_std           $ABL --f2 std ;;

  6) log "Fase 6: guardando el Conjunto Pareto final en PostgreSQL"
     python integrated_app.py --fuente db --algoritmo ctaea --n-obj 4 \
         --pop $POP --gen $GEN --seed 42 --max-archivo $MAX_ARCHIVO \
         --metodo-decision domina_mec --guardar-pareto seleccion \
         --exportar-pareto "$SALIDA/pareto_bd" >>"$SALIDA/logs/bd_final.log" 2>&1 \
       && log "FIN    BD actualizada (opt_run, pareto_solucion, asignacion_pareto, asignacion_mec_opt)" \
       || { log "FALLO  guardado en BD (ver $SALIDA/logs/bd_final.log)"; FALLOS+=("bd_final"); } ;;

  7) log "Fase 7: cotas exactas (componente de alumnos de f1 y f4)"
     if python cotas_exactas.py --csv-dir "$CSV_DIR" --exp "$SALIDA/principal/completo" \
          --salida "$SALIDA/cotas_exactas" >>"$SALIDA/logs/cotas_exactas.log" 2>&1; then
       log "FIN    cotas exactas ($SALIDA/cotas_exactas/)"
     else
       log "FALLO  cotas exactas (ver $SALIDA/logs/cotas_exactas.log)"; FALLOS+=("cotas_exactas")
     fi ;;

  8) log "Fase 8 (revisión R1-1): escenario completo con cruce de dos puntos"
     exp principal_cruce2p $BASE --escenario completo --cruce 2p \
         --algoritmos ctaea nsga3 nsga2 spea2 greedy --guardar-pareto frente ;;

  *) log "Fase desconocida: $f" ;;
esac; done

if [ ${#FALLOS[@]} -eq 0 ]; then log "Campaña terminada sin fallos. Resultados en $SALIDA/"
else log "Campaña terminada con fallos: ${FALLOS[*]}"; fi
