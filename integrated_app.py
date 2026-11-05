# ================================================================
# integrated_app.py
#
# Basado en:
#   Telesca, E.; López-Pires, F. (2025). "A Unified Framework for
#   Multi-Objective Assignment of Teachers and Students to Educational
#   Institutions". Preprint 202512.0426.v1. Universidad Internacional
#   Tres Fronteras (UNINTER), Paraguay.
#   Proyecto original Conacyt-Uninter: Dr. Fabio López (tutor investigador),
#   Ing. Eliana Telesca (investigadora en formación).
#
# Extensión many-objective (NSGA-III, U-NSGA-III, C-TAEA, RVEA, SPEA2; 3 a 6 objetivos):
#   "Many-Objective Assignment of Teachers and Students to Schools: An Experimental 
#   Evaluation of Algorithms" (K. Pistilli, F. López-Pires, R. Benítez — 
#   Facultad Politécnica, Universidad Nacional de Asunción)
#
# Ejecución "batch/CLI" de UNA optimización (para experimentos con varias
# corridas y estadística usar experiments.py):
#   python integrated_app.py                               # BD, NSGA-III, 4 objetivos
#   python integrated_app.py --algoritmo nsga2 --n-obj 3   # configuración del preprint
#   python integrated_app.py --fuente ejemplo --no-guardar # sin PostgreSQL
#   python integrated_app.py --fuente ejemplo --exportar-pareto pareto_out
#   python integrated_app.py --objetivos f1_distancia f2_balance f3_colocalizacion \
#          f4_distancia_max f6_estabilidad --metodo-decision rodilla
# El resultado es el CONJUNTO PARETO completo; la solución guardada en
# asignacion_mec_opt es una decisión a posteriori (--metodo-decision).
# ================================================================

import argparse
import logging
import sys
from pathlib import Path

# Permite ejecutar el script desde cualquier carpeta: los módulos del
# proyecto se importan desde la carpeta de este archivo.
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from instancia import cargar_desde_csv, cargar_desde_sql, x_desde_asignacion  # noqa: E402
from integrated_optimization import run_integrated_optimization, select_best_individual  # noqa: E402
from integrated_problem import OBJETIVOS, IntegratedProblem  # noqa: E402
from metrics import domina  # noqa: E402
from pareto import METODOS_DECISION, etiquetar_decisiones, exportar_pareto  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Opciones de línea de comandos:
      datos        --fuente (db | csv | ejemplo), --csv-dir
      modelo       --n-obj, --objetivos, --f4-modo, --cap-min
      algoritmo    --algoritmo, --pop, --gen, --seed
      salida       --metodo-decision, --guardar-pareto, --exportar-pareto, --no-guardar
    """
    ap = argparse.ArgumentParser(description="Optimización TSSA (una corrida)")
    ap.add_argument("--fuente", choices=["db", "csv", "ejemplo"], default="db")
    ap.add_argument("--csv-dir", default="datos")
    ap.add_argument("--algoritmo", default="nsga3",
                    choices=["nsga2", "nsga3", "unsga3", "ctaea", "rvea", "spea2", "agemoea"])
    ap.add_argument("--n-obj", type=int, default=4, choices=[3, 4, 5])
    ap.add_argument("--objetivos", nargs="+", default=None, choices=list(OBJETIVOS),
                    help="lista explícita de objetivos (reemplaza a --n-obj)")
    ap.add_argument("--f4-modo", choices=["suma", "max"], default="suma")
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gen", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap-min", type=int, default=0)
    ap.add_argument("--metodo-decision", default="compromiso",
                    help=f"solución a guardar en asignacion_mec_opt: {', '.join(METODOS_DECISION)}, "
                         "min_fk o domina_mec:<método> (domina_mec = la de compromiso entre las "
                         "soluciones que dominan al MEC)")
    ap.add_argument("--guardar-pareto", default="seleccion",
                    choices=["ninguna", "seleccion", "todas", "solo_compat"],
                    help="BD: asignaciones detalladas del Conjunto Pareto a guardar "
                         "(solo_compat = solo asignacion_mec_opt, como la v1.3)")
    ap.add_argument("--exportar-pareto", default=None,
                    help="carpeta donde exportar el Conjunto Pareto (CSV/NPZ), sin BD")
    ap.add_argument("--no-guardar", action="store_true", help="no escribir en la base de datos")
    ap.add_argument("--max-archivo", type=int, default=500,
                    help="tamaño máximo del archivo externo de Pareto")
    return ap.parse_args()


def main():
    """
    Flujo: cargar datos -> (si hay asignación MEC) convertirla en x_mec ->
    construir el problema -> optimizar (y guardar en BD si corresponde) ->
    imprimir el Conjunto Pareto, la solución elegida y la comparación con el MEC.
    """
    args = parse_args()
    # ---------- 1) Datos ----------
    if args.fuente == "db":
        from instancia import cargar_desde_db
        inst = cargar_desde_db()
    elif args.fuente == "csv":
        inst = cargar_desde_csv(args.csv_dir)
    else:
        inst = cargar_desde_sql(str(PROJECT_ROOT / "asignacion_mec.sql"))
    est, doc, cls = inst["estudiantes"], inst["docentes"], inst["clases"]
    if est.empty or doc.empty or cls.empty:
        logger.error("❌ No se pudo cargar los datos necesarios.")
        return
    logger.info(f"✅ Datos: {len(est)} estudiantes, {len(doc)} docentes, {len(cls)} clases")

    # ---------- 2) Asignación actual del MEC y problema ----------
    # x_mec sirve para f6 (x_base) y para comparar al final. Se necesita un
    # problema auxiliar porque x_desde_asignacion usa sus índices y candidatos.
    # Sus matrices de distancia U y V se reutilizan en el problema definitivo
    # para no recalcularlas (dependen sólo de los datos, no de los objetivos).
    tiene_mec = inst.get("asignacion") is not None and len(inst["asignacion"])
    x_mec, U, V = None, None, None
    if tiene_mec:
        base = IntegratedProblem(est, doc, cls, n_obj=3)
        x_mec, _ = x_desde_asignacion(base, inst["asignacion"])
        U, V = base.U, base.V
    problem = IntegratedProblem(est, doc, cls, n_obj=args.n_obj, objetivos=args.objetivos,
                                cap_min=args.cap_min, f4_modo=args.f4_modo, x_base=x_mec,
                                U=U, V=V)

    # ---------- 3) Configuración de BD (sólo si los datos vienen de la BD) ----------
    # Se usa la configuración ÚNICA de database.py (la misma con la que se leyeron
    # los datos). Import diferido: con --fuente csv/ejemplo no hace falta SQLAlchemy.
    db_config = None
    if args.fuente == "db" and not args.no_guardar:
        from database import DB_CONFIG
        db_config = dict(DB_CONFIG)

    # ---------- 4) Optimización (+ persistencia si db_config) ----------
    result = run_integrated_optimization(problem, pop_size=args.pop, n_gen=args.gen,
                                         algoritmo=args.algoritmo, seed=args.seed,
                                         db_config=db_config, metodo=args.metodo_decision,
                                         guardar_pareto=args.guardar_pareto,
                                         max_archivo=args.max_archivo)
    # ---------- 5) Reporte del Conjunto Pareto ----------
    # Se listan sólo las soluciones elegidas por algún método de decisión
    # (compromiso, rodilla, pseudo-pesos), con valores en signo natural.
    ps = result.pareto_set
    n_ps = ps["n_soluciones"] if ps else 0
    print(f"\n🧭 Conjunto Pareto: {n_ps} soluciones no dominadas factibles "
          f"({args.algoritmo.upper()}, {problem.n_obj} objetivos)")
    if ps:
        for i, metodos in sorted(etiquetar_decisiones(ps["F"]).items()):
            vals = problem.a_reportable(ps["F"][i])[0]
            print(f"   sol {i:>4} [{metodos}]: " + ", ".join(f"{o}={v:.3f}" for o, v in zip(problem.objetivos, vals)))
        if args.exportar_pareto:
            meta = exportar_pareto(problem, ps, args.exportar_pareto, f"{args.algoritmo}_seed{args.seed}")
            print(f"   Exportado en {args.exportar_pareto}/{meta['run_id']}")

    # ---------- 6) Solución elegida por --metodo-decision ----------
    _, best_x, best_F = select_best_individual(result, args.metodo_decision)
    _, _, det = problem.evaluar(best_x, detalle=True)
    print(f"\n📊 Solución elegida ({args.metodo_decision}):")
    for o, s, e in zip(problem.objetivos, problem.sentidos, problem.etiquetas):
        print(f"   ➤ {o:<20} [{s}] {det[o]:>12.4f}   {e}")

    # ---------- 7) Comparación con la asignación actual (MEC) ----------
    if tiene_mec:
        Fm, Gm, detm = problem.evaluar(x_mec, detalle=True)
        print("\n📋 Asignación actual (MEC):")
        for o in problem.objetivos:
            print(f"   ➤ {o:<20} {detm[o]:>12.4f}")
        print(f"   Violaciones MEC: {dict(zip(('g1','g2','g3','g4','g5'), map(int, Gm)))}")
        if ps:
            n_dom = sum(domina(f, Fm) for f in ps["F"])
            print(f"   Soluciones del Conjunto Pareto que dominan al MEC: {n_dom} de {n_ps}")
            if args.metodo_decision.startswith("domina_mec"):
                print("   La solución elegida " + ("DOMINA al MEC (mejor o igual en todos los objetivos)."
                      if domina(best_F, Fm) else "NO domina al MEC: ninguna lo hacía y se usó el método "
                      "sobre todo el frente (ver advertencia)."))


if __name__ == "__main__":
    main()
