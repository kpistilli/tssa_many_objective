# ================================================================
# integrated_optimization.py
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
#   Klaus Pistilli — Investigador en formación
#   Universidad Nacional de Asunción — Facultad Politécnica (FP-UNA)
#   Asunción, Paraguay — 2026
#
# Tesis de Maestría en TIC - TSSA many-objective
#
# - Fábrica de algoritmos: NSGA-II, NSGA-III, U-NSGA-III, C-TAEA, RVEA, SPEA2
#   (y AGE-MOEA si numba está instalado), con operadores específicos del TSSA
#   o los operadores "SBX + redondeo" de la versión anterior. La heurística
#   GREEDY (operators.solucion_greedy) se usa como referencia no evolutiva en
#   experiments.py, junto a estos algoritmos, para el estudio comparativo.
# - run_integrated_optimization(): API compatible con la v1.3 (la usan
#   integrated_app.py y el viewer Streamlit), con `algoritmo` configurable.
# - El resultado de una corrida es el CONJUNTO PARETO (result.pareto_set),
#   acumulado con un archivo externo en todas las generaciones (pareto.py).
# - select_best_individual(): decisión a posteriori (compromiso, ASF, rodilla,
#   pseudo-pesos, lexicográfico); tolera frentes vacíos (sin factibles).
# - DatabaseManager: persiste el Conjunto Pareto completo (opt_run,
#   pareto_solucion, asignacion_pareto) y la solución elegida en
#   asignacion_mec_opt (compatibilidad con el viewer), con sección derivada.
# ================================================================

import logging
from typing import Any, Dict, Optional

import numpy as np
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.callback import Callback
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from operators import TSSAMutation, TSSARepair, TSSASampling, semillas_estabilidad, solucion_greedy
from pareto import (ParetoArchive, escribir_asignacion_mec_opt, extraer_pareto_set,
                    guardar_pareto_db, ref_mec, seleccionar_solucion)

logger = logging.getLogger(__name__)

# Estudio comparativo: NSGA-II, NSGA-III, U-NSGA-III, C-TAEA, RVEA, SPEA2 (y
# opcionalmente AGE-MOEA si numba está instalado) contra la heurística GREEDY
# (esta última no es un algoritmo evolutivo: se corre una sola vez por
# escenario en experiments.py como línea de base, no con `--runs` corridas).
ALGORITMOS = ("nsga2", "nsga3", "unsga3", "ctaea", "rvea", "spea2", "agemoea", "greedy")
ALGORITMOS_EVOLUTIVOS = tuple(a for a in ALGORITMOS if a != "greedy")


# ------------------------------------------------------------------
# Construcción de algoritmos
# ------------------------------------------------------------------
def direcciones_referencia(n_obj, n_puntos, seed=1):
    """
    Direcciones de referencia para NSGA-III. Se usa el método 'energy' para
    obtener EXACTAMENTE n_puntos (= tamaño de población), lo que permite una
    comparación justa con NSGA-II a igual población (100, como en CACIC 2024).
    Das-Dennis con M=4 solo da 84 (p=6) o 120 (p=7) puntos.

    Es un cálculo de optimización propio (≈1–2 s para 100 puntos) que se hace
    una vez por algoritmo construido; `seed` lo hace determinista.
    """
    return get_reference_directions("energy", n_obj, n_puntos, seed=seed)


def construir_operadores(problem, operadores="tssa", reparar=True, semilla_greedy=True,
                         semillas_extra=None, cruce="ux", tasa_mut_est=0.005,
                         tasa_mut_doc=0.02):
    """
    Arma el diccionario de operadores (kwargs) que se pasa a cualquier algoritmo de pymoo.

    Args:
        operadores: "tssa" = operadores propios del problema (muestreo por grado,
            cruce uniforme/2 puntos y mutación categórica) o "sbx" = SBX + mutación
            polinomial + redondeo (enfoque de la v1.3, sólo para comparación).
        reparar: agrega TSSARepair (R1–R5) después de cruce y mutación.
        semilla_greedy: inyecta la solución greedy en la población inicial.
        semillas_extra: otros vectores x a inyectar (p. ej. la asignación MEC).
        cruce: "ux" (uniforme) o "2p" (dos puntos), ambos con probabilidad 0.9.
        tasa_mut_est, tasa_mut_doc: tasas de TSSAMutation.

    Returns:
        dict con sampling, crossover, mutation, (repair) y eliminate_duplicates.
    """
    if operadores == "tssa":
        semillas = []
        if semilla_greedy:
            semillas.append(solucion_greedy(problem))
        semillas += list(semillas_extra or [])
        ops = dict(
            sampling=TSSASampling(semillas=semillas),
            crossover=(UniformCrossover(prob=0.9) if cruce == "ux" else TwoPointCrossover(prob=0.9)),
            mutation=TSSAMutation(tasa_est=tasa_mut_est, tasa_doc=tasa_mut_doc),
        )
    elif operadores == "sbx":
        # operadores reales + redondeo (enfoque de la versión 1.3 / preprint)
        ops = dict(
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
            mutation=PM(eta=20, vtype=float, repair=RoundingRepair()),
        )
    else:
        raise ValueError("operadores debe ser 'tssa' o 'sbx'")
    if reparar:
        ops["repair"] = TSSARepair()
    # pymoo descarta hijos idénticos a otros hijos o a la población actual
    ops["eliminate_duplicates"] = True
    return ops


def construir_algoritmo(nombre, problem, pop_size=100, **kw_ops):
    """
    Fábrica de algoritmos evolutivos de pymoo con los operadores del TSSA.

    Args:
        nombre: nsga2 | nsga3 | unsga3 | ctaea | rvea | spea2 | agemoea.
        problem: IntegratedProblem (define n_obj para las direcciones de referencia).
        pop_size: tamaño de población (y nº de direcciones de referencia en los
            algoritmos que las usan, para comparar a igual población).
        **kw_ops: se pasan a construir_operadores().

    "greedy" no es un algoritmo evolutivo: se rechaza con un mensaje explicativo.
    """
    nombre = nombre.lower()
    ops = construir_operadores(problem, **kw_ops)
    if nombre == "nsga2":
        return NSGA2(pop_size=pop_size, **ops)
    if nombre == "nsga3":
        return NSGA3(ref_dirs=direcciones_referencia(problem.n_obj, pop_size), pop_size=pop_size, **ops)
    if nombre == "unsga3":
        return UNSGA3(ref_dirs=direcciones_referencia(problem.n_obj, pop_size), pop_size=pop_size, **ops)
    if nombre == "ctaea":
        # CTAEA fija pop_size = len(ref_dirs); no admite pop_size como parámetro aparte.
        ref_dirs = direcciones_referencia(problem.n_obj, pop_size)
        ops.pop("eliminate_duplicates", None)  # CTAEA no acepta este kwarg
        return CTAEA(ref_dirs=ref_dirs, **ops)
    if nombre == "rvea":
        return RVEA(ref_dirs=direcciones_referencia(problem.n_obj, pop_size), pop_size=pop_size, **ops)
    if nombre == "spea2":
        return SPEA2(pop_size=pop_size, **ops)
    if nombre == "agemoea":
        from pymoo.algorithms.moo.age import AGEMOEA  # requiere numba
        return AGEMOEA(pop_size=pop_size, **ops)
    if nombre == "greedy":
        raise ValueError("'greedy' es una heurística de referencia, no un algoritmo evolutivo; "
                         "se ejecuta con operators.solucion_greedy(), no con construir_algoritmo().")
    raise ValueError(f"Algoritmo desconocido '{nombre}'. Opciones: {ALGORITMOS}")


# ------------------------------------------------------------------
# Selección de un individuo representativo
# ------------------------------------------------------------------
def _extract_FX(result):
    """
    (F, X) del Conjunto Pareto de la corrida: archivo externo (si se usó) +
    población final, factibles y no dominados. Si no hay factibles, la
    solución de menor violación.

    Orden de búsqueda: result.pareto_set (lo arma run_integrated_optimization)
    -> se reconstruye con extraer_pareto_set() -> individuo de menor CV.
    No se usa result.F: en NSGA-III (= algorithm.opt) contiene sólo las
    soluciones más cercanas a cada dirección de referencia y subestima el frente.
    Siempre devuelve matrices 2D (una fila por solución).
    """
    ps = getattr(result, "pareto_set", None)
    if ps is None:
        ps = extraer_pareto_set(getattr(result, "problem", None), result,
                                getattr(result, "archivo", None))
    if ps is not None and ps["n_soluciones"]:
        return np.atleast_2d(ps["F"]), np.atleast_2d(ps["X"])
    pop = getattr(result, "pop", None)
    if pop is None:
        raise ValueError("Resultado vacío: no hay F/X ni población.")
    CV = pop.get("CV").ravel()
    k = int(np.argmin(CV))
    logger.warning(f"⚠️ Sin soluciones factibles; se usa la de menor violación (CV={CV[k]:.2f}).")
    return np.atleast_2d(pop.get("F")[k]), np.atleast_2d(pop.get("X")[k])


def indice_compromiso(F):
    """Índice de la solución con menor distancia de Tchebycheff al ideal (F normalizado)."""
    return seleccionar_solucion(F, "compromiso")


def select_best_individual(result, metodo="compromiso", pesos=None):
    """
    Selecciona UNA solución del Conjunto Pareto (decisión a posteriori).
    metodo: compromiso | asf | rodilla | pseudo_pesos | lexicografico | min_fk |
    domina_mec[:<método>] (ver pareto.seleccionar_solucion; domina_mec usa la
    asignación del MEC del problema, x_base, como referencia). El frente
    completo queda en result.pareto_set.
    """
    F, X = _extract_FX(result)
    problem = getattr(result, "problem", None)
    ref = ref_mec(problem) if (problem is not None and metodo.startswith("domina_mec")) else None
    idx = seleccionar_solucion(F, metodo, pesos, ref=ref)
    return idx, X[idx], F[idx]


# ------------------------------------------------------------------
# Persistencia
# ------------------------------------------------------------------
class DatabaseManager:
    """
    Persistencia en PostgreSQL (no toca la baseline asignacion_mec):
      - save_pareto_set(): Conjunto Pareto COMPLETO (opt_run, pareto_solucion,
        asignacion_pareto) + asignacion_mec_opt con la solución elegida.
      - save_asignaciones(): solo asignacion_mec_opt (comportamiento v1.3).
    """

    def __init__(self, db_config: Dict[str, Any]):
        """db_config: kwargs de psycopg2.connect (user, password, host, port, database)."""
        self.db_config = db_config
        self.conn = None

    def connect(self) -> bool:
        """Abre la conexión; devuelve False (y registra el error) si no se pudo."""
        import psycopg2  # import diferido: sólo hace falta si se persiste en BD
        try:
            self.conn = psycopg2.connect(**self.db_config)
            logger.info("✅ Conexión a la base de datos establecida")
            return True
        except psycopg2.Error as e:
            logger.error(f"❌ Error al conectar a la base de datos: {e}")
            return False

    def disconnect(self) -> None:
        """Cierra la conexión si está abierta (seguro de llamar más de una vez)."""
        if self.conn and not self.conn.closed:
            self.conn.close()

    def save_pareto_set(self, problem, result, metodo="compromiso", guardar_asignaciones="seleccion",
                        **meta) -> Optional[str]:
        """
        Guarda el Conjunto Pareto de `result` con pareto.guardar_pareto_db().

        Args:
            metodo: método de decisión para la solución que va a asignacion_mec_opt.
            guardar_asignaciones: ninguna | seleccion | todas (filas detalladas
                en asignacion_pareto).
            **meta: run_id, algoritmo, pop_size, n_gen, seed (tabla opt_run).

        Returns:
            run_id de la corrida guardada. Ante error hace rollback y relanza.
        """
        F, X = _extract_FX(result)
        ps = {"X": np.rint(X).astype(np.int32), "F": F, "n_soluciones": len(F),
              "ideal": F.min(0), "nadir": F.max(0), "obj_names": list(problem.objetivos)}
        try:
            return guardar_pareto_db(self.conn, problem, ps, metodo=metodo,
                                     guardar_asignaciones=guardar_asignaciones, **meta)
        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ Error al guardar el Conjunto Pareto: {e}", exc_info=True)
            raise

    def save_asignaciones(self, problem, result, metodo="compromiso") -> None:
        """
        Modo compatible con la v1.3: escribe SÓLO asignacion_mec_opt con la
        solución elegida por `metodo`. La tabla se crea con la misma estructura
        que asignacion_mec y se vacía (TRUNCATE) antes de cargarla con COPY.
        La sección se deriva por clase (no se fija en "A").
        """
        try:
            _, best_x, _ = select_best_individual(result, metodo)
            with self.conn.cursor() as cur:
                escribir_asignacion_mec_opt(cur, problem, best_x)
            self.conn.commit()
            logger.info("✅ Asignaciones guardadas en asignacion_mec_opt")
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            logger.error(f"❌ Error al guardar asignaciones: {e}", exc_info=True)
            raise


# ------------------------------------------------------------------
# Orquestación de una corrida
# ------------------------------------------------------------------
class _Callbacks(Callback):
    """Permite usar varios callbacks a la vez (p. ej. archivo Pareto + convergencia)."""

    def __init__(self, *cbs):
        super().__init__()
        self.cbs = [c for c in cbs if c is not None]

    def notify(self, algorithm):
        """pymoo lo llama al final de cada generación; se reenvía a cada callback."""
        for c in self.cbs:
            c(algorithm)


def run_integrated_optimization(problem, pop_size: int = 100, n_gen: int = 100, n_procs: int = 1,
                                db_config: Optional[Dict[str, Any]] = None, run_id: str = None,
                                metadata: Dict[str, Any] = None, algoritmo: str = "nsga3",
                                seed: int = 42, verbose: bool = True, callback=None,
                                save_history: bool = False, archivar: bool = True,
                                max_archivo: int = 500, metodo: str = "compromiso",
                                guardar_pareto: str = "seleccion", **kw_ops):
    """
    Ejecuta el algoritmo elegido sobre `problem` y (opcionalmente) persiste.
    Firma compatible con la v1.3: (problem, pop_size, n_gen, n_procs, db_config, ...).

    - archivar=True: archivo externo no dominado de todas las generaciones.
    - El Conjunto Pareto completo queda en result.pareto_set (dict X, F, ...).
    - db_config: guarda el Conjunto Pareto (guardar_pareto = ninguna | seleccion |
      todas, para las asignaciones detalladas) y asignacion_mec_opt con la
      solución elegida por `metodo`. guardar_pareto="solo_compat" escribe
      únicamente asignacion_mec_opt (v1.3).
    - n_procs: se acepta por compatibilidad con la v1.3, pero hoy NO se usa
      (la evaluación corre en un solo proceso). Para paralelizar varias
      corridas usar experiments.py --procesos.
    - seed: semilla de la corrida. Fija el generador de pymoo y TAMBIÉN np.random,
      que usan los operadores propios (muestreo, mutación y reparación): desde
      pymoo 0.6.2, minimize(seed=...) ya no siembra np.random.
    - callback: callback adicional de pymoo (p. ej. registro de convergencia).
    - **kw_ops: opciones de operadores (ver construir_operadores).

    Returns:
        Result de pymoo con dos atributos extra: `archivo` (ParetoArchive o None)
        y `pareto_set` (dict X, F, n_soluciones, ideal, nadir, obj_names, o None).
    """
    if n_procs and n_procs > 1:
        logger.info(f"ℹ️ n_procs={n_procs} se ignora: la corrida usa un solo proceso "
                    "(para corridas en paralelo usar experiments.py --procesos).")
    # 1) Algoritmo + callbacks (archivo externo y, opcionalmente, uno del usuario)
    # Con f6 se inyectan el MEC y soluciones intermedias (ver semillas_estabilidad)
    if "f6_estabilidad" in getattr(problem, "objetivos", ()) and "semillas_extra" not in kw_ops:
        kw_ops["semillas_extra"] = semillas_estabilidad(problem)
    algorithm = construir_algoritmo(algoritmo, problem, pop_size=pop_size, **kw_ops)
    archivo = ParetoArchive(max_size=max_archivo) if archivar else None
    cb = _Callbacks(archivo, callback)
    extra = {"callback": cb} if cb.cbs else {}
    # 2) Búsqueda evolutiva: criterio de parada = n_gen generaciones
    np.random.seed(seed)   # operadores propios (ver docstring): reproducibilidad exacta
    result = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=verbose,
                      save_history=save_history, **extra)
    # 3) Conjunto Pareto final = archivo externo ∪ población final (factibles, no dominados)
    result.archivo = archivo
    result.pareto_set = extraer_pareto_set(problem, result, archivo)
    # 4) Persistencia opcional en PostgreSQL
    if db_config:
        db = DatabaseManager(db_config)
        if db.connect():
            try:
                if guardar_pareto == "solo_compat":
                    db.save_asignaciones(problem, result, metodo)
                else:
                    db.save_pareto_set(problem, result, metodo=metodo,
                                       guardar_asignaciones=guardar_pareto, run_id=run_id,
                                       algoritmo=algoritmo, pop_size=pop_size, n_gen=n_gen, seed=seed)
            finally:
                db.disconnect()
    return result
