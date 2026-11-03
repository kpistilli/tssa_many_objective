# ================================================================
# experiments.py
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
# Protocolo experimental reproducible (inspirado en CACIC 2024 y ampliado):
#   * Escenarios: instancia completa y/o por grado.
#   * Algoritmos: NSGA-II, NSGA-III, U-NSGA-III, C-TAEA, RVEA, SPEA2 (y AGE-MOEA)
#     + la heurística greedy como línea de base. La sintaxis
#     "algoritmo@k" optimiza con k objetivos y luego RE-EVALÚA su frente en el
#     espacio de n_obj objetivos del experimento (ej. "nsga2@3" = preprint).
#   * R corridas independientes por algoritmo con semillas registradas
#     (semilla = --seed-base + nº de corrida; greedy se corre una sola vez).
#   * Métricas: HV, IGD+ y GD+ (normalización global), spacing, #no dominadas,
#     tasa de factibles y tiempo.
#   * Estadística: Kruskal–Wallis global + Mann–Whitney U por pares con
#     corrección Holm/Bonferroni + A12 de Vargha–Delaney.
#   * Referencias: asignación actual del MEC (si existe) y heurística greedy:
#     dominancia de Pareto (Tabla 4 CACIC) y % de mejora por objetivo (Tabla 5).
#   * Figuras: boxplots, convergencia de HV, coordenadas paralelas, matriz de dispersión.
#   * (v2.1, integrado desde run_experiments.py / pareto_export.py):
#     - el resultado de cada corrida es su CONJUNTO PARETO, acumulado con un
#       archivo externo no dominado en todas las generaciones;
#     - PF* por corrida en CSV (con la solución elegida por compromiso, rodilla y
#       pseudo-pesos) y, con --guardar-pareto conjunto|asignaciones, también P*;
#     - GD+, Kruskal–Wallis global y corrección Holm/Bonferroni en los pares;
#     - tasa de factibles de la población final (no del frente);
#     - objetivos configurables (--objetivos, incl. f6_estabilidad) y --f4-modo.
#
# Ejemplos:
#   python experiments.py --fuente ejemplo --runs 5 --gen 50
#   python experiments.py --fuente db --algoritmos nsga2@3 nsga2 nsga3 --runs 30 --gen 100
#   python experiments.py --fuente csv --csv-dir datos_cde --escenario por-grado --procesos 4
#   python experiments.py --fuente sintetica --sint-estudiantes 15763 --sint-establecimientos 90
#
# Estructura de la salida (--salida, por defecto "resultados/"):
#   config.json                       parámetros, versiones y fecha de la ejecución
#   <escenario>/referencias.csv        greedy y MEC evaluados (con detalle de mapeo)
#   <escenario>/corridas.csv           una fila por corrida con todos los indicadores
#   <escenario>/convergencia.csv       HV cada --conv-cada generaciones
#   <escenario>/resumen.csv            media, desvío, mediana, mín. y máx. por algoritmo
#   <escenario>/kruskal_wallis.csv, estadistica.csv   pruebas estadísticas
#   <escenario>/frente_referencia.csv, frente_union_*.csv, mejora_vs_referencias.csv
#   <escenario>/frentes/               PF* de cada corrida (NPZ + CSV)
#   <escenario>/conjuntos_pareto/      P* de cada corrida (con --guardar-pareto)
#   <escenario>/informe.txt y figuras .png
# ================================================================

import argparse
import json
import logging
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from pymoo.core.callback import Callback

import instancia as inst_mod
from integrated_optimization import _Callbacks, construir_algoritmo, indice_compromiso
from integrated_problem import CONJUNTOS_OBJETIVOS, OBJETIVOS, IntegratedProblem, NOMBRES_RESTRICCIONES
from metrics import (Indicadores, Normalizador, dominancia_vs_ref, estadistica_completa,
                     no_dominados, tabla_mejora)
from operators import semillas_estabilidad, solucion_greedy
from pareto import ParetoArchive, etiquetar_decisiones, exportar_pareto, extraer_pareto_set

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("experimentos")

# Silencia el aviso de módulos no compilados de pymoo. OJO: sin compilar,
# el ordenamiento no dominado de pymoo 0.6.1 es bastante más lento; conviene verificar
# con `from pymoo.util.function_loader import is_compiled; is_compiled()` (0.6.1) o
# `from pymoo.functions import is_compiled` (0.6.2). Desde pymoo 0.6.2 el
# hipervolumen y el ordenamiento no dominado usan moocore (compilado).
try:
    from pymoo.config import Config
    Config.warnings["not_compiled"] = False
except Exception:
    pass


# ------------------------------------------------------------------
# Argumentos
# ------------------------------------------------------------------
def parse_args(argv=None):
    """Define y lee las opciones de línea de comandos, agrupadas por tema."""
    ap = argparse.ArgumentParser(description="Experimentos TSSA many-objective")
    d = ap.add_argument_group("datos")
    d.add_argument("--fuente", choices=["db", "csv", "ejemplo", "sintetica"], default="ejemplo")
    d.add_argument("--csv-dir", default="datos")
    d.add_argument("--sql", default="asignacion_mec.sql", help="script SQL para --fuente ejemplo")
    d.add_argument("--exportar-csv", default=None, help="guarda un snapshot CSV de la instancia cargada")
    d.add_argument("--escenario", choices=["completo", "por-grado", "ambos"], default="completo")
    d.add_argument("--sint-estudiantes", type=int, default=2000)
    d.add_argument("--sint-establecimientos", type=int, default=20)
    d.add_argument("--sint-seed", type=int, default=0)

    m = ap.add_argument_group("modelo")
    m.add_argument("--n-obj", type=int, default=4, choices=[3, 4, 5])
    m.add_argument("--objetivos", nargs="+", default=None, choices=list(OBJETIVOS),
                   help="lista explícita de objetivos (reemplaza a --n-obj), p. ej. "
                        "f1_distancia f2_balance f3_colocalizacion f4_distancia_max f6_estabilidad")
    m.add_argument("--f4-modo", choices=["suma", "max"], default="suma",
                   help="suma = max_est + max_doc (defecto); max = max(max_est, max_doc) como la v2")
    m.add_argument("--cap-min", type=int, default=0, help="0 = sin mínimo (v1.3); 25 = preprint; 10 = CACIC")
    m.add_argument("--f2", choices=["var", "std"], default="var")

    a = ap.add_argument_group("algoritmos")
    a.add_argument("--algoritmos", nargs="+",
                   default=["nsga2", "nsga3", "unsga3", "ctaea", "rvea", "spea2", "greedy"],
                   help="nombres (nsga2, nsga3, unsga3, ctaea, rvea, spea2, agemoea, greedy); "
                        "sufijo @k = optimizar con k objetivos (no aplica a greedy)")
    a.add_argument("--runs", type=int, default=30)
    a.add_argument("--pop", type=int, default=100)
    a.add_argument("--gen", type=int, default=100)
    a.add_argument("--seed-base", type=int, default=1000)
    a.add_argument("--operadores", choices=["tssa", "sbx"], default="tssa")
    a.add_argument("--cruce", choices=["ux", "2p"], default="ux")
    a.add_argument("--sin-reparacion", action="store_true")
    a.add_argument("--sin-semilla-greedy", action="store_true")
    a.add_argument("--tasa-mut-est", type=float, default=0.005)
    a.add_argument("--tasa-mut-doc", type=float, default=0.02)
    a.add_argument("--max-archivo", type=int, default=500,
                   help="tamaño máximo del archivo externo no dominado por corrida")

    s = ap.add_argument_group("salida")
    s.add_argument("--salida", default="resultados")
    s.add_argument("--conv-cada", type=int, default=5, help="registrar convergencia cada k generaciones")
    s.add_argument("--procesos", type=int, default=1)
    s.add_argument("--sin-figuras", action="store_true")
    s.add_argument("--guardar-pareto", choices=["frente", "conjunto", "asignaciones"], default="frente",
                   help="frente = PF* por corrida (CSV); conjunto = además P* (X) en NPZ; "
                        "asignaciones = además las asignaciones de las soluciones seleccionadas")
    s.add_argument("--correccion", choices=["holm", "bonferroni"], default="holm",
                   help="corrección por comparaciones múltiples en Mann–Whitney")
    return ap.parse_args(argv)


def parse_spec(spec, n_obj_exp):
    """
    "nsga2@3" -> ("nsga2", 3); "nsga3" -> ("nsga3", n_obj_exp).
    El número tras '@' es la cantidad de objetivos con la que OPTIMIZA el algoritmo.
    """
    if "@" in spec:
        nombre, k = spec.split("@")
        return nombre.lower(), int(k)
    return spec.lower(), n_obj_exp


# ------------------------------------------------------------------
# Datos
# ------------------------------------------------------------------
def cargar_instancia(args):
    """Carga la instancia según --fuente y, si se pide, guarda un snapshot CSV."""
    if args.fuente == "db":
        inst = inst_mod.cargar_desde_db()
    elif args.fuente == "csv":
        inst = inst_mod.cargar_desde_csv(args.csv_dir)
    elif args.fuente == "ejemplo":
        inst = inst_mod.cargar_desde_sql(args.sql)
    else:
        inst = inst_mod.generar_instancia_sintetica(n_establecimientos=args.sint_establecimientos,
                                                    n_estudiantes=args.sint_estudiantes,
                                                    seed=args.sint_seed)
    if args.exportar_csv:
        inst_mod.exportar_csv(inst, args.exportar_csv)
    return inst


def escenarios(inst, args):
    """Lista de (nombre, instancia) a ejecutar: "completo" y/o uno por grado."""
    out = []
    if args.escenario in ("completo", "ambos"):
        out.append(("completo", inst))
    if args.escenario in ("por-grado", "ambos"):
        for g in inst_mod.grados_de(inst):
            out.append((f"grado_{g}", inst_mod.subinstancia_por_grado(inst, g)))
    return out


def objetivos_experimento(args):
    """Objetivos del experimento: --objetivos si se indicó; si no, el conjunto de --n-obj."""
    return tuple(args.objetivos) if args.objetivos else CONJUNTOS_OBJETIVOS[args.n_obj]


def objetivos_para(args, k):
    """@k: los primeros k objetivos del experimento (o el conjunto estándar de k)."""
    obj = objetivos_experimento(args)
    return obj[:k] if k <= len(obj) else CONJUNTOS_OBJETIVOS[k]


def construir_problema(inst, args, n_obj, U=None, V=None, x_base=None):
    """
    Crea el IntegratedProblem con n_obj objetivos y las opciones del experimento.

    U, V y x_base se pueden pasar ya calculadas para no repetir las matrices de
    distancia. Si hace falta f6 y no hay x_base, se construye desde la tabla
    asignacion_mec con un problema auxiliar (y se reutilizan sus U y V).
    """
    objetivos = objetivos_para(args, n_obj)
    kw = dict(objetivos=objetivos, cap_min=args.cap_min, f2_metrica=args.f2,
              f4_modo=getattr(args, "f4_modo", "suma"))
    if "f6_estabilidad" in objetivos and x_base is None:
        if inst.get("asignacion") is None or not len(inst["asignacion"]):
            raise ValueError("f6_estabilidad requiere la tabla asignacion_mec en la instancia")
        base = IntegratedProblem(inst["estudiantes"], inst["docentes"], inst["clases"], n_obj=3,
                                 U=U, V=V)
        x_base, _ = inst_mod.x_desde_asignacion(base, inst["asignacion"])
        U, V = base.U, base.V
    return IntegratedProblem(inst["estudiantes"], inst["docentes"], inst["clases"],
                             U=U, V=V, x_base=x_base, **kw)


# ------------------------------------------------------------------
# Convergencia
# ------------------------------------------------------------------
class RegistroConvergencia(Callback):
    """
    Guarda, cada k generaciones, el frente factible en el espacio de EVALUACIÓN.

    Con `archivo` (caso normal) registra el frente acumulado del archivo externo,
    que es monótono: el HV de la curva nunca baja. Sin archivo (caso "@k")
    toma los no dominados factibles de la población y, si el algoritmo optimiza
    en otro espacio (mismo_espacio=False), los re-evalúa en el del experimento.
    Los registros (gen, F) se convierten luego en HV en correr_escenario().
    """

    def __init__(self, problema_eval, cada=5, mismo_espacio=True, archivo=None):
        super().__init__()
        self.pe, self.cada, self.mismo, self.archivo = problema_eval, cada, mismo_espacio, archivo
        self.registros = []

    def notify(self, algorithm):
        g = algorithm.n_gen
        if not (g == 1 or g % self.cada == 0):
            return
        if self.archivo is not None:        # frente acumulado (monótono) del archivo externo
            F = self.archivo.F if self.archivo.F is not None else np.empty((0, self.pe.n_obj))
            self.registros.append((g, np.array(F, copy=True)))
            return
        pop = algorithm.pop
        CV = pop.get("CV").ravel()
        fact = CV <= 0
        if not fact.any():
            self.registros.append((g, np.empty((0, self.pe.n_obj))))
            return
        if self.mismo:
            F = pop.get("F")[fact]
        else:
            F = np.array([self.pe.evaluar(x)[0] for x in pop.get("X")[fact]])
        F = no_dominados(F)
        self.registros.append((g, np.array(F, copy=True)))


# ------------------------------------------------------------------
# Una corrida (se ejecuta en un proceso aparte si --procesos > 1)
# ------------------------------------------------------------------
# Contexto por proceso: instancia, argumentos y problemas ya construidos
# (uno por cantidad de objetivos). Se llena una vez por proceso en _init_worker
# para no reconstruir U, V y los candidatos en cada corrida.
_CTX = {}


def _init_worker(inst, args_dict):
    """Inicializa _CTX (se usa como `initializer` del ProcessPoolExecutor y en modo secuencial)."""
    import argparse as _ap
    args = _ap.Namespace(**args_dict)
    args.n_obj = len(objetivos_experimento(args))
    base = construir_problema(inst, args, args.n_obj)
    _CTX.update(inst=inst, args=args, problemas={args.n_obj: base})


def _problema(k):
    """Problema con k objetivos, creado a demanda y cacheado en _CTX (comparte U, V y x_base)."""
    P = _CTX["problemas"]
    if k not in P:
        base = P[_CTX["args"].n_obj]
        P[k] = construir_problema(_CTX["inst"], _CTX["args"], k, U=base.U, V=base.V, x_base=base.x_base)
    return P[k]


def ejecutar_corrida(spec, run, seed):
    """
    Ejecuta UNA corrida de un algoritmo (o evalúa greedy) y devuelve un dict
    serializable con: F del Conjunto Pareto (espacio del experimento), X
    (opcional), X de la solución de compromiso, tasa de factibles de la
    población final, nº de no dominados de la última población, tiempo y
    registros de convergencia. Usa el contexto _CTX del proceso.
    """
    from pymoo.optimize import minimize
    args = _CTX["args"]
    nombre, k = parse_spec(spec, args.n_obj)
    p_eval = _problema(args.n_obj)

    if nombre == "greedy":
        # Heurística determinista (Sección 6.2.2 del preprint): no es una
        # búsqueda evolutiva, así que se evalúa una sola vez (no depende de
        # la semilla) y se reporta como una corrida más para que participe
        # en las mismas tablas/figuras que los algoritmos evolutivos.
        # Se construye y evalúa UNA vez; el tiempo reportado incluye ambas cosas.
        t0 = time.time()
        xg = solucion_greedy(p_eval)
        F, G = p_eval.evaluar(xg)
        tiempo = time.time() - t0
        # Si el greedy viola alguna restricción (p. ej. con --cap-min, que el
        # greedy no contempla), NO se reporta como frente: un punto infactible
        # distorsionaría la normalización, el frente de referencia, HV e IGD+.
        # Sus valores quedan igual en referencias.csv como línea de base.
        factible = bool(np.all(G <= 0))
        if not factible:
            log.warning(f"⚠️ greedy infactible en este escenario (G = {G.tolist()}); "
                        "se excluye de los indicadores y del frente de referencia")
        F_rep = F[None, :] if factible else np.empty((0, len(F)))
        return {"spec": spec, "run": run, "seed": seed, "tiempo_s": tiempo,
                "F": F_rep, "X": xg[None, :].astype(np.int32) if factible else None,
                "X_comp": xg.astype(np.int32) if factible else None,
                "tasa_factibles": float(factible),
                "n_nd_ult_poblacion": int(factible), "conv": []}

    # Problema en el que OPTIMIZA el algoritmo (k objetivos) y problema en el que
    # se EVALÚA para comparar (n_obj del experimento); coinciden salvo con "@k".
    p_opt = _problema(k)
    # Con f6 (estabilidad) se inyectan el MEC y soluciones intermedias MEC/greedy
    # para que la búsqueda cubra todo el eje de f6 y no sólo la zona del greedy.
    extra = (semillas_estabilidad(p_opt) if "f6_estabilidad" in p_opt.objetivos else [])
    alg = construir_algoritmo(nombre, p_opt, pop_size=args.pop, operadores=args.operadores,
                              reparar=not args.sin_reparacion,
                              semilla_greedy=not args.sin_semilla_greedy, cruce=args.cruce,
                              tasa_mut_est=args.tasa_mut_est, tasa_mut_doc=args.tasa_mut_doc,
                              semillas_extra=extra)
    # Archivo externo no dominado de TODAS las generaciones (en el espacio en que
    # optimiza el algoritmo). Para "@k" el conjunto final se re-evalúa después.
    archivo = ParetoArchive(max_size=args.max_archivo)
    mismo = (k == args.n_obj)
    conv = RegistroConvergencia(p_eval, args.conv_cada, mismo_espacio=mismo,
                                archivo=archivo if mismo else None)
    t0 = time.time()
    # pymoo >= 0.6.2 no siembra np.random con `seed`; los operadores propios
    # (muestreo, mutación, reparación) lo usan, así que se fija aquí para que
    # cada corrida sea exactamente reproducible a partir de su semilla.
    np.random.seed(seed)
    res = minimize(p_opt, alg, ("n_gen", args.gen), seed=seed, verbose=False,
                   callback=_Callbacks(archivo, conv))
    tiempo = time.time() - t0

    # tasa de factibles de la POBLACIÓN final (no del frente, que siempre es factible)
    CV = res.pop.get("CV").ravel() if res.pop is not None else np.array([np.inf])
    tasa_fact = float(np.mean(CV <= 0))
    n_ult = 0
    if (CV <= 0).any():
        n_ult = len(no_dominados(res.pop.get("F")[CV <= 0]))

    ps = extraer_pareto_set(p_opt, res, archivo)
    if ps is None:
        F, X = np.empty((0, args.n_obj)), np.empty((0, p_eval.n_var), dtype=np.int32)
    else:
        F, X = ps["F"], ps["X"]
        if not mismo:  # re-evaluar el conjunto de k objetivos en el espacio del experimento
            from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
            F = np.array([p_eval.evaluar(x)[0] for x in X])
            _, idx = np.unique(np.round(F, 10), axis=0, return_index=True)
            F, X = F[idx], X[idx]
            nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
            F, X = F[nd], X[nd]
    return {"spec": spec, "run": run, "seed": seed, "tiempo_s": tiempo, "F": F,
            "X": X.astype(np.int32) if args.guardar_pareto != "frente" else None,
            "X_comp": X[indice_compromiso(F)].astype(np.int32) if len(F) else None,
            "tasa_factibles": tasa_fact, "n_nd_ult_poblacion": n_ult,
            "conv": conv.registros}


# ------------------------------------------------------------------
# Figuras
# ------------------------------------------------------------------
def figuras(dir_out, problem, df_runs, conv_df, frentes_union, refs):
    """
    Genera las figuras PNG del escenario: boxplots de HV e IGD+, convergencia
    del HV (mediana y rango intercuartílico), coordenadas paralelas y matriz de
    dispersión de los frentes unión, con MEC y greedy como referencias.
    """
    import matplotlib
    # backend sin ventana: permite generar PNG en servidores / procesos batch
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algos = list(frentes_union)
    colores = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    col = {a: colores[i % len(colores)] for i, a in enumerate(algos)}

    # Boxplots HV / IGD+
    for met in ("HV", "IGD+"):
        fig, ax = plt.subplots(figsize=(1.6 * len(algos) + 2, 4))
        datos = [df_runs.loc[df_runs.algoritmo == a, met].dropna() for a in algos]
        try:
            ax.boxplot(datos, tick_labels=algos)
        except TypeError:  # matplotlib < 3.9
            ax.boxplot(datos, labels=algos)
        ax.set_ylabel(met + (" (↑ mejor)" if met == "HV" else " (↓ mejor)"))
        ax.set_title(f"{met} por algoritmo ({df_runs.run.nunique()} corridas)")
        fig.tight_layout(); fig.savefig(os.path.join(dir_out, f"boxplot_{met.replace('+', 'plus')}.png"), dpi=150)
        plt.close(fig)

    # Convergencia HV (mediana y RIC)
    if conv_df is not None and len(conv_df):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for a in algos:
            g = conv_df[conv_df.algoritmo == a].groupby("gen")["HV"]
            med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
            ax.plot(med.index, med.values, label=a, color=col[a])
            ax.fill_between(med.index, q1.values, q3.values, alpha=0.2, color=col[a])
        ax.set_xlabel("Generación"); ax.set_ylabel("HV normalizado (mediana, RIC)")
        ax.legend(); ax.set_title("Convergencia del hipervolumen")
        fig.tight_layout(); fig.savefig(os.path.join(dir_out, "convergencia_hv.png"), dpi=150)
        plt.close(fig)

    # Coordenadas paralelas (valores normalizados; 0 = mejor)
    allF = [F for F in frentes_union.values() if len(F)] + [F for F in refs.values()]
    if allF:
        norm = Normalizador(allF)
        etiquetas = [f"{o}\n({s})" for o, s in zip(problem.objetivos, problem.sentidos)]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        xs = np.arange(problem.n_obj)
        for a in algos:
            if len(frentes_union[a]):
                for j, f in enumerate(norm(frentes_union[a])):
                    ax.plot(xs, f, color=col[a], alpha=0.35, lw=1, label=a if j == 0 else None)
        estilos = {"MEC": ("k", "--"), "greedy": ("tab:red", ":")}
        for nombre, Fr in refs.items():
            if nombre in algos:
                continue  # ya graficado como algoritmo del estudio comparativo (evita duplicar)
            c, ls = estilos.get(nombre, ("gray", "-."))
            ax.plot(xs, norm(Fr)[0], color=c, ls=ls, lw=2.5, label=nombre)
        ax.set_xticks(xs); ax.set_xticklabels(etiquetas, fontsize=8)
        ax.set_ylabel("valor normalizado (0 = mejor)")
        ax.set_title("Coordenadas paralelas: frentes no dominados (unión de corridas)")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout(); fig.savefig(os.path.join(dir_out, "coordenadas_paralelas.png"), dpi=150)
        plt.close(fig)

        # Matriz de dispersión (valores reportables)
        M = problem.n_obj
        fig, axs = plt.subplots(M, M, figsize=(2.4 * M, 2.4 * M))
        for i in range(M):
            for j in range(M):
                ax = axs[i, j]
                if i == j:
                    ax.text(0.5, 0.5, problem.objetivos[i], ha="center", va="center", fontsize=9)
                    ax.set_xticks([]); ax.set_yticks([])
                    continue
                for a in algos:
                    if len(frentes_union[a]):
                        R = problem.a_reportable(frentes_union[a])
                        ax.scatter(R[:, j], R[:, i], s=6, color=col[a], alpha=0.6)
                for nombre, Fr in refs.items():
                    if nombre in algos:
                        continue  # ya graficado como algoritmo del estudio comparativo
                    R = problem.a_reportable(Fr)
                    ax.scatter(R[:, j], R[:, i], s=40, marker="*" if nombre == "MEC" else "x",
                               color="k" if nombre == "MEC" else "tab:red")
                ax.tick_params(labelsize=6)
        fig.suptitle("Matriz de dispersión (★ MEC, × greedy)")
        fig.tight_layout(); fig.savefig(os.path.join(dir_out, "matriz_dispersion.png"), dpi=130)
        plt.close(fig)


# ------------------------------------------------------------------
# Escenario completo
# ------------------------------------------------------------------
def correr_escenario(nombre_esc, inst, args):
    """
    Ejecuta el protocolo completo sobre un escenario y escribe todos sus resultados.

    Etapas: referencias (greedy, MEC) -> corridas (secuenciales o en paralelo)
    -> normalización global y frente de referencia -> indicadores por corrida
    -> resumen y estadística -> tablas vs referencias -> informe -> figuras.

    Returns:
        DataFrame con una fila por corrida, o None si ninguna fue factible.
    """
    dir_out = os.path.join(args.salida, nombre_esc)
    os.makedirs(os.path.join(dir_out, "frentes"), exist_ok=True)
    log.info(f"=== Escenario {nombre_esc}: {len(inst['estudiantes'])} estudiantes, "
             f"{len(inst['docentes'])} docentes, {len(inst['clases'])} clases ===")

    _init_worker(inst, vars(args))
    problem = _problema(args.n_obj)

    # --- Soluciones de referencia ---
    refs, filas_ref = {}, []
    xg = solucion_greedy(problem)
    Fg, Gg, detg = problem.evaluar(xg, detalle=True)
    refs["greedy"] = Fg[None, :]
    filas_ref.append({"referencia": "greedy", **detg})
    if inst.get("asignacion") is not None and len(inst["asignacion"]):
        xm, info = inst_mod.x_desde_asignacion(problem, inst["asignacion"])
        Fm, Gm, detm = problem.evaluar(xm, detalle=True)
        filas_ref.append({"referencia": "MEC", **detm, **{f"mapeo_{k}": v for k, v in info.items()}})
        refs["MEC"] = Fm[None, :]
        if np.any(Gm > 0):
            log.warning(f"⚠️ La asignación MEC viola restricciones: "
                        f"{dict(zip(NOMBRES_RESTRICCIONES, Gm))} (se compara igualmente)")
    pd.DataFrame(filas_ref).to_csv(os.path.join(dir_out, "referencias.csv"), index=False)

    # --- Corridas ---
    # "greedy" es determinista (no depende de la semilla): se corre una sola vez
    # por escenario, no `--runs` veces, para no desperdiciar cómputo repitiendo
    # exactamente la misma solución.
    tareas = [(spec, r, args.seed_base + r)
              for spec in args.algoritmos
              for r in (range(1) if parse_spec(spec, args.n_obj)[0] == "greedy" else range(args.runs))]
    resultados = []
    t0 = time.time()
    if args.procesos > 1:
        # cada proceso recibe la instancia una sola vez (initializer) y arma su _CTX
        with ProcessPoolExecutor(max_workers=args.procesos, initializer=_init_worker,
                                 initargs=(inst, vars(args))) as ex:
            futs = [ex.submit(ejecutar_corrida, *t) for t in tareas]
            for n, f in enumerate(futs, 1):
                resultados.append(f.result())
                log.info(f"[{n}/{len(tareas)}] {resultados[-1]['spec']} run {resultados[-1]['run']} "
                         f"({resultados[-1]['tiempo_s']:.1f}s)")
    else:
        for n, t in enumerate(tareas, 1):
            resultados.append(ejecutar_corrida(*t))
            eta = (time.time() - t0) / n * (len(tareas) - n)
            log.info(f"[{n}/{len(tareas)}] {t[0]} run {t[1]} "
                     f"({resultados[-1]['tiempo_s']:.1f}s, ETA {eta / 60:.1f} min)")

    # --- Normalización global y frente de referencia ---
    todos = [r["F"] for r in resultados if len(r["F"])]
    if not todos:
        log.error("Ninguna corrida produjo soluciones factibles.")
        return None
    # ideal/nadir SOLO de los frentes (sin MEC/greedy, que distorsionarían la escala del HV)
    norm = Normalizador(todos)
    frente_ref = no_dominados(np.vstack(todos))
    ind = Indicadores(norm, frente_ref)
    pd.DataFrame(problem.a_reportable(frente_ref), columns=problem.objetivos).to_csv(
        os.path.join(dir_out, "frente_referencia.csv"), index=False)

    # --- Métricas por corrida ---
    # además de los indicadores: mejor valor por objetivo, valores de la solución
    # de compromiso y cuántas soluciones dominan a cada referencia
    filas, conv = [], []
    for r in resultados:
        F = r["F"]
        m = ind(F)
        fila = {"algoritmo": r["spec"], "run": r["run"], "seed": r["seed"],
                "tiempo_s": r["tiempo_s"], "factible": len(F) > 0,
                "tasa_factibles_pob_final": r.get("tasa_factibles", np.nan),
                "n_nd_ult_poblacion": r.get("n_nd_ult_poblacion", np.nan), **m}
        if len(F):
            R = problem.a_reportable(F)
            ic = indice_compromiso(F)
            for k, o in enumerate(problem.objetivos):
                fila[f"mejor_{o}"] = (R[:, k].min() if problem.sentidos[k] == "min" else R[:, k].max())
                fila[f"compromiso_{o}"] = R[ic, k]
            for nref, Fr in refs.items():
                d = dominancia_vs_ref(F, Fr[0])
                fila[f"dominan_{nref}"] = d["dominan_a_ref"]
        filas.append(fila)
        tag = f"{r['spec'].replace('@', '_at')}_run{r['run']:02d}"
        np.savez_compressed(os.path.join(dir_out, "frentes", tag + ".npz"),
                            F=F, X_compromiso=r["X_comp"] if r["X_comp"] is not None else np.empty(0))
        if len(F):  # PF* de la corrida en signo natural, con la solución elegida por cada método
            et = etiquetar_decisiones(F)
            dfF = pd.DataFrame(problem.a_reportable(F), columns=problem.objetivos)
            dfF.insert(0, "solucion_idx", np.arange(len(F)))
            dfF["seleccionada"] = dfF["solucion_idx"].map(et).fillna("")
            dfF.to_csv(os.path.join(dir_out, "frentes", tag + "_frente.csv"), index=False)
            if args.guardar_pareto != "frente" and r.get("X") is not None:
                ps = {"X": r["X"], "F": F, "n_soluciones": len(F), "ideal": F.min(0),
                      "nadir": F.max(0), "obj_names": list(problem.objetivos)}
                exportar_pareto(problem, ps, os.path.join(dir_out, "conjuntos_pareto"), tag,
                                guardar_asignaciones=("seleccion" if args.guardar_pareto == "asignaciones"
                                                      else "ninguna"), etiquetas=et)
        for g, Fg_ in r["conv"]:
            conv.append({"algoritmo": r["spec"], "run": r["run"], "gen": g,
                         "HV": ind(Fg_)["HV"] if len(Fg_) else 0.0})
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(dir_out, "corridas.csv"), index=False)
    conv_df = pd.DataFrame(conv)
    conv_df.to_csv(os.path.join(dir_out, "convergencia.csv"), index=False)

    # --- Resumen y estadística ---
    cols = ["HV", "IGD+", "GD+", "spacing", "n_nd", "n_nd_ult_poblacion",
            "tasa_factibles_pob_final", "tiempo_s"]
    resumen = df.groupby("algoritmo")[[c for c in cols if c in df]].agg(
        ["mean", "std", "median", "min", "max"])
    resumen.to_csv(os.path.join(dir_out, "resumen.csv"))
    kw, est = estadistica_completa(df, correccion=args.correccion)
    kw.to_csv(os.path.join(dir_out, "kruskal_wallis.csv"), index=False)
    est.to_csv(os.path.join(dir_out, "estadistica.csv"), index=False)

    # --- Tablas estilo CACIC (unión de corridas por algoritmo vs MEC / greedy) ---
    frentes_union, tablas = {}, []
    for spec in args.algoritmos:
        Fs = [r["F"] for r in resultados if r["spec"] == spec and len(r["F"])]
        frentes_union[spec] = no_dominados(np.vstack(Fs)) if Fs else np.empty((0, problem.n_obj))
        if not len(frentes_union[spec]):
            continue
        ic = indice_compromiso(frentes_union[spec])
        et = etiquetar_decisiones(frentes_union[spec])
        dfu = pd.DataFrame(problem.a_reportable(frentes_union[spec]), columns=problem.objetivos)
        dfu.insert(0, "solucion_idx", np.arange(len(dfu)))
        dfu["seleccionada"] = dfu["solucion_idx"].map(et).fillna("")
        dfu.to_csv(os.path.join(dir_out, f"frente_union_{spec.replace('@', '_at')}.csv"), index=False)
        for nref, Fr in refs.items():
            t = tabla_mejora(problem, frentes_union[spec], Fr[0], nref, ic)
            dom = dominancia_vs_ref(frentes_union[spec], Fr[0])
            t.insert(0, "algoritmo", spec); t.insert(1, "referencia", nref)
            for k_, v_ in dom.items():
                t[k_] = v_
            tablas.append(t)
    if tablas:
        pd.concat(tablas, ignore_index=True).to_csv(os.path.join(dir_out, "mejora_vs_referencias.csv"),
                                                    index=False)

    # --- Informe de texto ---
    with open(os.path.join(dir_out, "informe.txt"), "w", encoding="utf-8") as f:
        f.write(f"Escenario: {nombre_esc}\n")
        f.write(f"Instancia: {problem.na} estudiantes, {problem.nd} docentes, {problem.nc} clases\n")
        f.write(f"Objetivos: {', '.join(problem.objetivos)}\n")
        f.write(f"Corridas: {args.runs} x {len(args.algoritmos)} | pop={args.pop} gen={args.gen} "
                f"| tiempo total {(time.time() - t0) / 60:.1f} min\n\n")
        f.write("== Referencias ==\n" + pd.DataFrame(filas_ref).T.to_string() + "\n\n")
        f.write("== Resumen de indicadores ==\n" + resumen.round(4).to_string() + "\n\n")
        f.write("== Kruskal–Wallis (global) ==\n" + kw.round(4).to_string() + "\n\n")
        f.write(f"== Mann–Whitney U por pares (p corregido: {args.correccion}) y A12 ==\n"
                + est.round(4).to_string() + "\n\n")
        if tablas:
            f.write("== Mejora vs referencias (unión de corridas) ==\n"
                    + pd.concat(tablas).round(4).to_string() + "\n")

    if not args.sin_figuras:
        try:
            figuras(dir_out, problem, df, conv_df, frentes_union, refs)
        except Exception as e:
            log.warning(f"No se pudieron generar figuras: {e}")
    log.info(f"✅ Resultados en {dir_out}")
    return df


def main(argv=None):
    """Punto de entrada: guarda config.json, carga datos y corre cada escenario."""
    args = parse_args(argv)
    args.n_obj = len(objetivos_experimento(args))   # --objetivos manda sobre --n-obj
    os.makedirs(args.salida, exist_ok=True)
    cfg = dict(vars(args), python=sys.version, plataforma=platform.platform(),
               fecha=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        import pymoo
        cfg["pymoo"] = pymoo.__version__
    except Exception:
        pass
    with open(os.path.join(args.salida, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    inst = cargar_instancia(args)
    if args.runs <= 0:
        log.info("--runs 0: solo carga/exportación de datos; no se ejecutan algoritmos.")
        return
    for nombre, sub in escenarios(inst, args):
        correr_escenario(nombre, sub, args)


if __name__ == "__main__":
    main()
