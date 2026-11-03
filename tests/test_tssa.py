# ================================================================
# test_tssa.py
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
# Pruebas del modelo TSSA many-objective:  python -m pytest -q tests
#
# Cubren: valores exactos del ejemplo didáctico del preprint, regresión contra
# la implementación v1.3 (tests/legacy), factibilidad de reparación y greedy,
# carga del script SQL y de la asignación MEC, los 6 algoritmos, el archivo
# Pareto, los métodos de decisión, f4_modo/f6, Holm/Bonferroni y exportación.
# Las corridas usan poblaciones y generaciones chicas para que sean rápidas.
# ================================================================
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Carpeta del proyecto (padre de tests/), para importar los módulos sin instalar nada
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from instancia import cargar_desde_sql, generar_instancia_sintetica, x_desde_asignacion  # noqa: E402
from integrated_optimization import (run_integrated_optimization,  # noqa: E402
                                     select_best_individual)
from integrated_problem import IntegratedProblem  # noqa: E402
from operators import reparar, solucion_greedy  # noqa: E402

SQL = os.path.join(RAIZ, "asignacion_mec.sql")


@pytest.fixture(scope="module")
def ejemplo():
    """Instancia de ejemplo (24 alumnos, 9 docentes, 9 clases), cargada una vez por módulo."""
    return cargar_desde_sql(SQL)


def _problema(inst, **kw):
    """Atajo: IntegratedProblem de una instancia con las opciones indicadas."""
    return IntegratedProblem(inst["estudiantes"], inst["docentes"], inst["clases"], **kw)


# ------------------------------------------------------------------
# 1) Ejemplo didáctico del preprint (Sección 4): f1 = 1.65, f2 = 0.25
# ------------------------------------------------------------------
def test_ejemplo_didactico_preprint():
    # U y V se pasan explícitas (las del ejemplo del paper), sin coordenadas
    est = pd.DataFrame({"nombre": ["a1", "a2", "a3"], "grado": ["1"] * 3})
    doc = pd.DataFrame({"nombre": ["d1", "d2"], "grado": ["1"] * 2})
    cls = pd.DataFrame({"grado": ["1", "1"], "turno": ["M", "M"], "capacidad": [2, 2],
                        "establecimiento_id": [1, 2], "institucion_id": [1, 2]})
    U = np.array([[1.0, 3.0], [2.0, 1.5], [2.5, 0.8]])
    V = np.array([[0.6, 2.2], [1.8, 0.5]])
    p = IntegratedProblem(est, doc, cls, n_obj=4, U=U, V=V)
    x = np.array([0, 1, 1, 0, 1])          # alumnos -> c1,c2,c2 ; c1->d1, c2->d2
    F, G, det = p.evaluar(x, detalle=True)
    assert np.isclose(det["f1_distancia"], 1.65)
    assert np.isclose(det["f2_balance"], 0.25)
    assert det["f3_colocalizacion"] == 0.0   # ningún docente tiene 2 clases
    assert np.isclose(det["f4_distancia_max"], 1.5 + 0.6)
    assert np.all(G == 0)


# ------------------------------------------------------------------
# 2) Regresión contra la implementación original v1.3 (f2 = std)
# ------------------------------------------------------------------
def test_regresion_vs_codigo_original(ejemplo):
    # 200 soluciones aleatorias evaluadas con el código v1.3 y con el actual
    sys.path.insert(0, os.path.join(RAIZ, "tests", "legacy"))
    from original_integrated_problem import IntegratedProblem as Original
    orig = Original(ejemplo["estudiantes"], ejemplo["docentes"], ejemplo["clases"])
    nuevo = _problema(ejemplo, n_obj=3, f2_metrica="std")
    rng = np.random.default_rng(0)
    for _ in range(200):
        XA = rng.integers(0, nuevo.nc, nuevo.na)
        XD = rng.integers(0, nuevo.nd + 1, nuevo.nc)
        XD[np.bincount(XA, minlength=nuevo.nc) == 0] = nuevo.nd  # clases inactivas sin docente
        x = np.concatenate([XA, XD])
        out = {}
        orig._evaluate(x, out)
        F, G = nuevo.evaluar(x)
        assert np.allclose(F, out["F"], atol=1e-3), (F, out["F"])
        Go = np.asarray(out["G"], dtype=float)
        assert np.allclose(G[[0, 1, 2, 4]], Go[[0, 1, 2, 4]]), (G, Go)
        # g4: la v2.0 cuenta clases en conflicto (más fina); la v1.3 docentes en conflicto.
        assert (G[3] == 0) == (Go[3] == 0) and G[3] >= Go[3]


# ------------------------------------------------------------------
# 3) Reparación y greedy producen soluciones factibles
# ------------------------------------------------------------------
@pytest.mark.parametrize("fuente", ["ejemplo", "sintetica"])
def test_reparacion_factible(ejemplo, fuente):
    inst = ejemplo if fuente == "ejemplo" else generar_instancia_sintetica(10, 600, seed=3)
    p = _problema(inst, n_obj=4)
    np.random.seed(0)
    for _ in range(30):
        x = np.random.randint(p.xl, p.xu + 1)
        _, G = p.evaluar(reparar(p, x))
        assert np.all(G == 0), G
    _, G = p.evaluar(solucion_greedy(p))
    assert np.all(G == 0)


def test_cap_min_reparacion():
    inst = generar_instancia_sintetica(8, 500, seed=5, capacidad=40, holgura=1.6)
    p = _problema(inst, n_obj=4, cap_min=10)
    np.random.seed(1)
    viol = [p.evaluar(reparar(p, np.random.randint(p.xl, p.xu + 1)))[1][0] for _ in range(10)]
    assert np.mean(viol) < 5  # la reparación elimina casi todas las violaciones de cap_min


# ------------------------------------------------------------------
# 4) Datos del script SQL y asignación MEC
# ------------------------------------------------------------------
def test_sql_y_mec(ejemplo):
    assert len(ejemplo["estudiantes"]) == 24
    assert len(ejemplo["docentes"]) == 9
    assert len(ejemplo["clases"]) == 9
    p = _problema(ejemplo, n_obj=4)
    x, info = x_desde_asignacion(p, ejemplo["asignacion"])
    # cada alumno se mapea por exactamente un método (la clave
    # "seccion_sin_clase_propia" es un aviso, no un método de mapeo)
    metodos = ("exacto", "sin_turno", "aproximado", "estudiante_sin_asignacion")
    assert sum(info.get(m, 0) for m in metodos) == 24
    F, _ = p.evaluar(x)
    # la asignación "desbalanceada" del script es peor que la greedy en distancia
    Fg, _ = p.evaluar(solucion_greedy(p))
    assert F[0] > Fg[0] and F[3] > Fg[3]


# ------------------------------------------------------------------
# 5) Algoritmos: corrida corta, frente factible, selección de compromiso
# ------------------------------------------------------------------
@pytest.mark.parametrize("algoritmo", ["nsga2", "nsga3", "unsga3", "ctaea", "rvea", "spea2"])
def test_algoritmos(ejemplo, algoritmo):
    p = _problema(ejemplo, n_obj=4)
    res = run_integrated_optimization(p, 20, 10, algoritmo=algoritmo, verbose=False, seed=1)
    idx, x, F = select_best_individual(res)
    assert F.shape == (4,)
    assert np.all(p.evaluar(x)[1] == 0)


# ------------------------------------------------------------------
# 6) Integración v2: Conjunto Pareto, decisión, f4_modo, f6, estadística
# ------------------------------------------------------------------
def test_archivo_pareto(ejemplo):
    from metrics import no_dominados
    p = _problema(ejemplo, n_obj=4)
    res = run_integrated_optimization(p, 30, 20, algoritmo="nsga3", verbose=False, seed=3)
    ps = res.pareto_set
    assert ps is not None and ps["n_soluciones"] >= 1
    assert len(no_dominados(ps["F"])) == len(ps["F"])          # todas no dominadas
    for x in ps["X"]:
        assert np.all(p.evaluar(x)[1] == 0)                     # todas factibles
    # cada solución factible de la población final está cubierta (igualada o
    # dominada débilmente) por alguna del archivo: el archivo nunca es peor
    CV = res.pop.get("CV").ravel()
    for f in res.pop.get("F")[CV <= 0]:
        assert np.any(np.all(ps["F"] <= f + 1e-9, axis=1))


def test_metodos_decision():
    from pareto import seleccionar_solucion
    F = np.array([[0.0, 1.0], [0.2, 0.2], [1.0, 0.0], [0.6, 0.5]])
    assert seleccionar_solucion(F, "compromiso") == 1
    assert seleccionar_solucion(F, "rodilla") == 1
    assert seleccionar_solucion(F, "min_f1") == 0
    assert seleccionar_solucion(F, "min_f2") == 2
    assert seleccionar_solucion(F, "lexicografico") == 0
    assert 0 <= seleccionar_solucion(F, "pseudo_pesos") < len(F)
    assert seleccionar_solucion(F, "asf", pesos=[0.9, 0.1]) in (0, 1)


def test_f4_modo_y_f6(ejemplo):
    base = _problema(ejemplo, n_obj=3)
    xm, _ = x_desde_asignacion(base, ejemplo["asignacion"])
    obj = ("f1_distancia", "f4_distancia_max", "f6_estabilidad")
    ps = _problema(ejemplo, objetivos=obj, f4_modo="suma", x_base=xm)
    pm = _problema(ejemplo, objetivos=obj, f4_modo="max", x_base=xm)
    xg = solucion_greedy(ps)
    _, _, ds = ps.evaluar(xg, detalle=True)
    _, _, dm = pm.evaluar(xg, detalle=True)
    assert np.isclose(ds["f4_distancia_max"], ds["dist_max_est"] + ds["dist_max_doc"])
    assert np.isclose(dm["f4_distancia_max"], max(ds["dist_max_est"], ds["dist_max_doc"]))
    assert ps.evaluar(xm)[0][2] == 0.0                         # el MEC no cambia a nadie
    assert 0.0 < ds["f6_estabilidad"] <= 1.0
    with pytest.raises(ValueError):
        _problema(ejemplo, objetivos=obj)                      # f6 sin x_base


def test_haversine_pares_igual_a_matriz():
    # la distancia punto a punto que usa el viewer es la misma que la del modelo
    from integrated_problem import haversine_matrix, haversine_pares
    rng = np.random.default_rng(4)
    a = rng.uniform([-26, -55.5], [-25, -54], size=(50, 2))
    b = rng.uniform([-26, -55.5], [-25, -54], size=(50, 2))
    M = haversine_matrix(a[:, 0], a[:, 1], b[:, 0], b[:, 1])
    assert np.allclose(haversine_pares(a[:, 0], a[:, 1], b[:, 0], b[:, 1]), np.diag(M))


def test_greedy_mas_lejanos_primero():
    inst = generar_instancia_sintetica(8, 500, seed=7, holgura=1.05)
    p = _problema(inst, n_obj=4)
    for flag in (True, False):
        _, G = p.evaluar(solucion_greedy(p, mas_lejanos_primero=flag))
        assert np.all(G == 0)


def test_correccion_holm_bonferroni():
    from metrics import _ajustar
    p = np.array([0.01, 0.04, 0.03, np.nan])
    b = _ajustar(p, "bonferroni")
    h = _ajustar(p, "holm")
    assert np.allclose(b[:3], [0.03, 0.12, 0.09]) and np.isnan(b[3])
    assert np.allclose(h[:3], [0.03, 0.06, 0.06]) and np.isnan(h[3])


def test_exportar_pareto_y_seccion(ejemplo, tmp_path):
    # tmp_path: carpeta temporal que crea (y limpia) pytest para esta prueba
    from pareto import decodificar_asignaciones, derivar_seccion, exportar_pareto
    p = _problema(ejemplo, n_obj=4)
    secc = derivar_seccion(p.clases)
    assert set(secc) <= {"A", "B", "C"} and "A" in set(secc)
    res = run_integrated_optimization(p, 20, 10, algoritmo="nsga2", verbose=False, seed=1)
    df = decodificar_asignaciones(p, res.pareto_set["X"][0])
    assert len(df) == p.na and df["seccion"].notna().all()
    meta = exportar_pareto(p, res.pareto_set, str(tmp_path), "prueba")
    assert meta["n_soluciones"] == res.pareto_set["n_soluciones"]
    assert (tmp_path / "prueba" / "pareto_front.csv").exists()
    assert (tmp_path / "prueba" / "pareto_set.npz").exists()


# ------------------------------------------------------------------
# R4 sobre franjas horarias: "Mañana-Tarde" ocupa el día completo del docente
# ------------------------------------------------------------------
def test_franjas_manana_tarde():
    from integrated_problem import franjas_de_turno
    assert franjas_de_turno("Mañana") == ("mañana",)
    assert franjas_de_turno("Mañana-Tarde") == ("mañana", "tarde")
    assert franjas_de_turno("M") == ("m",)          # texto desconocido: franja propia

    est = pd.DataFrame({"nombre": ["a1", "a2", "a3"], "grado": ["1"] * 3})
    doc = pd.DataFrame({"nombre": ["d1", "d2", "d3"], "grado": ["1"] * 3})
    cls = pd.DataFrame({"grado": ["1"] * 3, "turno": ["Mañana", "Tarde", "Mañana-Tarde"],
                        "capacidad": [1, 1, 1], "establecimiento_id": [1, 1, 1],
                        "institucion_id": [1, 1, 1]})
    U = np.ones((3, 3)); V = np.ones((3, 3))
    p = IntegratedProblem(est, doc, cls, n_obj=4, U=U, V=V)
    g4 = lambda XD: p.evaluar(np.array([0, 1, 2] + XD))[1][3]
    assert g4([0, 0, 1]) == 0      # d1: Mañana + Tarde -> sin conflicto
    assert g4([0, 1, 0]) == 1      # d1: Mañana + Mañana-Tarde -> conflicto
    assert g4([1, 0, 0]) == 1      # d1: Tarde + Mañana-Tarde -> conflicto
    # la reparación y el greedy nunca dejan a un docente de Mañana-Tarde con otra clase
    np.random.seed(0)
    for _ in range(20):
        _, G = p.evaluar(reparar(p, np.random.randint(p.xl, p.xu + 1)))
        assert np.all(G == 0), G
    _, G = p.evaluar(solucion_greedy(p))
    assert np.all(G == 0)


# ------------------------------------------------------------------
# MEC -> x: secciones con letras no consecutivas no se juntan en una clase
# ------------------------------------------------------------------
def test_mapeo_secciones_no_consecutivas():
    est = pd.DataFrame({"estudiante_id": [1, 2, 3, 4, 5], "nombre": list("abcde"),
                        "grado": ["1"] * 5, "lat": [0.0] * 5, "lng": [0.0] * 5})
    doc = pd.DataFrame({"docente_id": [10, 11, 12], "nombre": ["d1", "d2", "d3"],
                        "grado": ["1"] * 3, "lat": [0.0] * 3, "lng": [0.0] * 3})
    # establecimiento 7: dos clases de Tarde; establecimiento 8: una clase de Mañana
    cls = pd.DataFrame({"grado": ["1"] * 3, "turno": ["Tarde", "Tarde", "Mañana"],
                        "capacidad": [2, 2, 2], "establecimiento_id": [7, 7, 8],
                        "institucion_id": [7, 7, 8], "lat": [0.0, 0.0, 0.1], "lng": [0.0, 0.0, 0.1]})
    asig = pd.DataFrame({"estudiante_id": [1, 2, 3, 4, 5], "docente_id": [10, 10, 11, 11, 12],
                         "establecimiento_id": [7, 7, 7, 7, 8], "grado": ["1"] * 5,
                         "seccion": ["A", "A", "C", "C", "B"],      # A y C (sin B); "B" sola
                         "turno": ["Tarde"] * 4 + ["Mañana"]})
    p = IntegratedProblem(est, doc, cls, n_obj=4)
    x, info = x_desde_asignacion(p, asig)
    XA = x[:p.na]
    assert XA[0] == XA[1] == 0          # sección A -> 1ª clase
    assert XA[2] == XA[3] == 1          # sección C -> 2ª clase (antes caía en la 1ª)
    assert XA[4] == 2                   # sección B sola -> la única clase de su clave
    assert info.get("seccion_sin_clase_propia", 0) == 0
    _, G = p.evaluar(x)
    assert G[0] == 0                    # sin exceso de capacidad


# ------------------------------------------------------------------
# experiments.py: un greedy infactible no entra en el frente ni en los indicadores
# ------------------------------------------------------------------
def test_greedy_infactible_excluido(ejemplo):
    from experiments import parse_args, _init_worker, ejecutar_corrida
    for cap_min, esperado in ((0, 1), (25, 0)):   # con cap_min=25 el greedy viola R2
        args = parse_args(["--fuente", "ejemplo", "--cap-min", str(cap_min)])
        args.n_obj = 4
        _init_worker(ejemplo, vars(args))
        r = ejecutar_corrida("greedy", 0, 1)
        assert len(r["F"]) == esperado
        assert r["tasa_factibles"] == float(esperado)


# ------------------------------------------------------------------
# Reproducibilidad: misma semilla -> mismo Conjunto Pareto
# (pymoo 0.6.2 ya no siembra np.random; se fija en cada corrida)
# ------------------------------------------------------------------
@pytest.mark.parametrize("alg", ["nsga2", "ctaea"])
def test_reproducibilidad_semilla(ejemplo, alg):
    p = _problema(ejemplo, n_obj=4)
    corrida = lambda s: run_integrated_optimization(p, pop_size=12, n_gen=5, algoritmo=alg,
                                                    seed=s, verbose=False).pareto_set["F"]
    np.random.seed(123)                  # estado global "sucio" distinto antes de cada corrida
    F1 = corrida(7)
    np.random.seed(999)
    F2 = corrida(7)
    assert F1.shape == F2.shape and np.allclose(F1, F2)


# ------------------------------------------------------------------
# f6: semillas MEC + intermedias cubren el eje de estabilidad
# ------------------------------------------------------------------
def test_semillas_estabilidad(ejemplo):
    from operators import semillas_estabilidad
    base = _problema(ejemplo, n_obj=3)
    x_mec, _ = x_desde_asignacion(base, ejemplo["asignacion"])
    p = IntegratedProblem(ejemplo["estudiantes"], ejemplo["docentes"], ejemplo["clases"],
                          objetivos=("f1_distancia", "f2_balance", "f6_estabilidad"), x_base=x_mec)
    estado = np.random.get_state()
    s1, s2 = semillas_estabilidad(p), semillas_estabilidad(p)
    assert np.array_equal(np.random.get_state()[1], estado[1])   # no altera np.random
    assert len(s1) == 4 and all(np.array_equal(a, b) for a, b in zip(s1, s2))  # determinista
    f6 = [p.evaluar(x, detalle=True)[2]["f6_estabilidad"] for x in s1]
    assert f6[0] == 0.0                                  # el MEC no mueve a nadie
    for x in s1[1:]:
        assert np.all(p.evaluar(x)[1] == 0)              # intermedias reparadas y factibles
    # el Conjunto Pareto de una corrida corta incluye la zona cercana al MEC
    r = run_integrated_optimization(p, pop_size=12, n_gen=3, algoritmo="nsga2", seed=1, verbose=False)
    assert r.pareto_set["F"][:, 2].min() == 0.0


# ------------------------------------------------------------------
# Decisión "domina_mec": la elegida mejora al MEC en todos los objetivos
# ------------------------------------------------------------------
def test_decision_domina_mec(ejemplo):
    from pareto import seleccionar_solucion
    from integrated_optimization import select_best_individual
    F = np.array([[1.0, 5.0], [2.0, 2.0], [3.0, 1.0], [0.5, 9.0]])
    assert seleccionar_solucion(F, "domina_mec", ref=[2.5, 3.0]) == 1        # única que domina
    assert seleccionar_solucion(F, "domina_mec:min_f1", ref=[3.5, 9.5]) == 3  # submétodo
    # ninguna domina -> mismo resultado que el submétodo sobre todo el frente
    assert seleccionar_solucion(F, "domina_mec", ref=[0.0, 0.0]) == seleccionar_solucion(F, "compromiso")
    # integración: con x_base (MEC) la solución elegida domina al MEC si alguna lo hace
    base = _problema(ejemplo, n_obj=3)
    x_mec, _ = x_desde_asignacion(base, ejemplo["asignacion"])
    p = IntegratedProblem(ejemplo["estudiantes"], ejemplo["docentes"], ejemplo["clases"],
                          n_obj=4, x_base=x_mec)
    r = run_integrated_optimization(p, pop_size=12, n_gen=5, seed=3, verbose=False)
    Fm = p.evaluar(x_mec)[0]
    dominan = [f for f in r.pareto_set["F"] if np.all(f <= Fm) and np.any(f < Fm)]
    _, _, Fsel = select_best_individual(r, "domina_mec")
    if dominan:
        assert np.all(Fsel <= Fm) and np.any(Fsel < Fm)
