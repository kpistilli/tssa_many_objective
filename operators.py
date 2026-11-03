# ================================================================
# operators.py
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
# Operadores específicos del problema TSSA para pymoo:
#   - reparar(): mecanismo de reparación R1–R5 descrito en la Sección 5.2
#                del preprint (en la v1.3 del código no estaba implementado).
#   - solucion_greedy(): heurística greedy de la Sección 6.2.2 del preprint.
#   - TSSASampling:  población inicial consciente del grado (+ semilla greedy).
#   - TSSAMutation:  mutación categórica (reasignar/intercambiar) para alumnos y docentes.
#   - TSSARepair:    envoltorio pymoo de reparar().
#
# Nota: todos los operadores usan np.random (el generador global que pymoo
# siembra con `seed`), de modo que las corridas son reproducibles.
#
# Todos reciben el problema `p` (IntegratedProblem) y usan sus estructuras
# precalculadas: p.cand (clases candidatas por alumno, de la más cercana a la
# más lejana), p.doc_orden (docentes por cercanía a cada clase), p.cap,
# p.franjas, p.estab, p.U y p.V. Ver integrated_problem.py.
# ================================================================

import numpy as np
from pymoo.core.mutation import Mutation
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling


# ------------------------------------------------------------------
# Reparación
# ------------------------------------------------------------------
def _reparar_estudiantes(p, XA):
    """
    Repara el bloque de estudiantes XA (se modifica en el lugar).

    Pasos, en este orden:
      1. Grado (g5): alumno en una clase de otro grado -> su clase compatible
         más cercana (sólo si su grado tiene clases).
      2. R2 máximo: en cada clase sobrecargada (orden aleatorio) se mueven
         primero los alumnos más lejanos a ella, cada uno a su clase compatible
         más cercana que tenga cupo.
      3. R2 mínimo (si p.cap_min > 0): se intenta CERRAR cada clase abierta con
         menos de cap_min alumnos, redistribuyéndolos a clases que ya cumplen
         el mínimo; se cierra sólo si todos tienen destino (si no, se deja igual).

    R1 (un alumno en una sola clase) se cumple por construcción: XA[i] es un
    único índice de clase.

    Returns:
        XA reparado y n (alumnos por clase), que usa luego la reparación de docentes.
    """
    # Grado: alumno en clase de otro grado -> su clase compatible más cercana
    malos = np.where((p.g_est != p.g_cls[XA]) & p.tiene_clase_grado)[0]
    for i in malos:
        XA[i] = p.cand[i][0]

    n = np.bincount(XA, minlength=p.nc)

    # R2 (máximo): mover el exceso a la clase compatible más cercana con cupo
    sobre = np.where(n > p.cap)[0]
    np.random.shuffle(sobre)
    for c in sobre:
        miembros = np.where(XA == c)[0]
        exceso = n[c] - p.cap[c]
        # se mueven primero los alumnos más lejanos a la clase sobrecargada
        orden = miembros[np.argsort(-p.U[miembros, c])]
        movidos = 0
        for i in orden:
            if movidos >= exceso:
                break
            for c2 in p.cand[i]:
                if c2 != c and n[c2] < p.cap[c2]:
                    XA[i] = c2
                    n[c] -= 1
                    n[c2] += 1
                    movidos += 1
                    break

    # R2 (mínimo, opcional): cerrar clases activas por debajo de cap_min.
    # Se empieza por las más chicas (menos alumnos que reubicar).
    if p.cap_min > 0:
        chicas = np.where((n > 0) & (n < p.cap_min))[0]
        chicas = chicas[np.argsort(n[chicas])]
        for c in chicas:
            # su estado pudo cambiar al recibir alumnos de otra clase cerrada antes
            if n[c] == 0 or n[c] >= p.cap_min:
                continue
            miembros = np.where(XA == c)[0]
            destino = {}                              # alumno -> clase destino tentativa
            ok = True
            usados = np.zeros(p.nc, dtype=int)        # cupos reservados en esta pasada
            for i in miembros:
                elegido = -1
                # destino válido: otra clase ya "viable" (>= cap_min) y con cupo libre
                for c2 in p.cand[i]:
                    if c2 != c and n[c2] >= p.cap_min and n[c2] + usados[c2] < p.cap[c2]:
                        elegido = c2
                        break
                if elegido < 0:
                    ok = False
                    break
                destino[i] = elegido
                usados[elegido] += 1
            if ok:  # cerrar la clase solo si todos sus alumnos tienen destino
                for i, c2 in destino.items():
                    XA[i] = c2
                n += usados
                n[c] = 0
    return XA, n


def _reparar_docentes(p, XD, n, p_coloc=0.5):
    """
    Repara el bloque de docentes XD (se modifica en el lugar) dado n = alumnos por clase.

    R3 (a lo sumo 2 clases por docente), R4 (las 2 clases sin franjas horarias superpuestas)
    y R5 (toda clase abierta tiene docente).

    Fase 1: se recorren las clases abiertas en orden aleatorio y se CONSERVA el
            docente actual si todavía es compatible (preserva la herencia genética).
    Fase 2: cada clase que quedó pendiente recibe:
            (a) con probabilidad p_coloc, un docente que ya tiene 1 clase en el
                MISMO establecimiento y el turno libre (favorece f3);
            (b) si no, el docente disponible más cercano (favorece f1).
    Si no queda ningún docente disponible, la clase queda sin docente (g2 > 0).
    """
    activa = n > 0
    XD[~activa] = p.nd                      # clase cerrada: libera al docente
    carga = np.zeros(p.nd, dtype=int)       # clases ya asignadas a cada docente
    ocupado = np.zeros((p.nd, p.nf), dtype=bool)   # ocupado[d, f]: el docente d ya tiene ocupada la franja f
    estab_doc = np.full(p.nd, -1)           # establecimiento de la (primera) clase del docente

    # Fase 1: conservar asignaciones compatibles
    pendientes = []
    for c in np.random.permutation(np.where(activa)[0]):
        d, t = XD[c], p.franjas[c]   # t: franjas que ocupa la clase c
        if d < p.nd and carga[d] < 2 and not ocupado[d, t].any():
            carga[d] += 1
            ocupado[d, t] = True
            estab_doc[d] = p.estab[c]
        else:
            pendientes.append(c)

    # Fase 2: reasignar las clases pendientes
    for c in pendientes:
        t = p.franjas[c]
        XD[c] = p.nd
        # (a) preferencia por co-localización (favorece f3): docente con 1 clase
        #     en el mismo establecimiento y el turno libre
        if np.random.random() < p_coloc:
            cands = np.where((estab_doc == p.estab[c]) & (carga == 1) & ~ocupado[:, t].any(axis=1))[0]
            if len(cands):
                d = cands[np.argmin(p.V[cands, c])]
                XD[c] = d
                carga[d] += 1
                ocupado[d, t] = True
                continue
        # (b) docente disponible más cercano
        for d in p.doc_orden[c]:
            if carga[d] < 2 and not ocupado[d, t].any():
                XD[c] = d
                carga[d] += 1
                ocupado[d, t] = True
                estab_doc[d] = p.estab[c]
                break
    return XD


def reparar(p, x, p_coloc=0.5):
    """
    Repara un individuo completo (Sección 5.2 del preprint).

    Primero los estudiantes (grado y capacidad) y luego los docentes, porque
    qué clases quedan abiertas depende de dónde terminaron los alumnos.
    Trabaja sobre copias: el vector x original no se modifica.

    Args:
        p: IntegratedProblem.
        x: vector de decisión (float o int).
        p_coloc: probabilidad de preferir co-localización al reasignar docentes.

    Returns:
        np.ndarray entero [XA | XD] reparado.
    """
    XA, XD = p.decodificar(x)
    XA, XD = XA.copy(), XD.copy()
    XA, n = _reparar_estudiantes(p, XA)
    XD = _reparar_docentes(p, XD, n, p_coloc=p_coloc)
    return np.concatenate([XA, XD])


class TSSARepair(Repair):
    """
    Envoltorio pymoo de reparar(): pymoo lo aplica a cada hijo después de
    cruce y mutación (y a la población inicial), antes de evaluarlo.
    Devuelve float porque así lo espera el resto del pipeline de pymoo.
    """

    def __init__(self, p_coloc=0.5, **kwargs):
        super().__init__(**kwargs)
        self.p_coloc = p_coloc

    def _do(self, problem, X, **kwargs):
        return np.array([reparar(problem, x, self.p_coloc) for x in X], dtype=float)


# ------------------------------------------------------------------
# Heurística greedy (preprint, Sección 6.2.2)
# ------------------------------------------------------------------
def solucion_greedy(p, orden_estudiantes=None, mas_lejanos_primero=True):
    """
    Asigna cada estudiante a la clase compatible más cercana con cupo y luego
    a cada clase activa el docente más cercano que respete R3 y R4.

    mas_lejanos_primero (adoptado de la v2): se atiende primero a los estudiantes
    cuya clase compatible más cercana está más lejos, para que no queden sin cupo
    cerca y terminen asignados muy lejos (favorece f4, la distancia máxima).
    Con False se usa el orden original de los datos.

    Args:
        orden_estudiantes: orden explícito en que se atiende a los alumnos
            (tiene prioridad sobre mas_lejanos_primero).

    Si ninguna clase compatible tiene cupo, el alumno va a la más cercana
    igualmente (la solución quedará con g1 > 0). Es determinista: no usa azar.

    Returns:
        np.ndarray entero [XA | XD]. Se usa como semilla de la población inicial
        y como línea de base no evolutiva en experiments.py.
    """
    # --- Fase 1: alumnos -> clase compatible más cercana con cupo ---
    XA = np.zeros(p.na, dtype=int)
    n = np.zeros(p.nc, dtype=int)
    if orden_estudiantes is not None:
        orden = orden_estudiantes
    elif mas_lejanos_primero:
        # distancia de cada alumno a su clase compatible más cercana; se atiende
        # de mayor a menor (orden estable para que el resultado sea reproducible)
        mas_cercana = np.array([p.U[i, p.cand[i][0]] for i in range(p.na)])
        orden = np.argsort(-mas_cercana, kind="stable")
    else:
        orden = np.arange(p.na)
    for i in orden:
        elegido = p.cand[i][0]        # por defecto: la más cercana (aunque esté llena)
        for c in p.cand[i]:
            if n[c] < p.cap[c]:
                elegido = c
                break
        XA[i] = elegido
        n[elegido] += 1
    # --- Fase 2: clase abierta -> docente más cercano que respete R3 (≤2) y R4 (turno) ---
    XD = np.full(p.nc, p.nd, dtype=int)
    carga = np.zeros(p.nd, dtype=int)
    ocupado = np.zeros((p.nd, p.nf), dtype=bool)
    for c in np.where(n > 0)[0]:
        t = p.franjas[c]
        for d in p.doc_orden[c]:
            if carga[d] < 2 and not ocupado[d, t].any():
                XD[c] = d
                carga[d] += 1
                ocupado[d, t] = True
                break
    return np.concatenate([XA, XD])


def semillas_estabilidad(p, fracciones=(0.25, 0.5, 0.75), seed=0):
    """
    Semillas para cubrir el eje de f6 (estabilidad respecto del MEC).

    Devuelve la asignación del MEC (f6 = 0) y una solución intermedia por cada
    fracción: esa proporción de alumnos conserva su clase del MEC y el resto
    toma su clase del greedy; luego se repara. Así la población inicial abarca
    desde "no mover a nadie" hasta el greedy (que mueve a casi todos), y el
    cruce puede completar la curva entre ambos extremos.

    Es determinista: usa su propio estado aleatorio (`seed`) y deja np.random
    como estaba, para no alterar la reproducibilidad de la corrida.

    Args:
        p: IntegratedProblem con x_base (asignación del MEC).
        fracciones: proporciones de alumnos que conservan su clase del MEC.

    Returns:
        lista de vectores x enteros (MEC primero).
    """
    if p.x_base is None:
        return []
    x_mec = np.asarray(p.x_base, dtype=int).copy()
    xg = solucion_greedy(p)
    XA_mec, XA_g = x_mec[:p.na], xg[:p.na]
    estado = np.random.get_state()
    try:
        np.random.seed(seed)
        orden = np.random.permutation(p.na)     # qué alumnos conservan su clase primero
        semillas = [x_mec]
        for f in fracciones:
            conservan = orden[:int(round(f * p.na))]
            XA = XA_g.copy()
            XA[conservan] = XA_mec[conservan]
            semillas.append(reparar(p, np.concatenate([XA, xg[p.na:]])))
    finally:
        np.random.set_state(estado)
    return semillas


# ------------------------------------------------------------------
# Muestreo inicial
# ------------------------------------------------------------------
class TSSASampling(Sampling):
    """
    Cada estudiante elige una clase de su grado; el individuo r usa una
    "temperatura" distinta: desde muy cercano (casi greedy) hasta casi uniforme,
    para cubrir distintas regiones del frente. Los docentes se inicializan al azar
    y la reparación los vuelve factibles.

    Args:
        semillas: lista de vectores x a inyectar al inicio (p. ej. greedy).

    Cómo funciona la "temperatura": para el alumno i se sortea una posición j
    en su lista de candidatas con una distribución exponencial de media
    `escala`. Con escala chica casi todos toman su clase más cercana (j≈0);
    con escala grande se reparten entre clases más lejanas. Cada individuo
    sortea su propia escala, lo que diversifica la población inicial.
    """

    def __init__(self, semillas=None):
        super().__init__()
        self.semillas = [] if semillas is None else list(semillas)

    def _do(self, problem, n_samples, **kwargs):
        p = problem
        X = np.empty((n_samples, p.n_var), dtype=float)
        largos = np.array([len(c) for c in p.cand])   # nº de clases compatibles por alumno
        for r in range(n_samples):
            if r < len(self.semillas):                 # primeras filas: semillas (greedy, etc.)
                X[r] = self.semillas[r]
                continue
            escala = np.random.uniform(0.2, max(1.0, largos.mean()))
            # posición sorteada en cand[i], acotada al largo de cada lista
            j = np.minimum(np.floor(np.random.exponential(escala, size=p.na)).astype(int), largos - 1)
            XA = np.fromiter((p.cand[i][j[i]] for i in range(p.na)), dtype=int, count=p.na)
            # docentes al azar (incluye nd = "sin docente"); la reparación los ajusta
            XD = np.random.randint(0, p.nd + 1, size=p.nc)
            X[r] = np.concatenate([XA, XD])
        return X


# ------------------------------------------------------------------
# Mutación categórica
# ------------------------------------------------------------------
class TSSAMutation(Mutation):
    """
    Movimientos:
      estudiantes: (a) reasignar a una de sus k clases compatibles más cercanas,
                   (b) intercambiar dos estudiantes del mismo grado.
      docentes:    (a) reasignar a uno de los k docentes más cercanos,
                   (b) intercambiar docentes entre dos clases.

    Args:
        tasa_est: fracción esperada de estudiantes mutados por individuo.
        tasa_doc: fracción esperada de clases con docente mutado.
        k: tamaño del vecindario de candidatos.
        prob: probabilidad de mutar cada individuo (pymoo).

    La cantidad de movimientos por individuo es Binomial(n, tasa), con al
    menos 1. Cada movimiento elige (a) o (b) con probabilidad 1/2. La mutación
    puede generar soluciones infactibles: TSSARepair las corrige después.
    """

    def __init__(self, tasa_est=0.005, tasa_doc=0.02, k=5, prob=1.0, **kwargs):
        super().__init__(prob=prob, **kwargs)
        self.tasa_est, self.tasa_doc, self.k = tasa_est, tasa_doc, k

    def _do(self, problem, X, **kwargs):
        p = problem
        X = np.rint(X).astype(int)
        for r in range(len(X)):
            # XA y XD son VISTAS de la fila r: modificarlas modifica X
            XA, XD = X[r, :p.na], X[r, p.na:]
            # --- estudiantes ---
            m_est = max(1, np.random.binomial(p.na, self.tasa_est))
            for i in np.random.randint(0, p.na, size=m_est):
                if np.random.random() < 0.5:
                    # (a) reasignar a una de sus k clases compatibles más cercanas
                    kk = min(self.k, len(p.cand[i]))
                    XA[i] = p.cand[i][np.random.randint(kk)]
                else:
                    # (b) intercambio con otro alumno al azar; sólo se concreta si
                    #     es del mismo grado (si no, el movimiento no hace nada)
                    mismos = p.cand[i]  # clases del mismo grado
                    j = np.random.randint(0, p.na)
                    if p.g_est[j] == p.g_est[i] and len(mismos) > 1:
                        XA[i], XA[j] = XA[j], XA[i]
            # --- docentes ---
            m_doc = max(1, np.random.binomial(p.nc, self.tasa_doc))
            for c in np.random.randint(0, p.nc, size=m_doc):
                if np.random.random() < 0.5:
                    # (a) uno de los k docentes más cercanos a la clase
                    XD[c] = p.doc_orden[c][np.random.randint(min(self.k, p.nd))]
                else:
                    # (b) intercambiar los docentes de dos clases al azar
                    c2 = np.random.randint(0, p.nc)
                    XD[c], XD[c2] = XD[c2], XD[c]
        return X.astype(float)
