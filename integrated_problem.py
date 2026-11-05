# ================================================================
# integrated_problem.py
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
# Descripción del módulo
# ----------------------
# Problema TSSA (Teacher and Student to School Assignment) para pymoo,
# con soporte para 3 a 6 funciones objetivo:
#
#   f1  min  Distancia promedio estudiantes + distancia promedio docentes   (paper, ec. 22)
#   f2  min  Balance de alumnos por clase: varianza (paper, ec. 23)
#            [opcional: desvío estándar, como en la versión 1.3 del código]
#   f3  max  Proporción de docentes con 2 clases en el MISMO establecimiento (paper, ec. 24)
#   f4  min  Distancia máxima (NUEVA: equidad territorial / criterio minimax).
#            f4_modo="suma" (defecto): max_est + max_doc -> ambos grupos cuentan.
#            f4_modo="max": max(max_est, max_doc) -> definición de la v2
#            (f4_formulacion_many_objective.md); ignora al grupo con menor máximo.
#   f5  max  Afinidad docente–grado: proporción de clases activas cuyo docente
#            tiene el mismo grado que la clase (OPCIONAL, usa G_j de D_j)
#   f6  min  Estabilidad: proporción de estudiantes que cambian de
#            establecimiento respecto de la asignación actual (asignacion_mec).
#            Propuesta de la v2 (costo político de implementar el plan).
#            Requiere x_base (instancia.x_desde_asignacion).
#
# Todas las funciones usan SOLO los datos de entrada ya existentes
# (coordenadas, grado, turno, capacidad, establecimiento).
#
# Codificación (igual que la versión 1.3, para mantener compatibilidad):
#   x = [ XA | XD ]
#   XA[i] ∈ {0..nc-1}  clase asignada al estudiante i
#   XD[l] ∈ {0..nd}    docente asignado a la clase l (nd = "sin docente")
#
# Restricciones (G <= 0 es factible en pymoo; aquí todas son conteos >= 0):
#   g1  R2: exceso de capacidad (+ faltante bajo cap_min en clases activas, si cap_min > 0)
#   g2  R5: clases activas sin docente
#   g3  R3: exceso de carga docente (> 2 clases activas)
#   g4  R4: conflictos de turno: un docente no puede tener dos clases cuyas
#       franjas horarias se superpongan. Cada turno se descompone en franjas
#       (Mañana = {mañana}, Tarde = {tarde}, Mañana-Tarde = {mañana, tarde}),
#       así una clase de jornada Mañana-Tarde ocupa el día completo del docente.
#   g5  compatibilidad de grado estudiante–clase
#
# Cambios respecto a la v1.3:
#   - Matrices de distancia U (na x nc) y V (nd x nc) precalculadas una sola vez.
#   - Evaluación vectorizada con numpy (sin iloc en bucles) -> escalable a ~16k alumnos.
#   - Las restricciones de docentes (g3, g4) y f3/f5 consideran solo clases ACTIVAS
#     (una clase sin alumnos se considera cerrada; su docente no cuenta).
#   - n_obj configurable; nombres/sentidos de objetivos expuestos para reportes.
#   - g4 cuenta CLASES en conflicto de turno (v1.3 contaba docentes en conflicto):
#     medida más fina de la violación, útil para la dominancia por restricciones.
#     Ambas son 0 exactamente en los mismos casos.
# ================================================================

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem

logger = logging.getLogger("integrated_problem")

# ------------------------------------------------------------------
# Catálogo de objetivos: clave -> (sentido, etiqueta legible)
#
# Convención de signos: pymoo siempre MINIMIZA. Por eso, en el vector F que
# ve el algoritmo, los objetivos de sentido "max" (f3, f5) se guardan
# NEGADOS (-valor). Para reportes, tablas y la BD se usa
# IntegratedProblem.a_reportable(), que devuelve el signo natural.
# El orden de las claves aquí NO define el orden de F: ese orden lo fija la
# tupla `objetivos` con la que se construye cada problema.
# ------------------------------------------------------------------
OBJETIVOS = {
    "f1_distancia":      ("min", "Distancia promedio est.+doc. (km)"),
    "f2_balance":        ("min", "Balance alumnos/clase (varianza)"),
    "f3_colocalizacion": ("max", "Docentes con 2 clases en mismo establecimiento (prop.)"),
    "f4_distancia_max":  ("min", "Distancia máxima est.+doc. (km)"),
    "f5_afinidad_grado": ("max", "Clases con docente del mismo grado (prop.)"),
    "f6_estabilidad":    ("min", "Estudiantes que cambian de establecimiento vs MEC (prop.)"),
}

# Conjuntos predefinidos para --n-obj (3 = preprint, 4 = recomendado, 5 = + afinidad).
# f6_estabilidad no forma parte de ningún conjunto predefinido porque exige
# la asignación actual del MEC (x_base); se activa con --objetivos.
CONJUNTOS_OBJETIVOS = {
    3: ("f1_distancia", "f2_balance", "f3_colocalizacion"),
    4: ("f1_distancia", "f2_balance", "f3_colocalizacion", "f4_distancia_max"),
    5: ("f1_distancia", "f2_balance", "f3_colocalizacion", "f4_distancia_max", "f5_afinidad_grado"),
}

# Restricciones de desigualdad (columnas de G, en este orden fijo).
# Todas son conteos de violaciones >= 0; pymoo considera factible G <= 0,
# de modo que una solución es factible sólo si las cinco valen 0.
N_RESTRICCIONES = 5
NOMBRES_RESTRICCIONES = ("g1_capacidad", "g2_clase_sin_docente", "g3_carga_docente",
                         "g4_conflicto_turno", "g5_grado")


# ------------------------------------------------------------------
# Utilitarios
# ------------------------------------------------------------------
# Radio terrestre (km) usado en TODAS las distancias del proyecto (modelo,
# asignaciones guardadas y viewer), para que los números sean comparables.
RADIO_TIERRA_KM = 6371.0


def haversine_matrix(lat1, lng1, lat2, lng2) -> np.ndarray:
    """
    Matriz de distancias Haversine (km, radio terrestre 6371 km).

    Args:
        lat1, lng1: coordenadas (grados) de n puntos de origen.
        lat2, lng2: coordenadas (grados) de m puntos de destino.

    Returns:
        np.ndarray de forma (n, m) con D[i, j] = distancia del origen i al destino j.

    Se calcula por broadcasting (columna x fila), sin bucles. Costo en memoria:
    n*m flotantes (p. ej. 15.763 alumnos x 456 clases ~ 57 MB). El np.clip
    evita NaN por errores de redondeo cuando `a` queda levemente fuera de [0, 1].
    """
    lat1 = np.radians(np.asarray(lat1, dtype=float))[:, None]
    lng1 = np.radians(np.asarray(lng1, dtype=float))[:, None]
    lat2 = np.radians(np.asarray(lat2, dtype=float))[None, :]
    lng2 = np.radians(np.asarray(lng2, dtype=float))[None, :]
    a = (np.sin((lat2 - lat1) / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2.0) ** 2)
    return RADIO_TIERRA_KM * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def haversine_pares(lat1, lng1, lat2, lng2) -> np.ndarray:
    """
    Distancia Haversine (km) punto a punto: D[k] = distancia entre (lat1[k], lng1[k])
    y (lat2[k], lng2[k]). Misma fórmula y radio que haversine_matrix (su diagonal),
    pero sin construir la matriz n x n. Devuelve NaN donde falte alguna coordenada.
    La usa el viewer para recalcular distancias de tablas de asignación.
    """
    lat1, lng1, lat2, lng2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lng1, lat2, lng2))
    a = (np.sin((lat2 - lat1) / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2.0) ** 2)
    return RADIO_TIERRA_KM * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def franjas_de_turno(turno: str) -> tuple:
    """
    Descompone un turno (texto ya normalizado) en las franjas horarias que ocupa.

        "mañana"        -> ("mañana",)
        "tarde"         -> ("tarde",)
        "mañana-tarde"  -> ("mañana", "tarde")   # jornada completa
        "noche"         -> ("noche",)

    Un texto sin ninguna franja reconocida se trata como una franja propia
    (equivale al comportamiento anterior: solo choca consigo mismo).
    """
    t = str(turno).strip().lower()
    fs = [f for f, claves in (("mañana", ("mañana", "manana")), ("tarde", ("tarde",)),
                              ("noche", ("noche",)))
          if any(k in t for k in claves)]
    return tuple(fs) if fs else (t,)


def normalizar_texto(s: pd.Series) -> np.ndarray:
    """
    Normaliza textos categóricos (grado, turno) antes de compararlos:
    convierte a str, quita espacios de los extremos y pasa a minúsculas.
    Así "Mañana ", "mañana" y "MAÑANA" se tratan como el mismo turno.
    """
    return s.astype(str).str.strip().str.lower().to_numpy()


class IntegratedProblem(Problem):
    """
    Problema TSSA many-objective.

    Args:
        estudiantes: DataFrame con columnas {'lat','lng','nombre','grado'}.
        docentes:    DataFrame con columnas {'lat','lng','nombre'} (+ 'grado' para f5).
        clases:      DataFrame con columnas {'lat','lng','grado','turno','capacidad',
                                             'establecimiento_id','institucion_id'}.
        n_obj:       3 (formulación del preprint), 4 (recomendada) o 5.
        objetivos:   lista explícita de claves de OBJETIVOS (tiene prioridad sobre n_obj).
        cap_min:     capacidad mínima por clase activa (0 = desactivada, como v1.3;
                     25 = formulación del preprint; 10 = CACIC 2024).
        f2_metrica:  "var" (paper) o "std" (código v1.3).
        f4_modo:     "suma" (max_est + max_doc) o "max" (max de ambos, como la v2).
        x_base:      vector x de la asignación actual (MEC); requerido por f6.
        U, V:        matrices de distancia opcionales (na x nc) y (nd x nc). Si se
                     omiten se calculan con Haversine desde lat/lng.

    Atributos precalculados que usan los operadores (operators.py) y la evaluación:
        na, nd, nc     número de estudiantes, docentes y clases.
        U, V           distancias estudiante->clase y docente->clase (km).
        cap, turno, estab, nt
                       capacidad, código de turno, código de establecimiento de
                       cada clase y cantidad de turnos distintos.
        franjas, nf    matriz booleana (nc x nf): franjas horarias que ocupa cada
                       clase (ver franjas_de_turno) y cantidad de franjas.
        g_est, g_cls, g_doc
                       grado de estudiantes, clases y docentes codificado como
                       entero sobre un vocabulario común (comparaciones rápidas).
        cand[i]        clases del mismo grado que el estudiante i, ordenadas de
                       la más cercana a la más lejana.
        doc_orden[l]   docentes ordenados por cercanía a la clase l.
        x_base, estab_base
                       asignación del MEC y establecimiento base de cada alumno (f6).
    """

    def __init__(self, estudiantes: pd.DataFrame, docentes: pd.DataFrame, clases: pd.DataFrame,
                 n_obj: int = 4, objetivos: Optional[Sequence[str]] = None,
                 cap_min: int = 0, f2_metrica: str = "var", f4_modo: str = "suma",
                 U: Optional[np.ndarray] = None, V: Optional[np.ndarray] = None,
                 x_base: Optional[np.ndarray] = None):
        if estudiantes.empty or docentes.empty or clases.empty:
            raise ValueError("❌ Los DataFrames de entrada no pueden estar vacíos")

        # Índices 0..n-1: la POSICIÓN de cada fila es la que se usa en el vector x
        # (XA[i] = posición de la clase, XD[l] = posición del docente).
        estudiantes = estudiantes.reset_index(drop=True)
        docentes = docentes.reset_index(drop=True)
        clases = clases.reset_index(drop=True)
        self._validar_dataframes(estudiantes, docentes, clases, U, V)

        self.estudiantes, self.docentes, self.clases = estudiantes, docentes, clases
        self.na = self.n_estudiantes = len(estudiantes)
        self.nd = self.n_docentes = len(docentes)
        self.nc = self.n_clases = len(clases)

        # ---------- Objetivos ----------
        # `objetivos` explícito tiene prioridad; si no, se toma el conjunto
        # predefinido de n_obj. Se guardan sentidos y etiquetas en el mismo
        # orden para que reportes y conversiones de signo sean consistentes.
        if objetivos is None:
            if n_obj not in CONJUNTOS_OBJETIVOS:
                raise ValueError(f"n_obj debe ser uno de {list(CONJUNTOS_OBJETIVOS)}")
            objetivos = CONJUNTOS_OBJETIVOS[n_obj]
        desconocidos = set(objetivos) - set(OBJETIVOS)
        if desconocidos:
            raise ValueError(f"Objetivos desconocidos: {desconocidos}")
        self.objetivos = tuple(objetivos)
        self.sentidos = tuple(OBJETIVOS[o][0] for o in self.objetivos)
        self.etiquetas = tuple(OBJETIVOS[o][1] for o in self.objetivos)
        if f2_metrica not in ("var", "std"):
            raise ValueError("f2_metrica debe ser 'var' o 'std'")
        self.f2_metrica = f2_metrica
        if f4_modo not in ("suma", "max"):
            raise ValueError("f4_modo debe ser 'suma' o 'max'")
        self.f4_modo = f4_modo
        self.cap_min = int(cap_min)
        if "f6_estabilidad" in self.objetivos and x_base is None:
            raise ValueError("f6_estabilidad requiere x_base (asignación actual del MEC)")

        # ---------- Matrices de distancia ----------
        # Se calculan UNA sola vez por problema (costo O(na*nc + nd*nc)); la
        # evaluación sólo indexa estas matrices. Pasar U y V ya calculadas
        # permite reutilizarlas entre problemas de la misma instancia.
        if U is None:
            U = haversine_matrix(estudiantes["lat"], estudiantes["lng"], clases["lat"], clases["lng"])
        if V is None:
            V = haversine_matrix(docentes["lat"], docentes["lng"], clases["lat"], clases["lng"])
        self.U = np.asarray(U, dtype=float)
        self.V = np.asarray(V, dtype=float)
        if self.U.shape != (self.na, self.nc) or self.V.shape != (self.nd, self.nc):
            raise ValueError("Dimensiones de U o V incompatibles con los datos")

        # ---------- Atributos de clases ----------
        # pd.factorize convierte cada texto en un código entero 0..k-1; turno y
        # establecimiento se comparan como enteros en la evaluación vectorizada.
        self.cap = clases["capacidad"].astype(int).to_numpy()
        self.turno, self.turnos_unicos = pd.factorize(normalizar_texto(clases["turno"]))
        self.nt = max(1, len(self.turnos_unicos))
        # Franjas horarias de cada clase (R4 se evalúa sobre franjas, no sobre
        # el texto del turno): "Mañana-Tarde" ocupa las franjas mañana y tarde.
        franjas_cls = [franjas_de_turno(t) for t in normalizar_texto(clases["turno"])]
        self.franjas_unicas = list(dict.fromkeys(f for fs in franjas_cls for f in fs))
        pos_f = {f: k for k, f in enumerate(self.franjas_unicas)}
        self.nf = max(1, len(self.franjas_unicas))
        self.franjas = np.zeros((self.nc, self.nf), dtype=bool)
        for c, fs in enumerate(franjas_cls):
            self.franjas[c, [pos_f[f] for f in fs]] = True
        self.estab, _ = pd.factorize(clases["establecimiento_id"])

        # Grados codificados sobre un vocabulario común (estudiantes, clases, docentes):
        # el mismo texto recibe el mismo entero en los tres grupos. Si los docentes
        # no tienen columna 'grado' se usa un valor centinela que no coincide con
        # ninguna clase (en ese caso f5 vale 0).
        g_est = normalizar_texto(estudiantes["grado"])
        g_cls = normalizar_texto(clases["grado"])
        g_doc = (normalizar_texto(docentes["grado"]) if "grado" in docentes.columns
                 else np.array(["__sin_grado__"] * self.nd))
        vocab = {g: k for k, g in enumerate(pd.unique(np.concatenate([g_cls, g_est, g_doc])))}
        self.g_est = np.array([vocab[g] for g in g_est])
        self.g_cls = np.array([vocab[g] for g in g_cls])
        self.g_doc = np.array([vocab[g] for g in g_doc])
        self.grados_vocab = list(vocab)

        # ---------- Candidatos (para operadores/reparación) ----------
        # cand[i]: clases compatibles (mismo grado) del estudiante i, ordenadas por distancia.
        # Se procesa un grado a la vez: un único argsort por fila sobre la
        # submatriz U[alumnos del grado, clases del grado]. Cada cand[i] es una
        # vista de esa matriz ordenada (no se copian datos por alumno).
        # Si un grado no tiene clases, se usan TODAS las clases como candidatas
        # para que el alumno tenga algún destino; g5 reportará la incompatibilidad.
        self.cand = [None] * self.na
        self.tiene_clase_grado = np.zeros(self.na, dtype=bool)
        for g in np.unique(self.g_est):
            idx_est = np.where(self.g_est == g)[0]
            cls_g = np.where(self.g_cls == g)[0]
            if len(cls_g) == 0:
                logger.warning(f"⚠️ Grado '{self.grados_vocab[g]}' sin clases: "
                               f"{len(idx_est)} estudiantes no pueden ser factibles (g5).")
                cls_g = np.arange(self.nc)
            else:
                self.tiene_clase_grado[idx_est] = True
            orden = cls_g[np.argsort(self.U[np.ix_(idx_est, cls_g)], axis=1)]
            for r, i in enumerate(idx_est):
                self.cand[i] = orden[r]
        # doc_orden[l]: docentes ordenados por cercanía a la clase l (matriz nc x nd)
        self.doc_orden = np.argsort(self.V, axis=0).T

        # ---------- Asignación base (MEC) para f6 ----------
        # estab_base[i] = establecimiento en el que el MEC ubica hoy al alumno i;
        # f6 cuenta cuántos alumnos terminan en un establecimiento distinto.
        self.x_base = None
        self.estab_base = None
        if x_base is not None:
            xb = np.clip(np.rint(np.asarray(x_base, dtype=float)), 0, None).astype(int)
            self.x_base = xb
            self.estab_base = self.estab[np.clip(xb[:self.na], 0, self.nc - 1)]

        # ---------- Límites y superclase ----------
        # XA[i] ∈ [0, nc-1] (una clase por alumno); XD[l] ∈ [0, nd], donde el
        # valor nd es el centinela "clase sin docente".
        xl = np.zeros(self.na + self.nc, dtype=int)
        xu = np.concatenate([np.full(self.na, self.nc - 1), np.full(self.nc, self.nd)]).astype(int)
        super().__init__(n_var=self.na + self.nc, n_obj=len(self.objetivos),
                         n_ieq_constr=N_RESTRICCIONES, xl=xl, xu=xu, vtype=int)

    # ------------------------------------------------------------------
    @staticmethod
    def _validar_dataframes(estudiantes, docentes, clases, U=None, V=None):
        """
        Verifica que cada DataFrame tenga las columnas mínimas. lat/lng sólo se
        exigen si hay que calcular la matriz de distancias correspondiente
        (si se pasan U y/o V ya calculadas, no hacen falta coordenadas).
        Acumula todos los faltantes y lanza un único ValueError.
        """
        req_est = {"grado"} | (set() if U is not None else {"lat", "lng"})
        req_doc = set() if V is not None else {"lat", "lng"}
        req_cls = {"grado", "turno", "capacidad", "establecimiento_id"}
        if U is None or V is None:
            req_cls |= {"lat", "lng"}
        errores = []
        for nombre, df, req in (("Estudiantes", estudiantes, req_est),
                                ("Docentes", docentes, req_doc),
                                ("Clases", clases, req_cls)):
            falt = req - set(df.columns)
            if falt:
                errores.append(f"{nombre} faltan columnas: {falt}")
        if errores:
            raise ValueError("❌ " + " | ".join(errores))

    # ------------------------------------------------------------------
    def decodificar(self, x):
        """
        Separa el vector de decisión en sus dos bloques y lo lleva a enteros válidos.

        Los operadores de pymoo trabajan con float; aquí se redondea (rint) y se
        recorta a [xl, xu] para que cualquier x (p. ej. de SBX) sea indexable.

        Returns:
            XA (na,): clase asignada a cada estudiante.
            XD (nc,): docente asignado a cada clase (nd = sin docente).
        """
        x = np.clip(np.rint(np.asarray(x, dtype=float)), self.xl, self.xu).astype(int)
        return x[:self.na], x[self.na:]

    def evaluar(self, x, detalle: bool = False):
        """
        Evalúa un vector de decisión.

        Returns:
            (F, G) en forma de minimización; con detalle=True agrega un dict
            con los valores "reportables" (maximizaciones con su signo natural).
        """
        XA, XD = self.decodificar(x)
        n_cls = np.bincount(XA, minlength=self.nc)     # alumnos por clase
        activa = n_cls > 0                             # clase abierta (tiene alumnos)
        con_doc = XD < self.nd                         # clase con docente asignado
        act_doc = activa & con_doc

        # ---------------- Restricciones ----------------
        # g1 (R2): alumnos por encima de la capacidad, sumados sobre todas las clases;
        #          si cap_min > 0, también los alumnos que FALTAN para llegar al
        #          mínimo en las clases abiertas (una clase vacía no se penaliza).
        g1 = float(np.maximum(0, n_cls - self.cap).sum())
        if self.cap_min > 0:
            g1 += float(np.maximum(0, self.cap_min - n_cls[activa]).sum())
        # g2 (R5): clases abiertas que quedaron sin docente.
        g2 = float(np.sum(activa & ~con_doc))

        idx = np.where(act_doc)[0]                      # clases activas con docente
        d_de = XD[idx]                                  # docente de cada una de ellas
        # g3 (R3): clases por encima de 2 por docente (carga = clases activas del docente).
        carga = np.bincount(d_de, minlength=self.nd)
        g3 = float(np.maximum(0, carga - 2).sum())
        # g4 (R4): conflictos de turno sobre FRANJAS horarias. ocup[d, f] = clases
        # activas del docente d que ocupan la franja f; cada clase de más en la
        # misma franja suma 1. Con turnos simples (Mañana/Tarde) coincide con la
        # versión anterior; una clase Mañana-Tarde choca con cualquier otra.
        ocup = np.zeros((self.nd, self.nf), dtype=int)
        np.add.at(ocup, d_de, self.franjas[idx].astype(int))
        g4 = float(np.maximum(0, ocup - 1).sum())
        # g5: alumnos en una clase de otro grado.
        g5 = float(np.sum(self.g_est != self.g_cls[XA]))

        # ---------------- Objetivos ----------------
        # Se calculan TODOS los objetivos del catálogo (son baratos) y al final se
        # arma F sólo con los activos, en el orden de self.objetivos.
        d_est = self.U[np.arange(self.na), XA]          # distancia de cada alumno a su clase
        d_doc = self.V[d_de, idx] if len(idx) else np.zeros(0)   # docente -> clase activa
        val = {}
        # f1 (ec. 22): promedio alumnos + promedio docentes (sólo clases activas).
        val["f1_distancia"] = d_est.mean() + (d_doc.mean() if len(d_doc) else 0.0)
        # f2 (ec. 23): dispersión de alumnos por clase sobre las nc clases
        # (las clases vacías cuentan con 0 alumnos).
        val["f2_balance"] = float(n_cls.var() if self.f2_metrica == "var" else n_cls.std())
        # f3: docentes con exactamente 2 clases activas, ambas en el mismo establecimiento.
        # Se ordenan los pares (docente, establecimiento) con lexsort; dos filas
        # consecutivas iguales = el docente tiene 2 clases en el mismo lugar.
        # Se divide por nd (TODOS los docentes), igual que en la v1.3.
        if len(idx):
            e = self.estab[idx]
            o = np.lexsort((e, d_de))
            t_s, e_s = d_de[o], e[o]
            mismo = (t_s[1:] == t_s[:-1]) & (e_s[1:] == e_s[:-1])
            docs_mismo = t_s[1:][mismo]
            val["f3_colocalizacion"] = float(np.sum(carga[docs_mismo] == 2)) / max(1, self.nd)
            # f5: proporción de clases activas con docente del mismo grado que la clase.
            val["f5_afinidad_grado"] = float(np.mean(self.g_doc[d_de] == self.g_cls[idx]))
        else:
            val["f3_colocalizacion"] = 0.0
            val["f5_afinidad_grado"] = 0.0
        # f4 (minimax): peor distancia de alumnos y de docentes, combinadas según f4_modo.
        max_doc = d_doc.max() if len(d_doc) else 0.0
        val["f4_distancia_max"] = (d_est.max() + max_doc if self.f4_modo == "suma"
                                   else max(d_est.max(), max_doc))
        # f6: proporción de alumnos cuyo establecimiento difiere del asignado por el MEC.
        if self.estab_base is not None:
            val["f6_estabilidad"] = float(np.mean(self.estab[XA] != self.estab_base))
        else:
            val["f6_estabilidad"] = 0.0

        # F en forma de minimización (los "max" negados); G en el orden de NOMBRES_RESTRICCIONES.
        F = np.array([val[o] if s == "min" else -val[o]
                      for o, s in zip(self.objetivos, self.sentidos)], dtype=float)
        G = np.array([g1, g2, g3, g4, g5], dtype=float)
        if detalle:
            # Valores para informes: objetivos con signo natural, restricciones
            # e indicadores descriptivos adicionales (no afectan la búsqueda).
            reportable = {o: val[o] for o in self.objetivos}
            reportable.update(dict(zip(NOMBRES_RESTRICCIONES, G)))
            reportable["dist_prom_est"] = float(d_est.mean())
            reportable["dist_prom_doc"] = float(d_doc.mean()) if len(d_doc) else 0.0
            reportable["dist_max_est"] = float(d_est.max())
            reportable["dist_max_doc"] = float(d_doc.max()) if len(d_doc) else 0.0
            reportable["clases_activas"] = int(activa.sum())
            reportable["docentes_usados"] = int((carga > 0).sum())
            if self.estab_base is not None:
                reportable["prop_cambio_establecimiento"] = val["f6_estabilidad"]
            return F, G, reportable
        return F, G

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Punto de entrada de pymoo: evalúa una población completa.

        X tiene forma (n_individuos, n_var). Cada fila se evalúa con evaluar()
        (ya vectorizada internamente sobre alumnos y clases) y los resultados se
        devuelven en out["F"] (n_individuos x n_obj) y out["G"] (n_individuos x 5).
        """
        F = np.empty((len(X), self.n_obj))
        G = np.empty((len(X), N_RESTRICCIONES))
        for r, x in enumerate(X):
            F[r], G[r] = self.evaluar(x)
        out["F"], out["G"] = F, G

    # ------------------------------------------------------------------
    def a_reportable(self, F: np.ndarray) -> np.ndarray:
        """
        Convierte F (forma de minimización) a valores con signo natural.

        Acepta un vector (un individuo) o una matriz (un frente); siempre
        devuelve una matriz 2D. Sólo cambia el signo de los objetivos "max".
        """
        F = np.atleast_2d(np.asarray(F, dtype=float)).copy()
        for k, s in enumerate(self.sentidos):
            if s == "max":
                F[:, k] = -F[:, k]
        return F

    @staticmethod
    def _hav(loc1, loc2) -> float:
        """Distancia Haversine (km) entre dos puntos (compatibilidad con v1.3)."""
        return float(haversine_matrix([loc1[0]], [loc1[1]], [loc2[0]], [loc2[1]])[0, 0])
