# ================================================================
# instancia.py
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
# Fuentes de datos para construir instancias del problema, SIN modificar
# los datos de entrada:
#   - cargar_desde_db():      PostgreSQL (usa database.py del proyecto).
#   - cargar_desde_csv(dir):  snapshot en CSV (reproducibilidad de la tesis).
#   - exportar_csv(inst, dir): guarda un snapshot de la BD en CSV.
#   - cargar_desde_sql(path): parsea los INSERT de asignacion_mec.sql
#                             (instancia de ejemplo, sin necesitar PostgreSQL).
#   - generar_instancia_sintetica(): SOLO para pruebas de escalabilidad.
#   - subinstancia_por_grado(): escenarios por grado, como en CACIC 2024
#                               (filtra filas, no altera valores).
#   - x_desde_asignacion(): convierte la asignación actual del MEC
#                            (tabla asignacion_mec) en un vector x evaluable.
#
# Una "instancia" es un dict con DataFrames:
#   estudiantes, docentes, clases, asignacion (esta última puede ser None).
# ================================================================

import ast
import logging
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger("instancia")

# Columnas (y orden) del DataFrame de clases, idénticas a las que devuelve
# database.cargar_datos_desde_db(); así todas las fuentes producen lo mismo.
COLS_CLASES = ["clase_id", "grado", "turno", "capacidad", "establecimiento_id",
               "lat", "lng", "nombre_institucion", "institucion_id"]


# ------------------------------------------------------------------
# PostgreSQL
# ------------------------------------------------------------------
def cargar_desde_db():
    """
    Carga la instancia desde PostgreSQL (tablas del proyecto).

    Usa database.cargar_datos_desde_db() para estudiantes, docentes y clases, y
    lee aparte la tabla baseline asignacion_mec. Si esa tabla no existe o falla
    la lectura, 'asignacion' queda en None (no es obligatoria para optimizar).
    """
    from database import cargar_datos_desde_db, engine  # import diferido
    est, doc, cls, _ = cargar_datos_desde_db()
    try:
        with engine.connect() as conn:
            asig = pd.read_sql("SELECT * FROM asignacion_mec", con=conn)
    except Exception as e:  # la tabla baseline es opcional
        logger.warning(f"No se pudo leer asignacion_mec: {e}")
        asig = None
    return {"estudiantes": est, "docentes": doc, "clases": cls, "asignacion": asig}


# ------------------------------------------------------------------
# CSV
# ------------------------------------------------------------------
def exportar_csv(inst, carpeta):
    """
    Guarda un snapshot de la instancia en CSV (un archivo por tabla:
    estudiantes.csv, docentes.csv, clases.csv y, si existe, asignacion.csv).
    Congelar los datos así permite reproducir los experimentos de la tesis
    aunque la BD cambie después.
    """
    os.makedirs(carpeta, exist_ok=True)
    for k in ("estudiantes", "docentes", "clases", "asignacion"):
        if inst.get(k) is not None:
            inst[k].to_csv(os.path.join(carpeta, f"{k}.csv"), index=False)


def cargar_desde_csv(carpeta):
    """Lee un snapshot creado con exportar_csv(); asignacion.csv es opcional."""
    leer = lambda n: pd.read_csv(os.path.join(carpeta, f"{n}.csv"))
    asig_path = os.path.join(carpeta, "asignacion.csv")
    return {"estudiantes": leer("estudiantes"), "docentes": leer("docentes"),
            "clases": leer("clases"),
            "asignacion": pd.read_csv(asig_path) if os.path.exists(asig_path) else None}


# ------------------------------------------------------------------
# Parser del script SQL de ejemplo (asignacion_mec.sql)
# ------------------------------------------------------------------
# Captura: tabla, lista de columnas y el bloque de VALUES hasta el ';' final
# (re.S: el bloque puede ocupar varias líneas; re.I: sin distinguir mayúsculas).
_INSERT_RE = re.compile(r"INSERT INTO\s+(\w+)\s*\(([^)]*)\)\s*VALUES\s*(.*?);", re.S | re.I)


def _parsear_inserts(sql_text):
    """
    Convierte las sentencias INSERT del script SQL en DataFrames {tabla: df}.

    Cada bloque VALUES (...), (...) se interpreta como una lista de tuplas de
    Python con ast.literal_eval (seguro: sólo acepta literales). Varios INSERT
    sobre la misma tabla se concatenan. Se agrega la columna 'id' 1..n para
    imitar el SERIAL de PostgreSQL (mismo orden de inserción).
    """
    # quitar comentarios "--" (el script no tiene "--" dentro de strings)
    sin_coment = "\n".join(l.split("--")[0] for l in sql_text.splitlines())
    tablas = {}
    for tabla, cols, valores in _INSERT_RE.findall(sin_coment):
        cols = [c.strip() for c in cols.split(",")]
        filas = ast.literal_eval("[" + valores.strip() + "]")
        filas = [f if isinstance(f, tuple) else (f,) for f in filas]   # tablas de 1 columna
        df = pd.DataFrame(filas, columns=cols)
        tablas[tabla] = pd.concat([tablas[tabla], df], ignore_index=True) if tabla in tablas else df
    for df in tablas.values():  # SERIAL: ids 1..n en orden de inserción
        df.insert(0, "id", np.arange(1, len(df) + 1))
    return tablas


def cargar_desde_sql(path="asignacion_mec.sql"):
    """
    Reproduce en memoria lo que database.cargar_datos_desde_db() leería de la BD,
    a partir del script asignacion_mec.sql (instancia de ejemplo, sin PostgreSQL).

    Las clases se unen con establecimientos (lat/lng) e instituciones
    (nombre_institucion), igual que el JOIN de database.py.
    """
    with open(path, encoding="utf-8") as f:
        t = _parsear_inserts(f.read())
    est = t["estudiantes"].rename(columns={"id": "estudiante_id"})
    doc = t["docentes"].rename(columns={"id": "docente_id"})
    cls = (t["clases"].rename(columns={"id": "clase_id"})
           .merge(t["establecimientos"].rename(columns={"id": "establecimiento_id"}),
                  on="establecimiento_id")
           .merge(t["instituciones"][["id", "nombre"]].rename(
                  columns={"id": "institucion_id", "nombre": "nombre_institucion"}),
                  on="institucion_id")
           .sort_values("clase_id").reset_index(drop=True))
    asig = t.get("asignacion_mec")
    return {"estudiantes": est, "docentes": doc, "clases": cls[COLS_CLASES], "asignacion": asig}


# ------------------------------------------------------------------
# Instancia sintética (solo pruebas de escalabilidad / depuración)
# ------------------------------------------------------------------
def generar_instancia_sintetica(n_establecimientos=20, n_estudiantes=2000, n_docentes=None,
                                grados=("1ro", "2do", "3ro"), turnos=("Mañana", "Tarde"),
                                capacidad=40, holgura=1.15, seed=0,
                                centro=(-25.51, -54.65), radio_km=15.0):
    """
    Genera una instancia aleatoria con estructura similar a Ciudad del Este.
    NO usar para resultados de la tesis: solo para probar el pipeline y la escala.

    Args:
        n_establecimientos, n_estudiantes: tamaño de la instancia.
        n_docentes: por defecto 0,75 x nº de clases (cada docente puede tener 2).
        grados, turnos: valores categóricos (se reparten en forma cíclica).
        capacidad: cupo de cada clase.
        holgura: cupo total / demanda por grado (1,15 = 15 % de cupos libres).
        seed: semilla del generador (instancia reproducible).
        centro, radio_km: centro geográfico y dispersión de los establecimientos.
    """
    rng = np.random.default_rng(seed)
    km = 1 / 111.0          # ≈ grados de latitud por km (aproximación suficiente aquí)
    # establecimientos: nube gaussiana alrededor del centro
    est_lat = centro[0] + rng.normal(0, radio_km * km / 2, n_establecimientos)
    est_lng = centro[1] + rng.normal(0, radio_km * km / 2, n_establecimientos)

    # alumnos: cada uno cerca (σ = 2,5 km) de un establecimiento al azar
    alumnos = []
    for i in range(n_estudiantes):
        k = rng.integers(n_establecimientos)
        alumnos.append((i + 1, f"Est {i + 1}", grados[i % len(grados)],
                        est_lat[k] + rng.normal(0, 2.5 * km), est_lng[k] + rng.normal(0, 2.5 * km)))
    est = pd.DataFrame(alumnos, columns=["estudiante_id", "nombre", "grado", "lat", "lng"])

    # clases: suficientes para absorber la demanda por grado (con holgura)
    filas, cid = [], 1
    for g in grados:
        demanda = int((est.grado == g).sum())
        n_cls = int(np.ceil(demanda * holgura / capacidad))
        for j in range(n_cls):
            k = rng.integers(n_establecimientos)
            filas.append((cid, g, turnos[j % len(turnos)], capacidad, k + 1, est_lat[k], est_lng[k],
                          f"Institución {k + 1}", k + 1))
            cid += 1
    cls = pd.DataFrame(filas, columns=COLS_CLASES)

    # docentes: más dispersos (σ = 4 km) alrededor de establecimientos al azar
    n_docentes = n_docentes or int(np.ceil(len(cls) * 0.75))
    ks = rng.integers(n_establecimientos, size=n_docentes)
    doc = pd.DataFrame({
        "docente_id": np.arange(1, n_docentes + 1),
        "nombre": [f"Doc {j + 1}" for j in range(n_docentes)],
        "grado": [grados[j % len(grados)] for j in range(n_docentes)],
        "lat": est_lat[ks] + rng.normal(0, 4 * km, n_docentes),
        "lng": est_lng[ks] + rng.normal(0, 4 * km, n_docentes),
    })

    # asignación "status quo" sintética: clase aleatoria del grado, docente aleatorio
    asig = []
    for _, a in est.iterrows():
        c = cls[cls.grado == a.grado].sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        asig.append((a.estudiante_id, int(rng.integers(1, n_docentes + 1)), c.establecimiento_id,
                     c.institucion_id, c.grado, "A", c.turno))
    asig = pd.DataFrame(asig, columns=["estudiante_id", "docente_id", "establecimiento_id",
                                       "institucion_id", "grado", "seccion", "turno"])
    return {"estudiantes": est, "docentes": doc, "clases": cls, "asignacion": asig}


# ------------------------------------------------------------------
# Escenarios por grado
# ------------------------------------------------------------------
def grados_de(inst):
    """Lista ordenada de grados distintos que aparecen en las clases."""
    return sorted(inst["clases"]["grado"].astype(str).str.strip().unique())


def subinstancia_por_grado(inst, grado, filtrar_docentes=True):
    """
    Sub-instancia de un solo grado (escenario "por-grado", como CACIC 2024).

    Filtra filas; no modifica valores. Estudiantes y clases: sólo las del grado.
    Docentes: los de ese grado si la columna existe y hay alguno (si no, todos).
    Asignación MEC: sólo las filas de los estudiantes que quedan.
    """
    norm = lambda s: s.astype(str).str.strip()
    est = inst["estudiantes"][norm(inst["estudiantes"]["grado"]) == grado]
    cls = inst["clases"][norm(inst["clases"]["grado"]) == grado]
    doc = inst["docentes"]
    if filtrar_docentes and "grado" in doc.columns:
        doc_g = doc[norm(doc["grado"]) == grado]
        if len(doc_g):
            doc = doc_g
    asig = inst.get("asignacion")
    if asig is not None:
        asig = asig[asig["estudiante_id"].isin(est["estudiante_id"])]
    r = lambda d: d.reset_index(drop=True)
    return {"estudiantes": r(est), "docentes": r(doc), "clases": r(cls),
            "asignacion": None if asig is None else r(asig)}


# ------------------------------------------------------------------
# Asignación actual (MEC) -> vector x
# ------------------------------------------------------------------
def x_desde_asignacion(problem, asig):
    """
    Construye el vector x = [XA | XD] a partir de la tabla asignacion_mec.

    - Estudiante -> clase: por (establecimiento_id, grado, turno); si hay varias
      clases con la misma clave, cada sección distinta de esa clave va a una
      clase distinta según su ORDEN dentro de la clave (no según su letra):
      con secciones {A, C} -> A a la 1ª clase y C a la 2ª. Así letras no
      consecutivas ("B" sola, o "A" y "C") no juntan dos secciones en una clase.
      Si no hay coincidencia exacta se busca (establecimiento, grado) y luego
      la clase del mismo grado más cercana al establecimiento (se informa).
    - Clase -> docente: el docente más frecuente entre las filas de esa clase.
      (La tabla asigna docente por estudiante; puede violar R3/R4: se reporta.)

    Returns:
        x (np.ndarray), info (dict con conteos de mapeo).
    """
    cls = problem.clases
    norm = lambda v: str(v).strip().lower()
    # Índices de búsqueda de clases (posiciones en problem.clases):
    #   clave3: (establecimiento, grado, turno) -> [clases]
    #   clave2: (establecimiento, grado)        -> [clases]  (respaldo sin turno)
    clave3, clave2 = {}, {}
    for k, c in cls.iterrows():
        clave3.setdefault((int(c.establecimiento_id), norm(c.grado), norm(c.turno)), []).append(k)
        clave2.setdefault((int(c.establecimiento_id), norm(c.grado)), []).append(k)
    # id de BD -> posición en el vector x
    pos_est = {int(e): i for i, e in enumerate(problem.estudiantes["estudiante_id"])}
    # Orden de las secciones dentro de cada clave (establecimiento, grado, turno):
    # rango_sec[clave][sección] = 0, 1, 2, ... (secciones ordenadas alfabéticamente)
    sec_txt = lambda v: str(v).strip().upper() if pd.notna(v) and str(v).strip() else "A"
    secs_por_clave = {}
    for _, a in asig.iterrows():
        secs_por_clave.setdefault((int(a.establecimiento_id), norm(a.grado), norm(a.turno)),
                                  set()).add(sec_txt(a.get("seccion", "A")))
    rango_sec = {k: {s: r for r, s in enumerate(sorted(v))} for k, v in secs_por_clave.items()}
    pos_doc = {int(d): j for j, d in enumerate(problem.docentes["docente_id"])}
    coords = cls[["lat", "lng"]].to_numpy(float)
    estab_coord = cls.groupby("establecimiento_id")[["lat", "lng"]].first()

    XA = np.full(problem.na, -1, dtype=int)   # -1 = alumno aún sin clase
    docentes_por_clase = {}                     # clase -> docentes vistos en sus filas
    info = Counter()                            # cuántas filas se mapearon de cada forma
    for _, a in asig.iterrows():
        i = pos_est.get(int(a.estudiante_id))
        if i is None:
            info["estudiante_no_encontrado"] += 1
            continue
        e, g, t = int(a.establecimiento_id), norm(a.grado), norm(a.turno)
        opciones = clave3.get((e, g, t))
        if opciones:
            # r-ésima sección de la clave -> r-ésima clase de la clave. El módulo
            # sólo actúa si hay más secciones que clases (se informa aparte).
            r = rango_sec[(e, g, t)][sec_txt(a.get("seccion", "A"))]
            if r >= len(opciones):
                info["seccion_sin_clase_propia"] += 1
            k = opciones[r % len(opciones)]
            info["exacto"] += 1
        elif clave2.get((e, g)):
            k = clave2[(e, g)][0]
            info["sin_turno"] += 1
        else:
            # último recurso: clase del grado del alumno más cercana (distancia
            # euclídea en grados; sólo se usa para elegir, no para reportar)
            mismas = np.where(problem.g_cls == problem.g_est[i])[0]
            if len(mismas) == 0:
                mismas = np.arange(problem.nc)
            ref = estab_coord.loc[e].to_numpy(float) if e in estab_coord.index else coords[mismas[0]]
            k = mismas[np.argmin(np.linalg.norm(coords[mismas] - ref, axis=1))]
            info["aproximado"] += 1
        XA[i] = k
        j = pos_doc.get(int(a.docente_id))
        if j is not None:
            docentes_por_clase.setdefault(k, []).append(j)

    # alumnos que no aparecen en asignacion_mec -> su clase compatible más cercana
    sin = np.where(XA < 0)[0]
    if len(sin):
        info["estudiante_sin_asignacion"] = len(sin)
        for i in sin:
            XA[i] = problem.cand[i][0]
    # docente de cada clase = el más frecuente en sus filas (moda); clases sin
    # filas quedan sin docente (valor nd)
    XD = np.full(problem.nc, problem.nd, dtype=int)
    for k, lista in docentes_por_clase.items():
        XD[k] = Counter(lista).most_common(1)[0][0]
    return np.concatenate([XA, XD]), dict(info)
