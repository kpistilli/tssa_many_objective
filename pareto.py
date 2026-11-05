# ================================================================
# pareto.py
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
# El resultado de una corrida ES un Conjunto Pareto (Algoritmo 1 del preprint:
# "Salida: Pareto Set"). La v1.3 lo colapsaba a una única solución. Este módulo:
#   1) ParetoArchive: archivo externo no dominado y factible acumulado en TODAS
#      las generaciones (con NSGA-III la última población tiene muchas menos
#      no dominadas que las halladas durante la búsqueda).
#   2) extraer_pareto_set(): P* (X) y PF* (F) completos, factibles y sin duplicados.
#   3) Decisión multicriterio A-POSTERIORI (separada de la optimización):
#      compromiso/ASF, punto rodilla, pseudo-pesos, lexicográfico, min_fk.
#   4) Decodificación de una solución a filas de asignación con SECCIÓN derivada.
#   5) Persistencia del conjunto completo: CSV/NPZ y PostgreSQL
#      (opt_run, pareto_solucion, asignacion_pareto + asignacion_mec_opt compatible).
# ================================================================

import csv
import io
import json
import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.core.callback import Callback
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

logger = logging.getLogger("pareto")

# Métodos de decisión a posteriori con nombre fijo. Además se acepta "min_fk"
# (k = 1..M: la mejor solución en el objetivo k) y "domina_mec:<método>"
# (el método aplicado sólo a las soluciones que dominan al MEC); ver
# seleccionar_solucion().
METODOS_DECISION = ("compromiso", "asf", "rodilla", "pseudo_pesos", "lexicografico", "domina_mec")


def ref_mec(problem):
    """F (forma de minimización) de la asignación del MEC, o None si el problema no tiene x_base."""
    x = getattr(problem, "x_base", None)
    return None if x is None else problem.evaluar(x)[0]


# =====================================================================
# 1) Archivo externo no dominado
# =====================================================================
class ParetoArchive(Callback):
    """
    Conjunto no dominado y factible acumulado durante toda la corrida.

    Args:
        max_size: tamaño máximo; si se supera se trunca por distancia de crowding.
        problema_eval: si se indica, las soluciones se re-evalúan en ese problema
            antes de archivarse (útil para "algoritmo@k": se optimiza con k
            objetivos pero el archivo se mantiene en el espacio del experimento).
        cada: archivar cada k generaciones (1 = todas).

    Atributos:
        X, F: soluciones archivadas (X int32; F en forma de minimización); None al inicio.
        historial: lista de {"n_gen", "n_pareto"} para seguir el crecimiento.

    Se usa como callback de pymoo: notify() se ejecuta al final de cada generación.
    """

    def __init__(self, max_size=500, problema_eval=None, cada=1):
        super().__init__()
        self.max_size, self.pe, self.cada = int(max_size), problema_eval, max(1, int(cada))
        self.X = self.F = None
        self.historial = []

    def notify(self, algorithm):
        """Agrega al archivo las soluciones FACTIBLES (CV <= 0) de la población actual."""
        # se archiva en la generación 1 y luego cada `cada` generaciones
        if algorithm.n_gen % self.cada and algorithm.n_gen != 1:
            return
        pop = algorithm.pop
        if pop is None or len(pop) == 0:
            return
        ok = pop.get("CV").ravel() <= 0
        if not ok.any():
            return
        X = np.atleast_2d(pop.get("X")[ok])
        if self.pe is not None:
            # re-evaluación en el espacio de objetivos del experimento ("algoritmo@k")
            F = np.array([self.pe.evaluar(x)[0] for x in X])
        else:
            F = np.atleast_2d(pop.get("F")[ok])
        self.agregar(X, F)
        self.historial.append({"n_gen": algorithm.n_gen, "n_pareto": len(self.F)})

    def agregar(self, X, F):
        """
        Une (X, F) al archivo y lo deja en su estado invariante:
        sólo no dominados, sin vectores F repetidos y con a lo sumo max_size filas.
        """
        # X se guarda como int32 (índices de clase/docente): la mitad de memoria
        # que float64 (con ~16.000 variables y 500 soluciones: ~32 MB en vez de ~65 MB)
        X, F = np.rint(np.atleast_2d(X)).astype(np.int32), np.atleast_2d(F)
        if self.F is None:
            self.X, self.F = X, F.copy()
        else:
            self.X, self.F = np.vstack([self.X, X]), np.vstack([self.F, F])
        # 1) conservar sólo el primer frente (no dominados)
        nd = NonDominatedSorting().do(self.F, only_non_dominated_front=True)
        self.X, self.F = self.X[nd], self.F[nd]
        # 2) quitar duplicados en el espacio de objetivos (redondeo a 1e-9 para
        #    tolerar ruido numérico); np.sort mantiene el orden de llegada
        _, u = np.unique(np.round(self.F, 9), axis=0, return_index=True)
        u = np.sort(u)
        self.X, self.F = self.X[u], self.F[u]
        # 3) si se excede max_size, conservar las de mayor distancia de crowding
        #    (las más aisladas), para no perder la diversidad del frente
        if len(self.F) > self.max_size:
            cd = get_crowding_function("cd").do(self.F, n_remove=len(self.F) - self.max_size)
            keep = np.argsort(-cd, kind="stable")[: self.max_size]
            self.X, self.F = self.X[keep], self.F[keep]


# =====================================================================
# 2) Extracción del Conjunto Pareto
# =====================================================================
def extraer_pareto_set(problem, result=None, archivo=None):
    """
    Une el archivo externo y la población final (factibles), y devuelve el
    Conjunto Pareto P* y el Frente PF* sin duplicados.

    Returns: dict(X, F, n_soluciones, ideal, nadir, obj_names) o None si no hay factibles.

    F queda en forma de minimización; ideal/nadir son el mínimo y el máximo por
    columna de ese F. X se devuelve como int32 (índices de clase/docente).
    Si el archivo re-evalúa en otro espacio ("@k", archivo.pe), no se mezcla
    la población final porque sus F están en un espacio de objetivos distinto.
    """
    Xs, Fs = [], []
    if archivo is not None and archivo.F is not None and len(archivo.F):
        Xs.append(archivo.X)
        Fs.append(archivo.F)
    pop = getattr(result, "pop", None) if result is not None else None
    if pop is not None and len(pop) and (archivo is None or archivo.pe is None):
        ok = pop.get("CV").ravel() <= 0
        if ok.any():
            Xs.append(pop.get("X")[ok])
            Fs.append(pop.get("F")[ok])
    if not Fs:
        return None
    X, F = np.vstack(Xs), np.vstack(Fs)
    nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
    X, F = X[nd], F[nd]
    _, u = np.unique(np.round(F, 9), axis=0, return_index=True)
    u = np.sort(u)
    X, F = X[u], F[u]
    nombres = list(getattr(problem, "objetivos", [f"f{k + 1}" for k in range(F.shape[1])]))
    return {"X": np.rint(X).astype(np.int32), "F": F, "n_soluciones": len(F),
            "ideal": F.min(0), "nadir": F.max(0), "obj_names": nombres[:F.shape[1]]}


def frente_a_dataframe(problem, ps):
    """PF* con valores en signo natural (maximizaciones positivas)."""
    R = problem.a_reportable(ps["F"]) if hasattr(problem, "a_reportable") else ps["F"]
    df = pd.DataFrame(R, columns=ps["obj_names"])
    df.insert(0, "solucion_idx", np.arange(len(df)))
    return df


# =====================================================================
# 3) Decisión multicriterio a posteriori
# =====================================================================
def _normalizar(F):
    """
    Normalización min–max por columna al rango [0, 1] usando el ideal y el nadir
    del PROPIO frente (0 = mejor valor, 1 = peor). Columnas constantes -> 0.
    """
    F = np.atleast_2d(F)
    ideal, nadir = F.min(0), F.max(0)
    return (F - ideal) / np.where(nadir - ideal > 0, nadir - ideal, 1.0)


def seleccionar_solucion(F, metodo="compromiso", pesos=None, ref=None):
    """
    Índice de una solución del frente F (forma de minimización).

    ref: F de una solución de referencia (la asignación del MEC); sólo la usa
    "domina_mec".

    metodo:
      - "compromiso": Tchebycheff/ASF con pesos iguales sobre F normalizado.
      - "asf": ASF con `pesos` (importancia relativa de cada objetivo).
      - "rodilla": punto rodilla (más alejado, hacia el ideal, del hiperplano
        que pasa por las soluciones extremas).
      - "pseudo_pesos": pseudo-pesos de Deb más cercanos a `pesos`.
      - "lexicografico": orden F[:,0], F[:,1], ... (comportamiento de la v1.3).
      - "min_fk" (k = 1..M): mejor en el objetivo k; desempata lexicográficamente.
      - "domina_mec" o "domina_mec:<método>": entre las soluciones que DOMINAN
        a `ref` (iguales o mejores en todos los objetivos y estrictamente mejores
        en alguno), la que elige <método> (por defecto "compromiso"), aplicado y
        normalizado sólo sobre ese subconjunto. Es la recomendación que mejora a
        la asignación actual en todos los criterios a la vez. Si ninguna la
        domina (o no hay `ref`), se usa <método> sobre todo el frente y se
        registra una advertencia.
    """
    F = np.atleast_2d(F)
    if metodo.startswith("domina_mec"):
        sub = metodo.split(":", 1)[1] if ":" in metodo else "compromiso"
        if ref is None:
            logger.warning("⚠️ domina_mec sin asignación del MEC de referencia: se usa "
                           f"'{sub}' sobre todo el frente")
            return seleccionar_solucion(F, sub, pesos)
        ref = np.asarray(ref, dtype=float)
        dom = np.where(np.all(F <= ref, axis=1) & np.any(F < ref, axis=1))[0]
        if len(dom) == 0:
            logger.warning("⚠️ Ninguna solución del frente domina al MEC: se usa "
                           f"'{sub}' sobre todo el frente")
            return seleccionar_solucion(F, sub, pesos)
        logger.info(f"domina_mec: {len(dom)} de {len(F)} soluciones dominan al MEC; "
                    f"se elige con '{sub}' entre ellas")
        return int(dom[seleccionar_solucion(F[dom], sub, pesos)])
    M = F.shape[1]
    if len(F) == 1:
        return 0
    Fn = _normalizar(F)
    # pesos normalizados a suma 1 (iguales si no se indican)
    w = np.ones(M) / M if pesos is None else np.asarray(pesos, float) / np.sum(pesos)
    if metodo in ("compromiso", "asf"):
        # Tchebycheff: minimizar el MAYOR desvío ponderado respecto del ideal
        ww = np.ones(M) / M if metodo == "compromiso" else w
        return int(np.argmin((Fn * np.maximum(ww, 1e-12)).max(axis=1)))
    if metodo == "rodilla":
        # hiperplano n·f = 1 que pasa por las M soluciones extremas (la mejor en
        # cada objetivo); la rodilla es la más "hundida" hacia el ideal (menor
        # distancia con signo). Si el sistema es singular se usa la menor suma.
        ext = Fn[np.argmin(Fn, axis=0)]
        try:
            n = np.linalg.lstsq(ext, np.ones(M), rcond=None)[0]
            if not np.all(np.isfinite(n)) or np.linalg.norm(n) == 0:
                raise np.linalg.LinAlgError
            d = (Fn @ n - 1.0) / np.linalg.norm(n)
            return int(np.argmin(d))
        except np.linalg.LinAlgError:
            return int(np.argmin(Fn.sum(axis=1)))
    if metodo == "pseudo_pesos":
        # pseudo-peso de cada objetivo = cuán cerca está la solución de su mejor
        # valor, relativo a los demás objetivos; se elige la más parecida a `w`
        pw = (1.0 - Fn) / np.maximum((1.0 - Fn).sum(axis=1, keepdims=True), 1e-12)
        return int(np.argmin(np.abs(pw - w).sum(axis=1)))
    if metodo == "lexicografico":
        return int(np.lexsort(tuple(F[:, k] for k in reversed(range(M))))[0])
    if metodo.startswith("min_f"):
        # "min_f1" -> objetivo 0; empates se resuelven por los demás en orden
        j = int(metodo[5:]) - 1
        orden = [j] + [k for k in range(M) if k != j]
        return int(np.lexsort(tuple(F[:, k] for k in reversed(orden)))[0])
    raise ValueError(f"Método de decisión desconocido: {metodo}. Opciones: {METODOS_DECISION}, "
                     "min_fk o domina_mec:<método>")


def etiquetar_decisiones(F, metodos=("compromiso", "rodilla", "pseudo_pesos")):
    """
    Aplica varios métodos de decisión al mismo frente y agrupa sus elecciones.

    Returns:
        dict {indice_solución: "compromiso,rodilla"}: si dos métodos eligen la
        misma solución, sus nombres se concatenan con coma.
    """
    etiquetas = {}
    for m in metodos:
        i = seleccionar_solucion(F, m)
        etiquetas[i] = f"{etiquetas[i]},{m}" if i in etiquetas else m
    return etiquetas


# =====================================================================
# 4) Decodificación -> filas de asignación
# =====================================================================
def derivar_seccion(clases):
    """
    La tabla `clases` no tiene `seccion`, pero la formulación define
    C_j = [G, T, S, I, E]. Se deriva la letra por orden dentro de cada
    (establecimiento, grado, turno). No requiere datos nuevos.
    """
    # cumcount numera 0, 1, 2... las clases de cada grupo según su orden en la
    # tabla: 0 -> "A", 1 -> "B"...; a partir de la 27ª se usa "A26", "A27"...
    orden = clases.groupby(["establecimiento_id", "grado", "turno"]).cumcount()
    return orden.map(lambda k: chr(ord("A") + k) if k < 26 else f"A{k}")


def decodificar_asignaciones(problem, x):
    """
    Traduce un vector x a una fila por estudiante, con el mismo formato que
    asignacion_mec: estudiante_id, docente_id, establecimiento_id,
    institucion_id, grado, sección (derivada), turno y distancia (km, 2 dec.).

    El docente de cada alumno es el docente de su clase (XD[XA[i]]). Si la
    clase no tiene docente (sólo en soluciones infactibles) se usa el docente
    más cercano para que la fila siga siendo insertable en la BD.
    Si los DataFrames no tienen estudiante_id/docente_id se usa su 1ª columna.
    """
    XA, XD = problem.decodificar(x)
    cls = problem.clases
    if "seccion" not in cls.columns:
        cls = cls.assign(seccion=derivar_seccion(cls).to_numpy())
    doc_de_clase = XD[XA].copy()
    sin = doc_de_clase >= problem.nd
    if sin.any():  # clase sin docente (no debería ocurrir en soluciones factibles)
        doc_de_clase[sin] = problem.doc_orden[XA[sin], 0]
    sub = cls.iloc[XA].reset_index(drop=True)          # datos de la clase de cada alumno
    col_est = "estudiante_id" if "estudiante_id" in problem.estudiantes.columns else problem.estudiantes.columns[0]
    col_doc = "docente_id" if "docente_id" in problem.docentes.columns else problem.docentes.columns[0]
    return pd.DataFrame({
        "estudiante_id": problem.estudiantes[col_est].to_numpy(),
        "docente_id": problem.docentes[col_doc].to_numpy()[doc_de_clase],
        "clase_idx": XA,
        "establecimiento_id": sub["establecimiento_id"].to_numpy(),
        "institucion_id": sub["institucion_id"].to_numpy() if "institucion_id" in sub else 0,
        "grado": sub["grado"].to_numpy(),
        "seccion": sub["seccion"].to_numpy(),
        "turno": sub["turno"].to_numpy(),
        "distancia": np.round(problem.U[np.arange(problem.na), XA], 2),
    })


# =====================================================================
# 5) Persistencia
# =====================================================================
def exportar_pareto(problem, ps, carpeta, run_id, guardar_asignaciones="seleccion",
                    indices=None, etiquetas=None):
    """
    Guarda en carpeta/run_id:
      pareto_front.csv (PF*, signo natural + etiqueta de decisión),
      pareto_set.npz   (P* comprimido en binario: mucho más liviano que CSV),
      metadata.json y, opcionalmente, asignaciones_solXXXX.csv.gz.
    guardar_asignaciones: 'ninguna' | 'seleccion' (índices/etiquetas) | 'todas'.
    indices: soluciones a exportar con 'seleccion' (por defecto, las etiquetadas).
    etiquetas: dict {índice: métodos}; si es None se calcula con etiquetar_decisiones().

    Returns:
        dict de metadatos (el mismo que se escribe en metadata.json).
    """
    out = Path(carpeta) / run_id
    out.mkdir(parents=True, exist_ok=True)
    etiquetas = etiquetas if etiquetas is not None else etiquetar_decisiones(ps["F"])
    df = frente_a_dataframe(problem, ps)
    df["seleccionada"] = df["solucion_idx"].map(etiquetas).fillna("")
    df.to_csv(out / "pareto_front.csv", index=False)
    np.savez_compressed(out / "pareto_set.npz", X=ps["X"], F=ps["F"])
    meta = {"run_id": run_id, "n_soluciones": int(ps["n_soluciones"]),
            "obj_names": ps["obj_names"], "ideal": np.asarray(ps["ideal"]).tolist(),
            "nadir": np.asarray(ps["nadir"]).tolist(), "n_var": int(ps["X"].shape[1]),
            "seleccionadas": {str(k): v for k, v in etiquetas.items()}}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if guardar_asignaciones != "ninguna":
        idxs = range(len(ps["X"])) if guardar_asignaciones == "todas" else (indices or list(etiquetas))
        for i in idxs:
            decodificar_asignaciones(problem, ps["X"][i]).to_csv(
                out / f"asignaciones_sol{i:04d}.csv.gz", index=False)
    return meta


# Esquema de persistencia del Conjunto Pareto (idempotente: IF NOT EXISTS).
#   opt_run            1 fila por corrida (parámetros, ideal/nadir en signo natural)
#   pareto_solucion    1 fila por solución del frente (vector f en signo natural)
#   asignacion_pareto  asignaciones detalladas de las soluciones elegidas (o de todas)
# Borrar una corrida de opt_run borra en cascada sus soluciones y asignaciones.
DDL = """
CREATE TABLE IF NOT EXISTS opt_run (
    run_id         TEXT PRIMARY KEY,
    algoritmo      TEXT NOT NULL,
    pop_size       INTEGER, n_gen INTEGER, seed INTEGER,
    n_obj          INTEGER, n_estudiantes INTEGER, n_docentes INTEGER, n_clases INTEGER,
    n_soluciones   INTEGER,
    obj_names      TEXT[],
    ideal          DOUBLE PRECISION[],
    nadir          DOUBLE PRECISION[],
    fecha          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS pareto_solucion (
    id             SERIAL PRIMARY KEY,
    run_id         TEXT NOT NULL REFERENCES opt_run(run_id) ON DELETE CASCADE,
    solucion_idx   INTEGER NOT NULL,
    f              DOUBLE PRECISION[] NOT NULL,
    seleccionada   TEXT,
    UNIQUE (run_id, solucion_idx)
);
CREATE TABLE IF NOT EXISTS asignacion_pareto (
    id                 SERIAL PRIMARY KEY,
    run_id             TEXT NOT NULL REFERENCES opt_run(run_id) ON DELETE CASCADE,
    solucion_idx       INTEGER NOT NULL,
    estudiante_id      INTEGER NOT NULL,
    docente_id         INTEGER NOT NULL,
    establecimiento_id INTEGER NOT NULL,
    institucion_id     INTEGER NOT NULL,
    grado              VARCHAR(20), seccion VARCHAR(10), turno VARCHAR(12),
    distancia          DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS ix_asig_pareto_run ON asignacion_pareto (run_id, solucion_idx);
"""


# Columnas de una fila de asignación (mismo formato que asignacion_mec).
COLS_ASIGNACION = ["estudiante_id", "docente_id", "establecimiento_id", "institucion_id",
                   "grado", "seccion", "turno", "distancia"]


def _tabla_asignacion(df, **fijas):
    """
    Prepara un DataFrame de decodificar_asignaciones() para cargarlo en la BD:
    ids como enteros, textos como str y distancia como float (mismas conversiones
    que hacía la inserción fila por fila). `fijas` agrega columnas constantes al
    inicio (p. ej. run_id y solucion_idx para asignacion_pareto).
    """
    out = df[COLS_ASIGNACION].copy()
    for c in ("estudiante_id", "docente_id", "establecimiento_id", "institucion_id"):
        out[c] = out[c].astype("int64")
    for c in ("grado", "seccion", "turno"):
        out[c] = out[c].astype(str)
    out["distancia"] = out["distancia"].astype(float)
    for pos, (col, valor) in enumerate(fijas.items()):
        out.insert(pos, col, valor)
    return out


def _copy_dataframe(cur, tabla, df):
    """
    Carga masiva con COPY ... FROM STDIN (formato CSV), mucho más rápida que
    INSERT por lotes para miles de filas. Los textos van entre comillas
    (QUOTE_NONNUMERIC), así un texto vacío se guarda como '' y no como NULL.
    En las columnas numéricas decimales un valor faltante (NaN) se guarda como
    NULL (FORCE_NULL), porque pandas lo escribe como "" entre comillas.
    `tabla` y las columnas son nombres internos del código (no datos de usuario).
    """
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, quoting=csv.QUOTE_NONNUMERIC)
    buf.seek(0)
    opciones = "FORMAT csv"
    decimales = [c for c in df.columns if df[c].dtype.kind == "f"]
    if decimales:
        opciones += f", FORCE_NULL ({', '.join(decimales)})"
    cur.copy_expert(f"COPY {tabla} ({', '.join(df.columns)}) FROM STDIN WITH ({opciones})", buf)


def escribir_asignacion_mec_opt(cur, problem, x):
    """
    Reemplaza el contenido de asignacion_mec_opt con la solución x (una fila por
    estudiante). La tabla se crea, si no existe, con la misma estructura que
    asignacion_mec. No hace commit: lo decide quien llama.
    La usan guardar_pareto_db() y DatabaseManager.save_asignaciones().
    """
    df = decodificar_asignaciones(problem, x)
    cur.execute("CREATE TABLE IF NOT EXISTS asignacion_mec_opt (LIKE asignacion_mec INCLUDING ALL)")
    cur.execute("TRUNCATE asignacion_mec_opt RESTART IDENTITY")
    _copy_dataframe(cur, "asignacion_mec_opt", _tabla_asignacion(df))


def guardar_pareto_db(conn, problem, ps, run_id=None, algoritmo="nsga3", pop_size=None,
                      n_gen=None, seed=None, guardar_asignaciones="seleccion", metodo="compromiso",
                      etiquetas=None, escribir_compat=True):
    """
    Persiste el Conjunto Pareto COMPLETO en PostgreSQL:
      opt_run (1 fila por corrida), pareto_solucion (1 fila por solución, F en
      signo natural), asignacion_pareto (asignaciones de las soluciones
      seleccionadas o de todas) y, por compatibilidad con el viewer,
      asignacion_mec_opt con la solución elegida por `metodo`.
    """
    from psycopg2.extras import execute_values  # inserción por lotes
    # identificador corto y único si no se provee uno
    run_id = run_id or uuid.uuid4().hex[:12]
    etiquetas = etiquetas if etiquetas is not None else etiquetar_decisiones(ps["F"])
    R = problem.a_reportable(ps["F"])
    X = ps["X"]
    elegido = seleccionar_solucion(ps["F"], metodo, ref=ref_mec(problem))
    # Todo dentro de una única transacción: se confirma con un solo commit al
    # final; si algo falla, el llamador hace rollback (DatabaseManager).
    with conn.cursor() as cur:
        cur.execute(DDL)
        cur.execute("""INSERT INTO opt_run (run_id, algoritmo, pop_size, n_gen, seed, n_obj,
                       n_estudiantes, n_docentes, n_clases, n_soluciones, obj_names, ideal, nadir)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (run_id) DO NOTHING""",
                    (run_id, algoritmo, pop_size, n_gen, seed, int(R.shape[1]), problem.na,
                     problem.nd, problem.nc, int(len(R)), list(ps["obj_names"]),
                     R.min(0).tolist(), R.max(0).tolist()))
        execute_values(cur, """INSERT INTO pareto_solucion (run_id, solucion_idx, f, seleccionada)
                               VALUES %s ON CONFLICT (run_id, solucion_idx) DO NOTHING""",
                       [(run_id, i, [float(v) for v in R[i]], etiquetas.get(i)) for i in range(len(R))])
        if guardar_asignaciones != "ninguna":
            idxs = range(len(X)) if guardar_asignaciones == "todas" else sorted(set(etiquetas) | {elegido})
            # borra asignaciones previas del mismo run_id (re-guardado idempotente)
            cur.execute("DELETE FROM asignacion_pareto WHERE run_id = %s", (run_id,))
            # un COPY por solución (~na filas cada uno): rápido y con memoria acotada
            for i in idxs:
                df = decodificar_asignaciones(problem, X[i])
                _copy_dataframe(cur, "asignacion_pareto",
                                _tabla_asignacion(df, run_id=run_id, solucion_idx=int(i)))
        if escribir_compat and len(X):
            # compatibilidad con el viewer: asignacion_mec_opt = solución elegida
            escribir_asignacion_mec_opt(cur, problem, X[elegido])
    conn.commit()
    logger.info(f"✅ Corrida {run_id}: {len(R)} soluciones del Conjunto Pareto guardadas")
    return run_id
