# ================================================================
# integrated_viewer_optimizado.py
#
# Basado en:
#   Telesca, E.; López-Pires, F. (2025). "A Unified Framework for
#   Multi-Objective Assignment of Teachers and Students to Educational
#   Institutions". Preprint 202512.0426.v1. Universidad Internacional
#   Tres Fronteras (UNINTER), Paraguay.
#   Proyecto original Conacyt-Uninter: Dr. Fabio López (tutor investigador),
#   Ing. Eliana Telesca (investigadora en formación). Viewer original v1.3.
#
# Extensión many-objective (NSGA-III, U-NSGA-III, C-TAEA, RVEA, SPEA2; 3 a 6 objetivos):
#   "Many-Objective Assignment of Teachers and Students to Schools: An Experimental 
#   Evaluation of Algorithms" (K. Pistilli, F. López-Pires, R. Benítez — 
#   Facultad Politécnica, Universidad Nacional de Asunción)
#
# Descripción del módulo (UI)
# ---------------------------
# Aplicación Streamlit con 3 pestañas:
#   1) "Visualización Actual" (ANTES):
#        - Asignación actual del MEC (tabla asignacion_mec) con distancias
#          recalculadas con la MISMA fórmula que el modelo
#        - Objetivos f1–f6 de la asignación MEC, evaluados con IntegratedProblem
#        - Mapa Folium (estudiantes, docentes, establecimientos y líneas opcionales)
#   2) "Optimización":
#        - Parámetros: algoritmo, objetivos (incl. f6), f4_modo, cap_min,
#          población, generaciones, método de decisión y qué guardar
#        - Resultado: el CONJUNTO PARETO completo, con la solución que elige cada
#          método; se puede cambiar el método y guardar otra solución
#        - Comparación objetivo por objetivo contra el MEC
#        - Tabla y mapa de asignacion_mec_opt (DESPUÉS)
#   3) "Comparación Antes/Después":
#        - Objetivos del modelo antes (MEC) y después (asignacion_mec_opt)
#        - Reasignados = alumnos que cambian de ESTABLECIMIENTO (mismo criterio que f6)
#        - Tabla filtrable, exportaciones, gráficos y mapa de trayectorias
#
# Cómo se alinea con el resto del proyecto:
#   - Los datos se cargan con instancia.cargar_desde_db(), igual que
#     integrated_app.py y experiments.py (incluye la asignación MEC).
#   - La asignación MEC y la guardada se convierten en vectores x con
#     instancia.x_desde_asignacion() y se evalúan con el mismo IntegratedProblem:
#     los valores de f1–f6 son los mismos que imprime integrated_app.py.
#   - Las distancias por fila se recalculan con integrated_problem.haversine_pares
#     (no se usa la columna `distancia` guardada, que puede venir de otra fuente).
#   - Los selectores de algoritmo/objetivos son nativos: parche_viewer.py ya
#     no hace falta (detecta la clave "opt_algo" y no modifica nada).
#
# Notas de diseño:
#  - Streamlit re-ejecuta el script completo ante cada interacción; por eso los
#    datos (st.cache_data) y el modelo base (st.cache_resource) están cacheados,
#    y los parámetros de la corrida van dentro de un formulario (st.form).
#  - Mapas livianos: líneas de asignación opcionales y muestreadas
#    (MAX_LINEAS_MAPA), un marcador por establecimiento y
#    st_folium(returned_objects=[]) para no re-ejecutar al mover el mapa.
#  - La corrida usa un solo proceso (run_integrated_optimization no paraleliza).
# ================================================================
import logging
import math
import uuid
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium import FeatureGroup
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sqlalchemy.exc import SQLAlchemyError
import plotly.express as px

# Módulos internos del proyecto
from integrated_problem import (IntegratedProblem, CONJUNTOS_OBJETIVOS,         # Modelo
                                NOMBRES_RESTRICCIONES, haversine_pares)
from integrated_optimization import (run_integrated_optimization,             # Orquestación
                                     ALGORITMOS_EVOLUTIVOS, DatabaseManager)
from pareto import (METODOS_DECISION, seleccionar_solucion, etiquetar_decisiones,  # Conjunto Pareto
                    frente_a_dataframe, escribir_asignacion_mec_opt)
from instancia import cargar_desde_db, x_desde_asignacion                      # Datos
from database import engine, DB_CONFIG                                         # BD y config única

# ================================
# CONFIGURACIÓN DE CONEXIÓN Y LOGGING
# -----------------------------
# - DB_CONFIG viene de database.py: es la ÚNICA configuración del proyecto
#   (variables DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, desde .env o
#   el entorno). Se usa la misma para leer datos y para escribir resultados.
# - El viewer es un punto de entrada: aquí se configura el logging global.
# ================================
logging.basicConfig(level=logging.INFO)

# ================================
# CONFIGURACIÓN GENERAL DE PÁGINA
# ================================
st.set_page_config(page_title="ASEDEE", layout="wide")
st.markdown("<h1 style='text-align:center'>📊 ASEDEE</h1>", unsafe_allow_html=True)

# ================================
# HELPERS (turnos, export, mapas)
# ================================
TURNOS_TO_ID = {
    "mañana": 0, "manana": 0, "matutino": 0,
    "tarde": 1, "vespertino": 1,
    "noche": 2, "nocturno": 2,
    0: 0, 1: 1, 2: 2
}
ID_TO_TURNO = {0: "Mañana", 1: "Tarde", 2: "Noche"}
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

def to_turno_id(v):
    """Mapea texto/entero de turno a id normalizado {0,1,2}; desconocido o NaN -> 0 (Mañana)."""
    if pd.isna(v): return 0
    if isinstance(v, (int, np.integer)): return TURNOS_TO_ID.get(int(v), 0)
    return TURNOS_TO_ID.get(str(v).strip().lower(), 0)

def turno_label(v):
    """Devuelve etiqueta humana del turno."""
    return ID_TO_TURNO.get(to_turno_id(v), str(v))

def exportar_excel(df: pd.DataFrame, sheet_name: str = "Datos") -> bytes:
    """Serializa un DataFrame a un archivo XLSX en memoria (bytes)."""
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name=sheet_name)
    return out.getvalue()

def select_cols(df: pd.DataFrame, cols: list) -> list:
    """Devuelve solo las columnas existentes (evita KeyError en vistas parciales)."""
    return [c for c in cols if c in df.columns]

def _valid_coords(*vals):
    """Valida que todos los valores de lat/lon existan y no sean NaN."""
    return all(v is not None and not pd.isna(v) for v in vals)

def fmt_km(v):
    """Número en km con 2 decimales, o 'n/d' si no hay dato."""
    return "n/d" if v is None or pd.isna(v) else f"{v:.2f} km"

# Tope de asignaciones con líneas dibujadas por mapa. Cada línea es un objeto
# en el HTML del mapa: con ~16.000 alumnos (2 líneas cada uno) el navegador
# se vuelve muy lento. Por encima del tope se toma una muestra regular.
MAX_LINEAS_MAPA = 2000

def muestrear(df: pd.DataFrame, max_filas: int = MAX_LINEAS_MAPA) -> pd.DataFrame:
    """Muestra regular (1 de cada k filas) con a lo sumo max_filas filas; determinista."""
    if df is None or len(df) <= max_filas:
        return df
    paso = math.ceil(len(df) / max_filas)
    return df.iloc[::paso]

def establecimientos_desde_clases(clases: pd.DataFrame) -> pd.DataFrame:
    """
    Una fila por establecimiento (en lugar de una por clase) para los marcadores
    de instituciones: coordenadas, institución, nº de clases, grados y turnos.
    """
    if clases is None or clases.empty or not {"establecimiento_id", "lat", "lng"}.issubset(clases.columns):
        return pd.DataFrame()
    g = clases.groupby("establecimiento_id")
    out = g[["lat", "lng"]].first()
    out["institucion"] = g["nombre_institucion"].first() if "nombre_institucion" in clases.columns else ""
    out["n_clases"] = g.size()
    if "grado" in clases.columns:
        out["grados"] = g["grado"].agg(lambda s: ", ".join(sorted(map(str, s.unique()))))
    if "turno" in clases.columns:
        out["turnos"] = g["turno"].agg(lambda s: ", ".join(sorted({turno_label(t) for t in s})))
    return out.reset_index()

def nuevo_mapa(lat, lng) -> folium.Map:
    """
    Mapa centrado en los datos: se crea en el centro de los puntos y se
    ajusta (fit_bounds) a su extensión, en lugar de mostrar todo Paraguay.
    """
    lat = pd.to_numeric(pd.Series(lat), errors="coerce").dropna()
    lng = pd.to_numeric(pd.Series(lng), errors="coerce").dropna()
    if lat.empty or lng.empty:
        return folium.Map(location=[-25.3, -57.6], zoom_start=7)   # sin datos: Paraguay
    mapa = folium.Map(location=[float(lat.mean()), float(lng.mean())], zoom_start=11)
    mapa.fit_bounds([[float(lat.min()), float(lng.min())], [float(lat.max()), float(lng.max())]])
    return mapa

def dibujar_lineas(df: pd.DataFrame, capa_est, capa_doc, sufijo: str) -> int:
    """
    Dibuja las líneas Estudiante→Establecimiento (azul continua) y
    Docente→Establecimiento (verde punteada) de una muestra de `df`
    (columnas est_*, doc_*, estb_*). Devuelve cuántas asignaciones se dibujaron.
    """
    muestra = muestrear(df)
    for _, fila in muestra.iterrows():
        if _valid_coords(fila.get("est_lat"), fila.get("est_lng"), fila.get("estb_lat"), fila.get("estb_lng")):
            folium.PolyLine(
                [(fila["est_lat"], fila["est_lng"]), (fila["estb_lat"], fila["estb_lng"])],
                color="#1f77b4", weight=2, opacity=0.85,
                tooltip=f"Estudiante → Establecimiento ({sufijo})"
            ).add_to(capa_est)
        if _valid_coords(fila.get("doc_lat"), fila.get("doc_lng"), fila.get("estb_lat"), fila.get("estb_lng")):
            folium.PolyLine(
                [(fila["doc_lat"], fila["doc_lng"]), (fila["estb_lat"], fila["estb_lng"])],
                color="#2ca02c", weight=2, opacity=0.85, dash_array="6,4",
                tooltip=f"Docente → Establecimiento ({sufijo})"
            ).add_to(capa_doc)
    return len(muestra)

# ================================
# DATOS Y MODELO (cacheados)
# -------------------------------
# - cargar_instancia(): estudiantes, docentes, clases y asignación MEC, con la
#   misma función que usan integrated_app.py y experiments.py.
# - preparar_modelo(): problema "de reporte" con todos los objetivos
#   disponibles (f1–f5 y f6 si hay asignación MEC), matrices U/V y el vector x
#   de la asignación MEC. Se construye una vez y se reutiliza en cada corrida.
# ================================
@st.cache_data(ttl=600)
def cargar_instancia():
    """Instancia desde PostgreSQL vía instancia.cargar_desde_db() (dict de DataFrames)."""
    return cargar_desde_db()

@st.cache_resource(ttl=600)
def preparar_modelo():
    """
    Devuelve dict con:
      base     IntegratedProblem de 3 objetivos (sólo para U, V y candidatas)
      reporte  IntegratedProblem con TODOS los objetivos disponibles; sirve para
               evaluar cualquier asignación con los mismos números que el resto
      x_mec    vector x de la asignación actual (None si no hay asignacion_mec)
      mapeo    cómo se mapeó cada fila de asignacion_mec (exacto, sin_turno, ...)
    """
    inst = cargar_instancia()
    est, doc, cls, asig = inst["estudiantes"], inst["docentes"], inst["clases"], inst.get("asignacion")
    base = IntegratedProblem(est, doc, cls, n_obj=3)
    x_mec, mapeo = None, {}
    if asig is not None and len(asig):
        x_mec, mapeo = x_desde_asignacion(base, asig)
    objetivos = list(CONJUNTOS_OBJETIVOS[5]) + (["f6_estabilidad"] if x_mec is not None else [])
    reporte = IntegratedProblem(est, doc, cls, objetivos=objetivos, U=base.U, V=base.V, x_base=x_mec)
    return {"base": base, "reporte": reporte, "x_mec": x_mec, "mapeo": mapeo}

def tabla_objetivos(rep: IntegratedProblem, x_antes, x_despues, nombre_antes="MEC (antes)",
                    nombre_despues="Solución") -> pd.DataFrame:
    """
    Evalúa dos asignaciones con el MISMO problema y arma una tabla por objetivo:
    valor antes, valor después y mejora % (positiva = mejor, según el sentido
    min/max de cada objetivo). Agrega distancias promedio/máximas y violaciones.
    """
    _, Ga, da = rep.evaluar(x_antes, detalle=True)
    _, Gd, dd = rep.evaluar(x_despues, detalle=True)
    filas = []
    for o, s, etq in zip(rep.objetivos, rep.sentidos, rep.etiquetas):
        a, d = float(da[o]), float(dd[o])
        mejora = ((a - d) if s == "min" else (d - a)) / abs(a) * 100 if a != 0 else np.nan
        filas.append({"objetivo": o, "descripción": etq, "sentido": s,
                      nombre_antes: a, nombre_despues: d, "mejora %": mejora})
    for k, etq in (("dist_prom_est", "Distancia promedio de estudiantes (km)"),
                   ("dist_prom_doc", "Distancia promedio de docentes (km)"),
                   ("dist_max_est", "Distancia máxima de estudiantes (km)")):
        a, d = float(da[k]), float(dd[k])
        filas.append({"objetivo": k, "descripción": etq, "sentido": "min",
                      nombre_antes: a, nombre_despues: d,
                      "mejora %": (a - d) / abs(a) * 100 if a else np.nan})
    filas.append({"objetivo": "violaciones", "descripción": "Suma de violaciones g1–g5 (0 = factible)",
                  "sentido": "min", nombre_antes: float(np.sum(Ga)), nombre_despues: float(np.sum(Gd)),
                  "mejora %": np.nan})
    return pd.DataFrame(filas)

def nota_mapeo(mapeo: dict) -> str:
    """
    Texto de aviso si alguna fila de asignacion_mec no coincide con una clase
    (establecimiento, grado, turno). Esas filas se aproximan a la clase más
    parecida (instancia.x_desde_asignacion), así que los objetivos del MEC y f6
    usan esa aproximación y pueden diferir de lo que dice la tabla literal.
    Devuelve "" si todas las filas coincidieron exactamente.
    """
    aprox = {k: v for k, v in mapeo.items() if k != "exacto" and v}
    if not aprox:
        return ""
    total = sum(mapeo.values())
    detalle = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in aprox.items())
    return (f"{sum(aprox.values())} de {total} filas de asignacion_mec no coinciden con ninguna clase "
            f"(establecimiento + grado + turno) y se aproximaron ({detalle}). Los objetivos del MEC y f6 "
            "usan esa aproximación; el conteo de reasignados de la pestaña de comparación usa la tabla tal cual.")

def mostrar_tabla_objetivos(df: pd.DataFrame):
    """Muestra la tabla de tabla_objetivos() con 4 decimales y la mejora con signo."""
    formatos = {c: "{:.4f}" for c in df.columns if c not in ("objetivo", "descripción", "sentido", "mejora %")}
    formatos["mejora %"] = "{:+.1f}%"
    st.dataframe(df.style.format(formatos, na_rep="—"), hide_index=True, width="stretch")

@st.cache_data(ttl=600)
def cargar_asignaciones_full(tabla="asignacion_mec", silencioso=False):
    """
    Vista FULL de una tabla de asignaciones (una fila por estudiante), con
    coordenadas de estudiante, docente y establecimiento.

    - `tabla`: asignacion_mec (antes) o asignacion_mec_opt (después). Se inserta
      en el SQL con f-string: sólo debe recibir estos nombres internos.
    - Distancias: `distancia` se RECALCULA con la fórmula del modelo
      (haversine_pares, estudiante → establecimiento) para que antes y después
      sean comparables; el valor guardado queda en `distancia_registrada`.
      También se agrega `distancia_docente` (docente → establecimiento).
    - silencioso=True: si la tabla no existe devuelve un DataFrame vacío sin
      mostrar error (asignacion_mec_opt puede no existir todavía).
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                f"""
                SELECT
                    a.id, a.grado, a.seccion, a.turno, a.distancia,
                    e.id AS estudiante_id, e.nombre AS estudiante, e.lat AS est_lat, e.lng AS est_lng,
                    d.id AS docente_id, d.nombre AS docente, d.lat AS doc_lat, d.lng AS doc_lng,
                    es.id AS establecimiento_id, es.lat AS estb_lat, es.lng AS estb_lng,
                    i.id AS institucion_id, i.nombre AS institucion,
                    i.departamento AS inst_departamento, i.localidad AS inst_localidad
                FROM {tabla} a
                JOIN estudiantes e       ON a.estudiante_id      = e.id
                JOIN docentes d          ON a.docente_id         = d.id
                JOIN establecimientos es ON a.establecimiento_id = es.id
                JOIN instituciones i     ON es.institucion_id    = i.id
                ORDER BY a.id DESC
                """,
                con=conn
            )
    except SQLAlchemyError as e:
        if not silencioso:
            st.error(f"❌ Error cargando asignaciones ({tabla}): {e}")
        return pd.DataFrame()
    if df.empty:
        return df
    df["distancia_registrada"] = pd.to_numeric(df["distancia"], errors="coerce")
    df["distancia"] = np.round(haversine_pares(df["est_lat"], df["est_lng"], df["estb_lat"], df["estb_lng"]), 2)
    df["distancia_docente"] = np.round(haversine_pares(df["doc_lat"], df["doc_lng"], df["estb_lat"], df["estb_lng"]), 2)
    return df

def x_despues():
    """
    Vector x de asignacion_mec_opt (lo guardado), o None. Se calcula una vez y se
    guarda en la sesión (con ~16.000 filas tarda casi un segundo); refrescar_despues()
    lo invalida cuando cambia la tabla.
    """
    if st.session_state.get("x_despues") is None:
        df = st.session_state.get("asignaciones_after")
        if df is None or df.empty:
            return None
        st.session_state.x_despues = x_desde_asignacion(preparar_modelo()["base"], df)[0]
    return st.session_state.x_despues

def refrescar_despues():
    """Vuelve a leer asignacion_mec_opt (después de guardar) en el estado de la sesión."""
    cargar_asignaciones_full.clear()
    st.session_state.asignaciones_after = cargar_asignaciones_full("asignacion_mec_opt", silencioso=True)
    st.session_state.x_despues = None

# Snapshots en session_state para ANTES/DESPUÉS.
# st.session_state sobrevive a las re-ejecuciones dentro de la misma sesión del
# navegador. "Después" arranca con lo que YA haya en asignacion_mec_opt (por
# ejemplo, lo guardado por integrated_app.py o por una sesión anterior).
if "asignaciones_before" not in st.session_state:
    st.session_state.asignaciones_before = cargar_asignaciones_full("asignacion_mec")
if "asignaciones_after" not in st.session_state:
    st.session_state.asignaciones_after = cargar_asignaciones_full("asignacion_mec_opt", silencioso=True)

# ================================
# TABS
# Nota: Streamlit ejecuta el contenido de LAS TRES pestañas en cada
# re-ejecución, aunque el usuario sólo esté viendo una.
# ================================
tab_actual, tab_opt, tab_comp = st.tabs(
    ["🔍 Visualización Actual", "🚀 Optimización", "📊 Comparación Antes/Después"]
)

# ================================================================
# TAB 1: VISUALIZACIÓN ACTUAL (ANTES)
# ================================================================
with tab_actual:
    inst = cargar_instancia()
    estudiantes, docentes, clases = inst["estudiantes"], inst["docentes"], inst["clases"]
    if estudiantes.empty or docentes.empty or clases.empty:
        st.error("❌ Error al cargar datos base (estudiantes/docentes/clases).")
        st.stop()
    modelo = preparar_modelo()
    rep, x_mec = modelo["reporte"], modelo["x_mec"]

    st.success(f"✅ Datos cargados: {len(estudiantes)} estudiantes, {len(docentes)} docentes, {len(clases)} clases")

    asig_full = st.session_state.asignaciones_before

    # KPIs del estado actual, calculados con el modelo (mismos valores que integrated_app.py)
    st.markdown("### 🔎 Estado actual (asignación del MEC)")
    if x_mec is None:
        st.info("No hay datos en asignacion_mec: no se puede evaluar el estado actual.")
    else:
        _, G_mec, det_mec = rep.evaluar(x_mec, detalle=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Asignaciones", f"{len(asig_full):,}")
        c2.metric("Distancia prom. estudiantes", fmt_km(det_mec["dist_prom_est"]))
        c3.metric("Distancia prom. docentes", fmt_km(det_mec["dist_prom_doc"]))
        c4.metric("f1 (est. + doc.)", fmt_km(det_mec["f1_distancia"]))
        if nota_mapeo(modelo["mapeo"]):
            st.warning("⚠️ " + nota_mapeo(modelo["mapeo"]))
        with st.expander("Objetivos f1–f6 y restricciones de la asignación MEC"):
            vals = pd.DataFrame({"objetivo": rep.objetivos, "sentido": rep.sentidos,
                                 "descripción": rep.etiquetas,
                                 "valor": [det_mec[o] for o in rep.objetivos]})
            st.dataframe(vals, hide_index=True, width="stretch")
            st.caption("Violaciones: " + ", ".join(f"{n} = {int(v)}" for n, v in zip(NOMBRES_RESTRICCIONES, G_mec))
                       + " · mapeo de filas: " + ", ".join(f"{k} = {v}" for k, v in modelo["mapeo"].items()))

    # Tabla de asignaciones (distancia del modelo + la registrada en la tabla)
    st.subheader("📋 Asignaciones Actuales (ANTES)")
    cols_antes = select_cols(asig_full, ["id", "estudiante", "docente", "institucion", "grado", "seccion",
                                         "turno", "distancia", "distancia_registrada", "distancia_docente"])
    if not asig_full.empty:
        st.dataframe(asig_full[cols_antes], height=360, width="stretch")
        dif = (asig_full["distancia"] - asig_full["distancia_registrada"]).abs()
        if dif.max(skipna=True) > 0.5:
            st.caption("ℹ️ `distancia` se recalcula con la fórmula del modelo (estudiante → establecimiento). "
                       f"La columna guardada en la tabla difiere en promedio {dif.mean():.2f} km; "
                       "se muestra como `distancia_registrada` pero no se usa en las comparaciones.")
        st.download_button(
            "📥 Exportar Asignaciones (antes)",
            data=exportar_excel(asig_full[cols_antes], "Asignaciones_antes"),
            file_name="asignaciones_antes.xlsx", mime=XLSX, key="btn_export_antes",
        )

    # ----- MAPA “ANTES”: puntos y líneas de unión (estilo igual al de Optimización)
    # Marcadores: uno por estudiante, docente y ESTABLECIMIENTO. Las líneas de
    # asignación son opcionales y se muestrean (ver MAX_LINEAS_MAPA).
    st.subheader("🗺️ Mapa de Estudiantes, Docentes e Instituciones (antes)")
    lineas_antes = st.checkbox("Mostrar líneas de asignación", value=False, key="lineas_antes")
    mapa = nuevo_mapa(pd.concat([estudiantes["lat"], docentes["lat"]]), pd.concat([estudiantes["lng"], docentes["lng"]]))

    capa_est = FeatureGroup(name="🎓 Estudiantes (Azul)", show=True)
    capa_doc = FeatureGroup(name="👩‍🏫 Docentes (Verde)", show=True)
    capa_inst = FeatureGroup(name="🏫 Instituciones (Rojo)", show=True)
    capa_linea_est = FeatureGroup(name="Líneas Estudiante → Establecimiento", show=True)
    capa_linea_doc = FeatureGroup(name="Líneas Docente → Establecimiento", show=True)
    cluster_est = MarkerCluster(name="Cluster Estudiantes").add_to(capa_est)
    cluster_doc = MarkerCluster(name="Cluster Docentes").add_to(capa_doc)
    cluster_ins = MarkerCluster(name="Cluster Instituciones").add_to(capa_inst)

    for _, r in estudiantes.iterrows():
        if _valid_coords(r.get("lat"), r.get("lng")):
            folium.CircleMarker([r["lat"], r["lng"]], radius=3, color="blue", fill=True, fill_opacity=0.9,
                                popup=f"🎓 {r.get('nombre', r.get('estudiante_id', ''))}").add_to(cluster_est)
    for _, r in docentes.iterrows():
        if _valid_coords(r.get("lat"), r.get("lng")):
            folium.CircleMarker([r["lat"], r["lng"]], radius=3, color="green", fill=True, fill_opacity=0.9,
                                popup=f"👩‍🏫 {r.get('nombre', r.get('docente_id', ''))}").add_to(cluster_doc)
    # Instituciones: un marcador por establecimiento (no uno por clase)
    for _, r in establecimientos_desde_clases(clases).iterrows():
        if _valid_coords(r["lat"], r["lng"]):
            folium.Marker(
                [r["lat"], r["lng"]],
                popup=(f"🏫 {r.get('institucion','')}<br>Clases: {r.get('n_clases','')}"
                       f"<br>Grados: {r.get('grados','')}<br>Turnos: {r.get('turnos','')}"),
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(cluster_ins)

    if lineas_antes and not asig_full.empty:
        n_dib = dibujar_lineas(asig_full, capa_linea_est, capa_linea_doc, "antes")
        if n_dib < len(asig_full):
            st.caption(f"Líneas de {n_dib:,} de {len(asig_full):,} asignaciones (muestra regular).")

    # returned_objects=[]: mover o hacer zoom no re-ejecuta toda la app
    for capa in (capa_est, capa_doc, capa_inst, capa_linea_est, capa_linea_doc):
        capa.add_to(mapa)
    folium.LayerControl(collapsed=False).add_to(mapa)
    st_folium(mapa, width=1000, height=520, key="mapa_antes", returned_objects=[])


# ================================================================
# TAB 2: OPTIMIZACIÓN (DESPUÉS)
# - Parámetros (en un formulario: no se re-ejecuta nada hasta "Ejecutar")
# - Conjunto Pareto, decisión a posteriori y comparación con el MEC
# - Tabla y mapa de asignacion_mec_opt
# ================================================================
with tab_opt:
    st.subheader("🚀 Optimización de Asignaciones")
    modelo = preparar_modelo()
    rep, x_mec = modelo["reporte"], modelo["x_mec"]
    metodos = list(METODOS_DECISION) + [f"min_f{k}" for k in range(1, 7)]
    algoritmos = list(ALGORITMOS_EVOLUTIVOS)

    with st.form("form_opt"):
        c1, c2, c3 = st.columns(3)
        algoritmo = c1.selectbox("Algoritmo", algoritmos, index=algoritmos.index("nsga3"),
                                 key="opt_algo", help="agemoea requiere tener numba instalado")
        n_obj = c2.radio("Objetivos base", [4, 3, 5], horizontal=True, key="opt_nobj",
                         help="3 = preprint (f1–f3) · 4 = + distancia máxima (f4) · 5 = + afinidad de grado (f5)")
        usar_f6 = c3.checkbox("Agregar f6 (estabilidad vs MEC)", value=False, key="opt_f6",
                              disabled=x_mec is None,
                              help="Proporción de estudiantes que cambian de establecimiento respecto del MEC")
        c4, c5, c6 = st.columns(3)
        f4_modo = c4.selectbox("f4: distancia máxima", ["suma", "max"], key="opt_f4",
                               help="suma = máx. estudiantes + máx. docentes · max = el mayor de los dos")
        cap_min = c5.number_input("Mínimo de alumnos por clase (cap_min)", 0, 100, 0, key="opt_capmin",
                                  help="0 = sin mínimo (v1.3) · 25 = preprint · 10 = CACIC")
        metodo = c6.selectbox("Solución a guardar (método de decisión)", metodos, key="opt_metodo")
        c7, c8, c9 = st.columns(3)
        pop_size = c7.slider("Tamaño de población", 10, 200, 50, key="opt_pop")
        n_gen = c8.slider("Generaciones", 10, 200, 30, key="opt_gen")
        guardar = c9.selectbox("Asignaciones detalladas a guardar", ["seleccion", "todas", "ninguna"], key="opt_guardar",
                               help="seleccion = las soluciones elegidas por algún método · todas = todo el Conjunto Pareto "
                                    "(puede ser mucho) · ninguna = sólo el frente. asignacion_mec_opt se escribe siempre.")
        ejecutar = st.form_submit_button("Ejecutar Optimización", type="primary")

    if ejecutar:
        objetivos = list(CONJUNTOS_OBJETIVOS[n_obj]) + (["f6_estabilidad"] if usar_f6 else [])
        if metodo.startswith("min_f") and int(metodo[5:]) > len(objetivos):
            st.warning(f"{metodo} no aplica con {len(objetivos)} objetivos; se usa 'compromiso'.")
            metodo = "compromiso"
        with st.status("⏳ Ejecutando optimización...", expanded=False) as status:
            try:
                inst = cargar_instancia()
                base = modelo["base"]
                problem = IntegratedProblem(inst["estudiantes"], inst["docentes"], inst["clases"],
                                            objetivos=objetivos, cap_min=int(cap_min), f4_modo=f4_modo,
                                            U=base.U, V=base.V, x_base=x_mec)
                run_id = f"viewer_{uuid.uuid4().hex[:8]}"
                result = run_integrated_optimization(
                    problem, pop_size, n_gen, 1, algoritmo=algoritmo, verbose=False,
                    db_config=dict(DB_CONFIG),   # misma configuración que la lectura
                    run_id=run_id, metodo=metodo, guardar_pareto=guardar,
                )
                st.session_state.corrida = {"problem": problem, "ps": result.pareto_set, "run_id": run_id,
                                            "algoritmo": algoritmo, "metodo": metodo}
                refrescar_despues()
                status.update(label="✅ Optimización completada", state="complete")
            except Exception as e:
                status.update(label="❌ Error durante la optimización", state="error")
                st.error(f"❌ Error durante la optimización: {e}")

    # ---------- Resultado de la última corrida de esta sesión ----------
    corrida = st.session_state.get("corrida")
    if corrida:
        problem, ps = corrida["problem"], corrida["ps"]
        if ps is None:
            st.error("La corrida no encontró soluciones factibles: se guardó la de menor violación. "
                     "Probá con más generaciones o sin cap_min.")
        else:
            st.subheader(f"🧭 Conjunto Pareto: {ps['n_soluciones']} soluciones no dominadas")
            st.caption(f"Corrida `{corrida['run_id']}` · {corrida['algoritmo'].upper()} · "
                       f"{problem.n_obj} objetivos · guardada en opt_run / pareto_solucion. "
                       "Valores con su signo natural (f3 y f5 se maximizan).")
            frente = frente_a_dataframe(problem, ps)
            frente["elegida por"] = frente["solucion_idx"].map(etiquetar_decisiones(ps["F"])).fillna("")
            st.dataframe(frente.round(4), hide_index=True, height=260, width="stretch")

            # Decisión a posteriori: cambiar de método no re-optimiza, sólo elige otra solución
            m_ok = [m for m in metodos if not m.startswith("min_f") or int(m[5:]) <= problem.n_obj]
            m_sel = st.selectbox("Ver la solución que elige el método", m_ok,
                                 index=m_ok.index(corrida["metodo"]) if corrida["metodo"] in m_ok else 0,
                                 key="metodo_post")
            idx = seleccionar_solucion(ps["F"], m_sel,
                                       ref=problem.evaluar(x_mec)[0] if x_mec is not None else None)
            x_sel = ps["X"][idx]
            G_sel = problem.evaluar(x_sel)[1]
            if np.any(G_sel > 0):
                st.warning("⚠️ Esta solución viola restricciones: "
                           + ", ".join(f"{n} = {int(v)}" for n, v in zip(NOMBRES_RESTRICCIONES, G_sel) if v > 0))
            st.markdown(f"**Solución {idx}** (método *{m_sel}*) comparada con la asignación actual:")
            if x_mec is not None:
                mostrar_tabla_objetivos(tabla_objetivos(rep, x_mec, x_sel))
            else:
                _, _, det = rep.evaluar(x_sel, detalle=True)
                st.dataframe(pd.DataFrame({"objetivo": rep.objetivos, "valor": [det[o] for o in rep.objetivos]}),
                             hide_index=True)
            if m_sel != corrida["metodo"]:
                if st.button(f"💾 Guardar la solución {idx} en asignacion_mec_opt", key="btn_guardar_sel"):
                    db = DatabaseManager(dict(DB_CONFIG))
                    if db.connect():
                        try:
                            with db.conn.cursor() as cur:
                                escribir_asignacion_mec_opt(cur, problem, x_sel)
                            db.conn.commit()
                            corrida["metodo"] = m_sel
                            refrescar_despues()
                            st.success(f"Solución {idx} guardada en asignacion_mec_opt.")
                        except Exception as e:
                            db.conn.rollback()
                            st.error(f"❌ No se pudo guardar: {e}")
                        finally:
                            db.disconnect()
                    else:
                        st.error("❌ No se pudo conectar a la base de datos.")
            else:
                st.caption("Esta es la solución que está guardada en asignacion_mec_opt.")

    # ---------- Asignación guardada (DESPUÉS) ----------
    opt_df = st.session_state.asignaciones_after
    if opt_df.empty:
        st.info("Todavía no hay resultados en asignacion_mec_opt. Ejecutá una optimización.")
    else:
        st.subheader("📋 Asignaciones Optimizadas (DESPUÉS) — tabla asignacion_mec_opt")
        cols_opt = select_cols(opt_df, ["id", "estudiante", "docente", "institucion", "grado", "seccion",
                                        "turno", "distancia", "distancia_docente"])
        st.dataframe(opt_df[cols_opt], height=360, width="stretch")
        st.download_button(
            "📥 Exportar Asignaciones Optimizadas",
            data=exportar_excel(opt_df[cols_opt], "Asignaciones_despues"),
            file_name="asignaciones_optimizadas.xlsx", mime=XLSX, key="btn_export_despues",
        )

        # Mapa “DESPUÉS” (mismo estilo que ANTES, para comparar visualmente)
        st.subheader("🗺️ Mapa de Estudiantes, Docentes e Instituciones (después)")
        lineas_despues = st.checkbox("Mostrar líneas de asignación", value=False, key="lineas_despues")
        mapa_opt = nuevo_mapa(pd.concat([opt_df["est_lat"], opt_df["doc_lat"]]),
                              pd.concat([opt_df["est_lng"], opt_df["doc_lng"]]))
        capa_est_opt = FeatureGroup(name="🎓 Estudiantes (Azul)", show=True)
        capa_doc_opt = FeatureGroup(name="👩‍🏫 Docentes (Verde)", show=True)
        capa_inst_opt = FeatureGroup(name="🏫 Instituciones (Rojo)", show=True)
        cluster_est_opt = MarkerCluster(name="Cluster Estudiantes").add_to(capa_est_opt)
        cluster_doc_opt = MarkerCluster(name="Cluster Docentes").add_to(capa_doc_opt)
        cluster_ins_opt = MarkerCluster(name="Cluster Instituciones").add_to(capa_inst_opt)
        capa_linea_est_opt = FeatureGroup(name="Líneas Estudiante → Establecimiento", show=True)
        capa_linea_doc_opt = FeatureGroup(name="Líneas Docente → Establecimiento", show=True)

        # opt_df tiene una fila por estudiante: docentes y establecimientos se
        # deduplican para dibujar un único marcador por cada uno.
        def _unicos(df, col):
            return df.drop_duplicates(col) if col in df.columns else df

        for _, fila in _unicos(opt_df, "estudiante_id").iterrows():
            if _valid_coords(fila.get("est_lat"), fila.get("est_lng")):
                folium.CircleMarker([fila["est_lat"], fila["est_lng"]], radius=3, color="blue",
                                    fill=True, fill_opacity=0.8, popup=f"🎓 {fila.get('estudiante','')}"
                                    ).add_to(cluster_est_opt)
        for _, fila in _unicos(opt_df, "docente_id").iterrows():
            if _valid_coords(fila.get("doc_lat"), fila.get("doc_lng")):
                folium.CircleMarker([fila["doc_lat"], fila["doc_lng"]], radius=3, color="green",
                                    fill=True, fill_opacity=0.8, popup=f"👩‍🏫 {fila.get('docente','')}"
                                    ).add_to(cluster_doc_opt)
        for _, fila in _unicos(opt_df, "establecimiento_id").iterrows():
            if _valid_coords(fila.get("estb_lat"), fila.get("estb_lng")):
                folium.Marker([fila["estb_lat"], fila["estb_lng"]], popup=f"🏫 {fila.get('institucion','')}",
                              icon=folium.Icon(color="red", icon="info-sign")).add_to(cluster_ins_opt)

        if lineas_despues:
            n_dib = dibujar_lineas(opt_df, capa_linea_est_opt, capa_linea_doc_opt, "después")
            if n_dib < len(opt_df):
                st.caption(f"Líneas de {n_dib:,} de {len(opt_df):,} asignaciones (muestra regular).")

        for capa in (capa_est_opt, capa_doc_opt, capa_inst_opt, capa_linea_est_opt, capa_linea_doc_opt):
            capa.add_to(mapa_opt)
        folium.LayerControl(collapsed=False).add_to(mapa_opt)
        st_folium(mapa_opt, width=1000, height=520, key="mapa_despues", returned_objects=[])

# ================================================================
# TAB 3: COMPARACIÓN ANTES/DESPUÉS
# - Objetivos del modelo: MEC vs asignacion_mec_opt (mismo problema de reporte)
# - Reasignados = cambian de establecimiento (mismo criterio que f6)
# - Tabla, exportaciones, gráficos y mapa de trayectorias
# ================================================================
with tab_comp:
    st.subheader("📊 Comparación Antes/Después")
    df_before = st.session_state.get("asignaciones_before", pd.DataFrame())
    df_after = st.session_state.get("asignaciones_after", pd.DataFrame())

    if df_after.empty or df_before.empty:
        st.info("Hace falta la asignación actual (asignacion_mec) y una solución guardada "
                "(asignacion_mec_opt). Ejecutá una optimización en la pestaña anterior.")
        st.stop()

    # Objetivos del modelo antes/después: las dos tablas se convierten en vectores x
    # y se evalúan con el mismo problema que usa integrated_app.py.
    modelo = preparar_modelo()
    x_antes, x_desp = modelo["x_mec"], x_despues()
    t_obj = tabla_objetivos(modelo["reporte"], x_antes, x_desp, "Antes (MEC)", "Después")
    v = t_obj.set_index("objetivo")

    def _delta(k):
        """Variación % para st.metric (negativa = la distancia baja = mejora)."""
        m = v.loc[k, "mejora %"]
        return None if pd.isna(m) else f"{-m:+.1f}%"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Distancia prom. estudiantes", fmt_km(v.loc["dist_prom_est", "Después"]),
              delta=_delta("dist_prom_est"), delta_color="inverse")
    c2.metric("Distancia prom. docentes", fmt_km(v.loc["dist_prom_doc", "Después"]),
              delta=_delta("dist_prom_doc"), delta_color="inverse")
    c3.metric("f1 (est. + doc.)", fmt_km(v.loc["f1_distancia", "Después"]),
              delta=_delta("f1_distancia"), delta_color="inverse")
    if "f6_estabilidad" in v.index:
        c4.metric("f6: cambian de establecimiento", f"{100 * v.loc['f6_estabilidad', 'Después']:.1f}%",
                  help="Según el modelo (asignación MEC llevada a clases). Ver el aviso de abajo si hay filas aproximadas.")
    with st.expander("Todos los objetivos y restricciones (antes vs después)", expanded=True):
        mostrar_tabla_objetivos(t_obj)

    # Unificar columnas clave para merge y deltas
    claves = ["estudiante_id", "estudiante", "docente_id", "docente", "establecimiento_id",
              "institucion_id", "institucion", "turno", "distancia",
              "est_lat", "est_lng", "doc_lat", "doc_lng", "estb_lat", "estb_lng",
              "inst_departamento", "inst_localidad"]

    def _sufijo(df, suf):
        """Renombra las columnas clave con _antes/_despues (distancia -> dist_antes/dist_despues)."""
        cols = select_cols(df, claves)
        return df[cols].rename(columns={c: (f"dist_{suf}" if c == "distancia" else f"{c}_{suf}")
                                        for c in cols if c not in ("estudiante_id", "estudiante")})
    A, B = _sufijo(df_before, "antes"), _sufijo(df_after, "despues")

    # Merge por estudiante + deltas y banderas.
    # "reasignado" = cambió de ESTABLECIMIENTO (mismo criterio que f6); cambio de
    # docente y de turno se muestran como columnas aparte.
    cmp = pd.merge(B, A, on=["estudiante_id", "estudiante"], how="left")
    cmp["delta_km"] = (cmp["dist_despues"] - cmp["dist_antes"]).round(2)
    cmp["reasignado"] = cmp["establecimiento_id_antes"] != cmp["establecimiento_id_despues"]
    cmp["cambio_docente"] = cmp["docente_id_antes"] != cmp["docente_id_despues"]
    cmp["cambio_turno"] = cmp["turno_antes"] != cmp["turno_despues"]

    st.write("### Detalle de estudiantes")
    only_reas = st.checkbox("Mostrar solo reasignados (cambian de establecimiento)", value=False, key="cmp_only_reas")
    view_df = cmp[cmp["reasignado"]] if only_reas else cmp
    cols_show = select_cols(view_df, [
        "estudiante_id", "estudiante", "inst_departamento_despues", "inst_localidad_despues",
        "dist_antes", "dist_despues", "delta_km", "institucion_antes", "institucion_despues",
        "turno_antes", "turno_despues", "docente_antes", "docente_despues",
        "reasignado", "cambio_docente", "cambio_turno"])
    st.caption(f"{int(cmp['reasignado'].sum()):,} de {len(cmp):,} estudiantes cambian de establecimiento "
               "según las tablas asignacion_mec y asignacion_mec_opt.")
    if nota_mapeo(modelo["mapeo"]):
        st.caption("ℹ️ " + nota_mapeo(modelo["mapeo"]))
    st.dataframe(view_df[cols_show].sort_values("delta_km"), height=360, width="stretch")

    cdl1, cdl2 = st.columns(2)
    cdl1.download_button("📥 Exportar Comparación (antes vs después)", data=exportar_excel(cmp[cols_show], "Comparacion"),
                         file_name="comparacion_antes_despues.xlsx", mime=XLSX, key="btn_cmp_todo")
    cdl2.download_button("📥 Exportar SOLO Reasignados", data=exportar_excel(cmp.loc[cmp["reasignado"], cols_show], "Reasignados"),
                         file_name="reasignados.xlsx", mime=XLSX, key="btn_cmp_reas")

    # Distribución de distancias (fórmula del modelo en ambos casos)
    dist_df = pd.concat([pd.DataFrame({"Distancia (km)": df_before["distancia"], "Estado": "Antes"}),
                         pd.DataFrame({"Distancia (km)": df_after["distancia"], "Estado": "Después"})],
                        ignore_index=True).dropna()
    if not dist_df.empty:
        fig1 = px.box(dist_df, x="Estado", y="Distancia (km)", points=False,
                      title="Distribución de distancias estudiante → establecimiento (km)")
        st.plotly_chart(fig1, width="stretch")

    # Barras: Δ promedio por departamento y turno
    if {"inst_departamento_despues", "turno_despues", "delta_km"}.issubset(cmp.columns):
        grp = cmp.assign(turno_norm=cmp["turno_despues"].map(turno_label))
        g = grp.groupby(["inst_departamento_despues", "turno_norm"], dropna=True)["delta_km"].mean().reset_index()
        if not g.empty:
            fig2 = px.bar(g, x="inst_departamento_despues", y="delta_km", color="turno_norm",
                          barmode="group", title="Δ promedio de distancia por departamento y turno")
            fig2.update_layout(yaxis_title="Δ km (después - antes)", xaxis_title="Departamento (después)")
            st.plotly_chart(fig2, width="stretch")

    # Mapa de trayectorias Antes → Después (sólo reasignados). Cada reasignado
    # aporta 2 líneas (antes gris punteada, después coloreada por Δ);
    # max_lines limita el total para no saturar el navegador.
    st.write("### 🗺️ Mapa de reasignados (Antes → Después)")
    max_lines = 1000
    reas = cmp[cmp["reasignado"]]
    if reas.empty:
        st.info("Ningún estudiante cambia de establecimiento en esta solución.")
    else:
        mcmp = nuevo_mapa(reas["est_lat_despues"], reas["est_lng_despues"])
        capa_lineas = FeatureGroup(name="Reasignaciones", show=True)

        def _color_por_delta(delta):
            """Verde si mejora (<0), rojo si empeora (>0), gris si igual/NaN."""
            if pd.isna(delta): return "#888888"
            return "#2ca02c" if delta < 0 else ("#d62728" if delta > 0 else "#888888")

        count = 0
        for _, r in reas.iterrows():
            if count >= max_lines:
                break
            est_pt = (r.get("est_lat_despues"), r.get("est_lng_despues"))
            estb_before = (r.get("estb_lat_antes"), r.get("estb_lng_antes"))
            estb_after = (r.get("estb_lat_despues"), r.get("estb_lng_despues"))
            if _valid_coords(*est_pt, *estb_before):
                folium.PolyLine([est_pt, estb_before], color="#888888", weight=2, opacity=0.7,
                                dash_array="5,5", tooltip="Antes").add_to(capa_lineas)
                count += 1
            if _valid_coords(*est_pt, *estb_after):
                folium.PolyLine([est_pt, estb_after], color=_color_por_delta(r.get("delta_km")), weight=3, opacity=0.9,
                                tooltip="Después",
                                popup=folium.Popup(
                                    f"<b>🎓 {r.get('estudiante','')}</b>"
                                    f"<br>🏫 {r.get('institucion_antes','')} ➜ <b>{r.get('institucion_despues','')}</b>"
                                    f"<br>👩‍🏫 {r.get('docente_antes','')} ➜ <b>{r.get('docente_despues','')}</b>"
                                    f"<br>Turno: {turno_label(r.get('turno_antes'))} ➜ <b>{turno_label(r.get('turno_despues'))}</b>"
                                    f"<br>Dist.: {r.get('dist_antes')} km ➜ <b>{r.get('dist_despues')} km</b>"
                                    f"<br>Δ km: <b>{r.get('delta_km')}</b>", max_width=360
                                )).add_to(capa_lineas)
                count += 1
        capa_lineas.add_to(mcmp)
        folium.LayerControl(collapsed=False).add_to(mcmp)
        st_folium(mcmp, width=1000, height=520, key="mapa_cmp_reasignados", returned_objects=[])
