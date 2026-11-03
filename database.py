# ================================================================
# database.py
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
# Descripción del módulo
# ----------------------
# Proveer:
#   1) Un "engine" de SQLAlchemy conectado a PostgreSQL, construido
#      desde variables de entorno (.env o entorno del sistema).
#   2) Funciones utilitarias para:
#        - Cargar los DataFrames base (estudiantes, docentes, clases,
#          establecimientos) con los JOIN necesarios.
#        - Probar la conectividad a la base (health check).
#
# Notas de diseño
# ---------------
# - El módulo se carga una única vez por proceso; por eso define
#   `engine` a nivel de módulo (patrón "singleton" de facto).
# - El fallo al cargar datos no debe interrumpir el proceso: devolvemos
#   DataFrames vacíos para que la UI/CLI maneje el error de forma amigable.
# - Las credenciales se obtienen con `dotenv`; en producción se recomienda
#   inyectarlas por variables de entorno del sistema (no commitear .env).
#
# Dependencias clave:
#   - SQLAlchemy (engine/lectura)
#   - pandas (tabular)
#   - python-dotenv (config)
#   - logging (observabilidad)
# ================================================================

import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.engine import URL
import logging
from dotenv import load_dotenv

# ================================
# LOGGING
# ----------------------------
# - Se usa el logger del módulo. Este módulo NO configura el logging global
#   (logging.basicConfig): eso lo hace cada punto de entrada (integrated_app.py,
#   experiments.py, el viewer), para no imponer formato/nivel a quien lo importe.
# ================================
logger = logging.getLogger(__name__)

# ================================
# CARGA DE VARIABLES DE ENTORNO
# -----------------------------
# - Prioriza variables del sistema; si existe un archivo .env
#   en el directorio de ejecución, también las carga.
# - Claves esperadas: DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME.
# - Se proveen defaults de desarrollo (no usar en producción).
# - load_dotenv() NO sobrescribe variables ya definidas en el sistema.
# ================================
load_dotenv()

# Contraseña por defecto SÓLO para desarrollo local (compatibilidad con la
# configuración anterior). Definir DB_PASSWORD en .env o en el entorno.
_PASSWORD_DESARROLLO = "postgres123"
if os.getenv("DB_PASSWORD") is None:
    logger.warning("⚠️ DB_PASSWORD no está definida; se usa la contraseña de desarrollo. "
                   "Definila en el archivo .env.")

# ÚNICA configuración de conexión del proyecto (claves compatibles con
# psycopg2.connect). La importan integrated_app.py y el viewer, de modo que
# lectura (SQLAlchemy) y escritura de resultados (psycopg2) usan lo mismo.
DB_CONFIG = {
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', _PASSWORD_DESARROLLO),
    'host': os.getenv('DB_HOST', '127.0.0.1'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'Asignacion_MEC')
}

# URL de SQLAlchemy (driver psycopg2) armada con URL.create: escapa
# automáticamente caracteres especiales del usuario/contraseña (@, :, /, #),
# que en un URI escrito a mano lo dejarían mal formado.
# Equivale a: postgresql+psycopg2://user:pass@host:port/dbname
DB_URI = URL.create(
    drivername="postgresql+psycopg2",
    username=DB_CONFIG['user'],
    password=DB_CONFIG['password'],
    host=DB_CONFIG['host'],
    port=int(DB_CONFIG['port']),
    database=DB_CONFIG['database'],
)

# ================================================================
# ENGINE GLOBAL
# --------------
# - Objeto reutilizable para todas las lecturas/consultas desde pandas.
# - Alternativas avanzadas (opcionales):
#     * pool_pre_ping=True para desconexiones ociosas
#     * pool_size / max_overflow para tunning de concurrencia
#     * connect_args={'options': '-c statement_timeout=30000'} para timeouts
# ================================================================
# Configuración elegida: pool_pre_ping verifica cada conexión antes de usarla
# (descarta conexiones cortadas por inactividad) y el pool mantiene hasta
# 5 conexiones abiertas + 10 temporales. create_engine no se conecta todavía:
# la primera conexión real ocurre en el primer engine.connect().
engine = create_engine(DB_URI, pool_pre_ping=True, pool_size=5, max_overflow=10)

def cargar_datos_desde_db():
    """
    Carga estudiantes, docentes, clases (con JOIN a establecimientos e instituciones)
    y establecimientos, usando una única conexión del pool.

    ¿Qué entrega?
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (estudiantes, docentes, clases, establecimientos)

    Estructuras mínimas esperadas:
        - estudiantes:  id->estudiante_id, nombre, grado, lat, lng, departamento, localidad, barrio
        - docentes:     id->docente_id,   nombre, grado, lat, lng, departamento, localidad, barrio
        - clases:       clase_id, grado, turno, capacidad, establecimiento_id,
                        lat (del establecimiento), lng (del establecimiento),
                        nombre_institucion, institucion_id
        - establecimientos: id, institucion_id, lat, lng

    Comportamiento ante error:
        - Loguea el error y retorna 4 DataFrames vacíos (evita romper el flujo
          en la UI). La capa superior decide cómo notificar al usuario.

    Importante:
        - Se hace reset_index(drop=True) para evitar efectos colaterales al
          iterar por posición en algoritmos posteriores.
    """
    try:
        logger.info("Cargando datos desde la base de datos...")

        with engine.connect() as conn:
            # -----------------------------
            # Tabla: ESTUDIANTES
            # -----------------------------
            estudiantes = pd.read_sql(
                """
                SELECT 
                    id AS estudiante_id,
                    nombre, grado, lat, lng,
                    departamento, localidad, barrio
                FROM estudiantes
                """,
                con=conn
            )

            # -----------------------------
            # Tabla: DOCENTES
            # -----------------------------
            docentes = pd.read_sql(
                """
                SELECT 
                    id AS docente_id,
                    nombre, grado, lat, lng,
                    departamento, localidad, barrio
                FROM docentes
                """,
                con=conn
            )

            # ---------------------------------------------------------
            # Tabla: CLASES + JOIN con ESTABLECIMIENTOS e INSTITUCIONES
            # ---------------------------------------------------------
            # - Aquí obtenemos lat/lng del establecimiento para cada clase,
            #   y el nombre/ID de la institución a la que pertenece.
            clases = pd.read_sql(
                """
                SELECT 
                    c.id AS clase_id,
                    c.grado, c.turno, c.capacidad,
                    c.establecimiento_id,
                    e.lat, e.lng,
                    i.nombre AS nombre_institucion,
                    e.institucion_id
                FROM clases c
                JOIN establecimientos e ON c.establecimiento_id = e.id
                JOIN instituciones i   ON e.institucion_id    = i.id
                """,
                con=conn
            )

            # -----------------------------
            # Tabla: ESTABLECIMIENTOS
            # -----------------------------
            establecimientos = pd.read_sql(
                """
                SELECT id, institucion_id, lat, lng
                FROM establecimientos
                """,
                con=conn
            )

            # Normalización de índices (evita problemas con iloc en optimización)
            estudiantes = estudiantes.reset_index(drop=True)
            docentes = docentes.reset_index(drop=True)
            clases = clases.reset_index(drop=True)
            establecimientos = establecimientos.reset_index(drop=True)

            logger.info(
                "Datos cargados correctamente: "
                f"{len(estudiantes)} estudiantes, "
                f"{len(docentes)} docentes, "
                f"{len(clases)} clases, "
                f"{len(establecimientos)} establecimientos"
            )

        return estudiantes, docentes, clases, establecimientos

    except Exception as e:
        # exc_info=True agrega el stack trace al log (útil para depuración)
        logger.error(f"Error al cargar datos: {e}", exc_info=True)
        # Fallback seguro: dataframes vacíos para que la UI/CLI lo maneje
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def test_conexion():
    """
    Health check de conectividad a la base de datos.

    ¿Qué hace?
        - Abre una conexión y lee la tabla `estudiantes` completa
          (SELECT * FROM estudiantes): además de validar la conexión,
          confirma que la tabla existe. Con muchos registros es más costoso que un SELECT 1.
        - Si la consulta se ejecuta sin excepciones, retorna True.
        - Cualquier error loguea y retorna False.

    Uso típico:
        >>> from database import test_conexion
        >>> test_conexion()
        True  # si todo ok

    Notas:
        - Usa el `engine` global (pool de conexiones). El context manager
          garantiza cerrar la conexión al salir del bloque.
    """
    try:
        # Alternativa más liviana (sólo conectividad), comentada:
        ##with engine.connect() as conn:
        ##    conn.execute(text("SELECT 1"))
        with engine.connect() as conn:
            df = pd.read_sql(
                "SELECT * from estudiantes", 
                con=conn)
        logger.info("✅ Conexión a la base de datos exitosa")
        return True
    except Exception as e:
        logger.error(f"❌ Error de conexión: {e}")
        return False

# Prueba manual: descomentar y ejecutar `python database.py`.
##test_conexion()