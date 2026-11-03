# ================================================================
# integrated_problem.py 
# Proyecto Conacyt-Uninter
# Tutor investigador: Dr. Fabio Lopez
# Investigador en formación: Ing. Eliana Telesca
# Versión: 1.3
#
# Descripción del módulo
# --------------
# Define el problema de optimización multiobjetivo para el asignador
# educativo usando Pymoo. Modela:
#   - Variables de decisión:
#       * XA: Para cada estudiante, el índice de clase asignada.
#       * XD_class: Para cada clase, el índice de docente asignado.
#         Nota: El valor == n_docentes indica "SIN DOCENTE" (solo válido
#               cuando la clase está inactiva, es decir, sin alumnos).
#   - Objetivos (F):
#       * F1: Distancia promedio estudiantes→establecimientos + 
#             distancia promedio docentes→establecimientos (solo clases activas).
#       * F2: Desvío estándar (std) del número de alumnos por clase
#             (minimizar → balancear carga).
#       * F3: -(proporción de docentes con exactamente 2 clases en el
#              MISMO establecimiento). Se usa negativo para poder minimizar.
#   - Restricciones (G >= 0 viola):
#       * g1: Exceso total de capacidad (sumatoria de overflow).
#       * g2: Clases activas sin docente asignado.
#       * g3: Docentes con más de 2 clases.
#       * g4: Si un docente tiene 2 clases, sus turnos deben ser distintos.
#       * g5: Incompatibilidades de grado (estudiante vs. clase).
#
# Dependencias clave:
#   - numpy, pandas, logging
#   - pymoo.core.problem.ElementwiseProblem
# ================================================================

import numpy as np
from pymoo.core.problem import ElementwiseProblem
import logging
import pandas as pd

# Logger específico del módulo (evita ruido de otros módulos)
logger = logging.getLogger("integrated_problem")
logger.setLevel(logging.INFO)


class IntegratedProblem(ElementwiseProblem):
    """
    Problema de optimización multiobjetivo para asignaciones educativas.

    Variables:
      - XA: vector de longitud n_estudiantes con enteros en [0, n_clases-1].
      - XD_class: vector de longitud n_clases con enteros en [0, n_docentes]
                  siendo n_docentes un valor centinela que significa "SIN DOCENTE".

    Objetivos:
      - F1: Distancia promedio estudiantes + docentes.
      - F2: Desvío estándar de alumnos por clase (balanceo).
      - F3: Negativo de la proporción de docentes con 2 clases en el MISMO establecimiento.

    Restricciones (G >= 0 viola):
      - g1: Suma de excesos de capacidad por clase.
      - g2: Clase activa sin docente asignado.
      - g3: Docentes con más de 2 clases.
      - g4: Mismo docente con 2 clases en turnos iguales.
      - g5: Estudiante asignado a clase de otro grado.
    """

    def __init__(self, estudiantes: pd.DataFrame, docentes: pd.DataFrame, clases: pd.DataFrame):
        """
        Inicializa el problema validando la presencia de datos y columnas mínimas.

        Args:
            estudiantes: df con columnas mínimas {'lat','lng','nombre','grado', ...}
            docentes:    df con columnas mínimas {'lat','lng','nombre', ...}
            clases:      df con columnas mínimas {'lat','lng','grado','turno','capacidad',
                                                  'establecimiento_id','institucion_id', ...}

        Raises:
            ValueError: si algún DataFrame está vacío o faltan columnas requeridas.
        """
        # 1) Verificación de no vacíos (evita fallos aguas abajo)
        if estudiantes.empty or docentes.empty or clases.empty:
            raise ValueError("❌ Los DataFrames de entrada no pueden estar vacíos")

        # 2) Índices limpios (previene efectos colaterales con posiciones)
        estudiantes = estudiantes.reset_index(drop=True)
        docentes    = docentes.reset_index(drop=True)
        clases      = clases.reset_index(drop=True)

        # 3) Validación de esquema mínimo
        self._validar_dataframes(estudiantes, docentes, clases)

        # 4) Guarda referencias y tallas del problema
        self.estudiantes = estudiantes
        self.docentes    = docentes
        self.clases      = clases

        self.n_estudiantes = len(estudiantes)
        self.n_docentes    = len(docentes)
        self.n_clases      = len(clases)

        # 5) Dimensión del vector de decisión: [XA | XD_class]
        n_var = self.n_estudiantes + self.n_clases

        # 6) Límites inferiores/superiores (enteros)
        #    XA: clases [0 .. n_clases-1]
        #    XD: docentes [0 .. n_docentes] (el último es "sin docente")
        xl = np.concatenate([
            np.zeros(self.n_estudiantes, dtype=int),
            np.zeros(self.n_clases, dtype=int)
        ])
        xu = np.concatenate([
            np.full(self.n_estudiantes, self.n_clases - 1, dtype=int),
            np.full(self.n_clases, self.n_docentes, dtype=int)
        ])

        # 7) Inicializa el problema base de Pymoo
        super().__init__(
            n_var=n_var,
            n_obj=3,            # F1, F2, F3
            n_constr=5,         # g1..g5
            xl=xl,
            xu=xu,
            elementwise=True    # _evaluate se llama por individuo
        )

    def _validar_dataframes(self, estudiantes: pd.DataFrame, docentes: pd.DataFrame, clases: pd.DataFrame):
        """
        Verifica que existan las columnas mínimas necesarias para el cálculo
        de objetivos y restricciones. Si falta alguna, acumula y lanza error.
        """
        columnas_requeridas_est = {'lat', 'lng', 'nombre', 'grado'}
        columnas_requeridas_doc = {'lat', 'lng', 'nombre'}
        columnas_requeridas_cls = {'lat', 'lng', 'grado', 'turno', 'capacidad',
                                   'establecimiento_id', 'institucion_id'}

        def faltantes(df: pd.DataFrame, req: set) -> set:
            return req - set(df.columns)

        errores = []
        fe = faltantes(estudiantes, columnas_requeridas_est)
        fd = faltantes(docentes,    columnas_requeridas_doc)
        fc = faltantes(clases,      columnas_requeridas_cls)
        if fe: errores.append(f"Estudiantes faltan columnas: {fe}")
        if fd: errores.append(f"Docentes faltan columnas: {fd}")
        if fc: errores.append(f"Clases faltan columnas: {fc}")

        if errores:
            # Se detalla todo en un único mensaje para acortar ciclo de corrección
            raise ValueError("❌ " + " | ".join(errores))

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evalúa un individuo (vector x) y produce:
            - out["F"] = [F1, F2, F3]
            - out["G"] = [g1, g2, g3, g4, g5]

        Notas:
            * Se protege con try/except para evitar que una excepción
              rompa toda la corrida (penalización alta si hay error).
        """
        try:
            # ----- Decodificación del vector de decisión -----
            n_est = self.n_estudiantes
            XA = x[:n_est].astype(int)           # asignaciones de estudiantes → clases
            XD = x[n_est:].astype(int)           # asignaciones de clases → docentes

            # ----- Carga por clase y clases activas -----
            alumnos_por_clase = np.zeros(self.n_clases, dtype=int)
            for i, clase_idx in enumerate(XA):
                alumnos_por_clase[clase_idx] += 1
            clase_activa = alumnos_por_clase > 0  # True si tiene ≥1 alumno

            # ========== Restricciones ==========
            # g1) Exceso de capacidad total (sumatoria de overflow)
            exceso_total = 0.0
            for clase_idx in range(self.n_clases):
                cap = int(self.clases.iloc[clase_idx]["capacidad"])
                overflow = alumnos_por_clase[clase_idx] - cap
                if overflow > 0:
                    exceso_total += overflow
            g1 = exceso_total

            # g2) Clases activas sin docente (XD == n_docentes)
            clases_sin_docente = 0
            for clase_idx in range(self.n_clases):
                if clase_activa[clase_idx] and XD[clase_idx] == self.n_docentes:
                    clases_sin_docente += 1
            g2 = clases_sin_docente

            # g3) Docente con más de 2 clases
            clases_por_docente = np.zeros(self.n_docentes, dtype=int)
            for clase_idx in range(self.n_clases):
                d = XD[clase_idx]
                if d < self.n_docentes:  # ignora el centinela "sin docente"
                    clases_por_docente[d] += 1
            exceso_clases = np.sum(np.maximum(0, clases_por_docente - 2))
            g3 = exceso_clases

            # g4) Si un docente tiene 2 clases, turnos deben ser distintos
            conflictos_turno = 0
            if np.any(clases_por_docente >= 2):
                for d in range(self.n_docentes):
                    if clases_por_docente[d] >= 2:
                        turnos = [self.clases.iloc[k]["turno"]
                                  for k in range(self.n_clases)
                                  if XD[k] == d]
                        # Si hay al menos dos y no todos son distintos → conflicto
                        if len(turnos) >= 2 and len(set(turnos)) < len(turnos):
                            conflictos_turno += 1
            g4 = conflictos_turno

            # g5) Grado estudiante ≠ grado clase
            incompat = 0
            for i, clase_idx in enumerate(XA):
                ge = self.estudiantes.iloc[i]["grado"]
                gc = self.clases.iloc[clase_idx]["grado"]
                if pd.notna(ge) and pd.notna(gc):
                    if str(ge).strip() != str(gc).strip():
                        incompat += 1
            g5 = incompat

            # ========== Objetivos ==========
            # F1) Distancias: estudiantes promedio + docentes promedio (solo clases activas con docente real)
            dist_est_total = 0.0
            for i, clase_idx in enumerate(XA):
                est   = self.estudiantes.iloc[i]
                clase = self.clases.iloc[clase_idx]
                dist_est_total += self._hav((est["lat"], est["lng"]),
                                            (clase["lat"], clase["lng"]))
            dist_est_prom = dist_est_total / max(1, self.n_estudiantes)

            dist_doc_total, cnt_doc = 0.0, 0
            for clase_idx in range(self.n_clases):
                d = XD[clase_idx]
                if clase_activa[clase_idx] and d < self.n_docentes:
                    clase  = self.clases.iloc[clase_idx]
                    doc    = self.docentes.iloc[int(d)]
                    dist_doc_total += self._hav((doc["lat"], doc["lng"]),
                                                (clase["lat"], clase["lng"]))
                    cnt_doc += 1
            dist_doc_prom = dist_doc_total / max(1, cnt_doc)
            F1 = dist_est_prom + dist_doc_prom

            # F2) Balance de carga (std alumnos/clase)
            F2 = float(np.std(alumnos_por_clase))

            # F3) Negativo de la proporción de docentes con 2 clases en el MISMO establecimiento
            docentes_mismo_est = 0
            for d in range(self.n_docentes):
                # Clases asignadas al docente d
                clases_d = [k for k in range(self.n_clases) if XD[k] == d]
                if len(clases_d) == 2:
                    e1 = self.clases.iloc[clases_d[0]]["establecimiento_id"]
                    e2 = self.clases.iloc[clases_d[1]]["establecimiento_id"]
                    if pd.notna(e1) and pd.notna(e2) and int(e1) == int(e2):
                        docentes_mismo_est += 1
            F3 = - (docentes_mismo_est / max(1, self.n_docentes))

            # Salida para Pymoo
            out["F"] = [F1, F2, F3]
            out["G"] = [g1, g2, g3, g4, g5]

        except Exception as e:
            # Penalización grande para evitar que una excepción invalide toda la corrida
            logger.error(f"❌ Error en evaluación: {e}", exc_info=True)
            out["F"] = [1e10, 1e10, 1e10]
            out["G"] = [1e10, 1e10, 1e10, 1e10, 1e10]

    @staticmethod
    def _hav(loc1, loc2) -> float:
        """
        Distancia Haversine entre dos puntos (lat, lng) en kilómetros.
        Usa radio medio de la Tierra = 6371 km.
        """
        RADIO_TIERRA = 6371.0
        lat1, lon1 = np.radians(loc1)
        lat2, lon2 = np.radians(loc2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return RADIO_TIERRA * c
