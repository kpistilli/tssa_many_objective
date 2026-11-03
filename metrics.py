# ================================================================
# metrics.py
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
# Métricas de calidad de frentes y comparación con soluciones de referencia:
#   - Normalización con ideal/nadir GLOBALES (de todas las corridas).
#   - Hipervolumen (HV), IGD+, GD+, Spacing, número de soluciones no dominadas.
#   - Dominancia de Pareto frente a la asignación del MEC (Tabla 4 de CACIC 2024).
#   - Tabla de mejora % por objetivo (Tabla 5 de CACIC 2024).
#   - Estadística: Kruskal–Wallis global; Mann–Whitney U por pares con
#     corrección por comparaciones múltiples (Holm por defecto, o Bonferroni)
#     y tamaño de efecto A12 de Vargha–Delaney (integrado desde la v2).
# Todas las funciones trabajan en forma de MINIMIZACIÓN (como pymoo).
# ================================================================

import numpy as np
import pandas as pd
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats import kruskal, mannwhitneyu


def no_dominados(F):
    """Filas de F que forman el primer frente (no dominadas); acepta F vacío."""
    F = np.atleast_2d(F)
    if len(F) == 0:
        return F
    return F[NonDominatedSorting().do(F, only_non_dominated_front=True)]


def domina(a, b):
    """True si a domina a b (minimización)."""
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a <= b) and np.any(a < b))


class Normalizador:
    """
    Normalización min–max con ideal/nadir fijos (comunes a todos los algoritmos).

    Se construye UNA vez con los frentes de todas las corridas de un escenario,
    de modo que HV, IGD+ y GD+ de distintos algoritmos queden en la misma escala
    y sean comparables. Un objetivo sin rango (constante) se divide por 1.
    """

    def __init__(self, conjuntos_F):
        allF = np.vstack([np.atleast_2d(F) for F in conjuntos_F if len(F)])
        self.ideal, self.nadir = allF.min(0), allF.max(0)
        self.rango = np.where(self.nadir - self.ideal > 1e-12, self.nadir - self.ideal, 1.0)

    def __call__(self, F):
        """Aplica la normalización: 0 = ideal global, 1 = nadir global."""
        return (np.atleast_2d(F) - self.ideal) / self.rango


def spacing(F):
    """
    Spacing de Schott (0 = distribución uniforme).

    d_i = distancia Manhattan de cada solución a su vecina más cercana;
    spacing = desvío estándar muestral de los d_i. Se aplica sobre F normalizado.
    """
    F = np.atleast_2d(F)
    if len(F) < 2:
        return 0.0
    # matriz n x n de distancias Manhattan (la diagonal se anula con inf)
    D = np.abs(F[:, None, :] - F[None, :, :]).sum(-1)
    np.fill_diagonal(D, np.inf)
    d = D.min(1)
    return float(np.sqrt(((d - d.mean()) ** 2).sum() / (len(d) - 1)))


class Indicadores:
    """
    Calcula todos los indicadores de calidad de un frente, ya normalizado.

    Args:
        normalizador: Normalizador común del escenario.
        frente_ref: frente de referencia (no dominados de la unión de todas las
            corridas); se usa para IGD+ y GD+.
        ref_point: punto de referencia del HV en el espacio normalizado
            (1.1 en cada eje: margen del 10 % sobre el nadir global).

    Uso: ind = Indicadores(norm, ref); ind(F) -> dict HV, IGD+, GD+, spacing, n_nd.
    """

    def __init__(self, normalizador, frente_ref, ref_point=1.1):
        self.norm = normalizador
        n_obj = len(normalizador.ideal)
        self.hv = HV(ref_point=np.full(n_obj, ref_point))
        self.igd = IGDPlus(normalizador(frente_ref))
        self.gd = GDPlus(normalizador(frente_ref))

    def __call__(self, F):
        # corrida sin soluciones factibles: HV = 0 y el resto indefinido
        if F is None or len(F) == 0:
            return {"HV": 0.0, "IGD+": np.nan, "GD+": np.nan, "spacing": np.nan, "n_nd": 0}
        Fn = self.norm(F)
        return {"HV": float(self.hv(Fn)), "IGD+": float(self.igd(Fn)), "GD+": float(self.gd(Fn)),
                "spacing": spacing(Fn), "n_nd": int(len(F))}


# ------------------------------------------------------------------
# Estadística
# ------------------------------------------------------------------
def a12(x, y):
    """
    A12 de Vargha–Delaney: P(X > Y) + 0.5 P(X = Y).

    Tamaño de efecto no paramétrico: 0,5 = sin diferencia; 1 = X siempre mayor.
    Compara todos los pares (x_i, y_j) por broadcasting.
    """
    x, y = np.asarray(x), np.asarray(y)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = (x[:, None] > y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(x) * len(y)))


def magnitud_a12(a):
    """Etiqueta del efecto según los umbrales de Vargha y Delaney (2000) sobre |A12 - 0,5|."""
    d = abs(a - 0.5)
    return "despreciable" if d < 0.06 else "pequeña" if d < 0.14 else "mediana" if d < 0.21 else "grande"


def _ajustar(p, metodo):
    """Corrección por comparaciones múltiples (Holm o Bonferroni); ignora NaN."""
    p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    adj = np.full_like(p, np.nan)
    if m == 0:
        return adj
    if metodo == "bonferroni":
        adj[ok] = np.minimum(1.0, p[ok] * m)
        return adj
    # Holm (step-down): los p se ordenan de menor a mayor y el de posición r
    # (r = 0, 1, ...) se multiplica por (m - r); además se
    # fuerza monotonía (un p ajustado nunca es menor que el anterior).
    idx = np.where(ok)[0][np.argsort(p[ok])]
    acumulado = 0.0
    for r, k in enumerate(idx):
        acumulado = max(acumulado, min(1.0, (m - r) * p[k]))
        adj[k] = acumulado
    return adj


def kruskal_global(df, metrica):
    """
    Kruskal–Wallis entre todos los algoritmos con al menos 2 corridas.

    df: una fila por corrida con columnas 'algoritmo' y `metrica`.
    Si p < alpha, al menos un algoritmo difiere; los pares se analizan con comparar_pares().
    """
    grupos = [g[metrica].dropna().to_numpy() for _, g in df.groupby("algoritmo")]
    grupos = [g for g in grupos if len(g) >= 2]
    # sin al menos 2 grupos, o si todos los valores son idénticos, el test no
    # está definido: se devuelve NaN en lugar de lanzar un error
    if len(grupos) < 2 or all(np.all(g == grupos[0][0]) for g in grupos):
        return {"metrica": metrica, "test": "Kruskal-Wallis", "estadistico": np.nan,
                "p_valor": np.nan, "k_algoritmos": len(grupos)}
    H, p = kruskal(*grupos)
    return {"metrica": metrica, "test": "Kruskal-Wallis", "estadistico": float(H),
            "p_valor": float(p), "k_algoritmos": len(grupos)}


def comparar_pares(df, metrica, mayor_es_mejor, correccion="holm", alpha=0.05):
    """
    Mann–Whitney bilateral por pares + A12, con p-valores corregidos por
    comparaciones múltiples (holm | bonferroni). A12 se expresa siempre como
    "probabilidad de que A sea MEJOR que B" (se invierte si menor es mejor).
    """
    algos = list(df["algoritmo"].unique())
    filas = []
    for i, a in enumerate(algos):
        for b in algos[i + 1:]:
            x = df.loc[df.algoritmo == a, metrica].dropna().to_numpy()
            y = df.loc[df.algoritmo == b, metrica].dropna().to_numpy()
            # sin datos suficientes o ambos grupos constantes e iguales -> p indefinido
            if len(x) < 2 or len(y) < 2 or (np.all(x == x[0]) and np.all(y == x[0])):
                p = np.nan
            else:
                p = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
            A = a12(x, y) if mayor_es_mejor else a12(-x, -y)
            filas.append({"metrica": metrica, "test": "Mann-Whitney", "A": a, "B": b, "p_valor": p,
                          "A12(A mejor que B)": A,
                          "efecto": magnitud_a12(A) if not np.isnan(A) else ""})
    out = pd.DataFrame(filas)
    if len(out):
        out[f"p_{correccion}"] = _ajustar(out["p_valor"].to_numpy(), correccion)
        out[f"significativo_{alpha}"] = out[f"p_{correccion}"] < alpha
    return out


def estadistica_completa(df, metricas=(("HV", True), ("IGD+", False), ("GD+", False)),
                         correccion="holm"):
    """
    Kruskal–Wallis global + pares corregidos, para varias métricas.

    metricas: tuplas (nombre, mayor_es_mejor); las ausentes en df se omiten.
    Returns: (DataFrame global, DataFrame de pares).
    """
    glob = pd.DataFrame([kruskal_global(df, m) for m, _ in metricas if m in df])
    pares = pd.concat([comparar_pares(df, m, mejor, correccion) for m, mejor in metricas if m in df],
                      ignore_index=True)
    return glob, pares


# ------------------------------------------------------------------
# Comparación contra una solución de referencia (MEC o greedy)
# ------------------------------------------------------------------
def tabla_mejora(problem, F_frente, F_ref, nombre_ref="MEC", idx_compromiso=None):
    """
    Tabla estilo CACIC 2024 (Tabla 5): valor de referencia, mejor valor del frente
    por objetivo, valor de la solución de compromiso y % de mejora.
    Valores reportados con su signo natural (maximizaciones en positivo).
    """
    F_frente = np.atleast_2d(F_frente)
    ref_r = problem.a_reportable(F_ref)[0]
    mejor_r = problem.a_reportable(F_frente.min(0))[0]
    filas = []
    for k, (obj, sentido, etq) in enumerate(zip(problem.objetivos, problem.sentidos, problem.etiquetas)):
        fila = {"objetivo": obj, "sentido": sentido, "descripcion": etq,
                "valor_referencia": ref_r[k], "mejor_del_frente": mejor_r[k]}
        # % de mejora en forma de minimización: positivo = el frente mejora a la
        # referencia (vale también para objetivos "max", que están negados)
        base = F_ref[k]
        fila["mejora_%_mejor"] = (100 * (base - F_frente[:, k].min()) / abs(base)) if base != 0 else np.nan
        if idx_compromiso is not None:
            comp = F_frente[idx_compromiso, k]
            fila["compromiso"] = problem.a_reportable(F_frente[idx_compromiso])[0][k]
            fila["mejora_%_compromiso"] = (100 * (base - comp) / abs(base)) if base != 0 else np.nan
        filas.append(fila)
    return pd.DataFrame(filas)


def dominancia_vs_ref(F_frente, F_ref):
    """
    Cuántas soluciones del frente dominan / son dominadas / no comparables con la referencia.
    (Tabla 4 de CACIC 2024.) Todo en forma de minimización.
    """
    F_frente = np.atleast_2d(F_frente)
    dom = sum(domina(f, F_ref) for f in F_frente)
    domdo = sum(domina(F_ref, f) for f in F_frente)
    return {"dominan_a_ref": int(dom), "dominadas_por_ref": int(domdo),
            "no_comparables": int(len(F_frente) - dom - domdo)}
