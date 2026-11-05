# ================================================================
# cotas_exactas_asu.py  —  Cotas exactas de la componente de alumnos
# 
# Extensión many-objective (NSGA-III, U-NSGA-III, C-TAEA, RVEA, SPEA2; 3 a 6 objetivos):
#   "Many-Objective Assignment of Teachers and Students to Schools: An Experimental 
#   Evaluation of Algorithms" (K. Pistilli, F. López-Pires, R. Benítez — 
#   Facultad Politécnica, Universidad Nacional de Asunción)
#
# No es una corrida evolutiva: es un cálculo determinista sobre los datos
# congelados (datos_asu/, fase 1). No depende de las fases 2–6.
#
#   (1) Óptimo exacto de la distancia MEDIA de alumnos (componente de f1)
#       bajo R1 (una clase por alumno), R2 (capacidad) y compatibilidad de grado.
#       Problema de transporte por grado (alumno -> establecimiento con la
#       capacidad agregada de sus clases del grado). Es totalmente unimodular:
#       el LP de HiGHS devuelve una solución entera.
#   (2) Óptimo exacto de la PEOR distancia de alumnos (componente de f4):
#       cuello de botella por búsqueda binaria sobre las distancias y
#       factibilidad por flujo máximo (scipy.sparse.csgraph.maximum_flow).
#       Además: media mínima sujeta a ese peor viaje (precio de la equidad).
#   Certificado: para cada óptimo se arma una asignación de alumnos a clases y
#       se verifica con un MILP (HiGHS) que existe una asignación de docentes
#       que cumple R3–R5 (franjas horarias). Si existe, la cota es ALCANZABLE,
#       es decir, es el óptimo exacto de esa componente en el problema completo.
#   (3) Brecha: mejor valor de cada componente en los conjuntos de Pareto
#       guardados de la fase 3 (escenario completo), por algoritmo.
#
# La componente de docentes y R7 no intervienen en (1) y (2): las cotas valen
# con y sin R7.
#
# Salidas (sólo agregados, sin datos personales), en --salida:
#   cotas.csv         una fila por grado + total
#   cotas_brecha.csv  una fila por algoritmo + referencias
#   cotas_runs.csv    caché por corrida (reanudable)
#
# Uso:  artículo:  python3 codigo/cotas_exactas_asu.py
#       tesis:     python cotas_exactas.py            (fase 7 de campania_tesis.sh)
#       --tmax N corta la brecha a los N s y devuelve código 3 (volver a ejecutar).
# Requiere sólo los datos congelados de la fase 1; la brecha usa la fase 3.
# ================================================================
import argparse, glob, json, os, sys, time
import numpy as np
import pandas as pd
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import maximum_flow

AQUI = os.path.dirname(os.path.abspath(__file__))
# Dos modos con el MISMO archivo:
#   - repositorio de la tesis (tssa_many_objective/cotas_exactas.py, fase 7 de
#     campania_tesis.sh): salida en resultados/tesis/cotas_exactas/
#   - artículo (articulo_tssa/codigo/cotas_exactas_asu.py): salida en resultados_asu/
if os.path.exists(os.path.join(AQUI, "instancia.py")):
    REPO_DEF = AQUI
    OUT_DEF = os.path.join(AQUI, "resultados", "tesis", "cotas_exactas")
else:
    ART = os.path.dirname(AQUI)
    REPO_DEF = os.path.join(os.path.dirname(ART), "tssa_many_objective")
    OUT_DEF = os.path.join(ART, "resultados_asu")
OUT = OUT_DEF


def args_():
    a = argparse.ArgumentParser()
    a.add_argument("--repo", default=REPO_DEF)
    a.add_argument("--csv-dir", default="datos_asu", help="relativo a --repo")
    a.add_argument("--exp", default="resultados/tesis/principal/completo",
                   help="experimento de la fase 3 (relativo a --repo) para la brecha")
    a.add_argument("--salida", default=OUT_DEF)
    a.add_argument("--tmax", type=float, default=1e9,
                   help="segundos máximos para la brecha (código 3 = reanudar)")
    return a.parse_args()


# ---------------------------------------------------------------- (1)
def transporte(D, cap, permitido=None):
    """min sum D*x  s.a. cada alumno 1, cada sitio <= cap.  Devuelve (costo, sitio por alumno)."""
    n, m = D.shape
    mask = np.ones_like(D, bool) if permitido is None else permitido
    ii, jj = np.nonzero(mask)
    nv = len(ii)
    A_eq = coo_matrix((np.ones(nv), (ii, np.arange(nv))), shape=(n, nv)).tocsr()
    A_ub = coo_matrix((np.ones(nv), (jj, np.arange(nv))), shape=(m, nv)).tocsr()
    r = linprog(D[ii, jj], A_ub=A_ub, b_ub=cap, A_eq=A_eq, b_eq=np.ones(n),
                bounds=(0, 1), method="highs")
    if r.status != 0:
        return np.inf, None
    x = r.x
    assert np.all(np.minimum(np.abs(x), np.abs(1 - x)) < 1e-6), "LP no entero"
    sel = x > 0.5
    sitio = np.full(n, -1)
    sitio[ii[sel]] = jj[sel]
    return float(r.fun), sitio


# ---------------------------------------------------------------- (2)
def flujo_factible(D, cap, tau):
    n, m = D.shape
    ii, jj = np.nonzero(D <= tau + 1e-12)
    s, t = 0, n + m + 1
    filas = np.concatenate([np.zeros(n, int), 1 + ii, 1 + n + np.arange(m)])
    cols = np.concatenate([1 + np.arange(n), 1 + n + jj, np.full(m, t)])
    capac = np.concatenate([np.ones(n, int), np.ones(len(ii), int), cap.astype(int)]).astype(np.int32)
    G = csr_matrix((capac, (filas, cols)), shape=(n + m + 2, n + m + 2))
    r = maximum_flow(G, s, t)
    if r.flow_value < n:
        return False, None
    F = r.flow.tocoo()
    sitio = np.full(n, -1)
    sel = (F.data > 0) & (F.row >= 1) & (F.row <= n) & (F.col > n) & (F.col <= n + m)
    sitio[F.row[sel] - 1] = F.col[sel] - 1 - n
    return True, sitio


def cuello_botella(D, cap):
    vals = np.unique(D)
    lo, hi = 0, len(vals) - 1          # hi siempre factible (capacidad total >= demanda)
    while lo < hi:
        mid = (lo + hi) // 2
        if flujo_factible(D, cap, vals[mid])[0]:
            hi = mid
        else:
            lo = mid + 1
    ok, sitio = flujo_factible(D, cap, vals[lo])
    return float(vals[lo]), sitio


# ---------------------------------------------------------------- certificado
def docentes_factibles(P, XA, tlim=120):
    """¿Existe XD que cumpla R3–R5 dadas las clases activas de XA?  MILP de factibilidad."""
    n_cls = np.bincount(XA, minlength=P.nc)
    act = np.where(n_cls > 0)[0]
    na_, nd = len(act), P.nd
    nv = na_ * nd                                   # y[l, d], índice l*nd + d
    filas, cols = [], []
    # cada clase activa: exactamente un docente
    for r, l in enumerate(act):
        filas.append(np.full(nd, r)); cols.append(r * nd + np.arange(nd))
    A1 = coo_matrix((np.ones(nv), (np.concatenate(filas), np.concatenate(cols))), shape=(na_, nv))
    # cada docente y franja: a lo sumo una clase (implica R3 <= 2 y R4)
    filas, cols, k = [], [], 0
    for f in range(P.nf):
        ls = np.where(P.franjas[act, f])[0]
        for d in range(nd):
            filas.append(np.full(len(ls), k)); cols.append(ls * nd + d); k += 1
    A2 = coo_matrix((np.ones(sum(len(c) for c in cols)), (np.concatenate(filas), np.concatenate(cols))),
                    shape=(k, nv))
    r = milp(np.zeros(nv), integrality=np.ones(nv), bounds=Bounds(0, 1),
             constraints=[LinearConstraint(A1, 1, 1), LinearConstraint(A2, 0, 1)],
             options={"time_limit": tlim})
    return bool(r.status == 0), int(len(act))


def xa_desde_sitios(P, idx_alum, clases_sitio, sitio):
    """Reparte los alumnos de cada sitio entre sus clases (las de una sola franja primero)."""
    XA = {}
    for j, cls in enumerate(clases_sitio):
        al = idx_alum[sitio == j]
        orden = sorted(cls, key=lambda c: (P.franjas[c].sum(), c))
        p = 0
        for c in orden:
            toma = al[p:p + P.cap[c]]
            for i in toma:
                XA[int(i)] = c
            p += len(toma)
        assert p == len(al)
    return XA


# ---------------------------------------------------------------- main
def main():
    global OUT
    a = args_()
    OUT = a.salida
    t0 = time.time()
    sys.path.insert(0, a.repo)
    from instancia import cargar_desde_csv
    from integrated_problem import IntegratedProblem
    inst = cargar_desde_csv(os.path.join(a.repo, a.csv_dir))
    P = IntegratedProblem(inst["estudiantes"], inst["docentes"], inst["clases"], n_obj=4)
    cls = P.clases
    os.makedirs(OUT, exist_ok=True)

    f_cotas = os.path.join(OUT, "cotas.csv")
    if not os.path.exists(f_cotas):
        filas = []
        XA_med, XA_max, XA_lex = np.full(P.na, -1), np.full(P.na, -1), np.full(P.na, -1)
        datos_g = []
        for g in np.unique(P.g_est):
            idx_alum = np.where(P.g_est == g)[0]
            cg = np.where(P.g_cls == g)[0]
            est_ids = cls.establecimiento_id.to_numpy()[cg]
            sitios = pd.unique(est_ids)
            clases_sitio = [cg[est_ids == e] for e in sitios]
            col = np.array([cs[0] for cs in clases_sitio])       # misma coordenada por sitio
            D = P.U[np.ix_(idx_alum, col)]
            cap = np.array([P.cap[cs].sum() for cs in clases_sitio])
            t1 = time.time()
            costo, s_med = transporte(D, cap)
            t_lp = time.time() - t1
            t1 = time.time()
            tau, s_max = cuello_botella(D, cap)
            t_bn = time.time() - t1
            datos_g.append((g, idx_alum, D, cap, clases_sitio))
            for XA, s in ((XA_med, s_med), (XA_max, s_max)):
                for i, c in xa_desde_sitios(P, idx_alum, clases_sitio, s).items():
                    XA[i] = c
            filas.append(dict(grado=P.grados_vocab[g], alumnos=len(idx_alum), sitios=len(sitios),
                              clases=len(cg), capacidad=int(cap.sum()),
                              media_sin_cap=float(D.min(1).mean()), max_sin_cap=float(D.min(1).max()),
                              suma_opt=costo, media_opt=costo / len(idx_alum), max_opt=tau,
                              t_lp_s=t_lp, t_cuello_s=t_bn))
        tau_glob = max(f["max_opt"] for f in filas)
        suma_lex = 0.0
        for (g, idx_alum, D, cap, clases_sitio), f in zip(datos_g, filas):
            c_lex, s_lex = transporte(D, cap, permitido=D <= tau_glob + 1e-12)
            f["media_con_max_opt"] = c_lex / len(idx_alum)
            suma_lex += c_lex
            for i, c in xa_desde_sitios(P, idx_alum, clases_sitio, s_lex).items():
                XA_lex[i] = c
        # certificados de alcanzabilidad (asignación de docentes factible)
        cert = {}
        for nom, XA in (("media", XA_med), ("max", XA_max), ("lex", XA_lex)):
            assert (XA >= 0).all()
            d = P.U[np.arange(P.na), XA]
            t1 = time.time()
            ok, nact = docentes_factibles(P, XA)
            cert[nom] = (ok, nact, float(d.mean()), float(d.max()), time.time() - t1)
            print(f"certificado {nom}: docentes factibles={ok} clases activas={nact} "
                  f"media={d.mean():.4f} max={d.max():.4f} ({time.time()-t1:.1f}s)", flush=True)
        N = P.na
        Dall_min = np.concatenate([dg[2].min(1) for dg in datos_g])
        filas.append(dict(grado="Total", alumnos=N, sitios=cls.establecimiento_id.nunique(),
                          clases=P.nc, capacidad=int(P.cap.sum()),
                          media_sin_cap=float(Dall_min.mean()), max_sin_cap=float(Dall_min.max()),
                          suma_opt=sum(f["suma_opt"] for f in filas),
                          media_opt=sum(f["suma_opt"] for f in filas) / N, max_opt=tau_glob,
                          media_con_max_opt=suma_lex / N,
                          t_lp_s=sum(f["t_lp_s"] for f in filas),
                          t_cuello_s=sum(f["t_cuello_s"] for f in filas)))
        df = pd.DataFrame(filas)
        for nom, (ok, nact, m, mx, tt) in cert.items():
            df[f"cert_{nom}_factible"] = ok
            df[f"cert_{nom}_clases_activas"] = nact
            df[f"cert_{nom}_media"] = m
            df[f"cert_{nom}_max"] = mx
        df.to_csv(f_cotas, index=False)
        print(df.T.to_string(), flush=True)
    else:
        print("cotas.csv ya existe (borrarlo para recalcular)")

    # ------------------------------------------------------------ (3) brecha
    f_runs = os.path.join(OUT, "cotas_runs.csv")
    hechas = pd.read_csv(f_runs) if os.path.exists(f_runs) else pd.DataFrame()
    ya = set(hechas.run) if len(hechas) else set()
    base = os.path.join(a.repo, a.exp, "conjuntos_pareto")
    if not os.path.isdir(base):
        print(f"Sin conjuntos de Pareto en {base}: se omite la brecha (correr antes la fase 3).")
        return
    nuevas = []
    corte = False
    ar = np.arange(P.na)
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        run = os.path.basename(d)
        if run in ya:
            continue
        if time.time() - t0 > a.tmax:
            corte = True
            break
        z = np.load(os.path.join(d, "pareto_set.npz"))
        X, F = z["X"], z["F"]
        XA = np.clip(X[:, :P.na], 0, P.nc - 1)
        dE = P.U[ar[None, :], XA]
        med, mx = dE.mean(1), dE.max(1)
        # control: f1 - media de alumnos debe ser la media de docentes (>= 0)
        nuevas.append(dict(run=run, algoritmo=run.split("_run")[0], n_sol=len(X),
                           min_media_est=float(med.min()), min_max_est=float(mx.min()),
                           min_f1=float(F[:, 0].min()), min_f4=float(F[:, 3].min()),
                           control_min_f1_menos_media=float((F[:, 0] - med).min()),
                           control_min_f4_menos_max=float((F[:, 3] - mx).min())))
        del X, XA, dE
    todas = pd.concat([hechas, pd.DataFrame(nuevas)], ignore_index=True)
    todas.to_csv(f_runs, index=False)
    print(f"corridas procesadas: {len(todas)} (+{len(nuevas)})", flush=True)
    if corte:
        print("CORTE POR TIEMPO: volver a ejecutar")
        sys.exit(3)

    # resumen por algoritmo + referencias (greedy y MEC de la fase 3)
    cot = pd.read_csv(f_cotas).set_index("grado").loc["Total"]
    ref = pd.read_csv(os.path.join(a.repo, a.exp, "referencias.csv")).set_index("referencia")
    res = []
    for alg, g in todas.groupby("algoritmo"):
        res.append(dict(algoritmo=alg, corridas=len(g),
                        mejor_media_est=g.min_media_est.min(), mediana_media_est=g.min_media_est.median(),
                        mejor_max_est=g.min_max_est.min(), mediana_max_est=g.min_max_est.median()))
    for r in ("MEC",):
        res.append(dict(algoritmo=r, corridas=0, mejor_media_est=ref.loc[r, "dist_prom_est"],
                        mediana_media_est=ref.loc[r, "dist_prom_est"],
                        mejor_max_est=ref.loc[r, "dist_max_est"], mediana_max_est=ref.loc[r, "dist_max_est"]))
    res = pd.DataFrame(res)
    for c in ("mejor_media_est", "mediana_media_est"):
        res["brecha_" + c] = 100 * (res[c] / cot.media_opt - 1)
    for c in ("mejor_max_est", "mediana_max_est"):
        res["brecha_" + c] = 100 * (res[c] / cot.max_opt - 1)
    res.to_csv(os.path.join(OUT, "cotas_brecha.csv"), index=False)
    print(res.to_string(), flush=True)
    print(f"controles (deben ser >= 0): f1-media {todas.control_min_f1_menos_media.min():.4f}, "
          f"f4-max {todas.control_min_f4_menos_max.min():.4f}")


if __name__ == "__main__":
    main()
