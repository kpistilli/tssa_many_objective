# Many-Objective Assignment of Teachers and Students to Schools

Code and aggregated results for the article *Many-Objective Assignment of Teachers
and Students to Schools: An Experimental Evaluation of Algorithms* (K. Pistilli,
F. López-Pires, R. Benítez — Facultad Politécnica, Universidad Nacional de
Asunción).

The repository implements the many-objective Teacher and Student to School
Assignment (TSSA) problem (objectives f1–f4, optional f6 stability), the solution
framework (grade-aware categorical variation, constraint repair, greedy seed,
external Pareto archive) on top of [pymoo](https://pymoo.org), six MOEAs
(NSGA-II, NSGA-III, U-NSGA-III, C-TAEA, RVEA, SPEA2) and the full experimental
protocol (30 runs, HV / IGD+ / GD+, Kruskal–Wallis, Mann–Whitney with Holm,
Vargha–Delaney A12). A detailed description in Spanish is in
[`README_TESIS.md`](README_TESIS.md).

## Data availability

The experiments use pseudonymised records of the Paraguayan Ministry of
Education and Sciences (MEC) for Asunción, 2026. **They contain personal data
and are not included in this repository**; they can be requested from the MEC.
`resultados/tesis/datos_sha256.txt` lists the SHA-256 fingerprints of the
snapshot used, so that a copy obtained from the MEC can be checked against it.

Without the MEC data, everything can be run on:

- the toy instance in `asignacion_mec.sql` (fictitious names and coordinates):
  `--fuente ejemplo`;
- a synthetic instance with the same structure: `--fuente sintetica`.

Published results are aggregated only (objective vectors, indicators, tests).
Decision vectors (`*.npz`) and individual assignments are excluded because they
map pseudonymised students and teachers to classes.

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests          # unit tests
```

## Reproduction

```bash
# Quick run on the synthetic instance
python experiments.py --fuente sintetica --algoritmos nsga3 ctaea greedy \
    --runs 5 --pop 100 --gen 100 --salida resultados/demo

# Full campaign of the article (requires the MEC snapshot in datos_asu/)
bash campania_tesis.sh
```

`campania_tesis.sh` runs the unit tests, the convergence pilot, the main
experiment (eight configurations × four scenarios × 30 runs), the five-objective
stability study and the eight ablations; it is resumable. The optional phase 7
(`bash campania_tesis.sh 7`, `cotas_exactas.py`) computes in seconds the exact
optima of the student components of f1 (transportation LP) and f4 (bottleneck
assignment via max-flow), certifies that a feasible teacher assignment exists,
and reports the gap of every algorithm's Pareto sets to those optima. Parameters and outputs
are described in `README_TESIS.md` (sections 4 and 5).

## License

MIT — see [`LICENSE`](LICENSE).

## How to cite

Pistilli, K., López-Pires, F., Benítez, R. *Many-Objective Assignment of Teachers
and Students to Schools: An Experimental Evaluation of Algorithms*. Manuscript
under review, 2026. Software DOI: to be added (Zenodo).
