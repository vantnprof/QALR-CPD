# QALR-CPD: Quantum Optimization for Adaptive Low-Rank CP Decomposition

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Code, run records and figures for

> Van Tien Nguyen and Panagiotis Markopoulos. *Quantum Optimization for Adaptive Low-Rank CP Decomposition.*
> Submitted to Quantum Machine Intelligence.

QALR-CPD starts CP alternating least squares (ALS) from an overfactored model and, every few sweeps, decides which
components to remove by solving a quadratic unconstrained binary optimization (QUBO) problem whose quadratic matrix is
the components' Gram matrix. The QUBO is solved by exact search, bit flipping, simulated annealing, tabu search, or
QAOA (simulated, or on IBM quantum hardware).

Every number, table and figure of the revised manuscript is regenerated from the records in this repository in
seconds, and every record can be regenerated from the code and data. See [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Setup

Python 3.13 on Linux x86_64; the versions used for the paper are pinned in `requirements.txt`. Nothing is
installed besides the requirements: the code runs from this folder.

```bash
git clone https://github.com/vantnprof/QALR-CPD.git
cd QALR-CPD
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py check      # regression checks, about 15 s
pytest                    # the same checks, plus: the records regenerate the released tables (pytest --smoke: 2 min more)
```

The code runs on the CPU. The PyPI `torch` wheel includes CUDA; the smaller CPU wheel works as well
(`pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu` before installing the requirements).
If importing `qiskit_aer` fails with `cannot allocate memory in static TLS block`, preload its OpenMP library:
`export LD_PRELOAD=$(ls .venv/lib/python3.13/site-packages/qiskit_aer.libs/libgomp-*.so.1.0.0)`.

## Usage

Everything goes through `main.py`:

```bash
python main.py tables                 # result tables and value report      -> reports/tables/revision/
python main.py figures                # figures added in the revision        -> reports/figures/revision/
python main.py synthetic              # check the generated synthetic tensors against the run records

python main.py run snr --jobs 8       # re-run one experiment stage          -> results/revision/snr/
python main.py run --help             # all stages, with run times and the tables they produce
python main.py hardware --dry-run     # QAOA on the FakeFez noise model (IBM account needed without --dry-run)
python main.py smoke                  # every stage on tiny stand-in tensors, about 2 minutes
```

From Python, with the repository root as the working directory (or on `sys.path`):

```python
from dataclasses import asdict
from src.experiments.runner import RunConfig, run_one
from src.pruning.solvers import solve_qubo

# QALR-CPD with exact search on a generated tensor (R_true = 6, +10 dB), started from R_max = 20, mu = 0.65
spec = "snr10=synth:snr_db=10,rho=0.5,R=6,shape=20x20x20x20,seed=0"
run = run_one(dict(dataset=spec, method="ES", mu=0.65, seed=0, rank_init=20, cfg=asdict(RunConfig())))
print(run["summary"]["final_rank"])          # 6

# any solver of the paper on a QUBO  min_z  z'Gz - 2c'z + mu sum(z)  (ES, BF, MS-BF, SA, Tabu, QAOA, Mag-IL, ...)
result = solve_qubo("SA", G, c, mu)           # result["mask"], result["energy"]
```

The first-version experiment scripts run as modules from the repository root, for example
`python -m legacy.v1.experiments.amino.acid_experiment --help`.

## Repository layout

```
main.py                  command line: python main.py <command>
src/                     source code
├── cp/                  CP-ALS with in-loop pruning (CPFactorization)
├── pruning/             pruning QUBO; exact search, bit flipping, SA, tabu, QAOA, magnitude baselines
├── data/                dataset loading, synthetic generator, export and check of the generated tensors
├── experiments/         configuration, one adaptive-CP run, the experiment stages, QUBO bank, IBM hardware
├── analysis/            penalty-selection protocol, CORCONDIA, tables, figures
├── checks.py            regression checks
├── cli.py               the commands of main.py
└── paths.py             default folders
tests/                   pytest (regression checks, records -> tables, command line; pytest --smoke for the smoke run)
data/                    Amino Acids, X60t, COVID-19 and first-version synthetic tensors, with converters
results/
├── revision/            per-run records of the revision (runs, pruning calls, sweeps, saved QUBOs, hardware jobs)
└── v1/                  records of the first version (summaries; per-sweep files gzip-compressed)
reports/
├── tables/              revision/ (written by python main.py tables), v1/
├── figures/             revision/ (written by python main.py figures), v1/
└── online_resource_1/   Online Resource 1 (full result tables): LaTeX source and PDF
legacy/v1/               first-version experiment and plotting scripts (they use src/)
docs/REPRODUCE.md        table and figure map, run times, hardware runs, what to expect on another machine
```

## IBM hardware

The hardware runs of the revision used **ibm_fez** (156-qubit IBM Heron r2); the first-version run used
**ibm_torino** (133-qubit Heron r1), which was no longer available to us. `python main.py hardware --dry-run` runs the same
protocol on the fake backend of the device named by `--backend` (FakeFez by default, FakeTorino with
`--backend ibm_torino`). Credentials are read from a saved Qiskit account or from the `IBM_API_KEY`, `IBM_CRN` and
`IBM_CHANNEL` environment variables, never from the repository.

## Data

The real tensors are public datasets: Amino Acids fluorescence (Bro, 1997) and X60t kinetic fluorescence
(Nikolajsen et al., 2003) from the University of Copenhagen chemometrics data sets, and COVID-19 systems serology
(Tan et al., 2021), distributed with TensorLy. Processed copies and the first-version synthetic tensors are in `data/`
and on Zenodo ([doi:10.5281/zenodo.23736430](https://doi.org/10.5281/zenodo.23736430)). The synthetic designs added in
the revision are generated from their seeds (`python main.py synthetic --write DIR` saves them).

## Citation

See [CITATION.cff](CITATION.cff). Code under the MIT License; please also cite the original data sources.
