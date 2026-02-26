# QALR-CPD: Quantum Adaptive Algorithm for Low-Rank CP Decomposition

This repository contains the implementation and experiment pipeline for the paper:

**"Quantum Adaptive Algorithm for Low-Rank CP Decomposition"**  

QALR-CPD is a hybrid CP/PARAFAC training pipeline:
- CP-ALS factor updates are done classically.
- Rank adaptation is done by periodic component pruning.
- The pruning subproblem is written as a QUBO and solved with:
  - exhaustive search (`ALS-LR (ES)`),
  - bit-flipping local search (`ALS-LR (BF)`),
  - QAOA (`QALR-CPD`),
  - plus a Top-K baseline (`ALS-TOPK`).

Datasets in this repo:
- Synthetic CP benchmark: shape `(20,20,20,20)`, true rank `6`.
- Amino acid fluorescence: shape `(5,201,61)`.
- Kinetic fluorescence X60t: shape `(64,12,10,60)`.
- COVID-19 serology tensor: shape `(438,6,11)`.

## Layout

- `src/classical_alternative_leastsquare/`: CP-ALS model and training loop.
- `src/rank_prunning/`: pruning solvers (exhaustive, bit-flip, QAOA, threshold/Top-K).
- `script/exp/`: experiment entrypoints.
- `script/plot/`: plotting + table generation scripts.
- `data/`: dataset `.pt` files, generators, and conversion utilities.
- `results/`: raw sweep and summary CSV outputs.
- `figs/`, `tables/`, `figure/result/`: generated figures/tables.

## Environment Setup

Tested with Python `3.11`.

```bash
conda create -n qalr-cpd python=3.11 -y
conda activate qalr-cpd

python -m pip install --upgrade pip
python -m pip install \
  torch numpy scipy pandas matplotlib \
  qiskit qiskit-aer qiskit-ibm-runtime \
  tensorly
```

Sanity check imports:

```bash
python - <<'PY'
from src.classical_alternative_leastsquare.classical_als import CPFactorization
print('CPFactorization import OK')
PY
```

## Data Preparation

Preprocessed `.pt` files are already present in `data/`.  
If you need to regenerate them:

### Convert real datasets

```bash
python data/amino/convert_mat_format_to_pt.py \
  --mat-path data/amino/amino.mat \
  --out-path data/amino/amino.pt

python data/kinetic/convert_mat_to_pt.py \
  --mat-path data/kinetic/X60t.mat \
  --out-path data/kinetic/X60t.pt

python data/covid19/convert_tensorly_to_pt.py \
  --out-path data/covid19/covid19_serology.pt
```

### Regenerate synthetic datasets

```bash
python data/synthetic/synthetic_generator.py \
  --shape 20,20,20,20 --rank-true 6 --noise-std 0.01 --corr-rho 0.5 \
  --num-seeds 20 --seed-start 0 --out-dir data/synthetic --overwrite

python data/synthetic_noise_varying/synthetic_noise_varying_generator.py \
  --shape 20,20,20,20 --rank-true 6 --corr-rho 0.5 \
  --noise-stds 0.0,0.001,0.01,0.05,0.1 \
  --num-seeds 20 --seed-start 0 \
  --out-dir data/synthetic_noise_varying --overwrite

python data/synthetic_collinearity_varying/synthetic_collinearity_varying_generator.py \
  --shape 20,20,20,20 --rank-true 6 --noise-std 0.01 \
  --corr-rhos 0.0,0.3,0.5,0.7,0.9 \
  --num-seeds 20 --seed-start 0 \
  --out-dir data/synthetic_collinearity_varying --overwrite
```

## Quick Smoke Run (Fast)

```bash
python script/exp/amino/acid_experiment.py \
  --num-seeds 1 --seed-start 0 \
  --ranks 3 --topk-values 3 --qaoa-rank-init 3 \
  --mu-values 0.1 --qaoa-shots 128 --qaoa-maxiter 20 \
  --t-max 20 --prune-period 10 \
  --results-csv results/smoke_acid.csv \
  --summary-csv results/smoke_acid_summary.csv

python script/plot/amino/plot_acid_experiment.py \
  --results-csv results/smoke_acid.csv \
  --summary-csv results/smoke_acid_summary.csv
```

## Reproduce Main Experiments (Paper Defaults)

The paper settings are reflected in script defaults:
- max ALS sweeps `Tmax=200`
- pruning period `Tprune=40`
- `qaoa_reps=2`, `qaoa_shots=4096`, `qaoa_maxiter=200`
- 20 random seeds (for real datasets)
- `mu_scale=xnorm2_over_rank`

### Run core experiments

```bash
python script/exp/synthetic/synthetic_experiment.py
python script/exp/amino/acid_experiment.py
python script/exp/kinetic/kinetic_experiment.py
python script/exp/covid/covid_experiment.py
```

### Generate core figures/tables

```bash
python script/plot/synthetic/plot_synthetic_experiment.py --paper-output
python script/plot/amino/plot_acid_experiment.py --paper-output
python script/plot/kinetic/plot_kinetic_experiment.py --paper-output
python script/plot/covid/plot_covid_experiment.py --paper-output
```

## Reproduce Ablations

### Run ablation experiments

```bash
python script/exp/ablation/synthetic_scale_robustness_experiment.py
python script/exp/ablation/synthetic_noise_varying_experiment.py
python script/exp/ablation/synthetic_collinearity_varying_experiment.py
python script/exp/ablation/ablation_acid_penalty_hyperparameter.py
python script/exp/ablation/ablation_acid_shot_circuit_depth.py
python script/exp/ablation/ablation_acid_ibm_device.py
```

Optional real IBM hardware run:

```bash
cat > ibm_config.json <<'JSON'
{
  "API_KEY": "YOUR_IBM_API_KEY",
  "CRN": "YOUR_IBM_INSTANCE_OR_CRN",
  "backend_name": "ibm_torino",
  "channel": "ibm_quantum_platform"
}
JSON

python script/exp/ablation/ablation_acid_real_ibm_device.py \
  --ibm-config ibm_config.json
```

If you only want simulator from that script:

```bash
python script/exp/ablation/ablation_acid_real_ibm_device.py --sim-only
```

### Generate ablation figures/tables

```bash
python script/plot/ablation/plot_synthetic_scale_robustness_experiment.py
python script/plot/ablation/plot_synthetic_noise_varying_experiment.py
python script/plot/ablation/plot_synthetic_collinearity_varying_experiment.py
python script/plot/ablation/plot_ablation_acid_penalty_hyperparameter.py
python script/plot/ablation/plot_ablation_acid_shot_circuit_depth.py
python script/plot/ablation/plot_ablation_acid_ibm_device.py
python script/plot/ablation/plot_ablation_acid_real_ibm_device.py
```

## Outputs

After runs, key artifacts are written to:
- `results/*.csv`: raw sweep logs + summary metrics.
- `figs/*.png`: figures for paper and diagnostics.
- `tables/*.tex` and `tables/*.csv`: aggregated result tables.
- `figure/result/*`: paper-output exports from plotting scripts.

**Contact information**: vantn.prof@gmail.com
