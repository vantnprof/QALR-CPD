# Reproducing the paper

Two levels: regenerate every table and figure from the released records (seconds), or re-run the experiments that
produced the records (hours, mostly for the QAOA stages). Commands run from the repository root after
`pip install -r requirements.txt`.

## 1. From the released records (seconds)

```bash
python main.py tables       # reports/tables/revision/*.tex and PLACEHOLDERS.md (every value quoted in the text)
python main.py figures      # reports/figures/revision/*.png
python main.py synthetic    # regenerates the 320 tensors of the revision designs and checks them against the records
pytest                      # includes: the records regenerate the released tables byte for byte
cd reports/online_resource_1 && pdflatex online_resource_1.tex && pdflatex online_resource_1.tex   # Online Resource 1
```

For the first-version figures, restore the per-sweep records once and run the plotting scripts from the repository
root (each takes `--help`; they write to `reports/figures/v1/` and `reports/tables/v1/`):

```bash
gunzip -k results/v1/*_experiment.csv.gz
python legacy/v1/plotting/amino/plot_acid_experiment.py
```

### Tables of the revised manuscript

| Table | Content | Produced by | File in `reports/tables/revision/` |
|---|---|---|---|
| 1–4 | Notation, solvers, complexity, synthetic configuration | text | — |
| 5 | SNR and noise floor of the first-version synthetic designs | `python main.py tables` | `PLACEHOLDERS.md` |
| 6 | Settings of every experiment | `src/experiments/config.py`, `runner.py` | `results/revision/environment.json` |
| 7, 8 | Exact-SNR grid; SNR-matched collinearity | `run snr`, `run mu-rules` | `tableR2-7_snr_and_rho.tex` |
| 9 | In-loop ablation on the four tensors | `run ablation` | `tableR1_ablation_*.tex` |
| 10 | COVID-19 | `run covid20`; first-version rows | `table7_covid_fixed_rows.tex`; `results/v1/covid_experiment_summary.csv` |
| 11–13 | Synthetic, Amino Acids, X60t (first-version operating points) | `legacy/v1/plotting/`; corrected rows from `run ablation` | `reports/tables/v1/` |
| 14, 15 | Penalty selection; sensitivity to R_max | `run mu`, `run mu-rules` | `tableC1_plateaus.tex`; split-half check `tableS_heldout.tex` |
| 16 | Solvers on identical saved QUBOs | `run bank`, `run solvebank`, `run qaoa-bank` | `tableR2_solver_bank.tex`, `tableR3-2_scaling.tex` |
| 17 | Cost of one pruning call | `run profile` | `tableR4-5_runtime.tex` |
| 18 | QAOA under the uniform non-empty rule | `run qaoa-noise` | `tableR2-6_qaoa_noise.tex` |
| 19 | Hardware repeats (ibm_fez) and noise-model repeats (FakeFez, FakeTorino) | `hardware` | `tableS_hardware.tex`, `PLACEHOLDERS.md` |
| 20, 21 | First-version noise and collinearity sweeps under one protocol | `run sweeps`, `run qaoa-noise` | `tableR4-1_sweeps.tex` |
| 22 | First-version shared-snapshot records (Amino Acids) | `legacy/v1/experiments/running_time/` | `results/v1/running_time_amino*.csv` |

Also quoted in the text: the noise-calibrated starting penalty (`run sigma`, `tableC1_sigma_rule.tex`), CORCONDIA
(`run corcondia`, `tableC1_corcondia.tex`), the Proposition 2 cross term and the software versions (`PLACEHOLDERS.md`).

### Figures of the revised manuscript

| Figure | File | Produced by |
|---|---|---|
| 1, 2, 12, 13, 19 | `reports/figures/revision/FigR_*.png`, `FigS1_covid_trajectories.png` | `python main.py figures` |
| 3 | `reports/figures/v1/covid_rank_trajectory_*` | `legacy/v1/plotting/covid/plot_covid_experiment.py` |
| 4, 5 | `reports/figures/v1/synthetic_convergence_*`, `synthetic_ranktraj_*` | `legacy/v1/plotting/synthetic/plot_synthetic_experiment.py` |
| 6, 7 | `reports/figures/v1/acid_convergence_*`, `acid_rank_trajectory_*` | `legacy/v1/plotting/amino/plot_acid_experiment.py` |
| 8, 9 | `reports/figures/v1/kinetic_convergence_*`, `kinetic_rank_trajectory_*` | `legacy/v1/plotting/kinetic/plot_kinetic_experiment.py` |
| 10, 11 | `reports/figures/v1/ablation_acid_penalty_hyperparameter/` | `legacy/v1/plotting/ablation/plot_ablation_acid_penalty_hyperparameter.py` |
| 14, 15 | `reports/figures/v1/ablation_acid_shot_circuit_depth/` | `legacy/v1/plotting/ablation/plot_ablation_acid_shot_circuit_depth.py` |
| 16 | `reports/figures/v1/ablation_real_ibm_*` (first-version run on ibm_torino) | `legacy/v1/plotting/ablation/plot_ablation_acid_real_ibm_device.py` |
| 17, 18 | `reports/figures/v1/result/synthesis/scale_sweep_*` | `legacy/v1/plotting/ablation/plot_synthetic_scale_robustness_experiment.py` |

## 2. Re-running the experiments

`python main.py run STAGE` writes `results/revision/<stage>/{runs.csv, calls.csv, sweeps.csv.gz}` and resumes where it
stopped. Common options: `--jobs N`, `--seeds 0-19`, `--datasets`, `--data-root`, `--out`, `--quick`. Keep `--jobs`
at 1–2 for the QAOA stages: Aer already uses all cores. Times are for 8 cores.

| Command | What it runs | Time | Manuscript |
|---|---|---|---|
| `python main.py run covid20` | fixed-rank ALS at R = 20 and Top-K from R_max = 20 on COVID-19 | 3 min | Table 10 |
| `python main.py run ablation` | in-loop baselines vs QALR-CPD on the four tensors | 10–15 min | Table 9 |
| `python main.py compare` | stage ablation against the first-version summaries | seconds | check |
| `python main.py run reproduce` | first-version protocol (original exact-search loop), then compared | ~1 h | optional |
| `python main.py run snr` | exact-SNR grid and SNR-matched collinearity designs | ~45 min | Tables 7–8, Fig. 1 |
| `python main.py run sweeps` | first-version sweeps under one protocol | 5–10 min | Tables 20–21 |
| `python main.py run mu` | 15-value penalty grid, R_max = 10, 15, 20 (ES) and 30 (SA) | ~1 h | Tables 14–15, Fig. 12 |
| `python main.py run mu-rules` | Mag-IL and Diag-x over the same grid | 1–1.5 h | Tables 7–8, 14 |
| `python main.py run sigma` | noise-calibrated starting penalty | ~10 min | Sec. 4.4 |
| `python main.py run corcondia` | core consistency at the plateau rank | ~10 min | Sec. 4.4 |
| `python main.py run bank` | saves every pruning QUBO of exact-search runs | ~5 min | Table 16 |
| `python main.py run solvebank` | classical solvers on the saved QUBOs | 5–10 min | Table 16, Fig. 13 |
| `python main.py run qaoa-bank` | QAOA on a subset of the saved QUBOs | 1–1.5 h | Table 16, Fig. 13 |
| `python main.py run qaoa-noise` | QAOA under the uniform non-empty rule | ~4 h | Table 18 |
| `python main.py run qaoa-tables --seeds 0-4` | in-loop QAOA rows under the uniform rule | ~6 h | Figs. 2, 19 |
| `python main.py run profile --jobs 1` | cost of one pruning call | 15–30 min | Table 17 |
| `python main.py hardware` | QAOA repeats on IBM hardware or its noise model | QPU queue | Table 19 |

The first-version experiments are in `legacy/v1/experiments/` and run as modules from the repository root, e.g.
`python -m legacy.v1.experiments.amino.acid_experiment --help`; they write to `results/v1/`.

## 3. IBM hardware

```bash
python main.py hardware --backend ibm_fez --repeats 5                  # device
python main.py hardware --dry-run --backend ibm_fez --repeats 5        # FakeFez noise model, no account
python main.py hardware --dry-run --backend ibm_torino --repeats 5     # FakeTorino noise model
```

Default instance: Amino Acids, R = 12, mu = 0.1, seed 0, first pruning call (`results/revision/bank/bank`). Angles are
optimized on the noiseless simulator with the in-loop settings and the optimized circuit is sampled on the backend
(`--protocol full` runs COBYLA on the device instead). The fake backends come from
`qiskit_ibm_runtime.fake_provider` (calibration snapshots dated 2025-02-26 in qiskit-ibm-runtime 0.50.0).
`results/revision/hardware/hardware_repeats.csv` holds five repeats each on ibm_fez, FakeFez and FakeTorino, with job
IDs, calibration time, transpiled depth, two-qubit gates, routing SWAPs, layout and QPU time. Credentials: a saved
account (`QiskitRuntimeService.save_account`), the `IBM_API_KEY` / `IBM_CRN` / `IBM_CHANNEL` environment variables,
or `--ibm-config file.json` kept outside the repository.

## 4. What to expect on another machine

- The records were produced on an AMD Ryzen Threadripper PRO 5955WX with the versions in
  `results/revision/environment.json`. ALS runs in float32, so another CPU or library version changes results in the
  last digits. Re-running 184 runs of stage ablation (seeds 0–1; synthetic, Amino Acids, COVID-19) on an Intel Xeon
  with the pinned versions reproduced the recorded rank of every synthetic and COVID-19 run, with errors within
  1.4e-5; on Amino Acids, whose components are nearly collinear, a few runs whose pruning decision is a near tie took
  a different path (exact search 5 of 6 runs with the recorded rank, Diag-x 1 of 6).
- `python main.py compare` differs from the first version on purpose on the default synthetic benchmark (the first-version
  scripts let ES and BF, but not QAOA, prune at the final sweep); on Amino Acids and X60t the two agree in
  distribution but not seed by seed.
- `python main.py synthetic` agrees with the recorded noise floors to about 1e-15.

## 5. Definitions

- **Penalty.** `mu` is dimensionless; every solver receives the QUBO with mu_eff = mu ||X||_F^2 / R_max.
- **Penalty grid** of the selection protocol: 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1,
  1.5, 2. A plateau is selected only if it spans at least three grid values; mu is its middle value (the upper of
  the two middle values for an even number).
- **Top-K.** TopK-PH truncates the R_max fit post hoc to the rank selected by exact search; TopK-PH+refit adds 50 ALS
  sweeps; TopK-IL truncates inside the loop with K matched call by call to exact search.
- **Gap.** `gap_pct` = 100 (f − f*) / f* with f the full penalized objective; `gap_pct_xnorm2` = 100 (f − f*) / ||X||^2.
- **SA and tabu** solve the unconstrained QUBO; the best non-empty read is polished by the constrained one-flip descent.
- **X60t missing entries** are set to zero for the ALS updates; every error on X60t is over the observed entries.

## 6. Changes to the first-version code

| Module | Change | Why |
|---|---|---|
| `src/pruning/exhaustive.py` | Exact search enumerates masks in vectorized chunks: same argmin and tie-breaking, about 100x faster. `QALR_ES_IMPL=loop` restores the per-mask loop. | Makes the sweeps affordable |
| `src/pruning/qaoa.py` | `require_nonempty` is honoured: QAOA returns the best non-empty sampled bitstring (else the single best component, as BF does). `qaoa_solve_qubo(G, c, mu)` is shared by the loop and the benchmark. | Same empty-mask rule for every solver |
| `legacy/v1/experiments/ablation/ablation_acid_real_ibm_device.py` | Credentials removed from the source. | Security |
| `legacy/v1/plotting/ablation/plot_ablation_acid_ibm_device.py` | Curves of the depolarizing noise model labelled as such. | Accuracy |

Module names of the first version: `src/classical_alternative_leastsquare/classical_als.py` is `src/cp/als.py`,
and `src/rank_prunning/{exhaustive_search,bit_flipping,qaoa,threshold}_rank_prunning.py` are
`src/pruning/{exhaustive,bitflip,qaoa,threshold}.py`.
