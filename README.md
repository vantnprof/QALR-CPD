# QALR-CPD — revision code (Quantum Machine Intelligence, submission a696c662)

The submitted code (`src/`, `script/`) plus one runnable script per revision task. Run them in number order from
anywhere: `python 02_ablation_inloop.py --jobs 8`. Each script prints what it answers, where it wrote its output,
and what to run next; `python <script> --help` shows its options. Shared code lives in `revision/`.

## 1. Setup (10 minutes)

```bash
unzip QALR-CPD-revision.zip && cd QALR-CPD-revision
ln -s /path/to/your/QALR-CPD/data data          # or copy it; same layout as the submitted repo
ln -s /path/to/your/QALR-CPD/results results    # only for step 03
pip install dwave-samplers                      # everything else is already in your environment
python 00_check_code.py                         # about 1 minute; every line must say PASS
python 99_smoke_test.py                         # every step on tiny stand-in data, about 3 minutes
```

Data expected under `data/` (edit `revision/config.py` if yours differ): `synthetic/manifest_*.json`,
`synthetic_noise_varying/**/manifest_*.json`, `amino/*.pt`, `kinetic/X60t.pt`, `covid19/covid19_serology.pt`.

## 2. The scripts

Times: one full-size classical run takes about 2 s; one QAOA pruning call at R = 20 takes about 5 min (2-core machine).
Every step resumes where it stopped, so you can interrupt and rerun the same command.

| Script | Answers | Time (8 cores) | Writes (under `revision_out/`) |
|---|---|---|---|
| `00_check_code.py` | code sanity | 1 min | — |
| `01_covid_rank20.py --jobs 4` | R4.1 (Table 7) | 3 min | `covid20/runs.csv` |
| `02_ablation_inloop.py --jobs 8` | C2: R1.3, R2.3, R3.3, R4.2, R5.6 | 10–15 min | `ablation/` |
| `03_check_reproduce.py` | does the new runner match your submitted numbers? | seconds | printed table |
| `04_synthetic_snr_rho.py --jobs 8` | R2.7 | ~45 min | `snr/` |
| `05_mu_selection_sweep.py --jobs 8` | C1, R4.4 | ~1 h | `mu/` |
| `06_mu_sigma_rule.py --jobs 8` | C1 | ~10 min | `sigma/` |
| `07_corcondia.py` | C1 | ~10 min | `corcondia/` |
| `08_qubo_bank_dump.py --jobs 8` | C3, R3.2 | ~5 min | `bank/bank/*.npz` |
| `09_solver_bank_classical.py --jobs 8` | C3, R3.2, R4.5 | 5–10 min | `bank/bank_results.csv` |
| `10_qaoa_noise_rerun.py` | R2.6 | ~4 h (overnight) | `qaoa_noise/` |
| `11_qaoa_solver_bank.py` | C3, R3.2 | 1–1.5 h | QAOA rows in `bank/bank_results.csv` |
| `12_qaoa_table_rows.py --seeds 0-4` | R2.6 (optional) | ~6 h | `qaoa_tables/` |
| `13_ibm_hardware_repeats.py --backend ibm_fez` | R3.2 | QPU queue + 5 sampler jobs | `hardware/hardware_repeats.csv` |
| `14_make_tables.py` | all | seconds | `tables/PLACEHOLDERS.md`, `tables/*.tex` |
| `99_smoke_test.py` | setup test | 3 min | `quick/` |

Suggested order: 00 → 01 → 02 → **03 (stop if it does not match)** → 04–09 (one sitting, or overnight) → 10 overnight
→ 11 → 13 → 14. Run 14 at any point to see what is done so far.

Common options for 01–12: `--jobs N`, `--seeds 0-19`, `--data-root`, `--out`, `--quick`; step-specific ones are in
`--help`. Keep `--jobs` at 1–2 for the QAOA steps (10–12): Aer already uses all cores.

## 3. What each step does

- **01** — fixed-rank ALS at R = 20 and Top-K (K = 6, 8, 10) from R_max = 20 on COVID-19, plus exact search at
  mu = 1e-4. The submitted Table 7 row labelled R = 20 was the R = 16 run; ALS at R = 20 must equal the ES row.
- **02** — on synth_default, amino, kinetic and covid at the penalties of Tables 4–7, with identical schedule, ALS
  budget and seeds: ALS(R_max), ES, BF, MS-BF, SA, tabu, **Mag-IL** (lambda_r > sqrt(mu_eff), the diagonal-only QUBO
  in the removal form, Proposition 2), **Diag-x** (diagonal-only QUBO in the z form); then **TopK-IL** (in-loop Top-K
  with K matched call by call to ES) and **TopK-PH / TopK-PH+refit** (post-hoc truncation to the ES rank, without and
  with 50 refit sweeps). Step 14 reports, per tensor and penalty, the error change from moving truncation into the loop
  (TopK-PH − TopK-IL) and from replacing magnitude by the QUBO at matched rank (TopK-IL − ES).
- **03** — compares the ES and BF runs of step 02 with `results/*_summary.csv` seed by seed.
- **04** — generated tensors (20x20x20x20, R_true = 6) with exact SNR: −3, 0, 5, 10, 15, 20 dB at rho = 0.5, and
  rho in {0, .3, .5, .7, .9} at 0 dB and +10 dB; methods as in 02, mu in {0.1, 0.3, 0.5}, plus ALS at R_true.
- **05** — exact search over a 15-value penalty grid for R_max in {10, 15, 20}, SA for R_max = 30; selected-rank
  distributions, widest stable plateau, exact-recovery rates.
- **06** — mu0 = beta * sigma_hat^2 * sum(I_n) * R_max / ||X||^2 for beta in {0.5, 1, 2, 4, 8}, with sigma_hat from the
  residual of the R_max fit; the rank exact search selects at mu0.
- **07** — CORCONDIA at the plateau rank from 05 and its neighbours.
- **08 / 09 / 11** — saves every pruning QUBO of exact-search runs (main bank at the table penalties; scaling bank at
  R = 8, ..., 20) and solves each with ES, BF, MS-BF (8/32/128 restarts), SA, tabu, Mag-IL, Diag-x (09) and QAOA on a
  stratified subset (11).
- **10** — QAOA under the uniform non-empty rule on the submitted noise-sweep tensors (sigma = 0.05, 0.1) with ES;
  counts every call where the empty mask would have won. `--with-legacy` also reruns the old behaviour.
- **13** — `sampled` (default) optimizes angles on the noiseless simulator exactly like the in-loop QAOA and samples
  on the device; `full` runs COBYLA on the device. Records job IDs, backend, calibration time, depth, two-qubit gates,
  routing SWAPs, layout, QPU seconds. Credentials: saved account, `IBM_API_KEY` / `IBM_CRN` / `IBM_CHANNEL`, or
  `--ibm-config file.json` kept outside the repository. `--dry-run` uses a fake Fez noise model.

## 4. What changed in the submitted code

| File | Change | Why |
|---|---|---|
| `src/rank_prunning/exhaustive_search_rank_prunning.py` | Exact search enumerates masks in vectorized chunks: same argmin, same tie-breaking, about 100x faster. `QALR_ES_IMPL=loop` restores the per-mask loop. | Makes the sweeps affordable |
| `src/rank_prunning/qaoa_rank_prunning.py` | `require_nonempty` is honoured: QAOA returns the best nonzero sampled bitstring (else the single best component, as BF does). `qaoa_solve_qubo(G, c, mu)` is shared by the loop and the benchmark. Circuit, initialization, optimizer and sampling unchanged. | R2.6 |

All other files in `src/` and `script/` are as submitted. The IBM API keys and CRNs hard-coded in
`script/exp/ablation/ablation_acid_real_ibm_device.py` and the two `ibm_runtime_config*.json` files were removed.
**Those keys were in the zip you shared: revoke them and create new ones before making any repository public.**

## 5. Definitions to keep straight in the text

- **Penalty.** mu_eff = mu ||X||_F^2 / R_max, as in Eq. (8) and the submitted scripts.
- **Top-K.** The submission used truncation without refit on synthetic, X60t and COVID-19, but truncation plus 50 refit
  sweeps on Amino. Both variants are now reported by name (TopK-PH, TopK-PH+refit).
- **Gap.** `gap_pct` = 100 (f − f*) / f*, with f = E + ||X||² the full penalized objective; tables use its median, plus
  the stable `gap_pct_xnorm2` = 100 (f − f*) / ||X||². `gap_pct_absE` = 100 (E − E*) / |E*| is saved too; check which
  convention the submitted Table 8 used before comparing old and new numbers.
- **SA and tabu** solve the unconstrained QUBO; the best nonzero read is polished by the constrained one-flip descent.
  Instances whose unconstrained optimum is the empty mask are flagged and excluded from their per-read P(opt).
- **COVID-19 missing entries.** ALS uses the zero-filled tensor and the error is computed on observed entries, as in
  the submission; CORCONDIA on COVID-19 uses the zero-filled tensor and is labelled accordingly.
- **New synthetic tensors** come from `revision/synth.py`. The generator of the original synthetic files is not in the
  repository, so the original files are used wherever existing synthetic numbers are compared.

## 6. Back to the paper

`python 14_make_tables.py` writes `revision_out/tables/PLACEHOLDERS.md` (one line per value the response letter
needs, grouped by reviewer comment; steps not yet run are marked MISSING) and `revision_out/tables/*.tex` (drop-in
table bodies). Send `revision_out/tables/` back for the response letter and the manuscript text.
