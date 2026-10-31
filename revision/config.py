"""Edit this file if your data live elsewhere. Paths are relative to --data-root (default: ./data)."""

# Real data and the original synthetic benchmark (same files as the submitted experiments).
DATASETS = {
    "synth_default": "manifest:{root}/synthetic/manifest_*.json?noise=0.01&rho=0.5&seed={seed}",
    "amino": "pt:{root}/amino/*.pt",
    "kinetic": "pt:{root}/kinetic/X60t.pt",
    "covid": "pt:{root}/covid19/covid19_serology.pt",
}

# Original noise-sweep tensors (for the QAOA rerun under the uniform empty-mask rule, R2.6).
NOISE_SWEEP = {
    f"noise_{s:g}": f"manifest:{{root}}/synthetic_noise_varying/**/manifest_*.json?noise={s}&rho=0.5&seed={{seed}}"
    for s in (0.0, 0.001, 0.01, 0.05, 0.1)
}

# The submitted noise, collinearity and scale sweeps (and COVID-19) stopped at |e_t - e_{t-1}| < 1e-12; the default
# synthetic benchmark, Amino Acids and X60t ran all 200 sweeps (tol 0). Re-runs of the sweeps keep their tolerance.
SWEEP_TOL = 1e-12

# Collinearity sweep of the submission (SNR varies with rho), re-run classically under one protocol (step 04b).
RHO_SWEEP_OLD = {
    f"rho_{r:g}": f"manifest:{{root}}/synthetic_collinearity_varying/**/manifest_*.json?noise=0.01&rho={r}&seed={{seed}}"
    for r in (0.0, 0.3, 0.5, 0.7, 0.9)
}

# Penalties reported in Tables 4-7 of the submission (used for the in-loop ablation, C2).
TABLE_MU = {
    "synth_default": [0.02, 0.10, 0.90],
    "amino": [0.05, 0.55, 0.75],
    "kinetic": [0.005, 0.1, 0.8],
    "covid": [0.0001, 0.65, 0.9],
}

# New synthetic designs (R2.7): exact target SNR; rho fixed for the grid, SNR fixed for the rho sweep.
SNR_GRID_DB = [-3, 0, 5, 10, 15, 20]
SNR_GRID_RHO = 0.5
RHO_GRID = [0.0, 0.3, 0.5, 0.7, 0.9]          # same levels as the submitted collinearity sweep
RHO_SWEEP_SNR_DB = [0, 10]
SYNTH_SHAPE = "20x20x20x20"
SYNTH_RANK = 6
SYNTH_MU = [0.1, 0.3, 0.5]                    # 0.1 is the value used in the submitted noise/rho sweeps

# Penalty grid for the selection protocol (C1). FULL_MU_GRID is the grid of the submission.
MU_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.5, 2.0]
FULL_MU_GRID = [0.0001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3,
                0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.5, 2.0]
RMAX_GRID_ES = [10, 15, 20]                   # exact search feasible
RMAX_GRID_HEURISTIC = [30]                    # SA / tabu only
SIGMA_RULE_BETAS = [0.5, 1.0, 2.0, 4.0, 8.0]

# Solver bank (C3, R3.2).
BANK_SCALING_R = [8, 10, 12, 14, 16, 18, 20]
BANK_SCALING_MU = 0.1
MSBF_RESTARTS = [8, 32, 128]

SEEDS = list(range(20))
