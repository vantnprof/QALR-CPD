#!/usr/bin/env python3
"""04b — Classical rows of the submitted noise and collinearity sweeps, re-run under one protocol (R2.6, R2.7, R4.1).

    python 04b_original_sweeps.py --jobs 8                       (~5-10 min)

The submitted synthetic scripts let ES and BF prune at the final sweep while QAOA could not. This step re-runs ALS(20),
Top-K (K = 3, 6, 10; collinearity K = 6), ES and BF at mu = 0.1 on the submitted noise-sweep tensors (sigma = 0, 1e-3,
1e-2, 5e-2, 1e-1) and collinearity-sweep tensors (rho = 0, 0.3, 0.5, 0.7, 0.9), with the submitted stopping tolerance
(1e-12) and no final-sweep pruning for any method, so these rows and the QAOA rows (step 10) follow one protocol.
Writes revision_out/sweeps/. Fills Table 19 and Table 21 of the revision.
Next: python 05_mu_selection_sweep.py --jobs 8
"""
from revision.cli import run

if __name__ == "__main__":
    run("sweeps", __doc__, outputs=["sweeps/runs.csv"], next_step="python 05_mu_selection_sweep.py --jobs 8")
