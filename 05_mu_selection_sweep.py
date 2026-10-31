#!/usr/bin/env python3
"""05 — Penalty selection and R_max sensitivity (C1, R4.4).

    python 05_mu_selection_sweep.py --jobs 8                (about 1 h on 8 cores; 7 h on one)
    python 05_mu_selection_sweep.py --jobs 8 --full-grid    (the 32-value grid of the submission)

Exact search over a 15-value penalty grid for R_max in {10, 15, 20}, and simulated annealing for R_max = 30, on the
four benchmark tensors and the six SNR-grid tensors. Gives selected-rank distributions, the widest stable plateau,
and exact-recovery rates.
Output: revision_out/mu/runs.csv
Next:   python 06_mu_sigma_rule.py --jobs 8
"""
from revision.cli import add_datasets, run


def opts(ap):
    add_datasets(ap)
    ap.add_argument("--full-grid", action="store_true", help="use the 32-value penalty grid of the submission")
    ap.add_argument("--no-synth", action="store_true", help="skip the generated SNR tensors")
    ap.add_argument("--data-seed", type=int, default=0)


if __name__ == "__main__":
    run("mu", __doc__, opts, outputs=["mu/runs.csv"], next_step="python 06_mu_sigma_rule.py --jobs 8")
