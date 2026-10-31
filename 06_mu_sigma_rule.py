#!/usr/bin/env python3
"""06 — Noise-calibrated starting penalty (C1).

    python 06_mu_sigma_rule.py --jobs 8                     (about 10 min on 8 cores)

Estimates the noise level from the residual of the R_max fit, sets
mu0 = beta * sigma_hat^2 * sum(I_n) * R_max / ||X||^2 for beta in {0.5, 1, 2, 4, 8}, and reports the rank exact search
selects at mu0 (and, on synthetic tensors, how often it equals R_true).
Output: revision_out/sigma/{mu0.csv, runs.csv}
Next:   python 07_corcondia.py
"""
from revision.cli import add_datasets, run


def opts(ap):
    add_datasets(ap)
    ap.add_argument("--no-synth", action="store_true", help="skip the generated SNR tensors")
    ap.add_argument("--data-seed", type=int, default=0)


if __name__ == "__main__":
    run("sigma", __doc__, opts, outputs=["sigma/mu0.csv", "sigma/runs.csv"], next_step="python 07_corcondia.py")
