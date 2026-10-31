#!/usr/bin/env python3
"""04 — Controlled synthetic designs (R2.7). No data needed (tensors are generated).

    python 04_synthetic_snr_rho.py --jobs 8                 (about 45 min on 8 cores; 6 h on one)

Exact-SNR grid (-3, 0, 5, 10, 15, 20 dB at rho = 0.5) and SNR-matched collinearity (rho = 0, .3, .5, .7, .9 at 0 and
+10 dB), 20x20x20x20, R_true = 6, mu in {0.1, 0.3, 0.5}; same methods as step 02 plus ALS at R_true.
Output: revision_out/snr/runs.csv
Next:   python 05_mu_selection_sweep.py --jobs 8
"""
from revision.cli import run


def opts(ap):
    ap.add_argument("--data-seed", type=int, default=0, help="seed of the generated tensors")


if __name__ == "__main__":
    run("snr", __doc__, opts, outputs=["snr/runs.csv"], next_step="python 05_mu_selection_sweep.py --jobs 8")
