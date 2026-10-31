#!/usr/bin/env python3
"""10 — QAOA under the uniform non-empty rule on the noise sweep (R2.6). Quantum simulation (Aer); long.

    python 10_qaoa_noise_rerun.py                           (about 4 h on a laptop; run overnight)
    python 10_qaoa_noise_rerun.py --all-noise               (all five noise levels instead of 0.05 and 0.1)
    python 10_qaoa_noise_rerun.py --with-legacy             (also rerun the submitted cancel-on-empty behaviour)

Needs data/synthetic_noise_varying/. QAOA (p=2, 4096 shots, 200 COBYLA evaluations) and exact search, 20 seeds,
mu = 0.1. Every pruning call where the empty mask would have won is counted. Keep --jobs at 1 or 2: Aer already uses
all cores.
Output: revision_out/qaoa_noise/runs.csv
Next:   python 11_qaoa_solver_bank.py
"""
from revision.cli import add_qaoa, run


def opts(ap):
    add_qaoa(ap)
    ap.add_argument("--all-noise", action="store_true", help="all five noise levels")
    ap.add_argument("--with-legacy", action="store_true", help="also rerun the submitted QAOA behaviour")


if __name__ == "__main__":
    run("qaoa-noise", __doc__, opts, outputs=["qaoa_noise/runs.csv"], next_step="python 11_qaoa_solver_bank.py")
