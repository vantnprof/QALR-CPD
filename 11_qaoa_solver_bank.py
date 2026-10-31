#!/usr/bin/env python3
"""11 — QAOA on the banked QUBOs, the same instances as step 09 (C3, R3.2). Quantum simulation (Aer).

    python 11_qaoa_solver_bank.py                            (about 1-1.5 h)
    python 11_qaoa_solver_bank.py --qaoa-per-r 3 --qaoa-main 5   (smaller)

QAOA (p=2, 4096 shots, 200 COBYLA evaluations) on qaoa-per-r instances for each scaling size R <= qaoa-max-r, plus
the first qaoa-main main-bank instances (R = 20). Records gap, whether the optimum is returned, and the per-shot
probability of the optimal bitstring.
Output: revision_out/bank/bank_results.csv (QAOA rows)
Next:   python 13_ibm_hardware_repeats.py   (optional 12 first)
"""
from revision.cli import add_qaoa, run


def opts(ap):
    add_qaoa(ap)
    ap.add_argument("--qaoa-per-r", type=int, default=5, help="instances per scaling size R")
    ap.add_argument("--qaoa-max-r", type=int, default=16, help="largest scaling R")
    ap.add_argument("--qaoa-main", type=int, default=10, help="main-bank instances (R = 20, ~5 min each)")


if __name__ == "__main__":
    run("solvebank", __doc__, opts, fixed={"with_qaoa": True}, outputs=["bank/bank_results.csv"],
        next_step="python 13_ibm_hardware_repeats.py")
