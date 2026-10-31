#!/usr/bin/env python3
"""09 — Classical solvers on identical QUBOs (C3, R3.2, R4.5). Run after 08.

    python 09_solver_bank_classical.py --jobs 8              (about 5-10 min)

Solves every banked QUBO with exact search (the optimum), BF, multi-start BF (8, 32, 128 restarts), simulated
annealing, tabu search, Mag-IL and Diag-x; records gap to the optimum, success rate, per-read P(opt), evaluations,
time and TTS99.
Output: revision_out/bank/bank_results.csv
Next:   python 10_qaoa_noise_rerun.py   (quantum, long)   or   python 14_make_tables.py   (to see results so far)
"""
from revision.cli import run

if __name__ == "__main__":
    run("solvebank", __doc__, fixed={"with_qaoa": False}, outputs=["bank/bank_results.csv"],
        next_step="python 10_qaoa_noise_rerun.py")
