#!/usr/bin/env python3
"""00 — Regression checks. Run this first; every line must say PASS.

    python 00_check_code.py              (about 1 minute)
    python 00_check_code.py --skip-qaoa  (if Qiskit/Aer are not installed yet)

Checks: vectorized exact search == the submitted loop; revision bit-flipping == submitted bit-flipping;
Proposition 2 identity; diagonal-only QUBO == magnitude threshold; a keep-all pruning call reproduces fixed-rank
ALS sweep by sweep (R4.1); SA/tabu feasibility; QAOA non-empty rule (R2.6).
Next: python 01_covid_rank20.py
"""
from revision.checks import main

if __name__ == "__main__":
    main()
