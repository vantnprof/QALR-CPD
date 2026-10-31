#!/usr/bin/env python3
"""15 — Cost of one pruning call, split by phase (Table 17; R4.5, R5.4).

    python 15_runtime_profile.py --jobs 1                        (~15-30 min; keep --jobs at 1 so timings are clean)

Runs every in-loop solver (ES, BF, MS-BF, SA, tabu, QAOA) on the four tensors at the middle table penalty, R_max = 20,
3 seeds, and records per pruning call the QUBO construction time, the solver time and, for QAOA, transpilation,
parameter optimization (with the number of estimator calls), sampling and mask selection. Also records the CPU model.
Writes revision_out/profile/. Then run 14_make_tables.py.
"""
from revision.cli import run

if __name__ == "__main__":
    run("profile", __doc__, outputs=["profile/calls.csv"], next_step="python 14_make_tables.py")
