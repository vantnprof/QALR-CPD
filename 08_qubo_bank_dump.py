#!/usr/bin/env python3
"""08 — Save the pruning QUBOs every solver will be compared on (C3, R3.2).

    python 08_qubo_bank_dump.py --jobs 8                     (about 5 min)

Runs exact-search pruning with QUBO dumping: main bank = the four tensors at the Table 4-7 penalties (10 seeds, every
pruning call); scaling bank = synth_default and amino with R_max = 8, 10, ..., 20 (first call, R = R_max).
Output: revision_out/bank/bank/*.npz
Next:   python 09_solver_bank_classical.py --jobs 8
"""
from revision.cli import add_datasets, run

if __name__ == "__main__":
    run("bank", __doc__, add_datasets, outputs=["bank/bank/"], next_step="python 09_solver_bank_classical.py --jobs 8")
