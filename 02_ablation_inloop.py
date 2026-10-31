#!/usr/bin/env python3
"""02 — In-loop ablation: does the pairwise QUBO term matter? (C2: R1.3, R2.3, R3.3, R4.2, R5.6)

    python 02_ablation_inloop.py --jobs 8                     (about 10-15 min on 8 cores; 1.5 h on one)
    python 02_ablation_inloop.py --jobs 8 --datasets amino     (one tensor only)

On synth_default, amino, kinetic, covid at the penalties of Tables 4-7, same schedule, ALS budget and seeds:
ALS(R_max), ES, BF, MS-BF, SA, tabu, Mag-IL (= diagonal-only QUBO), Diag-x, then TopK-IL (K matched call by call to
ES) and TopK-PH / TopK-PH+refit (post-hoc truncation to the ES rank).
Output: revision_out/ablation/{runs.csv, calls.csv, sweeps.csv.gz}
Next:   python 03_check_reproduce.py   (confirms the new runner reproduces your submitted ES/BF numbers)
"""
from revision.cli import add_datasets, run

if __name__ == "__main__":
    run("ablation", __doc__, add_datasets, outputs=["ablation/runs.csv"], next_step="python 03_check_reproduce.py")
