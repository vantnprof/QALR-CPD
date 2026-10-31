#!/usr/bin/env python3
"""01 — Table 7 fix (R4.1). Needs data/covid19/covid19_serology.pt.

    python 01_covid_rank20.py --jobs 4        (about 3 minutes)

Runs, for 20 seeds: fixed-rank ALS at R=20, Top-K (K=6,8,10) from R_max=20 with and without 50 refit sweeps, and
exact-search pruning at mu=1e-4. The submitted 'ALS (R=20)' row was the R=16 run; ALS at R=20 must equal the
ES mu=1e-4 row seed by seed.
Output: revision_out/covid20/runs.csv  ->  Table 7 rows in revision_out/tables/ after 14_make_tables.py
Next:   python 02_ablation_inloop.py --jobs 8
"""
from revision.cli import run

if __name__ == "__main__":
    run("covid20", __doc__, outputs=["covid20/runs.csv"], next_step="python 02_ablation_inloop.py --jobs 8")
