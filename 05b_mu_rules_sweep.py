#!/usr/bin/env python3
"""05b — Magnitude rules over the full penalty grid, for a fair best-over-mu comparison with the QUBO (C1, C2).

    python 05b_mu_rules_sweep.py --jobs 8                        (~1-1.5 h on 8 cores; resumes if interrupted)

Step 05 swept the 15-value mu grid with exhaustive search (ES) only. This step sweeps the same grid with Mag-IL (in-loop
magnitude threshold = diagonal-only QUBO) and Diag-x on every tensor and synthetic design at R_max = 20, and adds ES on
the SNR-matched collinearity designs. Each rule then gets the same plateau analysis, so the comparison is not tied to
one mu value. Writes revision_out/mu_rules/. Then run 14_make_tables.py.
"""
from revision.cli import run

if __name__ == "__main__":
    run("mu-rules", __doc__, outputs=["mu_rules/runs.csv"], next_step="python 14_make_tables.py")
