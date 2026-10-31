#!/usr/bin/env python3
"""03 — Does the revision runner reproduce the submitted numbers? Run after 02.

    python 03_check_reproduce.py                       (seconds; reads <repo>/results/*_summary.csv)
    python 03_check_reproduce.py --old-results /path/to/QALR-CPD/results

Compares ES and BF runs of step 02 with the submitted summary CSVs, seed by seed. Expect identical ranks and errors
equal to ~1e-7. If anything differs by more than 1e-4, stop and check before using new numbers.
Next: python 04_synthetic_snr_rho.py --jobs 8
"""
from revision.check_reproduce import main

if __name__ == "__main__":
    main()
