#!/usr/bin/env python3
"""07 — Core consistency (CORCONDIA) around the selected ranks (C1). Run after 05.

    python 07_corcondia.py                                   (about 10 min)

For each tensor, takes the widest stable rank plateau from step 05 and computes CORCONDIA for fixed-rank fits at
that rank and its two neighbours (5 seeds).
Output: revision_out/corcondia/corcondia.csv
Next:   python 08_qubo_bank_dump.py --jobs 8
"""
from revision.cli import add_datasets, run


def opts(ap):
    add_datasets(ap)
    ap.add_argument("--no-synth", action="store_true", help="skip the generated SNR tensors")
    ap.add_argument("--data-seed", type=int, default=0)


if __name__ == "__main__":
    run("corcondia", __doc__, opts, outputs=["corcondia/corcondia.csv"], next_step="python 08_qubo_bank_dump.py --jobs 8")
