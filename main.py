#!/usr/bin/env python3
"""QALR-CPD command line. Run from anywhere; `python main.py --help` lists the commands.

    python main.py check                 regression checks
    python main.py tables                tables from the released records   -> reports/tables/revision/
    python main.py figures               figures from the released records  -> reports/figures/revision/
    python main.py run snr --jobs 8      re-run one experiment stage        -> results/revision/snr/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
