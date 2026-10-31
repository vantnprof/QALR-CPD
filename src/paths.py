"""Default locations, relative to the repository root. Every command also takes explicit paths.

The repository root is the folder that contains src/ (or QALRCPD_ROOT, if set).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ["QALRCPD_ROOT"]).resolve() if os.environ.get("QALRCPD_ROOT") \
    else Path(__file__).resolve().parents[1]
DATA = ROOT / "data"                                  # real tensors and first-version synthetic tensors
RECORDS = ROOT / "results" / "revision"               # per-run records of the revision experiments
RECORDS_V1 = ROOT / "results" / "v1"                  # records of the first version
QUICK = ROOT / "results" / "quick"                    # smoke-test outputs (not tracked)
TABLES = ROOT / "reports" / "tables" / "revision"     # generated tables + PLACEHOLDERS.md
FIGURES = ROOT / "reports" / "figures" / "revision"   # generated figures
