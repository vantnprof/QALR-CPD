#!/usr/bin/env python3
"""99 — Smoke test: every step on tiny stand-in tensors, about 2 minutes, no data needed.

    python 99_smoke_test.py

Writes to revision_out/quick/ (separate from real results). Use it after installing, before the real runs.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
STEPS = [
    ["00_check_code.py", "--skip-qaoa"],
    ["01_covid_rank20.py", "--quick", "--jobs", "2"],
    ["02_ablation_inloop.py", "--quick", "--jobs", "2"],
    ["04_synthetic_snr_rho.py", "--quick", "--jobs", "2"],
    ["05_mu_selection_sweep.py", "--quick", "--jobs", "2"],
    ["06_mu_sigma_rule.py", "--quick", "--jobs", "2"],
    ["07_corcondia.py", "--quick"],
    ["08_qubo_bank_dump.py", "--quick", "--jobs", "2"],
    ["09_solver_bank_classical.py", "--quick", "--jobs", "2"],
    ["10_qaoa_noise_rerun.py", "--quick", "--with-legacy"],
    ["11_qaoa_solver_bank.py", "--quick", "--qaoa-per-r", "1", "--qaoa-main", "1"],
    ["13_ibm_hardware_repeats.py", "--dry-run", "--repeats", "1", "--shots", "512", "--maxiter", "20",
     "--bank-dir", str(REPO / "revision_out" / "quick" / "bank" / "bank"),
     "--out", str(REPO / "revision_out" / "quick" / "hardware")],
    ["14_make_tables.py", "--out", str(REPO / "revision_out" / "quick")],
]

if __name__ == "__main__":
    for step in STEPS:
        print(f"\n>>> python {' '.join(step)}", flush=True)
        r = subprocess.run([sys.executable, str(REPO / step[0])] + step[1:], cwd=REPO)
        if r.returncode != 0:
            sys.exit(f"smoke test failed at {step[0]}")
    print("\nsmoke test passed: see revision_out/quick/tables/PLACEHOLDERS.md")
