#!/usr/bin/env python3
"""13 — Repeated QAOA runs on IBM hardware with full provenance (R3.2). Run after 08.

    python 13_ibm_hardware_repeats.py --dry-run                         (fake Fez noise model, no account)
    python 13_ibm_hardware_repeats.py --backend ibm_fez --repeats 5   (real device; 5 sampler jobs)
    python 13_ibm_hardware_repeats.py --backend ibm_fez --protocol full --repeats 3   (COBYLA on the device)

Default instance: Amino, R = 12, seed 0, first pruning call, from revision_out/bank/bank (or pass --instance).
'sampled' (default): angles optimized on the noiseless simulator, hardware used to sample. Records job IDs, backend,
calibration time, depth, two-qubit gates, routing SWAPs, layout, QPU seconds, gap, P(opt) per shot.
Credentials: saved account, or IBM_API_KEY / IBM_CRN / IBM_CHANNEL, or --ibm-config file.json outside the repo.
Output: revision_out/hardware/hardware_repeats.csv
Next:   python 14_make_tables.py
"""
from revision.exp_hardware import main

if __name__ == "__main__":
    main()
