"""The synthetic tensors of the revision designs: check them against the run records, optionally write them.

    python main.py synthetic                                     (seconds; regenerate and check, writes nothing)
    python main.py synthetic --write data/synthetic_revision     (about 380 MB: .npz files + manifest.json)

The SNR grid (-3, 0, 5, 10, 15, 20 dB at rho = 0.5) and the SNR-matched collinearity designs (rho = 0, .3, .5, .7, .9
at 0 and +10 dB) are not stored as files: every run generates tensor s from seed s with src.data.synthetic
(numpy.random.default_rng). This command regenerates all of them and compares the realized noise floor ||N||/||X|| and
SNR with the values recorded by the runs in results/revision/{snr,mu,mu_rules,sigma}/runs.csv, which come from the
machine that ran the experiments. Agreement to ~1e-15 shows that these are the tensors the reported numbers came from.

--write saves one .npz per tensor (X and the unit-norm factors U1..U4, float64 as generated; the runs cast X to
float32; the noiseless tensor is the CP reconstruction of U1..U4 with unit weights) and manifest.json with each file's
design, seed, realized SNR, rho, floor and SHA-256. The first-version synthetic files (data/synthetic*, made by
data/synthetic/synthetic_generator.py) are not affected.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

from src.experiments import config as C
from src.data.synthetic import make_synthetic


def designs():
    """Dataset names exactly as in the run records -> (snr_db, rho)."""
    out = {f"snr{s:+d}dB_rho{C.SNR_GRID_RHO:g}": (float(s), C.SNR_GRID_RHO) for s in C.SNR_GRID_DB}
    out.update({f"rho{r:.1f}_snr{s:+d}dB": (float(s), r) for r in C.RHO_GRID for s in C.RHO_SWEEP_SNR_DB})
    return out


def recorded(out_dir: Path):
    rec = {}
    for stage in ("snr", "mu", "mu_rules", "sigma"):
        p = out_dir / stage / "runs.csv"
        if not p.exists():
            continue
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                # full-size designs only (the --quick stand-ins reuse the names with 8x8x8 tensors of rank 3)
                if (r.get("floor") and re.match(r"(snr|rho)", r["dataset"])
                        and r.get("rank_true") and int(float(r["rank_true"])) == C.SYNTH_RANK):
                    rec.setdefault((r["dataset"], int(r["seed"])), set()).add(
                        (float(r["floor"]), float(r["snr_db"]) if r.get("snr_db") else float("nan")))
    return rec


def main(argv=None):
    from src import paths

    ap = argparse.ArgumentParser(prog="python main.py synthetic", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default=str(paths.RECORDS), help="run records to check against")
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--write", default=None, help="folder to write the tensors to (default: check only)")
    a = ap.parse_args(argv)
    lo, _, hi = a.seeds.partition("-")
    seeds = range(int(lo), int(hi or lo) + 1)
    shape = tuple(int(s) for s in C.SYNTH_SHAPE.split("x"))
    rec = recorded(Path(a.records))
    if not rec:
        print(f"note: no run records under {a.records}; regenerating without the check")
    wdir = Path(a.write) if a.write else None
    if wdir:
        wdir.mkdir(parents=True, exist_ok=True)
    manifest, worst, checked = [], 0.0, 0
    for name, (snr, rho) in designs().items():
        for s in seeds:
            t = make_synthetic(shape=shape, rank_true=C.SYNTH_RANK, rho=rho, snr_db=snr, seed=s)
            for floor, snr_rec in rec.get((name, s), ()):
                worst = max(worst, abs(t["floor"] - floor) / floor, abs(t["snr_db"] - snr_rec) / max(abs(snr_rec), 1.0)
                            if snr_rec == snr_rec else 0.0)
                checked += 1
            entry = dict(design=name, seed=s, snr_db_target=snr, snr_db=t["snr_db"], rho=rho,
                         rho_realized=t["rho_realized"], rank_true=C.SYNTH_RANK, shape=list(shape),
                         floor=t["floor"], sigma=t["sigma"])
            if wdir:
                fn = wdir / name / f"{name}_seed{s:02d}.npz"
                fn.parent.mkdir(exist_ok=True)
                np.savez_compressed(fn, X=t["X"].numpy(),
                                    **{f"U{k + 1}": U.numpy() for k, U in enumerate(t["factors"])})
                entry["path"] = str(fn.relative_to(wdir))
                entry["sha256"] = hashlib.sha256(fn.read_bytes()).hexdigest()
            manifest.append(entry)
    print(f"{len(manifest)} tensors in {len(designs())} designs (seeds {a.seeds}); "
          f"{checked} recorded (floor, SNR) pairs checked; largest relative difference {worst:.1e}")
    if checked and worst > 1e-9:
        sys.exit("MISMATCH: these tensors differ from the ones recorded by the runs")
    if checked:
        print("PASS: regenerated tensors match the run records")
    if wdir:
        (wdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"wrote {wdir}/<design>/*.npz and {wdir}/manifest.json")


if __name__ == "__main__":
    main()
