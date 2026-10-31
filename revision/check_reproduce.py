"""Does the revision runner reproduce the submitted numbers?

    python 03_check_reproduce.py                 compares step 02 (revision protocol) with results/*_summary.csv
    python 03b_reproduce_exact.py                runs the submitted protocol exactly, then compares

Seed-by-seed: number of seeds with the same final rank, and the largest per-seed error difference.
Distribution: mean +- s.d. of the final rank and of the error, submitted vs new.

Two known protocol differences make step 02 differ from the submission on purpose:
  * the submitted synthetic scripts let ES and BF prune at the final sweep (QAOA did not); the revision suppresses
    the final-sweep call for every method;
  * the vectorized exhaustive search sums energies in a different order than the original loop, which can matter
    only where two masks tie to float32 precision.
Step 03b removes both and therefore tests the runner itself. The ALS (R=20) line involves no pruning: if it differs,
the difference is upstream of pruning (data file, initialization, or library numerics).
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, pstdev

OLD = {
    "synth_default": "synthetic_experiment_summary.csv",
    "amino": "acid_experiment_summary.csv",
    "kinetic": "kinetic_experiment_summary.csv",
    "covid": "covid_experiment_summary.csv",
}
METHOD_MAP = {"ES": "ALS-LR (ES)", "BF": "ALS-LR (BF)"}


def rd(p: Path):
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def _ms(v):
    v = list(v)
    return (mean(v), pstdev(v) if len(v) > 1 else 0.0) if v else (float("nan"), float("nan"))


def compare(new_runs: Path, old_dir: Path, label: str) -> float:
    new = rd(new_runs)
    if not new:
        raise SystemExit(f"no runs in {new_runs}")
    print(f"\n{label}: {new_runs}")
    print(f"{'dataset':14s} {'meth':4s} {'mu':>7s} {'n':>3s} {'same R':>7s} {'max|d err|':>10s} | "
          f"{'R submitted':>13s} {'R new':>13s} | {'err submitted':>19s} {'err new':>19s}")
    worst = 0.0
    for ds, fname in OLD.items():
        old = rd(old_dir / fname)
        if not old:
            print(f"{ds:14s} (no {fname} in {old_dir})")
            continue
        # Fixed-rank ALS at R=20: no pruning involved, so it isolates data, initialization and ALS numerics.
        na = {int(r["seed"]): r for r in new if r["dataset"] == ds and r["method"] == "ALS" and int(r["rank_init"]) == 20}
        oa = {int(float(r["seed"])): r for r in old if r["method"] == "ALS" and int(float(r["rank_init"])) == 20}
        ca = sorted(set(na) & set(oa))
        if ca:
            d = max(abs(float(na[s]["final_rel_error"]) - float(oa[s]["final_rel_error"])) for s in ca)
            eo = _ms(float(oa[s]["final_rel_error"]) for s in ca)
            en = _ms(float(na[s]["final_rel_error"]) for s in ca)
            print(f"{ds:14s} {'ALS':4s} {'R=20':>7s} {len(ca):3d} {'--':>7s} {d:10.2e} | {'':13s} {'':13s} | "
                  f"{eo[0]:.4e} ± {eo[1]:.1e} {en[0]:.4e} ± {en[1]:.1e}")
        for m_new, m_old in METHOD_MAP.items():
            nrows = [r for r in new if r["dataset"] == ds and r["method"] == m_new and int(r["rank_init"]) == 20]
            if not nrows:
                continue
            for mu in sorted({float(r["mu"]) for r in nrows}):
                sub = {int(r["seed"]): r for r in nrows if math.isclose(float(r["mu"]), mu)}
                ref = {int(float(r["seed"])): r for r in old if r["method"] == m_old and r.get("prune_mu")
                       and math.isclose(float(r["prune_mu"]), mu) and int(float(r["rank_init"])) == 20}
                common = sorted(set(sub) & set(ref))
                if not common:
                    print(f"{ds:14s} {m_new:4s} {mu:7g}   no matching submitted rows")
                    continue
                same = sum(int(sub[s]["final_rank"]) == int(float(ref[s]["final_rank"])) for s in common)
                d = max(abs(float(sub[s]["final_rel_error"]) - float(ref[s]["final_rel_error"])) for s in common)
                worst = max(worst, d)
                ro = _ms(float(ref[s]["final_rank"]) for s in common)
                rn = _ms(float(sub[s]["final_rank"]) for s in common)
                eo = _ms(float(ref[s]["final_rel_error"]) for s in common)
                en = _ms(float(sub[s]["final_rel_error"]) for s in common)
                print(f"{ds:14s} {m_new:4s} {mu:7g} {len(common):3d} {same:>3d}/{len(common):<3d} {d:10.2e} | "
                      f"{ro[0]:6.2f} ± {ro[1]:4.2f} {rn[0]:6.2f} ± {rn[1]:4.2f} | "
                      f"{eo[0]:.4e} ± {eo[1]:.1e} {en[0]:.4e} ± {en[1]:.1e}")
    print(f"largest per-seed error difference: {worst:.2e}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument("--old-results", default=str(repo / "results"), help="folder with the submitted *_summary.csv")
    ap.add_argument("--out", default=str(repo / "revision_out"))
    ap.add_argument("--stage", choices=["ablation", "reproduce"], default="ablation",
                    help="ablation: step 02 (revision protocol); reproduce: step 03b (submitted protocol)")
    a = ap.parse_args()
    if a.stage == "ablation" and not (Path(a.out) / "ablation" / "runs.csv").exists():
        raise SystemExit("run `python 02_ablation_inloop.py` first")
    worst = compare(Path(a.out) / a.stage / "runs.csv", Path(a.old_results),
                    "Revision protocol (step 02)" if a.stage == "ablation" else "Submitted protocol (step 03b)")
    if a.stage == "ablation":
        print("Step 02 differs from the submission on purpose on synthetic data (no ES/BF pruning at the final sweep).\n"
              "Run `python 03b_reproduce_exact.py` to test the runner against the submitted protocol.")
    else:
        print("Same rank on every seed and error differences ~1e-6 or below validate the runner.")


if __name__ == "__main__":
    main()
