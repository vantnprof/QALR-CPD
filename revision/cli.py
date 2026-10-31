"""Shared command-line plumbing for the numbered scripts at the repository root (01_..., 02_..., ...).

Every script gets the same core options; paths default to the repository root, so a script works from any working
directory:  --data-root <repo>/data   --out <repo>/revision_out
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

REPO = Path(__file__).resolve().parents[1]

# Every attribute any stage reads, with its default. Scripts expose the relevant ones as options.
DEFAULTS: Dict = dict(
    data_root=str(REPO / "data"), out=str(REPO / "revision_out"), jobs=1, seeds=None, datasets=None, data_seed=0,
    t_max=None, prune_period=None, dtype="float32", qaoa_shots=4096, qaoa_maxiter=200, with_qaoa=False,
    qaoa_per_r=5, qaoa_max_r=16, qaoa_main=10, with_legacy=False, all_noise=False, full_grid=False,
    no_synth=False, quick=False,
)


def parser(doc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=DEFAULTS["data_root"], help="folder holding amino/, kinetic/, covid19/, "
                    "synthetic/, synthetic_noise_varying/ (default: <repo>/data)")
    ap.add_argument("--out", default=DEFAULTS["out"], help="output folder (default: <repo>/revision_out)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes (e.g. 8)")
    ap.add_argument("--seeds", default=None, help="seed list, e.g. 0-19 (default) or 0,1,2")
    ap.add_argument("--quick", action="store_true", help="tiny smoke test on stand-in data (no real data needed)")
    return ap


def run(stage: str, doc: str, add_options: Optional[Callable[[argparse.ArgumentParser], None]] = None,
        fixed: Optional[Dict] = None, outputs: Iterable[str] = (), next_step: str = "") -> argparse.Namespace:
    """Parse options, then run one stage of revision.run."""
    ap = parser(doc)
    if add_options:
        add_options(ap)
    ns = ap.parse_args()
    args = argparse.Namespace(**{**DEFAULTS, **vars(ns), **(fixed or {})})
    from revision import run as R
    from revision.runner import write_env

    if args.quick:
        if Path(args.out).resolve() == Path(DEFAULTS["out"]).resolve():
            args.out = str(Path(args.out) / "quick")
        R._make_quick_data(args)
    write_env(Path(args.out))
    t0 = time.time()
    print(f"=== {stage} | data: {args.data_root} | out: {args.out} | jobs: {args.jobs}" +
          (" | QUICK" if args.quick else ""), flush=True)
    R.STAGES[stage](args)
    print(f"=== {stage} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    for o in outputs:
        print(f"    output: {Path(args.out) / o}")
    if next_step:
        print(f"    next:   {next_step}")
    return args


def add_datasets(ap):
    ap.add_argument("--datasets", default=None,
                    help="subset of synth_default,amino,kinetic,covid (names in revision/config.py)")


def add_qaoa(ap):
    ap.add_argument("--qaoa-shots", type=int, default=4096)
    ap.add_argument("--qaoa-maxiter", type=int, default=200, help="COBYLA evaluation budget")


def main_guard() -> None:
    """Make 'python NN_script.py' work from anywhere: put the repository root on sys.path."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
