"""Command-line interface, run through main.py at the repository root: ``python main.py <command> [options]``."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

USAGE = """usage: python main.py <command> [options]

Reproduce the paper from the released records (seconds):
  tables       result tables, value report and supplementary tables      -> reports/tables/revision/
  figures      figures added in the revision                             -> reports/figures/revision/
  synthetic    check (or --write) the generated synthetic tensors against the run records

Re-run the experiments:
  run STAGE    one experiment stage (`python main.py run --help` lists the stages)  -> results/revision/STAGE/
  hardware     QAOA on IBM hardware, or --dry-run on its fake-backend noise model
  compare      stage ablation against the first-version records

Checks:
  check        regression checks of the solvers and the loop (about 15 s)
  smoke        every stage on tiny stand-in tensors (about 2 min)       -> results/quick/

`python main.py <command> --help` shows the options of a command. Paths default to the repository layout
(data/, results/, reports/).
"""

# Stage name -> (description, typical time with --jobs 8, manuscript items)
STAGES: Dict[str, tuple] = {
    "covid20": ("fixed-rank ALS at R=20 and Top-K from R_max=20 on COVID-19", "3 min", "Table 10"),
    "ablation": ("in-loop baselines vs QALR-CPD on the four tensors, first-version penalties", "10-15 min", "Table 9"),
    "reproduce": ("first-version protocol re-run (original ES loop), then compared", "~1 h", "optional check"),
    "snr": ("exact-SNR grid and SNR-matched collinearity designs", "~45 min", "Tables 7-8, Fig. 1"),
    "sweeps": ("first-version noise and collinearity sweeps under one protocol", "5-10 min", "Tables 20-21"),
    "mu": ("15-value penalty grid with exact search; R_max = 10, 15, 20 (ES) and 30 (SA)", "~1 h", "Tables 14-15, Fig. 12"),
    "mu-rules": ("Mag-IL and Diag-x over the same penalty grid", "1-1.5 h", "Tables 7-8, 14"),
    "sigma": ("noise-calibrated starting penalty", "~10 min", "Sec. 4.4"),
    "corcondia": ("core consistency at the plateau rank and its neighbours", "~10 min", "Sec. 4.4"),
    "bank": ("saves every pruning QUBO of exact-search runs (+ scaling bank R = 8..20)", "~5 min", "Table 16"),
    "solvebank": ("classical solvers on every saved QUBO", "5-10 min", "Table 16, Fig. 13"),
    "qaoa-bank": ("QAOA on a subset of the saved QUBOs (keep --jobs 1-2)", "1-1.5 h", "Table 16, Fig. 13"),
    "qaoa-noise": ("QAOA under the uniform non-empty rule on the noise sweep (keep --jobs 1-2)", "~4 h", "Table 18"),
    "qaoa-tables": ("in-loop QAOA rows under the uniform rule (keep --jobs 1-2)", "~6 h for 5 seeds", "Figs. 2, 19"),
    "profile": ("cost of one pruning call, split by phase (use --jobs 1)", "15-30 min", "Table 17"),
}

# Defaults of every attribute a stage reads.
RUN_DEFAULTS = dict(
    jobs=1, seeds=None, datasets=None, data_seed=0, t_max=None, prune_period=None, dtype="float32",
    qaoa_shots=4096, qaoa_maxiter=200, with_qaoa=False, qaoa_per_r=5, qaoa_max_r=16, qaoa_main=10,
    with_legacy=False, all_noise=False, full_grid=False, no_synth=False, quick=False,
)


def _run_parser() -> argparse.ArgumentParser:
    from src import paths

    lines = [f"  {k:12s} {d}\n  {'':12s} [{t}; {m}]" for k, (d, t, m) in STAGES.items()]
    ap = argparse.ArgumentParser(
        prog="python main.py run", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run one experiment stage. Every stage resumes where it stopped.",
        epilog="stages (time with --jobs 8; what they produce):\n" + "\n".join(lines))
    ap.add_argument("stage", choices=list(STAGES), metavar="STAGE")
    g = ap.add_argument_group("common")
    g.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    g.add_argument("--seeds", default=None, help="e.g. 0-19 (default) or 0,1,2")
    g.add_argument("--datasets", default=None, help="subset of synth_default,amino,kinetic,covid")
    g.add_argument("--data-root", default=str(paths.DATA), help="default: data/")
    g.add_argument("--out", default=str(paths.RECORDS), help="records folder (default: results/revision)")
    g.add_argument("--quick", action="store_true", help="tiny stand-in tensors and 2 seeds (writes to results/quick)")
    g = ap.add_argument_group("synthetic designs")
    g.add_argument("--data-seed", type=int, default=0, help="k > 0 uses an independent family of tensors (seeds 1000k+s)")
    g.add_argument("--no-synth", action="store_true", help="mu, sigma, corcondia: skip the generated designs")
    g.add_argument("--full-grid", action="store_true", help="mu: the 32-value grid of the first version")
    g = ap.add_argument_group("QAOA")
    g.add_argument("--qaoa-shots", type=int, default=4096)
    g.add_argument("--qaoa-maxiter", type=int, default=200, help="COBYLA evaluation budget")
    g.add_argument("--qaoa-per-r", type=int, default=5, help="qaoa-bank: instances per scaling size R")
    g.add_argument("--qaoa-max-r", type=int, default=16, help="qaoa-bank: largest scaling R")
    g.add_argument("--qaoa-main", type=int, default=10, help="qaoa-bank: main-bank instances (R = 20)")
    g.add_argument("--all-noise", action="store_true", help="qaoa-noise: all five noise levels")
    g.add_argument("--with-legacy", action="store_true", help="qaoa-noise: also the first-version behaviour")
    g = ap.add_argument_group("loop settings (default: those of the paper)")
    g.add_argument("--t-max", type=int, default=None)
    g.add_argument("--prune-period", type=int, default=None)
    g.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    return ap


def cmd_run(argv: List[str]) -> None:
    from src import paths

    ns = _run_parser().parse_args(argv)
    args = argparse.Namespace(**{**RUN_DEFAULTS, **vars(ns)})
    stage = args.stage
    if stage == "qaoa-bank":                       # QAOA rows of the solver bank
        stage, args.with_qaoa = "solvebank", True
    from src.experiments import stages as S
    from src.experiments.runner import write_env

    if args.quick:
        if Path(args.out).resolve() == paths.RECORDS.resolve():
            args.out = str(paths.QUICK)
        S._make_quick_data(args)
    write_env(Path(args.out))
    t0 = time.time()
    print(f"=== {args.stage} | data: {args.data_root} | out: {args.out} | jobs: {args.jobs}"
          + (" | QUICK" if args.quick else ""), flush=True)
    S.STAGES[stage](args)
    folder = {"solvebank": "bank", "corcondia": "corcondia"}.get(stage, stage.replace("-", "_"))
    print(f"=== {args.stage} done in {(time.time() - t0) / 60:.1f} min -> {Path(args.out) / folder}", flush=True)
    if args.stage == "reproduce" and not args.quick:
        from src.experiments.reproduce import compare

        compare(Path(args.out) / "reproduce" / "runs.csv", paths.RECORDS_V1, "First-version protocol (stage reproduce)")
        print("Same rank on every seed and error differences ~1e-4 or below validate the runner.")


def cmd_tables(argv: List[str]) -> None:
    from src.analysis import supplementary, tables

    tables.main(argv)
    supplementary.main(argv)


def cmd_smoke(argv: List[str]) -> None:
    from src import paths

    argparse.ArgumentParser(prog="python main.py smoke",
                            description="Every stage on tiny stand-in tensors (no data needed); "
                                        "writes to results/quick/.").parse_args(argv)
    q = str(paths.QUICK)
    steps = [
        ["check", "--skip-qaoa"],
        ["run", "covid20", "--quick", "--jobs", "2"],
        ["run", "ablation", "--quick", "--jobs", "2"],
        ["run", "snr", "--quick", "--jobs", "2"],
        ["run", "mu", "--quick", "--jobs", "2"],
        ["run", "sigma", "--quick", "--jobs", "2"],
        ["run", "corcondia", "--quick"],
        ["run", "bank", "--quick", "--jobs", "2"],
        ["run", "solvebank", "--quick", "--jobs", "2"],
        ["run", "qaoa-noise", "--quick", "--with-legacy"],
        ["run", "qaoa-bank", "--quick", "--qaoa-per-r", "1", "--qaoa-main", "1"],
        ["hardware", "--dry-run", "--repeats", "1", "--shots", "512", "--maxiter", "20",
         "--bank-dir", str(Path(q) / "bank" / "bank"), "--out", str(Path(q) / "hardware")],
        ["tables", "--records", q, "--tables-dir", str(Path(q) / "tables")],
        ["synthetic", "--seeds", "0-1", "--records", q],
    ]
    for st in steps:
        print(f"\n>>> python main.py {' '.join(st)}", flush=True)
        if subprocess.run([sys.executable, str(paths.ROOT / "main.py")] + st).returncode != 0:
            sys.exit(f"smoke test failed at: python main.py {' '.join(st)}")
    print(f"\nsmoke test passed: see {Path(q) / 'tables' / 'PLACEHOLDERS.md'}")


def _lazy(module: str) -> Callable[[List[str]], None]:
    def f(argv: List[str]) -> None:
        import importlib

        importlib.import_module(module).main(argv)
    return f


COMMANDS: Dict[str, Callable[[List[str]], None]] = {
    "run": cmd_run,
    "tables": cmd_tables,
    "figures": _lazy("src.analysis.figures"),
    "synthetic": _lazy("src.data.export"),
    "hardware": _lazy("src.experiments.hardware"),
    "compare": _lazy("src.experiments.reproduce"),
    "check": _lazy("src.checks"),
    "smoke": cmd_smoke,
}


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return
    if argv[0] in ("-V", "--version"):
        from src import __version__

        print(f"QALR-CPD {__version__}")
        return
    cmd = COMMANDS.get(argv[0])
    if cmd is None:
        sys.exit(f"unknown command {argv[0]!r}\n\n{USAGE}")
    cmd(argv[1:])


if __name__ == "__main__":
    main()
