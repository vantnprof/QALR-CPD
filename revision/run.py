"""Revision experiments, one stage per reviewer request. Run from the repository root:

    python -m revision.run <stage> ...   (legacy entry point; use the numbered scripts at the repository root)

Stages (see README.md for what each one answers and how long it takes)
  covid20      R4.1   fixed-rank ALS at R=20 and Top-K from R_max=20 on COVID-19 (+ keep-all ES check)
  ablation     C2     in-loop baselines vs QALR-CPD on the four benchmark tensors at the Table 4-7 penalties
  snr          R2.7   exact-SNR grid and SNR-matched collinearity sweep (new synthetic tensors)
  mu           C1     penalty sweep with exact search, R_max sensitivity, SA at R_max = 30
  sigma        C1     noise-calibrated starting penalty mu0(beta) and the rank it selects
  corcondia    C1     core consistency around the selected ranks
  bank         C3     dump the pruning QUBOs of exact-search runs (+ scaling instances, R = 8..20)
  solvebank    C3     ES / BF / MS-BF / SA / tabu (/ QAOA with --with-qaoa) on every banked QUBO
  qaoa-noise   R2.6   QAOA under the uniform non-empty rule vs the submitted behaviour, noise sweep
  qaoa-tables  R2.6   optional: rerun the QAOA rows of Tables 4-7 under the uniform rule
  all-classical       covid20, ablation, snr, mu, sigma, corcondia, bank, solvebank (no quantum)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from revision import config as C
from revision.runner import RunConfig, run_tasks, write_env


# ----------------------------------------------------------------------------- helpers
def spec(name: str, args) -> str:
    if args.quick and name in QUICK_REAL:
        return QUICK_REAL[name]
    return f"{name}=" + C.DATASETS[name].format(root=args.data_root, seed="{seed}")


def synth_spec(name: str, args, snr_db=None, sigma=None, rho=0.5) -> str:
    shape = "8x8x8" if args.quick else C.SYNTH_SHAPE
    R = 3 if args.quick else C.SYNTH_RANK
    key = f"snr_db={snr_db}" if snr_db is not None else f"sigma={sigma}"
    # one generated tensor per run seed (seed s -> tensor s), as in the submitted synthetic experiments;
    # --data-seed k shifts to an independent family of tensors (seeds 1000k + s)
    return f"{name}=synth:{key},rho={rho},R={R},shape={shape},seed={{seed}},seed_offset={1000 * int(args.data_seed)}"


QUICK_REAL: Dict[str, str] = {}


def base_cfg(args, **over) -> Dict:
    cfg = RunConfig(t_max=args.t_max, prune_period=args.prune_period, dtype=args.dtype,
                    qaoa_shots=args.qaoa_shots, qaoa_maxiter=args.qaoa_maxiter)
    if args.quick:
        cfg.t_max, cfg.prune_period = 40, 10
        cfg.qaoa_shots, cfg.qaoa_maxiter = 512, 20
        cfg.sa_reads, cfg.sa_sweeps, cfg.tabu_reads = 20, 200, 5
    d = asdict(cfg)
    d.update(over)
    return d


def rmax(args) -> int:
    return 8 if args.quick else 20


def seeds(args) -> List[int]:
    if args.seeds:
        a, _, b = args.seeds.partition("-")
        return list(range(int(a), int(b) + 1)) if b else [int(x) for x in args.seeds.split(",")]
    return [0, 1] if args.quick else C.SEEDS


def mus(lst, args):
    return lst[:2] if args.quick else lst


def read_runs(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_calls(path: Path) -> List[Dict]:
    return read_runs(path)


def es_rank_info(out: Path):
    """Final rank and call-by-call ranks of the ES runs in `out` (for rank-matched baselines)."""
    final, sched = {}, defaultdict(list)
    for r in read_runs(out / "runs.csv"):
        if r["method"] == "ES":
            final[(r["dataset"], float(r["mu"]), int(r["seed"]), int(r["rank_init"]))] = int(r["final_rank"])
    for c in read_calls(out / "calls.csv"):
        if c["method"] == "ES":
            sched[(c["dataset"], float(c["mu"]), int(c["seed"]), int(c["rank_init"]))].append(
                (int(c["call"]), int(c["R_after"])))
    sched = {k: [r for _, r in sorted(v)] for k, v in sched.items()}
    return final, sched


def two_phase(datasets: List[str], mu_of, args, out: Path, phase1_methods, cfg, tag="", with_matched=True,
              fixed_ranks=None):
    """Phase 1: fixed-rank ALS + rank-adaptive solvers. Phase 2: rank-matched TopK-IL and TopK-PH from ES."""
    R0 = rmax(args)
    tasks = []
    for d in datasets:
        name = d.split("=", 1)[0]
        for s in seeds(args):
            for R in sorted(set([R0] + list(fixed_ranks or []))):
                tasks.append(dict(dataset=d, method="ALS", mu=None, seed=s, rank_init=R, cfg=cfg, tag=tag))
            for mu in mu_of(name):
                for m in phase1_methods:
                    tasks.append(dict(dataset=d, method=m, mu=mu, seed=s, rank_init=R0, cfg=cfg, tag=tag))
    run_tasks(tasks, out, jobs=args.jobs)
    if not with_matched:
        return
    final, sched = es_rank_info(out)
    tasks = []
    for d in datasets:
        name = d.split("=", 1)[0]
        for s in seeds(args):
            for mu in mu_of(name):
                key = (name, float(mu), s, R0)
                if key not in final:
                    continue
                tasks.append(dict(dataset=d, method="TopK-IL", mu=mu, seed=s, rank_init=R0, cfg=cfg,
                                  schedule=sched.get(key, []), tag=tag))
                tasks.append(dict(dataset=d, method="TopK-PH", mu=mu, seed=s, rank_init=R0, K=final[key], cfg=cfg,
                                  tag=tag))
                tasks.append(dict(dataset=d, method="TopK-PH+refit", mu=mu, seed=s, rank_init=R0, K=final[key],
                                  cfg=cfg, tag=tag))
    run_tasks(tasks, out, jobs=args.jobs)


CLASSICAL_ADAPTIVE = ["ES", "BF", "MS-BF", "SA", "Tabu", "Mag-IL", "Diag-x"]


# ----------------------------------------------------------------------------- stages
def stage_covid20(args):
    out = Path(args.out) / "covid20"
    cfg = base_cfg(args)
    d = spec("covid", args)
    R0 = rmax(args)
    tasks = []
    for s in seeds(args):
        tasks.append(dict(dataset=d, method="ALS", mu=None, seed=s, rank_init=R0, cfg=cfg))
        tasks.append(dict(dataset=d, method="ES", mu=0.0001, seed=s, rank_init=R0, cfg=cfg))
        for K in ([2, 3] if args.quick else [6, 8, 10]):
            tasks.append(dict(dataset=d, method="TopK-PH", mu=None, seed=s, rank_init=R0, K=K, cfg=cfg))
            tasks.append(dict(dataset=d, method="TopK-PH+refit", mu=None, seed=s, rank_init=R0, K=K, cfg=cfg))
    run_tasks(tasks, out, jobs=args.jobs)


def stage_ablation(args):
    out = Path(args.out) / "ablation"
    cfg = base_cfg(args)
    names = args.datasets.split(",") if args.datasets else list(C.DATASETS)
    datasets = [spec(n, args) for n in names]
    two_phase(datasets, lambda n: mus(C.TABLE_MU.get(n, [0.1]), args), args, out, CLASSICAL_ADAPTIVE, cfg)


def _synthetic_designs(args):
    ds = []
    grid = C.SNR_GRID_DB[:2] if args.quick else C.SNR_GRID_DB
    for snr in grid:
        ds.append(synth_spec(f"snr{snr:+d}dB_rho{C.SNR_GRID_RHO}", args, snr_db=snr, rho=C.SNR_GRID_RHO))
    rhos = C.RHO_GRID[:2] if args.quick else C.RHO_GRID
    for snr in (C.RHO_SWEEP_SNR_DB[:1] if args.quick else C.RHO_SWEEP_SNR_DB):
        for rho in rhos:
            ds.append(synth_spec(f"rho{rho}_snr{snr:+d}dB", args, snr_db=snr, rho=rho))
    return list(dict.fromkeys(ds))


def stage_snr(args):
    out = Path(args.out) / "snr"
    cfg = base_cfg(args)
    R_true = 3 if args.quick else C.SYNTH_RANK
    two_phase(_synthetic_designs(args), lambda n: mus(C.SYNTH_MU, args), args, out, CLASSICAL_ADAPTIVE, cfg,
              fixed_ranks=[R_true])


def stage_mu(args):
    out = Path(args.out) / "mu"
    cfg = base_cfg(args)
    names = args.datasets.split(",") if args.datasets else list(C.DATASETS)
    datasets = [spec(n, args) for n in names] + (_synthetic_designs(args)[:6] if not args.no_synth else [])
    grid = C.FULL_MU_GRID if args.full_grid else C.MU_GRID
    grid = grid[::5] if args.quick else grid
    rmax_es = [6, 8] if args.quick else C.RMAX_GRID_ES
    rmax_h = [10] if args.quick else C.RMAX_GRID_HEURISTIC
    tasks = []
    for d in datasets:
        for s in seeds(args):
            for mu in grid:
                for R in rmax_es:
                    tasks.append(dict(dataset=d, method="ES", mu=mu, seed=s, rank_init=R, cfg=cfg))
                for R in rmax_h:
                    tasks.append(dict(dataset=d, method="SA", mu=mu, seed=s, rank_init=R, cfg=cfg))
    run_tasks(tasks, out, jobs=args.jobs)


def stage_mu_rules(args):
    """Step 05b: the magnitude rules over the same penalty grid as step 05, so that every selection rule gets the same
    plateau analysis (C1, C2). Mag-IL and Diag-x on every tensor and synthetic design at R_max = 20, plus ES on the
    SNR-matched collinearity designs (step 05 ran ES on the SNR grid only)."""
    out = Path(args.out) / "mu_rules"
    cfg = base_cfg(args)
    names = args.datasets.split(",") if args.datasets else list(C.DATASETS)
    designs = _synthetic_designs(args)
    n_snr = len(C.SNR_GRID_DB[:2] if args.quick else C.SNR_GRID_DB)
    snr_designs, rho_designs = designs[:n_snr], designs[n_snr:]
    grid = C.FULL_MU_GRID if args.full_grid else C.MU_GRID
    grid = grid[::5] if args.quick else grid
    R0 = rmax(args)
    tasks = []
    for d in [spec(n, args) for n in names] + snr_designs + rho_designs:
        for s in seeds(args):
            for mu in grid:
                for m in ("Mag-IL", "Diag-x"):
                    tasks.append(dict(dataset=d, method=m, mu=mu, seed=s, rank_init=R0, cfg=cfg))
                if d in rho_designs:
                    tasks.append(dict(dataset=d, method="ES", mu=mu, seed=s, rank_init=R0, cfg=cfg))
    run_tasks(tasks, out, jobs=args.jobs)


def stage_sigma(args):
    """mu0 = beta * sigma_hat^2 * sum(I_n) * R_ref / ||X||^2 (in the dimensionless units of Eq. 8)."""
    from revision.data import load_dataset

    out = Path(args.out) / "sigma"
    cfg = base_cfg(args)
    names = args.datasets.split(",") if args.datasets else list(C.DATASETS)
    datasets = [spec(n, args) for n in names] + (_synthetic_designs(args)[:6] if not args.no_synth else [])
    R0 = rmax(args)
    run_tasks([dict(dataset=d, method="ALS", mu=None, seed=s, rank_init=R0, cfg=cfg)
               for d in datasets for s in seeds(args)], out, jobs=args.jobs)
    from revision.data import resolve_spec

    runs = [r for r in read_runs(out / "runs.csv") if r["method"] == "ALS"]
    rows, tasks = [], []
    for d in datasets:
        name = d.split("=", 1)[0]
        # One mu0 per tensor: the real tensors are shared by all seeds (median residual over the inits); a synthetic
        # design has one tensor per seed, so each seed gets the mu0 of its own tensor.
        groups: Dict[str, List[int]] = {}
        for s in seeds(args):
            groups.setdefault(resolve_spec(d, s), []).append(s)
        per_beta: Dict[float, List[float]] = {b: [] for b in C.SIGMA_RULE_BETAS}
        sig_hat, sig_true, rank_true = [], [], ""
        for d_s, ss in groups.items():
            ds = load_dataset(d_s, cfg["dtype"])
            n_obs = float(ds.Omega.sum().item()) if ds.Omega is not None else float(ds.X.numel())
            dof = n_obs - R0 * (sum(ds.shape) - len(ds.shape) + 1)
            errs = sorted(float(r["final_rel_error"]) for r in runs if r["dataset"] == name and int(r["seed"]) in ss)
            if not errs or dof <= 0:
                continue
            sigma2 = (errs[len(errs) // 2] * ds.xnorm) ** 2 / dof
            sig_hat.append(math.sqrt(sigma2))
            if ds.meta.get("sigma") not in (None, ""):
                sig_true.append(float(ds.meta["sigma"]))
            rank_true = ds.meta.get("rank_true", "")
            for beta in C.SIGMA_RULE_BETAS:
                mu0 = beta * sigma2 * sum(ds.shape) * R0 / ds.xnorm ** 2
                per_beta[beta].append(mu0)
                for s in ss:
                    tasks.append(dict(dataset=d, method="ES", mu=round(mu0, 12), seed=s, rank_init=R0, cfg=cfg,
                                      tag=f"beta={beta}"))
        if not sig_hat:
            continue
        for beta in C.SIGMA_RULE_BETAS:
            m0 = per_beta[beta]
            rows.append({"dataset": name, "beta": beta, "sigma_hat": float(np.median(sig_hat)),
                         "mu0": float(np.median(m0)), "mu0_min": min(m0), "mu0_max": max(m0), "n_tensors": len(m0),
                         "sigma_true": float(np.median(sig_true)) if sig_true else "", "rank_true": rank_true})
    with open(out / "mu0.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    run_tasks(tasks, out, jobs=args.jobs)


def stage_corcondia(args):
    from revision.corcondia import corcondia_for_runs

    corcondia_for_runs(Path(args.out), args, spec, _synthetic_designs, seeds, base_cfg, rmax)


def stage_bank(args):
    out = Path(args.out) / "bank"
    cfg = base_cfg(args, dump_qubo=True, log_sweeps=False)
    names = args.datasets.split(",") if args.datasets else list(C.DATASETS)
    sd = seeds(args)[: (2 if args.quick else 10)]
    tasks = []
    for n in names:
        for mu in mus(C.TABLE_MU.get(n, [0.1]), args):
            for s in sd:
                tasks.append(dict(dataset=spec(n, args), method="ES", mu=mu, seed=s, rank_init=rmax(args), cfg=cfg,
                                  tag="main"))
    scal = [6, 8] if args.quick else C.BANK_SCALING_R
    for n in [x for x in ("synth_default", "amino") if x in names]:
        for R in scal:
            for s in sd:
                tasks.append(dict(dataset=spec(n, args), method="ES", mu=C.BANK_SCALING_MU, seed=s, rank_init=R,
                                  cfg=cfg, tag="scaling"))
    run_tasks(tasks, out, jobs=args.jobs)


def stage_solvebank(args):
    from revision.bank import solve_bank

    solve_bank(Path(args.out) / "bank", args)


def stage_qaoa_noise(args):
    out = Path(args.out) / "qaoa_noise"
    cfg = base_cfg(args, tol=C.SWEEP_TOL)  # same stopping rule as the submitted noise sweep
    levels = list(C.NOISE_SWEEP) if args.all_noise else ["noise_0.05", "noise_0.1"]
    tasks = []
    for lvl in levels:
        if args.quick:
            d = synth_spec(lvl, args, sigma=float(lvl.split("_")[1]) or 1e-9)
        else:
            d = f"{lvl}=" + C.NOISE_SWEEP[lvl].format(root=args.data_root, seed="{seed}")
        methods = ("QAOA", "ES") + (("QAOA-legacy",) if args.with_legacy else ())
        for s in seeds(args):
            for m in methods:
                tasks.append(dict(dataset=d, method=m, mu=0.1, seed=s, rank_init=rmax(args), cfg=cfg))
    run_tasks(tasks, out, jobs=args.jobs)


def stage_qaoa_tables(args):
    out = Path(args.out) / "qaoa_tables"
    cfg = base_cfg(args)
    names = args.datasets.split(",") if args.datasets else list(C.DATASETS)
    # The submitted COVID-19 QAOA rows used three restarts; the other tensors used one.
    cfg_covid = dict(cfg, qaoa_restarts=3)
    tasks = [dict(dataset=spec(n, args), method="QAOA", mu=mu, seed=s, rank_init=rmax(args),
                  cfg=cfg_covid if n == "covid" else cfg)
             for n in names for mu in mus(C.TABLE_MU.get(n, [0.1]), args) for s in seeds(args)]
    run_tasks(tasks, out, jobs=args.jobs)


def stage_profile(args):
    """Step 15: per-call cost breakdown (R4.5, R5.4). Every in-loop solver at the middle table penalty, R_max = 20,
    3 seeds, with QUBO construction time and the QAOA phase timings recorded per pruning call."""
    out = Path(args.out) / "profile"
    cfg = base_cfg(args)
    names = args.datasets.split(",") if args.datasets else list(C.DATASETS)
    sd = seeds(args)[:3]
    tasks = []
    for n in names:
        mu = mus(C.TABLE_MU.get(n, [0.1]), args)
        mu = mu[len(mu) // 2]
        for s in sd:
            for m in ("ES", "BF", "MS-BF", "SA", "Tabu", "QAOA"):
                tasks.append(dict(dataset=spec(n, args), method=m, mu=mu, seed=s, rank_init=rmax(args),
                                  cfg=dict(cfg, qaoa_restarts=3) if (m == "QAOA" and n == "covid") else cfg))
    run_tasks(tasks, out, jobs=args.jobs)


def stage_reproduce(args):
    """Exact reproduction of the submitted ES/BF runs (step 03b): the original per-mask ES loop, and on synthetic data
    the submitted schedule that let ES/BF prune at the final sweep. Not used for any revision result."""
    out = Path(args.out) / "reproduce"
    names = args.datasets.split(",") if args.datasets else ["synth_default", "amino", "kinetic"]
    tasks = []
    for n in names:
        cfg = base_cfg(args, es_impl="loop", final_prune=(n == "synth_default"))
        for mu in mus(C.TABLE_MU.get(n, [0.1]), args):
            for s in seeds(args):
                for m in ("ES", "BF"):
                    tasks.append(dict(dataset=spec(n, args), method=m, mu=mu, seed=s, rank_init=rmax(args), cfg=cfg))
    run_tasks(tasks, out, jobs=args.jobs)


def stage_sweeps(args):
    """Classical rows of the submitted noise and collinearity sweeps, re-run under one protocol (step 04b):
    tolerance 1e-12 as submitted, no pruning at the final sweep for any method."""
    out = Path(args.out) / "sweeps"
    cfg = base_cfg(args, tol=C.SWEEP_TOL)
    if args.quick:
        groups = [(synth_spec("noise_0.05", args, sigma=0.05), [2, 3]), (synth_spec("rho_0.5", args, sigma=0.01), [3])]
    else:
        groups = [(f"{k}=" + v.format(root=args.data_root, seed="{seed}"), [3, 6, 10]) for k, v in C.NOISE_SWEEP.items()]
        groups += [(f"{k}=" + v.format(root=args.data_root, seed="{seed}"), [6]) for k, v in C.RHO_SWEEP_OLD.items()]
    R0 = rmax(args)
    tasks = []
    for d, ks in groups:
        for s in seeds(args):
            tasks.append(dict(dataset=d, method="ALS", mu=None, seed=s, rank_init=R0, cfg=cfg))
            for K in ks:
                tasks.append(dict(dataset=d, method="TopK-PH", mu=None, seed=s, rank_init=R0, K=K, cfg=cfg))
            for m in ("ES", "BF"):
                tasks.append(dict(dataset=d, method=m, mu=0.1, seed=s, rank_init=R0, cfg=cfg))
    run_tasks(tasks, out, jobs=args.jobs)


STAGES = {
    "reproduce": stage_reproduce, "sweeps": stage_sweeps, "mu-rules": stage_mu_rules, "profile": stage_profile,
    "covid20": stage_covid20, "ablation": stage_ablation, "snr": stage_snr, "mu": stage_mu, "sigma": stage_sigma,
    "corcondia": stage_corcondia, "bank": stage_bank, "solvebank": stage_solvebank, "qaoa-noise": stage_qaoa_noise,
    "qaoa-tables": stage_qaoa_tables,
}
CLASSICAL_ORDER = ["covid20", "ablation", "snr", "mu", "sigma", "corcondia", "bank", "solvebank"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES) + ["all-classical"])
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="revision_out")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seeds", default=None, help="e.g. 0-19 or 0,1,2")
    ap.add_argument("--datasets", default=None, help="comma list of names in revision/config.py DATASETS")
    ap.add_argument("--data-seed", type=int, default=0, help="seed of the generated synthetic tensors")
    ap.add_argument("--t-max", type=int, default=None)
    ap.add_argument("--prune-period", type=int, default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--qaoa-shots", type=int, default=4096)
    ap.add_argument("--qaoa-maxiter", type=int, default=200)
    ap.add_argument("--with-qaoa", action="store_true", help="solvebank: also run QAOA on a subset")
    ap.add_argument("--qaoa-per-r", type=int, default=5, help="solvebank: QAOA instances per scaling size R")
    ap.add_argument("--qaoa-max-r", type=int, default=16, help="solvebank: largest scaling R given to QAOA")
    ap.add_argument("--qaoa-main", type=int, default=10, help="solvebank: QAOA on this many main-bank instances")
    ap.add_argument("--with-legacy", action="store_true",
                    help="qaoa-noise: also rerun the submitted QAOA behaviour (cancel on empty mask)")
    ap.add_argument("--all-noise", action="store_true", help="qaoa-noise: all five noise levels")
    ap.add_argument("--full-grid", action="store_true", help="mu: the full 32-value grid of the submission")
    ap.add_argument("--no-synth", action="store_true", help="mu/sigma: skip the generated SNR datasets")
    ap.add_argument("--quick", action="store_true", help="tiny smoke test (small tensors, 2 seeds)")
    args = ap.parse_args()
    if args.quick:
        _make_quick_data(args)
    write_env(Path(args.out))
    for st in (CLASSICAL_ORDER if args.stage == "all-classical" else [args.stage]):
        print(f"=== stage {st} ===", flush=True)
        STAGES[st](args)


def _make_quick_data(args):
    """Small stand-ins for the real tensors so every stage can be smoke-tested without the data."""
    import torch

    from revision.synth import make_synthetic

    root = Path(args.out) / "quick_data"
    root.mkdir(parents=True, exist_ok=True)
    shapes = {"amino": (5, 12, 9), "kinetic": (6, 5, 4, 7), "covid": (14, 6, 5)}
    for i, (name, shp) in enumerate(shapes.items()):
        p = root / f"{name}.pt"
        if not p.exists():
            o = make_synthetic(shape=shp, rank_true=3, rho=0.3, snr_db=15, seed=100 + i)
            X = o["X"].float()
            if name == "covid":
                X[0, 0, 0] = float("nan")
            torch.save({"X": X, "config": {"shape": list(shp)}}, p)
        QUICK_REAL[name] = f"{name}=pt:{p}"
    QUICK_REAL["synth_default"] = "synth_default=synth:sigma=0.05,rho=0.5,R=3,shape=8x8x8,seed={seed}"


if __name__ == "__main__":
    main()
