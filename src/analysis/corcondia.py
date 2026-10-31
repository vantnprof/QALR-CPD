"""Core consistency diagnostic (Bro & Kiers, 2003) around the rank selected by the penalty plateau (C1).

CORCONDIA = 100 * (1 - ||G - T||_F^2 / R), where G is the least-squares Tucker core of X given the CP loadings
and T is the superdiagonal identity core. For tensors with missing entries (COVID-19) the zero-filled tensor is
used, as in the ALS updates of the submission; this is an approximation and is reported as such.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch


def corcondia(X: np.ndarray, factors, weights) -> float:
    A = [np.asarray(F, dtype=np.float64).copy() for F in factors]
    A[0] = A[0] * np.asarray(weights, dtype=np.float64)[None, :]
    G = np.asarray(X, dtype=np.float64)
    for n, F in enumerate(A):
        P = np.linalg.pinv(F)                       # (R, I_n)
        G = np.moveaxis(np.tensordot(P, G, axes=([1], [n])), 0, n)
    R = A[0].shape[1]
    T = np.zeros_like(G)
    idx = np.arange(R)
    T[tuple([idx] * G.ndim)] = 1.0
    return float(100.0 * (1.0 - np.sum((G - T) ** 2) / R))


def corcondia_for_runs(out_root: Path, args, spec, synth_designs, seeds, base_cfg, rmax):
    from src.experiments import config as C
    from src.analysis.plateau import rank_profile, widest_plateau
    from src.data.loaders import load_dataset, resolve_spec
    from src.experiments.stages import read_runs
    from src.cp.als import CPFactorization

    runs = read_runs(Path(out_root) / "mu" / "runs.csv")
    if not runs:
        raise SystemExit("run the 'mu' stage first")
    names = args.datasets.split(",") if args.datasets else list(C.DATASETS)
    datasets = [spec(n, args) for n in names] + (synth_designs(args)[:6] if not args.no_synth else [])
    R0 = rmax(args)
    rows = []
    for d in datasets:
        name = d.split("=", 1)[0]
        prof = rank_profile(runs, name, "ES", R0)
        plat = widest_plateau(prof, exclude_full=R0)
        if not plat:
            continue
        cfg = base_cfg(args)
        for R in sorted({max(1, plat["rank"] - 1), plat["rank"], plat["rank"] + 1}):
            vals = []
            for s in seeds(args)[:5]:
                ds = load_dataset(resolve_spec(d, s), "float64")   # seed s -> tensor s for the synthetic sets
                Xn = ds.X.cpu().numpy()
                torch.manual_seed(s)
                m = CPFactorization(shape=ds.shape, rank=R, ridge=1e-6, init="random", seed=s, dtype=ds.X.dtype)
                for _ in range(int(cfg["t_max"] or 200)):
                    m.als_step(ds.X, normalize=True)
                vals.append(corcondia(Xn, [A.detach().cpu().numpy() for A in m.factors], m.weights.cpu().numpy()))
            rows.append({"dataset": name, "plateau_rank": plat["rank"], "mu_lo": plat["mu_lo"], "mu_hi": plat["mu_hi"],
                         "R": R, "corcondia_median": float(np.median(vals)), "corcondia_min": float(np.min(vals)),
                         "missing_entries": int(ds.Omega is not None)})
            print(f"[corcondia] {name} R={R}: median {np.median(vals):.1f}", flush=True)
    out = Path(out_root) / "corcondia"
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out / "corcondia.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
