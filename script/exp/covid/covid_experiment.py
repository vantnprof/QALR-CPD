#!/usr/bin/env python3
"""
Run CP pruning experiments on the COVID-19 serology dataset.
Outputs per-sweep metrics and per-run summaries to CSV, mirroring acid_experiment.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "pandas is required to run the experiments. Please install pandas and retry."
    ) from exc

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - optional dependency
    linear_sum_assignment = None

from src.classical_alternative_leastsquare.classical_als import CPFactorization


def _parse_list_int(raw: str) -> List[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _parse_list_float(raw: str) -> List[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def _dataset_tag(shape: Sequence[int], rank_true: int, corr_rho: float, noise_std: float) -> str:
    dims = "x".join(str(s) for s in shape)
    return f"I{dims}_R{rank_true}_rho{corr_rho}_noise{noise_std}"


def _dtype_from_str(raw: str) -> torch.dtype:
    key = raw.strip().lower()
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{raw}'.")
    return mapping[key]


def _scale_mu(mu: float, mode: str, xnorm: float, rank_ref: int) -> float:
    mode_key = mode.strip().lower()
    if mode_key == "none":
        return float(mu)
    if mode_key == "xnorm":
        return float(mu) * float(xnorm)
    if mode_key == "xnorm2":
        return float(mu) * float(xnorm) ** 2
    if mode_key in ("xnorm2_over_rank", "xnorm2/rank", "xnorm2-over-rank"):
        denom = max(1, int(rank_ref))
        return float(mu) * (float(xnorm) ** 2) / float(denom)
    raise ValueError(f"Unknown mu scaling mode '{mode}'.")


def _sanitize_observations(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(X)
    if not finite.all():
        X = X.clone()
        X[~finite] = 0.0
    Omega = finite.to(dtype=X.dtype, device=X.device)
    return X, Omega


def _observed_norm(X: torch.Tensor, Omega: Optional[torch.Tensor]) -> torch.Tensor:
    if Omega is None:
        return torch.linalg.norm(X).clamp_min(1e-12)
    return torch.linalg.norm(Omega * X).clamp_min(1e-12)


def compute_component_recovery(
    gt_factors: Sequence[torch.Tensor],
    est_factors: Sequence[torch.Tensor],
) -> float:
    if len(gt_factors) == 0 or len(est_factors) == 0:
        return 0.0
    if len(gt_factors) != len(est_factors):
        raise ValueError("Factor order mismatch between ground truth and estimate.")

    gt = [A.detach().cpu() for A in gt_factors]
    est = [A.detach().cpu() for A in est_factors]
    R_true = gt[0].shape[1]
    R_hat = est[0].shape[1]
    if R_true == 0 or R_hat == 0:
        return 0.0

    sims = torch.ones((R_true, R_hat), dtype=gt[0].dtype)
    for n in range(len(gt)):
        gt_n = gt[n] / gt[n].norm(dim=0, keepdim=True).clamp_min(1e-12)
        est_n = est[n] / est[n].norm(dim=0, keepdim=True).clamp_min(1e-12)
        sims *= (gt_n.T @ est_n).abs()

    sims = sims.clamp_min(0.0) ** (1.0 / len(gt))
    sims_np = sims.numpy()

    if linear_sum_assignment is None:
        used_cols = set()
        total = 0.0
        for r in range(R_true):
            best = -1.0
            best_j = None
            for j in range(R_hat):
                if j in used_cols:
                    continue
                val = float(sims_np[r, j])
                if val > best:
                    best = val
                    best_j = j
            if best_j is not None:
                used_cols.add(best_j)
                total += best
        return total / float(R_true)

    row_ind, col_ind = linear_sum_assignment(-sims_np)
    matched_sum = float(sims_np[row_ind, col_ind].sum())
    return matched_sum / float(R_true)


def run_als(
    X: torch.Tensor,
    shape: Sequence[int],
    rank_init: int,
    Omega: Optional[torch.Tensor],
    ridge: float,
    n_sweeps: int,
    tol: float,
    seed: int,
    prune_method: str,
    prune_mu: Optional[float],
    prune_period: int,
    max_rank_prune: int,
    qaoa_reps: int,
    qaoa_shots_effective: int,
    qaoa_maxiter: int,
    qaoa_seed: int,
    qaoa_restarts: int,
    bitflip_budget: Optional[int],
    bitflip_restarts: int,
    bitflip_init_mode: str,
    bitflip_tie_break: str,
    bitflip_tol: float,
    bitflip_seed: Optional[int],
    bitflip_max_rank: Optional[int],
    log_every: int,
    verbose: bool,
    run_label: str,
) -> Tuple[List[Dict], Dict, List[torch.Tensor]]:
    torch.manual_seed(seed)
    model = CPFactorization(
        shape=shape,
        rank=rank_init,
        ridge=ridge,
        init="random",
        seed=seed,
        device=X.device,
        dtype=X.dtype,
    )

    rows: List[Dict] = []
    start = time.time()
    prunes_applied = 0
    prev_err: Optional[float] = None
    Xnorm = _observed_norm(X, Omega)

    method_key = prune_method.lower().strip() if prune_method is not None else "none"
    is_topk = method_key in ("topk", "top-k", "top_k")
    total_iters = int(n_sweeps)
    bitflip_seed_effective = seed if bitflip_seed is None else int(bitflip_seed)

    for sweep in range(total_iters):
        sweep_t0 = time.time()
        model.als_step(X, normalize=True)

        prune_applied = False
        if method_key != "none" and prune_mu is not None:
            last_sweep = sweep == total_iters - 1
            should_prune = last_sweep if is_topk else not last_sweep
            prune_frequency = 1 if is_topk else int(prune_period)

            if should_prune:
                before_rank = model.rank
                model._maybe_prune_components(
                    X=X,
                    iteration=sweep,
                    total_iters=total_iters,
                    prunning_enable=True,
                    prunning_mu=float(prune_mu),
                    prunning_frequency=prune_frequency,
                    prunning_require_nonempty=True,
                    prunning_max_rank_exhaustive=int(max_rank_prune),
                    verbose=verbose,
                    prunning_method=prune_method,
                    prunning_qaoa_reps=int(qaoa_reps),
                    prunning_qaoa_shots=int(qaoa_shots_effective),
                    prunning_qaoa_maxiter=int(qaoa_maxiter),
                    prunning_qaoa_seed=int(qaoa_seed),
                    prunning_qaoa_restarts=int(qaoa_restarts),
                    prunning_bitflip_budget=bitflip_budget,
                    prunning_bitflip_restarts=int(bitflip_restarts),
                    prunning_bitflip_init_mode=bitflip_init_mode,
                    prunning_bitflip_tie_break=bitflip_tie_break,
                    prunning_bitflip_tol=float(bitflip_tol),
                    prunning_bitflip_seed=int(bitflip_seed_effective),
                    prunning_bitflip_max_rank=bitflip_max_rank,
                )
                prune_applied = model.rank != before_rank
                if prune_applied:
                    prunes_applied += 1

        resid = X - model.forward()
        if Omega is not None:
            resid = Omega * resid
        err = (torch.linalg.norm(resid) / Xnorm).item()
        sweep_elapsed = time.time() - sweep_t0
        cumulative = time.time() - start

        if log_every <= 1 or (sweep + 1) % log_every == 0:
            rows.append(
                {
                    "sweep": sweep + 1,
                    "rel_error": err,
                    "rank": model.rank,
                    "prune_applied": prune_applied,
                    "sweep_time_s": sweep_elapsed,
                    "cumulative_s": cumulative,
                }
            )
            if verbose:
                print(
                    f"[sweep] {run_label} sweep={sweep + 1:03d} "
                    f"rel_error={err:.6e} rank={model.rank} prune={int(prune_applied)}"
                )

        if (not is_topk) and tol > 0 and prev_err is not None and abs(prev_err - err) < tol:
            break
        prev_err = err

    sweeps_done = sweep + 1
    summary = {
        "sweeps": sweeps_done,
        "final_rel_error": err,
        "final_rank": model.rank,
        "prunes_applied": prunes_applied,
        "total_time_s": cumulative,
    }
    factors_copy = [A.detach().cpu().clone() for A in model.factors]
    return rows, summary, factors_copy


def _log_run_done(run_label: str, summary: Dict) -> None:
    status = summary.get("status", "ok")
    if status != "ok":
        print(f"[done] {run_label} status=failed error={summary.get('error')}")
        return
    print(
        f"[done] {run_label} final_rel_error={summary.get('final_rel_error'):.6e} "
        f"final_rank={summary.get('final_rank')} sweeps={summary.get('sweeps')}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CP pruning experiments on COVID-19 serology data.")
    parser.add_argument(
        "--data-path",
        action="append",
        default=None,
        help="Path to COVID-19 .pt file (repeatable). If omitted, uses --data-glob.",
    )
    parser.add_argument(
        "--data-glob",
        type=str,
        default="data/covid19/covid19_serology.pt",
        help="Glob for COVID-19 .pt files.",
    )
    parser.add_argument("--results-csv", type=str, default="results/covid_experiment.csv")
    parser.add_argument("--summary-csv", type=str, default="results/covid_experiment_summary.csv")
    parser.add_argument("--ranks", type=str, default="3,4,5,6,8,10,16,20")
    parser.add_argument("--topk-values", type=str, default="3,4,5,6,7,8,9,10")
    parser.add_argument("--t-max", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--prune-period", type=int, default=40)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument(
        "--mu-values",
        type=str,
        default=(
            "0.0001,0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,"
            "0.15,0.2,0.25,0.30,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.70,0.75,0.8,"
            "0.85,0.9,0.95,1.0,1.5,2.0"
        ),
    )
    parser.add_argument(
        "--mu-scale",
        type=str,
        default="xnorm2_over_rank",
        choices=("none", "xnorm", "xnorm2", "xnorm2_over_rank"),
        help=(
            "Scale mu to match data magnitude. "
            "'xnorm2_over_rank' treats mu as a per-component penalty relative to ||X||_F^2."
        ),
    )
    parser.add_argument("--qaoa-reps", type=int, default=2)
    parser.add_argument("--qaoa-shots", type=str, default="4096")
    parser.add_argument("--qaoa-shots-fallback", type=int, default=4096)
    parser.add_argument("--qaoa-maxiter", type=int, default=200)
    parser.add_argument("--qaoa-restarts", type=int, default=1)
    parser.add_argument("--qaoa-rank-init", type=int, default=20)
    parser.add_argument(
        "--bitflip-budget",
        type=int,
        default=None,
        help="Bit-flip iteration budget. Defaults to --qaoa-maxiter when unset.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=20,
        help="Number of random seeds (use 20 for the paper table).",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_paths: List[Path] = []
    if args.data_path:
        data_paths = [Path(p) for p in args.data_path]
    else:
        data_paths = list(Path(".").glob(args.data_glob))

    if not data_paths:
        raise SystemExit("No data files found. Use --data-path or --data-glob.")

    ranks = _parse_list_int(args.ranks)
    topk_values = _parse_list_int(args.topk_values)
    mu_values = _parse_list_float(args.mu_values)
    qaoa_shots_list = _parse_list_int(args.qaoa_shots)
    dtype = _dtype_from_str(args.dtype)
    bitflip_budget = args.bitflip_budget if args.bitflip_budget is not None else args.qaoa_maxiter
    if args.seeds:
        run_seeds = _parse_list_int(args.seeds)
    else:
        run_seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    Path(args.results_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    summaries: List[Dict] = []

    for data_path in data_paths:
        if not data_path.exists():
            print(f"[warn] dataset file missing: {data_path}")
            continue

        payload = torch.load(data_path, map_location="cpu")
        X = payload["X"].to(device=args.device, dtype=dtype)
        X, Omega = _sanitize_observations(X)
        missing = int((Omega == 0).sum().item())
        if missing > 0:
            total = int(Omega.numel())
            print(f"[info] found {missing}/{total} non-finite entries; treating as missing.")

        gt_factors = payload.get("factors", None)
        if gt_factors is not None:
            gt_factors = [A.to(dtype=dtype) for A in gt_factors]

        cfg = payload.get("config", {})
        shape = tuple(cfg.get("shape", tuple(X.shape)))
        rank_true = int(cfg.get("rank_true", 0))
        corr_rho = float(cfg.get("corr_rho", 0.0))
        noise_std = float(cfg.get("noise_std", 0.0))
        dataset_tag = _dataset_tag(shape, rank_true, corr_rho, noise_std)
        data_seed = int(payload.get("seed", 0))
        xnorm = float(_observed_norm(X, Omega).item())

        print(f"[info] dataset={data_path} seed={data_seed} tag={dataset_tag}")
        if args.mu_scale != "none":
            scale_note = {
                "xnorm": f"||X||_F={xnorm:.3e}",
                "xnorm2": f"||X||_F^2={xnorm ** 2:.3e}",
                "xnorm2_over_rank": (
                    f"||X||_F^2/R={xnorm ** 2 / max(1, int(args.qaoa_rank_init)):.3e}"
                ),
            }.get(args.mu_scale, "")
            print(f"[info] mu_scale={args.mu_scale} ({scale_note})")

        qaoa_rank_init = int(args.qaoa_rank_init)
        for run_seed in run_seeds:
            print(f"[info] starting run seed={run_seed}")
            for rank_init in ranks:
                # No-pruning baseline.
                for method, prune_method, mu, qaoa_shots_requested in [
                    ("ALS", "none", None, None),
                ]:
                    run_label = f"method={method} rank_init={rank_init} seed={run_seed} mu={mu}"
                    print(
                        f"[run] method={method} rank_init={rank_init} seed={run_seed} mu={mu} "
                        f"qaoa_shots={qaoa_shots_requested}"
                    )
                    run_id = {
                        "dataset_tag": dataset_tag,
                        "dataset_path": str(data_path),
                        "data_seed": data_seed,
                        "seed": run_seed,
                        "rank_true": rank_true,
                        "rank_init": rank_init,
                        "top_k": None,
                        "method": method,
                        "prune_method": prune_method,
                        "prune_mu": mu,
                        "prune_mu_effective": None,
                        "mu_scale": args.mu_scale,
                        "prune_period": args.prune_period,
                        "ridge": args.ridge,
                        "qaoa_reps": args.qaoa_reps,
                        "qaoa_shots_requested": qaoa_shots_requested,
                        "qaoa_shots_effective": qaoa_shots_requested,
                        "qaoa_maxiter": args.qaoa_maxiter,
                        "qaoa_restarts": args.qaoa_restarts,
                        "qaoa_shots_note": None,
                    }
                    try:
                        rows, summary, est_factors = run_als(
                            X=X,
                            shape=shape,
                            rank_init=rank_init,
                            Omega=Omega,
                            ridge=args.ridge,
                            n_sweeps=args.t_max,
                            tol=args.tol,
                            seed=run_seed,
                            prune_method=prune_method,
                            prune_mu=mu,
                            prune_period=args.prune_period,
                            max_rank_prune=rank_init,
                            qaoa_reps=args.qaoa_reps,
                            qaoa_shots_effective=1,
                            qaoa_maxiter=args.qaoa_maxiter,
                            qaoa_seed=run_seed,
                            qaoa_restarts=args.qaoa_restarts,
                            bitflip_budget=bitflip_budget,
                            bitflip_restarts=1,
                            bitflip_init_mode="zeros",
                            bitflip_tie_break="first",
                            bitflip_tol=1e-12,
                            bitflip_seed=run_seed,
                            bitflip_max_rank=None,
                            log_every=args.log_every,
                            verbose=args.verbose,
                            run_label=run_label,
                        )
                        rec = (
                            compute_component_recovery(gt_factors, est_factors)
                            if gt_factors is not None
                            else None
                        )
                        summary.update({"final_rec": rec, "status": "ok", "error": ""})
                    except Exception as exc:
                        summary = {
                            "sweeps": 0,
                            "final_rel_error": None,
                            "final_rank": None,
                            "prunes_applied": None,
                            "total_time_s": None,
                            "final_rec": None,
                            "status": "failed",
                            "error": str(exc),
                        }
                        rows = []
                    _log_run_done(run_label, summary)
                    for r in rows:
                        all_rows.append({**run_id, **r})
                    summaries.append({**run_id, **summary})

                # Top-K pruning baseline.
                for top_k in topk_values:
                    if top_k <= 0 or top_k > rank_init:
                        continue
                    method = "ALS-TOPK"
                    prune_method = "top-k"
                    mu = 0.0
                    qaoa_shots_requested = None
                    run_label = f"method={method} rank_init={rank_init} seed={run_seed} top_k={top_k}"
                    print(
                        f"[run] method={method} rank_init={rank_init} seed={run_seed} top_k={top_k} "
                        f"qaoa_shots={qaoa_shots_requested}"
                    )
                    run_id = {
                        "dataset_tag": dataset_tag,
                        "dataset_path": str(data_path),
                        "data_seed": data_seed,
                        "seed": run_seed,
                        "rank_true": rank_true,
                        "rank_init": rank_init,
                        "top_k": top_k,
                        "method": method,
                        "prune_method": prune_method,
                        "prune_mu": mu,
                        "prune_mu_effective": None,
                        "mu_scale": args.mu_scale,
                        "prune_period": args.prune_period,
                        "ridge": args.ridge,
                        "qaoa_reps": args.qaoa_reps,
                        "qaoa_shots_requested": qaoa_shots_requested,
                        "qaoa_shots_effective": qaoa_shots_requested,
                        "qaoa_maxiter": args.qaoa_maxiter,
                        "qaoa_restarts": args.qaoa_restarts,
                        "qaoa_shots_note": None,
                    }
                    try:
                        rows, summary, est_factors = run_als(
                            X=X,
                            shape=shape,
                            rank_init=rank_init,
                            Omega=Omega,
                            ridge=args.ridge,
                            n_sweeps=args.t_max,
                            tol=args.tol,
                            seed=run_seed,
                            prune_method=prune_method,
                            prune_mu=mu,
                            prune_period=args.prune_period,
                            max_rank_prune=top_k,
                            qaoa_reps=args.qaoa_reps,
                            qaoa_shots_effective=1,
                            qaoa_maxiter=args.qaoa_maxiter,
                            qaoa_seed=run_seed,
                            qaoa_restarts=args.qaoa_restarts,
                            bitflip_budget=bitflip_budget,
                            bitflip_restarts=1,
                            bitflip_init_mode="zeros",
                            bitflip_tie_break="first",
                            bitflip_tol=1e-12,
                            bitflip_seed=run_seed,
                            bitflip_max_rank=None,
                            log_every=args.log_every,
                            verbose=args.verbose,
                            run_label=run_label,
                        )
                        rec = (
                            compute_component_recovery(gt_factors, est_factors)
                            if gt_factors is not None
                            else None
                        )
                        summary.update({"final_rec": rec, "status": "ok", "error": ""})
                    except Exception as exc:
                        summary = {
                            "sweeps": 0,
                            "final_rel_error": None,
                            "final_rank": None,
                            "prunes_applied": None,
                            "total_time_s": None,
                            "final_rec": None,
                            "status": "failed",
                            "error": str(exc),
                        }
                        rows = []
                    _log_run_done(run_label, summary)
                    for r in rows:
                        all_rows.append({**run_id, **r})
                    summaries.append({**run_id, **summary})

            for mu in mu_values:
                mu_effective = _scale_mu(mu, args.mu_scale, xnorm=xnorm, rank_ref=qaoa_rank_init)
                for method, prune_method in [
                    ("ALS-LR (ES)", "exhaustive"),
                    ("ALS-LR (BF)", "bitflip"),
                ]:
                    qaoa_shots_requested = None
                    run_label = f"method={method} rank_init={qaoa_rank_init} seed={run_seed} mu={mu}"
                    print(
                        f"[run] method={method} rank_init={qaoa_rank_init} seed={run_seed} mu={mu} "
                        f"qaoa_shots={qaoa_shots_requested}"
                    )
                    run_id = {
                        "dataset_tag": dataset_tag,
                        "dataset_path": str(data_path),
                        "data_seed": data_seed,
                        "seed": run_seed,
                        "rank_true": rank_true,
                        "rank_init": qaoa_rank_init,
                        "top_k": None,
                        "method": method,
                        "prune_method": prune_method,
                        "prune_mu": mu,
                        "prune_mu_effective": mu_effective,
                        "mu_scale": args.mu_scale,
                        "prune_period": args.prune_period,
                        "ridge": args.ridge,
                        "qaoa_reps": args.qaoa_reps,
                        "qaoa_shots_requested": qaoa_shots_requested,
                        "qaoa_shots_effective": qaoa_shots_requested,
                        "qaoa_maxiter": args.qaoa_maxiter,
                        "qaoa_restarts": args.qaoa_restarts,
                        "qaoa_shots_note": None,
                    }
                    try:
                        rows, summary, est_factors = run_als(
                            X=X,
                            shape=shape,
                            rank_init=qaoa_rank_init,
                            Omega=Omega,
                            ridge=args.ridge,
                            n_sweeps=args.t_max,
                            tol=args.tol,
                            seed=run_seed,
                            prune_method=prune_method,
                            prune_mu=mu_effective,
                            prune_period=args.prune_period,
                            max_rank_prune=qaoa_rank_init,
                            qaoa_reps=args.qaoa_reps,
                            qaoa_shots_effective=1,
                            qaoa_maxiter=args.qaoa_maxiter,
                            qaoa_seed=run_seed,
                            qaoa_restarts=args.qaoa_restarts,
                            bitflip_budget=bitflip_budget,
                            bitflip_restarts=1,
                            bitflip_init_mode="zeros",
                            bitflip_tie_break="first",
                            bitflip_tol=1e-12,
                            bitflip_seed=run_seed,
                            bitflip_max_rank=None,
                            log_every=args.log_every,
                            verbose=args.verbose,
                            run_label=run_label,
                        )
                        rec = (
                            compute_component_recovery(gt_factors, est_factors)
                            if gt_factors is not None
                            else None
                        )
                        summary.update({"final_rec": rec, "status": "ok", "error": ""})
                    except Exception as exc:
                        summary = {
                            "sweeps": 0,
                            "final_rel_error": None,
                            "final_rank": None,
                            "prunes_applied": None,
                            "total_time_s": None,
                            "final_rec": None,
                            "status": "failed",
                            "error": str(exc),
                        }
                        rows = []
                    _log_run_done(run_label, summary)
                    for r in rows:
                        all_rows.append({**run_id, **r})
                    summaries.append({**run_id, **summary})

            # QAOA pruning baselines (fixed init rank).
            for mu in mu_values:
                mu_effective = _scale_mu(mu, args.mu_scale, xnorm=xnorm, rank_ref=qaoa_rank_init)
                for qaoa_shots_requested in qaoa_shots_list:
                    qaoa_shots_effective = (
                        args.qaoa_shots_fallback
                        if qaoa_shots_requested <= 0
                        else qaoa_shots_requested
                    )
                    shots_note = (
                        f"fallback_to_{args.qaoa_shots_fallback}"
                        if qaoa_shots_requested <= 0
                        else "as_requested"
                    )
                    if qaoa_shots_requested <= 0:
                        print(
                            f"[warn] qaoa_shots={qaoa_shots_requested} not supported; "
                            f"using {qaoa_shots_effective} instead."
                        )
                    run_label = f"method=QALR-CPD rank_init={qaoa_rank_init} seed={run_seed} mu={mu}"
                    print(
                        f"[run] method=QALR-CPD rank_init={qaoa_rank_init} seed={run_seed} mu={mu} "
                        f"qaoa_shots={qaoa_shots_requested}"
                    )
                    run_id = {
                        "dataset_tag": dataset_tag,
                        "dataset_path": str(data_path),
                        "data_seed": data_seed,
                        "seed": run_seed,
                        "rank_true": rank_true,
                        "rank_init": qaoa_rank_init,
                        "top_k": None,
                        "method": "QALR-CPD",
                        "prune_method": "qaoa",
                        "prune_mu": mu,
                        "prune_mu_effective": mu_effective,
                        "mu_scale": args.mu_scale,
                        "prune_period": args.prune_period,
                        "ridge": args.ridge,
                        "qaoa_reps": args.qaoa_reps,
                        "qaoa_shots_requested": qaoa_shots_requested,
                        "qaoa_shots_effective": qaoa_shots_effective,
                        "qaoa_maxiter": args.qaoa_maxiter,
                        "qaoa_restarts": args.qaoa_restarts,
                        "qaoa_shots_note": shots_note,
                    }
                    try:
                        rows, summary, est_factors = run_als(
                            X=X,
                            shape=shape,
                            rank_init=qaoa_rank_init,
                            Omega=Omega,
                            ridge=args.ridge,
                            n_sweeps=args.t_max,
                            tol=args.tol,
                            seed=run_seed,
                            prune_method="qaoa",
                            prune_mu=mu_effective,
                            prune_period=args.prune_period,
                            max_rank_prune=qaoa_rank_init,
                            qaoa_reps=args.qaoa_reps,
                            qaoa_shots_effective=qaoa_shots_effective,
                            qaoa_maxiter=args.qaoa_maxiter,
                            qaoa_seed=run_seed,
                            qaoa_restarts=args.qaoa_restarts,
                            bitflip_budget=bitflip_budget,
                            bitflip_restarts=1,
                            bitflip_init_mode="zeros",
                            bitflip_tie_break="first",
                            bitflip_tol=1e-12,
                            bitflip_seed=run_seed,
                            bitflip_max_rank=None,
                            log_every=args.log_every,
                            verbose=args.verbose,
                            run_label=run_label,
                        )
                        rec = (
                            compute_component_recovery(gt_factors, est_factors)
                            if gt_factors is not None
                            else None
                        )
                        summary.update({"final_rec": rec, "status": "ok", "error": ""})
                    except Exception as exc:
                        summary = {
                            "sweeps": 0,
                            "final_rel_error": None,
                            "final_rank": None,
                            "prunes_applied": None,
                            "total_time_s": None,
                            "final_rec": None,
                            "status": "failed",
                            "error": str(exc),
                        }
                        rows = []
                    _log_run_done(run_label, summary)
                    for r in rows:
                        all_rows.append({**run_id, **r})
                    summaries.append({**run_id, **summary})

    if all_rows:
        pd.DataFrame(all_rows).to_csv(args.results_csv, index=False)
        print(f"[info] wrote sweep metrics to {args.results_csv}")
    if summaries:
        pd.DataFrame(summaries).to_csv(args.summary_csv, index=False)
        print(f"[info] wrote summary metrics to {args.summary_csv}")


if __name__ == "__main__":
    main()
