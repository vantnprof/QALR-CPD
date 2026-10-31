#!/usr/bin/env python3
"""
Run an ablation study over QAOA shots and circuit depth (reps) for QALR-CPD
on the amino acid fluorescence dataset.

This script sweeps shot counts and QAOA depth while keeping the pruning
penalty (mu) fixed. Outputs per-sweep metrics and per-run summaries to CSV.
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

from src.cp.als import CPFactorization


def _parse_list_int(raw: str) -> List[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


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
    Xnorm = torch.linalg.norm(X).clamp_min(1e-12)

    method_key = prune_method.lower().strip() if prune_method is not None else "none"
    is_topk = method_key in ("topk", "top-k", "top_k")
    total_iters = int(n_sweeps)

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
                )
                prune_applied = model.rank != before_rank
                if prune_applied:
                    prunes_applied += 1

        err = (torch.linalg.norm(X - model.forward()) / Xnorm).item()
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
    parser = argparse.ArgumentParser(
        description="Ablation study for QAOA shots and depth on QALR-CPD (amino acid dataset)."
    )
    parser.add_argument(
        "--data-path",
        action="append",
        default=None,
        help="Path to amino .pt file (repeatable). If omitted, uses --data-glob.",
    )
    parser.add_argument(
        "--data-glob",
        type=str,
        default="data/amino/*.pt",
        help="Glob for amino .pt files.",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default="results/v1/ablation_acid_shot_circuit_depth.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="results/v1/ablation_acid_shot_circuit_depth_summary.csv",
    )
    parser.add_argument("--rank-init", type=int, default=16)
    parser.add_argument("--mu", type=float, default=0.8)
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
    parser.add_argument("--t-max", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--prune-period", type=int, default=40)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument(
        "--qaoa-reps-list",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated list of QAOA depth (reps).",
    )
    parser.add_argument(
        "--qaoa-shots-list",
        type=str,
        default="128,256,512,1024,2048,4096,8912,16384",
        help="Comma-separated list of QAOA shot counts.",
    )
    parser.add_argument("--qaoa-shots-fallback", type=int, default=2048)
    parser.add_argument("--qaoa-maxiter", type=int, default=200)
    parser.add_argument("--qaoa-restarts", type=int, default=3)
    parser.add_argument(
        "--max-rank-prune",
        type=int,
        default=None,
        help="Max rank budget for pruning solvers (defaults to --rank-init).",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=20,
        help="Number of random seeds (use 20 for paper-style averages).",
    )
    parser.add_argument("--seed-start", type=int, default=100)
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

    qaoa_reps_list = _parse_list_int(args.qaoa_reps_list)
    qaoa_shots_list = _parse_list_int(args.qaoa_shots_list)
    if not qaoa_reps_list:
        raise SystemExit("Empty --qaoa-reps-list supplied.")
    if not qaoa_shots_list:
        raise SystemExit("Empty --qaoa-shots-list supplied.")

    dtype = _dtype_from_str(args.dtype)
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

        print(f"[info] dataset={data_path} seed={data_seed} tag={dataset_tag}")

        rank_init = int(args.rank_init)
        max_rank_prune = int(args.max_rank_prune or rank_init)
        xnorm = float(torch.linalg.norm(X).clamp_min(1e-12).item())
        mu_effective = _scale_mu(args.mu, args.mu_scale, xnorm=xnorm, rank_ref=rank_init)
        if args.mu_scale != "none":
            scale_note = {
                "xnorm": f"||X||_F={xnorm:.3e}",
                "xnorm2": f"||X||_F^2={xnorm ** 2:.3e}",
                "xnorm2_over_rank": f"||X||_F^2/R={xnorm ** 2 / max(1, rank_init):.3e}",
            }.get(args.mu_scale, "")
            print(f"[info] mu_scale={args.mu_scale} ({scale_note})")

        for run_seed in run_seeds:
            print(f"[info] starting run seed={run_seed}")
            for qaoa_reps in qaoa_reps_list:
                for qaoa_shots_requested in qaoa_shots_list:
                    qaoa_shots_effective = (
                        int(args.qaoa_shots_fallback)
                        if qaoa_shots_requested <= 0
                        else int(qaoa_shots_requested)
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

                    run_label = (
                        f"method=QALR-CPD seed={run_seed} "
                        f"mu={args.mu} depth={qaoa_reps} shots={qaoa_shots_requested}"
                    )
                    print(
                        f"[run] method=QALR-CPD seed={run_seed} mu={args.mu} "
                        f"mu_eff={mu_effective:.3e} depth={qaoa_reps} shots={qaoa_shots_requested}"
                    )
                    run_id = {
                        "dataset_tag": dataset_tag,
                        "dataset_path": str(data_path),
                        "data_seed": data_seed,
                        "seed": run_seed,
                        "rank_true": rank_true,
                        "rank_init": rank_init,
                        "method": "QALR-CPD",
                        "solver": "qaoa",
                        "prune_method": "qaoa",
                        "prune_mu": args.mu,
                        "prune_mu_effective": mu_effective,
                        "mu_scale": args.mu_scale,
                        "prune_period": args.prune_period,
                        "ridge": args.ridge,
                        "max_rank_prune": max_rank_prune,
                        "qaoa_reps": int(qaoa_reps),
                        "qaoa_shots_requested": int(qaoa_shots_requested),
                        "qaoa_shots_effective": int(qaoa_shots_effective),
                        "qaoa_maxiter": args.qaoa_maxiter,
                        "qaoa_restarts": args.qaoa_restarts,
                        "qaoa_shots_note": shots_note,
                    }
                    try:
                        rows, summary, est_factors = run_als(
                            X=X,
                            shape=shape,
                            rank_init=rank_init,
                            ridge=args.ridge,
                            n_sweeps=args.t_max,
                            tol=args.tol,
                            seed=run_seed,
                            prune_method="qaoa",
                            prune_mu=mu_effective,
                            prune_period=args.prune_period,
                            max_rank_prune=max_rank_prune,
                            qaoa_reps=int(qaoa_reps),
                            qaoa_shots_effective=qaoa_shots_effective,
                            qaoa_maxiter=args.qaoa_maxiter,
                            qaoa_seed=run_seed,
                            qaoa_restarts=args.qaoa_restarts,
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

        ok_rows = [s for s in summaries if s.get("status") == "ok"]
        if ok_rows:
            summary_df = pd.DataFrame(ok_rows)
            group_cols = ["dataset_tag", "qaoa_reps", "qaoa_shots_effective"]
            agg = (
                summary_df.groupby(group_cols, dropna=False)
                .agg(
                    final_rel_error_mean=("final_rel_error", "mean"),
                    final_rel_error_std=("final_rel_error", "std"),
                    final_rank_mean=("final_rank", "mean"),
                    final_rank_std=("final_rank", "std"),
                )
                .reset_index()
            )
            for col in ("final_rel_error_std", "final_rank_std"):
                if col in agg.columns:
                    agg[col] = agg[col].fillna(0.0)
            print("[info] mean+/-std summary by depth+shots:")
            for _, row in agg.iterrows():
                print(
                    "  "
                    f"depth={row['qaoa_reps']} shots={row['qaoa_shots_effective']} "
                    f"rel_error={row['final_rel_error_mean']:.6e}+/-{row['final_rel_error_std']:.6e} "
                    f"rank={row['final_rank_mean']:.2f}+/-{row['final_rank_std']:.2f}"
                )


if __name__ == "__main__":
    main()
