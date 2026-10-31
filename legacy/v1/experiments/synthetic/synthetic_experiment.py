#!/usr/bin/env python3
"""
Run synthetic CP experiments over pre-generated datasets and save results to CSV.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
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


def _resolve_dataset_path(path_str: str, manifest_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = manifest_dir / path
    return candidate


@dataclass
class DatasetRecord:
    path: Path
    seed: int
    shape: Tuple[int, ...]
    rank_true: int
    noise_std: float
    corr_rho: float
    dtype: str


def load_manifest(path: Path) -> List[DatasetRecord]:
    data = json.loads(path.read_text())
    records: List[DatasetRecord] = []
    for entry in data:
        resolved = _resolve_dataset_path(entry["path"], path.parent)
        records.append(
            DatasetRecord(
                path=resolved,
                seed=int(entry.get("seed", 0)),
                shape=tuple(entry["shape"]),
                rank_true=int(entry["rank_true"]),
                noise_std=float(entry["noise_std"]),
                corr_rho=float(entry["corr_rho"]),
                dtype=str(entry.get("dtype", "float32")),
            )
        )
    return records


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
        # Greedy fallback if SciPy is not available.
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
    bitflip_budget: Optional[int] = 200,
    bitflip_restarts: int = 1,
    bitflip_init_mode: str = "zeros",
    bitflip_tie_break: str = "first",
    bitflip_tol: float = 1e-12,
    bitflip_seed: Optional[int] = None,
    bitflip_max_rank: Optional[int] = None,
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
    is_qaoa = method_key == "qaoa"
    total_iters = int(n_sweeps)
    bitflip_budget_use = None if bitflip_budget is None else int(bitflip_budget)
    bitflip_restarts_use = int(bitflip_restarts)
    bitflip_seed_use = seed if bitflip_seed is None else int(bitflip_seed)
    bitflip_tol_use = float(bitflip_tol)
    bitflip_max_rank_use = (
        int(max_rank_prune) if bitflip_max_rank is None else int(bitflip_max_rank)
    )

    for sweep in range(total_iters):
        sweep_t0 = time.time()
        model.als_step(X, normalize=True)

        prune_applied = False
        if method_key != "none" and prune_mu is not None:
            should_prune = False
            prune_frequency = int(prune_period)
            if is_topk:
                should_prune = sweep == int(n_sweeps) - 1
                prune_frequency = 1
            elif prune_period > 0:
                should_prune = True
            if is_qaoa and sweep == int(n_sweeps) - 1:
                should_prune = False

            if should_prune:
                before_rank = model.rank
                try:
                    model._maybe_prune_components(
                        X=X,
                        iteration=sweep,
                        total_iters=total_iters,
                        prunning_enable=True,
                        prunning_mu=float(prune_mu),
                        prunning_frequency=prune_frequency,
                        prunning_require_nonempty=True,
                        prunning_max_rank_exhaustive=int(max_rank_prune),
                        verbose=False,
                        prunning_method=prune_method,
                        prunning_qaoa_reps=int(qaoa_reps),
                        prunning_qaoa_shots=int(qaoa_shots_effective),
                        prunning_qaoa_maxiter=int(qaoa_maxiter),
                        prunning_qaoa_seed=int(qaoa_seed),
                        prunning_qaoa_restarts=int(qaoa_restarts),
                        prunning_bitflip_budget=bitflip_budget_use,
                        prunning_bitflip_restarts=bitflip_restarts_use,
                        prunning_bitflip_init_mode=bitflip_init_mode,
                        prunning_bitflip_tie_break=bitflip_tie_break,
                        prunning_bitflip_tol=bitflip_tol_use,
                        prunning_bitflip_seed=bitflip_seed_use,
                        prunning_bitflip_max_rank=bitflip_max_rank_use,
                    )
                except Exception:
                    raise
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

    summary = {
        "sweeps": sweep + 1,
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
    parser = argparse.ArgumentParser(description="Run synthetic CP pruning experiments.")
    parser.add_argument(
        "--manifest",
        action="append",
        default=None,
        help="Path to a manifest JSON (repeatable). If omitted, uses --manifest-glob.",
    )
    parser.add_argument(
        "--manifest-glob",
        type=str,
        default="data/synthetic/manifest_*.json",
        help="Glob for manifest JSON files.",
    )
    parser.add_argument("--results-csv", type=str, default="results/v1/synthetic_experiment.csv")
    parser.add_argument("--summary-csv", type=str, default="results/v1/synthetic_experiment_summary.csv")
    parser.add_argument("--ranks", type=str, default="6,10,14,20")
    parser.add_argument(
        "--als-ranks",
        type=str,
        default=None,
        help="Comma-separated ALS baseline ranks. Defaults to --ranks when omitted.",
    )
    parser.add_argument("--t-max", type=int, default=200)
    parser.add_argument("--tol", type=float, default=0.0)
    parser.add_argument("--prune-period", type=int, default=40)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--mu-values", type=str, default="0.0001,0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.15,0.2,0.25,0.30,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.70,0.75,0.8,0.85,0.9,0.95,1.0,1.5,2.0")
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
    parser.add_argument(
        "--bitflip-budget",
        type=int,
        default=200,
        help="Iteration budget for bit-flipping pruning.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--topk-values",
        type=str,
        default="3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests: List[Path] = []
    if args.manifest:
        manifests = [Path(p) for p in args.manifest]
    else:
        manifests = list(Path(".").glob(args.manifest_glob))

    if not manifests:
        raise SystemExit("No manifest files found. Use --manifest or --manifest-glob.")

    prune_ranks = _parse_list_int(args.ranks)
    base_als_ranks = _parse_list_int(args.als_ranks) if args.als_ranks else None
    mu_values = _parse_list_float(args.mu_values)
    topk_values = _parse_list_int(args.topk_values)
    qaoa_shots_list = _parse_list_int(args.qaoa_shots)
    dtype = _dtype_from_str(args.dtype)

    Path(args.results_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    summaries: List[Dict] = []

    for manifest_path in manifests:
        records = load_manifest(manifest_path)
        for record in records:
            if not record.path.exists():
                print(f"[warn] dataset file missing: {record.path}")
                continue

            payload = torch.load(record.path, map_location="cpu")
            X = payload["X"].to(device=args.device, dtype=dtype)
            gt_factors = payload.get("factors", None)
            if gt_factors is not None:
                gt_factors = [A.to(dtype=dtype) for A in gt_factors]
            xnorm = float(torch.linalg.norm(X).clamp_min(1e-12).item())
            dataset_tag = payload.get(
                "config",
                {},
            )
            if dataset_tag:
                dataset_tag = _dataset_tag(
                    shape=tuple(dataset_tag.get("shape", record.shape)),
                    rank_true=int(dataset_tag.get("rank_true", record.rank_true)),
                    corr_rho=float(dataset_tag.get("corr_rho", record.corr_rho)),
                    noise_std=float(dataset_tag.get("noise_std", record.noise_std)),
                )
            else:
                dataset_tag = _dataset_tag(record.shape, record.rank_true, record.corr_rho, record.noise_std)

            print(f"[info] dataset={record.path} seed={record.seed} tag={dataset_tag}")
            if args.mu_scale != "none":
                scale_note = {
                    "xnorm": f"||X||_F={xnorm:.3e}",
                    "xnorm2": f"||X||_F^2={xnorm ** 2:.3e}",
                    "xnorm2_over_rank": "||X||_F^2/R (R=rank_init)",
                }.get(args.mu_scale, "")
                print(f"[info] mu_scale={args.mu_scale} ({scale_note})")

            als_ranks = (
                base_als_ranks if base_als_ranks is not None else sorted(set(prune_ranks) | {record.rank_true})
            )
            all_ranks = sorted(set(prune_ranks) | set(als_ranks))

            for rank_init in all_ranks:
                if rank_init in als_ranks:
                    # No-pruning baseline.
                    for method, prune_method, mu, qaoa_shots_requested in [
                        ("ALS", "none", None, None),
                    ]:
                        run_label = f"method={method} rank_init={rank_init} mu={mu}"
                        print(
                            f"[run] method={method} rank_init={rank_init} mu={mu} "
                            f"qaoa_shots={qaoa_shots_requested}"
                        )
                        run_id = {
                            "dataset_tag": dataset_tag,
                            "dataset_path": str(record.path),
                            "seed": record.seed,
                            "rank_true": record.rank_true,
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
                                shape=record.shape,
                                rank_init=rank_init,
                                ridge=args.ridge,
                                n_sweeps=args.t_max,
                                tol=args.tol,
                                seed=record.seed,
                                prune_method=prune_method,
                                prune_mu=mu,
                                prune_period=args.prune_period,
                                max_rank_prune=rank_init,
                                qaoa_reps=args.qaoa_reps,
                                qaoa_shots_effective=1,
                                qaoa_maxiter=args.qaoa_maxiter,
                                qaoa_seed=record.seed,
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

                if rank_init not in prune_ranks:
                    continue

                # Top-K pruning baseline.
                for top_k in topk_values:
                    if top_k <= 0 or top_k > rank_init:
                        continue
                    method = "ALS w/ Top-K"
                    prune_method = "top-k"
                    mu = 0.0
                    qaoa_shots_requested = None
                    run_label = f"method={method} rank_init={rank_init} top_k={top_k}"
                    print(
                        f"[run] method={method} rank_init={rank_init} top_k={top_k} "
                        f"qaoa_shots={qaoa_shots_requested}"
                    )
                    run_id = {
                        "dataset_tag": dataset_tag,
                        "dataset_path": str(record.path),
                        "seed": record.seed,
                        "rank_true": record.rank_true,
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
                            shape=record.shape,
                            rank_init=rank_init,
                            ridge=args.ridge,
                            n_sweeps=args.t_max,
                            tol=args.tol,
                            seed=record.seed,
                            prune_method=prune_method,
                            prune_mu=mu,
                            prune_period=args.prune_period,
                            max_rank_prune=top_k,
                            qaoa_reps=args.qaoa_reps,
                            qaoa_shots_effective=1,
                            qaoa_maxiter=args.qaoa_maxiter,
                            qaoa_seed=record.seed,
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

                # QUBO pruning baselines.
                for mu in mu_values:
                    mu_effective = _scale_mu(mu, args.mu_scale, xnorm=xnorm, rank_ref=rank_init)
                    for method, prune_method in [
                        ("ALS-LR (ES)", "exhaustive"),
                        ("ALS-LR (BF)", "bitflip"),
                    ]:
                        qaoa_shots_requested = None
                        run_label = f"method={method} rank_init={rank_init} mu={mu}"
                        print(
                            f"[run] method={method} rank_init={rank_init} mu={mu} "
                            f"qaoa_shots={qaoa_shots_requested}"
                        )
                        run_id = {
                            "dataset_tag": dataset_tag,
                            "dataset_path": str(record.path),
                            "seed": record.seed,
                            "rank_true": record.rank_true,
                            "rank_init": rank_init,
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
                                shape=record.shape,
                                rank_init=rank_init,
                                ridge=args.ridge,
                                n_sweeps=args.t_max,
                                tol=args.tol,
                                seed=record.seed,
                                prune_method=prune_method,
                                prune_mu=mu_effective,
                                prune_period=args.prune_period,
                                max_rank_prune=rank_init,
                                qaoa_reps=args.qaoa_reps,
                                qaoa_shots_effective=1,
                                qaoa_maxiter=args.qaoa_maxiter,
                                qaoa_seed=record.seed,
                                qaoa_restarts=args.qaoa_restarts,
                                bitflip_budget=args.bitflip_budget,
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

                    # QAOA.
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
                        run_label = f"method=QALR-CPD rank_init={rank_init} mu={mu}"
                        print(
                            f"[run] method=QALR-CPD rank_init={rank_init} mu={mu} "
                            f"qaoa_shots={qaoa_shots_requested}"
                        )
                        run_id = {
                            "dataset_tag": dataset_tag,
                            "dataset_path": str(record.path),
                            "seed": record.seed,
                            "rank_true": record.rank_true,
                            "rank_init": rank_init,
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
                                shape=record.shape,
                                rank_init=rank_init,
                                ridge=args.ridge,
                                n_sweeps=args.t_max,
                                tol=args.tol,
                                seed=record.seed,
                                prune_method="qaoa",
                                prune_mu=mu_effective,
                                prune_period=args.prune_period,
                                max_rank_prune=rank_init,
                                qaoa_reps=args.qaoa_reps,
                                qaoa_shots_effective=qaoa_shots_effective,
                                qaoa_maxiter=args.qaoa_maxiter,
                                qaoa_seed=record.seed,
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


if __name__ == "__main__":
    main()
