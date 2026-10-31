#!/usr/bin/env python3
"""
Runtime-focused CP-ALS pruning experiment on the amino acid dataset.

This script compares pruning solvers under the same CP-ALS schedule:
  - Exhaustive search (ALS-LR (ES))
  - Bit-flip local search (ALS-LR (BF))
  - QAOA (QALR-CPD)

Outputs:
  1) Per-sweep CSV (default: results/v1/running_time_amino.csv)
  2) Per-run summary CSV (default: results/v1/running_time_amino_summary.csv)
  3) Pruning-call-only CSV (default: results/v1/running_time_amino_prune_calls.csv)
  4) Tee log file mirroring console output in real time.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "pandas is required to run this experiment. Please install pandas and retry."
    ) from exc

from src.cp.als import CPFactorization
from src.pruning.bitflip import bit_flipping_prune_cp
from src.pruning.exhaustive import (
    exhaustive_prune_cp,
    inner_product_rank1,
    normalize_cp_factors,
)
from src.pruning.qaoa import qaoa_prune_cp


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


class _TeeStream:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextlib.contextmanager
def _tee_console_to_file(log_path: Optional[Path], overwrite: bool) -> None:
    if log_path is None:
        yield
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    with log_path.open(mode, encoding="utf-8", buffering=1) as log_handle:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        tee_stream = _TeeStream(old_stdout, log_handle)
        sys.stdout = tee_stream
        sys.stderr = tee_stream
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


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
    bitflip_budget: Optional[int],
    bitflip_restarts: int,
    bitflip_init_mode: str,
    bitflip_tie_break: str,
    bitflip_tol: float,
    bitflip_seed: int,
    bitflip_max_rank: Optional[int],
    log_every: int,
    verbose: bool,
    run_label: str,
) -> Tuple[List[Dict], Dict]:
    torch.manual_seed(seed)
    model = CPFactorization(
        shape=shape,
        rank=int(rank_init),
        ridge=float(ridge),
        init="random",
        seed=int(seed),
        device=X.device,
        dtype=X.dtype,
    )

    rows: List[Dict] = []
    start = time.time()
    prunes_applied = 0
    prune_calls = 0
    prune_solver_total_s = 0.0
    prev_err: Optional[float] = None
    xnorm = torch.linalg.norm(X).clamp_min(1e-12)
    xnorm_val = float(xnorm.item())
    total_iters = int(n_sweeps)
    if total_iters <= 0:
        raise ValueError("n_sweeps must be a positive integer.")

    method_key = prune_method.lower().strip() if prune_method is not None else "none"
    prune_period_eff = max(1, int(prune_period))
    bitflip_seed_effective = int(bitflip_seed)
    mu_for_objective = 0.0 if prune_mu is None else float(prune_mu)

    err = float("nan")
    f_approx = float("nan")
    residual_norm_sq = float("nan")
    cumulative = 0.0
    sweep = -1

    for sweep in range(total_iters):
        sweep_t0 = time.time()
        model.als_step(X, normalize=True)

        is_prune_call = False
        prune_applied = False
        prune_solver_runtime_s = 0.0
        prune_call_error = ""
        rank_before_prune = int(model.rank)

        if method_key != "none" and prune_mu is not None:
            last_sweep = sweep == total_iters - 1
            should_prune = (not last_sweep) and ((sweep + 1) % prune_period_eff == 0)
            if should_prune:
                is_prune_call = True
                prune_calls += 1
                prune_t0 = time.time()
                try:
                    model._maybe_prune_components(
                        X=X,
                        iteration=sweep,
                        total_iters=total_iters,
                        prunning_enable=True,
                        prunning_mu=float(prune_mu),
                        prunning_frequency=prune_period_eff,
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
                except Exception as exc:  # pragma: no cover - safety net
                    prune_call_error = str(exc)
                    if verbose:
                        print(
                            f"[warn] prune call failed ({run_label}, sweep={sweep + 1}): "
                            f"{prune_call_error}"
                        )
                prune_solver_runtime_s = time.time() - prune_t0
                prune_solver_total_s += prune_solver_runtime_s
                prune_applied = int(model.rank) != rank_before_prune
                if prune_applied:
                    prunes_applied += 1

        X_hat = model.forward()
        residual = X - X_hat
        residual_norm_sq = float(torch.sum(residual * residual).item())
        err = float((residual_norm_sq ** 0.5) / xnorm_val)
        f_approx = float(residual_norm_sq + mu_for_objective * float(model.rank))

        sweep_elapsed = time.time() - sweep_t0
        cumulative = time.time() - start

        if log_every <= 1 or (sweep + 1) % log_every == 0:
            rows.append(
                {
                    "sweep": sweep + 1,
                    "rel_error": err,
                    "f_approx": f_approx,
                    "residual_norm_sq": residual_norm_sq,
                    "rank": int(model.rank),
                    "rank_before_prune": rank_before_prune,
                    "is_prune_call": bool(is_prune_call),
                    "prune_call_idx": prune_calls if is_prune_call else None,
                    "prune_applied": bool(prune_applied),
                    "prune_solver_runtime_s": prune_solver_runtime_s,
                    "prune_call_error": prune_call_error,
                    "sweep_time_s": sweep_elapsed,
                    "cumulative_s": cumulative,
                }
            )
            if verbose:
                print(
                    f"[sweep] {run_label} sweep={sweep + 1:03d} rel_error={err:.6e} "
                    f"f_approx={f_approx:.6e} rank={model.rank} prune_call={int(is_prune_call)} "
                    f"prune_applied={int(prune_applied)} prune_runtime_s={prune_solver_runtime_s:.6e}"
                )

        if tol > 0 and prev_err is not None and abs(prev_err - err) < tol:
            break
        prev_err = err

    sweeps_done = sweep + 1
    summary = {
        "sweeps": sweeps_done,
        "final_rel_error": err,
        "final_f_approx": f_approx,
        "final_residual_norm_sq": residual_norm_sq,
        "final_rank": int(model.rank),
        "prunes_applied": int(prunes_applied),
        "prune_calls": int(prune_calls),
        "prune_solver_total_s": float(prune_solver_total_s),
        "prune_solver_mean_s": (
            float(prune_solver_total_s) / float(prune_calls) if prune_calls > 0 else 0.0
        ),
        "total_time_s": float(cumulative),
    }
    return rows, summary


def _log_run_done(run_label: str, summary: Dict) -> None:
    status = summary.get("status", "ok")
    if status != "ok":
        print(f"[done] {run_label} status=failed error={summary.get('error')}")
        return
    print(
        f"[done] {run_label} final_rel_error={summary.get('final_rel_error'):.6e} "
        f"final_f_approx={summary.get('final_f_approx'):.6e} "
        f"final_rank={summary.get('final_rank')} sweeps={summary.get('sweeps')} "
        f"prune_calls={summary.get('prune_calls')} "
        f"mean_prune_runtime_s={summary.get('prune_solver_mean_s'):.6e}"
    )


def _solve_prune_on_snapshot(
    method: str,
    X: torch.Tensor,
    factors: Sequence[torch.Tensor],
    weights: torch.Tensor,
    mu_effective: float,
    max_rank_exhaustive: int,
    max_rank_qaoa: int,
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
    bitflip_seed: int,
    bitflip_max_rank: Optional[int],
) -> Dict:
    solver_t0 = time.time()
    selected: List[int]
    lam: torch.Tensor
    U: List[torch.Tensor]
    diag: Dict

    if method == "exhaustive":
        selected, _, (lam, U), diag = exhaustive_prune_cp(
            X=X,
            factors=factors,
            mu=float(mu_effective),
            weights=weights,
            require_nonempty=True,
            max_R_exhaustive=int(max_rank_exhaustive),
            qubo_scale=1.0,
        )
    elif method == "bitflip":
        selected, _, (lam, U), diag = bit_flipping_prune_cp(
            X=X,
            factors=factors,
            mu=float(mu_effective),
            weights=weights,
            require_nonempty=True,
            max_R_bitflip=bitflip_max_rank,
            budget=bitflip_budget,
            restarts=int(bitflip_restarts),
            init_mode=bitflip_init_mode,
            tie_break=bitflip_tie_break,
            tol=float(bitflip_tol),
            seed=int(bitflip_seed),
            qubo_scale=1.0,
        )
    elif method == "qaoa":
        selected, _, (lam, U), diag = qaoa_prune_cp(
            X=X,
            factors=factors,
            mu=float(mu_effective),
            weights=weights,
            require_nonempty=True,
            max_R_qaoa=int(max_rank_qaoa),
            qaoa_reps=int(qaoa_reps),
            shots=int(qaoa_shots_effective),
            maxiter=int(qaoa_maxiter),
            seed=int(qaoa_seed),
            num_restarts=int(qaoa_restarts),
            qubo_scale=1.0,
        )
    else:
        raise ValueError(f"Unknown method '{method}'")

    solver_runtime_s = time.time() - solver_t0
    selected = sorted(int(i) for i in selected)
    if len(selected) == 0:
        raise ValueError("Pruning solver returned empty selection with require_nonempty=True.")

    return {
        "status": "ok",
        "error": "",
        "selected": selected,
        "lam": lam,
        "U": U,
        "rank_after": int(len(selected)),
        "solver_runtime_s": float(solver_runtime_s),
        "diagnostics": diag if isinstance(diag, dict) else {},
    }


def _build_snapshot_es_terms(
    X: torch.Tensor,
    factors: Sequence[torch.Tensor],
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build ES objective terms for a fixed pruning snapshot:
      E_no_const(z) = z^T G z - 2 z^T c + mu * sum(z)
    where full objective is ||X||_F^2 + E_no_const(z).
    """
    lam_ref, U_ref = normalize_cp_factors(factors=factors, weights=weights)
    R = U_ref[0].shape[1]
    device = X.device
    dtype = X.dtype
    lam_ref = lam_ref.to(device=device, dtype=dtype)
    U_ref = [u.to(device=device, dtype=dtype) for u in U_ref]

    overlaps = torch.ones((R, R), device=device, dtype=dtype)
    for n in range(len(U_ref)):
        overlaps = overlaps * (U_ref[n].T @ U_ref[n])
    G = (lam_ref[:, None] * lam_ref[None, :]) * overlaps

    c = torch.zeros((R,), device=device, dtype=dtype)
    for r in range(R):
        u_cols = [U_ref[n][:, r] for n in range(len(U_ref))]
        c[r] = lam_ref[r] * inner_product_rank1(X, u_cols)
    return G, c


def _mask_from_selected(selected: Sequence[int], R: int) -> List[int]:
    mask = [0] * int(R)
    for idx in selected:
        i = int(idx)
        if i < 0 or i >= int(R):
            raise ValueError(f"Selected index out of range: {i} for R={R}")
        mask[i] = 1
    return mask


def _objective_from_mask_es_terms(
    mask_bits: Sequence[int],
    G: torch.Tensor,
    c: torch.Tensor,
    mu_effective: float,
    xnorm_sq: float,
) -> Tuple[float, float, float]:
    z = torch.tensor(mask_bits, device=G.device, dtype=G.dtype)
    mu_val = float(mu_effective)
    obj_no_const_t = (z @ (G @ z)) - 2.0 * (z @ c) + mu_val * z.sum()
    obj_no_const = float(obj_no_const_t.detach().cpu())
    obj_full = float(float(xnorm_sq) + obj_no_const)
    residual_norm_sq = float(obj_no_const - mu_val * float(z.sum().item()) + float(xnorm_sq))
    if residual_norm_sq < 0.0 and residual_norm_sq > -1e-8:
        residual_norm_sq = 0.0
    return obj_full, obj_no_const, residual_norm_sq


def run_shared_input_benchmark(
    X: torch.Tensor,
    shape: Sequence[int],
    rank_init: int,
    ridge: float,
    n_sweeps: int,
    tol: float,
    seed: int,
    prune_mu_effective: float,
    prune_period: int,
    max_rank_exhaustive: int,
    max_rank_qaoa: int,
    qaoa_reps: int,
    qaoa_shots_effective: int,
    qaoa_maxiter: int,
    qaoa_restarts: int,
    bitflip_budget: Optional[int],
    bitflip_restarts: int,
    bitflip_init_mode: str,
    bitflip_tie_break: str,
    bitflip_tol: float,
    bitflip_max_rank: Optional[int],
    log_every: int,
    verbose: bool,
    run_label_prefix: str,
    max_prune_calls: Optional[int] = None,
) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
    methods = ["exhaustive", "bitflip", "qaoa"]
    method_rows: Dict[str, List[Dict]] = {m: [] for m in methods}
    xnorm = torch.linalg.norm(X).clamp_min(1e-12)
    xnorm_val = float(xnorm.item())
    xnorm_sq = float(torch.sum(X * X).item())

    model = CPFactorization(
        shape=shape,
        rank=int(rank_init),
        ridge=float(ridge),
        init="random",
        seed=int(seed),
        device=X.device,
        dtype=X.dtype,
    )

    start = time.time()
    prev_err: Optional[float] = None
    prune_call_idx = 0
    total_iters = int(n_sweeps)
    if total_iters <= 0:
        raise ValueError("n_sweeps must be a positive integer.")

    for sweep in range(total_iters):
        model.als_step(X, normalize=True)
        current_err = float((torch.linalg.norm(X - model.forward()) / xnorm).item())
        rank_before = int(model.rank)
        should_prune = (sweep < total_iters - 1) and ((sweep + 1) % max(1, int(prune_period)) == 0)
        if not should_prune:
            if tol > 0 and prev_err is not None and abs(prev_err - current_err) < tol:
                break
            prev_err = current_err
            continue

        prune_call_idx += 1
        factors_snapshot = [A.detach().clone() for A in model.factors]
        weights_snapshot = model.weights.detach().clone()
        G_es, c_es = _build_snapshot_es_terms(
            X=X,
            factors=factors_snapshot,
            weights=weights_snapshot,
        )
        R_snapshot = int(factors_snapshot[0].shape[1]) if len(factors_snapshot) > 0 else 0
        input_id = (
            f"{run_label_prefix}|seed={seed}|sweep={sweep + 1}|call={prune_call_idx}|"
            f"rank_before={rank_before}"
        )
        method_results: Dict[str, Dict] = {}

        for method in methods:
            try:
                result = _solve_prune_on_snapshot(
                    method=method,
                    X=X,
                    factors=factors_snapshot,
                    weights=weights_snapshot,
                    mu_effective=prune_mu_effective,
                    max_rank_exhaustive=max_rank_exhaustive,
                    max_rank_qaoa=max_rank_qaoa,
                    qaoa_reps=qaoa_reps,
                    qaoa_shots_effective=qaoa_shots_effective,
                    qaoa_maxiter=qaoa_maxiter,
                    qaoa_seed=seed,
                    qaoa_restarts=qaoa_restarts,
                    bitflip_budget=bitflip_budget,
                    bitflip_restarts=bitflip_restarts,
                    bitflip_init_mode=bitflip_init_mode,
                    bitflip_tie_break=bitflip_tie_break,
                    bitflip_tol=bitflip_tol,
                    bitflip_seed=seed,
                    bitflip_max_rank=bitflip_max_rank,
                )
            except Exception as exc:  # pragma: no cover - runtime safety
                result = {
                    "status": "failed",
                    "error": str(exc),
                    "selected": list(range(rank_before)),
                    "lam": weights_snapshot,
                    "U": factors_snapshot,
                    "rank_after": rank_before,
                    "solver_runtime_s": 0.0,
                    "diagnostics": {},
                }

            method_results[method] = result
            rank_after = int(result["rank_after"])
            mask_bits = _mask_from_selected(result["selected"], R_snapshot)
            mask_bitstring = "".join(str(b) for b in mask_bits)
            selected_indices = ",".join(str(i) for i in result["selected"])

            if result["status"] == "ok":
                f_approx, objective_no_const, residual_norm_sq = _objective_from_mask_es_terms(
                    mask_bits=mask_bits,
                    G=G_es,
                    c=c_es,
                    mu_effective=prune_mu_effective,
                    xnorm_sq=xnorm_sq,
                )
                rel_error = float((max(0.0, residual_norm_sq) ** 0.5) / xnorm_val)
            else:
                f_approx = float("nan")
                objective_no_const = float("nan")
                residual_norm_sq = float("nan")
                rel_error = float("nan")

            row = {
                "sweep": sweep + 1,
                "rel_error": rel_error,
                "f_approx": f_approx,
                "objective_no_const": objective_no_const,
                "objective_constant": xnorm_sq,
                "residual_norm_sq": residual_norm_sq,
                "rank": rank_after,
                "rank_before_prune": rank_before,
                "mask_bitstring": mask_bitstring,
                "selected_indices": selected_indices,
                "is_prune_call": True,
                "prune_call_idx": prune_call_idx,
                "prune_applied": bool(rank_after != rank_before),
                "prune_solver_runtime_s": float(result["solver_runtime_s"]),
                "prune_call_error": str(result["error"]),
                "sweep_time_s": float(result["solver_runtime_s"]),
                "cumulative_s": time.time() - start,
                "status": result["status"],
                "input_id": input_id,
            }
            if log_every <= 1 or (sweep + 1) % log_every == 0 or verbose:
                print(
                    f"[sweep] {run_label_prefix} method={method} sweep={sweep + 1:03d} "
                    f"call={prune_call_idx} rank_before={rank_before} rank_after={rank_after} "
                    f"f_approx={row['f_approx']:.6e} runtime_s={row['prune_solver_runtime_s']:.6e} "
                    f"status={row['status']}"
                )
            method_rows[method].append(row)

        # Keep trajectory identical across methods by advancing with ES decision only.
        es_res = method_results["exhaustive"]
        if es_res["status"] == "ok":
            es_selected = es_res["selected"]
            es_lam = es_res["lam"]
            es_U = es_res["U"]
            if len(es_selected) < rank_before:
                model._apply_pruning_selection(es_lam, es_U, es_selected)
            else:
                model._set_weights_tensor(es_lam)
        else:
            if verbose:
                print(
                    f"[warn] {run_label_prefix} ES failed at sweep={sweep + 1}; "
                    f"keeping current model state."
                )

        if max_prune_calls is not None and int(max_prune_calls) > 0:
            if prune_call_idx >= int(max_prune_calls):
                if verbose:
                    print(
                        f"[info] {run_label_prefix} reached max_prune_calls="
                        f"{int(max_prune_calls)}; stopping run."
                    )
                break

        if tol > 0 and prev_err is not None and abs(prev_err - current_err) < tol:
            break
        prev_err = current_err

    sweeps_done = sweep + 1
    summaries: Dict[str, Dict] = {}
    for method in methods:
        rows = method_rows[method]
        ok_rows = [r for r in rows if r.get("status") == "ok"]
        last_ok = ok_rows[-1] if ok_rows else None
        total_solver_s = float(sum(float(r.get("prune_solver_runtime_s", 0.0)) for r in rows))
        prune_calls = int(len(rows))
        summary = {
            "sweeps": sweeps_done,
            "final_rel_error": last_ok["rel_error"] if last_ok is not None else None,
            "final_f_approx": last_ok["f_approx"] if last_ok is not None else None,
            "final_residual_norm_sq": (
                last_ok["residual_norm_sq"] if last_ok is not None else None
            ),
            "final_rank": last_ok["rank"] if last_ok is not None else None,
            "prunes_applied": int(sum(int(bool(r.get("prune_applied"))) for r in rows)),
            "prune_calls": prune_calls,
            "prune_solver_total_s": total_solver_s,
            "prune_solver_mean_s": (total_solver_s / float(prune_calls)) if prune_calls > 0 else 0.0,
            "total_time_s": time.time() - start,
            "status": "ok" if len(ok_rows) == len(rows) else "failed",
            "error": "" if len(ok_rows) == len(rows) else "at least one prune call failed",
            "final_rec": None,
        }
        if prune_calls == 0:
            summary["status"] = "failed"
            summary["error"] = (
                "no prune call executed; increase --t-max or reduce --prune-period"
            )
        summaries[method] = summary

    return method_rows, summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run amino runtime experiment for ES/BF/QAOA pruning."
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
    parser.add_argument("--results-csv", type=str, default="results/v1/running_time_amino.csv")
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="results/v1/running_time_amino_summary.csv",
    )
    parser.add_argument(
        "--calls-csv",
        type=str,
        default="results/v1/running_time_amino_prune_calls.csv",
        help="Optional pruning-call-only CSV path. Set empty string to disable.",
    )
    parser.add_argument("--ranks", type=str, default="20,21,22,23,24,25,26,27,28,29,30")
    parser.add_argument("--t-max", type=int, default=200)
    parser.add_argument("--tol", type=float, default=0.0)
    parser.add_argument("--prune-period", type=int, default=40)
    parser.add_argument(
        "--max-prune-calls-per-run",
        type=int,
        default=0,
        help=(
            "Stop each (dataset, seed, rank, mu, shots) run after this many prune calls. "
            "Use 1 to benchmark only the first induced pruning problem at each R. "
            "Use 0 for no cap (default)."
        ),
    )
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--mu-values", type=str, default="0.1")
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
        "--max-rank-exhaustive",
        type=int,
        default=None,
        help="Cap for exhaustive pruning rank. Defaults to rank_init when unset.",
    )
    parser.add_argument(
        "--max-rank-qaoa",
        type=int,
        default=None,
        help="Cap for QAOA pruning rank. Defaults to rank_init when unset.",
    )
    parser.add_argument(
        "--bitflip-max-rank",
        type=int,
        default=None,
        help="Cap for bit-flip pruning rank. Defaults to no cap when unset.",
    )
    parser.add_argument(
        "--bitflip-budget",
        type=int,
        default=None,
        help="Bit-flip iteration budget. Defaults to --qaoa-maxiter when unset.",
    )
    parser.add_argument("--bitflip-restarts", type=int, default=1)
    parser.add_argument("--bitflip-init-mode", type=str, default="zeros")
    parser.add_argument("--bitflip-tie-break", type=str, default="first")
    parser.add_argument("--bitflip-tol", type=float, default=1e-12)
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of random seeds. Runtime sweeps at high R can be expensive.",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--log-file",
        type=str,
        default="results/v1/running_time_amino.log",
        help="Log file path. Use empty string to disable tee logging.",
    )
    parser.add_argument(
        "--overwrite-log",
        action="store_true",
        help="Overwrite --log-file instead of appending.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete existing output CSV files before starting.",
    )
    parser.add_argument(
        "--checkpoint-every-runs",
        type=int,
        default=1,
        help="Write partial CSV outputs every N completed runs (0 disables).",
    )
    parser.add_argument(
        "--plot-checkpoint-every-runs",
        type=int,
        default=0,
        help="Regenerate running-time plots every N completed runs (0 disables).",
    )
    parser.add_argument(
        "--plot-output-dir",
        type=str,
        default="reports/figures/v1/running_time",
        help="Output directory for checkpoint plots.",
    )
    parser.add_argument(
        "--plot-runtime-figure",
        type=str,
        default="runtime_per_pruning_call_vs_R.partial.png",
        help="Checkpoint runtime plot filename.",
    )
    parser.add_argument(
        "--plot-error-figure",
        type=str,
        default="normalized_error_vs_R.partial.png",
        help="Checkpoint normalized-error plot filename.",
    )
    parser.add_argument(
        "--plot-r-col",
        type=str,
        default="rank_before_prune",
        help="R column for checkpoint plots.",
    )
    parser.add_argument(
        "--plot-r-source-method",
        type=str,
        default="min",
        choices=["ES", "BF", "QAOA", "min", "max", "mean"],
        help="Method/aggregation used to derive dynamic R for checkpoint plots.",
    )
    return parser.parse_args()


def _run_experiment(args: argparse.Namespace) -> None:
    data_paths: List[Path] = []
    if args.data_path:
        data_paths = [Path(p) for p in args.data_path]
    else:
        data_paths = list(Path(".").glob(args.data_glob))

    if not data_paths:
        raise SystemExit("No data files found. Use --data-path or --data-glob.")

    ranks = _parse_list_int(args.ranks)
    mu_values = _parse_list_float(args.mu_values)
    qaoa_shots_list = _parse_list_int(args.qaoa_shots)
    if len(qaoa_shots_list) == 0:
        raise ValueError("--qaoa-shots must contain at least one integer.")

    dtype = _dtype_from_str(args.dtype)
    bitflip_budget = args.bitflip_budget if args.bitflip_budget is not None else args.qaoa_maxiter
    if args.seeds:
        run_seeds = _parse_list_int(args.seeds)
    else:
        run_seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    results_path = Path(args.results_csv)
    summary_path = Path(args.summary_csv)
    calls_path = Path(args.calls_csv) if args.calls_csv else None

    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if calls_path is not None:
        calls_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite_output:
        for out_path in (results_path, summary_path, calls_path):
            if out_path is not None and out_path.exists():
                out_path.unlink()

    print(f"[info] started_at={dt.datetime.now().isoformat(timespec='seconds')}")
    print(f"[info] data_paths={[str(p) for p in data_paths]}")
    print(f"[info] ranks={ranks}")
    print(f"[info] mu_values={mu_values}")
    print(f"[info] seeds={run_seeds}")
    print(f"[info] prune_period={args.prune_period} t_max={args.t_max}")
    if int(args.max_prune_calls_per_run) > 0:
        print(f"[info] max_prune_calls_per_run={int(args.max_prune_calls_per_run)}")
        if args.t_max <= args.prune_period:
            print(
                "[warn] t_max <= prune_period may skip pruning entirely because pruning "
                "is not applied on the last sweep; consider setting --t-max >= prune_period+1."
            )
    print(f"[info] qaoa_shots={qaoa_shots_list} qaoa_reps={args.qaoa_reps}")
    print(
        "[info] caution: exhaustive pruning scales exponentially with R; "
        "R in [20, 30] can be very expensive."
    )
    print(
        "[info] mode=shared_input (strict apple-to-apple): each prune call benchmarks "
        "ES/BF/QAOA on the same snapshot; trajectory update uses ES selection."
    )

    all_rows: List[Dict] = []
    summaries: List[Dict] = []
    total_runs = (
        len(data_paths)
        * len(run_seeds)
        * len(ranks)
        * len(mu_values)
        * (3 * len(qaoa_shots_list))
    )
    completed_runs = 0
    t_global_start = time.time()
    print(f"[info] total_planned_runs={total_runs}")

    method_specs = [
        ("ALS-LR (ES)", "exhaustive"),
        ("ALS-LR (BF)", "bitflip"),
        ("QALR-CPD", "qaoa"),
    ]

    def _format_hms(seconds: float) -> str:
        sec = max(0.0, float(seconds))
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _write_checkpoint_files() -> None:
        if all_rows:
            sweep_df = pd.DataFrame(all_rows)
            sweep_df.to_csv(results_path, index=False)
            if calls_path is not None:
                calls_df = sweep_df[sweep_df["is_prune_call"] == True].copy()  # noqa: E712
                calls_df.to_csv(calls_path, index=False)
        if summaries:
            pd.DataFrame(summaries).to_csv(summary_path, index=False)

    def _maybe_plot_checkpoint(force: bool = False) -> None:
        if args.plot_checkpoint_every_runs <= 0:
            return
        if (not force) and (completed_runs % int(args.plot_checkpoint_every_runs) != 0):
            return
        if not results_path.exists():
            return
        output_dir = Path(args.plot_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "legacy/v1/plotting/running_time/time_plot.py",
            "--csv-path",
            str(results_path),
            "--output-dir",
            str(output_dir),
            "--runtime-figure",
            args.plot_runtime_figure,
            "--error-figure",
            args.plot_error_figure,
            "--r-col",
            args.plot_r_col,
            "--r-source-method",
            args.plot_r_source_method,
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(
                f"[checkpoint] updated plots: {output_dir / args.plot_runtime_figure}, "
                f"{output_dir / args.plot_error_figure}"
            )
        except Exception as exc:
            print(f"[warn] checkpoint plotting skipped at run {completed_runs}: {exc}")

    for data_path in data_paths:
        if not data_path.exists():
            print(f"[warn] dataset file missing: {data_path}")
            continue

        payload = torch.load(data_path, map_location="cpu")
        X = payload["X"].to(device=args.device, dtype=dtype)
        cfg = payload.get("config", {})
        shape = tuple(cfg.get("shape", tuple(X.shape)))
        rank_true = int(cfg.get("rank_true", 0))
        corr_rho = float(cfg.get("corr_rho", 0.0))
        noise_std = float(cfg.get("noise_std", 0.0))
        dataset_tag = _dataset_tag(shape, rank_true, corr_rho, noise_std)
        data_seed = int(payload.get("seed", 0))
        xnorm = float(torch.linalg.norm(X).clamp_min(1e-12).item())

        print(f"[info] dataset={data_path} seed={data_seed} tag={dataset_tag} shape={shape}")
        if args.mu_scale != "none":
            print(f"[info] mu_scale={args.mu_scale} ||X||_F={xnorm:.6e}")

        for run_seed in run_seeds:
            print(f"[info] starting run seed={run_seed}")
            for rank_init in ranks:
                for mu in mu_values:
                    mu_effective = _scale_mu(mu, args.mu_scale, xnorm=xnorm, rank_ref=rank_init)
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

                        max_rank_exhaustive = (
                            int(args.max_rank_exhaustive)
                            if args.max_rank_exhaustive is not None
                            else int(rank_init)
                        )
                        max_rank_qaoa = (
                            int(args.max_rank_qaoa)
                            if args.max_rank_qaoa is not None
                            else int(rank_init)
                        )
                        run_label_prefix = (
                            f"shared_input rank_init={rank_init} seed={run_seed} "
                            f"mu={mu} shots={qaoa_shots_requested}"
                        )
                        print(
                            "[run] "
                            f"{run_label_prefix} (ES/BF/QAOA evaluated on identical input per call)"
                        )

                        method_rows: Dict[str, List[Dict]]
                        method_summaries: Dict[str, Dict]
                        try:
                            method_rows, method_summaries = run_shared_input_benchmark(
                                X=X,
                                shape=shape,
                                rank_init=rank_init,
                                ridge=args.ridge,
                                n_sweeps=args.t_max,
                                tol=args.tol,
                                seed=run_seed,
                                prune_mu_effective=mu_effective,
                                prune_period=args.prune_period,
                                max_rank_exhaustive=max_rank_exhaustive,
                                max_rank_qaoa=max_rank_qaoa,
                                qaoa_reps=args.qaoa_reps,
                                qaoa_shots_effective=int(qaoa_shots_effective),
                                qaoa_maxiter=args.qaoa_maxiter,
                                qaoa_restarts=args.qaoa_restarts,
                                bitflip_budget=bitflip_budget,
                                bitflip_restarts=args.bitflip_restarts,
                                bitflip_init_mode=args.bitflip_init_mode,
                                bitflip_tie_break=args.bitflip_tie_break,
                                bitflip_tol=args.bitflip_tol,
                                bitflip_max_rank=args.bitflip_max_rank,
                                log_every=args.log_every,
                                verbose=args.verbose,
                                run_label_prefix=run_label_prefix,
                                max_prune_calls=(
                                    int(args.max_prune_calls_per_run)
                                    if int(args.max_prune_calls_per_run) > 0
                                    else None
                                ),
                            )
                        except Exception as exc:  # pragma: no cover - runtime safety
                            method_rows = {m[1]: [] for m in method_specs}
                            method_summaries = {
                                m[1]: {
                                    "sweeps": 0,
                                    "final_rel_error": None,
                                    "final_f_approx": None,
                                    "final_residual_norm_sq": None,
                                    "final_rank": None,
                                    "prunes_applied": None,
                                    "prune_calls": None,
                                    "prune_solver_total_s": None,
                                    "prune_solver_mean_s": None,
                                    "total_time_s": None,
                                    "final_rec": None,
                                    "status": "failed",
                                    "error": str(exc),
                                }
                                for m in method_specs
                            }

                        for method_name, prune_method in method_specs:
                            run_label = (
                                f"method={method_name} rank_init={rank_init} seed={run_seed} "
                                f"mu={mu} prune_method={prune_method} shots={qaoa_shots_requested}"
                            )
                            run_id = {
                                "dataset_tag": dataset_tag,
                                "dataset_path": str(data_path),
                                "data_seed": data_seed,
                                "seed": run_seed,
                                "rank_true": rank_true,
                                "rank_init": rank_init,
                                "top_k": None,
                                "method": method_name,
                                "prune_method": prune_method,
                                "prune_mu": mu,
                                "prune_mu_effective": mu_effective,
                                "mu_scale": args.mu_scale,
                                "prune_period": args.prune_period,
                                "ridge": args.ridge,
                                "qaoa_reps": args.qaoa_reps,
                                "qaoa_shots_requested": qaoa_shots_requested,
                                # Keep identical shot metadata for all 3 methods to ensure same run grouping.
                                "qaoa_shots_effective": qaoa_shots_effective,
                                "qaoa_maxiter": args.qaoa_maxiter,
                                "qaoa_restarts": args.qaoa_restarts,
                                "qaoa_shots_note": shots_note,
                                "bitflip_budget": bitflip_budget,
                                "bitflip_restarts": args.bitflip_restarts,
                                "bitflip_init_mode": args.bitflip_init_mode,
                                "bitflip_tie_break": args.bitflip_tie_break,
                                "bitflip_tol": args.bitflip_tol,
                                "bitflip_max_rank": args.bitflip_max_rank,
                                "max_rank_prune": (
                                    max_rank_exhaustive
                                    if prune_method == "exhaustive"
                                    else (max_rank_qaoa if prune_method == "qaoa" else rank_init)
                                ),
                                "shared_input": True,
                            }

                            summary = method_summaries.get(prune_method, {})
                            _log_run_done(run_label, summary)
                            for row in method_rows.get(prune_method, []):
                                all_rows.append({**run_id, **row})
                            summaries.append({**run_id, **summary})
                            completed_runs += 1

                            elapsed = time.time() - t_global_start
                            avg_per_run = elapsed / max(1, completed_runs)
                            remaining = max(0, total_runs - completed_runs)
                            eta_s = avg_per_run * remaining
                            print(
                                f"[progress] runs={completed_runs}/{total_runs} "
                                f"elapsed={_format_hms(elapsed)} eta={_format_hms(eta_s)} "
                                f"avg_per_run={avg_per_run:.2f}s"
                            )

                            if args.checkpoint_every_runs > 0 and (
                                completed_runs % int(args.checkpoint_every_runs) == 0
                            ):
                                _write_checkpoint_files()
                                print(
                                    f"[checkpoint] wrote partial outputs at run "
                                    f"{completed_runs}/{total_runs}"
                                )
                                _maybe_plot_checkpoint()

    # Final write (also covers trailing runs not aligned with checkpoint interval).
    _write_checkpoint_files()
    if args.plot_checkpoint_every_runs > 0:
        _maybe_plot_checkpoint(force=True)

    if all_rows:
        print(f"[info] wrote sweep metrics to {results_path}")
        if calls_path is not None:
            print(f"[info] wrote pruning-call metrics to {calls_path}")
    else:
        print("[warn] no per-sweep rows were collected.")

    if summaries:
        print(f"[info] wrote summary metrics to {summary_path}")
        summary_df = pd.DataFrame(summaries)
        ok_df = summary_df[summary_df["status"] == "ok"].copy()
        if not ok_df.empty:
            group_cols = [
                "dataset_tag",
                "method",
                "rank_init",
                "prune_mu",
                "qaoa_shots_effective",
            ]
            agg = (
                ok_df.groupby(group_cols, dropna=False)
                .agg(
                    final_rel_error_mean=("final_rel_error", "mean"),
                    final_rel_error_std=("final_rel_error", "std"),
                    final_f_approx_mean=("final_f_approx", "mean"),
                    final_f_approx_std=("final_f_approx", "std"),
                    final_rank_mean=("final_rank", "mean"),
                    final_rank_std=("final_rank", "std"),
                    mean_prune_runtime_s=("prune_solver_mean_s", "mean"),
                    mean_total_runtime_s=("total_time_s", "mean"),
                )
                .reset_index()
            )
            for col in (
                "final_rel_error_std",
                "final_f_approx_std",
                "final_rank_std",
            ):
                if col in agg.columns:
                    agg[col] = agg[col].fillna(0.0)

            print("[info] mean+/-std summary (ok runs):")
            for _, row in agg.iterrows():
                print(
                    "  "
                    f"method={row['method']} R={row['rank_init']} mu={row['prune_mu']} "
                    f"shots={row['qaoa_shots_effective']} "
                    f"rel_error={row['final_rel_error_mean']:.6e}+/-{row['final_rel_error_std']:.6e} "
                    f"f_approx={row['final_f_approx_mean']:.6e}+/-{row['final_f_approx_std']:.6e} "
                    f"rank={row['final_rank_mean']:.2f}+/-{row['final_rank_std']:.2f} "
                    f"mean_prune_runtime_s={row['mean_prune_runtime_s']:.6e} "
                    f"mean_total_runtime_s={row['mean_total_runtime_s']:.6e}"
                )
        else:
            print("[warn] all runs failed; no aggregate statistics available.")
    else:
        print("[warn] no run summaries were collected.")

    print(f"[info] finished_at={dt.datetime.now().isoformat(timespec='seconds')}")


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_file) if args.log_file else None
    with _tee_console_to_file(log_path=log_path, overwrite=bool(args.overwrite_log)):
        if log_path is not None:
            mode = "overwrite" if args.overwrite_log else "append"
            print(f"[info] logging to {log_path} ({mode})")
        _run_experiment(args)


if __name__ == "__main__":
    main()
