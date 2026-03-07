#!/usr/bin/env python3
"""
Run lightweight amino poster experiments:
1) fit overfactored CP-ALS once per seed,
2) freeze factors and build a soft-rank QUBO,
3) solve with ES / BF / QAOA,
4) aggregate metrics and convergence traces.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "pandas is required to run this script. Please install pandas and retry."
    ) from exc


def _parse_list_int(raw: str) -> List[int]:
    return [int(x) for x in raw.split(",") if x.strip()]

def _resolve_data_paths(args: argparse.Namespace) -> List[Path]:
    if args.data_path:
        return [Path(p) for p in args.data_path]
    return sorted(Path(".").glob(args.data_glob))


def _resolve_run_seeds(args: argparse.Namespace) -> List[int]:
    if args.seed is not None:
        return [int(args.seed)]
    if args.seeds:
        return _parse_list_int(args.seeds)
    return list(range(int(args.seed_start), int(args.seed_start) + int(args.num_seeds)))


def _load_synth_module():
    try:
        return importlib.import_module("synthetic_poster_experiment")
    except Exception as exc:
        raise SystemExit(
            "Failed to import synthetic_poster_experiment helpers. "
            "Run from the repo root and ensure dependencies are installed."
        ) from exc


def _fmt_latex_pm(mean: float, std: float, sci: bool = False) -> str:
    if (not np.isfinite(mean)) or (not np.isfinite(std)):
        return r"\textit{--}"
    if sci:
        return f"{mean:.3e} $\\pm$ {std:.3e}"
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def write_latex_table(rows: Sequence[Dict[str, float]], output_path: Path) -> None:
    by_method = {str(r["solver"]): r for r in rows}

    def _row_values(solver_key: str, is_es: bool = False) -> Tuple[str, str, str]:
        row = by_method.get(solver_key)
        if row is None:
            return r"\textit{--}", r"\textit{--}", "0" if is_es else r"\textit{--}"
        rhat = _fmt_latex_pm(float(row["mean_Rhat"]), float(row["std_Rhat"]), sci=False)
        rel = _fmt_latex_pm(float(row["mean_relerr"]), float(row["std_relerr"]), sci=True)
        if is_es:
            delta = "0"
        else:
            delta = _fmt_latex_pm(
                float(row.get("mean_delta", float("nan"))),
                float(row.get("std_delta", float("nan"))),
                sci=True,
            )
        return rhat, rel, delta

    es_rhat, es_rel, es_delta = _row_values("ES", is_es=True)
    bf_rhat, bf_rel, bf_delta = _row_values("BF", is_es=False)
    sim_rhat, sim_rel, sim_delta = _row_values("QAOA", is_es=False)

    lines = [
        r"\begin{table}[t]",
        r"\caption{Amino Acids dataset: placeholder results for ES, BF, simulator-QAOA, and hardware-QAOA on the induced rank-selection QUBO.}",
        r"\label{tab:amino_metrics}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Method & \(\hat{R}\) & Rel.\ error \(\downarrow\) & \(\bar{\delta}\) \(\downarrow\) \\",
        r"\midrule",
        f"ES                               & {es_rhat} & {es_rel} & {es_delta} \\\\",
        f"BF                               & {bf_rhat} & {bf_rel} & {bf_delta} \\\\",
        f"QAOA (simulator)                 & {sim_rhat} & {sim_rel} & {sim_delta} \\\\",
        r"QAOA (real device)               & \textit{--} & \textit{--} & \textit{--} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Amino poster experiment pipeline.")
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
        "--seed",
        type=int,
        default=None,
        help="Run only one random seed for ALS/QUBO solves.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated run seeds. Ignored when --seed is set.",
    )
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--R-over", type=int, default=20, dest="R_over")
    parser.add_argument("--als-sweeps", type=int, default=200)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--mu-values", type=str, default="0.75")
    parser.add_argument("--plot-mu", type=float, default=0.75)
    parser.add_argument(
        "--mu-effect",
        "--mu-scale",
        dest="mu_effect",
        type=str,
        default="xnorm2_over_rank",
        choices=("none", "xnorm", "xnorm2", "xnorm2_over_rank"),
        help=(
            "Scale mu to match data magnitude. "
            "'xnorm2_over_rank' treats mu as a per-component penalty relative to ||X||_F^2."
        ),
    )
    parser.add_argument("--qaoa-p", type=int, default=2)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--qaoa-maxiter", type=int, default=40)
    parser.add_argument(
        "--qaoa-optimizer",
        type=str,
        default="COBYLA",
        help="Optimizer method passed to scipy.optimize.minimize.",
    )
    parser.add_argument("--bf-max-evals", type=int, default=40)
    parser.add_argument("--bf-restarts", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="script/poster/amino_result")
    parser.add_argument(
        "--figure-path",
        type=str,
        default="script/poster/amino_result/amino_convergence_qaoa_bf.png",
    )
    parser.add_argument(
        "--qaoa-loss-figure-path",
        type=str,
        default="script/poster/amino_result/amino_qaoa_loss_history.png",
    )
    parser.add_argument(
        "--als-figure-path",
        type=str,
        default="script/poster/amino_result/amino_als_relerr_history.png",
    )
    parser.add_argument(
        "--tex-table-path",
        type=str,
        default="script/poster/amino_result/amino_metrics.tex",
        help="Path to write the filled LaTeX metrics table (for --plot-mu).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    synth = _load_synth_module()
    dtype = synth._dtype_from_str(args.dtype)
    mu_values = synth._parse_list_float(args.mu_values)
    if len(mu_values) == 0:
        raise SystemExit("mu_values cannot be empty.")
    plot_mu = synth._resolve_plot_mu(args.plot_mu, mu_values)

    data_paths = _resolve_data_paths(args)
    if not data_paths:
        raise SystemExit("No data files found. Use --data-path or --data-glob.")

    run_seeds = _resolve_run_seeds(args)
    if len(run_seeds) == 0:
        raise SystemExit("No run seeds selected.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = Path(args.figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    qaoa_loss_figure_path = Path(args.qaoa_loss_figure_path)
    qaoa_loss_figure_path.parent.mkdir(parents=True, exist_ok=True)
    als_figure_path = Path(args.als_figure_path)
    als_figure_path.parent.mkdir(parents=True, exist_ok=True)
    tex_table_path = Path(args.tex_table_path)
    tex_table_path.parent.mkdir(parents=True, exist_ok=True)

    run_config = {
        "script": str(Path(__file__).resolve()),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": list(sys.argv),
        "args": dict(vars(args)),
        "resolved": {
            "data_paths": [str(p) for p in data_paths],
            "run_seeds": [int(s) for s in run_seeds],
            "mu_values": [float(mu) for mu in mu_values],
            "plot_mu": float(plot_mu),
        },
    }
    run_config_path = out_dir / "amino_run_config.json"
    run_config_path.write_text(json.dumps(run_config, indent=2) + "\n")
    print(f"[info] wrote run config to {run_config_path}")

    print(f"[info] loaded {len(data_paths)} amino tensor file(s).")
    print(
        f"[info] config R_over={args.R_over} als_sweeps={args.als_sweeps} "
        f"bf_max_evals={args.bf_max_evals} qaoa_maxiter={args.qaoa_maxiter} "
        f"num_run_seeds={len(run_seeds)} mu_effect={args.mu_effect}"
    )

    if int(args.R_over) > 20:
        raise SystemExit(
            "This workflow requires exhaustive search for every seed/mu; "
            "please set --R-over <= 20."
        )

    solvers = ("ES", "BF", "QAOA")
    metrics_by_mu: Dict[float, Dict[str, Dict[str, List[float]]]] = {}
    for mu in mu_values:
        metrics_by_mu[mu] = {
            solver: {"Rhat": [], "relerr": [], "energy": [], "gap": [], "delta": []}
            for solver in solvers
        }

    bf_plot_traces: List[np.ndarray] = []
    qaoa_plot_traces: List[np.ndarray] = []
    qaoa_loss_plot_traces: List[np.ndarray] = []
    qaoa_iter_plot_traces: List[np.ndarray] = []
    es_plot_values: List[float] = []
    als_plot_traces: List[np.ndarray] = []
    als_trace_rows: List[Dict[str, object]] = []
    processed_runs = 0

    for data_idx, data_path in enumerate(data_paths, start=1):
        if not data_path.exists():
            print(f"[warn] missing dataset file: {data_path}")
            continue

        payload = torch.load(data_path, map_location="cpu")
        if "X" not in payload:
            print(f"[warn] payload has no key 'X': {data_path}")
            continue

        X_base = payload["X"].to(device=args.device, dtype=dtype)
        cfg = payload.get("config", {})
        shape = tuple(cfg.get("shape", list(X_base.shape)))
        rank_true = int(cfg.get("rank_true", 0))
        noise_std = float(cfg.get("noise_std", 0.0))
        corr_rho = float(cfg.get("corr_rho", 0.0))
        dataset_seed = int(payload.get("seed", 0))

        print(
            f"[info] data={data_idx}/{len(data_paths)} path={data_path} "
            f"shape={shape} rank_true={rank_true} noise={noise_std} rho={corr_rho} "
            f"data_seed={dataset_seed}"
        )

        for seed_idx, run_seed in enumerate(run_seeds, start=1):
            X = X_base
            xnorm = float(torch.linalg.norm(X).clamp_min(1e-12).item())
            label = (
                f"data={data_idx}/{len(data_paths)} run_seed={run_seed} "
                f"({seed_idx}/{len(run_seeds)}) path={data_path.name}"
            )
            print(f"[als] start {label}")
            factors, weights, als_rel_error, als_time_s, als_relerr_trace = synth.run_overfactored_als(
                X=X,
                rank_over=args.R_over,
                sweeps=args.als_sweeps,
                ridge=args.ridge,
                seed=run_seed,
                log_every=args.log_every,
                verbose=args.verbose,
                run_label=label,
            )
            for sweep_idx, relerr in enumerate(als_relerr_trace.tolist(), start=1):
                als_trace_rows.append(
                    {
                        "dataset_path": str(data_path),
                        "data_seed": int(dataset_seed),
                        "run_seed": int(run_seed),
                        "sweep": int(sweep_idx),
                        "rel_error": float(relerr),
                        "rank_over": int(args.R_over),
                        "als_sweeps_requested": int(args.als_sweeps),
                        "als_time_s": float(als_time_s),
                    }
                )
            als_plot_traces.append(np.asarray(als_relerr_trace, dtype=np.float64))
            factors_w = synth._absorb_weights_into_factors(factors, weights)
            weighted_components, Q_base, q_base = synth.build_qubo_from_factors(X, factors_w)
            print(
                f"[als] done  {label} rel_error={als_rel_error:.6e} "
                f"time_s={als_time_s:.2f} rank={args.R_over}"
            )
            processed_runs += 1

            for mu_idx, mu in enumerate(mu_values):
                mu_effective = synth._scale_mu(mu, args.mu_effect, xnorm=xnorm, rank_ref=args.R_over)
                q_raw = q_base + float(mu_effective) * np.ones_like(q_base)
                Q, q = synth._prepare_qubo_inputs(Q_base, q_raw)
                base_seed = (
                    int(run_seed)
                    + 100_003 * int(mu_idx + 1)
                    + 10_000_019 * int(data_idx)
                )
                shared_solver_seed = int(base_seed % 65536)

                es_mask, es_energy = synth.exhaustive_search_qubo(Q, q)
                es_energy = synth.qubo_energy(es_mask, Q, q)
                es_metrics = synth.evaluate_mask_metrics(
                    X=X,
                    X_norm=xnorm,
                    weighted_components=weighted_components,
                    mask=es_mask,
                    energy=es_energy,
                    es_energy=es_energy,
                )
                for key, val in es_metrics.items():
                    metrics_by_mu[mu]["ES"][key].append(val)

                bf_mask, bf_energy, _bf_best_trace, bf_loss_trace = synth.bitflip_search_qubo(
                    Q=Q,
                    q=q,
                    max_evals=args.bf_max_evals,
                    restarts=args.bf_restarts,
                    seed=shared_solver_seed,
                )
                bf_energy_eval = synth.qubo_energy(bf_mask, Q, q)
                if abs(float(bf_energy) - float(bf_energy_eval)) > 1e-9:
                    print(
                        f"[warn] BF energy mismatch detected; using canonical value "
                        f"{bf_energy_eval:.6e} (solver reported {bf_energy:.6e})."
                    )
                bf_energy = bf_energy_eval
                bf_metrics = synth.evaluate_mask_metrics(
                    X=X,
                    X_norm=xnorm,
                    weighted_components=weighted_components,
                    mask=bf_mask,
                    energy=bf_energy,
                    es_energy=es_energy,
                )
                for key, val in bf_metrics.items():
                    metrics_by_mu[mu]["BF"][key].append(val)

                qaoa_mask, qaoa_energy, qaoa_trace, qaoa_loss_trace, qaoa_iter_trace = synth.qaoa_search_qubo(
                    Q=Q,
                    q=q,
                    qaoa_p=args.qaoa_p,
                    shots=args.shots,
                    maxiter=args.qaoa_maxiter,
                    seed=shared_solver_seed,
                    optimizer=args.qaoa_optimizer,
                )
                qaoa_energy_eval = synth.qubo_energy(qaoa_mask, Q, q)
                if abs(float(qaoa_energy) - float(qaoa_energy_eval)) > 1e-9:
                    print(
                        f"[warn] QAOA energy mismatch detected; using canonical value "
                        f"{qaoa_energy_eval:.6e} (solver reported {qaoa_energy:.6e})."
                    )
                qaoa_energy = qaoa_energy_eval
                qaoa_metrics = synth.evaluate_mask_metrics(
                    X=X,
                    X_norm=xnorm,
                    weighted_components=weighted_components,
                    mask=qaoa_mask,
                    energy=qaoa_energy,
                    es_energy=es_energy,
                )
                for key, val in qaoa_metrics.items():
                    metrics_by_mu[mu]["QAOA"][key].append(val)

                print(
                    f"[seed] data={data_path.name} run_seed={run_seed} mu={mu} method=ES "
                    f"mu_eff={mu_effective:.6e} "
                    f"R_hat={es_metrics['Rhat']:.0f} rel_err={es_metrics['relerr']:.6e} "
                    f"delta_bar={es_metrics['delta']:.6e}"
                )
                print(
                    f"[seed] data={data_path.name} run_seed={run_seed} mu={mu} method=BF "
                    f"mu_eff={mu_effective:.6e} "
                    f"R_hat={bf_metrics['Rhat']:.0f} rel_err={bf_metrics['relerr']:.6e} "
                    f"delta_bar={bf_metrics['delta']:.6e}"
                )
                print(
                    f"[seed] data={data_path.name} run_seed={run_seed} mu={mu} method=QAOA "
                    f"mu_eff={mu_effective:.6e} "
                    f"R_hat={qaoa_metrics['Rhat']:.0f} rel_err={qaoa_metrics['relerr']:.6e} "
                    f"delta_bar={qaoa_metrics['delta']:.6e}"
                )

                if synth._is_close(mu, plot_mu):
                    es_plot_values.append(float(es_energy))
                    bf_y = synth._pad_trace(
                        np.asarray(bf_loss_trace, dtype=np.float64),
                        int(args.bf_max_evals),
                    )
                    qaoa_y = synth._pad_trace(
                        np.asarray(qaoa_trace, dtype=np.float64),
                        int(args.qaoa_maxiter),
                    )
                    qaoa_loss_y = synth._pad_trace(
                        np.asarray(qaoa_loss_trace, dtype=np.float64),
                        int(args.qaoa_maxiter),
                    )
                    qaoa_iter_y = synth._pad_trace(
                        np.asarray(qaoa_iter_trace, dtype=np.float64),
                        int(args.qaoa_maxiter),
                    )
                    bf_plot_traces.append(bf_y)
                    qaoa_plot_traces.append(qaoa_y)
                    qaoa_loss_plot_traces.append(qaoa_loss_y)
                    qaoa_iter_plot_traces.append(qaoa_iter_y)

    if processed_runs == 0:
        raise SystemExit("No valid amino runs were processed.")

    als_trace_path = out_dir / "amino_als_relerr_history.csv"
    pd.DataFrame(als_trace_rows).to_csv(als_trace_path, index=False)
    print(f"[info] wrote ALS relative-error history to {als_trace_path}")
    if als_plot_traces:
        als_mat = np.vstack(als_plot_traces)
        als_png, als_pdf = synth.plot_als_relerr_history(als_mat, als_figure_path)
        print(f"[info] wrote ALS convergence figures to {als_png} and {als_pdf}")

    for mu in mu_values:
        rows = [
            synth._metric_row("ES", metrics_by_mu[mu]["ES"]),
            synth._metric_row("BF", metrics_by_mu[mu]["BF"]),
            synth._metric_row("QAOA", metrics_by_mu[mu]["QAOA"]),
        ]
        metrics_path = out_dir / f"amino_metrics_mu={synth._format_mu(mu)}.csv"
        pd.DataFrame(rows).to_csv(metrics_path, index=False)
        print(f"[info] wrote metrics to {metrics_path}")

        if synth._is_close(mu, plot_mu):
            write_latex_table(rows, tex_table_path)
            print(f"[info] wrote latex table to {tex_table_path}")

        for row in rows:
            print(
                f"[summary] mu={mu} method={row['solver']} "
                f"R_hat={row['mean_Rhat']:.4f}±{row['std_Rhat']:.4f} "
                f"rel_err={row['mean_relerr']:.6e}±{row['std_relerr']:.6e} "
                f"delta_bar={row['mean_delta']:.6e}±{row['std_delta']:.6e}"
            )

    if bf_plot_traces and qaoa_plot_traces:
        bf_mat = np.vstack(bf_plot_traces)
        qaoa_mat = np.vstack(qaoa_plot_traces)
        bf_mean = np.nanmean(bf_mat, axis=0)
        bf_std = np.nanstd(bf_mat, axis=0)
        qaoa_mean = np.nanmean(qaoa_mat, axis=0)
        qaoa_std = np.nanstd(qaoa_mat, axis=0)
    else:
        bf_mean = np.zeros((int(args.bf_max_evals),), dtype=np.float64)
        bf_std = np.zeros((int(args.bf_max_evals),), dtype=np.float64)
        qaoa_mean = np.zeros((int(args.qaoa_maxiter),), dtype=np.float64)
        qaoa_std = np.zeros((int(args.qaoa_maxiter),), dtype=np.float64)

    if qaoa_loss_plot_traces:
        qaoa_loss_mat = np.vstack(qaoa_loss_plot_traces)
        qaoa_loss_mean = np.nanmean(qaoa_loss_mat, axis=0)
        qaoa_loss_std = np.nanstd(qaoa_loss_mat, axis=0)
    else:
        qaoa_loss_mat = np.zeros((0, int(args.qaoa_maxiter)), dtype=np.float64)
        qaoa_loss_mean = np.zeros((int(args.qaoa_maxiter),), dtype=np.float64)
        qaoa_loss_std = np.zeros((int(args.qaoa_maxiter),), dtype=np.float64)

    if qaoa_iter_plot_traces:
        qaoa_iter_mat = np.vstack(qaoa_iter_plot_traces)
        qaoa_iter_mean = np.nanmean(qaoa_iter_mat, axis=0)
        qaoa_iter_std = np.nanstd(qaoa_iter_mat, axis=0)
    else:
        qaoa_iter_mat = np.zeros((0, int(args.qaoa_maxiter)), dtype=np.float64)
        qaoa_iter_mean = np.zeros((int(args.qaoa_maxiter),), dtype=np.float64)
        qaoa_iter_std = np.zeros((int(args.qaoa_maxiter),), dtype=np.float64)

    es_min_value_plot = (
        float(np.nanmean(np.asarray(es_plot_values, dtype=np.float64)))
        if len(es_plot_values) > 0
        else None
    )

    convergence_path = out_dir / f"amino_convergence_mu={synth._format_mu(plot_mu)}.npz"
    np.savez(
        convergence_path,
        mu=float(plot_mu),
        use_gap=np.array([1], dtype=np.int64),
        bf_x=np.arange(1, len(bf_mean) + 1, dtype=np.int64),
        bf_mean=bf_mean,
        bf_std=bf_std,
        qaoa_x=np.arange(1, len(qaoa_mean) + 1, dtype=np.int64),
        qaoa_mean=qaoa_mean,
        qaoa_std=qaoa_std,
        qaoa_loss_x=np.arange(1, len(qaoa_loss_mean) + 1, dtype=np.int64),
        qaoa_loss_mean=qaoa_loss_mean,
        qaoa_loss_std=qaoa_loss_std,
        qaoa_iter_x=np.arange(1, len(qaoa_iter_mean) + 1, dtype=np.int64),
        qaoa_iter_mean=qaoa_iter_mean,
        qaoa_iter_std=qaoa_iter_std,
        es_min_value=np.array(
            [es_min_value_plot if es_min_value_plot is not None else np.nan],
            dtype=np.float64,
        ),
    )
    print(f"[info] wrote convergence traces to {convergence_path}")

    synth.plot_convergence(
        bf_mean=bf_mean,
        bf_std=bf_std,
        qaoa_mean=qaoa_mean,
        qaoa_std=qaoa_std,
        use_gap=True,
        es_min_value=es_min_value_plot,
        output_path=figure_path,
        show_zoom=False,
    )
    print(f"[info] wrote convergence figure to {figure_path}")

    synth.plot_qaoa_loss_history(
        qaoa_loss_mat=qaoa_loss_mat,
        qaoa_loss_mean=qaoa_loss_mean,
        qaoa_loss_std=qaoa_loss_std,
        output_path=qaoa_loss_figure_path,
    )
    print(f"[info] wrote QAOA loss figure to {qaoa_loss_figure_path}")


if __name__ == "__main__":
    main()
