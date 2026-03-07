#!/usr/bin/env python3
"""
Plot amino poster outputs from saved experiment artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _format_mu(mu: float) -> str:
    return f"{float(mu):.6g}"


def _parse_figsize(raw: str) -> Tuple[float, float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("--figsize must be formatted as 'W,H', e.g. '3,2.5'.")
    return float(parts[0]), float(parts[1])


def _extract_mu_from_name(path: Path, prefix: str, suffix: str) -> Optional[float]:
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    middle = name[len(prefix) : len(name) - len(suffix)]
    try:
        return float(middle)
    except ValueError:
        return None


def _resolve_artifact_path(
    explicit_path: Optional[str],
    directory: Path,
    prefix: str,
    suffix: str,
    mu: Optional[float],
) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path

    candidates = sorted(directory.glob(f"{prefix}*{suffix}"))
    if not candidates:
        raise FileNotFoundError(f"No files found for pattern: {directory}/{prefix}*{suffix}")

    if mu is not None:
        target = directory / f"{prefix}{_format_mu(mu)}{suffix}"
        if target.exists():
            return target
        parsed = [(p, _extract_mu_from_name(p, prefix, suffix)) for p in candidates]
        parsed = [(p, m) for p, m in parsed if m is not None]
        if parsed:
            nearest = min(parsed, key=lambda pm: abs(float(pm[1]) - float(mu)))
            print(
                f"[warn] exact file for mu={mu} not found; using nearest mu={nearest[1]} "
                f"at {nearest[0]}"
            )
            return nearest[0]
        raise FileNotFoundError(f"Could not resolve file for mu={mu} in {directory}")

    if len(candidates) == 1:
        return candidates[0]

    raise SystemExit(
        "Multiple candidate files found. Please specify --mu or explicit file path.\n"
        + "\n".join(str(p) for p in candidates)
    )


def _fmt_latex_pm(mean: float, std: float, sci: bool = False) -> str:
    if (not np.isfinite(mean)) or (not np.isfinite(std)):
        return r"\textit{--}"
    if sci:
        return f"{mean:.3e} $\\pm$ {std:.3e}"
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def _paired_figure_paths(output_path: Path) -> Tuple[Path, Path]:
    base = output_path.with_suffix("") if output_path.suffix else output_path
    return base.with_suffix(".png"), base.with_suffix(".pdf")


def _pad_trace(trace: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(trace, dtype=np.float64).reshape(-1)
    target_len = int(target_len)
    if target_len <= 0:
        return np.zeros((0,), dtype=np.float64)
    if arr.size == 0:
        return np.full((target_len,), np.nan, dtype=np.float64)
    if arr.size >= target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - arr.size), mode="edge")


def write_latex_table(rows: Sequence[Dict[str, float]], output_path: Path) -> None:
    by_method = {str(r["solver"]): r for r in rows}

    es_row = by_method.get("ES")
    es_energy_mean = float(es_row.get("mean_energy", float("nan"))) if es_row is not None else float("nan")

    def _row_values(solver_key: str, is_es: bool = False) -> Tuple[str, str, str]:
        row = by_method.get(solver_key)
        required = ("mean_Rhat", "std_Rhat", "mean_relerr", "std_relerr")
        if row is None or any(key not in row for key in required):
            return r"\textit{--}", r"\textit{--}", "0" if is_es else r"\textit{--}"
        rhat = _fmt_latex_pm(float(row["mean_Rhat"]), float(row["std_Rhat"]), sci=False)
        rel = _fmt_latex_pm(float(row["mean_relerr"]), float(row["std_relerr"]), sci=True)
        if is_es:
            delta = "0"
        else:
            mean_delta = float(row.get("mean_delta", float("nan")))
            std_delta = float(row.get("std_delta", float("nan")))
            if (not np.isfinite(mean_delta)) and np.isfinite(es_energy_mean):
                mean_gap = float(row.get("mean_gap", float("nan")))
                std_gap = float(row.get("std_gap", float("nan")))
                denom = max(abs(es_energy_mean), 1e-12)
                mean_delta = abs(mean_gap) / denom if np.isfinite(mean_gap) else float("nan")
                std_delta = abs(std_gap) / denom if np.isfinite(std_gap) else float("nan")
            delta = _fmt_latex_pm(mean_delta, std_delta, sci=True)
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


def _compute_xticks(x_max: int) -> np.ndarray:
    x_max = max(int(x_max), 1)
    if x_max <= 10:
        return np.arange(1, x_max + 1, dtype=int)

    ticks: List[int] = [1]
    ticks.extend(range(10, x_max + 1, 10))
    if ticks[-1] != x_max:
        ticks.append(x_max)
    return np.asarray(ticks, dtype=int)


def _marker_indices_for_ticks(x_values: np.ndarray, x_ticks: np.ndarray) -> List[int]:
    x_vals = np.asarray(x_values, dtype=np.int64).reshape(-1)
    ticks = np.asarray(x_ticks, dtype=np.int64).reshape(-1)
    if x_vals.size == 0 or ticks.size == 0:
        return []
    return np.flatnonzero(np.isin(x_vals, ticks)).astype(int).tolist()


def _endpoint_nudge(x_max: int) -> float:
    span = max(int(x_max) - 1, 0)
    if span <= 0:
        return 0.0
    return min(0.9, 0.3 * span)


def _nudge_endpoint_positions(x: np.ndarray, x_max: int) -> np.ndarray:
    x_plot = np.asarray(x, dtype=np.float64).reshape(-1).copy()
    if x_plot.size == 0:
        return x_plot

    delta = _endpoint_nudge(x_max)
    if delta <= 0.0:
        return x_plot

    if np.isclose(x_plot[0], 1.0):
        x_plot[0] += delta
    if x_plot.size >= 2 and np.isclose(x_plot[-1], float(x_max)):
        x_plot[-1] -= delta
    return x_plot


def _set_inset_endpoint_ticks(ax, x_max: int) -> None:
    x_ticks = _compute_xticks(x_max)
    tick_pos = x_ticks.astype(np.float64)

    delta = _endpoint_nudge(x_max)
    if delta > 0.0 and tick_pos.size > 0:
        left_idx = np.where(np.isclose(tick_pos, 1.0))[0]
        if left_idx.size > 0:
            tick_pos[left_idx[0]] += delta

        right_idx = np.where(np.isclose(tick_pos, float(x_max)))[0]
        if right_idx.size > 0:
            tick_pos[right_idx[0]] -= delta

    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(int(v)) for v in x_ticks])


def plot_convergence(
    bf_x: np.ndarray,
    bf_mean: np.ndarray,
    bf_std: np.ndarray,
    qaoa_x: np.ndarray,
    qaoa_mean: np.ndarray,
    qaoa_std: np.ndarray,
    es_min_value: Optional[float],
    output_path: Path,
    figsize: Tuple[float, float],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bf_x = np.asarray(bf_x, dtype=np.int64).reshape(-1)
    bf_mean = np.asarray(bf_mean, dtype=np.float64).reshape(-1)
    bf_std = np.asarray(bf_std, dtype=np.float64).reshape(-1)
    qaoa_x = np.asarray(qaoa_x, dtype=np.int64).reshape(-1)
    qaoa_mean = np.asarray(qaoa_mean, dtype=np.float64).reshape(-1)
    qaoa_std = np.asarray(qaoa_std, dtype=np.float64).reshape(-1)
    x_max = int(
        max(
            int(np.max(bf_x)) if bf_x.size else 1,
            int(np.max(qaoa_x)) if qaoa_x.size else 1,
            1,
        )
    )
    x_ticks = _compute_xticks(x_max)
    bf_x_plot = _nudge_endpoint_positions(bf_x, x_max)
    qaoa_x_plot = _nudge_endpoint_positions(qaoa_x, x_max)

    fig, ax = plt.subplots(figsize=figsize)
    bf_markevery = _marker_indices_for_ticks(bf_x, x_ticks)
    qaoa_markevery = _marker_indices_for_ticks(qaoa_x, x_ticks)
    bf_line, = ax.plot(
        bf_x_plot,
        bf_mean,
        color="#1f77b4",
        linewidth=1.5,
        marker="o",
        markersize=3.0,
        markevery=bf_markevery,
        label="BF",
    )

    qaoa_line, = ax.plot(
        qaoa_x_plot,
        qaoa_mean,
        color="#d62728",
        linewidth=1.5,
        marker="s",
        markersize=3.0,
        markevery=qaoa_markevery,
        label="QAOA",
    )

    ax.set_xlim(1, x_max)
    _set_inset_endpoint_ticks(ax, x_max)

    es_line = None
    if es_min_value is not None and np.isfinite(es_min_value):
        x_es = np.arange(1, x_max + 1, dtype=np.int64)
        x_es_plot = _nudge_endpoint_positions(x_es, x_max)
        es_markevery = _marker_indices_for_ticks(x_es, x_ticks)
        es_line, = ax.plot(
            x_es_plot,
            np.full((len(x_es_plot),), float(es_min_value), dtype=np.float64),
            color="black",
            linestyle="--",
            linewidth=1.0,
            marker="*",
            markersize=6.0,
            markevery=es_markevery,
            label="ES",
        )

    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("Objective", fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="x", alpha=0.3)

    if es_line is not None:
        ax.legend(handles=[es_line, bf_line, qaoa_line], frameon=False, fontsize=9)
    else:
        ax.legend(handles=[bf_line, qaoa_line], frameon=False, fontsize=9)

    fig.tight_layout()
    png_path, pdf_path = _paired_figure_paths(output_path)
    fig.savefig(png_path, bbox_inches="tight", dpi=int(dpi))
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_qaoa_loss_history(
    qaoa_loss_x: np.ndarray,
    qaoa_loss_mean: np.ndarray,
    qaoa_loss_std: np.ndarray,
    output_path: Path,
    figsize: Tuple[float, float],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(qaoa_loss_x, dtype=np.int64).reshape(-1)
    mean = np.asarray(qaoa_loss_mean, dtype=np.float64).reshape(-1)
    std = np.asarray(qaoa_loss_std, dtype=np.float64).reshape(-1)
    x_max = int(max(int(np.max(x)) if x.size else 1, 1))
    x_plot = _nudge_endpoint_positions(x, x_max)
    markevery = max(1, len(x) // 8)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        x_plot,
        mean,
        color="#d62728",
        linewidth=1.5,
        marker="^",
        markersize=3.0,
        markevery=markevery,
        label="QAOA loss",
    )
    valid = np.isfinite(mean) & np.isfinite(std)
    if np.any(valid):
        xv = x_plot[valid]
        ax.fill_between(
            xv,
            mean[valid] - std[valid],
            mean[valid] + std[valid],
            color="#d62728",
            alpha=0.12,
        )

    ax.set_xlim(1, x_max)
    _set_inset_endpoint_ticks(ax, x_max)
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("QAOA Loss", fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    png_path, pdf_path = _paired_figure_paths(output_path)
    fig.savefig(png_path, bbox_inches="tight", dpi=int(dpi))
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_als_relerr_history(
    als_relerr_mat: np.ndarray,
    output_path: Path,
    figsize: Tuple[float, float],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    mat = np.asarray(als_relerr_mat, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] == 0:
        raise ValueError("als_relerr_mat must be a non-empty 2D array.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, mat.shape[1] + 1, dtype=np.int64)
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    x_max = int(max(len(x), 1))
    x_plot = _nudge_endpoint_positions(x, x_max)
    markevery = max(1, len(x) // 8)

    fig, ax = plt.subplots(figsize=figsize)
    for row in mat:
        ax.plot(x_plot, row, color="#aaaaaa", linewidth=0.7, alpha=0.35)
    ax.plot(
        x_plot,
        mean,
        color="#2ca02c",
        linewidth=1.5,
        marker="d",
        markersize=3.0,
        markevery=markevery,
        label="ALS rel. error",
    )
    valid = np.isfinite(mean) & np.isfinite(std)
    if np.any(valid):
        xv = x_plot[valid]
        ax.fill_between(xv, mean[valid] - std[valid], mean[valid] + std[valid], color="#2ca02c", alpha=0.12)

    ax.set_xlim(1, x_max)
    _set_inset_endpoint_ticks(ax, x_max)
    ax.set_xlabel("ALS Sweep", fontsize=9)
    ax.set_ylabel("Relative Error", fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    png_path, pdf_path = _paired_figure_paths(output_path)
    fig.savefig(png_path, bbox_inches="tight", dpi=int(dpi))
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot amino poster outputs from saved artifacts.")
    parser.add_argument("--in-dir", type=str, default="script/poster/amino_result")
    parser.add_argument("--mu", type=float, default=None, help="Target mu for selecting artifacts.")
    parser.add_argument("--convergence-npz", type=str, default=None)
    parser.add_argument("--metrics-csv", type=str, default=None)
    parser.add_argument("--als-history-csv", type=str, default=None)
    parser.add_argument("--figure-path", type=str, default="script/poster/amino_fig/amino_convergence_qaoa_bf.png")
    parser.add_argument("--qaoa-loss-figure-path", type=str, default="script/poster/amino_fig/amino_qaoa_loss_history.png")
    parser.add_argument("--als-figure-path", type=str, default="script/poster/amino_fig/amino_als_relerr_history.png")
    parser.add_argument("--tex-table-path", type=str, default="script/poster/amino_fig/amino_metrics.tex")
    parser.add_argument("--figsize", type=str, default="3,2.5")
    parser.add_argument("--dpi", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    figsize = _parse_figsize(args.figsize)

    convergence_path = _resolve_artifact_path(
        explicit_path=args.convergence_npz,
        directory=in_dir,
        prefix="amino_convergence_mu=",
        suffix=".npz",
        mu=args.mu,
    )
    metrics_path = _resolve_artifact_path(
        explicit_path=args.metrics_csv,
        directory=in_dir,
        prefix="amino_metrics_mu=",
        suffix=".csv",
        mu=args.mu,
    )

    conv = np.load(convergence_path)

    bf_x = conv["bf_x"] if "bf_x" in conv else np.arange(1, len(conv["bf_mean"]) + 1, dtype=np.int64)
    bf_std = conv["bf_std"] if "bf_std" in conv else np.zeros_like(conv["bf_mean"], dtype=np.float64)
    qaoa_x = conv["qaoa_x"] if "qaoa_x" in conv else np.arange(1, len(conv["qaoa_mean"]) + 1, dtype=np.int64)
    qaoa_std = conv["qaoa_std"] if "qaoa_std" in conv else np.zeros_like(conv["qaoa_mean"], dtype=np.float64)
    qaoa_loss_x = conv["qaoa_loss_x"] if "qaoa_loss_x" in conv else qaoa_x

    es_min_value = None
    if "es_min_value" in conv:
        arr = np.asarray(conv["es_min_value"], dtype=np.float64).reshape(-1)
        if arr.size > 0 and np.isfinite(arr[0]):
            es_min_value = float(arr[0])

    plot_convergence(
        bf_x=bf_x,
        bf_mean=conv["bf_mean"],
        bf_std=bf_std,
        qaoa_x=qaoa_x,
        qaoa_mean=conv["qaoa_mean"],
        qaoa_std=qaoa_std,
        es_min_value=es_min_value,
        output_path=Path(args.figure_path),
        figsize=figsize,
        dpi=args.dpi,
    )
    conv_png, conv_pdf = _paired_figure_paths(Path(args.figure_path))
    print(f"[info] wrote convergence figures to {conv_png} and {conv_pdf}")

    if "qaoa_loss_mean" in conv and "qaoa_loss_std" in conv:
        plot_qaoa_loss_history(
            qaoa_loss_x=qaoa_loss_x,
            qaoa_loss_mean=conv["qaoa_loss_mean"],
            qaoa_loss_std=conv["qaoa_loss_std"],
            output_path=Path(args.qaoa_loss_figure_path),
            figsize=figsize,
            dpi=args.dpi,
        )
        loss_png, loss_pdf = _paired_figure_paths(Path(args.qaoa_loss_figure_path))
        print(f"[info] wrote QAOA loss figures to {loss_png} and {loss_pdf}")
    else:
        print("[warn] qaoa_loss_mean/std not found in NPZ; skipped QAOA loss plot.")

    als_history_path = Path(args.als_history_csv) if args.als_history_csv else (in_dir / "amino_als_relerr_history.csv")
    if als_history_path.exists():
        als_df = pd.read_csv(als_history_path)
        required_als = {"sweep", "rel_error"}
        missing_als = sorted(required_als - set(als_df.columns))
        if missing_als:
            print(f"[warn] ALS history CSV missing columns {missing_als}; skipped ALS plot.")
        else:
            max_sweep = int(np.nanmax(np.asarray(als_df["sweep"], dtype=np.float64)))
            key_cols = [c for c in ("dataset_path", "data_seed", "run_seed") if c in als_df.columns]
            traces: List[np.ndarray] = []
            if key_cols:
                grouped = als_df.groupby(key_cols, sort=False, dropna=False)
                for _, sub in grouped:
                    vals = np.asarray(sub.sort_values("sweep")["rel_error"], dtype=np.float64)
                    traces.append(_pad_trace(vals, max_sweep))
            else:
                vals = np.asarray(als_df.sort_values("sweep")["rel_error"], dtype=np.float64)
                traces.append(_pad_trace(vals, max_sweep))
            if traces:
                als_mat = np.vstack(traces)
                plot_als_relerr_history(
                    als_relerr_mat=als_mat,
                    output_path=Path(args.als_figure_path),
                    figsize=figsize,
                    dpi=args.dpi,
                )
                als_png, als_pdf = _paired_figure_paths(Path(args.als_figure_path))
                print(f"[info] wrote ALS convergence figures to {als_png} and {als_pdf}")
    else:
        print(f"[warn] ALS history CSV not found: {als_history_path}; skipped ALS plot.")

    df = pd.read_csv(metrics_path)
    needed = {"solver", "mean_Rhat", "std_Rhat", "mean_relerr", "std_relerr"}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"metrics CSV is missing columns: {missing}")

    rows = []
    for method in ("ES", "BF", "QAOA"):
        sub = df[df["solver"].astype(str) == method]
        if sub.empty:
            rows.append({"solver": method})
        else:
            rows.append(sub.iloc[0].to_dict())
    write_latex_table(rows, Path(args.tex_table_path))
    print(f"[info] wrote LaTeX table to {args.tex_table_path}")


if __name__ == "__main__":
    main()
