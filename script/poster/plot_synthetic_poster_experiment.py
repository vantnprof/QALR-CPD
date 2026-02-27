#!/usr/bin/env python3
"""
Plot synthetic poster outputs from saved experiment artifacts.
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


def write_latex_table(rows: Sequence[Dict[str, float]], output_path: Path) -> None:
    by_method = {str(r["solver"]): r for r in rows}

    def _row_line(method: str) -> str:
        row = by_method.get(method)
        if row is None:
            return f"{method} & \\textit{{--}} & \\textit{{--}} & \\textit{{--}} & \\textit{{--}} \\\\"
        rhat = _fmt_latex_pm(float(row["mean_Rhat"]), float(row["std_Rhat"]), sci=False)
        rel = _fmt_latex_pm(float(row["mean_relerr"]), float(row["std_relerr"]), sci=True)
        energy = _fmt_latex_pm(float(row["mean_energy"]), float(row["std_energy"]), sci=True)
        gap = "0" if method == "ES" else _fmt_latex_pm(float(row["mean_gap"]), float(row["std_gap"]), sci=True)
        return f"{method} & {rhat} & {rel} & {energy} & {gap} \\\\"

    lines = [
        r"\begin{table}[t]",
        r"\caption{Result on synthetic dataset.}",
        r"\label{tab:synth_metrics}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Method & \(\hat R\) & Rel.\ error \(\downarrow\) & \(E(\mathbf{z})\) \(\downarrow\) & Gap to ES \(\downarrow\) \\",
        r"\midrule",
        _row_line("ES"),
        _row_line("BF"),
        _row_line("QAOA"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def _compute_xticks(x_max: int) -> np.ndarray:
    if x_max <= 10:
        return np.arange(1, x_max + 1, dtype=int)
    return np.unique(np.linspace(1, x_max, num=5, dtype=int))


def plot_convergence(
    bf_x: np.ndarray,
    bf_mean: np.ndarray,
    qaoa_x: np.ndarray,
    qaoa_mean: np.ndarray,
    es_min_value: Optional[float],
    output_path: Path,
    figsize: Tuple[float, float],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bf_x = np.asarray(bf_x, dtype=np.int64).reshape(-1)
    bf_mean = np.asarray(bf_mean, dtype=np.float64).reshape(-1)
    qaoa_x = np.asarray(qaoa_x, dtype=np.int64).reshape(-1)
    qaoa_mean = np.asarray(qaoa_mean, dtype=np.float64).reshape(-1)

    fig, ax = plt.subplots(figsize=figsize)
    bf_markevery = max(1, len(bf_x) // 8)
    qaoa_markevery = max(1, len(qaoa_x) // 8)
    bf_line, = ax.plot(
        bf_x,
        bf_mean,
        color="#1f77b4",
        linewidth=1.5,
        marker="o",
        markersize=3.0,
        markevery=bf_markevery,
        label="BF",
    )
    bf_valid = np.isfinite(bf_mean) & np.isfinite(bf_std)
    if np.any(bf_valid):
        xb = bf_x[bf_valid]
        ax.fill_between(
            xb,
            bf_mean[bf_valid] - bf_std[bf_valid],
            bf_mean[bf_valid] + bf_std[bf_valid],
            color="#1f77b4",
            alpha=0.15,
        )
    qaoa_line, = ax.plot(
        qaoa_x,
        qaoa_mean,
        color="#d62728",
        linewidth=1.5,
        marker="s",
        markersize=3.0,
        markevery=qaoa_markevery,
        label="QAOA",
    )
    q_valid = np.isfinite(qaoa_mean) & np.isfinite(qaoa_std)
    if np.any(q_valid):
        xq = qaoa_x[q_valid]
        ax.fill_between(
            xq,
            qaoa_mean[q_valid] - qaoa_std[q_valid],
            qaoa_mean[q_valid] + qaoa_std[q_valid],
            color="#d62728",
            alpha=0.15,
        )

    x_max = int(max(int(np.max(bf_x)) if bf_x.size else 1, int(np.max(qaoa_x)) if qaoa_x.size else 1, 1))
    ax.set_xlim(1, x_max)
    ax.set_xticks(_compute_xticks(x_max))
    optimum_line = None
    if es_min_value is not None and np.isfinite(es_min_value):
        x_star = float(max(1, int(round(0.9 * x_max))))
        optimum_line = ax.axhline(
            y=float(es_min_value),
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="Optimum",
        )
        ax.plot([x_star], [float(es_min_value)], marker="*", color="black", markersize=7.0, linestyle="None")

    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("Objective", fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.3)
    if optimum_line is not None:
        optimum_legend = Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            marker="*",
            markersize=7.0,
            linewidth=1.0,
            label="Optimum",
        )
        ax.legend(handles=[optimum_legend, bf_line, qaoa_line], frameon=False, fontsize=9)
    else:
        ax.legend(handles=[bf_line, qaoa_line], frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=int(dpi))
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
    markevery = max(1, len(x) // 8)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        x,
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
        xv = x[valid]
        ax.fill_between(xv, mean[valid] - std[valid], mean[valid] + std[valid], color="#d62728", alpha=0.12)

    x_max = int(max(int(np.max(x)) if x.size else 1, 1))
    ax.set_xlim(1, x_max)
    ax.set_xticks(_compute_xticks(x_max))
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("QAOA Loss", fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=int(dpi))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot synthetic poster outputs from saved artifacts.")
    parser.add_argument("--in-dir", type=str, default="script/poster")
    parser.add_argument("--mu", type=float, default=None, help="Target mu for selecting artifacts.")
    parser.add_argument("--convergence-npz", type=str, default=None)
    parser.add_argument("--metrics-csv", type=str, default=None)
    parser.add_argument("--figure-path", type=str, default="script/poster/synth_convergence_qaoa_bf.png")
    parser.add_argument("--qaoa-loss-figure-path", type=str, default="script/poster/synth_qaoa_loss_history.png")
    parser.add_argument("--tex-table-path", type=str, default="script/poster/synth_metrics.tex")
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
        prefix="convergence_mu=",
        suffix=".npz",
        mu=args.mu,
    )
    metrics_path = _resolve_artifact_path(
        explicit_path=args.metrics_csv,
        directory=in_dir,
        prefix="metrics_mu=",
        suffix=".csv",
        mu=args.mu,
    )

    conv = np.load(convergence_path)
    bf_x = conv["bf_x"] if "bf_x" in conv else np.arange(1, len(conv["bf_mean"]) + 1, dtype=np.int64)
    qaoa_x = conv["qaoa_x"] if "qaoa_x" in conv else np.arange(1, len(conv["qaoa_mean"]) + 1, dtype=np.int64)
    qaoa_loss_x = conv["qaoa_loss_x"] if "qaoa_loss_x" in conv else qaoa_x
    es_min_value = None
    if "es_min_value" in conv:
        arr = np.asarray(conv["es_min_value"], dtype=np.float64).reshape(-1)
        if arr.size > 0 and np.isfinite(arr[0]):
            es_min_value = float(arr[0])

    plot_convergence(
        bf_x=bf_x,
        bf_mean=conv["bf_mean"],
        qaoa_x=qaoa_x,
        qaoa_mean=conv["qaoa_mean"],
        es_min_value=es_min_value,
        output_path=Path(args.figure_path),
        figsize=figsize,
        dpi=args.dpi,
    )
    print(f"[info] wrote convergence figure to {args.figure_path}")

    if "qaoa_loss_mean" in conv and "qaoa_loss_std" in conv:
        plot_qaoa_loss_history(
            qaoa_loss_x=qaoa_loss_x,
            qaoa_loss_mean=conv["qaoa_loss_mean"],
            qaoa_loss_std=conv["qaoa_loss_std"],
            output_path=Path(args.qaoa_loss_figure_path),
            figsize=figsize,
            dpi=args.dpi,
        )
        print(f"[info] wrote QAOA loss figure to {args.qaoa_loss_figure_path}")
    else:
        print("[warn] qaoa_loss_mean/std not found in NPZ; skipped QAOA loss plot.")

    df = pd.read_csv(metrics_path)
    needed = {
        "solver",
        "mean_Rhat",
        "std_Rhat",
        "mean_relerr",
        "std_relerr",
        "mean_energy",
        "std_energy",
        "mean_gap",
        "std_gap",
    }
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
