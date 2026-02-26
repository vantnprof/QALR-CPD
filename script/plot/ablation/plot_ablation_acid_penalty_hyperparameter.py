"""
Visualization for the mu ablation on the amino acid dataset (QALR-CPD only).

Consumes CSV outputs from `ablation_acid_penalty_hyperparameter.py` and produces:
  1) rel. error vs mu (exhaustive vs QAOA)
  2) rank vs mu (exhaustive vs QAOA)
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOLVER_ORDER = ("exhaustive", "qaoa")
SOLVER_LABELS = {
    "exhaustive": "Ideal",
    "qaoa": "QALR-CPD",
}
SOLVER_STYLE = {
    "exhaustive": {"color": "black", "linestyle": "-", "marker": "*"},
    "qaoa": {"color": "C0", "linestyle": "-.", "marker": "o"},
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[warn] missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _parse_list_float(raw: str) -> List[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def _filter_status_ok(df: pd.DataFrame) -> pd.DataFrame:
    if "status" in df.columns:
        return df[df["status"] == "ok"].copy()
    return df.copy()


def _normalize_solver(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "solver" not in df.columns or df["solver"].isna().all():
        if "prune_method" in df.columns:
            df["solver"] = df["prune_method"]
        else:
            df["solver"] = "unknown"
    df["solver"] = df["solver"].astype(str).str.lower()
    df["solver"] = df["solver"].replace({"exa": "exhaustive", "exh": "exhaustive"})
    return df


def _sanitize_tag(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)


def _coerce_mu(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "prune_mu" in df.columns:
        df["prune_mu"] = pd.to_numeric(df["prune_mu"], errors="coerce")
    return df


def _aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = ["dataset_tag", "solver", "prune_mu"]
    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            final_rel_error_mean=("final_rel_error", "mean"),
            final_rel_error_std=("final_rel_error", "std"),
            final_rank_mean=("final_rank", "mean"),
            final_rank_std=("final_rank", "std"),
            count=("final_rel_error", "count"),
        )
        .reset_index()
    )
    for col in ("final_rel_error_std", "final_rank_std"):
        if col in agg.columns:
            agg[col] = agg[col].fillna(0.0)
    return agg


def _sorted_mu_values(df: pd.DataFrame, mu_values: Optional[List[float]]) -> List[float]:
    if mu_values:
        return list(mu_values)
    vals = df["prune_mu"].dropna().unique().tolist()
    return sorted(float(v) for v in vals)


def _series_for_solver(
    df: pd.DataFrame, solver: str, mu_values: Iterable[float], mean_col: str, std_col: str
) -> Tuple[List[float], List[float]]:
    sub = df[df["solver"] == solver].copy()
    if sub.empty:
        return [], []
    sub = sub.set_index("prune_mu")
    means: List[float] = []
    stds: List[float] = []
    for mu in mu_values:
        if mu in sub.index:
            means.append(float(sub.loc[mu, mean_col]))
            stds.append(float(sub.loc[mu, std_col]))
        else:
            means.append(float("nan"))
            stds.append(float("nan"))
    return means, stds


def _plot_metric(
    df: pd.DataFrame,
    dataset_tag: str,
    mu_values: List[float],
    mean_col: str,
    std_col: str,
    ylabel: str,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x_pos = np.arange(len(mu_values))

    for solver in SOLVER_ORDER:
        if solver not in df["solver"].unique():
            continue
        means, stds = _series_for_solver(df, solver, mu_values, mean_col, std_col)
        style = SOLVER_STYLE.get(solver, {})
        label = SOLVER_LABELS.get(solver, solver)
        ax.errorbar(
            x_pos,
            means,
            yerr=stds,
            label=label,
            capsize=3,
            linewidth=1.6,
            **style,
        )

    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="best")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(mu) for mu in mu_values], rotation=30, ha="right")
    fig.tight_layout()

    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=1000, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] wrote {out_path}")


def _emit_conclusion(df: pd.DataFrame, dataset_tag: str, mu_values: List[float]) -> None:
    sub = df[df["dataset_tag"] == dataset_tag]
    if sub.empty:
        return
    print("[note] conclusion:")
    print(
        "  QAOA is not exact like exhaustive search; as mu increases, the selected rank"
        " generally decreases, while QAOA stays close to the exhaustive trend."
    )

    if {"exhaustive", "qaoa"}.issubset(set(sub["solver"].unique())):
        ex_means, _ = _series_for_solver(
            sub, "exhaustive", mu_values, "final_rank_mean", "final_rank_std"
        )
        qa_means, _ = _series_for_solver(
            sub, "qaoa", mu_values, "final_rank_mean", "final_rank_std"
        )
        diffs = [
            abs(e - q)
            for e, q in zip(ex_means, qa_means)
            if not (np.isnan(e) or np.isnan(q))
        ]
        if diffs:
            print(f"  mean |rank_exhaustive - rank_qaoa| = {np.mean(diffs):.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mu ablation results for QALR-CPD on amino acid data."
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="results/ablation_acid_penalty_hyperparameter_summary.csv",
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default="figs/ablation_acid_penalty_hyperparameter",
    )
    parser.add_argument("--dataset-tag", type=str, default=None)
    parser.add_argument(
        "--mu-values",
        type=str,
        default="0.005,0.01,0.1,0.8,1.0",
        help="Comma-separated mu values to show on the x-axis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_csv(args.summary_csv)
    if df.empty:
        return

    df = _filter_status_ok(df)
    df = _normalize_solver(df)
    df = _coerce_mu(df)
    agg = _aggregate_summary(df)
    if agg.empty:
        print("[warn] no usable rows after aggregation")
        return

    if args.dataset_tag is not None:
        dataset_tags = [args.dataset_tag]
    else:
        dataset_tags = sorted(
            t for t in agg["dataset_tag"].dropna().unique().tolist()
        )

    mu_values = _parse_list_float(args.mu_values) if args.mu_values else None

    for dataset_tag in dataset_tags:
        sub = agg[agg["dataset_tag"] == dataset_tag].copy()
        if sub.empty:
            print(f"[warn] no rows for dataset_tag={dataset_tag}")
            continue

        mu_sorted = _sorted_mu_values(sub, mu_values)
        if not mu_sorted:
            print(f"[warn] no mu values for dataset_tag={dataset_tag}")
            continue

        tag_safe = _sanitize_tag(str(dataset_tag))
        err_path = os.path.join(
            args.fig_dir, f"ablation_mu_rel_error_{tag_safe}.png"
        )
        rank_path = os.path.join(
            args.fig_dir, f"ablation_mu_rank_{tag_safe}.png"
        )

        _plot_metric(
            sub,
            dataset_tag=str(dataset_tag),
            mu_values=mu_sorted,
            mean_col="final_rel_error_mean",
            std_col="final_rel_error_std",
            ylabel="Rel. error",
            out_path=err_path,
        )
        _plot_metric(
            sub,
            dataset_tag=str(dataset_tag),
            mu_values=mu_sorted,
            mean_col="final_rank_mean",
            std_col="final_rank_std",
            ylabel="Selected rank",
            out_path=rank_path,
        )
        _emit_conclusion(agg, str(dataset_tag), mu_sorted)


if __name__ == "__main__":
    main()
