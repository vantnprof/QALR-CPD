"""
Plotting utilities for ablation_acid_ibm_device.py outputs.

Produces:
  1) rel. error vs iteration (noiseless vs depolarizing noise model)
  2) selected rank vs pruning call (noiseless vs depolarizing noise model)
  3) QAOA optimization landscape (first/last prune in separate files)
  4) QAOA distribution bar charts (four separate files; sim/ibm first/last)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[warn] missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _sanitize_tag(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(tag))


def _select_summary_row(
    summary: pd.DataFrame,
    dataset_tag: Optional[str],
    seed: Optional[int],
    rank_init: Optional[int],
) -> Optional[pd.Series]:
    if summary.empty:
        return None
    df = summary.copy()
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    if dataset_tag is not None:
        df = df[df["dataset_tag"] == dataset_tag]
    if seed is not None:
        df = df[df["seed"] == seed]
    if rank_init is not None and "rank_init" in df.columns:
        df = df[df["rank_init"] == rank_init]
    if df.empty:
        return None
    if len(df) > 1:
        print("[warn] multiple runs match filter; using first row")
    return df.iloc[0]


def _load_diag(path: str) -> Dict:
    # the records store paths from before the first-version results moved to results/v1/
    if not os.path.exists(path) and str(path).startswith("results/") and not str(path).startswith("results/v1/"):
        path = "results/v1/" + str(path)[len("results/"):]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _plot_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_path: str,
) -> None:
    if df.empty:
        print("[warn] metric plot skipped (no data).")
        return

    if "backend" not in df.columns or "sweep" not in df.columns or metric not in df.columns:
        print("[warn] metric plot skipped (missing columns).")
        return

    agg = (
        df.groupby(["backend", "sweep"])
        .agg(value_mean=(metric, "mean"), value_std=(metric, "std"))
        .reset_index()
    )
    if agg.empty:
        print("[warn] metric plot skipped (no aggregated data).")
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    xticks = [1, 20, 40, 60, 80, 100]
    line_width = 1.6
    marker_size = 4
    mark_every = 12
    fill_alpha = 0.08

    for backend, style in {
        "sim": {"color": "C0", "linestyle": "-", "marker": "o", "label": "sim"},
        "ibm": {"color": "C1", "linestyle": "--", "marker": "s", "label": "noisy sim. (depolarizing)"},
    }.items():
        sub = agg[agg["backend"] == backend].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("sweep")
        x = sub["sweep"].to_numpy()
        if np.nanmin(x) <= 0:
            x = x + 1
        plot_style = dict(style)
        label = plot_style.pop("label")
        ax.plot(
            x,
            sub["value_mean"].to_numpy(),
            markevery=mark_every,
            label=label,
            linewidth=line_width,
            markersize=marker_size,
            **plot_style,
        )
        value_std = sub["value_std"].fillna(0.0)
        if value_std.notna().any():
            lo = sub["value_mean"] - value_std
            hi = sub["value_mean"] + value_std
            ax.fill_between(x, lo, hi, alpha=fill_alpha, color=plot_style.get("color", None))

    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title=None, framealpha=0.9, prop={"style": "italic", "size": 9})

    plot_max_sweep = 100
    ax.set_xlim(1, plot_max_sweep)
    ax.set_xticks(xticks)

    value_std = agg["value_std"].fillna(0.0)
    ymin = float((agg["value_mean"] - value_std).min())
    ymax = float((agg["value_mean"] + value_std).max())
    ypad = max(1e-4, 0.05 * (ymax - ymin))
    ax.set_ylim(max(0.0, ymin - ypad), ymax + ypad)

    fig.tight_layout(pad=0.3)
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=1000)
    plt.close(fig)
    print(f"[info] wrote {out_path}")


def _plot_rank_prune(df: pd.DataFrame, out_path: str) -> None:
    if df.empty:
        print("[warn] rank plot skipped (no data).")
        return

    prune_rows_list = []
    for backend in ("sim", "ibm"):
        sub = df[df["backend"] == backend].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("sweep")

        prune_rows = None
        if "prune_period" in sub.columns:
            periods = [p for p in sub["prune_period"].dropna().unique() if p > 0]
            if periods:
                prune_period = int(periods[0])
                prune_rows = sub[sub["sweep"] % prune_period == 0].copy()
                if not prune_rows.empty:
                    prune_rows["prune_call"] = (prune_rows["sweep"] / prune_period).astype(int)

        if (prune_rows is None or prune_rows.empty) and "prune_applied" in sub.columns:
            prune_rows = sub[sub["prune_applied"] == True].copy()
            if not prune_rows.empty:
                if "seed" in prune_rows.columns:
                    prune_rows["prune_call"] = prune_rows.groupby("seed").cumcount() + 1
                else:
                    prune_rows = prune_rows.sort_values("sweep")
                    prune_rows["prune_call"] = np.arange(1, len(prune_rows) + 1)

        if prune_rows is None or prune_rows.empty:
            continue

        prune_rows = prune_rows[["prune_call", "rank"]].copy()
        prune_rows["backend"] = backend
        prune_rows_list.append(prune_rows)

    if not prune_rows_list:
        print("[warn] rank plot skipped (no pruning calls found).")
        return

    prune_df = pd.concat(prune_rows_list, ignore_index=True)
    prune_df = prune_df[(prune_df["prune_call"] >= 1) & (prune_df["prune_call"] <= 4)]
    if prune_df.empty:
        print("[warn] rank plot skipped (no pruning calls in 1..4).")
        return

    agg = (
        prune_df.groupby(["backend", "prune_call"])
        .agg(rank_mean=("rank", "mean"), rank_std=("rank", "std"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    xticks = [1, 2, 3, 4]
    line_width = 1.6
    marker_size = 4
    mark_every = 1
    fill_alpha = 0.08

    for backend, style in {
        "sim": {"color": "C0", "linestyle": "-", "marker": "o", "label": "sim"},
        "ibm": {"color": "C1", "linestyle": "--", "marker": "s", "label": "noisy sim. (depolarizing)"},
    }.items():
        sub = agg[agg["backend"] == backend].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("prune_call")
        x = sub["prune_call"].to_numpy()
        plot_style = dict(style)
        label = plot_style.pop("label")
        ax.plot(
            x,
            sub["rank_mean"].to_numpy(),
            markevery=mark_every,
            label=label,
            linewidth=line_width,
            markersize=marker_size,
            **plot_style,
        )
        rank_std = sub["rank_std"].fillna(0.0)
        if rank_std.notna().any():
            lo = sub["rank_mean"] - rank_std
            hi = sub["rank_mean"] + rank_std
            ax.fill_between(x, lo, hi, alpha=fill_alpha, color=plot_style.get("color", None))

    ax.set_xlabel("Pruning call", fontsize=10)
    ax.set_ylabel("Selected rank", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title=None, framealpha=0.9, prop={"style": "italic", "size": 9})

    x_min, x_max = 1, 4
    pad = max(0.2, (x_max - x_min) * 0.01)
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_xticks(xticks)

    rank_std = agg["rank_std"].fillna(0.0)
    ymin = float((agg["rank_mean"] - rank_std).min())
    ymax = float((agg["rank_mean"] + rank_std).max())
    ypad = max(1e-4, 0.05 * (ymax - ymin))
    ax.set_ylim(max(0.0, ymin - ypad), ymax + ypad)

    fig.tight_layout(pad=0.3)
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=1000)
    plt.close(fig)
    print(f"[info] wrote {out_path}")


def _plot_landscape(diag: Dict, out_path: str) -> None:
    landscape = diag.get("qaoa_landscape", {})
    sim_first = landscape.get("sim_first", [])
    sim_last = landscape.get("sim_last", [])
    ibm_first = landscape.get("ibm_first", [])
    ibm_last = landscape.get("ibm_last", [])

    out_base, out_ext = os.path.splitext(out_path)
    if not out_ext:
        out_ext = ".png"
        out_base = out_path
    out_first = f"{out_base}_first{out_ext}"
    out_last = f"{out_base}_last{out_ext}"

    def _plot_single(
        sim_vals: List[float],
        ibm_vals: List[float],
        title: str,
        out_file: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        max_len = max(len(sim_vals), len(ibm_vals), 1)
        x = np.arange(1, max_len + 1)
        if sim_vals:
            ax.plot(x[: len(sim_vals)], sim_vals, color="C0", linestyle="-", label="sim")
        if ibm_vals:
            ax.plot(x[: len(ibm_vals)], ibm_vals, color="C1", linestyle="--", label="noisy sim. (depolarizing)")
        if title:
            ax.set_title(title, fontsize=9)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective")
        pad = max(0.2, (max_len - 1) * 0.01)
        ax.set_xlim(left=1 - pad, right=max_len + pad)
        tick_count = min(6, max_len)
        ticks = np.linspace(1, max_len, num=tick_count)
        ticks = np.unique(np.round(ticks).astype(int))
        if ticks[0] != 1:
            ticks[0] = 1
        if ticks[-1] != max_len:
            ticks[-1] = max_len
        ax.set_xticks(ticks)
        formatter = mticker.ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.get_offset_text().set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, loc="best", prop={"style": "italic", "size": 8})
        fig.tight_layout()
        ensure_dir(os.path.dirname(out_file))
        fig.savefig(out_file, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"[info] wrote {out_file}")

    _plot_single(sim_first, ibm_first, "", out_first)
    _plot_single(sim_last, ibm_last, "", out_last)


def _top_k_bitstrings(dist: Dict[str, float]) -> List[str]:
    if not dist:
        return []
    bit_len = len(next(iter(dist.keys())))
    top_k = bit_len if bit_len > 0 else len(dist)
    ordered = sorted(dist.keys(), key=lambda b: (-float(dist.get(b, 0.0)), int(b, 2)))
    return ordered[: min(top_k, len(ordered))]


def _plot_distributions(diag: Dict, out_path: str, max_bitstrings: Optional[int]) -> None:
    dist = diag.get("qaoa_distributions", {})
    categories = {
        "sim_first": dist.get("sim_first", {}),
        "sim_last": dist.get("sim_last", {}),
        "ibm_first": dist.get("ibm_first", {}),
        "ibm_last": dist.get("ibm_last", {}),
    }

    labels = [
        ("sim_first", "sim_first"),
        ("sim_last", "sim_last"),
        ("ibm_first", "ibm_first"),
        ("ibm_last", "ibm_last"),
    ]

    out_base, out_ext = os.path.splitext(out_path)
    if not out_ext:
        out_ext = ".png"
        out_base = out_path

    for key, suffix in labels:
        dist_cat = categories.get(key, {}) or {}
        bitstrings = _top_k_bitstrings(dist_cat)
        if not bitstrings:
            print(f"[warn] no distribution found for {key}; skipping plot")
            continue

        probs = [float(dist_cat.get(b, 0.0)) for b in bitstrings]
        x = np.arange(len(bitstrings))

        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        ax.bar(x, probs, color="C0")
        ax.set_xlabel("Bitstring")
        ax.set_ylabel("Probability")
        ax.set_xticks(x)
        ax.set_xticklabels(bitstrings, rotation=45, ha="right", fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        out_file = f"{out_base}_{suffix}{out_ext}"
        ensure_dir(os.path.dirname(out_file))
        fig.savefig(out_file, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"[info] wrote {out_file}")


def _summary_long(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    df = summary.copy()
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    rows: List[Dict[str, float]] = []
    for _, row in df.iterrows():
        for backend in ("sim", "ibm"):
            err = row.get(f"final_rel_error_{backend}")
            rank = row.get(f"final_rank_{backend}")
            if err is None or (isinstance(err, float) and np.isnan(err)):
                continue
            rows.append(
                {
                    "dataset_tag": row.get("dataset_tag"),
                    "seed": row.get("seed"),
                    "rank_init": row.get("rank_init"),
                    "backend": backend,
                    "final_rel_error": float(err),
                    "final_rank": float(rank) if rank is not None else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _fmt_sci_tex(value: Optional[float], digits: int = 3) -> str:
    if value is None or np.isnan(value):
        return r"\texttt{--}"
    if value == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(value))))
    mant = value / (10**exp)
    if abs(mant) >= 10:
        mant /= 10
        exp += 1
    sign = "-" if mant < 0 else ""
    mant = abs(mant)
    return f"{sign}{mant:.{digits}f}{{\\times}}10^{{{exp}}}"


def _fmt_err_pm(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None or np.isnan(mean):
        return r"\texttt{--}"
    std_val = 0.0 if std is None or np.isnan(std) else std
    return r"\(" + f"{_fmt_sci_tex(mean)} \\pm {_fmt_sci_tex(std_val)}" + r"\)"


def _fmt_rank_pm(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None or np.isnan(mean):
        return r"\texttt{--}"
    std_val = 0.0 if std is None or np.isnan(std) else std
    return r"\(" + f"{mean:.2f} \\pm {std_val:.2f}" + r"\)"


def build_table(
    summary: pd.DataFrame,
    dataset_tag: Optional[str],
    rank_init: Optional[int],
    out_csv: str,
    out_tex: str,
) -> None:
    if summary.empty:
        print("[warn] summary table skipped (no data).")
        return
    long_df = _summary_long(summary)
    if dataset_tag is not None and "dataset_tag" in long_df.columns:
        long_df = long_df[long_df["dataset_tag"] == dataset_tag]
    if rank_init is not None and "rank_init" in long_df.columns:
        long_df = long_df[long_df["rank_init"] == rank_init]
    if long_df.empty:
        print("[warn] summary table skipped (no matching rows).")
        return

    agg = (
        long_df.groupby("backend", dropna=False)
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

    backend_order = ["sim", "ibm"]
    agg["backend"] = pd.Categorical(agg["backend"], backend_order, ordered=True)
    agg = agg.sort_values("backend")

    ensure_dir(os.path.dirname(out_csv))
    agg.to_csv(out_csv, index=False)
    print(f"[info] wrote {out_csv}")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{QALR-CPD on amino acids: noiseless simulator vs depolarizing noise model (mean $\pm$ std over seeds).}",
        r"\label{tab:ablation_ibm}",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"Backend & \(\hat R\) & Rel.\ error \(\downarrow\) \\",
        r"\midrule",
    ]

    for _, row in agg.iterrows():
        backend = row.get("backend")
        backend_label = "sim" if backend == "sim" else "noisy sim."
        rank_fmt = _fmt_rank_pm(row.get("final_rank_mean"), row.get("final_rank_std"))
        err_fmt = _fmt_err_pm(row.get("final_rel_error_mean"), row.get("final_rel_error_std"))
        lines.append(f"{backend_label} & {rank_fmt} & {err_fmt} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    ensure_dir(os.path.dirname(out_tex))
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[info] wrote {out_tex}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ablation_acid_ibm_device results.")
    parser.add_argument("--results-csv", type=str, default="results/v1/ablation_acid_ibm_device.csv")
    parser.add_argument(
        "--summary-csv", type=str, default="results/v1/ablation_acid_ibm_device_summary.csv"
    )
    parser.add_argument("--diag-path", type=str, default=None)
    parser.add_argument("--dataset-tag", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rank-init", type=int, default=None)
    parser.add_argument("--fig-dir", type=str, default="reports/figures/v1")
    parser.add_argument("--tables-dir", type=str, default="reports/tables/v1")
    parser.add_argument("--max-bitstrings", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = load_csv(args.results_csv)
    summary = load_csv(args.summary_csv)

    diag_path = args.diag_path
    run_id: Dict = {}
    if diag_path is None:
        row = _select_summary_row(summary, args.dataset_tag, args.seed, args.rank_init)
        if row is None:
            raise SystemExit("No matching run found in summary CSV.")
        diag_path = row.get("diag_path")
        if not isinstance(diag_path, str) or not diag_path:
            raise SystemExit("No diag_path available in the selected summary row.")
        run_id = {
            "dataset_tag": row.get("dataset_tag"),
            "seed": row.get("seed"),
            "rank_init": row.get("rank_init"),
        }

    diag = _load_diag(diag_path)
    if not run_id:
        run_id = diag.get("run_id", {})

    dataset_tag = run_id.get("dataset_tag") or args.dataset_tag or "dataset"
    seed = run_id.get("seed")
    rank_init = run_id.get("rank_init")

    if not results.empty and dataset_tag is not None:
        sub = results.copy()
        if "dataset_tag" in sub.columns:
            sub = sub[sub["dataset_tag"] == dataset_tag]
        if seed is not None and "seed" in sub.columns:
            sub = sub[sub["seed"] == seed]
        if rank_init is not None and "rank_init" in sub.columns:
            sub = sub[sub["rank_init"] == rank_init]
    else:
        sub = pd.DataFrame()

    tag_safe = _sanitize_tag(dataset_tag)
    err_path = os.path.join(args.fig_dir, f"ablation_ibm_rel_error_{tag_safe}.png")
    rank_path = os.path.join(args.fig_dir, f"ablation_ibm_rank_{tag_safe}.png")
    landscape_path = os.path.join(args.fig_dir, f"ablation_ibm_landscape_{tag_safe}.png")
    dist_path = os.path.join(args.fig_dir, f"ablation_ibm_distributions_{tag_safe}.png")

    if not sub.empty:
        _plot_metric(sub, "rel_error", "Rel. error", err_path)
        _plot_rank_prune(sub, rank_path)
    else:
        print("[warn] results CSV missing or no matching rows; skipping error/rank curves.")

    _plot_landscape(diag, landscape_path)
    _plot_distributions(diag, dist_path, args.max_bitstrings)

    table_base = os.path.join(args.tables_dir, f"ablation_ibm_summary_{tag_safe}")
    build_table(
        summary,
        dataset_tag=dataset_tag,
        rank_init=rank_init,
        out_csv=f"{table_base}.csv",
        out_tex=f"{table_base}.tex",
    )


if __name__ == "__main__":
    main()
