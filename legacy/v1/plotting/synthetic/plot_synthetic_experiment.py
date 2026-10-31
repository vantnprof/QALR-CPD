"""
Visualization for synthetic CP pruning experiments.

Consumes CSV outputs from `synthetic_experiment.py` and produces:
  - Convergence curves (mean +/- std across seeds)
  - Rank trajectories for pruning methods
  - LaTeX table for the main comparison
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QALR_METHOD = "QALR-CPD"
LEGACY_QALR_METHOD = "ALS-QAOA"

PLOT_METHOD_LABELS = {
    "ALS": "ALS",
    "ALS-LR (ES)": "ALS-LR (ES)",
    "ALS-LR (BF)": "ALS-LR (BF)",
    "ALS-TOPK": "ALS w/ Top-K",
    "TREL1": "TREL1",
    QALR_METHOD: "QALR-CPD",
    LEGACY_QALR_METHOD: "QALR-CPD",
}

TABLE_METHOD_LABELS = {
    "ALS": r"ALS",
    "ALS-TOPK": r"ALS w/ Top-\(K\)",
    "TREL1": r"TREL1~\cite{shi2017tensorrankestimation}",
    QALR_METHOD: r"QALR-CPD (Ours)",
    LEGACY_QALR_METHOD: r"QALR-CPD (Ours)",
}

STYLE_MAP = {
    "ALS": {"color": "black", "linestyle": "-", "marker": "*"},
    "ALS-LR (ES)": {"color": "C2", "linestyle": "--", "marker": "x"},
    "ALS-LR (BF)": {"color": "C3", "linestyle": "-.", "marker": "+"},
    "ALS-TOPK": {"color": "C1", "linestyle": "--", "marker": "s"},
    "TREL1": {"color": "gray", "linestyle": ":", "marker": "d"},
    QALR_METHOD: {"color": "C0", "linestyle": "-.", "marker": "o"},
    LEGACY_QALR_METHOD: {"color": "C0", "linestyle": "-.", "marker": "o"},
}
METHOD_ORDER = ["ALS", "ALS-TOPK", "TREL1", QALR_METHOD]
PAPER_CONVERGENCE_NAME = "synthetic_convergence_placeholder.pdf"
PAPER_RANKTRAJ_NAME = "synthetic_ranktraj_placeholder.pdf"
PLOT_MAX_SWEEP = 200
ALS_TABLE_RANKS = (6, 20)
CONVERGENCE_ALS_RANK_BY_TAG = {
    "I20x20x20x20_R6_rho0.5_noise0.01": 20,
}
CONVERGENCE_QAOA_MU_BY_TAG = {
    "I20x20x20x20_R6_rho0.5_noise0.01": (0.1,),
}
CONVERGENCE_QAOA_SHOTS_BY_TAG = {
    "I20x20x20x20_R6_rho0.5_noise0.01": 4096,
}
CONVERGENCE_QAOA_RANK_BY_TAG = {
    "I20x20x20x20_R6_rho0.5_noise0.01": 20,
}
CONVERGENCE_LR_MU_BY_TAG = {
    "I20x20x20x20_R6_rho0.5_noise0.01": 0.1,
}
CONVERGENCE_METHODS_BY_TAG = {
    "I20x20x20x20_R6_rho0.5_noise0.01": ("ALS", "ALS-LR (ES)", "ALS-LR (BF)", QALR_METHOD),
}
RANKTRAJ_MAX_PRUNE_CALL_BY_TAG = {
    "I20x20x20x20_R6_rho0.5_noise0.01": 4,
}
RANKTRAJ_METHODS_BY_TAG = {
    "I20x20x20x20_R6_rho0.5_noise0.01": ("ALS-LR (ES)", "ALS-LR (BF)", QALR_METHOD),
}
RANKTRAJ_LR_MU_BY_TAG = {
    "I20x20x20x20_R6_rho0.5_noise0.01": 0.1,
}
X_PAD_FRAC = 0.01
X_PAD_MIN = 0.2
INSET_BOUNDS = (0.55, 0.26, 0.32, 0.32)
ZOOM_METHODS = ("ALS", QALR_METHOD)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[warn] missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def sanitize_tag(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)


def _mean_std(series: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    values = series.dropna()
    if values.empty:
        return None, None
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def _fmt_mean_std(mean: Optional[float], std: Optional[float], fmt: str) -> str:
    if mean is None:
        return r"\texttt{--}"
    std_val = 0.0 if std is None else std
    return f"{fmt.format(mean)} $\\pm$ {fmt.format(std_val)}"


def _filter_status_ok(df: pd.DataFrame) -> pd.DataFrame:
    if "status" in df.columns:
        return df[df["status"] == "ok"].copy()
    return df.copy()


def _normalize_method_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "method" in df.columns:
        df["method"] = df["method"].replace(
            {
                LEGACY_QALR_METHOD: QALR_METHOD,
                "ALS w/ Top-K": "ALS-TOPK",
                "ALS w/ Top-k": "ALS-TOPK",
            }
        )
    return df


def _apply_x_padding(ax: plt.Axes, xmin: float, xmax: float) -> None:
    pad = max(X_PAD_MIN, (xmax - xmin) * X_PAD_FRAC)
    ax.set_xlim(xmin - pad, xmax + pad)


def _select_best_group(
    df: pd.DataFrame,
    group_cols: List[str],
    metric_col: str,
) -> Optional[Dict[str, float]]:
    df = df.copy()
    for col in group_cols:
        if col in df.columns:
            df = df[df[col].notna()]
    if metric_col in df.columns:
        df = df[df[metric_col].notna()]
    if df.empty:
        return None
    agg = df.groupby(group_cols, dropna=False)[metric_col].mean().reset_index()
    if agg.empty:
        return None
    agg = agg.sort_values(metric_col, ascending=True)
    best = agg.iloc[0]
    return {col: float(best[col]) for col in group_cols}


def _select_rank_for_method(
    summary: pd.DataFrame,
    method: str,
    preferred_rank: Optional[float] = None,
) -> Optional[int]:
    sub = summary[summary["method"] == method]
    if sub.empty or "rank_init" not in sub.columns:
        return None
    ranks = sorted({int(r) for r in sub["rank_init"].dropna().unique()})
    if not ranks:
        return None
    if preferred_rank is not None and int(preferred_rank) in ranks:
        return int(preferred_rank)
    return max(ranks)


def select_main_config(summary: pd.DataFrame) -> Dict[str, float]:
    cfg: Dict[str, float] = {}
    summary = _filter_status_ok(summary)
    summary = _normalize_method_names(summary)
    if summary.empty:
        return cfg

    for col in ("top_k", "prune_mu", "qaoa_shots_effective", "trel1_lam"):
        if col not in summary.columns:
            summary[col] = np.nan

    topk_df = summary[summary["method"] == "ALS-TOPK"]
    best_topk = _select_best_group(topk_df, ["top_k"], "final_rel_error")
    if best_topk and "top_k" in best_topk:
        cfg["top_k"] = best_topk["top_k"]

    qaoa_df = summary[summary["method"] == QALR_METHOD]
    best_qaoa = _select_best_group(qaoa_df, ["prune_mu", "qaoa_shots_effective"], "final_rel_error")
    if best_qaoa:
        cfg["qaoa_mu"] = best_qaoa["prune_mu"]
        cfg["qaoa_shots"] = best_qaoa["qaoa_shots_effective"]

    trel1_df = summary[summary["method"] == "TREL1"]
    if not trel1_df.empty and "trel1_lam" in trel1_df.columns:
        best_trel1 = _select_best_group(trel1_df, ["trel1_lam"], "final_rel_error")
        if best_trel1 and "trel1_lam" in best_trel1:
            cfg["trel1_lam"] = best_trel1["trel1_lam"]

    return cfg


def _filter_qaoa_config(df: pd.DataFrame, cfg: Optional[Dict[str, float]]) -> pd.DataFrame:
    if cfg is None:
        return df
    df = _normalize_method_names(df)
    if "prune_mu" not in df.columns:
        df["prune_mu"] = np.nan
    if "qaoa_shots_effective" not in df.columns:
        df["qaoa_shots_effective"] = np.nan

    if "qaoa_mu" in cfg:
        mu = float(cfg["qaoa_mu"])
        mu_vals = df["prune_mu"].to_numpy(dtype=float)
        is_qaoa = df["method"] == QALR_METHOD
        df = df[~is_qaoa | np.isclose(mu_vals, mu, atol=1e-12)]
    if "qaoa_shots" in cfg:
        shots = float(cfg["qaoa_shots"])
        shot_vals = df["qaoa_shots_effective"].to_numpy(dtype=float)
        is_qaoa = df["method"] == QALR_METHOD
        df = df[~is_qaoa | np.isclose(shot_vals, shots, atol=1e-12)]
    return df


def _prepare_group_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "prune_mu" not in df.columns:
        df["prune_mu"] = np.nan
    if "qaoa_shots_effective" not in df.columns:
        df["qaoa_shots_effective"] = np.nan
    df["prune_mu_key"] = df["prune_mu"].fillna(-1.0)
    df["qaoa_shots_key"] = df["qaoa_shots_effective"].fillna(-1)
    return df


def _format_qaoa_suffix(mu: float, shots: float) -> str:
    mu_str = "na" if mu is None or np.isnan(mu) else f"{mu:g}"
    shots_str = "na" if shots is None or np.isnan(shots) else f"{int(shots)}"
    return f" (mu={mu_str}, shots={shots_str})"


def _format_float_tag(value: float) -> str:
    if value is None or np.isnan(value):
        return "na"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def _legend_label(label: str) -> str:
    return label.replace("--", "$--$")


def _qaoa_marker_map(mu_values: np.ndarray) -> Dict[float, str]:
    qaoa_markers = ["o", "^", "s", "D", "v", "P"]
    qaoa_mu_vals = sorted(v for v in mu_values if not np.isnan(v))
    return {val: qaoa_markers[i % len(qaoa_markers)] for i, val in enumerate(qaoa_mu_vals)}


def plot_convergence(
    df: pd.DataFrame,
    out_path: str,
    main_cfg: Optional[Dict[str, float]] = None,
) -> None:
    if df.empty:
        print("[warn] convergence plot skipped (no data).")
        return

    df = _normalize_method_names(df)
    df = df[
        df["method"].isin(("ALS", "ALS-LR (ES)", "ALS-LR (BF)", "TREL1", QALR_METHOD))
    ].copy()
    if df.empty:
        print("[warn] convergence plot skipped (no matching methods).")
        return

    df = _prepare_group_keys(df)
    if main_cfg:
        if "rank_init" in df.columns and "rank_init" in main_cfg:
            df = df[df["rank_init"] == float(main_cfg["rank_init"])]
        df = _filter_qaoa_config(df, main_cfg)
        if df.empty:
            print("[warn] convergence plot skipped (no data after main config filter).")
            return

    dataset_tag = None
    if "dataset_tag" in df.columns:
        tags = [str(t) for t in df["dataset_tag"].dropna().unique()]
        if len(tags) == 1:
            dataset_tag = tags[0]

    qaoa = df[df["method"] == QALR_METHOD]
    lr = df[df["method"].isin(("ALS-LR (ES)", "ALS-LR (BF)"))]
    force_qaoa_label = dataset_tag in CONVERGENCE_QAOA_MU_BY_TAG if dataset_tag else False
    force_lr_label = dataset_tag in CONVERGENCE_LR_MU_BY_TAG if dataset_tag else False
    include_qaoa_params = (
        not main_cfg
        and (
            force_qaoa_label
            or qaoa["prune_mu_key"].nunique(dropna=True) > 1
            or qaoa["qaoa_shots_key"].nunique(dropna=True) > 1
        )
    )
    include_lr_params = (
        not main_cfg
        and (force_lr_label or lr["prune_mu_key"].nunique(dropna=True) > 1)
    )

    als_rank_label = None
    if "rank_init" in df.columns:
        als_ranks = df.loc[df["method"] == "ALS", "rank_init"].dropna().unique()
        if len(als_ranks) == 1:
            als_rank_label = int(als_ranks[0])

    qaoa_rank_label = None
    if "rank_init" in df.columns:
        qaoa_ranks = df.loc[df["method"] == QALR_METHOD, "rank_init"].dropna().unique()
        if len(qaoa_ranks) == 1:
            qaoa_rank_label = int(qaoa_ranks[0])

    agg = (
        df.groupby(["method", "prune_mu_key", "qaoa_shots_key", "sweep"])
        .agg(rel_err_mean=("rel_error", "mean"), rel_err_std=("rel_error", "std"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    xticks = [1, 40, 80, 120, 160, 200]
    line_width = 1.6
    marker_size = 4
    mark_every = 12
    fill_alpha = 0.08
    method_order = ["ALS", "ALS-LR (ES)", "ALS-LR (BF)", "TREL1", QALR_METHOD]
    qaoa_marker_map = _qaoa_marker_map(df["prune_mu"].dropna().unique())
    grouped = []
    for (method, mu_key, shots_key), sub in agg.groupby(["method", "prune_mu_key", "qaoa_shots_key"]):
        grouped.append((method, mu_key, shots_key, sub))
    grouped.sort(
        key=lambda item: (
            method_order.index(item[0]) if item[0] in method_order else len(method_order),
            item[1],
            item[2],
        )
    )

    for method, mu_key, shots_key, sub in grouped:

        sub = sub.sort_values("sweep")
        if method == "ALS" and als_rank_label is not None:
            label = "ALS"
        elif method in ("ALS-LR (ES)", "ALS-LR (BF)") and include_lr_params:
            label = method
        elif method == QALR_METHOD and include_qaoa_params:
            label = PLOT_METHOD_LABELS.get(method, method)
        else:
            label = PLOT_METHOD_LABELS.get(method, method)
        label = _legend_label(label)
        style = dict(STYLE_MAP.get(method, {}))
        if method == QALR_METHOD and include_qaoa_params:
            style["marker"] = qaoa_marker_map.get(float(mu_key), style.get("marker", "o"))
        style.pop("label", None)
        ax.plot(
            sub["sweep"],
            sub["rel_err_mean"],
            markevery=mark_every,
            label=label,
            linewidth=line_width,
            markersize=marker_size,
            **style,
        )
        rel_std = sub["rel_err_std"].fillna(0.0)
        if rel_std.notna().any():
            lo = sub["rel_err_mean"] - rel_std
            hi = sub["rel_err_mean"] + rel_std
            ax.fill_between(sub["sweep"], lo, hi, alpha=fill_alpha, color=style.get("color", None))

    prune_period = None
    if "prune_period" in df.columns:
        periods = [p for p in df["prune_period"].dropna().unique() if p > 0]
        if periods:
            prune_period = int(periods[0])

    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("Rel. error", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title=None, fontsize=9, framealpha=0.9)
    if not agg.empty:
        max_sweep = int(agg["sweep"].max())
        plot_max_sweep = max(PLOT_MAX_SWEEP, max_sweep)
        _apply_x_padding(ax, 1, plot_max_sweep)
        ticks = [t for t in xticks if t <= plot_max_sweep]
        if plot_max_sweep not in ticks:
            ticks.append(plot_max_sweep)
        ax.set_xticks(sorted(ticks))
        rel_std = agg["rel_err_std"].fillna(0.0)
        ymin = float((agg["rel_err_mean"] - rel_std).min())
        ymax = float((agg["rel_err_mean"] + rel_std).max())
        ypad = max(1e-4, 0.05 * (ymax - ymin))
        ax.set_ylim(max(0.0, ymin - ypad), ymax + ypad)

        zoom_start = prune_period if prune_period is not None else max(1, int(0.2 * max_sweep))
        zoom_end = max_sweep
        zoom_df = agg[(agg["sweep"] >= zoom_start) & (agg["method"].isin(ZOOM_METHODS))]
        if not zoom_df.empty and zoom_df["sweep"].nunique() > 1 and zoom_start < zoom_end:
            ax_inset = ax.inset_axes(INSET_BOUNDS)
            for method, mu_key, shots_key, sub in grouped:
                if method not in ZOOM_METHODS:
                    continue
                sub_zoom = sub[sub["sweep"] >= zoom_start]
                if sub_zoom.empty:
                    continue
                style = dict(STYLE_MAP.get(method, {}))
                line_color = style.get("color", None)
                line_style = style.get("linestyle", "-")
                ax_inset.plot(
                    sub_zoom["sweep"],
                    sub_zoom["rel_err_mean"],
                    color=line_color,
                    linestyle=line_style,
                    linewidth=1.1,
                )
                rel_std = sub_zoom["rel_err_std"].fillna(0.0)
                if rel_std.notna().any():
                    lo = sub_zoom["rel_err_mean"] - rel_std
                    hi = sub_zoom["rel_err_mean"] + rel_std
                    ax_inset.fill_between(
                        sub_zoom["sweep"],
                        lo,
                        hi,
                        alpha=0.06,
                        color=line_color,
                    )

            zoom_std = zoom_df["rel_err_std"].fillna(0.0)
            zmin = float((zoom_df["rel_err_mean"] - zoom_std).min())
            zmax = float((zoom_df["rel_err_mean"] + zoom_std).max())
            zpad = max(1e-4, 0.1 * (zmax - zmin))
            ax_inset.set_xlim(zoom_start, zoom_end)
            ax_inset.set_ylim(zmin - zpad, zmax + zpad)
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])
            ax.indicate_inset_zoom(ax_inset, edgecolor="gray")

    # ax.set_yscale("log")
    fig.tight_layout(pad=0.45)
    fig.savefig(out_path, dpi=1000)
    print(f"[info] wrote {out_path}")
    plt.close(fig)


def plot_rank_trajectory(
    df: pd.DataFrame,
    out_path: str,
    main_cfg: Optional[Dict[str, float]] = None,
    mu_targets: Optional[Sequence[float]] = None,
    shots_target: Optional[int] = None,
    rank_init_target: Optional[int] = None,
    max_prune_call: Optional[int] = None,
) -> None:
    if df.empty:
        print("[warn] rank trajectory plot skipped (no data).")
        return

    df = _normalize_method_names(df)
    dataset_tag = None
    if "dataset_tag" in df.columns:
        tags = [str(t) for t in df["dataset_tag"].dropna().unique()]
        if len(tags) == 1:
            dataset_tag = tags[0]

    methods_target = RANKTRAJ_METHODS_BY_TAG.get(dataset_tag, (QALR_METHOD,))
    df = df[df["method"].isin(methods_target)].copy()
    if df.empty:
        print("[warn] rank trajectory plot skipped (no matching methods).")
        return

    df = _prepare_group_keys(df)
    df = _filter_qaoa_config(df, main_cfg)
    if df.empty:
        print("[warn] rank trajectory plot skipped (no data after config filter).")
        return

    if rank_init_target is not None and "rank_init" in df.columns:
        df = df[df["rank_init"] == float(rank_init_target)]
        if df.empty:
            print("[warn] rank trajectory plot skipped (no rows for requested rank).")
            return

    lr_mu_target = RANKTRAJ_LR_MU_BY_TAG.get(dataset_tag)

    ext = ".png"
    out_path_prefix = out_path
    if out_path.lower().endswith(".png"):
        ext = ".png"
        out_path_prefix = out_path[: -len(ext)]
    elif out_path.lower().endswith(".pdf"):
        ext = ".pdf"
        out_path_prefix = out_path[: -len(ext)]

    def _rank_agg_from_subset(sub: pd.DataFrame) -> Optional[pd.DataFrame]:
        prune_period = None
        if "prune_period" in sub.columns:
            periods = [p for p in sub["prune_period"].dropna().unique() if p > 0]
            if periods:
                prune_period = int(periods[0])

        if prune_period:
            sub = sub[sub["sweep"] % prune_period == 0].copy()
            if sub.empty:
                return None
            sub["prune_call"] = (sub["sweep"] / prune_period).astype(int)
        elif "prune_applied" in sub.columns:
            sub = sub[sub["prune_applied"] == True].copy()
            if sub.empty:
                return None
            sub["prune_call"] = sub.groupby("seed").cumcount() + 1
        else:
            return None

        agg = (
            sub.groupby("prune_call")
            .agg(rank_mean=("rank", "mean"), rank_std=("rank", "std"))
            .reset_index()
        )
        if agg.empty:
            return None
        if max_prune_call is not None:
            agg = agg[agg["prune_call"] <= int(max_prune_call)]
            if agg.empty:
                return None
        return agg

    multi_method = any(m != QALR_METHOD for m in methods_target) or len(methods_target) > 1
    if multi_method:
        fig, ax = plt.subplots(figsize=(3.0, 2.5))
        all_calls: List[int] = []
        y_min = None
        y_max = None

        qaoa_mu_target = None
        if mu_targets:
            qaoa_mu_target = float(mu_targets[0])

        for method in methods_target:
            sub = df[df["method"] == method].copy()
            if sub.empty:
                continue

            if method == QALR_METHOD:
                if shots_target is not None and "qaoa_shots_effective" in sub.columns:
                    shot_vals = sub["qaoa_shots_effective"].to_numpy(dtype=float)
                    sub = sub[np.isclose(shot_vals, float(shots_target), atol=1e-12)]
                if qaoa_mu_target is not None:
                    mu_vals = sub["prune_mu"].to_numpy(dtype=float)
                    sub = sub[np.isclose(mu_vals, float(qaoa_mu_target), atol=1e-12)]
            if method in ("ALS-LR (ES)", "ALS-LR (BF)") and lr_mu_target is not None:
                mu_vals = sub["prune_mu"].to_numpy(dtype=float)
                sub = sub[np.isclose(mu_vals, float(lr_mu_target), atol=1e-12)]

            agg = _rank_agg_from_subset(sub)
            if agg is None:
                continue

            all_calls.extend(int(x) for x in agg["prune_call"].tolist())
            style = STYLE_MAP.get(method, {})
            ax.plot(
                agg["prune_call"],
                agg["rank_mean"],
                color=style.get("color", None),
                linestyle=style.get("linestyle", "-"),
                marker=style.get("marker", "o"),
                linewidth=1.6,
                markersize=4,
                label=PLOT_METHOD_LABELS.get(method, method),
            )
            rank_std = agg["rank_std"].fillna(0.0)
            if rank_std.notna().any():
                ax.fill_between(
                    agg["prune_call"],
                    agg["rank_mean"] - rank_std,
                    agg["rank_mean"] + rank_std,
                    alpha=0.12,
                    color=style.get("color", None),
                )
            cur_min = float((agg["rank_mean"] - rank_std).min())
            cur_max = float((agg["rank_mean"] + rank_std).max())
            y_min = cur_min if y_min is None else min(y_min, cur_min)
            y_max = cur_max if y_max is None else max(y_max, cur_max)

        if not all_calls:
            print("[warn] rank trajectory plot skipped (no prune sweeps).")
            return

        ax.set_xlabel("Pruning call", fontsize=10)
        ax.set_ylabel("Selected rank", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(labelsize=9)
        x_min = min(all_calls)
        x_max = max(all_calls)
        _apply_x_padding(ax, x_min, x_max)
        ax.set_xticks(sorted(set(all_calls)))

        if y_min is None or y_max is None:
            y_min, y_max = 0.0, 1.0
        y_pad = max(0.5, 0.05 * (y_max - y_min))
        ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)

        ax.legend(title=None, fontsize=9, framealpha=0.9, loc="best")
        fig.tight_layout(pad=0.3)
        fig.savefig(out_path, dpi=1000)
        print(f"[info] wrote {out_path}")
        plt.close(fig)
        return

    qaoa = df[df["method"] == QALR_METHOD].copy()
    if qaoa.empty:
        print("[warn] rank trajectory plot skipped (no QAOA rows).")
        return

    if shots_target is not None and "qaoa_shots_effective" in qaoa.columns:
        shot_vals = qaoa["qaoa_shots_effective"].to_numpy(dtype=float)
        qaoa = qaoa[np.isclose(shot_vals, float(shots_target), atol=1e-12)]
        if qaoa.empty:
            print("[warn] rank trajectory plot skipped (no QAOA rows for requested shots).")
            return
    if mu_targets is not None:
        mu_vals = qaoa["prune_mu"].to_numpy(dtype=float)
        mu_mask = np.zeros_like(mu_vals, dtype=bool)
        for mu in mu_targets:
            mu_mask |= np.isclose(mu_vals, float(mu), atol=1e-12)
        qaoa = qaoa[mu_mask]
        if qaoa.empty:
            print("[warn] rank trajectory plot skipped (no QAOA rows for requested mu).")
            return

    if mu_targets is not None:
        mu_list = [float(mu) for mu in mu_targets]
        if not mu_list:
            print("[warn] rank trajectory plot skipped (empty mu_targets).")
            return
        markers = ["o", "^", "s", "D", "P", "X"]
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(mu_list))))
        mu_color = {float(mu): colors[i % len(colors)] for i, mu in enumerate(mu_list)}

        fig, ax = plt.subplots(figsize=(3.0, 2.5))
        all_calls = []
        y_min = None
        y_max = None

        for idx, mu in enumerate(mu_list):
            sub = qaoa[np.isclose(qaoa["prune_mu"].to_numpy(dtype=float), mu, atol=1e-12)]
            agg = _rank_agg_from_subset(sub)
            if agg is None:
                continue
            all_calls.extend(int(x) for x in agg["prune_call"].tolist())
            color = mu_color.get(float(mu), "C0")
            marker = markers[idx % len(markers)]
            ax.plot(
                agg["prune_call"],
                agg["rank_mean"],
                color=color,
                linestyle="-",
                marker=marker,
                linewidth=1.6,
                markersize=4,
                label="QALR-CPD",
            )
            rank_std = agg["rank_std"].fillna(0.0)
            if rank_std.notna().any():
                ax.fill_between(
                    agg["prune_call"],
                    agg["rank_mean"] - rank_std,
                    agg["rank_mean"] + rank_std,
                    alpha=0.12,
                    color=color,
                )
            cur_min = float((agg["rank_mean"] - rank_std).min())
            cur_max = float((agg["rank_mean"] + rank_std).max())
            y_min = cur_min if y_min is None else min(y_min, cur_min)
            y_max = cur_max if y_max is None else max(y_max, cur_max)

        if not all_calls:
            print("[warn] rank trajectory plot skipped (no prune sweeps).")
            return

        ax.set_xlabel("Pruning call", fontsize=10)
        ax.set_ylabel("Selected rank", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(labelsize=9)
        x_min = min(all_calls)
        x_max = max(all_calls)
        _apply_x_padding(ax, x_min, x_max)
        ax.set_xticks(sorted(set(all_calls)))

        if y_min is None or y_max is None:
            y_min, y_max = 0.0, 1.0
        y_pad = max(0.5, 0.05 * (y_max - y_min))
        ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)

        ax.legend(title=None, fontsize=9, framealpha=0.9, loc="best")
        fig.tight_layout(pad=0.3)
        fig.savefig(out_path, dpi=1000)
        print(f"[info] wrote {out_path}")
        plt.close(fig)
        return

    groups = []
    if main_cfg:
        groups = [(None, None, qaoa, out_path)]
    else:
        for (mu_key, shots_key), sub in qaoa.groupby(["prune_mu_key", "qaoa_shots_key"]):
            mu_tag = _format_float_tag(float(mu_key))
            shots_tag = _format_float_tag(float(shots_key))
            group_path = f"{out_path_prefix}_mu{mu_tag}_shots{shots_tag}{ext}"
            groups.append((mu_key, shots_key, sub, group_path))

    for _, _, sub, group_path in groups:
        agg = _rank_agg_from_subset(sub)
        if agg is None:
            print("[warn] rank trajectory plot skipped (no prune sweeps).")
            continue

        style = STYLE_MAP.get(QALR_METHOD, {})
        fig, ax = plt.subplots(figsize=(3.0, 2.5))
        ax.plot(
            agg["prune_call"],
            agg["rank_mean"],
            color=style.get("color", "C0"),
            linestyle="-",
            marker=style.get("marker", "o"),
            linewidth=1.6,
            markersize=4,
            label="Selected rank",
        )
        rank_std = agg["rank_std"].fillna(0.0)
        if rank_std.notna().any():
            ax.fill_between(
                agg["prune_call"],
                agg["rank_mean"] - rank_std,
                agg["rank_mean"] + rank_std,
                alpha=0.12,
                color=style.get("color", "C0"),
            )

        ax.set_xlabel("Pruning call", fontsize=10)
        ax.set_ylabel("Selected rank", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(labelsize=9)
        _apply_x_padding(ax, 1, int(agg["prune_call"].max()))
        ax.set_xticks(sorted(int(x) for x in agg["prune_call"].unique()))

        ymin = float((agg["rank_mean"] - rank_std).min())
        ymax = float((agg["rank_mean"] + rank_std).max())
        ypad = max(0.5, 0.05 * (ymax - ymin))
        ax.set_ylim(max(0.0, ymin - ypad), ymax + ypad)

        ax.legend(title=None, fontsize=9, framealpha=0.9, loc="best")
        fig.tight_layout(pad=0.3)
        fig.savefig(group_path, dpi=1000)
        print(f"[info] wrote {group_path}")
        plt.close(fig)


def plot_qaoa_tradeoff(df: pd.DataFrame, out_path_prefix: str) -> None:
    if df.empty:
        print("[warn] QAOA trade-off plot skipped (no data).")
        return

    df = _normalize_method_names(df)
    qaoa = df[df["method"] == QALR_METHOD].copy()
    if qaoa.empty:
        print("[warn] QAOA trade-off plot skipped (no QAOA rows).")
        return

    qaoa = _prepare_group_keys(qaoa)
    for (mu_key, shots_key), sub in qaoa.groupby(["prune_mu_key", "qaoa_shots_key"]):
        agg = (
            sub.groupby("sweep")
            .agg(
                rel_err_mean=("rel_error", "mean"),
                rel_err_std=("rel_error", "std"),
                rank_mean=("rank", "mean"),
                rank_std=("rank", "std"),
            )
            .reset_index()
        )
        if agg.empty:
            continue

        prune_period = None
        if "prune_period" in sub.columns:
            periods = [p for p in sub["prune_period"].dropna().unique() if p > 0]
            if periods:
                prune_period = int(periods[0])

        mu_tag = _format_float_tag(float(mu_key))
        shots_tag = _format_float_tag(float(shots_key))
        out_path = f"{out_path_prefix}_mu{mu_tag}_shots{shots_tag}.png"

        fig, ax_err = plt.subplots(figsize=(3.0, 2.5))
        ax_rank = ax_err.twinx()
        xticks = [1, 40, 80, 120, 160, 200]

        style = STYLE_MAP.get(QALR_METHOD, {"label": r"QALR-CPD"})
        line_width = 1.6
        rank_width = 1.4
        marker_size = 4
        mark_every = 12
        fill_alpha_err = 0.08
        fill_alpha_rank = 0.08

        ax_err.plot(
            agg["sweep"],
            agg["rel_err_mean"],
            color=style.get("color", None),
            linestyle=style.get("linestyle", "-"),
            marker=style.get("marker", "o"),
            markevery=mark_every,
            label="Rel. error",
            linewidth=line_width,
            markersize=marker_size,
            zorder=3,
        )
        rel_std = agg["rel_err_std"].fillna(0.0)
        ax_err.fill_between(
            agg["sweep"],
            agg["rel_err_mean"] - rel_std,
            agg["rel_err_mean"] + rel_std,
            alpha=fill_alpha_err,
            color=style.get("color", None),
            zorder=1,
        )

        ax_rank.step(
            agg["sweep"],
            agg["rank_mean"],
            where="post",
            color="gray",
            linestyle="--",
            label="Rank",
            linewidth=rank_width,
            alpha=0.8,
            zorder=2,
        )
        rank_std = agg["rank_std"].fillna(0.0)
        ax_rank.fill_between(
            agg["sweep"],
            agg["rank_mean"] - rank_std,
            agg["rank_mean"] + rank_std,
            step="post",
            alpha=fill_alpha_rank,
            color="gray",
        )

        ax_err.set_xlabel("Iteration", fontsize=10)
        ax_err.set_ylabel("Rel. error", fontsize=10)
        ax_rank.set_ylabel("Selected rank", fontsize=10)
        ax_err.grid(True, linestyle="--", alpha=0.35)
        ax_err.tick_params(labelsize=9)
        ax_rank.tick_params(labelsize=9)
        max_sweep = int(agg["sweep"].max())
        plot_max_sweep = max(PLOT_MAX_SWEEP, max_sweep)
        _apply_x_padding(ax_err, 1, plot_max_sweep)
        ticks = [t for t in xticks if t <= plot_max_sweep]
        if plot_max_sweep not in ticks:
            ticks.append(plot_max_sweep)
        ax_err.set_xticks(sorted(ticks))

        err_min = float((agg["rel_err_mean"] - rel_std).min())
        err_max = float((agg["rel_err_mean"] + rel_std).max())
        err_pad = max(1e-4, 0.05 * (err_max - err_min))
        ax_err.set_ylim(max(err_min - err_pad, 0.0), err_max + err_pad)

        rank_min = float((agg["rank_mean"] - rank_std).min())
        rank_max = float((agg["rank_mean"] + rank_std).max())
        ax_rank.set_ylim(max(rank_min - 0.5, 0.0), rank_max + 0.5)

        if prune_period is not None:
            max_sweep = int(agg["sweep"].max())
            for sweep in range(prune_period, max_sweep + 1, prune_period):
                ax_err.axvline(sweep, color="lightgray", linestyle=":", linewidth=0.8, zorder=0)

            zoom = agg[agg["sweep"] >= prune_period]
            if not zoom.empty:
                ax_inset = ax_err.inset_axes([0.55, 0.28, 0.4, 0.35])
                ax_inset.plot(
                    zoom["sweep"],
                    zoom["rel_err_mean"],
                    color=style.get("color", None),
                    linewidth=1.2,
                )
                zstd = zoom["rel_err_std"].fillna(0.0)
                ax_inset.fill_between(
                    zoom["sweep"],
                    zoom["rel_err_mean"] - zstd,
                    zoom["rel_err_mean"] + zstd,
                    alpha=0.08,
                    color=style.get("color", None),
                )
                zmin = float((zoom["rel_err_mean"] - zstd).min())
                zmax = float((zoom["rel_err_mean"] + zstd).max())
                zpad = max(1e-4, 0.1 * (zmax - zmin))
                ax_inset.set_xlim(prune_period, int(agg["sweep"].max()))
                ax_inset.set_ylim(max(zmin - zpad, 0.0), zmax + zpad)
                ax_inset.set_xticks([])
                ax_inset.set_yticks([])
                ax_err.indicate_inset_zoom(ax_inset, edgecolor="gray")

        handles, labels = [], []
        for ax in (ax_err, ax_rank):
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        ax_err.legend(handles, labels, loc="upper right", fontsize=9, framealpha=0.9)

        fig.tight_layout(pad=0.3)
        fig.savefig(out_path, dpi=1000)
        print(f"[info] wrote {out_path}")
        plt.close(fig)


def build_latex_table(
    summary: pd.DataFrame,
    out_path: str,
    main_cfg: Optional[Dict[str, float]] = None,
) -> None:
    if summary.empty:
        print("[warn] summary table skipped (no data).")
        return

    summary = _filter_status_ok(summary)
    summary = _normalize_method_names(summary)
    if summary.empty:
        print("[warn] summary table skipped (no valid rows).")
        return

    summary = summary.copy()
    for col in ("top_k", "prune_mu", "qaoa_shots_effective", "trel1_lam"):
        if col not in summary.columns:
            summary[col] = np.nan

    if main_cfg is None:
        main_cfg = select_main_config(summary)

    def _lookup_stats(
        method: str,
        rank_init: Optional[float] = None,
        top_k: Optional[float] = None,
        mu: Optional[float] = None,
        shots: Optional[float] = None,
        trel1_lam: Optional[float] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        sub = summary[summary["method"] == method].copy()
        if rank_init is not None and "rank_init" in sub.columns:
            sub = sub[sub["rank_init"] == float(rank_init)]
        if top_k is not None:
            sub = sub[sub["top_k"] == float(top_k)]
        if mu is not None:
            mu_vals = sub["prune_mu"].to_numpy(dtype=float)
            sub = sub[np.isclose(mu_vals, float(mu), atol=1e-12)]
        if shots is not None:
            shot_vals = sub["qaoa_shots_effective"].to_numpy(dtype=float)
            sub = sub[np.isclose(shot_vals, float(shots), atol=1e-12)]
        if trel1_lam is not None and "trel1_lam" in sub.columns:
            lam_vals = sub["trel1_lam"].to_numpy(dtype=float)
            sub = sub[np.isclose(lam_vals, float(trel1_lam), atol=1e-12)]
        rank_mean, rank_std = _mean_std(sub["final_rank"])
        err_mean, err_std = _mean_std(sub["final_rel_error"])
        return rank_mean, rank_std, err_mean, err_std

    preferred_rank = main_cfg.get("rank_init")
    topk_rank = _select_rank_for_method(summary, "ALS-TOPK", preferred_rank)
    trel1_rank = _select_rank_for_method(summary, "TREL1", preferred_rank)
    qalr_rank = _select_rank_for_method(summary, QALR_METHOD, preferred_rank)

    top_k = main_cfg.get("top_k")
    if top_k is None and topk_rank is not None:
        topk_df = summary[summary["method"] == "ALS-TOPK"]
        topk_df = topk_df[topk_df["rank_init"] == float(topk_rank)]
        best_topk = _select_best_group(topk_df, ["top_k"], "final_rel_error")
        if best_topk and "top_k" in best_topk:
            top_k = best_topk["top_k"]

    qaoa_mu = main_cfg.get("qaoa_mu")
    qaoa_shots = main_cfg.get("qaoa_shots")
    if qalr_rank is not None and (qaoa_mu is None or qaoa_shots is None):
        qaoa_df = summary[summary["method"] == QALR_METHOD]
        qaoa_df = qaoa_df[qaoa_df["rank_init"] == float(qalr_rank)]
        best_qaoa = _select_best_group(qaoa_df, ["prune_mu", "qaoa_shots_effective"], "final_rel_error")
        if best_qaoa:
            qaoa_mu = qaoa_mu if qaoa_mu is not None else best_qaoa.get("prune_mu")
            qaoa_shots = qaoa_shots if qaoa_shots is not None else best_qaoa.get("qaoa_shots_effective")

    trel1_lam = main_cfg.get("trel1_lam")
    if trel1_rank is not None and trel1_lam is None and "trel1_lam" in summary.columns:
        trel1_df = summary[summary["method"] == "TREL1"]
        trel1_df = trel1_df[trel1_df["rank_init"] == float(trel1_rank)]
        best_trel1 = _select_best_group(trel1_df, ["trel1_lam"], "final_rel_error")
        if best_trel1 and "trel1_lam" in best_trel1:
            trel1_lam = best_trel1["trel1_lam"]

    rows = []
    for als_rank in ALS_TABLE_RANKS:
        rank_mean, rank_std, err_mean, err_std = _lookup_stats("ALS", rank_init=als_rank)
        als_label = rf"ALS (\(R={als_rank}\))" if als_rank == ALS_TABLE_RANKS[0] else rf"ALS \(R={als_rank}\)"
        rows.append(
            {
                "method": als_label,
                "final_rank_mean": rank_mean,
                "final_rank_std": rank_std,
                "final_rel_error_mean": err_mean,
                "final_rel_error_std": err_std,
            }
        )

    topk_label = TABLE_METHOD_LABELS.get("ALS-TOPK", "ALS w/ Top-K")
    rank_mean, rank_std, err_mean, err_std = _lookup_stats(
        "ALS-TOPK",
        rank_init=topk_rank,
        top_k=top_k,
    )
    rows.append(
        {
            "method": topk_label,
            "final_rank_mean": rank_mean,
            "final_rank_std": rank_std,
            "final_rel_error_mean": err_mean,
            "final_rel_error_std": err_std,
        }
    )

    trel1_label = TABLE_METHOD_LABELS.get("TREL1", "TREL1")
    rank_mean, rank_std, err_mean, err_std = _lookup_stats(
        "TREL1",
        rank_init=trel1_rank,
        trel1_lam=trel1_lam,
    )
    rows.append(
        {
            "method": trel1_label,
            "final_rank_mean": rank_mean,
            "final_rank_std": rank_std,
            "final_rel_error_mean": err_mean,
            "final_rel_error_std": err_std,
        }
    )

    qalr_label = TABLE_METHOD_LABELS.get(QALR_METHOD, QALR_METHOD)
    rank_mean, rank_std, err_mean, err_std = _lookup_stats(
        QALR_METHOD,
        rank_init=qalr_rank,
        mu=qaoa_mu,
        shots=qaoa_shots,
    )
    rows.append(
        {
            "method": qalr_label,
            "final_rank_mean": rank_mean,
            "final_rank_std": rank_std,
            "final_rel_error_mean": err_mean,
            "final_rel_error_std": err_std,
        }
    )

    table_df = pd.DataFrame(rows)
    table_df.to_csv(out_path.replace(".tex", ".csv"), index=False)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Synthetic benchmark (Table~\ref{tab:synth_config}): selected rank \(\hat R\) "
        r"and relative error (mean \(\pm\) std over \(20\) seeds).}",
        r"\label{tab:results_synth}",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"Method & \(\hat R\) & Rel.\ error \(\downarrow\) \\",
        r"\midrule",
    ]

    for _, row in table_df.iterrows():
        rank_str = _fmt_mean_std(row["final_rank_mean"], row["final_rank_std"], "{:.2f}")
        err_str = _fmt_mean_std(row["final_rel_error_mean"], row["final_rel_error_std"], "{:.3e}")
        lines.append(f"{row['method']} & {rank_str} & {err_str} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[info] wrote {out_path}")


def build_all_results_table(
    summary: pd.DataFrame,
    out_path: str,
    rank_init: Optional[int] = None,
) -> None:
    if summary.empty:
        print("[warn] summary table skipped (no data).")
        return

    summary = _filter_status_ok(summary)
    summary = _normalize_method_names(summary)
    if summary.empty:
        print("[warn] summary table skipped (no valid rows).")
        return

    summary = summary.copy()
    for col in ("top_k", "prune_mu", "qaoa_shots_effective", "trel1_lam", "rank_init"):
        if col not in summary.columns:
            summary[col] = np.nan
    if rank_init is not None and "rank_init" in summary.columns:
        summary = summary[summary["rank_init"] == float(rank_init)]
        if summary.empty:
            print("[warn] summary table skipped (no rows for requested rank_init).")
            return

    method_specs = [
        ("ALS", "ALS", ["rank_init"]),
        ("ALS-TOPK", "ALS w/ Top-K", ["rank_init", "top_k"]),
        ("ALS-LR (ES)", "ALS-LR (ES)", ["rank_init", "prune_mu"]),
        ("ALS-LR (BF)", "ALS-LR (BF)", ["rank_init", "prune_mu"]),
        ("TREL1", "TREL1", ["rank_init", "trel1_lam"]),
        (QALR_METHOD, "QALR-CPD", ["rank_init", "prune_mu", "qaoa_shots_effective"]),
    ]

    rows_full: List[Dict[str, str]] = []
    for method_key, method_label, config_cols in method_specs:
        sub = summary[summary["method"] == method_key].copy()
        if sub.empty:
            continue
        for col in config_cols:
            if col not in sub.columns:
                sub[col] = np.nan
        sub = sub.dropna(subset=config_cols)
        if sub.empty:
            continue
        agg = (
            sub.groupby(config_cols, dropna=False)
            .agg(
                final_rank_mean=("final_rank", "mean"),
                final_rank_std=("final_rank", "std"),
                final_rel_error_mean=("final_rel_error", "mean"),
                final_rel_error_std=("final_rel_error", "std"),
            )
            .reset_index()
        )
        for col in ("final_rank_std", "final_rel_error_std"):
            if col in agg.columns:
                agg[col] = agg[col].fillna(0.0)
        agg = agg.sort_values(config_cols)

        for _, row in agg.iterrows():
            if method_key == "ALS":
                setting = r"\(" + f"R={int(row['rank_init'])}" + r"\)"
            elif method_key == "ALS-TOPK":
                setting = r"\(" + f"R={int(row['rank_init'])},\\ K={int(row['top_k'])}" + r"\)"
            elif method_key in ("ALS-LR (ES)", "ALS-LR (BF)"):
                setting = (
                    r"\("
                    + f"R_{{\\max}}={int(row['rank_init'])},\\ \\mu={row['prune_mu']:.3g}"
                    + r"\)"
                )
            elif method_key == "TREL1":
                setting = (
                    r"\("
                    + f"R_{{init}}={int(row['rank_init'])},\\ \\lambda={row['trel1_lam']:.3g}"
                    + r"\)"
                )
            else:
                shots = int(row["qaoa_shots_effective"])
                setting = (
                    r"\("
                    + f"R_{{\\max}}={int(row['rank_init'])},\\ \\mu={row['prune_mu']:.3g},\\ shots={shots}"
                    + r"\)"
                )
            rows_full.append(
                {
                    "method": method_label,
                    "setting": setting,
                    "rank": _fmt_mean_std(row["final_rank_mean"], row["final_rank_std"], "{:.2f}"),
                    "error": _fmt_mean_std(row["final_rel_error_mean"], row["final_rel_error_std"], "{:.3e}"),
                }
            )

    table_df = pd.DataFrame(rows_full)
    table_df.to_csv(out_path.replace(".tex", ".csv"), index=False)

    rank_note = f" (rank init {rank_init})" if rank_init is not None else ""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Synthetic benchmark: selected rank \(\hat R\) and relative error (mean \(\pm\) std "
        rf"over \(20\) seeds), all configurations{rank_note}.}}",
        r"\label{tab:results_synth_all}",
        r"\begin{tabular}{@{}llcc@{}}",
        r"\toprule",
        r"Method & Setting & \(\hat R\) & Rel.\ error \(\downarrow\) \\ \midrule",
    ]

    for row in rows_full:
        lines.append(f"{row['method']} & {row['setting']} & {row['rank']} & {row['error']} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[info] wrote {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot synthetic CP experiment results.")
    parser.add_argument("--results-csv", type=str, default="results/v1/synthetic_experiment.csv")
    parser.add_argument("--summary-csv", type=str, default="results/v1/synthetic_experiment_summary.csv")
    parser.add_argument("--fig-dir", type=str, default="reports/figures/v1")
    parser.add_argument("--tables-dir", type=str, default="reports/tables/v1")
    parser.add_argument("--dataset-tag", type=str, default=None)
    parser.add_argument("--rank-init", type=int, default=None)
    parser.add_argument("--paper-output", action="store_true")
    parser.add_argument("--paper-fig-dir", type=str, default="reports/figures/v1/result")
    parser.add_argument("--paper-table-path", type=str, default="reports/tables/v1/synthetic_results_main.tex")
    return parser.parse_args()


def _unique_pairs(df: pd.DataFrame) -> List[Tuple[str, int]]:
    if df.empty or "dataset_tag" not in df.columns or "rank_init" not in df.columns:
        return []
    pairs = df[["dataset_tag", "rank_init"]].dropna().drop_duplicates()
    return [(str(row["dataset_tag"]), int(row["rank_init"])) for _, row in pairs.iterrows()]


def main() -> None:
    args = parse_args()
    ensure_dir(args.fig_dir)
    ensure_dir(args.tables_dir)

    df = load_csv(args.results_csv)
    summary = load_csv(args.summary_csv)

    if df.empty and summary.empty:
        print("[warn] no data loaded; aborting.")
        return

    dataset_tags = []
    if "dataset_tag" in df.columns:
        dataset_tags = df["dataset_tag"].dropna().unique().tolist()
    if not dataset_tags and "dataset_tag" in summary.columns:
        dataset_tags = summary["dataset_tag"].dropna().unique().tolist()
    if args.dataset_tag is not None:
        dataset_tags = [t for t in dataset_tags if t == args.dataset_tag]
    if not dataset_tags:
        dataset_tags = ["synthetic"]
    if len(dataset_tags) > 1:
        print("[warn] multiple dataset tags found; using the first.")
    dataset_tag = dataset_tags[0]
    tag = sanitize_tag(dataset_tag)

    df_sub = df[df["dataset_tag"] == dataset_tag] if not df.empty else pd.DataFrame()
    summary_tag = summary[summary["dataset_tag"] == dataset_tag] if not summary.empty else pd.DataFrame()

    if summary_tag.empty and df_sub.empty:
        print("[warn] no rows for selected dataset tag.")
        return

    main_cfg = select_main_config(summary_tag)
    rank_init = args.rank_init
    if rank_init is None:
        rank_init = _select_rank_for_method(summary_tag, "ALS")
    if rank_init is None and "rank_init" in df_sub.columns:
        rank_vals = sorted({int(v) for v in df_sub["rank_init"].dropna().unique()})
        if rank_vals:
            rank_init = rank_vals[-1]
    if rank_init is not None:
        main_cfg["rank_init"] = float(rank_init)

    table_path = os.path.join(args.tables_dir, f"synthetic_results_{tag}.tex")
    build_all_results_table(summary_tag, table_path)

    df_conv = _normalize_method_names(df_sub.copy())
    conv_methods = CONVERGENCE_METHODS_BY_TAG.get(dataset_tag, ["ALS", QALR_METHOD])
    df_conv = df_conv[df_conv["method"].isin(conv_methods)]

    als_rank_target = CONVERGENCE_ALS_RANK_BY_TAG.get(dataset_tag, rank_init)
    qaoa_rank_target = CONVERGENCE_QAOA_RANK_BY_TAG.get(dataset_tag, als_rank_target)

    if als_rank_target is not None and "rank_init" in df_conv.columns:
        df_conv = df_conv[
            ((df_conv["method"] == "ALS") & (df_conv["rank_init"] == float(als_rank_target)))
            | (df_conv["method"] != "ALS")
        ]

    lr_mu_target = CONVERGENCE_LR_MU_BY_TAG.get(dataset_tag)
    if lr_mu_target is not None and "prune_mu" in df_conv.columns:
        mu_vals = df_conv["prune_mu"].to_numpy(dtype=float)
        is_lr = df_conv["method"].isin(["ALS-LR (ES)", "ALS-LR (BF)"])
        df_conv = df_conv[~is_lr | np.isclose(mu_vals, float(lr_mu_target), atol=1e-12)]

    if dataset_tag in CONVERGENCE_QAOA_MU_BY_TAG:
        mu_targets = CONVERGENCE_QAOA_MU_BY_TAG[dataset_tag]
        if "prune_mu" in df_conv.columns:
            mu_vals = df_conv["prune_mu"].to_numpy(dtype=float)
            mu_mask = np.zeros_like(mu_vals, dtype=bool)
            for mu in mu_targets:
                mu_mask |= np.isclose(mu_vals, float(mu), atol=1e-12)
            df_conv = df_conv[(df_conv["method"] != QALR_METHOD) | mu_mask]

    shots_target = CONVERGENCE_QAOA_SHOTS_BY_TAG.get(dataset_tag)
    if shots_target is not None and "qaoa_shots_effective" in df_conv.columns:
        shot_vals = df_conv["qaoa_shots_effective"].to_numpy(dtype=float)
        df_conv = df_conv[
            (df_conv["method"] != QALR_METHOD)
            | np.isclose(shot_vals, float(shots_target), atol=1e-12)
        ]

    if qaoa_rank_target is not None and "rank_init" in df_conv.columns:
        df_conv = df_conv[
            (df_conv["method"] == "ALS")
            | (df_conv["rank_init"] == float(qaoa_rank_target))
        ]

    fig_path = os.path.join(args.fig_dir, f"synthetic_convergence_{tag}.png")
    plot_convergence(df_conv, fig_path, main_cfg=None)

    ranktraj_path = os.path.join(
        args.fig_dir,
        f"synthetic_ranktraj_{tag}_R20_shots4096.png",
    )
    plot_rank_trajectory(
        df_sub,
        ranktraj_path,
        mu_targets=CONVERGENCE_QAOA_MU_BY_TAG.get(dataset_tag),
        shots_target=CONVERGENCE_QAOA_SHOTS_BY_TAG.get(dataset_tag),
        rank_init_target=CONVERGENCE_QAOA_RANK_BY_TAG.get(dataset_tag, 20),
        max_prune_call=RANKTRAJ_MAX_PRUNE_CALL_BY_TAG.get(dataset_tag),
    )


if __name__ == "__main__":
    main()
