"""
Visualization for kinetic fluorescence (X60t) CP pruning experiments.

Consumes CSV outputs from `kinetic_experiment.py` and produces:
  - Convergence curves for selected methods (mean +/- std across seeds)
  - Rank trajectories for QALR-CPD pruning calls
  - Paper-ready summary table
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
ALS_LR_ES_METHOD = "ALS-LR (ES)"
ALS_LR_BF_METHOD = "ALS-LR (BF)"

METHOD_LABELS = {
    "ALS": "ALS",
    "ALS-TOPK": "ALS w/ Top-K",
    ALS_LR_ES_METHOD: "ALS-LR (ES)",
    ALS_LR_BF_METHOD: "ALS-LR (BF)",
    QALR_METHOD: "QALR-CPD (ours)",
    LEGACY_QALR_METHOD: "QALR-CPD (ours)",
}

STYLE_MAP = {
    "ALS": {"color": "black", "linestyle": "-", "marker": "*"},
    "ALS-TOPK": {"color": "C1", "linestyle": "--", "marker": "s"},
    ALS_LR_ES_METHOD: {"color": "C2", "linestyle": "--", "marker": "x"},
    ALS_LR_BF_METHOD: {"color": "C3", "linestyle": "-.", "marker": "+"},
    QALR_METHOD: {"color": "C0", "linestyle": "-.", "marker": "o"},
    LEGACY_QALR_METHOD: {"color": "C0", "linestyle": "-.", "marker": "o"},
}

CONVERGENCE_ALS_RANKS = (6, 10)
CONVERGENCE_TOPK = (10, 4)
CONVERGENCE_QAOA_RMAX = 16
CONVERGENCE_QAOA_MU = 0.2
CONVERGENCE_QAOA_SHOTS = 4096
PLOT_MAX_SWEEP = 200
PAPER_CONVERGENCE_NAME = "x60t_convergence_placeholder.pdf"
PAPER_RANKTRAJ_NAME = "covid_ranktraj_placeholder.pdf"
ALS_RANK_BY_TAG = {"I64x12x10x60_R0_rho0.0_noise0.0": 20}
RANK_TRAJECTORY_MAX_CALL_BY_TAG = {"I64x12x10x60_R0_rho0.0_noise0.0": 4}
QALR_MU_BY_TAG = {"I64x12x10x60_R0_rho0.0_noise0.0": (0.1,)}
QALR_RMAX_BY_TAG = {"I64x12x10x60_R0_rho0.0_noise0.0": 20}
QALR_SHOTS_BY_TAG = {"I64x12x10x60_R0_rho0.0_noise0.0": 4096}
PLAIN_LEGEND_TAGS = {"I64x12x10x60_R0_rho0.0_noise0.0"}
RANK_TRAJECTORY_METHODS_BY_TAG = {
    "I64x12x10x60_R0_rho0.0_noise0.0": [ALS_LR_ES_METHOD, ALS_LR_BF_METHOD, QALR_METHOD]
}
RANK_TRAJECTORY_MU_TARGET_BY_TAG = {"I64x12x10x60_R0_rho0.0_noise0.0": 0.1}
RANK_TRAJECTORY_LABELS_BY_TAG = {
    "I64x12x10x60_R0_rho0.0_noise0.0": {
        ALS_LR_ES_METHOD: "ALS-LR (ES)",
        ALS_LR_BF_METHOD: "ALS-LR (BF)",
        QALR_METHOD: "QALR-CPD",
    }
}
RANK_TRAJECTORY_OUTPUT_BY_TAG = {
    "I64x12x10x60_R0_rho0.0_noise0.0": "kinetic_rank_trajectory_I64x12x10x60_R0_rho0.0_noise0.0_mu1_shots4096_mu0p1_shots4096.png"
}


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


def _filter_status_ok(df: pd.DataFrame) -> pd.DataFrame:
    if "status" in df.columns:
        return df[df["status"] == "ok"].copy()
    return df.copy()


def _normalize_method_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "method" in df.columns:
        df["method"] = df["method"].replace({LEGACY_QALR_METHOD: QALR_METHOD})
    return df


def _config_value(config: Optional[Dict[str, float]], key: str) -> Optional[float]:
    if config is None or key not in config:
        return None
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return float(value)


def _qaoa_marker_map(mu_values: np.ndarray) -> Dict[float, str]:
    qaoa_markers = ["o", "^", "s", "D", "v", "P"]
    qaoa_mu_vals = sorted(v for v in mu_values if not np.isnan(v))
    return {val: qaoa_markers[i % len(qaoa_markers)] for i, val in enumerate(qaoa_mu_vals)}


def _select_best_group(
    summary: pd.DataFrame,
    method: str,
    group_cols: List[str],
) -> Optional[Dict[str, float]]:
    sub = summary[summary["method"] == method].copy()
    if sub.empty:
        return None
    for col in group_cols:
        if col not in sub.columns:
            sub[col] = np.nan
    if "final_rel_error" in sub.columns:
        sub = sub[sub["final_rel_error"].notna()]
    if sub.empty:
        return None
    agg = sub.groupby(group_cols, dropna=False)["final_rel_error"].mean().reset_index()
    if agg.empty:
        return None
    agg = agg.sort_values("final_rel_error", ascending=True)
    best = agg.iloc[0]
    return {col: float(best[col]) for col in group_cols}


def _best_paper_configs(summary: pd.DataFrame) -> Dict[str, Optional[Dict[str, float]]]:
    if summary.empty:
        return {}
    summary = _normalize_method_names(summary)
    summary = _filter_status_ok(summary)
    for col in ("top_k", "prune_mu", "qaoa_shots_effective", "rank_init"):
        if col not in summary.columns:
            summary[col] = np.nan
    return {
        "ALS": _select_best_group(summary, "ALS", ["rank_init"]),
        "ALS-TOPK": _select_best_group(summary, "ALS-TOPK", ["rank_init", "top_k"]),
        ALS_LR_ES_METHOD: _select_best_group(summary, ALS_LR_ES_METHOD, ["rank_init", "prune_mu"]),
        ALS_LR_BF_METHOD: _select_best_group(summary, ALS_LR_BF_METHOD, ["rank_init", "prune_mu"]),
        QALR_METHOD: _select_best_group(
            summary,
            QALR_METHOD,
            ["rank_init", "prune_mu", "qaoa_shots_effective"],
        ),
    }


def _filter_by_value(series: pd.Series, value: float) -> pd.Series:
    if value is None or np.isnan(value):
        return series.isna()
    vals = series.to_numpy(dtype=float)
    return np.isclose(vals, float(value), atol=1e-12)


def _prepare_group_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "prune_mu" not in df.columns:
        df["prune_mu"] = np.nan
    if "qaoa_shots_effective" not in df.columns:
        df["qaoa_shots_effective"] = np.nan
    df["prune_mu_key"] = df["prune_mu"].fillna(-1.0)
    df["qaoa_shots_key"] = df["qaoa_shots_effective"].fillna(-1)
    return df


def _filter_qaoa_config(
    df: pd.DataFrame, config: Optional[Dict[str, float]]
) -> pd.DataFrame:
    if config is None:
        return df
    df = df.copy()
    for key in ("rank_init", "prune_mu", "qaoa_shots_effective"):
        val = _config_value(config, key)
        if val is None:
            continue
        if key not in df.columns:
            df[key] = np.nan
        df = df[_filter_by_value(df[key], val)]
    return df


def _filter_override_curves(df: pd.DataFrame, dataset_tag: str) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []

    als_rank = ALS_RANK_BY_TAG.get(dataset_tag)
    if als_rank is not None:
        als_df = df[(df["method"] == "ALS") & _filter_by_value(df["rank_init"], als_rank)].copy()
        parts.append(als_df)

    qaoa_mu = QALR_MU_BY_TAG.get(dataset_tag)
    if qaoa_mu:
        qaoa_df = df[df["method"] == QALR_METHOD].copy()
        mu_vals = qaoa_df["prune_mu"].to_numpy(dtype=float)
        mu_mask = np.zeros_like(mu_vals, dtype=bool)
        for mu in qaoa_mu:
            mu_mask |= np.isclose(mu_vals, float(mu), atol=1e-12)
        qaoa_df = qaoa_df[mu_mask]
        qaoa_rmax = QALR_RMAX_BY_TAG.get(dataset_tag, CONVERGENCE_QAOA_RMAX)
        qaoa_shots = QALR_SHOTS_BY_TAG.get(dataset_tag, CONVERGENCE_QAOA_SHOTS)
        if "rank_init" in qaoa_df.columns:
            qaoa_df = qaoa_df[_filter_by_value(qaoa_df["rank_init"], qaoa_rmax)]
        if "qaoa_shots_effective" in qaoa_df.columns:
            qaoa_df = qaoa_df[
                _filter_by_value(qaoa_df["qaoa_shots_effective"], qaoa_shots)
            ]
        parts.append(qaoa_df)

        for method in (ALS_LR_ES_METHOD, ALS_LR_BF_METHOD):
            lr_df = df[df["method"] == method].copy()
            if lr_df.empty:
                continue
            mu_vals = lr_df["prune_mu"].to_numpy(dtype=float)
            mu_mask = np.zeros_like(mu_vals, dtype=bool)
            for mu in qaoa_mu:
                mu_mask |= np.isclose(mu_vals, float(mu), atol=1e-12)
            lr_df = lr_df[mu_mask]
            qaoa_rmax = QALR_RMAX_BY_TAG.get(dataset_tag, CONVERGENCE_QAOA_RMAX)
            if "rank_init" in lr_df.columns:
                lr_df = lr_df[_filter_by_value(lr_df["rank_init"], qaoa_rmax)]
            parts.append(lr_df)

    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


def _filter_by_config_map(
    df: pd.DataFrame,
    config_map: Optional[Dict[str, Optional[Dict[str, float]]]],
) -> pd.DataFrame:
    if df.empty or not config_map:
        return df
    keep_idx = []
    for method, config in config_map.items():
        if config is None:
            continue
        sub = df[df["method"] == method].copy()
        if sub.empty:
            continue
        for col, value in config.items():
            if col not in sub.columns:
                sub[col] = np.nan
            sub = sub[_filter_by_value(sub[col], value)]
        if not sub.empty:
            keep_idx.append(sub.index)
    if not keep_idx:
        return df.iloc[0:0]
    idx = keep_idx[0]
    for extra in keep_idx[1:]:
        idx = idx.union(extra)
    return df.loc[idx]


def _stats_for_config(
    summary: pd.DataFrame,
    method: str,
    config: Optional[Dict[str, float]],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if config is None:
        return None, None, None, None
    sub = summary[summary["method"] == method].copy()
    if sub.empty:
        return None, None, None, None
    for col, value in config.items():
        if col not in sub.columns:
            sub[col] = np.nan
        sub = sub[_filter_by_value(sub[col], value)]
    rank_mean, rank_std = _mean_std(sub.get("final_rank", pd.Series(dtype=float)))
    err_mean, err_std = _mean_std(sub.get("final_rel_error", pd.Series(dtype=float)))
    return rank_mean, rank_std, err_mean, err_std


def _format_float_tag(value: float) -> str:
    if value is None or np.isnan(value):
        return "na"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def _fmt_sci_tex(value: Optional[float], digits: int = 4) -> str:
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


def _filter_convergence_defaults(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _normalize_method_names(df)
    for col in ("top_k", "prune_mu", "qaoa_shots_effective"):
        if col not in df.columns:
            df[col] = np.nan

    als_mask = (df["method"] == "ALS") & df["rank_init"].isin(CONVERGENCE_ALS_RANKS)
    topk_mask = df["method"] == "ALS-TOPK"
    if len(CONVERGENCE_TOPK) == 2:
        topk_mask &= df["rank_init"] == float(CONVERGENCE_TOPK[0])
        topk_mask &= df["top_k"] == float(CONVERGENCE_TOPK[1])
    lr_mask = df["method"].isin([ALS_LR_ES_METHOD, ALS_LR_BF_METHOD])
    if lr_mask.any():
        mu_vals = df["prune_mu"].to_numpy(dtype=float)
        lr_mask &= np.isclose(mu_vals, CONVERGENCE_QAOA_MU, atol=1e-12)
        lr_mask &= df["rank_init"] == float(CONVERGENCE_QAOA_RMAX)
    qaoa_mask = df["method"] == QALR_METHOD
    if qaoa_mask.any():
        mu_vals = df["prune_mu"].to_numpy(dtype=float)
        shots_vals = df["qaoa_shots_effective"].to_numpy(dtype=float)
        qaoa_mask &= np.isclose(mu_vals, CONVERGENCE_QAOA_MU, atol=1e-12)
        qaoa_mask &= df["rank_init"] == float(CONVERGENCE_QAOA_RMAX)
        qaoa_mask &= np.isclose(shots_vals, CONVERGENCE_QAOA_SHOTS, atol=1e-12)
    return df[als_mask | topk_mask | lr_mask | qaoa_mask]


def plot_convergence(
    df: pd.DataFrame,
    out_path: str,
    config_map: Optional[Dict[str, Optional[Dict[str, float]]]] = None,
) -> None:
    if df.empty:
        print("[warn] convergence plot skipped (no data).")
        return

    df = df.copy()
    for col in ("top_k", "prune_mu", "qaoa_shots_effective"):
        if col not in df.columns:
            df[col] = np.nan
    df = _normalize_method_names(df)
    df_all = df.copy()

    dataset_tag = None
    if "dataset_tag" in df_all.columns:
        tags = [str(t) for t in df_all["dataset_tag"].dropna().unique()]
        if len(tags) == 1:
            dataset_tag = tags[0]
    use_override = dataset_tag in ALS_RANK_BY_TAG or dataset_tag in QALR_MU_BY_TAG
    plain_legend = dataset_tag in PLAIN_LEGEND_TAGS

    df = df[df["method"].isin(["ALS", "ALS-TOPK", ALS_LR_ES_METHOD, ALS_LR_BF_METHOD, QALR_METHOD])]
    if df.empty:
        print("[warn] convergence plot skipped (no ALS/QALR rows).")
        return

    if config_map:
        df = _filter_by_config_map(df, config_map)
        if df.empty:
            print("[warn] convergence plot skipped (no data for selected configs).")
            return
    else:
        if use_override:
            df = _filter_override_curves(df_all, dataset_tag)
        else:
            df = _filter_convergence_defaults(df)
        if df.empty:
            print("[warn] convergence plot skipped (no matching defaults).")
            return

    agg = (
        df.groupby(
            ["method", "rank_init", "top_k", "prune_mu", "qaoa_shots_effective", "sweep"],
            dropna=False,
        )
        .agg(rel_err_mean=("rel_error", "mean"), rel_err_std=("rel_error", "std"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    xticks = [1, 40, 80, 120, 160, 200]
    line_width = 1.6
    marker_size = 4
    mark_every = 12
    fill_alpha = 0.08
    qaoa_marker_map = _qaoa_marker_map(df["prune_mu"].dropna().unique())

    method_order = ["ALS", "ALS-TOPK", ALS_LR_ES_METHOD, ALS_LR_BF_METHOD, QALR_METHOD]
    grouped = []
    for (method, r_init, top_k, mu, shots), sub in agg.groupby(
        ["method", "rank_init", "top_k", "prune_mu", "qaoa_shots_effective"],
        dropna=False,
    ):
        grouped.append((method, r_init, top_k, mu, shots, sub))
    grouped.sort(
        key=lambda item: (
            method_order.index(item[0]) if item[0] in method_order else len(method_order),
            item[1],
            item[2] if not np.isnan(item[2]) else -1.0,
        )
    )

    for method, r_init, top_k, _mu, _shots, sub in grouped:
        sub = sub.sort_values("sweep")
        if plain_legend:
            if method == "ALS":
                label = "ALS"
            elif method in (ALS_LR_ES_METHOD, ALS_LR_BF_METHOD):
                label = METHOD_LABELS.get(method, method)
            elif method == QALR_METHOD:
                label = "QALR-CPD"
            else:
                label = METHOD_LABELS.get(method, method)
        elif method == "ALS" and dataset_tag in ALS_RANK_BY_TAG and not np.isnan(r_init):
            label = "ALS"
        elif method in (ALS_LR_ES_METHOD, ALS_LR_BF_METHOD) and not np.isnan(_mu):
            label = f"{method} ($\\mu={_mu:g}$)"
        elif method == QALR_METHOD and not np.isnan(_mu):
            if np.isclose(float(_mu), 0.8, atol=1e-12):
                label = f"QALR-CPR ($\\mu={_mu:g}$)"
            else:
                label = f"QALR-CPD ($\\mu={_mu:g}$)"
        else:
            label = METHOD_LABELS.get(method, method)
        style = dict(STYLE_MAP.get(method, {}))
        if method == QALR_METHOD and not np.isnan(_mu):
            style["marker"] = qaoa_marker_map.get(float(_mu), style.get("marker", "o"))
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

    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("Rel. error", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(labelsize=9)
    if not agg.empty:
        max_sweep = int(agg["sweep"].max())
        plot_max_sweep = PLOT_MAX_SWEEP if PLOT_MAX_SWEEP else max_sweep
        xpad = 1.5
        ax.set_xlim(1 - xpad, plot_max_sweep + xpad)
        ticks = [t for t in xticks if t <= plot_max_sweep]
        if plot_max_sweep not in ticks:
            ticks.append(plot_max_sweep)
        ax.set_xticks(sorted(ticks))

    ax.set_ylim(0.02, 0.17)
    ax.legend(title=None, fontsize=8, framealpha=0.9, loc="upper right")
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=1000)
    print(f"[info] wrote {out_path}")
    plt.close(fig)


def _rank_agg_from_subset(
    sub: pd.DataFrame, max_prune_call: Optional[int] = None
) -> Optional[pd.DataFrame]:
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
    if max_prune_call is not None:
        agg = agg[agg["prune_call"] <= int(max_prune_call)].copy()
    if agg.empty:
        return None
    return agg


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
    qaoa = df[df["method"] == QALR_METHOD].copy()
    if qaoa.empty:
        print("[warn] rank trajectory plot skipped (no QAOA rows).")
        return

    qaoa = _prepare_group_keys(qaoa)
    qaoa = _filter_qaoa_config(qaoa, main_cfg)
    if qaoa.empty:
        print("[warn] rank trajectory plot skipped (no QAOA rows after filter).")
        return

    if rank_init_target is not None and "rank_init" in qaoa.columns:
        qaoa = qaoa[_filter_by_value(qaoa["rank_init"], float(rank_init_target))]
        if qaoa.empty:
            print("[warn] rank trajectory plot skipped (no QAOA rows for requested rank).")
            return
    if shots_target is not None and "qaoa_shots_effective" in qaoa.columns:
        qaoa = qaoa[_filter_by_value(qaoa["qaoa_shots_effective"], float(shots_target))]
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

    ext = ".png"
    out_path_prefix = out_path
    if out_path.lower().endswith(".png"):
        ext = ".png"
        out_path_prefix = out_path[: -len(ext)]
    elif out_path.lower().endswith(".pdf"):
        ext = ".pdf"
        out_path_prefix = out_path[: -len(ext)]

    if mu_targets is not None and len(mu_targets) > 1:
        markers = ["o", "^", "s", "D", "P", "X"]
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(mu_targets))))
        mu_color = {float(mu): colors[i % len(colors)] for i, mu in enumerate(mu_targets)}

        fig, ax = plt.subplots(figsize=(3.0, 2.5))
        all_calls: List[int] = []
        y_min = None
        y_max = None

        for idx, mu in enumerate(mu_targets):
            sub = qaoa[np.isclose(qaoa["prune_mu"].to_numpy(dtype=float), float(mu), atol=1e-12)]
            agg = _rank_agg_from_subset(sub, max_prune_call=max_prune_call)
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
                label=f"QALR-CPD ($\\mu={float(mu):g}$)",
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
        xpad = 0.5
        ax.set_xlim(x_min - xpad, x_max + xpad)
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

    groups: List[Tuple[Optional[float], Optional[float], pd.DataFrame, str]] = []
    if main_cfg:
        groups = [(None, None, qaoa, out_path)]
    else:
        for (mu_key, shots_key), sub in qaoa.groupby(["prune_mu_key", "qaoa_shots_key"]):
            mu_tag = _format_float_tag(float(mu_key))
            shots_tag = _format_float_tag(float(shots_key))
            group_path = f"{out_path_prefix}_mu{mu_tag}_shots{shots_tag}{ext}"
            groups.append((mu_key, shots_key, sub, group_path))

    for _, _, sub, group_path in groups:
        agg = _rank_agg_from_subset(sub, max_prune_call=max_prune_call)
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
        x_min = int(agg["prune_call"].min())
        x_max = int(agg["prune_call"].max())
        xpad = 0.5
        ax.set_xlim(x_min - xpad, x_max + xpad)
        ax.set_xticks(sorted(int(x) for x in agg["prune_call"].unique()))

        y_min = float((agg["rank_mean"] - rank_std).min())
        y_max = float((agg["rank_mean"] + rank_std).max())
        y_pad = max(0.5, 0.05 * (y_max - y_min))
        ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)

        ax.legend(title=None, fontsize=9, framealpha=0.9, loc="best")
        fig.tight_layout(pad=0.3)
    fig.savefig(group_path, dpi=1000)
    print(f"[info] wrote {group_path}")
    plt.close(fig)


def plot_rank_trajectory_methods(
    df: pd.DataFrame,
    out_path: str,
    methods: Sequence[str],
    mu_target: Optional[float] = None,
    rank_init_target: Optional[int] = None,
    shots_target: Optional[int] = None,
    max_prune_call: Optional[int] = None,
    label_overrides: Optional[Dict[str, str]] = None,
) -> None:
    if df.empty:
        print("[warn] rank trajectory plot skipped (no data).")
        return

    df = _normalize_method_names(df)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    all_calls: List[int] = []
    y_min = np.inf
    y_max = -np.inf

    for method in methods:
        sub = df[df["method"] == method].copy()
        if sub.empty:
            continue
        if mu_target is not None and "prune_mu" in sub.columns:
            mu_vals = sub["prune_mu"].to_numpy(dtype=float)
            sub = sub[np.isclose(mu_vals, float(mu_target), atol=1e-12)]
        if rank_init_target is not None and "rank_init" in sub.columns:
            r_vals = sub["rank_init"].to_numpy(dtype=float)
            sub = sub[np.isclose(r_vals, float(rank_init_target), atol=1e-12)]
        if (
            method == QALR_METHOD
            and shots_target is not None
            and "qaoa_shots_effective" in sub.columns
        ):
            shots_vals = sub["qaoa_shots_effective"].to_numpy(dtype=float)
            sub = sub[np.isclose(shots_vals, float(shots_target), atol=1e-12)]
        if sub.empty:
            continue

        agg = _rank_agg_from_subset(sub, max_prune_call=max_prune_call)
        if agg is None or agg.empty:
            continue

        style = dict(STYLE_MAP.get(method, {}))
        color = style.get("color", "C0")
        linestyle = style.get("linestyle", "-")
        marker = style.get("marker", "o")
        if label_overrides and method in label_overrides:
            label = label_overrides[method]
        else:
            base_label = METHOD_LABELS.get(method, method)
            if mu_target is None:
                label = base_label
            else:
                label = f"{base_label} ($\\mu={mu_target:g}$)"

        ax.plot(
            agg["prune_call"],
            agg["rank_mean"],
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.6,
            markersize=4,
            label=label,
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

        all_calls.extend(agg["prune_call"].tolist())
        y_min = min(y_min, float((agg["rank_mean"] - rank_std).min()))
        y_max = max(y_max, float((agg["rank_mean"] + rank_std).max()))

    if not all_calls:
        print("[warn] rank trajectory plot skipped (no prune sweeps).")
        return

    ax.set_xlabel("Pruning call", fontsize=10)
    ax.set_ylabel("Selected rank", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(labelsize=9)
    x_min = int(min(all_calls))
    x_max = int(max(all_calls))
    xpad = 0.5
    ax.set_xlim(x_min - xpad, x_max + xpad)
    ax.set_xticks(sorted(set(int(x) for x in all_calls)))

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        y_min = float(df["rank"].min())
        y_max = float(df["rank"].max())
    y_pad = max(0.5, 0.05 * (y_max - y_min))
    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)

    ax.legend(title=None, fontsize=9, framealpha=0.9, loc="best")
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=1000)
    print(f"[info] wrote {out_path}")
    plt.close(fig)


def build_table(
    summary: pd.DataFrame,
    out_path: str,
    caption: str,
    label: str,
    all_rows: bool = False,
) -> None:
    if summary.empty:
        print("[warn] summary table skipped (no data).")
        return

    summary = _normalize_method_names(summary)
    summary = _filter_status_ok(summary)
    for col in ("top_k", "prune_mu", "qaoa_shots_effective", "rank_init"):
        if col not in summary.columns:
            summary[col] = np.nan

    if not all_rows:
        configs = _best_paper_configs(summary)

        rows = [
            ("ALS", r"ALS"),
            ("ALS-TOPK", r"ALS w/ Top-\(K\)"),
            (ALS_LR_ES_METHOD, r"ALS-LR (ES)"),
            (ALS_LR_BF_METHOD, r"ALS-LR (BF)"),
            (QALR_METHOD, r"QALR-CPD (ours)"),
        ]

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\begin{tabular}{@{}lcc@{}}",
            r"\toprule",
            r"Method & \(\hat R\) & Rel.\ error \(\downarrow\) \\",
            r"\midrule",
        ]

        for method_key, label_text in rows:
            rank_mean, rank_std, err_mean, err_std = _stats_for_config(
                summary,
                method_key,
                configs.get(method_key),
            )
            rank_fmt = _fmt_rank_pm(rank_mean, rank_std)
            err_fmt = _fmt_err_pm(err_mean, err_std)
            lines.append(f"{label_text} & {rank_fmt} & {err_fmt} \\\\")

        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[info] wrote {out_path}")
        return

    rows_full: List[Dict[str, str]] = []
    method_specs = [
        ("ALS", "ALS", ["rank_init"]),
        ("ALS-TOPK", r"ALS w/ Top-\(K\)", ["rank_init", "top_k"]),
        (ALS_LR_ES_METHOD, "ALS-LR (ES)", ["rank_init", "prune_mu"]),
        (ALS_LR_BF_METHOD, "ALS-LR (BF)", ["rank_init", "prune_mu"]),
        (QALR_METHOD, "QALR-CPD", ["rank_init", "prune_mu", "qaoa_shots_effective"]),
    ]

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
            elif method_key in (ALS_LR_ES_METHOD, ALS_LR_BF_METHOD):
                setting = r"\(" + f"R_{{\\max}}={int(row['rank_init'])},\\ \\mu={row['prune_mu']:.3g}" + r"\)"
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
                    "rank": _fmt_rank_pm(row["final_rank_mean"], row["final_rank_std"]),
                    "error": _fmt_err_pm(row["final_rel_error_mean"], row["final_rel_error_std"]),
                }
            )

    table_df = pd.DataFrame(rows_full)
    table_df.to_csv(out_path.replace(".tex", ".csv"), index=False)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption} (all configurations).}}",
        rf"\label{{{label}}}",
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
    parser = argparse.ArgumentParser(description="Plot X60t experiment results.")
    parser.add_argument("--results-csv", type=str, default="results/kinetic_experiment.csv")
    parser.add_argument("--summary-csv", type=str, default="results/kinetic_experiment_summary.csv")
    parser.add_argument("--fig-dir", type=str, default="figs")
    parser.add_argument("--tables-dir", type=str, default="tables")
    parser.add_argument("--dataset-tag", type=str, default=None)
    parser.add_argument("--paper-output", action="store_true")
    parser.add_argument("--paper-fig-dir", type=str, default="figure/result")
    parser.add_argument("--paper-table-path", type=str, default="tables/kinetic_results_paper.tex")
    return parser.parse_args()


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
        dataset_tags = ["x60t"]

    paper_tag = None
    if args.paper_output:
        if len(dataset_tags) > 1:
            print("[warn] multiple dataset tags found; using the first for paper outputs.")
        paper_tag = dataset_tags[0]

    for dataset_tag in dataset_tags:
        tag = sanitize_tag(dataset_tag)
        df_sub = df[df["dataset_tag"] == dataset_tag] if not df.empty else pd.DataFrame()
        summary_sub = summary[summary["dataset_tag"] == dataset_tag] if not summary.empty else pd.DataFrame()
        paper_configs = _best_paper_configs(summary_sub) if args.paper_output else None

        fig_path = os.path.join(args.fig_dir, f"kinetic_convergence_{tag}.png")
        plot_convergence(df_sub, fig_path)

        rank_path = os.path.join(args.fig_dir, f"kinetic_rank_trajectory_{tag}.png")
        mu_targets = None
        shots_target = None
        rank_init_target = None
        max_prune_call = RANK_TRAJECTORY_MAX_CALL_BY_TAG.get(dataset_tag)
        rank_methods = RANK_TRAJECTORY_METHODS_BY_TAG.get(dataset_tag)
        mu_target = RANK_TRAJECTORY_MU_TARGET_BY_TAG.get(dataset_tag)
        label_overrides = RANK_TRAJECTORY_LABELS_BY_TAG.get(dataset_tag)
        rank_output_override = RANK_TRAJECTORY_OUTPUT_BY_TAG.get(dataset_tag)
        if dataset_tag in QALR_MU_BY_TAG:
            mu_targets = list(QALR_MU_BY_TAG[dataset_tag])
            shots_target = QALR_SHOTS_BY_TAG.get(dataset_tag, CONVERGENCE_QAOA_SHOTS)
            rank_init_target = QALR_RMAX_BY_TAG.get(dataset_tag, CONVERGENCE_QAOA_RMAX)
            if rank_methods is None:
                rank_path = os.path.join(
                    args.fig_dir,
                    f"kinetic_rank_trajectory_{tag}_mu1_shots{shots_target}.png",
                )
        if rank_methods:
            if mu_target is None and mu_targets and len(mu_targets) == 1:
                mu_target = float(mu_targets[0])
            if rank_init_target is None:
                rank_init_target = QALR_RMAX_BY_TAG.get(dataset_tag, CONVERGENCE_QAOA_RMAX)
            if shots_target is None:
                shots_target = QALR_SHOTS_BY_TAG.get(dataset_tag, CONVERGENCE_QAOA_SHOTS)
            if rank_output_override:
                rank_path = os.path.join(args.fig_dir, rank_output_override)
            plot_rank_trajectory_methods(
                df_sub,
                rank_path,
                methods=rank_methods,
                mu_target=mu_target,
                rank_init_target=rank_init_target,
                shots_target=shots_target,
                max_prune_call=max_prune_call,
                label_overrides=label_overrides,
            )
        else:
            plot_rank_trajectory(
                df_sub,
                rank_path,
                mu_targets=mu_targets,
                shots_target=shots_target,
                rank_init_target=rank_init_target,
                max_prune_call=max_prune_call,
            )

        table_path = os.path.join(args.tables_dir, f"kinetic_results_{tag}.tex")
        build_table(
            summary_sub,
            table_path,
            caption="X60t kinetic fluorescence tensor: relative error and selected rank (mean \\(\\pm\\) std).",
            label=f"tab:results_{tag}",
            all_rows=True,
        )

        if args.paper_output and dataset_tag == paper_tag:
            ensure_dir(args.paper_fig_dir)
            paper_table_dir = os.path.dirname(args.paper_table_path)
            if paper_table_dir:
                ensure_dir(paper_table_dir)

            paper_conv = os.path.join(args.paper_fig_dir, PAPER_CONVERGENCE_NAME)
            paper_conv_cfg = None
            if paper_configs:
                paper_conv_cfg = {
                    "ALS": paper_configs.get("ALS"),
                    "ALS-TOPK": paper_configs.get("ALS-TOPK"),
                    ALS_LR_ES_METHOD: paper_configs.get(ALS_LR_ES_METHOD),
                    ALS_LR_BF_METHOD: paper_configs.get(ALS_LR_BF_METHOD),
                    QALR_METHOD: paper_configs.get(QALR_METHOD),
                }
            plot_convergence(df_sub, paper_conv, config_map=paper_conv_cfg)

            paper_rank = os.path.join(args.paper_fig_dir, PAPER_RANKTRAJ_NAME)
            plot_rank_trajectory(
                df_sub,
                paper_rank,
                mu_targets=list(QALR_MU_BY_TAG.get(dataset_tag, ())),
                shots_target=QALR_SHOTS_BY_TAG.get(dataset_tag),
                rank_init_target=QALR_RMAX_BY_TAG.get(dataset_tag),
            )

            build_table(
                summary_sub,
                args.paper_table_path,
                caption="X60t kinetic fluorescence tensor: relative error and selected rank (mean \\(\\pm\\) std).",
                label="tab:results_x60t",
            )


if __name__ == "__main__":
    main()
