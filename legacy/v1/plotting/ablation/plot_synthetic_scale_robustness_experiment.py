"""
Visualization for synthetic scale-robustness CP experiments.

Consumes CSV outputs from `synthetic_scale_robustness_experiment.py` and produces:
  - Relative error vs. QUBO scale exponent plot (mean +/- std across seeds)
  - Selected rank vs. QUBO scale exponent plot (mean +/- std across seeds)
  - LaTeX + CSV tables summarizing the sweep
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QALR_METHOD = "QALR-CPD"
LEGACY_QALR_METHOD = "ALS-QAOA"
ALS_LR_ES_METHOD = "ALS-LR (ES)"
ALS_LR_BF_METHOD = "ALS-LR (BF)"

METHOD_ORDER = ["ALS", "ALS-TOPK", ALS_LR_ES_METHOD, ALS_LR_BF_METHOD, QALR_METHOD]
PLOT_METHOD_ORDER = [QALR_METHOD]
PLOT_LABELS = {
    "ALS": "ALS",
    "ALS-TOPK": "Top-K",
    ALS_LR_ES_METHOD: "ALS-LR (ES)",
    ALS_LR_BF_METHOD: "ALS-LR (BF)",
    QALR_METHOD: "QALR-CPD",
    LEGACY_QALR_METHOD: "QALR-CPD",
}
STYLE_MAP = {
    "ALS": {"color": "black", "linestyle": "-", "marker": "*"},
    "ALS-TOPK": {"color": "C1", "linestyle": "--", "marker": "s"},
    ALS_LR_ES_METHOD: {"color": "C2", "linestyle": "--", "marker": "x"},
    ALS_LR_BF_METHOD: {"color": "C3", "linestyle": "-.", "marker": "+"},
    QALR_METHOD: {"color": "C0", "linestyle": "-.", "marker": "o"},
    LEGACY_QALR_METHOD: {"color": "C0", "linestyle": "-.", "marker": "o"},
}
TABLE_METHOD_LABELS = {
    "ALS": "ALS",
    "ALS-TOPK": r"Top-\(K\)",
    ALS_LR_ES_METHOD: "ALS-LR (ES)",
    ALS_LR_BF_METHOD: "ALS-LR (BF)",
    QALR_METHOD: "QALR-CPD",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[warn] missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


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


def _coerce_scale_exp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "qubo_scale_exp" in df.columns:
        df["qubo_scale_exp"] = pd.to_numeric(df["qubo_scale_exp"], errors="coerce")
        if df["qubo_scale_exp"].notna().any():
            return df

    if "qubo_scale_factor" in df.columns:
        factors = pd.to_numeric(df["qubo_scale_factor"], errors="coerce")
        safe = factors.where(factors > 0)
        df["qubo_scale_exp"] = np.log10(safe)
    else:
        df["qubo_scale_exp"] = np.nan
    return df


def _coerce_noise_std(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "noise_std" in df.columns:
        df["noise_std"] = pd.to_numeric(df["noise_std"], errors="coerce")
        if df["noise_std"].notna().any():
            return df

    if "dataset_tag" in df.columns:
        extracted = df["dataset_tag"].astype(str).str.extract(r"noise([0-9eE+.-]+)")
        df["noise_std"] = pd.to_numeric(extracted[0], errors="coerce")
        if df["noise_std"].notna().any():
            return df

    if "dataset_path" in df.columns:
        extracted = df["dataset_path"].astype(str).str.extract(r"noise_([0-9eE+.-]+)")
        df["noise_std"] = pd.to_numeric(extracted[0], errors="coerce")
    return df


def _coerce_corr_rho(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "corr_rho" in df.columns:
        df["corr_rho"] = pd.to_numeric(df["corr_rho"], errors="coerce")
        if df["corr_rho"].notna().any():
            return df

    if "dataset_tag" in df.columns:
        extracted = df["dataset_tag"].astype(str).str.extract(r"rho([0-9eE+.-]+)")
        df["corr_rho"] = pd.to_numeric(extracted[0], errors="coerce")
        if df["corr_rho"].notna().any():
            return df

    if "dataset_path" in df.columns:
        extracted = df["dataset_path"].astype(str).str.extract(r"rho_([0-9eE+.-]+)")
        df["corr_rho"] = pd.to_numeric(extracted[0], errors="coerce")
    return df


def _method_config_cols(method: str) -> List[str]:
    if method == "ALS":
        return ["rank_init"]
    if method == "ALS-TOPK":
        return ["rank_init", "top_k"]
    if method in (ALS_LR_ES_METHOD, ALS_LR_BF_METHOD):
        return ["rank_init", "prune_mu"]
    if method == QALR_METHOD:
        return ["rank_init", "prune_mu", "qaoa_shots_effective"]
    return []


def _select_best_config(
    df: pd.DataFrame,
    config_cols: Sequence[str],
) -> Optional[Dict[str, float]]:
    if df.empty or "final_rel_error" not in df.columns:
        return None
    sub = df.copy()
    for col in config_cols:
        if col not in sub.columns:
            sub[col] = np.nan
    sub = sub[sub["final_rel_error"].notna()]
    if sub.empty:
        return None
    agg = sub.groupby(list(config_cols), dropna=False)["final_rel_error"].mean().reset_index()
    if agg.empty:
        return None
    best = agg.sort_values("final_rel_error", ascending=True).iloc[0]
    config: Dict[str, float] = {}
    for col in config_cols:
        value = best[col]
        config[col] = float(value) if value == value else np.nan
    return config


def _filter_by_value(series: pd.Series, value: float) -> pd.Series:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return series.isna()
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return np.isclose(vals, float(value), atol=1e-12)


def _parse_float_list(raw: str) -> List[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def _resolve_scale_exps(
    available: Sequence[float],
    preferred: Optional[Sequence[float]],
) -> List[float]:
    available_sorted = sorted(float(v) for v in available)
    if not preferred:
        return available_sorted
    return [float(v) for v in preferred]


def _format_scale_raw(exp_val: float) -> str:
    if exp_val is None or (isinstance(exp_val, float) and np.isnan(exp_val)):
        return "--"
    return f"{float(exp_val):g}"


def _format_scale_tex(exp_val: float) -> str:
    if exp_val is None or (isinstance(exp_val, float) and np.isnan(exp_val)):
        return "--"
    exp_val = float(exp_val)
    if np.isclose(exp_val, round(exp_val)):
        exp_val = int(round(exp_val))
    return rf"$10^{{{exp_val}}}$"


def _fmt_pm_inner(mean: Optional[float], std: Optional[float], fmt: str = "{:.3g}") -> Optional[str]:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return None
    std_val = 0.0 if std is None or (isinstance(std, float) and np.isnan(std)) else std
    return fmt.format(float(mean)) + r" \pm " + fmt.format(float(std_val))


def _fmt_pm(mean: Optional[float], std: Optional[float], fmt: str = "{:.3g}") -> str:
    inner = _fmt_pm_inner(mean, std, fmt=fmt)
    if inner is None:
        return "--"
    return r"\(" + inner + r"\)"


def _format_rank_err_cell(
    rank_mean: Optional[float],
    rank_std: Optional[float],
    err_mean: Optional[float],
    err_std: Optional[float],
) -> str:
    r_inner = _fmt_pm_inner(rank_mean, rank_std, fmt="{:.3g}")
    e_inner = _fmt_pm_inner(err_mean, err_std, fmt="{:.3g}")
    if r_inner is None and e_inner is None:
        return "--"
    if r_inner is None:
        return r"\(" + r"\mathrm{Err}=" + e_inner + r"\)"
    if e_inner is None:
        return r"\(" + r"\hat R=" + r_inner + r"\)"
    return r"\(" + r"\hat R=" + r_inner + r",\ \mathrm{Err}=" + e_inner + r"\)"


def _apply_config_filter(df: pd.DataFrame, config: Dict[str, float]) -> pd.DataFrame:
    sub = df.copy()
    for col, value in config.items():
        if col not in sub.columns:
            return sub.iloc[0:0].copy()
        sub = sub[_filter_by_value(sub[col], value)]
    return sub


def build_scale_stats(
    summary: pd.DataFrame,
    method_order: Sequence[str],
    scale_exps: Optional[Sequence[float]],
    baseline_scale_exp: float,
    corr_rho: Optional[float],
    noise_std: Optional[float],
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Optional[Dict[str, float]]], List[float]]:
    df = _filter_status_ok(summary)
    df = _normalize_method_names(df)
    df = _coerce_scale_exp(df)
    df = _coerce_noise_std(df)
    df = _coerce_corr_rho(df)

    if df.empty:
        return pd.DataFrame(), {}, []

    if noise_std is not None and not (isinstance(noise_std, float) and np.isnan(noise_std)):
        df = df[np.isclose(df["noise_std"].to_numpy(dtype=float), float(noise_std), atol=1e-12)]
    if corr_rho is not None and not (isinstance(corr_rho, float) and np.isnan(corr_rho)):
        df = df[np.isclose(df["corr_rho"].to_numpy(dtype=float), float(corr_rho), atol=1e-12)]

    if df.empty:
        return pd.DataFrame(), {}, []

    available_scales = df["qubo_scale_exp"].dropna().unique().tolist()
    resolved_scales = _resolve_scale_exps(available_scales, scale_exps)
    config_map: Dict[str, Optional[Dict[str, float]]] = {}
    rows: List[Dict[str, float]] = []

    overrides = overrides or {}

    for method in method_order:
        sub = df[df["method"] == method].copy()
        if sub.empty:
            continue

        config_cols = _method_config_cols(method)
        config: Optional[Dict[str, float]] = None

        override = overrides.get(method, {})
        if override:
            candidate: Dict[str, float] = {}
            for col in config_cols:
                if col in override and override[col] is not None:
                    val = override[col]
                    if not (isinstance(val, float) and np.isnan(val)):
                        candidate[col] = float(val)
            if candidate:
                filtered = _apply_config_filter(sub, candidate)
                if filtered.empty:
                    print(f"[warn] override for {method} produced no rows; falling back.")
                else:
                    config = candidate
                    sub = filtered

        if config is None:
            baseline = sub[np.isclose(sub["qubo_scale_exp"], float(baseline_scale_exp), atol=1e-12)]
            if baseline.empty:
                baseline = sub
            config = _select_best_config(baseline, config_cols)
            if config:
                sub = _apply_config_filter(sub, config)

        config_map[method] = config

        for scale in resolved_scales:
            scale_mask = np.isclose(sub["qubo_scale_exp"].to_numpy(dtype=float), float(scale), atol=1e-12)
            sub_scale = sub[scale_mask].copy()
            if sub_scale.empty:
                continue
            mean_err, std_err = _mean_std(sub_scale.get("final_rel_error", pd.Series(dtype=float)))
            mean_rank, std_rank = _mean_std(sub_scale.get("final_rank", pd.Series(dtype=float)))
            mean_rec, std_rec = _mean_std(sub_scale.get("final_rec", pd.Series(dtype=float)))
            rows.append(
                {
                    "method": method,
                    "qubo_scale_exp": float(scale),
                    "mean_rel_error": mean_err,
                    "std_rel_error": std_err,
                    "mean_rank": mean_rank,
                    "std_rank": std_rank,
                    "mean_rec": mean_rec,
                    "std_rec": std_rec,
                    "n_runs": int(len(sub_scale)),
                }
            )

    stats_df = pd.DataFrame(rows)
    return stats_df, config_map, resolved_scales


def _lookup_stats(
    stats_df: pd.DataFrame,
    method: str,
    scale_exp: float,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if stats_df.empty:
        return None, None, None, None
    sub = stats_df[stats_df["method"] == method].copy()
    if sub.empty:
        return None, None, None, None
    vals = sub["qubo_scale_exp"].to_numpy(dtype=float)
    mask = np.isclose(vals, float(scale_exp), atol=1e-12)
    if not mask.any():
        return None, None, None, None
    row = sub[mask].iloc[0]
    return (
        row.get("mean_rank"),
        row.get("std_rank"),
        row.get("mean_rel_error"),
        row.get("std_rel_error"),
    )


def build_wide_table(
    stats_df: pd.DataFrame,
    method_order: Sequence[str],
    scale_exps: Sequence[float],
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for scale_exp in scale_exps:
        row: Dict[str, float] = {"qubo_scale_exp": float(scale_exp)}
        for method in method_order:
            key = method.lower().replace("-", "_").replace(" ", "_")
            rank_mean, rank_std, err_mean, err_std = _lookup_stats(stats_df, method, scale_exp)
            row[f"{key}_rank_mean"] = rank_mean
            row[f"{key}_rank_std"] = rank_std
            row[f"{key}_error_mean"] = err_mean
            row[f"{key}_error_std"] = err_std
        rows.append(row)
    return pd.DataFrame(rows)


def _method_header_label(method: str, config_map: Dict[str, Optional[Dict[str, float]]]) -> str:
    label = TABLE_METHOD_LABELS.get(method, method)
    config = config_map.get(method)
    if not config:
        return label
    parts = []
    if method == "ALS" and "rank_init" in config:
        parts.append(f"R={config['rank_init']:.0f}")
    if method == "ALS-TOPK" and "top_k" in config:
        parts.append(f"K={config['top_k']:.0f}")
    if method in (ALS_LR_ES_METHOD, ALS_LR_BF_METHOD, QALR_METHOD) and "prune_mu" in config:
        parts.append(rf"$\mu={config['prune_mu']:.3g}$")
    if method == QALR_METHOD and "qaoa_shots_effective" in config:
        parts.append(rf"$S={config['qaoa_shots_effective']:.0f}$")
    if not parts:
        return label
    return label + " (" + ", ".join(parts) + ")"


def build_latex_table(
    stats_df: pd.DataFrame,
    out_path: str,
    method_order: Sequence[str],
    scale_exps: Sequence[float],
    config_map: Dict[str, Optional[Dict[str, float]]],
    corr_rho: Optional[float],
    noise_std: Optional[float],
) -> None:
    if stats_df.empty:
        print("[warn] summary table skipped (no data).")
        return

    scale_list = ",".join(_format_scale_raw(v) for v in scale_exps)
    rho_str = "--" if corr_rho is None or (isinstance(corr_rho, float) and np.isnan(corr_rho)) else f"{corr_rho:g}"
    sig_str = (
        "--"
        if noise_std is None or (isinstance(noise_std, float) and np.isnan(noise_std))
        else f"{noise_std:g}"
    )
    tabular_fmt = r"@{}l" + ("c" * len(scale_exps)) + r"@{}"
    scale_headers = [_format_scale_tex(v) for v in scale_exps]
    caption = (
        r"\caption{Synthetic scale-robustness ablation across "
        r"\(10^x\) with \(x\in\{"
        + scale_list
        + r"\}\), at fixed noise "
        r"\(\sigma="
        + sig_str
        + r"\) and collinearity "
        r"\(\rho="
        + rho_str
        + r"\). Mean \(\pm\) std across seeds.}"
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        caption,
        r"\label{tab:synth_scale_sweep}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{tabular_fmt}}}",
        r"\toprule",
        " & ".join([r"Method"] + scale_headers) + r" \\",
        r"\midrule",
    ]

    for method in method_order:
        if method not in stats_df["method"].unique():
            continue
        label = _method_header_label(method, config_map)
        row_items = [label]
        for scale_exp in scale_exps:
            rank_mean, rank_std, err_mean, err_std = _lookup_stats(stats_df, method, scale_exp)
            if method in ("ALS", "ALS-TOPK"):
                row_items.append(_fmt_pm(err_mean, err_std, "{:.4g}"))
            else:
                row_items.append(_format_rank_err_cell(rank_mean, rank_std, err_mean, err_std))
        lines.append(" & ".join(row_items) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
        ]
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[info] wrote {out_path}")


def plot_metric_vs_scale(
    stats_df: pd.DataFrame,
    out_path: str,
    method_order: Sequence[str],
    metric_col: str,
    std_col: str,
    scale_exps: Sequence[float],
    ylabel: str,
    dpi: int,
) -> None:
    if stats_df.empty:
        print(f"[warn] plot skipped (no data) -> {out_path}")
        return

    plt.figure(figsize=(3.0, 2.5))
    plotted = 0
    for method in method_order:
        sub = stats_df[stats_df["method"] == method].copy()
        if sub.empty or metric_col not in sub.columns:
            continue
        sub = sub[sub[metric_col].notna()].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("qubo_scale_exp")
        style = STYLE_MAP.get(method, {})
        label = PLOT_LABELS.get(method, method)
        yerr = sub[std_col].fillna(0.0) if std_col in sub.columns else None
        plt.errorbar(
            sub["qubo_scale_exp"],
            sub[metric_col],
            yerr=yerr,
            label=label,
            capsize=2,
            **style,
        )
        plotted += 1

    plt.xticks(scale_exps, [f"{s:g}" for s in scale_exps])
    plt.xlabel(r"$x$")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    if plotted > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"[info] wrote {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot synthetic scale-robustness experiment results."
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="results/v1/synthetic_scale_robustness_experiment_summary.csv",
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default="reports/figures/v1/result/synthesis",
    )
    parser.add_argument(
        "--error-fig-name",
        "--fig-name",
        dest="error_fig_name",
        type=str,
        default="scale_sweep_rel_error_vs_scale.png",
    )
    parser.add_argument(
        "--rank-fig-name",
        type=str,
        default="scale_sweep_rank_vs_scale.png",
    )
    parser.add_argument("--tables-dir", type=str, default="reports/tables/v1")
    parser.add_argument(
        "--table-name", type=str, default="synthetic_scale_robustness_results.tex"
    )
    parser.add_argument(
        "--scale-exps",
        type=str,
        default=None,
        help="Comma-separated scale exponents to include (defaults to those in CSV).",
    )
    parser.add_argument(
        "--baseline-scale-exp",
        type=float,
        default=0.0,
        help="Scale exponent used to pick the best config per method.",
    )
    parser.add_argument(
        "--corr-rho",
        type=float,
        default=0.5,
        help="Collinearity rho filter (set to NaN to disable).",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.01,
        help="Noise std filter (set to NaN to disable).",
    )
    parser.add_argument(
        "--als-rank",
        type=float,
        default=float("nan"),
        help="ALS rank to report (NaN selects best at baseline).",
    )
    parser.add_argument(
        "--topk",
        type=float,
        default=float("nan"),
        help="Top-K value to report (NaN selects best at baseline).",
    )
    parser.add_argument(
        "--alslr-mu",
        type=float,
        default=float("nan"),
        help="ALS-LR mu to report (NaN selects best at baseline).",
    )
    parser.add_argument(
        "--qalr-mu",
        type=float,
        default=float("nan"),
        help="QALR-CPD mu to report (NaN selects best at baseline).",
    )
    parser.add_argument(
        "--qalr-shots",
        type=float,
        default=float("nan"),
        help="QALR-CPD shots to report (NaN selects best at baseline).",
    )
    parser.add_argument("--dpi", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = load_csv(args.summary_csv)
    if summary.empty:
        print("[warn] summary table and plots skipped (no data).")
        return

    scale_exps = _parse_float_list(args.scale_exps) if args.scale_exps else None
    overrides = {
        "ALS": {"rank_init": args.als_rank},
        "ALS-TOPK": {"top_k": args.topk},
        ALS_LR_ES_METHOD: {"prune_mu": args.alslr_mu},
        ALS_LR_BF_METHOD: {"prune_mu": args.alslr_mu},
        QALR_METHOD: {"prune_mu": args.qalr_mu, "qaoa_shots_effective": args.qalr_shots},
    }
    stats_df, config_map, resolved_scales = build_scale_stats(
        summary,
        METHOD_ORDER,
        scale_exps,
        args.baseline_scale_exp,
        args.corr_rho,
        args.noise_std,
        overrides,
    )
    if stats_df.empty:
        print("[warn] summary table and plots skipped (no valid rows).")
        return

    ensure_dir(args.fig_dir)
    ensure_dir(args.tables_dir)

    fig_error_path = os.path.join(args.fig_dir, args.error_fig_name)
    fig_rank_path = os.path.join(args.fig_dir, args.rank_fig_name)
    table_path = os.path.join(args.tables_dir, args.table_name)
    root, ext = os.path.splitext(table_path)
    if ext.lower() != ".tex":
        table_path = f"{table_path}.tex"
        root = table_path[:-4]
    table_csv_path = f"{root}.csv"
    table_long_path = f"{root}_long.csv"

    plot_metric_vs_scale(
        stats_df,
        fig_error_path,
        PLOT_METHOD_ORDER,
        "mean_rel_error",
        "std_rel_error",
        resolved_scales,
        "Rel. error",
        args.dpi,
    )
    plot_metric_vs_scale(
        stats_df,
        fig_rank_path,
        PLOT_METHOD_ORDER,
        "mean_rank",
        "std_rank",
        resolved_scales,
        "Selected rank",
        args.dpi,
    )

    stats_df.to_csv(table_long_path, index=False)
    wide_df = build_wide_table(stats_df, METHOD_ORDER, resolved_scales)
    wide_df.to_csv(table_csv_path, index=False)
    build_latex_table(
        stats_df,
        table_path,
        METHOD_ORDER,
        resolved_scales,
        config_map,
        args.corr_rho,
        args.noise_std,
    )


if __name__ == "__main__":
    main()
