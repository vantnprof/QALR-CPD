"""
Visualization for synthetic noise-varying CP experiments.

Consumes CSV outputs from `synthetic_noise_varying_experiment.py` and produces:
  - Relative error vs. noise level plot (mean +/- std across seeds) for selected configs
  - Selected rank vs. noise level plot (mean +/- std across seeds) for selected configs
  - LaTeX table of selected rank and relative error per noise level
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
DEFAULT_NOISE_LEVELS = (0.0, 1e-3, 1e-2, 5e-2, 1e-1)
DEFAULT_PLOT_NOISE_LEVELS = (1e-3, 1e-2, 5e-2, 1e-1)
PLOT_LABELS = {
    "ALS": "ALS",
    "ALS-TOPK": "Top-K",
    "TREL1": "TREL1",
    ALS_LR_ES_METHOD: "ALS-LR (ES)",
    ALS_LR_BF_METHOD: "ALS-LR (BF)",
    QALR_METHOD: "QALR-CPD",
    LEGACY_QALR_METHOD: "QALR-CPD",
}
STYLE_MAP = {
    "ALS": {"color": "black", "linestyle": "-", "marker": "*"},
    "ALS-TOPK": {"color": "C1", "linestyle": "--", "marker": "s"},
    "TREL1": {"color": "gray", "linestyle": ":", "marker": "d"},
    ALS_LR_ES_METHOD: {"color": "C2", "linestyle": "--", "marker": "x"},
    ALS_LR_BF_METHOD: {"color": "C3", "linestyle": "-.", "marker": "+"},
    QALR_METHOD: {"color": "C0", "linestyle": "-.", "marker": "o"},
    LEGACY_QALR_METHOD: {"color": "C0", "linestyle": "-.", "marker": "o"},
}
TABLE_METHOD_LABELS = {
    "ALS": "ALS",
    "ALS-TOPK": r"Top-\(K\)",
    "TREL1": "TREL1",
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
    if method == "TREL1":
        return ["rank_init", "trel1_lam"]
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


def _resolve_noise_levels(
    available: Sequence[float],
    preferred: Optional[Sequence[float]],
) -> List[float]:
    available_sorted = sorted(float(v) for v in available)
    if not preferred:
        return available_sorted
    return [float(v) for v in preferred]


def _format_rho_raw(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{float(value):.1f}"


def _format_sigma_tex(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    val = float(value)
    if np.isclose(val, 0.0, atol=1e-12):
        return r"\(0\)"
    if abs(val) >= 1.0:
        return rf"\({val:g}\)"
    exp = int(np.floor(np.log10(abs(val))))
    mant = val / (10**exp)
    if np.isclose(mant, round(mant), atol=1e-6):
        mant_int = int(round(mant))
        if mant_int == 1:
            return rf"\(10^{{{exp}}}\)"
        return rf"\({mant_int}{{\times}}10^{{{exp}}}\)"
    return rf"\({mant:.3g}{{\times}}10^{{{exp}}}\)"


def _format_sigma_raw(value: Optional[float]) -> str:
    formatted = _format_sigma_tex(value)
    if formatted.startswith(r"\(") and formatted.endswith(r"\)"):
        return formatted[2:-2]
    return formatted


def _method_column_spec(method: str) -> Tuple[str, ...]:
    if method in ("ALS", "ALS-TOPK"):
        return ("error",)
    return ("rank", "error")


def _fmt_sci_tex(value: Optional[float], digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
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


def _fmt_pm(
    mean: Optional[float],
    std: Optional[float],
    fmt: str,
    sci: bool = False,
) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"
    std_val = 0.0 if std is None or (isinstance(std, float) and np.isnan(std)) else std
    if sci:
        mean_str = _fmt_sci_tex(mean)
        std_str = _fmt_sci_tex(std_val)
    else:
        mean_str = fmt.format(mean)
        std_str = fmt.format(std_val)
    return r"\(" + f"{mean_str} \\pm {std_str}" + r"\)"


def _format_config_value(value: Optional[float], fmt: str = "{:.3g}") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    val = float(value)
    if val.is_integer():
        return str(int(val))
    return fmt.format(val)


def _strip_math(value: str) -> str:
    if value.startswith(r"\(") and value.endswith(r"\)"):
        return value[2:-2]
    return value


def _format_rank_err_cell(
    rank_mean: Optional[float],
    rank_std: Optional[float],
    err_mean: Optional[float],
    err_std: Optional[float],
) -> str:
    rank_cell = _fmt_pm(rank_mean, rank_std, "{:.2f}", sci=False)
    err_cell = _fmt_pm(err_mean, err_std, "{:.4g}", sci=True)
    if rank_cell == "--" or err_cell == "--":
        return "--"
    return (
        r"\(\hat R="
        + _strip_math(rank_cell)
        + r",\ \mathrm{err}="
        + _strip_math(err_cell)
        + r"\)"
    )


def _has_override(value: Optional[float]) -> bool:
    return value is not None and not (isinstance(value, float) and np.isnan(value))


def _method_header_label(
    method: str, config_map: Dict[str, Optional[Dict[str, float]]]
) -> str:
    cfg = config_map.get(method) if config_map else None
    if method == "ALS":
        rank = _format_config_value(cfg.get("rank_init") if cfg else None, fmt="{:.0f}")
        return rf"{TABLE_METHOD_LABELS.get(method, method)} (\(R={rank}\))"
    if method == "ALS-TOPK":
        top_k = _format_config_value(cfg.get("top_k") if cfg else None, fmt="{:.0f}")
        return rf"{TABLE_METHOD_LABELS.get(method, method)} (\(K={top_k}\))"
    if method in (ALS_LR_ES_METHOD, ALS_LR_BF_METHOD):
        rank = _format_config_value(cfg.get("rank_init") if cfg else None, fmt="{:.0f}")
        mu = cfg.get("prune_mu") if cfg else None
        mu_str = _format_sigma_raw(mu)
        return (
            rf"{TABLE_METHOD_LABELS.get(method, method)} "
            rf"(\(R_{{\max}}={rank},\ \mu={mu_str}\))"
        )
    if method == "TREL1":
        lam = _format_config_value(cfg.get("trel1_lam") if cfg else None, fmt="{:.3g}")
        return rf"{TABLE_METHOD_LABELS.get(method, method)} (\(\lambda={lam}\))"
    if method == QALR_METHOD:
        mu = cfg.get("prune_mu") if cfg else None
        mu_str = _format_sigma_raw(mu)
        return rf"{TABLE_METHOD_LABELS.get(method, method)} (\(\mu={mu_str}\))"
    return TABLE_METHOD_LABELS.get(method, method)


def _apply_overrides(df: pd.DataFrame, overrides: Dict[str, Optional[float]]) -> pd.DataFrame:
    filtered = df
    for col, value in overrides.items():
        if not _has_override(value):
            continue
        if col not in filtered.columns:
            filtered[col] = np.nan
        filtered = filtered[_filter_by_value(filtered[col], value)]
    return filtered


def build_noise_sweep_stats(
    summary: pd.DataFrame,
    method_order: Sequence[str],
    noise_levels: Optional[Sequence[float]],
    corr_rho: Optional[float],
    overrides: Dict[str, Dict[str, Optional[float]]],
) -> Tuple[pd.DataFrame, Dict[str, Optional[Dict[str, float]]], List[float]]:
    summary = _normalize_method_names(summary)
    summary = _filter_status_ok(summary)
    summary = _coerce_noise_std(summary)
    summary = _coerce_corr_rho(summary)
    if summary.empty or "noise_std" not in summary.columns:
        return pd.DataFrame(), {}, []

    summary = summary[summary["noise_std"].notna()]
    if summary.empty:
        return pd.DataFrame(), {}, []

    if _has_override(corr_rho):
        if "corr_rho" not in summary.columns:
            print("[warn] corr_rho column missing; skipping rho filter.")
        else:
            rho_vals = summary["corr_rho"].to_numpy(dtype=float)
            rho_mask = np.isclose(rho_vals, float(corr_rho), atol=1e-12)
            summary = summary[rho_mask]
            if summary.empty:
                return pd.DataFrame(), {}, []

    summary = summary.copy()
    for col in ("rank_init", "top_k", "trel1_lam", "prune_mu", "qaoa_shots_effective"):
        if col not in summary.columns:
            summary[col] = np.nan

    resolved_levels = _resolve_noise_levels(
        summary["noise_std"].unique(),
        noise_levels,
    )
    if not resolved_levels:
        return pd.DataFrame(), {}, []

    rows: List[Dict] = []
    config_map: Dict[str, Optional[Dict[str, float]]] = {}
    for method in method_order:
        method_df = summary[summary["method"] == method].copy()
        method_overrides = overrides.get(method, {}) if overrides else {}
        method_df = _apply_overrides(method_df, method_overrides)
        if method_df.empty:
            cfg = {
                key: float(val)
                for key, val in method_overrides.items()
                if _has_override(val)
            }
            config_map[method] = cfg if cfg else None
            continue
        config_cols = _method_config_cols(method)
        best_cfg = _select_best_config(method_df, config_cols)
        if best_cfg is None:
            best_cfg = {}
        for col, value in method_overrides.items():
            if _has_override(value):
                best_cfg[col] = float(value)
        config_map[method] = best_cfg if best_cfg else None
        filtered = method_df
        if best_cfg is not None:
            for col, value in best_cfg.items():
                if col not in filtered.columns:
                    filtered[col] = np.nan
                filtered = filtered[_filter_by_value(filtered[col], value)]
        for noise in resolved_levels:
            noise_vals = filtered["noise_std"].to_numpy(dtype=float)
            noise_mask = np.isclose(noise_vals, float(noise), atol=1e-12)
            noise_df = filtered[noise_mask]
            if noise_df.empty:
                continue
            rank_mean, rank_std = _mean_std(noise_df["final_rank"])
            err_mean, err_std = _mean_std(noise_df["final_rel_error"])
            row = {
                "noise_std": float(noise),
                "method": method,
                "mean_rank": rank_mean,
                "std_rank": rank_std,
                "mean_rel_error": err_mean,
                "std_rel_error": err_std,
                "num_runs": len(noise_df),
            }
            if best_cfg:
                row.update(best_cfg)
            rows.append(row)

    stats_df = pd.DataFrame(rows)
    return stats_df, config_map, resolved_levels


def _lookup_stats(
    stats_df: pd.DataFrame,
    method: str,
    noise: float,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if stats_df.empty:
        return None, None, None, None
    sub = stats_df[stats_df["method"] == method].copy()
    if sub.empty:
        return None, None, None, None
    noise_vals = sub["noise_std"].to_numpy(dtype=float)
    mask = np.isclose(noise_vals, float(noise), atol=1e-12)
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
    noise_levels: Sequence[float],
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for noise in noise_levels:
        row: Dict[str, float] = {"noise_std": float(noise)}
        for method in method_order:
            key = method.lower().replace("-", "_")
            rank_mean, rank_std, err_mean, err_std = _lookup_stats(stats_df, method, noise)
            row[f"{key}_rank_mean"] = rank_mean
            row[f"{key}_rank_std"] = rank_std
            row[f"{key}_error_mean"] = err_mean
            row[f"{key}_error_std"] = err_std
        rows.append(row)
    return pd.DataFrame(rows)


def build_latex_table(
    stats_df: pd.DataFrame,
    out_path: str,
    method_order: Sequence[str],
    noise_levels: Sequence[float],
    config_map: Dict[str, Optional[Dict[str, float]]],
    corr_rho: Optional[float],
) -> None:
    if stats_df.empty:
        print("[warn] summary table skipped (no data).")
        return

    sigma_list = ",".join(_format_sigma_raw(v) for v in noise_levels)
    rho_str = _format_rho_raw(corr_rho)
    method_specs = [(method, _method_column_spec(method)) for method in method_order]
    header_labels = [_method_header_label(method, config_map) for method, _ in method_specs]
    tabular_fmt = r"@{}l" + ("c" * len(noise_levels)) + r"@{}"
    noise_headers = [_format_sigma_tex(v) for v in noise_levels]
    caption = (
        r"\caption{Synthetic noise ablation: performance across "
        r"\(\sigma\in\{"
        + sigma_list
        + r"\}\) at fixed collinearity "
        r"\(\rho="
        + rho_str
        + r"\). Mean \(\pm\) std over \(S=20\) seeds. "
        r"Cells show Rel.\ error for ALS/Top-\(K\); \(\hat R\) and Rel.\ error for ALS-LR/QALR-CPD.}"
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        caption,
        r"\label{tab:synth_noise_sweep_best}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{tabular_fmt}}}",
        r"\toprule",
        " & ".join([r"Method"] + noise_headers) + r" \\",
        r"\midrule",
    ]

    for (method, spec), label in zip(method_specs, header_labels):
        row_items = [label]
        for noise in noise_levels:
            rank_mean, rank_std, err_mean, err_std = _lookup_stats(stats_df, method, noise)
            if len(spec) == 1:
                row_items.append(_fmt_pm(err_mean, err_std, "{:.4g}", sci=True))
            else:
                row_items.append(
                    _format_rank_err_cell(rank_mean, rank_std, err_mean, err_std)
                )
        lines.append(" & ".join(row_items) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[info] wrote {out_path}")


def plot_metric_vs_noise(
    stats_df: pd.DataFrame,
    out_path: str,
    method_order: Sequence[str],
    metric_col: str,
    std_col: str,
    ylabel: str,
    x_ticks: Optional[Sequence[float]],
    dpi: int,
) -> None:
    if stats_df.empty:
        print("[warn] plot skipped (no data).")
        return

    use_discrete_x = bool(x_ticks)
    tick_positions: Optional[np.ndarray] = None
    tick_labels: Optional[List[str]] = None
    tick_values: Optional[List[float]] = None
    if use_discrete_x:
        tick_values = [float(v) for v in x_ticks]
        tick_positions = np.arange(len(tick_values), dtype=float)
        tick_labels = [
            _format_sigma_tex(v).replace(r"\(", "$").replace(r"\)", "$")
            for v in tick_values
        ]

    plt.figure(figsize=(3, 2.5))
    for method in method_order:
        sub = stats_df[stats_df["method"] == method].copy()
        if sub.empty:
            continue
        sub = sub[sub[metric_col].notna()].copy()
        if use_discrete_x and tick_values is not None:
            noise_vals = sub["noise_std"].to_numpy(dtype=float)
            xpos = np.full(noise_vals.shape, np.nan)
            for idx, val in enumerate(noise_vals):
                for jdx, tick in enumerate(tick_values):
                    if np.isclose(val, tick, atol=1e-12):
                        xpos[idx] = float(jdx)
                        break
            sub["xpos"] = xpos
            sub = sub[sub["xpos"].notna()].sort_values("xpos")
        else:
            sub = sub.sort_values("noise_std")
        if sub.empty:
            continue
        style = STYLE_MAP.get(method, {})
        label = PLOT_LABELS.get(method, method)
        yerr = sub[std_col].fillna(0.0)
        x_vals = sub["xpos"] if use_discrete_x else sub["noise_std"]
        plt.errorbar(
            x_vals,
            sub[metric_col],
            yerr=yerr,
            label=label,
            capsize=2,
            **style,
        )

    if use_discrete_x and tick_positions is not None and tick_labels is not None:
        plt.xticks(tick_positions, tick_labels)
        if len(tick_positions) > 1:
            plt.xlim(tick_positions[0] - 0.3, tick_positions[-1] + 0.3)
    plt.xlabel(r"$\sigma$")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"[info] wrote {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot synthetic noise-varying experiment results."
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="results/v1/synthetic_noise_varying_experiment_summary.csv",
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
        default="noise_sweep_best_error_vs_sigma.png",
    )
    parser.add_argument(
        "--rank-fig-name",
        type=str,
        default="noise_sweep_best_rank_vs_sigma.png",
    )
    parser.add_argument("--tables-dir", type=str, default="reports/tables/v1")
    parser.add_argument(
        "--table-name", type=str, default="synthetic_noise_varying_results.tex"
    )
    parser.add_argument(
        "--noise-levels",
        type=str,
        default=None,
        help="Comma-separated noise stds to include (defaults to paper sweep).",
    )
    parser.add_argument(
        "--corr-rho",
        type=float,
        default=0.5,
        help="Collinearity rho to filter for the noise sweep (set to NaN to disable filtering).",
    )
    parser.add_argument(
        "--als-rank",
        type=float,
        default=20,
        help="ALS rank to report (set to NaN to pick the best rank).",
    )
    parser.add_argument(
        "--topk",
        type=float,
        default=6,
        help="Top-K value to report (set to NaN to pick the best K).",
    )
    parser.add_argument(
        "--trel1-lam",
        type=float,
        default=0.01,
        help="TREL1 lambda to report (set to NaN to pick the best lambda).",
    )
    parser.add_argument(
        "--qalr-mu",
        type=float,
        default=0.1,
        help="QALR-CPD mu to report (set to NaN to pick the best mu).",
    )
    parser.add_argument("--dpi", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = load_csv(args.summary_csv)
    if summary.empty:
        print("[warn] summary table and plots skipped (no data).")
        return

    noise_levels = (
        _parse_float_list(args.noise_levels)
        if args.noise_levels
        else list(DEFAULT_NOISE_LEVELS)
    )
    overrides = {
        "ALS": {"rank_init": args.als_rank},
        "ALS-TOPK": {"top_k": args.topk},
        "TREL1": {"trel1_lam": args.trel1_lam},
        QALR_METHOD: {"prune_mu": args.qalr_mu},
    }
    stats_df, config_map, resolved_levels = build_noise_sweep_stats(
        summary,
        METHOD_ORDER,
        noise_levels,
        args.corr_rho,
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

    eps = 1e-12
    desired_plot_levels = [float(v) for v in DEFAULT_PLOT_NOISE_LEVELS]
    plot_levels = [
        float(v)
        for v in desired_plot_levels
        if any(np.isclose(float(v), float(r), atol=eps) for r in resolved_levels)
    ]
    if not plot_levels:
        plot_levels = [
            float(v)
            for v in resolved_levels
            if (float(v) + eps) >= 1e-3 and (float(v) - eps) <= 1e-1
        ]
        if not plot_levels:
            plot_levels = resolved_levels

    plot_metric_vs_noise(
        stats_df,
        fig_error_path,
        METHOD_ORDER,
        "mean_rel_error",
        "std_rel_error",
        "Rel. error",
        plot_levels,
        args.dpi,
    )
    plot_metric_vs_noise(
        stats_df,
        fig_rank_path,
        METHOD_ORDER,
        "mean_rank",
        "std_rank",
        "Selected rank",
        plot_levels,
        args.dpi,
    )

    stats_df.to_csv(table_long_path, index=False)
    wide_df = build_wide_table(stats_df, METHOD_ORDER, resolved_levels)
    wide_df.to_csv(table_csv_path, index=False)
    build_latex_table(
        stats_df,
        table_path,
        METHOD_ORDER,
        resolved_levels,
        config_map,
        args.corr_rho,
    )
    print(f"[info] wrote {table_long_path}")
    print(f"[info] wrote {table_csv_path}")


if __name__ == "__main__":
    main()
