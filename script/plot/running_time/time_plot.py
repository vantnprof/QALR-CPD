"""
Runtime/error plots for ES, BF, and QAOA pruning calls.

The script expects experiment CSVs with per-sweep logs (for example
results/acid_experiment.csv or results/synthetic_experiment.csv).

Outputs:
1) Runtime per pruning call (seconds, log-scale y) vs R.
2) Normalized approximation error (f_approx - f_ES) / (|f_ES| + 1e-12)
   for BF and QAOA vs the same R values used in plot (1).

By default the script enforces strict apple-to-apple tuples:
- same paired call id across methods
- same input_id (if available)
- same R across ES/BF/QAOA
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRUNE_TO_METHOD = {
    "exhaustive": "ES",
    "bitflip": "BF",
    "qaoa": "QAOA",
}
METHOD_ORDER = ["ES", "BF", "QAOA"]
METHOD_STYLE = {
    # Matches STYLE_MAP used in amino/synthetic experiment plotting scripts:
    # ALS-LR (ES): C2 -- x, ALS-LR (BF): C3 -. +, QALR-CPD: C0 -. o
    "ES": {"color": "C2", "linestyle": "--", "marker": "x"},
    "BF": {"color": "C3", "linestyle": "-.", "marker": "+"},
    "QAOA": {"color": "C0", "linestyle": "-.", "marker": "o"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot pruning-call runtime and normalized approximation error "
            "for ES/BF/QAOA from experiment CSV logs."
        )
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="results/acid_experiment.csv",
        help="Input experiment CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/running_time",
        help="Directory to write plots and summaries.",
    )
    parser.add_argument(
        "--runtime-figure",
        type=str,
        default="runtime_per_pruning_call_vs_R.png",
        help="Runtime figure filename.",
    )
    parser.add_argument(
        "--error-figure",
        type=str,
        default="normalized_error_vs_R.png",
        help="Normalized error figure filename.",
    )
    parser.add_argument(
        "--r-col",
        type=str,
        default="rank_before_prune",
        help=(
            "Rank column used for x-axis R. "
            "For pruning dynamics, use rank_before_prune (default)."
        ),
    )
    parser.add_argument(
        "--r-source-method",
        type=str,
        default="min",
        choices=["ES", "BF", "QAOA", "min", "max", "mean"],
        help=(
            "How to derive a single R for each paired tuple (run_key, call_idx): "
            "pick ES/BF/QAOA rank, or aggregate across methods with min/max/mean."
        ),
    )
    parser.add_argument(
        "--allow-input-mismatch",
        action="store_true",
        help=(
            "Allow tuples where method inputs differ (different input_id or rank spread). "
            "By default, such tuples are dropped for strict apple-to-apple comparison."
        ),
    )
    parser.add_argument(
        "--planned-sweeps",
        type=int,
        default=None,
        help=(
            "Planned number of sweeps used in the experiment. If unset, inferred "
            "as the mode of max sweep count per run."
        ),
    )
    parser.add_argument(
        "--default-prune-period",
        type=int,
        default=40,
        help="Fallback prune period if column is missing/invalid.",
    )
    parser.add_argument(
        "--min-runtime-s",
        type=float,
        default=1e-9,
        help="Lower bound for runtime values after ALS-baseline subtraction.",
    )
    parser.add_argument(
        "--objective-col",
        type=str,
        default=None,
        help=(
            "Objective column for normalized error. If unset, auto-picks from "
            "['f_approx','objective','obj_value','qubo_energy','energy','rel_error']."
        ),
    )
    parser.add_argument(
        "--qaoa-shots",
        type=float,
        default=None,
        help=(
            "Optional qaoa_shots_effective filter. If omitted and multiple shot values "
            "exist, the largest value is used."
        ),
    )
    parser.add_argument(
        "--min-r",
        type=float,
        default=None,
        help="Optional lower bound on R for plotting (e.g., 18 keeps only R >= 18).",
    )
    parser.add_argument(
        "--r-values",
        type=str,
        default=None,
        help=(
            "Optional comma-separated R values to keep exactly "
            "(e.g., '18,20,22,24,26,28,30')."
        ),
    )
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _pick_objective_col(df: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(f"Requested objective column '{requested}' not found.")
        return requested

    candidates = ["f_approx", "objective", "obj_value", "qubo_energy", "energy", "rel_error"]
    for col in candidates:
        if col in df.columns:
            if col == "rel_error":
                print(
                    "[warn] Using 'rel_error' as f_approx proxy because no explicit "
                    "QUBO/objective column was found."
                )
            return col
    raise ValueError(
        "No objective column found. Use --objective-col to specify one explicitly."
    )


def _build_run_key(df: pd.DataFrame) -> pd.Series:
    qaoa_only_cols = {"qaoa_shots_effective", "qaoa_reps", "qaoa_maxiter", "qaoa_restarts"}
    key_candidates = [
        "dataset_tag",
        "dataset_path",
        "data_seed",
        "seed",
        "rank_true",
        "rank_init",
        "prune_mu",
        "prune_mu_effective",
        "mu_scale",
        "prune_period",
        "ridge",
        "qaoa_shots_effective",
        "qaoa_reps",
        "qaoa_maxiter",
        "qaoa_restarts",
    ]
    key_cols: List[str] = []
    for c in key_candidates:
        if c not in df.columns:
            continue
        if c in qaoa_only_cols:
            non_qaoa = df[df["prune_method"] != "qaoa"]
            # Include qaoa config only when it is also populated for non-QAOA rows.
            # This keeps backward compatibility with old logs where these are NaN for ES/BF.
            if non_qaoa.empty or non_qaoa[c].notna().any():
                key_cols.append(c)
            continue
        key_cols.append(c)
    if not key_cols:
        raise ValueError("No run-key columns found for pairing method runs.")
    # String key avoids dtype inconsistencies across csv exports.
    return df[key_cols].astype(str).agg("|".join, axis=1)


def _filter_qaoa_shots(df: pd.DataFrame, requested_shots: float | None) -> pd.DataFrame:
    if "qaoa_shots_effective" not in df.columns:
        return df

    is_qaoa = df["prune_method"] == "qaoa"
    shot_vals = (
        pd.to_numeric(df.loc[is_qaoa, "qaoa_shots_effective"], errors="coerce")
        .dropna()
        .unique()
        .tolist()
    )
    if not shot_vals:
        return df

    shot_vals_sorted = sorted(float(v) for v in shot_vals)
    if requested_shots is None:
        chosen = shot_vals_sorted[-1]
        if len(shot_vals_sorted) > 1:
            print(
                f"[info] Multiple qaoa_shots_effective values found: {shot_vals_sorted}. "
                f"Using {chosen} for apple-to-apple pairing."
            )
    else:
        chosen = float(requested_shots)
        if chosen not in shot_vals_sorted:
            raise ValueError(
                f"--qaoa-shots={chosen} not found in qaoa_shots_effective values: {shot_vals_sorted}"
            )

    qaoa_shots_num = pd.to_numeric(df["qaoa_shots_effective"], errors="coerce")
    keep = (~is_qaoa) | np.isclose(qaoa_shots_num, chosen, atol=1e-12)
    return df.loc[keep].copy()


def _infer_planned_sweeps(df: pd.DataFrame) -> int | None:
    per_run = df.groupby(["run_key", "method_label"], dropna=False)["sweep"].max()
    if per_run.empty:
        return None
    modes = per_run.mode()
    if modes.empty:
        return int(per_run.max())
    return int(modes.iloc[0])


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y"})


def prepare_call_level_table(
    df: pd.DataFrame,
    r_col: str,
    planned_sweeps: int | None,
    default_prune_period: int,
    min_runtime_s: float,
) -> tuple[pd.DataFrame, int | None, str]:
    _require_columns(df, ["prune_method", "sweep", "sweep_time_s", r_col])
    sub = df[df["prune_method"].isin(PRUNE_TO_METHOD)].copy()
    if sub.empty:
        raise ValueError("No ES/BF/QAOA rows found in CSV.")

    sub["method_label"] = sub["prune_method"].map(PRUNE_TO_METHOD)
    sub["run_key"] = _build_run_key(sub)

    # Keep only run keys present for all 3 methods.
    method_counts = sub.groupby(["run_key", "method_label"], dropna=False).size().unstack(fill_value=0)
    valid_keys = method_counts.index[(method_counts[METHOD_ORDER] > 0).all(axis=1)]
    sub = sub[sub["run_key"].isin(valid_keys)].copy()
    if sub.empty:
        raise ValueError("No matched run keys with all ES/BF/QAOA methods.")

    has_explicit_prune_flags = "is_prune_call" in sub.columns

    # Infer planned sweeps only when prune calls must be inferred from sweep index.
    if planned_sweeps is None and (not has_explicit_prune_flags):
        planned_sweeps = _infer_planned_sweeps(sub)
    if planned_sweeps is not None:
        print(f"[info] planned_sweeps={planned_sweeps}")

    period = pd.to_numeric(sub.get("prune_period", default_prune_period), errors="coerce")
    period = period.fillna(default_prune_period).astype(int)
    period = period.where(period > 0, default_prune_period)
    sub["prune_period_eff"] = period

    # If present, trust explicit prune-call flags emitted by the experiment script.
    if has_explicit_prune_flags:
        sub["is_prune_call"] = _to_bool(sub["is_prune_call"])
    else:
        # Fallback: infer prune calls from sweep index and prune period.
        sub["is_prune_call"] = (sub["sweep"] % sub["prune_period_eff"] == 0)

    # When explicit call flags are present, trust them by default and avoid truncation.
    if (not has_explicit_prune_flags) and (planned_sweeps is not None):
        sub["is_prune_call"] = sub["is_prune_call"] & (sub["sweep"] < int(planned_sweeps))

    calls = sub.loc[sub["is_prune_call"]].copy()
    calls = calls.dropna(subset=["sweep_time_s", r_col]).copy()
    if calls.empty:
        raise ValueError("No pruning-call rows after filtering.")

    if "prune_call_idx" in calls.columns:
        call_idx = pd.to_numeric(calls["prune_call_idx"], errors="coerce")
        inferred = (calls["sweep"] // calls["prune_period_eff"]).astype(int)
        calls["call_idx"] = call_idx.fillna(inferred).astype(int)
    else:
        calls["call_idx"] = (calls["sweep"] // calls["prune_period_eff"]).astype(int)

    runtime_source = "baseline_subtracted_sweep_time_s"
    if "prune_solver_runtime_s" in calls.columns:
        direct_runtime = pd.to_numeric(calls["prune_solver_runtime_s"], errors="coerce")
    else:
        direct_runtime = pd.Series(np.nan, index=calls.index, dtype=float)

    if direct_runtime.notna().any():
        calls["solver_runtime_s"] = direct_runtime
        runtime_source = "direct_prune_solver_runtime_s"
    else:
        baseline = (
            sub.loc[~sub["is_prune_call"]]
            .groupby(["run_key", "method_label"], dropna=False)["sweep_time_s"]
            .median()
            .rename("als_baseline_s")
        )
        calls = calls.join(baseline, on=["run_key", "method_label"], how="left")
        calls["solver_runtime_s"] = (
            pd.to_numeric(calls["sweep_time_s"], errors="coerce")
            - pd.to_numeric(calls["als_baseline_s"], errors="coerce")
        )

    calls["solver_runtime_s"] = pd.to_numeric(calls["solver_runtime_s"], errors="coerce")
    calls = calls.dropna(subset=["solver_runtime_s"]).copy()
    calls["solver_runtime_s"] = calls["solver_runtime_s"].clip(lower=float(min_runtime_s))
    calls[r_col] = pd.to_numeric(calls[r_col], errors="coerce")
    calls = calls.dropna(subset=[r_col]).copy()
    return calls, planned_sweeps, runtime_source


def build_paired_runtime_table(
    calls: pd.DataFrame,
    r_col: str,
    r_source_method: str,
    allow_input_mismatch: bool,
) -> pd.DataFrame:
    idx_cols = ["run_key", "call_idx"]
    runtime_paired = calls.pivot_table(
        index=idx_cols,
        columns="method_label",
        values="solver_runtime_s",
        aggfunc="mean",
    )
    runtime_paired = runtime_paired.dropna(subset=METHOD_ORDER, how="any")
    if runtime_paired.empty:
        raise ValueError("No matched pruning-call tuples across ES/BF/QAOA for runtime.")

    r_paired = calls.pivot_table(
        index=idx_cols,
        columns="method_label",
        values=r_col,
        aggfunc="mean",
    )
    r_paired = r_paired.reindex(runtime_paired.index)

    if not allow_input_mismatch:
        rank_min = r_paired[METHOD_ORDER].apply(pd.to_numeric, errors="coerce").min(axis=1)
        rank_max = r_paired[METHOD_ORDER].apply(pd.to_numeric, errors="coerce").max(axis=1)
        same_rank_mask = (rank_max - rank_min).abs() <= 1e-12
        runtime_paired = runtime_paired.loc[same_rank_mask].copy()
        r_paired = r_paired.loc[runtime_paired.index].copy()

        if "input_id" in calls.columns:
            input_id_paired = calls.pivot_table(
                index=idx_cols,
                columns="method_label",
                values="input_id",
                aggfunc=lambda x: str(x.iloc[0]) if len(x) > 0 else "",
            ).reindex(runtime_paired.index)
            same_input_mask = (
                (input_id_paired["ES"] == input_id_paired["BF"])
                & (input_id_paired["ES"] == input_id_paired["QAOA"])
            )
            runtime_paired = runtime_paired.loc[same_input_mask].copy()
            r_paired = r_paired.loc[runtime_paired.index].copy()

        if runtime_paired.empty:
            raise ValueError(
                "No strict apple-to-apple tuples remain after input/rank consistency filtering."
            )

    r_key = str(r_source_method).strip()
    if r_key in METHOD_ORDER:
        if r_key not in r_paired.columns:
            raise ValueError(f"R source method '{r_key}' is not available in paired tuples.")
        r_value = pd.to_numeric(r_paired[r_key], errors="coerce")
    elif r_key == "min":
        r_value = r_paired[METHOD_ORDER].apply(pd.to_numeric, errors="coerce").min(axis=1)
    elif r_key == "max":
        r_value = r_paired[METHOD_ORDER].apply(pd.to_numeric, errors="coerce").max(axis=1)
    elif r_key == "mean":
        r_value = r_paired[METHOD_ORDER].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    else:
        raise ValueError(f"Unsupported --r-source-method={r_source_method}")

    runtime_paired["__r_value__"] = pd.to_numeric(r_value, errors="coerce")
    runtime_paired = runtime_paired.dropna(subset=["__r_value__"]).copy()
    if runtime_paired.empty:
        raise ValueError("No paired runtime tuples with valid dynamic R values.")
    return runtime_paired


def summarize_runtime(runtime_paired: pd.DataFrame, r_col: str) -> pd.DataFrame:
    runtime_long = runtime_paired[METHOD_ORDER].stack().rename("runtime_s").reset_index()
    if "method_label" in runtime_long.columns:
        runtime_long = runtime_long.rename(columns={"method_label": "method"})
    elif "level_2" in runtime_long.columns:
        runtime_long = runtime_long.rename(columns={"level_2": "method"})
    else:
        # Fallback: third index-reset column is the stacked method label.
        runtime_long = runtime_long.rename(columns={runtime_long.columns[2]: "method"})
    runtime_long = runtime_long.join(
        runtime_paired["__r_value__"].rename(r_col),
        on=["run_key", "call_idx"],
    )
    summary = (
        runtime_long.groupby([r_col, "method"], dropna=False)["runtime_s"]
        .agg(mean_s="mean", std_s="std", n="size")
        .reset_index()
        .sort_values([r_col, "method"])
    )
    return summary


def summarize_normalized_error(
    calls: pd.DataFrame,
    runtime_paired: pd.DataFrame,
    r_col: str,
    objective_col: str,
) -> pd.DataFrame:
    calls_obj = calls.copy()
    calls_obj["objective_value"] = pd.to_numeric(calls_obj[objective_col], errors="coerce")
    calls_obj = calls_obj.dropna(subset=["objective_value"]).copy()

    idx_cols = ["run_key", "call_idx"]
    obj_paired = calls_obj.pivot_table(
        index=idx_cols,
        columns="method_label",
        values="objective_value",
        aggfunc="mean",
    )

    # Use exact same tuples as runtime plot to keep the comparison apple-to-apple.
    obj_paired = obj_paired.reindex(runtime_paired.index)
    obj_paired = obj_paired.dropna(subset=METHOD_ORDER, how="any")
    if obj_paired.empty:
        raise ValueError(
            "No matched tuples for normalized error after runtime-index alignment."
        )

    eps = 1e-12
    norm = pd.DataFrame(index=obj_paired.index)
    norm[r_col] = runtime_paired.loc[obj_paired.index, "__r_value__"]
    norm["BF"] = (obj_paired["BF"] - obj_paired["ES"]) / (obj_paired["ES"].abs() + eps)
    norm["QAOA"] = (obj_paired["QAOA"] - obj_paired["ES"]) / (obj_paired["ES"].abs() + eps)

    norm_long = norm.reset_index().melt(
        id_vars=["run_key", "call_idx", r_col],
        value_vars=["BF", "QAOA"],
        var_name="method",
        value_name="norm_error",
    )
    summary = (
        norm_long.groupby([r_col, "method"], dropna=False)["norm_error"]
        .agg(mean="mean", std="std", n="size")
        .reset_index()
        .sort_values([r_col, "method"])
    )
    return summary


def _filter_runtime_pairs_by_min_r(
    runtime_paired: pd.DataFrame,
    min_r: float | None,
) -> pd.DataFrame:
    if min_r is None:
        return runtime_paired
    out = runtime_paired[pd.to_numeric(runtime_paired["__r_value__"], errors="coerce") >= float(min_r)].copy()
    if out.empty:
        raise ValueError(f"No paired tuples remain after applying --min-r={min_r}.")
    return out


def _parse_r_values(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    vals: set[int] = set()
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        vals.add(int(float(t)))
    if not vals:
        return None
    return vals


def _filter_runtime_pairs_by_r_values(
    runtime_paired: pd.DataFrame,
    r_values: set[int] | None,
) -> pd.DataFrame:
    if r_values is None:
        return runtime_paired
    r_num = pd.to_numeric(runtime_paired["__r_value__"], errors="coerce")
    keep = r_num.map(lambda x: int(round(float(x))) if pd.notna(x) else None).isin(r_values)
    out = runtime_paired.loc[keep].copy()
    if out.empty:
        raise ValueError(
            f"No paired tuples remain after applying --r-values={sorted(r_values)}."
        )
    return out


def _sorted_r_values(values: Sequence[object]) -> List[object]:
    def _coerce_rank_value(v: object) -> object:
        try:
            fv = float(v)
        except Exception:
            return v
        if np.isfinite(fv) and abs(fv - round(fv)) <= 1e-9:
            return int(round(fv))
        return fv

    normalized = [_coerce_rank_value(v) for v in values]
    try:
        return sorted(normalized, key=lambda x: float(x))
    except Exception:
        return sorted(normalized)


def _rank_tick_label(v: object) -> str:
    try:
        fv = float(v)
        if np.isfinite(fv) and abs(fv - round(fv)) <= 1e-9:
            return str(int(round(fv)))
    except Exception:
        pass
    return str(v)


def _coerce_rank_for_lookup(v: object) -> object:
    try:
        fv = float(v)
        if np.isfinite(fv) and abs(fv - round(fv)) <= 1e-9:
            return int(round(fv))
        return fv
    except Exception:
        return v


def plot_runtime(summary: pd.DataFrame, r_col: str, output_path: Path) -> List[object]:
    fig, ax = plt.subplots(figsize=(3, 2.5))
    r_values = _sorted_r_values(summary[r_col].dropna().unique().tolist())
    x_pos = {r: i for i, r in enumerate(r_values)}

    for method in METHOD_ORDER:
        sub = summary[summary["method"] == method].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(r_col)
        x = [x_pos[_coerce_rank_for_lookup(v)] for v in sub[r_col].tolist()]
        y = sub["mean_s"].to_numpy(dtype=float)
        style = METHOD_STYLE[method]
        ax.plot(
            x,
            y,
            label=method,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=1.8,
            markersize=5,
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"$R$")
    ax.set_ylabel("Running time (s)")
    if r_values:
        ax.set_xticks(range(len(r_values)))
        ax.set_xticklabels([_rank_tick_label(v) for v in r_values])
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=1000, bbox_inches="tight")
    plt.close(fig)
    return r_values


def plot_normalized_error(
    summary: pd.DataFrame,
    r_col: str,
    r_values: Sequence[object],
    output_path: Path,
    objective_col: str,
) -> None:
    fig, ax = plt.subplots(figsize=(3, 2.5))
    keep_r = set(r_values)
    x_pos = {r: i for i, r in enumerate(r_values)}

    for method in ["BF", "QAOA"]:
        sub = summary[(summary["method"] == method) & (summary[r_col].isin(keep_r))].copy()
        if sub.empty:
            sub = summary[
                (summary["method"] == method)
                & (summary[r_col].map(_coerce_rank_for_lookup).isin(keep_r))
            ].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(r_col)
        x = [x_pos[_coerce_rank_for_lookup(v)] for v in sub[r_col].tolist()]
        y = sub["mean"].to_numpy(dtype=float)
        style = METHOD_STYLE[method]
        ax.plot(
            x,
            y,
            label=method,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=1.8,
            markersize=5,
        )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlabel(r"$R$")
    ax.set_ylabel(r"$(f_{\mathrm{approx}} - f_{\mathrm{ES}}) / (|f_{\mathrm{ES}}| + 10^{-12})$")
    if r_values:
        ax.set_xticks(range(len(r_values)))
        ax.set_xticklabels([_rank_tick_label(v) for v in r_values])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=1000, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"Missing input CSV: {csv_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    objective_col = _pick_objective_col(df, args.objective_col)
    r_col = args.r_col
    if r_col not in df.columns:
        if r_col == "rank_before_prune" and "rank" in df.columns:
            r_col = "rank"
            print("[info] rank_before_prune not found; falling back to rank for x-axis.")
        elif r_col == "rank" and "rank_before_prune" in df.columns:
            r_col = "rank_before_prune"
            print("[info] rank not found; using rank_before_prune for x-axis.")
        else:
            raise ValueError(f"Requested --r-col={args.r_col} not found in CSV columns.")

    df = _filter_qaoa_shots(df, args.qaoa_shots)
    calls, planned, runtime_source = prepare_call_level_table(
        df=df,
        r_col=r_col,
        planned_sweeps=args.planned_sweeps,
        default_prune_period=args.default_prune_period,
        min_runtime_s=args.min_runtime_s,
    )
    runtime_paired = build_paired_runtime_table(
        calls,
        r_col=r_col,
        r_source_method=args.r_source_method,
        allow_input_mismatch=bool(args.allow_input_mismatch),
    )
    runtime_paired = _filter_runtime_pairs_by_min_r(runtime_paired, args.min_r)
    selected_r_values = _parse_r_values(args.r_values)
    runtime_paired = _filter_runtime_pairs_by_r_values(runtime_paired, selected_r_values)
    runtime_summary = summarize_runtime(runtime_paired, r_col=r_col)
    error_summary = summarize_normalized_error(
        calls=calls,
        runtime_paired=runtime_paired,
        r_col=r_col,
        objective_col=objective_col,
    )

    runtime_fig_path = out_dir / args.runtime_figure
    error_fig_path = out_dir / args.error_figure
    runtime_r_values = plot_runtime(runtime_summary, r_col=r_col, output_path=runtime_fig_path)
    plot_normalized_error(
        error_summary,
        r_col=r_col,
        r_values=runtime_r_values,
        output_path=error_fig_path,
        objective_col=objective_col,
    )

    runtime_summary.to_csv(out_dir / "runtime_summary.csv", index=False)
    error_summary.to_csv(out_dir / "normalized_error_summary.csv", index=False)

    print(f"[info] input_csv={csv_path}")
    print(f"[info] runtime_source={runtime_source}")
    print(f"[info] r_col_used={r_col}")
    print(f"[info] r_source_method={args.r_source_method}")
    print(f"[info] allow_input_mismatch={bool(args.allow_input_mismatch)}")
    if args.min_r is not None:
        print(f"[info] min_r={args.min_r}")
    if selected_r_values is not None:
        print(f"[info] r_values={sorted(selected_r_values)}")
    if planned is not None:
        print(f"[info] planned_sweeps_used={planned}")
    print(f"[info] runtime_pairs={len(runtime_paired)}")
    print(f"[info] R values used={runtime_r_values}")
    if len(runtime_r_values) < 2:
        print(
            "[warn] Only one R value found. For a full curve vs R, use a CSV with "
            "multiple rank_init values (e.g., results/synthetic_experiment.csv)."
        )
    print(f"[info] wrote: {runtime_fig_path}")
    print(f"[info] wrote: {error_fig_path}")
    print(f"[info] wrote: {out_dir / 'runtime_summary.csv'}")
    print(f"[info] wrote: {out_dir / 'normalized_error_summary.csv'}")


if __name__ == "__main__":
    main()
