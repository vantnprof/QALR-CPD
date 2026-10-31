"""
Visualization for QAOA shot-count and circuit-depth ablation on the amino acid dataset.

Consumes CSV outputs from `ablation_acid_shot_circuit_depth.py` and produces:
  1) rel. error vs shots (curves per depth)
  2) selected rank vs shots (curves per depth)
  3) normalized runtime vs shots (curves per depth; normalized by min shots)
"""

from __future__ import annotations

import argparse
import math
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLOR_CYCLE = ("C0", "C1", "C2", "C3", "C4", "C5")
MARKER_CYCLE = ("o", "s", "^", "D", "v", "P", "X")
LINESTYLE_CYCLE = ("-", "--", "-.", ":", "-", "--", "-.")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[warn] missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _parse_list_int(raw: str) -> List[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _filter_status_ok(df: pd.DataFrame) -> pd.DataFrame:
    if "status" in df.columns:
        return df[df["status"] == "ok"].copy()
    return df.copy()


def _sanitize_tag(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in (
        "qaoa_reps",
        "qaoa_shots_effective",
        "final_rel_error",
        "final_rank",
        "total_time_s",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _remap_shots(df: pd.DataFrame, from_value: int, to_value: int) -> pd.DataFrame:
    if "qaoa_shots_effective" not in df.columns:
        return df
    df = df.copy()
    mask = df["qaoa_shots_effective"] == from_value
    if mask.any():
        df.loc[mask, "qaoa_shots_effective"] = to_value
        print(f"[info] remapped shots {from_value} -> {to_value} for plotting")
    return df


def _aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = ["dataset_tag", "qaoa_reps", "qaoa_shots_effective"]
    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            final_rel_error_mean=("final_rel_error", "mean"),
            final_rel_error_std=("final_rel_error", "std"),
            final_rank_mean=("final_rank", "mean"),
            final_rank_std=("final_rank", "std"),
            total_time_mean=("total_time_s", "mean"),
            total_time_std=("total_time_s", "std"),
            count=("final_rel_error", "count"),
        )
        .reset_index()
    )
    for col in (
        "final_rel_error_std",
        "final_rank_std",
        "total_time_std",
    ):
        if col in agg.columns:
            agg[col] = agg[col].fillna(0.0)
    return agg


def _sorted_values(df: pd.DataFrame, col: str, values: Optional[List[int]]) -> List[int]:
    if values:
        return list(values)
    vals = df[col].dropna().unique().tolist()
    return sorted(int(v) for v in vals)


def _depth_style(depth: int, idx: int) -> Dict[str, str]:
    color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
    marker = MARKER_CYCLE[idx % len(MARKER_CYCLE)]
    linestyle = LINESTYLE_CYCLE[idx % len(LINESTYLE_CYCLE)]
    return {"color": color, "marker": marker, "linestyle": linestyle}


def _build_style_map(depths: Iterable[int]) -> Dict[int, Dict[str, str]]:
    depth_list = [int(d) for d in depths]
    style_map: Dict[int, Dict[str, str]] = {}
    for idx, depth in enumerate(depth_list):
        style_map[depth] = _depth_style(depth, idx)
    return style_map


def _shot_tick_labels(shot_values: List[int]) -> List[str]:
    labels: List[str] = []
    for v in shot_values:
        if isinstance(v, (int, np.integer)) and v > 0 and (v & (v - 1) == 0):
            exp = int(round(math.log2(int(v))))
            labels.append(rf"$2^{{{exp}}}$")
        else:
            labels.append(str(v))
    return labels


def _series_for_depth(
    df: pd.DataFrame,
    depth: int,
    shot_values: Iterable[int],
    mean_col: str,
    std_col: str,
) -> Tuple[List[float], List[float]]:
    sub = df[df["qaoa_reps"] == depth].copy()
    if sub.empty:
        return [], []
    sub = sub.set_index("qaoa_shots_effective")
    means: List[float] = []
    stds: List[float] = []
    for shots in shot_values:
        if shots in sub.index:
            means.append(float(sub.loc[shots, mean_col]))
            stds.append(float(sub.loc[shots, std_col]))
        else:
            means.append(float("nan"))
            stds.append(float("nan"))
    return means, stds


def _global_baseline(
    df: pd.DataFrame,
    depths: Iterable[int],
    shot_values: Optional[List[int]],
    mean_col: str,
) -> Optional[Tuple[int, int, float]]:
    if df.empty:
        return None
    depth_candidates = sorted(int(v) for v in df["qaoa_reps"].dropna().unique())
    shot_candidates = sorted(int(v) for v in df["qaoa_shots_effective"].dropna().unique())
    if not depth_candidates or not shot_candidates:
        return None
    depth_min = min(depth_candidates)
    if depth_min not in set(int(d) for d in depths):
        depth_min = min(int(d) for d in depths)
    if shot_values:
        available = [s for s in shot_values if s in set(shot_candidates)]
        shots_min = min(available) if available else min(shot_candidates)
    else:
        shots_min = min(shot_candidates)
    row = df[
        (df["qaoa_reps"] == depth_min)
        & (df["qaoa_shots_effective"] == shots_min)
    ]
    if row.empty:
        return None
    base = float(row[mean_col].iloc[0])
    return int(depth_min), int(shots_min), base


def _plot_metric(
    df: pd.DataFrame,
    depths: List[int],
    shot_values: List[int],
    mean_col: str,
    std_col: str,
    ylabel: str,
    out_path: str,
    style_map: Optional[Dict[int, Dict[str, str]]] = None,
    show_legend: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x_pos = np.arange(len(shot_values))

    for idx, depth in enumerate(depths):
        means, stds = _series_for_depth(df, depth, shot_values, mean_col, std_col)
        if not means:
            continue
        if style_map is not None and depth in style_map:
            style = style_map[depth]
        else:
            style = _depth_style(depth, idx)
        label = rf"$p={depth}$"
        ax.plot(
            x_pos,
            means,
            color=style.get("color", None),
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
            linewidth=1.2,
            markersize=4,
            label=label,
        )

    ax.set_xlabel("Shots")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(frameon=False, loc="best")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(_shot_tick_labels(shot_values), rotation=30, ha="right")
    fig.tight_layout()

    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=1000, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] wrote {out_path}")


def _plot_normalized_runtime(
    df: pd.DataFrame,
    depths: List[int],
    shot_values: List[int],
    out_path: str,
    style_map: Optional[Dict[int, Dict[str, str]]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x_pos = np.arange(len(shot_values))
    baseline = _global_baseline(df, depths, shot_values, "total_time_mean")
    if baseline is None:
        print("[warn] no baseline runtime found; skipping normalized runtime plot.")
        plt.close(fig)
        return
    base_depth, base_shots, base_time = baseline
    if base_time is None or np.isnan(base_time) or base_time <= 0:
        print("[warn] baseline runtime invalid; skipping normalized runtime plot.")
        plt.close(fig)
        return

    for idx, depth in enumerate(depths):
        means, stds = _series_for_depth(
            df, depth, shot_values, "total_time_mean", "total_time_std"
        )
        if not means:
            continue
        norm_means = [
            float(m) / base_time if not np.isnan(m) else float("nan") for m in means
        ]
        norm_stds = [
            float(s) / base_time if not np.isnan(s) else float("nan") for s in stds
        ]
        if style_map is not None and depth in style_map:
            style = style_map[depth]
        else:
            style = _depth_style(depth, idx)
        label = rf"$p={depth}$"
        ax.plot(
            x_pos,
            norm_means,
            color=style.get("color", None),
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
            linewidth=1.4,
            markersize=4,
            label=label,
        )

    ax.set_xlabel("Shots")
    ax.set_ylabel("Normalized running time")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="best")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(_shot_tick_labels(shot_values), rotation=30, ha="right")
    fig.tight_layout()

    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=1000, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] wrote {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot shot/depth ablation results for QALR-CPD on amino acid data."
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="results/v1/ablation_acid_shot_circuit_depth_summary.csv",
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default="reports/figures/v1/ablation_acid_shot_circuit_depth",
    )
    parser.add_argument("--dataset-tag", type=str, default=None)
    parser.add_argument(
        "--shots-list",
        type=str,
        default="128,2048,4096,8192,16384",
        help="Comma-separated shot values to show on the x-axis.",
    )
    parser.add_argument(
        "--depth-list",
        type=str,
        default="1,3,5",
        help="Comma-separated depth (reps) values to plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_csv(args.summary_csv)
    if df.empty:
        return

    df = _filter_status_ok(df)
    df = _coerce_numeric(df)
    df = _remap_shots(df, from_value=8912, to_value=8192)
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

    shot_values = _parse_list_int(args.shots_list) if args.shots_list else None
    depth_values = _parse_list_int(args.depth_list) if args.depth_list else None

    for dataset_tag in dataset_tags:
        sub = agg[agg["dataset_tag"] == dataset_tag].copy()
        if sub.empty:
            print(f"[warn] no rows for dataset_tag={dataset_tag}")
            continue

        shots_sorted = _sorted_values(sub, "qaoa_shots_effective", shot_values)
        depths_sorted = _sorted_values(sub, "qaoa_reps", depth_values)
        if not shots_sorted:
            print(f"[warn] no shots for dataset_tag={dataset_tag}")
            continue
        if not depths_sorted:
            print(f"[warn] no depths for dataset_tag={dataset_tag}")
            continue
        err_depths = [d for d in depths_sorted if d == 5]
        if not err_depths:
            print("[warn] depth p=5 not available; using all depths for rel. error plot.")
            err_depths = depths_sorted
        rank_depths = [d for d in depths_sorted if d == 5]
        if not rank_depths:
            print("[warn] depth p=5 not available; using all depths for rank plot.")
            rank_depths = depths_sorted
        style_map = _build_style_map(depths_sorted)

        tag_safe = _sanitize_tag(str(dataset_tag))
        err_path = os.path.join(
            args.fig_dir, f"ablation_shots_rel_error_{tag_safe}.png"
        )
        rank_path = os.path.join(
            args.fig_dir, f"ablation_shots_rank_{tag_safe}.png"
        )
        runtime_path = os.path.join(
            args.fig_dir, f"ablation_shots_runtime_{tag_safe}.png"
        )

        _plot_metric(
            sub,
            depths=err_depths,
            shot_values=shots_sorted,
            mean_col="final_rel_error_mean",
            std_col="final_rel_error_std",
            ylabel="Rel. error",
            out_path=err_path,
            style_map=style_map,
            show_legend=False,
        )
        _plot_metric(
            sub,
            depths=rank_depths,
            shot_values=shots_sorted,
            mean_col="final_rank_mean",
            std_col="final_rank_std",
            ylabel="Selected rank",
            out_path=rank_path,
            style_map=style_map,
        )
        _plot_normalized_runtime(
            sub,
            depths=depths_sorted,
            shot_values=shots_sorted,
            out_path=runtime_path,
            style_map=style_map,
        )


if __name__ == "__main__":
    main()
