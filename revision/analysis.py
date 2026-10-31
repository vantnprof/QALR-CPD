"""Shared analysis: penalty plateaus (C1) and summary statistics."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np


def mean_std(xs) -> Tuple[float, float]:
    xs = [float(x) for x in xs if x not in ("", None) and np.isfinite(float(x))]
    if not xs:
        return float("nan"), float("nan")
    return float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def rank_profile(runs: List[Dict], dataset: str, method: str = "ES", rank_init: int = 20) -> List[Dict]:
    """Per penalty: modal selected rank, its stability (share of seeds at the mode), mean rank, exact recovery."""
    by_mu = defaultdict(list)
    rt = None
    for r in runs:
        if r["dataset"] == dataset and r["method"] == method and int(r["rank_init"]) == rank_init and r["mu"] not in ("", "None"):
            if r.get("tag", "") not in ("", None):
                continue
            by_mu[float(r["mu"])].append(int(r["final_rank"]))
            if r.get("rank_true") not in ("", None):
                rt = int(float(r["rank_true"]))
    rows = []
    for mu in sorted(by_mu):
        ranks = by_mu[mu]
        mode, cnt = Counter(ranks).most_common(1)[0]
        rows.append({
            "mu": mu, "n": len(ranks), "mode": mode, "stability": cnt / len(ranks), "mean": float(np.mean(ranks)),
            "std": float(np.std(ranks, ddof=1)) if len(ranks) > 1 else 0.0,
            "exact": (sum(1 for x in ranks if x == rt) / len(ranks)) if rt is not None else float("nan"),
            "rank_true": rt if rt is not None else "",
        })
    return rows


def widest_plateau(profile: List[Dict], min_stability: float = 0.6, exclude_full: int = None) -> Dict:
    """Longest run of consecutive grid penalties with the same modal rank and stability >= min_stability.
    Ranks equal to exclude_full (R_max, i.e. nothing pruned) and rank 1 are not eligible."""
    best, cur = None, []
    for row in profile + [None]:
        ok = row is not None and row["stability"] >= min_stability and row["mode"] not in (exclude_full, 1)
        if ok and cur and row["mode"] == cur[-1]["mode"]:
            cur.append(row)
            continue
        if cur and (best is None or len(cur) > len(best)):
            best = cur
        cur = [row] if ok else []
    if not best:
        return {}
    return {"rank": best[0]["mode"], "mu_lo": best[0]["mu"], "mu_hi": best[-1]["mu"], "width": len(best),
            "mu_mid": best[len(best) // 2]["mu"], "stability": float(np.mean([r["stability"] for r in best]))}
