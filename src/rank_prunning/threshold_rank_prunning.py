from typing import List, Sequence, Tuple, Optional, Dict
import torch

from src.rank_prunning.exhaustive_search_rank_prunning import normalize_cp_factors


@torch.no_grad()
def threshold_prune_cp(
    factors: Sequence[torch.Tensor],
    weights: Optional[torch.Tensor] = None,
    top_k: Optional[int] = None,
    weight_threshold: Optional[float] = None,
    require_nonempty: bool = True,
) -> Tuple[List[int], float, Tuple[torch.Tensor, List[torch.Tensor]], Dict]:
    """
    Top-K / weight-threshold pruning using normalized CP weights.

    Returns:
      selected_indices, score, (lambda, U_factors), diagnostics
    """
    lam, U = normalize_cp_factors(factors, weights=weights)
    R = U[0].shape[1] if len(U) else 0

    if R == 0:
        diagnostics = {
            "R": 0,
            "best_mask": [],
            "top_k": top_k,
            "weight_threshold": weight_threshold,
            "weight_sum": 0.0,
        }
        return [], 0.0, (lam, U), diagnostics

    abs_lam = lam.abs().detach().cpu().tolist()
    sorted_idx = sorted(range(R), key=lambda i: (-abs_lam[i], i))

    selected = list(sorted_idx)
    if weight_threshold is not None:
        selected = [i for i in selected if abs_lam[i] >= weight_threshold]

    if top_k is not None:
        top_k = int(top_k)
        if top_k < 0:
            raise ValueError("top_k must be non-negative.")
        if top_k == 0:
            selected = []
        else:
            selected = selected[: min(top_k, len(selected))]

    if require_nonempty and len(selected) == 0:
        selected = [sorted_idx[0]]

    selected_sorted = sorted(selected)
    selected_set = set(selected_sorted)
    mask = [1 if i in selected_set else 0 for i in range(R)]
    weight_sum = float(sum(abs_lam[i] for i in selected_sorted))

    diagnostics = {
        "R": R,
        "best_mask": mask,
        "top_k": top_k,
        "weight_threshold": weight_threshold,
        "weight_sum": weight_sum,
    }
    return selected_sorted, weight_sum, (lam, U), diagnostics
