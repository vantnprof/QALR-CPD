"""QUBO terms for CP component pruning, identical to the submitted implementation.

E(z) = z^T G z - 2 z^T c + mu * 1^T z,   G_rs = lam_r lam_s <C_r, C_s>,  c_r = lam_r <X, C_r>
f(z) = E(z) + ||X||_F^2 = ||X - sum_r z_r lam_r C_r||_F^2 + mu * 1^T z   (full penalized objective)
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch

from src.pruning.exhaustive import normalize_cp_factors, inner_product_rank1


@torch.no_grad()
def cp_qubo_terms(
    X: torch.Tensor, factors: Sequence[torch.Tensor], weights: torch.Tensor
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Same arithmetic as exhaustive_prune_cp / bit_flipping_prune_cp / qaoa_prune_cp."""
    lam, U = normalize_cp_factors(factors, weights=weights)
    R = U[0].shape[1]
    lam = lam.to(device=X.device, dtype=X.dtype)
    U = [u.to(device=X.device, dtype=X.dtype) for u in U]
    overlaps = torch.ones((R, R), device=X.device, dtype=X.dtype)
    for n in range(len(U)):
        overlaps = overlaps * (U[n].T @ U[n])
    G = (lam[:, None] * lam[None, :]) * overlaps
    c = torch.zeros((R,), device=X.device, dtype=X.dtype)
    for r in range(R):
        c[r] = lam[r] * inner_product_rank1(X, [U[n][:, r] for n in range(len(U))])
    return lam, U, G, c


def energy(z: np.ndarray, G: np.ndarray, c: np.ndarray, mu: float) -> float:
    z = np.asarray(z, dtype=np.float64)
    return float(z @ G @ z - 2.0 * z @ c + mu * z.sum())


def energies_batch(Z: np.ndarray, G: np.ndarray, c: np.ndarray, mu: float) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64)
    return np.einsum("ij,jk,ik->i", Z, G, Z) - 2.0 * Z @ c + mu * Z.sum(axis=1)


def removal_terms(mask: np.ndarray, G: np.ndarray, c: np.ndarray) -> Tuple[float, float]:
    """Proposition 2 diagnostics for a selected mask (1 = kept).

    removed_energy = ||sum_{r in D} T_r||^2 = (1-x)^T G (1-x)
    cross          = 2 <E, sum_{r in D} T_r> = 2 sum_{r in D} (c_r - sum_s G_rs)
    (<E, T_r> = <X, T_r> - <Xhat, T_r> = c_r - sum_s G_rs, no tensor contraction needed.)
    """
    d = 1.0 - np.asarray(mask, dtype=np.float64)
    removed = float(d @ G @ d)
    resid_inner = c - G.sum(axis=1)
    cross = float(2.0 * d @ resid_inner)
    return removed, cross
