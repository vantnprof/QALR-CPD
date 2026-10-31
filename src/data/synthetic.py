"""Synthetic CP tensors with controlled collinearity and an exact target SNR (R2.7).

Factors: in every mode, u_r = sqrt(rho) g + sqrt(1-rho) e_r with a shared Gaussian g and independent e_r,
then column-normalized, so the expected within-mode cosine is about rho. Weights are 1 (manuscript Table 2).

Noise: i.i.d. Gaussian. With snr_db given, the realized noise is rescaled so that
||X_clean||^2 / ||N||^2 = 10^(snr_db/10) exactly for this instance (SNR-matched designs hold SNR fixed
while rho varies). With sigma given instead, N ~ N(0, sigma^2) as in the submitted experiments.
The realized noise floor eps* = ||N|| / ||X|| is returned for 'excess over floor'.

Note: the first-version synthetic tensors (data/synthetic*, made by data/synthetic/synthetic_generator.py) are used
as released wherever first-version numbers are compared; the tensors generated here are used for the SNR grid and the
SNR-matched collinearity designs of the revision. `python main.py synthetic` writes them to disk and checks them against
the noise floors recorded by the runs.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch


def _mode_factors(I: int, R: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    g = rng.standard_normal(I)
    E = rng.standard_normal((I, R))
    U = math.sqrt(max(rho, 0.0)) * g[:, None] + math.sqrt(max(1.0 - rho, 0.0)) * E
    return U / np.linalg.norm(U, axis=0, keepdims=True)


def mean_pairwise_cosine(U: np.ndarray) -> float:
    C = U.T @ U
    R = C.shape[0]
    return float((C.sum() - np.trace(C)) / (R * (R - 1))) if R > 1 else 1.0


def make_synthetic(
    shape: Sequence[int] = (20, 20, 20, 20),
    rank_true: int = 6,
    rho: float = 0.5,
    snr_db: Optional[float] = None,
    sigma: Optional[float] = None,
    seed: int = 0,
) -> Dict:
    if (snr_db is None) == (sigma is None):
        raise ValueError("give exactly one of snr_db or sigma")
    rng = np.random.default_rng(int(seed))
    factors = [_mode_factors(int(I), int(rank_true), float(rho), rng) for I in shape]
    letters = "abcdefghij"[: len(shape)]
    eq = ",".join(f"{l}r" for l in letters) + "->" + letters
    X_clean = np.einsum(eq, *factors)
    N = rng.standard_normal(X_clean.shape)
    if snr_db is not None:
        target = np.linalg.norm(X_clean) / (10.0 ** (snr_db / 20.0))
        N = N * (target / np.linalg.norm(N))
    else:
        N = N * float(sigma)
    X = X_clean + N
    snr_real = 10.0 * math.log10(np.linalg.norm(X_clean) ** 2 / max(np.linalg.norm(N) ** 2, 1e-300))
    return {
        "X": torch.from_numpy(X),
        "X_clean": torch.from_numpy(X_clean),
        "factors": [torch.from_numpy(U) for U in factors],
        "rank_true": int(rank_true),
        "rho": float(rho),
        "rho_realized": float(np.mean([mean_pairwise_cosine(U) for U in factors])),
        "snr_db": float(snr_real),
        "sigma": float(np.linalg.norm(N) / math.sqrt(N.size)),
        "floor": float(np.linalg.norm(N) / np.linalg.norm(X)),
        "seed": int(seed),
        "shape": list(int(I) for I in shape),
    }
