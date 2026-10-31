from typing import Callable, Dict, List, Optional
import torch


_EINSUM_LETTERS = "abcdefghijklmnopqrstuvwxy"  # up to 25 modes


def khatri_rao(mats: List[torch.Tensor]) -> torch.Tensor:
    """
    Khatri–Rao product (column-wise Kronecker) for a list of matrices.
    mats[k]: (I_k, R) -> output: (prod_k I_k, R)

    Ordering: last matrix varies fastest (matches row-major unfolding with
    remaining modes kept in increasing order).
    """
    if len(mats) == 0:
        raise ValueError("Need at least one matrix for Khatri-Rao.")
    R = mats[0].shape[1]
    for M in mats:
        if M.ndim != 2 or M.shape[1] != R:
            raise ValueError("All Khatri-Rao inputs must be 2D and share the same rank R.")

    KR = mats[0]
    for M in mats[1:]:
        # (I1,R) ⊙ (I2,R) -> (I1*I2, R) with M varying fastest
        KR = (KR[:, None, :] * M[None, :, :]).reshape(-1, R)
    return KR


def unfold(X: torch.Tensor, mode: int) -> torch.Tensor:
    """
    Mode-n unfolding:
      X_(mode) shape (I_mode, prod_{m!=mode} I_m)
    using permutation (mode, 0..mode-1, mode+1..N-1) and row-major reshape.
    """
    N = X.ndim
    order = [mode] + [m for m in range(N) if m != mode]
    Xp = X.permute(order).contiguous()
    return Xp.reshape(X.shape[mode], -1)


def mttkrp(X: torch.Tensor, factors: List[torch.Tensor], mode: int) -> torch.Tensor:
    """
    MTTKRP:
      M = X_(mode) @ ( ⊙_{m!=mode} A_m )
    Output: (I_mode, R)
    """
    N = X.ndim
    mats = [factors[m] for m in range(N) if m != mode]  # increasing order
    KR = khatri_rao(mats)  # last varies fastest -> matches unfold()
    Xn = unfold(X, mode)
    return Xn @ KR


def hadamard_gram(factors: List[torch.Tensor], mode: int) -> torch.Tensor:
    """
    Hadamard product of Gram matrices over all modes except `mode`:
      G = Π_{m!=mode} (A_m^T A_m)
    Output: (R, R)
    """
    R = factors[0].shape[1]
    G = torch.ones((R, R), device=factors[0].device, dtype=factors[0].dtype)
    for m, A in enumerate(factors):
        if m == mode:
            continue
        G = G * (A.T @ A)
    return G


def cp_reconstruct(factors: List[torch.Tensor], weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Reconstruct full tensor from CP factors.
    factors[n]: (I_n, R), weights: (R,)
    """
    N = len(factors)
    if N == 0:
        raise ValueError("Need at least one factor.")
    if N > len(_EINSUM_LETTERS):
        raise ValueError(f"Tensor order {N} too large for this einsum helper.")

    R = factors[0].shape[1]
    for A in factors:
        if A.shape[1] != R:
            raise ValueError("All factors must share the same rank R.")
    if weights is None:
        weights = torch.ones((R,), device=factors[0].device, dtype=factors[0].dtype)

    inds = _EINSUM_LETTERS[:N]
    r = "z"
    eq_in = ",".join([f"{inds[n]}{r}" for n in range(N)] + [r])
    eq_out = "->" + inds
    eq = eq_in + eq_out
    return torch.einsum(eq, *factors, weights)
