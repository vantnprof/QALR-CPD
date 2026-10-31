from typing import List, Sequence, Tuple, Optional
import torch

from src.utils.tensor import _EINSUM_LETTERS


@torch.no_grad()
def normalize_cp_factors(
    factors: Sequence[torch.Tensor],
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Return (lambda, normalized_factors) where each factor column has unit norm.
    Any provided weights are folded into lambda.
    """
    if len(factors) == 0:
        raise ValueError("factors must contain at least one matrix.")

    R = factors[0].shape[1]
    device = factors[0].device
    dtype = factors[0].dtype

    lam = torch.ones((R,), device=device, dtype=dtype)
    if weights is not None:
        lam = lam * weights.detach().to(device=device, dtype=dtype)

    normalized: List[torch.Tensor] = []
    for A in factors:
        if A.shape[1] != R:
            raise ValueError("All factors must share the same rank R.")
        cols = A.detach()
        col_norms = torch.linalg.norm(cols, dim=0).clamp_min(1e-12)
        normalized.append(cols / col_norms)
        lam = lam * col_norms

    return lam, normalized


@torch.no_grad()
def inner_product_rank1(X: torch.Tensor, u_cols: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Compute <X, u1 ⊗ u2 ⊗ ... ⊗ uN>, where u_cols[n] has shape (I_n,).
    """
    if X.ndim != len(u_cols):
        raise ValueError("Rank-1 vectors must match tensor order.")
    inds = _EINSUM_LETTERS[: X.ndim]
    eq = inds + "," + ",".join(inds[i] for i in range(X.ndim)) + "->"
    return torch.einsum(eq, X, *u_cols)


@torch.no_grad()
def compute_qubo_scale_ref(G: torch.Tensor, c: torch.Tensor, mu: float) -> float:
    """
    Compute QUBO normalization reference:
        scale_ref = max(max(|Q|), max(|lin|), 1.0)
    where Q is the upper-triangular QUBO matrix derived from G and
    lin = mu * 1 - 2 * c.
    """
    R = G.shape[0]
    if R == 0:
        return 1.0

    device = G.device
    dtype = G.dtype

    Q = torch.triu(G, diagonal=1) * 2.0
    Q = Q + torch.diag(torch.diagonal(G))
    lin = (float(mu) * torch.ones((R,), device=device, dtype=dtype)) - 2.0 * c

    q_max = float(Q.abs().max().item())
    lin_max = float(lin.abs().max().item())
    return max(q_max, lin_max, 1.0)


@torch.no_grad()
def exhaustive_prune_cp(
    X: torch.Tensor,
    factors: Sequence[torch.Tensor],
    mu: float,
    weights: Optional[torch.Tensor] = None,
    require_nonempty: bool = True,
    max_R_exhaustive: int = 20,
    qubo_scale: float = 1.0,
) -> Tuple[List[int], float, Tuple[torch.Tensor, List[torch.Tensor]], dict]:
    """
    Solve (exactly, by brute force):
        min_{z in {0,1}^R} || X - sum_r z_r * lambda_r * C_r ||_F^2 + mu * sum_r z_r

    where C_r are normalized rank-1 tensors built from normalized CP columns.
    This is exponential in R; we hard-stop if R is too large.

    Returns: (selected_indices, best_energy, (lambda, U_factors), diagnostics)
    """
    lam, U = normalize_cp_factors(factors, weights=weights)
    R = U[0].shape[1]
    if R > max_R_exhaustive:
        raise ValueError(
            f"R={R} too large for exhaustive search. "
            f"Set max_R_exhaustive higher (dangerous) or reduce R_max."
        )

    device = X.device
    dtype = X.dtype
    lam = lam.to(device=device, dtype=dtype)
    U = [u.to(device=device, dtype=dtype) for u in U]

    # Compute overlap Gram for C_r: <C_r, C_s> = Π_n <u_r^(n), u_s^(n)>
    overlaps = torch.ones((R, R), device=device, dtype=dtype)
    for n in range(len(U)):
        Sn = U[n].T @ U[n]  # (R,R)
        overlaps = overlaps * Sn

    # G_rs = lambda_r lambda_s <C_r, C_s>
    G = (lam[:, None] * lam[None, :]) * overlaps  # (R,R)

    # c_r = lambda_r <X, C_r>
    c = torch.zeros((R,), device=device, dtype=dtype)
    for r in range(R):
        u_cols = [U[n][:, r] for n in range(len(U))]
        c[r] = lam[r] * inner_product_rank1(X, u_cols)

    G_orig = G
    c_orig = c
    mu_orig = float(mu)

    scale_ref = compute_qubo_scale_ref(G_orig, c_orig, mu_orig)
    scale_factor = float(qubo_scale)
    if not (scale_factor > 0.0):
        raise ValueError("qubo_scale must be a positive, finite number.")
    scale_eff = scale_factor / scale_ref
    if scale_eff != 1.0:
        G = G_orig * scale_eff
        c = c_orig * scale_eff
        mu = mu_orig * scale_eff
    else:
        G = G_orig
        c = c_orig
        mu = mu_orig

    # Objective (dropping constant ||X||_F^2):
    # E(z) = z^T G z - 2 z^T c + mu * sum(z)
    best_mask, _ = exhaustive_solve_qubo(G, c, mu, require_nonempty=require_nonempty)

    if best_mask is None:
        # Can happen only if require_nonempty=True and R==0
        return [], float("inf"), (lam, U), {
            "R": R,
            "best_mask": None,
            "best_E_no_const": float("inf"),
            "qubo_scale": scale_factor,
            "qubo_scale_ref": scale_ref,
            "qubo_scale_eff": scale_eff,
        }

    selected = [i for i in range(R) if (best_mask >> i) & 1]
    z_bits = [(best_mask >> i) & 1 for i in range(R)]
    z = torch.tensor(z_bits, device=device, dtype=dtype)
    best_E = (z @ (G_orig @ z)) - 2.0 * (z @ c_orig) + mu_orig * z.sum()

    diagnostics = {
        "R": R,
        "best_mask": best_mask,
        "best_E_no_const": float(best_E.detach().cpu()),
        "qubo_scale": scale_factor,
        "qubo_scale_ref": scale_ref,
        "qubo_scale_eff": scale_eff,
    }
    return selected, float(best_E.detach().cpu()), (lam, U), diagnostics


# ---------------------------------------------------------------------------
# Revision: exact enumeration shared by the in-loop pruning call and the
# solver benchmark. Two implementations with the same contract:
#   argmin_{z in {0,1}^R, z != 0 if require_nonempty} z^T G z - 2 z^T c + mu 1^T z
# Ties are broken toward the smallest mask integer (bit i <-> component i),
# exactly as the original loop did (strict "<" while scanning masks in order).
#   impl="vectorized" (default): chunked batched evaluation, ~100x faster.
#   impl="loop": the original per-mask loop, kept for exact reproduction.
# Select with the environment variable QALR_ES_IMPL=loop|vectorized.
# ---------------------------------------------------------------------------
import os as _os


def _mask_bits(start: int, stop: int, R: int, device, dtype) -> torch.Tensor:
    idx = torch.arange(start, stop, device=device, dtype=torch.int64)
    shifts = torch.arange(R, device=device, dtype=torch.int64)
    return ((idx.unsqueeze(1) >> shifts) & 1).to(dtype)


@torch.no_grad()
def exhaustive_solve_qubo(
    G: torch.Tensor,
    c: torch.Tensor,
    mu: float,
    require_nonempty: bool = True,
    impl: Optional[str] = None,
    chunk_bits: int = 16,
) -> Tuple[Optional[int], float]:
    """Exact minimizer of the pruning QUBO. Returns (best_mask_int, best_energy)."""
    R = int(G.shape[0])
    if R == 0:
        return None, float("inf")
    impl = (impl or _os.environ.get("QALR_ES_IMPL", "vectorized")).lower().strip()
    start = 1 if require_nonempty else 0
    total = 1 << R
    best_E = torch.tensor(float("inf"), device=G.device, dtype=G.dtype)
    best_mask: Optional[int] = None
    if impl == "loop":
        for mask in range(start, total):
            z_bits = [(mask >> i) & 1 for i in range(R)]
            z = torch.tensor(z_bits, device=G.device, dtype=G.dtype)
            E = (z @ (G @ z)) - 2.0 * (z @ c) + mu * z.sum()
            if E < best_E:
                best_E = E
                best_mask = mask
        return best_mask, float(best_E)
    chunk = 1 << int(chunk_bits)
    for s0 in range(start, total, chunk):
        s1 = min(s0 + chunk, total)
        Z = _mask_bits(s0, s1, R, G.device, G.dtype)
        E = ((Z @ G) * Z).sum(dim=1) - 2.0 * (Z @ c) + mu * Z.sum(dim=1)
        i = int(torch.argmin(E).item())  # first occurrence within the chunk
        if E[i] < best_E:  # strict: earlier chunks win ties
            best_E = E[i]
            best_mask = s0 + i
    return best_mask, float(best_E)
