from __future__ import annotations

from typing import List, Sequence, Tuple, Optional, Dict
import random
import torch

from src.pruning.exhaustive import (
    normalize_cp_factors,
    inner_product_rank1,
    compute_qubo_scale_ref,
)


def _build_qubo_terms(
    X: torch.Tensor,
    factors: Sequence[torch.Tensor],
    weights: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
    lam, U = normalize_cp_factors(factors, weights=weights)
    R = U[0].shape[1] if len(U) else 0

    device = X.device
    dtype = X.dtype
    lam = lam.to(device=device, dtype=dtype)
    U = [u.to(device=device, dtype=dtype) for u in U]

    overlaps = torch.ones((R, R), device=device, dtype=dtype)
    for n in range(len(U)):
        overlaps = overlaps * (U[n].T @ U[n])

    G = (lam[:, None] * lam[None, :]) * overlaps

    c = torch.zeros((R,), device=device, dtype=dtype)
    for r in range(R):
        u_cols = [U[n][:, r] for n in range(len(U))]
        c[r] = lam[r] * inner_product_rank1(X, u_cols)

    return lam, U, G, c


def _init_binary_vector(
    R: int,
    init_mode: str,
    rng: random.Random,
    device: torch.device,
    dtype: torch.dtype,
    warm_start: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    mode = (init_mode or "zeros").lower().strip()
    if mode in ("zeros", "zero"):
        z = torch.zeros((R,), device=device, dtype=dtype)
    elif mode in ("random", "rand"):
        z = torch.tensor([rng.randint(0, 1) for _ in range(R)], device=device, dtype=dtype)
    elif mode in ("warm_start", "warm-start", "warm"):
        if warm_start is None:
            z = torch.zeros((R,), device=device, dtype=dtype)
        else:
            if len(warm_start) != R:
                raise ValueError("warm_start length must match R.")
            z = torch.tensor([1 if int(v) else 0 for v in warm_start], device=device, dtype=dtype)
    else:
        raise ValueError("init_mode must be one of: zeros, random, warm_start")
    return z


@torch.no_grad()
def bit_flipping_prune_cp(
    X: torch.Tensor,
    factors: Sequence[torch.Tensor],
    mu: float,
    weights: Optional[torch.Tensor] = None,
    require_nonempty: bool = True,
    max_R_bitflip: Optional[int] = None,
    budget: Optional[int] = 200,
    restarts: int = 1,
    init_mode: str = "zeros",
    tie_break: str = "first",
    tol: float = 1e-12,
    seed: int = 0,
    warm_start: Optional[Sequence[int]] = None,
    verbose: bool = False,
    qubo_scale: float = 1.0,
) -> Tuple[List[int], float, Tuple[torch.Tensor, List[torch.Tensor]], Dict]:
    """
    Bit-flipping local search for the CP rank-pruning QUBO:

        min_{z in {0,1}^R} || X - sum_r z_r * lambda_r * C_r ||_F^2 + mu * sum_r z_r

    Returns: (selected_indices, best_energy, (lambda, U_factors), diagnostics)
    """
    lam, U, G, c = _build_qubo_terms(X, factors, weights)
    R = U[0].shape[1] if len(U) else 0

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

    if R == 0:
        diagnostics = {
            "R": 0,
            "best_mask": [],
            "best_E_no_const": 0.0,
            "restarts": 0,
            "budget": 0,
            "init_mode": init_mode,
            "tie_break": tie_break,
            "tol": float(tol),
            "qubo_scale": scale_factor,
            "qubo_scale_ref": scale_ref,
            "qubo_scale_eff": scale_eff,
        }
        return [], 0.0, (lam, U), diagnostics

    if max_R_bitflip is not None and R > int(max_R_bitflip):
        raise ValueError(
            f"R={R} too large for bit-flip cap {max_R_bitflip}. "
            f"Increase max_R_bitflip or reduce rank."
        )

    device = X.device
    dtype = X.dtype
    lin = (mu * torch.ones((R,), device=device, dtype=dtype)) - 2.0 * c
    diag_G = torch.diagonal(G)

    budget = 0 if budget is None else int(budget)
    restarts = max(1, int(restarts))
    tol = float(tol)
    tie_break_mode = (tie_break or "first").lower().strip()

    rng = random.Random(int(seed))

    best_E_scaled = torch.tensor(float("inf"), device=device, dtype=dtype)
    best_mask: Optional[List[int]] = None
    best_selected: List[int] = []

    best_single_idx: Optional[int] = None
    if require_nonempty and R > 0:
        best_single_idx = int(torch.argmin(diag_G + lin).item())

    for r_idx in range(restarts):
        # Advance RNG between restarts to keep deterministic but different starts.
        if r_idx > 0:
            rng.random()

        z = _init_binary_vector(
            R=R,
            init_mode=init_mode,
            rng=rng,
            device=device,
            dtype=dtype,
            warm_start=warm_start,
        )

        if require_nonempty and R > 0 and z.sum().item() < 0.5:
            z[best_single_idx] = 1.0

        g = G @ z
        E = (z @ g) + (lin @ z)
        if verbose:
            print(f"[bitflip] restart={r_idx+1}/{restarts} init_sum={int(z.sum().item())} E={float(E):.6e}")

        for step in range(budget):
            s = 1.0 - 2.0 * z
            delta = s * (2.0 * g + lin) + diag_G

            if require_nonempty and R > 0 and z.sum().item() <= 1.0:
                one_idx = torch.nonzero(z > 0.5, as_tuple=False).flatten()
                if one_idx.numel() == 1:
                    delta[one_idx[0]] = float("inf")

            min_delta = torch.min(delta)
            if min_delta >= -tol:
                if verbose:
                    print(
                        f"[bitflip] restart={r_idx+1} step={step} stop: "
                        f"min_delta={float(min_delta):.6e} >= -tol"
                    )
                break

            if tie_break_mode in ("random", "random_among_best", "random-among-best"):
                candidates = torch.nonzero(delta <= min_delta + 1e-12, as_tuple=False).flatten().tolist()
                i_star = int(rng.choice(candidates))
            else:
                i_star = int(torch.argmin(delta).item())

            s_i = 1.0 - 2.0 * z[i_star]
            z[i_star] = 1.0 - z[i_star]
            E = E + delta[i_star]
            g = g + s_i * G[:, i_star]
            if verbose:
                print(
                    f"[bitflip] restart={r_idx+1} step={step} flip={i_star} "
                    f"delta={float(delta[i_star]):.6e} E={float(E):.6e} "
                    f"sum={int(z.sum().item())}"
                )

        if require_nonempty and R > 0 and z.sum().item() < 0.5:
            z[best_single_idx] = 1.0
            g = G[:, best_single_idx].clone()
            E = diag_G[best_single_idx] + lin[best_single_idx]
            if verbose:
                print(
                    f"[bitflip] restart={r_idx+1} corrected empty -> idx={best_single_idx} "
                    f"E={float(E):.6e}"
                )

        if E < best_E_scaled:
            best_E_scaled = E
            best_mask = [1 if float(v) > 0.5 else 0 for v in z.detach().cpu().tolist()]
            best_selected = [i for i, b in enumerate(best_mask) if b == 1]
            if verbose:
                print(
                    f"[bitflip] restart={r_idx+1} new_best "
                    f"E={float(best_E_scaled):.6e} selected={best_selected}"
                )

    if best_mask is None:
        best_E_val = float("inf")
    else:
        z_best = torch.tensor(best_mask, device=device, dtype=dtype)
        best_E = (z_best @ (G_orig @ z_best)) - 2.0 * (z_best @ c_orig) + mu_orig * z_best.sum()
        best_E_val = float(best_E.detach().cpu())

    diagnostics = {
        "R": R,
        "best_mask": best_mask,
        "best_E_no_const": best_E_val,
        "restarts": restarts,
        "budget": budget,
        "init_mode": init_mode,
        "tie_break": tie_break_mode,
        "tol": tol,
        "qubo_scale": scale_factor,
        "qubo_scale_ref": scale_ref,
        "qubo_scale_eff": scale_eff,
    }
    return best_selected, best_E_val, (lam, U), diagnostics


if __name__ == "__main__":
    torch.manual_seed(0)

    def _make_orthonormal_factors(shape, rank, seed=0, dtype=torch.float64):
        g = torch.Generator()
        g.manual_seed(seed)
        factors = []
        for I in shape:
            A = torch.randn((I, rank), generator=g, dtype=dtype)
            Q, _ = torch.linalg.qr(A, mode="reduced")
            factors.append(Q[:, :rank].contiguous())
        return factors

    def _build_tensor(weights, factors):
        letters = "abcdefghijklmnopqrstuvwxyz"
        inds = letters[: len(factors)]
        eq = "r," + ",".join(f"{i}r" for i in inds) + "->" + "".join(inds)
        return torch.einsum(eq, weights, *factors)

    def _run_basic_tests():
        # Test 1: matches expected selection on separable instance
        shape = (6, 5, 4)
        R = 4
        weights = torch.tensor([1.2, 0.8, 0.3, 0.1], dtype=torch.float64)
        factors = _make_orthonormal_factors(shape, R, seed=0, dtype=torch.float64)
        X = _build_tensor(weights, factors)
        mu = 0.5
        selected, E, _, diag = bit_flipping_prune_cp(
            X,
            factors,
            mu=mu,
            weights=weights,
            require_nonempty=True,
            budget=20,
            restarts=1,
            init_mode="zeros",
            verbose=True,
        )
        assert selected == [0, 1], f"unexpected selection: {selected} diag={diag}"
        assert E < 0.0, f"unexpected energy: {E}"

        # Test 2: require_nonempty forces a single selection for large mu
        mu = 10.0
        selected, _, _, _ = bit_flipping_prune_cp(
            X,
            factors,
            mu=mu,
            weights=weights,
            require_nonempty=True,
            budget=0,
            restarts=1,
            init_mode="zeros",
        )
        assert selected == [0], f"expected single best index, got {selected}"

        # Test 3: warm-start with zero budget preserves start
        warm_start = [1, 0, 1, 0]
        selected, _, _, _ = bit_flipping_prune_cp(
            X,
            factors,
            mu=0.1,
            weights=weights,
            require_nonempty=True,
            budget=0,
            restarts=1,
            init_mode="warm_start",
            warm_start=warm_start,
        )
        assert selected == [0, 2], f"warm-start not preserved: {selected}"

        print("bit_flipping_rank_prunning: basic tests passed.")

    def _run_objective_comparison():
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"[bitflip] matplotlib unavailable, skipping plot. Error: {exc}")
            return

        from src.pruning.exhaustive import exhaustive_prune_cp

        shape = (8, 9, 10)
        R = 6
        weights = torch.tensor([1.2, 0.9, 0.6, 0.4, 0.2, 0.1], dtype=torch.float64)
        factors = _make_orthonormal_factors(shape, R, seed=123, dtype=torch.float64)
        X = _build_tensor(weights, factors)

        mu_list = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 6.0]
        E_exh = []
        E_bf = []
        ranks_exh = []
        ranks_bf = []

        for mu in mu_list:
            selected_exh, E_exh_val, _, _ = exhaustive_prune_cp(
                X,
                factors,
                mu=mu,
                weights=weights,
                require_nonempty=True,
                max_R_exhaustive=12,
            )
            selected_bf, E_bf_val, _, _ = bit_flipping_prune_cp(
                X,
                factors,
                mu=mu,
                weights=weights,
                require_nonempty=True,
                budget=200,
                restarts=5,
                init_mode="random",
                tie_break="random_among_best",
                seed=0,
            )
            E_exh.append(E_exh_val)
            E_bf.append(E_bf_val)
            ranks_exh.append(len(selected_exh))
            ranks_bf.append(len(selected_bf))
            print(
                f"[bitflip] mu={mu:.3f} E_exh={E_exh_val:.6e} E_bf={E_bf_val:.6e} "
                f"gap={E_bf_val - E_exh_val:+.3e} "
                f"rank_exh={len(selected_exh)} rank_bf={len(selected_bf)}"
            )

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(mu_list, E_exh, "o-", label="Exhaustive")
        ax.plot(mu_list, E_bf, "s--", label="Bit-flip")
        ax.set_xlabel("mu")
        ax.set_ylabel("Objective (no const)")
        ax.set_title("Bit-flip vs Exhaustive Objective")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        out_path = "bitflip_vs_exhaustive_objective.png"
        plt.savefig(out_path, dpi=200)
        print(f"[bitflip] saved plot to {out_path}")

    _run_basic_tests()
    _run_objective_comparison()
