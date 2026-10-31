import argparse
import csv
import math
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch

from src.pruning.exhaustive import (
    exhaustive_prune_cp,
    inner_product_rank1,
    normalize_cp_factors,
)

try:
    from src.pruning.qaoa import qaoa_prune_cp

    _HAS_QAOA = True
    _QAOA_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on optional qiskit install
    _HAS_QAOA = False
    _QAOA_IMPORT_ERROR = str(exc)


def _parse_int_list(value: str) -> Tuple[int, ...]:
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def _parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _mask_to_bits(selected: Sequence[int], R: int) -> List[int]:
    bits = [0] * R
    for idx in selected:
        if 0 <= idx < R:
            bits[idx] = 1
    return bits


def _bits_to_str(bits: Sequence[int]) -> str:
    return "".join(str(int(b)) for b in bits)


def _build_frozen_instance(
    shape: Sequence[int],
    R: int,
    seed: int,
    noise_std: float = 0.0,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dtype = torch.float64
    factors = [torch.rand(s, R, dtype=dtype) for s in shape]
    weights = torch.ones(R, dtype=dtype)

    weak_count = max(1, R // 5)
    for r in range(weak_count):
        for A in factors:
            A[:, r] *= 0.15
        weights[r] *= 0.2

    letters = "abcdefghijklmnopqrstuvwxyz"
    inds = letters[: len(shape)]
    eq = "r," + ",".join(f"{i}r" for i in inds) + "->" + "".join(inds)
    X = torch.einsum(eq, weights, *factors)
    if noise_std > 0:
        X = X + noise_std * torch.randn_like(X)
    return X, factors, weights


def _build_shared_energy(
    X: torch.Tensor,
    factors: Sequence[torch.Tensor],
    weights: torch.Tensor,
):
    lam, U = normalize_cp_factors(factors, weights=weights)
    R = U[0].shape[1]

    overlaps = torch.ones((R, R), device=X.device, dtype=X.dtype)
    for n in range(len(U)):
        overlaps = overlaps * (U[n].T @ U[n])

    G = (lam[:, None] * lam[None, :]) * overlaps

    c = torch.zeros((R,), device=X.device, dtype=X.dtype)
    for r in range(R):
        u_cols = [U[n][:, r] for n in range(len(U))]
        c[r] = lam[r] * inner_product_rank1(X, u_cols)

    def energy(z_bits: Sequence[int], mu: float) -> float:
        z = torch.tensor(z_bits, device=X.device, dtype=X.dtype)
        E = (z @ (G @ z)) - 2.0 * (z @ c) + mu * z.sum()
        return float(E.detach().cpu())

    return energy


def _format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check CP pruning objective vs mu monotonicity.")
    parser.add_argument("--shape", type=str, default="6,7,5", help="Tensor shape as comma list.")
    parser.add_argument("--rank", type=int, default=10, help="CP rank (<= 20 for exhaustive).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--noise-std", type=float, default=0.0, help="Noise std to add to X.")
    parser.add_argument(
        "--mu-list",
        type=str,
        default="0.0,0.001,0.005,0.01,0.05,0.1,0.2,0.4,1.0,2.0,3.0,5.0,10.0,100.0",
        help="Comma-separated mu values.",
    )
    parser.add_argument("--max-r-exhaustive", type=int, default=20, help="Max R for exhaustive.")
    parser.add_argument("--max-r-qaoa", type=int, default=24, help="Max R for QAOA.")
    parser.add_argument("--qaoa-reps", type=int, default=2, help="QAOA depth (reps).")
    parser.add_argument("--shots", type=int, default=1024, help="QAOA shots.")
    parser.add_argument("--maxiter", type=int, default=200, help="QAOA optimizer iterations.")
    parser.add_argument("--num-restarts", type=int, default=1, help="QAOA restarts.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k bitstrings to evaluate.")
    parser.add_argument("--skip-qaoa", action="store_true", help="Skip QAOA pruning.")
    parser.add_argument(
        "--output-csv",
        type=str,
        default="debug_mu_sweep.csv",
        help="CSV output path.",
    )
    args = parser.parse_args()

    shape = _parse_int_list(args.shape)
    R = int(args.rank)
    if R > 20:
        raise ValueError("Rank must be <= 20 for exhaustive search.")
    mu_list = _parse_float_list(args.mu_list)
    if not mu_list:
        raise ValueError("mu_list must contain at least one value.")

    X, factors, weights = _build_frozen_instance(
        shape=shape,
        R=R,
        seed=args.seed,
        noise_std=args.noise_std,
    )
    shared_energy = _build_shared_energy(X, factors, weights)

    run_qaoa = _HAS_QAOA and not args.skip_qaoa
    if not run_qaoa:
        if args.skip_qaoa:
            print("QAOA disabled via --skip-qaoa.")
        elif not _HAS_QAOA:
            print(f"QAOA import unavailable; skipping QAOA. Error: {_QAOA_IMPORT_ERROR}")

    header = (
        "mu | rank_exh | rank_qaoa | E_exh | E_shared_exh | E_shared_qaoa | mask_exh | mask_qaoa"
    )
    print(header)
    print("-" * len(header))

    rows = []
    monotonic_fail = False
    energy_mismatch = False
    qaoa_energy_mismatch = False
    prev_rank = None
    tol = 1e-5

    for mu in mu_list:
        selected_exh, E_exh, _, _ = exhaustive_prune_cp(
            X,
            factors,
            mu,
            weights=weights,
            require_nonempty=True,
            max_R_exhaustive=args.max_r_exhaustive,
        )
        z_exh = _mask_to_bits(selected_exh, R)
        E_shared_exh = shared_energy(z_exh, mu)
        rank_exh = len(selected_exh)

        if abs(E_shared_exh - E_exh) > tol:
            energy_mismatch = True

        if prev_rank is not None and rank_exh > prev_rank:
            monotonic_fail = True
        prev_rank = rank_exh

        if run_qaoa:
            selected_qaoa, E_qaoa, _, _ = qaoa_prune_cp(
                X,
                factors,
                mu,
                weights=weights,
                require_nonempty=True,
                max_R_qaoa=args.max_r_qaoa,
                qaoa_reps=args.qaoa_reps,
                shots=args.shots,
                maxiter=args.maxiter,
                seed=args.seed,
                num_restarts=args.num_restarts,
                top_k=args.top_k,
            )
            z_qaoa = _mask_to_bits(selected_qaoa, R)
            E_shared_qaoa = shared_energy(z_qaoa, mu)
            rank_qaoa = len(selected_qaoa)
            if abs(E_shared_qaoa - E_qaoa) > tol:
                qaoa_energy_mismatch = True
        else:
            z_qaoa = []
            E_shared_qaoa = math.nan
            rank_qaoa = -1

        mask_exh = _bits_to_str(z_exh)
        mask_qaoa = _bits_to_str(z_qaoa) if z_qaoa else ""

        rows.append(
            {
                "mu": mu,
                "rank_exh": rank_exh,
                "rank_qaoa": rank_qaoa,
                "E_exh": E_exh,
                "E_shared_exh": E_shared_exh,
                "E_shared_qaoa": E_shared_qaoa,
                "mask_exh": mask_exh,
                "mask_qaoa": mask_qaoa,
            }
        )

        print(
            f"{mu:>6.3f} | {rank_exh:>8d} | {rank_qaoa:>9d} | "
            f"{_format_float(E_exh):>12} | {_format_float(E_shared_exh):>13} | "
            f"{_format_float(E_shared_qaoa):>15} | {mask_exh} | {mask_qaoa}"
        )

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mu",
                "rank_exh",
                "rank_qaoa",
                "E_exh",
                "E_shared_exh",
                "E_shared_qaoa",
                "mask_exh",
                "mask_qaoa",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if monotonic_fail:
        print(
            "FAIL: Exhaustive violates monotonicity on fixed instance -> "
            "mu term sign/convention likely wrong."
        )
    if energy_mismatch:
        print("FAIL: Exhaustive energy does not match shared objective.")
    if qaoa_energy_mismatch:
        print("WARNING: QAOA energy does not match shared objective.")

    if not monotonic_fail and not energy_mismatch:
        print("PASS: Exhaustive monotonicity holds and energy matches shared objective.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
