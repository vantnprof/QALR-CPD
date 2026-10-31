from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from src.utils.tensor import (
    mttkrp, hadamard_gram, cp_reconstruct, _EINSUM_LETTERS
)
from src.pruning.exhaustive import exhaustive_prune_cp
from src.pruning.bitflip import bit_flipping_prune_cp
from src.pruning.qaoa import qaoa_prune_cp
from src.pruning.threshold import threshold_prune_cp


class CPFactorization(nn.Module):
    """
    General CP/PARAFAC factorization module with CP-ALS fitting.

    - factors are stored as nn.Parameters (so you can also do gradient-based training if desired)
    - fit_als / als_step updates them in-place under torch.no_grad()
    - weights are stored as a buffer by default (set learn_weights=True to make them parameters)

    You can plug in a custom linear solver via `solve_fn(G, M)`:
      - G: (R,R) normal matrix
      - M: (I_mode, R) MTTKRP
      returns A_new: (I_mode, R)
    """
    def __init__(
        self,
        shape: Sequence[int],
        rank: int,
        ridge: float = 1e-8,
        init: Union[str, Sequence[torch.Tensor]] = "random",
        seed: int = 0,
        learn_weights: bool = False,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.shape = tuple(int(s) for s in shape)
        self.rank = int(rank)
        self.ridge = float(ridge)

        if len(self.shape) > len(_EINSUM_LETTERS):
            raise ValueError(f"Order {len(self.shape)} too large for this simple implementation.")

        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        custom_factors: Optional[Sequence[torch.Tensor]] = None
        init_mode: Optional[str] = None
        if isinstance(init, str):
            init_mode = init.lower()
        else:
            custom_factors = list(init)  # accepts list, tuple, ParameterList, etc.

        if custom_factors is not None and len(custom_factors) != len(self.shape):
            raise ValueError("Length of init factor list must match tensor order.")

        params = []
        for idx, I in enumerate(self.shape):
            if custom_factors is not None:
                A_init = custom_factors[idx]
                if A_init.shape != (I, self.rank):
                    raise ValueError(
                        f"Init factor shape {tuple(A_init.shape)} does not match required shape {(I, self.rank)}."
                    )
                A = A_init.detach().clone().to(device=device, dtype=dtype)
            elif init_mode == "random":
                A = torch.randn((I, self.rank), generator=g, dtype=dtype, device=device)
                A = A / (A.norm(dim=0, keepdim=True).clamp_min(1e-12))
            elif init_mode == "orthogonal":
                A0 = torch.randn((I, self.rank), generator=g, dtype=dtype, device=device)
                Q, _ = torch.linalg.qr(A0, mode="reduced")
                A = Q[:, : self.rank].contiguous()
            else:
                raise ValueError("init must be one of: random, orthogonal, or a sequence of factor matrices")
            params.append(nn.Parameter(A))

        self.factors = nn.ParameterList(params)

        if learn_weights:
            self.weights = nn.Parameter(torch.ones((self.rank,), dtype=dtype, device=device))
        else:
            self.register_buffer("weights", torch.ones((self.rank,), dtype=dtype, device=device))

    def _set_weights_tensor(self, new_weights: torch.Tensor) -> None:
        """
        Replace the weights parameter/buffer while preserving learnable/buffer semantics.
        """
        new_w = new_weights.detach()
        if isinstance(self.weights, nn.Parameter):
            self.weights = nn.Parameter(new_w, requires_grad=self.weights.requires_grad)
        else:
            self._buffers["weights"] = new_w
            self.weights = new_w

    def _apply_pruning_selection(self, lam: torch.Tensor, U: List[torch.Tensor], selected: List[int]) -> None:
        """
        Update factor matrices and weights to keep only the selected CP components.
        """
        if len(selected) == 0:
            raise ValueError("Cannot prune to an empty component set.")

        device = U[0].device
        dtype = U[0].dtype
        new_params = []
        for n in range(len(U)):
            cols = U[n][:, selected].detach().clone().to(device=device, dtype=dtype)
            new_params.append(nn.Parameter(cols))

        self.factors = nn.ParameterList(new_params)
        self.rank = len(selected)

        new_weights = lam[selected].detach().clone().to(device=device, dtype=dtype)
        self._set_weights_tensor(new_weights)

    def forward(self) -> torch.Tensor:
        return cp_reconstruct([A for A in self.factors], self.weights)

    @torch.no_grad()
    def als_step(
        self,
        X: torch.Tensor,
        solve_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        normalize: bool = True,
    ) -> None:
      
        if tuple(X.shape) != self.shape:
            raise ValueError(f"X shape {tuple(X.shape)} != module shape {self.shape}")

        N = len(self.shape)
        R = self.rank
        I_R = torch.eye(R, device=X.device, dtype=X.dtype)

        factors = [A.detach().clone() for A in self.factors]

        factors[0] = factors[0] * self.weights.view(1, -1)

        for mode in range(N):
            M = mttkrp(X, factors, mode)  # (I_mode, R)
            G = hadamard_gram(factors, mode) + self.ridge * I_R  # (R,R)

            A_new = torch.linalg.solve(G, M.T).T if solve_fn is None else solve_fn(G, M)

            factors[mode] = A_new
            self.factors[mode].copy_(A_new)

        # Normalize once per sweep (stable λ)
        if normalize:
            lam = torch.ones((R,), device=X.device, dtype=X.dtype)
            for n in range(N):
                col = self.factors[n].norm(dim=0).clamp_min(1e-12)
                self.factors[n].div_(col)
                lam.mul_(col)
            self.weights.copy_(lam)

    @torch.no_grad()
    def _maybe_prune_components(
        self,
        X: torch.Tensor,
        iteration: int,
        total_iters: int,
        prunning_enable: bool,
        prunning_mu: float,
        prunning_frequency: int,
        prunning_require_nonempty: bool,
        prunning_max_rank_exhaustive: int,
        verbose: bool,
        prunning_method: str,
        prunning_qaoa_reps: int,
        prunning_qaoa_shots: int,
        prunning_qaoa_maxiter: int,
        prunning_qaoa_seed: int,
        prunning_qaoa_restarts: int,
        prunning_bitflip_budget: Optional[int],
        prunning_bitflip_restarts: int,
        prunning_bitflip_init_mode: str,
        prunning_bitflip_tie_break: str,
        prunning_bitflip_tol: float,
        prunning_bitflip_seed: int,
        prunning_bitflip_max_rank: Optional[int],
        prunning_qubo_scale_exp: float = 0.0,
        prunning_rank_history: Optional[List[int]] = None,
        prunning_iter_history: Optional[List[int]] = None,
        prunning_log: bool = False,
    ) -> None:
        if (not prunning_enable) or prunning_mu is None:
            return
        if prunning_frequency is None or prunning_frequency <= 0:
            return
        if (iteration + 1) % prunning_frequency != 0:
            return
        if self.rank == 0:
            return

        method = prunning_method.lower().strip() if prunning_method is not None else "exhaustive"
        if method in ("topk", "top-k", "top_k") and iteration != total_iters - 1:
            return
        qubo_scale = 10.0 ** float(prunning_qubo_scale_exp)
        try:
            if method == "exhaustive":
                selected, _, (lam, U), diag = exhaustive_prune_cp(
                    X,
                    [A.detach() for A in self.factors],
                    mu=float(prunning_mu),
                    weights=self.weights,
                    require_nonempty=prunning_require_nonempty,
                    max_R_exhaustive=prunning_max_rank_exhaustive,
                    qubo_scale=qubo_scale,
                )
            elif method == "qaoa":
                selected, _, (lam, U), diag = qaoa_prune_cp(
                    X,
                    [A.detach() for A in self.factors],
                    mu=float(prunning_mu),
                    weights=self.weights,
                    require_nonempty=prunning_require_nonempty,
                    max_R_qaoa=prunning_max_rank_exhaustive,
                    qaoa_reps=prunning_qaoa_reps,
                    shots=prunning_qaoa_shots,
                    maxiter=prunning_qaoa_maxiter,
                    seed=prunning_qaoa_seed,
                    num_restarts=prunning_qaoa_restarts,
                    qubo_scale=qubo_scale,
                )
            elif method in ("bit", "bitflip", "bit-flipping", "bit_flipping", "bitflipping"):
                selected, _, (lam, U), diag = bit_flipping_prune_cp(
                    X,
                    [A.detach() for A in self.factors],
                    mu=float(prunning_mu),
                    weights=self.weights,
                    require_nonempty=prunning_require_nonempty,
                    max_R_bitflip=prunning_bitflip_max_rank,
                    budget=prunning_bitflip_budget,
                    restarts=prunning_bitflip_restarts,
                    init_mode=prunning_bitflip_init_mode,
                    tie_break=prunning_bitflip_tie_break,
                    tol=prunning_bitflip_tol,
                    seed=prunning_bitflip_seed,
                    qubo_scale=qubo_scale,
                )
            elif method in ("threshold", "weight", "weight-threshold", "weight_threshold"):
                selected, _, (lam, U), diag = threshold_prune_cp(
                    [A.detach() for A in self.factors],
                    weights=self.weights,
                    weight_threshold=float(prunning_mu),
                    top_k=None,
                    require_nonempty=prunning_require_nonempty,
                )
            elif method in ("topk", "top-k", "top_k"):
                selected, _, (lam, U), diag = threshold_prune_cp(
                    [A.detach() for A in self.factors],
                    weights=self.weights,
                    weight_threshold=None,
                    top_k=prunning_max_rank_exhaustive,
                    require_nonempty=prunning_require_nonempty,
                )
            else:
                raise ValueError(f"Unknown pruning method: {prunning_method}")
        except ValueError as exc:
            if verbose:
                print(f"prunning skipped at iter {iteration}: {exc}")
            return
        except ImportError as exc:
            if verbose:
                print(f"prunning skipped at iter {iteration} due to missing dependency: {exc}")
            return

        if len(selected) == 0:
            if prunning_iter_history is not None:
                prunning_iter_history.append(iteration)
            if prunning_rank_history is not None:
                prunning_rank_history.append(self.rank)
            if verbose:
                print(f"prunning skipped at iter {iteration}: empty selection.")
            return
        old_rank = self.rank
        if prunning_log:
            print(
                f"prunning call at iter {iteration}: "
                f"method={method} selected={len(selected)}/{old_rank} "
                f"mu={float(prunning_mu):.6g}"
            )
        if len(selected) < old_rank:
            self._apply_pruning_selection(lam, U, selected)
            if verbose:
                print(f"prunning applied at iter {iteration}: rank {old_rank}->{self.rank}, mask={diag.get('best_mask')}")
        else:
            # Keep rank but refresh weights to the normalized lam (no factor columns dropped).
            self._set_weights_tensor(lam)
        if prunning_iter_history is not None:
            prunning_iter_history.append(iteration)
        if prunning_rank_history is not None:
            prunning_rank_history.append(self.rank)

    @torch.no_grad()
    def fit_als(
        self,
        X: torch.Tensor,
        n_iter: int = 50,
        tol: float = 1e-7,
        solve_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        track_error: bool = True,
        verbose: bool = False,
        prunning_enable: bool = False,
        prunning_mu: float = 0.0,
        prunning_frequency: int = 1,
        prunning_require_nonempty: bool = True,
        prunning_max_rank_exhaustive: int = 20,
        prunning_method: str = "exhaustive",
        prunning_qaoa_reps: int = 2,
        prunning_qaoa_shots: int = 2048,
        prunning_qaoa_maxiter: int = 200,
        prunning_qaoa_seed: int = 0,
        prunning_qaoa_restarts: int = 1,
        prunning_bitflip_budget: Optional[int] = 200,
        prunning_bitflip_restarts: int = 1,
        prunning_bitflip_init_mode: str = "zeros",
        prunning_bitflip_tie_break: str = "first",
        prunning_bitflip_tol: float = 1e-12,
        prunning_bitflip_seed: int = 0,
        prunning_bitflip_max_rank: Optional[int] = None,
        prunning_rank_history: Optional[List[int]] = None,
        prunning_iter_history: Optional[List[int]] = None,
        prunning_log: bool = False,
    ) -> List[float]:

        normX = torch.linalg.norm(X).clamp_min(1e-12)
        errs: List[float] = []
        prev = None
        
        for it in range(int(n_iter)):
            self.als_step(X, solve_fn=solve_fn, normalize=True)
            
            if it < int(n_iter) - 1:
                self._maybe_prune_components(
                    X=X,
                    iteration=it,
                    total_iters=int(n_iter),
                    prunning_enable=prunning_enable,
                    prunning_mu=prunning_mu,
                    prunning_frequency=prunning_frequency,
                    prunning_require_nonempty=prunning_require_nonempty,
                    prunning_max_rank_exhaustive=prunning_max_rank_exhaustive,
                    verbose=verbose,
                    prunning_method=prunning_method,
                    prunning_qaoa_reps=prunning_qaoa_reps,
                    prunning_qaoa_shots=prunning_qaoa_shots,
                    prunning_qaoa_maxiter=prunning_qaoa_maxiter,
                    prunning_qaoa_seed=prunning_qaoa_seed,
                    prunning_qaoa_restarts=prunning_qaoa_restarts,
                    prunning_bitflip_budget=prunning_bitflip_budget,
                    prunning_bitflip_restarts=prunning_bitflip_restarts,
                    prunning_bitflip_init_mode=prunning_bitflip_init_mode,
                    prunning_bitflip_tie_break=prunning_bitflip_tie_break,
                    prunning_bitflip_tol=prunning_bitflip_tol,
                    prunning_bitflip_seed=prunning_bitflip_seed,
                    prunning_bitflip_max_rank=prunning_bitflip_max_rank,
                    prunning_rank_history=prunning_rank_history,
                    prunning_iter_history=prunning_iter_history,
                    prunning_log=prunning_log,
                )
            else:
                method = prunning_method.lower().strip() if prunning_method is not None else "exhaustive"
                if prunning_enable and method in ("topk", "top-k", "top_k"):
                    self._maybe_prune_components(
                        X=X,
                        iteration=it,
                        total_iters=int(n_iter),
                        prunning_enable=prunning_enable,
                        prunning_mu=prunning_mu,
                        prunning_frequency=prunning_frequency,
                        prunning_require_nonempty=prunning_require_nonempty,
                        prunning_max_rank_exhaustive=prunning_max_rank_exhaustive,
                        verbose=verbose,
                        prunning_method=prunning_method,
                        prunning_qaoa_reps=prunning_qaoa_reps,
                        prunning_qaoa_shots=prunning_qaoa_shots,
                        prunning_qaoa_maxiter=prunning_qaoa_maxiter,
                        prunning_qaoa_seed=prunning_qaoa_seed,
                        prunning_qaoa_restarts=prunning_qaoa_restarts,
                        prunning_bitflip_budget=prunning_bitflip_budget,
                        prunning_bitflip_restarts=prunning_bitflip_restarts,
                        prunning_bitflip_init_mode=prunning_bitflip_init_mode,
                        prunning_bitflip_tie_break=prunning_bitflip_tie_break,
                        prunning_bitflip_tol=prunning_bitflip_tol,
                        prunning_bitflip_seed=prunning_bitflip_seed,
                        prunning_bitflip_max_rank=prunning_bitflip_max_rank,
                        prunning_rank_history=prunning_rank_history,
                        prunning_iter_history=prunning_iter_history,
                        prunning_log=prunning_log,
                    )
                else:
                    # No pruning on last iteration
                    print("Final iteration reached; skipping pruning step.")

            if track_error:
                Xhat = self.forward()
                err = (torch.linalg.norm(X - Xhat) / normX).item()
                errs.append(err)
                if verbose:
                    print(f"iter {it:03d} | relerr={err:.6e}")

                if prev is not None and abs(prev - err) < tol:
                    break
                prev = err

        return errs


if __name__ == "__main__":
    torch.manual_seed(0)

    shape = (20, 20, 20, 20)
    R_true = 10
    noise_std = 0.1

    true_factors = [torch.rand(s, R_true) for s in shape]
    X_clean = torch.einsum("ir,jr,kr,lr->ijkl", *true_factors)
    X = X_clean + noise_std * torch.randn(*shape)

    prun_mu = 100
    prun_freq = 40
    n_iter = 200  # shorten so we can run all baselines
    top_k = R_true

    def run_fit(method: str, prunning_enable: bool = True, **kwargs):
        model = CPFactorization(
            shape,
            rank=20,
            ridge=1e-4,
            init="random",
            seed=123,
            dtype=X.dtype,
            device=X.device,
        )
        prune_ranks: List[int] = []
        prune_iters: List[int] = []
        errs = model.fit_als(
            X,
            n_iter=n_iter,
            tol=1e-12,
            prunning_enable=prunning_enable,
            prunning_mu=prun_mu,
            prunning_frequency=prun_freq,
            prunning_method=method,
            verbose=False,
            prunning_log=prunning_enable,
            prunning_rank_history=prune_ranks,
            prunning_iter_history=prune_iters,
            **kwargs,
        )
        factors_copy = [A.detach().clone() for A in model.factors]
        weights_copy = model.weights.detach().clone()
        return errs, model.rank, factors_copy, weights_copy, prune_ranks, prune_iters

    results: List[Tuple[str, List[float], int, List[torch.Tensor], torch.Tensor, List[int], List[int]]] = []

    print("=== ALS (no pruning) ===")
    errs_als, rank_als, factors_als, weights_als, prune_ranks_als, prune_iters_als = run_fit(
        "exhaustive",
        prunning_enable=False,
    )
    print(f"final relerr: {errs_als[-1]:.6e} | final rank: {rank_als}")
    results.append(("ALS", errs_als, rank_als, factors_als, weights_als, prune_ranks_als, prune_iters_als))

    print("\n=== ALS-ES (exhaustive pruning) ===")
    try:
        errs_exh, rank_exh, factors_exh, weights_exh, prune_ranks_exh, prune_iters_exh = run_fit(
            "exhaustive",
            prunning_max_rank_exhaustive=24,
        )
        print(f"final relerr: {errs_exh[-1]:.6e} | final rank: {rank_exh}")
        results.append(("ALS-ES", errs_exh, rank_exh, factors_exh, weights_exh, prune_ranks_exh, prune_iters_exh))
    except Exception as exc:
        print(f"ALS-ES pruning failed: {exc}")

    print("\n=== ALS-BF (bit-flipping pruning) ===")
    try:
        errs_bf, rank_bf, factors_bf, weights_bf, prune_ranks_bf, prune_iters_bf = run_fit(
            "bit_flipping",
            prunning_bitflip_budget=200,
            prunning_bitflip_restarts=5,
            prunning_bitflip_init_mode="random",
            prunning_bitflip_tie_break="random_among_best",
            prunning_bitflip_tol=1e-12,
            prunning_bitflip_seed=0,
        )
        print(f"final relerr: {errs_bf[-1]:.6e} | final rank: {rank_bf}")
        results.append(("ALS-BF", errs_bf, rank_bf, factors_bf, weights_bf, prune_ranks_bf, prune_iters_bf))
    except Exception as exc:
        print(f"ALS-BF pruning failed: {exc}")

    print("\n=== QALR-CPD (QAOA pruning) ===")
    try:
        errs_qaoa, rank_qaoa, factors_qaoa, weights_qaoa, prune_ranks_qaoa, prune_iters_qaoa = run_fit(
            "qaoa",
            prunning_max_rank_exhaustive=24,
            prunning_qaoa_reps=2,
            prunning_qaoa_shots=2048,
            prunning_qaoa_maxiter=200,
            prunning_qaoa_seed=0,
            prunning_qaoa_restarts=1,
        )
        print(f"final relerr: {errs_qaoa[-1]:.6e} | final rank: {rank_qaoa}")
        results.append(("QALR-CPD", errs_qaoa, rank_qaoa, factors_qaoa, weights_qaoa, prune_ranks_qaoa, prune_iters_qaoa))
    except ImportError as exc:
        print(f"QALR-CPD pruning skipped (missing dependency): {exc}")
    except Exception as exc:
        print(f"QALR-CPD pruning failed: {exc}")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    for label, errs, rank, _, _, _, _ in results:
        iterations = range(1, len(errs) + 1)
        plt.plot(iterations, errs, label=f"{label} (final R={rank})")
    plt.xlabel("ALS iteration")
    plt.ylabel("Relative error")
    plt.title("CP-ALS relative error per pruning method")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("cp_als_prunning_comparison.png", dpi=1000)
    print("Saved figure to cp_als_prunning_comparison.png")

    # print out the performance comparison
    for label, errs, rank, factors, weights, _, _ in results:
        print(f"{label}: final relerr={errs[-1]:.6e}, final rank={rank}")

    # Plot rank per pruning call
    plt.figure(figsize=(7, 4))
    has_rank_plot = False
    for label, _, _, _, _, prune_ranks, _ in results:
        if not prune_ranks:
            continue
        has_rank_plot = True
        calls = range(1, len(prune_ranks) + 1)
        plt.plot(calls, prune_ranks, marker="o", label=label)
    if has_rank_plot:
        plt.xlabel("Pruning call")
        plt.ylabel("Selected rank")
        plt.title("Rank per pruning call")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig("cp_als_prunning_rank.png", dpi=1000)
        print("Saved figure to cp_als_prunning_rank.png")
    else:
        print("No pruning calls recorded; skipping rank plot.")

    # # for each founded rank, factor of each method, please use found rank and factors as initial point
    # # to run ALS without pruning to see which method gives better final error
    # print("\n=== Final ALS refinement without pruning ===")
    # for label, errs, rank, f, w in results:
    #     print(f"\n--- Method: {label} | Starting rank: {rank} ---")
    #     model = CPFactorization(
    #         shape,
    #         rank=rank,
    #         ridge=1e-6,
    #         init=f,
    #         seed=123,
    #         dtype=X.dtype,
    #         device=X.device,
    #     )
    #     model.weights.copy_(w)
    #     errs_refine = model.fit_als(
    #         X,
    #         n_iter=n_iter,
    #         tol=1e-12,
    #         prunning_enable=False,
    #         track_error=True,
    #         verbose=True,
    #     )
    #     print(f"Refined final relerr: {errs_refine[-1]:.6e}")
