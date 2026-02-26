"""
QAOA-based rank pruning for CP decomposition.

Pattern closely follows the IBM Quantum QAOA tutorial:
  1) Build an Ising cost operator from the QUBO objective.
  2) Use a QAOAAnsatz + Estimator in a classical optimization loop (COBYLA).
  3) Sample the optimized ansatz with a Sampler to pick the best bitstring.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple, Optional, Dict
import math
import numpy as np
import torch

from src.rank_prunning.exhaustive_search_rank_prunning import (
    normalize_cp_factors,
    inner_product_rank1,
)

from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_ibm_runtime import (
    Session,
    # QiskitRuntimeService,
    EstimatorV2 as Estimator,
    SamplerV2 as Sampler,
)
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator


def _energy_value(mask: List[int], G: torch.Tensor, c: torch.Tensor, mu: float) -> float:
    z = torch.tensor(mask, device=G.device, dtype=G.dtype)
    E = (z @ (G @ z)) - 2.0 * (z @ c) + mu * z.sum()
    return float(E.detach().cpu())


def _add_term(pauli_dict: Dict[str, complex], label: str, coeff: float) -> None:
    if abs(coeff) < 1e-12:
        return
    pauli_dict[label] = pauli_dict.get(label, 0.0) + coeff


def _build_cost_operator(Q: np.ndarray, lin: np.ndarray):
    """
    Map the QUBO E = x^T Q x + lin^T x into an Ising Hamiltonian over Z Paulis.
    Returns SparsePauliOp with any constant shift embedded (adds identity term).
    """

    R = Q.shape[0]

    # Fold diagonal into linear part because x_i^2 = x_i
    lin_eff = lin.copy()
    lin_eff += np.diag(Q)

    pauli_terms: Dict[str, complex] = {}
    I_str = "I" * R
    const = 0.0

    # Linear terms: lin_eff_i * x_i = lin_eff_i/2 * (1 - Z_i)
    for i in range(R):
        const += 0.5 * lin_eff[i]
        z_label = list(I_str)
        z_label[R - 1 - i] = "Z"  # Qiskit uses little-endian; flip index
        _add_term(pauli_terms, "".join(z_label), -0.5 * lin_eff[i])

    # Quadratic off-diagonal terms: Q_ij x_i x_j = Q_ij/4 (1 - Z_i - Z_j + Z_i Z_j)
    for i in range(R):
        for j in range(i + 1, R):
            q = Q[i, j]
            if abs(q) < 1e-12:
                continue
            const += q / 4.0

            zi = list(I_str)
            zi[R - 1 - i] = "Z"
            zj = list(I_str)
            zj[R - 1 - j] = "Z"
            zij = list(I_str)
            zij[R - 1 - i] = "Z"
            zij[R - 1 - j] = "Z"

            _add_term(pauli_terms, "".join(zi), -q / 4.0)
            _add_term(pauli_terms, "".join(zj), -q / 4.0)
            _add_term(pauli_terms, "".join(zij), q / 4.0)

    # _add_term(pauli_terms, I_str, const)  # constant shift

    labels = list(pauli_terms.keys())
    coeffs = list(pauli_terms.values())
    return SparsePauliOp(labels, coeffs)


@torch.no_grad()
def qaoa_prune_cp(
    X: torch.Tensor,
    factors: Sequence[torch.Tensor],
    mu: float,
    weights: Optional[torch.Tensor] = None,
    require_nonempty: bool = True,
    max_R_qaoa: int = 24,
    qaoa_reps: int = 2,
    shots: int = 2048,
    maxiter: int = 200,
    seed: int = 0,
    num_restarts: int = 1,
    top_k: Optional[int] = None,
    qubo_scale: float = 1.0,
) -> Tuple[List[int], float, Tuple[torch.Tensor, List[torch.Tensor]], dict]:
    """
    QAOA approximation of:
        min_{z in {0,1}^R} || X - sum_r z_r * lambda_r * C_r ||_F^2 + mu * sum_r z_r

    Returns: (selected_indices, best_energy, (lambda, U_factors), diagnostics)
    """

    lam, U = normalize_cp_factors(factors, weights=weights)
    R = U[0].shape[1]
    if top_k is None:
        top_k = R

    if R > max_R_qaoa:
        raise ValueError(
            f"R={R} too large for QAOA search cap {max_R_qaoa}. "
            f"Increase max_R_qaoa or reduce rank."
        )

    device = X.device
    dtype = X.dtype
    lam = lam.to(device=device, dtype=dtype)
    U = [u.to(device=device, dtype=dtype) for u in U]

    overlaps = torch.ones((R, R), device=device, dtype=dtype)
    for n in range(len(U)):
        Sn = U[n].T @ U[n]
        overlaps = overlaps * Sn

    G_t = (lam[:, None] * lam[None, :]) * overlaps

    c_t = torch.zeros((R,), device=device, dtype=dtype)
    for r in range(R):
        u_cols = [U[n][:, r] for n in range(len(U))]
        c_t[r] = lam[r] * inner_product_rank1(X, u_cols)

    G_np = G_t.detach().cpu().double().numpy()
    c_np = c_t.detach().cpu().double().numpy()
    Q = np.triu(G_np, 1) * 2.0
    np.fill_diagonal(Q, np.diag(G_np))
    lin = mu * np.ones(R) - 2.0 * c_np
    scale_ref = max(np.max(np.abs(Q)), np.max(np.abs(lin)), 1.0)
    Q_s = Q / scale_ref
    lin_s = lin / scale_ref
    scale_factor = float(qubo_scale)
    if not (scale_factor > 0.0):
        raise ValueError("qubo_scale must be a positive, finite number.")
    if scale_factor != 1.0:
        Q_s = Q_s * scale_factor
        lin_s = lin_s * scale_factor

    cost_hamiltonian = _build_cost_operator(Q=Q_s, lin=lin_s)
 
    best_selected: List[int] = []
    best_val = math.inf
    best_mask_bits: Optional[List[int]] = None

    for _ in range(max(1, int(num_restarts))):

        circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=int(qaoa_reps))
        # circuit.measure_all()

        backend = AerSimulator(shots=shots)
        backend.seed_simulator = seed
        pm = generate_preset_pass_manager(optimization_level=0, backend=backend)

        candidate_circuit = pm.run(circuit)

        init_beta = math.pi / 2
        init_gamma = math.pi
        init_params = [init_beta] * qaoa_reps + [init_gamma] * qaoa_reps

        def cost_func_estimator(params, ansatz, hamiltonian, estimator):
    
            isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
        
            pub = (ansatz, isa_hamiltonian, params)
            job = estimator.run([pub])
        
            results = job.result()[0]
            cost = results.data.evs
        
            objective_func_vals.append(cost)
        
            return cost
        
        objective_func_vals = []
        with Session(backend=backend) as session:
            estimator = Estimator(mode=session)
            estimator.options.default_shots = shots
            estimator.seed_estimator = seed
       
            result = minimize(
                cost_func_estimator,
                init_params,
                args=(candidate_circuit, cost_hamiltonian, estimator),
                method="COBYLA",
                tol=1e-6,
                options={"maxiter": maxiter},
            )
        opt_params = result.x
        
        optimized_circuit = candidate_circuit.assign_parameters(opt_params)
        optimized_circuit.measure_all()

        sampler = Sampler(mode=backend)
        sampler.options.default_shots = shots
        sampler.seed_estimator = seed
    
        pub = (optimized_circuit,)
        job = sampler.run([pub], shots=shots)
        counts_int = job.result()[0].data.meas.get_int_counts()
        shots = sum(counts_int.values())
        final_distribution_int = {key: val / shots for key, val in counts_int.items()}

        # auxiliary functions to sample most likely bitstring
        def to_bitstring(integer, num_bits):
            result = np.binary_repr(integer, width=num_bits)
            return [int(digit) for digit in result]
        
        keys = list(final_distribution_int.keys())
        values = list(final_distribution_int.values())

        top_k_indices = np.argsort(values)[-top_k:]
        for idx in top_k_indices:
            most_likely = keys[idx]
            sol_bits = to_bitstring(most_likely, R)
            sol_bits.reverse()
            
            E_val = _energy_value(sol_bits, G_t, c_t, mu)
            
            if E_val < best_val:
                best_val = E_val
                best_mask_bits = sol_bits
                best_selected = [i for i, b in enumerate(sol_bits) if b == 1]

    diagnostics = {
        "R": R,
        "best_mask": best_mask_bits,
        "best_E_no_const": best_val,
        "qaoa_restarts": max(1, int(num_restarts)),
        "qubo_scale": float(scale_factor),
        "qubo_scale_ref": float(scale_ref),
    }
    return best_selected, best_val, (lam, U), diagnostics


if __name__ == "__main__":
    from src.rank_prunning.exhaustive_search_rank_prunning import exhaustive_prune_cp

    torch.manual_seed(42)
    # Toy example data
    shape = (8, 9, 10, 12)
    R = 8

    factors = [torch.rand(s, R, dtype=torch.float64) for s in shape]

    weak_r = 1
    for A in factors:
        A[:, weak_r] *= 0.15
    
    # CP weights (same length as R)
    weights = torch.ones(R, dtype=torch.float64)
    weights[weak_r] = 0.05

    X = torch.einsum("r,ir,jr,kr,lr->ijkl", weights, *factors)

    noise_std = 0.0
    if noise_std > 0:
        X = X + noise_std * torch.randn_like(X)

    mu = 0.05

    ref_mask, ref_E, _, _ = exhaustive_prune_cp(
        X,
        factors,
        mu,
        weights=weights,
        require_nonempty=True,
        max_R_exhaustive=24,
    )

    mask, E, _, diag = qaoa_prune_cp(
        X,
        factors,
        mu,
        weights=weights,
        require_nonempty=True,
        max_R_qaoa=24,
        qaoa_reps=3,
        shots=1024,
        maxiter=200,
        top_k=R,
        seed=42,
        num_restarts=1,
    )

    print("=== QAOA rank pruning sanity check ===")
    print(f"R={factors[0].shape[1]} | mu={mu}")
    print(f"Exhaustive best mask: {ref_mask} | E={ref_E:.6f}")
    print(f"QAOA best mask      : {mask} | E={E:.6f}")
    print("Diagnostics:", diag)

    if mask != ref_mask:
        print("WARNING: QAOA did not match exhaustive optimum on this toy instance.")
