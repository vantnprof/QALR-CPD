#!/usr/bin/env python3
"""
Run lightweight synthetic poster experiments:
1) fit overfactored CP-ALS once per dataset,
2) freeze factors and build a soft-rank QUBO,
3) solve with ES / BF / QAOA,
4) aggregate metrics and convergence traces.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "pandas is required to run this script. Please install pandas and retry."
    ) from exc

from scipy.optimize import minimize

from src.classical_alternative_leastsquare.classical_als import CPFactorization
from src.utils.tensor_operator_util import _EINSUM_LETTERS


def _parse_list_int(raw: str) -> List[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _parse_list_float(raw: str) -> List[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def _dtype_from_str(raw: str) -> torch.dtype:
    key = raw.strip().lower()
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{raw}'.")
    return mapping[key]


def _resolve_dataset_path(path_str: str, manifest_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return manifest_dir / path


def _format_mu(mu: float) -> str:
    return f"{float(mu):.6g}"


def _is_close(a: float, b: float, atol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= atol


@dataclass
class DatasetRecord:
    path: Path
    seed: int
    shape: Tuple[int, ...]
    rank_true: int
    noise_std: float
    corr_rho: float
    dtype: str


@dataclass
class RecordState:
    record: DatasetRecord
    X: torch.Tensor
    X_norm: float
    weighted_components: List[torch.Tensor]
    Q: np.ndarray
    q_base: np.ndarray
    als_rel_error: float
    als_time_s: float


def load_manifest(path: Path) -> List[DatasetRecord]:
    data = json.loads(path.read_text())
    records: List[DatasetRecord] = []
    for entry in data:
        resolved = _resolve_dataset_path(entry["path"], path.parent)
        records.append(
            DatasetRecord(
                path=resolved,
                seed=int(entry.get("seed", 0)),
                shape=tuple(entry["shape"]),
                rank_true=int(entry["rank_true"]),
                noise_std=float(entry["noise_std"]),
                corr_rho=float(entry["corr_rho"]),
                dtype=str(entry.get("dtype", "float32")),
            )
        )
    return records


def _rank1_tensor(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    order = len(vectors)
    if order == 0:
        raise ValueError("Need at least one mode vector.")
    if order > len(_EINSUM_LETTERS):
        raise ValueError(f"Tensor order {order} exceeds einsum letter budget.")
    inds = _EINSUM_LETTERS[:order]
    equation = ",".join(inds) + "->" + "".join(inds)
    return torch.einsum(equation, *vectors)


def _absorb_weights_into_factors(
    factors: Sequence[torch.Tensor],
    weights: torch.Tensor,
) -> List[torch.Tensor]:
    if len(factors) == 0:
        raise ValueError("factors cannot be empty.")
    out = [A.detach().clone() for A in factors]
    out[0] = out[0] * weights.detach().view(1, -1)
    return out


def run_overfactored_als(
    X: torch.Tensor,
    rank_over: int,
    sweeps: int,
    ridge: float,
    seed: int,
    log_every: int,
    verbose: bool,
    run_label: str,
) -> Tuple[List[torch.Tensor], torch.Tensor, float, float]:
    torch.manual_seed(int(seed))
    model = CPFactorization(
        shape=tuple(X.shape),
        rank=int(rank_over),
        ridge=float(ridge),
        init="random",
        seed=int(seed),
        device=X.device,
        dtype=X.dtype,
    )

    xnorm = torch.linalg.norm(X).clamp_min(1e-12)
    t0 = time.time()
    for sweep in range(int(sweeps)):
        model.als_step(X, normalize=True)
        if verbose and (log_every <= 1 or (sweep + 1) % int(log_every) == 0):
            err = (torch.linalg.norm(X - model.forward()) / xnorm).item()
            print(
                f"[als] {run_label} sweep={sweep + 1:03d} "
                f"rel_error={err:.6e} rank={model.rank}"
            )
    elapsed = time.time() - t0
    final_rel_error = (torch.linalg.norm(X - model.forward()) / xnorm).item()
    factors = [A.detach().clone() for A in model.factors]
    weights = model.weights.detach().clone()
    return factors, weights, float(final_rel_error), float(elapsed)


def build_qubo_from_factors(
    X: torch.Tensor,
    factors_with_weights: Sequence[torch.Tensor],
) -> Tuple[List[torch.Tensor], np.ndarray, np.ndarray]:
    if len(factors_with_weights) == 0:
        raise ValueError("factors_with_weights cannot be empty.")

    rank = factors_with_weights[0].shape[1]
    device = X.device
    dtype = X.dtype
    order = len(factors_with_weights)

    lamb = torch.ones((rank,), device=device, dtype=dtype)
    normalized: List[torch.Tensor] = []
    for A in factors_with_weights:
        if A.shape[1] != rank:
            raise ValueError("All factors must share the same rank.")
        col_norms = torch.linalg.norm(A, dim=0).clamp_min(1e-12)
        normalized.append(A / col_norms)
        lamb = lamb * col_norms

    gram_prod = torch.ones((rank, rank), device=device, dtype=dtype)
    for U in normalized:
        gram_prod = gram_prod * (U.T @ U)

    Q_t = (lamb[:, None] * lamb[None, :]) * gram_prod
    Q_t.fill_diagonal_(0.0)

    c = torch.zeros((rank,), device=device, dtype=dtype)
    weighted_components: List[torch.Tensor] = []
    for r in range(rank):
        cols = [normalized[n][:, r] for n in range(order)]
        cbar_r = _rank1_tensor(cols)
        c[r] = torch.sum(X * cbar_r)
        weighted_components.append(lamb[r] * cbar_r)

    q_base_t = (lamb**2) - (2.0 * lamb * c)
    Q = Q_t.detach().cpu().double().numpy()
    q_base = q_base_t.detach().cpu().double().numpy()
    return weighted_components, Q, q_base


def qubo_energy(mask: np.ndarray, Q: np.ndarray, q: np.ndarray) -> float:
    z = np.asarray(mask, dtype=np.float64)
    return float(z @ (Q @ z) + q @ z)


def exhaustive_search_qubo(Q: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, float]:
    rank = int(q.shape[0])
    best_energy = float("inf")
    best_mask = np.zeros((rank,), dtype=np.int8)

    for mask_int in range(1 << rank):
        z = np.array([(mask_int >> i) & 1 for i in range(rank)], dtype=np.float64)
        E = float(z @ (Q @ z) + q @ z)
        if E < best_energy:
            best_energy = E
            best_mask = z.astype(np.int8)

    return best_mask, best_energy


def bitflip_search_qubo(
    Q: np.ndarray,
    q: np.ndarray,
    max_evals: int,
    restarts: int,
    seed: int,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    # Adopt the same local-search dynamics as src/rank_prunning/bit_flipping_rank_prunning.py,
    # but run directly on a shared canonical QUBO (Q, q) so ES/BF/QAOA use identical inputs.
    Q_mat = np.asarray(Q, dtype=np.float64)
    q_vec = np.asarray(q, dtype=np.float64).reshape(-1)
    rank = int(q_vec.shape[0])
    rng = np.random.default_rng(int(seed))
    max_evals = int(max_evals)
    restarts = max(1, int(restarts))
    tol = 1e-12
    tie_break_mode = "first"
    init_mode = "zeros"
    require_nonempty = False

    if max_evals <= 0:
        z0 = np.zeros((rank,), dtype=np.int8)
        return (
            z0,
            qubo_energy(z0, Q_mat, q_vec),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )

    diag_G = np.diag(Q_mat)
    best_energy = float("inf")
    best_mask = np.zeros((rank,), dtype=np.int8)
    trace_best: List[float] = []
    trace_loss: List[float] = []
    eval_count = 0

    best_single_idx: Optional[int] = None
    if require_nonempty and rank > 0:
        best_single_idx = int(np.argmin(diag_G + q_vec))

    def _init_binary_vector_np() -> np.ndarray:
        if init_mode in ("zeros", "zero"):
            return np.zeros((rank,), dtype=np.float64)
        if init_mode in ("random", "rand"):
            return rng.integers(0, 2, size=rank, endpoint=False).astype(np.float64)
        raise ValueError(f"Unsupported BF init_mode '{init_mode}'.")

    for _ in range(restarts):
        if eval_count >= max_evals:
            break

        z = _init_binary_vector_np()
        if require_nonempty and rank > 0 and z.sum() < 0.5:
            z[best_single_idx] = 1.0

        g = Q_mat @ z
        E = float(z @ g + q_vec @ z)
        eval_count += 1
        trace_loss.append(E)
        if E < best_energy:
            best_energy = E
            best_mask = z.astype(np.int8).copy()
        trace_best.append(best_energy)

        while eval_count < max_evals:
            s = 1.0 - 2.0 * z
            delta = s * (2.0 * g + q_vec) + diag_G

            if require_nonempty and rank > 0 and z.sum() <= 1.0:
                one_idx = np.flatnonzero(z > 0.5)
                if one_idx.size == 1:
                    delta[int(one_idx[0])] = np.inf

            min_delta = float(np.min(delta))
            if not np.isfinite(min_delta) or min_delta >= -tol:
                break

            if tie_break_mode in ("random", "random_among_best", "random-among-best"):
                candidates = np.flatnonzero(delta <= min_delta + 1e-12)
                i_star = int(rng.choice(candidates))
            else:
                i_star = int(np.argmin(delta))

            s_i = 1.0 - 2.0 * z[i_star]
            z[i_star] = 1.0 - z[i_star]
            E = float(E + delta[i_star])
            g = g + s_i * Q_mat[:, i_star]

            eval_count += 1
            trace_loss.append(E)
            if E < best_energy:
                best_energy = E
                best_mask = z.astype(np.int8).copy()
            trace_best.append(best_energy)

    return (
        best_mask,
        float(best_energy),
        np.asarray(trace_best, dtype=np.float64),
        np.asarray(trace_loss, dtype=np.float64),
    )


def _add_pauli_term(pauli_dict: Dict[str, complex], label: str, coeff: float) -> None:
    if abs(float(coeff)) < 1e-12:
        return
    pauli_dict[label] = pauli_dict.get(label, 0.0) + float(coeff)


def _build_cost_operator_rank_pruning_style(Q: np.ndarray, lin: np.ndarray):
    from qiskit.quantum_info import SparsePauliOp

    # Matches src/rank_prunning/qaoa_rank_prunning.py implementation style.
    rank = int(Q.shape[0])
    lin_eff = np.asarray(lin, dtype=np.float64).copy()
    lin_eff += np.diag(np.asarray(Q, dtype=np.float64))

    I_str = "I" * rank
    pauli_terms: Dict[str, complex] = {}
    const = 0.0

    for i in range(rank):
        const += 0.5 * lin_eff[i]
        z_label = list(I_str)
        z_label[rank - 1 - i] = "Z"
        _add_pauli_term(pauli_terms, "".join(z_label), -0.5 * float(lin_eff[i]))

    for i in range(rank):
        for j in range(i + 1, rank):
            q_ij = float(Q[i, j])
            if abs(q_ij) < 1e-12:
                continue
            const += q_ij / 4.0

            zi = list(I_str)
            zj = list(I_str)
            zij = list(I_str)
            zi[rank - 1 - i] = "Z"
            zj[rank - 1 - j] = "Z"
            zij[rank - 1 - i] = "Z"
            zij[rank - 1 - j] = "Z"

            _add_pauli_term(pauli_terms, "".join(zi), -q_ij / 4.0)
            _add_pauli_term(pauli_terms, "".join(zj), -q_ij / 4.0)
            _add_pauli_term(pauli_terms, "".join(zij), q_ij / 4.0)

    # Keep the constant energy shift so estimator outputs match the full QUBO scale.
    _add_pauli_term(pauli_terms, I_str, const)

    if not pauli_terms:
        return SparsePauliOp([I_str], [0.0])

    labels = list(pauli_terms.keys())
    coeffs = list(pauli_terms.values())
    return SparsePauliOp(labels, coeffs)


def build_qaoa_cost_operator(
    Q: np.ndarray,
    q: np.ndarray,
) -> Tuple[object, Dict[str, float]]:
    """
    Convert canonical QUBO E(z)=z^TQz+q^Tz into the same upper-triangular/scaled
    form used in src/rank_prunning/qaoa_rank_prunning.py before building Ising.
    """
    Q_np = np.asarray(Q, dtype=np.float64)
    q_np = np.asarray(q, dtype=np.float64).reshape(-1)
    if Q_np.shape[0] != q_np.shape[0]:
        raise ValueError(f"Q/q mismatch: Q={Q_np.shape}, q={q_np.shape}.")

    # rank_prunning-style QUBO matrix: upper-triangular with doubled off-diagonal.
    Q_upper = np.triu(Q_np, 1) * 2.0
    np.fill_diagonal(Q_upper, np.diag(Q_np))
    lin = q_np.copy()

    q_max = float(np.max(np.abs(Q_upper))) if Q_upper.size > 0 else 0.0
    lin_max = float(np.max(np.abs(lin))) if lin.size > 0 else 0.0
    scale_ref = max(q_max, lin_max, 1.0)
    Q_scaled = Q_upper / scale_ref
    lin_scaled = lin / scale_ref

    op = _build_cost_operator_rank_pruning_style(Q=Q_scaled, lin=lin_scaled)
    return op, {"qubo_scale_ref": float(scale_ref)}


def _int_to_binary_vector(value: int, width: int) -> np.ndarray:
    bits = [int(c) for c in np.binary_repr(int(value), width=width)]
    bits.reverse()
    return np.asarray(bits, dtype=np.int8)


def qaoa_search_qubo(
    Q: np.ndarray,
    q: np.ndarray,
    qaoa_p: int,
    shots: int,
    maxiter: int,
    seed: int,
    optimizer: str,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    try:
        from qiskit.circuit.library import QAOAAnsatz
        from qiskit.transpiler import generate_preset_pass_manager
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime import (
            Session,
            EstimatorV2 as Estimator,
            SamplerV2 as Sampler,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "QAOA dependencies are missing. Install qiskit, qiskit-aer, and qiskit-ibm-runtime."
        ) from exc

    rank = int(q.shape[0])
    maxiter = max(1, int(maxiter))
    shots = max(1, int(shots))
    qaoa_p = max(1, int(qaoa_p))

    if rank == 0:
        z0 = np.zeros((0,), dtype=np.int8)
        return z0, 0.0, np.zeros((maxiter,), dtype=np.float64), np.zeros((maxiter,), dtype=np.float64)

    cost_hamiltonian, qubo_meta = build_qaoa_cost_operator(Q=Q, q=q)
    scale_ref = float(qubo_meta.get("qubo_scale_ref", 1.0))
    if not (scale_ref > 0.0):
        scale_ref = 1.0
    top_k = max(1, rank)

    backend = AerSimulator(shots=shots)
    backend.seed_simulator = int(seed)
    pass_manager = generate_preset_pass_manager(optimization_level=0, backend=backend)

    ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=qaoa_p)
    candidate_circuit = pass_manager.run(ansatz)
    sample_template = candidate_circuit.copy()
    sample_template.measure_all()

    init_beta = math.pi / 2.0
    init_gamma = math.pi
    init_params = np.array([init_beta] * qaoa_p + [init_gamma] * qaoa_p, dtype=np.float64)
    if init_params.size != candidate_circuit.num_parameters:
        init_params = np.zeros((candidate_circuit.num_parameters,), dtype=np.float64)

    best_energy = float("inf")
    best_mask = np.zeros((rank,), dtype=np.int8)
    trace: List[float] = []
    objective_func_vals: List[float] = []

    with Session(backend=backend) as session:
        estimator = Estimator(mode=session)
        estimator.options.default_shots = shots
        estimator.seed_estimator = int(seed)

        sampler = Sampler(mode=session)
        sampler.options.default_shots = shots
        sampler.seed_estimator = int(seed)

        def objective(params: np.ndarray) -> float:
            nonlocal best_energy, best_mask

            if len(trace) >= maxiter:
                if objective_func_vals:
                    return float(objective_func_vals[-1] / scale_ref)
                return 0.0

            isa_hamiltonian = cost_hamiltonian.apply_layout(candidate_circuit.layout)
            est_pub = (candidate_circuit, isa_hamiltonian, np.asarray(params, dtype=np.float64))
            est_job = estimator.run([est_pub])
            est_val_scaled = float(est_job.result()[0].data.evs)
            # Convert back to unscaled QUBO objective units for plotting.
            objective_func_vals.append(est_val_scaled * scale_ref)

            sampled_circuit = sample_template.assign_parameters(np.asarray(params, dtype=np.float64))
            sample_job = sampler.run([(sampled_circuit,)], shots=shots)
            counts = sample_job.result()[0].data.meas.get_int_counts()
            if counts:
                keys = list(counts.keys())
                values = np.asarray(list(counts.values()), dtype=np.float64)
                k = min(top_k, values.size)
                top_indices = np.argsort(values)[-k:]
                for idx in top_indices:
                    z = _int_to_binary_vector(int(keys[idx]), rank)
                    E = qubo_energy(z, Q, q)
                    if E < best_energy:
                        best_energy = E
                        best_mask = z.copy()
            trace.append(best_energy if np.isfinite(best_energy) else float("inf"))
            return est_val_scaled

        result = minimize(
            objective,
            init_params,
            method=str(optimizer),
            tol=1e-6,
            options={"maxiter": maxiter},
        )

        if not np.isfinite(best_energy):
            final_params = np.asarray(result.x, dtype=np.float64)
            sampled_circuit = sample_template.assign_parameters(final_params)
            sample_job = sampler.run([(sampled_circuit,)], shots=shots)
            counts = sample_job.result()[0].data.meas.get_int_counts()
            if counts:
                keys = list(counts.keys())
                values = np.asarray(list(counts.values()), dtype=np.float64)
                k = min(top_k, values.size)
                top_indices = np.argsort(values)[-k:]
                for idx in top_indices:
                    z = _int_to_binary_vector(int(keys[idx]), rank)
                    E = qubo_energy(z, Q, q)
                    if E < best_energy:
                        best_energy = E
                        best_mask = z.copy()

    if not np.isfinite(best_energy):
        best_mask = np.zeros((rank,), dtype=np.int8)
        best_energy = qubo_energy(best_mask, Q, q)

    trace_arr = np.asarray(trace[:maxiter], dtype=np.float64)
    if trace_arr.size < maxiter:
        fill_val = float(best_energy)
        trace_arr = np.pad(
            trace_arr,
            (0, maxiter - trace_arr.size),
            mode="constant",
            constant_values=fill_val,
        )
    obj_arr = np.asarray(objective_func_vals[:maxiter], dtype=np.float64)
    if obj_arr.size < maxiter:
        if obj_arr.size == 0:
            obj_arr = np.full((maxiter,), np.nan, dtype=np.float64)
        else:
            obj_arr = np.pad(
                obj_arr,
                (0, maxiter - obj_arr.size),
                mode="edge",
            )
    return best_mask, float(best_energy), trace_arr, obj_arr


def reconstruct_from_mask(
    mask: np.ndarray,
    weighted_components: Sequence[torch.Tensor],
    shape_like: torch.Tensor,
) -> torch.Tensor:
    X_hat = torch.zeros_like(shape_like)
    for idx, bit in enumerate(mask):
        if int(bit) == 1:
            X_hat = X_hat + weighted_components[idx]
    return X_hat


def evaluate_mask_metrics(
    X: torch.Tensor,
    X_norm: float,
    weighted_components: Sequence[torch.Tensor],
    mask: np.ndarray,
    energy: float,
    es_energy: Optional[float],
) -> Dict[str, float]:
    X_hat = reconstruct_from_mask(mask, weighted_components, X)
    rel_err = float(torch.linalg.norm(X - X_hat).item() / max(float(X_norm), 1e-12))
    R_hat = int(np.asarray(mask, dtype=np.int64).sum())
    gap = float(energy - es_energy) if es_energy is not None else float("nan")
    return {
        "Rhat": float(R_hat),
        "relerr": rel_err,
        "energy": float(energy),
        "gap": gap,
    }


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.std())


def _pad_trace(trace: np.ndarray, target_len: int) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float64).reshape(-1)
    target_len = int(target_len)
    if target_len <= 0:
        return np.zeros((0,), dtype=np.float64)
    if trace.size == 0:
        return np.full((target_len,), np.nan, dtype=np.float64)
    if trace.size >= target_len:
        return trace[:target_len]
    return np.pad(trace, (0, target_len - trace.size), mode="edge")


def _resolve_plot_mu(plot_mu: float, mu_values: Sequence[float]) -> float:
    if len(mu_values) == 0:
        raise ValueError("mu_values cannot be empty.")
    for mu in mu_values:
        if _is_close(mu, plot_mu):
            return float(mu)
    nearest = min(mu_values, key=lambda v: abs(float(v) - float(plot_mu)))
    print(
        f"[warn] plot_mu={plot_mu} not found in mu_values; "
        f"using nearest value {nearest}."
    )
    return float(nearest)


def _prepare_qubo_inputs(Q: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Q_arr = np.asarray(Q, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
    if Q_arr.ndim != 2:
        raise ValueError(f"Q must be 2D, got shape {Q_arr.shape}.")
    if Q_arr.shape[0] != Q_arr.shape[1]:
        raise ValueError(f"Q must be square, got shape {Q_arr.shape}.")
    if Q_arr.shape[0] != q_arr.shape[0]:
        raise ValueError(
            f"Q and q dimension mismatch: Q={Q_arr.shape}, q={q_arr.shape}."
        )
    if not np.all(np.isfinite(Q_arr)) or not np.all(np.isfinite(q_arr)):
        raise ValueError("Q and q must be finite.")
    # Use a symmetric QUBO matrix for all solvers to avoid tiny numerical asymmetry differences.
    Q_arr = 0.5 * (Q_arr + Q_arr.T)
    return np.ascontiguousarray(Q_arr), np.ascontiguousarray(q_arr)


def _metric_row(solver: str, values: Dict[str, List[float]]) -> Dict[str, float]:
    mean_rhat, std_rhat = _mean_std(values["Rhat"])
    mean_relerr, std_relerr = _mean_std(values["relerr"])
    mean_energy, std_energy = _mean_std(values["energy"])
    mean_gap, std_gap = _mean_std(values["gap"])
    return {
        "solver": solver,
        "mean_Rhat": mean_rhat,
        "std_Rhat": std_rhat,
        "mean_relerr": mean_relerr,
        "std_relerr": std_relerr,
        "mean_energy": mean_energy,
        "std_energy": std_energy,
        "mean_gap": mean_gap,
        "std_gap": std_gap,
    }


def _fmt_latex_pm(mean: float, std: float, sci: bool = False) -> str:
    if (not np.isfinite(mean)) or (not np.isfinite(std)):
        return r"\textit{--}"
    if sci:
        return f"{mean:.3e} $\\pm$ {std:.3e}"
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def write_latex_table(rows: Sequence[Dict[str, float]], output_path: Path) -> None:
    by_method = {str(r["solver"]): r for r in rows}

    def _row_line(method: str) -> str:
        row = by_method.get(method)
        if row is None:
            return (
                f"{method} & \\textit{{--}} & \\textit{{--}} & \\textit{{--}} & \\textit{{--}} \\\\"
            )
        rhat = _fmt_latex_pm(float(row["mean_Rhat"]), float(row["std_Rhat"]), sci=False)
        rel = _fmt_latex_pm(float(row["mean_relerr"]), float(row["std_relerr"]), sci=True)
        energy = _fmt_latex_pm(float(row["mean_energy"]), float(row["std_energy"]), sci=True)
        if method == "ES":
            gap = "0"
        else:
            gap = _fmt_latex_pm(float(row["mean_gap"]), float(row["std_gap"]), sci=True)
        return f"{method} & {rhat} & {rel} & {energy} & {gap} \\\\"

    lines = [
        r"\begin{table}[t]",
        r"\caption{Result on synthetic dataset.}",
        r"\label{tab:synth_metrics}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Method & \(\hat R\) & Rel.\ error \(\downarrow\) & \(E(\mathbf{z})\) \(\downarrow\) & Gap to ES \(\downarrow\) \\",
        r"\midrule",
        _row_line("ES"),
        _row_line("BF"),
        _row_line("QAOA"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def plot_convergence(
    bf_mean: np.ndarray,
    bf_std: np.ndarray,
    qaoa_mean: np.ndarray,
    qaoa_std: np.ndarray,
    use_gap: bool,
    es_min_value: Optional[float],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("matplotlib is required to create the convergence plot.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_bf = np.arange(1, len(bf_mean) + 1)
    x_qaoa = np.arange(1, len(qaoa_mean) + 1)
    bf_markevery = max(1, len(x_bf) // 8)
    qaoa_markevery = max(1, len(x_qaoa) // 8)

    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    bf_line, = ax.plot(
        x_bf,
        bf_mean,
        color="#1f77b4",
        linewidth=1.5,
        marker="o",
        markersize=3.0,
        markevery=bf_markevery,
        label="BF",
    )
    bf_valid = np.isfinite(bf_mean) & np.isfinite(bf_std)
    if np.any(bf_valid):
        xb = x_bf[bf_valid]
        ax.fill_between(
            xb,
            bf_mean[bf_valid] - bf_std[bf_valid],
            bf_mean[bf_valid] + bf_std[bf_valid],
            color="#1f77b4",
            alpha=0.15,
        )
    qaoa_line, = ax.plot(
        x_qaoa,
        qaoa_mean,
        color="#d62728",
        linewidth=1.5,
        marker="s",
        markersize=3.0,
        markevery=qaoa_markevery,
        label="QAOA",
    )
    q_valid = np.isfinite(qaoa_mean) & np.isfinite(qaoa_std)
    if np.any(q_valid):
        xq = x_qaoa[q_valid]
        ax.fill_between(
            xq,
            qaoa_mean[q_valid] - qaoa_std[q_valid],
            qaoa_mean[q_valid] + qaoa_std[q_valid],
            color="#d62728",
            alpha=0.15,
        )
    x_max = int(max(len(x_bf), len(x_qaoa), 1))
    ax.set_xlim(1, x_max)
    if x_max <= 10:
        x_ticks = np.arange(1, x_max + 1, dtype=int)
    else:
        x_ticks = np.unique(np.linspace(1, x_max, num=5, dtype=int))
    ax.set_xticks(x_ticks)
    optimum_line = None
    if es_min_value is not None and np.isfinite(es_min_value):
        x_es = float(max(1, int(round(0.9 * x_max))))
        optimum_line = ax.axhline(
            y=float(es_min_value),
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="Optimum",
        )
        ax.plot(
            [x_es],
            [float(es_min_value)],
            marker="*",
            color="black",
            markersize=7.0,
            linestyle="None",
        )
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("Objective", fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.3)
    if optimum_line is not None:
        optimum_legend = Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            marker="*",
            markersize=7.0,
            linewidth=1.0,
            label="Optimum",
        )
        ax.legend(handles=[optimum_legend, bf_line, qaoa_line], frameon=False, fontsize=9)
    else:
        ax.legend(handles=[bf_line, qaoa_line], frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=1000)
    plt.close(fig)


def plot_qaoa_loss_history(
    qaoa_loss_mat: np.ndarray,
    qaoa_loss_mean: np.ndarray,
    qaoa_loss_std: np.ndarray,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("matplotlib is required to create the QAOA loss plot.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, len(qaoa_loss_mean) + 1)
    markevery = max(1, len(x) // 8)

    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    if qaoa_loss_mat.size > 0:
        for row in qaoa_loss_mat:
            ax.plot(x, row, color="#aaaaaa", linewidth=0.7, alpha=0.35)
    ax.plot(
        x,
        qaoa_loss_mean,
        color="#d62728",
        linewidth=1.5,
        marker="^",
        markersize=3.0,
        markevery=markevery,
        label="QAOA loss",
    )
    x_max = int(max(len(x), 1))
    ax.set_xlim(1, x_max)
    if x_max <= 10:
        x_ticks = np.arange(1, x_max + 1, dtype=int)
    else:
        x_ticks = np.unique(np.linspace(1, x_max, num=5, dtype=int))
    ax.set_xticks(x_ticks)
    valid = np.isfinite(qaoa_loss_mean) & np.isfinite(qaoa_loss_std)
    if np.any(valid):
        x_valid = x[valid]
        y_lo = qaoa_loss_mean[valid] - qaoa_loss_std[valid]
        y_hi = qaoa_loss_mean[valid] + qaoa_loss_std[valid]
        ax.fill_between(x_valid, y_lo, y_hi, color="#d62728", alpha=0.12)

    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("QAOA Loss", fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=1000)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic poster experiment pipeline.")
    parser.add_argument(
        "--manifest",
        action="append",
        default=None,
        help="Path to a manifest JSON (repeatable). If omitted, uses --manifest-glob.",
    )
    parser.add_argument(
        "--manifest-glob",
        type=str,
        default="data/synthetic/manifest_*.json",
        help="Glob for manifest JSON files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If set, keep only records with this seed.",
    )
    parser.add_argument("--R-over", type=int, default=20, dest="R_over")
    parser.add_argument("--als-sweeps", type=int, default=200)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--mu-values", type=str, default="0.05,0.10,0.20")
    parser.add_argument("--plot-mu", type=float, default=0.10)
    parser.add_argument("--qaoa-p", type=int, default=2)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--qaoa-maxiter", type=int, default=200)
    parser.add_argument(
        "--qaoa-optimizer",
        type=str,
        default="COBYLA",
        help="Optimizer method passed to scipy.optimize.minimize.",
    )
    parser.add_argument("--bf-max-evals", type=int, default=200)
    parser.add_argument("--bf-restarts", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="script/poster")
    parser.add_argument(
        "--figure-path",
        type=str,
        default="script/poster/synth_convergence_qaoa_bf.png",
    )
    parser.add_argument(
        "--qaoa-loss-figure-path",
        type=str,
        default="script/poster/synth_qaoa_loss_history.png",
    )
    parser.add_argument(
        "--tex-table-path",
        type=str,
        default="script/poster/synth_metrics.tex",
        help="Path to write the filled LaTeX metrics table (for --plot-mu).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = _dtype_from_str(args.dtype)
    mu_values = _parse_list_float(args.mu_values)
    if len(mu_values) == 0:
        raise SystemExit("mu_values cannot be empty.")
    plot_mu = _resolve_plot_mu(args.plot_mu, mu_values)

    manifests: List[Path]
    if args.manifest:
        manifests = [Path(p) for p in args.manifest]
    else:
        manifests = list(Path(".").glob(args.manifest_glob))
    if not manifests:
        raise SystemExit("No manifest files found. Use --manifest or --manifest-glob.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = Path(args.figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    qaoa_loss_figure_path = Path(args.qaoa_loss_figure_path)
    qaoa_loss_figure_path.parent.mkdir(parents=True, exist_ok=True)
    tex_table_path = Path(args.tex_table_path)
    tex_table_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[DatasetRecord] = []
    for manifest_path in manifests:
        records.extend(load_manifest(manifest_path))
    if not records:
        raise SystemExit("No dataset records found in the selected manifests.")
    if args.seed is not None:
        records = [r for r in records if int(r.seed) == int(args.seed)]
        if not records:
            raise SystemExit(f"No dataset records found for seed={args.seed}.")

    print(f"[info] loaded {len(records)} dataset records.")
    print(
        f"[info] config R_over={args.R_over} als_sweeps={args.als_sweeps} "
        f"bf_max_evals={args.bf_max_evals} qaoa_maxiter={args.qaoa_maxiter}"
    )
    if int(args.R_over) > 20:
        raise SystemExit(
            "This workflow requires exhaustive search for every seed/mu; "
            "please set --R-over <= 20."
        )

    solvers = ("ES", "BF", "QAOA")
    metrics_by_mu: Dict[float, Dict[str, Dict[str, List[float]]]] = {}
    for mu in mu_values:
        metrics_by_mu[mu] = {
            solver: {"Rhat": [], "relerr": [], "energy": [], "gap": []}
            for solver in solvers
        }

    bf_plot_traces: List[np.ndarray] = []
    qaoa_plot_traces: List[np.ndarray] = []
    qaoa_loss_plot_traces: List[np.ndarray] = []
    es_plot_values: List[float] = []
    processed_records = 0
    for idx, record in enumerate(records, start=1):
        if not record.path.exists():
            print(f"[warn] missing dataset file: {record.path}")
            continue

        payload = torch.load(record.path, map_location="cpu")
        if "X" not in payload:
            print(f"[warn] payload has no key 'X': {record.path}")
            continue

        X = payload["X"].to(device=args.device, dtype=dtype)
        xnorm = float(torch.linalg.norm(X).clamp_min(1e-12).item())
        label = f"record={idx}/{len(records)} seed={record.seed} path={record.path.name}"
        print(f"[als] start {label}")
        factors, weights, als_rel_error, als_time_s = run_overfactored_als(
            X=X,
            rank_over=args.R_over,
            sweeps=args.als_sweeps,
            ridge=args.ridge,
            seed=record.seed,
            log_every=args.log_every,
            verbose=args.verbose,
            run_label=label,
        )
        factors_w = _absorb_weights_into_factors(factors, weights)
        weighted_components, Q_base, q_base = build_qubo_from_factors(X, factors_w)
        print(
            f"[als] done  {label} rel_error={als_rel_error:.6e} "
            f"time_s={als_time_s:.2f} rank={args.R_over}"
        )
        processed_records += 1

        for mu_idx, mu in enumerate(mu_values):
            q_raw = q_base + float(mu) * np.ones_like(q_base)
            Q, q = _prepare_qubo_inputs(Q_base, q_raw)
            base_seed = int(record.seed) + 100_003 * int(mu_idx + 1)

            es_mask, es_energy = exhaustive_search_qubo(Q, q)
            es_energy = qubo_energy(es_mask, Q, q)
            es_metrics = evaluate_mask_metrics(
                X=X,
                X_norm=xnorm,
                weighted_components=weighted_components,
                mask=es_mask,
                energy=es_energy,
                es_energy=es_energy,
            )
            for key, val in es_metrics.items():
                metrics_by_mu[mu]["ES"][key].append(val)

            bf_mask, bf_energy, _bf_best_trace, bf_loss_trace = bitflip_search_qubo(
                Q=Q,
                q=q,
                max_evals=args.bf_max_evals,
                restarts=args.bf_restarts,
                seed=base_seed,
            )
            bf_energy_eval = qubo_energy(bf_mask, Q, q)
            if abs(float(bf_energy) - float(bf_energy_eval)) > 1e-9:
                print(
                    f"[warn] BF energy mismatch detected; using canonical value "
                    f"{bf_energy_eval:.6e} (solver reported {bf_energy:.6e})."
                )
            bf_energy = bf_energy_eval
            bf_metrics = evaluate_mask_metrics(
                X=X,
                X_norm=xnorm,
                weighted_components=weighted_components,
                mask=bf_mask,
                energy=bf_energy,
                es_energy=es_energy,
            )
            for key, val in bf_metrics.items():
                metrics_by_mu[mu]["BF"][key].append(val)

            qaoa_mask, qaoa_energy, qaoa_trace, qaoa_loss_trace = qaoa_search_qubo(
                Q=Q,
                q=q,
                qaoa_p=args.qaoa_p,
                shots=args.shots,
                maxiter=args.qaoa_maxiter,
                seed=base_seed + 17,
                optimizer=args.qaoa_optimizer,
            )
            qaoa_energy_eval = qubo_energy(qaoa_mask, Q, q)
            if abs(float(qaoa_energy) - float(qaoa_energy_eval)) > 1e-9:
                print(
                    f"[warn] QAOA energy mismatch detected; using canonical value "
                    f"{qaoa_energy_eval:.6e} (solver reported {qaoa_energy:.6e})."
                )
            qaoa_energy = qaoa_energy_eval
            qaoa_metrics = evaluate_mask_metrics(
                X=X,
                X_norm=xnorm,
                weighted_components=weighted_components,
                mask=qaoa_mask,
                energy=qaoa_energy,
                es_energy=es_energy,
            )
            for key, val in qaoa_metrics.items():
                metrics_by_mu[mu]["QAOA"][key].append(val)

            print(
                f"[seed] seed={record.seed} mu={mu} method=ES "
                f"R_hat={es_metrics['Rhat']:.0f} rel_err={es_metrics['relerr']:.6e} "
                f"shift_ES={es_metrics['gap']:.6e}"
            )
            print(
                f"[seed] seed={record.seed} mu={mu} method=BF "
                f"R_hat={bf_metrics['Rhat']:.0f} rel_err={bf_metrics['relerr']:.6e} "
                f"shift_ES={bf_metrics['gap']:.6e}"
            )
            print(
                f"[seed] seed={record.seed} mu={mu} method=QAOA "
                f"R_hat={qaoa_metrics['Rhat']:.0f} rel_err={qaoa_metrics['relerr']:.6e} "
                f"shift_ES={qaoa_metrics['gap']:.6e}"
            )

            if _is_close(mu, plot_mu):
                es_plot_values.append(float(es_energy))
                bf_y = _pad_trace(np.asarray(bf_loss_trace, dtype=np.float64), int(args.bf_max_evals))
                qaoa_y = _pad_trace(np.asarray(qaoa_trace, dtype=np.float64), int(args.qaoa_maxiter))
                qaoa_loss_y = _pad_trace(
                    np.asarray(qaoa_loss_trace, dtype=np.float64),
                    int(args.qaoa_maxiter),
                )
                bf_plot_traces.append(bf_y)
                qaoa_plot_traces.append(qaoa_y)
                qaoa_loss_plot_traces.append(qaoa_loss_y)

    if processed_records == 0:
        raise SystemExit("No valid dataset payloads were loaded.")

    for mu in mu_values:
        rows = [
            _metric_row("ES", metrics_by_mu[mu]["ES"]),
            _metric_row("BF", metrics_by_mu[mu]["BF"]),
            _metric_row("QAOA", metrics_by_mu[mu]["QAOA"]),
        ]
        metrics_path = out_dir / f"metrics_mu={_format_mu(mu)}.csv"
        pd.DataFrame(rows).to_csv(metrics_path, index=False)
        print(f"[info] wrote metrics to {metrics_path}")
        if _is_close(mu, plot_mu):
            write_latex_table(rows, tex_table_path)
            print(f"[info] wrote latex table to {tex_table_path}")
        for row in rows:
            print(
                f"[summary] mu={mu} method={row['solver']} "
                f"R_hat={row['mean_Rhat']:.4f}±{row['std_Rhat']:.4f} "
                f"rel_err={row['mean_relerr']:.6e}±{row['std_relerr']:.6e} "
                f"shift_ES={row['mean_gap']:.6e}±{row['std_gap']:.6e}"
            )

    if bf_plot_traces and qaoa_plot_traces:
        bf_mat = np.vstack(bf_plot_traces)
        qaoa_mat = np.vstack(qaoa_plot_traces)
        bf_mean = np.nanmean(bf_mat, axis=0)
        bf_std = np.nanstd(bf_mat, axis=0)
        qaoa_mean = np.nanmean(qaoa_mat, axis=0)
        qaoa_std = np.nanstd(qaoa_mat, axis=0)
    else:
        bf_mean = np.zeros((int(args.bf_max_evals),), dtype=np.float64)
        bf_std = np.zeros((int(args.bf_max_evals),), dtype=np.float64)
        qaoa_mean = np.zeros((int(args.qaoa_maxiter),), dtype=np.float64)
        qaoa_std = np.zeros((int(args.qaoa_maxiter),), dtype=np.float64)

    if qaoa_loss_plot_traces:
        qaoa_loss_mat = np.vstack(qaoa_loss_plot_traces)
        qaoa_loss_mean = np.nanmean(qaoa_loss_mat, axis=0)
        qaoa_loss_std = np.nanstd(qaoa_loss_mat, axis=0)
    else:
        qaoa_loss_mat = np.zeros((0, int(args.qaoa_maxiter)), dtype=np.float64)
        qaoa_loss_mean = np.zeros((int(args.qaoa_maxiter),), dtype=np.float64)
        qaoa_loss_std = np.zeros((int(args.qaoa_maxiter),), dtype=np.float64)
    es_min_value_plot = (
        float(np.nanmean(np.asarray(es_plot_values, dtype=np.float64)))
        if len(es_plot_values) > 0
        else None
    )

    convergence_path = out_dir / f"convergence_mu={_format_mu(plot_mu)}.npz"
    np.savez(
        convergence_path,
        mu=float(plot_mu),
        use_gap=np.array([1], dtype=np.int64),
        bf_x=np.arange(1, len(bf_mean) + 1, dtype=np.int64),
        bf_mean=bf_mean,
        bf_std=bf_std,
        qaoa_x=np.arange(1, len(qaoa_mean) + 1, dtype=np.int64),
        qaoa_mean=qaoa_mean,
        qaoa_std=qaoa_std,
        qaoa_loss_x=np.arange(1, len(qaoa_loss_mean) + 1, dtype=np.int64),
        qaoa_loss_mean=qaoa_loss_mean,
        qaoa_loss_std=qaoa_loss_std,
        es_min_value=np.array(
            [es_min_value_plot if es_min_value_plot is not None else np.nan],
            dtype=np.float64,
        ),
    )
    print(f"[info] wrote convergence traces to {convergence_path}")

    plot_convergence(
        bf_mean=bf_mean,
        bf_std=bf_std,
        qaoa_mean=qaoa_mean,
        qaoa_std=qaoa_std,
        use_gap=True,
        es_min_value=es_min_value_plot,
        output_path=figure_path,
    )
    print(f"[info] wrote convergence figure to {figure_path}")
    plot_qaoa_loss_history(
        qaoa_loss_mat=qaoa_loss_mat,
        qaoa_loss_mean=qaoa_loss_mean,
        qaoa_loss_std=qaoa_loss_std,
        output_path=qaoa_loss_figure_path,
    )
    print(f"[info] wrote QAOA loss figure to {qaoa_loss_figure_path}")


if __name__ == "__main__":
    main()
