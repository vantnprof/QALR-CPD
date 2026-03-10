#!/usr/bin/env python3
"""
Run lightweight synthetic poster experiments:
1) fit overfactored CP-ALS once per dataset,
2) freeze factors and build a soft-rank QUBO,
3) solve with ES / BF / QAOA (sim) / QAOA (IBM, one run),
4) aggregate metrics and convergence traces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _scale_mu(mu: float, mode: str, xnorm: float, rank_ref: int) -> float:
    mode_key = mode.strip().lower()
    if mode_key == "none":
        return float(mu)
    if mode_key == "xnorm":
        return float(mu) * float(xnorm)
    if mode_key == "xnorm2":
        return float(mu) * (float(xnorm) ** 2)
    if mode_key in ("xnorm2_over_rank", "xnorm2/rank", "xnorm2-over-rank"):
        denom = max(1, int(rank_ref))
        return float(mu) * (float(xnorm) ** 2) / float(denom)
    raise ValueError(f"Unknown mu effect mode '{mode}'.")


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


def _load_ibm_runtime_config(path_str: Optional[str]) -> Dict[str, str]:
    config: Dict[str, str] = {}

    if path_str:
        path = Path(path_str)
        if path.exists():
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict):
                config.update({str(k): str(v) for k, v in loaded.items()})
        else:
            print(
                f"[warn] IBM config file not found: {path}. "
                "Falling back to environment variables or saved account."
            )

    env_map = {
        "IBM_API_KEY": "API_KEY",
        "IBM_CRN": "CRN",
        "IBM_INSTANCE": "CRN",
        "IBM_BACKEND": "backend_name",
        "IBM_CHANNEL": "channel",
    }
    for env_key, cfg_key in env_map.items():
        val = os.getenv(env_key)
        if val:
            config[cfg_key] = str(val)
    return config


def _resolve_ibm_backend(
    config_path: Optional[str],
    backend_override: Optional[str],
    channel_override: Optional[str],
    instance_override: Optional[str],
) -> Tuple[Any, str]:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "IBM Runtime dependencies are missing. Install qiskit-ibm-runtime and retry."
        ) from exc

    config = _load_ibm_runtime_config(config_path)
    if backend_override:
        config["backend_name"] = str(backend_override)
    if channel_override:
        config["channel"] = str(channel_override)
    if instance_override:
        config["CRN"] = str(instance_override)

    backend_name = str(config.get("backend_name", "ibm_fez"))
    channel = str(config.get("channel", "ibm_quantum_platform"))
    token = config.get("API_KEY")
    instance = config.get("CRN")

    service_kwargs: Dict[str, str] = {"channel": channel}
    if token:
        service_kwargs["token"] = str(token)
    if instance:
        service_kwargs["instance"] = str(instance)

    service = QiskitRuntimeService(**service_kwargs)
    backend = service.backend(backend_name)
    return backend, backend_name


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


def _select_records_by_seed(args: argparse.Namespace, records: Sequence[DatasetRecord]) -> List[DatasetRecord]:
    if args.seed is not None:
        selected = [r for r in records if int(r.seed) == int(args.seed)]
    elif args.seeds:
        requested = set(_parse_list_int(args.seeds))
        selected = [r for r in records if int(r.seed) in requested]
    else:
        requested = set(range(int(args.seed_start), int(args.seed_start) + int(args.num_seeds)))
        selected = [r for r in records if int(r.seed) in requested]
    return sorted(selected, key=lambda r: (int(r.seed), str(r.path)))


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
) -> Tuple[List[torch.Tensor], torch.Tensor, float, float, np.ndarray]:
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
    relerr_trace: List[float] = []
    for sweep in range(int(sweeps)):
        model.als_step(X, normalize=True)
        err = (torch.linalg.norm(X - model.forward()) / xnorm).item()
        relerr_trace.append(float(err))
        if verbose and (log_every <= 1 or (sweep + 1) % int(log_every) == 0):
            print(
                f"[als] {run_label} sweep={sweep + 1:03d} "
                f"rel_error={err:.6e} rank={model.rank}"
            )
    elapsed = time.time() - t0
    final_rel_error = (
        float(relerr_trace[-1])
        if len(relerr_trace) > 0
        else (torch.linalg.norm(X - model.forward()) / xnorm).item()
    )
    factors = [A.detach().clone() for A in model.factors]
    weights = model.weights.detach().clone()
    return (
        factors,
        weights,
        float(final_rel_error),
        float(elapsed),
        np.asarray(relerr_trace, dtype=np.float64),
    )


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
    verify_inputs: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    verify_context: str = "",
) -> Tuple[object, Dict[str, float]]:
    """
    Convert canonical QUBO E(z)=z^TQz+q^Tz into the same upper-triangular form
    used in src/rank_prunning/qaoa_rank_prunning.py before building Ising.
    """
    Q_np = np.asarray(Q, dtype=np.float64)
    q_np = np.asarray(q, dtype=np.float64).reshape(-1)
    if Q_np.shape[0] != q_np.shape[0]:
        raise ValueError(f"Q/q mismatch: Q={Q_np.shape}, q={q_np.shape}.")

    if verify_inputs:
        qaoa_Q = np.ascontiguousarray(Q_np)
        qaoa_q = np.ascontiguousarray(q_np)
        labels = ("ES", "BF", "QAOA")
        all_ok = True
        checks: List[str] = []
        for label in labels:
            ref = verify_inputs.get(label)
            if ref is None:
                checks.append(f"{label}=missing")
                all_ok = False
                continue

            ref_Q, ref_q = ref
            ref_Q_np = np.ascontiguousarray(np.asarray(ref_Q, dtype=np.float64))
            ref_q_np = np.ascontiguousarray(np.asarray(ref_q, dtype=np.float64).reshape(-1))
            Q_same = bool(np.array_equal(qaoa_Q, ref_Q_np))
            q_same = bool(np.array_equal(qaoa_q, ref_q_np))
            all_ok = all_ok and Q_same and q_same
            checks.append(
                f"{label}(Q={Q_same},q={q_same},dQ={_max_abs_diff(qaoa_Q, ref_Q_np):.3e},"
                f"dq={_max_abs_diff(qaoa_q, ref_q_np):.3e},"
                f"Qh={_array_digest(qaoa_Q)}/{_array_digest(ref_Q_np)},"
                f"qh={_array_digest(qaoa_q)}/{_array_digest(ref_q_np)})"
            )

        status = "PASS" if all_ok else "FAIL"
        prefix = f"{verify_context} " if verify_context else ""
        print(
            f"[debug] {prefix}build_qaoa_cost_operator_input_check={status} "
            + " ".join(checks)
        )

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


def _measure_logical_only(circuit, num_logical: int):
    from qiskit.circuit import QuantumCircuit

    measured = circuit.copy()
    if hasattr(measured, "remove_final_measurements"):
        measured.remove_final_measurements()
    if int(num_logical) > int(measured.num_qubits):
        raise ValueError(
            f"Requested {num_logical} logical qubits, but circuit has {measured.num_qubits} qubits."
        )
    wrapped = QuantumCircuit(measured.num_qubits, int(num_logical))
    wrapped.compose(measured, inplace=True)
    wrapped.measure(list(range(int(num_logical))), list(range(int(num_logical))))
    return wrapped


def _counts_to_int(counts: Dict) -> Dict[int, int]:
    if not counts:
        return {}
    sample_key = next(iter(counts.keys()))
    if isinstance(sample_key, int):
        return {int(k): int(v) for k, v in counts.items()}
    if isinstance(sample_key, str):
        return {int(str(k), 2): int(v) for k, v in counts.items()}
    return {}


def _extract_counts_int(result_item) -> Dict[int, int]:
    data = result_item.data
    for attr in ("meas", "m"):
        if hasattr(data, attr):
            obj = getattr(data, attr)
            if hasattr(obj, "get_int_counts"):
                return {int(k): int(v) for k, v in obj.get_int_counts().items()}
            if hasattr(obj, "get_counts"):
                return _counts_to_int(obj.get_counts())
    for name in dir(data):
        if name.startswith("_"):
            continue
        obj = getattr(data, name, None)
        if obj is None:
            continue
        if hasattr(obj, "get_int_counts"):
            return {int(k): int(v) for k, v in obj.get_int_counts().items()}
        if hasattr(obj, "get_counts"):
            return _counts_to_int(obj.get_counts())
    return {}


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
    verify_inputs_for_cost_op: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    verify_context: str = "",
    backend: Optional[Any] = None,
    pass_manager_opt_level: int = 0,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from qiskit.circuit.library import QAOAAnsatz
        from qiskit.transpiler import generate_preset_pass_manager
        from qiskit_ibm_runtime import (
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
        zhist = np.zeros((maxiter,), dtype=np.float64)
        return z0, 0.0, zhist, zhist.copy(), zhist.copy()

    cost_hamiltonian, qubo_meta = build_qaoa_cost_operator(
        Q=Q,
        q=q,
        verify_inputs=verify_inputs_for_cost_op,
        verify_context=verify_context,
    )
    scale_ref = float(qubo_meta.get("qubo_scale_ref", 1.0))
    if not (scale_ref > 0.0):
        scale_ref = 1.0
    top_k = max(1, rank)

    backend_runtime = backend
    if backend_runtime is None:
        try:
            from qiskit_aer import AerSimulator
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "QAOA simulator backend requires qiskit-aer. Install qiskit-aer and retry."
            ) from exc
        backend_runtime = AerSimulator(shots=shots)
        if hasattr(backend_runtime, "seed_simulator"):
            backend_runtime.seed_simulator = int(seed)
    pass_manager = generate_preset_pass_manager(
        optimization_level=max(0, int(pass_manager_opt_level)),
        backend=backend_runtime,
    )

    ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=qaoa_p)
    candidate_circuit = pass_manager.run(ansatz)
    sample_template = _measure_logical_only(candidate_circuit, rank)

    init_beta = math.pi / 2.0
    init_gamma = math.pi
    init_params = np.array([init_beta] * qaoa_p + [init_gamma] * qaoa_p, dtype=np.float64)
    if init_params.size != candidate_circuit.num_parameters:
        init_params = np.zeros((candidate_circuit.num_parameters,), dtype=np.float64)

    best_energy = float("inf")
    best_mask = np.zeros((rank,), dtype=np.int8)
    trace: List[float] = []
    iter_trace: List[float] = []
    objective_func_vals: List[float] = []

    estimator = Estimator(mode=backend_runtime)
    if hasattr(estimator, "options"):
        estimator.options.default_shots = int(shots)
    if hasattr(estimator, "seed_estimator"):
        estimator.seed_estimator = int(seed)

    sampler = Sampler(mode=backend_runtime)
    if hasattr(sampler, "options"):
        sampler.options.default_shots = int(shots)
    if hasattr(sampler, "seed_estimator"):
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
        objective_func_vals.append(est_val_scaled * scale_ref)

        sampled_circuit = sample_template.assign_parameters(np.asarray(params, dtype=np.float64))
        sample_job = sampler.run([(sampled_circuit,)], shots=shots)
        counts = _extract_counts_int(sample_job.result()[0])
        iter_energy = float("inf")
        if counts:
            keys = list(counts.keys())
            values = np.asarray(list(counts.values()), dtype=np.float64)
            k = min(top_k, values.size)
            top_indices = np.argsort(values)[-k:]
            for idx in top_indices:
                z = _int_to_binary_vector(int(keys[idx]), rank)
                E = qubo_energy(z, Q, q)
                if E < iter_energy:
                    iter_energy = E
                if E < best_energy:
                    best_energy = E
                    best_mask = z.copy()
        trace.append(best_energy if np.isfinite(best_energy) else float("inf"))
        iter_trace.append(iter_energy if np.isfinite(iter_energy) else float("nan"))
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
        counts = _extract_counts_int(sample_job.result()[0])
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
    iter_arr = np.asarray(iter_trace[:maxiter], dtype=np.float64)
    if iter_arr.size < maxiter:
        if iter_arr.size == 0:
            iter_arr = np.full((maxiter,), np.nan, dtype=np.float64)
        else:
            iter_arr = np.pad(
                iter_arr,
                (0, maxiter - iter_arr.size),
                mode="edge",
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
    return best_mask, float(best_energy), trace_arr, obj_arr, iter_arr


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
    delta = (
        float(abs(float(energy) - float(es_energy)) / max(abs(float(es_energy)), 1e-12))
        if es_energy is not None
        else float("nan")
    )
    return {
        "Rhat": float(R_hat),
        "relerr": rel_err,
        "energy": float(energy),
        "gap": gap,
        "delta": delta,
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


def _array_digest(arr: np.ndarray) -> str:
    arr_c = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    return hashlib.sha256(arr_c.tobytes(order="C")).hexdigest()[:12]


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.shape != b_arr.shape:
        return float("inf")
    if a_arr.size == 0:
        return 0.0
    return float(np.max(np.abs(a_arr - b_arr)))


def _debug_verify_qubo_inputs(
    Q_es: np.ndarray,
    q_es: np.ndarray,
    Q_bf: np.ndarray,
    q_bf: np.ndarray,
    Q_qaoa: np.ndarray,
    q_qaoa: np.ndarray,
    seed: int,
    mu: float,
) -> None:
    q_same = bool(np.array_equal(q_es, q_bf) and np.array_equal(q_es, q_qaoa))
    Q_same = bool(np.array_equal(Q_es, Q_bf) and np.array_equal(Q_es, Q_qaoa))
    status = "PASS" if (Q_same and q_same) else "FAIL"

    q_diff = max(_max_abs_diff(q_es, q_bf), _max_abs_diff(q_es, q_qaoa))
    Q_diff = max(_max_abs_diff(Q_es, Q_bf), _max_abs_diff(Q_es, Q_qaoa))
    print(
        f"[debug] seed={seed} mu={_format_mu(mu)} qubo_input_check={status} "
        f"Q_equal={Q_same} q_equal={q_same} "
        f"Q_hash[ES/BF/QAOA]={_array_digest(Q_es)}/{_array_digest(Q_bf)}/{_array_digest(Q_qaoa)} "
        f"q_hash[ES/BF/QAOA]={_array_digest(q_es)}/{_array_digest(q_bf)}/{_array_digest(q_qaoa)} "
        f"max_abs_diff_Q={Q_diff:.3e} max_abs_diff_q={q_diff:.3e}"
    )


def _metric_row(solver: str, values: Dict[str, List[float]]) -> Dict[str, float]:
    mean_rhat, std_rhat = _mean_std(values["Rhat"])
    mean_relerr, std_relerr = _mean_std(values["relerr"])
    mean_energy, std_energy = _mean_std(values["energy"])
    mean_gap, std_gap = _mean_std(values["gap"])
    mean_delta, std_delta = _mean_std(values["delta"] if "delta" in values else [])
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
        "mean_delta": mean_delta,
        "std_delta": std_delta,
    }


def _fmt_latex_pm(mean: float, std: float, sci: bool = False) -> str:
    if (not np.isfinite(mean)) or (not np.isfinite(std)):
        return r"\textit{--}"
    if sci:
        return f"{mean:.3e} $\\pm$ {std:.3e}"
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def write_latex_table(
    rows: Sequence[Dict[str, float]],
    output_path: Path,
    qaoa_ibm_solver_key: str = "QAOA (ibm_fez)",
) -> None:
    by_method = {str(r["solver"]): r for r in rows}

    def _row_values(solver_key: str, is_es: bool = False) -> Tuple[str, str, str]:
        row = by_method.get(solver_key)
        if row is None:
            return r"\textit{--}", r"\textit{--}", "0" if is_es else r"\textit{--}"
        rhat = _fmt_latex_pm(float(row["mean_Rhat"]), float(row["std_Rhat"]), sci=False)
        rel = _fmt_latex_pm(float(row["mean_relerr"]), float(row["std_relerr"]), sci=True)
        if is_es:
            delta = "0"
        else:
            delta = _fmt_latex_pm(
                float(row.get("mean_delta", float("nan"))),
                float(row.get("std_delta", float("nan"))),
                sci=True,
            )
        return rhat, rel, delta

    es_rhat, es_rel, es_delta = _row_values("ES", is_es=True)
    bf_rhat, bf_rel, bf_delta = _row_values("BF", is_es=False)
    sim_rhat, sim_rel, sim_delta = _row_values("QAOA", is_es=False)
    ibm_rhat, ibm_rel, ibm_delta = _row_values(qaoa_ibm_solver_key, is_es=False)
    qaoa_ibm_label_tex = qaoa_ibm_solver_key.replace("_", r"\_")

    lines = [
        r"\begin{table}[t]",
        r"\caption{Synthetic dataset: results for ES, BF, simulator-QAOA, and IBM hardware-QAOA on the induced rank-selection QUBO.}",
        r"\label{tab:synth_metrics}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Method & \(\hat{R}\) & Rel.\ error \(\downarrow\) & \(\bar{\delta}\) \(\downarrow\) \\",
        r"\midrule",
        f"ES                               & {es_rhat} & {es_rel} & {es_delta} \\\\",
        f"BF                               & {bf_rhat} & {bf_rel} & {bf_delta} \\\\",
        f"QAOA (sim)                       & {sim_rhat} & {sim_rel} & {sim_delta} \\\\",
        f"{qaoa_ibm_label_tex}                 & {ibm_rhat} & {ibm_rel} & {ibm_delta} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def _compute_xticks(x_max: int) -> np.ndarray:
    x_max = max(int(x_max), 1)
    if x_max <= 10:
        return np.arange(1, x_max + 1, dtype=int)

    ticks: List[int] = [1]
    ticks.extend(range(10, x_max + 1, 10))
    if ticks[-1] != x_max:
        ticks.append(x_max)
    return np.asarray(ticks, dtype=int)


def _marker_indices_for_ticks(x_values: np.ndarray, x_ticks: np.ndarray) -> List[int]:
    x_vals = np.asarray(x_values, dtype=np.int64).reshape(-1)
    ticks = np.asarray(x_ticks, dtype=np.int64).reshape(-1)
    if x_vals.size == 0 or ticks.size == 0:
        return []
    return np.flatnonzero(np.isin(x_vals, ticks)).astype(int).tolist()


def _endpoint_nudge(x_max: int) -> float:
    span = max(int(x_max) - 1, 0)
    if span <= 0:
        return 0.0
    return min(0.9, 0.3 * span)


def _nudge_endpoint_positions(x: np.ndarray, x_max: int) -> np.ndarray:
    x_plot = np.asarray(x, dtype=np.float64).reshape(-1).copy()
    if x_plot.size == 0:
        return x_plot

    delta = _endpoint_nudge(x_max)
    if delta <= 0.0:
        return x_plot

    if np.isclose(x_plot[0], 1.0):
        x_plot[0] += delta
    if x_plot.size >= 2 and np.isclose(x_plot[-1], float(x_max)):
        x_plot[-1] -= delta
    return x_plot


def _set_inset_endpoint_ticks(ax, x_max: int) -> None:
    x_ticks = _compute_xticks(x_max)

    tick_pos = x_ticks.astype(np.float64)
    delta = _endpoint_nudge(x_max)
    if delta > 0.0 and tick_pos.size > 0:
        left_idx = np.where(np.isclose(tick_pos, 1.0))[0]
        if left_idx.size > 0:
            tick_pos[left_idx[0]] += delta

        right_idx = np.where(np.isclose(tick_pos, float(x_max)))[0]
        if right_idx.size > 0:
            tick_pos[right_idx[0]] -= delta

    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(int(v)) for v in x_ticks])


def plot_convergence(
    bf_mean: np.ndarray,
    bf_std: np.ndarray,
    qaoa_mean: np.ndarray,
    qaoa_std: np.ndarray,
    use_gap: bool,
    es_min_value: Optional[float],
    output_path: Path,
    qaoa_ibm_mean: Optional[np.ndarray] = None,
    qaoa_ibm_std: Optional[np.ndarray] = None,
    qaoa_ibm_label: str = "QAOA (ibm_fez)",
    show_zoom: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("matplotlib is required to create the convergence plot.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bf_mean = np.asarray(bf_mean, dtype=np.float64).reshape(-1)
    qaoa_mean = np.asarray(qaoa_mean, dtype=np.float64).reshape(-1)
    if qaoa_ibm_mean is None:
        qaoa_ibm_vals = np.zeros((0,), dtype=np.float64)
    else:
        qaoa_ibm_vals = np.asarray(qaoa_ibm_mean, dtype=np.float64).reshape(-1)
    x_bf = np.arange(1, bf_mean.size + 1, dtype=np.int64)
    x_qaoa = np.arange(1, qaoa_mean.size + 1, dtype=np.int64)
    x_qaoa_ibm = np.arange(1, qaoa_ibm_vals.size + 1, dtype=np.int64)
    x_max = int(max(len(x_bf), len(x_qaoa), len(x_qaoa_ibm), 1))
    x_ticks = _compute_xticks(x_max)
    x_bf_plot = _nudge_endpoint_positions(x_bf, x_max)
    x_qaoa_plot = _nudge_endpoint_positions(x_qaoa, x_max)
    x_qaoa_ibm_plot = _nudge_endpoint_positions(x_qaoa_ibm, x_max)
    bf_markevery = _marker_indices_for_ticks(x_bf, x_ticks)
    qaoa_markevery = _marker_indices_for_ticks(x_qaoa, x_ticks)
    qaoa_ibm_markevery = _marker_indices_for_ticks(x_qaoa_ibm, x_ticks)

    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    bf_line, = ax.plot(
        x_bf_plot,
        bf_mean,
        color="#1f77b4",
        linewidth=1.5,
        marker="o",
        markersize=3.0,
        markevery=bf_markevery,
        label="BF",
    )
    qaoa_line, = ax.plot(
        x_qaoa_plot,
        qaoa_mean,
        color="#d62728",
        linewidth=1.5,
        marker="s",
        markersize=3.0,
        markevery=qaoa_markevery,
        label="QAOA (sim)",
    )
    qaoa_ibm_line = None
    if qaoa_ibm_vals.size > 0 and np.any(np.isfinite(qaoa_ibm_vals)):
        qaoa_ibm_line, = ax.plot(
            x_qaoa_ibm_plot,
            qaoa_ibm_vals,
            color="#2ca02c",
            linewidth=1.5,
            marker="^",
            markersize=3.0,
            markevery=qaoa_ibm_markevery,
            label=qaoa_ibm_label,
        )
    ax.set_xlim(1, x_max)
    _set_inset_endpoint_ticks(ax, x_max)
    es_line = None
    if es_min_value is not None and np.isfinite(es_min_value):
        x_es = np.arange(1, x_max + 1, dtype=np.int64)
        x_es_plot = _nudge_endpoint_positions(x_es, x_max)
        es_markevery = _marker_indices_for_ticks(x_es, x_ticks)
        es_line, = ax.plot(
            x_es_plot,
            np.full((len(x_es_plot),), float(es_min_value), dtype=np.float64),
            color="black",
            linestyle="--",
            linewidth=1.0,
            marker="*",
            markersize=6.0,
            markevery=es_markevery,
            label="ES",
        )
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("Objective", fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="x", alpha=0.3)
    legend_handles = []
    if es_line is not None:
        legend_handles.append(es_line)
    legend_handles.extend([bf_line, qaoa_line])
    if qaoa_ibm_line is not None:
        legend_handles.append(qaoa_ibm_line)
    ax.legend(handles=legend_handles, frameon=False, fontsize=9)

    zoom_start = 10
    if show_zoom and x_max > zoom_start:
        bf_zoom_mask = x_bf >= int(zoom_start)
        qaoa_zoom_mask = x_qaoa >= int(zoom_start)
        qaoa_ibm_zoom_mask = x_qaoa_ibm >= int(zoom_start)
        if np.any(bf_zoom_mask) or np.any(qaoa_zoom_mask) or np.any(qaoa_ibm_zoom_mask):
            axins = ax.inset_axes([0.50, 0.18, 0.44, 0.34])

            if np.any(bf_zoom_mask):
                xb = x_bf[bf_zoom_mask]
                yb = bf_mean[bf_zoom_mask]
                axins.plot(xb, yb, color="#1f77b4", linewidth=1.0)

            if np.any(qaoa_zoom_mask):
                xq = x_qaoa[qaoa_zoom_mask]
                yq = qaoa_mean[qaoa_zoom_mask]
                axins.plot(xq, yq, color="#d62728", linewidth=1.0)
            if np.any(qaoa_ibm_zoom_mask):
                xqi = x_qaoa_ibm[qaoa_ibm_zoom_mask]
                yqi = qaoa_ibm_vals[qaoa_ibm_zoom_mask]
                axins.plot(xqi, yqi, color="#2ca02c", linewidth=1.0)

            if es_min_value is not None and np.isfinite(es_min_value):
                axins.axhline(float(es_min_value), color="black", linestyle="--", linewidth=0.8)

            axins.set_xlim(float(zoom_start), float(x_max))
            zoom_ticks = [int(t) for t in _compute_xticks(x_max).tolist() if int(t) >= int(zoom_start)]
            if len(zoom_ticks) == 0:
                zoom_ticks = [int(zoom_start), int(x_max)]
            axins.set_xticks(zoom_ticks)
            axins.set_xticklabels([""] * len(zoom_ticks))
            axins.tick_params(axis="x", which="both", bottom=True, labelbottom=False, length=2.0)
            axins.tick_params(axis="y", labelsize=7)
            axins.grid(axis="x", alpha=0.2)

            y_candidates: List[np.ndarray] = []
            if np.any(bf_zoom_mask):
                y_candidates.append(np.asarray(bf_mean[bf_zoom_mask], dtype=np.float64))
            if np.any(qaoa_zoom_mask):
                y_candidates.append(np.asarray(qaoa_mean[qaoa_zoom_mask], dtype=np.float64))
            if np.any(qaoa_ibm_zoom_mask):
                y_candidates.append(np.asarray(qaoa_ibm_vals[qaoa_ibm_zoom_mask], dtype=np.float64))
            if es_min_value is not None and np.isfinite(es_min_value):
                y_candidates.append(np.asarray([float(es_min_value)], dtype=np.float64))

            if y_candidates:
                y_all = np.concatenate([arr[np.isfinite(arr)] for arr in y_candidates if arr.size > 0])
                if y_all.size > 0:
                    y_min = float(np.min(y_all))
                    y_max = float(np.max(y_all))
                    span = max(y_max - y_min, 1e-12)
                    pad = 0.08 * span
                    axins.set_ylim(y_min - pad, y_max + pad)

            if hasattr(ax, "indicate_inset_zoom"):
                ax.indicate_inset_zoom(axins, edgecolor="0.5", alpha=0.8)

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
    x_max = int(max(len(x), 1))
    x_plot = _nudge_endpoint_positions(x, x_max)
    markevery = max(1, len(x) // 8)

    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    if qaoa_loss_mat.size > 0:
        for row in qaoa_loss_mat:
            ax.plot(x_plot, row, color="#aaaaaa", linewidth=0.7, alpha=0.35)
    ax.plot(
        x_plot,
        qaoa_loss_mean,
        color="#d62728",
        linewidth=1.5,
        marker="^",
        markersize=3.0,
        markevery=markevery,
        label="QAOA loss",
    )
    ax.set_xlim(1, x_max)
    _set_inset_endpoint_ticks(ax, x_max)
    valid = np.isfinite(qaoa_loss_mean) & np.isfinite(qaoa_loss_std)
    if np.any(valid):
        x_valid = x_plot[valid]
        y_lo = qaoa_loss_mean[valid] - qaoa_loss_std[valid]
        y_hi = qaoa_loss_mean[valid] + qaoa_loss_std[valid]
        ax.fill_between(x_valid, y_lo, y_hi, color="#d62728", alpha=0.12)

    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("QAOA Loss", fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=1000)
    plt.close(fig)


def _paired_output_paths(output_path: Path) -> Tuple[Path, Path]:
    base = output_path.with_suffix("") if output_path.suffix else output_path
    return base.with_suffix(".png"), base.with_suffix(".pdf")


def plot_als_relerr_history(
    als_relerr_mat: np.ndarray,
    output_path: Path,
) -> Tuple[Path, Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("matplotlib is required to create ALS convergence plots.") from exc

    mat = np.asarray(als_relerr_mat, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] == 0:
        raise ValueError("als_relerr_mat must be a non-empty 2D array.")

    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    x = np.arange(1, mat.shape[1] + 1, dtype=np.int64)
    x_max = int(max(len(x), 1))
    x_plot = _nudge_endpoint_positions(x, x_max)
    markevery = max(1, len(x) // 8)

    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    for row in mat:
        ax.plot(x_plot, row, color="#aaaaaa", linewidth=0.7, alpha=0.35)
    ax.plot(
        x_plot,
        mean,
        color="#2ca02c",
        linewidth=1.5,
        marker="d",
        markersize=3.0,
        markevery=markevery,
        label="ALS rel. error",
    )
    valid = np.isfinite(mean) & np.isfinite(std)
    if np.any(valid):
        xv = x_plot[valid]
        ax.fill_between(xv, mean[valid] - std[valid], mean[valid] + std[valid], color="#2ca02c", alpha=0.12)

    ax.set_xlim(1, x_max)
    _set_inset_endpoint_ticks(ax, x_max)
    ax.set_xlabel("ALS Sweep", fontsize=9)
    ax.set_ylabel("Relative Error", fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    png_path, pdf_path = _paired_output_paths(output_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight", dpi=1000)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


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
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated dataset seeds. Ignored when --seed is set.",
    )
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--R-over", type=int, default=20, dest="R_over")
    parser.add_argument("--als-sweeps", type=int, default=200)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--mu-values", type=str, default="0.10")
    parser.add_argument("--plot-mu", type=float, default=0.10)
    parser.add_argument(
        "--mu-effect",
        "--mu-scale",
        dest="mu_effect",
        type=str,
        default="xnorm2_over_rank",
        choices=("none", "xnorm", "xnorm2", "xnorm2_over_rank"),
        help=(
            "Scale mu to match data magnitude. "
            "'xnorm2_over_rank' treats mu as a per-component penalty relative to ||X||_F^2."
        ),
    )
    parser.add_argument("--qaoa-p", type=int, default=2)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--qaoa-maxiter", type=int, default=40)
    parser.add_argument(
        "--qaoa-ibm-maxiter",
        type=int,
        default=None,
        help=(
            "Max iterations for the single IBM hardware QAOA run. "
            "Defaults to --qaoa-maxiter when omitted."
        ),
    )
    parser.add_argument(
        "--qaoa-optimizer",
        type=str,
        default="COBYLA",
        help="Optimizer method passed to scipy.optimize.minimize.",
    )
    parser.add_argument(
        "--disable-qaoa-ibm",
        action="store_true",
        help="Skip the single IBM hardware QAOA run.",
    )
    parser.add_argument(
        "--ibm-config",
        type=str,
        default="script/poster/ibm_runtime_config.json",
        help="Path to IBM Runtime config JSON (API_KEY, CRN, backend_name, channel).",
    )
    parser.add_argument(
        "--ibm-backend",
        type=str,
        default="ibm_fez",
        help="IBM backend name used for the hardware QAOA run.",
    )
    parser.add_argument(
        "--ibm-channel",
        type=str,
        default=None,
        help="Optional IBM Runtime channel override.",
    )
    parser.add_argument(
        "--ibm-instance",
        type=str,
        default=None,
        help="Optional IBM Runtime instance/CRN override.",
    )
    parser.add_argument("--bf-max-evals", type=int, default=40)
    parser.add_argument("--bf-restarts", type=int, default=1)
    parser.add_argument(
        "--debug-verify-qubo-inputs",
        action="store_true",
        help=(
            "Print strict verification that ES/BF/QAOA receive identical QUBO inputs "
            "(Q matrix and q vector)."
        ),
    )
    parser.add_argument("--out-dir", type=str, default="script/poster/synth_result")
    parser.add_argument(
        "--figure-path",
        type=str,
        default="script/poster/synth_result/synth_convergence_qaoa_bf.png",
    )
    parser.add_argument(
        "--qaoa-loss-figure-path",
        type=str,
        default="script/poster/synth_result/synth_qaoa_loss_history.png",
    )
    parser.add_argument(
        "--als-figure-path",
        type=str,
        default="script/poster/synth_result/synth_als_relerr_history.png",
    )
    parser.add_argument(
        "--tex-table-path",
        type=str,
        default="script/poster/synth_result/synth_metrics.tex",
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
    als_figure_path = Path(args.als_figure_path)
    als_figure_path.parent.mkdir(parents=True, exist_ok=True)
    tex_table_path = Path(args.tex_table_path)
    tex_table_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[DatasetRecord] = []
    for manifest_path in manifests:
        records.extend(load_manifest(manifest_path))
    if not records:
        raise SystemExit("No dataset records found in the selected manifests.")
    records = _select_records_by_seed(args, records)
    if not records:
        if args.seed is not None:
            raise SystemExit(f"No dataset records found for seed={args.seed}.")
        raise SystemExit("No dataset records found for the selected seed range/list.")

    qaoa_ibm_maxiter = (
        int(args.qaoa_ibm_maxiter)
        if args.qaoa_ibm_maxiter is not None
        else int(args.qaoa_maxiter)
    )
    qaoa_ibm_maxiter = max(1, qaoa_ibm_maxiter)

    run_qaoa_ibm = not bool(args.disable_qaoa_ibm)
    qaoa_ibm_backend = None
    qaoa_ibm_backend_name = str(args.ibm_backend)
    qaoa_ibm_solver_key = f"QAOA ({qaoa_ibm_backend_name})"
    if run_qaoa_ibm:
        try:
            qaoa_ibm_backend, qaoa_ibm_backend_name = _resolve_ibm_backend(
                config_path=args.ibm_config,
                backend_override=args.ibm_backend,
                channel_override=args.ibm_channel,
                instance_override=args.ibm_instance,
            )
            qaoa_ibm_solver_key = f"QAOA ({qaoa_ibm_backend_name})"
            print(f"[info] IBM backend={qaoa_ibm_backend_name}")
        except Exception as exc:
            run_qaoa_ibm = False
            print(f"[warn] failed to initialize IBM backend; hardware QAOA disabled: {exc}")

    run_config = {
        "script": str(Path(__file__).resolve()),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": list(sys.argv),
        "args": dict(vars(args)),
        "resolved": {
            "manifest_paths": [str(p) for p in manifests],
            "num_records": int(len(records)),
            "record_seeds": sorted({int(r.seed) for r in records}),
            "mu_values": [float(mu) for mu in mu_values],
            "plot_mu": float(plot_mu),
            "qaoa_ibm_enabled": bool(run_qaoa_ibm),
            "qaoa_ibm_backend": str(qaoa_ibm_backend_name),
            "qaoa_ibm_solver_key": str(qaoa_ibm_solver_key),
            "qaoa_ibm_maxiter": int(qaoa_ibm_maxiter),
        },
    }
    run_config_path = out_dir / "synth_run_config.json"
    run_config_path.write_text(json.dumps(run_config, indent=2) + "\n")
    print(f"[info] wrote run config to {run_config_path}")

    print(f"[info] loaded {len(records)} dataset records.")
    print(
        f"[info] config R_over={args.R_over} als_sweeps={args.als_sweeps} "
        f"bf_max_evals={args.bf_max_evals} qaoa_maxiter={args.qaoa_maxiter} "
        f"qaoa_ibm_maxiter={qaoa_ibm_maxiter} mu_effect={args.mu_effect} "
        f"qaoa_ibm_enabled={run_qaoa_ibm}"
    )
    if int(args.R_over) > 20:
        raise SystemExit(
            "This workflow requires exhaustive search for every seed/mu; "
            "please set --R-over <= 20."
        )

    solvers = ("ES", "BF", "QAOA", qaoa_ibm_solver_key)
    metrics_by_mu: Dict[float, Dict[str, Dict[str, List[float]]]] = {}
    for mu in mu_values:
        metrics_by_mu[mu] = {
            solver: {"Rhat": [], "relerr": [], "energy": [], "gap": [], "delta": []}
            for solver in solvers
        }

    bf_plot_traces: List[np.ndarray] = []
    qaoa_plot_traces: List[np.ndarray] = []
    qaoa_ibm_plot_traces: List[np.ndarray] = []
    qaoa_loss_plot_traces: List[np.ndarray] = []
    qaoa_iter_plot_traces: List[np.ndarray] = []
    es_plot_values: List[float] = []
    als_plot_traces: List[np.ndarray] = []
    als_trace_rows: List[Dict[str, object]] = []
    processed_records = 0
    qaoa_ibm_attempted = False
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
        factors, weights, als_rel_error, als_time_s, als_relerr_trace = run_overfactored_als(
            X=X,
            rank_over=args.R_over,
            sweeps=args.als_sweeps,
            ridge=args.ridge,
            seed=record.seed,
            log_every=args.log_every,
            verbose=args.verbose,
            run_label=label,
        )
        for sweep_idx, relerr in enumerate(als_relerr_trace.tolist(), start=1):
            als_trace_rows.append(
                {
                    "dataset_path": str(record.path),
                    "data_seed": int(record.seed),
                    "run_seed": int(record.seed),
                    "sweep": int(sweep_idx),
                    "rel_error": float(relerr),
                    "rank_over": int(args.R_over),
                    "als_sweeps_requested": int(args.als_sweeps),
                    "als_time_s": float(als_time_s),
                }
            )
        als_plot_traces.append(np.asarray(als_relerr_trace, dtype=np.float64))
        factors_w = _absorb_weights_into_factors(factors, weights)
        weighted_components, Q_base, q_base = build_qubo_from_factors(X, factors_w)
        print(
            f"[als] done  {label} rel_error={als_rel_error:.6e} "
            f"time_s={als_time_s:.2f} rank={args.R_over}"
        )
        processed_records += 1

        for mu_idx, mu in enumerate(mu_values):
            mu_effective = _scale_mu(mu, args.mu_effect, xnorm=xnorm, rank_ref=args.R_over)
            q_raw = q_base + float(mu_effective) * np.ones_like(q_base)
            Q, q = _prepare_qubo_inputs(Q_base, q_raw)
            base_seed = int(record.seed) + 100_003 * int(mu_idx + 1)
            shared_solver_seed = int(base_seed % 65536)

            if args.debug_verify_qubo_inputs:
                Q_es = np.array(Q, dtype=np.float64, copy=True, order="C")
                q_es = np.array(q, dtype=np.float64, copy=True, order="C")
                Q_bf = np.array(Q, dtype=np.float64, copy=True, order="C")
                q_bf = np.array(q, dtype=np.float64, copy=True, order="C")
                Q_qaoa = np.array(Q, dtype=np.float64, copy=True, order="C")
                q_qaoa = np.array(q, dtype=np.float64, copy=True, order="C")
                _debug_verify_qubo_inputs(
                    Q_es=Q_es,
                    q_es=q_es,
                    Q_bf=Q_bf,
                    q_bf=q_bf,
                    Q_qaoa=Q_qaoa,
                    q_qaoa=q_qaoa,
                    seed=int(record.seed),
                    mu=float(mu),
                )
            else:
                Q_es, q_es = Q, q
                Q_bf, q_bf = Q, q
                Q_qaoa, q_qaoa = Q, q

            qaoa_verify_inputs = (
                {
                    "ES": (Q_es, q_es),
                    "BF": (Q_bf, q_bf),
                    "QAOA": (Q_qaoa, q_qaoa),
                }
                if args.debug_verify_qubo_inputs
                else None
            )
            qaoa_verify_context = (
                f"seed={int(record.seed)} mu={_format_mu(float(mu))}"
                if args.debug_verify_qubo_inputs
                else ""
            )

            es_mask, es_energy = exhaustive_search_qubo(Q_es, q_es)
            es_energy = qubo_energy(es_mask, Q_es, q_es)
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
                Q=Q_bf,
                q=q_bf,
                max_evals=args.bf_max_evals,
                restarts=args.bf_restarts,
                seed=shared_solver_seed,
            )
            bf_energy_eval = qubo_energy(bf_mask, Q_bf, q_bf)
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

            qaoa_mask, qaoa_energy, qaoa_trace, qaoa_loss_trace, qaoa_iter_trace = qaoa_search_qubo(
                Q=Q_qaoa,
                q=q_qaoa,
                qaoa_p=args.qaoa_p,
                shots=args.shots,
                maxiter=args.qaoa_maxiter,
                seed=shared_solver_seed,
                optimizer=args.qaoa_optimizer,
                verify_inputs_for_cost_op=qaoa_verify_inputs,
                verify_context=qaoa_verify_context,
            )
            qaoa_energy_eval = qubo_energy(qaoa_mask, Q_qaoa, q_qaoa)
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

            qaoa_ibm_metrics: Optional[Dict[str, float]] = None
            if run_qaoa_ibm and (not qaoa_ibm_attempted) and _is_close(mu, plot_mu):
                qaoa_ibm_attempted = True
                try:
                    (
                        qaoa_ibm_mask,
                        qaoa_ibm_energy,
                        qaoa_ibm_trace,
                        _qaoa_ibm_loss_trace,
                        _qaoa_ibm_iter_trace,
                    ) = qaoa_search_qubo(
                        Q=Q_qaoa,
                        q=q_qaoa,
                        qaoa_p=args.qaoa_p,
                        shots=args.shots,
                        maxiter=qaoa_ibm_maxiter,
                        seed=shared_solver_seed,
                        optimizer=args.qaoa_optimizer,
                        verify_inputs_for_cost_op=qaoa_verify_inputs,
                        verify_context=qaoa_verify_context,
                        backend=qaoa_ibm_backend,
                    )
                    qaoa_ibm_energy_eval = qubo_energy(qaoa_ibm_mask, Q_qaoa, q_qaoa)
                    if abs(float(qaoa_ibm_energy) - float(qaoa_ibm_energy_eval)) > 1e-9:
                        print(
                            f"[warn] {qaoa_ibm_solver_key} energy mismatch detected; "
                            f"using canonical value {qaoa_ibm_energy_eval:.6e} "
                            f"(solver reported {qaoa_ibm_energy:.6e})."
                        )
                    qaoa_ibm_energy = qaoa_ibm_energy_eval
                    qaoa_ibm_metrics = evaluate_mask_metrics(
                        X=X,
                        X_norm=xnorm,
                        weighted_components=weighted_components,
                        mask=qaoa_ibm_mask,
                        energy=qaoa_ibm_energy,
                        es_energy=es_energy,
                    )
                    for key, val in qaoa_ibm_metrics.items():
                        metrics_by_mu[mu][qaoa_ibm_solver_key][key].append(val)
                    qaoa_ibm_plot_traces.append(
                        _pad_trace(np.asarray(qaoa_ibm_trace, dtype=np.float64), int(qaoa_ibm_maxiter))
                    )
                except Exception as exc:
                    print(f"[warn] {qaoa_ibm_solver_key} run failed: {exc}")

            print(
                f"[seed] seed={record.seed} mu={mu} method=ES "
                f"mu_eff={mu_effective:.6e} "
                f"R_hat={es_metrics['Rhat']:.0f} rel_err={es_metrics['relerr']:.6e} "
                f"delta_bar={es_metrics['delta']:.6e}"
            )
            print(
                f"[seed] seed={record.seed} mu={mu} method=BF "
                f"mu_eff={mu_effective:.6e} "
                f"R_hat={bf_metrics['Rhat']:.0f} rel_err={bf_metrics['relerr']:.6e} "
                f"delta_bar={bf_metrics['delta']:.6e}"
            )
            print(
                f"[seed] seed={record.seed} mu={mu} method=QAOA "
                f"mu_eff={mu_effective:.6e} "
                f"R_hat={qaoa_metrics['Rhat']:.0f} rel_err={qaoa_metrics['relerr']:.6e} "
                f"delta_bar={qaoa_metrics['delta']:.6e}"
            )
            if qaoa_ibm_metrics is not None:
                print(
                    f"[seed] seed={record.seed} mu={mu} method={qaoa_ibm_solver_key} "
                    f"mu_eff={mu_effective:.6e} "
                    f"R_hat={qaoa_ibm_metrics['Rhat']:.0f} "
                    f"rel_err={qaoa_ibm_metrics['relerr']:.6e} "
                    f"delta_bar={qaoa_ibm_metrics['delta']:.6e}"
                )

            if _is_close(mu, plot_mu):
                es_plot_values.append(float(es_energy))
                bf_y = _pad_trace(np.asarray(bf_loss_trace, dtype=np.float64), int(args.bf_max_evals))
                qaoa_y = _pad_trace(np.asarray(qaoa_trace, dtype=np.float64), int(args.qaoa_maxiter))
                qaoa_loss_y = _pad_trace(
                    np.asarray(qaoa_loss_trace, dtype=np.float64),
                    int(args.qaoa_maxiter),
                )
                qaoa_iter_y = _pad_trace(
                    np.asarray(qaoa_iter_trace, dtype=np.float64),
                    int(args.qaoa_maxiter),
                )
                bf_plot_traces.append(bf_y)
                qaoa_plot_traces.append(qaoa_y)
                qaoa_loss_plot_traces.append(qaoa_loss_y)
                qaoa_iter_plot_traces.append(qaoa_iter_y)

    if processed_records == 0:
        raise SystemExit("No valid dataset payloads were loaded.")

    als_trace_path = out_dir / "synth_als_relerr_history.csv"
    pd.DataFrame(als_trace_rows).to_csv(als_trace_path, index=False)
    print(f"[info] wrote ALS relative-error history to {als_trace_path}")
    if als_plot_traces:
        als_mat = np.vstack(als_plot_traces)
        als_png, als_pdf = plot_als_relerr_history(als_mat, als_figure_path)
        print(f"[info] wrote ALS convergence figures to {als_png} and {als_pdf}")

    for mu in mu_values:
        rows = [
            _metric_row("ES", metrics_by_mu[mu]["ES"]),
            _metric_row("BF", metrics_by_mu[mu]["BF"]),
            _metric_row("QAOA", metrics_by_mu[mu]["QAOA"]),
            _metric_row(qaoa_ibm_solver_key, metrics_by_mu[mu][qaoa_ibm_solver_key]),
        ]
        metrics_path = out_dir / f"synth_metrics_mu={_format_mu(mu)}.csv"
        pd.DataFrame(rows).to_csv(metrics_path, index=False)
        print(f"[info] wrote metrics to {metrics_path}")
        if _is_close(mu, plot_mu):
            write_latex_table(
                rows,
                tex_table_path,
                qaoa_ibm_solver_key=qaoa_ibm_solver_key,
            )
            print(f"[info] wrote latex table to {tex_table_path}")
        for row in rows:
            print(
                f"[summary] mu={mu} method={row['solver']} "
                f"R_hat={row['mean_Rhat']:.4f}±{row['std_Rhat']:.4f} "
                f"rel_err={row['mean_relerr']:.6e}±{row['std_relerr']:.6e} "
                f"delta_bar={row['mean_delta']:.6e}±{row['std_delta']:.6e}"
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

    if qaoa_ibm_plot_traces:
        qaoa_ibm_mat = np.vstack(qaoa_ibm_plot_traces)
        qaoa_ibm_mean = np.nanmean(qaoa_ibm_mat, axis=0)
        qaoa_ibm_std = np.nanstd(qaoa_ibm_mat, axis=0)
    else:
        qaoa_ibm_mat = np.zeros((0, qaoa_ibm_maxiter), dtype=np.float64)
        qaoa_ibm_mean = np.zeros((0,), dtype=np.float64)
        qaoa_ibm_std = np.zeros((0,), dtype=np.float64)

    if qaoa_loss_plot_traces:
        qaoa_loss_mat = np.vstack(qaoa_loss_plot_traces)
        qaoa_loss_mean = np.nanmean(qaoa_loss_mat, axis=0)
        qaoa_loss_std = np.nanstd(qaoa_loss_mat, axis=0)
    else:
        qaoa_loss_mat = np.zeros((0, int(args.qaoa_maxiter)), dtype=np.float64)
        qaoa_loss_mean = np.zeros((int(args.qaoa_maxiter),), dtype=np.float64)
        qaoa_loss_std = np.zeros((int(args.qaoa_maxiter),), dtype=np.float64)

    if qaoa_iter_plot_traces:
        qaoa_iter_mat = np.vstack(qaoa_iter_plot_traces)
        qaoa_iter_mean = np.nanmean(qaoa_iter_mat, axis=0)
        qaoa_iter_std = np.nanstd(qaoa_iter_mat, axis=0)
    else:
        qaoa_iter_mat = np.zeros((0, int(args.qaoa_maxiter)), dtype=np.float64)
        qaoa_iter_mean = np.zeros((int(args.qaoa_maxiter),), dtype=np.float64)
        qaoa_iter_std = np.zeros((int(args.qaoa_maxiter),), dtype=np.float64)
    es_min_value_plot = (
        float(np.nanmean(np.asarray(es_plot_values, dtype=np.float64)))
        if len(es_plot_values) > 0
        else None
    )

    convergence_path = out_dir / f"synth_convergence_mu={_format_mu(plot_mu)}.npz"
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
        qaoa_ibm_x=np.arange(1, len(qaoa_ibm_mean) + 1, dtype=np.int64),
        qaoa_ibm_mean=qaoa_ibm_mean,
        qaoa_ibm_std=qaoa_ibm_std,
        qaoa_ibm_solver=np.asarray([qaoa_ibm_solver_key]),
        qaoa_loss_x=np.arange(1, len(qaoa_loss_mean) + 1, dtype=np.int64),
        qaoa_loss_mean=qaoa_loss_mean,
        qaoa_loss_std=qaoa_loss_std,
        qaoa_iter_x=np.arange(1, len(qaoa_iter_mean) + 1, dtype=np.int64),
        qaoa_iter_mean=qaoa_iter_mean,
        qaoa_iter_std=qaoa_iter_std,
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
        qaoa_ibm_mean=qaoa_ibm_mean,
        qaoa_ibm_std=qaoa_ibm_std,
        qaoa_ibm_label=qaoa_ibm_solver_key,
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
