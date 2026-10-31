#!/usr/bin/env python3
"""
Run QALR-CPD (QAOA pruning) on the amino acid dataset with two backends:
  1) Ideal simulator (Aer)
  2) Noisy simulator to mimic an IBM NISQ device

This script records:
  - Per-iteration rel. error + rank (sim vs noisy)
  - QAOA optimization landscapes for first/last pruning calls (sim vs noisy)
  - QAOA measurement distributions for first/last pruning calls (sim vs noisy)
  - Exhaustive-search reference (first/last pruning calls)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "pandas is required to run the experiments. Please install pandas and retry."
    ) from exc

try:
    import numpy as np
    from scipy.optimize import minimize
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "numpy and scipy are required to run QAOA. Please install them and retry."
    ) from exc

try:
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator, SamplerV2 as Sampler
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "qiskit, qiskit-aer, and qiskit-ibm-runtime are required to run QAOA. "
        "Please install them and retry."
    ) from exc

from src.cp.als import CPFactorization
from src.pruning.exhaustive import (
    normalize_cp_factors,
    inner_product_rank1,
    exhaustive_prune_cp,
)


def _parse_list_int(raw: str) -> List[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _dataset_tag(shape: Sequence[int], rank_true: int, corr_rho: float, noise_std: float) -> str:
    dims = "x".join(str(s) for s in shape)
    return f"I{dims}_R{rank_true}_rho{corr_rho}_noise{noise_std}"


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
        return float(mu) * float(xnorm) ** 2
    if mode_key in ("xnorm2_over_rank", "xnorm2/rank", "xnorm2-over-rank"):
        denom = max(1, int(rank_ref))
        return float(mu) * (float(xnorm) ** 2) / float(denom)
    raise ValueError(f"Unknown mu scaling mode '{mode}'.")


def _sanitize_tag(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)


def _init_factors(shape: Sequence[int], rank: int, seed: int, device, dtype) -> List[torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    factors: List[torch.Tensor] = []
    for dim in shape:
        A = torch.randn((int(dim), int(rank)), generator=g, dtype=dtype, device=device)
        A = A / (A.norm(dim=0, keepdim=True).clamp_min(1e-12))
        factors.append(A)
    return factors


def _bits_from_int(mask: int, num_bits: int) -> List[int]:
    bits = [int(b) for b in format(mask, f"0{num_bits}b")]
    bits.reverse()  # match little-endian ordering used in qaoa_rank_prunning
    return bits


def _bits_to_str(bits: Sequence[int]) -> str:
    return "".join(str(int(b)) for b in bits)


def _build_cost_operator(Q: np.ndarray, lin: np.ndarray) -> SparsePauliOp:
    R = Q.shape[0]
    lin_eff = lin.copy()
    lin_eff += np.diag(Q)

    pauli_terms: Dict[str, complex] = {}
    I_str = "I" * R
    const = 0.0

    def add_term(label: str, coeff: float) -> None:
        if abs(coeff) < 1e-12:
            return
        pauli_terms[label] = pauli_terms.get(label, 0.0) + coeff

    for i in range(R):
        const += 0.5 * lin_eff[i]
        z_label = list(I_str)
        z_label[R - 1 - i] = "Z"
        add_term("".join(z_label), -0.5 * lin_eff[i])

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
            add_term("".join(zi), -q / 4.0)
            add_term("".join(zj), -q / 4.0)
            add_term("".join(zij), q / 4.0)

    add_term(I_str, const)
    labels = list(pauli_terms.keys())
    coeffs = list(pauli_terms.values())
    return SparsePauliOp(labels, coeffs)


def _energy_value(mask_bits: List[int], G: torch.Tensor, c: torch.Tensor, mu: float) -> float:
    z = torch.tensor(mask_bits, device=G.device, dtype=G.dtype)
    E = (z @ (G @ z)) - 2.0 * (z @ c) + mu * z.sum()
    return float(E.detach().cpu())


def _build_noise_model(p1: float, p2: float, readout: float) -> NoiseModel:
    noise_model = NoiseModel()
    error_1q = depolarizing_error(float(p1), 1)
    error_2q = depolarizing_error(float(p2), 2)
    noise_model.add_all_qubit_quantum_error(
        error_1q, ["u1", "u2", "u3", "rx", "ry", "rz", "sx", "x"]
    )
    noise_model.add_all_qubit_quantum_error(
        error_2q, ["cx", "cz", "cy", "swap"]
    )
    if readout > 0:
        ro = float(readout)
        ro_err = ReadoutError([[1 - ro, ro], [ro, 1 - ro]])
        noise_model.add_all_qubit_readout_error(ro_err)
    return noise_model


def _build_backend(
    shots: int,
    seed: int,
    noisy: bool,
    noise_p1: float,
    noise_p2: float,
    readout: float,
) -> AerSimulator:
    if noisy:
        noise_model = _build_noise_model(noise_p1, noise_p2, readout)
        backend = AerSimulator(
            shots=int(shots),
            noise_model=noise_model,
            basis_gates=noise_model.basis_gates,
        )
    else:
        backend = AerSimulator(shots=int(shots))
    backend.seed_simulator = int(seed)
    return backend


def qaoa_prune_cp_backend(
    X: torch.Tensor,
    factors: Sequence[torch.Tensor],
    mu: float,
    weights: Optional[torch.Tensor],
    require_nonempty: bool,
    max_R_qaoa: int,
    qaoa_reps: int,
    shots: int,
    maxiter: int,
    seed: int,
    num_restarts: int,
    top_k: Optional[int],
    backend: AerSimulator,
    backend_label: str,
    log_qaoa: bool = False,
    log_prefix: Optional[str] = None,
) -> Tuple[List[int], float, Tuple[torch.Tensor, List[torch.Tensor]], Dict]:
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
        overlaps = overlaps * (U[n].T @ U[n])

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
    scale = max(np.max(np.abs(Q)), np.max(np.abs(lin)), 1.0)
    
    Q_s = Q / scale
    lin_s = lin / scale
    
    cost_hamiltonian = _build_cost_operator(Q=Q_s, lin=lin_s)

    best_selected: List[int] = []
    best_val = math.inf
    best_mask_bits: Optional[List[int]] = None
    best_obj_vals: Optional[List[float]] = None
    best_distribution: Optional[Dict[str, float]] = None

    num_restarts_eff = max(1, int(num_restarts))
    for restart_idx in range(num_restarts_eff):
        objective_vals: List[float] = []
        circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=int(qaoa_reps))
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
            objective_vals.append(float(cost))
            if log_qaoa:
                iter_idx = len(objective_vals)
                prefix = log_prefix or f"{backend_label}"
                print(
                    f"[qaoa] {prefix} restart={restart_idx + 1}/{num_restarts_eff} "
                    f"iter={iter_idx:03d} cost={float(cost):.6e}"
                )
            return float(cost)

        with Session(backend=backend) as session:
            estimator = Estimator(mode=session)
            estimator.options.default_shots = int(shots)
            estimator.seed_estimator = int(seed)
            result = minimize(
                cost_func_estimator,
                init_params,
                args=(candidate_circuit, cost_hamiltonian, estimator),
                method="COBYLA",
                tol=1e-6,
                options={"maxiter": int(maxiter)},
            )

        opt_params = result.x
        optimized_circuit = candidate_circuit.assign_parameters(opt_params)
        optimized_circuit.measure_all()

        sampler = Sampler(mode=backend)
        sampler.options.default_shots = int(shots)
        sampler.seed_estimator = int(seed)

        job = sampler.run([(optimized_circuit,)], shots=int(shots))
        counts_int = job.result()[0].data.meas.get_int_counts()
        shots_total = sum(counts_int.values())
        distribution_int = {int(k): v / shots_total for k, v in counts_int.items()}
        distribution_str = {
            _bits_to_str(_bits_from_int(k, R)): float(v)
            for k, v in distribution_int.items()
        }

        values = list(distribution_int.values())
        keys = list(distribution_int.keys())
        if values:
            top_k_eff = min(int(top_k), len(values))
            top_idx = np.argsort(values)[-top_k_eff:]
        else:
            top_idx = []

        for idx in top_idx:
            sol_bits = _bits_from_int(int(keys[idx]), R)
            E_val = _energy_value(sol_bits, G_t, c_t, mu)
            if E_val < best_val:
                best_val = E_val
                best_mask_bits = sol_bits
                best_selected = [i for i, b in enumerate(sol_bits) if b == 1]
                best_obj_vals = list(objective_vals)
                best_distribution = dict(distribution_str)

    diagnostics = {
        "backend": backend_label,
        "R": int(R),
        "best_mask_bits": best_mask_bits,
        "best_mask_str": _bits_to_str(best_mask_bits) if best_mask_bits is not None else None,
        "best_E_no_const": float(best_val),
        "objective_values": best_obj_vals or [],
        "distribution": best_distribution or {},
        "qaoa_restarts": int(max(1, int(num_restarts))),
        "shots": int(shots),
    }
    return best_selected, float(best_val), (lam, U), diagnostics


def _apply_pruning(
    model: CPFactorization,
    lam: torch.Tensor,
    U: List[torch.Tensor],
    selected: List[int],
) -> bool:
    if len(selected) == 0:
        return False
    old_rank = model.rank
    if len(selected) < old_rank:
        model._apply_pruning_selection(lam, U, selected)
        return True
    model._set_weights_tensor(lam)
    return False


def _build_exhaustive_info(
    X: torch.Tensor,
    factors: Sequence[torch.Tensor],
    mu: float,
    weights: Optional[torch.Tensor],
    max_r: int,
) -> Optional[Dict]:
    try:
        selected, best_E, _, diag = exhaustive_prune_cp(
            X,
            factors,
            mu=float(mu),
            weights=weights,
            require_nonempty=True,
            max_R_exhaustive=int(max_r),
        )
    except Exception:
        return None

    mask_int = diag.get("best_mask")
    if mask_int is None:
        return None
    bits = _bits_from_int(int(mask_int), int(factors[0].shape[1]))
    bit_str = _bits_to_str(bits)
    return {
        "selected": selected,
        "best_E_no_const": float(best_E),
        "best_mask_bits": bits,
        "best_mask_str": bit_str,
        "distribution": {bit_str: 1.0},
    }


def run_als_dual(
    X: torch.Tensor,
    shape: Sequence[int],
    rank_init: int,
    ridge: float,
    n_sweeps: int,
    tol: float,
    seed: int,
    prune_mu: float,
    prune_period: int,
    max_rank_prune: int,
    qaoa_reps: int,
    qaoa_shots: int,
    qaoa_maxiter: int,
    qaoa_seed: int,
    qaoa_restarts: int,
    log_every: int,
    verbose: bool,
    backend_sim: AerSimulator,
    backend_ibm: Optional[AerSimulator],
    exhaustive_max_r: int,
    run_label: str,
    sim_only: bool,
) -> Tuple[List[Dict], Dict, Dict]:
    torch.manual_seed(seed)
    init_factors = _init_factors(shape, rank_init, seed, X.device, X.dtype)
    sim_model = CPFactorization(
        shape=shape,
        rank=rank_init,
        ridge=ridge,
        init=[f.clone() for f in init_factors],
        seed=seed,
        device=X.device,
        dtype=X.dtype,
    )
    ibm_model = None
    if not sim_only:
        ibm_model = CPFactorization(
            shape=shape,
            rank=rank_init,
            ridge=ridge,
            init=[f.clone() for f in init_factors],
            seed=seed,
            device=X.device,
            dtype=X.dtype,
        )

    rows: List[Dict] = []
    Xnorm = torch.linalg.norm(X).clamp_min(1e-12)
    prunes_applied = {"sim": 0, "ibm": 0}
    prev_err_sim: Optional[float] = None
    prev_err_ibm: Optional[float] = None
    cumulative_sim = 0.0
    cumulative_ibm = 0.0

    prune_events: Dict[str, List[Dict]] = {"sim": [], "ibm": []}

    total_iters = int(n_sweeps)

    for sweep in range(total_iters):
        # Simulator branch
        sim_t0 = time.time()
        sim_model.als_step(X, normalize=True)

        prune_applied_sim = False
        if prune_mu is not None and prune_period > 0:
            last_sweep = sweep == total_iters - 1
            should_prune = not last_sweep
            prune_frequency = int(prune_period)
            if should_prune and (sweep + 1) % prune_frequency == 0:
                rank_before = sim_model.rank
                selected, _, (lam, U), diag = qaoa_prune_cp_backend(
                    X=X,
                    factors=[A.detach() for A in sim_model.factors],
                    mu=float(prune_mu),
                    weights=sim_model.weights,
                    require_nonempty=True,
                    max_R_qaoa=int(max_rank_prune),
                    qaoa_reps=int(qaoa_reps),
                    shots=int(qaoa_shots),
                    maxiter=int(qaoa_maxiter),
                    seed=int(qaoa_seed),
                    num_restarts=int(qaoa_restarts),
                    top_k=None,
                    backend=backend_sim,
                    backend_label="sim",
                    log_qaoa=verbose,
                    log_prefix=f"sim sweep={sweep + 1}",
                )
                exhaustive_info = _build_exhaustive_info(
                    X=X,
                    factors=[A.detach() for A in sim_model.factors],
                    mu=float(prune_mu),
                    weights=sim_model.weights,
                    max_r=int(exhaustive_max_r),
                )
                prune_applied_sim = _apply_pruning(sim_model, lam, U, selected)
                if prune_applied_sim:
                    prunes_applied["sim"] += 1
                prune_events["sim"].append(
                    {
                        "sweep": sweep + 1,
                        "rank_before": rank_before,
                        "rank_after": sim_model.rank,
                        "selected": selected,
                        "qaoa": diag,
                        "exhaustive": exhaustive_info,
                    }
                )

        err_sim = (torch.linalg.norm(X - sim_model.forward()) / Xnorm).item()
        sim_elapsed = time.time() - sim_t0
        cumulative_sim += sim_elapsed

        err_ibm = None
        ibm_elapsed = 0.0
        prune_applied_ibm = False
        if not sim_only:
            # IBM noisy branch
            ibm_t0 = time.time()
            ibm_model.als_step(X, normalize=True)
            if prune_mu is not None and prune_period > 0:
                last_sweep = sweep == total_iters - 1
                should_prune = not last_sweep
                prune_frequency = int(prune_period)
                if should_prune and (sweep + 1) % prune_frequency == 0:
                    rank_before = ibm_model.rank
                    selected, _, (lam, U), diag = qaoa_prune_cp_backend(
                        X=X,
                        factors=[A.detach() for A in ibm_model.factors],
                        mu=float(prune_mu),
                        weights=ibm_model.weights,
                        require_nonempty=True,
                        max_R_qaoa=int(max_rank_prune),
                        qaoa_reps=int(qaoa_reps),
                        shots=int(qaoa_shots),
                        maxiter=int(qaoa_maxiter),
                        seed=int(qaoa_seed),
                        num_restarts=int(qaoa_restarts),
                        top_k=None,
                        backend=backend_ibm,
                        backend_label="ibm",
                        log_qaoa=verbose,
                        log_prefix=f"ibm sweep={sweep + 1}",
                    )
                    exhaustive_info = _build_exhaustive_info(
                        X=X,
                        factors=[A.detach() for A in ibm_model.factors],
                        mu=float(prune_mu),
                        weights=ibm_model.weights,
                        max_r=int(exhaustive_max_r),
                    )
                    prune_applied_ibm = _apply_pruning(ibm_model, lam, U, selected)
                    if prune_applied_ibm:
                        prunes_applied["ibm"] += 1
                    prune_events["ibm"].append(
                        {
                            "sweep": sweep + 1,
                            "rank_before": rank_before,
                            "rank_after": ibm_model.rank,
                            "selected": selected,
                            "qaoa": diag,
                            "exhaustive": exhaustive_info,
                        }
                    )

            err_ibm = (torch.linalg.norm(X - ibm_model.forward()) / Xnorm).item()
            ibm_elapsed = time.time() - ibm_t0
            cumulative_ibm += ibm_elapsed

        if log_every <= 1 or (sweep + 1) % log_every == 0:
            rows.append(
                {
                    "sweep": sweep + 1,
                    "backend": "sim",
                    "rel_error": err_sim,
                    "rank": sim_model.rank,
                    "prune_applied": prune_applied_sim,
                    "sweep_time_s": sim_elapsed,
                    "cumulative_s": cumulative_sim,
                }
            )
            if not sim_only:
                rows.append(
                    {
                        "sweep": sweep + 1,
                        "backend": "ibm",
                        "rel_error": err_ibm,
                        "rank": ibm_model.rank,
                        "prune_applied": prune_applied_ibm,
                        "sweep_time_s": ibm_elapsed,
                        "cumulative_s": cumulative_ibm,
                    }
                )
            if verbose:
                if sim_only:
                    print(
                        f"[sweep] {run_label} sweep={sweep + 1:03d} "
                        f"sim_err={err_sim:.6e} sim_rank={sim_model.rank}"
                    )
                else:
                    print(
                        f"[sweep] {run_label} sweep={sweep + 1:03d} "
                        f"sim_err={err_sim:.6e} sim_rank={sim_model.rank} "
                        f"ibm_err={err_ibm:.6e} ibm_rank={ibm_model.rank}"
                    )

        if tol > 0:
            stop_sim = prev_err_sim is not None and abs(prev_err_sim - err_sim) < tol
            stop_ibm = True
            if not sim_only:
                stop_ibm = prev_err_ibm is not None and abs(prev_err_ibm - err_ibm) < tol
            if stop_sim and stop_ibm:
                break
        prev_err_sim = err_sim
        prev_err_ibm = err_ibm

    summary = {
        "sweeps": sweep + 1,
        "final_rel_error_sim": err_sim,
        "final_rel_error_ibm": err_ibm,
        "final_rank_sim": sim_model.rank,
        "final_rank_ibm": ibm_model.rank if ibm_model is not None else None,
        "prunes_applied_sim": prunes_applied["sim"],
        "prunes_applied_ibm": prunes_applied["ibm"] if not sim_only else None,
        "total_time_s_sim": cumulative_sim,
        "total_time_s_ibm": cumulative_ibm if not sim_only else None,
    }

    diagnostics = {"sim": prune_events["sim"], "ibm": prune_events["ibm"]}
    return rows, summary, diagnostics


def _pick_first_last(prune_events: List[Dict]) -> Tuple[Optional[Dict], Optional[Dict]]:
    if not prune_events:
        return None, None
    return prune_events[0], prune_events[-1]


def _extract_landscape(prune_event: Optional[Dict]) -> List[float]:
    if not prune_event:
        return []
    qaoa = prune_event.get("qaoa") or {}
    return [float(v) for v in qaoa.get("objective_values", [])]


def _extract_distribution(prune_event: Optional[Dict]) -> Dict[str, float]:
    if not prune_event:
        return {}
    qaoa = prune_event.get("qaoa") or {}
    dist = qaoa.get("distribution") or {}
    return {str(k): float(v) for k, v in dist.items()}


def _extract_exhaustive(prune_event: Optional[Dict]) -> Optional[Dict]:
    if not prune_event:
        return None
    exhaustive = prune_event.get("exhaustive")
    if not exhaustive:
        return None
    best_E = exhaustive.get("best_E_no_const")
    if best_E is None:
        return None
    return {
        "best_E_no_const": float(best_E),
        "best_mask_str": exhaustive.get("best_mask_str"),
        "distribution": exhaustive.get("distribution") or {},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation: QALR-CPD with simulator vs IBM-noisy backend."
    )
    parser.add_argument(
        "--data-path",
        action="append",
        default=None,
        help="Path to amino .pt file (repeatable). If omitted, uses --data-glob.",
    )
    parser.add_argument(
        "--data-glob",
        type=str,
        default="data/amino/*.pt",
        help="Glob for amino .pt files.",
    )
    parser.add_argument("--results-csv", type=str, default="results/v1/ablation_acid_ibm_device.csv")
    parser.add_argument(
        "--summary-csv", type=str, default="results/v1/ablation_acid_ibm_device_summary.csv"
    )
    parser.add_argument(
        "--diag-dir", type=str, default="results/v1/ablation_acid_ibm_device_diagnostics"
    )
    parser.add_argument("--rank-init", type=int, default=12)
    parser.add_argument("--t-max", type=int, default=100)
    parser.add_argument("--tol", type=float, default=0.0)
    parser.add_argument("--prune-period", type=int, default=40)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--mu", type=float, default=0.8)
    parser.add_argument(
        "--mu-scale",
        type=str,
        default="xnorm2_over_rank",
        choices=("none", "xnorm", "xnorm2", "xnorm2_over_rank"),
        help=(
            "Scale mu by tensor norm. 'xnorm2_over_rank' treats mu as a per-component "
            "penalty relative to ||X||_F^2."
        ),
    )
    parser.add_argument("--qaoa-reps", type=int, default=2)
    parser.add_argument("--qaoa-shots", type=int, default=1024)
    parser.add_argument("--qaoa-maxiter", type=int, default=20)
    parser.add_argument("--qaoa-restarts", type=int, default=1)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--exhaustive-max-r", type=int, default=24)
    parser.add_argument("--ibm-noise-p1", type=float, default=0.001)
    parser.add_argument("--ibm-noise-p2", type=float, default=0.01)
    parser.add_argument("--ibm-readout", type=float, default=0.02)
    parser.add_argument(
        "--sim-only",
        action="store_true",
        help="Run only the simulator backend (skip IBM-noisy backend).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_paths: List[Path] = []
    if args.data_path:
        data_paths = [Path(p) for p in args.data_path]
    else:
        data_paths = list(Path(".").glob(args.data_glob))

    if not data_paths:
        raise SystemExit("No data files found. Use --data-path or --data-glob.")

    dtype = _dtype_from_str(args.dtype)
    if args.seeds:
        run_seeds = _parse_list_int(args.seeds)
    else:
        run_seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    Path(args.results_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.diag_dir).mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    summaries: List[Dict] = []

    for data_path in data_paths:
        if not data_path.exists():
            print(f"[warn] dataset file missing: {data_path}")
            continue

        payload = torch.load(data_path, map_location="cpu")
        X = payload["X"].to(device=args.device, dtype=dtype)
        cfg = payload.get("config", {})
        shape = tuple(cfg.get("shape", tuple(X.shape)))
        rank_true = int(cfg.get("rank_true", 0))
        corr_rho = float(cfg.get("corr_rho", 0.0))
        noise_std = float(cfg.get("noise_std", 0.0))
        dataset_tag = _dataset_tag(shape, rank_true, corr_rho, noise_std)
        data_seed = int(payload.get("seed", 0))

        print(f"[info] dataset={data_path} seed={data_seed} tag={dataset_tag}")

        rank_init = int(args.rank_init)
        xnorm = float(torch.linalg.norm(X).clamp_min(1e-12).item())
        mu_effective = _scale_mu(args.mu, args.mu_scale, xnorm=xnorm, rank_ref=rank_init)
        if args.mu_scale != "none":
            scale_note = {
                "xnorm": f"||X||_F={xnorm:.3e}",
                "xnorm2": f"||X||_F^2={xnorm ** 2:.3e}",
                "xnorm2_over_rank": f"||X||_F^2/R={xnorm ** 2 / max(1, rank_init):.3e}",
            }.get(args.mu_scale, "")
            print(f"[info] mu_scale={args.mu_scale} ({scale_note})")

        for run_seed in run_seeds:
            run_label = f"QALR-CPD sim vs ibm seed={run_seed} rank={args.rank_init}"
            print(f"[run] {run_label}")
            run_id = {
                "dataset_tag": dataset_tag,
                "dataset_path": str(data_path),
                "data_seed": data_seed,
                "seed": run_seed,
                "rank_init": rank_init,
                "method": "QALR-CPD",
                "prune_method": "qaoa",
                "prune_mu": float(args.mu),
                "prune_mu_effective": float(mu_effective),
                "mu_scale": str(args.mu_scale),
                "prune_period": int(args.prune_period),
                "ridge": float(args.ridge),
                "qaoa_reps": int(args.qaoa_reps),
                "qaoa_shots": int(args.qaoa_shots),
                "qaoa_maxiter": int(args.qaoa_maxiter),
                "qaoa_restarts": int(args.qaoa_restarts),
                "ibm_noise_p1": float(args.ibm_noise_p1),
                "ibm_noise_p2": float(args.ibm_noise_p2),
                "ibm_readout": float(args.ibm_readout),
            }

            backend_sim = _build_backend(
                shots=args.qaoa_shots,
                seed=run_seed,
                noisy=False,
                noise_p1=args.ibm_noise_p1,
                noise_p2=args.ibm_noise_p2,
                readout=args.ibm_readout,
            )
            backend_ibm = None
            if not args.sim_only:
                backend_ibm = _build_backend(
                    shots=args.qaoa_shots,
                    seed=run_seed,
                    noisy=True,
                    noise_p1=args.ibm_noise_p1,
                    noise_p2=args.ibm_noise_p2,
                    readout=args.ibm_readout,
                )

            try:
                rows, summary, diagnostics = run_als_dual(
                    X=X,
                    shape=shape,
                    rank_init=args.rank_init,
                    ridge=args.ridge,
                    n_sweeps=args.t_max,
                    tol=args.tol,
                    seed=run_seed,
                    prune_mu=mu_effective,
                    prune_period=args.prune_period,
                    max_rank_prune=args.rank_init,
                    qaoa_reps=args.qaoa_reps,
                    qaoa_shots=args.qaoa_shots,
                    qaoa_maxiter=args.qaoa_maxiter,
                    qaoa_seed=run_seed,
                    qaoa_restarts=args.qaoa_restarts,
                    log_every=args.log_every,
                    verbose=args.verbose,
                    backend_sim=backend_sim,
                    backend_ibm=backend_ibm,
                    exhaustive_max_r=args.exhaustive_max_r,
                    run_label=run_label,
                    sim_only=bool(args.sim_only),
                )
                summary.update({"status": "ok", "error": ""})
            except Exception as exc:
                summary = {
                    "sweeps": 0,
                    "final_rel_error_sim": None,
                    "final_rel_error_ibm": None,
                    "final_rank_sim": None,
                    "final_rank_ibm": None,
                    "prunes_applied_sim": None,
                    "prunes_applied_ibm": None,
                    "total_time_s_sim": None,
                    "total_time_s_ibm": None,
                    "status": "failed",
                    "error": str(exc),
                }
                rows = []
                diagnostics = {"sim": [], "ibm": []}

            first_sim, last_sim = _pick_first_last(diagnostics.get("sim", []))
            first_ibm, last_ibm = _pick_first_last(diagnostics.get("ibm", []))

            diag_payload = {
                "run_id": run_id,
                "sim": {"first": first_sim, "last": last_sim},
                "ibm": {"first": first_ibm, "last": last_ibm},
                "qaoa_landscape": {
                    "sim_first": _extract_landscape(first_sim),
                    "sim_last": _extract_landscape(last_sim),
                    "ibm_first": _extract_landscape(first_ibm),
                    "ibm_last": _extract_landscape(last_ibm),
                },
                "qaoa_distributions": {
                    "sim_first": _extract_distribution(first_sim),
                    "sim_last": _extract_distribution(last_sim),
                    "ibm_first": _extract_distribution(first_ibm),
                    "ibm_last": _extract_distribution(last_ibm),
                },
                "exhaustive": {
                    "first": _extract_exhaustive(first_sim),
                    "last": _extract_exhaustive(last_sim),
                },
                "meta": {
                    "sim_first_sweep": first_sim.get("sweep") if first_sim else None,
                    "sim_last_sweep": last_sim.get("sweep") if last_sim else None,
                    "ibm_first_sweep": first_ibm.get("sweep") if first_ibm else None,
                    "ibm_last_sweep": last_ibm.get("sweep") if last_ibm else None,
                },
            }

            tag_safe = _sanitize_tag(dataset_tag)
            diag_path = Path(args.diag_dir) / f"ablation_ibm_{tag_safe}_seed{run_seed}_rank{args.rank_init}.json"
            with diag_path.open("w", encoding="utf-8") as f:
                json.dump(diag_payload, f, indent=2)

            summary["diag_path"] = str(diag_path)

            for r in rows:
                all_rows.append({**run_id, **r})
            summaries.append({**run_id, **summary})

            if summary.get("status") == "ok":
                if args.sim_only:
                    print(
                        "[done] "
                        f"seed={run_seed} sim_err={summary['final_rel_error_sim']:.6e} "
                        f"sim_rank={summary['final_rank_sim']}"
                    )
                else:
                    print(
                        "[done] "
                        f"seed={run_seed} sim_err={summary['final_rel_error_sim']:.6e} "
                        f"ibm_err={summary['final_rel_error_ibm']:.6e} "
                        f"sim_rank={summary['final_rank_sim']} ibm_rank={summary['final_rank_ibm']}"
                    )
            else:
                print(f"[done] seed={run_seed} status=failed error={summary.get('error')}")

    if all_rows:
        pd.DataFrame(all_rows).to_csv(args.results_csv, index=False)
        print(f"[info] wrote sweep metrics to {args.results_csv}")
    if summaries:
        pd.DataFrame(summaries).to_csv(args.summary_csv, index=False)
        print(f"[info] wrote summary metrics to {args.summary_csv}")


if __name__ == "__main__":
    main()
