"""Mask-update solvers for the CP-pruning QUBO, all under one non-empty rule (1^T z >= 1).

Every solver receives (G, c, mu) for E(z) = z^T G z - 2 z^T c + mu 1^T z and returns a dict:
  mask      np.ndarray[int]  (1 = keep)
  energy    float            E(mask)
  evals     int              objective evaluations / reads / circuit evaluations
  time_s    float            wall-clock of the solver call
  reads     list[float]      per-read (or per-restart) energies, for P(opt) and TTS
  info      dict             solver-specific diagnostics

Solvers
  ES      exact enumeration (src.pruning.exhaustive.exhaustive_solve_qubo)
  BF      single-start 1-flip local search, identical to the submitted bit-flipping
  MS-BF   BF with M random restarts
  SA      simulated annealing   (dwave-samplers), best nonzero read polished by constrained 1-flip descent
  Tabu    tabu search           (dwave-samplers), same post-processing
          (both solve the unconstrained QUBO; when its optimum is the empty mask every read is a miss, and the
           result comes from the non-empty fallback + polish; the bank flags such instances)
  QAOA    src.pruning.qaoa.qaoa_solve_qubo (Aer)
  Mag-IL  keep r iff lam_r^2 = G_rr > mu   (diagonal-only QUBO in the removal form, Prop. 2)
  Diag-x  keep r iff G_rr - 2 c_r + mu < 0 (diagonal-only QUBO in the z form)
  TopK    keep the K largest lam_r (K given)
"""
from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from src.pruning.qubo import energies_batch, energy


def _nonempty_fallback(G: np.ndarray, c: np.ndarray, mu: float) -> np.ndarray:
    d = np.diag(G) - 2.0 * c + mu
    z = np.zeros(G.shape[0], dtype=np.int64)
    z[int(np.argmin(d))] = 1
    return z


def _scale_ref(G: np.ndarray, c: np.ndarray, mu: float) -> float:
    Q = np.triu(G, 1) * 2.0
    np.fill_diagonal(Q, np.diag(G))
    lin = mu * np.ones(G.shape[0]) - 2.0 * c
    return float(max(np.max(np.abs(Q)), np.max(np.abs(lin)), 1.0))


def _qubo_dict(G: np.ndarray, c: np.ndarray, mu: float, scale: float) -> Dict:
    R = G.shape[0]
    Q = {}
    for i in range(R):
        Q[(i, i)] = (G[i, i] - 2.0 * c[i] + mu) / scale
        for j in range(i + 1, R):
            Q[(i, j)] = 2.0 * G[i, j] / scale
    return Q


def _pack(mask, G, c, mu, t0, evals, reads=None, info=None) -> Dict:
    mask = np.asarray(mask, dtype=np.int64)
    return {
        "mask": mask,
        "energy": energy(mask, G, c, mu),
        "evals": int(evals),
        "time_s": time.perf_counter() - t0,
        "reads": list(reads) if reads is not None else [],
        "info": info or {},
    }


# --------------------------------------------------------------------------- exact
def solve_es(G, c, mu, **_):
    from src.pruning.exhaustive import exhaustive_solve_qubo

    t0 = time.perf_counter()
    R = G.shape[0]
    best_mask, _ = exhaustive_solve_qubo(
        torch.as_tensor(G, dtype=torch.float64), torch.as_tensor(c, dtype=torch.float64), float(mu), True
    )
    z = np.array([(best_mask >> i) & 1 for i in range(R)], dtype=np.int64)
    return _pack(z, G, c, mu, t0, evals=(1 << R) - 1, info={"mask_int": int(best_mask)})


# --------------------------------------------------------------------------- local search
def _bitflip_one(G, lin, diag_G, z, budget, tol, tie_break, rng, best_single_idx):
    R = G.shape[0]
    if z.sum() < 0.5:
        z[best_single_idx] = 1.0
    g = G @ z
    E = z @ g + lin @ z
    steps = 0
    for _ in range(budget):
        s = 1.0 - 2.0 * z
        delta = s * (2.0 * g + lin) + diag_G
        if z.sum() <= 1.0:
            one = np.flatnonzero(z > 0.5)
            if one.size == 1:
                delta[one[0]] = np.inf
        min_delta = np.min(delta)
        steps += 1
        if min_delta >= -tol:
            break
        if tie_break == "random_among_best":
            cand = np.flatnonzero(delta <= min_delta + 1e-12).tolist()
            i_star = int(rng.choice(cand))
        else:
            i_star = int(np.argmin(delta))
        s_i = 1.0 - 2.0 * z[i_star]
        z[i_star] = 1.0 - z[i_star]
        E = E + delta[i_star]
        g = g + s_i * G[:, i_star]
    if z.sum() < 0.5:
        z[:] = 0.0
        z[best_single_idx] = 1.0
    return z, steps


def solve_bf(G, c, mu, seed=0, budget=200, restarts=1, init_mode="zeros", tie_break="first", tol=1e-12, **_):
    """Same algorithm as src.pruning.bitflip (normalized QUBO, absolute tol)."""
    t0 = time.perf_counter()
    R = G.shape[0]
    s = _scale_ref(G, c, mu)
    Gs, cs, mus = G / s, c / s, mu / s
    lin = mus * np.ones(R) - 2.0 * cs
    diag_G = np.diag(Gs).copy()
    best_single_idx = int(np.argmin(diag_G + lin))
    rng = random.Random(int(seed))
    best_z, best_E, reads, evals = None, math.inf, [], 0
    for r_idx in range(max(1, int(restarts))):
        if r_idx > 0:
            rng.random()
        if init_mode == "random":
            z = np.array([rng.randint(0, 1) for _ in range(R)], dtype=np.float64)
        else:
            z = np.zeros(R, dtype=np.float64)
        z, steps = _bitflip_one(Gs, lin, diag_G, z, int(budget), float(tol), tie_break, rng, best_single_idx)
        evals += steps * R
        E = energy(z, G, c, mu)
        reads.append(E)
        if E < best_E:
            best_E, best_z = E, z.copy()
    return _pack(best_z, G, c, mu, t0, evals, reads, {"restarts": int(restarts), "init_mode": init_mode})


def solve_msbf(G, c, mu, seed=0, restarts=32, budget=200, **_):
    return solve_bf(G, c, mu, seed=seed, budget=budget, restarts=restarts, init_mode="random",
                    tie_break="random_among_best")


# --------------------------------------------------------------------------- dwave samplers
def _polish(z, G, c, mu):
    """Constrained 1-flip descent (the BF move set with the non-empty rule) started from z."""
    R = G.shape[0]
    s = _scale_ref(G, c, mu)
    Gs, cs, mus = G / s, c / s, mu / s
    lin = mus * np.ones(R) - 2.0 * cs
    diag_G = np.diag(Gs).copy()
    zz, _ = _bitflip_one(Gs, lin, diag_G, np.asarray(z, dtype=np.float64).copy(), 10 * R, 1e-12, "first",
                         random.Random(0), int(np.argmin(diag_G + lin)))
    return zz.astype(np.int64)


def _best_feasible(samples: np.ndarray, G, c, mu):
    """Best nonzero read, then polished under the non-empty rule. Reads that are all-zero count as misses.
    Returns (mask, per-read energies, fallback_used, polish_improved)."""
    E = energies_batch(samples, G, c, mu)
    feas = samples.sum(axis=1) > 0
    reads = [float(e) if f else math.inf for e, f in zip(E, feas)]
    if feas.any():
        z0 = samples[int(np.argmin(np.where(feas, E, np.inf)))].astype(np.int64)
        fb = False
    else:
        z0 = _nonempty_fallback(G, c, mu)
        fb = True
    z1 = _polish(z0, G, c, mu)
    improved = energy(z1, G, c, mu) < energy(z0, G, c, mu) - 1e-12 * max(1.0, abs(energy(z0, G, c, mu)))
    return (z1 if improved else z0), reads, fb, bool(improved)


def solve_sa(G, c, mu, seed=0, num_reads=100, num_sweeps=1000, **_):
    from dwave.samplers import SimulatedAnnealingSampler

    t0 = time.perf_counter()
    s = _scale_ref(G, c, mu)
    ss = SimulatedAnnealingSampler().sample_qubo(
        _qubo_dict(G, c, mu, s), num_reads=int(num_reads), num_sweeps=int(num_sweeps), seed=int(seed) % (2**32)
    )
    R = G.shape[0]
    samples = np.array([[rec[i] for i in range(R)] for rec in ss.samples()], dtype=np.int64)
    z, reads, fb, pol = _best_feasible(samples, G, c, mu)
    return _pack(z, G, c, mu, t0, evals=int(num_reads) * int(num_sweeps) * R, reads=reads,
                 info={"num_reads": int(num_reads), "num_sweeps": int(num_sweeps), "fallback": fb, "polished": pol})


def solve_tabu(G, c, mu, seed=0, num_reads=20, timeout_ms=20, **_):
    from dwave.samplers import TabuSampler

    t0 = time.perf_counter()
    s = _scale_ref(G, c, mu)
    ss = TabuSampler().sample_qubo(
        _qubo_dict(G, c, mu, s), num_reads=int(num_reads), timeout=int(timeout_ms), seed=int(seed) % (2**32)
    )
    R = G.shape[0]
    samples = np.array([[rec[i] for i in range(R)] for rec in ss.samples()], dtype=np.int64)
    z, reads, fb, pol = _best_feasible(samples, G, c, mu)
    return _pack(z, G, c, mu, t0, evals=int(num_reads), reads=reads,
                 info={"num_reads": int(num_reads), "timeout_ms": int(timeout_ms), "fallback": fb, "polished": pol})


# --------------------------------------------------------------------------- quantum
def solve_qaoa(G, c, mu, seed=0, reps=2, shots=4096, maxiter=200, restarts=1, return_distribution=True, **_):
    from src.pruning.qaoa import qaoa_solve_qubo

    t0 = time.perf_counter()
    R = G.shape[0]
    sel, _, diag = qaoa_solve_qubo(
        torch.as_tensor(G, dtype=torch.float64), torch.as_tensor(c, dtype=torch.float64), float(mu),
        require_nonempty=True, qaoa_reps=int(reps), shots=int(shots), maxiter=int(maxiter), seed=int(seed),
        num_restarts=int(restarts), return_distribution=return_distribution,
    )
    z = np.zeros(R, dtype=np.int64)
    z[sel] = 1
    dist = diag.pop("final_distribution", {}) if return_distribution else {}
    info = dict(diag)
    info["distribution"] = {int(k): float(v) for k, v in dist.items()}
    return _pack(z, G, c, mu, t0, evals=diag.get("n_estimator_evals", 0), info=info)


# --------------------------------------------------------------------------- baselines
def solve_mag(G, c, mu, **_):
    t0 = time.perf_counter()
    lam2 = np.diag(G)
    z = (lam2 > mu).astype(np.int64)
    if z.sum() == 0:
        z[int(np.argmax(lam2))] = 1
    return _pack(z, G, c, mu, t0, evals=G.shape[0])


def solve_diagx(G, c, mu, **_):
    t0 = time.perf_counter()
    d = np.diag(G) - 2.0 * c + mu
    z = (d < 0).astype(np.int64)
    if z.sum() == 0:
        z[int(np.argmin(d))] = 1
    return _pack(z, G, c, mu, t0, evals=G.shape[0])


def solve_topk(G, c, mu, rank_target=None, **_):
    t0 = time.perf_counter()
    R = G.shape[0]
    K = R if rank_target is None else max(1, min(int(rank_target), R))
    lam = np.sqrt(np.clip(np.diag(G), 0.0, None))
    order = sorted(range(R), key=lambda i: (-lam[i], i))
    z = np.zeros(R, dtype=np.int64)
    z[order[:K]] = 1
    return _pack(z, G, c, mu, t0, evals=R, info={"K": K})


SOLVERS = {
    "ES": solve_es,
    "BF": solve_bf,
    "MS-BF": solve_msbf,
    "SA": solve_sa,
    "Tabu": solve_tabu,
    "QAOA": solve_qaoa,
    "Mag-IL": solve_mag,
    "Diag-x": solve_diagx,
    "TopK-IL": solve_topk,
}


def solve_qubo(method: str, G: np.ndarray, c: np.ndarray, mu: float, **kw) -> Dict:
    if method not in SOLVERS:
        raise ValueError(f"unknown solver {method}; choose from {sorted(SOLVERS)}")
    G = np.asarray(G, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    return SOLVERS[method](G, c, float(mu), **kw)
