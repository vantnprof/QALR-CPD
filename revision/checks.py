"""Regression checks. Run before any experiment:   python 00_check_code.py [--skip-qaoa]

 1. vectorized exact search == the submitted per-mask loop (same mask, same tie-breaking)
 2. revision BF (on G, c) == submitted bit_flipping_prune_cp (on X, factors)
 3. Proposition 2: removed-energy identity at an ALS stationary point (all masks)
 4. diagonal-only QUBO in the removal form == magnitude threshold lam_r > sqrt(mu)  (Mag-IL)
 5. keep-all regression: an ES run that keeps every component reproduces fixed-rank ALS sweep by sweep (R4.1)
 6. SA / tabu return feasible masks and reach the exact optimum on small instances
 7. QAOA: non-empty rule honoured; legacy flag reproduces the submitted cancellation (needs Qiskit + Aer)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import itertools
import sys

import numpy as np
import torch

from revision.qubo import cp_qubo_terms, energy, removal_terms
from revision.solvers import solve_qubo

OK, FAIL = "PASS", "FAIL"
results = []


def report(name, ok, detail=""):
    results.append((name, ok))
    print(f"[{OK if ok else FAIL}] {name} {detail}", flush=True)


def random_qubo(R, rng):
    A = rng.standard_normal((R + 3, R))
    G = A.T @ A
    c = rng.standard_normal(R) * 2.0
    mu = float(abs(rng.standard_normal()) * 2.0)
    return G, c, mu


def check_es(rng):
    from src.rank_prunning.exhaustive_search_rank_prunning import exhaustive_solve_qubo

    bad = ties = 0
    for t in range(60):
        R = int(rng.integers(3, 13))
        G, c, mu = random_qubo(R, rng)
        if t % 7 == 0:
            mu = 0.0
        Gt, ct = torch.tensor(G, dtype=torch.float32), torch.tensor(c, dtype=torch.float32)
        m1, e1 = exhaustive_solve_qubo(Gt, ct, mu, True, impl="loop")
        m2, e2 = exhaustive_solve_qubo(Gt, ct, mu, True, impl="vectorized")
        if m1 != m2:
            if abs(e1 - e2) <= 1e-5 * max(1.0, abs(e1)):
                ties += 1
            else:
                bad += 1
    report("1 vectorized ES == loop ES", bad == 0, f"(60 instances, {ties} float32 near-ties)")


def _random_cp(shape, R, rng, scale=1.0):
    fac = [torch.tensor(rng.standard_normal((I, R)), dtype=torch.float32) for I in shape]
    w = torch.tensor(np.abs(rng.standard_normal(R)) * scale + 0.1, dtype=torch.float32)
    return fac, w


def check_bf(rng):
    from src.rank_prunning.bit_flipping_rank_prunning import bit_flipping_prune_cp

    bad = 0
    for t in range(40):
        shape, R = (6, 5, 4), int(rng.integers(3, 9))
        X = torch.tensor(rng.standard_normal(shape), dtype=torch.float32)
        fac, w = _random_cp(shape, R, rng)
        mu = float(abs(rng.standard_normal()))
        sel, _, _, _ = bit_flipping_prune_cp(X, fac, mu=mu, weights=w, require_nonempty=True, budget=200, restarts=1,
                                             init_mode="zeros", tie_break="first", tol=1e-12, seed=t)
        _, _, G, c = cp_qubo_terms(X, fac, w)
        res = solve_qubo("BF", G.double().numpy(), c.double().numpy(), mu, seed=t)
        if sorted(sel) != [int(i) for i in np.flatnonzero(res["mask"])]:
            bad += 1
    report("2 revision BF == submitted BF", bad <= 1, f"({bad}/40 differ; float32 vs float64 near-ties allowed: 1)")


def _als_to_stationarity(X, R, seed=0, sweeps=3000):
    from src.classical_alternative_leastsquare.classical_als import CPFactorization

    torch.manual_seed(seed)
    m = CPFactorization(shape=tuple(X.shape), rank=R, ridge=0.0, init="random", seed=seed, dtype=X.dtype)
    for _ in range(sweeps):
        m.als_step(X, normalize=True)
    return m


def check_prop2_and_mag(rng):
    shape, Rt, R = (9, 8, 7), 3, 5
    fac = [rng.standard_normal((I, Rt)) for I in shape]
    X = np.einsum("ir,jr,kr->ijk", *fac) + 0.2 * rng.standard_normal(shape)
    Xt = torch.tensor(X, dtype=torch.float64)
    m = _als_to_stationarity(Xt, R)
    _, _, G, c = cp_qubo_terms(Xt, [A.detach() for A in m.factors], m.weights)
    G, c = G.numpy(), c.numpy()
    xn2 = float((Xt ** 2).sum())
    resid = float(((Xt - m.forward().detach()) ** 2).sum())
    mu = 0.3 * float(np.mean(np.diag(G)))
    worst, worst_cross = 0.0, 0.0
    for bits in itertools.product([0, 1], repeat=R):
        x = np.array(bits)
        f = energy(x, G, c, mu) + xn2
        removed, cross = removal_terms(x, G, c)
        g = resid + removed + mu * x.sum()
        worst = max(worst, abs(f - g) / max(f, 1e-12))
        worst_cross = max(worst_cross, abs(cross) / max(removed, 1e-12) if removed > 0 else 0.0)
    report("3 Proposition 2 identity at stationarity", worst < 1e-6,
           f"(max rel. mismatch {worst:.1e}, max |cross|/removed {worst_cross:.1e})")
    # 4: diagonal-only removal form == magnitude threshold
    best = min(itertools.product([0, 1], repeat=R),
               key=lambda b: sum(G[r, r] * (1 - b[r]) for r in range(R)) + mu * sum(b))
    mag = solve_qubo("Mag-IL", G, c, mu)["mask"]
    report("4 diagonal QUBO (removal form) == Mag-IL", list(best) == mag.tolist() or sum(best) == 0)


def check_keep_all():
    from revision.runner import run_one

    # A pruning call that keeps every component must leave the ALS trajectory unchanged (up to float32 rounding
    # of the re-normalized weights). TopK-IL with K = R at every call forces the keep-all path on every call;
    # ES at a negligible penalty is checked too whenever it keeps everything.
    spec = "kc=synth:sigma=0.05,rho=0.5,R=3,shape=8x8x8,seed=3"
    cfg = dict(t_max=60, prune_period=15, dtype="float32", log_sweeps=True)
    worst, n_es = 0.0, 0
    for s in range(3):
        a = run_one(dict(dataset=spec, method="ALS", mu=None, seed=s, rank_init=8, cfg=cfg))
        ea = np.array([r["rel_error"] for r in a["sweeps"]])
        runs = [run_one(dict(dataset=spec, method="TopK-IL", mu=0.0, seed=s, rank_init=8, cfg=cfg, schedule=[8, 8, 8]))]
        b = run_one(dict(dataset=spec, method="ES", mu=1e-9, seed=s, rank_init=8, cfg=cfg))
        if b["summary"]["final_rank"] == 8:
            runs.append(b)
            n_es += 1
        for rr in runs:
            eb = np.array([r["rel_error"] for r in rr["sweeps"]])
            worst = max(worst, float(np.max(np.abs(ea - eb))))
    report("5 keep-all pruning call == fixed-rank ALS, sweep by sweep", worst <= 1e-6,
           f"(max |diff| {worst:.1e}; {n_es}/3 ES runs kept all components)")


def check_sa_tabu(rng):
    hit = {"SA": 0, "Tabu": 0}
    feas = True
    from src.rank_prunning.exhaustive_search_rank_prunning import exhaustive_solve_qubo

    n = 0
    while n < 15:
        G, c, mu = random_qubo(int(rng.integers(6, 12)), rng)
        m_unc, _ = exhaustive_solve_qubo(torch.tensor(G), torch.tensor(c), mu, require_nonempty=False)
        if m_unc == 0:
            continue  # pruning-like instances: the unconstrained optimum keeps something
        t = n
        n += 1
        es = solve_qubo("ES", G, c, mu)
        for m in hit:
            r = solve_qubo(m, G, c, mu, seed=t)
            feas &= int(r["mask"].sum()) >= 1
            hit[m] += int(abs(r["energy"] - es["energy"]) <= 1e-9 * max(1, abs(es["energy"])))
    report("6 SA / tabu feasible and near-exact", feas and hit["SA"] >= 13 and hit["Tabu"] >= 13, f"(hits {hit} of 15)")


def check_qaoa():
    from src.rank_prunning.qaoa_rank_prunning import qaoa_solve_qubo

    R = 4
    G = np.eye(R) * 1.0 + 0.1
    c = np.zeros(R) + 0.05
    mu = 5.0  # every component costs more than it explains: the all-zero mask is optimal
    Gt, ct = torch.tensor(G), torch.tensor(c)
    sel_new, _, d_new = qaoa_solve_qubo(Gt, ct, mu, require_nonempty=True, qaoa_reps=1, shots=256, maxiter=10, seed=0)
    sel_old, _, _ = qaoa_solve_qubo(Gt, ct, mu, require_nonempty=False, qaoa_reps=1, shots=256, maxiter=10, seed=0)
    report("9 QAOA non-empty rule (legacy returns empty)", len(sel_new) >= 1 and len(sel_old) == 0,
           f"(new {sel_new}, legacy {sel_old}, empty_would_win={d_new['empty_would_win']})")


def check_seed_per_tensor():
    """Manifest datasets: run seed s must load the tensor whose manifest entry has seed s (as the submitted runs)."""
    import json
    import tempfile

    from revision.data import load_dataset, resolve_spec

    with tempfile.TemporaryDirectory() as tmp:
        recs = []
        for s in range(3):
            X = torch.full((4, 4, 4), float(s + 1))
            torch.save({"X": X}, Path(tmp) / f"synthetic_I4x4x4_R2_rho0.5_noise0.01_seed{s:02d}.pt")
            recs.append({"path": f"synthetic_I4x4x4_R2_rho0.5_noise0.01_seed{s:02d}.pt", "seed": s, "shape": [4, 4, 4],
                         "rank_true": 2, "noise_std": 0.01, "corr_rho": 0.5})
        (Path(tmp) / "manifest_test.json").write_text(json.dumps(recs[::-1]))
        spec = f"t=manifest:{tmp}/manifest_*.json?noise=0.01&rho=0.5&seed={{seed}}"
        got = [float(load_dataset(resolve_spec(spec, s)).X[0, 0, 0]) for s in range(3)]
        load_dataset.cache_clear()
    report("7 seed s runs on tensor s (manifest datasets)", got == [1.0, 2.0, 3.0], f"(loaded {got})")


def check_reproduction_mode():
    """Step 03b: final_prune lets ES/BF (not QAOA) prune at the last sweep; es_impl='loop' selects the original loop."""
    import os

    import src.rank_prunning.exhaustive_search_rank_prunning as esm
    from revision.runner import RunConfig, run_one
    from dataclasses import asdict

    seen = []
    orig = esm.exhaustive_solve_qubo

    def spy(*a, **k):
        seen.append(os.environ.get("QALR_ES_IMPL", "vectorized"))
        return orig(*a, **k)

    esm.exhaustive_solve_qubo = spy
    try:
        spec = "rp=synth:sigma=0.05,rho=0.5,R=3,shape=8x8x8,seed=2"
        base = asdict(RunConfig(t_max=40, prune_period=10, log_sweeps=False))
        legacy = dict(base, final_prune=True, es_impl="loop")
        n_rev = run_one(dict(dataset=spec, method="ES", mu=0.05, seed=0, rank_init=8, cfg=base))["summary"]["n_calls"]
        impl_rev = set(seen)
        seen.clear()
        n_leg = run_one(dict(dataset=spec, method="ES", mu=0.05, seed=0, rank_init=8, cfg=legacy))["summary"]["n_calls"]
        impl_leg = set(seen)
    finally:
        esm.exhaustive_solve_qubo = orig
    ok = n_rev == 3 and n_leg == 4 and impl_rev == {"vectorized"} and impl_leg == {"loop"} \
        and os.environ.get("QALR_ES_IMPL") is None
    report("8 reproduction mode (final-sweep call, original ES loop)", ok,
           f"(calls {n_rev} vs {n_leg}; impl {sorted(impl_rev)} vs {sorted(impl_leg)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-qaoa", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(12345)
    check_es(rng)
    check_bf(rng)
    check_prop2_and_mag(rng)
    check_keep_all()
    check_sa_tabu(rng)
    check_seed_per_tensor()
    check_reproduction_mode()
    if not a.skip_qaoa:
        try:
            check_qaoa()
        except ImportError as exc:
            report("9 QAOA non-empty rule", False, f"(Qiskit missing: {exc})")
    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
