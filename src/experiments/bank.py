"""Solver benchmark on banked pruning QUBOs (C3, R3.2, R4.5).

Every solver sees exactly the same QUBO instance (G, c, mu_eff) taken from an exact-search run.
Reported per (instance, solver):
  gap_pct      100 * (f - f*) / f*, with f = E + ||X||^2 the full penalized objective (>= 0); large when the
               fit is nearly exact (f* small), so the tables report its median
  gap_pct_xnorm2  100 * (f - f*) / ||X||^2  (excess objective as a share of the data energy; stable)
  gap_pct_absE    100 * (E - E*) / |E*|     (energy without the constant, possibly the convention of the
               submitted Table 8; check before comparing with the old numbers)
  hit          the returned mask attains the exact optimum
  p_read       share of independent reads/restarts that attain the optimum (SA, tabu, MS-BF)
  p_shot       QAOA: probability of the optimal bitstring in the final sampled distribution
  tts99_s      time per read * ceil(ln 0.01 / ln(1 - p_read))   (Ronnow et al., 2014)
QAOA wall-clock is statevector-simulation time and is not reported as a cost; use evals (circuit evaluations).
"""
from __future__ import annotations

import csv
import glob
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.experiments import config as C
from src.pruning.qubo import energy
from src.pruning.solvers import solve_qubo


def load_instances(bank_dir: Path) -> List[Dict]:
    inst = []
    for f in sorted(glob.glob(str(Path(bank_dir) / "bank" / "*.npz"))):
        z = np.load(f)
        meta = json.loads(bytes(z["meta"]).decode())
        for cm in meta["calls"]:
            j = cm["call"]
            if meta.get("tag") == "scaling" and j != 0:
                continue  # scaling uses the first call, where R = R_max
            inst.append({
                "id": f"{Path(f).stem}__c{j}", "dataset": meta["dataset"], "mu": meta["mu"], "seed": meta["seed"],
                "rank_init": meta["rank_init"], "tag": meta.get("tag", "main"), "call": j, "R": cm["R"],
                "mu_eff": cm["mu_eff"], "xnorm2": cm["xnorm2"], "G": z[f"G_{j}"], "c": z[f"c_{j}"],
            })
    return inst


def _tts(p: float, t_read: float) -> float:
    if p >= 1.0:
        return t_read
    if p <= 0.0:
        return float("inf")
    return t_read * math.ceil(math.log(0.01) / math.log(1.0 - p))


def _row(inst, solver, res, E_star, mask_star_int, extra=None):
    G, c, mu = inst["G"], inst["c"], inst["mu_eff"]
    f_star = E_star + inst["xnorm2"]
    E = energy(res["mask"], G, c, mu)
    tol = 1e-9 * max(1.0, abs(E_star))
    reads = res.get("reads") or []
    finite = [r for r in reads if np.isfinite(r)]
    p_read = (sum(1 for r in reads if np.isfinite(r) and abs(r - E_star) <= tol) / len(reads)) if reads else float("nan")
    n_reads = len(reads) if reads else 1
    t_read = res["time_s"] / n_reads
    row = {
        "id": inst["id"], "dataset": inst["dataset"], "tag": inst["tag"], "mu": inst["mu"], "seed": inst["seed"],
        "call": inst["call"], "R": inst["R"], "solver": solver, "energy": E, "f": E + inst["xnorm2"],
        "gap_pct": 100.0 * (E - E_star) / f_star if f_star > 0 else float("nan"),
        "gap_pct_xnorm2": 100.0 * (E - E_star) / inst["xnorm2"],
        "gap_pct_absE": 100.0 * (E - E_star) / abs(E_star) if E_star != 0 else float("nan"),
        "hit": int(abs(E - E_star) <= tol), "p_read": p_read, "n_reads": n_reads, "evals": res["evals"],
        "time_s": res["time_s"], "time_per_read_s": t_read,
        "tts99_s": _tts(p_read, t_read) if reads else float("nan"),
        "p_shot": "", "retained": int(np.sum(res["mask"])), "retained_opt": int(bin(mask_star_int).count("1")),
        "best_read_gap_pct": 100.0 * (min(finite) - E_star) / f_star if finite and f_star > 0 else float("nan"),
        "unconstrained_optimum_empty": inst.get("unc_empty", ""),
    }
    if extra:
        row.update(extra)
    return row


def _unconstrained_empty(inst: Dict) -> int:
    import torch

    from src.pruning.exhaustive import exhaustive_solve_qubo

    m, _ = exhaustive_solve_qubo(torch.tensor(inst["G"]), torch.tensor(inst["c"]), float(inst["mu_eff"]),
                                 require_nonempty=False)
    return int(m == 0)


def _classical(inst: Dict) -> List[Dict]:
    rows = []
    es = solve_qubo("ES", inst["G"], inst["c"], inst["mu_eff"])
    E_star = es["energy"]
    m_int = es["info"]["mask_int"]
    inst = dict(inst, unc_empty=_unconstrained_empty(inst))
    rows.append(_row(inst, "ES", es, E_star, m_int))
    seed = int(inst["seed"]) * 1000 + int(inst["call"])
    rows.append(_row(inst, "BF", solve_qubo("BF", inst["G"], inst["c"], inst["mu_eff"], seed=seed), E_star, m_int))
    for M in C.MSBF_RESTARTS:
        r = solve_qubo("MS-BF", inst["G"], inst["c"], inst["mu_eff"], seed=seed, restarts=M)
        rows.append(_row(inst, f"MS-BF(M={M})", r, E_star, m_int))
    rows.append(_row(inst, "SA", solve_qubo("SA", inst["G"], inst["c"], inst["mu_eff"], seed=seed,
                                            num_reads=100, num_sweeps=1000), E_star, m_int))
    rows.append(_row(inst, "Tabu", solve_qubo("Tabu", inst["G"], inst["c"], inst["mu_eff"], seed=seed,
                                              num_reads=20, timeout_ms=20), E_star, m_int))
    for b in ("Mag-IL", "Diag-x"):
        rows.append(_row(inst, b, solve_qubo(b, inst["G"], inst["c"], inst["mu_eff"]), E_star, m_int))
    return rows


def _qaoa(inst: Dict, shots: int, maxiter: int) -> List[Dict]:
    es = solve_qubo("ES", inst["G"], inst["c"], inst["mu_eff"])
    inst = dict(inst, unc_empty=_unconstrained_empty(inst))
    m_int = es["info"]["mask_int"]
    r = solve_qubo("QAOA", inst["G"], inst["c"], inst["mu_eff"], seed=int(inst["seed"]), shots=shots, maxiter=maxiter)
    p_shot = float(r["info"]["distribution"].get(m_int, 0.0))
    row = _row(inst, "QAOA", r, es["energy"], m_int)
    row.update({"p_shot": p_shot, "p_read": float("nan"), "tts99_s": float("nan")})
    return [row]


def _write(path: Path, rows: List[Dict]):
    """Append rows; classical and QAOA rows have different columns, so align them on the union header."""
    from src.experiments.runner import _append_csv

    _append_csv(path, rows)


def solve_bank(bank_dir: Path, args) -> None:
    out = Path(bank_dir) / "bank_results.csv"
    done = set()
    if out.exists():
        with open(out, newline="") as f:
            done = {(r["id"], r["solver"]) for r in csv.DictReader(f)}
    inst = load_instances(bank_dir)
    if not inst:
        raise SystemExit(f"no banked QUBOs under {bank_dir}/bank; run the 'bank' stage first")
    todo = [i for i in inst if (i["id"], "ES") not in done]
    print(f"[solvebank] {len(inst)} instances, {len(todo)} to solve classically", flush=True)
    t0 = time.time()
    if args.jobs <= 1:
        for k, i in enumerate(todo):
            _write(out, _classical(i))
            if (k + 1) % 20 == 0:
                print(f"  {k + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(_classical, i) for i in todo]
            for k, f in enumerate(as_completed(futs)):
                _write(out, f.result())
                if (k + 1) % 20 == 0:
                    print(f"  {k + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    if getattr(args, "with_qaoa", False):
        # Deterministic, stratified subset: qaoa_per_r instances for each scaling size R <= qaoa_max_r,
        # then the first qaoa_main instances of the main bank (first pruning call, R = R_max).
        pool = []
        for R in sorted({i["R"] for i in inst if i["tag"] == "scaling" and i["R"] <= int(args.qaoa_max_r)}):
            cand = sorted([i for i in inst if i["tag"] == "scaling" and i["R"] == R], key=lambda i: (i["seed"], i["dataset"]))
            pool += cand[: int(args.qaoa_per_r)]
        main_c = sorted([i for i in inst if i["tag"] != "scaling" and i["call"] == 0],
                        key=lambda i: (i["seed"], i["dataset"], i["mu"]))
        pool += main_c[: int(args.qaoa_main)]
        pool = [i for i in pool if (i["id"], "QAOA") not in done]
        print(f"[solvebank] QAOA on {len(pool)} instances", flush=True)
        shots = 512 if args.quick else args.qaoa_shots
        maxiter = 20 if args.quick else args.qaoa_maxiter
        for k, i in enumerate(pool):
            _write(out, _qaoa(i, shots, maxiter))
            print(f"  QAOA {k + 1}/{len(pool)} R={i['R']} {time.time() - t0:.0f}s", flush=True)
