"""One adaptive-CP run, replicating the loop of script/exp/*/..._experiment.py::run_als exactly.

Loop (unchanged from the submitted experiments):
  for sweep in range(t_max):
      model.als_step(X, normalize=True)
      prune if (not last sweep) and (sweep+1) % prune_period == 0          [rank-adaptive methods]
      top-K truncation at the last sweep                                    [post-hoc Top-K]
      error  = ||Omega*(X - Xhat)|| / ||Omega*X||
      stop if tol > 0 and |err - prev| < tol                                [not for Top-K]
  optional Top-K refit sweeps

Methods
  ALS                fixed rank (rank_init)
  TopK-PH            post-hoc truncation to K at the last sweep, no refit
  TopK-PH+refit      the same followed by `topk_refit` ALS sweeps (the Amino setting of the submission)
  ES, BF, QAOA       the submitted pruning functions, called with the submitted arguments
  QAOA-legacy        QAOA with the submitted empty-mask behaviour (cancelled call), for comparison only
  MS-BF, SA, Tabu    stronger classical solvers on the same QUBO (revision C3)
  Mag-IL             in-loop magnitude threshold lam_r > sqrt(mu_eff) = diagonal-only QUBO, removal form (C2)
  Diag-x             diagonal-only QUBO, z form (C2)
  TopK-IL            in-loop Top-K with K matched, call by call, to the ES run of the same seed and mu (C2)
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from revision.data import PAPER_SETTINGS, load_dataset, mu_effective, resolve_spec
from revision.qubo import cp_qubo_terms, removal_terms
from revision.solvers import solve_qubo

LEGACY = {"ES": "exhaustive", "BF": "bitflip", "QAOA": "qaoa", "QAOA-legacy": "qaoa"}
NEW_SOLVERS = {"MS-BF", "SA", "Tabu", "Mag-IL", "Diag-x", "TopK-IL"}
TOPK_PH = {"TopK-PH", "TopK-PH+refit"}
ALL_METHODS = ["ALS", *sorted(TOPK_PH), *LEGACY, *sorted(NEW_SOLVERS)]


@dataclass
class RunConfig:
    t_max: Optional[int] = None          # None -> paper setting for the dataset kind
    prune_period: Optional[int] = None
    ridge: Optional[float] = None
    tol: Optional[float] = None
    topk_refit: int = 50                 # used by TopK-PH+refit only
    rank_ref: str = "init"               # mu_eff = mu ||X||^2 / R with R = rank_init ("init") or an integer
    dtype: str = "float32"
    qaoa_reps: int = 2
    qaoa_shots: int = 4096
    qaoa_maxiter: int = 200
    qaoa_restarts: int = 1
    bf_budget: int = 200
    msbf_restarts: int = 32
    sa_reads: int = 100
    sa_sweeps: int = 1000
    tabu_reads: int = 20
    tabu_timeout_ms: int = 20
    dump_qubo: bool = False
    # Reproduction only (step 03b): the submitted synthetic scripts let ES/BF prune at the final sweep (QAOA did not);
    # es_impl='loop' uses the original per-mask exhaustive-search loop. The revision itself uses neither.
    final_prune: bool = False
    es_impl: str = ""
    log_sweeps: bool = True

    def resolved(self, kind: str) -> Dict:
        d = asdict(self)
        base = PAPER_SETTINGS.get(kind, PAPER_SETTINGS["synthetic"])
        for k_cfg, k_base in (("t_max", "t_max"), ("prune_period", "prune_period"), ("ridge", "ridge"), ("tol", "tol")):
            if d[k_cfg] is None:
                d[k_cfg] = base[k_base]
        return d


def component_recovery(gt, est) -> Optional[float]:
    """Same metric as the submitted scripts (Hungarian matching of geometric-mean |cos|)."""
    if gt is None or est is None or len(gt) == 0:
        return None
    from scipy.optimize import linear_sum_assignment

    sims = torch.ones((gt[0].shape[1], est[0].shape[1]), dtype=torch.float64)
    for A, B in zip(gt, est):
        A = A.double() / A.double().norm(dim=0, keepdim=True).clamp_min(1e-12)
        B = B.double() / B.double().norm(dim=0, keepdim=True).clamp_min(1e-12)
        sims *= (A.T @ B).abs()
    sims = sims.clamp_min(0.0) ** (1.0 / len(gt))
    r, c = linear_sum_assignment(-sims.numpy())
    return float(sims.numpy()[r, c].sum() / gt[0].shape[1])


def _legacy_prune(method, model, X, mu_eff, seed, rank_init, cfg):
    from src.rank_prunning.bit_flipping_rank_prunning import bit_flipping_prune_cp
    from src.rank_prunning.exhaustive_search_rank_prunning import exhaustive_prune_cp

    factors = [A.detach() for A in model.factors]
    if method == "ES":
        sel, E, (lam, U), diag = exhaustive_prune_cp(
            X, factors, mu=float(mu_eff), weights=model.weights, require_nonempty=True,
            max_R_exhaustive=int(rank_init), qubo_scale=1.0)
    elif method == "BF":
        sel, E, (lam, U), diag = bit_flipping_prune_cp(
            X, factors, mu=float(mu_eff), weights=model.weights, require_nonempty=True, max_R_bitflip=None,
            budget=cfg["bf_budget"], restarts=1, init_mode="zeros", tie_break="first", tol=1e-12, seed=int(seed),
            qubo_scale=1.0)
    else:
        from src.rank_prunning.qaoa_rank_prunning import qaoa_prune_cp

        sel, E, (lam, U), diag = qaoa_prune_cp(
            X, factors, mu=float(mu_eff), weights=model.weights,
            require_nonempty=(method == "QAOA"), max_R_qaoa=int(rank_init), qaoa_reps=cfg["qaoa_reps"],
            shots=cfg["qaoa_shots"], maxiter=cfg["qaoa_maxiter"], seed=int(seed), num_restarts=cfg["qaoa_restarts"],
            qubo_scale=1.0)
    return sel, lam, U, diag


QAOA_TIMES = ("n_estimator_evals", "t_transpile_s", "t_optimize_s", "t_sample_s", "t_select_s")


def run_one(task: Dict) -> Dict:
    """task: dataset (spec), method, mu, seed, rank_init, K (Top-K), schedule (TopK-IL), cfg (RunConfig dict)."""
    impl = (task.get("cfg") or {}).get("es_impl") or ""
    old = os.environ.get("QALR_ES_IMPL")
    if impl:
        os.environ["QALR_ES_IMPL"] = impl
    try:
        return _run_one(task)
    finally:
        if impl:
            if old is None:
                os.environ.pop("QALR_ES_IMPL", None)
            else:
                os.environ["QALR_ES_IMPL"] = old


def _run_one(task: Dict) -> Dict:
    torch.set_num_threads(int(os.environ.get("QALR_TORCH_THREADS", "1")))
    from src.classical_alternative_leastsquare.classical_als import CPFactorization

    t_start = time.time()
    ds = load_dataset(resolve_spec(task["dataset"], task["seed"]), task["cfg"].get("dtype", "float32"))
    cfg = RunConfig(**task["cfg"]).resolved(ds.kind)
    method, seed, rank_init = task["method"], int(task["seed"]), int(task["rank_init"])
    mu = task.get("mu")
    X, Omega = ds.X, ds.Omega
    xnorm = ds.xnorm
    rank_ref = rank_init if cfg["rank_ref"] == "init" else int(cfg["rank_ref"])
    mu_eff = mu_effective(mu, xnorm, rank_ref) if mu is not None else None
    xnorm2_zf = float(torch.linalg.norm(X).item()) ** 2  # constant of the QUBO (zero-filled X)

    torch.manual_seed(seed)
    model = CPFactorization(shape=ds.shape, rank=rank_init, ridge=cfg["ridge"], init="random", seed=seed,
                            device=X.device, dtype=X.dtype)
    sweeps, calls, dumps = [], [], []
    prev, err, solver_time = None, None, 0.0
    t_max, period = int(cfg["t_max"]), int(cfg["prune_period"])
    is_topk = method in TOPK_PH
    prunes_applied = n_keep_all = n_cancel = n_fallback = n_empty_win = 0
    sweep = -1
    for sweep in range(t_max):
        model.als_step(X, normalize=True)
        last = sweep == t_max - 1
        applied = False
        final_ok = bool(cfg.get("final_prune")) and method not in ("QAOA", "QAOA-legacy")
        if method != "ALS" and not is_topk and (not last or final_ok) and (sweep + 1) % period == 0 and model.rank > 0:
            R_before = model.rank
            t_q = time.perf_counter()
            lam_l, U_l, G_t, c_t = cp_qubo_terms(X, [A.detach() for A in model.factors], model.weights)
            G = G_t.double().cpu().numpy()
            c = c_t.double().cpu().numpy()
            qubo_build_s = time.perf_counter() - t_q
            t0 = time.perf_counter()
            info: Dict = {}
            try:
                if method in LEGACY:
                    sel, lam, U, diag = _legacy_prune(method, model, X, mu_eff, seed, rank_init, cfg)
                    info = {k: v for k, v in diag.items() if k in ("empty_would_win", "fallback") + QAOA_TIMES}
                else:
                    kw = dict(seed=seed)
                    if method == "MS-BF":
                        kw.update(restarts=cfg["msbf_restarts"], budget=cfg["bf_budget"])
                    elif method == "SA":
                        kw.update(num_reads=cfg["sa_reads"], num_sweeps=cfg["sa_sweeps"])
                    elif method == "Tabu":
                        kw.update(num_reads=cfg["tabu_reads"], timeout_ms=cfg["tabu_timeout_ms"])
                    elif method == "TopK-IL":
                        sched = task.get("schedule") or []
                        j = len(calls)
                        kw.update(rank_target=sched[min(j, len(sched) - 1)] if sched else R_before)
                    res = solve_qubo(method, G, c, mu_eff, **kw)
                    sel = [int(i) for i in np.flatnonzero(res["mask"])]
                    lam, U = lam_l, U_l
                    info = {k: v for k, v in res["info"].items() if k in ("fallback", "K", "empty_would_win") + QAOA_TIMES}
            except ValueError as exc:  # same behaviour as _maybe_prune_components: skip the call
                sel, lam, U, info = None, None, None, {"skipped": str(exc)}
            dt = time.perf_counter() - t0
            solver_time += dt
            if sel is None:
                outcome = "skipped"
            elif len(sel) == 0:
                outcome = "cancelled"
                n_cancel += 1
            elif len(sel) < R_before:
                model._apply_pruning_selection(lam, U, sel)
                outcome = "pruned"
                prunes_applied += 1
                applied = True
            else:
                model._set_weights_tensor(lam)
                outcome = "keep_all"
                n_keep_all += 1
            if info.get("empty_would_win"):
                n_empty_win += 1
            if info.get("fallback") not in (None, "none", False):
                n_fallback += 1
            mask = np.zeros(R_before, dtype=np.int64)
            if sel:
                mask[sel] = 1
            removed, cross = removal_terms(mask, G, c) if sel else (float("nan"), float("nan"))
            calls.append({
                "call": len(calls), "sweep": sweep + 1, "R_before": R_before, "R_after": model.rank,
                "outcome": outcome, "energy": float(mask @ G @ mask - 2 * mask @ c + mu_eff * mask.sum()) if sel else float("nan"),
                "removed_energy": removed, "cross_term": cross,
                "cross_ratio": abs(cross) / removed if sel and removed > 0 else float("nan"),
                "empty_would_win": bool(info.get("empty_would_win", False)), "fallback": str(info.get("fallback", "")),
                "K": info.get("K", ""), "solver_time_s": dt, "qubo_build_s": qubo_build_s,
                **{k: info.get(k, "") for k in QAOA_TIMES},
            })
            if cfg["dump_qubo"]:
                dumps.append({"G": G, "c": c, "lam": lam_l.double().cpu().numpy(), "mu_eff": mu_eff,
                              "xnorm2": xnorm2_zf, "call": len(calls) - 1, "sweep": sweep + 1, "R": R_before})
        if is_topk and last:
            from src.rank_prunning.threshold_rank_prunning import threshold_prune_cp

            sel, _, (lam, U), _ = threshold_prune_cp([A.detach() for A in model.factors], weights=model.weights,
                                                     weight_threshold=None, top_k=int(task["K"]), require_nonempty=True)
            if len(sel) < model.rank:
                model._apply_pruning_selection(lam, U, sel)
                prunes_applied += 1
                applied = True
            else:
                model._set_weights_tensor(lam)
        resid = X - model.forward()
        if Omega is not None:
            resid = Omega * resid
        err = float((torch.linalg.norm(resid) / xnorm).item())
        if cfg["log_sweeps"]:
            sweeps.append({"sweep": sweep + 1, "rel_error": err, "rank": model.rank, "prune_applied": applied})
        if (not is_topk) and cfg["tol"] > 0 and prev is not None and abs(prev - err) < cfg["tol"]:
            break
        prev = err
    n_sweeps = sweep + 1
    if method == "TopK-PH+refit" and cfg["topk_refit"] > 0:
        for _ in range(int(cfg["topk_refit"])):
            model.als_step(X, normalize=True)
            resid = X - model.forward()
            if Omega is not None:
                resid = Omega * resid
            err = float((torch.linalg.norm(resid) / xnorm).item())
            n_sweeps += 1
            if cfg["log_sweeps"]:
                sweeps.append({"sweep": n_sweeps, "rel_error": err, "rank": model.rank, "prune_applied": False})

    meta = ds.meta
    floor = meta.get("floor")
    rank_true = meta.get("rank_true")
    summary = {
        "dataset": task["dataset"].split("=", 1)[0], "method": method, "mu": mu, "mu_eff": mu_eff, "seed": seed,
        "rank_init": rank_init, "K": task.get("K", ""), "sweeps": n_sweeps, "final_rel_error": err,
        "final_rank": model.rank, "prunes_applied": prunes_applied, "n_calls": len(calls), "n_keep_all": n_keep_all,
        "n_cancelled": n_cancel, "n_fallback": n_fallback, "n_empty_would_win": n_empty_win,
        "solver_time_s": solver_time, "total_time_s": time.time() - t_start,
        "rank_true": rank_true if rank_true is not None else "", "floor": floor if floor is not None else "",
        "excess_over_floor": (err - floor) / floor if floor else "",
        "exact_rank": int(model.rank == rank_true) if rank_true is not None else "",
        "snr_db": meta.get("snr_db", ""), "rho": meta.get("rho", ""),
        "final_rec": component_recovery(ds.gt_factors, [A.detach() for A in model.factors]) if ds.gt_factors else "",
        "tag": task.get("tag", ""),
    }
    return {"summary": summary, "sweeps": sweeps, "calls": calls, "dumps": dumps}


# ----------------------------------------------------------------------------- batch execution
def task_key(t: Dict) -> str:
    return "|".join(str(x) for x in (t["dataset"].split("=", 1)[0], t["method"], t.get("mu"), t["seed"],
                                     t["rank_init"], t.get("K", ""), t.get("tag", "")))


def _done_keys(runs_csv: Path) -> set:
    if not runs_csv.exists():
        return set()
    keys = set()
    with open(runs_csv, newline="") as f:
        for r in csv.DictReader(f):
            mu = r["mu"]
            mu = None if mu in ("", "None") else float(mu)
            keys.add("|".join(str(x) for x in (r["dataset"], r["method"], mu, int(r["seed"]), int(r["rank_init"]),
                                               r["K"], r.get("tag", ""))))
    return keys


def _append_csv(path: Path, rows: List[Dict], compress: bool = False) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    if not path.exists():
        opener = (lambda p: gzip.open(p, "wt", newline="")) if compress else (lambda p: open(p, "w", newline=""))
        with opener(path) as f:
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            w.writeheader()
            w.writerows(rows)
        return
    reader = (lambda p: gzip.open(p, "rt", newline="")) if compress else (lambda p: open(p, newline=""))
    with reader(path) as f:
        header = next(csv.reader(f), [])
    if not set(keys) <= set(header):
        # A newer version of the code logs extra columns: rewrite the file once with the union of the columns,
        # so old and new rows stay aligned (old rows get empty values in the new columns).
        with reader(path) as f:
            old = list(csv.DictReader(f))
        header = header + [k for k in keys if k not in header]
        opener = (lambda p: gzip.open(p, "wt", newline="")) if compress else (lambda p: open(p, "w", newline=""))
        with opener(path) as f:
            w = csv.DictWriter(f, fieldnames=header, restval="")
            w.writeheader()
            w.writerows(old)
    opener = (lambda p: gzip.open(p, "at", newline="")) if compress else (lambda p: open(p, "a", newline=""))
    with opener(path) as f:
        w = csv.DictWriter(f, fieldnames=header, restval="")
        w.writerows(rows)


def _save_dumps(bank_dir: Path, summary: Dict, dumps: List[Dict]) -> None:
    if not dumps:
        return
    bank_dir.mkdir(parents=True, exist_ok=True)
    name = (f"{summary['dataset']}__{summary['method']}__mu{summary['mu']}__R{summary['rank_init']}"
            f"__s{summary['seed']}__{summary.get('tag') or 'main'}.npz")
    arrays = {}
    for d in dumps:
        j = d["call"]
        arrays[f"G_{j}"], arrays[f"c_{j}"], arrays[f"lam_{j}"] = d["G"], d["c"], d["lam"]
    meta = {k: summary[k] for k in ("dataset", "method", "mu", "seed", "rank_init", "tag")}
    meta["calls"] = [{k: d[k] for k in ("call", "sweep", "R", "mu_eff", "xnorm2")} for d in dumps]
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(bank_dir / name, **arrays)


def run_tasks(tasks: Iterable[Dict], out_dir: Path, jobs: int = 1, resume: bool = True) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = out_dir / "runs.csv"
    done = _done_keys(runs_csv) if resume else set()
    todo = [t for t in tasks if task_key(t) not in done]
    print(f"[{out_dir.name}] {len(todo)} runs to do ({len(done)} already done)", flush=True)
    if not todo:
        return runs_csv

    def handle(t, res):
        s = res["summary"]
        _append_csv(runs_csv, [s])
        key = {k: s[k] for k in ("dataset", "method", "mu", "seed", "rank_init", "K", "tag")}
        _append_csv(out_dir / "calls.csv", [{**key, **c} for c in res["calls"]])
        _append_csv(out_dir / "sweeps.csv.gz", [{**key, **r} for r in res["sweeps"]], compress=True)
        _save_dumps(out_dir / "bank", s, res["dumps"])

    t0, n = time.time(), 0
    if jobs <= 1:
        for t in todo:
            try:
                handle(t, run_one(t))
            except Exception:
                print(f"[error] {task_key(t)}\n{traceback.format_exc()}", flush=True)
            n += 1
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} done, {time.time() - t0:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(run_one, t): t for t in todo}
            for f in as_completed(futs):
                t = futs[f]
                try:
                    handle(t, f.result())
                except Exception:
                    print(f"[error] {task_key(t)}\n{traceback.format_exc()}", flush=True)
                n += 1
                if n % 10 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)} done, {time.time() - t0:.0f}s", flush=True)
    return runs_csv


def write_env(out_dir: Path) -> None:
    """Software versions for the parameter table (R5.4)."""
    import platform
    import sys

    vers = {"python": sys.version.split()[0], "platform": platform.platform()}
    cpu = platform.processor() or ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    if not cpu and sys.platform == "darwin":
        try:
            import subprocess
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        except Exception:
            pass
    vers["cpu"] = cpu or "unknown"
    vers["logical_cores"] = os.cpu_count()
    for mod in ("torch", "numpy", "scipy", "pandas", "qiskit", "qiskit_aer", "qiskit_ibm_runtime", "dwave.samplers"):
        try:
            m = __import__(mod, fromlist=["__version__"])
            vers[mod] = getattr(m, "__version__", "?")
        except Exception:
            vers[mod] = "not installed"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "environment.json").write_text(json.dumps(vers, indent=2))
