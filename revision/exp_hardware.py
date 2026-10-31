"""Repeated QAOA runs on IBM hardware for one banked QUBO, with full provenance (R3.2, R5.5).

    python 13_ibm_hardware_repeats.py --instance revision_out/bank/bank/<file>.npz --call 0 \
        --backend ibm_torino --repeats 5 --protocol sampled --execution-mode job

Protocols
  sampled  (default, cheap)  angles are optimized on the noiseless Aer simulator with the settings of the paper
           (p, shots, COBYLA budget); the hardware is used to sample the optimized circuit `repeats` times.
  full     the COBYLA loop itself runs on the hardware (as in the submitted diagnostic run, p=2, 1024 shots,
           20 evaluations); expensive: about maxiter estimator jobs per repeat.

Recorded per repeat: job ids, backend name, calibration timestamp (if exposed), transpiled depth, two-qubit gate
count, SWAPs inserted by routing, final physical layout, QPU usage (if exposed), returned mask, gap to the exact
optimum, probability of sampling the optimum, and the hardware time-to-solution estimate.

Credentials: a saved account (QiskitRuntimeService.save_account), or IBM_API_KEY / IBM_CRN / IBM_CHANNEL
environment variables, or --ibm-config path/to/json with API_KEY, CRN, channel. Never commit that file.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np


def _service(args):
    from qiskit_ibm_runtime import QiskitRuntimeService

    if args.ibm_config:
        cfg = json.loads(Path(args.ibm_config).read_text())
        return QiskitRuntimeService(channel=cfg.get("channel", "ibm_quantum_platform"), token=cfg["API_KEY"],
                                    instance=cfg.get("CRN"))
    if os.environ.get("IBM_API_KEY"):
        return QiskitRuntimeService(channel=os.environ.get("IBM_CHANNEL", "ibm_quantum_platform"),
                                    token=os.environ["IBM_API_KEY"], instance=os.environ.get("IBM_CRN"))
    return QiskitRuntimeService()


def _swap_count(circuit, pm) -> int:
    """SWAPs inserted by the routing stage of the preset pass manager (before basis translation)."""
    try:
        from qiskit.transpiler import StagedPassManager

        sub = StagedPassManager(stages=["init", "layout", "routing"], init=pm.init, layout=pm.layout,
                                routing=pm.routing)
        return int(sub.run(circuit).count_ops().get("swap", 0))
    except Exception:
        return -1


def _two_qubit_count(circ) -> int:
    ops = circ.count_ops()
    return int(sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx", "rzz", "iswap")))


def _usage_seconds(job):
    for getter in (lambda j: j.usage(), lambda j: j.metrics()["usage"]["quantum_seconds"],
                   lambda j: j.usage_estimation.get("quantum_seconds")):
        try:
            v = getter(job)
            if v is not None:
                return float(v)
        except Exception:
            pass
    return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument("--instance", default=None,
                    help="a .npz written by the bank stage (default: Amino, R=12, seed 0, from <out>/bank/bank)")
    ap.add_argument("--bank-dir", default=str(repo / "revision_out" / "bank" / "bank"))
    ap.add_argument("--call", type=int, default=0)
    ap.add_argument("--backend", default="ibm_torino")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--protocol", choices=["sampled", "full"], default="sampled")
    ap.add_argument("--execution-mode", choices=["job", "batch", "session"], default="job")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--sim-shots", type=int, default=4096)
    ap.add_argument("--maxiter", type=int, default=200, help="COBYLA budget (simulator for 'sampled')")
    ap.add_argument("--hw-maxiter", type=int, default=20, help="COBYLA budget on hardware for 'full'")
    ap.add_argument("--optimization-level", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ibm-config", default=None)
    ap.add_argument("--out", default=str(repo / "revision_out" / "hardware"))
    ap.add_argument("--dry-run", action="store_true", help="use a fake backend (no IBM account needed)")
    args = ap.parse_args()

    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import Batch, EstimatorV2, SamplerV2, Session
    from scipy.optimize import minimize

    from revision.qubo import energy
    from revision.solvers import solve_qubo
    from src.rank_prunning.qaoa_rank_prunning import _build_cost_operator, qaoa_normalized_qubo

    if args.instance is None:
        import glob

        cands = sorted(glob.glob(str(Path(args.bank_dir) / "amino__ES__mu0.1__R12__s0__scaling.npz")))
        cands = cands or sorted(glob.glob(str(Path(args.bank_dir) / "*__R12__*scaling.npz")))
        cands = cands or sorted(glob.glob(str(Path(args.bank_dir) / "*scaling.npz")))
        if not cands:
            raise SystemExit(f"no banked QUBO in {args.bank_dir}; run 08_qubo_bank_dump.py first or pass --instance")
        args.instance = cands[0]
        print(f"[hardware] instance: {args.instance}")
    z = np.load(args.instance)
    meta = json.loads(bytes(z["meta"]).decode())
    j = args.call
    G, c = z[f"G_{j}"], z[f"c_{j}"]
    cm = [m for m in meta["calls"] if m["call"] == j][0]
    mu, xnorm2, R = float(cm["mu_eff"]), float(cm["xnorm2"]), int(cm["R"])
    es = solve_qubo("ES", G, c, mu)
    E_star, m_star = es["energy"], es["info"]["mask_int"]
    f_star = E_star + xnorm2
    Q_s, lin_s, scale_ref = qaoa_normalized_qubo(G, c, mu)
    H = _build_cost_operator(Q=Q_s, lin=lin_s)
    ansatz = QAOAAnsatz(cost_operator=H, reps=args.reps)
    init = [math.pi / 2] * args.reps + [math.pi] * args.reps

    def select(dist):
        """In-loop rule: lowest energy among the R most frequent bitstrings, skipping all-zero;
        if none is nonzero, lowest energy among all sampled nonzero bitstrings."""
        top = sorted(dist, key=lambda k: -dist[k])
        cands = [k for k in top[:R] if k != 0] or [k for k in top if k != 0]
        best = (math.inf, None)
        for k in cands:
            E = energy(np.array([(k >> i) & 1 for i in range(R)]), G, c, mu)
            if E < best[0]:
                best = (E, k)
        return best

    # Angles on the noiseless simulator, exactly as the in-loop QAOA (Aer, optimization level 0, COBYLA).
    from qiskit_aer import AerSimulator

    aer = AerSimulator(shots=args.sim_shots)
    aer.seed_simulator = args.seed
    acirc = generate_preset_pass_manager(optimization_level=0, backend=aer).run(ansatz)
    with Session(backend=aer) as s_aer:
        e_aer = EstimatorV2(mode=s_aer)
        e_aer.options.default_shots = args.sim_shots
        Ha = H.apply_layout(acirc.layout)
        res_sim = minimize(lambda p: float(e_aer.run([(acirc, Ha, p)]).result()[0].data.evs), init,
                           method="COBYLA", tol=1e-6, options={"maxiter": args.maxiter})
    params_sim = res_sim.x
    acm = acirc.assign_parameters(params_sim)
    acm.measure_all()
    sim_counts = SamplerV2(mode=aer).run([(acm,)], shots=args.sim_shots).result()[0].data.meas.get_int_counts()
    sim_dist = {k: v / sum(sim_counts.values()) for k, v in sim_counts.items()}
    sim_E, _ = select(sim_dist)

    if args.dry_run:
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        backend = FakeTorino()
    else:
        backend = _service(args).backend(args.backend)
    pm = generate_preset_pass_manager(optimization_level=args.optimization_level, backend=backend,
                                      seed_transpiler=args.seed)
    isa = pm.run(ansatz)                      # for the estimator ('full' protocol)
    logical_meas = ansatz.copy()
    logical_meas.measure_all()                # measure the R logical qubits only, then transpile
    meas = pm.run(logical_meas)               # what the sampler runs; depth/gates/SWAPs are reported for it
    swaps = _swap_count(logical_meas, pm)
    try:
        calib = str(backend.properties().last_update_date)
    except Exception:
        calib = ""
    layout = meas.layout.final_index_layout() if meas.layout is not None else []

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rep in range(args.repeats):
        t0 = time.time()
        job_ids, trace = [], []
        ctx = None
        if not args.dry_run and args.execution_mode != "job":
            ctx = Session(backend=backend) if args.execution_mode == "session" else Batch(backend=backend)
        mode = ctx if ctx is not None else backend
        try:
            if args.protocol == "full":
                est = EstimatorV2(mode=mode)
                est.options.default_shots = args.shots
                Hl = H.apply_layout(isa.layout)

                def cost(p):
                    job = est.run([(isa, Hl, p)])
                    job_ids.append(job.job_id())
                    v = float(job.result()[0].data.evs)
                    trace.append(v)
                    return v

                params = minimize(cost, init, method="COBYLA", tol=1e-6, options={"maxiter": args.hw_maxiter}).x
            else:
                params = params_sim
            job = SamplerV2(mode=mode).run([(meas.assign_parameters(params),)], shots=args.shots)
            job_ids.append(job.job_id())
            counts = job.result()[0].data.meas.get_int_counts()
            usage = _usage_seconds(job)
        finally:
            if ctx is not None:
                ctx.close()
        n = sum(counts.values())
        dist = {k: v / n for k, v in counts.items()}
        p_shot = dist.get(m_star, 0.0)
        best_E, best_k = select(dist)
        rows.append({
            "repeat": rep, "backend": getattr(backend, "name", str(backend)), "calibration": calib,
            "protocol": args.protocol, "R": R, "reps": args.reps, "shots": args.shots,
            "isa_depth": meas.depth(), "two_qubit_gates": _two_qubit_count(meas), "swaps_routing": swaps,
            "layout": json.dumps(layout), "job_ids": json.dumps(job_ids), "qpu_seconds": usage,
            "wall_s": time.time() - t0, "mask_int": best_k, "energy": best_E,
            "gap_pct": 100.0 * (best_E - E_star) / f_star,
            "hit": int(abs(best_E - E_star) <= 1e-9 * max(1.0, abs(E_star))), "p_shot_opt": p_shot,
            "sim_gap_pct": 100.0 * (sim_E - E_star) / f_star, "sim_p_shot_opt": float(sim_dist.get(m_star, 0.0)),
            "hw_trace": json.dumps(trace), "instance": Path(args.instance).name, "call": j, "mu_eff": mu,
            "shot_tts99_s": ((usage / n) * math.ceil(math.log(0.01) / math.log(1 - p_shot))
                             if 0 < p_shot < 1 and np.isfinite(usage) else float("nan")),
        })
        print(f"[hardware] repeat {rep}: gap {rows[-1]['gap_pct']:.3f}%  p_shot(opt) {p_shot:.4f}  jobs {job_ids}",
              flush=True)
    path = out / "hardware_repeats.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"[hardware] wrote {path}")


if __name__ == "__main__":
    main()
