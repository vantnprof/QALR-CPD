"""Supplementary tables: split-half check of the penalty-selection protocol, and the hardware summary.

Run by ``python main.py tables`` after the main tables.

Split-half check (manuscript Sec. 4.4, limitation vii; Online Resource 1, Table S12). On synthetic data each seed
draws a new tensor, so the plateau and its exact-recovery rate in Table 14 are in-sample. Here the 20 seeds of each of
the 15 distinct synthetic designs are split into seeds 0-9 and 10-19; the protocol of Sec. 3.7 (widest plateau of
stability >= 0.6, at least three grid values, mu at its middle value) selects on one half and exact recovery is
measured at the selected mu on the other half, in both directions. ES runs come from stage mu (SNR grid, default
benchmark) and stage mu-rules (collinearity designs); Mag-IL and Diag-x from stage mu-rules. The two collinearity
designs with rho = 0.5 are the 0 and +10 dB SNR-grid designs (same tensors and seeds) and are not counted twice.

Hardware (Table 19, Sec. 4.6.3; Table S13): per backend (ibm_fez, FakeFez, FakeTorino), five repeats from
``python main.py hardware``.

Writes <tables-dir>/tableS_heldout.tex and tableS_hardware.tex and prints the summary quoted in the text.
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

from src.experiments import config as C
from src.analysis.plateau import rank_profile, widest_plateau

SNR = [f"snr{s:+d}dB_rho{C.SNR_GRID_RHO:g}" for s in C.SNR_GRID_DB]
RHO = [f"rho{r:.1f}_snr{s:+d}dB" for r in C.RHO_GRID if r != C.SNR_GRID_RHO for s in C.RHO_SWEEP_SNR_DB]
DESIGNS = SNR + ["synth_default"] + RHO
RULES = ["ES", "Mag-IL", "Diag-x"]
MIN_WIDTH = 3


def rd(p: Path):
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def select(rows, ds, method, seeds, rmax=20):
    sub = [r for r in rows if int(r["seed"]) in seeds]
    prof = rank_profile(sub, ds, method, rmax)
    pl = widest_plateau(prof, exclude_full=rmax)
    return prof, (pl if pl and pl["width"] >= MIN_WIDTH else None)


def tex(s: str) -> str:
    return s.replace("_", r"\_")


def heldout(out: Path):
    mu, rules = rd(out / "mu" / "runs.csv"), rd(out / "mu_rules" / "runs.csv")
    halves = [(set(range(0, 10)), set(range(10, 20))), (set(range(10, 20)), set(range(0, 10)))]
    lines, summary = [], {}
    for m in RULES:
        n_true = n_wrong = n_abst = 0
        rec = []
        for ds in DESIGNS:
            rows = mu if (m == "ES" and ds in SNR + ["synth_default"]) else rules
            for S, T in halves:
                prof, pl = select(rows, ds, m, S)
                rt = prof[0]["rank_true"] if prof else None
                if pl is None:
                    n_abst += 1
                    lines.append((m, ds, min(S), "abstain", "", ""))
                    continue
                tprof = rank_profile([r for r in rows if int(r["seed"]) in T], ds, m, 20)
                ex = next(p["exact"] for p in tprof if abs(p["mu"] - pl["mu_mid"]) < 1e-12)
                if pl["rank"] == rt:
                    n_true += 1
                    rec.append(ex)
                else:
                    n_wrong += 1
                lines.append((m, ds, min(S), pl["rank"], pl["mu_mid"], ex))
        summary[m] = (n_true, n_wrong, n_abst, rec)
    return lines, summary


def hardware(out: Path):
    rows = rd(out / "hardware" / "hardware_repeats.csv")
    res = []
    for b in ("ibm_fez", "fake_fez", "fake_torino"):
        sub = [r for r in rows if r["backend"] == b]
        if not sub:
            continue
        shots = sum(int(r["shots"]) for r in sub)
        hits = sum(round(float(r["p_shot_opt"]) * int(r["shots"])) for r in sub)
        gaps = [float(r["gap_pct"]) for r in sub]
        res.append(dict(backend=b, n=len(sub), depth=sub[0]["isa_depth"], cx=sub[0]["two_qubit_gates"],
                        swaps=sub[0]["swaps_routing"], cal=sub[0]["calibration"][:10], hits=hits, shots=shots,
                        gap=(st.mean(gaps), st.stdev(gaps) if len(gaps) > 1 else 0.0), returned=sum(int(r["hit"]) for r in sub)))
    return res


def main(argv=None):
    from src import paths

    ap = argparse.ArgumentParser(prog="python main.py tables (supplementary)", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default=str(paths.RECORDS))
    ap.add_argument("--tables-dir", default=str(paths.TABLES))
    a = ap.parse_args(argv)
    out = Path(a.records)
    tdir = Path(a.tables_dir)
    tdir.mkdir(parents=True, exist_ok=True)

    lines, summary = heldout(out)
    print(f"Split-half check over {len(DESIGNS)} distinct synthetic designs, 2 directions:")
    for m, (t, w, ab, rec) in summary.items():
        r = f"held-out exact recovery {min(rec):.0%}-{max(rec):.0%}, mean {st.mean(rec):.0%}" if rec else ""
        print(f"  {m:7s} true {t}/{2 * len(DESIGNS)}, wrong {w}, abstain {ab}; {r}")
    body = [r"\begin{longtable}{@{}lllccc@{}}", r"\toprule",
            r"Rule & Design & Selected on & Selected \(\hat R\) & \(\mu\) & Exact on the other half \\", r"\midrule",
            r"\endhead"]
    for m, ds, s0, rank, mu_, ex in lines:
        half = "seeds 0--9" if s0 == 0 else "seeds 10--19"
        if rank == "abstain":
            body.append(f"{m} & {tex(ds)} & {half} & abstain & -- & -- \\\\")
        else:
            body.append(f"{m} & {tex(ds)} & {half} & \\({rank}\\) & \\({mu_:g}\\) & {ex:.0%} \\\\".replace("%", r"\%"))
    body += [r"\midrule"]
    for m, (t, w, ab, rec) in summary.items():
        mean = f"mean {st.mean(rec):.0%}".replace("%", r"\%") if rec else ""
        body.append(f"\\multicolumn{{6}}{{@{{}}l}}{{{m}: true rank {t}, wrong rank {w}, abstain {ab} of "
                    f"{2 * len(DESIGNS)}; {mean}}} \\\\")
    body += [r"\bottomrule", r"\end{longtable}"]
    (tdir / "tableS_heldout.tex").write_text("% Split-half check of the penalty-selection protocol\n" + "\n".join(body) + "\n")

    hw = hardware(out)
    names = {"ibm_fez": r"\texttt{ibm\_fez} (device)", "fake_fez": r"FakeFez (noise model)",
             "fake_torino": r"FakeTorino (noise model)"}
    body = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
            r"Backend & Calibration & Depth & Two-qubit gates & SWAPs & Optimum sampled & Gap of returned mask (\%) \\",
            r"\midrule"]
    print("Hardware and noise-model repeats (one instance, 5 x 1024 shots):")
    for h in hw:
        body.append(f"{names[h['backend']]} & {h['cal']} & \\({h['depth']}\\) & \\({h['cx']}\\) & \\({h['swaps']}\\) & "
                    f"{h['hits']} of {h['shots']} & \\({h['gap'][0]:.0f}\\pm{h['gap'][1]:.0f}\\) \\\\")
        print(f"  {h['backend']:12s} optimum sampled {h['hits']}/{h['shots']}, gap {h['gap'][0]:.0f} +- {h['gap'][1]:.0f}%, "
              f"optimum returned {h['returned']}/{h['n']}, depth {h['depth']}, 2q {h['cx']}, swaps {h['swaps']}")
    body += [r"\bottomrule", r"\end{tabular}"]
    (tdir / "tableS_hardware.tex").write_text("% Hardware and noise-model repeats\n" + "\n".join(body) + "\n")
    print(f"wrote {tdir / 'tableS_heldout.tex'} and {tdir / 'tableS_hardware.tex'}")


if __name__ == "__main__":
    main()
