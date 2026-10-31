"""Figures added in the revision, drawn from the run records.

    python main.py figures                         (seconds; writes into reports/figures/revision/)
    python main.py figures --fig-dir /tmp/figs

FigR_rhat_distribution.png (manuscript Fig. 1; stage snr), FigR_covid_convergence.png (Fig. 2; stages covid20,
ablation, qaoa-tables), FigR_tradeoff.png (Fig. 12; stage mu, plateaus as reported in PLACEHOLDERS.md),
FigR_scaling.png (Fig. 13; stage solvebank) and FigS1_covid_trajectories.png (Fig. 19). The figures of the first
version are drawn by legacy/v1/plotting/ (see docs/REPRODUCE.md).
"""
import argparse
import csv
import gzip
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
REF = "#52514e"   # neutral for reference curves (fixed-rank ALS, exact search time)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "lines.linewidth": 1.4, "lines.markersize": 4,
})


def rd(p):
    p = Path(p)
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt") as fh:
        return list(csv.DictReader(fh))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------- R-hat distribution (SNR grid)
def fig_rhat(out, F):
    runs = rd(out / "snr" / "runs.csv")
    snrs = [-3, 0, 5, 10, 15, 20]
    names = {s: f"snr{s:+d}dB_rho0.5" for s in snrs}
    meths = [("Mag-IL", SLOT[1]), ("Diag-x", SLOT[2]), ("ES", SLOT[0])]
    label = {"ES": "QALR-CPD[ES]", "Mag-IL": "Mag-IL", "Diag-x": "Diag-x"}
    mus = [0.1, 0.3, 0.5]
    fig, axes = plt.subplots(len(mus), len(meths), figsize=(6.8, 5.6), sharex=True, sharey=True)
    for i, mu in enumerate(mus):
        for j, (m, col) in enumerate(meths):
            ax = axes[i, j]
            for k, s in enumerate(snrs):
                rr = [int(fnum(r["final_rank"])) for r in runs if r["dataset"] == names[s] and r["method"] == m
                      and math.isclose(fnum(r["mu"]), mu) and r["rank_init"] == "20"]
                cnt = defaultdict(int)
                for v in rr:
                    cnt[v] += 1
                for v, c in cnt.items():
                    ax.scatter(k, v, s=6 + 9 * c, color=col, alpha=0.85, edgecolor="white", linewidth=0.6, zorder=3)
                ex = 100 * sum(v == 6 for v in rr) / max(1, len(rr))
                ax.text(k, 17.2, f"{ex:.0f}", ha="center", va="center", fontsize=6.3, color=INK2)
            ax.axhline(6, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
            ax.set_ylim(0.5, 18.0)
            ax.set_xticks(range(len(snrs)), [f"{s:+d}" if s else "0" for s in snrs])
            ax.set_yticks([1, 3, 6, 9, 12])
            if i == 0:
                ax.set_title(label[m], color=INK, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"$\\mu={mu:g}$\nselected rank $\\hat R$")
            if i == len(mus) - 1:
                ax.set_xlabel("SNR (dB)")
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    fig.savefig(F / "FigR_rhat_distribution.png")
    plt.close(fig)


# ---------------------------------------------------------------- trade-off along the mu grid
def fig_tradeoff(out, F):
    runs = rd(out / "mu" / "runs.csv")
    # plateaus as reported by `python main.py tables` (same rule as Sec. 3.7), R_max = 20, exact search
    import re
    plateaus = {}
    for line in (_TABLES / "PLACEHOLDERS.md").read_text().splitlines():
        m_ = re.match(r"- \*\*(\S+) \| R_max=20 \| ES plateau\*\*: rank \d+ over mu in \[([\d.e-]+), ([\d.e-]+)\]", line)
        if m_:
            plateaus[m_.group(1)] = (float(m_.group(2)), float(m_.group(3)))
    ds_list = [("synth_default", "Default synthetic (−3.1 dB)"), ("amino", "Amino Acids"), ("kinetic", "X60t"),
               ("covid", "COVID-19")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.3))
    for ax, (ds, title) in zip(axes, ds_list):
        rr = [r for r in runs if r["dataset"] == ds and r["method"] == "ES" and r["rank_init"] == "20"]
        mus = sorted({fnum(r["mu"]) for r in rr})
        prof = []
        for mu in mus:
            g = [r for r in rr if math.isclose(fnum(r["mu"]), mu)]
            R = np.array([fnum(r["final_rank"]) for r in g])
            E = np.array([fnum(r["final_rel_error"]) for r in g])
            vals, counts = np.unique(R, return_counts=True)
            prof.append(dict(mu=mu, R=R, E=E, mode=int(vals[np.argmax(counts)]), stab=counts.max() / len(R)))
        x = np.array([np.median(p["R"]) for p in prof])
        y = np.array([np.median(p["E"]) for p in prof])
        xl = x - np.array([np.percentile(p["R"], 25) for p in prof])
        xh = np.array([np.percentile(p["R"], 75) for p in prof]) - x
        yl = y - np.array([np.percentile(p["E"], 25) for p in prof])
        yh = np.array([np.percentile(p["E"], 75) for p in prof]) - y
        ax.errorbar(x, y, xerr=[xl, xh], yerr=[yl, yh], fmt="none", ecolor=SLOT[0], alpha=0.35, elinewidth=0.8, zorder=2)
        ax.plot(x, y, color=SLOT[0], linewidth=1.2, zorder=3)
        ax.scatter(x, y, s=16, facecolor="white", edgecolor=SLOT[0], linewidth=1.0, zorder=4)
        pl = plateaus.get(ds)
        if pl:
            sel = [k for k, p in enumerate(prof) if pl[0] - 1e-12 <= p["mu"] <= pl[1] + 1e-12]
            ax.scatter(x[sel], y[sel], s=22, color=SLOT[0], zorder=5)
        for k in (0, len(prof) - 1):
            ax.annotate(f"μ={prof[k]['mu']:g}", (x[k], y[k]), xytext=(4, 4), textcoords="offset points", fontsize=6.3, color=INK2)
        ax.set_title(title)
        ax.set_xlabel("selected rank $\\hat R$ (median, IQR)")
        if ds in ("amino", "kinetic"):
            ax.set_yscale("log")
        ax.set_xlim(0, 21)
    axes[0].set_ylabel("relative error (median, IQR)")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(F / "FigR_tradeoff.png")
    plt.close(fig)


# ---------------------------------------------------------------- scaling bank
def fig_scaling(out, F):
    rows = rd(out / "bank" / "bank_results.csv")
    sc = [r for r in rows if r["tag"] == "scaling"]
    Rs = sorted({int(r["R"]) for r in sc})
    series = [("BF", "BF (single start)", SLOT[0], "o", "-"), ("MS-BF(M=32)", "MS-BF (M=32)", SLOT[1], "s", "-"),
              ("SA", "SA", SLOT[2], "^", "-"), ("Tabu", "tabu", SLOT[3], "D", "--"), ("QAOA", "QAOA (p=2)", SLOT[4], "v", ":")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for key, lab, col, mk, ls in series:
        xs, ps, ts = [], [], []
        for R in Rs:
            g = [r for r in sc if r["solver"] == key and int(r["R"]) == R]
            if g:
                xs.append(R)
                ps.append(100 * np.mean([fnum(r["hit"]) for r in g]))
                ts.append(np.median([fnum(r["time_s"]) for r in g]))
        a1.plot(xs, ps, color=col, marker=mk, linestyle=ls, label=lab, markeredgecolor="white", markeredgewidth=0.6)
        a2.plot(xs, ts, color=col, marker=mk, linestyle=ls, label=lab + (" (simulated)" if key == "QAOA" else ""),
                markeredgecolor="white", markeredgewidth=0.6)
    main_q = [r for r in rows if r["tag"] == "main" and r["solver"] == "QAOA"]
    if main_q:
        a1.scatter([20], [100 * np.mean([fnum(r["hit"]) for r in main_q])], s=26, facecolor="white", edgecolor=SLOT[4],
                   linewidth=1.2, zorder=5, label="QAOA, main bank (R = 20)")
    a1.set_ylim(-4, 104)
    a1.set_xlabel("number of QUBO variables $R$")
    a1.set_ylabel("instances solved to optimality (%)")
    a2.set_yscale("log")
    a2.set_xlabel("number of QUBO variables $R$")
    a2.set_ylabel("median time per call (s)")
    for a in (a1, a2):
        a.set_xticks(Rs)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, handlelength=2.2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=1.2, rect=(0, 0.09, 1, 1))
    fig.savefig(F / "FigR_scaling.png")
    plt.close(fig)


# ---------------------------------------------------------------- COVID-19 convergence and per-seed trajectories
def _curves(rows, method, mu, T=200):
    seeds = defaultdict(dict)
    for r in rows:
        if r["dataset"] != "covid" or r["method"] != method:
            continue
        if mu is None:
            if r["mu"] not in ("", "None"):
                continue
        elif r["mu"] in ("", "None") or not math.isclose(fnum(r["mu"]), mu):
            continue
        seeds[r["seed"]][int(r["sweep"])] = fnum(r["rel_error"])
    M = []
    for s, d in sorted(seeds.items(), key=lambda kv: int(kv[0])):
        v = np.full(T, np.nan)
        last = np.nan
        for t in range(1, T + 1):
            if t in d:
                last = d[t]
            v[t - 1] = last          # carry the last value after an early stop (tolerance)
        M.append(v)
    return np.array(M)


def fig_covid(out, F):
    als_rows = rd(out / "covid20" / "sweeps.csv.gz")
    abl = [r for r in rd(out / "ablation" / "sweeps.csv.gz") if r["dataset"] == "covid"]
    qa = [r for r in rd(out / "qaoa_tables" / "sweeps.csv.gz") if r["dataset"] == "covid"]
    T = 200
    t = np.arange(1, T + 1)
    curves = [
        ("ALS(20) = QALR-CPD[ES], μ=10⁻⁴", _curves(als_rows, "ALS", None), REF, "-"),
        ("QALR-CPD[ES], μ=0.65", _curves(abl, "ES", 0.65), SLOT[0], "-"),
        ("Mag-IL, μ=0.65", _curves(abl, "Mag-IL", 0.65), SLOT[1], "-"),
        ("TopK-IL, μ=0.65", _curves(abl, "TopK-IL", 0.65), SLOT[2], "--"),
        ("QALR-CPD[QAOA], μ=0.65 (5 seeds)", _curves(qa, "QAOA", 0.65), SLOT[3], ":"),
    ]
    es0 = _curves(abl, "ES", 1e-4)
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for p in (40, 80, 120, 160):
        ax.axvline(p, color=GRID, linewidth=0.8, zorder=0)
    for lab, M, col, ls in curves:
        m, s = np.nanmean(M, 0), np.nanstd(M, 0)
        ax.fill_between(t, m - s, m + s, color=col, alpha=0.12, linewidth=0)
        ax.plot(t, m, color=col, linestyle=ls, label=lab)
    ax.set_ylim(0.18, 0.66)
    ax.set_xlim(1, T)
    ax.set_xlabel("ALS sweep")
    ax.set_ylabel("relative error (observed entries)")
    ax.legend(loc="upper right", fontsize=6.3, handlelength=2.4)
    fig.tight_layout()
    fig.savefig(F / "FigR_covid_convergence.png")
    plt.close(fig)
    if es0.size and curves[0][1].size:
        print("max |ES(1e-4) - ALS(20)| over sweeps and seeds:", float(np.nanmax(np.abs(es0 - curves[0][1]))))

    panels = [("ALS(20) and QALR-CPD[ES], μ=10⁻⁴", curves[0][1], REF), ("QALR-CPD[ES], μ=0.65", curves[1][1], SLOT[0]),
              ("Mag-IL, μ=0.65", curves[2][1], SLOT[1]), ("TopK-IL, μ=0.65", curves[3][1], SLOT[2])]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.4), sharex=True, sharey=True)
    for ax, (lab, M, col) in zip(axes.ravel(), panels):
        for p in (40, 80, 120, 160):
            ax.axvline(p, color=GRID, linewidth=0.8, zorder=0)
        for v in M:
            ax.plot(t, v, color=col, linewidth=0.7, alpha=0.75)
        ax.set_title(lab)
        ax.set_ylim(0.18, 0.75)
    for ax in axes[1]:
        ax.set_xlabel("ALS sweep")
    for ax in axes[:, 0]:
        ax.set_ylabel("relative error")
    fig.tight_layout()
    fig.savefig(F / "FigS1_covid_trajectories.png")
    plt.close(fig)


_TABLES = None  # set by main(): folder with PLACEHOLDERS.md


def main(argv=None):
    global _TABLES
    from src import paths

    ap = argparse.ArgumentParser(prog="python main.py figures", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default=str(paths.RECORDS), help="run records (default: results/revision)")
    ap.add_argument("--tables-dir", default=str(paths.TABLES), help="where `python main.py tables` wrote PLACEHOLDERS.md")
    ap.add_argument("--fig-dir", default=str(paths.FIGURES), help="output (default: reports/figures/revision)")
    a = ap.parse_args(argv)
    out, F, _TABLES = Path(a.records), Path(a.fig_dir), Path(a.tables_dir)
    F.mkdir(parents=True, exist_ok=True)
    for fn in (fig_rhat, fig_tradeoff, fig_scaling, fig_covid):
        fn(out, F)
        print("done", fn.__name__)
    print(f"figures written to {F}")


if __name__ == "__main__":
    main()
