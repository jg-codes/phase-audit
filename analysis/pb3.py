# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""Table 3 and Figures 1b and 3: registered test PB3 (stride recovery on two production APIs), rebuilt from the derived per-probe tables.

Inputs in data/pb3/: base margins (probe x 16 shifts), per-probe H3 and H4 tables, the kept-probe tables and the post hoc noise-control
tables. Raw API rows are not part of the release; the derived tables are the registered analysis' own intermediate tables.
H1/H2 use 10,000 permutations and 10,000 two-level bootstrap resamples with the registered seed; every value is deterministic
for this seed, but bootstrap interval ends can differ from the paper by Monte Carlo noise if the numpy generator changes.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pb3_stats as A  # noqa: E402
from common import DATA, FIG, panel_letter, plot_style, write_result  # noqa: E402

D3 = f"{DATA}/pb3"


def analyse(model):
    S = A.STRIDE[model]
    Mdf = pd.read_csv(f"{D3}/base_margins_{model}.csv")
    srcs = Mdf.source.to_numpy()
    M = Mdf[[c for c in Mdf.columns if c.startswith("j")]]
    M.columns = range(16)
    keep = pd.read_csv(f"{D3}/keep_{model}.csv")
    out = dict(n_kept=int(keep.kept.sum()), n_h1=len(M))
    ER = A.excess_ratios(M, B=A.B_PERM, seed=A.SEED)
    out["ratios"] = {P: dict(ratio=v["ratio"], ratio_mean=v["ratio_mean"]) for P, v in ER.items()}
    idx = {P: i for i, P in enumerate(A.PERIODS)}
    obs_m = np.stack([ER[P]["obs_i"] for P in A.PERIODS], 1)
    nul_m = np.stack([ER[P]["null_i"] for P in A.PERIODS], 1)
    bs = np.array([obs_m[ix].sum(0) / nul_m[ix].sum(0) for ix in A.two_level_indices(srcs, A.B_BOOT, A.SEED)])
    out["ratio_ci95"] = {P: [float(np.percentile(bs[:, idx[P]], 2.5)), float(np.percentile(bs[:, idx[P]], 97.5))] for P in A.PERIODS}
    rv = np.array([ER[P]["ratio"] for P in A.PERIODS])
    ctrl = [3, 5, 7]
    mult = [P for P in A.PERIODS if P % S == 0]
    non = [P for P in A.PERIODS if P % S != 0]
    c1 = bool(rv[idx[S]] > 1.5 and ER[S]["p_perm"] < 0.01)
    c2 = bool(all(rv[idx[P]] < 1.2 for P in ctrl))
    pH1 = max(ER[S]["p_perm"], A.p_le(bs[:, idx[S]], 1.5), max(A.p_ge(bs[:, idx[P]], 1.2) for P in ctrl))
    out["H1"] = dict(ratio_true=float(rv[idx[S]]), controls={P: float(rv[idx[P]]) for P in ctrl}, p=float(pH1), kill=bool(not (c1 and c2)))

    def h2_ok(r):
        top = max(A.PERIODS, key=lambda P: r[idx[P]])
        okA = (top == S) or (top in mult and r[idx[S]] >= 0.9 * r[idx[top]])
        okB = all(r[idx[S]] > r[idx[P]] for P in non)
        return okA, okB

    okA, okB = h2_ok(rv)
    fails = np.array([not all(h2_ok(b_)) for b_ in bs])
    out["H2"] = dict(argmax=int(max(A.PERIODS, key=lambda P: rv[idx[P]])), clause_argmax_ok=bool(okA), p=float((fails.sum() + 1) / (len(bs) + 1)),
                     kill=bool(not okB), share_ok=float(1 - fails.mean()))
    T3 = pd.read_csv(f"{D3}/h3_per_probe_{model}.csv")
    out["H3"] = A.h3(T3, A.B_BOOT, A.SEED)
    out["H3"]["kill"] = bool(out["H3"]["lb95_onesided"] <= 0.3)
    out["H3"]["p"] = out["H3"]["p_le_0p3"]
    T4 = pd.read_csv(f"{D3}/h4_per_probe_{model}.csv")
    out["H4"] = A.h4(T4, "single", A.B_BOOT, A.SEED)
    out["H4"]["kill"] = bool(out["H4"]["lb95_onesided"] <= 0)
    out["H4"]["p"] = out["H4"]["p_le_0"]
    return out, Mdf


def noise_controls(model):
    """Post hoc noise-matched controls for H4 (pa3): excess of shift averaging over noise-only averaging, two-level bootstrap."""
    S = A.STRIDE[model]
    N = pd.read_csv(f"{D3}/noise_table_{model}.csv")
    res = {}

    def boot(ch, cn, label):
        d = N.dropna(subset=[ch[0], ch[1], cn[0], cn[1]])
        srcs = d.source.to_numpy()
        hs, ha, ns_, na = (d[c].to_numpy() for c in (ch[0], ch[1], cn[0], cn[1]))
        f = lambda ix: np.array([1 - ha[ix].mean() / hs[ix].mean(), 1 - na[ix].mean() / ns_[ix].mean()])
        b = np.array([f(ix) for ix in A.two_level_indices(srcs, 10000, 20261025)])
        pt = f(np.arange(len(d)))
        ex = b[:, 0] - b[:, 1]
        res[label] = dict(n=int(len(d)), excess_pp=float(100 * (pt[0] - pt[1])), excess_ci_pp=[float(100 * np.percentile(ex, 2.5)), float(100 * np.percentile(ex, 97.5))],
                          excess_lb95_pp=float(100 * np.percentile(ex, 5)))

    for K in range(2, S + 1):
        boot(("single", f"avg{K}"), ("M1", f"M{K}"), f"M_K{K}")
    boot(("single", f"avg{S}"), ("A_single", "A_avg"), "A_KS")
    boot(("single", f"avg{S}"), ("B_single", "B_avg"), "B_KS")
    return res


def main():
    R, Ms, NC = {}, {}, {}
    for m in ("flash", "pro"):
        R[m], Ms[m] = analyse(m)
        NC[m] = noise_controls(m)
    V = A.verdicts(R)
    v = {}
    for m in ("flash", "pro"):
        o = R[m]
        v[f"{m}|kept"] = o["n_kept"]
        v[f"{m}|complete_sweeps"] = o["n_h1"]
        v[f"{m}|H1_ratio"] = round(o["H1"]["ratio_true"], 2)
        v[f"{m}|H1_ci_lo"] = round(o["ratio_ci95"][A.STRIDE[m]][0], 2)
        v[f"{m}|H1_ci_hi"] = round(o["ratio_ci95"][A.STRIDE[m]][1], 2)
        v[f"{m}|controls_max"] = round(max(o["H1"]["controls"].values()), 2)
        v[f"{m}|H1_holm_p"] = round(V[(m, "H1")]["p_holm"], 3)
        v[f"{m}|H1_verdict"] = "SUPPORTED" if V[(m, "H1")]["verdict"] == "SUPPORTED" else V[(m, "H1")]["verdict"]
        v[f"{m}|H2_argmax"] = o["H2"]["argmax"]
        v[f"{m}|H2_share_pct"] = round(100 * o["H2"]["share_ok"], 2)
        v[f"{m}|H2_holm_p"] = round(V[(m, "H2")]["p_holm"], 3)
        v[f"{m}|H3_slope"] = round(o["H3"]["slope"], 3)
        v[f"{m}|H3_lb"] = round(o["H3"]["lb95_onesided"], 3)
        v[f"{m}|H3_holm_p"] = round(V[(m, "H3")]["p_holm"], 3)
        v[f"{m}|H3_verdict"] = V[(m, "H3")]["verdict"].split(" (")[0]
        v[f"{m}|H4_reduction_pct"] = round(100 * o["H4"]["reduction"], 1)
        v[f"{m}|H4_ci_lo_pct"] = round(100 * o["H4"]["ci95_two_sided"][0], 1)
        v[f"{m}|H4_ci_hi_pct"] = round(100 * o["H4"]["ci95_two_sided"][1], 1)
        v[f"{m}|H4_holm_p"] = round(V[(m, "H4")]["p_holm"], 4)
        v[f"{m}|H4_verdict"] = V[(m, "H4")]["verdict"].split(" (")[0]
        v[f"{m}|mean_of_ratios"] = round(o["ratios"][A.STRIDE[m]]["ratio_mean"], 2)
        for k, r in NC[m].items():
            v[f"{m}|noise_{k}_excess_pp"] = round(r["excess_pp"], 1)
            v[f"{m}|noise_{k}_ci_lo_pp"] = round(r["excess_ci_pp"][0], 1)
            v[f"{m}|noise_{k}_ci_hi_pp"] = round(r["excess_ci_pp"][1], 1)
            v[f"{m}|noise_{k}_lb_pp"] = round(r["excess_lb95_pp"], 1)
    write_result("pb3", v)
    for k in sorted(v):
        print(k, v[k])
    # ---- Figure 3 (a, b: periodogram-type ratios; c, d: typo contrast) ----
    plot_style()
    import matplotlib.pyplot as plt
    PL = A.PERIODS
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), gridspec_kw=dict(hspace=0.5, wspace=0.35))
    for col, (m, lab, c) in enumerate([("flash", "DeepSeek-V4.1-Flash (S = 2)", "#3a6ea5"), ("pro", "DeepSeek-V4-Pro (S = 4)", "#7b4f9e")]):
        S = A.STRIDE[m]
        ax = axes[0, col]
        rr = np.array([R[m]["ratios"][P]["ratio"] for P in PL])
        ci = np.array([R[m]["ratio_ci95"][P] for P in PL])
        ax.errorbar(PL, rr, yerr=[rr - ci[:, 0], ci[:, 1] - rr], fmt="none", ecolor=c, elinewidth=0.8, capsize=1.5)
        ax.scatter(PL, rr, s=[34 if P == S else 16 for P in PL], facecolors=["#1b7a5a" if P == S else "white" for P in PL], edgecolors=c, zorder=3)
        ax.axhline(1, color="#333333", lw=0.6)
        ax.axhline(1.5, color="#888888", ls=":", lw=0.8)
        ax.axhline(1.2, color="#888888", ls=":", lw=0.8)
        ax.set_xticks(PL)
        ax.set_xlabel("period tested (tokens)")
        ax.set_ylabel("shift effect / permutation null")
        ax.set_title(lab)
        panel_letter(ax, "ab"[col])
        T3 = pd.read_csv(f"{D3}/h3_per_probe_{m}.csv")
        ax = axes[1, col]
        ax.scatter(T3.pred, T3.obs, s=8, color=c, alpha=0.7)
        xs = np.array([T3.pred.min(), T3.pred.max()])
        sl = R[m]["H3"]["slope"]
        ax.plot(xs, sl * xs + (T3.obs.mean() - sl * T3.pred.mean()), color="#333333")
        ax.plot(xs, 0.3 * xs, color="#888888", ls="--")
        ax.set_xlabel("contrast predicted from the probe's shift profile")
        ax.set_ylabel("observed typo contrast")
        ax.set_title(f"slope {sl:.3f}")
        panel_letter(ax, "cd"[col])
    fig.savefig(f"{FIG}/fig3_pb3.png", dpi=200)
    # ---- Figure 1b data for the flagship figure ----
    pd.DataFrame([dict(model=m, period=P, ratio=R[m]["ratios"][P]["ratio"], lo=R[m]["ratio_ci95"][P][0], hi=R[m]["ratio_ci95"][P][1])
                  for m in R for P in PL]).to_csv(f"{os.path.dirname(FIG)}/out/pb3_ratios.csv", index=False)


if __name__ == "__main__":
    main()
