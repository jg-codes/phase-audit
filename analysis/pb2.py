# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""Appendix A (PB2 numbers quoted in Results) and Figure 6: registered test PB2.

The parity contrast P1 and the repeat-call flip rate of Flash (1k context) are recomputed from the per-probe flip table
(data/pb2/flip_per_probe_flash_1k.csv) with the registered two-level bootstrap (10,000 resamples, seed 20261024).
Figure 6 is drawn from the registered summary data/pb2/pb2_results.json (derived statistics only; the raw API rows are not
released). The PB1 exploratory count (33 of 48) is not rebuilt (see README).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import DATA, FIG, panel_letter, plot_style, write_result  # noqa: E402

B_BOOT, SEED = 10000, 20261024


def boot_means(per_probe, cols, B=B_BOOT, seed=SEED):
    """Two-level cluster bootstrap (source, then probe) of the probe-pooled means (reference copy of the registered function)."""
    rng = np.random.default_rng(seed)
    srcs = sorted(per_probe.source.unique())
    arr = {s: per_probe.loc[per_probe.source == s, cols].to_numpy(float) for s in srcs}
    point = per_probe[cols].to_numpy(float).mean(0)
    out = np.empty((B, len(cols)))
    for b in range(B):
        pick = rng.integers(0, len(srcs), len(srcs))
        parts = []
        for k in pick:
            a = arr[srcs[k]]
            parts.append(a[rng.integers(0, len(a), len(a))])
        out[b] = np.concatenate(parts).mean(0)
    return point, out


def main():
    f = pd.read_csv(f"{DATA}/pb2/flip_per_probe_flash_1k.csv")
    f["source"] = f.probe.str.rsplit("-", n=1).str[0]
    cols = ["p1", "repeat"]
    pt, bt = boot_means(f, cols)
    lo, hi = np.percentile(bt, [2.5, 97.5], axis=0)
    v = {"flash_1k|kept_probes": len(f), "flash_1k|P1_diff_pp": round(100 * pt[0], 2), "flash_1k|P1_ci_lo_pp": round(100 * lo[0], 2),
         "flash_1k|P1_ci_hi_pp": round(100 * hi[0], 2), "flash_1k|repeat_flip_pct": round(100 * pt[1], 1)}
    R = json.load(open(f"{DATA}/pb2/pb2_results.json"))
    v["flash_1k|P1_diff_pp_from_registered_json"] = round(100 * R["flips"]["flash_1k"]["P1"]["diff"], 2)
    write_result("pb2", v)
    for k in sorted(v):
        print(k, v[k])
    # Figure 6
    plot_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    a = ax[0, 0]
    arms = ["typo_d0", "typo_d1", "typo_d2", "typo_d1_pad", "orig_pad1", "repeat"]
    lab = ["typo d0\n(net 0)", "typo d1\n(+1)", "typo d2\n(+2)", "typo d1+pad\n(+2)", "orig+pad\n(+1)", "repeat"]
    for off, (k, c) in enumerate([("flash_1k", "#1f77b4"), ("pro_1k", "#d62728")]):
        fr = R["flips"][k]["flip_rates"]
        pts = [fr[x]["point"] * 100 for x in arms]
        lo_ = [fr[x]["ci95"][0] * 100 for x in arms]
        hi_ = [fr[x]["ci95"][1] * 100 for x in arms]
        xs = np.arange(len(arms)) + (off - 0.5) * 0.3
        a.errorbar(xs, pts, yerr=[np.array(pts) - lo_, np.array(hi_) - pts], fmt="o", color=c, capsize=3, label=k.split("_")[0] + f" (n={R['flips'][k]['n_kept']})")
    a.set_xticks(range(len(arms)))
    a.set_xticklabels(lab, fontsize=8)
    a.set_ylabel("flip rate vs original answer (%)")
    a.set_title("Flip rates by arm, 1k context, kept probes")
    a.legend()
    panel_letter(a, "a")
    b = ax[0, 1]
    fl = R["flips"]["flash_1k"]
    items = [("P1 odd-even\n(thr +15 pp)", fl["P1"]["diff"], fl["P1"]["ci95"]),
             ("P2a d1pad-d0\n(|.|<=10 pp)", fl["P2a_d1pad_minus_d0"]["diff"], fl["P2a_d1pad_minus_d0"]["ci95"]),
             ("P2b d1-d1pad\n(>=15 pp)", fl["P2b_d1_minus_d1pad"]["diff"], fl["P2b_d1_minus_d1pad"]["ci95"])]
    for i, (n, p, ci) in enumerate(items):
        b.errorbar(i, p * 100, yerr=[[(p - ci[0]) * 100], [(ci[1] - p) * 100]], fmt="o", color="k", capsize=4)
    b.axhline(0, color="grey", lw=0.8)
    b.axhline(15, color="C2", ls="--", lw=0.8)
    b.axhline(10, color="C1", ls=":", lw=0.8)
    b.axhline(-10, color="C1", ls=":", lw=0.8)
    b.set_xticks(range(3))
    b.set_xticklabels([n for n, _, _ in items], fontsize=8)
    b.set_ylabel("difference in flip rate (pp)")
    b.set_title("Flash P1/P2: point and 95% two-level CI")
    panel_letter(b, "b")
    c = ax[1, 0]
    w = 0.35
    for off, (k, col) in enumerate([("flash_heldout", "#1f77b4"), ("pro_heldout", "#d62728")]):
        vv = R["alignment"][k]["worst_phase_correct"]
        names = ["base", "avg_S", "aligned_0", "aligned_cal"]
        c.bar(np.arange(4) + (off - 0.5) * w, [vv[n] * 100 for n in names], w, color=col, label=f"{k.split('_')[0]} (n={R['alignment'][k]['n_probes']})")
    c.set_xticks(range(4))
    c.set_xticklabels(["base", "avg_S", "aligned_0", "aligned_cal"])
    c.set_ylim(90, 101)
    c.set_ylabel("worst-phase greedy correct (%)")
    c.set_title("Held-out sources, worst phase")
    c.legend(loc="upper center", ncol=2)
    panel_letter(c, "c")
    d = ax[1, 1]
    mp = R["P4"]["margin_by_phase_centered"]
    d.errorbar(range(4), [mp[str(i)]["mean"] for i in range(4)], yerr=[1.96 * mp[str(i)]["sem"] for i in range(4)], fmt="o-", color="#d62728", capsize=3)
    d.axhline(0, color="grey", lw=0.8)
    d.set_xticks(range(4))
    d.set_xlabel("segment start index mod 4")
    d.set_ylabel("probe-centred margin (logit)")
    d.set_title(f"Pro base margin by phase: F={R['P4']['period4_probe_cluster']['F']:.2f}, p={R['P4']['period4_probe_cluster']['p']:.2f}")
    panel_letter(d, "d")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig6_pb2.png", dpi=160)


if __name__ == "__main__":
    main()
