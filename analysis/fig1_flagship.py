# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""Figure 1: (a) periodogram of ChunkKV and SnapKV on published predictions (periods 2-64), (b) PB3 stride recovery, (c) pa2 offset profile.
Depends on the outputs of pb3.py (out/pb3_ratios.csv) and pa2.py (out/pa2_per_phase.csv), which read data/ only; run them first."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import DATA, FIG, OUT, panel_letter, plot_style  # noqa: E402


def main():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plot_style()
    pa = pd.read_csv(f"{DATA}/periodogram_F_periods2to64.csv")
    pb = pd.read_csv(f"{OUT}/pb3_ratios.csv")
    pp = pd.read_csv(f"{OUT}/pa2_per_phase.csv")
    full_cache = json.load(open(f"{OUT}/pa2.json"))["full_cache|mean"]
    GREEN, GREY, ORANGE, DARK, BLUE, PURPLE = "#1b7a5a", "#8c8c8c", "#d55e00", "#333333", "#3a6ea5", "#7b4f9e"

    def ratio(press):
        d = pa[(pa.press == press) & (pa.model == "Llama-3.1-8B-Instruct") & np.isclose(pa.cr, 0.88)].sort_values("P")
        return d.P.values, d.F.values / np.median(d.F.values)

    fig = plt.figure(figsize=(7.2, 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.58, left=0.08, right=0.99, top=0.82, bottom=0.33)
    ax = fig.add_subplot(gs[0])
    P1, R1 = ratio("chunkkv__Q")
    P2, R2 = ratio("snapkv__Q")
    ax.plot(P2, R2, color=GREY, lw=1.0, label="SnapKV, no blocks")
    ax.plot(P1, R1, color=DARK, lw=1.3, label="ChunkKV, 20-token chunks")
    ax.axvline(20, color=DARK, ls=":", lw=0.7)
    ax.set_xlabel("period tested (tokens)")
    ax.set_ylabel("periodic signal (F / median F)")
    ax.set_title("Leaderboard predictions reveal\nthe chunk size (20 tokens)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.25))
    panel_letter(ax, "a")
    ax = fig.add_subplot(gs[1])
    hs = []
    Ps = np.arange(2, 9)
    for m_, S, col, off, lab in [("flash", 2, BLUE, -0.13, "V4.1-Flash (stride 2)"), ("pro", 4, PURPLE, 0.13, "V4-Pro (stride 4)")]:
        d = pb[pb.model == m_].sort_values("period")
        rr, lo, hi = d.ratio.values, d.lo.values, d.hi.values
        ax.errorbar(Ps + off, rr, yerr=[rr - lo, hi - rr], fmt="none", ecolor=col, elinewidth=0.8, capsize=1.5, alpha=0.8)
        ax.scatter(Ps + off, rr, s=[30 if k == S else 14 for k in Ps], facecolors=[col if k == S else "white" for k in Ps], edgecolors=col, lw=0.9, zorder=3)
        hs.append(Line2D([], [], marker="o", ls="", color=col, label=lab, ms=4))
    hs.append(Line2D([], [], marker="o", ls="", mfc="white", mec=GREY, label="other periods", ms=3.5))
    ax.axhline(1, color=DARK, lw=0.6)
    ax.set_xticks(Ps)
    ax.set_xlabel("period tested (tokens)")
    ax.set_ylabel("shift effect / permutation null")
    ax.set_title("DeepSeek API: each model peaks\nat its published stride")
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.45, -0.25))
    panel_letter(ax, "b")
    ax = fig.add_subplot(gs[2])
    q = pp[np.isclose(pp.cr, 0.875)]
    for arm, lab, col, lw in [("A1_chunkkv", "one grid, chunk 20 (ChunkKV)", DARK, 1.2), ("A7_chunk10", "one grid, chunk 10", ORANGE, 1.0),
                              ("A3_twogrid", "two grids, chunk 20 (offsets 0, 10)", GREEN, 1.6)]:
        d = q[q.arm == arm].sort_values("phase")
        ax.plot(d.phase, d.acc, color=col, lw=lw, marker="o", ms=2.5, label=lab)
    ax.axhline(full_cache, color=GREY, ls=":", lw=0.8)
    ax.text(19, 0.985, "full cache", ha="right", va="bottom", fontsize=6, color=GREY)
    ax.set_xlabel("needle offset in the chunk (tokens)")
    ax.set_ylabel("share of correct answers")
    ax.set_xticks([0, 5, 10, 15, 19])
    ax.set_ylim(-0.06, 1.1)
    ax.set_title("Open model, same memory: two grids\nnarrow the gap, finer chunks do not")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.25))
    panel_letter(ax, "c")
    fig.savefig(f"{FIG}/fig1_flagship.png", dpi=200)


if __name__ == "__main__":
    main()
