# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""Figure 4 and the pa3 numbers in Results: iso-memory sweep (Qwen3-0.6B, CPU, exploratory) and selection timing on the test CPU.

Runs the pa3 statistics (exploratory run, not pre-registered) (pa3_stats.py, item-cluster bootstrap, 5,000 resamples, seed 20261025) on data/pa3/pa3_outcomes.csv.gz,
then reads the result for the expected values and the figure. Timing: medians over the three timed contexts per mode (data/pa3/timing_*.jsonl).
"""
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import DATA, FIG, OUT, plot_style, write_result  # noqa: E402


def main():
    subprocess.run([sys.executable, f"{HERE}/pa3_stats.py", f"{OUT}/pa3", f"{DATA}/pa3/pa3_outcomes.csv.gz"], check=True, capture_output=True)
    r = json.load(open(f"{OUT}/pa3_iso_results.json"))
    ref = str(0.875)
    v = {"n_items": r["n_items"], "full_cache_mean": round(r["full_cache_acc"], 3)}
    for arm, tag in (("A3_twogrid", "twogrid"), ("A1_chunkkv", "chunkkv"), ("A15_chunk40", "chunk40")):
        v[f"{tag}|cr0.875|mean"] = round(r["curves"][arm][ref]["mean"], 3)
    v["iso_cr_star"] = round(r["iso_score"]["mean"]["cr_star"], 2)
    lo, hi = r["iso_score"]["mean"]["cr_star_boot_ci_interpolated_only"]
    v["iso_cr_star_ci_lo"], v["iso_cr_star_ci_hi"] = round(lo, 2), round(hi, 2)
    kept_ratio = (1 - r["iso_score"]["mean"]["cr_star"]) / (1 - r["eff_cr"][ref])
    v["iso_memory_factor"] = round(kept_ratio, 1)
    fa, fb = (1 - lo) / (1 - r["eff_cr"][ref]), (1 - hi) / (1 - r["eff_cr"][ref])
    v["iso_memory_factor_lo"], v["iso_memory_factor_hi"] = round(min(fa, fb), 1), round(max(fa, fb), 1)
    v["iso_boot_share_never_reaching_pct"] = round(100 * r["iso_score"]["mean"]["boot_share"]["below_grid_min"] / 5000, 1)
    # timing
    files = sorted(glob.glob(f"{DATA}/pa3/timing_*.jsonl"))
    tm = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    tm["cr"] = tm.cr.fillna(0.0)
    g = tm.groupby(["mode", "cr"]).agg(n=("phase", "size"), prefill_med=("t_prefill_total", "median"), sel_med=("t_select_gather", "median")).reset_index()
    g.to_csv(f"{OUT}/pa3_timing_table.csv", index=False)
    s = g.set_index(["mode", "cr"])
    v["second_grid_overhead_s"] = round(float(s.loc[("twogrid", 0.875), "sel_med"] - s.loc[("c20", 0.875), "sel_med"]), 2)
    v["prefill_s"] = round(float(s.loc[("c20", 0.875), "prefill_med"]), 0)
    write_result("pa3", v)
    for k in sorted(v):
        print(k, v[k])
    # Figure 4
    plot_style()
    import matplotlib.pyplot as plt
    arms = [("A3_twogrid", "Two-grid, chunk 20 (offsets 0, 10)", "#D55E00", 2.2, "-"), ("A1_chunkkv", "ChunkKV, chunk 20", "#333333", 1.4, "-"),
            ("A15_chunk40", "ChunkKV, chunk 40", "#8a8a8a", 1.2, "--"), ("A7_chunk10", "ChunkKV, chunk 10", "#b5b5b5", 1.2, ":")]
    crs = [str(c) for c in r["crs"]]
    x = [r["eff_cr"][c] for c in crs]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    titles = {"mean": "Mean accuracy over 20 needle offsets", "worst": "Accuracy at the worst needle offset"}
    for ax, key, ci in ((axs[0], "mean", "mean_ci"), (axs[1], "worst", "worst_ci")):
        for a, lab, col, lw, ls in arms:
            y = [r["curves"][a][c][key] for c in crs]
            lo_ = [r["curves"][a][c][ci][0] for c in crs]
            hi_ = [r["curves"][a][c][ci][1] for c in crs]
            ax.plot(x, y, color=col, lw=lw, ls=ls, marker="o", ms=3, label=lab)
            if a in ("A3_twogrid", "A1_chunkkv"):
                ax.fill_between(x, lo_, hi_, color=col, alpha=0.10, lw=0)
        t = r["iso_score"]["mean" if key == "mean" else "worst_offset"]
        ax.axhline(t["target_acc"], color="#D55E00", lw=0.7, ls=(0, (3, 3)), alpha=0.8)
        ax.plot([t["cr_star"]], [t["target_acc"]], marker="D", color="#333333", ms=5, zorder=5)
        ax.text(t["cr_star"], t["target_acc"] - 0.09, f"CR* {t['cr_star']:.2f}", ha="center", va="top", color="#333333", fontsize=7)
        ax.set_xlabel("Effective compression ratio")
        ax.set_xlim(0.48, 1.07)
        ax.set_ylim(-0.04, 1.05)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
        ax.set_title(titles[key], fontsize=8)
    axs[0].set_ylabel("Exact-match accuracy (higher is better)")
    axs[1].legend(loc="upper right", fontsize=6)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.92, bottom=0.2, wspace=0.06)
    fig.savefig(f"{FIG}/fig4_iso.png", dpi=200)


if __name__ == "__main__":
    main()
