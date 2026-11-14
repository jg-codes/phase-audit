# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""Table 4 and the pa2 numbers in Results ("Two-grid selection on an open model"): exploratory run pa2, Qwen3-0.6B, CPU.

Reads data/pa2/pa2_outcomes.csv.gz (one row per context and arm: correct 0/1, kept-token shares). Mean = share correct over the 600 contexts
(30 complete items x 20 needle offsets); worst = minimum over the 20 per-offset means. Paired contrasts: item bootstrap (5,000 resamples,
seed 5), as in the independent recomputation of the run. The full set of pa2 statistics (exploratory run, not pre-registered) is in pa2_stats.py (not needed for the tables).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import DATA, OUT, write_result  # noqa: E402

ARMS = {"A1_chunkkv": "one_grid_c20", "A12_chunk20_off10": "one_grid_c20_moved10", "A7_chunk10": "one_grid_c10", "A8_chunk5": "one_grid_c5",
        "A11_token_shared": "token_level", "A9_twogrid10": "two_grids_c10_0_5", "A14_twogrid20_0_5": "two_grids_c20_0_5",
        "A3_twogrid": "two_grids_c20_0_10", "A10_threegrid20": "three_grids_c20", "A13_fourgrid20": "four_grids_c20"}


def load_matrices():
    d = pd.read_csv(f"{DATA}/pa2/pa2_outcomes.csv.gz")
    full = d[d.arm == "full"]
    c = d[(d.arm != "full") & (d.correct >= 0)].copy()
    cnt = c.groupby("item").phase_needle.nunique()
    # complete items: all 20 offsets decoded for every arm and ratio
    per = c.groupby(["cr", "arm", "item"]).phase_needle.nunique().unstack(["cr", "arm"])
    ok = sorted(per.index[(per == 20).all(axis=1)])
    c = c[c.item.isin(ok)]
    full = full[full.item.isin(ok)]
    M = {}
    for (cr, a), g in c.groupby(["cr", "arm"]):
        p = g.pivot(index="item", columns="phase_needle", values="correct").loc[ok]
        assert p.shape == (len(ok), 20) and not p.isna().any().any()
        M[(cr, a)] = p.to_numpy(float)
    hay = c.drop_duplicates("item").set_index("item").essay.loc[ok].to_numpy()
    return M, full, ok, hay


def main():
    M, full, ok, hay = load_matrices()
    n = len(ok)
    v = {"complete_items": n, "contexts_per_arm": n * 20}
    fp = full.groupby("phase_needle").correct.mean()
    v["full_cache|mean"] = round(full.correct.mean(), 3)
    v["full_cache|worst"] = round(fp.min(), 3)
    for (cr, a), m in sorted(M.items()):
        v[f"{ARMS[a]}|cr{cr}|mean"] = round(m.mean(), 3)
        v[f"{ARMS[a]}|cr{cr}|worst"] = round(m.mean(0).min(), 3)
    rng = np.random.default_rng(5)
    B = 5000
    BI = [rng.integers(0, n, n) for _ in range(B)]

    def dl(cr, a, b):
        x = (M[(cr, a)] - M[(cr, b)]).mean(1)
        lo, hi = np.percentile([x[i].mean() for i in BI], [2.5, 97.5])
        return round(x.mean(), 3), round(lo, 3), round(hi, 3)

    for cr in (0.75, 0.875):
        for a, b, tag in [("A3_twogrid", "A1_chunkkv", "twogrid_minus_chunkkv"), ("A12_chunk20_off10", "A1_chunkkv", "moved10_minus_chunkkv"),
                          ("A10_threegrid20", "A3_twogrid", "third_grid_gain"), ("A13_fourgrid20", "A3_twogrid", "fourth_grid_gain")]:
            d, lo, hi = dl(cr, a, b)
            v[f"{tag}|cr{cr}|delta"], v[f"{tag}|cr{cr}|lo"], v[f"{tag}|cr{cr}|hi"] = d, lo, hi
        p1, p12 = M[(cr, "A1_chunkkv")].mean(0), M[(cr, "A12_chunk20_off10")].mean(0)
        v[f"weak_offset_shift_corr|cr{cr}"] = round(float(np.corrcoef(p12, np.roll(p1, 10))[0, 1]), 2)
    write_result("pa2", v)
    # per-phase table for Figure 1c
    rows = [dict(cr=cr, arm=a, phase=p, acc=float(m.mean(0)[p])) for (cr, a), m in M.items() for p in range(20)]
    pd.DataFrame(rows).to_csv(f"{OUT}/pa2_per_phase.csv", index=False)
    for k in sorted(v):
        print(k, v[k])


if __name__ == "__main__":
    main()
