# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""Table 6: options for reliable needle retrieval under block-based compression. Every cell is a value computed by another script;
this script gathers them from out/*.json (run c2g.py, pa2.py, pa3.py and pb3.py first) and derives the selection overhead range."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, write_result  # noqa: E402


def load(n):
    return json.load(open(f"{OUT}/{n}.json"))


def main():
    c, a2, a3, p3 = load("c2g"), load("pa2"), load("pa3"), load("pb3")
    v = {}
    over, share = [], []
    for cell in ("llama_75", "qwen_75", "qwen_875"):
        d = c[f"cost|{cell}|two_grid|select_s"] - c[f"cost|{cell}|chunkkv_repro|select_s"]
        over.append(d)
        share.append(100 * d / c[f"cost|{cell}|chunkkv_repro|total_s"])
    v["second_grid_overhead_s_min"], v["second_grid_overhead_s_max"] = round(min(over), 2), round(max(over), 2)
    v["second_grid_overhead_pct_min"], v["second_grid_overhead_pct_max"] = round(min(share), 0), round(max(share), 0)
    v["ruler_mean_diff_qwen_75"], v["ruler_mean_diff_llama_75"], v["ruler_mean_diff_qwen_875"] = c["qwen_75|diff"], c["llama_75|diff"], c["qwen_875|diff"]
    v["lower_cr_memory_factor"] = a3["iso_memory_factor"]
    v["chunk10_cr875"], v["chunk5_cr875"] = a2["one_grid_c10|cr0.875|mean"], a2["one_grid_c5|cr0.875|mean"]
    v["chunk40_cr88"], v["twogrid_cr88"] = a3["chunk40|cr0.875|mean"], a3["twogrid|cr0.875|mean"]
    v["pro_typo_change_reduction_pct"] = p3["pro|H4_reduction_pct"]
    write_result("table6", v)
    for k in sorted(v):
        print(k, v[k])


if __name__ == "__main__":
    main()
