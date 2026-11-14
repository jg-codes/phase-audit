# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""Extract the printed numbers of the paper's tables into expected/*.json (one file per analysis script).

Usage: python tools/make_expected.py PAPER.tex expected/
The paper source is not part of this repository. Table cells are parsed from the LaTeX; the few numbers that the paper states only in
running text (abstract, Results, Appendix A) are listed in TEXT_CLAIMS below with their location. Each expected entry is
{"value": v, "tol": t}; the tolerance is half a unit of the last printed digit (relative 6% for p-values), and entries marked
"mc": true are bootstrap interval ends that may differ by Monte Carlo noise.
"""
import json
import os
import re
import sys

tex = open(sys.argv[1]).read()
OUT = sys.argv[2]
MODEL = {"Llama-3.1-8B": "Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct", "Qwen3-8B": "Qwen3-8B"}
CELL = {("Qwen3-8B", "0.875"): "qwen_875", ("Qwen3-8B", "0.75"): "qwen_75", ("Llama-3.1-8B-Instruct", "0.75"): "llama_75", ("Llama-3.1-8B-Instruct", "0.875"): "llama_875"}
exp = {k: {} for k in ["periodogram", "period20", "pb3", "pb2", "pa2", "pa2_mechanism", "pa3", "c2g", "table6"]}


def put(name, key, val, tol=None, mc=False):
    if isinstance(val, str):
        exp[name][key] = {"value": val}
        return
    if tol is None:
        tol = 1e-9
    e = {"value": val, "tol": tol}
    if mc:
        e["mc"] = True
    exp[name][key] = e


def dec(s):
    s = s.strip().replace("$-$", "-").replace("\u2212", "-").replace(",", "")
    m = re.search(r"[-+]?\d*\.?(\d*)", s)
    return float(re.sub(r"^\+", "", s.split()[0])), len(m.group(1)) if m else 0


def num(name, key, s, mc=False):
    v, d = dec(s)
    put(name, key, v, 0.5 * 10 ** -d + 1e-9, mc)


def pnum(name, key, s):
    v = float(s.strip())
    put(name, key, v, abs(v) * 0.06 + 1e-12)


def rows_of(block):
    out = []
    for line in block.split("\n"):
        line = line.strip()
        if line.endswith("\\\\") and "&" in line:
            cells = [c.strip().replace("\\_", "_") for c in line[:-2].split("&")]
            out.append(cells)
    return out


def section(start, end):
    a = tex.index(start)
    return tex[a: tex.index(end, a)]


# ---- Table 7 (Appendix B): all periodogram entries; Table 1 rows are a subset of the same values
blk = section("\\caption{All periodogram entries", "\\end{longtable}")
for c in rows_of(blk):
    if len(c) != 8 or c[0] in ("Press",):
        continue
    press, model, cr, acc, block, top, rt, rb = c
    k = f"{MODEL[model]}|{press}|{cr}"
    num("periodogram", k + "|acc", acc)
    put("periodogram", k + "|top", float(top), 0.5)
    num("periodogram", k + "|ratio_top", rt)
    if rb != "n/a":
        num("periodogram", k + "|ratio_block", rb)
# ---- Table 2
blk = section("\\caption{Period-20 tests with position control", "\\end{longtable}")
for c in rows_of(blk):
    if len(c) != 7 or c[0] == "Model":
        continue
    model, cr, ck, tw, n, diff, pf = c
    m = MODEL[model]
    k = f"{m}|{float(cr):.2f}"
    for tag, cell in (("chunkkv", ck), ("twin", tw)):
        if cell in ("excl.", "n.e."):
            put("period20", f"{k}|{tag}_LR", cell)
        else:
            lr, p = re.match(r"([\d.]+) \(([\d.e-]+)\)", cell).groups()
            num("period20", f"{k}|{tag}_LR", lr)
            pnum("period20", f"{k}|{tag}_p", p)
    put("period20", f"{k}|n", float(n.replace(",", "")), 0.5)
    num("period20", f"{k}|diff_pp", diff)
    F, p = re.match(r"([\d.]+) \(([\d.e-]+)\)", pf).groups()
    num("period20", f"{k}|paired_F", F)
    pnum("period20", f"{k}|paired_p", p)
# ---- Table 3 (PB3)
blk = section("\\caption{Registered test PB3", "\\end{tabular}")
rows = [r for r in rows_of(blk) if len(r) == 3 and r[0].startswith("H") and not r[0].startswith("Hyp")]
for h, r in zip(("H1", "H2", "H3", "H4"), rows):
    for model, cell in (("flash", r[1]), ("pro", r[2])):
        k = f"{model}|"
        hp = re.search(r"Holm p ([\d.]+)", cell).group(1)
        num("pb3", k + f"{h}_holm_p", hp)
        verdict = re.search(r"(NOT SUPPORTED|SUPPORTED|REFUTED)", cell).group(1)
        if h in ("H1", "H3", "H4"):
            put("pb3", k + f"{h}_verdict", verdict)
        if h == "H1":
            a, lo, hi = re.match(r"([\d.]+) \(([\d.]+) to ([\d.]+)\)", cell).groups()
            num("pb3", k + "H1_ratio", a)
            num("pb3", k + "H1_ci_lo", lo, mc=True)
            num("pb3", k + "H1_ci_hi", hi, mc=True)
            num("pb3", k + "controls_max", re.search(r"at most ([\d.]+)", cell).group(1))
        if h == "H2":
            put("pb3", k + "H2_argmax", float(re.search(r"P = (\d)", cell).group(1)), 0.5)
            num("pb3", k + "H2_share_pct", re.search(r"\(([\d.]+)\\% of", cell).group(1))
        if h == "H3":
            num("pb3", k + "H3_slope", re.search(r"slope ([\d.]+)", cell).group(1))
            num("pb3", k + "H3_lb", re.search(r"lower bound ([\d.]+)", cell).group(1), mc=True)
        if h == "H4":
            a, lo, hi = re.match(r"([\d.]+)\\% \(([\d.]+) to ([\d.]+)\)", cell).groups()
            num("pb3", k + "H4_reduction_pct", a)
            num("pb3", k + "H4_ci_lo_pct", lo, mc=True)
            num("pb3", k + "H4_ci_hi_pct", hi, mc=True)
cap = re.search(r"gives ([\d.]+) \(Flash\) and ([\d.]+) \(Pro\)", blk)
num("pb3", "flash|mean_of_ratios", cap.group(1))
num("pb3", "pro|mean_of_ratios", cap.group(2))
m = re.search(r"Kept probes: (\d+) \(Flash\) and (\d+) \(Pro\); H1 and H2 use the (\d+) and (\d+)", blk)
for key, g in (("flash|kept", 1), ("pro|kept", 2), ("flash|complete_sweeps", 3), ("pro|complete_sweeps", 4)):
    put("pb3", key, float(m.group(g)), 0.5)
# ---- Table 4 (pa2)
blk = section("\\caption{Granularity controls", "\\end{longtable}")
ARM4 = {"Full cache (reference)": "full_cache", "One grid, chunk 20 (ChunkKV)": "one_grid_c20", "One grid, chunk 20, moved by 10": "one_grid_c20_moved10",
        "One grid, chunk 10": "one_grid_c10", "One grid, chunk 5": "one_grid_c5", "Token-level top-k (own implementation)": "token_level",
        "Two grids, chunk 10 (offsets 0, 5)": "two_grids_c10_0_5", "Two grids, chunk 20 (offsets 0, 5)": "two_grids_c20_0_5",
        "Two grids, chunk 20 (offsets 0, 10)": "two_grids_c20_0_10", "Three grids, chunk 20 (0, 7, 14)": "three_grids_c20", "Four grids, chunk 20 (0, 5, 10, 15)": "four_grids_c20"}
for c in rows_of(blk):
    if c[0] in ARM4 and len(c) == 5:
        a = ARM4[c[0]]
        if a == "full_cache":
            num("pa2", "full_cache|mean", c[1])
            num("pa2", "full_cache|worst", c[2])
            continue
        for (cr, i) in (("0.75", 1), ("0.875", 3)):
            num("pa2", f"{a}|cr{cr}|mean", c[i])
            num("pa2", f"{a}|cr{cr}|worst", c[i + 1])
# ---- Table 5 (C2G)
blk = section("\\caption{Registered test C2G.", "\\end{longtable}")
for c in rows_of(blk):
    if len(c) != 8 or c[0] == "Model":
        continue
    model, cr, ck, tg, diff, lead, dl, ni = c
    cell = CELL[(MODEL[model], cr)]
    if cell == "llama_875":
        num("c2g", "llama_875|chunkkv_published", ck)
        num("c2g", "llama_875|leader", lead)
        continue
    num("c2g", f"{cell}|chunkkv", ck)
    num("c2g", f"{cell}|two_grid", tg)
    d, lo, hi = re.match(r"([+\-\u2212]?[\d.]+) \(([+\-\u2212]?[\d.]+), ([+\-\u2212]?[\d.]+)\)", diff.replace("$-$", "-")).groups()
    num("c2g", f"{cell}|diff", d)
    num("c2g", f"{cell}|diff_lo", lo, mc=True)
    num("c2g", f"{cell}|diff_hi", hi, mc=True)
    num("c2g", f"{cell}|leader", lead)
    d, lo, hi = re.match(r"([+\-\u2212]?[\d.]+) \(([+\-\u2212]?[\d.]+), ([+\-\u2212]?[\d.]+)\)", dl).groups()
    num("c2g", f"{cell}|diff_leader", d)
    num("c2g", f"{cell}|diff_leader_lo", lo, mc=True)
    num("c2g", f"{cell}|diff_leader_hi", hi, mc=True)
    put("c2g", f"{cell}|tasks_noninferior", float(ni.split()[0]), 0.5)
m = re.search(r"Uncompressed model: ([\d.]+) \(Qwen3-8B\) and ([\d.]+) \(Llama", blk)
num("c2g", "qwen|uncompressed", m.group(1))
num("c2g", "llama|uncompressed", m.group(2))
# ---- Appendix C per-task tables and cost table
app = tex[tex.index("\\section{Benchmark details}"):]
parts = re.split(r"\\emph\{RULER-4096 query-aware", app)[1:]
for part in parts:
    model = "Qwen3-8B" if "Qwen3-8B" in part.split("\n")[0] else "Llama-3.1-8B-Instruct"
    cr = re.search(r"CR ([\d.]+)", part.split("\n")[0]).group(1)
    cell = CELL[(model, cr)]
    for c in rows_of(part.split("\\end{longtable}")[0]):
        if len(c) != 8 or c[0] == "Task":
            continue
        t = c[0]
        for i, nm in enumerate(["chunkkv", "two_grid", "leader", "delta", "lb", "m"], start=1):
            s = c[i].replace("+", "")
            num("c2g", f"{cell}|{t}|{nm}", s, mc=(nm == "lb"))
        put("c2g", f"{cell}|{t}|noninferior", c[7])
cost = app[app.index("Selection cost per item"):]
for c in rows_of(cost):
    if len(c) != 9 or c[0] == "Model":
        continue
    cell = CELL[(MODEL[c[0]], "0.875" if c[1] == "0.875" else c[1])]
    arm = c[2]
    for i, nm in enumerate(["score_s", "select_s", "prefill_s", "decode_s", "total_s"], start=3):
        num("c2g", f"cost|{cell}|{arm}|{nm}", c[i])
    num("c2g", f"cost|{cell}|{arm}|peak_gb", c[8])

# ---- numbers stated only in running text (location in brackets); values as printed
T = lambda name, key, val, tol: put(name, key, val, tol)
# Results 1 / Table 1 [Results, Block sizes recovered]
T("periodogram", "blocked_entries", 16, 0.5)
T("periodogram", "top_equals_block", 8, 0.5)
T("periodogram", "top_multiple_of_block", 10, 0.5)
T("periodogram", "block_ratio_above_bg_max", 7, 0.5)
T("periodogram", "unblocked_entries", 18, 0.5)
T("periodogram", "unblocked_ratio_min", 3.1, 0.05)
T("periodogram", "unblocked_ratio_max", 8.9, 0.05)
T("periodogram", "compactor_max_ratio", 8.2, 0.05)
T("periodogram", "compactor_max_ratio_at_256", 3.4, 0.05)
T("periodogram", "other_entries_median_ratio_2to64", 4.0, 0.05)
T("periodogram", "fig1a_llama_chunkkv_ratio_2to64", 33.3, 0.05)  # Figure 1a caption
# PB2 [Results, parity law; Appendix A]
T("pb2", "flash_1k|P1_diff_pp", 0.84, 0.005)
T("pb2", "flash_1k|P1_ci_lo_pp", -1.36, 0.005)
T("pb2", "flash_1k|P1_ci_hi_pp", 4.40, 0.005)
T("pb2", "flash_1k|repeat_flip_pct", 3.8, 0.05)
T("pb2", "flash_1k|kept_probes", 99, 0.5)
# pa2 text [Results, two-grid selection on an open model]
for k, d, lo, hi in [("twogrid_minus_chunkkv|cr0.75", 0.223, 0.143, 0.290), ("twogrid_minus_chunkkv|cr0.875", 0.393, 0.335, 0.440),
                     ("moved10_minus_chunkkv|cr0.75", 0.023, -0.023, 0.068), ("moved10_minus_chunkkv|cr0.875", 0.002, -0.028, 0.030),
                     ("third_grid_gain|cr0.875", 0.033, 0.015, 0.055), ("fourth_grid_gain|cr0.875", 0.013, -0.033, 0.048)]:
    T("pa2", k + "|delta", d, 0.0005)
    put("pa2", k + "|lo", lo, 0.0005, True)
    put("pa2", k + "|hi", hi, 0.0005, True)
T("pa2", "weak_offset_shift_corr|cr0.75", 0.98, 0.005)
T("pa2_mechanism", "A1_chunkkv|cr0.875|auc_kv", 0.916, 0.0005)
T("pa2_mechanism", "A1_chunkkv|cr0.875|auc_kv_ci_lo", 0.866, 0.0005)
T("pa2_mechanism", "A1_chunkkv|cr0.875|auc_kv_ci_hi", 0.953, 0.0005)
T("pa2_mechanism", "A1_chunkkv|cr0.875|auc_v", 0.784, 0.0005)
T("pa2_mechanism", "A1_chunkkv|cr0.875|err_in_split_pct", 96.2, 0.05)
T("pa2_mechanism", "twogrid_split_offsets", 4, 0.5)
# pa3 text [Results; Figure 4 caption; Table 6]
T("pa3", "twogrid|cr0.875|mean", 0.838, 0.0005)
T("pa3", "chunkkv|cr0.875|mean", 0.579, 0.0005)
T("pa3", "full_cache_mean", 0.904, 0.0005)
T("pa3", "chunk40|cr0.875|mean", 0.642, 0.0005)
T("pa3", "iso_cr_star", 0.56, 0.005)
T("pa3", "iso_cr_star_ci_lo", 0.50, 0.005)
T("pa3", "iso_cr_star_ci_hi", 0.67, 0.005)
T("pa3", "iso_memory_factor", 3.6, 0.05)
T("pa3", "iso_memory_factor_lo", 2.7, 0.05)
T("pa3", "iso_memory_factor_hi", 4.2, 0.05)
T("pa3", "iso_boot_share_never_reaching_pct", 14.9, 0.05)
T("pa3", "second_grid_overhead_s", 0.10, 0.005)
T("pa3", "prefill_s", 25, 0.5)
T("pa3", "n_items", 12, 0.5)
# PB3 noise controls [Results; Limitations]
for k, e, lo, hi in [("pro|noise_M_K2", 2.8, -10.2, 14.4), ("pro|noise_M_K3", 9.4, -3.3, 20.5)]:
    T("pb3", k + "_excess_pp", e, 0.05)
    put("pb3", k + "_ci_lo_pp", lo, 0.05, True)
    put("pb3", k + "_ci_hi_pp", hi, 0.05, True)
T("pb3", "pro|noise_M_K4_excess_pp", 14.8, 0.05)
T("pb3", "pro|noise_A_KS_excess_pp", 27.3, 0.05)
T("pb3", "pro|noise_B_KS_excess_pp", 46.6, 0.05)
put("pb3", "pro|noise_M_K4_ci_lo_pp", 1.8, 0.05, True)  # lowest lower bound over the three baselines
# C2G text [Results; Appendix A]
for cell, a, b, c3, d_ in [("qwen_75", 0.05, -0.67, -10.60, -4.80), ("llama_75", 0.14, -1.92, -11.38, -9.40)]:
    T("c2g", f"{cell}|rerun_diff_points", a, 0.005)
    T("c2g", f"{cell}|other12_vs_leader", b, 0.005)
    T("c2g", f"{cell}|cwe_vs_leader", c3, 0.005)
    T("c2g", f"{cell}|multikey3_vs_leader", d_, 0.005)
T("c2g", "qwen_875|rerun_diff_points", -0.22, 0.005)
T("c2g", "qwen_875|rerun_tasks", 6, 0.5)
for cell, lo, hi in [("qwen_875", 82, 82), ("qwen_75", 84, 84), ("llama_75", 88, 88)]:
    pass
T("c2g", "qwen_875|rerun_same_answer_pct", 82, 0.5)  # "82% to 88%" over the three cells: ends 82 and 88
T("c2g", "llama_75|rerun_same_answer_pct", 88, 0.5)
for cell in ("qwen_875", "qwen_75", "llama_75"):
    T("c2g", f"{cell}|holm_p1", 4e-5, 5e-6)
T("c2g", "qwen_875|worst_phase_chunkkv", 0.183, 0.0005)
T("c2g", "qwen_875|worst_phase_two_grid", 0.420, 0.0005)
T("c2g", "qwen_75|worst_phase_chunkkv", 0.694, 0.0005)
T("c2g", "qwen_75|worst_phase_two_grid", 0.977, 0.0005)
T("c2g", "qwen_875|worst_phase_chunkkv_at", 10, 0.5)
T("c2g", "qwen_875|worst_phase_two_grid_at", 10, 0.5)
T("c2g", "qwen_875|items_per_phase_min", 47, 0.5)
T("c2g", "qwen_875|items_per_phase_max", 318, 0.5)
T("c2g", "P4|qwen_875|published_ratio20", 31.34, 0.005)
T("c2g", "P4|qwen_875|two_grid_ratio20", 3.29, 0.005)
T("c2g", "P4|qwen_background_max", 5.41, 0.005)
T("c2g", "qwen_75|best_context_only", 95.22, 0.005)
put("c2g", "llama_875|smoke_kill_tasks", "fwe,niah_single_3")
# Table 6 [Discussion]
T("table6", "second_grid_overhead_s_min", 0.03, 0.005)
T("table6", "second_grid_overhead_s_max", 0.06, 0.005)
T("table6", "second_grid_overhead_pct_min", 3, 0.5)
T("table6", "second_grid_overhead_pct_max", 5, 0.5)
T("table6", "ruler_mean_diff_qwen_75", 5.74, 0.005)
T("table6", "ruler_mean_diff_llama_75", 2.01, 0.005)
T("table6", "ruler_mean_diff_qwen_875", 10.27, 0.005)
T("table6", "lower_cr_memory_factor", 3.6, 0.05)
T("table6", "chunk10_cr875", 0.202, 0.0005)
T("table6", "chunk5_cr875", 0.020, 0.0005)
T("table6", "chunk40_cr88", 0.642, 0.0005)
T("table6", "twogrid_cr88", 0.838, 0.0005)
T("table6", "pro_typo_change_reduction_pct", 51.0, 0.05)
# Known deviations: two timing medians of Appendix C (Qwen3-8B, CR 0.875, two-grid arm) differ by 0.001 s in the third decimal
# from the printed table when rebuilt from the released per-item table (0.6095 and 1.1197 against 0.609 and 1.121). The per-item
# timing rows are recorded by the runner; the median depends on which rows count as the first three of a run.
for k in ("cost|qwen_875|two_grid|decode_s", "cost|qwen_875|two_grid|total_s"):
    exp["c2g"][k]["known"] = "median over per-item timings differs by 0.001 s after rounding (order of rows within a run)"
os.makedirs(OUT, exist_ok=True)
for name, d in exp.items():
    json.dump(d, open(os.path.join(OUT, name + ".json"), "w"), indent=1, sort_keys=True)
    print(name, len(d))
