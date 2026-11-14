# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""Table 5, Appendix C (per-task scores, selection cost), Appendix A numbers for C2G, and Figure 5: registered benchmark test C2G.

Statistics are the registered ones (c2g_stats.py: item-level sign-flip test with 100,000 flips, task-stratified paired bootstrap with
10,000 resamples, fixed seeds, per-task non-inferiority margins frozen from the 15 smoke items per task). Reads data/ only.
Bootstrap interval ends are deterministic for the registered seeds.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from patsy import dmatrix  # noqa: E402

import c2g_stats as cs  # noqa: E402
from common import DATA, FIG, OUT, panel_letter, plot_style, write_result  # noqa: E402

CELLS = {"qwen_875": ("Qwen3-8B", 0.875, "0.88"), "qwen_75": ("Qwen3-8B", 0.75, "0.75"), "llama_75": ("Llama-3.1-8B-Instruct", 0.75, "0.75")}
TASKS = cs.TASKS


def mean13(df, col):
    return float(np.mean([df.loc[df.task == t, col].mean() for t in TASKS]))


def ratio_at_20(pos, y0, ys, periods=range(2, 65)):
    """Periodogram ratio F(20) / median F over periods 2..64 for each correctness vector (dict name -> Series item -> 0/1)."""
    S = pos[pos.item.isin(y0.index[y0 == 1])].sort_values("item").reset_index(drop=True)
    X0 = np.asarray(dmatrix("C(task)+ntok+bs(pos,df=40)", S, return_type="dataframe"))
    Q0, _ = np.linalg.qr(X0)
    n, k0 = X0.shape
    FP = {}
    for P in periods:
        w = 2 * np.pi * S.pos.values / P
        F = np.column_stack([np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])
        F = F - Q0 @ (Q0.T @ F)
        FP[P] = np.linalg.qr(F)[0]
    out = {}
    for name, ysr in ys.items():
        y = ysr.reindex(S.item.values).to_numpy().astype(float)
        acc = y.mean()
        if not (0.03 < acc < 0.97):
            out[name] = (acc, None, None)
            continue
        r = y - Q0 @ (Q0.T @ y)
        rss0 = r @ r
        Fs = pd.Series({P: (np.sum((FP[P].T @ r) ** 2) / 4) / ((rss0 - np.sum((FP[P].T @ r) ** 2)) / (n - k0 - 4)) for P in periods})
        out[name] = (acc, Fs[20] / Fs.median(), Fs.max() / Fs.median())
    return out


def main():
    F = pd.read_csv(f"{DATA}/c2g/c2g_item_scores.csv.gz")
    S = pd.read_csv(f"{DATA}/c2g/c2g_smoke_item_scores.csv.gz")
    LB = pd.read_csv(f"{DATA}/leaderboard_item_scores_13task.csv.gz")
    SOTA = pd.read_csv(f"{DATA}/leaderboard_sota_rows.csv")
    pos = pd.read_csv(f"{DATA}/ruler_needle_positions.csv")
    lbc = pd.read_csv(f"{DATA}/leaderboard_item_correct.csv.gz")
    v, A = {}, {}
    pvals = {}
    for cell, (mname, cr, crtag) in CELLS.items():
        T = F[(F.cell == cell) & (F.arm == "two_grid")].sort_values("item_id")
        assert len(T) == 6500
        T = cs.frame(T.to_dict("records"))
        pub = LB[(LB.model == mname) & (LB.entry == "chunkkv") & (LB.cr == float(crtag))].sort_values("item")
        assert np.allclose(T.sort_values("item_id").pub_score.values, pub.score.values, atol=1e-3), "published scores differ from leaderboard rescoring"
        lead = LB[(LB.model == mname) & (LB.entry == "leader_adakv_dropkv") & (LB.cr == float(crtag))].sort_values("item").score.to_numpy()
        sm = S[(S.cell == cell) & (S.arm == "two_grid")].copy()
        sm["delta"] = sm.score - sm.pub_score
        marg = cs.margins_from_smoke(sm)
        a = dict(P1=cs.p1_cell(T), P2=cs.p2_cell(T, marg), P3=cs.p3_cell(T, lead))
        a["two_grid"], a["published"] = mean13(T, "score"), mean13(T, "pub_score")
        a["leader"] = float(np.mean([lead[(T.sort_values("item_id").task.values == t)].mean() for t in TASKS]))
        A[cell] = a
        pvals[cell] = a["P1"]["p_signflip"]
    # Holm over the four cells (the fourth, Llama CR 0.875, was stopped at the smoke stage: p = 1)
    pvals["llama_875"] = 1.0
    holm = cs.holm(pvals)
    for cell, a in A.items():
        k = cell
        v[f"{k}|chunkkv"] = round(a["published"], 2)
        v[f"{k}|two_grid"] = round(a["two_grid"], 2)
        v[f"{k}|diff"] = round(a["P1"]["mean13_delta"], 2)
        v[f"{k}|diff_lo"] = round(a["P1"]["boot_lo"], 2)
        v[f"{k}|diff_hi"] = round(a["P1"]["boot_hi"], 2)
        v[f"{k}|leader"] = round(a["leader"], 2)
        v[f"{k}|diff_leader"] = round(a["P3"]["mean13_delta"], 2)
        v[f"{k}|diff_leader_lo"] = round(a["P3"]["lo"], 2)
        v[f"{k}|diff_leader_hi"] = round(a["P3"]["hi"], 2)
        v[f"{k}|tasks_noninferior"] = a["P2"]["_n_pass"]
        v[f"{k}|holm_p1"] = float(f"{holm[k]:.2g}")
        v[f"{k}|n_items"] = a["P1"]["n"]
        for t in TASKS:
            p2 = a["P2"][t]
            Tt = F[(F.cell == k) & (F.arm == "two_grid") & (F.task == t)]
            lt = LB[(LB.model == CELLS[k][0]) & (LB.entry == "leader_adakv_dropkv") & (LB.cr == float(CELLS[k][2])) & (LB.task == t)].score.mean()
            v[f"{k}|{t}|chunkkv"] = round(Tt.pub_score.mean(), 2)
            v[f"{k}|{t}|two_grid"] = round(Tt.score.mean(), 2)
            v[f"{k}|{t}|leader"] = round(lt, 2)
            v[f"{k}|{t}|delta"] = round(p2["mean"], 2)
            v[f"{k}|{t}|lb"] = round(p2["lb95"], 2)
            v[f"{k}|{t}|m"] = round(p2["m"], 2)
            v[f"{k}|{t}|noninferior"] = "yes" if p2["noninferior"] else ("NO (kill)" if p2["kill"] else "NO")
        # leader contrasts used in the text
        lead = LB[(LB.model == CELLS[k][0]) & (LB.entry == "leader_adakv_dropkv") & (LB.cr == float(CELLS[k][2]))]
        ours = F[(F.cell == k) & (F.arm == "two_grid")]
        pt = {t: ours[ours.task == t].score.mean() - lead[lead.task == t].score.mean() for t in TASKS}
        v[f"{k}|other12_vs_leader"] = round(float(np.mean([pt[t] for t in TASKS if t != "niah_single_3"])), 2)
        v[f"{k}|cwe_vs_leader"] = round(pt["cwe"], 2)
        v[f"{k}|multikey3_vs_leader"] = round(pt["niah_multikey_3"], 2)
        # rerun of the published press in the local environment
        R = F[(F.cell == k) & (F.arm == "chunkkv_repro")]
        if len(R):
            tasks_done = sorted(R.task.unique())
            d = np.mean([R[R.task == t].score.mean() - R[R.task == t].pub_score.mean() for t in tasks_done])
            v[f"{k}|rerun_diff_points"] = round(float(d), 2)
            v[f"{k}|rerun_tasks"] = len(tasks_done)
            v[f"{k}|rerun_same_answer_pct"] = round(100 * R.same_answer.mean(), 0)
    # Llama CR 0.875: smoke-stage kill rule
    sm = S[(S.cell == "llama_875") & (S.arm == "two_grid")].copy()
    sm["delta"] = sm.score - sm.pub_score
    fired = [t for t, o in cs.kill_check_smoke(sm).items() if o["fired"]]
    v["llama_875|smoke_kill_tasks"] = ",".join(sorted(fired))
    v["llama_875|chunkkv_published"] = round(float(SOTA[(SOTA.model == "meta-llama/Llama-3.1-8B-Instruct") & (SOTA.cr_nominal == 0.875) & (SOTA.row_type == "chunkkv")].mean13.iloc[0]), 2)
    v["llama_875|leader"] = round(float(SOTA[(SOTA.model == "meta-llama/Llama-3.1-8B-Instruct") & (SOTA.cr_nominal == 0.875) & (SOTA.row_type == "best_mean13_in_track") & (SOTA.track == "query-aware")].mean13.iloc[0]), 2)
    for mname, short in (("Qwen/Qwen3-8B", "qwen"), ("meta-llama/Llama-3.1-8B-Instruct", "llama")):
        v[f"{short}|uncompressed"] = round(float(SOTA[(SOTA.model == mname) & (SOTA.row_type == "reference_no_press")].mean13.iloc[0]), 2)
    q = SOTA[(SOTA.model == "Qwen/Qwen3-8B") & (SOTA.cr_nominal == 0.75) & (SOTA.row_type == "best_mean13_in_track") & (SOTA.track == "context-only")].iloc[0]
    v["qwen_75|best_context_only"] = round(float(q.mean13), 2)
    # selection cost: medians per run, first three items of each run dropped
    cost = []
    for (cell, arm), g in F.groupby(["cell", "arm"]):
        g = g.groupby("run_id", sort=False).apply(lambda x: x.iloc[3:], include_groups=False).reset_index(drop=True) if False else pd.concat([x.iloc[3:] for _, x in g.groupby("run_id", sort=False)])
        cost.append(dict(cell=cell, arm=arm, **{c: g[c].median() for c in ["score_s", "select_s", "prefill_s", "decode_s", "total_s", "peak_mem_gb"]}))
    cost = pd.DataFrame(cost)
    cost.to_csv(f"{OUT}/c2g_cost.csv", index=False)
    for _, r in cost.iterrows():
        for c in ["score_s", "select_s", "prefill_s", "decode_s", "total_s"]:
            v[f"cost|{r.cell}|{r.arm}|{c}"] = round(r[c], 3)
        v[f"cost|{r.cell}|{r.arm}|peak_gb"] = round(r.peak_mem_gb, 2)
    # phase table (needle position modulo 20) and P4 periodogram ratio, six needle tasks, items solved without compression
    pq = pos[pos.model == "Qwen3-8B"]
    y0 = lbc[(lbc.model == "Qwen3-8B") & (lbc.press == "no_press")].set_index("item").y
    for cell, crl in (("qwen_875", 0.88), ("qwen_75", 0.75)):
        Tg = F[(F.cell == cell) & (F.arm == "two_grid")].set_index("item_id").score
        yt = (Tg.reindex(pq.item.values) >= 100 - 1e-9).astype(int)
        yt.index = pq.item.values
        yp = lbc[(lbc.model == "Qwen3-8B") & (lbc.press == "chunkkv__Q") & (lbc.cr == crl)].set_index("item").y
        Sx = pq[pq.item.isin(y0.index[y0 == 1])].sort_values("item").reset_index(drop=True)
        ph = (Sx.pos.values % 20).astype(int)
        tabs = {}
        for nm, ys in (("chunkkv", yp), ("two_grid", yt)):
            yv = ys.reindex(Sx.item.values).to_numpy().astype(float)
            tabs[nm] = np.array([yv[ph == p].mean() for p in range(20)])
            v[f"{cell}|worst_phase_{nm}"] = round(float(tabs[nm].min()), 3)
            v[f"{cell}|worst_phase_{nm}_at"] = int(tabs[nm].argmin())
        v[f"{cell}|items_per_phase_min"] = int(min((ph == p).sum() for p in range(20)))
        v[f"{cell}|items_per_phase_max"] = int(max((ph == p).sum() for p in range(20)))
        pd.DataFrame(dict(phase=range(20), n=[int((ph == p).sum()) for p in range(20)], published_chunkkv=tabs["chunkkv"], two_grid=tabs["two_grid"])).to_csv(f"{OUT}/c2g_phase_{cell}.csv", index=False)
    ys = {"published_chunkkv": lbc[(lbc.model == "Qwen3-8B") & (lbc.press == "chunkkv__Q") & (lbc.cr == 0.88)].set_index("item").y,
          "two_grid": (F[(F.cell == "qwen_875") & (F.arm == "two_grid")].set_index("item_id").score >= 100 - 1e-9).astype(int)}
    for press in ("knorm", "snapkv__Q", "streaming_llm"):
        for crl in (0.75, 0.88):
            ys[f"bg|{press}|{crl}"] = lbc[(lbc.model == "Qwen3-8B") & (lbc.press == press) & (lbc.cr == crl)].set_index("item").y
    rr = ratio_at_20(pq, y0, ys)
    v["P4|qwen_875|published_ratio20"] = round(rr["published_chunkkv"][1], 2)
    v["P4|qwen_875|two_grid_ratio20"] = round(rr["two_grid"][1], 2)
    v["P4|qwen_background_max"] = round(max(r[2] for k, r in rr.items() if k.startswith("bg|") and r[2] is not None), 2)
    write_result("c2g", v)
    pd.DataFrame([dict(k=k, v=x) for k, x in v.items()]).to_csv(f"{OUT}/c2g_values.csv", index=False)
    for k in sorted(v):
        if k.count("|") <= 1 or k.startswith("P4"):
            print(k, v[k])
    # ---- Figure 5 ----
    plot_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    FOC, GREY, BLUE = "#D55E00", "#7A7A7A", "#0072B2"
    cells3 = [("qwen_875", "Qwen3-8B, CR 0.875"), ("qwen_75", "Qwen3-8B, CR 0.75"), ("llama_75", "Llama-3.1-8B, CR 0.75")]
    tasks_plot = sorted(TASKS, key=lambda t: A["qwen_875"]["P2"][t]["mean"])
    mk = ["o", "s", "^"]
    fig = plt.figure(figsize=(7.6, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.6, wspace=0.42)
    axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    for i, (cell, lab) in enumerate(cells3):
        a = A[cell]
        axa.plot([i - 0.2], [a["published"]], "o", color=GREY, ms=6)
        axa.plot([i], [a["leader"]], "s", color=BLUE, ms=6)
        axa.errorbar([i + 0.2], [a["two_grid"]], yerr=[[a["P1"]["mean13_delta"] - a["P1"]["boot_lo"]], [a["P1"]["boot_hi"] - a["P1"]["mean13_delta"]]], fmt="D", color=FOC, ms=6, capsize=2)
        npm = v["qwen_uncompressed" if False else ("qwen|uncompressed" if "qwen" in cell else "llama|uncompressed")]
        axa.hlines(npm, i - 0.33, i + 0.33, color="k", lw=0.8, ls=":")
    axa.set_xticks(range(3))
    axa.set_xticklabels(["Qwen3-8B\nCR 0.875", "Qwen3-8B\nCR 0.75", "Llama-3.1-8B\nCR 0.75"])
    axa.set_ylim(55, 101)
    axa.set_xlim(-0.5, 2.5)
    axa.set_ylabel("mean of 13 tasks (higher = better)")
    axa.set_title("Two-grid beats ChunkKV in all three cells run")
    for i, (cell, lab) in enumerate(cells3):
        for j, t in enumerate(tasks_plot):
            p2 = A[cell]["P2"][t]
            y = j + (i - 1) * 0.22
            axb.plot([p2["lb95"], p2["mean"]], [y, y], color=FOC, lw=0.8, alpha=0.6)
            axb.plot([p2["mean"]], [y], mk[i], color=FOC, ms=3.5, mfc=(FOC if p2["noninferior"] else "white"))
    axb.axvline(0, color="k", lw=0.6)
    axb.set_yticks(range(13))
    axb.set_yticklabels([t.replace("niah_", "n.") for t in tasks_plot])
    axb.set_ylim(-0.7, 12.7)
    axb.set_xlabel("two-grid minus ChunkKV (score points)")
    axb.set_title("Gains differ by task; cwe and vt lose at CR 0.875 only")
    axb.legend(handles=[Line2D([0], [0], marker=mk[i], color=FOC, lw=0, ms=4, label=cells3[i][1]) for i in range(3)] + [Line2D([0], [0], marker="o", mfc="white", color=FOC, lw=0, ms=4, label="fails non-inferiority")], loc="lower right", fontsize=6)
    tab = pd.read_csv(f"{OUT}/c2g_phase_qwen_875.csv")
    axc.plot(tab.phase, tab.published_chunkkv, "-o", color=GREY, ms=3)
    axc.plot(tab.phase, tab.two_grid, "-D", color=FOC, ms=3)
    axc.set_xticks(range(0, 20, 2))
    axc.set_xlabel("needle position mod 20 tokens (chunk phase)")
    axc.set_ylabel("share of items still correct")
    axc.set_ylim(0, 1.05)
    axc.set_title("Qwen3-8B, CR 0.875: ChunkKV accuracy dips at several chunk phases; two-grid raises most of them")
    axc.text(5.2, 0.93, "two-grid ChunkKV", color=FOC)
    axc.text(12.2, 0.22, "ChunkKV (published)", color=GREY)
    axa.legend(handles=[Line2D([0], [0], marker="o", color=GREY, lw=0, ms=5, label="ChunkKV (published)"), Line2D([0], [0], marker="s", color=BLUE, lw=0, ms=5, label="best published (AdaKV-DropKV)"),
                        Line2D([0], [0], marker="D", color=FOC, lw=0, ms=5, label="two-grid ChunkKV, 95% CI"), Line2D([0], [0], color="k", ls=":", label="no compression")], loc="lower right", fontsize=6.5)
    for ax, ltr in ((axa, "a"), (axb, "b"), (axc, "c")):
        panel_letter(ax, ltr)
    fig.savefig(f"{FIG}/fig5_c2g.png", dpi=200)


if __name__ == "__main__":
    main()
