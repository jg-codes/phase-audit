# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""Tables 1 and 7: periodogram audit of the published kvpress RULER-4096 predictions.

Statistic: for each leaderboard entry, items solved by the uncompressed model; the correctness indicator is regressed on
task, key length and a 40-df B-spline of needle position; two harmonics (sin/cos of 2*pi*pos/P and of twice that) at period P
are tested as an F statistic (4 df). Ratio = F(P) / median F over periods 2..260. Entries with accuracy outside (0.03, 0.97)
are not estimable. Reads data/ only.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from patsy import dmatrix  # noqa: E402

from common import DATA, OUT, write_result  # noqa: E402

PERIODS = list(range(2, 261))
BLOCK = {"chunkkv__Q": 20, "cur": 16, "lagkv": 128, "adakv_compactor": 256}
NAME = {"chunkkv__Q": "ChunkKV", "cur": "CUR", "lagkv": "LagKV", "adakv_compactor": "Compactor", "knorm": "Knorm",
        "snapkv__Q": "SnapKV (query-aware)", "streaming_llm": "StreamingLLM"}


def audit():
    pos = pd.read_csv(f"{DATA}/ruler_needle_positions.csv")
    lb = pd.read_csv(f"{DATA}/leaderboard_item_correct.csv.gz")
    rows = []
    for model in ["Qwen3-8B", "Llama-3.1-8B-Instruct"]:
        base = lb[(lb.model == model) & (lb.press == "no_press")].set_index("item").y
        S = pos[(pos.model == model) & pos.item.isin(base.index[base == 1])].sort_values("item").reset_index(drop=True)
        X0 = np.asarray(dmatrix("C(task)+ntok+bs(pos,df=40)", S, return_type="dataframe"))
        Q0, _ = np.linalg.qr(X0)
        n, k0 = X0.shape
        FP = {}
        for P in PERIODS:
            w = 2 * np.pi * S.pos.values / P
            F = np.column_stack([np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])
            F = F - Q0 @ (Q0.T @ F)
            FP[P] = np.linalg.qr(F)[0]
        for (press, cr), g in lb[(lb.model == model) & (lb.press != "no_press")].groupby(["press", "cr"]):
            y = g.set_index("item").y.reindex(S.item.values).to_numpy().astype(float)
            acc = y.mean()
            if not (0.03 < acc < 0.97):
                continue
            r = y - Q0 @ (Q0.T @ y)
            rss0 = r @ r
            df2 = n - k0 - 4
            Fs = pd.Series({P: (np.sum((FP[P].T @ r) ** 2) / 4) / ((rss0 - np.sum((FP[P].T @ r) ** 2)) / df2) for P in PERIODS})
            med = Fs.median()
            rows.append(dict(model=model, press=press, cr=cr, acc=acc, n=n, top=int(Fs.idxmax()), ratio_top=Fs.max() / med,
                             ratio_block=(Fs[BLOCK[press]] / med if press in BLOCK else np.nan)))
    return pd.DataFrame(rows)


def main():
    T = audit()
    T.to_csv(f"{OUT}/periodogram_entries.csv", index=False)
    v = {}
    for _, r in T.iterrows():
        k = f"{r.model}|{NAME[r.press]}|{r.cr:.2f}"
        v[k + "|acc"] = round(r.acc, 3)
        v[k + "|top"] = r.top
        v[k + "|ratio_top"] = round(r.ratio_top, 1)
        if r.press in BLOCK:
            v[k + "|ratio_block"] = round(r.ratio_block, 1)
    B = T[T.press.isin(BLOCK)]
    v["blocked_entries"] = len(B)
    v["top_equals_block"] = int((B.top == B.press.map(BLOCK)).sum())
    v["top_multiple_of_block"] = int((B.top % B.press.map(BLOCK) == 0).sum())
    v["block_ratio_above_bg_max"] = int((B.ratio_block > T[~T.press.isin(BLOCK)].ratio_top.max()).sum())
    U = T[~T.press.isin(BLOCK)]
    v["unblocked_entries"] = len(U)
    v["unblocked_ratio_min"] = round(U.ratio_top.min(), 1)
    v["unblocked_ratio_max"] = round(U.ratio_top.max(), 1)
    C = T[T.press == "adakv_compactor"]
    v["compactor_max_ratio"] = round(C.ratio_top.max(), 1)
    v["compactor_max_ratio_at_256"] = round(C.ratio_block.max(), 1)
    # wider leaderboard: entries of the other presses, periods 2..64 (derived F table), and the Figure 1a ratio
    F64 = pd.read_csv(f"{DATA}/periodogram_F_periods2to64.csv")
    ent = F64.groupby(["model", "press", "cr"]).F.agg(lambda f: f.max() / f.median()).reset_index(name="ratio")
    oth = ent[~ent.press.isin(list(BLOCK))]
    v["other_entries_2to64"] = len(oth)
    v["other_entries_median_ratio_2to64"] = round(oth.ratio.median(), 1)
    d = F64[(F64.press == "chunkkv__Q") & (F64.model == "Llama-3.1-8B-Instruct") & (F64.cr == 0.88)]
    v["fig1a_llama_chunkkv_ratio_2to64"] = round(d.F.max() / d.F.median(), 1)
    write_result("periodogram", v)
    print(T.round(2).to_string())
    print({k: x for k, x in v.items() if "|" not in k})


if __name__ == "__main__":
    main()
