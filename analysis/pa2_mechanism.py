# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""Mechanism of the grid effect (Results, "Two-grid selection on an open model"): does the 13-token key-and-value span sit intact inside
at least one kept chunk? Reads data/pa2/pa2_outcomes.csv.gz (run pa2, 30 complete items x 20 offsets). AUC of the longest span fraction
kept intact against a correct answer, item-cluster bootstrap (400 resamples, seed 7), and the share of errors in which every grid splits the span.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import DATA, OUT, write_result  # noqa: E402

R = pd.read_csv(f"{DATA}/pa2/pa2_outcomes.csv.gz")
R = R[R.arm != "full"].copy()
cnt = R.groupby("item").phase_needle.nunique()
complete = cnt[cnt == 20].index
R = R[R.item.isin(complete)].copy()
R["correct"] = R.correct.astype(int)
assert len(complete) == 30 and (R.needle_start % 20 == R.phase_needle).all()

GR = {"A1_chunkkv": (20, (0,)), "A3_twogrid": (20, (0, 10)), "A7_chunk10": (10, (0,)), "A8_chunk5": (5, (0,)),
      "A9_twogrid10": (10, (0, 5)), "A10_threegrid20": (20, (0, 7, 14)), "A11_token_shared": (1, (0,)),
      "A12_chunk20_off10": (20, (10,)), "A13_fourgrid20": (20, (0, 5, 10, 15)), "A14_twogrid20_0_5": (20, (0, 5))}

def split_on(a, b, c, o):
    first = a + 1; k = -((o - first) // c); return (o + c * k) < b

spans = {'val7': (13, 20), 'keyval13': (7, 20), 'needle21': (0, 21)}
for nm, (lo, hi) in spans.items():
    R['split_all_' + nm] = [all(split_on(s + lo, s + hi, GR[a][0], o) for o in GR[a][1]) for s, a in zip(R.needle_start, R.arm)]

def maxin(s, arm, lo, hi):
    c, offs = GR[arm]; a, b = s + lo, s + hi; best = 0
    for o in offs:
        j0 = (a - o) // c
        for j in range(j0, j0 + 3):
            lo_c = o + c * j; best = max(best, max(0, min(b, lo_c + c) - max(a, lo_c)))
    return best

for nm, (lo, hi) in {'kv': (7, 20), 'v': (13, 20), 'n': (0, 21)}.items():
    R['maxin_' + nm] = [maxin(s, a, lo, hi) for s, a in zip(R.needle_start, R.arm)]

rng = np.random.default_rng(7)

def auc(score, y):
    y = np.asarray(y); s = np.asarray(score, float)
    if y.min() == y.max(): return np.nan
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

items_all = sorted(R.item.unique())

def cluster_boot(g, fn, B=1000):
    by = {i: d for i, d in g.groupby('item')}; ks = list(by); v = []
    for _ in range(B):
        pick = rng.choice(len(ks), len(ks)); d = pd.concat([by[ks[p]] for p in pick]); v.append(fn(d))
    v = np.array(v, float); return np.nanpercentile(v, [2.5, 97.5])

rowsm = []
for arm in ['A1_chunkkv', 'A12_chunk20_off10', 'A14_twogrid20_0_5', 'A3_twogrid', 'A7_chunk10', 'A9_twogrid10', 'A8_chunk5']:
    for cr in (0.75, 0.875):
        g = R[(R.arm == arm) & (R.cr == cr)].copy()
        errs = (g.correct == 0)
        r = dict(arm=arm, cr=cr, acc=g.correct.mean(), err_n=int(errs.sum()))
        r['full_cache_gap'] = 0.945 - g.correct.mean()
        for nm in ('val7', 'keyval13'):
            s = g['split_all_' + nm]; r['err_in_split_' + nm] = float((errs & s).sum() / max(1, errs.sum()))
        for nm in ('v', 'kv'):
            r['auc_maxin_' + nm] = auc(g['maxin_' + nm], g.correct)
            lo, hi = cluster_boot(g, lambda d: auc(d['maxin_' + nm], d.correct), 400)
            r['auc_maxin_' + nm + '_ci'] = (round(lo, 3), round(hi, 3))
        vk = g.val_kept
        r['P_valkept1'] = float((vk >= 0.9999).mean()); r['acc|valkept=1'] = g.correct[vk >= 0.9999].mean() if (vk >= 0.9999).any() else np.nan
        r['acc|valkept<1'] = g.correct[vk < 0.9999].mean() if (vk < 0.9999).any() else np.nan
        r['share_correct_with_valkept1'] = float(((vk >= 0.9999) & (g.correct == 1)).sum() / max(1, (g.correct == 1).sum()))
        r['auc_valkept'] = auc(vk, g.correct)
        rowsm.append(r)

M = pd.DataFrame(rowsm)
M.to_csv(f"{OUT}/pa2_mechanism_predictors.csv", index=False)
v = {}
for _, r in M.iterrows():
    k = f"{r.arm}|cr{r.cr}"
    v[k + "|auc_kv"] = round(r.auc_maxin_kv, 3)
    v[k + "|auc_kv_ci_lo"] = r.auc_maxin_kv_ci[0]
    v[k + "|auc_kv_ci_hi"] = r.auc_maxin_kv_ci[1]
    v[k + "|auc_v"] = round(r.auc_maxin_v, 3)
    v[k + "|err_in_split_pct"] = round(100 * r.err_in_split_keyval13, 1)
# offsets at which both grids of the two-grid arm split the 13-token span
g = R[(R.arm == "A3_twogrid") & (R.cr == 0.875)]
v["twogrid_split_offsets"] = int(g.groupby("phase_needle").split_all_keyval13.first().sum())
write_result("pa2_mechanism", v)
print(M.round(3).to_string())
print(v["twogrid_split_offsets"])