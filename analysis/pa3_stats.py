# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0

# pa3 iso-memory analysis (reference copy, adapted to the derived table). Usage: python pa3_stats.py OUT_PREFIX data/pa3/pa3_outcomes.csv.gz
import sys, json, numpy as np, pandas as pd

out = sys.argv[1]
D = pd.read_csv(sys.argv[2])
D["offset"] = D.needle_start % 20
D["uid"] = D.src + "|" + D["item"].astype(str)
ARMS = ["A1_chunkkv", "A3_twogrid", "A7_chunk10", "A15_chunk40"]
REF_CR = 0.875
FULL = D[D.arm == "full"].copy()
A = D[D.arm != "full"].copy()
CRS = sorted(A.cr.unique())

# complete items: 20 phases for every arm x CR and for the full-cache row
complete = []
for uid, g in A.groupby("uid"):
    cnt = g.groupby(["arm", "cr"]).ctx.nunique()
    full_ok = FULL[FULL.uid == uid].ctx.nunique() == 20
    if full_ok and all(cnt.get((a, c), 0) == 20 for a in ARMS for c in CRS):
        complete.append(uid)
complete = sorted(complete)
n = len(complete)
assert n >= 1, "no complete item"
A = A[A.uid.isin(complete)].copy()
FULL = FULL[FULL.uid.isin(complete)].copy()

# effective CR is constant across arms and contexts (whole-chunk rounding), asserted
eff_rng = A.groupby("cr").eff_cr.agg(["min", "max"])
assert (eff_rng["max"] - eff_rng["min"]).max() < 1e-9, eff_rng
EFF = A.groupby("cr").eff_cr.mean().to_dict()

uid_ix = {u: i for i, u in enumerate(complete)}
arm_ix = {a: i for i, a in enumerate(ARMS)}
cr_ix = {c: i for i, c in enumerate(CRS)}
T = np.full((n, len(ARMS), len(CRS), 20), np.nan)
for r in A.itertuples():
    T[uid_ix[r.uid], arm_ix[r.arm], cr_ix[r.cr], r.offset] = r.correct
F = np.full((n, 20), np.nan)
for r in FULL.itertuples():
    F[uid_ix[r.uid], r.offset] = r.correct
assert not np.isnan(T).any() and not np.isnan(F).any()
assert set(np.unique(T)) <= {0.0, 1.0}

rng = np.random.default_rng(20261025)
NB = 5000
BOOT = [rng.integers(0, n, n) for _ in range(NB)]


def interval(v):
    v = np.asarray(v, float)
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def mean_curve(t):            # [arm, cr]
    return t.mean(axis=(0, 3))


def worst_curve(t):           # [arm, cr]: minimum over the 20 needle offsets of the item-mean accuracy
    return t.mean(axis=0).min(axis=2)


mc, wc = mean_curve(T), worst_curve(T)
bm = np.array([mean_curve(T[b]) for b in BOOT])
bw = np.array([worst_curve(T[b]) for b in BOOT])
res = dict(n_items=n, items=complete, n_ctx_per_arm_cr=int(n * 20), crs=[float(c) for c in CRS],
           eff_cr={str(c): float(EFF[c]) for c in CRS}, full_cache_acc=float(F.mean()))
res["curves"] = {a: {str(c): dict(eff_cr=float(EFF[c]), mean=float(mc[arm_ix[a], cr_ix[c]]), mean_ci=interval(bm[:, arm_ix[a], cr_ix[c]]),
                                  worst=float(wc[arm_ix[a], cr_ix[c]]), worst_ci=interval(bw[:, arm_ix[a], cr_ix[c]])) for c in CRS} for a in ARMS}

# paired contrasts: two-grid minus each other arm at every CR (item-cluster bootstrap)
res["twogrid_minus"] = {}
for a in ARMS:
    if a == "A3_twogrid":
        continue
    res["twogrid_minus"][a] = {}
    for c in CRS:
        d = T[:, arm_ix["A3_twogrid"], cr_ix[c]].mean(axis=1) - T[:, arm_ix[a], cr_ix[c]].mean(axis=1)
        res["twogrid_minus"][a][str(c)] = dict(delta=float(d.mean()), ci=interval([d[b].mean() for b in BOOT]),
                                               items_pos=int((d > 0).sum()), items_neg=int((d < 0).sum()))


def crossing(eff_x, acc_y, target):
    """CR* by linear interpolation in effective CR: the largest CR at which the curve is still at or above the target."""
    x, y = np.asarray(eff_x, float), np.asarray(acc_y, float)
    if y[-1] >= target:
        return float(x[-1]), "at_or_above_grid_max"
    for i in range(len(x) - 1, 0, -1):
        if y[i - 1] >= target > y[i]:
            return float(x[i - 1] + (y[i - 1] - target) / (y[i - 1] - y[i]) * (x[i] - x[i - 1])), "interpolated"
    return float("nan"), "below_grid_min"


eff_list = [float(EFF[c]) for c in CRS]
j1, j3, kref = arm_ix["A1_chunkkv"], arm_ix["A3_twogrid"], cr_ix[REF_CR]
res["iso_score"] = {}
for name, fn in (("mean", mean_curve), ("worst_offset", worst_curve)):
    cv = fn(T)
    target = float(cv[j3, kref])
    cs, how = crossing(eff_list, cv[j1], target)
    shares = {"at_or_above_grid_max": 0, "interpolated": 0, "below_grid_min": 0}
    vals = []
    for b in BOOT:
        cb = fn(T[b])
        v, h = crossing(eff_list, cb[j1], cb[j3, kref])
        shares[h] += 1
        if h == "interpolated":
            vals.append(v)
    entry = dict(target_acc=target, ref_cr_eff=float(EFF[REF_CR]), chunkkv_at_ref=float(cv[j1, kref]), cr_star=cs, how=how,
                 boot_share=shares, cr_star_boot_ci_interpolated_only=interval(vals) if vals else None,
                 chunkkv_at_grid={str(c): float(cv[j1, cr_ix[c]]) for c in CRS})
    if name == "mean":
        base, full = float(cv[j1, kref]), float(F.mean())
        entry["recovery_at_ref"] = float((target - base) / (full - base)) if full > base else None
    res["iso_score"][name] = entry

# grid-based match: ChunkKV at each CR against two-grid at the reference CR, paired by item
tg = T[:, j3, kref].mean(axis=1)
res["chunkkv_vs_twogrid_ref"] = {}
for c in CRS:
    d = T[:, j1, cr_ix[c]].mean(axis=1) - tg
    ci_ = interval([d[b].mean() for b in BOOT])
    res["chunkkv_vs_twogrid_ref"][str(c)] = dict(diff=float(d.mean()), ci=ci_, within_ci_or_above=bool(ci_[1] >= 0))

# mechanism with the corrected span: fully kept key+value span (13 tokens) and value span (7 digits)
M = A.copy()
M["kv_all"] = (M.kv_full >= 0.9999).astype(int)
M["val_all"] = (M.val_full >= 0.9999).astype(int)
rows = []
for (arm, cr), g in M.groupby(["arm", "cr"]):
    rows.append(dict(arm=arm, cr=cr, acc=g.correct.mean(), p_kv_full_all_layers=g.kv_all.mean(), p_val_full_all_layers=g.val_all.mean(),
                     acc_given_kv_full=g.correct[g.kv_all == 1].mean() if (g.kv_all == 1).any() else np.nan,
                     acc_given_kv_not_full=g.correct[g.kv_all == 0].mean() if (g.kv_all == 0).any() else np.nan,
                     acc_given_val_full=g.correct[g.val_all == 1].mean() if (g.val_all == 1).any() else np.nan,
                     acc_given_val_not_full=g.correct[g.val_all == 0].mean() if (g.val_all == 0).any() else np.nan))
pd.DataFrame(rows).to_csv(out + "_mechanism_new_span.csv", index=False)

# cost from the sweep: selection time summed over layers (contexts without real-press verification), decode per generated token
A["dec_per_tok"] = A.t_dec / A.n_gen.clip(lower=1)
Q = A[A.verified_ctx == 0]
cost = Q.groupby(["arm", "cr"]).agg(t_sel_median_s=("t_sel", "median"), t_sel_q1=("t_sel", lambda x: x.quantile(0.25)),
                                    t_sel_q3=("t_sel", lambda x: x.quantile(0.75)), n_ctx=("t_sel", "size")).reset_index()
cost["dec_per_token_median_s"] = [A[(A.arm == r.arm) & (A.cr == r.cr)].dec_per_tok.median() for r in cost.itertuples()]
cost.to_csv(out + "_cost_from_sweep.csv", index=False)
ref1 = Q[(Q.arm == "A1_chunkkv") & (Q.cr == REF_CR)]
res["ctx_level"] = dict(t_score_median_s=float(ref1.t_score.median()), t_prefill_recorded_median_s=float(ref1.t_prefill_rec.median()), n_ctx=int(len(ref1)),
                        load1_median=float(A.load1.median()), load1_max=float(A.load1.max()))

tab = [dict(arm=a, cr_nominal=float(c), cr_eff=float(EFF[c]), acc_mean=e["mean"], acc_mean_lo=e["mean_ci"][0], acc_mean_hi=e["mean_ci"][1],
            acc_worst_offset=e["worst"], worst_lo=e["worst_ci"][0], worst_hi=e["worst_ci"][1])
       for a in ARMS for c in CRS for e in [res["curves"][a][str(c)]]]
pd.DataFrame(tab).to_csv(out + "_iso_memory_table.csv", index=False)
pd.DataFrame([dict(arm=a, cr=float(c), offset=o, acc=float(T[:, arm_ix[a], cr_ix[c], o].mean())) for a in ARMS for c in CRS for o in range(20)]).to_csv(out + "_per_offset.csv", index=False)
json.dump(res, open(out + "_iso_results.json", "w"), indent=1)
print(json.dumps(dict(n_items=n, iso={k: {kk: v[kk] for kk in ("target_acc", "chunkkv_at_ref", "cr_star", "how")} for k, v in res["iso_score"].items()})))
