# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""Reference copy of the pa2 analysis script (statistics of the exploratory run pa2 (no pre-registration)), adapted to read the derived outcome table
data/pa2/pa2_outcomes.csv.gz instead of the runner's row file.
pa2 analysis. Same statistics as pa1_analysis.py (3-harmonic peak-to-peak with item fixed effects and a linear phase trend,
phase-mean range, within-item permutation tests, item-cluster bootstrap, worst-phase accuracy), re-implemented on the balanced
item x phase matrix so that 2000 permutations and 2000 resamples run in seconds. `--selfcheck` asserts equality with the pa1
least-squares implementation on random data. Additions requested after the pa1 review: level-free harmonic endpoints (odd =
sqrt(a1^2 + a3^2); a2 = period-10 diagnostic; a4 = period-5; phase variance share), arm-vs-arm contrasts with item-cluster AND
haystack-cluster bootstrap, per-haystack results, native-period range, grid-shift check."""
import argparse, json
import numpy as np, pandas as pd

S = 20
ARM_LABEL = {"A1_chunkkv": "1 ChunkKV c20 (reference)", "A3_twogrid": "3 two-grid c20 {0,10}", "A7_chunk10": "7 ChunkKV c10",
             "A8_chunk5": "8 ChunkKV c5", "A9_twogrid10": "9 two-grid c10 {0,5}", "A10_threegrid20": "10 three-grid c20 {0,7,14}",
             "A11_token_shared": "11 token-level (shared score)", "A12_chunk20_off10": "12 ChunkKV c20 offset 10 (shift control)",
             "A13_fourgrid20": "13 four-grid c20 {0,5,10,15}", "A14_twogrid20_0_5": "14 two-grid c20 {0,5} (asymmetric)"}
GRAN = {"A1_chunkkv": 20, "A3_twogrid": 10, "A7_chunk10": 10, "A8_chunk5": 5, "A9_twogrid10": 5, "A10_threegrid20": 20 / 3,
        "A11_token_shared": 1, "A12_chunk20_off10": 20, "A13_fourgrid20": 5, "A14_twogrid20_0_5": 10}
PERIOD = {"A1_chunkkv": 20, "A3_twogrid": 10, "A7_chunk10": 10, "A8_chunk5": 5, "A9_twogrid10": 5, "A10_threegrid20": 20,
          "A11_token_shared": 20, "A12_chunk20_off10": 20, "A13_fourgrid20": 5, "A14_twogrid20_0_5": 20}
PAIRS = [("A3_twogrid", "A7_chunk10"), ("A3_twogrid", "A8_chunk5"), ("A9_twogrid10", "A8_chunk5"), ("A10_threegrid20", "A8_chunk5"),
         ("A9_twogrid10", "A7_chunk10"), ("A3_twogrid", "A1_chunkkv"), ("A7_chunk10", "A1_chunkkv"), ("A8_chunk5", "A1_chunkkv"),
         ("A11_token_shared", "A8_chunk5"), ("A12_chunk20_off10", "A1_chunkkv"), ("A9_twogrid10", "A1_chunkkv"),
         ("A10_threegrid20", "A1_chunkkv"), ("A13_fourgrid20", "A8_chunk5"), ("A13_fourgrid20", "A3_twogrid"),
         ("A13_fourgrid20", "A1_chunkkv"), ("A14_twogrid20_0_5", "A1_chunkkv"), ("A14_twogrid20_0_5", "A3_twogrid"),
         ("A9_twogrid10", "A13_fourgrid20")]
EPS = 1e-12

G = np.arange(S)
PH = G.copy()  # observed phase positions (all 20 in the main run; a subset, e.g. even phases, for the 1.7B run)


def set_phases(ph):
    """(Re)build the fit design and harmonic basis for the observed phase positions ph (subset of 0..19)."""
    global PH, _X, _PINV, _HARM, _COS, _SIN, NH
    PH = np.asarray(sorted(ph), int)
    n = len(PH)
    # pa1 fit on the phase means: intercept (item FE), linear trend, harmonics 1..3; evaluated on the full 0..19 circle
    _X = np.column_stack([np.ones(n), PH - PH.mean()] + [f(2 * np.pi * h * PH / S) for h in (1, 2, 3) for f in (np.cos, np.sin)])
    _PINV = np.linalg.pinv(_X)
    _HARM = np.column_stack([f(2 * np.pi * h * G / S) for h in (1, 2, 3) for f in (np.cos, np.sin)])
    NH = 10 if n == S else max(1, n // 2 - 1)  # harmonics reported: all 10 for 20 phases; 4 for 10 even phases (alias h <-> 10-h)
    hh = np.arange(1, NH + 1)
    _COS = np.cos(2 * np.pi * hh[:, None] * PH[None, :] / S)
    _SIN = np.sin(2 * np.pi * hh[:, None] * PH[None, :] / S)


set_phases(G)


def p2p_from_pm(pm):
    """3-harmonic peak-to-peak (pa1 definition) from phase means [..., n_phases]; balanced design => same as the lstsq fit."""
    beta = pm @ _PINV.T
    fit = beta[..., 2:] @ _HARM.T
    return fit.max(-1) - fit.min(-1)


def harm_amps(pm):
    """amplitude of harmonic h (period 20/h phases), h = 1..NH, by projection on cos/sin at the observed phases"""
    n = pm.shape[-1]
    a = 2 * np.hypot(pm @ _COS.T, pm @ _SIN.T) / n
    if n == S:
        a[..., S // 2 - 1] /= 2  # Nyquist harmonic
    return a


def endpoints(dm, w=None):
    """dm: item-demeaned outcome [n, 20]; w: optional item weights [..., n] (bootstrap counts). Returns dict of arrays [...]."""
    if w is None:
        w = np.ones((1, dm.shape[0]))
    sw = w.sum(-1)
    pm = (w @ dm) / sw[..., None]
    tot = (w @ (dm ** 2).sum(1)) / (dm.shape[1] * sw)
    a = harm_amps(pm)
    vp = pm.var(-1)
    return dict(range=pm.max(-1) - pm.min(-1), p2p=p2p_from_pm(pm), a1=a[..., 0], a2=a[..., 1], a3=a[..., 2], a4=a[..., 3],
                odd=np.hypot(a[..., 0], a[..., 2]), odd_all=np.sqrt((a[..., 0::2] ** 2).sum(-1)),
                even_all=np.sqrt((a[..., 1::2] ** 2).sum(-1)), var_phase=vp, var_share=np.where(tot > EPS, vp / np.maximum(tot, EPS), np.nan))


def perm_dm(dm, nperm, rng):
    """within-item permutation of the phase labels: returns [nperm, n, 20]"""
    n = dm.shape[0]
    order = np.argsort(rng.random((nperm, n, dm.shape[1])), axis=-1)
    return np.take_along_axis(np.broadcast_to(dm, (nperm, n, dm.shape[1])), order, axis=-1)


def wrap_range(pm, period):
    """range of the phase means after wrapping phases modulo the native period of the grid set (observed residues only)"""
    res = PH % period
    cols = [pm[..., res == r].mean(-1) for r in np.unique(res)]
    f = np.stack(cols, -1)
    return f.max(-1) - f.min(-1)


def perm_endpoints(dm, nperm, seed=0, wrap=None):
    rng = np.random.default_rng(seed)
    out = {}
    chunks = []
    step = 250
    for s0 in range(0, nperm, step):
        P = perm_dm(dm, min(step, nperm - s0), rng)
        pm = P.mean(1)  # [b, 20]
        tot = (P ** 2).mean((1, 2))
        a = harm_amps(pm)
        vp = pm.var(-1)
        e = dict(range=pm.max(-1) - pm.min(-1), p2p=p2p_from_pm(pm), a1=a[:, 0], a2=a[:, 1], a3=a[:, 2], a4=a[:, 3],
                 odd=np.hypot(a[:, 0], a[:, 2]), odd_all=np.sqrt((a[:, 0::2] ** 2).sum(-1)), even_all=np.sqrt((a[:, 1::2] ** 2).sum(-1)),
                 var_phase=vp, var_share=np.where(tot > EPS, vp / np.maximum(tot, EPS), np.nan))
        if wrap:
            e["range_native"] = wrap_range(pm, wrap)
        chunks.append(e)
    return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}


def pval(null, obs):
    null = null[np.isfinite(null)]
    return float((1 + (null >= obs - EPS).sum()) / (len(null) + 1)) if np.isfinite(obs) else np.nan


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def boot_weights(n_items, groups, nboot, seed, by="item"):
    """bootstrap count matrix [nboot, n_items]; by='hay' resamples haystacks and takes all their items."""
    rng = np.random.default_rng(seed)
    if by == "item":
        idx = rng.integers(0, n_items, (nboot, n_items))
        W = np.zeros((nboot, n_items))
        for b in range(nboot):
            np.add.at(W[b], idx[b], 1)
        return W
    hays = np.unique(groups)
    pick = rng.integers(0, len(hays), (nboot, len(hays)))
    cnt = np.zeros((nboot, len(hays)))
    for b in range(nboot):
        np.add.at(cnt[b], pick[b], 1)
    memb = (groups[None, :] == hays[:, None]).astype(float)  # [hays, items]
    return cnt @ memb


def to_matrix(g, y):
    piv = g.pivot(index="item", columns="phase", values=y).sort_index()
    assert piv.shape[1] == len(PH) and not piv.isna().any().any(), "unbalanced item x phase"
    return piv.to_numpy(float), piv.index.to_numpy()


def demean(Y):
    return Y - Y.mean(1, keepdims=True)


def worst_of(Y, w=None):
    if w is None:
        w = np.ones((1, Y.shape[0]))
    return ((w @ Y) / w.sum(-1)[..., None]).min(-1)


def selfcheck():
    rng = np.random.default_rng(3)
    for _ in range(5):
        n = 7
        Y = (rng.random((n, S)) < 0.5).astype(float) + 0.3 * np.sin(2 * np.pi * G / 10)[None, :] + rng.normal(0, .1, (n, S))
        phase = np.tile(G, n)
        item = np.repeat(np.arange(n), S)
        y = Y.ravel()
        ids = np.unique(item)
        Fm = (item[:, None] == ids[None, :]).astype(float)
        cols = []
        for h in (1, 2, 3):
            cols += [np.cos(2 * np.pi * h * phase / S), np.sin(2 * np.pi * h * phase / S)]
        X = np.column_stack([Fm, (phase - phase.mean())[:, None], np.column_stack(cols)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        c = beta[Fm.shape[1] + 1:]
        per = sum(c[2 * h] * np.cos(2 * np.pi * (h + 1) * G / S) + c[2 * h + 1] * np.sin(2 * np.pi * (h + 1) * G / S) for h in range(3))
        ref = float(per.max() - per.min())
        mine = float(endpoints(demean(Y))["p2p"][0])
        assert abs(ref - mine) < 1e-9, (ref, mine)
    print("selfcheck ok: vectorised 3-harmonic peak-to-peak equals the pa1 least-squares fit")


def main(rows, out, prefix, nperm, nboot):
    d = pd.read_csv(rows)
    d = d.copy()
    d["phase"] = d["phase_needle"].astype(int)
    full = d[d.arm == "full"]
    comp = d[(d.arm != "full") & (d.correct >= 0)].copy()  # arms that were not decoded carry correct = -1
    set_phases(sorted(comp.phase.unique()))
    print("phases observed:", len(PH))
    cnt = comp.groupby(["cr", "arm", "item"]).phase.nunique().unstack(["cr", "arm"])
    ok_items = cnt.index[(cnt == len(PH)).all(axis=1)]
    comp = comp[comp.item.isin(ok_items)]
    full = full[full.item.isin(ok_items)]
    item_hay = comp.drop_duplicates("item").set_index("item")["essay"].sort_index()
    items = item_hay.index.to_numpy()
    hay = item_hay.to_numpy()
    n = len(items)
    print("complete items:", n, "of", d.item.nunique(), "| haystacks:", len(np.unique(hay)))
    Wi = boot_weights(n, hay, nboot, 11, "item")
    Wh = boot_weights(n, hay, nboot, 13, "hay")
    MAT = {}
    for (cr, arm), g in comp.groupby(["cr", "arm"]):
        for y in ("correct", "val_kept", "needle_kept"):
            Y, ids = to_matrix(g, y)
            assert (ids == items).all()
            MAT[(cr, arm, y)] = Y
    crs = sorted(comp.cr.unique())
    arms = sorted(comp.arm.unique())
    res, perphase = [], []
    for cr in crs:
        for arm in arms:
            for y in ("correct", "val_kept", "needle_kept"):
                Y = MAT[(cr, arm, y)]
                dm = demean(Y)
                ep = endpoints(dm)
                ep = {k: float(v[0]) for k, v in ep.items()}
                nl = perm_endpoints(dm, nperm, seed=1, wrap=PERIOD[arm])
                eb = endpoints(dm, Wi)
                row = dict(cr=cr, arm=arm, outcome=y, n=Y.size, n_items=n, mean=float(Y.mean()))
                for k in ("range", "p2p", "a1", "a2", "a3", "a4", "odd", "odd_all", "even_all", "var_phase", "var_share"):
                    row[k] = ep[k]
                    row[k + "_null"] = float(np.nanmean(nl[k]))
                    row[k + "_p"] = pval(nl[k], ep[k])
                    row[k + "_lo"], row[k + "_hi"] = ci(eb[k])
                if y == "correct":
                    wm = (Wi @ Y.mean(1)) / Wi.sum(1)
                    row.update(mean_lo=ci(wm)[0], mean_hi=ci(wm)[1], worst=float(worst_of(Y)[0]), worst_lo=ci(worst_of(Y, Wi))[0],
                               worst_hi=ci(worst_of(Y, Wi))[1], period=PERIOD[arm])
                    rn = float(wrap_range(dm.mean(0), PERIOD[arm]))
                    row.update(range_native=rn, range_native_null=float(np.mean(nl["range_native"])), range_native_p=pval(nl["range_native"], rn))
                res.append(row)
            Y = MAT[(cr, arm, "correct")]
            pp = pd.DataFrame(dict(phase=PH, n=n, acc=Y.mean(0), val_kept=MAT[(cr, arm, "val_kept")].mean(0), needle_kept=MAT[(cr, arm, "needle_kept")].mean(0)))
            pp["cr"], pp["arm"] = cr, arm
            perphase.append(pp)
    res = pd.DataFrame(res)
    perphase = pd.concat(perphase)
    # paired contrasts
    con = []
    for cr in crs:
        for (A, B) in PAIRS:
            if A not in arms or B not in arms:
                continue
            for y in ("correct", "val_kept"):
                YA, YB = MAT[(cr, A, y)], MAT[(cr, B, y)]
                dA, dB = demean(YA), demean(YB)

                def stats(W):
                    ea, eb_ = endpoints(dA, W), endpoints(dB, W)
                    sw = W.sum(-1)
                    r = dict(delta=(W @ (YA - YB).mean(1)) / sw, drange=ea["range"] - eb_["range"], dodd=ea["odd"] - eb_["odd"],
                             da2=ea["a2"] - eb_["a2"], da4=ea["a4"] - eb_["a4"], dshare=ea["var_share"] - eb_["var_share"],
                             dvar=ea["var_phase"] - eb_["var_phase"])
                    if y == "correct":
                        r["dworst"] = worst_of(YA, W) - worst_of(YB, W)
                    return r
                obs = {k: float(v[0]) for k, v in stats(np.ones((1, n))).items()}
                bi, bh = stats(Wi), stats(Wh)
                row = dict(cr=cr, A=A, B=B, outcome=y, n_items=n, n_hay=len(np.unique(hay)))
                for k, v in obs.items():
                    row[k] = v
                    row[k + "_lo"], row[k + "_hi"] = ci(bi[k])
                    row[k + "_hay_lo"], row[k + "_hay_hi"] = ci(bh[k])
                if y == "correct":
                    pit = (YA - YB).mean(1)
                    dif = (YA - YB).ravel()
                    row.update(n_disc_pos=int((dif > 0).sum()), n_disc_neg=int((dif < 0).sum()), items_pos=int((pit > 0).sum()), items_neg=int((pit < 0).sum()))
                con.append(row)
    con = pd.DataFrame(con)
    # endpoint ratios arm / A1
    rat = []
    for cr in (crs if "A1_chunkkv" in arms else []):
        d1 = demean(MAT[(cr, "A1_chunkkv", "correct")])
        e1 = endpoints(d1, Wi)
        for arm in arms:
            if arm == "A1_chunkkv":
                continue
            ea = endpoints(demean(MAT[(cr, arm, "correct")]), Wi)
            for k in ("odd", "a2", "var_share"):
                r_ = ea[k] / np.maximum(e1[k], 1e-9)
                lo, hi = ci(r_)
                rat.append(dict(cr=cr, arm=arm, endpoint=k, ratio_med=float(np.nanmedian(r_)), ratio_lo=lo, ratio_hi=hi))
    rat = pd.DataFrame(rat)
    # per haystack
    phr = []
    for cr in crs:
        for arm in arms:
            Y = MAT[(cr, arm, "correct")]
            for h in np.unique(hay):
                Yh = Y[hay == h]
                e = {k: float(v[0]) for k, v in endpoints(demean(Yh)).items()}
                phr.append(dict(cr=cr, arm=arm, haystack=int(h), n_items=len(Yh), acc=float(Yh.mean()), worst=float(worst_of(Yh)[0]), range=e["range"],
                                odd=e["odd"], a2=e["a2"], a4=e["a4"], var_share=e["var_share"]))
    phr = pd.DataFrame(phr)
    sh = []
    for cr in (crs if ({"A1_chunkkv", "A12_chunk20_off10"} <= set(arms) and len(PH) == S) else []):
        for y, col in (("correct", "acc"), ("val_kept", "val_kept")):
            p1 = MAT[(cr, "A1_chunkkv", y)].mean(0)
            p12 = MAT[(cr, "A12_chunk20_off10", y)].mean(0)
            sh.append(dict(cr=cr, outcome=y, r_unshifted=float(np.corrcoef(p12, p1)[0, 1]), r_shifted10=float(np.corrcoef(p12, np.roll(p1, 10))[0, 1])))
    sh = pd.DataFrame(sh)
    fa = full.groupby("phase").agg(acc=("correct", "mean"), n=("correct", "size")).reset_index()
    for nm, df in (("amplitudes", res), ("per_phase", perphase), ("contrasts", con), ("endpoint_ratio", rat), ("per_haystack", phr),
                   ("grid_shift", sh), ("fullcache_per_phase", fa)):
        df.to_csv(f"{out}/{prefix}_{nm}.csv", index=False)
    print("full-cache acc", full.correct.mean(), len(full))
    print(res[res.outcome == "correct"][["cr", "arm", "mean", "worst", "range", "range_null", "range_p", "odd", "a2", "a4", "var_share"]].round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows")
    ap.add_argument("--out", default=".")
    ap.add_argument("--prefix", default="pa2")
    ap.add_argument("--nperm", type=int, default=2000)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck()
    else:
        main(a.rows, a.out, a.prefix, a.nperm, a.nboot)
