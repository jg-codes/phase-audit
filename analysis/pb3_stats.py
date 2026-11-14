# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""Statistics of the registered test PB3 (reference copies of the functions of the registered analysis script pb3_analysis.py,
frozen before any model call), operating on the derived per-probe tables in data/pb3/. Constants as registered."""
import numpy as np, pandas as pd

B_BOOT = 10000; B_PERM = 10000; SEED = 20261024
STRIDE = {"flash": 2, "pro": 4}
PERIODS = [2, 3, 4, 5, 6, 7, 8]
def two_level_indices(sources, B, seed):
    """yield index arrays (into the probe table) of a two-level cluster bootstrap: sources with replacement, then probes within source"""
    rng = np.random.default_rng(seed); srcs = sorted(set(sources)); sources = np.asarray(sources)
    pos = {s: np.where(sources == s)[0] for s in srcs}
    for _ in range(B):
        pick = rng.integers(0, len(srcs), len(srcs))
        yield np.concatenate([pos[srcs[k]][rng.integers(0, len(pos[srcs[k]]), len(pos[srcs[k]]))] for k in pick])

def boot_stat(sources, fn, B=B_BOOT, seed=SEED):
    return np.array([fn(ix) for ix in two_level_indices(sources, B, seed)])

def p_le(b, null): return (np.sum(b <= null) + 1) / (len(b) + 1)
def p_ge(b, null): return (np.sum(b >= null) + 1) / (len(b) + 1)
def bcv(M, P):
    """M: (..., 16) margins; classes j mod P; between-class variance = sum_c n_c (mean_c - grand)^2 / N"""
    n = M.shape[-1]; cls = np.arange(n) % P; g = M.mean(-1, keepdims=True); tot = 0
    for c in range(P):
        mk = cls == c; tot = tot + mk.sum() * (M[..., mk].mean(-1) - g[..., 0]) ** 2
    return tot / n

def excess_ratios(M, periods=PERIODS, B=B_PERM, seed=SEED):
    """between-class variance (classes j mod P) of each probe's base margins vs its permutation null (labels permuted within probe, independent per probe).
    'pooled' (primary; pb2_postrun_review.md section 4; reproduces its 3.5x / 2.1x): sum_i BCV_i / sum_i E_perm[BCV_i].
    'mean'   (literal reading of the registration, secondary): mean_i BCV_i / E_perm[BCV_i].
    p = permutation p of the statistic against the same joint null."""
    rng = np.random.default_rng(seed); X = M.to_numpy(float); n = X.shape[0]
    obs = {P: bcv(X, P) for P in periods}; perm_vals = {P: np.zeros((n, B)) for P in periods}
    for i in range(n):
        idx = np.argsort(rng.random((B, X.shape[1])), axis=1); Xp = X[i][idx]
        for P in periods: perm_vals[P][i] = bcv(Xp, P)
    res = {}
    for P in periods:
        nm = perm_vals[P].mean(1); r_pool = obs[P].sum() / nm.sum(); null_pool = perm_vals[P].sum(0) / nm.sum()
        ratio_i = obs[P] / nm; r_mean = ratio_i.mean(); null_mean = (perm_vals[P] / nm[:, None]).mean(0)
        res[P] = dict(ratio=float(r_pool), p_perm=float((np.sum(null_pool >= r_pool) + 1) / (B + 1)), ratio_mean=float(r_mean), p_perm_mean=float((np.sum(null_mean >= r_mean) + 1) / (B + 1)),
                      obs_i=obs[P], null_i=nm, per_probe=ratio_i)
    return res

# ---------------------------------------------------------------- H3: profile prediction
def ols_slope(x, y, intercept=True):
    if intercept: xm, ym = x.mean(), y.mean(); sxx = ((x - xm) ** 2).sum(); return float(((x - xm) * (y - ym)).sum() / sxx) if sxx > 0 else np.nan
    return float((x * y).sum() / (x * x).sum())

def h3(T, B=B_BOOT, seed=SEED):
    x, y = T.pred.to_numpy(), T.obs.to_numpy(); sl = ols_slope(x, y); sl0 = ols_slope(x, y, False)
    bs = boot_stat(T.source.to_numpy(), lambda ix: ols_slope(x[ix], y[ix]), B, seed); ok = ~np.isnan(bs)
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else np.nan
    return dict(n=int(len(T)), slope=sl, slope_through_origin=sl0, r=r, ci95_two_sided=[float(np.percentile(bs[ok], 2.5)), float(np.percentile(bs[ok], 97.5))],
                lb95_onesided=float(np.percentile(bs[ok], 5)), lb_twosided=float(np.percentile(bs[ok], 2.5)), p_le_0p3=float(p_le(bs[ok], 0.3)), p_le_0=float(p_le(bs[ok], 0.0)), boot=bs)

# ---------------------------------------------------------------- H4: avg_S remedy
def h4(T, col="single", B=B_BOOT, seed=SEED):
    s, a = T[col].to_numpy(), T.avg.to_numpy()
    f = lambda ix: 1 - a[ix].mean() / s[ix].mean()
    bs = boot_stat(T.source.to_numpy(), f, B, seed); pt = float(1 - a.mean() / s.mean())
    return dict(n=int(len(T)), single_mean=float(s.mean()), avg_mean=float(a.mean()), reduction=pt, ci95_two_sided=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                lb95_onesided=float(np.percentile(bs, 5)), lb_twosided=float(np.percentile(bs, 2.5)), p_le_0=float(p_le(bs, 0.0)), boot=bs)

# ---------------------------------------------------------------- per-model analysis
def holm(ps):
    ks = sorted(ps, key=lambda k: ps[k]); m = len(ks); run = 0; adj = {}
    for r, k in enumerate(ks): run = max(run, min(1.0, (m - r) * ps[k])); adj[k] = run
    return adj


# ---------------------------------------------------------------- driver
def verdicts(res, alpha=0.05):
    """res[model] = analyse_model output. Holm over the 8 tests (H1-H4 x 2 models). returns dict[(model,H)] -> row"""
    ps = {}
    for m, o in res.items():
        for h in ("H1", "H2", "H3", "H4"):
            if h in o and "p" in o[h]: ps[(m, h)] = float(o[h]["p"])
    adj = holm({f"{m}|{h}": p for (m, h), p in ps.items()}); out = {}
    for (m, h), p in ps.items():
        o = res[m][h]; ph = adj[f"{m}|{h}"]
        if h == "H1": kill = o["kill"]; ok = (not kill) and ph < alpha
        elif h == "H2": kill = o["kill"]; ok = (not kill) and o["clause_argmax_ok"] and ph < alpha
        elif h == "H3": kill = o["kill"]; ok = (not kill) and ph < alpha
        else: kill = o["kill"]; ok = (not kill) and o["reduction"] >= 0.30 and ph < alpha
        lab = "REFUTED (kill rule)" if kill else ("SUPPORTED" if ok else ("NOT SUPPORTED (kill rule not fired; Holm p >= 0.05 or threshold not met)"))
        out[(m, h)] = dict(p=p, p_holm=float(ph), kill=bool(kill), verdict=lab)
    return out
