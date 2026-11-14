# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""C2G analysis: registered statistics P1-P4 and descriptive tables (numpy/pandas/scipy/patsy).

All functions take rows (list of dicts from the runner's *_rows.jsonl) and published scores computed with the scorer below.
Seeds and sizes are the registered ones (prereg_C2G_confirm.json, statistics block).
"""
import glob
import json
import re

import numpy as np
import pandas as pd

TASKS = ["niah_single_1", "niah_multikey_2", "fwe", "qa_1", "niah_single_3", "niah_multikey_1", "niah_single_2", "cwe", "qa_2", "vt",
         "niah_multivalue", "niah_multikey_3", "niah_multiquery"]
NP_PATTERN = re.compile(r"[\x00-\x1f]")
SEEDS = dict(signflip=20261026, boot_p1=20261027, boot_p2=20261028, boot_p3=20261029, boot_p4=20261030)
N_FLIPS, N_BOOT = 100_000, 10_000
M_FLOOR, M_Z = 1.0, 1.96


def item_score(task, pred, answers):
    pred = NP_PATTERN.sub("", str(pred).strip()).strip().lower()
    refs = [r.lower() for r in answers]
    if task.split("_")[0] == "qa":
        return 100.0 * max(1.0 if r in pred else 0.0 for r in refs)
    return 100.0 * sum(1.0 if r in pred else 0.0 for r in refs) / len(refs)


def read_rows(paths):
    rows, seen = [], set()
    for p in paths:
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r["run_id"], r["arm"], r["item_id"])
            assert k not in seen, f"duplicate row {k}"
            seen.add(k)
            rows.append(r)
    return rows


def frame(rows):
    df = pd.DataFrame(rows)
    df["delta"] = df["score"] - df["pub_score"]
    return df


# ---------- registered tests ------------------------------------------------------------------------------------

def task_index(task_arr):
    return {t: np.nonzero(task_arr == t)[0] for t in TASKS}


def signflip_p(d, n_flips=N_FLIPS, seed=SEEDS["signflip"]):
    """One-sided item-level sign-flip permutation p for mean(d) > 0."""
    rng = np.random.default_rng(seed)
    obs = d.mean()
    n = len(d)
    ge = 0
    chunk = 2000
    for _ in range(n_flips // chunk):
        sg = rng.integers(0, 2, size=(chunk, n), dtype=np.int8) * 2 - 1
        ge += int(((sg @ d) / n >= obs - 1e-12).sum())
    return (1 + ge) / (1 + (n_flips // chunk) * chunk)


def boot_task_means(d, task_arr, n_boot=N_BOOT, seed=0):
    """Stratified paired bootstrap: returns (n_boot, 13) matrix of task-mean deltas (items resampled within task)."""
    rng = np.random.default_rng(seed)
    idx = task_index(task_arr)
    out = np.empty((n_boot, len(TASKS)))
    for j, t in enumerate(TASKS):
        ii = idx[t]
        draw = rng.integers(0, len(ii), size=(n_boot, len(ii)))
        out[:, j] = d[ii][draw].mean(axis=1)
    return out


def p1_cell(df, seed_flip=SEEDS["signflip"], seed_boot=SEEDS["boot_p1"]):
    d = df.sort_values("item_id")["delta"].to_numpy()
    tasks = df.sort_values("item_id")["task"].to_numpy()
    assert len(d) == 6500 and all((tasks == t).sum() == 500 for t in TASKS)
    mean13 = float(np.mean([d[tasks == t].mean() for t in TASKS]))
    p = signflip_p(d, seed=seed_flip)
    bm = boot_task_means(d, tasks, seed=seed_boot).mean(axis=1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return dict(mean13_delta=mean13, p_signflip=p, boot_lo=float(lo), boot_hi=float(hi), n=len(d))


def holm(pvals, alpha=0.05):
    keys = list(pvals)
    order = sorted(keys, key=lambda k: pvals[k])
    adj, run = {}, 0.0
    m = len(keys)
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        adj[k] = run
    return adj


def margins_from_smoke(df_smoke):
    """m_task = max(1.96 * sd_task / sqrt(500), 1.0 pp); sd_task = ddof-1 sd of the smoke paired deltas."""
    out = {}
    for t in TASKS:
        d = df_smoke.loc[df_smoke.task == t, "delta"].to_numpy()
        assert len(d) == 15, (t, len(d))
        sd = float(d.std(ddof=1))
        out[t] = dict(sd=sd, m=max(M_Z * sd / np.sqrt(500), M_FLOOR))
    return out


def p2_cell(df, margins, seed=SEEDS["boot_p2"]):
    d = df.sort_values("item_id")["delta"].to_numpy()
    tasks = df.sort_values("item_id")["task"].to_numpy()
    bm = boot_task_means(d, tasks, seed=seed)
    res = {}
    for j, t in enumerate(TASKS):
        lb = float(np.percentile(bm[:, j], 5))
        m = margins[t]["m"]
        res[t] = dict(mean=float(d[tasks == t].mean()), lb95=lb, m=m, noninferior=bool(lb > -m), kill=bool(lb < -2 * m))
    res["_cell_pass"] = all(res[t]["noninferior"] for t in TASKS)
    res["_n_pass"] = sum(res[t]["noninferior"] for t in TASKS)
    return res


def kill_check_smoke(df_smoke, seed=SEEDS["boot_p2"]):
    """Smoke-stage kill rule: bootstrap one-sided 95% lower bound of the 15 deltas < -2 * m_smoke, m_smoke = 1.96 sd/sqrt(15)."""
    rng = np.random.default_rng(seed)
    out = {}
    for t in TASKS:
        d = df_smoke.loc[df_smoke.task == t, "delta"].to_numpy()
        sd = float(d.std(ddof=1))
        m_s = M_Z * sd / np.sqrt(15)
        bm = d[rng.integers(0, 15, size=(N_BOOT, 15))].mean(axis=1)
        lb = float(np.percentile(bm, 5))
        out[t] = dict(mean=float(d.mean()), sd=sd, lb95=lb, m_smoke=m_s, fired=bool(sd > 0 and lb < -2 * m_s))
    return out


def p3_cell(df_treat, leader_scores, seed=SEEDS["boot_p3"]):
    """Treatment minus published leader, paired by item, task-stratified bootstrap, two-sided 95% percentile CI."""
    t = df_treat.sort_values("item_id")
    d = t["score"].to_numpy() - leader_scores[t["item_id"].to_numpy()]
    tasks = t["task"].to_numpy()
    bm = boot_task_means(d, tasks, seed=seed).mean(axis=1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return dict(mean13_delta=float(np.mean([d[tasks == tk].mean() for tk in TASKS])), lo=float(lo), hi=float(hi))


# ---------- P4 periodogram (code of artifact 8e95e335, lineage of periodogram_audit_all.csv) ----------------------

def needle_positions(ruler_df, tok):
    """pos = tokens of the context up to 'for <key> is' + 4 chars; ntok = tokens of ' <key>'. tok = tokenizers.Tokenizer."""
    P4_TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multivalue"]
    R = ruler_df.copy()
    R["item"] = range(len(R))
    R["key"] = R.question.str.extract(r"special magic (?:number|numbers|uuid) for (.+?) mentioned")[0]
    R = R[R.task.isin(P4_TASKS)]
    recs = []
    for _, row in R.iterrows():
        m_ = re.search(r"for " + re.escape(row.key) + r" is", row.context)
        pre = row.context[: m_.start() + 4]
        recs.append(dict(item=row["item"], task=row.task, pos=len(tok.encode(pre, add_special_tokens=False).ids),
                         ntok=len(tok.encode(" " + row.key, add_special_tokens=False).ids)))
    return pd.DataFrame(recs)


def periodogram_ratios(PH, y_by_entry, y0, periods=range(2, 65)):
    """PH: DataFrame(item, task, pos, ntok) for one model; y0: Series item -> bool solved by the uncompressed model;
    y_by_entry: {name: Series item -> bool correct}. Returns {name: dict(acc, ratio={P: F/median F}, estimable)}."""
    import patsy
    from patsy import dmatrix

    S = PH[PH.item.isin(y0.index[y0.values])].sort_values("item").reset_index(drop=True)
    X0 = np.asarray(dmatrix("C(task)+ntok+bs(pos,df=40)", S, return_type="dataframe", eval_env=patsy.EvalEnvironment([{}])))
    Q0, _ = np.linalg.qr(X0)
    n, k0 = X0.shape
    FP = {}
    for P in periods:
        w = 2 * np.pi * S.pos.values / P
        F = np.column_stack([np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])
        F = F - Q0 @ (Q0.T @ F)
        q, _ = np.linalg.qr(F)
        FP[P] = q
    out = {}
    for name, ys in y_by_entry.items():
        y = ys.reindex(S.item.values).to_numpy().astype(float)
        assert not np.isnan(y).any(), name
        acc = float(y.mean())
        if not (0.03 < acc < 0.97):
            out[name] = dict(acc=acc, estimable=False)
            continue
        r = y - Q0 @ (Q0.T @ y)
        rss0 = r @ r
        F = {}
        for P in periods:
            ss = np.sum((FP[P].T @ r) ** 2)
            df2 = n - k0 - 4
            F[P] = (ss / 4) / ((rss0 - ss) / df2)
        med = float(np.median(list(F.values())))
        out[name] = dict(acc=acc, estimable=True, n=n, ratio={P: F[P] / med for P in F}, max_ratio=max(F[P] / med for P in F),
                         argmax=max(F, key=F.get))
    return out
