# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""Table 2 and Figure 2: period-20 tests under position control on the published ChunkKV and SnapKV predictions.

LR: likelihood-ratio statistic (6 df) of three harmonics at period 20 added to a logistic model with task, key length and a
40-df B-spline of needle position. Entries with accuracy outside 0.05..0.95 (items solved by the uncompressed model) are
excluded ('excl.'); a failed fit is reported as 'n.e.'. Paired test: OLS of the per-item difference ChunkKV minus SnapKV on
the same terms, F test of the harmonics. Reads data/ only.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402

from common import DATA, FIG, panel_letter, plot_style, write_result  # noqa: E402

warnings.filterwarnings("ignore")
BASE = "C(task)+ntok+bs(pos,df=40)"
HARM = "+s1+c1+s2+c2+s3+c3"


def harmonics(g):
    w = 2 * np.pi * g.pos / 20
    for h in (1, 2, 3):
        g[f"s{h}"] = np.sin(h * w)
        g[f"c{h}"] = np.cos(h * w)
    return g


def load():
    pos = pd.read_csv(f"{DATA}/ruler_needle_positions.csv")
    lb = pd.read_csv(f"{DATA}/leaderboard_item_correct.csv.gz")
    out = {}
    for model in ["Qwen3-8B", "Llama-3.1-8B-Instruct"]:
        base = lb[(lb.model == model) & (lb.press == "no_press")].set_index("item").y
        P = pos[(pos.model == model) & pos.item.isin(base.index[base == 1])]
        for press in ["chunkkv__Q", "snapkv__Q"]:
            for cr in [0.5, 0.75, 0.88]:
                d = lb[(lb.model == model) & (lb.press == press) & (lb.cr == cr)][["item", "y"]]
                out[(model, press, cr)] = P.merge(d, on="item")
    return out


def lr_test(g):
    if not (0.05 < g.y.mean() < 0.95):
        return "excl.", None
    g = harmonics(g.copy())
    try:
        b = smf.logit(f"y ~ {BASE}", g).fit(disp=0, maxiter=200)
        f = smf.logit(f"y ~ {BASE}{HARM}", g).fit(disp=0, maxiter=200)
        lr = 2 * (f.llf - b.llf)
        if not np.isfinite(lr):
            return "n.e.", None
        return lr, float(stats.chi2.sf(lr, 6))
    except Exception:
        return "n.e.", None


def periodic_curve(g):
    g = harmonics(g.copy())
    f = smf.logit(f"y ~ {BASE}{HARM}", g).fit(disp=0, maxiter=300)
    ph = np.linspace(0, 19, 96)
    X = np.column_stack([fn(h * 2 * np.pi * ph / 20) for h in (1, 2, 3) for fn in (np.sin, np.cos)])
    names = [f"{a}{h}" for h in (1, 2, 3) for a in ("s", "c")]
    b = f.params[names].values
    V = f.cov_params().loc[names, names].values
    return ph, X @ b, np.sqrt(np.einsum("ij,jk,ik->i", X, V, X))


def main():
    D = load()
    v = {}
    for model in ["Qwen3-8B", "Llama-3.1-8B-Instruct"]:
        for cr in [0.5, 0.75, 0.88]:
            k = f"{model}|{cr:.2f}"
            c, s = D[(model, "chunkkv__Q", cr)], D[(model, "snapkv__Q", cr)]
            for tag, g in (("chunkkv", c), ("twin", s)):
                lr, p = lr_test(g)
                if isinstance(lr, str):
                    v[f"{k}|{tag}_LR"] = lr
                else:
                    v[f"{k}|{tag}_LR"] = round(lr, 1)
                    v[f"{k}|{tag}_p"] = float(f"{p:.2g}")
            m = c.merge(s[["item", "y"]], on="item", suffixes=("_c", "_s"))
            m["d"] = m.y_c - m.y_s
            m = harmonics(m)
            f0 = smf.ols(f"d ~ {BASE}", m).fit()
            f1 = smf.ols(f"d ~ {BASE}{HARM}", m).fit()
            F, p, _ = f1.compare_f_test(f0)
            v[f"{k}|n"] = len(m)
            v[f"{k}|diff_pp"] = round(100 * m.d.mean(), 1)
            v[f"{k}|paired_F"] = round(F, 1)
            v[f"{k}|paired_p"] = float(f"{p:.2g}")
    write_result("period20", v)
    for k in sorted(v):
        print(k, v[k])
    # Figure 2
    plot_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True, gridspec_kw=dict(wspace=0.06))
    for ax, model in zip(axes, ["Qwen3-8B", "Llama-3.1-8B-Instruct"]):
        for press, col, lab in [("chunkkv__Q", "#c2410c", "ChunkKV (20-token chunks)"), ("snapkv__Q", "#6b7280", "SnapKV, same window, no chunks")]:
            try:
                ph, est, se = periodic_curve(D[(model, press, 0.88)])
            except Exception as e:  # not estimable
                print("figure 2:", model, press, "not estimable:", str(e)[:60])
                continue
            ax.fill_between(ph, est - 1.96 * se, est + 1.96 * se, color=col, alpha=0.18, lw=0)
            ax.plot(ph, est, color=col, lw=1.4, label=lab)
        ax.axhline(0, color="#888888", lw=0.7)
        ax.set_title(f"{model}, CR 0.88")
        ax.set_xlabel("needle key position modulo 20 (tokens)")
        ax.set_xticks(range(0, 20, 4))
        ax.margins(x=0.02)
    axes[0].set_ylabel("periodic effect on log-odds of a correct answer")
    axes[0].legend(loc="lower left")
    fig.savefig(f"{FIG}/fig2_chunkkv_phase.png", dpi=200)


if __name__ == "__main__":
    main()
