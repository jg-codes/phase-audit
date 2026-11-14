# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the analysis scripts: paths, result writer, plot style."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
FIG = os.path.join(ROOT, "figures")
for _d in (OUT, FIG):
    os.makedirs(_d, exist_ok=True)


def write_result(name, values):
    """Write computed values (flat dict: key -> number or string) to out/<name>.json."""
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(values, f, indent=1, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def plot_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.titleweight": "normal",
        "axes.titlelocation": "left", "lines.linewidth": 1.2, "pdf.fonttype": 42,
    })


def panel_letter(ax, letter, dx=-0.18, dy=1.02):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom", ha="left")
