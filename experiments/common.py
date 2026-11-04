"""Shared helpers for the experiment scripts (paths, plotting style, result I/O)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
FIG = RES / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# Validated categorical slots (light surface); text never wears a series colour.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SHADE = "#e9e7e1"          # tunnel / region shading
GT = INK


def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.6, "figure.dpi": 130,
        "savefig.dpi": 220, "savefig.bbox": "tight", "axes.prop_cycle": plt.cycler(color=[BLUE, ORANGE, AQUA, YELLOW]),
    })


style()


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    return name


def save_json(name, obj):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(RES / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=2, default=conv)


def load_json(name):
    with open(RES / f"{name}.json") as f:
        return json.load(f)


def shade(ax, intervals, y_text=None, labels=None, color=SHADE):
    for k, (a, b) in enumerate(intervals):
        ax.axvspan(a, b, color=color, lw=0, zorder=0)
        if labels:
            ax.text((a + b) / 2, y_text if y_text is not None else ax.get_ylim()[1], labels[k], ha="center",
                    va="top", fontsize=7.5, color=INK2)


def ms(x):
    x = np.asarray(x, float)
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def pool_map(fn, args, procs=2):
    import multiprocessing as mp
    with mp.get_context("fork").Pool(procs) as p:
        return p.map(fn, args, chunksize=1)
