"""Figures of the note. python -m kuramoto_cot.figures --cftc <dir> --out <dir>"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import cotsync as cs
from .exploratory.original_sfb import compute_kuramoto_sync

HERE = Path(__file__).parent
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
INK, MUTED, ACCENT = "#1f2937", "#9ca3af", "#b45309"


def fig_r_real(cftc: Path, out: Path):
    res = pd.read_json(HERE / "results" / "results.json", typ="series")
    x = cs.load_cftc(cftc)[res["universe"]]
    th = cs.phases(x, "plane").loc["2007-06-01":]   # first full z-window; H1 itself starts 2009-06
    rng = np.random.default_rng(7)
    null = np.array([cs.order_parameter(cs.circular_shift(th, rng)).rolling(4).mean().to_numpy()
                     for _ in range(500)])
    r = cs.order_parameter(th)
    lo, hi = np.nanquantile(null, 0.05, axis=0), np.nanquantile(null, 0.95, axis=0)
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    ax.fill_between(r.index, lo, hi, color=MUTED, alpha=0.35, lw=0, label="circular-shift null, 5-95% (4-week mean)")
    ax.plot(r.index, r.rolling(4).mean(), color=INK, lw=1.0, label="r(t), 4-week mean")
    ax.axhline(np.nanmean(null), color=MUTED, lw=0.8, ls="--")
    for d, lab in (("2008-10-28", "GFC"), ("2010-11-02", "QE2 to May 2011"),
                   ("2015-12-08", "Dec 2015"), ("2021-02-23", "Reflation")):
        ax.annotate(lab, (pd.Timestamp(d), r.rolling(4).mean().loc[:d].iloc[-1]), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=7, color=ACCENT)
    ax.set_ylabel("order parameter r")
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_r_real.pdf")
    plt.close(fig)


def fig_lookahead(out: Path):
    rng = np.random.default_rng(3)
    durations = rng.integers(12, 90, 30)                       # irregular holding periods
    pos = np.concatenate([np.full(d, 1.0 if k % 2 else -1.0) for k, d in enumerate(durations)])[:400]
    t = np.arange(len(pos))
    orig = compute_kuramoto_sync(pos[None, :], np.ones(1)).phases[0]
    caus = cs.causal_sfb(pd.Series(pos), W=120, dt_min=2, T_max=60).to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 3.0), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    axes[0].step(t, pos, color=INK, lw=0.9, where="post")
    axes[0].set_ylabel("position")
    axes[1].plot(t, orig, color=ACCENT, lw=1.0, label="original SFB (interpolates to the next peak)")
    axes[1].plot(t, caus, color=INK, lw=1.0, ls="--", label="causal SFB (past peaks only)")
    flips = np.where(np.diff(pos) != 0)[0] + 1
    for f in flips[(flips > 150) & (flips < 400)]:
        for a in axes:
            a.axvline(f, color=MUTED, lw=0.5)
    axes[1].set_ylabel("phase (rad)")
    axes[1].set_xlabel("day")
    axes[1].set_xlim(150, 400)
    axes[0].legend(*axes[1].get_legend_handles_labels(), frameon=False, fontsize=7, ncol=2,
                   loc="lower left", bbox_to_anchor=(0, 1.02))
    fig.tight_layout()
    fig.savefig(out / "fig_lookahead.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cftc", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_r_real(Path(a.cftc), out)
    fig_lookahead(out)


if __name__ == "__main__":
    main()
