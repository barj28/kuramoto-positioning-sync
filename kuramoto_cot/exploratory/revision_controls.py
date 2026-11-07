"""Exploratory, after review: do the sub-universe effects survive aggregate positioning controls?

For each sub-universe, H2 on r with the registered controls plus (a) |mean z| (signed aggregate
positioning, the Brunnermeier-Nagel-Pedersen 2008 channel for currencies) and (b) mean z (its
sign). Also the same regressions with |mean z| in place of r, to see which carries the effect.

python -m kuramoto_cot.exploratory.revision_controls --cftc <dir> --prices <dir> --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import cotsync as cs
from .. import run_prereg as rp
from .revision import FX, RATES_EQ, h2_detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cftc", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(20261109)
    rp.N_BOOT = 5000
    x_all = cs.load_cftc(Path(a.cftc))
    prices = cs.load_prices(Path(a.prices), x_all.columns)
    days = prices["ES"].index
    uni = cs.select_universe(x_all, prices, days)
    subsets = {"fx8": [r for r in uni if r in FX], "rates_equity8": [r for r in uni if r in RATES_EQ],
               "commodity_dis16": [r for r in uni if cs.MAPPING[r][0] == "dis"]}
    res = {"note": "exploratory, after review; not pre-registered"}
    for name, cols in subsets.items():
        x = x_all[cols]
        rows = rp.weekly_rows(x, {c: prices[c] for c in cols}, days)
        rows["r_plane"] = cs.order_parameter(cs.phases(x, "plane"))
        z, _ = cs.standardize(x)
        rows["c_abs_mean_z"] = z.mean(axis=1).abs()
        rows["c_mean_z"] = z.mean(axis=1)
        keep = rows["entry"].between(*map(pd.Timestamp, rp.ENTRY_SPAN)) & ~rows.index.map(cs.in_shutdown)
        df = rows[keep].dropna(subset=list(rp.OUTCOMES) + rp.CONTROLS + ["r_plane", "c_abs_mean_z"])
        res[name] = {"corr_r_abs_mean_z": float(df[["r_plane", "c_abs_mean_z"]].corr(method="spearman").iloc[0, 1]),
                     "r_with_abs_mean_z_and_mean_z": h2_detail(df, "r_plane", rng, extra=["c_abs_mean_z", "c_mean_z"]),
                     "abs_mean_z_alone": h2_detail(df, "c_abs_mean_z", rng)}
        print(name, "done", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
