"""Exploratory analyses added after the hostile review (not pre-registered).

1. H1 against a multivariate Fourier-transform null that keeps every cross-correlation of weekly
   position changes, by sub-universe.
2. What r measures: correlation with breadth of positioning (|mean sign z|, |mean z|) and r with
   the level sign alone.
3. H2/H3 by sub-universe (registered S3 split, and the financial split into currencies, where
   leveraged funds are directional, versus rates and equity indices, where their net short is
   largely basis trades and hedges), with effects in SD of the outcome, NW 95% intervals, the
   minimum detectable effect at 80% power and TOST against +/-0.10 SD.
4. The registered H2 with signed aggregate positioning |mean z| as an extra control.

python -m kuramoto_cot.exploratory.revision --cftc <dir> --prices <dir> --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from .. import cotsync as cs
from .. import run_prereg as rp
from .. import stats as st

N_FT = 500
EQUIV = 0.10
RATES_EQ = ["ZT", "ZF", "ZN", "ZB", "UB", "ES", "NQ", "YM"]
FX = ["EUR.FUT", "JPY.FUT", "GBP.FUT", "CAD.FUT", "AUD.FUT", "CHF.FUT", "MXN.FUT", "NZD.FUT"]


def _h1_span_mean(th: pd.DataFrame) -> float:
    return float(cs.order_parameter(th.loc[rp.H1_SPAN[0]:rp.H1_SPAN[1]]).mean())


def ft_null(x: pd.DataFrame, rng) -> dict:
    """Complete-case panel of positions; surrogates of the weekly changes, integrated from the
    observed first level, then the same causal phase. Observed value on the same panel."""
    p = x.ffill(limit=4)
    p = p.loc[p.notna().all(axis=1).idxmax():].dropna()
    dx = p.diff().iloc[1:]
    th_obs = cs.phases(p, "plane")
    obs = _h1_span_mean(th_obs)
    th_span = th_obs.loc[rp.H1_SPAN[0]:]
    sur, circ = [], []
    for _ in range(N_FT):
        s = cs.ft_surrogate(dx.to_numpy(), rng)
        lev = np.vstack([p.iloc[[0]].to_numpy(), p.iloc[0].to_numpy() + np.cumsum(s, axis=0)])
        ps = pd.DataFrame(lev, index=p.index, columns=p.columns)
        sur.append(_h1_span_mean(cs.phases(ps, "plane")))
        circ.append(float(cs.order_parameter(cs.circular_shift(th_span, rng)).mean()))
    sur, circ = np.array(sur), np.array(circ)
    floor = float(np.mean(circ))
    return {"panel_start": str(p.index[0].date()), "n_markets": int(p.shape[1]), "obs": obs,
            "ft_mean": float(sur.mean()), "ft_p99": float(np.quantile(sur, 0.99)),
            "p_ft": float((1 + np.sum(sur >= obs)) / (1 + N_FT)),
            "circular_mean": floor, "p_circular": float((1 + np.sum(circ >= obs)) / (1 + N_FT)),
            "share_excess_explained_by_comovement": float((sur.mean() - floor) / (obs - floor))}


def breadth(x: pd.DataFrame) -> dict:
    z, _ = cs.standardize(x)
    th = cs.phases(x, "plane")
    span = slice(*rp.H1_SPAN)
    r = cs.order_parameter(th).loc[span]
    sign_only = cs.order_parameter(pd.DataFrame(np.where(z > 0, 0.0, np.pi), index=z.index,
                                                columns=z.columns).where(th.notna())).loc[span]
    bs = np.sign(z).where(th.notna()).mean(axis=1).abs().loc[span]
    bz = z.where(th.notna()).mean(axis=1).abs().loc[span]
    zz = z.where(th.notna()).abs().loc[span]
    _, d = cs.standardize(x)
    dd = d.where(th.notna()).abs().loc[span]
    rho = lambda a, b: float(pd.concat([a, b], axis=1).dropna().corr(method="spearman").iloc[0, 1])
    return {"r_mean": float(r.mean()), "r_sign_only_mean": float(sign_only.mean()),
            "spearman_r_abs_mean_sign_z": rho(r, bs), "spearman_r_abs_mean_z": rho(r, bz),
            "share_abs_z_gt_abs_d": float((zz > dd).sum().sum() / zz.notna().sum().sum())}


def h2_detail(df: pd.DataFrame, rcol: str, rng, extra: list[str] | None = None) -> dict:
    cols = [rcol] + rp.CONTROLS + (extra or [])
    X = rp._design(df, cols)
    out = {"n": int(len(df))}
    Y = df[list(rp.OUTCOMES)].to_numpy()
    b_hat = np.array([st.ols(Y[:, k], X)[1] for k in range(Y.shape[1])])
    boots = np.empty((rp.N_BOOT, Y.shape[1]))
    for b in range(rp.N_BOOT):
        i = st.stationary_bootstrap_indices(len(df), rp.BLOCK, rng)
        boots[b] = np.linalg.lstsq(X[i], Y[i], rcond=None)[0][1]
    p_rw, _ = st.romano_wolf(b_hat, boots)
    for k, (name, sign) in enumerate(rp.OUTCOMES.items()):
        sd = float(df[name].std())
        se = float(st.newey_west_se(Y[:, k], X, rp.NW_LAG)[1])
        b, s = b_hat[k] / sd, se / sd
        p_tost = max(1 - sps.norm.cdf((b + EQUIV) / s), sps.norm.cdf((b - EQUIV) / s))
        out[name] = {"b_sd": float(b), "ci95_sd": [float(b - 1.96 * s), float(b + 1.96 * s)],
                     "t_nw": float(b / s), "p_romano_wolf": float(p_rw[k]), "mde80_sd": float(2.8 * s),
                     "p_tost_0.10sd": float(p_tost), "predicted_sign": sign}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cftc", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(20261108)
    rp.N_BOOT = 5000
    x_all = cs.load_cftc(Path(a.cftc))
    prices = cs.load_prices(Path(a.prices), x_all.columns)
    days = prices["ES"].index
    uni = cs.select_universe(x_all, prices, days)
    subsets = {"all32": uni,
               "financial_tff16": [r for r in uni if cs.MAPPING[r][0] == "tff"],
               "commodity_dis16": [r for r in uni if cs.MAPPING[r][0] == "dis"],
               "fx8": [r for r in uni if r in FX],
               "rates_equity8": [r for r in uni if r in RATES_EQ]}
    res = {"note": "exploratory, after review; not pre-registered", "subsets": subsets}
    res["breadth_all32"] = breadth(x_all[uni])
    for name, cols in subsets.items():
        x = x_all[cols]
        r = cs.order_parameter(cs.phases(x, "plane"))
        rows = rp.weekly_rows(x, {c: prices[c] for c in cols}, days)
        rows["r_plane"] = r
        z, _ = cs.standardize(x)
        rows["c_abs_mean_z"] = z.mean(axis=1).abs()
        keep = rows["entry"].between(*map(pd.Timestamp, rp.ENTRY_SPAN)) & ~rows.index.map(cs.in_shutdown)
        df = rows[keep].dropna(subset=list(rp.OUTCOMES) + rp.CONTROLS + ["r_plane", "c_abs_mean_z"])
        res[name] = {"h1_ft": ft_null(x, rng), "h2": h2_detail(df, "r_plane", rng), "h3": rp.h3(df, "r_plane")}
        if name == "all32":
            res[name]["h2_with_abs_mean_z"] = h2_detail(df, "r_plane", rng, extra=["c_abs_mean_z"])
        print(name, "done", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
