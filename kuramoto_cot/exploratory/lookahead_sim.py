"""Exploratory (not pre-registered): does the paper's non-causal SFB manufacture predictability?

Markets with NO predictability (independent zero-mean returns, stochastic volatility), a
universe of trend strategies (TSMOM and MA crossovers) as in the paper, then the paper's
statistic: Spearman of r(t) with the 21-day forward maximum drawdown of the equal-weight
strategy portfolio, for the original SFB and for the causal SFB. Any correlation is spurious.

python -m research.kuramoto_cot.exploratory.lookahead_sim --reps 100 --out <json>
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats as sps

from research.kuramoto_cot import cotsync as cs
from research.kuramoto_cot.exploratory.original_sfb import compute_kuramoto_sync

M, T, FWD = 20, 3000, 21
TSMOM = (20, 60, 120, 250)
MA = ((10, 50), (20, 100))


def _market(rng):
    lv = np.empty(T)
    lv[0] = 0.0
    for t in range(1, T):
        lv[t] = 0.98 * lv[t - 1] + 0.15 * rng.standard_normal()
    return 0.01 * np.exp(lv) * rng.standard_normal(T)


def _positions(ret):
    p = np.cumsum(ret)
    s = pd.Series(p)
    out = [np.sign(s - s.shift(L)).to_numpy() for L in TSMOM]
    out += [np.sign(s.rolling(a).mean() - s.rolling(b).mean()).to_numpy() for a, b in MA]
    return np.nan_to_num(np.array(out))


def one(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rets = np.array([_market(rng) for _ in range(M)])
    pos = np.vstack([_positions(r) for r in rets])                     # (N, T)
    strat_ret = np.vstack([pos[k * 6:(k + 1) * 6, :-1] * rets[k, 1:] for k in range(M)])
    port = np.r_[0.0, strat_ret.mean(axis=0)]
    cum = np.cumsum(port)
    fdd = np.full(T, np.nan)
    for t in range(T - FWD):
        path = cum[t:t + FWD + 1] - cum[t]
        fdd[t] = (path - np.maximum.accumulate(path)).min()
    r_orig = compute_kuramoto_sync(pos, np.ones(len(pos))).r
    th_c = pd.DataFrame({i: cs.causal_sfb(pd.Series(pos[i]), W=120, dt_min=2, T_max=60) for i in range(len(pos))})
    r_caus = cs.order_parameter(th_c).to_numpy()
    out = {}
    for name, r in (("original", r_orig), ("causal", r_caus)):
        m = np.isfinite(r) & np.isfinite(fdd)
        m[:300] = False                                                   # warm-up
        rho, p = sps.spearmanr(r[m], fdd[m])
        out[name] = (float(rho), float(p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with ProcessPoolExecutor(max_workers=16) as ex:
        res = list(ex.map(one, range(1000, 1000 + a.reps)))
    summary = {}
    for name in ("original", "causal"):
        rho = np.array([r[name][0] for r in res])
        p = np.array([r[name][1] for r in res])
        summary[name] = {"mean_rho": float(rho.mean()), "sd_rho": float(rho.std(ddof=1)),
                         "share_p_lt_05": float((p < 0.05).mean()), "share_p_lt_1e6": float((p < 1e-6).mean()),
                         "min_rho": float(rho.min()), "max_rho": float(rho.max())}
    summary["reps"] = a.reps
    summary["design"] = {"markets": M, "days": T, "strategies": M * 6, "fwd_days": FWD,
                         "returns": "iid zero-mean, log-AR(1) stochastic vol, no predictability"}
    open(a.out, "w").write(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
