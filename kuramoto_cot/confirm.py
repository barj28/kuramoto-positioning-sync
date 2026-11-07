"""Confirmatory tests of CONFIRM_PREREG.md. Refuses to run before the OSF registration is recorded
in OSF_REGISTRATION.txt, and refuses the time hold-out before 2029-02-01.

python -m kuramoto_cot.confirm ch2a --cftc <dir> --prices <dir> --out <json>
python -m kuramoto_cot.confirm time --cftc <dir> --prices <extended dir> --out <json>   (from 2029-02-01)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import cotsync as cs
from . import run_prereg as rp
from . import stats as st

HERE = Path(__file__).parent
FX = ["AUD.FUT", "CAD.FUT", "CHF.FUT", "EUR.FUT", "GBP.FUT", "JPY.FUT", "MXN.FUT", "NZD.FUT"]
HOLDOUT_COMMODITIES = ["CC", "CT", "KC", "SB", "KE", "PA"]
TIME_HOLDOUT = ("2026-03-02", "2028-12-29")
UNLOCK = pd.Timestamp("2029-02-01")
ALPHA = 0.05 / 3
EXTRA = ["c_abs_mean_z", "c_mean_z"]


def registration_url(folder: Path = HERE) -> str:
    f = Path(folder) / "OSF_REGISTRATION.txt"
    url = f.read_text().strip() if f.exists() else ""
    if not re.match(r"^https://osf\.io/\S+$", url):
        raise SystemExit("no OSF registration recorded in OSF_REGISTRATION.txt: deposit CONFIRM_PREREG.md first")
    return url


def check_time_holdout_date(today: pd.Timestamp):
    if today < UNLOCK:
        raise SystemExit("the time hold-out (CH1, CH2b) is run once, from 2029-02-01")


def _rows(x_book: pd.DataFrame, prices: dict, days, r: pd.Series) -> pd.DataFrame:
    rows = rp.weekly_rows(x_book, prices, days)
    rows["r"] = r
    z, _ = cs.standardize(x_book)
    rows["c_abs_mean_z"] = z.mean(axis=1).abs()
    rows["c_mean_z"] = z.mean(axis=1)
    return rows[~rows.index.map(cs.in_shutdown)]


def test_one(df: pd.DataFrame, outcome: str, sign: int, rng) -> dict:
    cols = ["r"] + rp.CONTROLS + EXTRA
    df = df.dropna(subset=cols + [outcome])
    X = rp._design(df, cols)
    y = df[outcome].to_numpy()
    sd = float(df[outcome].std())
    b = st.ols(y, X)[1]
    se = st.newey_west_se(y, X, rp.NW_LAG)[1]
    t = b / se
    p_one = float(1 - sps.norm.cdf(sign * t))
    boots = np.array([st.ols(y[i], X[i])[1] for i in
                      (st.stationary_bootstrap_indices(len(y), rp.BLOCK, rng) for _ in range(5000))])
    p_boot = float((1 + np.sum(sign * (boots - b) >= sign * b)) / 5001)
    return {"n": int(len(df)), "first_entry": str(df["entry"].min().date()), "last_entry": str(df["entry"].max().date()),
            "b_sd": float(b / sd), "ci95_sd": [float((b - 1.96 * se) / sd), float((b + 1.96 * se) / sd)],
            "t_nw": float(t), "p_one_sided": p_one, "alpha": ALPHA, "confirmed": bool(p_one < ALPHA),
            "p_bootstrap_one_sided": p_boot, "predicted_sign": sign}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["ch2a", "time"])
    ap.add_argument("--cftc", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    url = registration_url()
    if a.which == "time":
        check_time_holdout_date(pd.Timestamp.today().normalize())
    rng = np.random.default_rng(20290201 if a.which == "time" else 20261110)
    reg = json.loads((HERE / "results" / "results.json").read_text())
    dis16 = [m for m in reg["universe"] if cs.MAPPING[m][0] == "dis"]
    x_all = cs.load_cftc(Path(a.cftc))
    roots = sorted(set(dis16) | set(FX) | set(HOLDOUT_COMMODITIES) | {"ES"})
    prices = cs.load_prices(Path(a.prices), roots)
    days = prices["ES"].index
    r_dis = cs.order_parameter(cs.phases(x_all[dis16], "plane"))
    res = {"osf": url, "prices": str(a.prices), "which": a.which}
    if a.which == "ch2a":
        rows = _rows(x_all[HOLDOUT_COMMODITIES], {m: prices[m] for m in HOLDOUT_COMMODITIES}, days, r_dis)
        rows = rows[rows["entry"] <= pd.Timestamp("2026-01-31")]
        res["CH2a"] = test_one(rows, "Y1_return", +1, rng)
    else:
        span = lambda d: d[d["entry"].between(*map(pd.Timestamp, TIME_HOLDOUT))]
        r_fx = cs.order_parameter(cs.phases(x_all[FX], "plane"))
        res["CH1"] = test_one(span(_rows(x_all[FX], {m: prices[m] for m in FX}, days, r_fx)), "Y2_maxdd", -1, rng)
        res["CH2b"] = test_one(span(_rows(x_all[dis16], {m: prices[m] for m in dis16}, days, r_dis)), "Y1_return", +1, rng)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
