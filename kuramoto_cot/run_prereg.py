"""Run H1, H2, H3 exactly as frozen in PREREG.md (commit 676c300).

python -m research.kuramoto_cot.run_prereg --cftc <unzipped CFTC dir> --prices <data/futures> --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.kuramoto_cot import cotsync as cs
from research.kuramoto_cot import stats as st

H = 20
H1_SPAN = ("2009-06-01", "2026-11-03")
ENTRY_SPAN = ("2010-06-07", "2026-01-31")
TRAIN_END = pd.Timestamp("2017-12-31")
N_SURR, N_BOOT, BLOCK, NW_LAG = 2000, 5000, 26, 8
CONTROLS = ["c_ret4w", "c_volratio", "c_corr13", "c_pc1_26", "c_absz"]
OUTCOMES = {"Y1_return": -1, "Y2_maxdd": -1, "Y3_volspike": +1}   # predicted sign of b


def h1(th: pd.DataFrame, rng, harmonic=1) -> dict:
    span = th.loc[H1_SPAN[0]:H1_SPAN[1]]
    obs = cs.order_parameter(span, harmonic).mean()
    null = np.array([cs.order_parameter(cs.circular_shift(span, rng), harmonic).mean() for _ in range(N_SURR)])
    n_bar = span.notna().sum(axis=1)
    n_bar = float(n_bar[n_bar >= 2].mean())
    return {"r_mean": float(obs), "null_mean": float(null.mean()), "null_p95": float(np.quantile(null, 0.95)),
            "null_p99": float(np.quantile(null, 0.99)), "ratio": float(obs / null.mean()),
            "p_one_sided": float((1 + np.sum(null >= obs)) / (1 + N_SURR)),
            "iid_floor": float(np.sqrt(np.pi / (4 * n_bar))), "n_markets_mean": n_bar, "holds": None}


def weekly_rows(x: pd.DataFrame, prices: dict, days: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per report date: entry date, outcomes and controls, all point in time."""
    P = pd.DataFrame({r: prices[r].reindex(days) for r in x.columns})
    dP = P.diff()
    sig = dP.ewm(halflife=20, min_periods=60).std()
    rv20, rv60 = dP.rolling(20, min_periods=15).std(), dP.rolling(60, min_periods=45).std()
    z, _ = cs.standardize(x)
    dx = x.diff()
    A, S, R20, R60 = dP.to_numpy(), sig.to_numpy(), rv20.to_numpy(), rv60.to_numpy()
    rows = []
    for k, t in enumerate(x.index):
        e = cs.entry_date(t, days)
        if e is None:
            continue
        i = days.get_loc(e)
        if i + H >= len(days) or i < 60:
            continue
        sgn = np.sign(x.loc[t].to_numpy())
        sgn = np.where(np.isfinite(sgn), sgn, 0.0)
        s_e = S[i]
        ok = (sgn != 0) & np.isfinite(s_e) & (s_e > 0)
        if ok.sum() < 5:
            continue
        fwd = A[i + 1:i + 1 + H][:, ok] * sgn[ok] / s_e[ok]
        back = A[i - H + 1:i + 1][:, ok] * sgn[ok] / s_e[ok]
        book = np.nanmean(fwd, axis=1)
        cum = np.cumsum(np.nan_to_num(book))
        peak = np.maximum.accumulate(np.r_[0.0, cum])[1:]
        fwd_rv = np.nanstd(A[i + 1:i + 1 + H], axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            volspike = np.nanmean(np.log(fwd_rv / R60[i]))
            volratio = np.nanmean(np.log(R20[i] / R60[i]))
        win13 = dx.iloc[max(0, k - 12):k + 1].dropna(axis=1)
        win26 = dx.iloc[max(0, k - 25):k + 1].dropna(axis=1)
        c13 = np.nan
        if win13.shape[0] == 13 and win13.shape[1] >= 5:
            C = np.corrcoef(win13.to_numpy().T)
            c13 = C[np.triu_indices_from(C, 1)].mean()
        pc1 = np.nan
        if win26.shape[0] == 26 and win26.shape[1] >= 5:
            ev = np.linalg.eigvalsh(np.corrcoef(win26.to_numpy().T))
            pc1 = ev[-1] / win26.shape[1]
        rows.append({"report": t, "entry": e, "Y1_return": float(np.nansum(book)),
                     "Y2_maxdd": float((cum - peak).min()), "Y3_volspike": float(volspike),
                     "c_ret4w": float(np.nansum(np.nanmean(back, axis=1))), "c_volratio": float(volratio),
                     "c_corr13": float(c13), "c_pc1_26": float(pc1),
                     "c_absz": float(np.nanmean(np.abs(z.loc[t].to_numpy())))})
    return pd.DataFrame(rows).set_index("report")


def _design(df: pd.DataFrame, cols, ref: pd.DataFrame | None = None) -> np.ndarray:
    ref = df if ref is None else ref
    Z = (df[cols] - ref[cols].mean()) / ref[cols].std()
    return np.column_stack([np.ones(len(df)), Z.to_numpy()])


def h2(df: pd.DataFrame, rcol: str, rng) -> dict:
    cols = [rcol] + CONTROLS
    X = _design(df, cols)
    Y = df[list(OUTCOMES)].to_numpy()
    b_hat = np.array([st.ols(Y[:, k], X)[1] for k in range(Y.shape[1])])
    nw_t = np.array([st.ols(Y[:, k], X)[1] / st.newey_west_se(Y[:, k], X, NW_LAG)[1] for k in range(Y.shape[1])])
    boots = np.empty((N_BOOT, Y.shape[1]))
    for b in range(N_BOOT):
        i = st.stationary_bootstrap_indices(len(df), BLOCK, rng)
        boots[b] = np.linalg.lstsq(X[i], Y[i], rcond=None)[0][1]
    p_rw, t_boot = st.romano_wolf(b_hat, boots)
    out = {"n": int(len(df)), "first_entry": str(df["entry"].min().date()), "last_entry": str(df["entry"].max().date())}
    for k, (name, sign) in enumerate(OUTCOMES.items()):
        out[name] = {"b_std": float(b_hat[k]), "t_nw": float(nw_t[k]), "t_boot": float(t_boot[k]),
                     "p_romano_wolf": float(p_rw[k]), "predicted_sign": sign,
                     "supports": bool(p_rw[k] < 0.05 and np.sign(b_hat[k]) == sign),
                     "rejects_thesis": bool(p_rw[k] < 0.05 and np.sign(b_hat[k]) == -sign)}
    out["holds"] = any(out[n]["supports"] for n in OUTCOMES)
    return out


def h3(df: pd.DataFrame, rcol: str) -> dict:
    # training rows whose 4-week outcome window would cross into the test period are dropped
    train = df[df["entry"] <= TRAIN_END - pd.Timedelta(days=31)]
    test = df[df["entry"] > TRAIN_END]
    big, small = [rcol] + CONTROLS, CONTROLS
    out = {}
    for name in OUTCOMES:
        Xb, Xs = _design(train, big), _design(train, small)
        bb, bs = st.ols(train[name].to_numpy(), Xb), st.ols(train[name].to_numpy(), Xs)
        fb, fs = _design(test, big, train) @ bb, _design(test, small, train) @ bs
        out[name] = st.clark_west(test[name].to_numpy(), fs, fb, NW_LAG)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cftc", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261107)

    x_all = cs.load_cftc(Path(a.cftc))
    prices = cs.load_prices(Path(a.prices), x_all.columns)
    days = prices["ES"].index
    uni = cs.select_universe(x_all, prices, days)
    x = x_all[uni]
    kinds = {r: cs.MAPPING[r][0] for r in uni}
    res = {"prereg_commit": "676c300", "universe": uni, "n_universe": len(uni),
           "excluded": sorted(set(cs.MAPPING) - set(uni))}

    th_plane = cs.phases(x, "plane")
    th_sfb = cs.phases(x, "sfb")
    r_plane = cs.order_parameter(th_plane)
    r_nem = cs.order_parameter(th_plane, 2)
    r_sfb = cs.order_parameter(th_sfb)
    pd.DataFrame({"r_plane": r_plane, "r_nematic": r_nem, "r_sfb": r_sfb,
                  "n_valid": th_plane.notna().sum(axis=1)}).to_parquet(out / "r_series.parquet")

    res["H1_primary"] = h1(th_plane, rng)
    res["H1_primary"]["holds"] = res["H1_primary"]["p_one_sided"] < 0.01
    res["H1_S1_nematic"] = h1(th_plane, rng, harmonic=2)
    res["H1_S2_sfb"] = h1(th_sfb, rng)
    for kind in ("tff", "dis"):
        cols = [r for r in uni if kinds[r] == kind]
        res[f"H1_S3_{kind}"] = h1(th_plane[cols], rng) | {"markets": cols}

    rows = weekly_rows(x, prices, days)
    rows["r_plane"], rows["r_nematic"], rows["r_sfb"] = r_plane, r_nem, r_sfb
    keep = rows["entry"].between(*map(pd.Timestamp, ENTRY_SPAN)) & ~rows.index.map(cs.in_shutdown)
    rows = rows[keep]
    rows.to_parquet(out / "weekly_rows.parquet")
    need = list(OUTCOMES) + CONTROLS
    res["rows_before_dropna"] = int(len(rows))
    for rcol, tag in (("r_plane", "primary"), ("r_nematic", "S1_nematic"), ("r_sfb", "S2_sfb")):
        df = rows.dropna(subset=need + [rcol])
        res[f"H2_{tag}"] = h2(df, rcol, rng)
        res[f"H3_{tag}"] = h3(df, rcol)
    prim = res["H2_primary"]
    res["H3_primary_holds"] = any(prim[n]["supports"] and res["H3_primary"][n]["p_one_sided"] < 0.05
                                  for n in OUTCOMES)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
