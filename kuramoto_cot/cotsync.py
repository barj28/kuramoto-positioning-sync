"""Synchronization of real speculative positioning (CFTC), as pre-registered in PREREG.md.

Everything here is causal: a value at report date t uses reports up to t only,
and prices up to the entry date (first trading day on or after t + 6 days).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# root -> (report, CFTC contract market code). Leveraged Funds in TFF for
# financials, Managed Money in the disaggregated report for commodities.
MAPPING: dict[str, tuple[str, str]] = {
    "ES": ("tff", "13874A"), "NQ": ("tff", "209742"), "YM": ("tff", "124603"),
    "RTY": ("tff", "239742"), "EMD": ("tff", "33874A"), "NKD": ("tff", "240741"),
    "ZT": ("tff", "042601"), "ZF": ("tff", "044601"), "ZN": ("tff", "043602"),
    "TN": ("tff", "043607"), "ZB": ("tff", "020601"), "UB": ("tff", "020604"),
    "EUR.FUT": ("tff", "099741"), "JPY.FUT": ("tff", "097741"), "GBP.FUT": ("tff", "096742"),
    "CAD.FUT": ("tff", "090741"), "AUD.FUT": ("tff", "232741"), "CHF.FUT": ("tff", "092741"),
    "MXN.FUT": ("tff", "095741"), "NZD.FUT": ("tff", "112741"),
    "CL": ("dis", "067651"), "NG": ("dis", "023651"), "HO": ("dis", "022651"),
    "RB": ("dis", "111659"), "GC": ("dis", "088691"), "SI": ("dis", "084691"),
    "HG": ("dis", "085692"), "PL": ("dis", "076651"), "PA": ("dis", "075651"),
    "ZC": ("dis", "002602"), "ZW": ("dis", "001602"), "KE": ("dis", "001612"),
    "ZS": ("dis", "005602"), "ZL": ("dis", "007601"), "ZM": ("dis", "026603"),
    "LE": ("dis", "057642"), "HE": ("dis", "054642"), "GF": ("dis", "061641"),
    "SB": ("dis", "080732"), "KC": ("dis", "083731"), "CC": ("dis", "073732"),
    "CT": ("dis", "033661"),
}
_LONG_SHORT = {
    "tff": ("Lev_Money_Positions_Long_All", "Lev_Money_Positions_Short_All"),
    "dis": ("M_Money_Positions_Long_All", "M_Money_Positions_Short_All"),
}
_FILES = {"tff": ("fin_fut_txt_*/*.txt",), "dis": ("fut_disagg_txt_*/*.txt",)}

SHUTDOWNS = (("2013-10-01", "2013-11-30"), ("2018-12-18", "2019-03-08"), ("2025-09-30", "2025-12-31"))
Z_WIN, Z_MIN = 156, 52
COVERAGE_MIN = 0.85


def _code(c) -> str:
    c = str(c).strip()
    return c.zfill(6) if c.isdigit() else c


def load_cftc(raw_dir: Path) -> pd.DataFrame:
    """Weekly net speculative position / open interest, report dates x roots."""
    want = {(k, c): root for root, (k, c) in MAPPING.items()}
    frames = []
    for kind, patterns in _FILES.items():
        lo, sh = _LONG_SHORT[kind]
        for pat in patterns:
            for f in sorted(Path(raw_dir).glob(pat)):
                d = pd.read_csv(f, low_memory=False, usecols=[
                    "Report_Date_as_YYYY-MM-DD", "CFTC_Contract_Market_Code", "Open_Interest_All", lo, sh])
                d["code"] = d["CFTC_Contract_Market_Code"].map(_code)
                d["root"] = [want.get((kind, c)) for c in d["code"]]
                d = d[d["root"].notna()]
                d["date"] = pd.to_datetime(d["Report_Date_as_YYYY-MM-DD"], format="mixed").dt.normalize()
                d["x"] = (d[lo] - d[sh]) / d["Open_Interest_All"].where(d["Open_Interest_All"] > 0)
                frames.append(d[["date", "root", "x"]])
    long = pd.concat(frames).drop_duplicates(["date", "root"], keep="last")
    return long.pivot(index="date", columns="root", values="x").sort_index()


def load_prices(price_dir: Path, roots) -> dict[str, pd.Series]:
    out = {}
    for r in roots:
        f = Path(price_dir) / f"{r}_daily.parquet"
        if f.exists():
            d = pd.read_parquet(f, columns=["date", "close"])
            out[r] = pd.Series(d["close"].to_numpy(), index=pd.to_datetime(d["date"]).dt.normalize()).sort_index()
    return out


def select_universe(x: pd.DataFrame, prices: dict[str, pd.Series], trading_days: pd.DatetimeIndex,
                    cftc_span=("2009-01-01", "2026-11-03"), price_span=("2010-06-07", "2026-01-31")) -> list[str]:
    """Mechanical filter of PREREG section 1: >= 85% coverage for both the CFTC series and the prices."""
    w = x.loc[cftc_span[0]:cftc_span[1]]
    days = trading_days[(trading_days >= price_span[0]) & (trading_days <= price_span[1])]
    keep = []
    for r in x.columns:
        if r not in prices:
            continue
        cftc_ok = w[r].notna().mean() >= COVERAGE_MIN
        price_ok = prices[r].reindex(days).notna().mean() >= COVERAGE_MIN
        if cftc_ok and price_ok:
            keep.append(r)
    return keep


# ── phases ──────────────────────────────────────────────────────────────────

def standardize(x):
    """Trailing z-score of the position level and of its weekly change (Series or DataFrame)."""
    z = (x - x.rolling(Z_WIN, min_periods=Z_MIN).mean()) / x.rolling(Z_WIN, min_periods=Z_MIN).std()
    dx = x.diff()
    d = dx / dx.rolling(Z_WIN, min_periods=Z_MIN).std()
    return z, d


def phase_plane(x: pd.Series) -> pd.Series:
    """theta = atan2(-d, z): 0 = maximal long, pi/2 = selling, pi = maximal short, 3pi/2 = buying."""
    z, d = standardize(x)
    th = np.mod(np.arctan2(-d, z), 2 * np.pi)
    return th.where(z.notna() & d.notna())


def causal_sfb(x: pd.Series, W: int = 26, nu: float = 0.0, dt_min: int = 2, T_max: int = 13) -> pd.Series:
    """The paper's SFB made causal: a flux peak at t is known at t + 1, and the phase after the
    last known peak is extrapolated at pi per median of the last 4 half-cycles, capped at pi."""
    v = x.to_numpy(dtype=float)
    q = np.r_[np.nan, np.diff(v)]
    phi = np.abs(q)
    T = len(v)
    th = np.full(T, np.nan)
    peaks: list[int] = []
    flat = 0
    for s in range(2, T):
        t = s - 1                                    # candidate peak, confirmed with q[s]
        win = phi[max(1, t - W + 1):t + 1]
        win = win[np.isfinite(win)]
        if win.size and np.isfinite(q[t - 1:s + 1]).all() and phi[t] > win.mean() + nu * win.std():
            is_pk = (q[t] > 0 and q[t] > q[t - 1] and q[t] > q[s]) or (q[t] < 0 and q[t] < q[t - 1] and q[t] < q[s])
            if is_pk:
                if peaks and np.sign(q[t]) == np.sign(q[peaks[-1]]):
                    if phi[t] > phi[peaks[-1]]:
                        peaks[-1] = t
                elif peaks and t - peaks[-1] < dt_min:
                    if phi[t] > phi[peaks[-1]]:
                        peaks[-1] = t
                else:
                    peaks.append(t)
        flat = flat + 1 if (np.isfinite(v[s]) and abs(v[s]) < 1e-10) else 0
        if len(peaks) < 2 or flat > T_max:
            continue
        halves = np.diff(peaks[-5:])
        tau = float(np.median(halves))
        k = peaks[-1]
        base = 0.0 if q[k] > 0 else np.pi
        th[s] = (base + np.pi * min((s - k) / tau, 1.0)) % (2 * np.pi)
    return pd.Series(th, index=x.index)


def phases(x: pd.DataFrame, method: str = "plane") -> pd.DataFrame:
    fn = phase_plane if method == "plane" else causal_sfb
    return pd.DataFrame({c: fn(x[c]) for c in x.columns}, index=x.index)


def order_parameter(th: pd.DataFrame, harmonic: int = 1) -> pd.Series:
    a = th.to_numpy(dtype=float) * harmonic
    ok = np.isfinite(a)
    n = ok.sum(axis=1)
    z = np.where(ok, np.exp(1j * np.where(ok, a, 0.0)), 0).sum(axis=1)
    r = np.where(n >= 2, np.abs(z) / np.maximum(n, 1), np.nan)
    return pd.Series(r, index=th.index)


def circular_shift(th: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Roll each market's phase series by an independent offset within its own valid span."""
    out = th.copy()
    for c in th.columns:
        s = th[c]
        i0, i1 = s.first_valid_index(), s.last_valid_index()
        if i0 is None:
            continue
        a, b = th.index.get_loc(i0), th.index.get_loc(i1) + 1
        seg = s.iloc[a:b].to_numpy()
        out.iloc[a:b, out.columns.get_loc(c)] = np.roll(seg, int(rng.integers(0, len(seg))))
    return out


# ── point in time ───────────────────────────────────────────────────────────

def entry_date(report: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    i = trading_days.searchsorted(report + pd.Timedelta(days=6))
    return trading_days[i] if i < len(trading_days) else None


def in_shutdown(d: pd.Timestamp) -> bool:
    return any(pd.Timestamp(a) <= d <= pd.Timestamp(b) for a, b in SHUTDOWNS)
