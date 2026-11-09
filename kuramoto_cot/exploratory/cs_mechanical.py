"""Is beta_CS mechanical? Simulate a market whose TRUE spread is constant
(liquidity never responds to volatility, true beta = 0), then run the paper's
exact estimator: Corwin-Schultz spread from daily H/L, V = 1/S (S>0 kept),
sigma = 21-day rolling close-to-close vol, OLS log V = a - beta log sigma.
Also: noise floor E[r] for N iid uniform phases."""
import numpy as np

SQ = np.sqrt(2.0) - 1.0
rng = np.random.default_rng(7)


def cs_spread(h, l):
    s = np.full(len(h), np.nan)
    for t in range(1, len(h)):
        b = np.log(h[t] / l[t]) ** 2 + np.log(h[t - 1] / l[t - 1]) ** 2
        g = np.log(max(h[t], h[t - 1]) / min(l[t], l[t - 1])) ** 2
        a = (np.sqrt(b) - np.sqrt(g)) / SQ
        s[t] = max(2 * (np.exp(a) - 1) / (1 + np.exp(a)), 0.0)
    return s


def run(spread_bps, days=2500, steps=390, lagged=False):
    # persistent log-vol AR(1), annualised vol around 20%
    lv = np.empty(days)
    lv[0] = np.log(0.20)
    for t in range(1, days):
        lv[t] = 0.98 * lv[t - 1] + 0.02 * np.log(0.20) + 0.12 * rng.standard_normal()
    sd_step = np.exp(lv) / np.sqrt(252 * steps)
    half = spread_bps * 1e-4 / 2
    logmid = 0.0
    H, L, C = np.empty(days), np.empty(days), np.empty(days)
    for d in range(days):
        path = logmid + np.cumsum(sd_step[d] * rng.standard_normal(steps))
        trade = path + half * rng.choice([-1.0, 1.0], steps)  # bid-ask bounce
        H[d], L[d], C[d] = np.exp(trade.max()), np.exp(trade.min()), np.exp(trade[-1])
        logmid = path[-1]
    S = cs_spread(H, L)
    lr = np.r_[np.nan, np.diff(np.log(C))]
    sig = np.full(days, np.nan)
    for t in range(21 if lagged else 20, days):
        w = lr[t - 21:t] if lagged else lr[t - 20:t + 1]   # lagged: the 21 days before t
        sig[t] = np.sqrt(np.nanmean(w ** 2) * 252)
    m = (S > 0) & np.isfinite(sig)
    y, x = np.log(1 / S[m]), np.log(sig[m])
    slope = np.polyfit(x, y, 1)[0]
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    return -slope, r2, m.mean()


for bps in (0.0, 2.0, 10.0):
    b, r2, kept = run(bps)
    print(f"true spread {bps:4.1f} bps (true beta=0): beta_CS={b:+.2f}  R2={r2:.3f}  share S>0 kept={kept:.2f}")

for N in (53, 4786):
    th = rng.uniform(0, 2 * np.pi, (5000, N))
    r = np.abs(np.exp(1j * th).mean(axis=1))
    print(f"noise floor N={N}: E[r]={r.mean():.4f}  sqrt(pi/(4N))={np.sqrt(np.pi / (4 * N)):.4f}  1/sqrt(N)={1/np.sqrt(N):.4f}")

# Same estimator with sigma over the 21 days BEFORE t (the earlier paper's equity specification)
for bps in (0.0, 2.0, 10.0):
    b, r2, kept = run(bps, lagged=True)
    print(f"lagged sigma, true spread {bps:4.1f} bps (true beta=0): beta_CS={b:+.2f}  R2={r2:.3f}  share S>0 kept={kept:.2f}")
